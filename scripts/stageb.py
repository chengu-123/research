"""Stage B CLI: SCAR + BMCSA K-parallel sampling on TRELLIS SS.

Accepts EITHER of two input sources (selected automatically by file type):
  (a) Stage A video tensor file (.pt), e.g.
        outputs/<name>/stage_a/wan_video_target_3FHW_uint8.pt
      6 frames are sampled at indices [0, 4, 8, 12, 16, 20].
  (b) Directory containing K=6 segmented PNGs named {i:02d}_seg.png
      (FreeArt3D convention), e.g. example/30857/.

Two-pass pipeline (method.md section 5.2 / pipeline.md section 6.2):
  Pass 1: K-parallel SS sampling with SCAR mix on x_0_pred (v3.3 default;
          mix_space='x_0' preserves position alignment + ODE consistency).
  Pass 2: SDEdit from t*=0.5 with BMCSA on all 24 DiT self-attn blocks.
          M-gate computed dynamically per-block from current modulated hidden
          (v3.3 default; M_compute_mode='dynamic').

Outputs under <output_dir>/:
  O_stack.npy / O_stack_soft.npy          [K, 64, 64, 64]
  O_stack_pass1.npy / _soft.npy           Pass-1 output (before BMCSA)
  z_final.pt                              [K, 8, 16, 16, 16]
  dit_hidden.pt                           {block_id: [K, 4096, 1024]}
  dynamic_M_log.pt                        per-step per-block dynamic M maps
  scar_diagnostics.json                   per-step mix / push stats
  sdedit_report.json                      mode, t_star, voxel deltas
  meta.json                               input source, sampler config
  viz/                                    O_stack.html, base_move_preview.html
  viz/bmcsa/                              M_dynamic_*.html, M_attn_*.html
  viz/guide/                              per-state P_guide / O_guide_bin HTMLs
  per_step/, pass2_per_step/              per-step Tweedie occupancy

Usage:
    conda activate mine
    python scripts/stageb.py \\
        --input      outputs/30857/stage_a/wan_video_target_3FHW_uint8.pt \\
        --output_dir outputs/30857/stage_b \\

    # Or with 6 segmented images:
    python scripts/stageb.py \\
        --input      example/30857 \\
        --output_dir outputs/30857/stage_b \\
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MINE_DIR = os.path.dirname(_THIS_DIR)
if _MINE_DIR not in sys.path:
    sys.path.insert(0, _MINE_DIR)
_TRELLIS_DIR = os.path.join(_MINE_DIR, "TRELLIS")
if _TRELLIS_DIR not in sys.path:
    sys.path.insert(0, _TRELLIS_DIR)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from omegaconf import OmegaConf

from pipelines.recon import build_trellis_pipeline
from pipelines.stage_b_sampler import attach_stage_b_sampler
from pipelines.stage_b_scar import run_scar
from pipelines.utils.seeding import seed_everything
from pipelines.utils.state_input import (
    describe_input_source,
    load_K_state_images,
)


def _parse_state_indices(arg: str) -> List[int]:
    """Parse '0,4,8,12,16,20' into [0, 4, 8, 12, 16, 20]."""
    parts = [s.strip() for s in arg.split(",") if s.strip() != ""]
    return [int(x) for x in parts]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage B driver: SCAR + BMCSA K=6 sampling on TRELLIS SS."
    )

    # ---- required ----
    p.add_argument("--input", required=True, dest="input_source",
                   help="Stage A video tensor .pt OR directory of {i:02d}_seg.png.")
    p.add_argument("--output_dir", required=True,
                   help="Destination directory for Stage B artifacts.")

    # ---- config + device ----
    p.add_argument("--config", default=os.path.join(_MINE_DIR, "configs/v1.yaml"),
                   help="OmegaConf YAML providing scar / stage_b_sdedit defaults.")
    p.add_argument("--device", default="cuda",
                   help="Device for TRELLIS pipeline (cuda or cuda:N).")

    # ---- input geometry ----
    p.add_argument("--K", type=int, default=6,
                   help="Number of articulation states. Default 6.")
    p.add_argument("--state_indices", default="0,4,8,12,16,20",
                   help="Comma-separated frame indices to sample from a Stage A .pt video. "
                        "Ignored when --input is a directory. Default 0,4,8,12,16,20 "
                        "(matches Stage A keyframes_6.png convention for F=21).")
    p.add_argument("--image_pattern", default="{i:02d}_seg.png",
                   help="Filename template for image-directory mode.")

    # ---- v3.3 mode toggles (override config defaults) ----
    p.add_argument("--mix_space", choices=["x_0", "z_t"], default=None,
                   help="SCAR mix space. 'x_0' = v3.3 default (clean signal estimate); "
                        "'z_t' = v4.3 legacy (noisy latent). When unset, falls back to "
                        "config 'scar.mix_space' (default 'x_0').")
    p.add_argument("--M_compute_mode", choices=["dynamic", "static"], default=None,
                   help="BMCSA M-mask compute mode. 'dynamic' = v3.3 default "
                        "(per-block from current hidden); 'static' = v4.3 legacy "
                        "(precomputed M_base from Pass-1). When unset, falls back to "
                        "config 'stage_b_sdedit.M_compute_mode' (default 'dynamic').")
    p.add_argument("--seed", type=int, default=None,
                   help="Shared-noise seed (overrides config io.seed if set).")

    # ---- safety / debugging ----
    p.add_argument("--no_remove_disk", action="store_true",
                   help="Disable grounding-disk / carpet removal post-decode "
                        "(default: enabled per FreeArt3D convention).")
    p.add_argument("--force_sampler", choices=["scar", "bcac", "vgcf"], default=None,
                   help="Override config sampler choice. Default reads 'stage_b.sampler' "
                        "from --config (typically 'scar').")
    p.add_argument("--joint_type", choices=["revolute", "prismatic"], default=None,
                   help="Ignored by Stage B; kept for symmetry with run_v1.py CLI.")

    return p.parse_args()


def _build_sampler(pipe, cfg, args, total_steps: int) -> str:
    """Attach the requested Stage-B sampler to ``pipe`` and return its name.

    The sampler name is returned so that downstream config blocks can be
    selected (e.g. 'scar' uses cfg.scar; 'vgcf' uses cfg.vgcf).
    """
    return attach_stage_b_sampler(
        pipe=pipe,
        cfg=cfg,
        total_steps=total_steps,
        force_sampler=args.force_sampler,
        mix_space=args.mix_space,
    )


def _build_sdedit_cfg(cfg, args) -> Dict[str, Any]:
    """Resolve stage_b_sdedit config (BMCSA / SDEdit Pass-2)."""
    raw = cfg.get("stage_b_sdedit", {}) or {}
    sdedit_cfg = dict(raw)
    # v3.3 defaults: enabled + dynamic-M.
    sdedit_cfg.setdefault("enabled", True)
    sdedit_cfg.setdefault("mode", "bmcsa")
    sdedit_cfg.setdefault("t_star", 0.5)
    sdedit_cfg.setdefault("pass2_steps", 12)
    sdedit_cfg.setdefault("tau_M", 0.05)
    sdedit_cfg.setdefault("token_resolution", 16)
    sdedit_cfg.setdefault("exclude_state_0", False)
    sdedit_cfg.setdefault("bmcsa_blocks", "all")
    sdedit_cfg.setdefault("bmcsa_strength", 1.0)
    sdedit_cfg.setdefault("attn_m_enabled", True)
    sdedit_cfg.setdefault("attn_m_apply_at", "guide")
    sdedit_cfg.setdefault("attn_m_threshold", 0.7)
    sdedit_cfg.setdefault("attn_m_tau", 0.05)
    sdedit_cfg.setdefault("guide_mode", "augmented_intersection")
    sdedit_cfg.setdefault("guide_excl_threshold", 0.83)
    # v3.3 BMCSA M-compute defaults (method.md section 5.2).
    sdedit_cfg.setdefault("M_compute_mode", "dynamic")
    sdedit_cfg.setdefault("tau_M_dynamic", 0.7)
    sdedit_cfg.setdefault("kappa_M_dynamic", 0.05)
    sdedit_cfg.setdefault("capture_dynamic_M", True)

    if args.M_compute_mode is not None:
        sdedit_cfg["M_compute_mode"] = args.M_compute_mode
    return sdedit_cfg


def _save_input_keyframes(images, out_dir: str) -> None:
    """Save the K input images side-by-side as a debugging artifact."""
    import matplotlib.pyplot as plt

    K = len(images)
    fig, axes = plt.subplots(1, K, figsize=(2.4 * K, 2.6))
    if K == 1:
        axes = [axes]
    for j, img in enumerate(images):
        axes[j].imshow(img)
        axes[j].set_title(f"state_{j}", fontsize=10)
        axes[j].set_xticks([])
        axes[j].set_yticks([])
    plt.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "input_keyframes.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"--config not a file: {args.config}")
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    OmegaConf.save(cfg, os.path.join(args.output_dir, "config.yaml"))

    seed = int(args.seed) if args.seed is not None else int(cfg.io.seed)
    seed_everything(seed)

    # ---- 1) Build TRELLIS pipeline -------------------------------------------------
    pipe = build_trellis_pipeline(device=args.device)
    if "sparse_structure_encoder" not in pipe.models:
        raise RuntimeError(
            "TRELLIS SS encoder failed to load (check pretrained path). "
            "Stage B Pass-2 SDEdit requires the encoder; see pipelines/recon.py."
        )

    # ---- 2) Load K state images (multiplex video.pt or image dir) -----------------
    state_indices = _parse_state_indices(args.state_indices)
    if len(state_indices) != int(args.K):
        raise ValueError(
            f"--state_indices ({state_indices}) must have length K={args.K}; "
            f"got len={len(state_indices)}"
        )
    images = load_K_state_images(
        args.input_source,
        K=int(args.K),
        state_indices=state_indices,
        image_pattern=args.image_pattern,
        out_mode="RGBA",
    )
    src_info = describe_input_source(args.input_source)
    inputs_dir = os.path.join(args.output_dir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(inputs_dir, f"{i:02d}_seg.png"))
    _save_input_keyframes(images, args.output_dir)

    # ---- 3) DINOv2 cond ----------------------------------------------------------
    preprocessed = [pipe.preprocess_image(img) for img in images]
    cond = pipe.get_cond(preprocessed)

    # ---- 4) Attach sampler --------------------------------------------------------
    sampler_params = dict(pipe.sparse_structure_sampler_params)
    total_steps_default = int(sampler_params.get("steps", 25))
    sampler_choice = _build_sampler(pipe, cfg, args, total_steps=total_steps_default)
    # NB: when SCARSampler set scar.steps, run_scar will pick it up via cfg_scar.

    # ---- 5) Resolve scar + sdedit cfgs --------------------------------------------
    scar_cfg = dict(cfg.get("scar", {})) if sampler_choice == "scar" else {}
    if "mix_space" not in scar_cfg or args.mix_space is not None:
        scar_cfg["mix_space"] = (
            args.mix_space
            if args.mix_space is not None
            else scar_cfg.get("mix_space", "x_0")
        )
    sdedit_cfg = _build_sdedit_cfg(cfg, args)

    remove_disk_flag = (not args.no_remove_disk) and bool(scar_cfg.get("remove_disk", True))

    # ---- 6) Persist meta.json before running (so a crash leaves trail) ----------
    meta = {
        "stage": "B",
        "input_source": src_info,
        "K": int(args.K),
        "state_indices": state_indices,
        "image_pattern": args.image_pattern,
        "seed": seed,
        "sampler": sampler_choice,
        "device": args.device,
        "scar_cfg": OmegaConf.to_container(OmegaConf.create(scar_cfg), resolve=True),
        "sdedit_cfg": OmegaConf.to_container(OmegaConf.create(sdedit_cfg), resolve=True),
        "sampler_params": OmegaConf.to_container(
            OmegaConf.create(dict(sampler_params)), resolve=True,
        ),
        "remove_disk": remove_disk_flag,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- 7) Run Stage B ------------------------------------------------------------
    print("=" * 78)
    print(f"[stage_b] input source : {src_info}")
    print(f"[stage_b] K            : {args.K}  state_indices={state_indices}")
    print(f"[stage_b] sampler      : {sampler_choice}")
    print(f"[stage_b] mix_space    : {scar_cfg.get('mix_space', 'n/a')}")
    print(f"[stage_b] M_compute    : {sdedit_cfg.get('M_compute_mode', 'n/a')}")
    print(f"[stage_b] sdedit       : enabled={sdedit_cfg.get('enabled')} "
          f"mode={sdedit_cfg.get('mode')} t*={sdedit_cfg.get('t_star')} "
          f"steps={sdedit_cfg.get('pass2_steps')}")
    print(f"[stage_b] output_dir   : {args.output_dir}")
    print(f"[stage_b] seed         : {seed}")
    print("=" * 78)

    res = run_scar(
        pipe=pipe,
        cond=cond,
        K=int(args.K),
        out_dir=args.output_dir,
        cfg_scar=scar_cfg,
        seed=seed,
        remove_disk_flag=remove_disk_flag,
        cfg_sdedit=sdedit_cfg,
    )

    # ---- 8) Final summary --------------------------------------------------------
    K_actual = res.O_stack.shape[0]
    voxel_counts = [int((res.O_stack[k] > 0.5).sum().item()) for k in range(K_actual)]
    print("=" * 78)
    print(f"[stage_b] DONE  -> {args.output_dir}")
    print(f"[stage_b] O_stack shape  : {tuple(res.O_stack.shape)}")
    print(f"[stage_b] voxel counts   : {voxel_counts}")
    print(f"[stage_b] z_final shape  : {tuple(res.z_final.shape)}")
    print("[stage_b] artifacts:")
    for fname in (
        "O_stack.npy", "O_stack_soft.npy", "z_final.pt",
        "dit_hidden.pt", "dynamic_M_log.pt",
        "scar_diagnostics.json", "sdedit_report.json", "meta.json",
        "viz/O_stack.html",
        "viz/bmcsa/dynamic_M_diagnostics.html",
        "per_step/per_step_curves.html",
    ):
        full = os.path.join(args.output_dir, fname)
        if os.path.exists(full):
            print(f"  + {full}")
    print("=" * 78)


if __name__ == "__main__":
    main()

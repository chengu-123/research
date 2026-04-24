"""v1 pipeline entrypoint: VGCF + SAJO + URDF on single-state input images.

Reads K segmented images from ``--input_dir`` (pattern governed by
``configs/v1.yaml:io.image_pattern``), drives Stages B/C/F end-to-end,
and writes the result under ``--output_dir`` with the v1 layout::

    outputs/<name>/
        config.yaml
        inputs/
        stage_b_vgcf/
        stage_c_sajo/
        stage_d_placeholder/
        stage_f/

Stages A (Wan2.2 video), D (dual Stage-2), and E (ACTF fusion) are
deferred; this entrypoint uses a single standard Stage-2 pass on the
state-0 occupancy grid as a placeholder.

Example
-------
    python run_v1.py \
        --input_dir  example/some_object \
        --output_dir outputs/some_object \
        --config     configs/v1.yaml \
        --joint_type revolute
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback

# Ensure TRELLIS is on the path before any trellis imports below.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_REPO_ROOT, "TRELLIS") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "TRELLIS"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True,
                   help="Directory containing K seg images (default pattern 00_seg.png..05_seg.png).")
    p.add_argument("--output_dir", required=True,
                   help="Destination directory (will be created).")
    p.add_argument("--config", default="configs/v1.yaml")
    p.add_argument("--joint_type", choices=["revolute", "prismatic"], default=None,
                   help="If set, skip BIC and force this joint type.")
    p.add_argument("--stage", choices=["all", "b", "c", "d", "f"], default="all",
                   help="Run a single stage (assumes prior stages already wrote artifacts).")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    OmegaConf.save(cfg, os.path.join(args.output_dir, "config.yaml"))

    from pipelines.utils.seeding import seed_everything
    seed_everything(int(cfg.io.seed))

    from pipelines.recon import build_trellis_pipeline
    from pipelines.stage_b_vgcf import run_vgcf
    from pipelines.stage_c_sajo import run_sajo
    from pipelines.stage_d_placeholder import run_stage_d_placeholder
    from pipelines.stage_f_assemble import run_stage_f
    from pipelines.mesh_split import split_mesh_by_masks
    from pipelines.utils.voxel_io import load_seg_images
    from trellis.pipelines.samplers import (
        VGCFSampler, BCACSampler, SCARSampler, generate_alpha_schedule,
    )
    from pipelines.stage_b_scar import run_scar, SCARResult

    pipe = build_trellis_pipeline(device=args.device)

    # Select Stage-B sampler: SCAR (default), BCAC, or VGCF (legacy).
    sampler_choice = str(cfg.get("stage_b", {}).get("sampler", "scar")).lower()
    sigma_min = pipe.sparse_structure_sampler.sigma_min

    if sampler_choice == "scar":
        scar_cfg = cfg.get("scar", {})
        # Resolve alpha schedule: explicit list (if given) wins; otherwise
        # generate from alpha_peak + alpha_decay over total sampler steps.
        total_steps = int(pipe.sparse_structure_sampler_params.get("steps", 25))
        if "alpha_schedule" in scar_cfg and scar_cfg["alpha_schedule"] is not None:
            alpha_schedule = tuple(scar_cfg["alpha_schedule"])
        else:
            alpha_schedule = tuple(generate_alpha_schedule(
                peak=float(scar_cfg.get("alpha_peak", 0.5)),
                total_steps=total_steps,
                decay=str(scar_cfg.get("alpha_decay", "quadratic")),
            ))
        pipe.sparse_structure_sampler = SCARSampler(
            sigma_min=sigma_min,
            alpha_schedule=alpha_schedule,
            active_fraction=float(scar_cfg.get("active_fraction", 0.1)),
            tau_percentile=float(scar_cfg.get("tau_percentile", 0.65)),
            eps_log=float(scar_cfg.get("eps_log", 1.0e-6)),
            eta=float(scar_cfg.get("eta", 0.5)),
            mix_steps=int(scar_cfg.get("mix_steps", 0)),
            mix_weights=tuple(scar_cfg.get("mix_weights", [0.3, 0.4, 0.3])),
            extreme_mix_mode=str(scar_cfg.get("extreme_mix_mode", "symmetric")),
            w_floor=float(scar_cfg.get("w_floor", 0.0)),
            scar_enabled=True,
        )
    elif sampler_choice == "bcac":
        bcac_cfg = cfg.get("bcac", {})
        pipe.sparse_structure_sampler = BCACSampler(
            sigma_min=sigma_min,
            block_start=int(bcac_cfg.get("block_start", 4)),
            block_end=int(bcac_cfg.get("block_end", 9)),
            t_full=float(bcac_cfg.get("t_full", 0.7)),
            t_release=float(bcac_cfg.get("t_release", 0.3)),
            alpha_max=float(bcac_cfg.get("alpha_max", 1.0)),
            tau_percentile=float(bcac_cfg.get("tau_percentile", 0.65)),
            active_fraction=float(bcac_cfg.get("active_fraction", 0.8)),
            eps_log=float(bcac_cfg.get("eps_log", 1e-6)),
            eta=float(bcac_cfg.get("eta", 0.5)),
        )
    elif sampler_choice == "vgcf":
        pipe.sparse_structure_sampler = VGCFSampler(
            sigma_min=sigma_min,
            lambda_max=float(cfg.vgcf.lambda_max),
            t_stop=float(cfg.vgcf.t_stop),
            eta=float(cfg.vgcf.eta),
            vgcf_enabled=bool(cfg.vgcf.enabled),
            active_fraction=float(cfg.vgcf.get("active_fraction", 0.8)),
            tau_percentile=float(cfg.vgcf.get("tau_percentile", 0.65)),
            eps_log=float(cfg.vgcf.get("eps_log", 1.0e-6)),
            lambda_schedule=str(cfg.vgcf.get("lambda_schedule", "warmup")),
        )
    else:
        raise ValueError(f"unknown stage_b.sampler: {sampler_choice}")

    # Load K images and encode conditioning.
    images = load_seg_images(
        args.input_dir,
        K=int(cfg.io.K),
        pattern=str(cfg.io.image_pattern),
    )
    inputs_dir = os.path.join(args.output_dir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(inputs_dir, f"{i:02d}_seg.png"))

    preprocessed = [pipe.preprocess_image(img) for img in images]
    cond = pipe.get_cond(preprocessed)

    # ------------------ Stage B: SCAR / BCAC / VGCF ------------------
    stage_b_dir = os.path.join(args.output_dir, "stage_b")
    need_stage_b = args.stage in ("all", "b") or \
                   not os.path.exists(os.path.join(stage_b_dir, "O_stack.npy"))

    if need_stage_b:
        if sampler_choice == "scar":
            scar_cfg = dict(cfg.scar) if "scar" in cfg else {}
            sdedit_cfg = dict(cfg.stage_b_sdedit) if "stage_b_sdedit" in cfg else None
            stage_b_res = run_scar(
                pipe=pipe,
                cond=cond,
                K=int(cfg.io.K),
                out_dir=stage_b_dir,
                cfg_scar=scar_cfg,
                seed=int(cfg.io.seed),
                remove_disk_flag=bool(scar_cfg.get("remove_disk", True)),
                cfg_sdedit=sdedit_cfg,
            )
        else:
            stage_b_res = run_vgcf(
                pipe=pipe,
                cond=cond,
                K=int(cfg.io.K),
                cfg_vgcf=cfg.vgcf,
                out_dir=stage_b_dir,
                cfg_stage_b=cfg.get("stage_b"),
            )
    else:
        import numpy as np
        from pipelines.stage_b_vgcf import VGCFResult
        O_np = np.load(os.path.join(stage_b_dir, "O_stack.npy")).astype(float)
        soft_np = np.load(os.path.join(stage_b_dir, "O_stack_soft.npy")).astype(float)
        z = torch.load(os.path.join(stage_b_dir, "z_final.pt"), map_location=args.device)
        # Use VGCFResult shape for reload (same fields that downstream uses)
        stage_b_res = VGCFResult(
            O_stack=torch.from_numpy(O_np).to(args.device),
            O_stack_soft=torch.from_numpy(soft_np).to(args.device),
            z_final=z.to(args.device),
            diagnostics=[],
        )

    if args.stage == "b":
        return

    # ------------------ Stage C: SAJO ------------------
    stage_c_dir = os.path.join(args.output_dir, "stage_c_sajo")
    sajo_res = run_sajo(
        O_stack=stage_b_res.O_stack,
        cfg_sajo=cfg.sajo,
        out_dir=stage_c_dir,
        joint_type_override=args.joint_type,
    )
    if args.stage == "c":
        return

    # ------------------ Stage D placeholder ------------------
    stage_d_dir = os.path.join(args.output_dir, "stage_d_placeholder")

    if bool(cfg.stage_d.use_vgcf_O_0):
        O_0 = stage_b_res.O_stack[0]
    else:
        import warnings
        warnings.warn(
            "stage_d.use_vgcf_O_0=false is not yet implemented; "
            "falling back to VGCF O_0. Set use_vgcf_O_0=true or "
            "implement clean Stage-1 re-run.",
            stacklevel=2,
        )
        O_0 = stage_b_res.O_stack[0]

    # Merge the trellis Stage-2 sampler overrides into cfg_stage_d so
    # the placeholder driver can pick them up.
    stage_d_cfg = OmegaConf.merge(cfg.stage_d, cfg.get("trellis", {}))
    cond_0 = {"cond": cond["cond"][:1], "neg_cond": cond["neg_cond"][:1]}
    d_res = run_stage_d_placeholder(
        pipe=pipe,
        cond_0=cond_0,
        O_0=O_0,
        out_dir=stage_d_dir,
        cfg_stage_d=stage_d_cfg,
    )
    if args.stage == "d":
        return

    # ------------------ Vertex-level split ------------------
    base_mesh, move_mesh = split_mesh_by_masks(
        d_res.mesh,
        sajo_res.M_base.detach().cpu().numpy(),
        sajo_res.M_move.detach().cpu().numpy(),
    )

    # ------------------ Stage F: URDF + pybullet ------------------
    stage_f_dir = os.path.join(args.output_dir, "stage_f")
    f_res = run_stage_f(
        base_mesh=base_mesh,
        move_mesh=move_mesh,
        sajo_result=sajo_res,
        out_dir=stage_f_dir,
        cfg_py=cfg.pybullet,
    )

    print(f"\n[run_v1] done. URDF: {f_res.urdf_path}")
    print(f"[run_v1] pybullet valid: {f_res.pybullet_report.get('valid')}")
    print(f"[run_v1] joint_type: {sajo_res.joint_type}")
    print(f"[run_v1] BIC confidence: {sajo_res.bic.get('confidence')}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

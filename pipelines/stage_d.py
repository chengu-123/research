"""Stage D entry point: thin wrapper that runs ONLY Stage D.

Stage D consumes Bootstrap artifacts (Stage B output) and refines the
canonical articulated 3D via W-RFSDS against Wan2.2 I2V. This file is a
flat CLI / function entry that callers (run_v1.py, ablation scripts, or
``python -m pipelines.stage_d``) can use without importing the deeper
package layout under ``pipelines/stage_d/``.

Two public entry points:

  1. ``run_stage_d(bootstrap_dir, out_dir, wan_ckpt_dir, **overrides)``
     Function call form. Loads ``StageDConfig`` defaults, applies the
     overrides (small Python dict), forwards to the package's
     ``pipelines.stage_d.run_stage_d.run_stage_d_main``.

  2. CLI: ``python -m pipelines.stage_d --bootstrap_dir ... --out_dir ...
     --wan_ckpt_dir ... [--total_iters 10000] [--world_up_axis Y|Z] ...``
     Useful when you want to bench Stage D on a Bootstrap artifact set
     produced earlier by ``python -m pipelines.stage_b_scar ...`` without
     spinning up the full ``run_v1.py`` orchestrator.

Bootstrap artifact contract (loaded by
``pipelines.stage_d.run_stage_d.load_bootstrap_bundle``): see
``pipeline.md`` section 6.1 for the file list expected under
``bootstrap_dir/`` (z_s0, z_slat0, slat_mean/std/shell_mask, U_object,
gaussian_parent_idx, psi_0, phi_0 [canonical-shifted], anchors_object,
M_attn_boot_64, is_carpet_mask, trellis_cond_can, wan_cond_cached,
z_wan_target, wan_video_target_3FHW, s_0_pure).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

# Make sure TRELLIS / Wan2.2 are on sys.path before any deep import; matches
# run_v1.py / stage_a_wan.py convention.
_REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TRELLIS_ROOT = os.path.join(_REPO_ROOT, "TRELLIS")
if os.path.isdir(_TRELLIS_ROOT) and _TRELLIS_ROOT not in sys.path:
    sys.path.insert(0, _TRELLIS_ROOT)

from pipelines.stage_d.config import StageDConfig
from pipelines.stage_d.render import StageDCameraConfig
from pipelines.stage_d.run_stage_d import run_stage_d_main


logger = logging.getLogger(__name__)


# =============================================================================
# Function entry
# =============================================================================

def run_stage_d(
    bootstrap_dir: str,
    out_dir: str,
    wan_ckpt_dir: str,
    *,
    device: str = "cuda",
    device_id: int = 0,
    trellis_pretrained: str = "JeffreyXiang/TRELLIS-image-large",
    camera: Optional[StageDCameraConfig] = None,
    lpips_net: str = "vgg",
    cfg_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run only Stage D given Bootstrap artifacts on disk.

    Parameters
    ----------
    bootstrap_dir : str
        Directory containing Stage B / Bootstrap output files (see
        ``pipeline.md`` section 6.1 for the contract). Typical layout under
        a v1 run: ``outputs/<sample_id>/stage_b/``.
    out_dir : str
        Destination directory for Stage D artifacts. Created if it does
        not exist. Receives ``learnable_p1_final.pt``, ``viz/``,
        ``logs.jsonl``, ``summary.json``, periodic ``ckpt_*.pt``.
    wan_ckpt_dir : str
        Local Wan2.2-I2V-A14B weight directory (same as Stage A).
    device : str           default ``"cuda"``
    device_id : int        default 0
    trellis_pretrained : str
        TRELLIS pipeline pretrained tag, default
        ``"JeffreyXiang/TRELLIS-image-large"`` (matches ``recon.py``).
    camera : Optional[StageDCameraConfig]
        Explicit camera override. When None, defaults to
        ``StageDCameraConfig.freeart3d_canonical()`` (+Z up, FreeArt3D
        rendering convention). For non-PartNet inputs where the source
        camera differs, pass an explicit config.
    lpips_net : str        default ``"vgg"``
    cfg_overrides : Optional[Dict[str, Any]]
        Dict of fields to override on the default ``StageDConfig``.
        Example: ``{"total_iters": 5000, "lr_adapter": 2e-3}``. Unknown
        keys raise ``KeyError``.

    Returns
    -------
    summary : Dict[str, Any]
        Training summary (committed_type, n_iters_run, iter_0_camera_iou,
        type_vote / dual_clone info if applicable). Also written to
        ``<out_dir>/summary.json`` by ``run_stage_d_main``.
    """
    if not os.path.isdir(bootstrap_dir):
        raise FileNotFoundError(f"bootstrap_dir not found: {bootstrap_dir!r}")
    if not os.path.isdir(wan_ckpt_dir):
        raise FileNotFoundError(f"wan_ckpt_dir not found: {wan_ckpt_dir!r}")
    os.makedirs(out_dir, exist_ok=True)

    # Build StageDConfig from defaults + overrides.
    cfg = StageDConfig()
    if cfg_overrides:
        valid_fields = {f.name for f in dataclasses.fields(cfg)}
        for k, v in cfg_overrides.items():
            if k not in valid_fields:
                raise KeyError(
                    f"unknown StageDConfig field: {k!r}; valid: {sorted(valid_fields)}"
                )
            setattr(cfg, k, v)

    logger.info(
        "[pipelines.stage_d] run_stage_d: bootstrap_dir=%s, out_dir=%s, "
        "total_iters=%d",
        bootstrap_dir, out_dir, cfg.total_iters,
    )

    summary = run_stage_d_main(
        bootstrap_dir=bootstrap_dir,
        out_dir=out_dir,
        cfg=cfg,
        wan_ckpt_dir=wan_ckpt_dir,
        repo_root=_REPO_ROOT,
        device=device,
        device_id=device_id,
        trellis_pretrained=trellis_pretrained,
        camera=camera,
        lpips_net=lpips_net,
    )
    return summary


# =============================================================================
# CLI entry
# =============================================================================

def _parse_kv_overrides(items: Optional[list]) -> Dict[str, Any]:
    """Parse ``--cfg key=value`` items into a typed override dict.

    Values are parsed as JSON ('1.5', '"vgg"', 'true', '5000') so the user
    can pass ints, floats, strings, bools, lists. Unknown keys are not
    rejected here; ``run_stage_d`` validates against ``StageDConfig``.
    """
    if not items:
        return {}
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--cfg expects key=value, got {item!r}")
        k, raw = item.split("=", 1)
        try:
            out[k.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            # Fall back to string when raw is not valid JSON
            # (e.g. ``world_up_axis=Y`` rather than ``world_up_axis="Y"``).
            out[k.strip()] = raw
    return out


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pipelines.stage_d",
        description=(
            "Run only Stage D (W-RFSDS geometry refinement) given existing "
            "Stage B / Bootstrap artifacts. Produces learnable_p1_final.pt, "
            "logs.jsonl, summary.json, and viz/ HTML+PNG debug outputs."
        ),
    )
    p.add_argument("--bootstrap_dir", type=str, required=True,
                    help="Path to Bootstrap (Stage B) output dir.")
    p.add_argument("--out_dir", type=str, required=True,
                    help="Destination dir for Stage D artifacts.")
    p.add_argument("--wan_ckpt_dir", type=str, required=True,
                    help="Local Wan2.2-I2V-A14B checkpoint dir (same as Stage A).")
    p.add_argument("--device", type=str, default="cuda",
                    help="Torch device string. Default 'cuda'.")
    p.add_argument("--device_id", type=int, default=0,
                    help="CUDA device index. Default 0.")
    p.add_argument("--trellis_pretrained", type=str,
                    default="JeffreyXiang/TRELLIS-image-large",
                    help="TRELLIS pretrained tag (HF hub or local cache key).")
    p.add_argument("--lpips_net", type=str, default="vgg",
                    choices=["vgg", "alex", "squeeze"],
                    help="LPIPS backbone for L_rgb / L_first.")
    p.add_argument("--cfg", action="append", default=None,
                    metavar="key=value",
                    help=("Override a StageDConfig field. Repeat for multiple. "
                          "Values are parsed as JSON; bare strings work too. "
                          "Examples: --cfg total_iters=5000 "
                          "--cfg lr_adapter=2e-3 "
                          "--cfg cfg_warmup_g0=22.0"))
    p.add_argument("--log_level", type=str, default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Python logging level.")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    overrides = _parse_kv_overrides(args.cfg)
    summary = run_stage_d(
        bootstrap_dir=args.bootstrap_dir,
        out_dir=args.out_dir,
        wan_ckpt_dir=args.wan_ckpt_dir,
        device=args.device,
        device_id=args.device_id,
        trellis_pretrained=args.trellis_pretrained,
        lpips_net=args.lpips_net,
        cfg_overrides=overrides,
    )
    # Pretty-print summary to stdout for shell consumers.
    print(json.dumps(summary, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()


__all__ = ["run_stage_d", "main"]

"""Command line entry for ``python -m pipelines.stage_d``."""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

from .config import StageDConfig
_DEFAULT_TRELLIS_PRETRAINED = os.path.abspath(
    os.path.expanduser(os.environ.get("TRELLIS_PRETRAINED", "~/hf_models/TRELLIS-image-large"))
)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def _parse_kv_overrides(items: Optional[list]) -> Dict[str, Any]:
    if not items:
        return {}
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--cfg expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        out[key.strip()] = json.loads(raw)
    return out


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pipelines.stage_d",
        description="Run Stage D P1 from a Bootstrap artifact directory.",
    )
    p.add_argument("--bootstrap_dir", required=True,
                   help="Directory containing Bootstrap B12 files.")
    p.add_argument("--out_dir", required=True,
                   help="Destination directory for Stage D artifacts.")
    p.add_argument("--wan_ckpt_dir", required=True,
                   help="Local Wan2.2-I2V-A14B checkpoint directory.")
    p.add_argument("--device", default="cuda",
                   help="Torch device string. Default cuda.")
    p.add_argument("--device_id", type=int, default=0,
                   help="CUDA device index used by Wan. Default 0.")
    p.add_argument("--trellis_pretrained", default=_DEFAULT_TRELLIS_PRETRAINED,
                   help="Local TRELLIS-image-large directory containing pipeline.json.")
    p.add_argument("--lpips_net", default="vgg", choices=["vgg", "alex", "squeeze"],
                   help="LPIPS backbone. Default vgg.")
    p.add_argument("--cfg", action="append", default=None, metavar="key=json",
                   help="Override a StageDConfig field. Example: --cfg total_iters=5000")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Python logging level.")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    repo_root = _repo_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    trellis_root = os.path.join(repo_root, "TRELLIS")
    if os.path.isdir(trellis_root) and trellis_root not in sys.path:
        sys.path.insert(0, trellis_root)

    cfg = StageDConfig()
    overrides = _parse_kv_overrides(args.cfg)
    valid_fields = {f.name for f in dataclasses.fields(cfg)}
    for key, value in overrides.items():
        if key not in valid_fields:
            raise KeyError(f"unknown StageDConfig field: {key!r}")
        setattr(cfg, key, value)

    from .run_stage_d import run_stage_d_main

    summary = run_stage_d_main(
        bootstrap_dir=os.path.abspath(args.bootstrap_dir),
        out_dir=os.path.abspath(args.out_dir),
        cfg=cfg,
        wan_ckpt_dir=os.path.abspath(args.wan_ckpt_dir),
        repo_root=repo_root,
        device=str(args.device),
        device_id=int(args.device_id),
        trellis_pretrained=str(args.trellis_pretrained),
        camera=None,
        lpips_net=str(args.lpips_net),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()

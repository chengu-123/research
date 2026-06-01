"""Build the Stage D bootstrap bundle from Stage A video or six state images.

The input can be either:

    <output_dir>/stage_a/wan_video_target_3FHW_uint8.pt

or six image files named by --image_dir/--image_pattern, matching the Stage B
input convention.

It runs the Bootstrap B3-B12 path:
Stage B SCAR/BMCSA, Stage C joint init, U_object expansion, SLAT sampling,
Wan I2V condition caching, Wan VAE target encoding, and final persistence to:

    <output_dir>/bootstrap/

Usage:
    python scripts/bootstrap.py \
        --output_dir outputs/30857 \
        --input_mode stagea_video \
        --motion "A brown wooden desk. The right drawer slides open." \
        --wan_ckpt /path/to/Wan2.2-I2V-A14B \
        --s0_pure /path/to/PartNet/30857/00_pure.png

    python scripts/bootstrap.py \
        --output_dir outputs/30857 \
        --input_mode six_images \
        --image_dir /path/to/PartNet/30857 \
        --image_pattern "rendering_joint_00_state_{i:02d}.png" \
        --motion "A brown wooden desk. The right drawer slides open." \
        --wan_ckpt /path/to/Wan2.2-I2V-A14B \
        --s0_pure /path/to/PartNet/30857/00_pure.png
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MINE_DIR = os.path.dirname(_THIS_DIR)
if _MINE_DIR not in sys.path:
    sys.path.insert(0, _MINE_DIR)
_TRELLIS_DIR = os.path.join(_MINE_DIR, "TRELLIS")
if _TRELLIS_DIR not in sys.path:
    sys.path.insert(0, _TRELLIS_DIR)
_DEFAULT_TRELLIS_PRETRAINED = os.path.abspath(
    os.path.expanduser(os.environ.get("TRELLIS_PRETRAINED", "~/hf_models/TRELLIS-image-large"))
)

import torch
from omegaconf import OmegaConf


def _section(cfg: Any, name: str) -> Dict[str, Any]:
    if name not in cfg:
        return {}
    value = OmegaConf.to_container(cfg[name], resolve=True)
    return dict(value) if value is not None else {}


def _load_wan_condition_modules(
    wan_ckpt_dir: str,
    device_id: int,
    convert_model_dtype: bool,
) -> Any:
    from pipelines.stage_a_wan import _load_wan_i2v_components

    wan_cfg, _WanI2V = _load_wan_i2v_components()
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_1 import Wan2_1_VAE

    del _WanI2V
    del convert_model_dtype

    device = torch.device(f"cuda:{int(device_id)}")
    text_encoder = T5EncoderModel(
        text_len=wan_cfg.text_len,
        dtype=wan_cfg.t5_dtype,
        device=device,
        checkpoint_path=os.path.join(wan_ckpt_dir, wan_cfg.t5_checkpoint),
        tokenizer_path=os.path.join(wan_ckpt_dir, wan_cfg.t5_tokenizer),
        shard_fn=None,
    )
    vae = Wan2_1_VAE(
        vae_pth=os.path.join(wan_ckpt_dir, wan_cfg.vae_checkpoint),
        device=device,
    )
    return SimpleNamespace(
        config=wan_cfg,
        device=device,
        text_encoder=text_encoder,
        vae=vae,
        vae_stride=wan_cfg.vae_stride,
        patch_size=wan_cfg.patch_size,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Bootstrap B3-B12 artifacts from Stage A video or six images."
    )
    p.add_argument("--output_dir", required=True,
                   help="Top-level run directory. Created for six_images mode; "
                        "must contain stage_a/ for default stagea_video mode.")
    p.add_argument("--input_mode", default="stagea_video",
                   choices=["stagea_video", "six_images"],
                   help="Bootstrap input source. stagea_video reads a Stage A "
                        "uint8 tensor; six_images reads K state images.")
    p.add_argument("--stagea_video", default=None,
                   help="Optional path to Stage A wan_video_target_3FHW_uint8.pt. "
                        "When unset, reads output_dir/stage_a/wan_video_target_3FHW_uint8.pt.")
    p.add_argument("--image_dir", default=None,
                   help="Directory containing K observed state images. In "
                        "stagea_video mode, these images drive Stage B/C while "
                        "the dense target still comes from --stagea_video.")
    p.add_argument("--image_pattern", default="rendering_joint_00_state_{i:02d}.png",
                   help="Filename template for observed state images.")
    p.add_argument("--state_images", nargs=6, default=None,
                   help="Six explicit observed state image paths. Overrides "
                        "--image_dir/--image_pattern when provided.")
    p.add_argument("--pure_image_dir", default=None,
                   help="Directory containing the six no-background pure target images.")
    p.add_argument("--pure_image_pattern", default="{i:02d}_pure.png",
                   help="Filename template for the six pure target images.")
    p.add_argument("--pure_state_images", nargs=6, default=None,
                   help="Six explicit no-background pure target image paths.")
    p.add_argument("--s0_pure", required=True,
                   help="No-carpet frame-0 image, typically 00_pure.png in the "
                        "same source directory as 00_seg.png. Used only for "
                        "Stage D supervision and cached Wan conditioning.")
    p.add_argument("--s5_pure", default=None,
                   help="No-carpet final-state image, typically 05_pure.png. "
                        "Required for --wan_backend fun_inp and Stage D L_last.")
    p.add_argument("--wan_backend", default="fun_inp", choices=["i2v", "fun_inp"],
                   help="Cached Wan condition backend for Stage D W-RFSDS.")
    p.add_argument("--motion", required=True,
                   help="Same user motion prompt used for Stage A.")
    p.add_argument("--wan_ckpt", required=True, dest="wan_ckpt_dir",
                   help="Local Wan checkpoint directory. Use Wan2.2-Fun-A14B-InP "
                        "for the active Stage D contract.")
    p.add_argument("--config", default=os.path.join(_MINE_DIR, "configs/v1.yaml"),
                   help="Config file providing scar and stage_b_sdedit sections.")
    p.add_argument("--device", default="cuda",
                   help="Torch device for TRELLIS and Bootstrap. Default cuda.")
    p.add_argument("--device_id", type=int, default=0,
                   help="CUDA device index used by Wan modules. Default 0.")
    p.add_argument("--trellis_pretrained", default=_DEFAULT_TRELLIS_PRETRAINED,
                   help="Local TRELLIS-image-large directory containing pipeline.json.")
    p.add_argument("--stage_a_size", default="832*480",
                   help="Wan area profile used by the existing Stage A artifact.")
    p.add_argument("--stage_a_frame_num", type=int, default=21,
                   help="Expected Stage A frame count.")
    p.add_argument("--lang", default="en", choices=["zh", "en"],
                   help="Prompt-template language for cached Wan condition.")
    p.add_argument("--guide_scale", type=float, default=3.5,
                   help="Recorded Stage A guide scale for metadata.")
    p.add_argument("--seed", type=int, default=None,
                   help="Shared seed for Stage A metadata and Stage B SCAR. "
                        "Default reads io.seed from --config.")
    p.add_argument("--no_convert_model_dtype", action="store_true",
                   help="Keep Wan models in their checkpoint dtype.")
    p.add_argument("--no_remove_disk", action="store_true",
                   help="Do not remove the FreeArt3D grounding disk in Stage B.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    wan_ckpt_dir = os.path.abspath(args.wan_ckpt_dir)
    config_path = os.path.abspath(args.config)
    s0_pure_path = os.path.abspath(args.s0_pure)
    s5_pure_path = os.path.abspath(args.s5_pure) if args.s5_pure is not None else None

    stagea_video_path = None
    image_dir = None
    state_images = ()
    pure_image_dir = None
    pure_state_images = ()
    if str(args.input_mode) == "stagea_video":
        if args.stagea_video is not None:
            os.makedirs(output_dir, exist_ok=True)
            stagea_video_path = os.path.abspath(args.stagea_video)
        else:
            if not os.path.isdir(output_dir):
                raise FileNotFoundError(f"--output_dir not found: {output_dir}")
            stagea_video_path = os.path.join(
                output_dir, "stage_a", "wan_video_target_3FHW_uint8.pt"
            )
        if not os.path.isfile(stagea_video_path):
            raise FileNotFoundError(f"missing Stage A video tensor: {stagea_video_path}")
    else:
        os.makedirs(output_dir, exist_ok=True)
    if args.state_images is not None:
        state_images = tuple(os.path.abspath(p) for p in args.state_images)
        for p in state_images:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"missing state image: {p}")
    elif args.image_dir is not None:
        image_dir = os.path.abspath(args.image_dir)
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"--image_dir not found: {image_dir}")
    elif str(args.input_mode) == "six_images":
        raise ValueError("--image_dir or --state_images is required for --input_mode six_images")
    if args.pure_state_images is not None:
        pure_state_images = tuple(os.path.abspath(p) for p in args.pure_state_images)
        for p in pure_state_images:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"missing pure state image: {p}")
    elif args.pure_image_dir is not None:
        pure_image_dir = os.path.abspath(args.pure_image_dir)
        if not os.path.isdir(pure_image_dir):
            raise FileNotFoundError(f"--pure_image_dir not found: {pure_image_dir}")
    if not os.path.isdir(wan_ckpt_dir):
        raise FileNotFoundError(f"--wan_ckpt not found: {wan_ckpt_dir}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"--config not found: {config_path}")
    if not os.path.isfile(s0_pure_path):
        raise FileNotFoundError(f"--s0_pure not found: {s0_pure_path}")
    if str(args.wan_backend) == "fun_inp":
        if s5_pure_path is None:
            raise ValueError("--s5_pure is required when --wan_backend fun_inp")
        if not os.path.isfile(s5_pure_path):
            raise FileNotFoundError(f"--s5_pure not found: {s5_pure_path}")
    elif s5_pure_path is not None and not os.path.isfile(s5_pure_path):
        raise FileNotFoundError(f"--s5_pure not found: {s5_pure_path}")

    from pipelines.bootstrap import BootstrapConfig, run_bootstrap
    from pipelines.recon import build_trellis_pipeline

    raw_cfg = OmegaConf.load(config_path)
    seed = int(args.seed) if args.seed is not None else int(raw_cfg.io.seed)
    boot_cfg = BootstrapConfig(
        skip_b1_stage_a=True,
        wan_ckpt_dir=wan_ckpt_dir,
        bootstrap_input_mode=str(args.input_mode),
        stage_a_video_path=stagea_video_path,
        stage_image_dir=image_dir,
        stage_image_pattern=str(args.image_pattern),
        stage_image_paths=state_images,
        stage_pure_image_dir=pure_image_dir,
        stage_pure_image_pattern=str(args.pure_image_pattern),
        stage_pure_image_paths=pure_state_images,
        s_0_pure_path=s0_pure_path,
        s_5_pure_path=s5_pure_path,
        wan_condition_backend=str(args.wan_backend),
        stage_a_wan_size=str(args.stage_a_size),
        stage_a_frame_num=int(args.stage_a_frame_num),
        stage_a_seed=seed,
        stage_a_lang=str(args.lang),
        stage_a_guide_scale=float(args.guide_scale),
        stage_a_convert_model_dtype=not bool(args.no_convert_model_dtype),
        stage_a_device_id=int(args.device_id),
        cfg_scar=_section(raw_cfg, "scar"),
        cfg_sdedit=_section(raw_cfg, "stage_b_sdedit"),
        stage_b_remove_disk=not bool(args.no_remove_disk),
        device=str(args.device),
    )

    print(f"[bootstrap_cli] loading TRELLIS on {args.device}")
    pipe = build_trellis_pipeline(
        device=str(args.device),
        pretrained=str(args.trellis_pretrained),
    )

    print("[bootstrap_cli] loading Wan T5/VAE for cached condition and target latent")
    wan_owner = _load_wan_condition_modules(
        wan_ckpt_dir=wan_ckpt_dir,
        device_id=int(args.device_id),
        convert_model_dtype=not bool(args.no_convert_model_dtype),
    )

    result = run_bootstrap(
        s_0_with_carpet=None,
        user_motion_prompt=str(args.motion),
        out_dir=output_dir,
        pipe=pipe,
        cfg=boot_cfg,
        wan_t5=wan_owner.text_encoder,
        wan_vae=wan_owner.vae,
    )
    print("=" * 78)
    print(f"[bootstrap_cli] wrote Stage D bundle: {os.path.join(output_dir, 'bootstrap')}")
    print(f"[bootstrap_cli] joint_type={result.joint_init.joint_type()}")
    print(f"[bootstrap_cli] n_U_object={int(result.U_object.shape[0])}")
    print("=" * 78)


if __name__ == "__main__":
    main()

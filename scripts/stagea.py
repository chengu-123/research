"""Stage A CLI: read a single closed-state image and generate a 21-frame
articulated-object video via Wan2.2 I2V.

The user supplies one RGB(A) image of the object in its closed state (with
the FreeArt3D grounding disk / carpet already baked in) and a short text
description of the moving part's motion (e.g. "the drawer slowly slides
outward"). This script appends the universal locked-camera addon, runs
Wan2.2-I2V-A14B once (single fixed seed), runs the background-static
optical-flow sanity check, and writes the Stage A artifact bundle:

    <output_dir>/
        input_s_0_with_carpet.png          RGB input fed to Wan
        wan_video_target.mp4              the generated 16 fps video
        wan_video_grid.png                all 21 frames as a labelled grid
        keyframes_6.png                   states [0,4,8,12,16,20] for Stage B
        optical_flow_per_frame.png        per-transition mean flow + threshold
        meta.json                         prompts, seed, sanity report
        wan_video_target_3FHW_uint8.pt    [3, 21, H, W] uint8 (Stage B input)

Usage:
    conda activate mine
    python scripts/stagea.py \\
        --image      example/30857/05_pure.png \\
        --motion     "the drawer slowly slides outward in a continuous motion" \\
        --wan_ckpt   /path/to/Wan2.2-I2V-A14B \\
        --output_dir outputs/30857/stage_a \\
        --lang       en

The negative prompt is built from pipelines/wan_helpers/prompt_templates/<lang>.json
and deliberately omits the tokens "static" / "motionless" that appear in
Wan's default sample_neg_prompt (those would actively penalise the
locked-camera behaviour we require). Carpet stays baked in throughout the
video; Stage G Export is the only place is_carpet_mask filters it out.
"""

# -----------------------------------------------------------------------------
# Offline guards: must precede any HuggingFace lib import. pipelines.stage_a_wan
# sets the same vars at module load, but we set them here too so that any
# argparse / PIL / dependency import order remains safe.
# -----------------------------------------------------------------------------
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import argparse
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MINE_DIR = os.path.dirname(_THIS_DIR)
if _MINE_DIR not in sys.path:
    sys.path.insert(0, _MINE_DIR)

from PIL import Image

from pipelines.stage_a_wan import SUPPORTED_WAN_SIZE_LABELS, run_stage_a
from pipelines.wan_helpers import build_articulated_prompts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage A driver: Wan2.2 I2V articulated-object video generation."
    )

    # ---- required ----
    p.add_argument("--image", required=True,
                   help="Path to the closed-state input image (RGB or RGBA, "
                        "carpet/grounding disk already baked in).")
    p.add_argument("--motion", required=True,
                   help="Per-object motion description, e.g. "
                        "\"the drawer slowly slides outward in a continuous motion\". "
                        "The universal camera-lock addon is appended automatically.")
    p.add_argument("--wan_ckpt", required=True, dest="wan_ckpt_dir",
                   help="Local Wan2.2-I2V-A14B checkpoint directory containing "
                        "low_noise_model/, high_noise_model/, google/umt5-xxl/, "
                        "models_t5_umt5-xxl-enc-bf16.pth, Wan2.1_VAE.pth.")
    p.add_argument("--output_dir", required=True,
                   help="Destination directory (created if missing). Receives the "
                        "five debug artifacts plus the uint8 tensor consumed by Stage B.")

    # ---- prompt / language ----
    p.add_argument("--lang", default="en", choices=["zh", "en"],
                   help="Language of the universal camera-lock addon and negative "
                        "prompt template. The user motion string can be in any "
                        "language (Wan T5 is multilingual).")

    # ---- Wan2.2 hyperparameters (pipeline_v3 Section 1.3 / 5.3 fix the defaults) ----
    p.add_argument("--seed", type=int, default=42,
                   help="Wan2.2 RNG seed. pipeline_v3 fixes single-seed=42 "
                        "(no multi-candidate selection, aligned with CHORD A.1).")
    p.add_argument("--frame_num", type=int, default=21,
                   help="Output video frame count; must satisfy 4n+1. Default 21 "
                        "yields F_lat=6 matching Stage B K=6 conditioning.")
    p.add_argument("--size", default="832*480", choices=SUPPORTED_WAN_SIZE_LABELS,
                   help="Official Wan2.2 I2V area profile. This is max-area, "
                        "not fixed output H/W; actual H/W follows input aspect. "
                        "Default 832*480 is the 480P profile.")
    p.add_argument("--sampling_steps", type=int, default=50,
                   help="Wan ODE sampling steps. Default 50.")
    p.add_argument("--guide_scale", type=float, default=5.0,
                   help="Wan CFG scale for video generation. Default 5.0 (Wan "
                        "convention). NB: CHORD 25->12 is for W-RFSDS distillation "
                        "in Stage D/F, NOT for video generation here.")
    p.add_argument("--sample_shift", type=float, default=5.0,
                   help="Wan flow scheduler shift. Default 5.0 (Wan i2v_A14B default).")
    p.add_argument("--sample_solver", default="unipc", choices=["unipc", "dpm++"],
                   help="Wan ODE solver. Default unipc.")
    p.add_argument("--fps", type=int, default=16,
                   help="MP4 writer frame rate. Default 16 (Wan sample_fps default).")
    p.add_argument("--device_id", type=int, default=0,
                   help="CUDA device index. CPU is not supported (Wan VAE requires CUDA).")

    # ---- VRAM controls ----
    p.add_argument("--no_offload_model", action="store_true",
                   help="Disable per-step CPU offload of inactive sub-models. "
                        "Lower latency but adds ~24 GB peak VRAM.")
    p.add_argument("--no_convert_model_dtype", action="store_true",
                   help="Keep DiT/VAE in their original mixed-precision dtype. "
                        "Default converts to bf16, which saves VRAM with no quality loss.")
    p.add_argument("--t5_cpu", action="store_true",
                   help="Run T5 text encoder on CPU. Saves ~10 GB VRAM at "
                        "the cost of ~5 s extra per prompt encode.")

    # ---- sanity check ----
    p.add_argument("--no_sanity_check", action="store_true",
                   help="Skip background-static optical-flow sanity check. "
                        "Not recommended for production runs.")
    p.add_argument("--sanity_threshold_ratio", type=float, default=0.0015,
                   help="Per-transition mean flow / bbox_diagonal threshold for "
                        "calling a frame 'moved'. Default 0.0015 (ViPS arxiv "
                        "2604.17623 Section 0.F.3).")
    p.add_argument("--sanity_max_moved_fraction", type=float, default=0.10,
                   help="Pass criterion: at most this fraction of inter-frame "
                        "transitions exceed the threshold. Default 0.10 "
                        "(ViPS reports 0.71%% reject rate at these defaults).")
    p.add_argument("--no_raise_on_sanity_fail", action="store_true",
                   help="On sanity fail, write artifacts and return normally "
                        "instead of raising WanQualityError. Useful for debugging "
                        "a known-bad video against Stage B.")

    # ---- prompt preview (no Wan load) ----
    p.add_argument("--dry_run", action="store_true",
                   help="Build and print the full positive and negative prompts "
                        "without loading Wan2.2. Use this to iterate on the "
                        "--motion wording before paying for a video generation.")

    return p.parse_args()


def _validate_inputs(args: argparse.Namespace) -> None:
    """Fail fast on path / shape errors before importing torch / Wan."""
    image_path = os.path.abspath(args.image)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"--image not a file: {image_path}")
    args.image = image_path

    if not args.dry_run:
        wan_ckpt_dir = os.path.abspath(args.wan_ckpt_dir)
        if not os.path.isdir(wan_ckpt_dir):
            raise FileNotFoundError(f"--wan_ckpt not a directory: {wan_ckpt_dir}")
        args.wan_ckpt_dir = wan_ckpt_dir

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    if args.frame_num <= 1 or (args.frame_num - 1) % 4 != 0:
        raise ValueError(
            f"--frame_num must be of the form 4n+1 with n>=1; got {args.frame_num}"
        )

def _print_prompts(pos_prompt: str, neg_prompt: str, user_motion_prompt: str, lang: str) -> None:
    print("=" * 78)
    print(f"[stage_a]  lang={lang}")
    print(f"[stage_a]  user_motion : {user_motion_prompt!r}")
    print("-" * 78)
    print("[stage_a]  POSITIVE PROMPT (user_motion + universal camera-lock addon):")
    print(f"    {pos_prompt}")
    print("-" * 78)
    print("[stage_a]  NEGATIVE PROMPT (universal, omits 'static'/'motionless'):")
    print(f"    {neg_prompt}")
    print("=" * 78)


def main() -> None:
    args = parse_args()
    _validate_inputs(args)

    pos_prompt, neg_prompt = build_articulated_prompts(args.motion, lang=args.lang)
    _print_prompts(pos_prompt, neg_prompt, args.motion, args.lang)

    if args.dry_run:
        print("[stage_a]  --dry_run: prompts above; skipping Wan generation.")
        return

    image = Image.open(args.image)

    result = run_stage_a(
        image=image,
        user_motion_prompt=args.motion,
        wan_ckpt_dir=args.wan_ckpt_dir,
        out_dir=args.output_dir,
        seed=int(args.seed),
        frame_num=int(args.frame_num),
        wan_size_label=str(args.size),
        sampling_steps=int(args.sampling_steps),
        guide_scale=float(args.guide_scale),
        sample_shift=float(args.sample_shift),
        sample_solver=str(args.sample_solver),
        lang=str(args.lang),
        fps=int(args.fps),
        offload_model=not bool(args.no_offload_model),
        convert_model_dtype=not bool(args.no_convert_model_dtype),
        t5_cpu=bool(args.t5_cpu),
        device_id=int(args.device_id),
        sanity_check=not bool(args.no_sanity_check),
        sanity_threshold_ratio=float(args.sanity_threshold_ratio),
        sanity_max_moved_fraction=float(args.sanity_max_moved_fraction),
        raise_on_sanity_fail=not bool(args.no_raise_on_sanity_fail),
    )

    print("=" * 78)
    print(f"[stage_a]  Wrote {len(result.artifact_paths)} artifacts to:")
    print(f"    {args.output_dir}")
    print("-" * 78)
    print(f"  seed                 = {result.seed}")
    print(f"  frame_num            = {result.frame_num}")
    print(f"  resolution_hw        = {result.resolution_hw}")
    print(f"  sampling_steps       = {result.sampling_steps}")
    print(f"  guide_scale          = {result.guide_scale}")
    print(f"  sample_shift         = {result.sample_shift}")
    print(f"  sample_solver        = {result.sample_solver}")
    rep = result.sanity_report
    print(f"  sanity_passed        = {rep.passed}")
    print(f"  sanity_moved_frac    = {rep.moved_fraction:.4f}  "
          f"(max allowed = {rep.max_moved_fraction:.4f})")
    print(f"  sanity_threshold_px  = {rep.threshold_pixels:.2f}  "
          f"(bbox_diagonal = {rep.bbox_diagonal:.2f})")
    print("-" * 78)
    print("Artifacts:")
    for p in result.artifact_paths:
        print(f"  {p}")
    print("=" * 78)


if __name__ == "__main__":
    main()

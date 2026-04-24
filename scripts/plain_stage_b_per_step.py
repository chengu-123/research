"""Plain TRELLIS Stage-1 baseline: K-parallel sampling with shared noise,
no SCAR, no mix, no push — purely unmodified TRELLIS.

Produces per-step Tweedie decoded HTML at K=6 states with dropdown. Intended
as a baseline to compare against SCAR runs and against diagnose_trellis_steps
(which includes Option A mixing). Use this to answer "what does plain TRELLIS
look like at 25 steps, step by step".

Usage:
    conda activate mine
    python scripts/plain_stage_b_per_step.py \\
        --input_dir outputs/30857 \\
        --output_dir outputs/30857_plain_25 \\
        --K 6

Outputs under <output_dir>/per_step/:
    O_step_00.html .. O_step_24.html    (25 per-step HTMLs, one per Euler step)
    per_step_stats.json                   (voxel counts, pairwise IoU, always/ever)
Also under <output_dir>/:
    O_final.html                          (final converged K states)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MINE_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _MINE_DIR)
sys.path.insert(0, os.path.join(_MINE_DIR, "TRELLIS"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True,
                   help="Dir containing K images named by --pattern.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--K", type=int, default=6,
                   help="Number of articulation states (images).")
    p.add_argument("--pattern", default="rendering_joint_00_state_{i:02d}.png",
                   help="Filename pattern; {i} is state index.")
    p.add_argument("--seed", type=int, default=0,
                   help="Shared-noise seed.")
    p.add_argument("--steps", type=int, default=25,
                   help="Total Euler steps (TRELLIS-image-large default = 25).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load K images unchanged
    images = []
    for i in range(args.K):
        fpath = os.path.join(args.input_dir, args.pattern.format(i=i))
        images.append(Image.open(fpath).convert("RGBA"))
        print(f"[plain] loaded {fpath}")

    # Unmodified TRELLIS pipeline
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    pipe = TrellisImageTo3DPipeline.from_pretrained(
        "JeffreyXiang/TRELLIS-image-large"
    )
    pipe.cuda()

    preprocessed = [pipe.preprocess_image(img) for img in images]
    cond = pipe.get_cond(preprocessed)

    flow_model = pipe.models["sparse_structure_flow_model"]
    decoder = pipe.models["sparse_structure_decoder"]

    # Shared noise init (K tracks share the same initial noise)
    gen = torch.Generator(device=device).manual_seed(int(args.seed))
    eps = torch.randn(
        (1, 8, flow_model.resolution, flow_model.resolution, flow_model.resolution),
        device=device, generator=gen,
    )
    noise = eps.repeat(args.K, 1, 1, 1, 1)

    # Force the base sampler (in case pipe was swapped to SCAR/VGCF/BCAC elsewhere)
    sigma_min = pipe.sparse_structure_sampler.sigma_min
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=sigma_min)

    # Inherit pipeline's default params (cfg_strength, cfg_interval, rescale_t),
    # override only step count to user's choice (default 25).
    sampler_params = dict(pipe.sparse_structure_sampler_params)
    sampler_params["steps"] = int(args.steps)
    print(f"[plain] running sampler with params: {sampler_params}")

    out = sampler.sample(
        flow_model, noise,
        cond=cond["cond"], neg_cond=cond["neg_cond"],
        verbose=True,
        **sampler_params,
    )

    # Per-step Tweedie decode
    from pipelines.utils.voxel_viz import save_voxel_stack_html
    per_step_dir = os.path.join(args.output_dir, "per_step")
    os.makedirs(per_step_dir, exist_ok=True)

    t_seq = np.linspace(1.0, 0.0, args.steps + 1)
    rescale_t = float(sampler_params.get("rescale_t", 1.0))
    t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)

    pred_x0_list = list(getattr(out, "pred_x_0", []))
    per_step_stats = []

    for step_idx, x0_hat in enumerate(pred_x0_list):
        with torch.no_grad():
            logits_step = decoder(x0_hat.to(device))                 # (K, 1, 64, 64, 64)
            occ_step = (torch.sigmoid(logits_step).squeeze(1) > 0.5).float()

        occ_np = occ_step.detach().cpu().numpy().astype(np.float32)
        K_s = occ_np.shape[0]

        counts = [int(occ_np[k].sum()) for k in range(K_s)]
        ious = []
        for i in range(K_s):
            for j in range(i + 1, K_s):
                inter = ((occ_np[i] > 0.5) & (occ_np[j] > 0.5)).sum()
                union = ((occ_np[i] > 0.5) | (occ_np[j] > 0.5)).sum()
                ious.append(float(inter / max(union, 1)))

        all_occ = (occ_np > 0.5).all(axis=0)
        any_occ = (occ_np > 0.5).any(axis=0)
        always = int(all_occ.sum())
        ever = int(any_occ.sum())

        t = float(t_seq[step_idx])
        step_stat = {
            "step": step_idx,
            "t": round(t, 4),
            "voxel_counts": counts,
            "pairwise_iou_mean": round(float(np.mean(ious)) if ious else 0.0, 4),
            "pairwise_iou_min": round(float(np.min(ious)) if ious else 0.0, 4),
            "always_occupied": always,
            "ever_occupied": ever,
            "always_ever_ratio": round(float(always / max(ever, 1)), 4),
        }
        per_step_stats.append(step_stat)
        print(f"  step {step_idx:2d} t={t:.3f}  "
              f"IoU={step_stat['pairwise_iou_mean']:.3f}  "
              f"always/ever={step_stat['always_ever_ratio']:.3f}  "
              f"counts={counts}")

        save_voxel_stack_html(
            occ_np,
            os.path.join(per_step_dir, f"O_step_{step_idx:02d}.html"),
            title=f"Plain TRELLIS step {step_idx}/{args.steps - 1}  t={t:.3f}",
        )

    with open(os.path.join(per_step_dir, "per_step_stats.json"), "w") as f:
        json.dump(per_step_stats, f, indent=2)

    # Final-state visualization
    with torch.no_grad():
        logits_final = decoder(out.samples)
        occ_final = (torch.sigmoid(logits_final).squeeze(1) > 0.5).float()
    save_voxel_stack_html(
        occ_final.detach().cpu().numpy().astype(np.float32),
        os.path.join(args.output_dir, "O_final.html"),
        title=f"Plain TRELLIS final ({args.steps} steps, K={args.K})",
    )

    print(f"\n[plain] Done.")
    print(f"  Per-step HTMLs : {per_step_dir}/O_step_XX.html")
    print(f"  Per-step stats : {per_step_dir}/per_step_stats.json")
    print(f"  Final HTML     : {args.output_dir}/O_final.html")


if __name__ == "__main__":
    main()

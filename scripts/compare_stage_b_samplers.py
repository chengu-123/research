"""Compare SCAR v3, VGCF (legacy), BCAC (legacy), and plain K-parallel
sampling on a given input directory, producing a side-by-side metric table.

Usage:
    conda activate mine
    python scripts/compare_stage_b_samplers.py \\
        --input_dir outputs/30857 \\
        --output_dir outputs/30857_ablation \\
        --K 6

Outputs under `<output_dir>/`:
    scar/        # SCAR v3 result + diagnostics
    vgcf/        # legacy VGCF (unchanged)
    bcac/        # legacy BCAC (unchanged)
    plain/       # plain K-parallel (SCAR with scar_enabled=False)
    summary.json # per-sampler: pairwise_iou_mean, base_iou, icp_magnitude, wall_time
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image
import torch


def run_one_sampler(name, pipe, cond, K, out_dir):
    from pipelines.stage_b_scar import run_scar
    from pipelines.stage_b_vgcf import run_vgcf
    from trellis.pipelines.samplers import SCARSampler, VGCFSampler, BCACSampler
    from omegaconf import OmegaConf

    sigma_min = pipe.sparse_structure_sampler.sigma_min
    t0 = time.time()
    if name == "scar":
        pipe.sparse_structure_sampler = SCARSampler(sigma_min=sigma_min)
        pipe.sparse_structure_sampler_params["steps"] = 12
        res = run_scar(
            pipe=pipe, cond=cond, K=K, out_dir=out_dir,
            cfg_scar={"alpha_schedule": [0.7, 1.0, 1.0, 0.5],
                      "icp_enabled": True, "icp_max_translation": 1.5,
                      "icp_vote_threshold": 0.83},
            seed=0, remove_disk_flag=True,
        )
        O = res.O_stack
        icp_offsets = res.icp_offsets
    elif name == "plain":
        pipe.sparse_structure_sampler = SCARSampler(sigma_min=sigma_min, scar_enabled=False)
        pipe.sparse_structure_sampler_params["steps"] = 12
        res = run_scar(
            pipe=pipe, cond=cond, K=K, out_dir=out_dir,
            cfg_scar={"alpha_schedule": [0.0, 0.0, 0.0, 0.0],
                      "icp_enabled": True, "icp_max_translation": 1.5,
                      "icp_vote_threshold": 0.83},
            seed=0, remove_disk_flag=True,
        )
        O = res.O_stack
        icp_offsets = res.icp_offsets
    elif name == "vgcf":
        pipe.sparse_structure_sampler = VGCFSampler(sigma_min=sigma_min)
        vgcf_cfg = OmegaConf.create({
            "enabled": True, "steps": 12, "cfg_strength": 7.5,
            "cfg_interval": [0.0, 1.0], "rescale_t": 1.0,
            "lambda_max": 1.0, "t_stop": 0.2, "eta": 0.5, "seed": 0,
            "active_fraction": 0.8, "tau_percentile": 0.65,
            "eps_log": 1.0e-6, "lambda_schedule": "warmup",
        })
        res = run_vgcf(pipe=pipe, cond=cond, K=K, cfg_vgcf=vgcf_cfg, out_dir=out_dir)
        O = res.O_stack
        icp_offsets = [np.zeros(3)] * K
    elif name == "bcac":
        pipe.sparse_structure_sampler = BCACSampler(sigma_min=sigma_min)
        vgcf_cfg = OmegaConf.create({
            "enabled": True, "steps": 12, "cfg_strength": 7.5,
            "cfg_interval": [0.0, 1.0], "rescale_t": 1.0,
            "lambda_max": 1.0, "t_stop": 0.2, "eta": 0.5, "seed": 0,
            "active_fraction": 0.8, "tau_percentile": 0.65,
            "eps_log": 1.0e-6, "lambda_schedule": "warmup",
        })
        res = run_vgcf(pipe=pipe, cond=cond, K=K, cfg_vgcf=vgcf_cfg, out_dir=out_dir)
        O = res.O_stack
        icp_offsets = [np.zeros(3)] * K
    else:
        raise ValueError(f"unknown sampler: {name}")
    wall = time.time() - t0

    O_np = O.detach().cpu().numpy().astype(bool)

    ious = []
    for i in range(K):
        for j in range(i + 1, K):
            inter = (O_np[i] & O_np[j]).sum()
            union = (O_np[i] | O_np[j]).sum()
            ious.append(float(inter / max(union, 1)))
    pairwise_iou_mean = float(np.mean(ious))

    votes = O_np.sum(axis=0)
    base_mask = votes >= max(5, int(0.83 * K))
    if base_mask.sum() > 0:
        base_ious = [float((O_np[k] & base_mask).sum() / max(base_mask.sum(), 1)) for k in range(K)]
        base_iou_mean = float(np.mean(base_ious))
    else:
        base_iou_mean = 0.0

    icp_magnitudes = [float(np.linalg.norm(o)) for o in icp_offsets]

    return {
        "name": name,
        "pairwise_iou_mean": pairwise_iou_mean,
        "base_iou_mean": base_iou_mean,
        "icp_magnitude_mean": float(np.mean(icp_magnitudes)),
        "icp_magnitude_max": float(np.max(icp_magnitudes)),
        "wall_time_sec": wall,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--samplers", nargs="+",
                   default=["scar", "plain", "vgcf", "bcac"])
    p.add_argument("--pattern", default="rendering_joint_00_state_{i:02d}.png")
    args = p.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "TRELLIS"))

    from trellis.pipelines import TrellisImageTo3DPipeline

    os.makedirs(args.output_dir, exist_ok=True)

    images = [
        Image.open(os.path.join(args.input_dir, args.pattern.format(i=i))).convert("RGBA")
        for i in range(args.K)
    ]
    pipe = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipe.cuda()
    preprocessed = [pipe.preprocess_image(im) for im in images]
    cond = pipe.get_cond(preprocessed)

    summary = []
    for name in args.samplers:
        out = os.path.join(args.output_dir, name)
        row = run_one_sampler(name, pipe, cond, args.K, out)
        print(f"[{name}] pairwise_iou={row['pairwise_iou_mean']:.3f}  "
              f"base_iou={row['base_iou_mean']:.3f}  "
              f"icp_mag_mean={row['icp_magnitude_mean']:.3f}  "
              f"wall={row['wall_time_sec']:.1f}s")
        summary.append(row)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()

"""Diagnose TRELLIS Stage-1: per-step Tweedie decode + cross-state consistency.

Uses the UNMODIFIED TRELLIS pipeline (shared noise, standard sampler,
NO VGCF, NO BCAC) to show how 6 states diverge during the 12-step
Euler flow from t=1 to t=0.

Outputs under <output_dir>/trellis_step_diag/:
  per_step_stats.json     — per-step pairwise IoU, always/ever ratio, latent var
  O_step_XX.html          — interactive voxel viewer (dropdown over 6 states)
  per_step_curves.html    — IoU / consistency / variance curves over steps

Usage:
  conda activate mine
  python diagnose_trellis_steps.py \
    --input_dir outputs/outputs/30857 \
    --output_dir outputs/30857_step_diag \
    --K 6
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "TRELLIS"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True,
                   help="Dir with rendering_joint_00_state_{i:02d}.png (or {i:02d}_seg.png)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--pattern", default=None,
                   help="Image filename pattern. Auto-detected if not set.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg_strength", type=float, default=7.5)
    p.add_argument("--cfg_interval", nargs=2, type=float, default=[0.0, 1.0])
    p.add_argument("--steps", type=int, default=25,
                   help="Total Euler steps. TRELLIS-image-large default is 25 (also "
                        "what FreeArt3D uses); methods.md erroneously said 12.")
    p.add_argument("--mix_steps", type=int, default=8,
                   help="Blend state tokens for the first N Euler steps (Option A: "
                        "advance the mixed trajectory). 0 disables. For --steps=25, "
                        "8 covers t in [0.72, 1.0] (shape-formation window).")
    p.add_argument("--mix_weights", nargs=3, type=float, default=[0.3, 0.4, 0.3],
                   help="(w_first, w_self, w_last). Sum should be 1.0.")
    return p.parse_args()


def detect_pattern(input_dir, K):
    for pat in [
        "rendering_joint_00_state_{i:02d}.png",
        "{i:02d}_seg.png",
        "{i:02d}_pure.png",
    ]:
        if os.path.isfile(os.path.join(input_dir, pat.format(i=0))):
            return pat
    raise FileNotFoundError(f"Cannot detect image pattern in {input_dir}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pattern = args.pattern or detect_pattern(args.input_dir, args.K)
    print(f"[diag] pattern: {pattern}, K={args.K}, device={device}")

    # Load images
    images = []
    for i in range(args.K):
        fpath = os.path.join(args.input_dir, pattern.format(i=i))
        images.append(Image.open(fpath).convert("RGBA"))
        print(f"  loaded {fpath}")

    # Build TRELLIS pipeline (UNMODIFIED)
    from trellis.pipelines import TrellisImageTo3DPipeline
    pipe = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipe.cuda()

    # Encode conditions
    preprocessed = [pipe.preprocess_image(img) for img in images]
    cond = pipe.get_cond(preprocessed)

    flow_model = pipe.models["sparse_structure_flow_model"]
    decoder = pipe.models["sparse_structure_decoder"]

    # Shared noise
    gen = torch.Generator(device=device).manual_seed(args.seed)
    eps = torch.randn((1, 8, flow_model.resolution, flow_model.resolution, flow_model.resolution),
                      device=device, generator=gen)
    noise = eps.repeat(args.K, 1, 1, 1, 1)

    # ---- Manual Euler loop (replicating FlowEulerGuidanceIntervalSampler) ----
    # We do this manually so we can intercept pred_x_0 at every step.
    from trellis.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler

    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=pipe.sparse_structure_sampler.sigma_min)
    sample = noise.clone()
    t_seq = np.linspace(1.0, 0.0, args.steps + 1)
    t_pairs = list(zip(t_seq[:-1], t_seq[1:]))

    per_step_stats = []
    per_step_occ = []   # keep voxels in memory for HTML; we don't persist npy anymore
    step_dir = args.output_dir
    os.makedirs(step_dir, exist_ok=True)

    w_first, w_self, w_last = args.mix_weights
    print(f"\n[diag] Running {args.steps}-step Euler with cfg_strength={args.cfg_strength}, "
          f"cfg_interval={args.cfg_interval}")
    if args.mix_steps > 0:
        print(f"[diag] Option-A state-token mixing: first {args.mix_steps} steps use "
              f"x_in[k] = {w_first}*x[0] + {w_self}*x[k] + {w_last}*x[K-1]; "
              f"mixed trajectory is propagated (sample <- x_in - dt*v).")

    for step_idx, (t, t_prev) in enumerate(t_pairs):
        t = float(t)
        t_prev = float(t_prev)
        dt = t - t_prev

        mixed = step_idx < args.mix_steps
        if mixed:
            # Blend state tokens: anchor each state with the two extremes (state 0 and state K-1).
            x_in = w_first * sample[0:1] + w_self * sample + w_last * sample[-1:]
        else:
            x_in = sample

        with torch.no_grad():
            # Standard CFG + guidance interval forward (uses x_in; CFG mixin passes it to
            # both cond and neg_cond forwards unchanged).
            out = sampler.sample_once(
                flow_model, x_in, t, t_prev,
                cond=cond["cond"],
                neg_cond=cond["neg_cond"],
                cfg_strength=args.cfg_strength,
                cfg_interval=tuple(args.cfg_interval),
            )
            pred_x_prev = out.pred_x_prev   # (K, 8, D, H, W)  == x_in - dt * v
            pred_x_0 = out.pred_x_0         # Tweedie clean estimate (from x_in)

            # Decode Tweedie estimate through VAE → 64³
            logits = decoder(pred_x_0)                          # (K, 1, 64, 64, 64)
            occ = (torch.sigmoid(logits).squeeze(1) > 0.5).float()  # (K, 64, 64, 64)

        occ_np = occ.detach().cpu().numpy()
        K_s = occ_np.shape[0]

        # ---- Statistics ----
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

        voxel_std = occ_np.std(axis=0)

        # Latent-level variance
        x0_np = pred_x_0.detach().cpu().float().numpy()
        latent_var = x0_np.var(axis=0).sum(axis=0)  # (D, H, W) in 16³

        stat = {
            "step": step_idx,
            "t": round(t, 4),
            "mixed_input": mixed,
            "voxel_counts": counts,
            "pairwise_iou_mean": round(float(np.mean(ious)), 4) if ious else 0.0,
            "pairwise_iou_min": round(float(np.min(ious)), 4) if ious else 0.0,
            "pairwise_iou_max": round(float(np.max(ious)), 4) if ious else 0.0,
            "always_occupied": always,
            "ever_occupied": ever,
            "always_ever_ratio": round(float(always / max(ever, 1)), 4),
            "voxel_std_mean": round(float(voxel_std.mean()), 6),
            "latent_var_mean": round(float(latent_var.mean()), 6),
            "latent_var_max": round(float(latent_var.max()), 4),
        }
        per_step_stats.append(stat)

        tag = "[mix]" if mixed else "     "
        print(f"  step {step_idx:2d} {tag} t={t:.3f}  "
              f"IoU={stat['pairwise_iou_mean']:.3f} (min {stat['pairwise_iou_min']:.3f})  "
              f"always/ever={stat['always_ever_ratio']:.3f}  "
              f"latent_var_mean={stat['latent_var_mean']:.4f}")

        # Keep voxels in memory (uint8, ~6*64³ = 1.5MB/step — trivial)
        per_step_occ.append(occ_np.astype(np.uint8))

        # Take the Euler step
        sample = pred_x_prev

    # ---- Persist stats ----
    with open(os.path.join(step_dir, "per_step_stats.json"), "w") as f:
        json.dump(per_step_stats, f, indent=2)

    # ---- HTML visualizations ----
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Per-step voxel HTMLs (dropdown over K states)
        for step_idx in range(len(per_step_stats)):
            occ_step = per_step_occ[step_idx].astype(float)
            K_s = occ_step.shape[0]
            traces = []
            for k in range(K_s):
                x, y, z = np.where(occ_step[k] > 0.5)
                tr = go.Scatter3d(x=x, y=y, z=z, mode="markers",
                                  marker=dict(size=1.8, color="rgba(255,122,0,1.0)"),
                                  name=f"state {k}", visible=(k == 0))
                traces.append(tr)
            buttons = [dict(label=f"state {k}", method="update",
                            args=[{"visible": [i == k for i in range(K_s)]},
                                  {"title": f"Step {step_idx} (t={per_step_stats[step_idx]['t']:.2f}) — state {k}"}])
                       for k in range(K_s)]
            fig = go.Figure(data=traces)
            fig.update_layout(
                title=f"Step {step_idx} (t={per_step_stats[step_idx]['t']:.2f}) — state 0",
                scene=dict(xaxis=dict(range=[0, 63]), yaxis=dict(range=[0, 63]),
                           zaxis=dict(range=[0, 63]), aspectmode="cube"),
                updatemenus=[dict(type="dropdown", showactive=True, buttons=buttons,
                                  x=1.02, y=1.0, xanchor="left")],
            )
            fig.write_html(os.path.join(step_dir, f"O_step_{step_idx:02d}.html"),
                           include_plotlyjs="cdn")

        # Summary curves
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=("Pairwise IoU (6 states)", "Always/Ever Ratio",
                                            "16³ Latent Variance"))
        steps = [s["step"] for s in per_step_stats]
        fig.add_trace(go.Scatter(x=steps, y=[s["pairwise_iou_mean"] for s in per_step_stats],
                                 mode="lines+markers", name="IoU mean"), row=1, col=1)
        fig.add_trace(go.Scatter(x=steps, y=[s["pairwise_iou_min"] for s in per_step_stats],
                                 mode="lines", name="IoU min", line=dict(dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=steps, y=[s["pairwise_iou_max"] for s in per_step_stats],
                                 mode="lines", name="IoU max", line=dict(dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=steps, y=[s["always_ever_ratio"] for s in per_step_stats],
                                 mode="lines+markers", name="always/ever"), row=2, col=1)
        fig.add_trace(go.Scatter(x=steps, y=[s["latent_var_mean"] for s in per_step_stats],
                                 mode="lines+markers", name="var mean"), row=3, col=1)
        fig.add_trace(go.Scatter(x=steps, y=[s["latent_var_max"] for s in per_step_stats],
                                 mode="lines", name="var max", line=dict(dash="dot")), row=3, col=1)
        fig.update_layout(title="TRELLIS Stage-1 (unmodified) — cross-state consistency over steps",
                          height=800)
        fig.update_xaxes(title_text="Euler step", row=3, col=1)
        fig.write_html(os.path.join(step_dir, "per_step_curves.html"), include_plotlyjs="cdn")
        print(f"\n[diag] HTML curves saved to {step_dir}/per_step_curves.html")

    except ImportError:
        print("[diag] plotly not available, skipping HTML generation")

    print(f"\n[diag] Done. Results in {step_dir}/")
    print(f"  per_step_stats.json  — numbers")
    print(f"  O_step_XX.html       — per-step voxel viewer (dropdown over 6 states)")
    print(f"  per_step_curves.html — IoU / consistency curves")


if __name__ == "__main__":
    main()

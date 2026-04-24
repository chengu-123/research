"""Generate interactive HTML visualizations of Stage C v8.1 final outputs + DiT priors.

Produces one combined HTML per object under scripts/viz_out/{obj}/ showing:
  * canonical_base  (green dots)
  * canonical_move  (red dots)
  * per_state_assignment (6 panels, each state's move/base colored)
  * DiT p_move      (heatmap over active voxels)
  * DiT s_boundary  (heatmap over active voxels)
  * fitted joint axis (arrow/line through canonical frame)
  * GT joint axis (arrow/line, different color, for comparison)

Run:
    KMP_DUPLICATE_LIB_OK=TRUE /d/anaconda3/envs/library/python.exe scripts/viz_final.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as ps
import torch

ROOT = Path(r"C:/Users/管晨皓/Desktop/temp/standard/mine")
STAGE_C_DIR = ROOT / "outputs" / "experiment_c_v8_dit"
STAGE_B_DIR = ROOT / "outputs" / "experiment_b_v8_hook"
REF_DIR = ROOT / "outputs" / "reference"
OUT_DIR = ROOT / "scripts" / "viz_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OBJECTS = ["30857", "7201", "7128", "26525"]
RES = 64


def _voxel_coords(mask: np.ndarray) -> np.ndarray:
    """Return (N, 3) voxel indices where mask is True."""
    if mask.dtype != bool:
        mask = mask.astype(bool)
    return np.argwhere(mask)


def _to_scatter(coords: np.ndarray, color: str, name: str, size: int = 2, opacity: float = 0.6):
    if len(coords) == 0:
        return go.Scatter3d(x=[], y=[], z=[], mode="markers", name=name + " (empty)")
    return go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers",
        marker=dict(size=size, color=color, opacity=opacity),
        name=name,
    )


def _axis_line_scatter(omega, q, phi_range, n_samples=30, color="blue", name="axis"):
    """Draw the joint axis as a line segment through q in direction omega, spanning phi_range."""
    omega = np.asarray(omega).reshape(3)
    q = np.asarray(q).reshape(3)
    omega = omega / max(np.linalg.norm(omega), 1e-8)
    ts = np.linspace(-phi_range * 0.5, phi_range * 0.5, n_samples)
    # Place axis in voxel coord: map world-[-0.5, 0.5] to [0, RES-1]
    pts = q[None, :] + ts[:, None] * omega[None, :]
    vx = (pts[:, 0] + 0.5) * (RES - 1)
    vy = (pts[:, 1] + 0.5) * (RES - 1)
    vz = (pts[:, 2] + 0.5) * (RES - 1)
    return go.Scatter3d(
        x=vx, y=vy, z=vz, mode="lines",
        line=dict(color=color, width=8),
        name=name,
    )


def viz_object(obj: str):
    stage_c = STAGE_C_DIR / obj / "stage_c"
    if not stage_c.exists():
        print(f"[viz] skip {obj}: stage_c not found")
        return

    # Load pipeline outputs
    cb = np.load(stage_c / "canonical_base.npy").astype(bool)
    cm = np.load(stage_c / "canonical_move.npy").astype(bool)
    psa = np.load(stage_c / "per_state_assignment.npy")
    jp = torch.load(stage_c / "joint_params.pt", weights_only=False)
    try:
        meta = json.load(open(stage_c / "meta.json"))
    except Exception:
        meta = {}

    # Load GT joint info
    gt_info = json.load(open(REF_DIR / obj / "joint_info.json"))[0]

    # Load DiT priors (recompute from Stage B hidden + partition on-the-fly)
    import sys
    sys.path.insert(0, str(ROOT))
    from pipelines.stage_c_segmatch.features import load_O_stack, load_dit_hidden
    from pipelines.stage_c_segmatch.dit_prior import compute_dit_priors
    from pipelines.stage_c_segmatch.partition import count_based_partition

    O = load_O_stack(str(STAGE_B_DIR / f"{obj}_b" / "stage_b"), device=torch.device("cpu"), dtype=torch.float32)
    count = (O > 0.5).to(torch.int32).sum(dim=0)
    shell_t = (count > 0) & (count < O.shape[0])
    always_on_t = count == O.shape[0]
    footprint_t = count > 0

    dit_hidden = load_dit_hidden(str(STAGE_B_DIR / f"{obj}_b" / "stage_b"),
                                  device=torch.device("cpu"), dtype=torch.float32)
    if dit_hidden is not None:
        dit = compute_dit_priors(
            dit_hidden, shell_t, always_on_t, footprint_t,
            target_blocks=None, d_latent=16, out_size=RES,
        )
        p_move = dit.p_move.numpy()                # (64, 64, 64)
        s_boundary = dit.s_boundary.numpy()
    else:
        p_move = np.zeros((RES, RES, RES))
        s_boundary = np.zeros((RES, RES, RES))
        dit = None

    footprint = footprint_t.numpy().astype(bool)
    K = int(psa.shape[0])

    # ---------- Panel 1: canonical base + move ----------
    fig1 = go.Figure()
    fig1.add_trace(_to_scatter(_voxel_coords(cb), "green", "canonical_base", size=2, opacity=0.25))
    fig1.add_trace(_to_scatter(_voxel_coords(cm), "red", "canonical_move", size=2, opacity=0.9))

    # Overlay fitted axis (red) + GT axis (blue, indicative — frame may differ)
    jt = jp["joint_type"]
    omega = jp["omega"].cpu().numpy()
    q = jp["q"].cpu().numpy()
    phi_max = float(jp["phi_k"][-1].item())
    if jt == "revolute":
        axis_extent = 0.5
    else:
        axis_extent = max(phi_max * 2, 0.3)
    fig1.add_trace(_axis_line_scatter(
        omega, q, axis_extent, color="red", name=f"fit {jt} axis (phi_5={phi_max:.3f})",
    ))
    gt_dir = np.asarray(gt_info["axis"]["direction"])
    gt_orig = np.asarray(gt_info["axis"]["origin"])
    fig1.add_trace(_axis_line_scatter(
        gt_dir, gt_orig, 0.5, color="royalblue",
        name=f"GT {gt_info['type']} axis (dir={gt_dir.tolist()})",
    ))
    fig1.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="cube"),
        title=f"{obj} canonical (GT={gt_info['type']}, pred={jt}, phi_5={phi_max:.3f})",
        width=900, height=700,
    )
    fig1.write_html(OUT_DIR / f"{obj}_canonical.html")
    print(f"  wrote {OUT_DIR}/{obj}_canonical.html")

    # ---------- Panel 2: per-state assignment grid (6 subplots) ----------
    specs = [[{"type": "scene"}] * 3, [{"type": "scene"}] * 3]
    fig2 = ps.make_subplots(rows=2, cols=3, specs=specs,
                             subplot_titles=[f"state {k}" for k in range(K)])
    for k in range(K):
        r, c = k // 3 + 1, k % 3 + 1
        move_coords = _voxel_coords(psa[k] == 1)
        base_coords = _voxel_coords(psa[k] == 0)
        fig2.add_trace(_to_scatter(base_coords, "gray", f"base[{k}]", size=1.5, opacity=0.15),
                       row=r, col=c)
        fig2.add_trace(_to_scatter(move_coords, "red", f"move[{k}]", size=2, opacity=0.9),
                       row=r, col=c)
    fig2.update_layout(
        title=f"{obj} per_state_assignment (move red, base gray)",
        width=1500, height=900, showlegend=False,
    )
    fig2.write_html(OUT_DIR / f"{obj}_per_state.html")
    print(f"  wrote {OUT_DIR}/{obj}_per_state.html")

    # ---------- Panel 3: DiT priors ----------
    if dit is not None:
        fig3 = ps.make_subplots(rows=1, cols=2,
                                  specs=[[{"type": "scene"}, {"type": "scene"}]],
                                  subplot_titles=["p_move_dit (sigmoid score)", "s_boundary (normalized cross-seed std)"])

        # p_move on footprint voxels, colored by value
        foot_coords = _voxel_coords(footprint)
        if len(foot_coords) > 0:
            vals_pm = p_move[foot_coords[:, 0], foot_coords[:, 1], foot_coords[:, 2]]
            fig3.add_trace(go.Scatter3d(
                x=foot_coords[:, 0], y=foot_coords[:, 1], z=foot_coords[:, 2],
                mode="markers",
                marker=dict(size=2.5, color=vals_pm, cmin=0, cmax=1,
                            colorscale="RdYlGn_r", opacity=0.7,
                            colorbar=dict(x=0.45, title="p_move")),
                name="p_move_dit",
            ), row=1, col=1)

            vals_sb = s_boundary[foot_coords[:, 0], foot_coords[:, 1], foot_coords[:, 2]]
            fig3.add_trace(go.Scatter3d(
                x=foot_coords[:, 0], y=foot_coords[:, 1], z=foot_coords[:, 2],
                mode="markers",
                marker=dict(size=2.5, color=vals_sb, cmin=0, cmax=1,
                            colorscale="Viridis", opacity=0.7,
                            colorbar=dict(x=1.0, title="s_boundary")),
                name="s_boundary",
            ), row=1, col=2)
        fig3.update_layout(
            title=f"{obj} DiT 1024-dim priors (v8.1 AAAI novelty)",
            width=1600, height=700,
        )
        fig3.write_html(OUT_DIR / f"{obj}_dit_prior.html")
        print(f"  wrote {OUT_DIR}/{obj}_dit_prior.html")

    # ---------- Panel 4: summary numbers ----------
    summary = {
        "object": obj,
        "gt_type": gt_info["type"],
        "gt_range": float(gt_info["range"][1] - gt_info["range"][0]),
        "gt_axis_dir": gt_info["axis"]["direction"],
        "gt_axis_origin": gt_info["axis"]["origin"],
        "pred_type": jt,
        "pred_phi_5": phi_max,
        "pred_omega": omega.tolist(),
        "pred_q": q.tolist(),
        "canonical_move_n": int(cm.sum()),
        "canonical_base_n": int(cb.sum()),
        "per_state_move_n": [int((psa[k] == 1).sum()) for k in range(K)],
        "per_state_base_n": [int((psa[k] == 0).sum()) for k in range(K)],
        "dit_prior_used": dit is not None,
        "dit_prior_meta": dit.meta if dit is not None else None,
    }
    with open(OUT_DIR / f"{obj}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {OUT_DIR}/{obj}_summary.json")


if __name__ == "__main__":
    for obj in OBJECTS:
        print(f"=== {obj} ===")
        viz_object(obj)
    print("\nAll done. HTMLs under:", OUT_DIR)

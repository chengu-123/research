"""Per-stage HTML visualizations for Stage C SegMatch v5.

Reuses shared helpers from ``pipelines/utils/voxel_viz.py`` where
possible (voxel stacks, soft fields, mask overlay, axis overlay, EM
traces) and adds v5-specific viz:

* **C.1 footprint** — base/move initial + footprint overlay + centroid
* **C.2 moment trajectory** — K centroid points + fitted arc/line
* **C.4 volumetric loss** — rev/pris L-traces + BIC bar
* **C.5 motion-consistency data term** — soft field
* **C.7 overlap cleanup** — before / after / contact / deleted overlay
* **C.8 trajectory** — canonical_move through K states + fitted axis

All writers silently no-op when plotly is missing.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..utils.voxel_viz import (
    _PLOTLY_OK,
    _scene,
    _write,
    save_axis_overlay_html,
    save_em_traces_html,
    save_soft_voxel_html,
    save_two_masks_html,
    save_voxel_html,
    save_voxel_stack_html,
)

if _PLOTLY_OK:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots


# ---- Helpers ---------------------------------------------------------


def _np(t):
    if t is None:
        return None
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _world_to_voxel(pts: np.ndarray, resolution: int = 64) -> np.ndarray:
    return (pts + 0.5) * (resolution - 1)


def _apply_T_to_points(pts_world: np.ndarray, T: np.ndarray) -> np.ndarray:
    h = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1))], axis=-1)
    return (T @ h.T).T[:, :3]


def _voxel_to_world(vox: np.ndarray, resolution: int = 64) -> np.ndarray:
    return vox.astype(np.float64) / float(resolution - 1) - 0.5


# ---- C.2 moment trajectory --------------------------------------------


def save_centroid_trajectory_html(
    centroids_world: np.ndarray,
    warm_joint_hint: str,
    axis_world: np.ndarray,
    q_world: np.ndarray,
    phi_k: np.ndarray,
    out_path: str,
    resolution: int = 64,
    title: str = "C.2 centroid trajectory + warm-start fit",
) -> None:
    """K per-state move centroids + the chosen warm-start curve (prismatic
    line or revolute arc)."""
    if not _PLOTLY_OK:
        return
    R = int(resolution)
    K = centroids_world.shape[0]
    centroids_idx = _world_to_voxel(centroids_world, R)

    traces: List[Any] = [go.Scatter3d(
        x=centroids_idx[:, 0],
        y=centroids_idx[:, 1],
        z=centroids_idx[:, 2],
        mode="markers+text",
        marker=dict(size=8, color="rgba(255,80,80,0.95)"),
        text=[f"state {k}" for k in range(K)],
        textposition="top center",
        name="move centroid",
    )]

    axis_np = np.asarray(axis_world, dtype=np.float64)
    norm = np.linalg.norm(axis_np)
    if norm > 1e-8:
        axis_np = axis_np / norm
    q_idx = _world_to_voxel(np.asarray(q_world, dtype=np.float64), R)

    if warm_joint_hint == "revolute":
        # Draw the fitted axis
        ts = np.linspace(-R, R, 200)
        line_pts = q_idx[None, :] + ts[:, None] * axis_np[None, :] * (R - 1) / R
        in_range = np.all((line_pts >= 0) & (line_pts <= R - 1), axis=1)
        line_pts = line_pts[in_range]
        if line_pts.shape[0] >= 2:
            traces.append(go.Scatter3d(
                x=line_pts[:, 0], y=line_pts[:, 1], z=line_pts[:, 2],
                mode="lines",
                line=dict(color="rgb(30,180,30)", width=5, dash="dash"),
                name="fitted rev axis",
            ))
        # Connect the centroids in order
        traces.append(go.Scatter3d(
            x=centroids_idx[:, 0], y=centroids_idx[:, 1], z=centroids_idx[:, 2],
            mode="lines",
            line=dict(color="rgba(255,80,80,0.5)", width=3),
            name="centroid path",
        ))
    else:
        # Prismatic: line from centroid_0 along axis
        t_range = max(phi_k.max() if phi_k.size > 0 else 0.0, 0.5)
        ts = np.linspace(-t_range, t_range, 50)
        line_pts = centroids_idx[0:1] + ts[:, None] * axis_np[None, :] * (R - 1)
        traces.append(go.Scatter3d(
            x=line_pts[:, 0], y=line_pts[:, 1], z=line_pts[:, 2],
            mode="lines",
            line=dict(color="rgb(30,180,30)", width=5, dash="dash"),
            name="fitted pris line",
        ))
        traces.append(go.Scatter3d(
            x=centroids_idx[:, 0], y=centroids_idx[:, 1], z=centroids_idx[:, 2],
            mode="lines",
            line=dict(color="rgba(255,80,80,0.5)", width=3),
            name="centroid path",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{title} — hint: {warm_joint_hint}",
        scene=_scene(R),
    )
    _write(fig, out_path, png=False)


# ---- C.4 volumetric loss -----------------------------------------------


def save_volumetric_loss_html(
    rev_trace: List[float],
    pris_trace: List[float],
    bic_rev: float,
    bic_pris: float,
    selected: str,
    out_path: str,
    title: str = "C.4 joint-constrained volumetric Adam",
) -> None:
    if not _PLOTLY_OK:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("volumetric variance loss", "BIC"),
        specs=[[{"type": "xy"}, {"type": "xy"}]],
    )
    if rev_trace:
        fig.add_trace(go.Scatter(
            y=list(rev_trace), mode="lines",
            name="revolute", line=dict(color="rgb(255,122,0)")
        ), row=1, col=1)
    if pris_trace:
        fig.add_trace(go.Scatter(
            y=list(pris_trace), mode="lines",
            name="prismatic", line=dict(color="rgb(0,122,255)")
        ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=["revolute", "prismatic"],
        y=[float(bic_rev), float(bic_pris)],
        marker=dict(color=["rgb(255,122,0)", "rgb(0,122,255)"]),
        showlegend=False,
    ), row=1, col=2)
    fig.update_xaxes(title_text="Adam step", row=1, col=1)
    fig.update_yaxes(title_text="variance sum", row=1, col=1)
    fig.update_yaxes(title_text="BIC", row=1, col=2)
    fig.update_layout(
        title=f"{title} — selected: {selected}",
        height=420,
    )
    _write(fig, out_path, png=False)


# ---- C.5 data term overlay --------------------------------------------


def save_data_term_html(
    data_term: np.ndarray,
    M_attn_64: np.ndarray,
    out_path: str,
    title: str = "C.5 motion-consistency data term",
    resolution: int = 64,
    threshold: float = 0.1,
) -> None:
    """data_term (high = likely move under rigid hypothesis) overlaid on
    M_attn_64 (high = likely base from diffusion). Ideal: the two are
    complementary."""
    if not _PLOTLY_OK:
        return
    R = int(resolution)
    xs, ys, zs = np.where((data_term > threshold) | (M_attn_64 > threshold))
    vals_mot = data_term[xs, ys, zs]
    vals_attn = M_attn_64[xs, ys, zs]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=("motion_consistency(v)", "M_attn_64(v)"),
    )
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="markers",
        marker=dict(size=2.0, color=vals_mot, colorscale="Viridis",
                    cmin=0.0, cmax=1.0, colorbar=dict(title="mot",
                                                       x=0.45)),
        name="motion",
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="markers",
        marker=dict(size=2.0, color=vals_attn, colorscale="Plasma",
                    cmin=0.0, cmax=1.0, colorbar=dict(title="attn",
                                                       x=1.02)),
        name="attn",
        showlegend=False,
    ), row=1, col=2)
    fig.update_layout(title=title,
                       scene=_scene(R), scene2=_scene(R),
                       height=500)
    _write(fig, out_path, png=False)


# ---- C.7 overlap cleanup ---------------------------------------------


def save_overlap_cleanup_html(
    canonical_base: np.ndarray,
    canonical_move_before: np.ndarray,
    canonical_move_after: np.ndarray,
    contact_region: np.ndarray,
    overlap_before: np.ndarray,
    out_path: str,
    resolution: int = 64,
    title: str = "C.7 canonical base-base overlap cleanup",
) -> None:
    """Dropdown view with 4 states: before (base+move with overlap in red),
    after (clean), contact band (green), and the deleted 'containment'
    voxels alone."""
    if not _PLOTLY_OK:
        return
    deleted = canonical_move_before & (~canonical_move_after)

    R = int(resolution)
    # Trace groups
    def _scatter(binary, color, name, visible=False, size=1.8):
        x, y, z = np.where(binary > 0.5)
        return go.Scatter3d(
            x=x, y=y, z=z, mode="markers",
            marker=dict(size=size, color=color),
            name=name, visible=visible,
        )

    traces = []
    # View 0: "before" — base + move + overlap highlighted
    view0 = [
        _scatter(canonical_base, "rgba(255,122,0,0.55)", "base", visible=True),
        _scatter(canonical_move_before, "rgba(0,122,255,0.55)", "move_before",
                 visible=True),
        _scatter(overlap_before, "rgba(255,30,30,0.9)", "overlap_before",
                 visible=True, size=2.4),
    ]
    # View 1: "after" — base + cleaned move + contact
    view1 = [
        _scatter(canonical_base, "rgba(255,122,0,0.55)", "base"),
        _scatter(canonical_move_after, "rgba(0,122,255,0.55)", "move_after"),
        _scatter(contact_region, "rgba(40,200,40,0.9)", "contact_region",
                 size=2.2),
    ]
    # View 2: "deleted" — only the containment voxels
    view2 = [
        _scatter(canonical_base, "rgba(200,200,200,0.25)", "base_bg"),
        _scatter(deleted, "rgba(255,30,30,0.95)", "deleted (containment)",
                 size=2.6),
    ]

    views = [view0, view1, view2]
    view_names = ["before (overlap in red)", "after cleanup (contact green)",
                  "deleted containment voxels"]
    buttons = []
    start = 0
    for i, view in enumerate(views):
        visibility = [False] * sum(len(v) for v in views)
        for j, trace in enumerate(view):
            trace.visible = (i == 0)  # only view 0 initially
            traces.append(trace)
            visibility[start + j] = True
        buttons.append(dict(
            label=view_names[i], method="update",
            args=[{"visible": []}, {"title": f"{title} — {view_names[i]}"}],
        ))
        start += len(view)
    # Fix the visibility arrays for each button
    trace_count = len(traces)
    for i, btn in enumerate(buttons):
        vis = [False] * trace_count
        start_i = sum(len(v) for v in views[:i])
        end_i = start_i + len(views[i])
        for t in range(start_i, end_i):
            vis[t] = True
        btn["args"][0]["visible"] = vis

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{title} — {view_names[0]}",
        scene=_scene(R),
        updatemenus=[dict(type="dropdown", showactive=True, buttons=buttons,
                          x=1.02, y=1.0, xanchor="left")],
    )
    _write(fig, out_path, png=False)


# ---- C.8 trajectory ---------------------------------------------------


def save_trajectory_html(
    canonical_move: np.ndarray,
    canonical_base: np.ndarray,
    T_k: np.ndarray,
    joint_type: str,
    omega_world: np.ndarray,
    q_world: np.ndarray,
    out_path: str,
    title: str = "C.8 trajectory — T_k(canonical_move)",
    resolution: int = 64,
) -> None:
    """Same as the old v3 trajectory viz: base grey, axis red, per-state
    move with dropdown, plus an 'all states' overlay."""
    if not _PLOTLY_OK:
        return
    R = int(resolution)
    K = T_k.shape[0]

    xb, yb, zb = np.where(canonical_base > 0.5)
    base_trace = go.Scatter3d(
        x=xb, y=yb, z=zb, mode="markers",
        marker=dict(size=1.4, color="rgba(160,160,160,0.35)"),
        name="canonical_base", visible=True,
    )
    traces = [base_trace]

    q_idx = _world_to_voxel(np.asarray(q_world, dtype=np.float64), R)
    omega_np = np.asarray(omega_world, dtype=np.float64)
    norm = np.linalg.norm(omega_np)
    if norm > 1e-8:
        omega_np = omega_np / norm
    if joint_type == "revolute":
        ts = np.linspace(-R, R, 200)
        line_pts = q_idx[None, :] + ts[:, None] * omega_np[None, :] * (R - 1) / R
        in_range = np.all((line_pts >= 0) & (line_pts <= R - 1), axis=1)
        line_pts = line_pts[in_range]
        if line_pts.shape[0] >= 2:
            traces.append(go.Scatter3d(
                x=line_pts[:, 0], y=line_pts[:, 1], z=line_pts[:, 2],
                mode="lines",
                line=dict(color="rgb(255,30,30)", width=6),
                name="axis", visible=True,
            ))
    else:
        xs, ys, zs = np.where(canonical_move > 0.5)
        if xs.size > 0:
            cx, cy, cz = float(xs.mean()), float(ys.mean()), float(zs.mean())
        else:
            cx = cy = cz = R / 2.0
        tip = np.array([cx, cy, cz]) + omega_np * (R / 3.0)
        traces.append(go.Scatter3d(
            x=[cx, tip[0]], y=[cy, tip[1]], z=[cz, tip[2]],
            mode="lines+markers",
            line=dict(color="rgb(255,30,30)", width=8),
            marker=dict(size=[3, 8], color="rgb(255,30,30)"),
            name="v_hat", visible=True,
        ))

    move_vox = np.argwhere(canonical_move > 0.5).astype(np.float64)
    move_world = _voxel_to_world(move_vox, R) if move_vox.shape[0] > 0 else move_vox

    state_trace_start = len(traces)
    palette = [
        "rgba(255,50,50,0.75)", "rgba(255,180,0,0.75)",
        "rgba(100,220,40,0.75)", "rgba(0,200,255,0.75)",
        "rgba(170,80,255,0.75)", "rgba(255,100,220,0.75)",
    ]
    for k in range(K):
        if move_world.shape[0] > 0:
            proj = _apply_T_to_points(move_world, T_k[k])
            proj_idx = _world_to_voxel(proj, R)
        else:
            proj_idx = move_world
        traces.append(go.Scatter3d(
            x=proj_idx[:, 0] if proj_idx.size > 0 else [],
            y=proj_idx[:, 1] if proj_idx.size > 0 else [],
            z=proj_idx[:, 2] if proj_idx.size > 0 else [],
            mode="markers",
            marker=dict(size=1.8, color=palette[k % len(palette)]),
            name=f"T_k={k}(move)",
            visible=(k == 0),
        ))

    all_x, all_y, all_z, all_c = [], [], [], []
    for k in range(K):
        if move_world.shape[0] > 0:
            proj = _apply_T_to_points(move_world, T_k[k])
            proj_idx = _world_to_voxel(proj, R)
            all_x.extend(proj_idx[:, 0].tolist())
            all_y.extend(proj_idx[:, 1].tolist())
            all_z.extend(proj_idx[:, 2].tolist())
            all_c.extend([palette[k % len(palette)]] * proj_idx.shape[0])
    traces.append(go.Scatter3d(
        x=all_x, y=all_y, z=all_z,
        mode="markers",
        marker=dict(size=1.4, color=all_c),
        name="T_k (all)", visible=False,
    ))
    all_idx = len(traces) - 1

    n_fixed = state_trace_start
    buttons = []
    for k in range(K):
        vis = [i < n_fixed for i in range(len(traces))]
        vis[state_trace_start + k] = True
        buttons.append(dict(
            label=f"state {k}", method="update",
            args=[{"visible": vis},
                  {"title": f"{title} — state {k}"}],
        ))
    vis_all = [i < n_fixed for i in range(len(traces))]
    vis_all[all_idx] = True
    buttons.append(dict(
        label="all states", method="update",
        args=[{"visible": vis_all},
              {"title": f"{title} — all K states"}],
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{title} — state 0",
        scene=_scene(R),
        updatemenus=[dict(type="dropdown", showactive=True, buttons=buttons,
                          x=1.02, y=1.0, xanchor="left")],
    )
    _write(fig, out_path, png=False)


# ---- Per-state assignment dropdown -----------------------------------


def save_assignment_stack_html(
    per_state_assignment: np.ndarray,
    out_path: str,
    resolution: int = 64,
    title: str = "per-state assignment",
) -> None:
    if not _PLOTLY_OK:
        return
    K = per_state_assignment.shape[0]
    R = int(resolution)
    traces = []
    for k in range(K):
        a = per_state_assignment[k]
        xb, yb, zb = np.where(a == 0)
        xm, ym, zm = np.where(a == 1)
        traces.append(go.Scatter3d(
            x=xb, y=yb, z=zb, mode="markers",
            marker=dict(size=1.6, color="rgba(255,122,0,0.80)"),
            name=f"base @ state {k} (N={xb.size})", visible=(k == 0),
        ))
        traces.append(go.Scatter3d(
            x=xm, y=ym, z=zm, mode="markers",
            marker=dict(size=1.6, color="rgba(0,122,255,0.80)"),
            name=f"move @ state {k} (N={xm.size})", visible=(k == 0),
        ))
    buttons = []
    for k in range(K):
        vis = [False] * len(traces)
        vis[2 * k] = True
        vis[2 * k + 1] = True
        buttons.append(dict(
            label=f"state {k}", method="update",
            args=[{"visible": vis},
                  {"title": f"{title} — state {k}"}],
        ))
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{title} — state 0",
        scene=_scene(R),
        updatemenus=[dict(type="dropdown", showactive=True, buttons=buttons,
                          x=1.02, y=1.0, xanchor="left")],
    )
    _write(fig, out_path, png=False)


# ---- Top-level bundle ------------------------------------------------


def write_all_viz(
    viz_dir: str,
    O_stack,
    M_attn_64,
    move_mask_k_initial,
    base_mask_k_initial,
    p_base,
    p_move,
    footprint,
    base_centroid,
    warm,
    fit_result,
    seg_result,
    refine_result,
    agg,
    overlap_result,
    diag,
    resolution: int = 64,
) -> None:
    """Write the full per-stage HTML bundle for one sample."""
    if not _PLOTLY_OK:
        return
    os.makedirs(viz_dir, exist_ok=True)

    # Input
    input_dir = os.path.join(viz_dir, "input")
    save_voxel_stack_html(
        _np(O_stack > 0.5).astype(np.float32),
        os.path.join(input_dir, "O_stack.html"),
        title="Stage B O_stack (K-state)",
        resolution=resolution,
    )
    save_soft_voxel_html(
        _np(M_attn_64),
        os.path.join(input_dir, "M_attn_64.html"),
        title="Stage B M_attn_64 (semantic base prior)",
        resolution=resolution,
    )

    # C.1 partition
    c1_dir = os.path.join(viz_dir, "c1_partition")
    save_two_masks_html(
        _np(p_base), _np(p_move),
        os.path.join(c1_dir, "p_base_p_move.html"),
        title="C.1 joint_free_split (footprint × M_attn): base (orange) vs move (blue)",
        threshold=0.3, resolution=resolution,
    )
    save_soft_voxel_html(
        _np(footprint),
        os.path.join(c1_dir, "footprint.html"),
        title="C.1 footprint = max_k O_k (ever-occupied)",
        resolution=resolution,
    )
    save_voxel_stack_html(
        _np(move_mask_k_initial).astype(np.float32),
        os.path.join(c1_dir, "move_mask_k_initial.html"),
        title="C.1 initial move_mask_k per state",
        color="rgba(0,122,255,0.9)",
        resolution=resolution,
    )
    # Base centroid marker
    if _PLOTLY_OK:
        M = _np(M_attn_64)
        xs, ys, zs = np.where(M > 0.3)
        traces = [go.Scatter3d(
            x=xs, y=ys, z=zs, mode="markers",
            marker=dict(size=1.6, color=M[xs, ys, zs],
                        colorscale="Plasma", cmin=0.0, cmax=1.0,
                        colorbar=dict(title="M_attn")),
            name="M_attn>0.3",
        )]
        c_vox = _world_to_voxel(_np(base_centroid), resolution=resolution)
        traces.append(go.Scatter3d(
            x=[c_vox[0]], y=[c_vox[1]], z=[c_vox[2]],
            mode="markers",
            marker=dict(size=10, color="rgb(255,30,30)", symbol="cross"),
            name="base_centroid",
        ))
        fig = go.Figure(data=traces)
        fig.update_layout(title="C.1 base centroid (canonical frame origin)",
                          scene=_scene(resolution))
        _write(fig, os.path.join(c1_dir, "base_centroid.html"), png=False)

    # C.2 warm start
    c2_dir = os.path.join(viz_dir, "c2_warm_start")
    save_centroid_trajectory_html(
        _np(warm.centroids_world),
        warm.joint_type_hint,
        _np(warm.axis),
        _np(warm.q),
        _np(warm.phi_k),
        os.path.join(c2_dir, "centroid_trajectory.html"),
        resolution=resolution,
    )

    # C.4 volumetric
    c4_dir = os.path.join(viz_dir, "c4_volumetric")
    save_volumetric_loss_html(
        fit_result.rev.L_trace,
        fit_result.pris.L_trace,
        fit_result.bic_rev,
        fit_result.bic_pris,
        fit_result.joint_fit.joint_type,
        os.path.join(c4_dir, "loss_and_bic.html"),
    )

    # C.5 seg_refine
    c5_dir = os.path.join(viz_dir, "c5_seg_refine")
    save_data_term_html(
        _np(seg_result.data_term),
        _np(M_attn_64),
        os.path.join(c5_dir, "motion_consistency_vs_attn.html"),
        resolution=resolution,
    )
    save_assignment_stack_html(
        _np(agg.per_state_assignment),
        os.path.join(c5_dir, "per_state_assignment.html"),
        resolution=resolution,
    )

    # C.6 axis refine
    c6_dir = os.path.join(viz_dir, "c6_axis_refine")
    save_axis_overlay_html(
        _np(agg.canonical_move.float() if hasattr(agg.canonical_move, "float")
            else agg.canonical_move),
        refine_result.joint_fit.joint_type,
        _np(refine_result.joint_fit.omega),
        _np(refine_result.joint_fit.q),
        os.path.join(c6_dir, "refined_axis.html"),
        title="C.6 axis after refine",
        resolution=resolution,
    )

    # C.7 overlap cleanup
    c7_dir = os.path.join(viz_dir, "c7_overlap")
    save_overlap_cleanup_html(
        _np(overlap_result.canonical_base),
        _np(agg.canonical_move),                                # before
        _np(overlap_result.canonical_move),                     # after
        _np(overlap_result.contact_region),
        _np(overlap_result.overlap_before),
        os.path.join(c7_dir, "cleanup.html"),
        resolution=resolution,
    )

    # C.8 trajectory
    c8_dir = os.path.join(viz_dir, "c8_trajectory")
    save_trajectory_html(
        _np(overlap_result.canonical_move.float()
            if hasattr(overlap_result.canonical_move, "float")
            else overlap_result.canonical_move),
        _np(overlap_result.canonical_base.float()
            if hasattr(overlap_result.canonical_base, "float")
            else overlap_result.canonical_base),
        _np(refine_result.joint_fit.T_k),
        refine_result.joint_fit.joint_type,
        _np(refine_result.joint_fit.omega),
        _np(refine_result.joint_fit.q),
        os.path.join(c8_dir, "trajectory.html"),
        resolution=resolution,
    )

    # Diagnostics summary
    _save_diag_table_html(diag, os.path.join(viz_dir, "diagnostics.html"))


def _save_diag_table_html(diag, out_path: str) -> None:
    if not _PLOTLY_OK:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows = [
        ("joint_type_selected", diag.joint_type_selected),
        ("bic_rev", f"{diag.bic_rev:.3f}"),
        ("bic_pris", f"{diag.bic_pris:.3f}"),
        ("bic_margin", f"{diag.bic_margin:.4f}"),
        ("volumetric_loss_rev", f"{diag.volumetric_loss_rev:.4f}"),
        ("volumetric_loss_pris", f"{diag.volumetric_loss_pris:.4f}"),
        ("n_move_voxels_initial", str(diag.n_move_voxels_initial)),
        ("n_move_voxels_final", str(diag.n_move_voxels_final)),
        ("n_base_voxels_initial", str(diag.n_base_voxels_initial)),
        ("n_base_voxels_final", str(diag.n_base_voxels_final)),
        ("n_flips", str(diag.n_flips)),
        ("n_overlap_deleted", str(diag.n_overlap_deleted)),
        ("warm_start_used", diag.warm_start_used),
        ("icp_used", str(diag.icp_used)),
    ]
    fig = go.Figure(data=[go.Table(
        header=dict(values=["metric", "value"],
                    fill_color="rgba(100,100,100,0.3)",
                    align="left"),
        cells=dict(values=[[r[0] for r in rows], [r[1] for r in rows]],
                   align="left"),
    )])
    fig.update_layout(title="Stage C diagnostics (v5)")
    _write(fig, out_path, png=False)

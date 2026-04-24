"""Plotly-based visualization helpers for the v1 pipeline.

All writers produce ``.html`` files (interactive, no server required) and
optionally a matching ``.png`` via kaleido if available. The module
works without kaleido — PNG writes are silently skipped if the backend
is missing, so `run_v1.py` never hard-fails on visualization.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_OK = True
except Exception:
    _PLOTLY_OK = False


# ---- Shared camera / layout ------------------------------------------

_CAMERA = {
    "eye": {"x": 1.396, "y": -1.574, "z": 0.458},
    "center": {"x": 0, "y": 0, "z": 0},
    "up": {"x": 0, "y": 0, "z": 1},
}


def _scene(resolution: int = 64) -> dict:
    return dict(
        xaxis=dict(title="X", range=[0, resolution - 1]),
        yaxis=dict(title="Y", range=[0, resolution - 1]),
        zaxis=dict(title="Z", range=[0, resolution - 1]),
        aspectmode="cube",
        camera=_CAMERA,
    )


def _write(fig: Any, out_path: str, width: int = 800, height: int = 800,
           png: bool = True) -> None:
    """Write HTML (always) and PNG (if kaleido is available)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    html_path = out_path if out_path.endswith(".html") else out_path.replace(".png", ".html")
    if not html_path.endswith(".html"):
        html_path = html_path + ".html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    if png:
        try:
            png_path = html_path.replace(".html", ".png")
            fig.write_image(png_path, width=width, height=height)
        except Exception:
            pass  # kaleido missing / static export failed — HTML is always written


# ---- Binary voxel visualization --------------------------------------


def _scatter_from_binary(voxels: np.ndarray,
                         color: str = "rgba(255,122,0,1.0)",
                         name: str = "voxels") -> "go.Scatter3d":
    """Scatter3d trace from a binary/float occupancy grid."""
    x, y, z = np.where(voxels > 0.5)
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=1.8, color=color, line=dict(width=0)),
        name=name,
        visible=True,
    )


def save_voxel_html(
    voxels: np.ndarray,
    out_path: str,
    title: str = "voxel",
    color: str = "rgba(255,122,0,1.0)",
    resolution: int = 64,
) -> None:
    """Single binary voxel grid -> HTML (+ PNG if kaleido available)."""
    if not _PLOTLY_OK:
        return
    fig = go.Figure(data=[_scatter_from_binary(voxels, color=color, name=title)])
    fig.update_layout(title=title, scene=_scene(resolution))
    _write(fig, out_path)


def save_voxel_stack_html(
    voxel_stack: np.ndarray,
    out_path: str,
    title: str = "O_k stack",
    color: str = "rgba(255,122,0,1.0)",
    resolution: int = 64,
) -> None:
    """``(K, D, H, W)`` binary stack -> single HTML with a state dropdown.

    Only one state is visible at a time; the dropdown lets a reviewer
    flip through the K articulation states without opening K files.
    """
    if not _PLOTLY_OK:
        return
    K = voxel_stack.shape[0]
    traces = []
    for k in range(K):
        tr = _scatter_from_binary(voxel_stack[k], color=color, name=f"state {k}")
        tr.visible = (k == 0)
        traces.append(tr)

    buttons = []
    for k in range(K):
        vis = [i == k for i in range(K)]
        buttons.append(dict(label=f"state {k}",
                            method="update",
                            args=[{"visible": vis},
                                  {"title": f"{title} — state {k}"}]))
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{title} — state 0",
        scene=_scene(resolution),
        updatemenus=[dict(type="dropdown", showactive=True, buttons=buttons,
                          x=1.02, y=1.0, xanchor="left")],
    )
    _write(fig, out_path)


# ---- Soft (probability) voxel visualization --------------------------


def save_soft_voxel_html(
    p: np.ndarray,
    out_path: str,
    title: str = "p(voxel)",
    threshold: float = 0.15,
    colorscale: str = "Viridis",
    resolution: int = 64,
) -> None:
    """Soft probability field `(D, H, W) in [0,1]` -> HTML with
    continuous color mapped to the probability value.

    Points below ``threshold`` are dropped to keep the HTML small.
    """
    if not _PLOTLY_OK:
        return
    active = p > threshold
    xs, ys, zs = np.where(active)
    vals = p[active]
    fig = go.Figure(data=[go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        marker=dict(
            size=2.0,
            color=vals,
            colorscale=colorscale,
            cmin=0.0, cmax=1.0,
            colorbar=dict(title="p"),
            opacity=0.75,
        ),
        name=title,
    )])
    fig.update_layout(title=title, scene=_scene(resolution))
    _write(fig, out_path)


def save_two_masks_html(
    M_base: np.ndarray,
    M_move: np.ndarray,
    out_path: str,
    title: str = "base vs move",
    threshold: float = 0.3,
    resolution: int = 64,
) -> None:
    """Overlay the two soft masks in orange (base) / blue (move)."""
    if not _PLOTLY_OK:
        return
    xb, yb, zb = np.where(M_base > threshold)
    xm, ym, zm = np.where(M_move > threshold)
    traces = [
        go.Scatter3d(x=xb, y=yb, z=zb, mode="markers",
                     marker=dict(size=1.8, color="rgba(255,122,0,0.75)"),
                     name="M_base"),
        go.Scatter3d(x=xm, y=ym, z=zm, mode="markers",
                     marker=dict(size=1.8, color="rgba(0,122,255,0.75)"),
                     name="M_move"),
    ]
    fig = go.Figure(data=traces)
    fig.update_layout(title=title, scene=_scene(resolution))
    _write(fig, out_path)


# ---- Anchors + fitted axis overlay -----------------------------------


def save_anchors_html(
    mu_O: np.ndarray,
    anchor_coords: np.ndarray,
    anchor_weights: np.ndarray,
    out_path: str,
    title: str = "contact anchors",
    bg_threshold: float = 0.3,
    resolution: int = 64,
) -> None:
    """Visualize the mean occupancy as grey background + anchors colored
    by their weight."""
    if not _PLOTLY_OK:
        return
    xs, ys, zs = np.where(mu_O > bg_threshold)
    bg = go.Scatter3d(x=xs, y=ys, z=zs, mode="markers",
                      marker=dict(size=1.4, color="rgba(160,160,160,0.25)"),
                      name="mean occupancy")
    traces = [bg]
    if anchor_coords is not None and len(anchor_coords) > 0:
        ac = np.asarray(anchor_coords)
        aw = np.asarray(anchor_weights).astype(np.float32)
        aw_norm = aw / max(float(aw.max()), 1e-8)
        traces.append(go.Scatter3d(
            x=ac[:, 0], y=ac[:, 1], z=ac[:, 2],
            mode="markers",
            marker=dict(
                size=4.5,
                color=aw_norm,
                colorscale="Plasma",
                cmin=0.0, cmax=1.0,
                colorbar=dict(title="w"),
            ),
            name=f"anchors (N={len(ac)})",
        ))
    fig = go.Figure(data=traces)
    fig.update_layout(title=title, scene=_scene(resolution))
    _write(fig, out_path)


def save_axis_overlay_html(
    M_move: np.ndarray,
    joint_type: str,
    omega_world: np.ndarray,
    q_world: np.ndarray,
    out_path: str,
    title: str = "fitted axis",
    threshold: float = 0.3,
    resolution: int = 64,
) -> None:
    """Plot the fitted screw axis overlaid on the move-region voxels.

    Everything is rendered in voxel-index space ``[0, R-1]``. The caller
    passes ``omega_world, q_world`` in the SAJO world frame
    (``(i,j,k)/(R-1) - 0.5``) and we map them back to indices for display.
    """
    if not _PLOTLY_OK:
        return
    xs, ys, zs = np.where(M_move > threshold)
    move_trace = go.Scatter3d(
        x=xs, y=ys, z=zs, mode="markers",
        marker=dict(size=1.8, color="rgba(0,122,255,0.75)"),
        name="M_move",
    )

    R = int(resolution)
    # World -> voxel index: idx = (world + 0.5) * (R - 1)
    q_idx = (np.asarray(q_world, dtype=np.float64) + 0.5) * (R - 1)
    omega_np = np.asarray(omega_world, dtype=np.float64)
    if np.linalg.norm(omega_np) > 1e-8:
        omega_np = omega_np / np.linalg.norm(omega_np)

    traces: List[Any] = [move_trace]
    if joint_type == "revolute":
        # Draw the axis line as q +/- t * omega, t in [-R, R] voxel units.
        ts = np.linspace(-R, R, 200)
        line_pts = q_idx[None, :] + ts[:, None] * omega_np[None, :] * (R - 1) / R
        in_range = np.all((line_pts >= 0) & (line_pts <= R - 1), axis=1)
        line_pts = line_pts[in_range]
        if line_pts.shape[0] >= 2:
            traces.append(go.Scatter3d(
                x=line_pts[:, 0], y=line_pts[:, 1], z=line_pts[:, 2],
                mode="lines",
                line=dict(color="rgb(255,30,30)", width=6),
                name="axis (revolute)",
            ))
    elif joint_type == "prismatic":
        # Draw an arrow from the move centroid along v_hat.
        center = np.array([xs.mean() if xs.size > 0 else R / 2,
                           ys.mean() if ys.size > 0 else R / 2,
                           zs.mean() if zs.size > 0 else R / 2], dtype=np.float64)
        tip = center + omega_np * (R / 3.0)
        traces.append(go.Scatter3d(
            x=[center[0], tip[0]],
            y=[center[1], tip[1]],
            z=[center[2], tip[2]],
            mode="lines+markers",
            line=dict(color="rgb(255,30,30)", width=8),
            marker=dict(size=[3, 8], color="rgb(255,30,30)"),
            name="v_hat (prismatic)",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(title=title, scene=_scene(resolution))
    _write(fig, out_path)


# ---- 2D diagnostic curves --------------------------------------------


def save_diagnostics_curves_html(
    diagnostics: Sequence[Dict[str, Any]],
    out_path: str,
    title: str = "VGCF diagnostics",
    keys: Sequence[str] = ("sigma2_median", "sigma2_max", "M_mean", "lambda"),
) -> None:
    """Line plot of per-step diagnostic keys (x = step index)."""
    if not _PLOTLY_OK or not diagnostics:
        return
    steps = [int(d.get("step", i)) for i, d in enumerate(diagnostics)]
    ts = [float(d.get("t", 0.0)) for d in diagnostics]

    fig = make_subplots(rows=len(keys), cols=1, shared_xaxes=True,
                        subplot_titles=list(keys))
    for row, key in enumerate(keys, start=1):
        ys = [float(d.get(key, float("nan"))) for d in diagnostics]
        fig.add_trace(go.Scatter(x=steps, y=ys, mode="lines+markers", name=key),
                      row=row, col=1)
    fig.update_layout(title=title, height=250 * len(keys),
                      showlegend=False)
    fig.update_xaxes(title_text="step", row=len(keys), col=1)
    # secondary info: t value per step as a text trace in the first subplot
    fig.add_trace(go.Scatter(x=steps, y=ts, mode="lines", name="t",
                             line=dict(dash="dot", color="rgba(120,120,120,0.6)"),
                             showlegend=True), row=1, col=1)
    _write(fig, out_path, png=False)


def save_em_traces_html(
    rev_trace: Sequence[float],
    pris_trace: Sequence[float],
    bic: Dict[str, Any],
    out_path: str,
    title: str = "SAJO EM traces",
) -> None:
    """Two EM L_reg traces + BIC bar chart."""
    if not _PLOTLY_OK:
        return
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("L_reg trace", "BIC comparison"),
                        specs=[[{"type": "xy"}, {"type": "xy"}]])
    if rev_trace:
        fig.add_trace(go.Scatter(y=list(rev_trace), mode="lines+markers",
                                 name="revolute", line=dict(color="rgb(255,122,0)")),
                      row=1, col=1)
    if pris_trace:
        fig.add_trace(go.Scatter(y=list(pris_trace), mode="lines+markers",
                                 name="prismatic", line=dict(color="rgb(0,122,255)")),
                      row=1, col=1)
    bic_rev = bic.get("bic_rev")
    bic_pris = bic.get("bic_pris")
    if bic_rev is not None and bic_pris is not None:
        fig.add_trace(go.Bar(
            x=["revolute", "prismatic"],
            y=[float(bic_rev), float(bic_pris)],
            marker=dict(color=["rgb(255,122,0)", "rgb(0,122,255)"]),
            name="BIC",
            showlegend=False,
        ), row=1, col=2)
    selected = bic.get("joint_type", "?")
    conf = bic.get("confidence")
    conf_str = f"{conf:.3f}" if isinstance(conf, (int, float)) else "n/a"
    fig.update_layout(
        title=f"{title} — selected: {selected} (conf={conf_str})",
        height=420,
    )
    fig.update_xaxes(title_text="outer iter", row=1, col=1)
    fig.update_yaxes(title_text="L_reg", row=1, col=1)
    fig.update_yaxes(title_text="BIC", row=1, col=2)
    _write(fig, out_path, png=False)

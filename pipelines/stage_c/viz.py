"""Stage C visualisation.

Produces HTML (+ PNG via kaleido if available) artifacts that let the user
verify the joint init at a glance:

  joint_overview_3d.html      — K per-state centroids + best-fit primitive
                                 (line or arc) + joint axis + anchors,
                                 all in voxel-index space.
  M_motion_corridor_64.html   — soft voxel cloud of swept volume.
  anchors_overlay.html        — anchors on base background.
  axis_overlay.html           — axis line on motion-corridor voxels.
  phi_progression.html        — 2D charts: u_raw, u_normalised, phi_0_shifted,
                                 delta_u_init.
  type_fit_diagnostics.html   — bar chart of residual_line vs residual_arc
                                 plus type_logit / confidence breakdown.
  summary.html                — text summary of JointInit + diagnostics.

All writers silently skip when plotly is missing, matching the
`pipelines/utils/voxel_viz.py` convention so Stage C never blocks on viz.

Called from `run_stage_c_init.run_stage_c_joint_init` when
`StageCConfig.save_viz=True`. Bootstrap B12 separately persists the npys
this module reads; Stage C viz is a debug aid, not a downstream
dependency.
"""

from __future__ import annotations

import html
import os
from typing import Any, Optional

import numpy as np
import torch

from pipelines.utils.voxel_viz import (
    _PLOTLY_OK,
    _CAMERA,
    _scene,
    _write,
    save_anchors_html,
    save_axis_overlay_html,
    save_soft_voxel_html,
    save_voxel_html,
)


# ---------------------------------------------------------------------------
# Voxel-space helpers (world-coord input -> voxel-index output for plotting)
# ---------------------------------------------------------------------------


def _world_to_voxel_idx_f(p_world: np.ndarray, res: int) -> np.ndarray:
    """Map world coords (-0.5, 0.5) -> continuous voxel index [0, res - 1]."""
    return (np.asarray(p_world, dtype=np.float64) + 0.5) * (res - 1)


def _norm_np(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    return float(np.sqrt((arr * arr).sum()))


def _color_for_k(k: int, K: int) -> str:
    """Per-state colour for K=6: red -> orange -> yellow -> green -> blue -> purple."""
    palette = [
        "rgb(231, 76, 60)",    # red       state 0
        "rgb(241, 153, 76)",   # orange    state 1
        "rgb(241, 196, 15)",   # yellow    state 2
        "rgb(46, 204, 113)",   # green     state 3
        "rgb(52, 152, 219)",   # blue      state 4
        "rgb(155, 89, 182)",   # purple    state 5
    ]
    return palette[k % len(palette)] if k < len(palette) else "rgb(120, 120, 120)"


# ---------------------------------------------------------------------------
# 1) Joint overview 3D
# ---------------------------------------------------------------------------


def save_joint_overview_html(
    out_path: str,
    geom: Any,
    type_result: Any,
    axis_result: Any,
    anchors_object: Optional[torch.Tensor],
    O_base_canonical: Optional[torch.Tensor],
    M_motion_corridor_64: Optional[torch.Tensor],
    resolution: int = 64,
    title: str = "Stage C joint overview",
) -> None:
    """3D scatter combining centroids + best-fit primitive + axis + anchors.

    All in voxel-index space [0, resolution-1].
    """
    if not _PLOTLY_OK:
        return

    import plotly.graph_objects as go

    traces = []

    # Layer 1: base voxels as light gray background (low alpha)
    if O_base_canonical is not None:
        base_np = O_base_canonical.detach().cpu().numpy()
        if base_np.ndim > 3:
            base_np = base_np.squeeze()
        bxs, bys, bzs = np.where(base_np > 0)
        if bxs.size > 0:
            # Subsample for performance
            n = bxs.size
            stride = max(1, n // 3000)
            traces.append(go.Scatter3d(
                x=bxs[::stride], y=bys[::stride], z=bzs[::stride],
                mode="markers",
                marker=dict(size=1.4, color="rgba(180, 180, 180, 0.2)"),
                name=f"O_base_canonical ({n} voxels)",
            ))

    # Layer 2: motion corridor as red-tinted soft voxels (where strong)
    if M_motion_corridor_64 is not None:
        mcorr_np = M_motion_corridor_64.detach().cpu().numpy()
        if mcorr_np.ndim > 3:
            mcorr_np = mcorr_np.squeeze()
        mxs, mys, mzs = np.where(mcorr_np > 0.1)
        if mxs.size > 0:
            mvals = mcorr_np[mxs, mys, mzs]
            n = mxs.size
            stride = max(1, n // 1500)
            traces.append(go.Scatter3d(
                x=mxs[::stride], y=mys[::stride], z=mzs[::stride],
                mode="markers",
                marker=dict(
                    size=2.0,
                    color=mvals[::stride],
                    colorscale="Reds",
                    cmin=0.0, cmax=1.0,
                    opacity=0.55,
                    showscale=False,
                ),
                name=f"M_motion_corridor (>0.1, {n} voxels)",
            ))

    # Layer 3: per-state centroids (color-coded), in voxel index space
    cents_w = geom.centroid_world.detach().cpu().numpy()
    valid = geom.valid_mask.detach().cpu().numpy().astype(bool)
    K = int(cents_w.shape[0])
    cents_idx = _world_to_voxel_idx_f(cents_w, resolution)
    for k in range(K):
        if not valid[k]:
            continue
        traces.append(go.Scatter3d(
            x=[cents_idx[k, 0]], y=[cents_idx[k, 1]], z=[cents_idx[k, 2]],
            mode="markers+text",
            marker=dict(size=10.0, color=_color_for_k(k, K),
                        line=dict(color="black", width=1)),
            text=[f"s{k}"],
            textposition="top center",
            textfont=dict(color="black", size=12),
            name=f"centroid s{k}",
        ))

    # Layer 4: best-fit primitive (line or arc)
    valid_k = np.where(valid)[0]
    if valid_k.size >= 2:
        if type_result.type_str == "revolute" and type_result.arc_center is not None:
            # Draw the circle in axis-perp plane through arc_center with arc_radius
            ac = _world_to_voxel_idx_f(type_result.arc_center, resolution)
            an = type_result.arc_normal
            an = np.asarray(an, dtype=np.float64)
            an = an / (_norm_np(an) + 1e-12)
            # Two basis vectors in plane perpendicular to an
            tmp = np.array([1.0, 0.0, 0.0]) if abs(an[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = np.cross(an, tmp); u = u / (_norm_np(u) + 1e-12)
            v = np.cross(an, u)
            r_world = float(type_result.arc_radius)
            r_idx = r_world * (resolution - 1)
            thetas = np.linspace(0, 2 * np.pi, 100)
            circle_pts = ac[None, :] + r_idx * (np.cos(thetas)[:, None] * u[None, :] +
                                                  np.sin(thetas)[:, None] * v[None, :])
            traces.append(go.Scatter3d(
                x=circle_pts[:, 0], y=circle_pts[:, 1], z=circle_pts[:, 2],
                mode="lines",
                line=dict(color="rgb(255, 165, 0)", width=4, dash="dot"),
                name=f"best-fit arc (r={r_world:.3f})",
            ))
        else:
            # Prismatic / fallback: best-fit line through line_origin along line_direction
            lo = _world_to_voxel_idx_f(type_result.line_origin, resolution)
            ld = np.asarray(type_result.line_direction, dtype=np.float64)
            ld = ld / (_norm_np(ld) + 1e-12)
            # Extend ±2x trajectory length
            cents_along = ((cents_w[valid] - type_result.line_origin) * ld[None, :]).sum(axis=1)
            t_max = float(np.abs(cents_along).max()) * 1.5 if cents_along.size > 0 else 0.2
            t_max_idx = t_max * (resolution - 1)
            ts = np.linspace(-t_max_idx, t_max_idx, 60)
            line_pts = lo[None, :] + ts[:, None] * ld[None, :]
            in_range = np.all((line_pts >= 0) & (line_pts <= resolution - 1), axis=1)
            line_pts = line_pts[in_range]
            if line_pts.shape[0] >= 2:
                traces.append(go.Scatter3d(
                    x=line_pts[:, 0], y=line_pts[:, 1], z=line_pts[:, 2],
                    mode="lines",
                    line=dict(color="rgb(255, 165, 0)", width=4, dash="dot"),
                    name="best-fit line",
                ))

    # Layer 5: joint axis (the resolved axis_result.axis through origin)
    ax_world = axis_result.axis.detach().cpu().numpy()
    ax_world = ax_world / (_norm_np(ax_world) + 1e-12)
    orig_world = axis_result.origin.detach().cpu().numpy()
    orig_idx = _world_to_voxel_idx_f(orig_world, resolution)
    ts_axis = np.linspace(-0.6, 0.6, 60)
    ax_pts = orig_idx[None, :] + ts_axis[:, None] * ax_world[None, :] * (resolution - 1)
    in_range = np.all((ax_pts >= 0) & (ax_pts <= resolution - 1), axis=1)
    ax_pts = ax_pts[in_range]
    if ax_pts.shape[0] >= 2:
        traces.append(go.Scatter3d(
            x=ax_pts[:, 0], y=ax_pts[:, 1], z=ax_pts[:, 2],
            mode="lines",
            line=dict(color="rgb(220, 20, 60)", width=8),
            name=f"joint axis ({type_result.type_str})",
        ))
        # Origin marker
        traces.append(go.Scatter3d(
            x=[orig_idx[0]], y=[orig_idx[1]], z=[orig_idx[2]],
            mode="markers",
            marker=dict(size=6.5, color="rgb(220, 20, 60)",
                        line=dict(color="black", width=1)),
            name="axis origin",
        ))

    # Layer 6: anchors as small black dots
    if anchors_object is not None and int(anchors_object.shape[0]) > 0:
        ac = anchors_object.detach().cpu().numpy().astype(np.float64)
        traces.append(go.Scatter3d(
            x=ac[:, 0], y=ac[:, 1], z=ac[:, 2],
            mode="markers",
            marker=dict(size=3.5, color="rgb(0, 0, 0)", symbol="diamond"),
            name=f"anchors (N={ac.shape[0]})",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=_scene(resolution),
        legend=dict(x=0.02, y=0.98, font=dict(size=10)),
        width=1100, height=800,
    )
    _write(fig, out_path)


# ---------------------------------------------------------------------------
# 2) Phi progression 2D chart
# ---------------------------------------------------------------------------


def save_phi_progression_html(
    out_path: str,
    phi_result: Any,
    joint_init: Any,
    canonical_state_idx: int = 2,
    title: str = "Stage C phi_0 progression",
) -> None:
    """2D chart of u_raw, u_normalised, phi_0_shifted, delta_u_init across K states."""
    if not _PLOTLY_OK:
        return

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    u_raw = phi_result.u_raw.detach().cpu().numpy()
    u_norm = phi_result.u_normalized.detach().cpu().numpy()
    phi_shifted = phi_result.phi_0_shifted.detach().cpu().numpy()
    delta_u = phi_result.delta_u_init.detach().cpu().numpy()
    K = len(u_raw)
    ks = list(range(K))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "u_raw (signed projection)",
            "u_normalised in [0, 1]",
            f"phi_0_shifted (phi_0[c={canonical_state_idx}]=0)",
            "delta_u_init (softplus-inverted diffs)",
        ),
        vertical_spacing=0.16,
    )
    fig.add_trace(go.Scatter(
        x=ks, y=u_raw.tolist(),
        mode="lines+markers", marker=dict(size=10), name="u_raw",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ks, y=u_norm.tolist(),
        mode="lines+markers", marker=dict(size=10), name="u_normalised",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=ks, y=phi_shifted.tolist(),
        mode="lines+markers", marker=dict(size=10), name="phi_0_shifted",
        line=dict(color="rgb(220, 20, 60)"),
    ), row=2, col=1)
    # Mark canonical state
    fig.add_trace(go.Scatter(
        x=[canonical_state_idx], y=[float(phi_shifted[canonical_state_idx])],
        mode="markers", marker=dict(size=18, color="rgba(220, 20, 60, 0.3)",
                                     symbol="circle-open", line=dict(width=3)),
        name=f"c={canonical_state_idx} (=0)", showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=list(range(len(delta_u))), y=delta_u.tolist(),
        name="delta_u_init",
        marker=dict(color="rgba(46, 204, 113, 0.8)"),
    ), row=2, col=2)

    fig.update_xaxes(title_text="state k", row=2, col=1)
    fig.update_xaxes(title_text="step i", row=2, col=2)
    fig.update_layout(
        title=(
            f"{title} — type={joint_init.joint_type()}, "
            f"confidence={joint_init.confidence:.3f}"
        ),
        height=700, width=1100,
        showlegend=False,
    )
    _write(fig, out_path, png=True)


# ---------------------------------------------------------------------------
# 3) Type fit diagnostics
# ---------------------------------------------------------------------------


def save_type_fit_diagnostics_html(
    out_path: str,
    type_result: Any,
    axis_result: Any,
    joint_init: Any,
    title: str = "Stage C joint type detection diagnostics",
) -> None:
    """Bar chart: residual_line vs residual_arc, with type_logit annotation."""
    if not _PLOTLY_OK:
        return

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    res_line = float(type_result.residual_line)
    res_arc = float(type_result.residual_arc) if np.isfinite(type_result.residual_arc) else None
    type_logit = float(type_result.type_logit)
    sub_conf = joint_init.sub_confidence

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Fit residuals (line vs arc)",
            "Sub-confidence breakdown",
        ),
        column_widths=[0.55, 0.45],
    )

    # Residuals bar
    res_x = ["line"]
    res_y = [res_line]
    res_color = ["rgb(52, 152, 219)"]
    if res_arc is not None:
        res_x.append("arc")
        res_y.append(res_arc)
        res_color.append("rgb(155, 89, 182)")
    fig.add_trace(go.Bar(
        x=res_x, y=res_y, marker=dict(color=res_color),
        name="residual", text=[f"{v:.6f}" for v in res_y], textposition="outside",
    ), row=1, col=1)

    # Sub-confidence bar
    sub_keys = ["type", "axis", "centroid", "base"]
    sub_vals = [float(sub_conf.get(k, 0.0)) for k in sub_keys]
    fig.add_trace(go.Bar(
        x=sub_keys, y=sub_vals,
        marker=dict(color=["rgb(231,76,60)", "rgb(241,196,15)",
                          "rgb(46,204,113)", "rgb(52,152,219)"]),
        name="sub_conf",
        text=[f"{v:.3f}" for v in sub_vals], textposition="outside",
    ), row=1, col=2)

    fig.update_yaxes(range=[0, 1.05], row=1, col=2)
    fig.update_layout(
        title=(
            f"{title}<br>"
            f"<sub>joint_type={joint_init.joint_type()}, "
            f"type_logit={type_logit:.3f}, "
            f"axis_fit_source={axis_result.fit_source}, "
            f"overall_confidence={joint_init.confidence:.3f}</sub>"
        ),
        height=480, width=1000, showlegend=False,
    )
    _write(fig, out_path, png=True)


# ---------------------------------------------------------------------------
# 4) Summary HTML (plain table, no plotly dependency)
# ---------------------------------------------------------------------------


def save_summary_html(
    out_path: str,
    joint_init: Any,
    type_result: Any,
    axis_result: Any,
    phi_result: Any,
    title: str = "Stage C summary",
) -> None:
    """Plain HTML table of the JointInit + key sub-result fields. Works
    even without plotly — useful as a single-page report."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(_fmt(x) for x in v) + "]"
        return html.escape(str(v))

    psi = joint_init.psi
    axis_list = psi.axis.detach().cpu().tolist()
    origin_list = psi.origin.detach().cpu().tolist()
    delta_u_list = psi.delta_u_init.detach().cpu().tolist()
    phi_list = joint_init.phi_0.detach().cpu().tolist()

    rows = [
        ("joint_type", joint_init.joint_type()),
        ("confidence (overall)", joint_init.confidence),
        ("psi.axis", axis_list),
        ("psi.origin", origin_list),
        ("psi.type_logit", psi.type_logit),
        ("psi.theta_limit_raw (softplus_inv)", psi.theta_limit_raw),
        ("psi.disp_limit_raw (softplus_inv)", psi.disp_limit_raw),
        ("psi.delta_u_init", delta_u_list),
        ("phi_0 (c-shifted)", phi_list),
        ("phi_0[c]", phi_list[2] if len(phi_list) > 2 else None),
        ("anchors count", int(joint_init.anchors_object.shape[0])),
        ("--- sub_confidence ---", ""),
        ("conf type", joint_init.sub_confidence.get("type", 0.0)),
        ("conf axis", joint_init.sub_confidence.get("axis", 0.0)),
        ("conf centroid", joint_init.sub_confidence.get("centroid", 0.0)),
        ("conf base", joint_init.sub_confidence.get("base", 0.0)),
        ("--- diagnostics ---", ""),
        ("type_str", type_result.type_str),
        ("type_logit (raw)", type_result.type_logit),
        ("residual_line", type_result.residual_line),
        ("residual_arc", type_result.residual_arc),
        ("n_valid_states", type_result.n_valid_states),
        ("axis_fit_source", axis_result.fit_source),
        ("axis_fit_residual", axis_result.fit_residual),
        ("phi monotone_enforced", phi_result.monotone_enforced),
        ("observed_max_angle (rad)", phi_result.observed_max_angle),
        ("observed_max_disp (world)", phi_result.observed_max_disp),
    ]

    body_rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{_fmt(v)}</td></tr>"
        for k, v in rows
    )
    htmltxt = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:'Source Sans Pro',Arial,sans-serif;"
        "max-width:900px;margin:30px auto;color:#222;}"
        "h1{font-size:20px;}table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #ddd;padding:6px 10px;text-align:left;font-size:13px;}"
        "th{background:#f3f3f3;width:35%;}"
        "tr:nth-child(even){background:#fafafa;}"
        ".notes{color:#666;font-size:12px;margin-top:14px;}</style></head>"
        f"<body><h1>{html.escape(title)}</h1><table>{body_rows}</table>"
        "<p class='notes'>Generated by pipelines/stage_c/viz.save_summary_html. "
        "Bootstrap B6 persists the same content to <code>stage_c_joint_init.json</code>.</p>"
        "</body></html>"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(htmltxt)


# ---------------------------------------------------------------------------
# Public entrypoint called by run_stage_c_init
# ---------------------------------------------------------------------------


def save_stage_c_viz(
    viz_dir: str,
    inputs: Any,
    geom: Any,
    type_result: Any,
    axis_result: Any,
    phi_result: Any,
    anchors_object: torch.Tensor,
    joint_init: Any,
) -> None:
    """Produce the full Stage C viz bundle under {viz_dir}/.

    Always writes summary.html (no plotly needed). Other writers
    silently skip when plotly is missing — matching the
    pipelines/utils/voxel_viz.py convention.
    """
    os.makedirs(viz_dir, exist_ok=True)
    res = int(inputs.O_init.shape[-1]) if inputs.O_init is not None else 64

    # Always: text summary
    save_summary_html(
        os.path.join(viz_dir, "summary.html"),
        joint_init=joint_init,
        type_result=type_result,
        axis_result=axis_result,
        phi_result=phi_result,
        title="Stage C joint init summary",
    )

    if not _PLOTLY_OK:
        return  # remaining viz needs plotly

    # 3D joint overview
    save_joint_overview_html(
        os.path.join(viz_dir, "joint_overview_3d.html"),
        geom=geom,
        type_result=type_result,
        axis_result=axis_result,
        anchors_object=anchors_object,
        O_base_canonical=inputs.O_base_canonical,
        M_motion_corridor_64=inputs.M_motion_corridor_64,
        resolution=res,
    )

    # phi progression chart
    save_phi_progression_html(
        os.path.join(viz_dir, "phi_progression.html"),
        phi_result=phi_result,
        joint_init=joint_init,
        canonical_state_idx=int(geom.centroid_world.shape[0]) // 3 if False else 2,
    )

    # Type detection diagnostics
    save_type_fit_diagnostics_html(
        os.path.join(viz_dir, "type_fit_diagnostics.html"),
        type_result=type_result,
        axis_result=axis_result,
        joint_init=joint_init,
    )

    # Motion corridor heatmap (reuse soft voxel writer)
    if inputs.M_motion_corridor_64 is not None:
        mcorr_np = inputs.M_motion_corridor_64.detach().cpu().to(torch.float32).numpy()
        if mcorr_np.ndim > 3:
            mcorr_np = mcorr_np.squeeze()
        save_soft_voxel_html(
            mcorr_np,
            os.path.join(viz_dir, "M_motion_corridor_64.html"),
            title="M_motion_corridor_64 (Stage B Pass-1 footprint*(1-shared))",
            threshold=0.1,
            resolution=res,
        )

    # Anchors overlaid on base (use existing save_anchors_html convention).
    # We synthesise a "background" by using O_base_canonical (or zeros) and
    # uniform weights for anchors. Anchor coords are already voxel indices.
    if anchors_object is not None and int(anchors_object.shape[0]) > 0:
        bg = None
        if inputs.O_base_canonical is not None:
            bg = inputs.O_base_canonical.detach().cpu().to(torch.float32).numpy()
            if bg.ndim > 3:
                bg = bg.squeeze()
        else:
            bg = np.zeros((res, res, res), dtype=np.float32)
        ac = anchors_object.detach().cpu().numpy().astype(np.float32)
        weights = np.ones(ac.shape[0], dtype=np.float32)
        save_anchors_html(
            mu_O=bg,
            anchor_coords=ac,
            anchor_weights=weights,
            out_path=os.path.join(viz_dir, "anchors_overlay.html"),
            title=f"Stage C anchors on O_base_canonical (N={ac.shape[0]})",
            bg_threshold=0.5,
            resolution=res,
        )

    # Axis overlay on motion corridor (uses existing save_axis_overlay_html
    # which expects world-space axis + origin; perfect for axis_result).
    if inputs.M_motion_corridor_64 is not None:
        mcorr_np = inputs.M_motion_corridor_64.detach().cpu().to(torch.float32).numpy()
        if mcorr_np.ndim > 3:
            mcorr_np = mcorr_np.squeeze()
        ax_w = axis_result.axis.detach().cpu().numpy()
        or_w = axis_result.origin.detach().cpu().numpy()
        save_axis_overlay_html(
            M_move=mcorr_np,
            joint_type=type_result.type_str if type_result.type_str != "uncertain" else "prismatic",
            omega_world=ax_w,
            q_world=or_w,
            out_path=os.path.join(viz_dir, "axis_overlay.html"),
            title=f"Stage C axis ({type_result.type_str}) on motion corridor",
            threshold=0.1,
            resolution=res,
        )

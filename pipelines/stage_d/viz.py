"""Per-iter visualization snapshots for Stage D.

Code conventions (record/readme.md) require every major experimental
stage to produce visualizations for debugging. Stage D writes:

  viz/iter_{it:06d}/
    rendered.png        horizontal strip of 21 rendered RGB frames
    target.png          horizontal strip of 21 Wan video target frames
    diff.png            abs(rendered - target) strip (x4 contrast)
    state_3d.html       plotly: U_object voxels colored by g*(1-m) base /
                        g*m move + joint axis overlay (interactive)
    phi_curve.html      plotly: u_shifted, phi_rev, phi_pri as a 2D
                        line plot across 6 states / 21 frames
    gates.npz           raw per-voxel (g, m) + U_object for offline tooling
    phi_curve.npz       raw u_shifted / phi_rev / phi_pri arrays
    metrics.json        schedule snapshot + loss components

Plus at training start:
  viz/iter_0_camera_diag.png    iter-0 camera sanity-check 3-panel strip

PNG strips use PIL; HTML viz uses ``pipelines.utils.voxel_viz`` (plotly).
If plotly is missing, HTML saves are silently skipped (utility's own
fallback) so training never hard-fails on visualization alone.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from .config import TRELLIS_OCC_RES

# Optional plotly-based HTML viz. Import is guarded so the rest of viz.py
# (PNG strips + metrics.json) still works in minimal environments.
try:
    from pipelines.utils import voxel_viz as _vxz
    _VOXEL_VIZ_OK = True
except Exception:
    _VOXEL_VIZ_OK = False


def _frames_to_strip(frames_T3HW_01: torch.Tensor) -> np.ndarray:
    """Concatenate F frames of shape [F, 3, H, W] in [0, 1] into a single strip.

    Returns ``np.uint8`` of shape ``[H, F * W, 3]`` (RGB order) ready for
    ``Image.fromarray``. Float input is clamped to [0, 1] first.
    """
    if frames_T3HW_01.ndim != 4 or frames_T3HW_01.shape[1] != 3:
        raise ValueError(
            f"expected [F, 3, H, W]; got {tuple(frames_T3HW_01.shape)}"
        )
    f = frames_T3HW_01.detach().cpu().float().clamp(0.0, 1.0)
    f = (f * 255.0).round().to(torch.uint8).numpy()       # [F, 3, H, W]
    F_, C_, H_, W_ = f.shape
    # F frames side-by-side horizontally
    strip = f.transpose(0, 2, 3, 1).reshape(F_, H_, W_, 3)
    strip = strip.transpose(1, 0, 2, 3).reshape(H_, F_ * W_, 3)
    return strip


def save_iter_snapshot(
    out_root: str,
    iter_idx: int,
    rgb_T3HW_01: torch.Tensor,
    wan_target_T3HW_01: torch.Tensor,
    metrics: Dict[str, float],
    U_object: Optional[np.ndarray] = None,
    g_per_voxel: Optional[torch.Tensor] = None,
    m_per_voxel: Optional[torch.Tensor] = None,
) -> None:
    """Write the iter ``it`` snapshot under ``<out_root>/viz/iter_{it:06d}/``.

    Skips the heavy items (frames, gates) when their inputs are ``None``
    so the train loop can defer them to a sparser schedule (e.g. only
    save full snapshots every ``viz_every`` iters but log metrics every
    ``log_every``).
    """
    snap_dir = os.path.join(out_root, "viz", f"iter_{iter_idx:06d}")
    os.makedirs(snap_dir, exist_ok=True)

    # Metrics (always)
    with open(os.path.join(snap_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=True)

    # Frame strips
    if rgb_T3HW_01 is not None:
        rendered = _frames_to_strip(rgb_T3HW_01)
        Image.fromarray(rendered).save(os.path.join(snap_dir, "rendered.png"))
    if wan_target_T3HW_01 is not None:
        target = _frames_to_strip(wan_target_T3HW_01)
        Image.fromarray(target).save(os.path.join(snap_dir, "target.png"))

    if rgb_T3HW_01 is not None and wan_target_T3HW_01 is not None:
        diff = (rgb_T3HW_01 - wan_target_T3HW_01).abs().clamp(0.0, 0.25) * 4.0
        Image.fromarray(_frames_to_strip(diff)).save(
            os.path.join(snap_dir, "diff.png")
        )

    # Gate arrays (for offline HTML voxel viz)
    if U_object is not None and g_per_voxel is not None and m_per_voxel is not None:
        np.savez_compressed(
            os.path.join(snap_dir, "gates.npz"),
            U_object=U_object,
            g=g_per_voxel.detach().cpu().float().numpy(),
            m=m_per_voxel.detach().cpu().float().numpy(),
        )


def save_phi_curve(out_root: str, iter_idx: int,
                    u_shifted: torch.Tensor,
                    phi_rev: torch.Tensor,
                    phi_pri: torch.Tensor) -> None:
    """Persist the current iter's phi roll-out as a small npz for plotting.

    u_shifted is the canonical-state-shifted normalized progress in
    ``[u_shifted[c] = 0]``; ``phi_rev / phi_pri`` are signed angles /
    displacements per the two-branch convention.
    """
    snap_dir = os.path.join(out_root, "viz", f"iter_{iter_idx:06d}")
    os.makedirs(snap_dir, exist_ok=True)
    np.savez(
        os.path.join(snap_dir, "phi_curve.npz"),
        u_shifted=u_shifted.detach().cpu().float().numpy(),
        phi_rev=phi_rev.detach().cpu().float().numpy(),
        phi_pri=phi_pri.detach().cpu().float().numpy(),
    )


# =============================================================================
# 3D interactive HTML viz (plotly via pipelines.utils.voxel_viz)
# =============================================================================

def _build_per_voxel_field(
    U_object: np.ndarray,
    values: np.ndarray,
    resolution: int = TRELLIS_OCC_RES,
) -> np.ndarray:
    """Scatter per-voxel values into a dense ``[resolution]^3`` array.

    Used to turn (U_object, g, m) into the dense soft-mask fields that
    ``voxel_viz.save_two_masks_html`` / ``save_axis_overlay_html`` expect.

    Voxels outside ``U_object`` get value 0 (background).
    """
    if U_object.ndim != 2 or U_object.shape[1] != 3:
        raise ValueError(f"U_object must be [N, 3]; got {U_object.shape}")
    if values.ndim != 1 or values.shape[0] != U_object.shape[0]:
        raise ValueError(
            f"values must be [{U_object.shape[0]}]; got {values.shape}"
        )
    field = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    u = U_object.astype(np.int64)
    field[u[:, 0], u[:, 1], u[:, 2]] = values.astype(np.float32)
    return field


def save_3d_state_html(
    out_root: str,
    iter_idx: int,
    U_object_np: np.ndarray,           # [N, 3] int voxel coords
    g_per_voxel_np: np.ndarray,        # [N] in [0, 1]  presence
    m_per_voxel_np: np.ndarray,        # [N] in [0, 1]  move probability
    joint_axis_world_np: np.ndarray,   # [3] unit vector (TRELLIS world frame)
    joint_origin_world_np: np.ndarray, # [3] world-space point
    joint_type: str,                   # "revolute" or "prismatic"
    resolution: int = TRELLIS_OCC_RES,
    threshold: float = 0.3,
) -> None:
    """Write two interactive HTML files: base/move overlay and axis overlay.

    Files written under ``<out_root>/viz/iter_{it:06d}/``:
        state_base_move.html : voxels colored by g*(1-m) base / g*m move
        state_axis.html      : move voxels + fitted joint axis overlay

    Both are interactive (plotly Scatter3d); the user can rotate/zoom in
    the browser to inspect base/move segmentation and axis alignment.

    The function is a no-op if ``plotly`` is unavailable (silently skipped,
    matching ``voxel_viz`` convention — see ``_VOXEL_VIZ_OK`` import guard).
    """
    if not _VOXEL_VIZ_OK:
        return
    snap_dir = os.path.join(out_root, "viz", f"iter_{iter_idx:06d}")
    os.makedirs(snap_dir, exist_ok=True)

    # Build dense [R, R, R] soft fields for base and move.
    g_field = _build_per_voxel_field(U_object_np, g_per_voxel_np, resolution)
    m_field = _build_per_voxel_field(U_object_np, m_per_voxel_np, resolution)
    base_field = g_field * (1.0 - m_field)        # presence AND not move
    move_field = g_field * m_field                 # presence AND move

    # Two-mask overlay HTML.
    _vxz.save_two_masks_html(
        M_base=base_field,
        M_move=move_field,
        out_path=os.path.join(snap_dir, "state_base_move.html"),
        title=f"iter {iter_idx}: base (orange) / move (blue), thresh={threshold}",
        threshold=threshold,
        resolution=resolution,
    )

    # Axis overlay HTML (only meaningful when move region is non-empty).
    if (move_field > threshold).sum() > 0:
        _vxz.save_axis_overlay_html(
            M_move=move_field,
            joint_type=str(joint_type),
            omega_world=np.asarray(joint_axis_world_np, dtype=np.float64),
            q_world=np.asarray(joint_origin_world_np, dtype=np.float64),
            out_path=os.path.join(snap_dir, "state_axis.html"),
            title=f"iter {iter_idx}: joint axis ({joint_type})",
            threshold=threshold,
            resolution=resolution,
        )


def save_phi_curve_html(
    out_root: str,
    iter_idx: int,
    u_shifted_np: np.ndarray,           # [K=6]
    phi_rev_np: np.ndarray,             # [F=21]
    phi_pri_np: np.ndarray,             # [F=21]
    type_soft: float,                   # sigmoid(type_logit), for plot title
) -> None:
    """Plot the per-frame phi schedule across the 21 frames + 6 control u.

    Interactive plotly line plot (one trace per branch). Useful for spotting
    sign flips, off-center canonical, or insufficient phi range.
    """
    if not _VOXEL_VIZ_OK:
        return
    snap_dir = os.path.join(out_root, "viz", f"iter_{iter_idx:06d}")
    os.makedirs(snap_dir, exist_ok=True)
    # Inline plotly (no helper in voxel_viz for line plots).
    try:
        import plotly.graph_objects as go
    except Exception:
        return

    K = int(u_shifted_np.shape[0])
    F = int(phi_rev_np.shape[0])
    u_x = np.linspace(0, F - 1, K)
    frame_x = np.arange(F)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=u_x.tolist(), y=u_shifted_np.tolist(),
        mode="markers+lines",
        marker=dict(size=10, color="black", symbol="diamond"),
        line=dict(color="rgba(0,0,0,0.4)", dash="dot"),
        name="u_shifted (K control points)",
    ))
    fig.add_trace(go.Scatter(
        x=frame_x.tolist(), y=phi_rev_np.tolist(),
        mode="lines", line=dict(color="rgb(255,30,30)", width=2),
        name="phi_rev (radians)",
    ))
    fig.add_trace(go.Scatter(
        x=frame_x.tolist(), y=phi_pri_np.tolist(),
        mode="lines", line=dict(color="rgb(0,122,255)", width=2),
        name="phi_pri (world units)",
    ))
    fig.update_layout(
        title=f"iter {iter_idx}: phi rollout (type_soft = {type_soft:.3f})",
        xaxis_title="frame index (0..20)",
        yaxis_title="signed angle / displacement",
        hovermode="x unified",
    )
    out_path = os.path.join(snap_dir, "phi_curve.html")
    try:
        fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    except Exception:
        # Silently skip; npz is still present via save_phi_curve.
        pass


# =============================================================================
# End-of-P1 final summary
# =============================================================================

def save_p1_final_summary(
    out_root: str,
    final_rgb_T3HW_01: torch.Tensor,
    wan_target_T3HW_01: torch.Tensor,
    U_object_np: np.ndarray,
    g_per_voxel_np: np.ndarray,
    m_per_voxel_np: np.ndarray,
    joint_axis_world_np: np.ndarray,
    joint_origin_world_np: np.ndarray,
    committed_type: str,
    summary: Dict[str, float],
    resolution: int = TRELLIS_OCC_RES,
) -> None:
    """Write end-of-P1 deliverables under ``<out_root>/viz/p1_final/``.

    Files:
        final_rendered.png       21-frame strip of final P1 render
        final_diff.png           pixel difference vs Wan target (x4 contrast)
        final_base_move.html     committed base/move 3D viz
        final_axis.html          committed joint axis viz
        summary.json             dict of (committed_type, losses, IoU, ...)
    """
    snap_dir = os.path.join(out_root, "viz", "p1_final")
    os.makedirs(snap_dir, exist_ok=True)

    # Final RGB strips.
    Image.fromarray(_frames_to_strip(final_rgb_T3HW_01)).save(
        os.path.join(snap_dir, "final_rendered.png")
    )
    Image.fromarray(_frames_to_strip(wan_target_T3HW_01)).save(
        os.path.join(snap_dir, "final_target.png")
    )
    diff = (final_rgb_T3HW_01 - wan_target_T3HW_01).abs().clamp(0.0, 0.25) * 4.0
    Image.fromarray(_frames_to_strip(diff)).save(
        os.path.join(snap_dir, "final_diff.png")
    )

    # 3D state HTMLs (reuse save_3d_state_html by writing to a different dir).
    if _VOXEL_VIZ_OK:
        g_field = _build_per_voxel_field(U_object_np, g_per_voxel_np, resolution)
        m_field = _build_per_voxel_field(U_object_np, m_per_voxel_np, resolution)
        base_field = g_field * (1.0 - m_field)
        move_field = g_field * m_field
        _vxz.save_two_masks_html(
            M_base=base_field, M_move=move_field,
            out_path=os.path.join(snap_dir, "final_base_move.html"),
            title=f"P1 final: base (orange) / move (blue), committed={committed_type}",
            threshold=0.3, resolution=resolution,
        )
        if (move_field > 0.3).sum() > 0:
            _vxz.save_axis_overlay_html(
                M_move=move_field, joint_type=committed_type,
                omega_world=np.asarray(joint_axis_world_np, dtype=np.float64),
                q_world=np.asarray(joint_origin_world_np, dtype=np.float64),
                out_path=os.path.join(snap_dir, "final_axis.html"),
                title=f"P1 final axis ({committed_type})",
                threshold=0.3, resolution=resolution,
            )

    # Summary JSON.
    with open(os.path.join(snap_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)


# =============================================================================
# Training loss curves (from logs.jsonl)
# =============================================================================

# Default keys to plot. Aggregator writes L_sds, L_lat, L_rgb, L_first,
# L_contact, L_gate, L_shell, L_m_prior, L_z, L_total (see losses.py).
_DEFAULT_LOSS_KEYS: tuple = (
    "L_total", "L_sds", "L_first", "L_rgb", "L_contact",
    "L_gate", "L_shell", "L_m_prior", "L_z",
)

# Schedule keys (logged every cfg.log_every iter alongside losses).
_DEFAULT_SCHED_KEYS: tuple = (
    "cfg_scale", "T_g", "T_m",
    "lambda_sup", "lambda_part", "lambda_joint",
    "lambda_sds", "lambda_rgb", "lambda_shell",
)


def save_loss_curves_html(
    out_root: str,
    logs_jsonl_path: Optional[str] = None,
    loss_keys: tuple = _DEFAULT_LOSS_KEYS,
    sched_keys: tuple = _DEFAULT_SCHED_KEYS,
    log_y_for_losses: bool = True,
) -> None:
    """Parse ``logs.jsonl`` and render interactive curve HTML files.

    Reads the JSON-lines training log (one record per ``cfg.log_every``
    iter) produced by ``train_stage_d_p1`` and writes:

        viz/p1_final/loss_curves.html      one trace per loss component
        viz/p1_final/schedule_curves.html  CFG / T_g / T_m / lambdas vs iter

    Both files are interactive plotly HTMLs; hover shows exact values and
    click-on-legend toggles individual traces. Useful for spotting:
      - loss spikes (NaN, gradient explosion, schedule mis-tuned)
      - phase transitions (lambda ramps -> losses should respond)
      - whether L_first / L_contact actually decrease across training

    If plotly is unavailable, silently skips (matches voxel_viz convention).

    Parameters
    ----------
    out_root : str             Stage D ``out_dir``.
    logs_jsonl_path : Optional[str]
        Path to logs.jsonl. Defaults to ``<out_root>/logs.jsonl``.
    loss_keys : tuple of str
        Loss components to plot. Missing keys are silently skipped.
    sched_keys : tuple of str
        Schedule values to plot in a separate HTML.
    log_y_for_losses : bool    Use log-scale Y axis for losses (helpful
        because L_total ranges several orders of magnitude across phases).
    """
    if not _VOXEL_VIZ_OK:
        return
    try:
        import plotly.graph_objects as go
    except Exception:
        return

    if logs_jsonl_path is None:
        logs_jsonl_path = os.path.join(out_root, "logs.jsonl")
    if not os.path.isfile(logs_jsonl_path):
        return

    # Parse JSON lines (skip malformed lines silently; training log may be
    # mid-write if called concurrently).
    records: list = []
    with open(logs_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return

    iters = [int(r.get("it", i)) for i, r in enumerate(records)]
    snap_dir = os.path.join(out_root, "viz", "p1_final")
    os.makedirs(snap_dir, exist_ok=True)

    # ---- Loss curves ----
    fig_loss = go.Figure()
    for key in loss_keys:
        ys = [r.get(key) for r in records]
        # Drop None entries so plotly doesn't break the trace.
        x_clean = [it for it, y in zip(iters, ys) if y is not None]
        y_clean = [y for y in ys if y is not None]
        if not y_clean:
            continue
        # Clamp log-y to a small positive to avoid log(0) issues.
        if log_y_for_losses:
            y_clean = [max(float(y), 1.0e-8) for y in y_clean]
        fig_loss.add_trace(go.Scatter(
            x=x_clean, y=y_clean, mode="lines", name=key,
            hovertemplate=f"{key}: %{{y:.4g}}<br>iter %{{x}}<extra></extra>",
        ))
    fig_loss.update_layout(
        title="Stage D P1 loss components vs iter",
        xaxis_title="iter",
        yaxis_title="loss value" + (" (log)" if log_y_for_losses else ""),
        yaxis_type="log" if log_y_for_losses else "linear",
        hovermode="x unified",
    )
    try:
        fig_loss.write_html(
            os.path.join(snap_dir, "loss_curves.html"),
            include_plotlyjs="cdn", full_html=True,
        )
    except Exception:
        pass

    # ---- Schedule curves ----
    fig_sched = go.Figure()
    for key in sched_keys:
        ys = [r.get(key) for r in records]
        x_clean = [it for it, y in zip(iters, ys) if y is not None]
        y_clean = [float(y) for y in ys if y is not None]
        if not y_clean:
            continue
        fig_sched.add_trace(go.Scatter(
            x=x_clean, y=y_clean, mode="lines", name=key,
            hovertemplate=f"{key}: %{{y:.4g}}<br>iter %{{x}}<extra></extra>",
        ))
    fig_sched.update_layout(
        title="Stage D P1 schedule values vs iter",
        xaxis_title="iter",
        yaxis_title="schedule value",
        hovermode="x unified",
    )
    try:
        fig_sched.write_html(
            os.path.join(snap_dir, "schedule_curves.html"),
            include_plotlyjs="cdn", full_html=True,
        )
    except Exception:
        pass


__all__ = [
    "save_iter_snapshot", "save_phi_curve",
    "save_3d_state_html", "save_phi_curve_html",
    "save_p1_final_summary",
    "save_loss_curves_html",
]

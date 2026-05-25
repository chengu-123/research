"""Joint type detection via geometric fit (per user direction).

Replaces v8.1's BIC loss-ratio (which needed full Phase-EM, ~30 sec)
with a direct geometric fit on per-state move centroids (~1 sec):

  centroid_k = centroid of state k's move voxels (world space)
  residual_line = sum_k dist(centroid_k, best_fit_line)^2
  residual_arc  = sum_k dist(centroid_k, best_fit_circle)^2
  type_logit = log(residual_arc + eps) - log(residual_line + eps)
               > 0 -> linear fit better -> prismatic
               < 0 -> arc fit better    -> revolute

Confidence is derived from the separation margin between the two residuals,
normalised against the centroid spread length scale.

method/pipeline.md spec: this function consumes Stage B v3.3.6's
O_move_per_state directly; no Phase-EM dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from .move_geometry import (
    PerStateMoveGeom,
    centroid_trajectory_length,
    overall_move_extent,
    valid_centroid_subset,
)


# ---------------------------------------------------------------------------
# Geometric fits (numpy, no autograd needed)
# ---------------------------------------------------------------------------


def fit_line_3d_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Best-fit 3D line via PCA.

    points : (n, 3)
    Returns:
        origin    : (3,) line passes through this point (centroid of input)
        direction : (3,) unit vector along principal axis
        residual  : float, mean squared perpendicular distance to line
    """
    if points.shape[0] < 2:
        raise ValueError("fit_line_3d_pca needs >= 2 points")
    centroid = points.mean(axis=0)
    centred = points - centroid
    # PCA: largest singular vector of centred = line direction
    # SVD: centred = U S Vt; first row of Vt = principal axis
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    direction = vt[0]
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    # Residual: perpendicular distance from each point to line
    proj = (centred @ direction)[:, None] * direction[None, :]
    perp = centred - proj
    residual = float((perp ** 2).sum(axis=-1).mean())
    return centroid.astype(np.float64), direction.astype(np.float64), residual


def fit_circle_3d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Best-fit 3D circle to >=3 points.

    Algorithm:
        1) Fit best-fit plane via PCA (normal = smallest-variance axis).
        2) Project points into that plane (2D coords in plane basis).
        3) Algebraic least-squares circle fit in 2D:
              min over (a, b, r) of sum [(x_i-a)^2 + (y_i-b)^2 - r^2]^2
           -> linear system in (a, b, c=a^2+b^2-r^2):
              2*a*x_i + 2*b*y_i - c = x_i^2 + y_i^2
        4) Lift 2D centre back to 3D.

    points : (n, 3); n >= 3
    Returns:
        center_3d  : (3,) circle centre in 3D
        normal_3d  : (3,) unit vector normal to circle plane (rotation axis)
        radius     : float
        residual   : float, mean squared 3D distance from point to circle
                     ( = (in-plane radial error)^2 + (out-of-plane error)^2 )
    """
    n = points.shape[0]
    if n < 3:
        raise ValueError("fit_circle_3d needs >= 3 points")
    # 1) plane fit via PCA
    centroid = points.mean(axis=0)
    centred = points - centroid
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    # vt rows = principal axes; smallest variance = plane normal (last)
    normal = vt[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    # Two in-plane basis vectors = first two rows of vt (orthonormal)
    e1 = vt[0] / (np.linalg.norm(vt[0]) + 1e-12)
    e2 = vt[1] / (np.linalg.norm(vt[1]) + 1e-12)

    # 2) project to 2D in (e1, e2) basis
    pts2d = np.stack([centred @ e1, centred @ e2], axis=1)        # (n, 2)
    x = pts2d[:, 0]
    y = pts2d[:, 1]

    # 3) algebraic LS circle fit
    A = np.stack([2.0 * x, 2.0 * y, -np.ones(n)], axis=1)         # (n, 3)
    b = x * x + y * y                                              # (n,)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a2d, b2d, c = sol[0], sol[1], sol[2]
    r2 = a2d * a2d + b2d * b2d - c
    radius = float(np.sqrt(max(r2, 1e-12)))

    # 4) lift centre back to 3D
    center_3d = centroid + a2d * e1 + b2d * e2

    # Residual: in-plane radial error squared + out-of-plane distance squared
    in_plane = np.sqrt((x - a2d) ** 2 + (y - b2d) ** 2) - radius     # (n,)
    out_plane = centred @ normal                                      # (n,)
    residual = float(np.mean(in_plane ** 2 + out_plane ** 2))
    return (
        center_3d.astype(np.float64),
        normal.astype(np.float64),
        radius,
        residual,
    )


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------


@dataclass
class JointTypeResult:
    type_str: str                  # "prismatic" | "revolute" | "uncertain"
    type_logit: float              # raw log-ratio (positive -> prismatic)
    confidence: float              # in [0, 1]
    residual_line: float
    residual_arc: float
    line_origin: np.ndarray        # (3,) world space
    line_direction: np.ndarray     # (3,) unit
    arc_center: Optional[np.ndarray]    # (3,) world space, None if arc fit unavailable
    arc_normal: Optional[np.ndarray]    # (3,) unit
    arc_radius: Optional[float]
    n_valid_states: int            # how many centroids were usable


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def detect_joint_type(
    geom: PerStateMoveGeom,
    type_decision_margin: float = 0.15,
    arc_min_states: int = 4,
    eps: float = 1e-8,
) -> JointTypeResult:
    """Decide joint type from per-state move centroids.

    Logic:
      1. Need >= 2 valid centroids to even fit a line. If less, return
         "uncertain" with confidence=0 (Stage D will dual-clone branch).
      2. Need >= arc_min_states centroids for arc fit; if fewer, only line fit
         available -> default prismatic with confidence depending on line
         residual / length-scale ratio.
      3. Otherwise fit both, compute type_logit = log(arc) - log(line),
         convert margin to confidence in [0, 1].

    The length scale used for confidence normalisation is the centroid
    trajectory length (sum of consecutive pairwise distances), which is the
    natural "how far did things move" magnitude.

    Parameters mirror StageCConfig.
    """
    cents_t, ks = valid_centroid_subset(geom)
    n_valid = int(cents_t.shape[0])

    if n_valid < 2:
        # Not enough data to fit anything
        return JointTypeResult(
            type_str="uncertain",
            type_logit=0.0,
            confidence=0.0,
            residual_line=float("inf"),
            residual_arc=float("inf"),
            line_origin=np.zeros(3, dtype=np.float64),
            line_direction=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            arc_center=None,
            arc_normal=None,
            arc_radius=None,
            n_valid_states=n_valid,
        )

    points = cents_t.detach().cpu().numpy().astype(np.float64)
    length_scale = max(centroid_trajectory_length(geom), eps)

    line_origin, line_dir, res_line = fit_line_3d_pca(points)

    if n_valid < arc_min_states:
        # Arc fit not reliable -- default to prismatic.
        # Confidence = how well the line fits relative to overall move size.
        norm_res = res_line / (length_scale * length_scale + eps)
        # norm_res small -> confident prismatic; large -> uncertain
        conf = float(np.exp(-norm_res * 50.0))
        return JointTypeResult(
            type_str="prismatic",
            type_logit=type_decision_margin * 2.0,   # mild prismatic bias
            confidence=max(0.0, min(conf, 1.0)),
            residual_line=res_line,
            residual_arc=float("inf"),
            line_origin=line_origin,
            line_direction=line_dir,
            arc_center=None,
            arc_normal=None,
            arc_radius=None,
            n_valid_states=n_valid,
        )

    # Full case: both fits available
    arc_center, arc_normal, arc_radius, res_arc = fit_circle_3d(points)

    # Normalised log-ratio: log(res_arc/L^2 + eps) - log(res_line/L^2 + eps)
    # Equivalently: log(res_arc) - log(res_line) (L^2 cancels)
    type_logit = float(np.log(res_arc + eps) - np.log(res_line + eps))

    # Confidence from margin |type_logit| relative to decision threshold
    # confidence = clamp((|type_logit| - margin/2) / margin, 0, 1) + small offset
    abs_logit = abs(type_logit)
    if abs_logit < type_decision_margin * 0.5:
        # Near-zero margin: residuals are comparable -> low confidence
        conf = float(abs_logit / max(type_decision_margin, eps))
        type_str = "uncertain"
    else:
        # Beyond margin: confidence saturates as logit grows
        # Saturate around 3*margin -> conf ~ 0.95
        conf = float(1.0 - np.exp(-abs_logit / max(type_decision_margin, eps)))
        type_str = "prismatic" if type_logit > 0.0 else "revolute"

    # Also weight by absolute residual quality (both residuals should be
    # small relative to length-scale; if both are large, neither fits well).
    min_res = min(res_line, res_arc)
    norm_min_res = min_res / (length_scale * length_scale + eps)
    fit_quality = float(np.exp(-norm_min_res * 50.0))   # 1 when good fit, ->0 when bad
    conf = conf * fit_quality

    conf = max(0.0, min(conf, 1.0))

    return JointTypeResult(
        type_str=type_str,
        type_logit=type_logit,
        confidence=conf,
        residual_line=res_line,
        residual_arc=res_arc,
        line_origin=line_origin,
        line_direction=line_dir,
        arc_center=arc_center,
        arc_normal=arc_normal,
        arc_radius=arc_radius,
        n_valid_states=n_valid,
    )

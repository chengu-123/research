"""Axis + origin fit for revolute / prismatic joints.

Given the joint type detection result, produce a final (axis, origin) pair
in world space ready to package into psi_0:

- prismatic: axis = line direction (already from joint_type_detect line fit)
             origin = line_origin (any point on the axis; we use the
                      centroid-of-centroids for stability)
- revolute:  axis = circle plane normal (sign-corrected to point along
                    positive rotation direction inferred from state order)
             origin = circle centre

A fallback path uses motion-corridor PCA when centroid-based fits fail
(e.g., only 2-3 valid states or arc fit residual is huge).

All outputs are world-space float tensors. Units: world ([-0.5, 0.5] cube).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .joint_type_detect import JointTypeResult
from .move_geometry import (
    PerStateMoveGeom,
    valid_centroid_subset,
    voxel_to_world,
)


@dataclass
class AxisResult:
    axis: torch.Tensor          # (3,) unit vector, world space
    origin: torch.Tensor        # (3,) world space, on axis line for revolute
    fit_source: str             # 'centroid_circle' | 'centroid_line' | 'corridor_pca'
    fit_residual: float         # mean squared distance (world units squared)


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        # Pathological: return canonical x-axis as a safe fallback
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return v / n


def _sign_correct_revolute_axis(
    arc_center: np.ndarray,
    arc_normal: np.ndarray,
    centroids_ordered: np.ndarray,    # (n_valid, 3) sorted by state index
) -> np.ndarray:
    """Flip the arc normal so that the rotation from state k to state k+1
    is positive (right-hand rule).

    For consecutive points p_k, p_{k+1} on the circle:
      v_k = p_k - center
      v_{k+1} = p_{k+1} - center
    Positive rotation direction = normalize(v_k x v_{k+1})  -- this should
    align with arc_normal. If on average it points the OPPOSITE way, flip
    arc_normal.

    Returns the (possibly flipped) unit normal.
    """
    n_pts = centroids_ordered.shape[0]
    if n_pts < 2:
        return _normalize(arc_normal)

    # Accumulate cross products of consecutive radius vectors
    vec = centroids_ordered - arc_center[None, :]
    cross_sum = np.zeros(3, dtype=np.float64)
    for i in range(n_pts - 1):
        cross_sum = cross_sum + np.cross(vec[i], vec[i + 1])
    cross_dir = _normalize(cross_sum)

    if np.dot(cross_dir, arc_normal) < 0.0:
        return -_normalize(arc_normal)
    return _normalize(arc_normal)


def _corridor_pca_axis(
    M_motion_corridor_64: torch.Tensor,
    threshold: float,
    res: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fallback: principal axis of high-motion-corridor voxels."""
    mask = M_motion_corridor_64 > threshold
    if int(mask.sum().item()) < 4:
        # Pathological: too few corridor voxels -> return canonical x-axis through origin
        return (
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            float("inf"),
        )
    idx = torch.nonzero(mask, as_tuple=False)              # (n, 3) int
    pts = voxel_to_world(idx, res=res).detach().cpu().numpy().astype(np.float64)

    centroid = pts.mean(axis=0)
    centred = pts - centroid
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    direction = _normalize(vt[0])
    proj = (centred @ direction)[:, None] * direction[None, :]
    perp = centred - proj
    residual = float((perp ** 2).sum(axis=-1).mean())
    return direction, centroid, residual


def fit_axis_origin(
    type_result: JointTypeResult,
    geom: PerStateMoveGeom,
    M_motion_corridor_64: Optional[torch.Tensor] = None,
    corridor_pca_threshold: float = 0.1,
    res: int = 64,
    arc_fit_residual_reject: float = 0.05,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> AxisResult:
    """Resolve (axis, origin) from the type detection + geometry.

    Priority:
      revolute:
        1. If arc_center + arc_normal available AND residual < threshold,
           use those (with sign correction from state ordering).
        2. Else corridor PCA fallback (axis from PCA, origin = corridor centroid).
        3. Else canonical x-axis (sentinel; should not happen with v3.3.6).
      prismatic:
        1. Use line_direction + a point on the line.
           Origin choice: project the geometric centroid of move centroids
           onto the line -> stable point used as ψ.origin.
        2. Else corridor PCA fallback.
    """
    if device is None:
        device = (
            geom.centroid_world.device
            if geom.centroid_world.numel() > 0
            else torch.device("cpu")
        )

    type_str = type_result.type_str
    cents_t, ks = valid_centroid_subset(geom)
    n_valid = int(cents_t.shape[0])

    axis_np: np.ndarray
    origin_np: np.ndarray
    fit_source: str
    fit_residual: float

    if type_str == "revolute" and type_result.arc_center is not None \
            and type_result.residual_arc < arc_fit_residual_reject:
        # Path 1: centroid-based circle fit
        # Sort centroids by state index for proper sign correction
        order = torch.argsort(ks).detach().cpu().numpy()
        cents_np = cents_t.detach().cpu().numpy().astype(np.float64)
        cents_sorted = cents_np[order]
        axis_np = _sign_correct_revolute_axis(
            type_result.arc_center, type_result.arc_normal, cents_sorted,
        )
        origin_np = type_result.arc_center.astype(np.float64)
        fit_source = "centroid_circle"
        fit_residual = float(type_result.residual_arc)

    elif type_str == "prismatic" and n_valid >= 2:
        # Path 2: centroid-based line fit
        axis_np = _normalize(type_result.line_direction.astype(np.float64))
        # Origin = centroid of centroids projected onto line (lies on line by construction)
        cents_np = cents_t.detach().cpu().numpy().astype(np.float64)
        c_mean = cents_np.mean(axis=0)
        # project c_mean onto line through type_result.line_origin along axis_np
        t = float(np.dot(c_mean - type_result.line_origin, axis_np))
        origin_np = type_result.line_origin.astype(np.float64) + t * axis_np
        fit_source = "centroid_line"
        fit_residual = float(type_result.residual_line)

    else:
        # Fallback: corridor PCA (only works for prismatic-like swept volumes;
        # for revolute with bad data this still gives a reasonable axis hint)
        if M_motion_corridor_64 is None:
            # No corridor available -> last-resort canonical fallback
            axis_np = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            origin_np = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            fit_source = "canonical_fallback"
            fit_residual = float("inf")
        else:
            axis_np, origin_np, fit_residual = _corridor_pca_axis(
                M_motion_corridor_64, corridor_pca_threshold, res,
            )
            fit_source = "corridor_pca"

    axis_t = torch.tensor(axis_np, device=device, dtype=dtype)
    origin_t = torch.tensor(origin_np, device=device, dtype=dtype)
    return AxisResult(
        axis=axis_t,
        origin=origin_t,
        fit_source=fit_source,
        fit_residual=fit_residual,
    )

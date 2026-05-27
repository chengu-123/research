"""Axis + origin extraction from v3 joint_type_detect.

v3 simplification: joint_type_detect.detect_joint_type_v3 ALREADY enumerates
6 cardinal axes per type and selects the best by voxel-physical score. This
module is now a thin adapter that:

  - Reads the chosen primary candidate from JointTypeResult
  - Packages (axis, origin) as an AxisResult for downstream confidence + viz
  - Provides a fallback for the "uncertain" type case (use highest-score
    candidate regardless of type)

The old centroid-circle/line fit paths are GONE; cardinal enumeration +
voxel-level reverse-warp scoring is the only path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .joint_type_detect import CandidateResult, JointTypeResult


@dataclass
class AxisResult:
    axis: torch.Tensor          # (3,) unit, world space (cardinal +/-X/Y/Z)
    origin: torch.Tensor        # (3,) world space, on axis line for revolute
    fit_source: str             # 'cardinal_v3_pris' | 'cardinal_v3_rev' | 'cardinal_v3_uncertain'
    fit_residual: float         # 1 - candidate.score.score (lower = better)


def fit_axis_origin(
    type_result: JointTypeResult,
    geom=None,                                          # unused in v3, kept for API compat
    M_motion_corridor_64: Optional[torch.Tensor] = None,  # unused
    corridor_pca_threshold: float = 0.1,                  # unused
    res: int = 64,
    arc_fit_residual_reject: float = 0.05,                # unused
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> AxisResult:
    """Extract axis/origin from JointTypeResult's selected primary candidate.

    v3 logic:
      type=prismatic -> use best_pris
      type=revolute  -> use best_rev
      type=uncertain -> pick the higher-scoring of best_pris vs best_rev
                        (Stage D dual-clone will check both anyway)
    """
    if type_result.best_pris is None and type_result.best_rev is None:
        # Pathological -- joint_type_detect failed to produce any candidate.
        # Return canonical fallback so Stage D dual-clone can still run.
        if device is None:
            device = torch.device("cpu")
        return AxisResult(
            axis=torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype),
            origin=torch.zeros(3, device=device, dtype=dtype),
            fit_source="cardinal_v3_empty",
            fit_residual=float("inf"),
        )

    if type_result.type_str == "prismatic":
        chosen = type_result.best_pris
        source = "cardinal_v3_pris"
    elif type_result.type_str == "revolute":
        chosen = type_result.best_rev
        source = "cardinal_v3_rev"
    else:
        # Uncertain: take higher score; both candidates exist by this point
        if type_result.best_pris is not None and (
            type_result.best_rev is None
            or type_result.best_pris.score.score >= type_result.best_rev.score.score
        ):
            chosen = type_result.best_pris
        else:
            chosen = type_result.best_rev
        source = "cardinal_v3_uncertain"

    # chosen should now be non-None due to the empty-case guard above
    if chosen is None:
        # Fallback (shouldn't happen due to earlier guard)
        if device is None:
            device = torch.device("cpu")
        return AxisResult(
            axis=torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype),
            origin=torch.zeros(3, device=device, dtype=dtype),
            fit_source="cardinal_v3_empty",
            fit_residual=float("inf"),
        )

    if device is not None:
        axis_t = chosen.axis.to(device=device, dtype=dtype)
        origin_t = chosen.origin.to(device=device, dtype=dtype)
    else:
        axis_t = chosen.axis.to(dtype=dtype)
        origin_t = chosen.origin.to(dtype=dtype)

    # fit_residual := 1 - score for compatibility with downstream confidence
    fit_residual = float(max(0.0, 1.0 - chosen.score.score))

    return AxisResult(
        axis=axis_t,
        origin=origin_t,
        fit_source=source,
        fit_residual=fit_residual,
    )


__all__ = ["AxisResult", "fit_axis_origin"]

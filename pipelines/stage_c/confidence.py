"""Confidence aggregation.

Per user direction "总是返回 + 加 confidence 字段": Stage C always returns
a JointInit (never raises NoArticulationError); the `confidence` field in
[0, 1] tells Stage D how much to trust this init.

The overall confidence is a weighted combination of 4 sub-confidences:

  - type_conf:     from joint_type_detect (residual margin between line / arc)
  - axis_conf:     from axis_fit  (1 / (1 + normalised_residual))
  - centroid_conf: from move_geometry (n_valid / K) * (1 - non_monotone_fraction)
  - base_conf:     from O_base_canonical count vs expected object size

Weights are configurable via StageCConfig (default 0.30 / 0.30 / 0.20 / 0.20).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .axis_fit import AxisResult
from .joint_type_detect import JointTypeResult
from .move_geometry import (
    PerStateMoveGeom,
    centroid_trajectory_length,
    overall_move_extent,
)
from .phi_fit import PhiResult


@dataclass
class ConfidenceBundle:
    overall: float
    type_conf: float
    axis_conf: float
    centroid_conf: float
    base_conf: float
    notes: str = ""


def _axis_residual_to_conf(residual: float, length_scale: float, eps: float = 1e-9) -> float:
    """Map axis fit residual to [0, 1]. Smaller residual = higher conf."""
    if not np.isfinite(residual):
        return 0.0
    norm = residual / (length_scale * length_scale + eps)
    # exp(-50 * norm) -> 1 at norm=0, 0.6 at norm=0.01, ~0.01 at norm=0.09
    return float(np.exp(-50.0 * norm))


def _centroid_conf(geom: PerStateMoveGeom, phi: Optional[PhiResult]) -> float:
    """Centroid quality: (n_valid / K) penalised by monotonicity enforcement."""
    K = int(geom.valid_mask.shape[0])
    n_valid = int(geom.valid_mask.sum().item())
    if K == 0:
        return 0.0
    completeness = n_valid / K
    if phi is None:
        return completeness
    mono_penalty = 0.7 if phi.monotone_enforced else 1.0
    return float(completeness * mono_penalty)


def _base_conf(
    O_base_canonical: Optional[torch.Tensor],
    P_base_canonical: Optional[torch.Tensor],
    res: int = 64,
    expected_min_voxels: int = 500,
    expected_max_voxels: int = 8000,
) -> float:
    """Crude proxy: does O_base_canonical have a plausible voxel count?

    Empirically v3.3.6 produces 2000-5000 base voxels for typical objects.
    Outside [500, 8000] suggests Stage B failure and we down-weight Stage C
    confidence.
    """
    if O_base_canonical is not None:
        n = int((O_base_canonical > 0).sum().item())
    elif P_base_canonical is not None:
        n = int((P_base_canonical > 0.5).sum().item())
    else:
        return 0.5  # no info -> neutral

    if n < expected_min_voxels:
        return float(max(0.0, n / max(expected_min_voxels, 1)))
    if n > expected_max_voxels:
        # Too many base voxels -> object may have absorbed move
        return float(max(0.0, expected_max_voxels / max(n, 1)))
    return 1.0


def aggregate_confidence(
    type_result: JointTypeResult,
    axis_result: AxisResult,
    geom: PerStateMoveGeom,
    phi_result: PhiResult,
    O_base_canonical: Optional[torch.Tensor] = None,
    P_base_canonical: Optional[torch.Tensor] = None,
    w_type: float = 0.30,
    w_axis: float = 0.30,
    w_centroid: float = 0.20,
    w_base: float = 0.20,
    res: int = 64,
) -> ConfidenceBundle:
    length_scale = max(centroid_trajectory_length(geom), overall_move_extent(geom), 1e-6)

    c_type = float(type_result.confidence)
    c_axis = _axis_residual_to_conf(axis_result.fit_residual, length_scale)
    c_cent = _centroid_conf(geom, phi_result)
    c_base = _base_conf(O_base_canonical, P_base_canonical, res=res)

    w_total = w_type + w_axis + w_centroid + w_base
    if w_total <= 0.0:
        overall = 0.0
    else:
        overall = (
            w_type * c_type + w_axis * c_axis + w_centroid * c_cent + w_base * c_base
        ) / w_total

    notes = ""
    if axis_result.fit_source == "canonical_fallback":
        notes += " axis_fallback=canonical;"
        overall = min(overall, 0.1)
    elif axis_result.fit_source == "corridor_pca":
        notes += " axis_fallback=corridor_pca;"
        overall = overall * 0.7
    if type_result.type_str == "uncertain":
        notes += " type=uncertain;"

    overall = float(max(0.0, min(1.0, overall)))

    return ConfidenceBundle(
        overall=overall,
        type_conf=c_type,
        axis_conf=c_axis,
        centroid_conf=c_cent,
        base_conf=c_base,
        notes=notes.strip(),
    )

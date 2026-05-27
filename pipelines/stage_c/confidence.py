"""Confidence aggregation for v3 Stage C output.

v3 confidence is derived from the cardinal-candidate voxel-scoring framework:

  - type_conf:     from JointTypeResult.confidence (clamped |type_logit| margin)
  - axis_conf:     from selected CandidateResult.score.consistency
                   (higher = warped fragments align better in canonical)
  - centroid_conf: from valid_state count / K
  - base_conf:     from O_base_canonical voxel count plausibility

The legacy centroid-monotonicity penalty is removed (v3 enforces monotone via
PAV by construction, so monotone_enforced is always feasible).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .axis_fit import AxisResult
from .joint_type_detect import JointTypeResult
from .phi_fit import PhiResult


@dataclass
class ConfidenceBundle:
    overall: float
    type_conf: float
    axis_conf: float
    centroid_conf: float
    base_conf: float
    notes: str = ""


def _state_completeness(valid_state, K: int) -> float:
    """Fraction of states that contributed valid evidence."""
    if K <= 0:
        return 0.0
    if isinstance(valid_state, (list, tuple)):
        return float(sum(bool(v) for v in valid_state)) / float(K)
    if isinstance(valid_state, torch.Tensor):
        return float(valid_state.float().mean().item())
    return 0.5


def _base_conf(
    O_base_canonical: Optional[torch.Tensor],
    P_base_canonical: Optional[torch.Tensor],
    expected_min_voxels: int = 500,
    expected_max_voxels: int = 8000,
) -> float:
    if O_base_canonical is not None:
        n = int((O_base_canonical > 0).sum().item())
    elif P_base_canonical is not None:
        n = int((P_base_canonical > 0.5).sum().item())
    else:
        return 0.5
    if n < expected_min_voxels:
        return float(max(0.0, n / max(expected_min_voxels, 1)))
    if n > expected_max_voxels:
        return float(max(0.0, expected_max_voxels / max(n, 1)))
    return 1.0


def aggregate_confidence(
    type_result: JointTypeResult,
    axis_result: AxisResult,
    geom=None,                              # unused in v3 (legacy)
    phi_result: Optional[PhiResult] = None, # unused for centroid_conf in v3
    O_base_canonical: Optional[torch.Tensor] = None,
    P_base_canonical: Optional[torch.Tensor] = None,
    valid_state=None,                       # v3 input: list of K bool
    K: int = 6,
    w_type: float = 0.35,
    w_axis: float = 0.35,
    w_centroid: float = 0.15,
    w_base: float = 0.15,
    res: int = 64,
) -> ConfidenceBundle:
    """Aggregate to a single [0, 1] confidence + per-component breakdown."""

    c_type = float(type_result.confidence)

    # axis_conf: use the selected candidate's voxel-physical consistency
    if type_result.type_str == "prismatic" and type_result.best_pris is not None:
        selected = type_result.best_pris
    elif type_result.type_str == "revolute" and type_result.best_rev is not None:
        selected = type_result.best_rev
    elif type_result.best_pris is not None and type_result.best_rev is not None:
        # uncertain: pick higher
        selected = (
            type_result.best_pris
            if type_result.best_pris.score.score >= type_result.best_rev.score.score
            else type_result.best_rev
        )
    else:
        selected = type_result.best_pris or type_result.best_rev

    if selected is not None:
        c_axis = float(selected.score.consistency)
        contact_compat = float(selected.score.contact_compat)
        # Penalize if conflict (base intersection) high
        conflict_penalty = 1.0 - float(selected.score.conflict)
        c_axis = c_axis * conflict_penalty
        # Reward contact-compatible axis
        c_axis = c_axis * contact_compat
    else:
        c_axis = 0.0

    c_cent = _state_completeness(valid_state, K) if valid_state is not None else 0.5
    c_base = _base_conf(O_base_canonical, P_base_canonical)

    w_total = w_type + w_axis + w_centroid + w_base
    if w_total <= 0.0:
        overall = 0.0
    else:
        overall = (
            w_type * c_type + w_axis * c_axis + w_centroid * c_cent + w_base * c_base
        ) / w_total

    notes = ""
    if axis_result.fit_source == "cardinal_v3_uncertain":
        notes += " type_uncertain;"
        overall = overall * 0.7
    elif axis_result.fit_source == "cardinal_v3_empty":
        notes += " axis_empty_fallback;"
        overall = min(overall, 0.1)
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


__all__ = ["ConfidenceBundle", "aggregate_confidence"]

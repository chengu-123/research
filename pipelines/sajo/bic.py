"""Dual-model BIC selection for SAJO.

Runs both the revolute and prismatic EMs and picks the joint type with
the lower BIC score. ``L_data_final`` is taken from the unweighted warp
data term at convergence (see spec §2.8), excluding the anchor and
prior terms — this is the likelihood-proxy standard for BIC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from .em import EMHParams, EMResult, em_prismatic, em_revolute


@dataclass
class BICResult:
    joint_type: str
    bic_rev: float
    bic_pris: float
    confidence: Optional[float]
    N: int
    k_rev: int
    k_pris: int
    em_rev: EMResult = field(default_factory=lambda: None)  # type: ignore
    em_pris: EMResult = field(default_factory=lambda: None)  # type: ignore
    selected: EMResult = field(default_factory=lambda: None)  # type: ignore


def _count_active_voxels(p_move: torch.Tensor, thresh: float) -> int:
    return int((p_move > thresh).sum().item())


def run_dual_em_and_select(
    O_stack: torch.Tensor,
    p_base: torch.Tensor,
    p_move: torch.Tensor,
    anchor_coords: torch.Tensor,
    anchor_weights: torch.Tensor,
    rev_init: Dict[str, torch.Tensor],
    pris_init: Dict[str, torch.Tensor],
    hp: EMHParams,
    k_rev_const: int = 4,
    k_pris_const: int = 2,
    min_N: int = 50,
    resolution: int = 64,
) -> BICResult:
    """Run both EMs sequentially and return the BIC-selected result.

    ``k_rev = k_rev_const + K`` (per-state rotation magnitudes)
    ``k_pris = k_pris_const + K`` (per-state translation magnitudes)
    ``N = K * |{x : p_move(x) > active_voxel_thresh}|``
    """
    K = O_stack.shape[0]

    em_rev = em_revolute(
        O_stack, p_base, p_move,
        anchor_coords, anchor_weights,
        rev_init, hp, resolution,
    )
    em_pris = em_prismatic(
        O_stack, p_base, p_move,
        anchor_coords, anchor_weights,
        pris_init, hp, resolution,
    )

    N_per_state = _count_active_voxels(p_move, hp.active_voxel_thresh)
    N = max(N_per_state * K, 1)

    k_rev = k_rev_const + K
    k_pris = k_pris_const + K

    bic_rev = 2.0 * em_rev.L_data_final + k_rev * math.log(float(N))
    bic_pris = 2.0 * em_pris.L_data_final + k_pris * math.log(float(N))

    selected_type = "revolute" if bic_rev <= bic_pris else "prismatic"
    selected = em_rev if selected_type == "revolute" else em_pris

    if N < min_N:
        confidence: Optional[float] = None
    else:
        # softmax(-BIC/2)
        m = min(bic_rev, bic_pris)
        exp_rev = math.exp(-0.5 * (bic_rev - m))
        exp_pris = math.exp(-0.5 * (bic_pris - m))
        total = exp_rev + exp_pris
        p_rev = exp_rev / total
        confidence = float(p_rev if selected_type == "revolute" else (1.0 - p_rev))

    return BICResult(
        joint_type=selected_type,
        bic_rev=float(bic_rev),
        bic_pris=float(bic_pris),
        confidence=confidence,
        N=N,
        k_rev=k_rev,
        k_pris=k_pris,
        em_rev=em_rev,
        em_pris=em_pris,
        selected=selected,
    )

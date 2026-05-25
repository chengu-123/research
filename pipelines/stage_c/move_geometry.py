"""Per-state move geometry extraction.

Given Stage B v3.3.6's per-state move evidence (`O_move_per_state` hard or
`P_move_evidence_per_state` soft), compute geometric summaries that drive
joint type detection and axis fitting downstream:

- centroid_k: weighted mean voxel position of state k's move
- extent_k:   bbox diagonal length of state k's move
- count_k:    number of move voxels in state k

All positions are returned in WORLD space ([-0.5, 0.5] cube), via the
voxel-to-world convention from method.md section 5.5 / 4.11:
    world(u) = (u_int + 0.5) / res - 0.5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def voxel_to_world(coords_int: torch.Tensor, res: int = 64) -> torch.Tensor:
    """Convert integer voxel coords to world-space float coords.

    coords_int : (..., 3) int  in [0, res-1]
    Returns    : (..., 3) float in (-0.5, 0.5)  (voxel-CENTER convention)
    """
    return (coords_int.float() + 0.5) / float(res) - 0.5


@dataclass
class PerStateMoveGeom:
    """Geometric summary per K state."""

    centroid_world: torch.Tensor      # (K, 3) world-space mean move position
    count: torch.Tensor               # (K,) int   move voxel count per state
    extent_world: torch.Tensor        # (K,) float bbox diagonal in world units
    valid_mask: torch.Tensor          # (K,) bool  state has enough move voxels
                                      #            (>= min_move_voxels_per_state)
    weighted: bool                    # True if soft-weighted, False if hard-binary


def _hard_centroid_per_state(
    O_move_per_state: torch.Tensor,    # (K, D, H, W) uint8/bool, hard mask
    res: int,
    min_voxels: int,
) -> PerStateMoveGeom:
    """Equal-weight centroid over hard move voxels."""
    K = int(O_move_per_state.shape[0])
    device = O_move_per_state.device

    centroid_world = torch.zeros(K, 3, device=device, dtype=torch.float32)
    count = torch.zeros(K, device=device, dtype=torch.long)
    extent_world = torch.zeros(K, device=device, dtype=torch.float32)
    valid = torch.zeros(K, device=device, dtype=torch.bool)

    O_bool = O_move_per_state > 0
    for k in range(K):
        occ_k = O_bool[k]
        n_k = int(occ_k.sum().item())
        count[k] = n_k
        if n_k < min_voxels:
            continue
        idx = torch.nonzero(occ_k, as_tuple=False)               # (n_k, 3) int
        idx_world = voxel_to_world(idx, res=res)                  # (n_k, 3) float
        centroid_world[k] = idx_world.mean(dim=0)
        bbox_min = idx_world.min(dim=0).values
        bbox_max = idx_world.max(dim=0).values
        extent_world[k] = (bbox_max - bbox_min).norm()
        valid[k] = True
    return PerStateMoveGeom(
        centroid_world=centroid_world,
        count=count,
        extent_world=extent_world,
        valid_mask=valid,
        weighted=False,
    )


def _soft_centroid_per_state(
    P_move_evidence_per_state: torch.Tensor,   # (K, D, H, W) float
    res: int,
    soft_threshold: float,
    min_voxels: int,
) -> PerStateMoveGeom:
    """Weighted centroid using soft per-state evidence.

    Voxels with P_move < soft_threshold are excluded entirely; the remaining
    voxels contribute with weight = P_move.
    """
    K = int(P_move_evidence_per_state.shape[0])
    device = P_move_evidence_per_state.device

    centroid_world = torch.zeros(K, 3, device=device, dtype=torch.float32)
    count = torch.zeros(K, device=device, dtype=torch.long)
    extent_world = torch.zeros(K, device=device, dtype=torch.float32)
    valid = torch.zeros(K, device=device, dtype=torch.bool)

    for k in range(K):
        Pk = P_move_evidence_per_state[k]
        mask = Pk >= soft_threshold
        n_k = int(mask.sum().item())
        count[k] = n_k
        if n_k < min_voxels:
            continue
        idx = torch.nonzero(mask, as_tuple=False)                 # (n_k, 3) int
        w = Pk[mask].float().clamp_min(0.0)                       # (n_k,)
        w_sum = w.sum().clamp_min(1e-6)
        idx_world = voxel_to_world(idx, res=res)                  # (n_k, 3) float
        centroid_world[k] = (idx_world * w.unsqueeze(-1)).sum(dim=0) / w_sum
        # Extent: weighted bbox is awkward; use plain bbox over above-threshold voxels.
        bbox_min = idx_world.min(dim=0).values
        bbox_max = idx_world.max(dim=0).values
        extent_world[k] = (bbox_max - bbox_min).norm()
        valid[k] = True
    return PerStateMoveGeom(
        centroid_world=centroid_world,
        count=count,
        extent_world=extent_world,
        valid_mask=valid,
        weighted=True,
    )


def compute_per_state_move_geom(
    O_move_per_state: Optional[torch.Tensor],
    P_move_evidence_per_state: Optional[torch.Tensor],
    res: int = 64,
    min_voxels: int = 30,
    soft_threshold: float = 0.1,
    prefer_soft: bool = True,
) -> PerStateMoveGeom:
    """Public entrypoint: choose soft or hard centroid depending on availability.

    Prefers soft (P_move_evidence_per_state) when available -- it yields more
    stable centroids than hard binary, especially for thin parts where hard
    binary count varies sharply across K. Falls back to hard if only
    O_move_per_state is provided.

    Raises ValueError if both inputs are None.
    """
    if prefer_soft and P_move_evidence_per_state is not None:
        return _soft_centroid_per_state(
            P_move_evidence_per_state, res, soft_threshold, min_voxels,
        )
    if O_move_per_state is not None:
        return _hard_centroid_per_state(O_move_per_state, res, min_voxels)
    if P_move_evidence_per_state is not None:
        return _soft_centroid_per_state(
            P_move_evidence_per_state, res, soft_threshold, min_voxels,
        )
    raise ValueError(
        "compute_per_state_move_geom: both O_move_per_state and "
        "P_move_evidence_per_state are None; need at least one Stage B output"
    )


# ---------------------------------------------------------------------------
# Helpers used by downstream modules
# ---------------------------------------------------------------------------


def valid_centroid_subset(
    geom: PerStateMoveGeom,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (centroids, k_indices) only for states with valid move.

    Used by joint_type_detect / axis_fit / phi_fit so they operate on the
    densely-defined subset rather than risking nans / zero rows.
    """
    valid_k = torch.nonzero(geom.valid_mask, as_tuple=False).squeeze(-1)  # (n_valid,)
    return geom.centroid_world[valid_k], valid_k


def overall_move_extent(geom: PerStateMoveGeom) -> float:
    """Max per-state extent across all valid states (world units).

    Used as a length scale for type detection margins and theta/disp init.
    """
    if not geom.valid_mask.any():
        return 0.0
    valid_extent = geom.extent_world[geom.valid_mask]
    return float(valid_extent.max().item())


def centroid_trajectory_length(geom: PerStateMoveGeom) -> float:
    """Sum of pairwise distances between consecutive valid centroids.

    Coarse proxy for "how far did the part move across all states".
    """
    cents, ks = valid_centroid_subset(geom)
    if cents.shape[0] < 2:
        return 0.0
    # Sort by k so distances are consecutive-in-state.
    order = torch.argsort(ks)
    cents_sorted = cents[order]
    diffs = cents_sorted[1:] - cents_sorted[:-1]
    return float(diffs.norm(dim=-1).sum().item())

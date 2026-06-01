"""Anchor extraction from M_motion_corridor_64 and O_base_canonical.

Per v3.3.6 spec (user-directed Plan C+ in this critical-thinking session):

    anchor_candidates = voxels where:
        M_motion_corridor_64 > anchor_corridor_threshold       # in swept volume
        AND O_base_canonical (or P_base_canonical) > base_thresh  # still base
        AND NOT is_carpet_mask                                    # exclude carpet

This is the "base/move contact band" -- exactly the voxel set that the
joint axis must pass through for physical plausibility (method.md target.md
G3: revolute hinge axis must pierce the anchor set's narrow neighborhood).

After intersection we dilate by 1 voxel (capture boundary) and farthest-point-
sample down to anchor_target_count for downstream loss tractability.

Output: anchors_object (N_a, 3) int32 voxel coords (NOT world space) --
Bootstrap converts to world via voxel_to_world later.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_to_xyz(flat_idx: torch.Tensor, res: int) -> torch.Tensor:
    """Convert flat indices (in row-major DHW order) to (x, y, z) int32.

    Matches the SS-VAE decoder output convention used everywhere else:
    voxel_flat = x * res * res + y * res + z.
    """
    x = flat_idx // (res * res)
    rem = flat_idx % (res * res)
    y = rem // res
    z = rem % res
    return torch.stack([x, y, z], dim=-1).to(torch.int32)


def _voxel_dilate_3d(
    coords_int: torch.Tensor,    # (N, 3) int
    radius: int,
    res: int,
) -> torch.Tensor:
    """3D voxel dilation via dense scatter + max_pool3d.

    Returns deduplicated (N', 3) int32 coords.
    """
    if radius <= 0 or coords_int.shape[0] == 0:
        return coords_int.to(torch.int32)
    device = coords_int.device

    # Dense bool grid
    grid = torch.zeros(res, res, res, device=device, dtype=torch.bool)
    grid[coords_int[:, 0].long(), coords_int[:, 1].long(), coords_int[:, 2].long()] = True

    # max_pool3d with kernel=2*radius+1, stride=1 to dilate
    kernel = 2 * radius + 1
    pad = radius
    grid_f = grid.unsqueeze(0).unsqueeze(0).float()
    dilated = torch.nn.functional.max_pool3d(
        grid_f, kernel_size=kernel, stride=1, padding=pad,
    ).squeeze(0).squeeze(0) > 0.5

    new_coords = torch.nonzero(dilated, as_tuple=False).to(torch.int32)
    return new_coords


def _farthest_point_sample(
    points_world: torch.Tensor,    # (N, 3) float (world space)
    n_target: int,
    seed: int = 0,
) -> torch.Tensor:
    """FPS down-sampling. Returns indices of selected points.

    Greedy: pick first point at random, then repeatedly pick the point
    farthest from the already-selected set.
    """
    N = points_world.shape[0]
    if N <= n_target:
        return torch.arange(N, device=points_world.device, dtype=torch.long)

    device = points_world.device
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    first = int(torch.randint(0, N, (1,), generator=gen).item())

    selected = torch.zeros(n_target, dtype=torch.long, device=device)
    selected[0] = first
    # min distance from each point to the selected set so far
    dist_to_set = torch.full((N,), float("inf"), device=device, dtype=torch.float32)
    last_picked = first
    for i in range(1, n_target):
        d_new = (points_world - points_world[last_picked]).pow(2).sum(dim=-1)
        dist_to_set = torch.minimum(dist_to_set, d_new)
        last_picked = int(torch.argmax(dist_to_set).item())
        selected[i] = last_picked
        # mark picked as 0 so we don't pick it again
        dist_to_set[last_picked] = -1.0
    return selected


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def extract_anchors(
    M_motion_corridor_64: Optional[torch.Tensor],   # (64, 64, 64) float, optional
    O_base_canonical: Optional[torch.Tensor],       # (64, 64, 64) uint8/bool, optional
    P_base_canonical: Optional[torch.Tensor],       # (64, 64, 64) float, optional fallback
    move_union_voxel: Optional[torch.Tensor],       # (N, 3) int cleaned move evidence
    is_carpet_mask_flat: torch.Tensor,              # (res^3,) bool
    corridor_threshold: float = 0.1,
    base_threshold: float = 0.5,
    dilate_radius: int = 1,
    near_move_radius: int = 3,
    target_count: int = 48,                          # v3: ignored (no FPS subsample)
    min_count: int = 8,
    res: int = 64,
    fps_seed: int = 0,                               # v3: ignored
) -> tuple[torch.Tensor, dict]:
    """v3 change: no FPS subsample. Keep ALL contact-band voxels.

    Reason: the 48-anchor cap was an arbitrary tractability heuristic from
    when anchors fed an expensive EM loss. v3 voxel-physical scoring uses
    anchors only for the contact_compat term (distance from axis line to
    anchor band), which is cheap regardless of |anchors|. Keeping all
    contact-band voxels gives more reliable physical constraint.
    """
    """Extract anchor voxel set.

    Returns
    -------
    anchors_object : (N_a, 3) int32 voxel coords, N_a == min(target_count, candidates_after_dilate)
    diag           : dict with intermediate counts for viz / sanity
    """
    if O_base_canonical is not None:
        base_mask_for_contact = O_base_canonical > 0
    elif P_base_canonical is not None:
        base_mask_for_contact = P_base_canonical > base_threshold
    else:
        base_mask_for_contact = None

    cand_mask = None
    diag = {}
    if (
        move_union_voxel is not None
        and move_union_voxel.shape[0] > 0
        and base_mask_for_contact is not None
    ):
        move_grid = torch.zeros(res, res, res, device=move_union_voxel.device, dtype=torch.bool)
        coords = move_union_voxel.to(device=move_grid.device, dtype=torch.long)
        coords = coords[
            (coords[:, 0] >= 0) & (coords[:, 0] < res)
            & (coords[:, 1] >= 0) & (coords[:, 1] < res)
            & (coords[:, 2] >= 0) & (coords[:, 2] < res)
        ]
        move_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = True
        near_move = torch.nn.functional.max_pool3d(
            move_grid.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=2 * near_move_radius + 1,
            stride=1,
            padding=near_move_radius,
        ).squeeze(0).squeeze(0) > 0.5
        base_float = base_mask_for_contact.float()
        avg = torch.nn.functional.avg_pool3d(
            base_float.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1,
        ).squeeze(0).squeeze(0)
        base_surface = (avg > 0.1) & (avg < 0.9) & base_mask_for_contact
        local_contact = base_surface & near_move
        if int(local_contact.sum().item()) >= min_count:
            cand_mask = local_contact
            diag = {
                "source": "local_base_move_contact",
                "move_union_count": int(coords.shape[0]),
                "base_count": int(base_mask_for_contact.sum().item()),
                "candidates_pre_dilate": int(cand_mask.sum().item()),
            }
            if M_motion_corridor_64 is not None:
                corridor_mask = M_motion_corridor_64 > corridor_threshold
                diag["corridor_count"] = int(corridor_mask.sum().item())

    if cand_mask is None and M_motion_corridor_64 is None:
        # Without motion corridor we cannot do v3.3.6 anchor extraction.
        # Fall back to: just use O_base_canonical boundary voxels.
        if O_base_canonical is None:
            raise ValueError(
                "extract_anchors: need at least one of "
                "M_motion_corridor_64 / O_base_canonical to extract anchors"
            )
        # Use base surface (gradient magnitude) as anchor candidates
        base = (O_base_canonical > 0).float()
        kernel = 3
        avg = torch.nn.functional.avg_pool3d(
            base.unsqueeze(0).unsqueeze(0), kernel_size=kernel, stride=1, padding=1,
        ).squeeze(0).squeeze(0)
        boundary = (avg > 0.1) & (avg < 0.9) & (base > 0.5)
        cand_mask = boundary
        diag = {
            "source": "base_boundary_fallback",
            "candidates_pre_dilate": int(cand_mask.sum().item()),
        }
    elif cand_mask is None:
        # v3.3.6 normal path: M_motion_corridor_64 > thr AND base voxel
        corridor_mask = M_motion_corridor_64 > corridor_threshold
        if O_base_canonical is not None:
            base_mask = O_base_canonical > 0
        elif P_base_canonical is not None:
            base_mask = P_base_canonical > base_threshold
        else:
            # If no base info at all, just use corridor
            base_mask = torch.ones_like(corridor_mask, dtype=torch.bool)
        cand_mask = corridor_mask & base_mask
        diag = {
            "source": "corridor_intersect_base",
            "corridor_count": int(corridor_mask.sum().item()),
            "base_count": int(base_mask.sum().item()),
            "candidates_pre_dilate": int(cand_mask.sum().item()),
        }

    # If we don't have enough candidates, relax corridor threshold once
    if int(cand_mask.sum().item()) < min_count and M_motion_corridor_64 is not None:
        relaxed_thr = max(corridor_threshold * 0.3, 1e-3)
        corridor_mask_relaxed = M_motion_corridor_64 > relaxed_thr
        if O_base_canonical is not None:
            base_mask = O_base_canonical > 0
        elif P_base_canonical is not None:
            base_mask = P_base_canonical > base_threshold
        else:
            base_mask = torch.ones_like(corridor_mask_relaxed, dtype=torch.bool)
        cand_mask = corridor_mask_relaxed & base_mask
        diag["candidates_after_relax"] = int(cand_mask.sum().item())
        diag["relaxed_corridor_threshold"] = relaxed_thr

    # Get integer coords (3D index)
    cand_coords = torch.nonzero(cand_mask, as_tuple=False).to(torch.int32)

    # Strip carpet voxels
    if cand_coords.shape[0] > 0 and is_carpet_mask_flat is not None:
        # Convert to flat index, mask out carpet
        flat = (
            cand_coords[:, 0].long() * res * res
            + cand_coords[:, 1].long() * res
            + cand_coords[:, 2].long()
        )
        keep = ~is_carpet_mask_flat[flat]
        cand_coords = cand_coords[keep]
        diag["candidates_after_carpet_strip"] = int(cand_coords.shape[0])

    # Dilate to capture boundary band
    cand_coords = _voxel_dilate_3d(cand_coords, radius=dilate_radius, res=res)

    # Strip carpet again post-dilate (dilation may pull carpet voxels in)
    if cand_coords.shape[0] > 0 and is_carpet_mask_flat is not None:
        flat = (
            cand_coords[:, 0].long() * res * res
            + cand_coords[:, 1].long() * res
            + cand_coords[:, 2].long()
        )
        keep = ~is_carpet_mask_flat[flat]
        cand_coords = cand_coords[keep]
    diag["candidates_after_dilate"] = int(cand_coords.shape[0])

    if cand_coords.shape[0] == 0:
        diag["status"] = "empty"
        return cand_coords, diag

    # v3: NO FPS subsample. Keep all contact-band voxels.
    # The contact_compat scoring term only needs min-distance from axis line
    # to anchor cloud, which is O(N_a) regardless of count.
    diag["final_anchor_count"] = int(cand_coords.shape[0])
    diag["status"] = "ok" if cand_coords.shape[0] >= min_count else "below_min"
    diag["fps_skipped"] = True
    return cand_coords, diag

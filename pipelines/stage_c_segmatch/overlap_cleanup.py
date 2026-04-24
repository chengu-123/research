"""C.7 Canonical base-base overlap cleanup (target.md G2).

After the rigid motion is fit and canonical geometry aggregated, we may
find voxels that land in both ``canonical_base`` AND ``canonical_move``:

* **Containment conflict** (target.md P2.1): the move part, inverse-warped
  to canonical, lands inside the base housing (e.g., drawer interior
  under cabinet wall). These voxels are swept-volume artifacts from the
  K discrete states and should be removed from ``canonical_move`` so that
  ``canonical_move ∩ canonical_base == contact band only``.

* **Boundary collision** (target.md P2.2): near the contact band (``d ≤ r``),
  the overlap is legitimate and represents the physical joint surface —
  we keep both.

Distinction: distance to the ``canonical_base`` surface/boundary.
Far from boundary → containment conflict, delete from move. Near
boundary → contact band voxel, keep both.

This is a geometric post-processing step, not an optimization; no
learning, no iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class OverlapCleanupResult:
    canonical_base: torch.Tensor          # (D, H, W) bool — unchanged
    canonical_move: torch.Tensor          # (D, H, W) bool — after cleanup
    contact_region: torch.Tensor          # (D, H, W) bool — overlap ∩ contact band
    overlap_before: torch.Tensor          # (D, H, W) bool — original intersection
    n_containment_deleted: int
    n_contact_kept: int


def _morph_erode(mask: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """26-connectivity erosion via min-pool (= -max-pool on complement)."""
    inv = (~mask).to(torch.float32)[None, None]
    dilated_inv = F.max_pool3d(inv, kernel_size=kernel, stride=1,
                                padding=kernel // 2).squeeze(0).squeeze(0)
    return dilated_inv <= 0.5


def _morph_dilate(mask: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """26-connectivity dilation via 3x3x3 max-pool."""
    x = mask.to(torch.float32)[None, None]
    out = F.max_pool3d(x, kernel_size=kernel, stride=1,
                       padding=kernel // 2).squeeze(0).squeeze(0)
    return out > 0.5


def _base_boundary(canonical_base: torch.Tensor) -> torch.Tensor:
    """Surface voxels of the base: ``dilate(base) - base``. 1-voxel thick."""
    dil = _morph_dilate(canonical_base, kernel=3)
    return dil & (~canonical_base)


def _signed_distance_to_base_boundary(
    canonical_base: torch.Tensor,
    max_radius: int = 4,
) -> torch.Tensor:
    """Approximate signed distance (in voxel units) from each voxel to
    the base surface. Positive outside the base, zero on the boundary,
    negative inside. Capped at ``max_radius``.

    Implemented as a simple BFS-like morphological chamfer over
    successive dilations. For ``max_radius = 4`` this is 4
    iterations of 3x3x3 max-pool, plenty fast for 64³.
    """
    device = canonical_base.device
    dist = torch.full(canonical_base.shape, float(max_radius + 1),
                      device=device, dtype=torch.float32)
    boundary = _base_boundary(canonical_base)
    dist[boundary] = 0.0
    current = boundary.clone()
    for r in range(1, max_radius + 1):
        dilated = _morph_dilate(current, kernel=3)
        new_ring = dilated & (~current)
        dist[new_ring & (dist > float(r))] = float(r)
        current = dilated
    # Negative distance inside the base
    dist[canonical_base] = -dist[canonical_base]
    return dist


def cleanup_overlap(
    canonical_base: torch.Tensor,
    canonical_move: torch.Tensor,
    contact_band_radius: int = 2,
) -> OverlapCleanupResult:
    """Run containment vs contact classification on the base-move overlap.

    Parameters
    ----------
    canonical_base, canonical_move : (D, H, W) bool
    contact_band_radius : int — voxels; overlap voxels within this
        radius of the base surface are kept as contact band, overlap
        voxels deeper inside the base are deleted from ``canonical_move``.
    """
    overlap_before = canonical_base & canonical_move
    if not overlap_before.any():
        return OverlapCleanupResult(
            canonical_base=canonical_base,
            canonical_move=canonical_move,
            contact_region=torch.zeros_like(canonical_base),
            overlap_before=overlap_before,
            n_containment_deleted=0,
            n_contact_kept=0,
        )

    # Distance from each voxel to the base surface (signed; negative inside base)
    signed_dist = _signed_distance_to_base_boundary(
        canonical_base, max_radius=contact_band_radius + 1,
    )
    # Containment: overlap AND strictly inside base (dist < 0)
    # AND far from boundary (|dist| > contact_band_radius)
    deep_inside = overlap_before & (signed_dist < -float(contact_band_radius))
    # Contact: overlap AND near boundary
    contact = overlap_before & (~deep_inside)

    # Apply cleanup: remove deep-inside voxels from canonical_move
    canonical_move_clean = canonical_move & (~deep_inside)

    return OverlapCleanupResult(
        canonical_base=canonical_base,
        canonical_move=canonical_move_clean,
        contact_region=contact,
        overlap_before=overlap_before,
        n_containment_deleted=int(deep_inside.sum().item()),
        n_contact_kept=int(contact.sum().item()),
    )

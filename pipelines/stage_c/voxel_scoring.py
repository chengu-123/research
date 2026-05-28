"""Voxel-level physical scoring for Stage C candidate joints.

This module implements the core "physical evaluator" that GPT identified as
the missing piece in the centroid-only implementation. Given a hypothesized
joint (type, axis, origin, phi_k), it:

  1. Builds SE(3) transform T_k for each state k
  2. Reverse-warps per-state move evidence E_k back to canonical state c
  3. Scores the candidate by checking whether the reverse-warped fragments
     form a consistent canonical support (high consistency = joint hypothesis
     explains the observations)

The score combines five physical quantities:

  consistency:  mean_k IoU(E_k_warp_to_c, fragments_union)
                -> high if fragments overlap after reverse warp

  conflict:     |fragments_union & O_base_canonical| / |fragments_union|
                -> high if warped fragments pierce the base voxels
                   (physically impossible; penalizes wrong hypotheses)

  coverage:     |fragments_union| / |V_union|
                -> fraction of observed move evidence explained by this joint

  contact_compat: 1.0 if axis line passes within dilated anchor band
                  0.3 otherwise (physical constraint: revolute hinge MUST be
                  at base-move contact)

  monotone_quality: 1.0 if phi_k strictly monotone in k, else penalty

Final score = consistency * (1 - conflict) * coverage * contact_compat *
              monotone_quality

References: method.md sec 7.8 SE(3) warp convention; pipeline.md 4.11
voxel/world coordinate handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def voxel_to_world(coords_int: torch.Tensor, res: int = 64) -> torch.Tensor:
    """(N, 3) int voxel coords in [0, res-1] -> (N, 3) float in (-0.5, 0.5)."""
    return (coords_int.float() + 0.5) / float(res) - 0.5


def world_to_voxel(coords_world: torch.Tensor, res: int = 64) -> torch.Tensor:
    """Inverse of voxel_to_world; returns (N, 3) long, clipped to [0, res-1]."""
    vox = ((coords_world + 0.5) * float(res) - 0.5).round().long()
    return vox.clamp(0, res - 1)


# ---------------------------------------------------------------------------
# SE(3) builders (no autograd; pure numpy-like ops on torch tensors)
# ---------------------------------------------------------------------------


def rotation_matrix_from_axis_angle(axis: torch.Tensor, angle: float) -> torch.Tensor:
    """Rodrigues formula. axis: (3,) unit vector. angle: scalar radians.

    Returns (3, 3) rotation matrix on same device/dtype as axis.
    """
    a = axis / axis.norm().clamp_min(1e-12)
    c = torch.cos(torch.tensor(angle, device=a.device, dtype=a.dtype))
    s = torch.sin(torch.tensor(angle, device=a.device, dtype=a.dtype))
    K = torch.tensor([
        [0.0, -a[2].item(), a[1].item()],
        [a[2].item(), 0.0, -a[0].item()],
        [-a[1].item(), a[0].item(), 0.0],
    ], device=a.device, dtype=a.dtype)
    I3 = torch.eye(3, device=a.device, dtype=a.dtype)
    return I3 + s * K + (1.0 - c) * (K @ K)


def se3_prismatic(axis: torch.Tensor, displacement: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure translation along axis. Returns (R=I, t=displacement*axis)."""
    R = torch.eye(3, device=axis.device, dtype=axis.dtype)
    t = float(displacement) * axis
    return R, t


def se3_revolute(
    axis: torch.Tensor, origin: torch.Tensor, angle: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotation by angle around line {origin + t*axis}. Returns (R, t).

    Mapping: p -> R @ (p - origin) + origin = R @ p + (origin - R @ origin)
    So in canonical (R, t) form: t = origin - R @ origin.
    """
    R = rotation_matrix_from_axis_angle(axis, angle)
    t = origin - R @ origin
    return R, t


# ---------------------------------------------------------------------------
# Voxel warp + IoU
# ---------------------------------------------------------------------------


def warp_voxel_set(
    V_world: torch.Tensor,    # (N, 3) world coords
    R: torch.Tensor,           # (3, 3)
    t: torch.Tensor,           # (3,)
) -> torch.Tensor:
    """Apply SE(3) to a voxel cloud in world space."""
    return (R @ V_world.T).T + t


def rasterize_world_to_voxel_grid(
    V_world: torch.Tensor,    # (N, 3) world coords
    res: int = 64,
) -> torch.Tensor:
    """Build a dense (res, res, res) bool occupancy grid from a sparse point cloud.

    Out-of-range points are silently dropped.
    """
    device = V_world.device
    vox = ((V_world + 0.5) * float(res) - 0.5).round().long()
    in_range = ((vox >= 0) & (vox < res)).all(dim=-1)
    vox_in = vox[in_range]
    grid = torch.zeros(res, res, res, dtype=torch.bool, device=device)
    if vox_in.shape[0] > 0:
        grid[vox_in[:, 0], vox_in[:, 1], vox_in[:, 2]] = True
    return grid


def voxel_iou(grid_a: torch.Tensor, grid_b: torch.Tensor) -> float:
    """IoU on dense (res, res, res) bool grids."""
    a = grid_a.bool()
    b = grid_b.bool()
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


# ---------------------------------------------------------------------------
# Reverse-align fragments + score
# ---------------------------------------------------------------------------


@dataclass
class CandidateScore:
    """Per-candidate physical score breakdown."""

    score: float                    # final scalar in [0, 1]
    consistency: float              # mean IoU of fragments with union
    conflict: float                 # base-overlap fraction
    coverage: float                 # fragments_union / V_union ratio
    contact_compat: float           # 1.0 if axis near anchors, 0.3 else
    monotone_quality: float         # 1.0 if phi strictly monotone, else penalty
    valid_states_used: int          # number of states that contributed
    boundary_path_score: float = 0.0
    axis_base_support: float = 0.0
    arc_balance: float = 0.0
    radius_stability: float = 0.0
    axis_stability: float = 0.0


def reverse_align_and_score(
    joint_type: str,                                  # "prismatic" | "revolute"
    axis: torch.Tensor,                                # (3,) unit world
    origin: torch.Tensor,                              # (3,) world
    phi_k: torch.Tensor,                               # (K,) per-state progress
    canonical_state_idx: int,                         # c, e.g., 2
    V_per_state_voxel: List[torch.Tensor],            # K lists of (N_k, 3) int voxel coords
    V_union_voxel: torch.Tensor,                       # (N_u, 3) int voxel coords
    O_base_canonical: Optional[torch.Tensor],         # (res, res, res) bool, may be None
    anchors_world: Optional[torch.Tensor],            # (N_a, 3) world, may be None
    res: int = 64,
    anchor_band_radius: float = 0.06,                  # world units, ~4 voxels at res=64
) -> CandidateScore:
    """Reverse-warp each state's evidence to canonical and score the alignment.

    Algorithm:
      For each state k != c (canonical):
        delta = phi_k[c] - phi_k[k]   # apply this much "extra motion" to k's
                                       # evidence to bring it back to canonical
        (For prismatic: subtract phi displacement; for revolute: rotate by
         -(phi_k[k] - phi_k[c]) around axis through origin.)
        E_k_warp = warp(V_k_world, T(delta))
        Rasterize to dense grid

      fragments_union = OR over all warped states (including k=c which is identity)
      consistency = mean_k IoU(E_k_warp_grid, fragments_union_grid)

    For prismatic:  warp = translate by delta * axis
    For revolute:   warp = rotate by delta around (axis, origin)
    """
    if joint_type not in ("prismatic", "revolute"):
        raise ValueError(f"joint_type must be prismatic or revolute, got {joint_type}")
    K = len(V_per_state_voxel)
    if K == 0 or phi_k.numel() != K:
        raise ValueError(f"phi_k length {phi_k.numel()} must equal K={K}")
    device = axis.device
    dtype = axis.dtype

    phi_c = float(phi_k[canonical_state_idx].item())

    # Warp each state's voxel set back to canonical
    fragments_grids: List[torch.Tensor] = []
    valid_states = 0
    iou_each: List[float] = []
    for k in range(K):
        V_k_int = V_per_state_voxel[k]
        if V_k_int is None or V_k_int.shape[0] == 0:
            continue
        valid_states += 1
        V_k_world = voxel_to_world(V_k_int, res=res).to(device=device, dtype=dtype)
        # Apply T(phi_c - phi_k[k]) to warp state k -> canonical
        delta = phi_c - float(phi_k[k].item())
        if joint_type == "prismatic":
            R, t = se3_prismatic(axis, delta)
        else:
            R, t = se3_revolute(axis, origin, delta)
        V_warp_world = warp_voxel_set(V_k_world, R, t)
        grid_warp = rasterize_world_to_voxel_grid(V_warp_world, res=res)
        fragments_grids.append(grid_warp)

    if not fragments_grids:
        return CandidateScore(
            score=0.0, consistency=0.0, conflict=1.0, coverage=0.0,
            contact_compat=0.0, monotone_quality=0.0, valid_states_used=0,
        )

    # Union of fragments
    fragments_union = fragments_grids[0].clone()
    for g in fragments_grids[1:]:
        fragments_union = fragments_union | g

    # Consistency = mean IoU of each fragment with the union
    for g in fragments_grids:
        iou_each.append(voxel_iou(g, fragments_union))
    consistency = float(sum(iou_each) / len(iou_each))

    # Conflict = fragments union piercing base
    if O_base_canonical is not None:
        base_b = O_base_canonical.bool().to(fragments_union.device)
        inter_base = (fragments_union & base_b).sum().item()
        union_size = fragments_union.sum().item()
        conflict = float(inter_base) / float(union_size) if union_size > 0 else 1.0
    else:
        conflict = 0.0

    # Coverage = how much of V_union the fragments_union actually covers
    V_union_grid = torch.zeros_like(fragments_union)
    if V_union_voxel is not None and V_union_voxel.shape[0] > 0:
        vu = V_union_voxel.long()
        V_union_grid[vu[:, 0], vu[:, 1], vu[:, 2]] = True
    union_count = V_union_grid.sum().item()
    if union_count > 0:
        coverage = float(fragments_union.sum().item()) / float(union_count)
        coverage = min(coverage, 1.0)
    else:
        coverage = 0.0

    # Contact compat = does axis line pass through anchor band?
    # Axis line: {origin + t * axis : t in R}; check min distance from anchors to line
    if anchors_world is not None and anchors_world.shape[0] > 0:
        # Distance from point p to line (origin, axis):
        #   d = || (p - origin) - ((p - origin) . axis) * axis ||
        diff = anchors_world.to(device=device, dtype=dtype) - origin.unsqueeze(0)
        proj = (diff * axis.unsqueeze(0)).sum(dim=-1, keepdim=True) * axis.unsqueeze(0)
        perp = diff - proj
        dists = perp.norm(dim=-1)
        min_dist = float(dists.min().item())
        if min_dist < anchor_band_radius:
            contact_compat = 1.0
        else:
            # Soft penalty: linearly decay to 0.3 at 3x radius
            ratio = (min_dist - anchor_band_radius) / (2.0 * anchor_band_radius)
            contact_compat = float(max(0.3, 1.0 - 0.7 * min(ratio, 1.0)))
    else:
        contact_compat = 1.0  # No anchors -> can't penalize

    # Monotone quality: check if phi is monotone-increasing in k
    phi_np = phi_k.detach().cpu()
    diffs = phi_np[1:] - phi_np[:-1]
    n_neg = int((diffs < 0).sum().item())
    n_pos = int((diffs > 0).sum().item())
    # Allow either monotone increasing OR decreasing (sign convention can flip)
    n_against_dominant = min(n_neg, n_pos)
    monotone_quality = 1.0 - float(n_against_dominant) / max(int(diffs.numel()), 1)

    final_score = (
        consistency
        * max(1.0 - conflict, 0.0)
        * coverage
        * contact_compat
        * monotone_quality
    )

    return CandidateScore(
        score=float(final_score),
        consistency=consistency,
        conflict=conflict,
        coverage=coverage,
        contact_compat=contact_compat,
        monotone_quality=monotone_quality,
        valid_states_used=valid_states,
    )


# ---------------------------------------------------------------------------
# FreeArt3D-style cardinal axis invariant (no point correspondences needed)
# ---------------------------------------------------------------------------


def freeart3d_axis_invariant(
    V_per_state_world: List[torch.Tensor],   # K (N_k, 3) world-coord voxel sets
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute axis-aligned discrimination signals from per-state voxel sets.

    Reproduces FreeArt3D estimate.py:387-401 logic in the voxel setting
    (no GIM-DKM point correspondences available; we use voxel-set centroid
    pair differences as a stand-in for the diff_vectors there).

    For each pair (i, j), compute diff = centroid(V_j) - centroid(V_i) in
    world space. Weight by sqrt(min(|V_i|, |V_j|)) to suppress outlier states.

    Returns
    -------
    pris_scores : (3,) float, axis-wise abs projection sum for prismatic
                  candidate; cardinal axis with MAX value = best prismatic axis
                  (parallel to motion). FreeArt3D estimate.py:399
                  `diff_axis = mean(abs(diff_vectors), axis=0);
                   argmax = prismatic axis`.

    rev_scores  : (3,) float, abs projection sum for revolute candidate;
                  cardinal axis with MIN value = best revolute axis
                  (perpendicular to motion plane). FreeArt3D estimate.py:393
                  `dots = sum(abs(diff @ candidate)); argmin = revolute axis`.
    """
    diffs = []
    weights = []
    K = len(V_per_state_world)
    for i in range(K):
        Vi = V_per_state_world[i]
        if Vi is None or Vi.shape[0] == 0:
            continue
        ci = Vi.mean(dim=0)
        for j in range(K):
            if i == j:
                continue
            Vj = V_per_state_world[j]
            if Vj is None or Vj.shape[0] == 0:
                continue
            cj = Vj.mean(dim=0)
            d = cj - ci
            if d.norm().item() < 1e-8:
                continue
            diffs.append(d)
            weights.append(float(min(Vi.shape[0], Vj.shape[0]) ** 0.5))

    if not diffs:
        zero = torch.zeros(3)
        return zero, zero

    D = torch.stack(diffs, dim=0)              # (M, 3)
    W = torch.tensor(weights, dtype=D.dtype).unsqueeze(-1)   # (M, 1)
    # Weighted sum of |proj| per cardinal axis
    abs_proj = (D.abs() * W).sum(dim=0) / W.sum()            # (3,)
    return abs_proj, abs_proj.clone()  # pris uses argmax, rev uses argmin -> same vector


def envelope_advance_along_axis(
    V_per_state_world: List[torch.Tensor],   # K voxel sets in world coords
    axis: torch.Tensor,                        # (3,) unit world
    percentile: float = 0.9,
) -> torch.Tensor:
    """For each state k, compute the (signed) percentile-90 of voxel projections
    on axis. Returns (K,) tensor with NaN for missing states.

    Used as the "envelope edge" signal for prismatic phi_k (more occlusion-
    robust than centroid).
    """
    K = len(V_per_state_world)
    out = torch.full((K,), float("nan"))
    for k in range(K):
        Vk = V_per_state_world[k]
        if Vk is None or Vk.shape[0] == 0:
            continue
        proj = (Vk * axis.unsqueeze(0)).sum(dim=-1)        # (N_k,)
        q = float(torch.quantile(proj, percentile).item())
        out[k] = q
    return out


def angular_median_around_axis(
    V_per_state_world: List[torch.Tensor],
    axis: torch.Tensor,                        # (3,) unit world
    origin: torch.Tensor,                      # (3,) world point on axis
    ref_perp_unit: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """For each state k, compute median angle (in radians) around the axis line
    through origin, measured from a fixed reference perpendicular direction.

    If ref_perp_unit is None, uses the perpendicular component of the canonical
    state's centroid relative to origin.

    Returns (K,) tensor with NaN for missing states.
    """
    K = len(V_per_state_world)
    out = torch.full((K,), float("nan"))
    device = axis.device
    a = axis / axis.norm().clamp_min(1e-12)

    # Pick a stable in-plane reference if none provided
    if ref_perp_unit is None:
        # Use first non-empty state's centroid
        for k in range(K):
            Vk = V_per_state_world[k]
            if Vk is None or Vk.shape[0] == 0:
                continue
            ck = Vk.mean(dim=0)
            v = ck - origin
            v_perp = v - (v @ a) * a
            if v_perp.norm().item() > 1e-6:
                ref_perp_unit = v_perp / v_perp.norm()
                break
    if ref_perp_unit is None:
        return out

    # Build right-hand-rule perpendicular companion: cross(ref, axis)
    sign_dir = torch.linalg.cross(ref_perp_unit.to(device), a)
    sign_dir = sign_dir / sign_dir.norm().clamp_min(1e-12)

    for k in range(K):
        Vk = V_per_state_world[k]
        if Vk is None or Vk.shape[0] == 0:
            continue
        diff = Vk.to(device) - origin.unsqueeze(0)
        proj_axis = (diff * a.unsqueeze(0)).sum(dim=-1, keepdim=True) * a.unsqueeze(0)
        perp = diff - proj_axis
        perp_norm = perp.norm(dim=-1).clamp_min(1e-12)
        perp_unit = perp / perp_norm.unsqueeze(-1)
        cos_t = (perp_unit * ref_perp_unit.to(device).unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)
        sin_t = (perp_unit * sign_dir.unsqueeze(0)).sum(dim=-1)
        angles = torch.atan2(sin_t, cos_t)                  # (N_k,) in (-pi, pi]
        # Use median (robust to outliers)
        out[k] = float(angles.median().item())
    return out


__all__ = [
    "voxel_to_world", "world_to_voxel",
    "rotation_matrix_from_axis_angle", "se3_prismatic", "se3_revolute",
    "warp_voxel_set", "rasterize_world_to_voxel_grid", "voxel_iou",
    "CandidateScore", "reverse_align_and_score",
    "freeart3d_axis_invariant",
    "envelope_advance_along_axis", "angular_median_around_axis",
]

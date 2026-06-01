"""Joint type + axis detection via cardinal-axis enumeration + voxel scoring.

v3 (cardinal-cand + voxel-physical-scoring; supersedes centroid-only v2):

Algorithm:
  Step A. Build per-state move voxel sets V_k (in voxel int and world coords)
          and the swept union V_union.
  Step B. FreeArt3D-style cardinal axis invariant (estimate.py:387-401):
          - Compute pair-wise centroid displacement vectors
          - prismatic axis = cardinal with MAX |sum proj| (parallel to motion)
          - revolute axis  = cardinal with MIN |sum proj| (perpendicular)
          These give us geometric prior; physical scoring is the final arbiter.
  Step C. For BOTH types (prismatic, revolute), generate K_cardinal candidates
          on the 6 cardinal axes (sign convention: ensure phi monotone-increasing
          in k by flipping axis if needed). For revolute, origin = anchor band
          centroid projected onto axis line through swept centroid (physical
          constraint: hinge must be at base-move contact).
  Step D. For each candidate, run voxel reverse-warp scoring (voxel_scoring.py):
              score = consistency * (1 - conflict) * coverage *
                      contact_compat * monotone_quality
  Step E. best_pris = argmax score over 6 prismatic candidates
          best_rev  = argmax score over 6 revolute candidates
          type_logit = log(best_pris.score / best_rev.score)
          recommended = sign(type_logit) when margin > threshold

The result is a JointTypeResult that records BOTH best_pris and best_rev so
axis_fit / phi_fit / run_stage_c_init can construct primary AND secondary
JointInit (Stage D dual-clone gets two well-initialized branches).
"""

from __future__ import annotations

import math
import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .voxel_scoring import (
    CandidateScore,
    angular_median_around_axis,
    envelope_advance_along_axis,
    freeart3d_axis_invariant,
    reverse_align_and_score,
    voxel_to_world,
)


# ---------------------------------------------------------------------------
# Cardinal set
# ---------------------------------------------------------------------------


def cardinal_axes(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Six cardinal axes as a (6, 3) tensor: +X, -X, +Y, -Y, +Z, -Z."""
    return torch.tensor([
        [+1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, +1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0, +1.0], [0.0, 0.0, -1.0],
    ], device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Per-state voxel set builder
# ---------------------------------------------------------------------------


def build_per_state_voxel_sets(
    O_move_per_state: Optional[torch.Tensor],         # (K, D, H, W) uint8/bool
    P_move_evidence_per_state: Optional[torch.Tensor],# (K, D, H, W) float
    O_base_canonical: Optional[torch.Tensor],         # (D, H, W) uint8/bool
    is_carpet_mask_flat: torch.Tensor,                # (res^3,) bool
    res: int = 64,
    soft_threshold: float = 0.1,
    min_voxels: int = 20,
    prefer_soft: bool = True,
) -> Tuple[List[torch.Tensor], torch.Tensor, List[bool]]:
    """Build per-state cleaned voxel coordinate lists.

    Returns
    -------
    V_per_state_voxel : list of K tensors (N_k, 3) int voxel coords
                        Empty tensor if state's evidence below min_voxels.
    V_union_voxel     : (N_union, 3) int voxel coords of OR_k(V_k)
    valid_mask        : list of K bool, True if state has >= min_voxels
    """
    device = is_carpet_mask_flat.device
    carpet_3d = is_carpet_mask_flat.reshape(res, res, res).to(device)
    base_3d = (
        O_base_canonical.to(device).bool()
        if O_base_canonical is not None
        else torch.zeros(res, res, res, dtype=torch.bool, device=device)
    )

    if prefer_soft and P_move_evidence_per_state is not None:
        src = P_move_evidence_per_state.to(device)
        K = int(src.shape[0])
        if O_move_per_state is not None:
            hard = O_move_per_state.to(device).bool()
            masks = [hard[k] | (src[k] >= soft_threshold) for k in range(K)]
        else:
            masks = [(src[k] >= soft_threshold) for k in range(K)]
    elif O_move_per_state is not None:
        src = O_move_per_state.to(device)
        K = int(src.shape[0])
        masks = [src[k].bool() for k in range(K)]
    elif P_move_evidence_per_state is not None:
        src = P_move_evidence_per_state.to(device)
        K = int(src.shape[0])
        masks = [(src[k] >= soft_threshold) for k in range(K)]
    else:
        raise ValueError(
            "build_per_state_voxel_sets: need at least one of "
            "O_move_per_state or P_move_evidence_per_state"
        )

    V_per_state: List[torch.Tensor] = []
    valid: List[bool] = []
    union_mask = torch.zeros(res, res, res, dtype=torch.bool, device=device)
    for k in range(K):
        m = masks[k] & (~base_3d) & (~carpet_3d)
        coords = torch.nonzero(m, as_tuple=False).to(torch.int32)
        if coords.shape[0] >= min_voxels:
            V_per_state.append(coords)
            valid.append(True)
            union_mask = union_mask | m
        else:
            V_per_state.append(torch.zeros(0, 3, dtype=torch.int32, device=device))
            valid.append(False)
    V_union = torch.nonzero(union_mask, as_tuple=False).to(torch.int32)
    return V_per_state, V_union, valid


# ---------------------------------------------------------------------------
# Candidate generation: phi_k per state given (type, axis, origin)
# ---------------------------------------------------------------------------


def _project_phi_prismatic(
    V_per_state_world: List[torch.Tensor],
    axis_unit: torch.Tensor,
    valid: List[bool],
    canonical_state_idx: int,
    percentile: float = 0.5,
) -> torch.Tensor:
    """Per-state phi from percentile projection of voxels on axis (signed).

    Uses median (percentile=0.5) which is robust to occlusion-revealed extra
    voxels; the user noted Stage B per-state count is NOT monotone for prismatic
    so the median per-state position along axis is more stable than mean or
    extreme percentile.

    Returns (K,) tensor; NaN for invalid states (filled later by interpolation).
    """
    K = len(V_per_state_world)
    out = torch.full((K,), float("nan"))
    for k in range(K):
        if not valid[k]:
            continue
        Vk = V_per_state_world[k]
        proj = (Vk * axis_unit.unsqueeze(0)).sum(dim=-1)
        out[k] = float(torch.quantile(proj, percentile).item())
    # Fill NaN by linear interpolation over valid k indices
    return _interp_nan_linear(out)


def _project_phi_revolute(
    V_per_state_world: List[torch.Tensor],
    axis_unit: torch.Tensor,
    origin: torch.Tensor,
    valid: List[bool],
    canonical_state_idx: int,
) -> torch.Tensor:
    """Per-state phi from angular median around axis. Returns (K,) radians.

    Reference perpendicular = canonical state's centroid - origin, projected
    perpendicular to axis. NaN-filled by interpolation.
    """
    K = len(V_per_state_world)
    a = axis_unit / axis_unit.norm().clamp_min(1e-12)
    # Build reference perp from canonical state
    ref_perp = None
    if 0 <= canonical_state_idx < K and valid[canonical_state_idx]:
        Vc = V_per_state_world[canonical_state_idx]
        cc = Vc.mean(dim=0)
        v = cc - origin
        v_perp = v - (v @ a) * a
        if v_perp.norm().item() > 1e-6:
            ref_perp = v_perp / v_perp.norm()
    if ref_perp is None:
        # Fallback: use first valid state
        for k in range(K):
            if not valid[k]:
                continue
            ck = V_per_state_world[k].mean(dim=0)
            v = ck - origin
            v_perp = v - (v @ a) * a
            if v_perp.norm().item() > 1e-6:
                ref_perp = v_perp / v_perp.norm()
                break
    if ref_perp is None:
        return torch.zeros(K)

    out = angular_median_around_axis(
        V_per_state_world, axis_unit, origin, ref_perp_unit=ref_perp,
    )
    return _interp_nan_linear(out)


def _interp_nan_linear(x: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate NaN entries in a 1D tensor."""
    arr = x.detach().cpu().numpy().astype(np.float64)
    K = len(arr)
    valid_idx = np.where(~np.isnan(arr))[0]
    if len(valid_idx) == K:
        return torch.from_numpy(arr).float()
    if len(valid_idx) < 2:
        # Degenerate: <= 1 valid -> linear ramp 0..1
        return torch.linspace(0.0, 1.0, K)
    arr_filled = np.interp(np.arange(K), valid_idx, arr[valid_idx])
    return torch.from_numpy(arr_filled).float()


def _resolve_revolute_origin(
    axis_unit: torch.Tensor,
    swept_centroid: torch.Tensor,
    anchors_world: Optional[torch.Tensor],
) -> torch.Tensor:
    """Compute origin = point on axis line {swept_centroid + t*axis} nearest to
    anchor centroid. If anchors absent, return swept_centroid.

    Physical motivation: hinge axis must pass through base-move contact band.
    """
    if anchors_world is None or anchors_world.shape[0] == 0:
        return swept_centroid.clone()
    anchor_centroid = anchors_world.mean(dim=0)
    t_star = float((anchor_centroid - swept_centroid) @ axis_unit)
    return swept_centroid + t_star * axis_unit


# ---------------------------------------------------------------------------
# Branched voxel candidates used by v4/v5 production Stage C
# ---------------------------------------------------------------------------


@dataclass
class _Component:
    coords: np.ndarray
    count: int
    centroid: np.ndarray
    bbox_lo: np.ndarray
    bbox_hi: np.ndarray
    corridor_mean: float
    score: float


@dataclass
class _PrismaticSpec:
    axis_dim: int
    sign: int
    score: float
    phi_world: np.ndarray
    advance: float
    monotone: float
    perp_iou: float
    compactness: float
    bbox_stability: float


@dataclass
class _RevoluteSpec:
    axis_dim: int
    center: Tuple[float, float]
    score: float
    phi_angle: np.ndarray
    contact: float
    radius_stability: float
    axis_stability: float
    angle_span: float
    angle_mono: float
    compactness: float
    hinge_support: float
    axis_base_support: float
    boundary_path_score: float
    arc_balance: float


def _plane_dims(axis_dim: int) -> Tuple[int, int]:
    dims = [0, 1, 2]
    dims.remove(axis_dim)
    return dims[0], dims[1]


def _fill_nan_linear_np(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float64).copy()
    valid = np.isfinite(arr)
    if valid.all():
        return arr
    if valid.sum() < 2:
        return np.linspace(0.0, 1.0, arr.shape[0], dtype=np.float64)
    idx = np.arange(arr.shape[0])
    arr[~valid] = np.interp(idx[~valid], idx[valid], arr[valid])
    return arr


def _monotone_quality_np(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    diffs = np.diff(arr)
    pos = int((diffs > 0).sum())
    neg = int((diffs < 0).sum())
    return max(pos, neg) / max(len(diffs), 1)


def _stability_from_series_np(series: np.ndarray, scale: float) -> float:
    if series.size == 0:
        return 0.0
    std = np.nanmean(np.nanstd(series, axis=0))
    return float(math.exp(-std / max(scale, 1e-6)))


def _raster2_np(coords_2d: np.ndarray, res: int) -> np.ndarray:
    grid = np.zeros((res, res), dtype=bool)
    if coords_2d.size == 0:
        return grid
    xy = np.rint(coords_2d).astype(np.int32)
    valid = np.all((xy >= 0) & (xy < res), axis=1)
    xy = xy[valid]
    grid[xy[:, 0], xy[:, 1]] = True
    return grid


def _raster3_np(coords: np.ndarray, res: int) -> np.ndarray:
    grid = np.zeros((res, res, res), dtype=bool)
    if coords.size == 0:
        return grid
    xyz = np.rint(coords).astype(np.int32)
    valid = np.all((xyz >= 0) & (xyz < res), axis=1)
    xyz = xyz[valid]
    grid[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True
    return grid


def _iou_np(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def _mean_pair_iou_np(grids: Sequence[np.ndarray]) -> float:
    vals: List[float] = []
    for i in range(len(grids)):
        for j in range(i + 1, len(grids)):
            vals.append(_iou_np(grids[i], grids[j]))
    return float(np.mean(vals)) if vals else 0.0


def _compactness_np(grids: Sequence[np.ndarray]) -> float:
    valid = [g for g in grids if g.sum() > 0]
    if not valid:
        return 0.0
    union = valid[0].copy()
    total = 0.0
    for g in valid:
        union |= g
        total += float(g.sum())
    denom = max(float(union.sum()) * len(valid), 1.0)
    return total / denom


def _shift_or_np(mask: np.ndarray, offset: Tuple[int, int, int], out: np.ndarray) -> None:
    src_slices = []
    dst_slices = []
    for dim, delta in enumerate(offset):
        size = mask.shape[dim]
        if delta >= 0:
            src_slices.append(slice(0, size - delta))
            dst_slices.append(slice(delta, size))
        else:
            src_slices.append(slice(-delta, size))
            dst_slices.append(slice(0, size + delta))
    out[tuple(dst_slices)] |= mask[tuple(src_slices)]


def _binary_dilation_np(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                _shift_or_np(mask, (dx, dy, dz), out)
    return out


def _binary_erosion_np(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    return ~_binary_dilation_np(~mask, radius=radius)


def _smooth2_np(heat: np.ndarray, rounds: int = 2) -> np.ndarray:
    out = heat.astype(np.float64).copy()
    kernel = ((0, 0, 0.25), (-1, 0, 0.125), (1, 0, 0.125),
              (0, -1, 0.125), (0, 1, 0.125),
              (-1, -1, 0.0625), (-1, 1, 0.0625),
              (1, -1, 0.0625), (1, 1, 0.0625))
    for _ in range(rounds):
        nxt = np.zeros_like(out)
        for da, db, w in kernel:
            src0 = slice(max(0, -da), min(out.shape[0], out.shape[0] - da))
            dst0 = slice(max(0, da), min(out.shape[0], out.shape[0] + da))
            src1 = slice(max(0, -db), min(out.shape[1], out.shape[1] - db))
            dst1 = slice(max(0, db), min(out.shape[1], out.shape[1] + db))
            nxt[dst0, dst1] += w * out[src0, src1]
        out = nxt
    return out


def _connected_components_np(mask: np.ndarray) -> List[np.ndarray]:
    coords = np.argwhere(mask)
    active = {tuple(int(v) for v in row) for row in coords}
    components: List[np.ndarray] = []
    neighbors = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    while active:
        seed = active.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            x, y, z = stack.pop()
            for dx, dy, dz in neighbors:
                nb = (x + dx, y + dy, z + dz)
                if nb in active:
                    active.remove(nb)
                    stack.append(nb)
                    comp.append(nb)
        components.append(np.asarray(comp, dtype=np.int32))
    return components


def _component_list_np(mask: np.ndarray, corridor: np.ndarray, min_count: int) -> List[_Component]:
    out: List[_Component] = []
    for coords in _connected_components_np(mask):
        count = int(coords.shape[0])
        if count < min_count:
            continue
        centroid = coords.mean(axis=0)
        bbox_lo = coords.min(axis=0)
        bbox_hi = coords.max(axis=0)
        corr = float(corridor[coords[:, 0], coords[:, 1], coords[:, 2]].mean())
        score = math.sqrt(count) * (0.25 + corr)
        out.append(_Component(coords, count, centroid, bbox_lo, bbox_hi, corr, score))
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def _clean_components_from_voxels(
    V_per_state_voxel: Sequence[torch.Tensor],
    O_base_canonical: Optional[torch.Tensor],
    M_motion_corridor_64: Optional[torch.Tensor],
    res: int,
    min_component: int = 8,
    keep_mass_frac: float = 0.92,
) -> Tuple[List[np.ndarray], List[List[_Component]]]:
    if O_base_canonical is None:
        base_np = np.zeros((res, res, res), dtype=bool)
    else:
        base_np = O_base_canonical.detach().cpu().numpy().astype(bool)
    if M_motion_corridor_64 is None:
        corridor_np = np.zeros((res, res, res), dtype=np.float32)
    else:
        corridor_np = M_motion_corridor_64.detach().cpu().numpy().astype(np.float32)

    state_sets: List[np.ndarray] = []
    component_sets: List[List[_Component]] = []
    for V in V_per_state_voxel:
        mask = np.zeros((res, res, res), dtype=bool)
        if V is not None and V.shape[0] > 0:
            coords = V.detach().cpu().numpy().astype(np.int32)
            valid = np.all((coords >= 0) & (coords < res), axis=1)
            coords = coords[valid]
            mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
        mask &= ~base_np
        comps = _component_list_np(mask, corridor_np, min_component)
        if not comps:
            state_sets.append(np.zeros((0, 3), dtype=np.int32))
            component_sets.append([])
            continue

        best_score = comps[0].score
        kept = [
            comp for comp in comps
            if comp.count >= max(min_component, int(0.06 * comps[0].count))
            and comp.score >= 0.18 * best_score
            and (comp.corridor_mean >= 0.045 or comp.count >= 0.30 * comps[0].count)
        ]
        kept.sort(key=lambda c: c.count, reverse=True)

        total = float(sum(c.count for c in kept))
        acc = 0.0
        final: List[_Component] = []
        for comp in kept:
            final.append(comp)
            acc += comp.count
            if total > 0 and acc / total >= keep_mass_frac:
                break
        coords = np.concatenate([c.coords for c in final], axis=0).astype(np.int32)
        state_sets.append(coords)
        component_sets.append(final)
    return state_sets, component_sets


def _score_prismatic_specs(state_sets: Sequence[np.ndarray], res: int) -> List[_PrismaticSpec]:
    out: List[_PrismaticSpec] = []
    for axis_dim in range(3):
        other = _plane_dims(axis_dim)
        for sign in (-1, 1):
            edges: List[float] = []
            perp_grids: List[np.ndarray] = []
            extents: List[np.ndarray] = []
            for coords in state_sets:
                if coords.size == 0:
                    edges.append(float("nan"))
                    perp_grids.append(np.zeros((res, res), dtype=bool))
                    extents.append(np.zeros(2, dtype=np.float64))
                    continue
                signed = sign * coords[:, axis_dim].astype(np.float64)
                edges.append(float(np.quantile(signed, 0.95)))
                perp = coords[:, other]
                perp_grids.append(_raster2_np(perp, res=res))
                extents.append(perp.max(axis=0) - perp.min(axis=0))

            edge_arr = _fill_nan_linear_np(np.asarray(edges, dtype=np.float64))
            edge_shift = edge_arr - edge_arr[2 if len(edge_arr) > 2 else 0]
            aligned = []
            for coords, shift in zip(state_sets, edge_shift):
                moved = coords.astype(np.float64).copy()
                if moved.size:
                    moved[:, axis_dim] -= sign * shift
                aligned.append(_raster3_np(moved, res=res))

            advance = float(edge_arr[-1] - edge_arr[0])
            positive_advance = max(advance, 0.0)
            extent_scale = max(8.0, float(np.nanmax(edge_arr) - np.nanmin(edge_arr)))
            advance_score = min(positive_advance / extent_scale, 1.0)
            mono = _monotone_quality_np(edge_arr)
            perp = _mean_pair_iou_np(perp_grids)
            comp = _compactness_np(aligned)
            bbox_stab = _stability_from_series_np(np.stack(extents, axis=0), scale=6.0)
            score = (
                0.30 * advance_score
                + 0.20 * mono
                + 0.25 * perp
                + 0.15 * comp
                + 0.10 * bbox_stab
            )
            phi_world = (edge_arr - edge_arr[0]) / float(res)
            out.append(
                _PrismaticSpec(
                    axis_dim=axis_dim,
                    sign=sign,
                    score=float(score),
                    phi_world=phi_world,
                    advance=advance,
                    monotone=mono,
                    perp_iou=perp,
                    compactness=comp,
                    bbox_stability=bbox_stab,
                )
            )
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def _base_surface_np(base: np.ndarray) -> np.ndarray:
    dil = _binary_dilation_np(base, radius=1)
    ero = _binary_erosion_np(base, radius=1)
    return dil & (~ero)


def _candidate_centers_np(
    base: np.ndarray,
    state_sets: Sequence[np.ndarray],
    corridor: np.ndarray,
    axis_dim: int,
    res: int,
    top_n: int,
) -> List[Tuple[float, float, float]]:
    move_union = np.zeros_like(base, dtype=bool)
    for coords in state_sets:
        if coords.size:
            move_union[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    near_move = _binary_dilation_np(move_union, radius=3)
    contact = _base_surface_np(base) & near_move
    if int(contact.sum()) == 0 and corridor.size:
        contact = base & _binary_dilation_np(corridor > 0.12, radius=1) & near_move
    d0, d1 = _plane_dims(axis_dim)
    coords = np.argwhere(contact)
    if coords.size == 0:
        mid = 0.5 * (res - 1)
        return [(mid, mid, 0.0)]

    weights = 1.0 + corridor[coords[:, 0], coords[:, 1], coords[:, 2]]
    heat = np.zeros((res, res), dtype=np.float64)
    np.add.at(heat, (coords[:, d0], coords[:, d1]), weights)
    smooth = _smooth2_np(heat, rounds=2)
    flat_idx = np.argsort(smooth.ravel())[::-1]
    centers: List[Tuple[float, float, float]] = []
    used: List[Tuple[int, int]] = []
    for idx in flat_idx:
        val = float(smooth.ravel()[idx])
        if val <= 0:
            break
        a, b = np.unravel_index(idx, smooth.shape)
        if any((a - ua) ** 2 + (b - ub) ** 2 < 9 for ua, ub in used):
            continue
        centers.append((float(a), float(b), val))
        used.append((int(a), int(b)))
        if len(centers) >= top_n:
            break

    boundary_vals = [set(), set()]
    for coords_state in state_sets:
        if coords_state.size == 0:
            continue
        plane = coords_state[:, [d0, d1]]
        for local_dim in range(2):
            vals = plane[:, local_dim].astype(np.float64)
            for q in (0.02, 0.05, 0.10, 0.90, 0.95, 0.98):
                boundary_vals[local_dim].add(int(round(float(np.quantile(vals, q)))))
            boundary_vals[local_dim].add(int(vals.min()))
            boundary_vals[local_dim].add(int(vals.max()))

    extra: List[Tuple[float, float, float]] = []
    for a in boundary_vals[0]:
        for b in boundary_vals[1]:
            if 0 <= a < res and 0 <= b < res:
                aa = int(np.clip(a, 0, res - 1))
                bb = int(np.clip(b, 0, res - 1))
                extra.append((float(aa), float(bb), float(smooth[aa, bb])))

    extra.sort(key=lambda x: x[2], reverse=True)
    for a, b, val in extra:
        if any((a - ua) ** 2 + (b - ub) ** 2 < 4 for ua, ub in used):
            continue
        centers.append((a, b, val))
        used.append((int(a), int(b)))
    return centers


def _axis_prior_from_geom_np(
    scores: np.ndarray,
    axis_dim: int,
    mode: str,
) -> Tuple[float, float]:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.shape[0] != 3 or not np.isfinite(arr).all():
        return 1.0, 0.0

    span = float(arr.max() - arr.min())
    if span < 1e-8:
        return 1.0, 0.0

    if mode == "revolute":
        order = np.argsort(arr)
        best = float(arr[order[0]])
        second = float(arr[order[1]])
        diff = float(arr[axis_dim] - best)
    elif mode == "prismatic":
        order = np.argsort(-arr)
        best = float(arr[order[0]])
        second = float(arr[order[1]])
        diff = float(best - arr[axis_dim])
    else:
        raise ValueError(f"mode must be prismatic or revolute, got {mode!r}")

    separation = max(second - best, 0.0)
    confidence = float(np.clip(separation / max(0.35 * span, 1e-8), 0.0, 1.0))
    scale = max(0.35 * span, 1e-3)
    prior = float(np.exp(-max(diff, 0.0) / scale))
    return prior, confidence


def _component_boundary_support(comp: _Component, center: np.ndarray, d0: int, d1: int) -> float:
    lo = comp.bbox_lo[[d0, d1]].astype(np.float64)
    hi = comp.bbox_hi[[d0, d1]].astype(np.float64)
    lower_gap = np.maximum(lo - center, 0.0)
    upper_gap = np.maximum(center - hi, 0.0)
    outside_vec = lower_gap + upper_gap
    outside_dist = float(np.sqrt((outside_vec * outside_vec).sum()))
    on_boundary_dist = float(
        min(
            abs(center[0] - lo[0]),
            abs(center[0] - hi[0]),
            abs(center[1] - lo[1]),
            abs(center[1] - hi[1]),
        )
    )
    support = math.exp(-outside_dist / 2.0) * math.exp(-on_boundary_dist / 2.0)
    support *= 0.5 + 0.5 * min(comp.corridor_mean / 0.35, 1.0)
    return float(support)


def _boundary_points_for_component(
    comp: _Component,
    center: np.ndarray,
    d0: int,
    d1: int,
    max_points: int = 4,
) -> List[Tuple[float, float]]:
    lo = comp.bbox_lo[[d0, d1]].astype(np.float64)
    hi = comp.bbox_hi[[d0, d1]].astype(np.float64)
    mid = (lo + hi) * 0.5
    points: List[Tuple[float, float]] = [
        (float(lo[0]), float(lo[1])),
        (float(lo[0]), float(hi[1])),
        (float(hi[0]), float(lo[1])),
        (float(hi[0]), float(hi[1])),
        (float(mid[0]), float(lo[1])),
        (float(mid[0]), float(hi[1])),
        (float(lo[0]), float(mid[1])),
        (float(hi[0]), float(mid[1])),
    ]
    plane = comp.coords[:, [d0, d1]].astype(np.float64)
    rel = plane - center
    if rel.size:
        angles = np.arctan2(rel[:, 1], rel[:, 0])
        order = np.argsort(angles)
        for q in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            idx = order[int(round(q * (len(order) - 1)))]
            points.append((float(plane[idx, 0]), float(plane[idx, 1])))

    unique: List[Tuple[float, float]] = []
    for point in points:
        if all((point[0] - old[0]) ** 2 + (point[1] - old[1]) ** 2 > 1.0 for old in unique):
            unique.append(point)
    unique.sort(
        key=lambda p: -float(np.sqrt((((np.asarray(p, dtype=np.float64) - center) ** 2).sum())))
    )
    return unique[:max_points]


def _best_boundary_path(
    component_sets: Sequence[Sequence[_Component]],
    axis_dim: int,
    center: Tuple[float, float],
) -> Tuple[float, np.ndarray, List[Tuple[float, float]], float]:
    d0, d1 = _plane_dims(axis_dim)
    center_arr = np.asarray(center, dtype=np.float64)
    candidate_lists: List[List[Tuple[float, float]]] = []
    support_each: List[float] = []
    for comps in component_sets:
        if not comps:
            return 0.0, np.zeros(len(component_sets), dtype=np.float64), [], 0.0
        supports = [_component_boundary_support(comp, center_arr, d0, d1) for comp in comps]
        best_idx = int(np.argmax(supports))
        support_each.append(float(supports[best_idx]))
        candidate_lists.append(
            _boundary_points_for_component(comps[best_idx], center_arr, d0, d1, max_points=4)
        )

    arrays = [np.asarray(points, dtype=np.float64) for points in candidate_lists]
    index_grid = np.asarray(
        list(itertools.product(*[range(arr.shape[0]) for arr in arrays])),
        dtype=np.int32,
    )
    if index_grid.size == 0:
        return 0.0, np.zeros(len(component_sets), dtype=np.float64), [], float(np.mean(support_each))

    paths = np.empty((index_grid.shape[0], len(arrays), 2), dtype=np.float64)
    for state_idx, arr in enumerate(arrays):
        paths[:, state_idx, :] = arr[index_grid[:, state_idx]]

    rel = paths - center_arr.reshape(1, 1, 2)
    radius = np.sqrt((rel * rel).sum(axis=2))
    angles = np.unwrap(np.arctan2(rel[:, :, 1], rel[:, :, 0]), axis=1)
    diffs = np.diff(angles, axis=1)
    span = np.abs(angles[:, -1] - angles[:, 0])
    span_score = np.minimum(span / (math.pi / 2.0), 1.0) * np.exp(-np.maximum(span - math.pi, 0.0))
    mono = np.maximum((diffs >= -0.08).mean(axis=1), (-diffs >= -0.08).mean(axis=1))
    radius_stability = np.exp(-np.std(radius, axis=1) / 7.0)
    smoothness = np.exp(-np.std(diffs, axis=1) / 1.0) if diffs.shape[1] else np.zeros_like(span)
    endpoint_radius = np.exp(-np.abs(radius[:, -1] - radius[:, 0]) / 7.0)
    scores = (
        0.35 * span_score
        + 0.25 * mono
        + 0.20 * radius_stability
        + 0.10 * smoothness
        + 0.10 * endpoint_radius
    )
    scores[np.median(radius, axis=1) < 4.0] = -1.0
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score < 0.0:
        return 0.0, np.zeros(len(component_sets), dtype=np.float64), [], float(np.mean(support_each))
    path = [(float(p[0]), float(p[1])) for p in paths[best_idx]]
    return best_score, angles[best_idx], path, float(np.mean(support_each))


def _axis_base_support_np(base: np.ndarray, axis_dim: int, center: Tuple[float, float]) -> float:
    d0, d1 = _plane_dims(axis_dim)
    a, b = int(round(center[0])), int(round(center[1]))
    if not (0 <= a < base.shape[d0] and 0 <= b < base.shape[d1]):
        return 0.0
    line_idx: List[object] = [slice(None), slice(None), slice(None)]
    line_idx[d0] = a
    line_idx[d1] = b
    line_count = float(base[tuple(line_idx)].sum())
    near_count = 0.0
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            aa = a + da
            bb = b + db
            if not (0 <= aa < base.shape[d0] and 0 <= bb < base.shape[d1]):
                continue
            idx: List[object] = [slice(None), slice(None), slice(None)]
            idx[d0] = aa
            idx[d1] = bb
            near_count += float(base[tuple(idx)].sum())
    return float(0.70 * min(line_count / 18.0, 1.0) + 0.30 * min(near_count / 72.0, 1.0))


def _score_revolute_specs(
    base: np.ndarray,
    state_sets: Sequence[np.ndarray],
    component_sets: Sequence[Sequence[_Component]],
    corridor: np.ndarray,
    axis_dim: int,
    res: int,
    axis_prior: float = 1.0,
    axis_prior_confidence: float = 0.0,
    top_n_centers: int = 160,
) -> List[_RevoluteSpec]:
    d0, d1 = _plane_dims(axis_dim)
    centers = _candidate_centers_np(base, state_sets, corridor, axis_dim, res, top_n_centers)
    out: List[_RevoluteSpec] = []
    for c0, c1, contact in centers:
        radius_stats: List[np.ndarray] = []
        axis_stats: List[np.ndarray] = []
        boundary_score, boundary_angles, boundary_path, boundary_support = _best_boundary_path(
            component_sets, axis_dim, (c0, c1)
        )
        for coords in state_sets:
            if coords.size == 0:
                radius_stats.append(np.full(5, np.nan))
                axis_stats.append(np.full(3, np.nan))
                continue
            plane = coords[:, [d0, d1]].astype(np.float64)
            rel = plane - np.array([c0, c1], dtype=np.float64)
            rad = np.sqrt((rel * rel).sum(axis=1))
            radius_stats.append(np.quantile(rad, [0.10, 0.30, 0.50, 0.70, 0.90]))
            axis_vals = coords[:, axis_dim].astype(np.float64)
            axis_stats.append(np.quantile(axis_vals, [0.10, 0.50, 0.90]))

        radius_arr = np.stack(radius_stats, axis=0)
        axis_arr = np.stack(axis_stats, axis=0)
        radius_stab = _stability_from_series_np(radius_arr, scale=4.0)
        axis_stab = _stability_from_series_np(axis_arr, scale=5.0)
        angle_arr = _fill_nan_linear_np(np.asarray(boundary_angles, dtype=np.float64))
        angle_span = float(abs(angle_arr[-1] - angle_arr[0]))
        span_score = min(angle_span / (math.pi / 2.0), 1.0)
        angle_mono = _monotone_quality_np(angle_arr)

        aligned = []
        theta_c = angle_arr[2 if len(angle_arr) > 2 else 0]
        rot_center = np.array([c0, c1], dtype=np.float64)
        for coords, theta in zip(state_sets, angle_arr):
            moved = coords.astype(np.float64).copy()
            if moved.size:
                delta = theta_c - theta
                plane = moved[:, [d0, d1]]
                rel0 = plane[:, 0] - rot_center[0]
                rel1 = plane[:, 1] - rot_center[1]
                cos_d = math.cos(delta)
                sin_d = math.sin(delta)
                moved[:, d0] = rel0 * cos_d - rel1 * sin_d + rot_center[0]
                moved[:, d1] = rel0 * sin_d + rel1 * cos_d + rot_center[1]
            aligned.append(_raster3_np(moved, res=res))
        comp = _compactness_np(aligned)
        contact_score = min(float(contact) / 8.0, 1.0)
        base_support = _axis_base_support_np(base, axis_dim, (c0, c1))
        path_arr = np.asarray(boundary_path, dtype=np.float64)
        if path_arr.size and np.isfinite(path_arr).all():
            path_range = np.ptp(path_arr, axis=0)
            arc_balance = float(np.min(path_range) / max(float(np.max(path_range)), 1e-6))
        else:
            arc_balance = 0.0
        hinge_support = boundary_support
        score = (
            0.20 * radius_stab
            + 0.15 * axis_stab
            + 0.13 * span_score
            + 0.07 * angle_mono
            + 0.08 * boundary_score
            + 0.08 * arc_balance
            + 0.05 * comp
            + 0.06 * contact_score
            + 0.10 * hinge_support
            + 0.08 * base_support
        )
        prior_w = 0.18 * float(np.clip(axis_prior_confidence, 0.0, 1.0))
        score = (1.0 - prior_w) * score + prior_w * float(axis_prior)
        phi_angle = angle_arr - angle_arr[0]
        out.append(
            _RevoluteSpec(
                axis_dim=axis_dim,
                center=(float(c0), float(c1)),
                score=float(score),
                phi_angle=phi_angle,
                contact=float(contact),
                radius_stability=radius_stab,
                axis_stability=axis_stab,
                angle_span=angle_span,
                angle_mono=angle_mono,
                compactness=comp,
                hinge_support=hinge_support,
                axis_base_support=base_support,
                boundary_path_score=boundary_score,
                arc_balance=arc_balance,
            )
        )
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def _axis_tensor(axis_dim: int, sign: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    axis = torch.zeros(3, device=device, dtype=dtype)
    axis[axis_dim] = float(sign)
    return axis


def _origin_from_revolute_center(
    axis_dim: int,
    center: Tuple[float, float],
    res: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    d0, d1 = _plane_dims(axis_dim)
    origin = torch.zeros(3, device=device, dtype=dtype)
    origin[d0] = float((center[0] + 0.5) / float(res) - 0.5)
    origin[d1] = float((center[1] + 0.5) / float(res) - 0.5)
    return origin


# ---------------------------------------------------------------------------
# Public result schema
# ---------------------------------------------------------------------------


@dataclass
class CandidateResult:
    """One (type, axis, origin, phi_k) candidate + its score."""

    type_str: str                        # "prismatic" | "revolute"
    axis: torch.Tensor                   # (3,) cardinal unit vector
    origin: torch.Tensor                 # (3,) world
    phi_k: torch.Tensor                  # (K,) per-state progress
    score: CandidateScore                # voxel-level physical score breakdown


@dataclass
class JointTypeResult:
    """v3 dual-candidate output (replaces v2 single line-vs-arc result)."""

    # Selected primary
    type_str: str                        # "prismatic" | "revolute" | "uncertain"
    type_logit: float                    # log(best_pris.score / best_rev.score)
    confidence: float                    # |type_logit| / margin_norm, clamped [0,1]

    # Best candidate per type (Stage D dual-clone uses both)
    best_pris: Optional[CandidateResult] = None
    best_rev: Optional[CandidateResult] = None

    # All candidates for diagnostics
    all_candidates: List[CandidateResult] = field(default_factory=list)

    # Voxel-statistical diagnostics
    n_valid_states: int = 0
    pris_geom_scores: Optional[torch.Tensor] = None  # (6,) FreeArt3D argmax values
    rev_geom_scores: Optional[torch.Tensor] = None   # (6,) FreeArt3D argmin values

    # Legacy fields for confidence.py / run_stage_c_init.py compatibility
    residual_line: float = float("inf")
    residual_arc: float = float("inf")
    line_origin: Optional[np.ndarray] = None
    line_direction: Optional[np.ndarray] = None
    arc_center: Optional[np.ndarray] = None
    arc_normal: Optional[np.ndarray] = None
    arc_radius: Optional[float] = None


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def detect_joint_type_v3(
    V_per_state_voxel: List[torch.Tensor],            # K tensors of (N_k, 3) int
    V_union_voxel: torch.Tensor,                       # (N_u, 3) int
    valid_state: List[bool],
    O_base_canonical: Optional[torch.Tensor],         # (res, res, res) for conflict check
    anchors_voxel: Optional[torch.Tensor],            # (N_a, 3) int, may be None
    canonical_state_idx: int = 2,
    res: int = 64,
    type_margin: float = 0.20,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    M_motion_corridor_64: Optional[torch.Tensor] = None,
) -> JointTypeResult:
    """v3 cardinal-cand + voxel-scoring joint type detection.

    Replaces v2's centroid line-vs-arc residual ratio.
    """
    if device is None:
        device = V_union_voxel.device
    K = len(V_per_state_voxel)
    n_valid = int(sum(valid_state))

    if n_valid < 2 or V_union_voxel.shape[0] < 10:
        # Degenerate input
        return JointTypeResult(
            type_str="uncertain", type_logit=0.0, confidence=0.0,
            n_valid_states=n_valid,
        )

    # ---- Step A: World coordinates ----
    V_per_state_world: List[torch.Tensor] = []
    for V_int in V_per_state_voxel:
        if V_int.shape[0] == 0:
            V_per_state_world.append(torch.zeros(0, 3, device=device, dtype=dtype))
        else:
            V_per_state_world.append(
                voxel_to_world(V_int, res=res).to(device=device, dtype=dtype)
            )
    V_union_world = voxel_to_world(V_union_voxel, res=res).to(device=device, dtype=dtype)
    swept_centroid = V_union_world.mean(dim=0)
    anchors_world = (
        voxel_to_world(anchors_voxel, res=res).to(device=device, dtype=dtype)
        if anchors_voxel is not None and anchors_voxel.shape[0] > 0
        else None
    )

    # ---- Step B: FreeArt3D-style axis invariant (geometric prior) ----
    pris_geom, rev_geom = freeart3d_axis_invariant(V_per_state_world)

    # ---- Step C+D: Branched candidate generation + physical scoring ----
    all_candidates: List[CandidateResult] = []
    pris_candidates: List[CandidateResult] = []
    rev_candidates: List[CandidateResult] = []

    if O_base_canonical is None:
        base_np = np.zeros((res, res, res), dtype=bool)
    else:
        base_np = O_base_canonical.detach().cpu().numpy().astype(bool)
    if M_motion_corridor_64 is None:
        corridor_np = np.zeros((res, res, res), dtype=np.float32)
    else:
        corridor_np = M_motion_corridor_64.detach().cpu().numpy().astype(np.float32)

    clean_state_sets, component_sets = _clean_components_from_voxels(
        V_per_state_voxel=V_per_state_voxel,
        O_base_canonical=O_base_canonical,
        M_motion_corridor_64=M_motion_corridor_64,
        res=res,
    )

    pris_geom_np = pris_geom.detach().cpu().numpy().astype(np.float64)
    rev_geom_np = rev_geom.detach().cpu().numpy().astype(np.float64)

    for spec in _score_prismatic_specs(clean_state_sets, res=res):
        axis = _axis_tensor(spec.axis_dim, spec.sign, device, dtype)
        phi_pris = torch.from_numpy(spec.phi_world).to(device=device, dtype=dtype)
        advance_score = min(max(spec.advance, 0.0) / max(abs(spec.advance), 8.0), 1.0)
        axis_prior, axis_prior_conf = _axis_prior_from_geom_np(
            pris_geom_np, spec.axis_dim, mode="prismatic",
        )
        prior_w = 0.12 * axis_prior_conf
        score = (1.0 - prior_w) * float(spec.score) + prior_w * float(axis_prior)
        score_pris = CandidateScore(
            score=float(score),
            consistency=float(spec.perp_iou),
            conflict=float(max(0.0, 1.0 - spec.compactness)),
            coverage=float(advance_score),
            contact_compat=float(spec.bbox_stability),
            monotone_quality=float(spec.monotone),
            valid_states_used=n_valid,
            radius_stability=float(spec.bbox_stability),
            axis_stability=float(spec.bbox_stability),
            axis_prior=float(axis_prior),
            axis_prior_confidence=float(axis_prior_conf),
        )
        c_pris = CandidateResult(
            type_str="prismatic", axis=axis, origin=swept_centroid.clone(),
            phi_k=phi_pris, score=score_pris,
        )
        pris_candidates.append(c_pris)
        all_candidates.append(c_pris)

    for axis_dim in range(3):
        rev_axis_prior, rev_axis_prior_conf = _axis_prior_from_geom_np(
            rev_geom_np, axis_dim, mode="revolute",
        )
        for spec in _score_revolute_specs(
            base=base_np,
            state_sets=clean_state_sets,
            component_sets=component_sets,
            corridor=corridor_np,
            axis_dim=axis_dim,
            res=res,
            axis_prior=rev_axis_prior,
            axis_prior_confidence=rev_axis_prior_conf,
            top_n_centers=160,
        ):
            phi_np = spec.phi_angle.astype(np.float64)
            axis_sign = 1
            if phi_np[-1] < phi_np[0]:
                axis_sign = -1
                phi_np = -phi_np
            phi_np = phi_np - phi_np[0]
            axis = _axis_tensor(axis_dim, axis_sign, device, dtype)
            origin = _origin_from_revolute_center(axis_dim, spec.center, res, device, dtype)
            score_rev = CandidateScore(
                score=float(spec.score),
                consistency=float(spec.radius_stability),
                conflict=float(max(0.0, 1.0 - spec.compactness)),
                coverage=float(spec.boundary_path_score),
                contact_compat=float(max(spec.hinge_support, spec.axis_base_support)),
                monotone_quality=float(spec.angle_mono),
                valid_states_used=n_valid,
                boundary_path_score=float(spec.boundary_path_score),
                axis_base_support=float(spec.axis_base_support),
                arc_balance=float(spec.arc_balance),
                radius_stability=float(spec.radius_stability),
                axis_stability=float(spec.axis_stability),
                axis_prior=float(rev_axis_prior),
                axis_prior_confidence=float(rev_axis_prior_conf),
            )
            c_rev = CandidateResult(
                type_str="revolute", axis=axis, origin=origin,
                phi_k=torch.from_numpy(phi_np).to(device=device, dtype=dtype),
                score=score_rev,
            )
            rev_candidates.append(c_rev)
            all_candidates.append(c_rev)

    # ---- Step E: Best per type + type decision ----
    best_pris = max(pris_candidates, key=lambda c: c.score.score)
    best_rev = max(rev_candidates, key=lambda c: c.score.score)

    eps = 1e-6
    s_pris = max(best_pris.score.score, eps)
    s_rev = max(best_rev.score.score, eps)
    type_logit = float(math.log(s_pris / s_rev))
    prismatic_edge_evidence = (
        best_pris.score.coverage
        * best_pris.score.monotone_quality
        * best_pris.score.axis_prior
    )
    if s_pris >= 0.94 * s_rev and prismatic_edge_evidence > 0.75:
        type_logit += 0.18 * min((prismatic_edge_evidence - 0.75) / 0.25, 1.0)

    if type_logit > type_margin:
        type_str = "prismatic"
    elif type_logit < -type_margin:
        type_str = "revolute"
    else:
        type_str = "uncertain"

    # Confidence from margin
    confidence = float(min(abs(type_logit) / max(type_margin * 2.0, eps), 1.0))

    return JointTypeResult(
        type_str=type_str,
        type_logit=type_logit,
        confidence=confidence,
        best_pris=best_pris,
        best_rev=best_rev,
        all_candidates=all_candidates,
        n_valid_states=n_valid,
        pris_geom_scores=pris_geom.detach().cpu(),
        rev_geom_scores=rev_geom.detach().cpu(),
    )


# ---------------------------------------------------------------------------
# Backward-compat wrapper for run_stage_c_init.py + confidence.py
# ---------------------------------------------------------------------------


def detect_joint_type(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Legacy entry name kept for compatibility with existing imports.

    New code should call `detect_joint_type_v3` directly with explicit args.
    This wrapper is intentionally minimal -- callers pass the v3 signature.
    """
    return detect_joint_type_v3(*args, **kwargs)


__all__ = [
    "JointTypeResult", "CandidateResult", "detect_joint_type_v3", "detect_joint_type",
    "build_per_state_voxel_sets", "cardinal_axes",
]

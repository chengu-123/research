"""Research-only Stage C candidate scoring.

This script does not modify production Stage C. It inspects Stage B artifacts
and evaluates physically constrained prismatic/revolute candidates under
partial, noisy, non-corresponded voxel observations.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage


SAMPLE_GT = {
    "30857": {
        "type": "prismatic",
        "axis_dim": 1,
        "signed_axis": -1,
        "path_dim": 1,
        "path_start": 27.0,
        "path_end": 16.0,
    },
    "7201": {
        "type": "revolute",
        "axis_dim": 0,
        "center_plane": (30.0, 18.0),  # (y, z)
        "path_start": (29.0, 31.0),
        "path_end": (15.0, 17.0),
    },
    "7128": {
        "type": "revolute",
        "axis_dim": 2,
        "center_plane": (19.0, 32.0),  # (x, y)
        "path_start": (37.0, 32.0),
        "path_end": (19.0, 14.0),
    },
}


@dataclass
class Component:
    coords: np.ndarray
    count: int
    centroid: np.ndarray
    bbox_lo: np.ndarray
    bbox_hi: np.ndarray
    corridor_mean: float
    score: float


@dataclass
class PrismaticCandidate:
    axis_dim: int
    sign: int
    score: float
    edge: List[float]
    centroid_path: List[float]
    advance: float
    monotone: float
    perp_iou: float
    compactness: float
    bbox_stability: float


@dataclass
class RevoluteCandidate:
    axis_dim: int
    center: Tuple[float, float]
    score: float
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
    angles: List[float]
    path: List[Tuple[float, float]]


def load_artifacts(root: str, sample_id: str) -> Dict[str, np.ndarray]:
    stage_b = os.path.join(root, sample_id, "stage_b")
    return {
        "base": np.load(os.path.join(stage_b, "O_base_canonical.npy")).astype(bool),
        "move": np.load(os.path.join(stage_b, "O_move_per_state.npy")).astype(bool),
        "p_move": np.load(os.path.join(stage_b, "P_move_evidence_per_state.npy")).astype(np.float32),
        "p_base": np.load(os.path.join(stage_b, "P_base_canonical.npy")).astype(np.float32),
        "corridor": np.load(
            os.path.join(stage_b, "viz", "bmcsa", "M_motion_corridor_64.npy")
        ).astype(np.float32),
    }


def component_list(mask: np.ndarray, corridor: np.ndarray, min_count: int) -> List[Component]:
    labels, n_labels = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    out: List[Component] = []
    for idx in range(1, n_labels + 1):
        coords = np.argwhere(labels == idx)
        count = int(coords.shape[0])
        if count < min_count:
            continue
        centroid = coords.mean(axis=0)
        bbox_lo = coords.min(axis=0)
        bbox_hi = coords.max(axis=0)
        corr = float(corridor[coords[:, 0], coords[:, 1], coords[:, 2]].mean())
        score = math.sqrt(count) * (0.25 + corr)
        out.append(Component(coords, count, centroid, bbox_lo, bbox_hi, corr, score))
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def build_clean_state_sets(
    base: np.ndarray,
    move: np.ndarray,
    p_move: np.ndarray,
    corridor: np.ndarray,
    soft_tau: float,
    min_component: int,
    keep_mass_frac: float,
) -> Tuple[List[np.ndarray], List[List[Component]]]:
    state_sets: List[np.ndarray] = []
    component_sets: List[List[Component]] = []
    for k in range(move.shape[0]):
        raw = (move[k] | (p_move[k] >= soft_tau)) & (~base)
        comps = component_list(raw, corridor, min_component)
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
        final: List[Component] = []
        for comp in kept:
            final.append(comp)
            acc += comp.count
            if total > 0 and acc / total >= keep_mass_frac:
                break

        coords = np.concatenate([c.coords for c in final], axis=0).astype(np.int32)
        state_sets.append(coords)
        component_sets.append(final)
    return state_sets, component_sets


def plane_dims(axis_dim: int) -> Tuple[int, int]:
    dims = [0, 1, 2]
    dims.remove(axis_dim)
    return dims[0], dims[1]


def raster2(coords_2d: np.ndarray, res: int = 64) -> np.ndarray:
    grid = np.zeros((res, res), dtype=bool)
    if coords_2d.size == 0:
        return grid
    xy = np.rint(coords_2d).astype(np.int32)
    valid = np.all((xy >= 0) & (xy < res), axis=1)
    xy = xy[valid]
    grid[xy[:, 0], xy[:, 1]] = True
    return grid


def raster3(coords: np.ndarray, res: int = 64) -> np.ndarray:
    grid = np.zeros((res, res, res), dtype=bool)
    if coords.size == 0:
        return grid
    xyz = np.rint(coords).astype(np.int32)
    valid = np.all((xyz >= 0) & (xyz < res), axis=1)
    xyz = xyz[valid]
    grid[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True
    return grid


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def mean_pair_iou(grids: Sequence[np.ndarray]) -> float:
    vals: List[float] = []
    for i in range(len(grids)):
        for j in range(i + 1, len(grids)):
            vals.append(iou(grids[i], grids[j]))
    return float(np.mean(vals)) if vals else 0.0


def compactness(grids: Sequence[np.ndarray]) -> float:
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


def monotone_quality(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    diffs = np.diff(arr)
    pos = int((diffs > 0).sum())
    neg = int((diffs < 0).sum())
    return max(pos, neg) / max(len(diffs), 1)


def stability_from_series(series: np.ndarray, scale: float) -> float:
    if series.size == 0:
        return 0.0
    std = np.nanmean(np.nanstd(series, axis=0))
    return float(math.exp(-std / max(scale, 1e-6)))


def score_prismatic(state_sets: Sequence[np.ndarray], res: int = 64) -> List[PrismaticCandidate]:
    out: List[PrismaticCandidate] = []
    for axis_dim in range(3):
        other = plane_dims(axis_dim)
        for sign in (-1, 1):
            edges: List[float] = []
            cent_path: List[float] = []
            perp_grids: List[np.ndarray] = []
            extents: List[np.ndarray] = []
            for coords in state_sets:
                if coords.size == 0:
                    edges.append(float("nan"))
                    cent_path.append(float("nan"))
                    perp_grids.append(np.zeros((res, res), dtype=bool))
                    extents.append(np.zeros(2, dtype=np.float64))
                    continue
                signed = sign * coords[:, axis_dim].astype(np.float64)
                edges.append(float(np.quantile(signed, 0.95)))
                cent_path.append(float(coords[:, axis_dim].mean()))
                perp = coords[:, other]
                perp_grids.append(raster2(perp, res=res))
                extents.append(perp.max(axis=0) - perp.min(axis=0))

            edge_arr = fill_nan_linear(np.asarray(edges, dtype=np.float64))
            edge_shift = edge_arr - edge_arr[2]
            aligned = []
            for coords, shift in zip(state_sets, edge_shift):
                moved = coords.astype(np.float64).copy()
                if moved.size:
                    moved[:, axis_dim] -= sign * shift
                aligned.append(raster3(moved, res=res))

            advance = float(edge_arr[-1] - edge_arr[0])
            positive_advance = max(advance, 0.0)
            extent_scale = max(8.0, np.nanmax(edge_arr) - np.nanmin(edge_arr))
            advance_score = min(positive_advance / extent_scale, 1.0)
            mono = monotone_quality(edge_arr)
            perp = mean_pair_iou(perp_grids)
            comp = compactness(aligned)
            bbox_stab = stability_from_series(np.stack(extents, axis=0), scale=6.0)
            score = (
                0.30 * advance_score
                + 0.20 * mono
                + 0.25 * perp
                + 0.15 * comp
                + 0.10 * bbox_stab
            )
            out.append(
                PrismaticCandidate(
                    axis_dim=axis_dim,
                    sign=sign,
                    score=float(score),
                    edge=[float(x) for x in edge_arr],
                    centroid_path=[float(x) for x in cent_path],
                    advance=advance,
                    monotone=mono,
                    perp_iou=perp,
                    compactness=comp,
                    bbox_stability=bbox_stab,
                )
            )
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def fill_nan_linear(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float64).copy()
    valid = np.isfinite(arr)
    if valid.all():
        return arr
    if valid.sum() < 2:
        return np.linspace(0.0, 1.0, arr.shape[0], dtype=np.float64)
    idx = np.arange(arr.shape[0])
    arr[~valid] = np.interp(idx[~valid], idx[valid], arr[valid])
    return arr


def base_surface(base: np.ndarray) -> np.ndarray:
    dil = ndimage.binary_dilation(base, structure=np.ones((3, 3, 3), dtype=bool))
    ero = ndimage.binary_erosion(base, structure=np.ones((3, 3, 3), dtype=bool))
    return dil & (~ero)


def candidate_centers(
    base: np.ndarray,
    state_sets: Sequence[np.ndarray],
    corridor: np.ndarray,
    axis_dim: int,
    top_n: int,
) -> List[Tuple[float, float, float]]:
    move_union = np.zeros_like(base, dtype=bool)
    for coords in state_sets:
        if coords.size:
            move_union[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    near_move = ndimage.binary_dilation(move_union, structure=np.ones((5, 5, 5), dtype=bool))
    contact = base_surface(base) & near_move
    contact |= base & ndimage.binary_dilation(corridor > 0.08, structure=np.ones((3, 3, 3), dtype=bool))
    d0, d1 = plane_dims(axis_dim)
    coords = np.argwhere(contact)
    if coords.size == 0:
        return [(31.5, 31.5, 0.0)]
    weights = 1.0 + corridor[coords[:, 0], coords[:, 1], coords[:, 2]]
    heat = np.zeros((64, 64), dtype=np.float64)
    np.add.at(heat, (coords[:, d0], coords[:, d1]), weights)
    smooth = ndimage.gaussian_filter(heat, sigma=1.2)
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

    # Add boundary-intersection centers. A revolute hinge lies on the near-base
    # extremal boundary of the moving part, so its projected center is often an
    # intersection of a stable low/high edge from different states. Heatmap-only
    # candidates overfit thick contact bands and miss this sparse hinge cue.
    boundary_vals = [set(), set()]
    for coords in state_sets:
        if coords.size == 0:
            continue
        plane = coords[:, [d0, d1]]
        for local_dim in range(2):
            vals = plane[:, local_dim].astype(np.float64)
            for q in (0.02, 0.05, 0.10, 0.90, 0.95, 0.98):
                boundary_vals[local_dim].add(int(round(float(np.quantile(vals, q)))))
            boundary_vals[local_dim].add(int(vals.min()))
            boundary_vals[local_dim].add(int(vals.max()))

    extra: List[Tuple[float, float, float]] = []
    for a in boundary_vals[0]:
        for b in boundary_vals[1]:
            if 0 <= a < 64 and 0 <= b < 64:
                aa = int(np.clip(a, 0, 63))
                bb = int(np.clip(b, 0, 63))
                heat_val = float(smooth[aa, bb])
                extra.append((float(aa), float(bb), heat_val))

    extra.sort(key=lambda x: x[2], reverse=True)
    for a, b, val in extra:
        if any((a - ua) ** 2 + (b - ub) ** 2 < 4 for ua, ub in used):
            continue
        centers.append((a, b, val))
        used.append((int(a), int(b)))
    return centers


def angle_unwrap(values: np.ndarray) -> np.ndarray:
    return np.unwrap(values.astype(np.float64))


def component_boundary_support(comp: Component, center: np.ndarray, d0: int, d1: int) -> float:
    lo = comp.bbox_lo[[d0, d1]].astype(np.float64)
    hi = comp.bbox_hi[[d0, d1]].astype(np.float64)
    lower_gap = np.maximum(lo - center, 0.0)
    upper_gap = np.maximum(center - hi, 0.0)
    outside_dist = float(np.linalg.norm(lower_gap + upper_gap))
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


def boundary_points_for_component(
    comp: Component,
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
    unique.sort(key=lambda p: -float(np.linalg.norm(np.asarray(p, dtype=np.float64) - center)))
    return unique[:max_points]


def best_boundary_path(
    component_sets: Sequence[Sequence[Component]],
    axis_dim: int,
    center: Tuple[float, float],
) -> Tuple[float, List[float], List[Tuple[float, float]], float]:
    d0, d1 = plane_dims(axis_dim)
    center_arr = np.asarray(center, dtype=np.float64)
    candidate_lists: List[List[Tuple[float, float]]] = []
    support_each: List[float] = []
    for comps in component_sets:
        if not comps:
            support_each.append(0.0)
            candidate_lists.append([(float("nan"), float("nan"))])
            continue
        supports = [component_boundary_support(comp, center_arr, d0, d1) for comp in comps]
        best_idx = int(np.argmax(supports))
        support_each.append(float(supports[best_idx]))
        candidate_lists.append(
            boundary_points_for_component(comps[best_idx], center_arr, d0, d1, max_points=4)
        )

    arrays = [np.asarray(points, dtype=np.float64) for points in candidate_lists]
    index_grid = np.asarray(
        list(itertools.product(*[range(arr.shape[0]) for arr in arrays])),
        dtype=np.int32,
    )
    if index_grid.size == 0:
        support_mean = float(np.mean(support_each)) if support_each else 0.0
        return 0.0, [0.0 for _ in candidate_lists], [], support_mean

    paths = np.empty((index_grid.shape[0], len(arrays), 2), dtype=np.float64)
    for state_idx, arr in enumerate(arrays):
        paths[:, state_idx, :] = arr[index_grid[:, state_idx]]

    rel = paths - center_arr.reshape(1, 1, 2)
    radius = np.linalg.norm(rel, axis=2)
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
    best_angles = angles[best_idx]
    best_path_arr = paths[best_idx]
    best_path = [(float(p[0]), float(p[1])) for p in best_path_arr]

    support_mean = float(np.mean(support_each)) if support_each else 0.0
    if best_score < 0.0:
        return 0.0, [0.0 for _ in candidate_lists], best_path, support_mean
    return best_score, [float(x) for x in best_angles], best_path, support_mean


def axis_base_support(base: np.ndarray, axis_dim: int, center: Tuple[float, float]) -> float:
    d0, d1 = plane_dims(axis_dim)
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


def score_revolute(
    base: np.ndarray,
    state_sets: Sequence[np.ndarray],
    component_sets: Sequence[Sequence[Component]],
    corridor: np.ndarray,
    axis_dim: int,
    top_n_centers: int,
    res: int = 64,
) -> List[RevoluteCandidate]:
    d0, d1 = plane_dims(axis_dim)
    centers = candidate_centers(base, state_sets, corridor, axis_dim, top_n_centers)
    out: List[RevoluteCandidate] = []
    for c0, c1, contact in centers:
        radius_stats: List[np.ndarray] = []
        axis_stats: List[np.ndarray] = []
        hinge_support_each: List[float] = []
        boundary_score, boundary_angles, boundary_path, boundary_support = best_boundary_path(
            component_sets, axis_dim, (c0, c1)
        )
        for coords in state_sets:
            if coords.size == 0:
                radius_stats.append(np.full(5, np.nan))
                axis_stats.append(np.full(3, np.nan))
                hinge_support_each.append(0.0)
                continue
            plane = coords[:, [d0, d1]].astype(np.float64)
            rel = plane - np.array([c0, c1], dtype=np.float64)
            rad = np.linalg.norm(rel, axis=1)
            radius_stats.append(np.quantile(rad, [0.10, 0.30, 0.50, 0.70, 0.90]))
            axis_vals = coords[:, axis_dim].astype(np.float64)
            axis_stats.append(np.quantile(axis_vals, [0.10, 0.50, 0.90]))

        # A physical revolute axis should be close to the moving component's
        # near-axis boundary in most states. This rejects false centers inside a
        # thick swept/contact region that can explain translation as a large arc.
        center_arr = np.array([c0, c1], dtype=np.float64)
        for comps in component_sets:
            best = 0.0
            for comp in comps:
                best = max(best, component_boundary_support(comp, center_arr, d0, d1))
            hinge_support_each.append(best)

        radius_arr = np.stack(radius_stats, axis=0)
        axis_arr = np.stack(axis_stats, axis=0)
        radius_stab = stability_from_series(radius_arr, scale=4.0)
        axis_stab = stability_from_series(axis_arr, scale=5.0)
        angle_arr = fill_nan_linear(np.asarray(boundary_angles, dtype=np.float64))
        angle_span = float(abs(angle_arr[-1] - angle_arr[0]))
        span_score = min(angle_span / (math.pi / 2.0), 1.0)
        angle_mono = monotone_quality(angle_arr)

        aligned = []
        theta_c = angle_arr[2]
        rot_center = np.array([c0, c1], dtype=np.float64)
        for coords, theta in zip(state_sets, angle_arr):
            moved = coords.astype(np.float64).copy()
            if moved.size:
                delta = theta_c - theta
                R = np.array(
                    [
                        [math.cos(delta), -math.sin(delta)],
                        [math.sin(delta), math.cos(delta)],
                    ],
                    dtype=np.float64,
                )
                plane = moved[:, [d0, d1]]
                moved[:, [d0, d1]] = (plane - rot_center) @ R.T + rot_center
            aligned.append(raster3(moved, res=res))
        comp = compactness(aligned)
        contact_score = min(contact / 8.0, 1.0)
        hinge_support = max(
            boundary_support,
            float(np.mean(hinge_support_each)) if hinge_support_each else 0.0,
        )
        base_support = axis_base_support(base, axis_dim, (c0, c1))
        path_arr = np.asarray(boundary_path, dtype=np.float64)
        if path_arr.size and np.isfinite(path_arr).all():
            path_range = np.ptp(path_arr, axis=0)
            arc_balance = float(np.min(path_range) / max(float(np.max(path_range)), 1e-6))
        else:
            arc_balance = 0.0
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
        out.append(
            RevoluteCandidate(
                axis_dim=axis_dim,
                center=(float(c0), float(c1)),
                score=float(score),
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
                angles=[float(x) for x in angle_arr],
                path=boundary_path,
            )
        )
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def candidate_dict_pri(c: PrismaticCandidate) -> Dict[str, object]:
    return {
        "axis_dim": c.axis_dim,
        "sign": c.sign,
        "score": round(c.score, 6),
        "edge": [round(x, 3) for x in c.edge],
        "advance": round(c.advance, 3),
        "monotone": round(c.monotone, 3),
        "perp_iou": round(c.perp_iou, 3),
        "compactness": round(c.compactness, 3),
        "bbox_stability": round(c.bbox_stability, 3),
    }


def candidate_dict_rev(c: RevoluteCandidate) -> Dict[str, object]:
    return {
        "axis_dim": c.axis_dim,
        "center": [round(c.center[0], 3), round(c.center[1], 3)],
        "score": round(c.score, 6),
        "contact": round(c.contact, 3),
        "radius_stability": round(c.radius_stability, 3),
        "axis_stability": round(c.axis_stability, 3),
        "angle_span_deg": round(math.degrees(c.angle_span), 2),
        "angle_mono": round(c.angle_mono, 3),
        "compactness": round(c.compactness, 3),
        "hinge_support": round(c.hinge_support, 3),
        "axis_base_support": round(c.axis_base_support, 3),
        "boundary_path_score": round(c.boundary_path_score, 3),
        "arc_balance": round(c.arc_balance, 3),
        "path": [[round(a, 3), round(b, 3)] for a, b in c.path],
    }


def evaluate_gt(sample_id: str, pri: PrismaticCandidate, rev: RevoluteCandidate) -> Dict[str, object]:
    gt = SAMPLE_GT.get(sample_id)
    if gt is None:
        return {}
    out: Dict[str, object] = {}
    if gt["type"] == "prismatic":
        out["gt_type"] = "prismatic"
        out["pri_axis_match"] = pri.axis_dim == gt["axis_dim"] and pri.sign == gt["signed_axis"]
        out["pri_path_start_end"] = [round(-pri.sign * pri.edge[0], 3), round(-pri.sign * pri.edge[-1], 3)]
    else:
        center = np.asarray(rev.center)
        gt_center = np.asarray(gt["center_plane"], dtype=np.float64)
        out["gt_type"] = "revolute"
        out["rev_axis_match"] = rev.axis_dim == gt["axis_dim"]
        out["rev_center_l2"] = round(float(np.linalg.norm(center - gt_center)), 3)
        out["rev_path_start_end"] = [
            [round(x, 3) for x in rev.path[0]],
            [round(x, 3) for x in rev.path[-1]],
        ]
    return out


def summarize_components(component_sets: Sequence[Sequence[Component]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for k, comps in enumerate(component_sets):
        rows.append(
            {
                "state": k,
                "components": [
                    {
                        "count": c.count,
                        "centroid": [round(float(x), 2) for x in c.centroid],
                        "bbox_lo": c.bbox_lo.astype(int).tolist(),
                        "bbox_hi": c.bbox_hi.astype(int).tolist(),
                        "corridor_mean": round(c.corridor_mean, 4),
                    }
                    for c in comps
                ],
            }
        )
    return rows


def run_sample(root: str, sample_id: str, args: argparse.Namespace) -> Dict[str, object]:
    art = load_artifacts(root, sample_id)
    state_sets, components = build_clean_state_sets(
        art["base"],
        art["move"],
        art["p_move"],
        art["corridor"],
        soft_tau=args.soft_tau,
        min_component=args.min_component,
        keep_mass_frac=args.keep_mass_frac,
    )
    pris = score_prismatic(state_sets)
    rev_all: List[RevoluteCandidate] = []
    for axis_dim in range(3):
        rev_all.extend(
            score_revolute(
                art["base"],
                state_sets,
                components,
                art["corridor"],
                axis_dim,
                top_n_centers=args.top_centers,
            )
        )
    rev_all.sort(key=lambda c: c.score, reverse=True)
    best_pri = pris[0]
    best_rev = rev_all[0]
    type_margin = best_pri.score - best_rev.score
    if type_margin > args.type_margin:
        pred_type = "prismatic"
    elif type_margin < -args.type_margin:
        pred_type = "revolute"
    else:
        pred_type = "uncertain"
    return {
        "sample_id": sample_id,
        "pred_type": pred_type,
        "type_margin_pri_minus_rev": round(type_margin, 6),
        "best_prismatic": candidate_dict_pri(best_pri),
        "top_prismatic": [candidate_dict_pri(c) for c in pris[:6]],
        "best_revolute": candidate_dict_rev(best_rev),
        "top_revolute": [candidate_dict_rev(c) for c in rev_all[:10]],
        "gt_eval": evaluate_gt(sample_id, best_pri, best_rev),
        "components": summarize_components(components),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.path.join("mine", "outputs"))
    parser.add_argument("--samples", nargs="+", default=["30857", "7201", "7128"])
    parser.add_argument("--soft_tau", type=float, default=0.10)
    parser.add_argument("--min_component", type=int, default=8)
    parser.add_argument("--keep_mass_frac", type=float, default=0.92)
    parser.add_argument("--top_centers", type=int, default=160)
    parser.add_argument("--type_margin", type=float, default=0.035)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    results = [run_sample(args.root, sample, args) for sample in args.samples]
    text = json.dumps(results, indent=2, ensure_ascii=True)
    print(text)
    if args.out_json is not None:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

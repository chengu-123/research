"""Filter Stage B move masks with adjacent-state component consistency.

This script operates on an already-produced Stage B output directory. It does
not rerun TRELLIS, SCAR, BMCSA, Stage C, or bootstrap.

Default behavior is non-destructive:
  - source: O_move_per_state_raw.npy if present, else O_move_per_state.npy
  - writes: O_move_per_state_temporal.npy, O_move_temporal_removed.npy,
            move_temporal_filter_report.json
  - if P_move_evidence_per_state.npy exists, writes the filtered soft field
    as P_move_evidence_per_state_temporal.npy

Use --overwrite to also replace O_move_per_state.npy and
P_move_evidence_per_state.npy with the filtered versions. The original arrays
are preserved as *_raw.npy when those files do not already exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MINE_DIR = os.path.dirname(_THIS_DIR)
if _MINE_DIR not in sys.path:
    sys.path.insert(0, _MINE_DIR)

from pipelines.utils.voxel_io import save_voxel_grid
from pipelines.utils.voxel_viz import save_voxel_stack_html


def _dilate_mask_3d(mask: np.ndarray, radius: int) -> np.ndarray:
    """Chebyshev dilation for a 3D boolean mask."""
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 3:
        raise ValueError(f"mask must be 3D; got shape {mask_bool.shape}")
    r = int(radius)
    if r < 0:
        raise ValueError(f"radius must be >= 0; got {radius}")
    if r == 0:
        return mask_bool.copy()
    padded = np.pad(mask_bool, ((r, r), (r, r), (r, r)), mode="constant")
    D, H, W = mask_bool.shape
    out = np.zeros_like(mask_bool, dtype=bool)
    for dx in range(2 * r + 1):
        for dy in range(2 * r + 1):
            for dz in range(2 * r + 1):
                out |= padded[dx:dx + D, dy:dy + H, dz:dz + W]
    return out


def _connected_components_26(mask: np.ndarray) -> List[np.ndarray]:
    """Return 26-connected components as int16 coordinate arrays."""
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 3:
        raise ValueError(f"mask must be 3D; got shape {mask_bool.shape}")
    D, H, W = mask_bool.shape
    visited = np.zeros_like(mask_bool, dtype=bool)
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if dx != 0 or dy != 0 or dz != 0
    ]

    components: List[np.ndarray] = []
    seeds = np.argwhere(mask_bool)
    for seed in seeds:
        sx, sy, sz = int(seed[0]), int(seed[1]), int(seed[2])
        if visited[sx, sy, sz]:
            continue
        queue: deque[Tuple[int, int, int]] = deque([(sx, sy, sz)])
        visited[sx, sy, sz] = True
        comp: List[Tuple[int, int, int]] = []
        while queue:
            x, y, z = queue.pop()
            comp.append((x, y, z))
            for dx, dy, dz in offsets:
                nx, ny, nz = x + dx, y + dy, z + dz
                if nx < 0 or nx >= D or ny < 0 or ny >= H or nz < 0 or nz >= W:
                    continue
                if mask_bool[nx, ny, nz] and not visited[nx, ny, nz]:
                    visited[nx, ny, nz] = True
                    queue.append((nx, ny, nz))
        components.append(np.asarray(comp, dtype=np.int16))
    return components


def filter_move_components_temporal(
    O_move_per_state: np.ndarray,
    dilation_radius: int,
    min_adjacent_overlap: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Keep components that overlap a dilated adjacent-state move mask."""
    move = np.asarray(O_move_per_state, dtype=bool)
    if move.ndim != 4:
        raise ValueError(f"O_move_per_state must be 4D; got shape {move.shape}")
    if int(min_adjacent_overlap) < 1:
        raise ValueError(
            f"min_adjacent_overlap must be >= 1; got {min_adjacent_overlap}"
        )

    K = int(move.shape[0])
    filtered = np.zeros_like(move, dtype=bool)
    dilated = [_dilate_mask_3d(move[k], radius=dilation_radius) for k in range(K)]
    report: Dict[str, Any] = {
        "enabled": True,
        "source_shape": list(move.shape),
        "rule": (
            "Keep each 26-connected component if it overlaps any dilated "
            "adjacent-state move mask."
        ),
        "dilation_radius": int(dilation_radius),
        "min_adjacent_overlap": int(min_adjacent_overlap),
        "states": [],
    }

    for k in range(K):
        comps = _connected_components_26(move[k])
        state_report: Dict[str, Any] = {
            "state": int(k),
            "input_voxels": int(move[k].sum()),
            "component_count": int(len(comps)),
            "kept_component_count": 0,
            "removed_component_count": 0,
            "kept_voxels": 0,
            "removed_voxels": 0,
            "removed_components": [],
        }
        has_left = k > 0
        has_right = k < K - 1
        for comp_idx, comp in enumerate(comps):
            if comp.size == 0:
                continue
            xs = comp[:, 0]
            ys = comp[:, 1]
            zs = comp[:, 2]
            left_overlap = int(dilated[k - 1][xs, ys, zs].sum()) if has_left else 0
            right_overlap = int(dilated[k + 1][xs, ys, zs].sum()) if has_right else 0
            keep = (
                max(left_overlap, right_overlap) >= int(min_adjacent_overlap)
            )
            if not has_left and not has_right:
                keep = True

            n_voxels = int(comp.shape[0])
            if keep:
                filtered[k, xs, ys, zs] = True
                state_report["kept_component_count"] += 1
                state_report["kept_voxels"] += n_voxels
            else:
                state_report["removed_component_count"] += 1
                state_report["removed_voxels"] += n_voxels
                state_report["removed_components"].append({
                    "component_index": int(comp_idx),
                    "voxel_count": n_voxels,
                    "left_overlap": left_overlap,
                    "right_overlap": right_overlap,
                })
        report["states"].append(state_report)

    removed = move & ~filtered
    report["input_voxels"] = int(move.sum())
    report["kept_voxels"] = int(filtered.sum())
    report["removed_voxels"] = int(removed.sum())
    return filtered.astype(np.uint8), removed.astype(np.uint8), report


def _resolve_source(stage_b_dir: str, source: str | None) -> str:
    if source is not None:
        return os.path.abspath(source)
    raw_path = os.path.join(stage_b_dir, "O_move_per_state_raw.npy")
    if os.path.isfile(raw_path):
        return raw_path
    return os.path.join(stage_b_dir, "O_move_per_state.npy")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Apply adjacent-state 26-component filtering to existing Stage B "
            "move masks."
        )
    )
    p.add_argument(
        "stage_b_dir",
        help="Directory containing Stage B outputs such as O_move_per_state.npy.",
    )
    p.add_argument(
        "--source",
        default=None,
        help=(
            "Optional source move mask npy. Default: O_move_per_state_raw.npy "
            "if present, else O_move_per_state.npy."
        ),
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Default: same as stage_b_dir.",
    )
    p.add_argument(
        "--radius",
        type=int,
        default=1,
        help="Chebyshev dilation radius for adjacent-state overlap. Default 1.",
    )
    p.add_argument(
        "--min_adjacent_overlap",
        type=int,
        default=1,
        help="Minimum overlap voxels with either adjacent dilated mask. Default 1.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Also replace O_move_per_state.npy and P_move_evidence_per_state.npy "
            "with filtered versions. Raw copies are preserved when absent."
        ),
    )
    p.add_argument(
        "--no_viz",
        action="store_true",
        help="Do not write HTML visualizations.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stage_b_dir = os.path.abspath(args.stage_b_dir)
    out_dir = os.path.abspath(args.output_dir or stage_b_dir)
    os.makedirs(out_dir, exist_ok=True)

    source_path = _resolve_source(stage_b_dir, args.source)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"source move mask not found: {source_path}")

    move_raw = np.load(source_path)
    move_filtered, move_removed, report = filter_move_components_temporal(
        move_raw,
        dilation_radius=int(args.radius),
        min_adjacent_overlap=int(args.min_adjacent_overlap),
    )
    report["stage_b_dir"] = stage_b_dir
    report["source_path"] = source_path
    report["output_dir"] = out_dir
    report["overwrite"] = bool(args.overwrite)

    raw_out = os.path.join(out_dir, "O_move_per_state_raw.npy")
    if not os.path.isfile(raw_out):
        save_voxel_grid(raw_out, np.asarray(move_raw, dtype=np.uint8))
    save_voxel_grid(
        os.path.join(out_dir, "O_move_per_state_temporal.npy"),
        move_filtered,
    )
    save_voxel_grid(
        os.path.join(out_dir, "O_move_temporal_removed.npy"),
        move_removed,
    )

    p_move_path = os.path.join(stage_b_dir, "P_move_evidence_per_state.npy")
    if os.path.isfile(p_move_path):
        p_move_raw = np.load(p_move_path).astype(np.float32)
        if p_move_raw.shape != move_filtered.shape:
            raise ValueError(
                f"P_move shape {p_move_raw.shape} does not match move shape "
                f"{move_filtered.shape}"
            )
        p_move_filtered = p_move_raw * move_filtered.astype(np.float32)
        p_raw_out = os.path.join(out_dir, "P_move_evidence_per_state_raw.npy")
        if not os.path.isfile(p_raw_out):
            save_voxel_grid(p_raw_out, p_move_raw.astype(np.float32))
        save_voxel_grid(
            os.path.join(out_dir, "P_move_evidence_per_state_temporal.npy"),
            p_move_filtered.astype(np.float32),
        )
        save_voxel_grid(
            os.path.join(out_dir, "move_confidence_temporal.npy"),
            p_move_filtered.astype(np.float32),
        )
        report["p_move_source_path"] = p_move_path
    else:
        report["p_move_source_path"] = None

    if args.overwrite:
        save_voxel_grid(os.path.join(out_dir, "O_move_per_state.npy"), move_filtered)
        if os.path.isfile(p_move_path):
            save_voxel_grid(
                os.path.join(out_dir, "P_move_evidence_per_state.npy"),
                p_move_filtered.astype(np.float32),
            )
            save_voxel_grid(
                os.path.join(out_dir, "move_confidence.npy"),
                p_move_filtered.astype(np.float32),
            )

    report_path = os.path.join(out_dir, "move_temporal_filter_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if not args.no_viz:
        viz_dir = os.path.join(out_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)
        save_voxel_stack_html(
            np.asarray(move_raw, dtype=np.float32),
            os.path.join(viz_dir, "O_move_per_state_raw.html"),
            title="O_move_per_state raw source",
        )
        save_voxel_stack_html(
            move_filtered.astype(np.float32),
            os.path.join(viz_dir, "O_move_per_state_temporal.html"),
            title=(
                f"O_move_per_state temporal filtered "
                f"(radius={int(args.radius)})"
            ),
        )
        save_voxel_stack_html(
            move_removed.astype(np.float32),
            os.path.join(viz_dir, "O_move_temporal_removed.html"),
            title=(
                f"O_move_temporal_removed "
                f"(radius={int(args.radius)})"
            ),
        )

    print("[filter_stageb_move_temporal] done")
    print(f"  source      : {source_path}")
    print(f"  output_dir  : {out_dir}")
    print(f"  input voxels: {report['input_voxels']}")
    print(f"  kept voxels : {report['kept_voxels']}")
    print(f"  removed     : {report['removed_voxels']}")
    print(f"  report      : {report_path}")


if __name__ == "__main__":
    main()

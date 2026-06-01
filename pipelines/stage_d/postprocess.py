"""Stage D segmentation post-processing.

Connectivity purge for the move (articulated) ownership. For a single-DoF part
the moving region is ONE connected component, so any detached move component --
a leaked static structure (e.g. a rod/tray) or isolated specks -- is relabeled
to base. This is a pure post-process on the per-voxel move mask: it does not
touch the joint or the geometry, only the move/base ownership consumed by the
downstream mesh split.

Diagnosis backing this (run 5891, jclamp it700): the leaked rod was a coherent
51-voxel component 9 voxels from the door, initialized base (alpha_m=-1.39) but
dragged to move by the 2D ownership loss's ray-ambiguity. It forms its own
component separate from the 513-voxel door, so a connectivity purge removes it
cleanly. Mirrors FreeArt3D's ndimage.label purge (pipelines/sds.py:638-662).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _connected_components_6(
    coords: np.ndarray,
    sel_mask: np.ndarray,
) -> List[List[int]]:
    """6-connectivity components over the voxels selected by ``sel_mask``.

    Parameters
    ----------
    coords : np.ndarray
        [N, 3] integer voxel-grid coordinates.
    sel_mask : np.ndarray
        [N] bool selection over ``coords``.

    Returns
    -------
    list of index-lists, largest component first.
    """
    idx: Dict[Tuple[int, int, int], int] = {}
    for i in range(coords.shape[0]):
        idx[(int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2]))] = i
    sel = np.where(sel_mask)[0].tolist()
    parent = {i: i for i in sel}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    nbrs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for i in sel:
        c = coords[i]
        for d in nbrs:
            j = idx.get((int(c[0] + d[0]), int(c[1] + d[1]), int(c[2] + d[2])))
            if j is not None and j in parent:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    comp: Dict[int, List[int]] = {}
    for i in sel:
        comp.setdefault(find(i), []).append(i)
    return sorted(comp.values(), key=len, reverse=True)


def purge_move_components(
    coords: np.ndarray,
    move_mask: np.ndarray,
    merge_dist: float = 2.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Keep the largest move component plus any component within ``merge_dist``
    voxels of it; relabel every other (detached) move voxel to base.

    Parameters
    ----------
    coords : np.ndarray
        [N, 3] integer voxel-grid coordinates (the Stage D ``U_object``).
    move_mask : np.ndarray
        [N] bool per-voxel move (articulated) ownership.
    merge_dist : float
        Voxel distance used to bridge thin necks so a legitimately split part is
        not over-purged. The default 2.0 keeps only voxels within 2 of the
        dominant move component.

    Returns
    -------
    cleaned : np.ndarray
        [N] bool purged move mask.
    info : dict
        ``sizes`` (all component sizes, largest first), ``kept`` (kept component
        ids), ``purged`` (number of move voxels relabeled to base).
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be [N, 3], got {tuple(coords.shape)}")
    move_mask = np.asarray(move_mask, dtype=bool)
    if move_mask.shape != (coords.shape[0],):
        raise ValueError(
            f"move_mask must be [N]={coords.shape[0]}, got {tuple(move_mask.shape)}"
        )
    comps = _connected_components_6(coords, move_mask)
    if not comps:
        return move_mask.copy(), {"sizes": [], "kept": [], "purged": 0}

    main = comps[0]
    main_xyz = coords[main].astype(np.float64)
    kept = set(main)
    kept_ids: List[int] = [0]
    for ci, c in enumerate(comps[1:], start=1):
        cxyz = coords[c].astype(np.float64)
        dmin = float(np.min(np.linalg.norm(cxyz[:, None, :] - main_xyz[None, :, :], axis=2)))
        if dmin <= float(merge_dist):
            kept.update(c)
            kept_ids.append(ci)

    cleaned = np.zeros_like(move_mask)
    cleaned[list(kept)] = True
    info: Dict[str, object] = {
        "sizes": [len(c) for c in comps],
        "kept": kept_ids,
        "purged": int(move_mask.sum() - cleaned.sum()),
    }
    return cleaned, info


__all__ = ["purge_move_components"]

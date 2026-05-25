"""Empirical comparison: drawer bottom Y in v3.3.2 (physical separation OK)
vs other versions. Goal: extract numerical mechanism that v3.3.4 must reproduce.
"""
import os
import sys
import numpy as np


def slice_object_extent(O_stack):
    """For each K state, find object bounding-box extent along Y axis."""
    K = O_stack.shape[0]
    print(f"  state | Y_min | Y_max | n_voxel | bottom-row count")
    print(f"  ------+-------+-------+---------+----------------")
    extents = []
    for k in range(K):
        occ = O_stack[k] > 0
        if not occ.any():
            extents.append((0, 0, 0, 0))
            print(f"   s{k}   |  -    |  -    |    0   |     -")
            continue
        ys = np.where(occ.any(axis=(0, 2)))[0]
        y_min, y_max = int(ys.min()), int(ys.max())
        n_voxel = int(occ.sum())
        # Bottom row: count voxels at the absolute lowest Y level present
        bottom_row = int(occ[:, y_min, :].sum())
        # Also check what's at y_min + 1
        next_row = int(occ[:, min(y_min + 1, 63), :].sum()) if y_min + 1 < 64 else 0
        extents.append((y_min, y_max, n_voxel, bottom_row))
        print(f"   s{k}   |  {y_min:2d}   |  {y_max:2d}   |  {n_voxel:5d}  |   {bottom_row:5d}  (next={next_row})")
    return extents


def base_cabinet_floor_y(O_stack, k_vote_thresh=5):
    """K-vote derived base mask; find cabinet floor Y as min Y of base voxels."""
    K = O_stack.shape[0]
    votes = (O_stack > 0).sum(axis=0)
    base = votes >= k_vote_thresh
    if not base.any():
        return None, 0
    ys = np.where(base.any(axis=(0, 2)))[0]
    return int(ys.min()), int(base.sum())


def analyze(run_dir, name):
    p = os.path.join(run_dir, "O_stack.npy")
    if not os.path.isfile(p):
        print(f"[{name}] no O_stack.npy at {p}")
        return None
    O = np.load(p)
    if O.dtype == np.uint8:
        O = O.astype(bool)
    print(f"\n===== {name} =====  shape={O.shape}, dtype={O.dtype}, n_occupied={int((O>0).sum())}")
    print("Per-state Y extent:")
    extents = slice_object_extent(O)
    floor_y, base_n = base_cabinet_floor_y(O, k_vote_thresh=5)
    print(f"\n  Base (K-vote >=5/6) bottom Y = {floor_y}, base voxel count = {base_n}")
    # The diagnostic: in each state, is the OBJECT bottom voxel layer at the SAME Y as base bottom (collision)
    # or 1 layer above (no-collision)?
    print(f"\n  Drawer-vs-cabinet floor (object Y_min - base Y_min):")
    for k, (y_min, _, _, _) in enumerate(extents):
        delta = y_min - floor_y if floor_y is not None else None
        verdict = ("Y_min < floor (unusual)" if delta < 0
                   else "MERGE (drawer on floor)" if delta == 0
                   else "DRAWER ABOVE FLOOR (correct physics)" if delta == 1
                   else f"+{delta} layers above floor")
        print(f"   s{k}: object Y_min={y_min}, floor Y_min={floor_y}, delta={delta} -> {verdict}")
    return extents, floor_y


ROOT = r"C:\Users\管晨皓\Desktop\temp\standard\mine\outputs\30857"
analyze(os.path.join(ROOT, "stageb_3.3.2"), "v3.3.2 (physical separation visible per user)")
# v1 and v2 may not have O_stack.npy at top level — try anyway, will report missing
analyze(os.path.join(ROOT, "stageb_v1"), "v3.3.4 v1 (Plan C+ off)")
analyze(os.path.join(ROOT, "stageb_v2"), "v3.3.4 v2 (Plan C+ on)")

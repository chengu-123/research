"""Inspect v3.3.2 Pass-1 vs Pass-2 K-state geometry for drawer-Y separation."""
import os
import numpy as np

ROOT = r"C:\Users\管晨皓\Desktop\temp\standard\mine\outputs\30857\stageb_3.3.2"


def per_state_y(O, name):
    print(f"\n=== {name} ===  shape={O.shape}")
    print("  state | n_voxel | Y_min | Y_max | bottom-row")
    print("  ------+---------+-------+-------+----------")
    for k in range(O.shape[0]):
        occ = (O[k] > 0)
        if not occ.any():
            print(f"   s{k}   |    0    |  -    |  -    |    -")
            continue
        ys = np.where(occ.any(axis=(0, 2)))[0]
        y_min, y_max = int(ys.min()), int(ys.max())
        bottom = int(occ[:, y_min, :].sum())
        next_bot = int(occ[:, min(y_min + 1, 63), :].sum())
        print(f"   s{k}   |  {int(occ.sum()):5d}  |  {y_min:2d}   |  {y_max:2d}   |  {bottom:5d} (next={next_bot})")


# Pass 1 (per-state move evidence; per-state different)
O1 = np.load(os.path.join(ROOT, "O_stack_pass1.npy"))
if O1.dtype == np.uint8: O1 = O1.astype(bool)
per_state_y(O1, "Pass-1 binary (per-state, before BMCSA)")

# Pass 2 (BMCSA K-share output)
O2 = np.load(os.path.join(ROOT, "O_stack.npy"))
if O2.dtype == np.uint8: O2 = O2.astype(bool)
per_state_y(O2, "Pass-2 binary (after BMCSA, K-share dominant)")

# Pass-1 SOFT (might reveal sub-threshold drawer position)
S1 = np.load(os.path.join(ROOT, "O_stack_pass1_soft.npy")).astype(np.float32)
print(f"\n=== Pass-1 SOFT inspection at drawer trajectory levels ===")
print(f"  Total soft mass per state, per Y slice (Y=24 floor neighborhood):")
# Look at Y in 23..27 (near floor) for each state
print("  state |  Y=23  |  Y=24  |  Y=25  |  Y=26  |  Y=27")
for k in range(S1.shape[0]):
    rows = [float((S1[k, :, y, :] > 0.5).sum()) for y in range(23, 28)]
    print(f"   s{k}   |  {rows[0]:4.0f}  |  {rows[1]:4.0f}  |  {rows[2]:4.0f}  |  {rows[3]:4.0f}  |  {rows[4]:4.0f}")

# Per-step Pass-2 evolution (where K converges)
P2_DIR = os.path.join(ROOT, "pass2_per_step")
if os.path.isdir(P2_DIR):
    print(f"\n=== Pass-2 trajectory: K-state Y_min per step ===")
    print("  step | s0-Y | s1-Y | s2-Y | s3-Y | s4-Y | s5-Y | converged?")
    # Re-decode would need TRELLIS — but per_step_stats.json should have voxel_counts
    import json
    with open(os.path.join(P2_DIR, "per_step_stats.json")) as f:
        stats = json.load(f)
    for s in stats[:6] + stats[-3:]:
        cnts = s["voxel_counts"]
        unique = len(set(cnts))
        print(f"   {s['step']:2d}  | counts={cnts} | {unique} unique values")

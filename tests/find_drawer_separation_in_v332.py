"""Where exactly in v3.3.2 outputs does drawer-bottom-Y differ from cabinet-floor-Y?"""
import os
import json
import numpy as np

ROOT = r"C:\Users\管晨皓\Desktop\temp\standard\mine\outputs\30857\stageb_3.3.2"


def y_profile(arr_3d, name, threshold=0.5):
    """Per-Y voxel count for a 3D array (soft or binary)."""
    binary = (arr_3d > threshold).astype(int)
    if binary.sum() == 0:
        print(f"  {name}: EMPTY")
        return None
    yc = binary.sum(axis=(0, 2))
    ys_occ = np.where(yc > 0)[0]
    y_min, y_max = int(ys_occ.min()), int(ys_occ.max())
    print(f"  {name}: Y_min={y_min}, Y_max={y_max}, n_voxel={int(binary.sum())}")
    # show first 6 occupied Y layers and their counts
    head = ys_occ[:6]
    print(f"    bottom layers: " + ", ".join(f"Y={y}({yc[y]})" for y in head))
    return y_min


print("=== Per-state P_guide bottom Y (Pass-2 INPUT, before BMCSA) ===")
print("(P_guide = augmented intersection from Pass-1 soft, encoded -> Pass-2 starting point)")
for k in range(6):
    p = os.path.join(ROOT, "viz", "guide", f"state{k}_P_guide.npy")
    if os.path.isfile(p):
        arr = np.load(p).astype(np.float32)
        y_profile(arr, f"state{k}_P_guide")

print("\n=== Pass-1 SOFT per-state bottom-Y ===")
S1 = np.load(os.path.join(ROOT, "O_stack_pass1_soft.npy")).astype(np.float32)
print(f"shape={S1.shape}")
for k in range(6):
    y_profile(S1[k], f"Pass-1 s{k} soft")

print("\n=== Pass-1 BINARY per-state bottom-Y ===")
B1 = np.load(os.path.join(ROOT, "O_stack_pass1.npy")).astype(bool)
for k in range(6):
    y_profile(B1[k].astype(np.float32), f"Pass-1 s{k} bin", threshold=0.5)

print("\n=== Pass-2 BINARY per-state bottom-Y ===")
B2 = np.load(os.path.join(ROOT, "O_stack.npy")).astype(bool)
for k in range(6):
    y_profile(B2[k].astype(np.float32), f"Pass-2 s{k} bin", threshold=0.5)

# Try p_base_preview and p_move_preview
print("\n=== p_base / p_move preview ===")
pb = np.load(os.path.join(ROOT, "p_base_preview.npy")).astype(np.float32)
pm = np.load(os.path.join(ROOT, "p_move_preview.npy")).astype(np.float32)
y_profile(pb, "p_base_preview (3D)", threshold=0.5)
y_profile(pm, "p_move_preview (3D)", threshold=0.5)

# Compute: cabinet base Y vs drawer move Y
print("\n=== DRAWER-Y vs CABINET-Y from p_base/p_move ===")
pb_bin = pb > 0.5
pm_bin = pm > 0.5
if pb_bin.any():
    base_ymin = int(np.where(pb_bin.any(axis=(0,2)))[0].min())
    print(f"  cabinet base bottom-Y (from p_base > 0.5): {base_ymin}")
if pm_bin.any():
    move_ymin = int(np.where(pm_bin.any(axis=(0,2)))[0].min())
    print(f"  drawer move bottom-Y (from p_move > 0.5): {move_ymin}")
    if pb_bin.any():
        delta = move_ymin - base_ymin
        verdict = ("MERGE" if delta == 0 else
                   f"DRAWER {delta} LAYERS ABOVE CABINET (correct)" if delta > 0 else
                   f"drawer BELOW cabinet (weird), delta={delta}")
        print(f"  delta = {delta} -> {verdict}")

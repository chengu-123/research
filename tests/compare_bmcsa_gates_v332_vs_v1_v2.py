"""Extract BMCSA gate numerical signatures from v3.3.2 vs v1 vs v2.
Goal: identify what makes v3.3.2 produce drawer-above-cabinet (cabinet floor
z=36, drawer floor z=37) while v1/v2 don't.

Per user: stop inspecting raw voxel layout. Look at BMCSA gates, dynamic-M
scalars, configuration JSONs, signal-level statistics.
"""
import os
import json
import numpy as np

ROOT = r"C:\Users\管晨皓\Desktop\temp\standard\mine\outputs\30857"
V332 = os.path.join(ROOT, "stageb_3.3.2", "viz", "bmcsa")
V1 = os.path.join(ROOT, "stageb_v1", "viz", "bmcsa")
V2 = os.path.join(ROOT, "stageb_v2", "viz", "bmcsa")


def load_if_exists(p):
    return np.load(p).astype(np.float32) if os.path.isfile(p) else None


def stats(arr, name):
    if arr is None:
        print(f"  {name:<32} MISSING")
        return
    print(f"  {name:<32} shape={str(arr.shape):<18} "
          f"min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.4f} "
          f">0.5={int((arr>0.5).sum()):>6}")


print("=" * 80)
print("INPUT GATES TO BMCSA (static, computed once per run)")
print("=" * 80)
for label, root in [("v3.3.2", V332), ("v1 (v3.3.4 occ)", V1), ("v2 (Plan C+)", V2)]:
    print(f"\n--- {label} ---")
    stats(load_if_exists(os.path.join(root, "M_base_64.npy")), "M_base_64")
    stats(load_if_exists(os.path.join(root, "M_attn_64.npy")), "M_attn_64")
    stats(load_if_exists(os.path.join(root, "M_motion_corridor_64.npy")), "M_motion_corridor_64")
    stats(load_if_exists(os.path.join(root, "M_motion_enhanced_64.npy")), "M_motion_enhanced_64 (Plan C+)")

print()
print("=" * 80)
print("DYNAMIC-M (S=12, B=24) — what BMCSA actually applied per step per block")
print("=" * 80)
for label, root in [("v3.3.2", V332), ("v1 (v3.3.4 occ)", V1), ("v2 (Plan C+)", V2)]:
    M = load_if_exists(os.path.join(root, "M_dynamic_scalar_step_block.npy"))
    if M is None:
        print(f"\n--- {label} ---  MISSING")
        continue
    print(f"\n--- {label} ---  shape={M.shape}")
    print(f"  per-step mean (across 24 blocks):")
    for s in range(M.shape[0]):
        row = M[s]
        print(f"    step {s:2d}: mean={row.mean():.4f}, std={row.std():.4f}, "
              f"min={row.min():.4f}, max={row.max():.4f}")
    print(f"  Overall: mean={M.mean():.4f}, std={M.std():.4f}")


print()
print("=" * 80)
print("SDEDIT CONFIG (what gate composition was actually applied)")
print("=" * 80)
for label, base in [("v3.3.2", os.path.join(ROOT, "stageb_3.3.2")),
                    ("v1 (v3.3.4 occ)", os.path.join(ROOT, "stageb_v1")),
                    ("v2 (Plan C+)", os.path.join(ROOT, "stageb_v2"))]:
    rp = os.path.join(base, "sdedit_report.json")
    if not os.path.isfile(rp):
        print(f"\n--- {label} --- NO sdedit_report.json")
        continue
    print(f"\n--- {label} ---")
    with open(rp) as f:
        rpt = json.load(f)
    for k in ["mode", "t_star", "pass2_steps", "M_compute_mode",
              "M_dynamic_signal", "tau_M_dynamic", "kappa_M_dynamic",
              "var_percentile", "var_eta", "kv_floor",
              "attn_m_enabled", "attn_m_apply_at", "use_motion_corridor",
              "motion_corridor_source"]:
        if k in rpt:
            print(f"  {k:<24} = {rpt[k]}")

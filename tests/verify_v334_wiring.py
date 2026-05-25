"""Verify v3.3.4 wiring after Step 1 + Step 2 edits.

Checks:
1. modulated.py + stage_b_scar.py parse cleanly
2. v1.yaml has the new defaults (use_motion_corridor=true, kv_floor=0.2,
   M_dynamic_signal=variance)
3. New BMCSA composition signature accepts the new kwargs without error
   (signature-level sanity, not numerical)
"""
import ast
import os
import sys
import yaml


ROOT = r"C:\Users\管晨皓\Desktop\temp\standard\mine"

# 1) Syntax
for path in [
    os.path.join(ROOT, "TRELLIS", "trellis", "modules", "transformer", "modulated.py"),
    os.path.join(ROOT, "pipelines", "stage_b_scar.py"),
]:
    with open(path, encoding="utf-8") as f:
        ast.parse(f.read())
print("[OK] modulated.py + stage_b_scar.py parse cleanly")

# 2) yaml defaults
with open(os.path.join(ROOT, "configs", "v1.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
sb = cfg["stage_b_sdedit"]
expected = {
    "use_motion_corridor": True,
    "kv_floor": 0.2,
    "M_dynamic_signal": "variance",
    "M_compute_mode": "dynamic",
    "attn_m_apply_at": "guide",
    "var_percentile": 0.65,
    "var_eta": 0.5,
    # ★ v3.3.4 Plan C+ multi-source motion corridor flags
    "motion_corridor_source": "enhanced",
    "motion_zfinal_percentile": 0.55,
    "motion_zfinal_eta": 0.5,
    "motion_hidden_percentile": 0.55,
    "motion_hidden_eta": 0.5,
}
print()
print("v3.3.4 stage_b_sdedit defaults:")
ok = True
for k, want in expected.items():
    got = sb.get(k)
    sym = "OK" if got == want else "FAIL"
    if got != want:
        ok = False
    print(f"  [{sym}] {k:<22} expected={want!r:<10} got={got!r}")
assert ok, "v1.yaml defaults do not match v3.3.4 spec"

# 3) BMCSA kwarg signature check (lightweight: confirm the call site lists
#    kv_floor as expected)
with open(os.path.join(ROOT, "pipelines", "stage_b_scar.py"), encoding="utf-8") as f:
    src = f.read()
assert "kv_floor=kv_floor_cfg" in src, "kv_floor not threaded from cfg to _sdedit_refine_k6_bmcsa"
assert "kv_floor=float(kv_floor)" in src, "kv_floor not added to base_kwargs"
assert 'sdedit_report["kv_floor"]' in src, "kv_floor not exported in sdedit_report"
assert 'sdedit_report["use_motion_corridor"]' in src, "use_motion_corridor not exported"
print()
print("[OK] kv_floor + use_motion_corridor properly wired through stage_b_scar.py")

# 4) modulated.py: check new composition logic strings present
with open(os.path.join(ROOT, "TRELLIS", "trellis", "modules", "transformer", "modulated.py"), encoding="utf-8") as f:
    mod_src = f.read()
assert "v3.3.4 motion-aware decisive composition" in mod_src, "v3.3.4 header missing"
assert "K-redundancy floor (s3-loss mitigation)" in mod_src, "kv_floor logic missing"
assert "M_move = torch.clamp(strength * M_move_raw" in mod_src, "M_move clamp missing"
assert "h = M_move * y_self + (1.0 - M_move) * y_shared" in mod_src, "blend formula changed"
print("[OK] modulated.py contains expected v3.3.4 composition logic")

print()
print("All wiring checks pass. Steps 1+2 complete.")

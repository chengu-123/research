"""Verify v3.3.4 v2 revert: Plan C+ off, v1 baseline active."""
import yaml

cfg = yaml.safe_load(open(
    r"C:\Users\管晨皓\Desktop\temp\standard\mine\configs\v1.yaml",
    encoding="utf-8",
))
sb = cfg["stage_b_sdedit"]
print("Reverted to v3.3.4 v1 (Plan C+ disabled):")
print(f"  motion_corridor_source = {sb['motion_corridor_source']!r}  (occupancy_only = v1 baseline)")
print(f"  use_motion_corridor    = {sb['use_motion_corridor']!r}")
print(f"  M_dynamic_signal       = {sb['M_dynamic_signal']!r}  (F1 fix retained)")
print(f"  kv_floor               = {sb['kv_floor']!r}")
print()
assert sb["motion_corridor_source"] == "occupancy_only", "Plan C+ revert incomplete"
print("[OK] revert verified")

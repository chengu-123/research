"""v3.3.5 default-per-state composition synthetic check.

Predicts M_move at the 6 critical voxel categories under v3.3.5 vs v3.3.4.

Goal: cabinet floor (M_base=1 AND motion=0.9) MUST get M_move > 0.7 in v3.3.5
       (per-state attention -> physical effect can emerge) — failed in v3.3.4 (M_move ~= 0.2).
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def v334_compose(M_static, M_base, M_dyn_motion=0.4, kv_floor=0.2):
    """v3.3.4 v1 composition: M_move = max(motion) * (1 - M_base)."""
    M_motion = max(M_static, M_dyn_motion)
    M_move = M_motion * (1.0 - M_base)
    return min(M_move, 1.0 - kv_floor)


def v335_compose(M_static, M_base, M_dyn_motion=0.4, alpha=0.9, kv_floor=0.2):
    """v3.3.5: M_move = 1 - alpha * M_base * (1 - max(motion))."""
    M_motion = max(M_static, M_dyn_motion)
    P_definite_base = M_base * (1.0 - M_motion)
    M_move = 1.0 - alpha * P_definite_base
    return min(M_move, 1.0 - kv_floor)


def v332_estimate(M_base, M_attn, M_motion, M_dyn_sat=0.9976):
    """v3.3.2 estimated M_eff = M_base * M_attn * (1-M_motion) * M_dyn_saturated.
    Returns M_move (= 1 - M_eff in v3.3.4 semantics for direct comparison)."""
    M_eff = M_base * M_attn * (1.0 - M_motion) * M_dyn_sat
    return 1.0 - M_eff  # convert to "per-state strength"


print(f"{'voxel category':<55} {'v3.3.4 M_move':>14} {'v3.3.5 M_move':>14} {'v3.3.2 est':>11}")
print("-" * 100)

cases = [
    # (name, M_static, M_base, M_attn, M_dyn_motion)
    ("1. far-base (M_base=1, motion=0)",              0.0, 1.0, 0.95, 0.0),
    ("2. base back face (M_base=0.7, M_attn=0.6, motion=0)", 0.0, 0.7, 0.6, 0.0),
    ("3. cabinet floor (M_base=1, motion=0.5)",       0.5, 1.0, 0.9,  0.5),
    ("4. cabinet floor STRONG motion (M_base=1, motion=0.9)", 0.9, 1.0, 0.9, 0.9),
    ("5. motion corridor far from base (M_base=0, motion=0.8)", 0.8, 0.0, 0.5, 0.8),
    ("6. air (M_base=0, no motion)",                  0.0, 0.0, 0.1, 0.0),
    ("7. drawer interior consistent (M_base=0.2, motion=0.3)", 0.3, 0.2, 0.8, 0.2),
]

interpretation = {
    1: "K-share dominant -> base preserved",
    2: "K-share dominant -> back face preserved (v3.3.2 lost this)",
    3: "Mixed -> partial physical detail",
    4: "PER-STATE DOMINANT -> drawer Y can emerge (KEY)",
    5: "Per-state (irrelevant, air around motion)",
    6: "Doesn't matter (air)",
    7: "Mostly per-state -> drawer interior allowed",
}

for idx, (name, m_static, m_base, m_attn, m_dyn_motion) in enumerate(cases, 1):
    v334 = v334_compose(m_static, m_base, m_dyn_motion)
    v335 = v335_compose(m_static, m_base, m_dyn_motion)
    # v3.3.2 estimated:
    v332 = v332_estimate(m_base, m_attn, max(m_static, 0.0))  # ignore dyn motion (cosine sat)
    v332_clipped = min(v332, 0.8)  # apply same kv_floor for fair comparison
    print(f"{name:<55} {v334:>14.3f} {v335:>14.3f} {v332_clipped:>11.3f}")
    print(f"  -> expect: {interpretation[idx]}")

print()
print("KEY ACCEPTANCE for v3.3.5 vs v3.3.4:")
print("  Cat 4 (cabinet floor STRONG motion): v3.3.4=0.0, v3.3.5 must be >=0.7")
print("    (this is where drawer-Y physical detail emerges)")
print("  Cat 1 (far base): v3.3.5 must be <=0.2 (K-share dominant for base completeness)")
print("  Cat 2 (back face): v3.3.5 must be <=0.4 (better than v3.3.2's failure)")

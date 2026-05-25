"""Synthetic sanity check for v3.3.4 motion-aware BMCSA composition.

Verifies that the new composition produces the expected M_move (per-state
preservation strength) across 6 voxel categories that span the real Stage B
sample distribution. Standalone numpy script -- no torch required.

Run:
    python tests/test_v334_composition_synthetic.py
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def v334_compose(M_static, M_base, var_div_log_token, var_div_log_full,
                 var_pct=0.65, var_eta=0.5, kv_floor=0.2):
    """Mirror modulated.py v3.3.4 BMCSA composition for a single token."""
    tau = np.quantile(var_div_log_full, var_pct)
    M_dyn = sigmoid((tau - var_div_log_token) / var_eta)
    M_dyn_motion = float(np.clip(1.0 - M_dyn, 0.0, 1.0))
    M_move_raw = max(M_static, M_dyn_motion)
    M_move_raw = M_move_raw * (1.0 - M_base)
    M_move = float(np.clip(M_move_raw, 0.0, 1.0 - kv_floor))
    return M_move, M_dyn_motion


def main():
    np.random.seed(42)
    # Simulate per-token log-variance distribution; base-like center.
    all_log_var = np.random.normal(loc=-1.5, scale=0.8, size=4096)
    log_var_base = -3.0
    log_var_move = -0.5

    cases = [
        ("1. far-base       (no motion, full base)", 0.0, 1.0, log_var_base),
        ("2. corridor mid   (static motion=0.5)",    0.5, 0.0, log_var_move),
        ("3. corridor end   (static motion=0.2)",    0.2, 0.0, -1.5),
        ("4. hinge axis     (M_base=0.5)",           0.0, 0.5, log_var_base),
        ("5. base back face (M_base=0.7)",           0.0, 0.7, log_var_base),
        ("6. drawer inside  (always-on base-like)",  0.0, 0.3, log_var_base - 1),
    ]

    print(f"{'category':<48} {'M_dyn_mot':>10} {'M_move':>8}   attention")
    print("-" * 100)
    expected_behavior = {
        1: "y_avg",  # far base: K/V averaging
        2: "self",   # corridor mid: per-state for physical detail
        3: "y_avg",  # corridor endpoint: mostly average
        4: "y_avg",  # hinge axis: base floor protects
        5: "y_avg",  # base back face: base floor protects (fixes v3.3.2 bug)
        6: "y_avg",  # drawer interior: low variance, no static motion
    }
    pass_count = 0
    for idx, (name, m_static, m_base, log_v) in enumerate(cases, start=1):
        M_move, M_dyn_mot = v334_compose(
            m_static, m_base, log_v, all_log_var,
        )
        if M_move < 0.1:
            att = "~100% y_avg (K averaged)"
            kind = "y_avg"
        elif M_move < 0.4:
            att = "mostly y_avg (light self)"
            kind = "y_avg"
        elif M_move < 0.7:
            att = "mixed (preserve some detail)"
            kind = "mix"
        else:
            att = "mostly y_self (per-state)"
            kind = "self"
        match = "PASS" if kind == expected_behavior[idx] or (
            expected_behavior[idx] == "self" and kind in ("self", "mix")
        ) else "FAIL"
        if match == "PASS":
            pass_count += 1
        print(f"{name:<48} {M_dyn_mot:>10.3f} {M_move:>8.3f}   {att:<35} [{match}]")
    print()
    print(f"Result: {pass_count}/{len(cases)} cases match expected behavior")
    print()
    print("Key acceptance:")
    print("  - cat 2 (corridor mid):       M_move should be >= 0.4 -> physical detail preserved")
    print("  - cat 5 (base back face):     M_move should be ~ 0    -> v3.3.2 bug fixed")
    print("  - cat 1, 4, 6 (base regions): M_move should be ~ 0    -> base completeness")


if __name__ == "__main__":
    main()

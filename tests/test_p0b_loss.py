"""Offline unit test for loss_motion_ownership_3d (P0b core ownership loss).

Standalone script (repo convention: tests/ are standalone, not pytest). Run:
    PYTHONPATH=. python tests/test_p0b_loss.py

Synthetic scene (no rendering): a static cabinet + a static rod (both foreground in
ALL states) + a door that translates with state (canonical c=2). Feeds three
(move_sil, base_sil) hypotheses and checks the loss behaves:

  A (correct: rod->base, door->move): base covers STATIC, move covers emerged door
     -> L_base_static ~ 0, L_emerge low.
  B (leak: rod->move): base MISSES the rod -> L_base_static >> A  (the term that fixes
     the rod leak; its gradient pushes the rod to base).
  C (move misses emerged door): move is only a shifted rod -> L_emerge >> A.
"""
import torch

from pipelines.stage_d.losses import loss_motion_ownership_3d

S, V, H, W = 6, 2, 48, 48
C = 2  # canonical index (a MIDDLE state, per R3)


def rect(mask, r0, r1, c0, c1):
    mask[r0:r1, max(0, c0):max(0, c1)] = 1.0


def door_cols(k):
    base0 = 20 + (k - C) * 4
    return base0, base0 + 8


def build_obs():
    obs = torch.zeros(S, V, 1, H, W)
    for k in range(S):
        for v in range(V):
            img = torch.zeros(H, W)
            rect(img, 10, 40, 4, 16)          # cabinet (static)
            rect(img, 20, 24, 40, 44)         # rod (static)
            c0, c1 = door_cols(k)
            rect(img, 18, 30, c0, c1)         # door (moves with k)
            obs[k, v, 0] = img
    return obs


def door_only(transformed=True):
    m = torch.zeros(S, V, 1, H, W)
    for k in range(S):
        c0, c1 = door_cols(k if transformed else C)
        for v in range(V):
            rect(m[k, v, 0], 18, 30, c0, c1)
    return m


def rod_shifted():
    m = torch.zeros(S, V, 1, H, W)
    for k in range(S):
        off = (k - C) * 4
        for v in range(V):
            rect(m[k, v, 0], 20, 24, 40 + off, 44 + off)
    return m


def base_with_rod():
    b = torch.zeros(V, 1, H, W)
    for v in range(V):
        rect(b[v, 0], 10, 40, 4, 16)          # cabinet
        rect(b[v, 0], 20, 24, 40, 44)         # rod
    return b


def base_no_rod():
    b = torch.zeros(V, 1, H, W)
    for v in range(V):
        rect(b[v, 0], 10, 40, 4, 16)          # cabinet only
    return b


def run(tag, move, base, obs):
    move = move.clone().requires_grad_(True)
    base = base.clone().requires_grad_(True)
    Le, Lbs, Lv = loss_motion_ownership_3d(move, base, obs, canonical_idx=C,
                                           erode_px=1, dilate_px=1)
    (Le + Lbs + Lv).backward()
    print("  [%s] L_emerge=%.4f  L_base_static=%.4f  L_vacate=%.4f  "
          "grad(base) nonzero=%s grad(move) nonzero=%s"
          % (tag, Le.item(), Lbs.item(), Lv.item(),
             bool(base.grad.abs().sum() > 0), bool(move.grad.abs().sum() > 0)))
    return Le.item(), Lbs.item(), Lv.item()


def main():
    obs = build_obs()
    print("== P0b loss_motion_ownership_3d unit test ==")
    LeA, LbsA, LvA = run("A correct (rod->base, door->move)", door_only(), base_with_rod(), obs)
    LeB, LbsB, LvB = run("B leak   (rod->move, base misses rod)",
                         door_only() + rod_shifted(), base_no_rod(), obs)
    LeC, LbsC, LvC = run("C move misses emerged (move = shifted rod only)",
                         rod_shifted(), base_with_rod(), obs)

    print("\n== assertions ==")
    ok = True
    a1 = LbsA < 0.02
    print("  A: L_base_static ~ 0 (base covers STATIC):           %s (%.4f)" % (a1, LbsA)); ok &= a1
    a2 = LbsB > 10 * max(LbsA, 1e-4)
    print("  B >> A: rod-as-move penalized by L_base_static:      %s (B=%.4f A=%.4f)" % (a2, LbsB, LbsA)); ok &= a2
    a3 = LeA < 0.15
    print("  A: L_emerge low (move covers emerged door):          %s (%.4f)" % (a3, LeA)); ok &= a3
    a4 = LeC > 3 * max(LeA, 1e-4)
    print("  C >> A: move-misses-emerged penalized by L_emerge:   %s (C=%.4f A=%.4f)" % (a4, LeC, LeA)); ok &= a4
    print("\nRESULT:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

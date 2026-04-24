"""Unit tests for the 2026-04-22 bug fix in pipelines.sajo.anchors.joint_free_split.

Verifies:
1. Legacy formula reproduced exactly under ``mode='legacy'``.
2. Footprint formulation recovers revolute endpoint move voxels that the
   legacy formula silently drops.
3. M_attn prior pulls long-prismatic-trajectory middle voxels out of base.
4. Air voxels (max_k O_k == 0) remain zero in both formulations.
5. Base voxels (all states occupied, std == 0) get p_base ≈ 1 in both.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipelines.sajo.anchors import joint_free_split


# ---- Fixtures --------------------------------------------------------


def _make_revolute_toy(D=16, K=6, device="cpu", dtype=torch.float32):
    """Synthesize a revolute K-state occupancy where the move part at each
    state occupies a DIFFERENT spatial region (rotation sweep).

    - Base cuboid: always occupied at (2..5, :, :)
    - Move cuboid: sweeps through rotation positions, one per state:
        state k: occupies (8..10, k*2:k*2+2, 5:7)   # distinct per k
    """
    O = torch.zeros((K, D, D, D), device=device, dtype=dtype)
    # Base
    O[:, 2:5, 2:10, 2:10] = 1.0
    # Move sweep (K distinct positions)
    for k in range(K):
        j0 = k * 2
        O[k, 8:10, j0: j0 + 2, 5:7] = 1.0
    return O


def _make_prismatic_long_drawer_toy(D=16, K=6, device="cpu", dtype=torch.float32):
    """Long drawer where the trajectory middle is occupied by 5/6 states.

    - Base cuboid: (2..5, :, :)
    - Drawer body length L=4 voxels at z direction, sliding by 1 voxel per state.
      - state k occupies (8..10, 5:7, k:k+4)
    - Trajectory voxels k=4 and k=5 are covered by 5/6 states each.
    """
    O = torch.zeros((K, D, D, D), device=device, dtype=dtype)
    O[:, 2:5, 2:10, 2:10] = 1.0  # base
    for k in range(K):
        O[k, 8:10, 5:7, k: k + 4] = 1.0
    return O


# ---- Tests -----------------------------------------------------------


def test_legacy_mode_is_unchanged_from_original_formula():
    """Legacy must reproduce the pre-fix formula exactly."""
    O = _make_revolute_toy()
    p_base, p_move = joint_free_split(
        O, sigma_b=0.25, sigma_m=0.15, tau_b=0.05, tau_m=0.05,
        mode="legacy",
    )
    # Reproduce the legacy formula manually
    mu = O.mean(dim=0)
    std = torch.sqrt(((O - mu) ** 2).mean(dim=0).clamp_min(1e-12))
    expected_base = mu * (1.0 - torch.sigmoid((std - 0.25) / 0.05))
    expected_move = mu * torch.sigmoid((std - 0.15) / 0.05)
    np.testing.assert_allclose(p_base.cpu().numpy(), expected_base.cpu().numpy(), atol=1e-6)
    np.testing.assert_allclose(p_move.cpu().numpy(), expected_move.cpu().numpy(), atol=1e-6)


def test_footprint_mode_recovers_revolute_endpoint_voxels():
    """Endpoint-state-only voxel should have p_move ~1 under footprint mode,
    but p_move < 0.5 under legacy mode (documented bug)."""
    O = _make_revolute_toy()
    # Pick a move voxel that only state 0 occupies
    j0 = 0
    v = (9, j0, 5)  # state-0-exclusive move voxel
    assert O[0, v[0], v[1], v[2]].item() == 1.0
    for k in range(1, O.shape[0]):
        assert O[k, v[0], v[1], v[2]].item() == 0.0, (
            f"expected voxel {v} to be exclusive to state 0, but state {k} also occupies"
        )

    # Legacy: p_move should be BELOW the 0.5 classification threshold (documents the bug)
    p_base_leg, p_move_leg = joint_free_split(O, mode="legacy")
    assert p_move_leg[v].item() < 0.5, (
        f"Legacy mode should UNDERESTIMATE endpoint move "
        f"(got p_move={p_move_leg[v].item():.3f})"
    )

    # Footprint: p_move should be HIGH (above 0.5)
    p_base_fp, p_move_fp = joint_free_split(O, mode="footprint")
    assert p_move_fp[v].item() > 0.5, (
        f"Footprint mode should RECOVER endpoint move "
        f"(got p_move={p_move_fp[v].item():.3f})"
    )


def test_footprint_mode_keeps_base_voxel_as_base():
    """Voxel occupied in all K states (pure base) should have p_base ≈ 1
    and p_move ≈ 0 under both modes."""
    O = _make_revolute_toy()
    v = (3, 5, 5)  # pure base voxel (all K states occupy)
    for k in range(O.shape[0]):
        assert O[k, v[0], v[1], v[2]].item() == 1.0

    for mode in ("legacy", "footprint"):
        p_base, p_move = joint_free_split(O, mode=mode)
        assert p_base[v].item() > 0.95, f"{mode}: p_base at pure base = {p_base[v].item():.3f}"
        assert p_move[v].item() < 0.05, f"{mode}: p_move at pure base = {p_move[v].item():.3f}"


def test_footprint_mode_zero_at_air_voxels():
    """Voxels never occupied in any state must give p_base == p_move == 0."""
    O = _make_revolute_toy()
    v = (15, 15, 15)  # corner air voxel
    for k in range(O.shape[0]):
        assert O[k, v[0], v[1], v[2]].item() == 0.0

    for mode in ("legacy", "footprint"):
        p_base, p_move = joint_free_split(O, mode=mode)
        assert p_base[v].item() < 1e-6
        assert p_move[v].item() < 1e-6


def test_long_drawer_middle_handled_by_footprint_alone():
    """Long-prismatic trajectory middle (mu high but std also high) is correctly
    classified by the footprint formula WITHOUT needing M_attn — the std-gate
    on p_base already drives p_base low, hence p_move = footprint - p_base ≈ 1.

    This documents the 2026-04-23 audit conclusion that M_attn is unnecessary
    in joint_free_split.
    """
    O = _make_prismatic_long_drawer_toy()
    # Voxel (9, 5, 4) is in drawer at states 1-4 (4/6 covered)
    v = (9, 5, 4)
    count = sum(1 for k in range(O.shape[0]) if O[k, v[0], v[1], v[2]].item() > 0.5)
    assert count >= 4, f"expected v={v} covered by many states; got {count}"

    p_base, p_move = joint_free_split(O, mode="footprint")
    # The std-gate alone should classify this trajectory voxel as move
    assert p_move[v].item() > p_base[v].item(), (
        f"Long-drawer middle should be move; got p_base={p_base[v].item()}, "
        f"p_move={p_move[v].item()}"
    )


def test_footprint_mode_clamps_p_move_to_nonnegative():
    """p_move = (footprint - p_base).clamp_min(0) must be ≥ 0 everywhere."""
    O = _make_revolute_toy()
    _, p_move = joint_free_split(O, mode="footprint")
    assert (p_move >= 0).all()


def test_p_move_plus_p_base_bounded_by_footprint_in_footprint_mode():
    """By construction: p_base + p_move = footprint when p_base ≤ footprint."""
    O = _make_revolute_toy()
    p_base, p_move = joint_free_split(O, mode="footprint")
    footprint = O.max(dim=0).values
    # p_move = footprint - p_base (clamped), so p_base + p_move == footprint when p_base ≤ footprint
    # Since p_base = mu * gate, and mu ≤ max_k = footprint, p_base ≤ footprint
    np.testing.assert_allclose(
        (p_base + p_move).cpu().numpy(),
        footprint.cpu().numpy(),
        atol=1e-6,
    )


def test_M_attn_parameter_is_now_a_no_op():
    """2026-04-23 audit: M_attn is no longer applied inside joint_free_split.

    Passing M_attn must NOT change the output (regression test: previous
    multiply-by-M_attn or even soft-gate variants were prone to flipping
    hinge-near-door voxels for revolute joints (mu=1, std=0, low M_attn)
    out of base. M_attn now lives in Stage C's downstream graph-cut
    refinement only.
    """
    O = _make_revolute_toy()
    p_base_no, p_move_no = joint_free_split(O, mode="footprint")

    M_attn = torch.full_like(O[0], 0.5)
    p_base_m, p_move_m = joint_free_split(O, mode="footprint", M_attn=M_attn)
    np.testing.assert_allclose(p_base_no.cpu().numpy(), p_base_m.cpu().numpy(), atol=1e-6)
    np.testing.assert_allclose(p_move_no.cpu().numpy(), p_move_m.cpu().numpy(), atol=1e-6)

    # Even a very low M_attn (which would previously flip true-base to move)
    # must now leave the output untouched.
    M_attn = torch.full_like(O[0], 0.05)
    p_base_lo, p_move_lo = joint_free_split(O, mode="footprint", M_attn=M_attn)
    np.testing.assert_allclose(p_base_no.cpu().numpy(), p_base_lo.cpu().numpy(), atol=1e-6)
    np.testing.assert_allclose(p_move_no.cpu().numpy(), p_move_lo.cpu().numpy(), atol=1e-6)


def test_M_attn_shape_still_validated():
    """M_attn shape validation is preserved even though the value is ignored,
    so a stale caller passing the wrong shape gets a clear error."""
    O = _make_revolute_toy()
    bad = torch.ones((4, 4, 4))  # wrong shape
    with pytest.raises(ValueError):
        joint_free_split(O, mode="footprint", M_attn=bad)


def test_invalid_mode_raises():
    O = _make_revolute_toy()
    with pytest.raises(ValueError):
        joint_free_split(O, mode="invalid_mode")


def test_M_attn_shape_mismatch_raises():
    O = _make_revolute_toy()
    bad_attn = torch.ones((8, 8, 8))  # wrong size
    with pytest.raises(ValueError):
        joint_free_split(O, M_attn=bad_attn, mode="footprint")


def test_M_attn_ignored_in_legacy_mode():
    """Legacy mode must not apply M_attn even if provided."""
    O = _make_revolute_toy()
    M_attn = torch.full_like(O[0], 0.1)  # would dramatically change base if applied

    p_base_no_m, p_move_no_m = joint_free_split(O, mode="legacy")
    p_base_m, p_move_m = joint_free_split(O, mode="legacy", M_attn=M_attn)

    np.testing.assert_allclose(p_base_no_m.cpu().numpy(), p_base_m.cpu().numpy(), atol=1e-6)
    np.testing.assert_allclose(p_move_no_m.cpu().numpy(), p_move_m.cpu().numpy(), atol=1e-6)


def test_footprint_default_mode():
    """Default mode (no arg) must be 'footprint'."""
    O = _make_revolute_toy()
    p_base_default, p_move_default = joint_free_split(O)
    p_base_fp, p_move_fp = joint_free_split(O, mode="footprint")
    np.testing.assert_allclose(p_base_default.cpu().numpy(), p_base_fp.cpu().numpy(), atol=1e-6)
    np.testing.assert_allclose(p_move_default.cpu().numpy(), p_move_fp.cpu().numpy(), atol=1e-6)


def test_shape_validation():
    with pytest.raises(ValueError):
        joint_free_split(torch.zeros((8, 8, 8)))  # 3D instead of 4D
    with pytest.raises(ValueError):
        joint_free_split(torch.zeros((2, 3, 8, 8, 8)))  # 5D instead of 4D

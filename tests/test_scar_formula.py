import sys
import os

import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "TRELLIS"))

from trellis.pipelines.samplers.scar import (
    compute_tweedie_variance_mask,
    apply_scar_gradient_push,
    generate_alpha_schedule,
)


def test_alpha_schedule_quadratic_peaks_at_zero_zeros_at_end():
    """Quadratic decay: alpha(0) = peak, alpha(total-1) = 0."""
    sched = generate_alpha_schedule(peak=0.5, total_steps=25, decay="quadratic")
    assert len(sched) == 25
    assert abs(sched[0] - 0.5) < 1e-9
    assert abs(sched[-1] - 0.0) < 1e-9
    # Middle value: alpha(12) = 0.5 * (1 - 12/24)^2 = 0.5 * 0.25 = 0.125
    assert abs(sched[12] - 0.125) < 1e-9


def test_alpha_schedule_linear_matches_formula():
    sched = generate_alpha_schedule(peak=1.0, total_steps=11, decay="linear")
    # At s=0: 1.0, at s=5: 0.5, at s=10: 0.0
    assert abs(sched[0] - 1.0) < 1e-9
    assert abs(sched[5] - 0.5) < 1e-9
    assert abs(sched[-1] - 0.0) < 1e-9


def test_alpha_schedule_cosine_starts_and_ends_correctly():
    sched = generate_alpha_schedule(peak=0.5, total_steps=25, decay="cosine")
    assert len(sched) == 25
    assert abs(sched[0] - 0.5) < 1e-9
    assert abs(sched[-1] - 0.0) < 1e-9
    # Monotonic non-increasing
    for i in range(1, len(sched)):
        assert sched[i] <= sched[i - 1] + 1e-9


def test_alpha_schedule_rejects_unknown_decay():
    with pytest.raises(ValueError):
        generate_alpha_schedule(peak=0.5, total_steps=25, decay="unknown")


def test_mask_active_fraction_controls_object_ratio():
    """active_fraction=0.1 should keep ~10% of latent voxels as object."""
    K, C, D, H, W = 3, 8, 16, 16, 16
    torch.manual_seed(0)
    # Simulate sparse object: high energy in a cube, low elsewhere.
    x_0 = torch.randn(K, C, D, H, W) * 0.1
    # Object region has stronger latent signal (object cube ~8x8x8 = 512 voxels,
    # out of 4096 total = 12.5%, so top-10% threshold should catch most of it).
    x_0[:, :, 4:12, 4:12, 4:12] += 3.0

    _, active = compute_tweedie_variance_mask(
        x_0, tau_percentile=0.65, eta=0.5,
        active_fraction=0.1, eps_log=1e-6,
    )
    n_active = active.sum().item()
    # Top 10% of 4096 = ~410 voxels
    assert 350 <= n_active <= 450, \
        f"active_fraction=0.1 should keep ~410 voxels, got {n_active}"
    # Most active voxels should fall within the object cube
    object_region = active[4:12, 4:12, 4:12]
    assert object_region.sum().item() > 0.8 * n_active, \
        f"only {object_region.sum().item()}/{n_active} active voxels in object region"


def test_mask_discriminates_base_vs_move_in_latent_space():
    """Base region (K states agree in latent) should get M~1;
    move region (K states differ) should get M~0."""
    K, C, D, H, W = 3, 8, 16, 16, 16
    torch.manual_seed(0)
    # All voxels are "object" (high energy) to avoid air filter confusion.
    x_0 = torch.randn(1, C, D, H, W).repeat(K, 1, 1, 1, 1) + 2.0
    # Inject per-state variation in the "move" half only
    x_0[:, :, 8:, :, :] = x_0[:, :, 8:, :, :] + torch.randn(K, C, 8, H, W) * 3.0

    M, active = compute_tweedie_variance_mask(
        x_0, tau_percentile=0.5, eta=0.5,
        active_fraction=1.0, eps_log=1e-6,  # keep all voxels
    )
    base_region_M = M[:8]    # stable half (K states agree)
    move_region_M = M[8:]    # varying half

    assert base_region_M.mean().item() > 0.7, \
        f"base M mean too low: {base_region_M.mean().item():.3f}"
    assert move_region_M.mean().item() < 0.3, \
        f"move M mean too high: {move_region_M.mean().item():.3f}"
    assert base_region_M.mean().item() - move_region_M.mean().item() > 0.5


def test_mask_is_bounded_in_unit_interval():
    """M values must always be in [0, 1] regardless of input statistics."""
    torch.manual_seed(0)
    K, C, D, H, W = 3, 8, 4, 4, 4
    # Mixed: some uniform, some random
    x_0 = torch.randn(K, C, D, H, W) * 3.0
    M, active = compute_tweedie_variance_mask(
        x_0, tau_percentile=0.65, eta=0.5,
        active_fraction=0.8, eps_log=1e-6,
    )
    assert M.shape == (D, H, W)
    assert (M >= 0).all() and (M <= 1).all(), \
        f"M out of [0,1]: min={M.min().item()}, max={M.max().item()}"


def test_mask_bimodal_distribution_separates():
    """Low-variance slice (base) should have higher M than high-variance slice (move).

    Uses comparable mean magnitudes so the active energy filter does not
    bias toward one slice. The discrimination comes from cross-state variance.
    """
    K, C, D, H, W = 5, 8, 4, 4, 4
    x_0 = torch.zeros(K, C, D, H, W)
    # Base: all K states have value 5.0 (zero cross-state variance)
    x_0[:, :, :2, :, :] = 5.0
    # Move: mean=5.0 but with per-state randn*2 (nonzero cross-state variance);
    # x0_bar at move is ~5.0 so energy is comparable to base's energy, which
    # means neither is excluded by the active-fraction energy filter.
    torch.manual_seed(0)
    x_0[:, :, 2:, :, :] = 5.0 + torch.randn(K, C, 2, H, W) * 2.0
    # Use tau_percentile=0.5 (median) for a balanced 50/50 base/move split.
    M, active = compute_tweedie_variance_mask(
        x_0, tau_percentile=0.5, eta=0.5,
        active_fraction=1.0, eps_log=1e-6,
    )
    assert M[:2].mean() > M[2:].mean() + 0.3, \
        f"base M={M[:2].mean().item():.3f}, move M={M[2:].mean().item():.3f}"


def test_gradient_push_w_floor_preserves_some_push_in_move():
    """With w_floor>0, push is non-zero even where M=0 (move region)."""
    K, C, D, H, W = 3, 8, 4, 4, 4
    torch.manual_seed(0)
    v_cfg = torch.randn(K, C, D, H, W)
    x_0 = torch.randn(K, C, D, H, W)
    # Half-base (M=1), half-move (M=0)
    M = torch.zeros(D, H, W)
    M[:2] = 1.0  # base half
    M[2:] = 0.0  # move half
    active = torch.ones(D, H, W, dtype=torch.bool)  # all within object
    w_floor = 0.2

    v_aug = apply_scar_gradient_push(
        v_cfg=v_cfg, x_0=x_0, M=M, alpha=1.0, t=0.5,
        w_floor=w_floor, active=active,
    )
    push = v_aug - v_cfg

    # Base region (M=1): weight = 1 * 0.8 + 0.2 = 1.0 -> full push
    # Move region (M=0): weight = 0 + 0.2 = 0.2 -> push at 0.2^2 = 4% strength
    base_push_norm = push[:, :, :2].abs().mean().item()
    move_push_norm = push[:, :, 2:].abs().mean().item()
    # Move push is non-zero
    assert move_push_norm > 0, "move region should receive non-zero push when w_floor > 0"
    # Base push is stronger than move push
    assert base_push_norm > move_push_norm
    # Ratio: (1.0^2) / (0.2^2) = 25
    ratio = base_push_norm / max(move_push_norm, 1e-9)
    assert 20 < ratio < 30, f"base/move push ratio expected ~25, got {ratio:.1f}"


def test_gradient_push_w_floor_zero_equals_hard_gate():
    """w_floor=0 should match original hard-mask behavior."""
    K, C, D, H, W = 3, 8, 4, 4, 4
    torch.manual_seed(0)
    v_cfg = torch.randn(K, C, D, H, W)
    x_0 = torch.randn(K, C, D, H, W)
    M = torch.rand(D, H, W)
    active = torch.ones(D, H, W, dtype=torch.bool)

    v_soft = apply_scar_gradient_push(
        v_cfg=v_cfg, x_0=x_0, M=M, alpha=1.0, t=0.5,
        w_floor=0.0, active=active,
    )
    v_hard = apply_scar_gradient_push(
        v_cfg=v_cfg, x_0=x_0, M=M, alpha=1.0, t=0.5,
    )
    torch.testing.assert_close(v_soft, v_hard)


def test_gradient_push_w_floor_does_not_leak_to_air():
    """Outside object (active=0), push should be zero regardless of w_floor."""
    K, C, D, H, W = 3, 8, 4, 4, 4
    torch.manual_seed(0)
    v_cfg = torch.randn(K, C, D, H, W)
    x_0 = torch.randn(K, C, D, H, W)
    M = torch.zeros(D, H, W)
    # Only half the voxels are active (inside object)
    active = torch.zeros(D, H, W, dtype=torch.bool)
    active[:2] = True

    v_aug = apply_scar_gradient_push(
        v_cfg=v_cfg, x_0=x_0, M=M, alpha=1.0, t=0.5,
        w_floor=0.2, active=active,
    )
    # Inactive region (air) should be untouched
    air_push = (v_aug[:, :, 2:] - v_cfg[:, :, 2:]).abs().mean().item()
    assert air_push == 0.0, f"air voxels received push: {air_push}"


def test_gradient_push_zero_alpha_is_identity():
    """alpha=0 -> v_aug == v_cfg."""
    K, C, D, H, W = 3, 8, 4, 4, 4
    v_cfg = torch.randn(K, C, D, H, W)
    x_0 = torch.randn(K, C, D, H, W)
    M = torch.ones(D, H, W) * 0.8
    v_aug = apply_scar_gradient_push(
        v_cfg=v_cfg, x_0=x_0, M=M, alpha=0.0, t=1.0,
    )
    torch.testing.assert_close(v_aug, v_cfg)


def test_gradient_push_mask_zero_is_identity():
    """M=0 everywhere -> v_aug == v_cfg regardless of alpha."""
    K, C, D, H, W = 3, 8, 4, 4, 4
    v_cfg = torch.randn(K, C, D, H, W)
    x_0 = torch.randn(K, C, D, H, W)
    M = torch.zeros(D, H, W)
    v_aug = apply_scar_gradient_push(
        v_cfg=v_cfg, x_0=x_0, M=M, alpha=1.0, t=1.0,
    )
    torch.testing.assert_close(v_aug, v_cfg)


def test_gradient_push_pulls_toward_consensus():
    """At base voxels (M=1), v_aug should be closer to v_bar than v_cfg."""
    torch.manual_seed(42)
    K, C, D, H, W = 6, 8, 4, 4, 4
    v_cfg = torch.randn(K, C, D, H, W)
    t = 0.92
    z_shared = torch.randn(1, C, D, H, W).repeat(K, 1, 1, 1, 1)
    x_0 = z_shared - t * v_cfg
    M = torch.ones(D, H, W)
    alpha = 1.0
    v_aug = apply_scar_gradient_push(v_cfg=v_cfg, x_0=x_0, M=M, alpha=alpha, t=t)
    v_bar = v_cfg.mean(dim=0, keepdim=True)
    pre_distance = (v_cfg - v_bar).norm(dim=1).mean().item()
    post_distance = (v_aug - v_bar).norm(dim=1).mean().item()
    assert post_distance < pre_distance * 0.2, \
        f"push should reduce distance to v_bar by >80%, got {pre_distance:.4f} -> {post_distance:.4f}"


def test_mask_M_squared_selectivity():
    """M^2 should be more selective than M: mid-range (M=0.5) -> M^2=0.25."""
    K, C, D, H, W = 3, 2, 2, 2, 2
    v_cfg = torch.ones(K, C, D, H, W)
    v_cfg[1] = 3.0
    x_0 = torch.zeros(K, C, D, H, W)
    x_0[1] = 2.0
    M_full = torch.ones(D, H, W)
    M_half = torch.ones(D, H, W) * 0.5
    v_aug_full = apply_scar_gradient_push(v_cfg, x_0, M_full, alpha=1.0, t=1.0)
    v_aug_half = apply_scar_gradient_push(v_cfg, x_0, M_half, alpha=1.0, t=1.0)
    diff_full = (v_aug_full - v_cfg).abs().mean().item()
    diff_half = (v_aug_half - v_cfg).abs().mean().item()
    assert 3.5 < diff_full / max(diff_half, 1e-9) < 4.5, \
        f"M^2 scaling wrong: diff_full/diff_half = {diff_full / max(diff_half, 1e-9):.2f}, expected ~4"

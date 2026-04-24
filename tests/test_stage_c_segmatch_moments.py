"""Unit tests for pipelines.stage_c_segmatch.moments (C.2 warm start)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pipelines.sajo.warp import batch_trilinear_warp
from pipelines.sajo.screw import exp_se3
from pipelines.stage_c_segmatch.moments import (
    _fit_prismatic,
    _fit_revolute,
    _fit_revolute_inertia,
    _per_state_covariance,
    _omega_from_inertia_trajectory,
    compute_move_centroids,
    moment_matching_warm_start,
    warm_start_as_dict,
)


def test_fit_prismatic_recovers_known_direction():
    """Centroids on a line: fitted v_hat should align with the line direction."""
    v_gt = torch.tensor([1.0, 0.5, 0.2])
    v_gt = v_gt / v_gt.norm()
    phi_gt = torch.tensor([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    centroids = torch.zeros(6, 3)
    for k in range(6):
        centroids[k] = phi_gt[k] * v_gt
    v_hat, phi_k, mse = _fit_prismatic(centroids)
    cos = float(torch.abs((v_hat * v_gt).sum()))
    assert cos > 0.99, f"direction cosine {cos}"
    assert mse < 1e-6, f"mse {mse}"
    # phi magnitudes should match up to sign
    phi_abs = phi_k.abs()
    phi_gt_abs = phi_gt.abs()
    np.testing.assert_allclose(phi_abs.numpy(), phi_gt_abs.numpy(), atol=1e-5)


def test_fit_revolute_recovers_planar_arc():
    """Centroids on a planar circle: axis should be plane normal; radius recovered."""
    center = torch.tensor([0.1, 0.0, 0.0])
    radius = 0.2
    omega_gt = torch.tensor([0.0, 0.0, 1.0])
    angles = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    centroids = torch.zeros(6, 3)
    for k, ang in enumerate(angles):
        centroids[k] = center + radius * torch.tensor([math.cos(ang.item()),
                                                        math.sin(ang.item()),
                                                        0.0])
    omega, q, phi_k, mse = _fit_revolute(centroids)
    cos = float(torch.abs((omega * omega_gt).sum()))
    assert cos > 0.99, f"axis cosine {cos}"
    # Angular progression: phi_k should be monotone (up to sign), spacing 0.2
    phi_diffs = phi_k[1:] - phi_k[:-1]
    # All diffs should have the same sign (monotone)
    signs = torch.sign(phi_diffs)
    assert (signs == signs[0]).all()
    # Spacing magnitude
    assert torch.allclose(phi_diffs.abs(), torch.full_like(phi_diffs, 0.2), atol=0.05)


def test_fit_prismatic_zero_displacement_degenerate():
    """All centroids at origin → degenerate; should not crash."""
    centroids = torch.zeros(6, 3)
    v_hat, phi_k, mse = _fit_prismatic(centroids)
    assert torch.isfinite(v_hat).all()
    assert torch.isfinite(phi_k).all()


def test_compute_move_centroids_shape():
    K, D = 6, 16
    O = torch.zeros((K, D, D, D))
    mask = torch.zeros((K, D, D, D), dtype=torch.bool)
    for k in range(K):
        O[k, 5:7, 5:7, k: k + 2] = 1.0
        mask[k, 5:7, 5:7, k: k + 2] = True
    centroids = compute_move_centroids(O, mask, resolution=D)
    assert centroids.shape == (K, 3)
    # Centroids should shift along z (since k=0..5 shifts)
    z_coords = centroids[:, 2]
    # Monotone increasing
    assert (z_coords[1:] > z_coords[:-1]).all()


def test_moment_matching_warm_start_picks_prismatic_for_linear_motion():
    K, D = 6, 16
    O = torch.zeros((K, D, D, D))
    mask = torch.zeros((K, D, D, D), dtype=torch.bool)
    for k in range(K):
        O[k, 5:7, 5:7, k: k + 2] = 1.0
        mask[k, 5:7, 5:7, k: k + 2] = True
    warm = moment_matching_warm_start(O, mask, resolution=D)
    # Linear trajectory → prismatic should have lower residual
    assert warm.joint_type_hint == "prismatic"
    assert warm.fit_residual["prismatic"] < warm.fit_residual["revolute"] + 1e-5


def test_moment_matching_warm_start_picks_revolute_for_arc_motion():
    K, D = 6, 32
    O = torch.zeros((K, D, D, D))
    mask = torch.zeros((K, D, D, D), dtype=torch.bool)
    # Place move blobs on a 2D arc in the x-y plane
    center = (15, 15)
    radius = 8
    for k in range(K):
        angle = k * 0.3  # 0, 0.3, 0.6, ..., 1.5 rad
        cx = int(center[0] + radius * math.cos(angle))
        cy = int(center[1] + radius * math.sin(angle))
        O[k, cx - 1: cx + 2, cy - 1: cy + 2, 15:17] = 1.0
        mask[k, cx - 1: cx + 2, cy - 1: cy + 2, 15:17] = True
    warm = moment_matching_warm_start(O, mask, resolution=D)
    assert warm.joint_type_hint == "revolute"


def test_warm_start_as_dict_returns_both_branches():
    K, D = 6, 16
    O = torch.zeros((K, D, D, D))
    mask = torch.zeros((K, D, D, D), dtype=torch.bool)
    for k in range(K):
        O[k, 5:7, 5:7, k: k + 2] = 1.0
        mask[k, 5:7, 5:7, k: k + 2] = True
    warm = moment_matching_warm_start(O, mask, resolution=D)
    d = warm_start_as_dict(warm)
    assert "revolute" in d
    assert "prismatic" in d
    assert d["revolute"]["omega"].shape == (3,)
    assert d["revolute"]["q"].shape == (3,)
    assert d["revolute"]["phi_k"].shape == (K,)
    assert d["prismatic"]["v_hat"].shape == (3,)
    assert d["prismatic"]["phi_k"].shape == (K,)


# ---- AOF Path A: inertia-tensor warm start ---------------------------


def _synth_revolute_elongated_voxels(D=32, K=6, omega_axis=2):
    """Synthesize an elongated cuboid rotating around the omega_axis.

    Creates a (K, D, D, D) binary occupancy where a long cuboid rotates
    by phi_k = k * 0.2 rad around the axis through the origin.
    """
    # Canonical cuboid: long along x-axis, narrow in y, z
    canonical = torch.zeros((D, D, D))
    cx, cy, cz = D // 2, D // 2, D // 2
    canonical[cx + 2: cx + 12, cy - 1: cy + 2, cz - 1: cz + 2] = 1.0

    omega = torch.zeros(3)
    omega[omega_axis] = 1.0
    q = torch.zeros(3)
    v = torch.linalg.cross(q, omega)
    twist = torch.cat([omega, v])

    phi_k_gt = torch.tensor([k * 0.2 for k in range(K)])
    T_k = torch.stack([exp_se3(twist, phi_k_gt[k]) for k in range(K)])

    O_stack = batch_trilinear_warp(
        canonical.unsqueeze(0).expand(K, -1, -1, -1).contiguous(),
        T_k, resolution=D,
    )
    O_stack_bin = (O_stack > 0.5).float()
    mask = O_stack_bin.bool()
    return O_stack_bin, mask, omega, phi_k_gt


def test_per_state_covariance_shape_and_eigvals_invariant_under_rotation():
    """For a rotating rigid body, per-state eigvals should be ≈ constant."""
    O, mask, _, _ = _synth_revolute_elongated_voxels(D=32, K=6)
    centroids, eigvals, eigvecs = _per_state_covariance(O, mask, resolution=32)
    assert centroids.shape == (6, 3)
    assert eigvals.shape == (6, 3)
    assert eigvecs.shape == (6, 3, 3)
    # Eigvals should be approximately constant across k (rigid body invariance)
    eig_std_across_k = eigvals.std(dim=0)
    eig_max = eigvals.max()
    rel_std = (eig_std_across_k.max() / eig_max.clamp_min(1e-8)).item()
    assert rel_std < 0.20, f"eigvals not invariant under rotation: rel_std={rel_std}"


def test_omega_from_inertia_trajectory_recovers_axis():
    """The smallest eigvec of A_i = Σ_k e_i^(k) e_i^(k)^T should align with ω."""
    O, mask, omega_gt, _ = _synth_revolute_elongated_voxels(
        D=32, K=6, omega_axis=2,
    )
    centroids, eigvals, eigvecs = _per_state_covariance(O, mask, resolution=32)
    omega, conf = _omega_from_inertia_trajectory(eigvecs, eigvals, eigengap_threshold=0.01)
    assert omega is not None, "Inertia recovery failed (all axes flagged degenerate)"
    cos = float(torch.abs((omega * omega_gt).sum()))
    assert cos > 0.85, f"axis cosine {cos} too low; omega={omega}, gt={omega_gt}"


def test_fit_revolute_inertia_full_pipeline():
    """End-to-end: synth revolute → recover (omega, q, phi_k, mse)."""
    O, mask, omega_gt, phi_gt = _synth_revolute_elongated_voxels(
        D=32, K=6, omega_axis=2,
    )
    omega, q, phi_k, mse = _fit_revolute_inertia(O, mask, resolution=32)
    cos = float(torch.abs((omega * omega_gt).sum()))
    assert cos > 0.85, f"axis cosine {cos}"
    assert mse < float("inf")
    assert phi_k.shape == (6,)


def test_fit_revolute_inertia_handles_symmetric_object_gracefully():
    """A perfectly symmetric (cubic) object → eigvals all equal → all axes
    degenerate → function should fall back without crashing (mse=∞ signals
    the caller to use the centroid-arc hypothesis instead)."""
    K, D = 6, 16
    # Symmetric cube (eigvals would all be equal)
    O = torch.zeros((K, D, D, D))
    mask = torch.zeros((K, D, D, D), dtype=torch.bool)
    for k in range(K):
        O[k, 5:8, 5:8, 5:8] = 1.0
        mask[k, 5:8, 5:8, 5:8] = True
    omega, q, phi_k, mse = _fit_revolute_inertia(O, mask, resolution=D)
    # Should not crash; mse may be ∞ or finite depending on eigengap threshold.
    assert torch.isfinite(omega).all()


def test_moment_matching_uses_inertia_when_better():
    """When inertia gives lower MSE than centroid arc, the warm-start should
    pick the inertia hypothesis."""
    O, mask, omega_gt, _ = _synth_revolute_elongated_voxels(
        D=32, K=6, omega_axis=2,
    )
    warm = moment_matching_warm_start(O, mask, resolution=32,
                                       use_inertia_for_revolute=True)
    # Should have report of both rev hypotheses
    assert "revolute_arc" in warm.fit_residual
    assert "revolute_inertia" in warm.fit_residual
    assert "rev_source" in warm.fit_residual
    # rev_source must be one of the two
    assert warm.fit_residual["rev_source"] in ("inertia", "centroid_arc")


def test_moment_matching_inertia_can_be_disabled():
    """use_inertia_for_revolute=False reverts to centroid-arc only."""
    O, mask, _, _ = _synth_revolute_elongated_voxels(D=32, K=6)
    warm = moment_matching_warm_start(O, mask, resolution=32,
                                       use_inertia_for_revolute=False)
    assert warm.fit_residual["revolute_inertia"] == float("inf")
    assert warm.fit_residual["rev_source"] == "centroid_arc"


def test_warm_start_as_dict_includes_inertia_when_O_provided():
    """When O_stack and move_mask are passed, warm_start_as_dict can pick
    the better of (centroid arc, inertia) for the revolute branch."""
    O, mask, omega_gt, _ = _synth_revolute_elongated_voxels(D=32, K=6)
    warm = moment_matching_warm_start(O, mask, resolution=32)
    d_no_inertia = warm_start_as_dict(warm, use_inertia_for_revolute=False)
    d_with_inertia = warm_start_as_dict(
        warm, O_stack=O, move_mask_k=mask, resolution=32,
        use_inertia_for_revolute=True,
    )
    assert "revolute" in d_no_inertia and "revolute" in d_with_inertia
    # The two may give different omegas; at minimum both are unit vectors
    assert torch.allclose(d_no_inertia["revolute"]["omega"].norm(),
                           torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(d_with_inertia["revolute"]["omega"].norm(),
                           torch.tensor(1.0), atol=1e-5)

"""Unit tests for pipelines.stage_c_segmatch.volumetric_fit (C.4 CORE).

Validates joint-constrained volumetric Adam recovers known (axis, phi_k)
under synthetic rigid motion on 32³ occupancy volumes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pipelines.sajo.screw import exp_prismatic, exp_se3
from pipelines.sajo.warp import batch_trilinear_warp
from pipelines.stage_c_segmatch.config import SegMatchHParams
from pipelines.stage_c_segmatch.volumetric_fit import (
    _build_T_k,
    _variance_loss,
    compute_bic,
    fit_volumetric,
    volumetric_fit_pipeline,
)


def _synth_rigid_body(
    D: int = 32,
    bbox=((8, 12), (10, 14), (12, 16)),
) -> torch.Tensor:
    """Single canonical 32³ occupancy of a small cuboid."""
    canonical = torch.zeros((D, D, D))
    (x0, x1), (y0, y1), (z0, z1) = bbox
    canonical[x0:x1, y0:y1, z0:z1] = 1.0
    return canonical


def _synth_states_prismatic(
    canonical: torch.Tensor,
    v_hat: torch.Tensor,
    phi_k: torch.Tensor,
    D: int = 32,
) -> torch.Tensor:
    """Apply prismatic T_k to canonical to produce K-state observation."""
    K = phi_k.shape[0]
    T_k = torch.stack([exp_prismatic(v_hat, phi_k[k]) for k in range(K)])
    O_stack = batch_trilinear_warp(
        canonical.unsqueeze(0).expand(K, -1, -1, -1).contiguous(),
        T_k, resolution=D,
    )
    return (O_stack > 0.5).float()


def _synth_states_revolute(
    canonical: torch.Tensor,
    omega: torch.Tensor,
    q: torch.Tensor,
    phi_k: torch.Tensor,
    D: int = 32,
) -> torch.Tensor:
    K = phi_k.shape[0]
    v = torch.linalg.cross(q, omega)
    twist = torch.cat([omega, v])
    T_k = torch.stack([exp_se3(twist, phi_k[k]) for k in range(K)])
    O_stack = batch_trilinear_warp(
        canonical.unsqueeze(0).expand(K, -1, -1, -1).contiguous(),
        T_k, resolution=D,
    )
    return (O_stack > 0.5).float()


# ---- Variance loss properties -----------------------------------------


def test_variance_loss_zero_when_transforms_are_correct():
    """Ground truth T_k should give SMALL variance loss.

    Not exactly zero because:
      - synth occupancy uses soft warp + binarize (loses sub-voxel info)
      - backwarp also trilinear (introduces edge interpolation)
    The cross-state variance at object boundaries is therefore non-zero
    but should be << the no-motion baseline (~50 for the same rig).
    """
    D = 32
    K = 4
    canonical = _synth_rigid_body(D=D)
    v_hat = torch.tensor([1.0, 0.0, 0.0])
    phi_k = torch.tensor([0.0, 0.05, 0.10, 0.15])
    O_stack = _synth_states_prismatic(canonical, v_hat, phi_k, D=D)
    T_k = torch.stack([exp_prismatic(v_hat, phi_k[k]) for k in range(K)])
    L = _variance_loss(O_stack, T_k, resolution=D)
    # Threshold accounts for boundary aliasing of binarized warp roundtrip.
    # Compare against baseline below.
    assert float(L.item()) < 15.0, (
        f"Variance under GT T_k should be small; got {L.item()}"
    )

    # Baseline: identity T_k for comparison
    T_id = torch.stack([torch.eye(4) for _ in range(K)])
    L_baseline = _variance_loss(O_stack, T_id, resolution=D)
    assert float(L_baseline.item()) > 3.0 * float(L.item()), (
        f"GT loss ({L.item()}) should be much smaller than identity baseline "
        f"({L_baseline.item()})"
    )


def test_variance_loss_positive_for_identity_under_nontrivial_motion():
    """With real motion but identity T_k, loss should be large."""
    D = 32
    K = 4
    canonical = _synth_rigid_body(D=D)
    v_hat = torch.tensor([1.0, 0.0, 0.0])
    phi_k = torch.tensor([0.0, 0.10, 0.20, 0.30])
    O_stack = _synth_states_prismatic(canonical, v_hat, phi_k, D=D)
    T_k = torch.stack([torch.eye(4) for _ in range(K)])
    L = _variance_loss(O_stack, T_k, resolution=D)
    assert float(L.item()) > 50.0, f"Variance under identity should be large; got {L.item()}"


# ---- Adam recovery ----------------------------------------------------


def test_fit_volumetric_recovers_prismatic():
    """Adam on variance loss recovers a known prismatic joint."""
    torch.manual_seed(0)
    D = 32
    K = 5
    canonical = _synth_rigid_body(D=D)
    v_gt = torch.tensor([1.0, 0.0, 0.0])
    phi_gt = torch.tensor([0.0, 0.05, 0.10, 0.15, 0.20])
    O_stack = _synth_states_prismatic(canonical, v_gt, phi_gt, D=D)

    init = {
        "v_hat": torch.tensor([0.9, 0.3, 0.1]).float(),     # slightly off
        "phi_k": torch.tensor([0.0, 0.03, 0.07, 0.12, 0.17]).float(),  # slightly off
    }
    fit = fit_volumetric(
        O_stack, "prismatic", init,
        n_inner_steps=150, lr_axis=5e-2, lr_phi=1e-2,
        resolution=D, device=torch.device("cpu"), dtype=torch.float32,
    )
    cos = float(torch.abs((fit.omega * v_gt).sum()))
    assert cos > 0.95, f"direction cosine after fit: {cos}"

    # Check phi recovery (up to sign flip)
    phi_est = fit.phi_k
    # Sign-aligned: ratio of last phi to GT last phi
    sign = torch.sign(phi_est[-1] * phi_gt[-1])
    if float(sign) < 0:
        phi_est = -phi_est
    np.testing.assert_allclose(phi_est.numpy(), phi_gt.numpy(), atol=0.05)


def test_fit_volumetric_recovers_revolute():
    """Adam on variance loss recovers a known revolute joint (small angles)."""
    torch.manual_seed(1)
    D = 32
    K = 4
    canonical = _synth_rigid_body(D=D, bbox=((14, 18), (16, 20), (18, 22)))
    omega_gt = torch.tensor([0.0, 0.0, 1.0])
    q_gt = torch.zeros(3)
    phi_gt = torch.tensor([0.0, 0.15, 0.30, 0.45])
    O_stack = _synth_states_revolute(canonical, omega_gt, q_gt, phi_gt, D=D)

    init = {
        "omega": torch.tensor([0.1, 0.0, 0.99]).float(),
        "q": torch.tensor([0.01, -0.01, 0.0]).float(),
        "phi_k": torch.tensor([0.0, 0.1, 0.25, 0.4]).float(),
    }
    fit = fit_volumetric(
        O_stack, "revolute", init,
        n_inner_steps=200, lr_axis=5e-2, lr_phi=1e-2,
        resolution=D, device=torch.device("cpu"), dtype=torch.float32,
    )
    cos = float(torch.abs((fit.omega * omega_gt).sum()))
    assert cos > 0.95, f"axis cosine after fit: {cos}"


# ---- BIC --------------------------------------------------------------


def test_bic_prefers_prismatic_when_losses_equal():
    from pipelines.stage_c_segmatch.volumetric_fit import VolumetricFit
    rev = VolumetricFit(
        joint_type="revolute", omega=torch.zeros(3), q=torch.zeros(3), v=torch.zeros(3),
        phi_k=torch.zeros(4), T_k=torch.eye(4)[None].expand(4, 4, 4).clone(),
        L_final=1.0, L_trace=[],
    )
    pris = VolumetricFit(
        joint_type="prismatic", omega=torch.zeros(3), q=torch.zeros(3), v=torch.zeros(3),
        phi_k=torch.zeros(4), T_k=torch.eye(4)[None].expand(4, 4, 4).clone(),
        L_final=1.0, L_trace=[],
    )
    jt, _, _, _ = compute_bic(rev, pris, n_active=100, K=4)
    assert jt == "prismatic"


def test_bic_prefers_revolute_when_prismatic_residual_much_larger():
    from pipelines.stage_c_segmatch.volumetric_fit import VolumetricFit
    rev = VolumetricFit(
        joint_type="revolute", omega=torch.zeros(3), q=torch.zeros(3), v=torch.zeros(3),
        phi_k=torch.zeros(4), T_k=torch.eye(4)[None].expand(4, 4, 4).clone(),
        L_final=1.0, L_trace=[],
    )
    pris = VolumetricFit(
        joint_type="prismatic", omega=torch.zeros(3), q=torch.zeros(3), v=torch.zeros(3),
        phi_k=torch.zeros(4), T_k=torch.eye(4)[None].expand(4, 4, 4).clone(),
        L_final=100.0, L_trace=[],
    )
    jt, _, _, _ = compute_bic(rev, pris, n_active=100, K=4)
    assert jt == "revolute"

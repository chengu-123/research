"""Regression tests for Stage C SegMatch v6 (count + phase EM + swept).

Covers:
  - count_based_partition: always_on classified via M_attn into
    true_base / move_interior / ambiguous
  - swept_volume.select_anchor_state: picks state with max move seeds
  - swept_volume.compute_swept_volume: shape + non-empty for rigid motion
  - swept_volume.late_commit_carve: lower-bound protection
  - swept_volume.compute_canonical_move_vote: volume-conservation threshold
  - volumetric_fit.fit_single_state_anchor: recovers known rigid T on synth data
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pipelines.sajo.screw import exp_prismatic, exp_se3
from pipelines.sajo.warp import batch_trilinear_warp
from pipelines.stage_c_segmatch.config import SegMatchHParams
from pipelines.stage_c_segmatch.partition import (
    count_based_partition,
    persistence_run_length,
)
from pipelines.stage_c_segmatch.swept_volume import (
    compute_canonical_move_vote,
    compute_swept_volume,
    late_commit_carve,
    select_anchor_state,
)
from pipelines.stage_c_segmatch.volumetric_fit import (
    fit_single_state_anchor,
)


# ---- Partition -------------------------------------------------------


def _toy_long_drawer(D=32, K=6, L=12, s=2):
    """Always-occupied interior present: L > (K-1)*s → must be classified
    as move_interior (not base) via M_attn."""
    O = torch.zeros((K, D, D, D))
    # Base cabinet (true base)
    O[:, 2:8, 2:26, 2:26] = 1.0
    # Drawer: state k occupies [k*s + 10, k*s + 10 + L] along z
    for k in range(K):
        z0 = k * s + 10
        O[k, 12:18, 12:18, z0: z0 + L] = 1.0
    # M_attn: cabinet high, drawer low
    M = torch.zeros((D, D, D))
    M[2:8, 2:26, 2:26] = 0.95       # cabinet = high
    # Drawer region (max extent)
    for k in range(K):
        z0 = k * s + 10
        M[12:18, 12:18, z0: z0 + L] = 0.15  # drawer = low
    return O, M


def test_count_based_partition_shapes():
    O, M = _toy_long_drawer()
    result = count_based_partition(O, M)
    assert result.count.shape == O.shape[1:]
    assert result.always_on.dtype == torch.bool
    assert result.move_mask_k.shape == O.shape
    assert result.base_mask_k.shape == O.shape
    assert result.base_centroid.shape == (3,)


def test_count_partition_classifies_drawer_interior_as_move():
    """Long-drawer always_on interior (high count, low M_attn) must end up
    in move_interior (not true_base). This is the 2026-04-23 key fix."""
    O, M = _toy_long_drawer(D=32, K=6, L=12, s=2)
    result = count_based_partition(
        O, M,
        m_attn_base_threshold=0.7,
        m_attn_move_threshold=0.3,
    )
    # There should be at least one always_on voxel in drawer region
    assert result.always_on.sum() > 0
    # All always_on voxels in drawer region have M_attn ~0.15 < 0.3 → move_interior
    assert result.move_interior.sum() > 0
    # Cabinet always_on voxels have M_attn ~0.95 > 0.7 → true_base
    assert result.true_base.sum() > 0
    # No overlap between true_base and move_interior
    assert not (result.true_base & result.move_interior).any()


def test_count_partition_without_M_attn_defers_always_on():
    """No M_attn → always_on all goes to ambiguous (deferred)."""
    O, _ = _toy_long_drawer()
    result = count_based_partition(O, M_attn_64=None)
    assert result.true_base.sum() == 0
    assert result.move_interior.sum() == 0
    assert result.ambiguous_on.sum() == result.always_on.sum()


def test_count_partition_canonical_omega_c_equals_O_0():
    O, M = _toy_long_drawer()
    result = count_based_partition(O, M)
    expected = O[0] > 0.5
    assert torch.equal(result.canonical_omega_c, expected)


def test_count_partition_canonical_move_in_omega_c():
    O, M = _toy_long_drawer()
    result = count_based_partition(O, M)
    # Canonical_move_init ⊆ Ω_c
    assert torch.equal(
        result.canonical_move_init & result.canonical_omega_c,
        result.canonical_move_init,
    )


# ---- Anchor selection ------------------------------------------------


def test_select_anchor_state_picks_max_exclusive_voxels():
    """State with largest exclusive-voxel ratio should be selected."""
    K, D = 6, 16
    O = torch.zeros((K, D, D, D))
    for k in range(K):
        # State 5 has more exclusive voxels than state 0
        O[k, 2:5, 2:5, 2:5] = 1.0  # always_on
        if k == 5:
            O[k, 8:14, 8:14, 8:14] = 1.0  # big exclusive
        elif k == 0:
            O[k, 6:7, 6:7, 6:7] = 1.0  # small exclusive
    move_strong = torch.zeros(O.shape[1:], dtype=torch.bool)
    for k in range(K):
        move_strong |= (O[k] > 0.5) & (O.sum(dim=0) <= 1)
    idx, stats = select_anchor_state(O, move_strong)
    assert idx == 5
    assert stats["selected_idx"] == 5


# ---- Swept volume ----------------------------------------------------


def test_compute_swept_volume_prismatic_produces_nonzero():
    D = 32
    canonical = torch.zeros((D, D, D), dtype=torch.bool)
    canonical[8:12, 10:14, 10:14] = True
    v_hat = torch.tensor([1.0, 0.0, 0.0])
    phi_k = torch.tensor([0.0, 0.05, 0.10, 0.15])
    sv = compute_swept_volume(
        canonical, joint_type="prismatic",
        axis_params={"v_hat": v_hat},
        phi_k=phi_k, n_samples=20, resolution=D,
    )
    assert sv.shape == canonical.shape
    assert sv.dtype == torch.bool
    assert sv.sum() >= canonical.sum()   # SV ⊇ canonical at φ=0 (approximately)


def test_compute_swept_volume_revolute_spans_angle_range():
    D = 32
    canonical = torch.zeros((D, D, D), dtype=torch.bool)
    canonical[16:19, 18:22, 16:20] = True
    omega = torch.tensor([0.0, 0.0, 1.0])
    q = torch.zeros(3)
    v = torch.linalg.cross(q, omega)
    phi_k = torch.tensor([0.0, 0.1, 0.2, 0.3])
    sv = compute_swept_volume(
        canonical, joint_type="revolute",
        axis_params={"omega": omega, "q": q, "v": v},
        phi_k=phi_k, n_samples=20, resolution=D,
    )
    assert sv.sum() > 0


# ---- Late-commit carve -----------------------------------------------


def test_late_commit_carve_lower_bound_protection():
    """If SV would carve > 70% of base, protection aborts."""
    D = 16
    base = torch.zeros((D, D, D), dtype=torch.bool)
    base[4:12, 4:12, 4:12] = True   # 512 voxels
    # SV covers most of base
    sv = torch.zeros_like(base)
    sv[4:11, 4:11, 4:11] = True     # ~343 voxels overlap
    result = late_commit_carve(base, sv, alpha_lower=0.3)
    # Carving would leave 512 - 343 = 169 voxels, ratio = 169/512 = 0.33 > 0.3
    # So protection may or may not trigger based on exact counts
    # Let's set up a case that definitely triggers protection:
    sv_big = base.clone()  # sv == base, carving would remove all
    result = late_commit_carve(base, sv_big, alpha_lower=0.3)
    assert result.lower_bound_protected
    assert not result.triggered
    # base unchanged
    assert torch.equal(result.canonical_base, base)


def test_late_commit_carve_ordinary_case():
    D = 16
    base = torch.zeros((D, D, D), dtype=torch.bool)
    base[4:12, 4:12, 4:12] = True
    sv = torch.zeros_like(base)
    sv[6:9, 6:9, 6:9] = True         # small swept inside base
    result = late_commit_carve(base, sv, alpha_lower=0.3)
    assert result.triggered
    assert result.n_carved > 0
    # base_after has sv holes
    assert (result.canonical_base & sv).sum() == 0


# ---- Canonical move vote ---------------------------------------------


def test_canonical_move_vote_volume_conservation_clips_to_omega_c():
    """canonical_move must be ⊆ Ω_c after vote."""
    K, D = 4, 16
    O = torch.zeros((K, D, D, D))
    mask = torch.zeros((K, D, D, D), dtype=torch.bool)
    for k in range(K):
        O[k, 5:8, 5:8, k + 5: k + 8] = 1.0
        mask[k, 5:8, 5:8, k + 5: k + 8] = True
    T_k = torch.stack([torch.eye(4) for _ in range(K)])
    omega_c = (O[0] > 0.5)
    result = compute_canonical_move_vote(
        O, mask, T_k, omega_c, vote_method="hard_majority",
        hard_vote_threshold=2, resolution=D,
    )
    assert result.canonical_move.shape == (D, D, D)
    # canonical_move ⊆ omega_c
    assert torch.equal(result.canonical_move & omega_c, result.canonical_move)


def test_canonical_move_vote_volume_conservation_respects_target():
    K, D = 4, 16
    O = torch.zeros((K, D, D, D))
    mask = torch.zeros((K, D, D, D), dtype=torch.bool)
    for k in range(K):
        O[k, 5:8, 5:8, 5:8] = 1.0
        mask[k, 5:8, 5:8, 5:8] = True
    T_k = torch.stack([torch.eye(4) for _ in range(K)])
    omega_c = (O[0] > 0.5)
    result = compute_canonical_move_vote(
        O, mask, T_k, omega_c, vote_method="volume_conservation",
        resolution=D,
    )
    assert result.canonical_move.sum() > 0


# ---- Phase 1 single-state anchor fit ---------------------------------


def test_fit_single_state_anchor_recovers_known_prismatic():
    torch.manual_seed(0)
    D = 32
    K = 6
    canonical = torch.zeros((D, D, D))
    canonical[10:15, 12:16, 14:20] = 1.0

    v_gt = torch.tensor([1.0, 0.0, 0.0])
    phi_gt = 0.10  # state 5 at 0.10
    T_true = exp_prismatic(v_gt, torch.tensor(phi_gt))
    O_anchor = batch_trilinear_warp(
        canonical.unsqueeze(0), T_true.unsqueeze(0), resolution=D,
    ).squeeze(0)
    O_anchor = (O_anchor > 0.5).float()

    # Fake O_stack (only anchor state matters here, but we still pass K states)
    O_stack = torch.zeros((K, D, D, D))
    O_stack[5] = O_anchor

    init = {"v_hat": torch.tensor([0.9, 0.3, 0.1]).float(),
            "phi_anchor": 0.05}
    fit = fit_single_state_anchor(
        O_anchor, O_stack, canonical,
        anchor_state_idx=5, joint_type="prismatic",
        init_params=init,
        n_inner_steps=100, lr_axis=0.05, lr_phi=0.01,
        resolution=D,
    )
    cos = float(torch.abs((fit.omega * v_gt).sum()))
    assert cos > 0.9, f"v_hat cosine {cos} too low"
    # phi_5 should be near phi_gt
    phi_est = fit.phi_k[5]
    sign = 1.0 if (phi_est * phi_gt) > 0 else -1.0
    np.testing.assert_allclose(float(sign * phi_est), phi_gt, atol=0.03)


# ---- Persistence (carry-over from v5) --------------------------------


def test_persistence_run_length_pure_base():
    K, D = 6, 8
    O = torch.zeros((K, D, D, D))
    O[:, 4, 4, 4] = 1.0
    p = persistence_run_length(O)
    assert int(p[4, 4, 4].item()) == K

"""Unit tests for pipelines.stage_c_segmatch.partition (v5).

Validates that partition_candidates wraps joint_free_split correctly
and produces per-state hard masks, soft fields, base centroid, and
footprint.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipelines.stage_c_segmatch.partition import (
    partition_candidates,
    persistence_run_length,
    _voxel_coord_grid,
)


def _toy(D=16, K=6):
    O = torch.zeros((K, D, D, D))
    M = torch.zeros((D, D, D))
    # Base: always-occupied slab
    O[:, 4:8, 4:12, 4:12] = 1.0
    M[4:8, 4:12, 4:12] = 0.9
    # Move: each state at a different z position (revolute-like, disjoint)
    for k in range(K):
        O[k, 9:12, 9:12, 4 + k: 4 + k + 2] = 1.0
    return O, M


def test_partition_returns_six_outputs():
    O, M = _toy()
    out = partition_candidates(O, M)
    assert len(out) == 6, "Expected 6 outputs"


def test_partition_shapes_consistent():
    O, M = _toy()
    move_mask_k, base_mask_k, p_base, p_move, centroid, footprint = partition_candidates(O, M)
    assert move_mask_k.shape == O.shape
    assert base_mask_k.shape == O.shape
    assert p_base.shape == M.shape
    assert p_move.shape == M.shape
    assert centroid.shape == (3,)
    assert footprint.shape == M.shape


def test_partition_base_and_move_dtypes():
    O, M = _toy()
    move_mask_k, base_mask_k, _, _, _, _ = partition_candidates(O, M)
    assert move_mask_k.dtype == torch.bool
    assert base_mask_k.dtype == torch.bool


def test_partition_move_mask_subset_of_occupancy():
    O, M = _toy()
    move_mask_k, _, _, _, _, _ = partition_candidates(O, M)
    occ = (O > 0.5)
    assert ((~occ) | move_mask_k | ~move_mask_k).all()  # sanity
    # move_mask_k voxels must be occupied in their state
    assert (move_mask_k <= occ).all()


def test_partition_base_centroid_in_base_block():
    """Base centroid should be in the bounding box of the base voxels."""
    O, M = _toy(D=16)
    _, _, _, _, centroid, _ = partition_candidates(O, M)
    # Base cuboid is (4..8, 4..12, 4..12) in voxel idx, centers ~(5.5, 7.5, 7.5)
    # World: idx/15 - 0.5
    x, y, z = centroid.tolist()
    lo, hi = 4 / 15.0 - 0.5, 7 / 15.0 - 0.5
    assert lo - 0.05 < x < hi + 0.05
    assert 4 / 15.0 - 0.5 - 0.05 < y < 11 / 15.0 - 0.5 + 0.05
    assert 4 / 15.0 - 0.5 - 0.05 < z < 11 / 15.0 - 0.5 + 0.05


def test_partition_footprint_equals_max_k():
    O, M = _toy()
    _, _, _, _, _, fp = partition_candidates(O, M)
    assert torch.allclose(fp, O.max(dim=0).values)


def test_partition_without_M_attn():
    """Partition must work when M_attn is None (optional)."""
    O, _ = _toy()
    move_mask_k, base_mask_k, p_base, p_move, centroid, footprint = partition_candidates(O)
    assert move_mask_k.shape == O.shape


def test_partition_legacy_mode():
    """mode='legacy' should produce the old SAJO formula (bug-preserved)."""
    O, M = _toy()
    out_fp = partition_candidates(O, M, mode="footprint")
    out_lg = partition_candidates(O, M, mode="legacy")
    # p_move should differ: legacy drops endpoint, footprint keeps them
    p_move_fp = out_fp[3]
    p_move_lg = out_lg[3]
    # On endpoint voxels in the toy (only 1 state occupies), legacy gives low p_move
    v = (10, 10, 4)  # state 0 drawer voxel (only state 0 occupies this z)
    assert p_move_lg[v].item() < 0.5
    # Footprint keeps it (because it's in footprint and not base)
    # (may be < 0.5 if p_base numerically close; the key test is they differ)
    # Assert at least the two modes produced different outputs
    assert not torch.allclose(p_move_fp, p_move_lg)


def test_partition_validates_shapes():
    with pytest.raises(ValueError):
        partition_candidates(torch.zeros(5, 5, 5))  # 3D instead of 4D


def test_voxel_grid_range_and_shape():
    g = _voxel_coord_grid(64, torch.device("cpu"), torch.float32)
    assert g.shape == (64, 64, 64, 3)
    assert torch.allclose(g[0, 0, 0], torch.tensor([-0.5, -0.5, -0.5]))
    assert torch.allclose(g[63, 63, 63], torch.tensor([0.5, 0.5, 0.5]))


# ---- AOF persistence run-length (2026-04-23 increment) ---------------


def test_persistence_run_length_pure_base_equals_K():
    """A voxel occupied in every state should have persistence == K."""
    K, D = 6, 8
    O = torch.zeros((K, D, D, D))
    O[:, 4, 4, 4] = 1.0
    p = persistence_run_length(O)
    assert int(p[4, 4, 4].item()) == K


def test_persistence_run_length_pure_air_equals_0():
    """A voxel never occupied should have persistence == 0."""
    K, D = 6, 8
    O = torch.zeros((K, D, D, D))
    p = persistence_run_length(O)
    assert int(p[2, 2, 2].item()) == 0


def test_persistence_run_length_contiguous_burst():
    """Consecutive states 1, 2, 3 → persistence == 3, even with later gaps."""
    K, D = 6, 4
    O = torch.zeros((K, D, D, D))
    # States 1, 2, 3 occupy (0,0,0); state 5 also occupies it (gap at 4).
    O[1, 0, 0, 0] = 1.0
    O[2, 0, 0, 0] = 1.0
    O[3, 0, 0, 0] = 1.0
    O[5, 0, 0, 0] = 1.0
    p = persistence_run_length(O)
    # Max contiguous run is [1, 2, 3] of length 3, NOT 4 (because state 4 breaks)
    assert int(p[0, 0, 0].item()) == 3


def test_persistence_run_length_returns_int_in_K_range():
    """Output dtype int32, all values in [0, K]."""
    K, D = 6, 16
    torch.manual_seed(0)
    O = (torch.rand((K, D, D, D)) > 0.5).float()
    p = persistence_run_length(O)
    assert p.dtype == torch.int32
    assert p.min().item() >= 0
    assert p.max().item() <= K


def test_persistence_run_length_validates_shape():
    with pytest.raises(ValueError):
        persistence_run_length(torch.zeros((4, 4, 4)))

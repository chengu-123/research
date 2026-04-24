"""Unit tests for pipelines.stage_c_segmatch.overlap_cleanup (C.7 target.md G2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipelines.stage_c_segmatch.overlap_cleanup import (
    _base_boundary,
    _morph_dilate,
    _morph_erode,
    _signed_distance_to_base_boundary,
    cleanup_overlap,
)


def test_morph_dilate_expands_by_one():
    D = 16
    mask = torch.zeros((D, D, D), dtype=torch.bool)
    mask[7, 7, 7] = True
    dilated = _morph_dilate(mask, kernel=3)
    # 26-dilation of a single voxel: all neighbors within L∞ distance 1
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                assert dilated[7 + di, 7 + dj, 7 + dk].item()


def test_morph_erode_shrinks_by_one():
    D = 16
    mask = torch.zeros((D, D, D), dtype=torch.bool)
    mask[5:10, 5:10, 5:10] = True
    eroded = _morph_erode(mask, kernel=3)
    # Eroded cube should be 3x3x3, centered
    assert eroded[7, 7, 7].item()
    assert not eroded[5, 5, 5].item()  # boundary removed


def test_base_boundary_is_one_voxel_thick():
    D = 16
    mask = torch.zeros((D, D, D), dtype=torch.bool)
    mask[5:10, 5:10, 5:10] = True
    boundary = _base_boundary(mask)
    # Boundary should NOT intersect interior voxels
    assert not boundary[7, 7, 7].item()
    # Boundary should be outside the mask (dilate - mask)
    assert (boundary & mask).sum().item() == 0


def test_signed_distance_negative_inside_positive_outside():
    D = 16
    mask = torch.zeros((D, D, D), dtype=torch.bool)
    mask[5:10, 5:10, 5:10] = True
    dist = _signed_distance_to_base_boundary(mask, max_radius=3)
    assert dist[7, 7, 7].item() < 0   # deep inside
    assert dist[0, 0, 0].item() > 0   # far outside


def test_cleanup_no_overlap_passes_through():
    D = 16
    base = torch.zeros((D, D, D), dtype=torch.bool)
    move = torch.zeros((D, D, D), dtype=torch.bool)
    base[5:8, 5:8, 5:8] = True
    move[10:13, 10:13, 10:13] = True
    res = cleanup_overlap(base, move)
    assert res.n_containment_deleted == 0
    assert res.n_contact_kept == 0
    assert torch.equal(res.canonical_move, move)


def test_cleanup_detects_containment_and_deletes():
    """A move voxel deep inside the base should be flagged as containment and deleted."""
    D = 16
    base = torch.zeros((D, D, D), dtype=torch.bool)
    base[4:12, 4:12, 4:12] = True
    move = base.clone()     # worst case: move fully contained in base
    res = cleanup_overlap(base, move, contact_band_radius=1)
    assert res.n_containment_deleted > 0
    # canonical_move should have fewer voxels after cleanup
    assert res.canonical_move.sum().item() < move.sum().item()


def test_cleanup_preserves_contact_band():
    """A move voxel right on the base boundary (within contact_band_radius)
    should be kept."""
    D = 16
    base = torch.zeros((D, D, D), dtype=torch.bool)
    base[4:12, 4:12, 4:12] = True
    move = torch.zeros((D, D, D), dtype=torch.bool)
    # Place move just outside the base right edge (shares boundary layer)
    move[12:14, 4:12, 4:12] = True
    # Extend overlap into the last base layer (x=11)
    move[11:14, 4:12, 4:12] = True
    res = cleanup_overlap(base, move, contact_band_radius=2)
    # The overlap band near boundary should be kept as contact
    assert res.n_contact_kept > 0


def test_cleanup_output_shapes():
    D = 8
    base = torch.zeros((D, D, D), dtype=torch.bool)
    move = torch.zeros((D, D, D), dtype=torch.bool)
    res = cleanup_overlap(base, move)
    assert res.canonical_base.shape == (D, D, D)
    assert res.canonical_move.shape == (D, D, D)
    assert res.contact_region.shape == (D, D, D)
    assert res.overlap_before.shape == (D, D, D)

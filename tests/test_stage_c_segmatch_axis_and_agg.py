"""Unit tests for pipelines.stage_c_segmatch.axis_refine and aggregation."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pipelines.sajo.screw import exp_prismatic, exp_se3
from pipelines.stage_c_segmatch.aggregation import aggregate_canonical
from pipelines.stage_c_segmatch.axis_refine import (
    contact_principal_axis,
    extract_contact_region,
    refine_axis,
)
from pipelines.stage_c_segmatch.config import SegMatchHParams
from pipelines.stage_c_segmatch.volumetric_fit import VolumetricFit as JointFit


def test_extract_contact_region_shape():
    D = 16
    base = torch.zeros((D, D, D), dtype=torch.bool)
    move = torch.zeros((D, D, D), dtype=torch.bool)
    base[5:10, 5:10, 5:10] = True
    move[9:14, 5:10, 5:10] = True
    contact = extract_contact_region(base, move)
    assert contact.shape == (D, D, D)
    assert contact.dtype == torch.bool
    assert contact.any()


def test_contact_principal_axis_known_line():
    """Contact voxels lying along the z-axis should produce a z-aligned
    principal direction.
    """
    D = 16
    contact = torch.zeros((D, D, D), dtype=torch.bool)
    for z in range(3, 13):
        contact[8, 8, z] = True
    M_attn = torch.ones((D, D, D)) * 0.5
    axis, pos, _coords = contact_principal_axis(contact, M_attn, resolution=D)
    assert axis is not None
    # The z-direction corresponds to last dim of (i, j, k) indexing
    assert abs(float(axis[2])) > 0.95


def test_refine_axis_revolute_pulls_axis_toward_contact_principal():
    D = 32
    # Construct a revolute fit whose axis is mis-aligned with the contact.
    contact = torch.zeros((D, D, D), dtype=torch.bool)
    # Contact along y-axis
    for y in range(10, 22):
        contact[15, y, 15] = True
    move = contact.clone()
    base = torch.zeros((D, D, D), dtype=torch.bool)
    for y in range(10, 22):
        base[14, y, 15] = True

    M_attn = torch.ones((D, D, D)) * 0.5

    omega_bad = torch.tensor([1.0, 0.0, 0.0])
    q = torch.zeros(3)
    v = torch.linalg.cross(q, omega_bad)
    phi_k = torch.tensor([0.0, 0.3, 0.6])
    T_k = torch.stack([exp_se3(torch.cat([omega_bad, v]), phi_k[k]) for k in range(3)])

    fit = JointFit(
        joint_type="revolute",
        omega=omega_bad, q=q, v=v, phi_k=phi_k, T_k=T_k,
        L_final=0.0, L_trace=[],
    )
    hp = SegMatchHParams()
    hp.axis_refine_iters = 100
    hp.adam_lr_axis = 5e-2

    result = refine_axis(fit, base, move, M_attn, hp, resolution=D)
    refined = result.joint_fit
    assert refined.joint_type == "revolute"
    # Axis should have moved toward y direction
    cos_y_after = float(torch.abs((refined.omega * torch.tensor([0.0, 1.0, 0.0])).sum()))
    cos_y_before = float(torch.abs((omega_bad * torch.tensor([0.0, 1.0, 0.0])).sum()))
    assert cos_y_after > cos_y_before


def test_aggregate_canonical_identity_transform_median():
    """With T_k = I for all k, median aggregation should just median O_stack."""
    K, D = 4, 16
    O = torch.zeros((K, D, D, D))
    # Base cuboid occupied in all K states
    O[:, 2:6, 2:10, 2:10] = 1.0
    # Move cuboid in different positions per state
    for k in range(K):
        O[k, 8:12, 6:10, 2 + k: 2 + k + 3] = 1.0

    base_mask = torch.zeros((D, D, D), dtype=torch.bool)
    base_mask[2:6, 2:10, 2:10] = True
    move_mask = torch.zeros((D, D, D), dtype=torch.bool)
    move_mask[8:12, 6:10, 2:8] = True

    T_k = torch.stack([torch.eye(4) for _ in range(K)])

    hp = SegMatchHParams()
    hp.warp_resolution = D

    agg = aggregate_canonical(O, base_mask, move_mask, T_k, hp)
    assert agg.canonical_base.shape == (D, D, D)
    assert agg.canonical_move.shape == (D, D, D)
    # Base should be fully filled in its bounding box (identity median of all-1s)
    assert agg.canonical_base[3, 5, 5].item() == 1
    # Per-state assignment: voxel (3,5,5) should be base for all K
    assert (agg.per_state_assignment[:, 3, 5, 5] == 0).all()


def test_aggregate_canonical_median_rejects_outlier_state():
    """Median should reject a single outlier state."""
    K, D = 5, 16
    O = torch.zeros((K, D, D, D))
    # 4 states have a cube at (5..7), 1 state has it at (10..12) — median says (5..7)
    for k in range(4):
        O[k, 5:8, 5:8, 5:8] = 1.0
    O[4, 10:13, 10:13, 10:13] = 1.0

    base_mask = torch.ones((D, D, D), dtype=torch.bool)   # treat all occupancy as "base"
    move_mask = torch.zeros((D, D, D), dtype=torch.bool)
    T_k = torch.stack([torch.eye(4) for _ in range(K)])

    hp = SegMatchHParams()
    hp.warp_resolution = D
    hp.aggregator = "median"

    agg = aggregate_canonical(O, base_mask, move_mask, T_k, hp)
    # Majority location (5..7) should be 1 in canonical_base
    assert agg.canonical_base[6, 6, 6].item() == 1
    # Outlier location (10..12) should be 0 (median across 5 values: 0,0,0,0,1 => 0)
    assert agg.canonical_base[11, 11, 11].item() == 0

"""Unit tests for Stage D motion ownership supervision."""

import os
import sys
import types

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if "lpips" not in sys.modules:
    lpips_stub = types.ModuleType("lpips")

    class _UnusedLPIPS(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, pred, target):
            return torch.zeros((pred.shape[0], 1, 1, 1), device=pred.device, dtype=pred.dtype)

    lpips_stub.LPIPS = _UnusedLPIPS
    sys.modules["lpips"] = lpips_stub

from pipelines.stage_d.config import StageDConfig
from pipelines.stage_d.losses import (
    dynamic_mask,
    loss_m_prior,
    loss_motion_ownership,
    loss_move_ceiling,
)
from pipelines.stage_d.schedules import schedule_lambda_gate, schedule_lambda_m_prior


def _refs_with_static_and_moving_regions():
    first = torch.zeros(3, 8, 8)
    last = torch.zeros(3, 8, 8)
    first[:, 0:2, 0:2] = 0.8
    last[:, 0:2, 0:2] = 0.8
    first[:, 3:5, 1:3] = 0.8
    last[:, 3:5, 5:7] = 0.8
    return first, last


def test_dynamic_mask_tracks_changed_pixels_only():
    first, last = _refs_with_static_and_moving_regions()

    dyn = dynamic_mask(first, last, dilate_px=0, thresh=0.04)

    assert dyn.shape == (1, 8, 8)
    assert torch.all(dyn[:, 3:5, 1:3] > 0.5)
    assert torch.all(dyn[:, 3:5, 5:7] > 0.5)
    assert torch.all(dyn[:, 0:2, 0:2] == 0.0)


def test_motion_ownership_penalizes_missing_dynamic_and_static_leakage():
    first, last = _refs_with_static_and_moving_regions()
    dyn = dynamic_mask(first, last, dilate_px=0, thresh=0.04)

    empty_move = torch.zeros(2, 1, 8, 8)
    missing_cover, no_static_leak = loss_motion_ownership(
        empty_move, first, last, dilate_px=0, thresh=0.04
    )
    assert missing_cover > 0.9
    assert torch.isclose(no_static_leak, torch.tensor(0.0), atol=1.0e-6)

    dynamic_move = dyn.unsqueeze(0).repeat(2, 1, 1, 1)
    covered, clean_static = loss_motion_ownership(
        dynamic_move, first, last, dilate_px=0, thresh=0.04
    )
    assert torch.isclose(covered, torch.tensor(0.0), atol=1.0e-6)
    assert torch.isclose(clean_static, torch.tensor(0.0), atol=1.0e-6)

    leaky_move = dynamic_move.clone()
    leaky_move[:, :, 0:2, 0:2] = 1.0
    _, static_leak = loss_motion_ownership(
        leaky_move, first, last, dilate_px=0, thresh=0.04
    )
    assert static_leak > 0.9


def test_gate_weight_stays_off_until_hardening_window():
    cfg = StageDConfig(
        lambda_gate=0.05,
        gate_hardening_start_frac=0.45,
        gate_hardening_ramp_frac=0.15,
    )

    assert schedule_lambda_gate(0.44, cfg) == 0.0
    assert 0.0 < schedule_lambda_gate(0.525, cfg) < 0.05
    assert schedule_lambda_gate(0.61, cfg) == 0.05


def test_m_prior_decays_to_main_anchor():
    cfg = StageDConfig(lambda_m_prior_warmup=0.10, lambda_m_prior_main=0.02)

    assert schedule_lambda_m_prior(0.00, cfg) == 0.0
    assert schedule_lambda_m_prior(0.06, cfg) == 0.10
    assert 0.02 < schedule_lambda_m_prior(0.20, cfg) < 0.10
    assert schedule_lambda_m_prior(0.31, cfg) == 0.02


def test_m_prior_penalizes_effective_move_logits():
    base_like_prior = torch.tensor([1.0, 0.0])
    correct_logits = torch.tensor([-3.0, 3.0])
    wrong_logits = torch.tensor([3.0, -3.0])

    assert loss_m_prior(correct_logits, base_like_prior) < loss_m_prior(wrong_logits, base_like_prior)


def test_move_ceiling_penalizes_whole_object_escape_only():
    low_mass_logits = torch.tensor([-2.0, -2.0, -2.0])
    high_mass_logits = torch.tensor([2.0, 2.0, 2.0])

    assert torch.isclose(
        loss_move_ceiling(low_mass_logits, ceiling=0.20),
        torch.tensor(0.0),
        atol=1.0e-6,
    )
    assert loss_move_ceiling(high_mass_logits, ceiling=0.20) > 0.5

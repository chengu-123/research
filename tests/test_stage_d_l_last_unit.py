"""Unit tests for Stage D final-frame anchoring.

The LPIPS package is not required for these tests; a tiny zero module keeps the
test focused on frame selection and aggregation.
"""

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

from pipelines.stage_d.losses import (
    LossInputs,
    aggregate_loss,
    loss_last_frame_anchor,
)


class ZeroLPIPS(torch.nn.Module):
    def forward(self, pred, target):
        return torch.zeros((), device=pred.device, dtype=pred.dtype)


def test_loss_last_frame_anchor_uses_final_frame_only():
    rgb = torch.zeros(5, 3, 4, 4)
    rgb[0] = 1.0
    rgb[-1] = 0.25
    target_last = torch.ones(3, 4, 4)

    loss = loss_last_frame_anchor(rgb, target_last, ZeroLPIPS())

    assert torch.isclose(loss, torch.tensor(0.75))


def test_aggregate_loss_adds_l_last_when_target_is_present():
    rgb = torch.zeros(5, 3, 4, 4)
    rgb[-1] = 0.5
    s0 = torch.zeros(3, 4, 4)
    s5 = torch.ones(3, 4, 4)
    dummy_vec = torch.zeros(1)

    inp = LossInputs(
        rgb_T3HW=rgb,
        r=dummy_vec,
        b=dummy_vec,
        axis=torch.tensor([1.0, 0.0, 0.0]),
        origin=torch.zeros(3),
        Delta_z_s=torch.zeros(1),
        alpha_m=dummy_vec,
        wan_video_target_T3HW_01=rgb.detach(),
        s_0_pure_3HW_01=s0,
        s_5_pure_3HW_01=s5,
        z_wan_target=torch.zeros(1, 1, 1, 1),
        anchors_world=torch.zeros(1, 3),
        shell_mask=torch.zeros(1, dtype=torch.bool),
        m_attn_at_U=torch.zeros(1),
        wan_cond={},
        tau=0.5,
        cfg_scale=0.0,
    )

    total, log = aggregate_loss(
        inp,
        ctx=object(),
        lpips_module=ZeroLPIPS(),
        cfg_lambdas_first=0.0,
        cfg_lambdas_last=2.0,
        cfg_lambdas_contact=0.0,
        cfg_lambdas_gate=0.0,
        cfg_lambdas_z=0.0,
        sched_lambdas_w_rfsds=(0.0, 0.0, 0.0),
        sched_lambda_shell=0.0,
        sched_lambda_m_prior=0.0,
    )

    assert torch.isclose(torch.tensor(log["L_last"]), torch.tensor(0.5))
    assert torch.isclose(total, torch.tensor(1.0))

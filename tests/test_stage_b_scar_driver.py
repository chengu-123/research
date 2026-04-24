import os
import json
import tempfile

import numpy as np
import torch
import torch.nn as nn
import pytest

from pipelines.stage_b_scar import run_scar, SCARResult


class MockDecoder(nn.Module):
    def forward(self, z):
        K = z.shape[0]
        logits = torch.full((K, 1, 64, 64, 64), -5.0)
        logits[:, :, 20:44, 20:44, 20:44] = 5.0
        return logits


class MockFlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resolution = 16
        self.patch_size = 2

    def forward(self, x, t, cond=None, **kwargs):
        return torch.zeros_like(x)


class MockPipeline:
    def __init__(self, device="cpu"):
        self.device = device
        self.models = {
            "sparse_structure_flow_model": MockFlowModel(),
            "sparse_structure_decoder": MockDecoder(),
        }
        from trellis.pipelines.samplers import SCARSampler
        self.sparse_structure_sampler = SCARSampler(sigma_min=0.0)
        self.sparse_structure_sampler_params = {
            "steps": 12,
            "cfg_strength": 7.5,
            "cfg_interval": (0.0, 1.0),
            "rescale_t": 1.0,
        }


def test_run_scar_returns_correct_shapes():
    """End-to-end driver produces K x 64^3 occupancy grids."""
    K = 3
    cond_tensors = {
        "cond": torch.randn(K, 1374, 1024),
        "neg_cond": torch.zeros(K, 1374, 1024),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_scar(
            pipe=MockPipeline(),
            cond=cond_tensors,
            K=K,
            out_dir=tmpdir,
            cfg_scar={
                "alpha_schedule": [0.7, 1.0, 1.0, 0.5],
                "icp_enabled": True,
                "icp_max_translation": 1.5,
            },
            seed=42,
            remove_disk_flag=False,
        )
        assert result.O_stack.shape == (K, 64, 64, 64)
        assert result.O_stack_soft.shape == (K, 64, 64, 64)
        assert result.z_final.shape == (K, 8, 16, 16, 16)
        assert len(result.icp_offsets) == K
        assert result.icp_offsets[0].shape == (3,)
        assert os.path.exists(os.path.join(tmpdir, "O_stack.npy"))
        assert os.path.exists(os.path.join(tmpdir, "scar_diagnostics.json"))
        assert os.path.exists(os.path.join(tmpdir, "icp_report.json"))


def test_run_scar_with_shifted_input_corrects_offset():
    """Driver's ICP should detect and correct an injected shift."""
    K = 2
    cond = {"cond": torch.randn(K, 1374, 1024),
            "neg_cond": torch.zeros(K, 1374, 1024)}

    class ShiftedDecoder(MockDecoder):
        """State 1 is shifted by (+1, 0, 0) relative to state 0."""
        def forward(self, z):
            logits = torch.full((z.shape[0], 1, 64, 64, 64), -5.0)
            logits[0, :, 20:44, 20:44, 20:44] = 5.0
            logits[1, :, 21:45, 20:44, 20:44] = 5.0
            return logits

    pipe = MockPipeline()
    pipe.models["sparse_structure_decoder"] = ShiftedDecoder()

    with tempfile.TemporaryDirectory() as tmpdir:
        # NOTE: with K=2, default vote_threshold=0.83 makes the intersection mask
        # unanimous (both must agree) which degenerates to geometric overlap, so
        # masked centroids coincide and ICP yields 0. We lower vote_threshold to
        # 0.5 here (= "any voxel occupied by >=1 of 2 states" = union) so the
        # shift is detectable. Real K=6 runs use the default 0.83 (>=5/6) unchanged.
        result = run_scar(
            pipe=pipe, cond=cond, K=K, out_dir=tmpdir,
            cfg_scar={"alpha_schedule": [0.7, 1.0, 1.0, 0.5],
                      "icp_enabled": True, "icp_max_translation": 1.5,
                      "icp_vote_threshold": 0.5},
            seed=0, remove_disk_flag=False,
        )
        offset_1 = result.icp_offsets[1]
        assert offset_1[0] < -0.5 and offset_1[0] > -1.5
        assert abs(offset_1[1]) < 0.2
        assert abs(offset_1[2]) < 0.2

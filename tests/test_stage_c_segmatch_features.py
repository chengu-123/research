"""Unit tests for pipelines.stage_c_segmatch.features.

Validates C.2 feature upsample + L2 normalization.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipelines.stage_c_segmatch.features import (
    l2_normalize,
    upsample_features,
)


def test_upsample_shape():
    z = torch.randn(6, 8, 16, 16, 16)
    up = upsample_features(z, out_size=64)
    assert up.shape == (6, 8, 64, 64, 64)


def test_upsample_preserves_value_at_corners_align_true():
    z = torch.randn(2, 4, 8, 8, 8)
    up = upsample_features(z, out_size=64)
    # align_corners=True: corner voxels should equal corner voxels.
    np.testing.assert_allclose(
        up[:, :, 0, 0, 0].cpu().numpy(),
        z[:, :, 0, 0, 0].cpu().numpy(),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        up[:, :, -1, -1, -1].cpu().numpy(),
        z[:, :, -1, -1, -1].cpu().numpy(),
        atol=1e-5,
    )


def test_l2_normalize_unit_norm():
    F = torch.randn(2, 8, 16, 16, 16)
    Fn = l2_normalize(F)
    norms = Fn.norm(dim=1)
    np.testing.assert_allclose(norms.cpu().numpy(),
                                np.ones_like(norms.cpu().numpy()),
                                atol=1e-5)


def test_l2_normalize_preserves_direction():
    F = torch.randn(1, 4, 2, 2, 2)
    Fn = l2_normalize(F)
    # Normalized vector should be a scalar multiple of the input vector.
    for i in range(2):
        for j in range(2):
            for k in range(2):
                v = F[0, :, i, j, k]
                vn = Fn[0, :, i, j, k]
                scale = v.norm().item()
                np.testing.assert_allclose(vn.cpu().numpy(),
                                            (v / scale).cpu().numpy(),
                                            atol=1e-5)


def test_shape_guards():
    with pytest.raises(ValueError):
        upsample_features(torch.zeros(4, 4, 4))
    with pytest.raises(ValueError):
        l2_normalize(torch.zeros(8, 16, 16, 16))

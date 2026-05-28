"""Unit tests for the Wan2.2-Fun-A14B-InP Stage A helpers.

These tests intentionally avoid importing VideoX-Fun. They validate the local
artifact contract and first/last-frame conditioning semantics that Stage ABCD
depends on.
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.stage_a_fun_inp import (
    _fun_inp_video_to_float01_uint8,
    _prepare_fun_inp_condition,
)


def _solid_image(value: int, size=(4, 6)) -> Image.Image:
    arr = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_prepare_fun_inp_condition_anchors_only_first_and_last_frames():
    start = _solid_image(32)
    end = _solid_image(224)

    video, mask, clip_image = _prepare_fun_inp_condition(
        start_image=start,
        end_image=end,
        frame_num=5,
        sample_hw=(8, 8),
    )

    assert video.shape == (1, 3, 5, 8, 8)
    assert mask.shape == (1, 1, 5, 8, 8)
    assert clip_image.size == (8, 8)
    assert torch.all(mask[:, :, 0] == 0)
    assert torch.all(mask[:, :, -1] == 0)
    assert torch.all(mask[:, :, 1:-1] == 255)
    assert float(video[:, :, 0].mean()) < float(video[:, :, -1].mean())


def test_fun_inp_video_conversion_accepts_pipeline_bfchw_output():
    raw = torch.linspace(-1.0, 1.0, steps=1 * 5 * 3 * 8 * 8).reshape(1, 5, 3, 8, 8)

    video_float, video_uint8 = _fun_inp_video_to_float01_uint8(
        raw,
        expected_F=5,
        expected_hw=(8, 8),
    )

    assert video_float.shape == (3, 5, 8, 8)
    assert video_uint8.shape == (3, 5, 8, 8)
    assert video_uint8.dtype == torch.uint8
    assert int(video_uint8.min()) == 0
    assert int(video_uint8.max()) == 255

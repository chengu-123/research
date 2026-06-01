"""Stage A smoke test: prompt builder + Stage A visualisation helpers.

Does NOT load Wan2.2 weights. Verifies that the new modules import
cleanly, prompts assemble correctly (zh + en), and the visualisation helpers
write the expected artifacts.

Run from repo root:
    python tests/test_stage_a_smoke.py
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import tempfile

import numpy as np
from PIL import Image

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipelines.utils.visualization_a import save_all_stage_a_visualisations
from pipelines.wan_helpers import build_articulated_prompts


def _check_prompts():
    pos_zh, neg_zh = build_articulated_prompts(
        "the drawer slowly slides outward in a continuous motion", lang="zh"
    )
    assert pos_zh.startswith("the drawer slowly slides outward")
    assert len(pos_zh) > len("the drawer slowly slides outward")
    assert len(neg_zh) > 20
    pos_en, neg_en = build_articulated_prompts("the drawer slides", lang="en")
    assert "Locked-off camera" in pos_en
    assert "static" not in neg_en.lower()
    assert "motionless" not in neg_en.lower()
    try:
        build_articulated_prompts("anything", lang="fr")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unsupported lang")
    try:
        build_articulated_prompts("", lang="zh")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty motion prompt")
    print("[smoke] prompts OK")


def test_stage_a_preserves_input_aspect_ratio_for_square_images():
    from pipelines.stage_a_wan import (
        _DEFAULT_WAN_SIZE_LABEL,
        _coerce_input_image,
        _predict_wan_output_hw,
        _WAN_MAX_AREA_CONFIGS,
    )

    image = Image.fromarray(
        np.zeros((800, 800, 3), dtype=np.uint8),
        mode="RGB",
    )
    pil = _coerce_input_image(image)
    assert pil.size == (800, 800)

    max_area = int(_WAN_MAX_AREA_CONFIGS[_DEFAULT_WAN_SIZE_LABEL])
    assert _predict_wan_output_hw((pil.height, pil.width), max_area) == (624, 624)


def test_stage_a_composites_rgba_input_on_white():
    from pipelines.stage_a_wan import _coerce_input_image

    arr = np.zeros((2, 2, 4), dtype=np.uint8)
    arr[:, :, :3] = 197
    arr[:, :, 3] = 0
    arr[0, 0] = [10, 20, 30, 255]
    image = Image.fromarray(arr, mode="RGBA")

    pil = _coerce_input_image(image)
    out = np.array(pil)
    assert pil.mode == "RGB"
    assert out[1, 1].tolist() == [255, 255, 255]
    assert out[0, 0].tolist() == [10, 20, 30]


def test_stage_a_shape_check_uses_predicted_wan_output_shape():
    import torch

    from pipelines.stage_a_wan import _wan_video_to_float01_uint8

    video = torch.zeros((3, 21, 624, 624), dtype=torch.float32)
    _float01, uint8 = _wan_video_to_float01_uint8(
        video,
        expected_F=21,
        expected_hw=(624, 624),
    )
    assert tuple(uint8.shape) == (3, 21, 624, 624)


def _make_synthetic_static_video(F=21, H=464, W=832, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 0.8, size=(3, H, W)).astype(np.float32)
    video = np.broadcast_to(base[:, None], (3, F, H, W)).copy()
    return video


def _check_visualisations(video):
    tmp = tempfile.mkdtemp(prefix="stage_a_smoke_")
    try:
        artifacts = save_all_stage_a_visualisations(
            video_3fhw_float01=video,
            out_dir=tmp,
            pos_prompt="dummy pos",
            neg_prompt="dummy neg",
            user_motion_prompt="dummy motion",
            lang="zh",
            seed=42,
            frame_num=video.shape[1],
            resolution_hw=(video.shape[2], video.shape[3]),
            sampling_steps=50,
            guide_scale=3.5,
            sample_shift=5.0,
            sample_solver="unipc",
            wan_ckpt_dir="/nonexistent/dummy",
        )
        expected = {
            "wan_video_target.mp4",
            "wan_video_grid.png",
            "keyframes_6.png",
            "meta.json",
        }
        produced = {os.path.basename(p) for p in artifacts}
        missing = expected - produced
        if missing:
            raise AssertionError(f"missing artifacts: {missing}")
        for p in artifacts:
            assert os.path.exists(p), f"artifact not on disk: {p}"
            assert os.path.getsize(p) > 0, f"artifact empty: {p}"
        with open(os.path.join(tmp, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["frame_num"] == video.shape[1]
        assert meta["guide_scale"] == 3.5
        assert "sanity_check" not in meta
        print(f"[smoke] visualisations OK in {tmp}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    _check_prompts()
    static_v = _make_synthetic_static_video()
    _check_visualisations(static_v)
    print("[smoke] all checks passed")


if __name__ == "__main__":
    main()

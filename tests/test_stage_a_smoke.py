"""Stage A smoke test: prompt builder + optical-flow sanity check + viz.

Does NOT load Wan2.2 weights. Verifies that the new modules import
cleanly, prompts assemble correctly (zh + en), the optical-flow check
classifies a synthetic static-vs-moving video correctly, and the
visualisation helpers write the five expected artifacts.

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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipelines.utils.optical_flow import background_static_check
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


def test_stage_a_wan_shape_contract_is_strict():
    stage_a_path = os.path.join(_REPO_ROOT, "pipelines", "stage_a_wan.py")
    with open(stage_a_path, encoding="utf-8") as f:
        source = f.read()

    assert "resolution_hw: Tuple[int, int] = (464, 832)" in source
    assert "_WAN_OUTPUT_TO_SIZE_LABEL" in source
    assert "expected_hw=(H, W)" in source
    assert "max_area=wan_max_area" in source
    assert "Wan output shape mismatch" in source


def _make_synthetic_static_video(F=21, H=464, W=832, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 0.8, size=(3, H, W)).astype(np.float32)
    video = np.broadcast_to(base[:, None], (3, F, H, W)).copy()
    return video


def _make_synthetic_moving_video(F=21, H=464, W=832, seed=0, shift_per_frame=5):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 0.8, size=(3, H, W)).astype(np.float32)
    video = np.zeros((3, F, H, W), dtype=np.float32)
    for f in range(F):
        dx = f * shift_per_frame
        shifted = np.roll(base, shift=dx, axis=-1)
        video[:, f] = shifted
    return video


def _check_optical_flow():
    static_v = _make_synthetic_static_video()
    report_static = background_static_check(static_v)
    assert report_static.passed, (
        f"static video failed sanity check: {report_static}"
    )

    moving_v = _make_synthetic_moving_video()
    report_moving = background_static_check(moving_v)
    assert not report_moving.passed, (
        f"moving video unexpectedly passed sanity check: {report_moving}"
    )
    print(
        f"[smoke] optical flow OK: "
        f"static moved_fraction={report_static.moved_fraction:.3f}, "
        f"moving moved_fraction={report_moving.moved_fraction:.3f}"
    )
    return static_v, report_static


def _check_visualisations(video, report):
    tmp = tempfile.mkdtemp(prefix="stage_a_smoke_")
    try:
        artifacts = save_all_stage_a_visualisations(
            video_3fhw_float01=video,
            out_dir=tmp,
            report=report,
            pos_prompt="dummy pos",
            neg_prompt="dummy neg",
            user_motion_prompt="dummy motion",
            lang="zh",
            seed=42,
            frame_num=video.shape[1],
            resolution_hw=(video.shape[2], video.shape[3]),
            sampling_steps=50,
            guide_scale=5.0,
            sample_shift=5.0,
            sample_solver="unipc",
            wan_ckpt_dir="/nonexistent/dummy",
        )
        expected = {
            "wan_video_target.mp4",
            "wan_video_grid.png",
            "keyframes_6.png",
            "optical_flow_per_frame.png",
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
        assert meta["sanity_check"]["passed"] == report.passed
        print(f"[smoke] visualisations OK in {tmp}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    _check_prompts()
    static_v, static_report = _check_optical_flow()
    _check_visualisations(static_v, static_report)
    print("[smoke] all checks passed")


if __name__ == "__main__":
    main()

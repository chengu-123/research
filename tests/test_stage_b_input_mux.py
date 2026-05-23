"""Smoke test for Stage B input multiplexer.

Does NOT load TRELLIS weights. Verifies that load_K_state_images correctly
dispatches to (a) a Stage A .pt video tensor and (b) a directory of segmented
PNGs, and returns K PIL Images of the expected shape and mode.

Run from repo root:
    python tests/test_stage_b_input_mux.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipelines.utils.state_input import (
    describe_input_source,
    load_K_state_images,
)


def _make_synthetic_video_pt(F: int, H: int, W: int, path: str, dtype=torch.uint8) -> None:
    """Write a Stage-A-like video tensor at ``path``."""
    rng = np.random.default_rng(seed=42)
    arr = (rng.integers(0, 256, size=(3, F, H, W), dtype=np.uint8))
    if dtype == torch.uint8:
        video = torch.from_numpy(arr)
    else:
        video = torch.from_numpy(arr).to(torch.float32) / 255.0
    torch.save(video, path)


def _make_synthetic_seg_dir(K: int, H: int, W: int, dir_path: str,
                            pattern: str = "{i:02d}_seg.png") -> None:
    """Write K dummy RGBA segmented PNGs at ``dir_path``."""
    os.makedirs(dir_path, exist_ok=True)
    rng = np.random.default_rng(seed=7)
    for i in range(K):
        rgba = rng.integers(0, 256, size=(H, W, 4), dtype=np.uint8)
        Image.fromarray(rgba, mode="RGBA").save(os.path.join(dir_path, pattern.format(i=i)))


def test_video_pt_mode() -> None:
    """Stage A .pt path: should sample K frames at state_indices."""
    tmp = tempfile.mkdtemp(prefix="stage_b_mux_video_")
    try:
        pt_path = os.path.join(tmp, "wan_video_target_3FHW_uint8.pt")
        _make_synthetic_video_pt(F=21, H=288, W=512, path=pt_path)

        for indices in [(0, 4, 8, 12, 16, 20), (0, 5, 10, 15, 18, 20)]:
            images = load_K_state_images(
                pt_path, K=6, state_indices=indices, out_mode="RGBA",
            )
            assert len(images) == 6, f"expected 6 images, got {len(images)}"
            for img in images:
                assert img.size == (512, 288), f"unexpected size {img.size}"
                assert img.mode == "RGBA", f"unexpected mode {img.mode}"

        # RGB mode
        images_rgb = load_K_state_images(
            pt_path, K=6, state_indices=(0, 4, 8, 12, 16, 20), out_mode="RGB",
        )
        for img in images_rgb:
            assert img.mode == "RGB"

        # float .pt also accepted
        pt_path_f = os.path.join(tmp, "video_float.pt")
        _make_synthetic_video_pt(F=21, H=64, W=64, path=pt_path_f, dtype=torch.float32)
        images_f = load_K_state_images(
            pt_path_f, K=6, state_indices=(0, 4, 8, 12, 16, 20),
        )
        assert all(img.size == (64, 64) for img in images_f)

        # describe_input_source on .pt
        info = describe_input_source(pt_path)
        assert info["mode"] == "video_pt"
        assert info["shape"] == (3, 21, 288, 512)

        print("[mux] video_pt mode OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_image_dir_mode() -> None:
    """Directory path: should load K PNGs per pattern."""
    tmp = tempfile.mkdtemp(prefix="stage_b_mux_dir_")
    try:
        _make_synthetic_seg_dir(K=6, H=320, W=320, dir_path=tmp)
        images = load_K_state_images(tmp, K=6, out_mode="RGBA")
        assert len(images) == 6
        for img in images:
            assert img.size == (320, 320)
            assert img.mode == "RGBA"

        info = describe_input_source(tmp)
        assert info["mode"] == "image_dir"

        # Wrong K -> FileNotFoundError
        try:
            load_K_state_images(tmp, K=7)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected FileNotFoundError for K=7")

        # Bad pattern -> ValueError
        try:
            load_K_state_images(tmp, K=6, image_pattern="seg_{x}.png")
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for bad image_pattern")

        print("[mux] image_dir mode OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bad_input_raises() -> None:
    """Neither file nor directory -> FileNotFoundError."""
    try:
        load_K_state_images("/nonexistent/path/that/does/not/exist", K=6)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError for nonexistent path")
    print("[mux] bad input OK")


def test_video_indices_range() -> None:
    """Out-of-range state_indices -> ValueError."""
    tmp = tempfile.mkdtemp(prefix="stage_b_mux_range_")
    try:
        pt_path = os.path.join(tmp, "v.pt")
        _make_synthetic_video_pt(F=9, H=32, W=32, path=pt_path)

        try:
            load_K_state_images(pt_path, K=6, state_indices=(0, 4, 8, 12, 16, 20))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for indices out of range")
        print("[mux] indices range OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_state_indices_length_mismatch() -> None:
    """len(state_indices) != K -> ValueError."""
    tmp = tempfile.mkdtemp(prefix="stage_b_mux_len_")
    try:
        pt_path = os.path.join(tmp, "v.pt")
        _make_synthetic_video_pt(F=21, H=32, W=32, path=pt_path)
        try:
            load_K_state_images(pt_path, K=6, state_indices=(0, 4, 8, 12, 16))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for K=6 vs len=5")
        print("[mux] indices length OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    test_video_pt_mode()
    test_image_dir_mode()
    test_bad_input_raises()
    test_video_indices_range()
    test_state_indices_length_mismatch()
    print("[mux] all checks passed")


if __name__ == "__main__":
    main()

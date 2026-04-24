import numpy as np
import pytest

from pipelines.utils.icp import (
    compute_translation_offset,
    apply_translation,
    align_to_reference,
)


def _make_cube(center, size, grid=64):
    """Return a (grid, grid, grid) binary occupancy of a cube centered at `center`."""
    g = np.zeros((grid, grid, grid), dtype=np.float32)
    cx, cy, cz = center
    h = size // 2
    g[cx - h: cx + h, cy - h: cy + h, cz - h: cz + h] = 1.0
    return g


def test_zero_offset_returns_zero():
    target = _make_cube((32, 32, 32), 10)
    source = target.copy()
    offset = compute_translation_offset(source, target, mask=None)
    np.testing.assert_allclose(offset, [0.0, 0.0, 0.0], atol=1e-6)


def test_known_translation_recovered():
    target = _make_cube((32, 32, 32), 10)
    source = _make_cube((32 + 2, 32 - 1, 32), 10)
    offset = compute_translation_offset(source, target, mask=None)
    np.testing.assert_allclose(offset, [-2.0, 1.0, 0.0], atol=0.05)


def test_mask_restricts_region():
    """Mask should exclude the shifted cube's move-region from centroid computation."""
    grid = 64
    target = np.zeros((grid, grid, grid), dtype=np.float32)
    source = np.zeros((grid, grid, grid), dtype=np.float32)
    target[15:25, 27:37, 27:37] = 1.0
    source[15:25, 27:37, 27:37] = 1.0
    target[45:55, 27:37, 27:37] = 1.0
    source[40:50, 27:37, 27:37] = 1.0
    mask = np.zeros((grid, grid, grid), dtype=bool)
    mask[15:25, 27:37, 27:37] = True
    offset = compute_translation_offset(source, target, mask=mask)
    np.testing.assert_allclose(offset, [0.0, 0.0, 0.0], atol=0.05)


def test_apply_translation_zero_is_identity():
    grid = _make_cube((32, 32, 32), 10)
    shifted = apply_translation(grid, np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(shifted, grid, atol=1e-6)


def test_apply_translation_integer_shift():
    grid = _make_cube((32, 32, 32), 10)
    shifted = apply_translation(grid, np.array([2.0, 0.0, 0.0]))
    expected = _make_cube((34, 32, 32), 10)
    np.testing.assert_allclose(shifted, expected, atol=1e-4)


def test_apply_translation_roundtrip():
    grid = _make_cube((32, 32, 32), 10)
    shifted = apply_translation(grid, np.array([1.3, -0.7, 0.5]))
    back = apply_translation(shifted, np.array([-1.3, 0.7, -0.5]))
    assert np.abs(back - grid).mean() < 0.02


def test_align_to_reference_basic():
    target = _make_cube((32, 32, 32), 10)
    source = _make_cube((32 + 1, 32, 32), 10)
    aligned, offset, capped = align_to_reference(
        source, target, base_mask=None, max_translation=1.5,
    )
    np.testing.assert_allclose(offset, [-1.0, 0.0, 0.0], atol=0.05)
    assert not capped
    assert np.abs(aligned - target).mean() < 0.02


def test_align_to_reference_caps_large_translation():
    target = _make_cube((32, 32, 32), 10)
    source = _make_cube((32 + 5, 32, 32), 10)
    _, offset, capped = align_to_reference(
        source, target, base_mask=None, max_translation=1.5,
    )
    assert np.isclose(np.linalg.norm(offset), 1.5, atol=0.01)
    assert capped


def test_align_to_reference_uses_base_mask():
    grid = 64
    target = np.zeros((grid, grid, grid), dtype=np.float32)
    source = np.zeros((grid, grid, grid), dtype=np.float32)
    target[20:30, 27:37, 27:37] = 1.0
    source[20:30, 27:37, 27:37] = 1.0
    target[50:60, 27:37, 27:37] = 1.0
    source[45:55, 27:37, 27:37] = 1.0
    base_mask = np.zeros((grid, grid, grid), dtype=bool)
    base_mask[20:30, 27:37, 27:37] = True
    _, offset, _ = align_to_reference(
        source, target, base_mask=base_mask, max_translation=1.5,
    )
    np.testing.assert_allclose(offset, [0.0, 0.0, 0.0], atol=0.05)

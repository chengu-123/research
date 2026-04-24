"""Translation-only rigid ICP for cross-state occupancy grid alignment.

Used by Stage B to correct residual rigid drift after SCAR sampling.
Operates on 64^3 binary or soft occupancy grids. The problem is greatly
simplified compared to full ICP because: (a) the shapes are already
near-aligned (< 2 voxel drift), (b) we only search translations (no
rotation), and (c) weighted centroid alignment is optimal for
translation-only alignment.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import shift as _scipy_shift


def compute_translation_offset(
    source: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute translation to bring `source` centroid onto `target` centroid.

    Parameters
    ----------
    source : np.ndarray
        Shape ``(D, H, W)``, float occupancy in [0, 1] or binary.
    target : np.ndarray
        Same shape as `source`.
    mask : np.ndarray, optional
        Shape ``(D, H, W)``, bool or 0/1 mask restricting which voxels
        contribute to centroid computation. If ``None``, all occupied
        voxels contribute.

    Returns
    -------
    np.ndarray
        Shape ``(3,)``, translation ``[dx, dy, dz]`` in voxel units that
        should be added to `source` to align with `target`.
        Returns zeros if either side has no mass.
    """
    assert source.shape == target.shape, \
        f"shape mismatch: {source.shape} vs {target.shape}"
    assert source.ndim == 3, f"expected 3D grid, got {source.ndim}D"

    s = source.astype(np.float64)
    t = target.astype(np.float64)
    if mask is not None:
        m = mask.astype(np.float64)
        s = s * m
        t = t * m

    s_sum = s.sum()
    t_sum = t.sum()
    if s_sum < 1e-8 or t_sum < 1e-8:
        return np.zeros(3, dtype=np.float32)

    dd, hh, ww = np.meshgrid(
        np.arange(source.shape[0], dtype=np.float64),
        np.arange(source.shape[1], dtype=np.float64),
        np.arange(source.shape[2], dtype=np.float64),
        indexing="ij",
    )
    s_centroid = np.array([
        (dd * s).sum() / s_sum,
        (hh * s).sum() / s_sum,
        (ww * s).sum() / s_sum,
    ])
    t_centroid = np.array([
        (dd * t).sum() / t_sum,
        (hh * t).sum() / t_sum,
        (ww * t).sum() / t_sum,
    ])
    return (t_centroid - s_centroid).astype(np.float32)


def apply_translation(
    grid: np.ndarray,
    offset: np.ndarray,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """Shift a 3D grid by a sub-voxel translation via trilinear interpolation.

    Parameters
    ----------
    grid : np.ndarray
        Shape ``(D, H, W)``.
    offset : np.ndarray
        Shape ``(3,)``, translation in voxel units.
    mode : str, default 'constant'
        Boundary handling for ``scipy.ndimage.shift``.
    cval : float, default 0.0
        Fill value when ``mode='constant'``.

    Returns
    -------
    np.ndarray
        Shape ``(D, H, W)``, dtype float32.
    """
    shifted = _scipy_shift(
        grid.astype(np.float32),
        shift=offset,
        order=1,
        mode=mode,
        cval=cval,
    )
    return shifted.astype(np.float32)


def align_to_reference(
    source: np.ndarray,
    target: np.ndarray,
    base_mask: Optional[np.ndarray] = None,
    max_translation: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Align `source` to `target` via translation-only centroid ICP.

    Parameters
    ----------
    source : np.ndarray
        Shape ``(D, H, W)``.
    target : np.ndarray
        Same shape as `source`.
    base_mask : np.ndarray, optional
        If provided, restrict centroid computation to base voxels.
    max_translation : float, default 1.5
        Clip translation magnitude to this many voxels to avoid large jumps
        from imperfect mask / mismatched move regions.

    Returns
    -------
    aligned : np.ndarray
        Shape ``(D, H, W)``, the translated source grid.
    offset : np.ndarray
        Shape ``(3,)``, applied offset (after capping).
    capped : bool
        True if the raw offset magnitude exceeded `max_translation`.
    """
    raw_offset = compute_translation_offset(source, target, mask=base_mask)
    magnitude = float(np.linalg.norm(raw_offset))
    capped = magnitude > max_translation
    if capped:
        offset = raw_offset * (max_translation / magnitude)
    else:
        offset = raw_offset
    aligned = apply_translation(source, offset)
    return aligned.astype(source.dtype), offset.astype(np.float32), capped

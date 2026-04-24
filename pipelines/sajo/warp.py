"""Differentiable trilinear warp of a 64^3 occupancy volume under SE(3).

The world-coordinate convention matches
:mod:`pipelines.utils.transforms`:``GridTransformer.rotate_grid``: voxel
index ``(i, j, k) in [0, 63]^3`` maps to world coordinates
``p = (i, j, k) / 63 - 0.5``, i.e. the unit cube centered at the origin.

Gradients flow through the SE(3) transform ``T`` because (a) the grid
construction is deterministic, (b) ``torch.linalg.inv`` is
differentiable, and (c) :func:`torch.nn.functional.grid_sample` is
differentiable in its ``grid`` argument.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Cache of normalized world-coord grids keyed by (device, dtype, resolution).
_GRID_CACHE: dict = {}


def _world_grid(resolution: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build the base world-coord grid ``(D, H, W, 3)``.

    Indexing mode ``ij`` means the first axis is "depth" (i), second is
    "height" (j), third is "width" (k). These are simply voxel indices;
    the world transform below normalizes them to ``[-0.5, 0.5]``.
    """
    key = (device, dtype, resolution)
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached
    idx = torch.arange(resolution, device=device, dtype=dtype)
    ii, jj, kk = torch.meshgrid(idx, idx, idx, indexing="ij")
    p_world = torch.stack([ii, jj, kk], dim=-1) / float(resolution - 1) - 0.5
    _GRID_CACHE[key] = p_world
    return p_world


def trilinear_warp(
    O_source: torch.Tensor,
    T: torch.Tensor,
    resolution: int = 64,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Warp the source volume by ``T``: ``O_warped(x) = O_source(T^{-1} x)``.

    Parameters
    ----------
    O_source : torch.Tensor
        ``(D, H, W)`` or ``(1, 1, D, H, W)``; float.
    T : torch.Tensor
        ``(4, 4)`` rigid transform acting on world coords
        ``p = (i, j, k)/(R-1) - 0.5``.
    resolution : int, default 64
        Volume side length ``R``.
    padding_mode : str
        Passed through to ``grid_sample``. ``'zeros'`` is the correct
        default for occupancy (voxels outside the source volume are empty).

    Returns
    -------
    torch.Tensor
        ``(D, H, W)`` warped float volume with the same dtype/device as
        ``O_source``, autograd-connected to ``T``.
    """
    if O_source.dim() == 3:
        vol = O_source[None, None]
    elif O_source.dim() == 5 and O_source.shape[0] == 1 and O_source.shape[1] == 1:
        vol = O_source
    else:
        raise ValueError(
            f"trilinear_warp expects (D,H,W) or (1,1,D,H,W), got {tuple(O_source.shape)}"
        )
    device = vol.device
    dtype = vol.dtype

    p_world = _world_grid(resolution, device, dtype)          # (D,H,W,3)
    ones = torch.ones_like(p_world[..., :1])
    p_h = torch.cat([p_world, ones], dim=-1)                  # (D,H,W,4)

    T_inv = torch.linalg.inv(T.to(device=device, dtype=dtype))  # (4,4)
    p_src = p_h.reshape(-1, 4) @ T_inv.T                      # (D*H*W, 4)
    p_src = p_src[..., :3].reshape(resolution, resolution, resolution, 3)

    # Normalize to grid_sample convention [-1, 1].
    grid = 2.0 * (p_src + 0.5) - 1.0                          # (D,H,W,3)

    # grid_sample's last-dim order is (x, y, z) corresponding to (W, H, D).
    # Our p_world ordering is (i, j, k) -> (D, H, W), so flip the last dim.
    grid_xyz = grid[..., [2, 1, 0]].unsqueeze(0)              # (1, D, H, W, 3)

    warped = F.grid_sample(
        vol,
        grid_xyz,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )                                                          # (1,1,D,H,W)
    return warped.squeeze(0).squeeze(0)


def batch_trilinear_warp(
    O_sources: torch.Tensor,
    Ts: torch.Tensor,
    resolution: int = 64,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Warp a batch of volumes by a batch of transforms.

    Parameters
    ----------
    O_sources : torch.Tensor
        ``(B, D, H, W)`` or ``(B, 1, D, H, W)``.
    Ts : torch.Tensor
        ``(B, 4, 4)``.

    Returns
    -------
    torch.Tensor
        ``(B, D, H, W)`` warped volumes.
    """
    if O_sources.dim() == 4:
        vol = O_sources.unsqueeze(1)
    elif O_sources.dim() == 5 and O_sources.shape[1] == 1:
        vol = O_sources
    else:
        raise ValueError(
            f"batch_trilinear_warp expects (B,D,H,W) or (B,1,D,H,W), got {tuple(O_sources.shape)}"
        )
    B = vol.shape[0]
    if Ts.shape != (B, 4, 4):
        raise ValueError(f"Ts must be (B,4,4)=({B},4,4), got {tuple(Ts.shape)}")
    device = vol.device
    dtype = vol.dtype

    p_world = _world_grid(resolution, device, dtype)          # (D,H,W,3)
    ones = torch.ones_like(p_world[..., :1])
    p_h = torch.cat([p_world, ones], dim=-1)                  # (D,H,W,4)

    T_inv = torch.linalg.inv(Ts.to(device=device, dtype=dtype))  # (B,4,4)
    # (B, N, 4) = (1, N, 4) @ (B, 4, 4)^T
    p_h_flat = p_h.reshape(-1, 4).unsqueeze(0)                 # (1, N, 4)
    p_src = torch.matmul(p_h_flat, T_inv.transpose(1, 2))     # (B, N, 4)
    p_src = p_src[..., :3].reshape(B, resolution, resolution, resolution, 3)

    grid = 2.0 * (p_src + 0.5) - 1.0
    grid_xyz = grid[..., [2, 1, 0]]                            # (B, D, H, W, 3)

    warped = F.grid_sample(
        vol,
        grid_xyz,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )                                                          # (B, 1, D, H, W)
    return warped.squeeze(1)

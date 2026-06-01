"""Per-voxel feature extraction for the H_sup / H_part / H_joint heads.

Given the captured SS-DiT hidden states from blocks {14, 16, 18} and the
fixed voxel set ``U_object`` (integer coords in [0, 63]^3), this module
produces a per-voxel feature ``feat[N_obj, feat_dim]`` where:

  feat_dim = 3 * 1024 (block hidden via grid_sample)
           + 36       (Fourier positional encoding, 6 freqs * 2 (sin/cos) * 3 axes)
           +  1       (SS-VAE decoded occupancy logit at the voxel)
           = 3109

Critical implementation details (method.md section 7.3):

1. **Hidden token ordering**: TRELLIS pos_emb is built with
   ``torch.meshgrid(*[arange(R)]*3, indexing='ij')``
   (sparse_structure_flow.py:102). This means token index
   ``i = D*16*16 + H*16 + W``, NOT W-major. To reshape
   ``hidden [B, 4096, 1024]`` back to ``[B, 1024, 16, 16, 16]`` we MUST
   use ``permute(0, 2, 1).contiguous().view(B, 1024, D=16, H=16, W=16)``,
   not ``view(...)`` directly (which would mix tokens and channels).

2. **grid_sample 5D axis order**: PyTorch's 5D grid_sample reads
   ``grid[..., 0]`` as the *W* (last spatial) axis, ``[..., 1]`` as *H*,
   ``[..., 2]`` as *D*. So the per-voxel coordinate must be
   ``stack([w_norm, h_norm, d_norm], dim=-1)``, NOT ``[d, h, w]``.

3. **Coordinate normalization**: voxels are int in [0, 63]; we map to
   [-1, 1] via ``(v + 0.5) / 64 * 2 - 1`` (voxel-center convention).
   The ``+ 0.5`` shift matches the D_GS world-space convention
   (decoder_gs.py:101 `xyz = (coords + 0.5) / resolution`) and the
   ``align_corners`` choice below.

4. **align_corners=True**: matches the pos_emb interpolation. Combined
   with the ``(v + 0.5) / R`` voxel-center mapping above, this yields
   trilinear interpolation that is exact at voxel centres.

5. **occupancy logit lookup**: ``occ_logits [1, 1, 64, 64, 64]`` is the
   raw conv3d output of SS-VAE-decoder (NOT sigmoided). We index
   ``occ_logits.view(-1)[flat_idx]`` where ``flat_idx = z*64*64 + y*64 + x``
   to match the natural flat-index of a [D, H, W] tensor.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .config import (
    TRELLIS_LATENT_GRID,
    TRELLIS_OCC_RES,
    TRELLIS_SS_MODEL_CH,
)


# =============================================================================
# Coordinate conversions
# =============================================================================

def voxel_to_world(u_xyz_int: torch.Tensor, res: int = TRELLIS_OCC_RES) -> torch.Tensor:
    """Convert integer voxel coords in [0, res-1] to world space in (-0.5, 0.5).

    Voxel-center convention (matches D_GS decoder_gs.py:101):
        world = (voxel + 0.5) / res - 0.5

    Parameters
    ----------
    u_xyz_int : Tensor [N, 3] of int
    res : int   default 64

    Returns
    -------
    Tensor [N, 3] of float, range (-0.5, 0.5)
    """
    return (u_xyz_int.float() + 0.5) / float(res) - 0.5


def world_to_voxel(w_xyz: torch.Tensor, res: int = TRELLIS_OCC_RES) -> torch.Tensor:
    """Inverse of ``voxel_to_world``, clamped to valid grid range."""
    v = ((w_xyz + 0.5) * float(res) - 0.5).round().long()
    return v.clamp(0, res - 1)


# =============================================================================
# Fourier positional encoding
# =============================================================================

def fourier_pe(coords_norm: torch.Tensor, num_freqs: int = 6) -> torch.Tensor:
    """Fourier features ``[sin(2^i * 2*pi * x), cos(2^i * 2*pi * x)]_{i=0..F-1}``.

    Standard NeRF-style PE on per-axis coordinates already normalized to
    roughly [-1, 1]. For ``num_freqs=6`` the output dim is
    ``3 axes * 2 (sin, cos) * 6 freqs = 36``.

    Parameters
    ----------
    coords_norm : Tensor [N, 3] in roughly [-1, 1]
    num_freqs : int

    Returns
    -------
    pe : Tensor [N, 6 * num_freqs]
    """
    if coords_norm.ndim != 2 or coords_norm.shape[-1] != 3:
        raise ValueError(
            f"fourier_pe expects coords [N, 3], got {tuple(coords_norm.shape)}"
        )
    device = coords_norm.device
    dtype = coords_norm.dtype
    freqs = 2.0 ** torch.arange(num_freqs, device=device, dtype=dtype)  # [F]
    # coords: [N, 3]; outer product with freqs -> [N, 3, F]
    scaled = coords_norm.unsqueeze(-1) * freqs.view(1, 1, -1) * torch.pi
    sin = torch.sin(scaled)   # [N, 3, F]
    cos = torch.cos(scaled)   # [N, 3, F]
    pe = torch.stack([sin, cos], dim=-1).reshape(coords_norm.shape[0], -1)
    return pe                 # [N, 3 * F * 2]


# =============================================================================
# Hidden state -> per-voxel feature
# =============================================================================

def _hidden_token_to_grid(h_token: torch.Tensor,
                           model_channels: int = TRELLIS_SS_MODEL_CH,
                           grid_res: int = TRELLIS_LATENT_GRID) -> torch.Tensor:
    """Reshape ``[B, L, C]`` -> ``[B, C, D, H, W]`` with token-order honoured.

    TRELLIS token order is ``meshgrid(indexing='ij')`` -> the flat index ``i``
    decomposes as ``i = d * R * R + h * R + w``, i.e. *D-major then H then W*.
    A direct ``.view(B, L, C).view(B, C, R, R, R)`` would be wrong: it would
    interpret the L axis as fast-varying-channel + slow-varying-token, mixing
    them. We use ``permute(0, 2, 1).contiguous().view(B, C, R, R, R)`` to
    avoid that. The ``contiguous()`` is required because ``permute`` returns
    a non-contiguous view; the subsequent ``.view`` will raise without it.
    """
    if h_token.ndim != 3:
        raise ValueError(
            f"hidden state must be [B, L, C], got {tuple(h_token.shape)}"
        )
    B, L, C = h_token.shape
    expected_L = grid_res ** 3
    if L != expected_L:
        raise ValueError(
            f"hidden length {L} != grid_res^3 = {expected_L}"
        )
    if C != model_channels:
        raise ValueError(
            f"hidden channels {C} != model_channels = {model_channels}"
        )
    return (
        h_token.permute(0, 2, 1).contiguous()                # [B, C, L]
               .view(B, C, grid_res, grid_res, grid_res)     # [B, C, D, H, W]
    )


def _voxel_to_normalized_grid(
    U_object_xyz: torch.Tensor,
    voxel_res: int = TRELLIS_OCC_RES,
    latent_grid: int = TRELLIS_LATENT_GRID,
) -> torch.Tensor:
    """Build the ``grid`` tensor used by ``F.grid_sample`` for 5D sampling.

    PyTorch 5D ``F.grid_sample`` expects ``grid[..., 0]`` = W, ``[..., 1]`` = H,
    ``[..., 2]`` = D in the [-1, 1] target sampler range. We map the U_object
    voxel-centre coords ``(v + 0.5) / voxel_res`` (still in [0, 1]) into the
    [-1, 1] sampler range, then stack as ``[w, h, d]``.

    Returns a tensor of shape ``[1, N, 1, 1, 3]`` ready for grid_sample's
    expected ``[B, D_out, H_out, W_out, 3]`` shape (we use 1D output here
    by setting ``H_out = W_out = 1``).
    """
    if U_object_xyz.ndim != 2 or U_object_xyz.shape[-1] != 3:
        raise ValueError(
            f"U_object_xyz must be [N, 3], got {tuple(U_object_xyz.shape)}"
        )
    N = U_object_xyz.shape[0]
    device = U_object_xyz.device

    # Voxel centre in [0, 1] (voxel grid is [0, voxel_res - 1])
    vc = (U_object_xyz.float() + 0.5) / float(voxel_res)   # [N, 3] in [0, 1]
    # Map to [-1, 1] sampler range (grid_sample uses [-1, 1] regardless of
    # the spatial dims of the input; we sample from a latent_grid^3 volume).
    vc_norm = vc * 2.0 - 1.0                                # [N, 3] in [-1, 1]

    d_norm = vc_norm[:, 0]                                  # [N]
    h_norm = vc_norm[:, 1]
    w_norm = vc_norm[:, 2]

    # IMPORTANT: PyTorch 5D grid_sample axis order is (W, H, D).
    grid = torch.stack([w_norm, h_norm, d_norm], dim=-1)    # [N, 3]
    # Reshape to grid_sample's expected [B=1, D_out=N, H_out=1, W_out=1, 3]
    grid = grid.view(1, N, 1, 1, 3)
    return grid


def sample_hidden_at_U(
    captured: Dict[int, torch.Tensor],
    U_object_xyz: torch.Tensor,
    occ_logits: torch.Tensor,
    capture_blocks: Tuple[int, ...] = (14, 16, 18),
    fourier_num_freqs: int = 6,
    voxel_res: int = TRELLIS_OCC_RES,
    latent_grid: int = TRELLIS_LATENT_GRID,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build per-voxel feature ``[N_obj, feat_dim]`` and per-voxel occ logit.

    Parameters
    ----------
    captured : dict
        Output of ``SS_DiT_WithAdapters.forward_capture``: block_idx ->
        ``[B=1, 4096, 1024]`` tensor (post-adapter hidden).
    U_object_xyz : Tensor [N_obj, 3] int
        Voxel coordinates in [0, voxel_res - 1].
    occ_logits : Tensor [1, 1, 64, 64, 64]
        Raw output of ``ss_vae_decoder(z_s_base)`` (NOT sigmoided).
    capture_blocks : tuple
        Same ordering as the ``adapters`` ModuleDict.

    Returns
    -------
    feat : Tensor [N_obj, feat_dim]
        Concatenation of: per-block grid-sampled hidden (3*1024) +
        Fourier PE of U_object coords (3*2*F=36) + occ logit at voxel (1).
    occ_at_U : Tensor [N_obj]
        Convenience copy of the raw occupancy logit at each U voxel. It is
        appended to ``feat`` as evidence, but not used as a calibrated support
        gate logit.
    """
    if not all(b in captured for b in capture_blocks):
        missing = [b for b in capture_blocks if b not in captured]
        raise KeyError(f"captured is missing blocks: {missing}")
    if occ_logits.shape != (1, 1, voxel_res, voxel_res, voxel_res):
        raise ValueError(
            f"occ_logits must be [1, 1, {voxel_res}, {voxel_res}, {voxel_res}], "
            f"got {tuple(occ_logits.shape)}"
        )
    N = U_object_xyz.shape[0]

    # 1) Build the grid (single 5D tensor reused across all blocks)
    grid = _voxel_to_normalized_grid(
        U_object_xyz, voxel_res=voxel_res, latent_grid=latent_grid,
    )                                                          # [1, N, 1, 1, 3]

    # 2) Sample per-block hidden via 5D bilinear (== trilinear in 3D)
    per_block_feats = []
    for block_idx in capture_blocks:
        h_token = captured[block_idx]                          # [1, 4096, 1024]
        h_grid = _hidden_token_to_grid(
            h_token, model_channels=TRELLIS_SS_MODEL_CH, grid_res=latent_grid,
        )                                                       # [1, 1024, 16, 16, 16]
        # F.grid_sample expects [B, C, D, H, W] input + [B, D_out, H_out, W_out, 3] grid
        # Output: [B, C, D_out=N, H_out=1, W_out=1]
        sampled = F.grid_sample(
            h_grid.float(),                                    # cast to fp32 for sampling stability
            grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        )                                                       # [1, 1024, N, 1, 1]
        sampled = sampled.squeeze(-1).squeeze(-1).squeeze(0)   # [1024, N]
        sampled = sampled.transpose(0, 1)                       # [N, 1024]
        per_block_feats.append(sampled)

    # 3) Fourier PE of voxel coords normalized to [-1, 1]
    coords_norm = (U_object_xyz.float() / float(voxel_res - 1)) * 2.0 - 1.0
    pe = fourier_pe(coords_norm, num_freqs=fourier_num_freqs)  # [N, 36]

    # 4) SS-VAE occupancy logit at each U_object voxel
    # Flat indexing convention: idx = d * R*R + h * R + w (matches token order)
    u = U_object_xyz.long()
    flat_idx = (
        u[:, 0] * voxel_res * voxel_res
        + u[:, 1] * voxel_res
        + u[:, 2]
    )
    occ_at_U = occ_logits.view(-1).index_select(0, flat_idx)   # [N]

    # 5) Concatenate -> [N, 3*1024 + 36 + 1] = [N, 3109]
    feat = torch.cat(
        per_block_feats + [pe.to(per_block_feats[0].dtype),
                           occ_at_U.unsqueeze(-1).to(per_block_feats[0].dtype)],
        dim=-1,
    )
    return feat, occ_at_U


# =============================================================================
# BinaryConcrete with straight-through estimator
# =============================================================================

class BinaryConcreteSTE(torch.autograd.Function):
    """Hard-binary forward, sigmoid-derivative backward (straight-through).

    Forward:  ``g = (sigmoid(logit) > 0.5).float()``  (hard 0/1)
    Backward: ``dg/dlogit = T * sigmoid(logit) * (1 - sigmoid(logit))``
              (i.e. gradient of ``sigmoid(logit / T)`` w.r.t. ``logit``)

    The temperature ``T`` controls how sharp the surrogate gradient is.
    Annealed from 1.5 (warmup) to 0.2 (transition) by the schedule.
    """

    @staticmethod
    def forward(ctx, logit: torch.Tensor, T: float) -> torch.Tensor:
        prob = torch.sigmoid(logit / max(float(T), 1.0e-6))
        ctx.save_for_backward(prob)
        ctx.T = float(T)
        return (prob > 0.5).to(logit.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (prob,) = ctx.saved_tensors
        # d(sigmoid(x/T))/dx = sigmoid(x/T) * (1 - sigmoid(x/T)) / T
        # We pass prob = sigmoid(x/T), so grad = grad_out * prob * (1 - prob) / T
        T = ctx.T if ctx.T > 1.0e-6 else 1.0e-6
        grad_logit = grad_out * prob * (1.0 - prob) / T
        return grad_logit, None


def binary_concrete_ste(logit: torch.Tensor, T: float) -> torch.Tensor:
    """Convenience wrapper for the autograd Function."""
    return BinaryConcreteSTE.apply(logit, T)


__all__ = [
    "voxel_to_world", "world_to_voxel",
    "fourier_pe",
    "sample_hidden_at_U",
    "binary_concrete_ste", "BinaryConcreteSTE",
]

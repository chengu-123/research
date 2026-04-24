"""SS-DiT 1024-dim hidden-state priors for Stage C MRF segmentation.

**v8.1 NOVELTY (2026-04-24)** — AAAI submission.

Background
----------
The 8-dim ``z_final`` VAE latent that Stage B already saves is a severe
bottleneck (128:1 compression from DiT's 1024 model_channels), collapsing
most per-voxel DINOv2-aligned semantic content into geometry-only codes.
For training-free articulated-part discrimination this is insufficient
(earlier experiments confirmed z_final-based classifier is empirically
equivalent to an EDT threshold — see material_classifier.py deprecation).

We instead hook the SS-DiT's PRE-OUTPUT hidden state at DIFT-analogous
mid-late blocks (14, 16, 18 of 24) at flow-time t ≈ 0.3. Each voxel then
carries a 1024-dim (or 3*1024 concat) feature that:
  * retains absolute voxel coordinate via sinusoidal APE through residuals
  * absorbs DINOv2 patch-level semantics through 24 cross-attention blocks
  * encodes local VAE-latent-driven geometry

This module converts that feature tensor into **two MRF priors** for
``seg_refine.py``:

1. **Drawer-vs-cabinet material prior** ``p_move_dit(v) ∈ [0, 1]``
   via prototype projection:
     drawer_proto   = mean of H over shell voxels (definite move)
     cabinet_proto  = mean of H over far-EDT always_on voxels (definite base)
     axis           = drawer_proto − cabinet_proto  (unit-normed discriminant)
     score(v)       = ⟨H(v) − (drawer_proto+cabinet_proto)/2, axis⟩
     p_move_dit(v)  = sigmoid(score / τ)
   This is a linear discriminant in the 1024-dim feature space — under
   Fisher-LDA assumptions (seed-class Gaussians with shared covariance)
   it is the Bayes-optimal decision surface, normalized to a probability.

2. **Cross-seed articulation-boundary prior** ``s_boundary(v) ∈ [0, 1]``:
     s(v) = ‖std_k(H[:, v, :])‖₂ / ‖mean_k(H[:, v, :])‖₂
     s_boundary(v) = clip((s(v) − s_q50) / (s_q95 − s_q50), 0, 1)
   Articulation-boundary voxels (where the moving part fluctuates across
   TRELLIS's K stochastic reconstructions) show the HIGHEST cross-seed
   variance of their 1024-dim DiT hidden state. This is a novel
   articulation-discovery signal that cannot be computed from z_final
   alone (whose K variants are over-smoothed by the VAE bottleneck).

These priors are fed as additive unary terms in the 2-label Potts MRF:
  U_base(v) += λ_dit_proto · sigmoid(⟨feat(v), axis⟩)
  U_move(v) += λ_dit_proto · (1 − sigmoid(⟨feat(v), axis⟩))
             − λ_dit_boundary · s_boundary(v)

Ablation (``use_dit_prior = False``) recovers the pure differential-
residual MRF and serves as the paper's baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class DiTPrior:
    """Per-voxel DiT-derived priors at resolution D=64."""

    p_move: torch.Tensor            # (D, D, D) in [0, 1] — drawer-prototype projection
    s_boundary: torch.Tensor        # (D, D, D) in [0, 1] — cross-seed variance boundary cue
    feat_mean: torch.Tensor         # (D, D, D, C_total) mean feature for downstream pairwise weights
    meta: Dict[str, float]          # n_shell_seeds, n_far_seeds, axis_norm, etc.


def _token_to_voxel_grid(
    hidden: torch.Tensor,            # (K, L=d^3, C)
    d_latent: int = 16,
    out_size: int = 64,
) -> torch.Tensor:
    """Reshape (K, L, C) tokens to (K, C, d, d, d), then trilinear upsample to (K, C, D, D, D).

    TRELLIS SS-DiT token indexing is row-major over (i, j, k); we reshape
    directly. Upsampling preserves align_corners=True to match the warp
    coord convention (``pipelines/sajo/warp.py``).
    """
    K, L, C = hidden.shape
    if L != d_latent ** 3:
        raise ValueError(f"token count L={L} != d_latent^3={d_latent**3}")
    grid = hidden.reshape(K, d_latent, d_latent, d_latent, C).permute(0, 4, 1, 2, 3).contiguous()
    # Upsample spatially to out_size^3
    if out_size == d_latent:
        return grid
    up = F.interpolate(
        grid,
        size=(out_size, out_size, out_size),
        mode="trilinear",
        align_corners=True,
    )
    return up


def _concat_blocks(
    hidden_dict: Dict[int, torch.Tensor],
    blocks: List[int],
    d_latent: int = 16,
    out_size: int = 64,
) -> torch.Tensor:
    """Concat multiple DiT blocks along channel dim → (K, C_total, D, D, D)."""
    tensors = []
    for idx in blocks:
        if idx not in hidden_dict:
            continue
        t = _token_to_voxel_grid(hidden_dict[idx], d_latent=d_latent, out_size=out_size)
        tensors.append(t)
    if not tensors:
        raise ValueError(f"no blocks from {blocks} found in hidden_dict")
    return torch.cat(tensors, dim=1)   # (K, sum_C, D, D, D)


def _compute_edt_mask(mask: torch.Tensor) -> torch.Tensor:
    """3D Euclidean distance transform (from True voxels outward).

    Returns distance from nearest True voxel, in voxel units.
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    m_np = mask.detach().cpu().numpy()
    # distance_transform_edt gives distance to nearest ZERO; we want distance
    # from TRUE voxels, so invert.
    if m_np.sum() == 0:
        return torch.zeros_like(mask, dtype=torch.float32)
    d_np = distance_transform_edt(~m_np)
    return torch.from_numpy(d_np).to(device=mask.device, dtype=torch.float32)


def compute_dit_priors(
    dit_hidden: Dict[int, torch.Tensor],        # {block_idx: (K, L, C)}
    shell: torch.Tensor,                          # (D, D, D) bool — definite move seed
    always_on: torch.Tensor,                      # (D, D, D) bool
    footprint: torch.Tensor,                      # (D, D, D) bool
    target_blocks: Optional[List[int]] = None,
    d_latent: int = 16,
    out_size: int = 64,
    far_aon_edt_threshold: float = 3.0,
    min_seeds: int = 20,
    projection_temperature: float = 0.1,
    s_boundary_quantiles: Tuple[float, float] = (0.50, 0.95),
) -> DiTPrior:
    """Compute per-voxel DiT-derived priors for MRF segmentation.

    Parameters
    ----------
    dit_hidden : dict loaded by ``features.load_dit_hidden``. Keys are
        integer block indices; values are ``(K, L=d_latent^3, C)``.
    shell : definite move seed mask (count 0<...<K).
    always_on : count == K.
    footprint : count > 0.
    target_blocks : which blocks to concat. Defaults to all numeric keys.
    far_aon_edt_threshold : always_on voxel is a definite cabinet seed
        if its EDT distance from shell exceeds this many voxels.
    min_seeds : minimum required seeds per class; falls back gracefully
        if fewer (returns priors at 0.5 everywhere).

    Returns
    -------
    DiTPrior dataclass with per-voxel priors at ``out_size``^3.
    """
    device = shell.device
    # Pick blocks (ignore _meta key if present)
    if target_blocks is None:
        target_blocks = sorted([k for k in dit_hidden.keys() if isinstance(k, int)])
    else:
        target_blocks = [int(b) for b in target_blocks]

    feat_K = _concat_blocks(dit_hidden, target_blocks, d_latent=d_latent, out_size=out_size)
    # feat_K: (K, C_total, D, D, D) on dit_hidden's device
    feat_K = feat_K.to(device)
    K = feat_K.shape[0]
    C_total = feat_K.shape[1]

    # ─── Cross-seed statistics ────────────────────────────────────────
    feat_mean = feat_K.mean(dim=0)                     # (C, D, D, D)
    feat_std = feat_K.std(dim=0, unbiased=False)       # (C, D, D, D)
    std_mag = feat_std.norm(dim=0)                     # (D, D, D)
    mean_mag = feat_mean.norm(dim=0).clamp_min(1e-6)   # (D, D, D)
    s = (std_mag / mean_mag)                            # normalized cross-seed std
    # Clip s_boundary to a 0..1 range around its q50/q95 inside footprint
    s_foot = s[footprint]
    if s_foot.numel() > 10:
        q_lo = float(torch.quantile(s_foot, s_boundary_quantiles[0]).item())
        q_hi = float(torch.quantile(s_foot, s_boundary_quantiles[1]).item())
    else:
        q_lo, q_hi = 0.0, 1.0
    s_boundary = ((s - q_lo) / max(q_hi - q_lo, 1e-6)).clamp(0.0, 1.0)

    # ─── Prototype-based drawer-vs-cabinet projection ────────────────
    # Drawer seeds: shell (definitely moves)
    drawer_seed_mask = shell
    # Cabinet seeds: always_on voxels FAR from shell in EDT
    edt_from_shell = _compute_edt_mask(shell)
    cabinet_seed_mask = always_on & (edt_from_shell > far_aon_edt_threshold)

    n_shell = int(drawer_seed_mask.sum().item())
    n_far = int(cabinet_seed_mask.sum().item())

    if n_shell < min_seeds or n_far < min_seeds:
        # Fall back to uniform 0.5 prior
        p_move = torch.full((out_size, out_size, out_size), 0.5,
                             device=device, dtype=torch.float32)
        feat_mean_voxel = feat_mean.permute(1, 2, 3, 0).contiguous()  # (D,D,D,C)
        return DiTPrior(
            p_move=p_move,
            s_boundary=s_boundary,
            feat_mean=feat_mean_voxel,
            meta={
                "n_shell_seeds": n_shell, "n_far_seeds": n_far,
                "axis_norm": 0.0, "fallback": 1.0,
                "s_q50": q_lo, "s_q95": q_hi,
            },
        )

    # Compute prototypes as mean feature over seed voxels
    # feat_mean is (C, D, D, D). We want to index by seed_mask:
    feat_flat = feat_mean.permute(1, 2, 3, 0).reshape(-1, C_total)    # (D^3, C)
    shell_flat = drawer_seed_mask.reshape(-1)
    far_flat = cabinet_seed_mask.reshape(-1)

    drawer_proto = feat_flat[shell_flat].mean(dim=0)       # (C,)
    cabinet_proto = feat_flat[far_flat].mean(dim=0)        # (C,)
    axis = drawer_proto - cabinet_proto                    # (C,)
    axis_norm = float(axis.norm().item())
    if axis_norm < 1e-8:
        # Seeds indistinguishable
        p_move = torch.full((out_size, out_size, out_size), 0.5,
                             device=device, dtype=torch.float32)
    else:
        axis_u = axis / axis_norm                                       # (C,)
        midpoint = 0.5 * (drawer_proto + cabinet_proto)                 # (C,)
        # Project every voxel's mean feature onto axis, centered at midpoint
        rel_feat = feat_flat - midpoint[None, :]                        # (D^3, C)
        score = (rel_feat * axis_u[None, :]).sum(dim=-1)                # (D^3,)
        # Normalize score to reasonable sigmoid range using seed-class spread
        shell_scores = (feat_flat[shell_flat] - midpoint[None, :]) @ axis_u
        far_scores = (feat_flat[far_flat] - midpoint[None, :]) @ axis_u
        spread = (shell_scores.std() + far_scores.std()).clamp_min(1e-3)
        score_normalized = score / (spread * projection_temperature)
        p_move_flat = torch.sigmoid(score_normalized)                   # (D^3,)
        p_move = p_move_flat.reshape(out_size, out_size, out_size)

    # Per-voxel feat_mean for potential pairwise use
    feat_mean_voxel = feat_mean.permute(1, 2, 3, 0).contiguous()        # (D,D,D,C)

    return DiTPrior(
        p_move=p_move.to(torch.float32),
        s_boundary=s_boundary.to(torch.float32),
        feat_mean=feat_mean_voxel.to(torch.float32),
        meta={
            "n_shell_seeds": float(n_shell),
            "n_far_seeds": float(n_far),
            "axis_norm": axis_norm,
            "fallback": 0.0,
            "s_q50": q_lo,
            "s_q95": q_hi,
            "projection_spread": float(spread.item()) if axis_norm >= 1e-8 else 0.0,
        },
    )

"""BCAC: Base-Consistent Articulation-aware Conditioning.

Replaces VGCF's weak velocity perturbation with **in-network attention
modification** inside the frozen TRELLIS Stage-1 DiT. Two mechanisms
work together:

1. **Cross-Attention Consensus**: In selected blocks, for non-anchor
   states (k > 0), blend the per-state cross-attention output with the
   output produced using the anchor state-0's DINOv2 conditioning.
   This eliminates the per-state conditioning divergence at the source
   (each layer's cross-attn is where the conditions get injected).

2. **Self-Attention Output Blending**: In the same blocks, blend each
   state's self-attention output toward the K-state mean, gated by a
   soft base mask M(p). This reinforces geometric consensus among base
   tokens.

Both modifications operate at the **output level** — each individual
attention forward pass sees in-distribution inputs. This avoids the OOD
risks of attention inflation (MVDream) or K/V fusion (MorphAny3D §5.1).

The base mask M(p) and the injection strength α are controlled by a
three-phase time schedule:
  t > t_full : α=1, M=1  — full consensus (everything still similar)
  t ∈ [t_release, t_full] : α decays, M from Tweedie variance
  t < t_release : α=0  — free evolution (move differentiates)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from tqdm import tqdm

from .flow_euler import FlowEulerGuidanceIntervalSampler


# ---- BCAC context passed to patched blocks ---------------------------


@dataclass
class BCACContext:
    """Shared mutable state read by each patched block at every step."""
    active: bool = False
    alpha: float = 0.0
    M_tokens: Optional[torch.Tensor] = None   # (1, N, 1), N = num tokens
    cond_anchor: Optional[torch.Tensor] = None  # (K, ctx_len, ctx_dim), all = state-0


# ---- Block monkey-patch ---------------------------------------------


def _make_bcac_forward(original_forward, ctx: BCACContext):
    """Return a replacement ``_forward`` for a ``ModulatedTransformerCrossBlock``
    that inserts BCAC self-attn blending and cross-attn consensus."""

    def _bcac_forward(self_block, x, mod, context):
        # Infer the compute dtype from x (fp16 when model.use_fp16=True).
        _dtype = x.dtype

        # --- AdaLN modulation (same as original) ---
        if self_block.share_mod:
            chunks = mod.chunk(6, dim=1)
        else:
            chunks = self_block.adaLN_modulation(mod).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

        # --- Self-attention ---
        h = self_block.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self_block.self_attn(h)           # (K, N, C)
        h = h * gate_msa.unsqueeze(1)

        # BCAC: self-attn output blending toward K-state mean.
        if ctx.active and ctx.M_tokens is not None:
            M = ctx.M_tokens.to(dtype=_dtype, device=h.device)
            h_mean = h.mean(dim=0, keepdim=True)      # (1, N, C)
            h = h + ctx.alpha * M * (h_mean - h)

        x = x + h

        # --- Cross-attention ---
        h_norm = self_block.norm2(x)                   # (K, N, C)
        h_ca = self_block.cross_attn(h_norm, context)  # per-state condition

        # BCAC: cross-attn consensus — blend toward anchor (state-0) output.
        if ctx.active and ctx.cond_anchor is not None and ctx.M_tokens is not None:
            # Cast anchor conditioning to the model's compute dtype.
            anchor = ctx.cond_anchor.to(dtype=_dtype, device=h_norm.device)
            h_ca_anchor = self_block.cross_attn(h_norm, anchor)
            M = ctx.M_tokens.to(dtype=_dtype, device=h_ca.device)
            # For state 0: cond_anchor[0] == context[0] → h_ca_anchor[0] ≈ h_ca[0]
            # → blend is a near-zero no-op. For k>0: pushes toward anchor.
            h_ca = h_ca + ctx.alpha * M * (h_ca_anchor - h_ca)

        x = x + h_ca

        # --- FFN (same as original) ---
        h = self_block.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self_block.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h

        return x

    return _bcac_forward


class BCACBlockPatcher:
    """Install / remove BCAC patches on a list of transformer blocks."""

    def __init__(self, blocks: torch.nn.ModuleList, block_indices: List[int],
                 ctx: BCACContext):
        self.blocks = blocks
        self.indices = list(block_indices)
        self.ctx = ctx
        self._originals: Dict[int, Any] = {}

    def install(self) -> None:
        for idx in self.indices:
            block = self.blocks[idx]
            self._originals[idx] = block._forward
            patched = _make_bcac_forward(block._forward, self.ctx)
            # Bind as a method so `self_block` resolves to the block instance.
            import types
            block._forward = types.MethodType(patched, block)

    def remove(self) -> None:
        for idx, orig in self._originals.items():
            self.blocks[idx]._forward = orig
        self._originals.clear()


# ---- BCACSampler -----------------------------------------------------


class BCACSampler(FlowEulerGuidanceIntervalSampler):
    """BCAC sampler: cross-attn consensus + self-attn output blending.

    Drop-in replacement for ``VGCFSampler``. The Euler integration loop
    is the same; the difference is that base consistency comes from
    **in-network attention modification** (via monkey-patched blocks)
    rather than post-hoc velocity perturbation.
    """

    def __init__(
        self,
        sigma_min: float,
        # Block selection
        block_start: int = 4,
        block_end: int = 9,      # exclusive → blocks 4,5,6,7,8
        # Time schedule (three-phase)
        t_full: float = 0.7,      # above this: M=1, alpha=1
        t_release: float = 0.3,   # below this: alpha=0
        # Mask
        tau_percentile: float = 0.65,
        active_fraction: float = 0.8,
        eps_log: float = 1e-6,
        eta: float = 0.5,
        # Strength
        alpha_max: float = 1.0,
    ) -> None:
        super().__init__(sigma_min=sigma_min)
        self.block_start = int(block_start)
        self.block_end = int(block_end)
        self.t_full = float(t_full)
        self.t_release = float(t_release)
        self.tau_percentile = float(tau_percentile)
        self.active_fraction = float(active_fraction)
        self.eps_log = float(eps_log)
        self.eta = float(eta)
        self.alpha_max = float(alpha_max)

    def _compute_alpha(self, t: float) -> float:
        if t > self.t_full:
            return self.alpha_max
        if t < self.t_release:
            return 0.0
        # Linear decay from alpha_max to 0 over [t_release, t_full]
        return self.alpha_max * (t - self.t_release) / max(self.t_full - self.t_release, 1e-6)

    def _compute_delta_mask(
        self,
        z_current: torch.Tensor,
        z_previous: torch.Tensor,
        token_spatial_res: int,
    ) -> torch.Tensor:
        """Compute a real-time base/move mask from the cross-state
        variance of the per-step latent delta Δz = z_current - z_previous.

        Core insight (from the user's observation of raw TRELLIS
        step-by-step data): base voxels have similar Δz across K states
        (the model pushes them the same way regardless of condition);
        move voxels have divergent Δz (condition-driven). This signal
        is immune to BCAC's consensus feedback because Δz reflects the
        model's VELOCITY at this step — even if z_previous was
        consensus'd, the velocity re-introduces condition differences.

        Args:
            z_current:  (K, C, D, H, W) — latent AFTER the Euler step.
            z_previous: (K, C, D, H, W) — latent BEFORE the Euler step.
            token_spatial_res: int — 8 for patch_size=2 on 16³.

        Returns:
            (1, N, 1) mask: 1 = base (low delta variance), 0 = move.
        """
        delta = z_current - z_previous                         # (K, C, D, H, W)

        # Cross-state variance of the delta, summed over C channels.
        delta_bar = delta.mean(dim=0, keepdim=True)            # (1, C, D, H, W)
        diff = delta - delta_bar                                # (K, C, D, H, W)
        sigma2 = (diff * diff).sum(dim=1).mean(dim=0)         # (D, H, W)

        # Log-space percentile threshold (same P0 approach).
        log_s2 = torch.log(sigma2 + self.eps_log)
        tau = torch.quantile(log_s2.flatten(), self.tau_percentile)

        M_16 = torch.sigmoid((tau - log_s2) / self.eta)       # (D, H, W)

        # Pool to token resolution: 16³ → 8³ (if patch_size=2).
        pool_k = M_16.shape[0] // token_spatial_res
        if pool_k > 1:
            M_tok = F.avg_pool3d(M_16[None, None], kernel_size=pool_k).squeeze(0).squeeze(0)
        else:
            M_tok = M_16
        return M_tok.reshape(1, -1, 1)                        # (1, N, 1)

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        noise: torch.Tensor,
        cond: torch.Tensor,
        neg_cond: torch.Tensor,
        steps: int = 12,
        rescale_t: float = 1.0,
        cfg_strength: float = 7.5,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs: Any,
    ) -> edict:
        K = noise.shape[0]
        sample = noise.clone()

        t_seq = np.linspace(1.0, 0.0, steps + 1)
        t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
        t_pairs = list(zip(t_seq[:-1], t_seq[1:]))

        # Infer token spatial resolution from the model's config.
        patch_size = getattr(model, "patch_size", 2)
        resolution = getattr(model, "resolution", noise.shape[-1])
        token_spatial_res = resolution // patch_size

        # Prepare anchor conditioning: state-0's DINOv2 features for all K.
        cond_anchor = cond[0:1].expand(K, -1, -1).contiguous()

        # BCAC context + patcher
        ctx = BCACContext(cond_anchor=cond_anchor)
        block_indices = list(range(
            min(self.block_start, len(model.blocks)),
            min(self.block_end, len(model.blocks)),
        ))
        patcher = BCACBlockPatcher(model.blocks, block_indices, ctx)
        patcher.install()

        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": [],
                     "bcac_diagnostics": []})
        z_previous: Optional[torch.Tensor] = None

        try:
            pbar = tqdm(t_pairs, desc="BCAC sampling", disable=not verbose)
            for step_idx, (t, t_prev) in enumerate(pbar):
                t = float(t)
                t_prev = float(t_prev)
                dt = t - t_prev

                alpha = self._compute_alpha(t)

                # BCAC is active whenever α > 0 AND we have a delta mask
                # (i.e. from step 1 onward, since step 0 has no z_previous).
                ctx.active = alpha > 0 and z_previous is not None
                ctx.alpha = alpha

                # If we have z_previous, compute the Δz mask for THIS step.
                if z_previous is not None and alpha > 0:
                    ctx.M_tokens = self._compute_delta_mask(
                        sample, z_previous, token_spatial_res,
                    )
                else:
                    ctx.M_tokens = None

                z_before_step = sample.clone()

                pred_x_0, _pred_eps, pred_v = self._get_model_prediction(
                    model, sample, t,
                    cond=cond, neg_cond=neg_cond,
                    cfg_strength=cfg_strength,
                    cfg_interval=cfg_interval,
                    **kwargs,
                )

                # Euler step.
                pred_x_prev = sample - dt * pred_v
                sample = pred_x_prev

                # Store z_previous for next step's delta computation.
                z_previous = z_before_step

                M_mean = float(ctx.M_tokens.mean().item()) if ctx.M_tokens is not None else 0.0
                diag: Dict[str, Any] = {
                    "step": step_idx,
                    "t": t,
                    "alpha": float(ctx.alpha),
                    "M_mean": M_mean,
                    "active_blocks": list(block_indices) if ctx.active else [],
                    "mask_source": "delta" if ctx.M_tokens is not None else "none",
                }
                ret.pred_x_t.append(pred_x_prev)
                ret.pred_x_0.append(pred_x_0)
                ret.bcac_diagnostics.append(diag)

        finally:
            patcher.remove()

        ret.samples = sample
        return ret

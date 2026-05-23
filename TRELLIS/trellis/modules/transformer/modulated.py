from typing import *
import torch
import torch.nn as nn
from ..attention import MultiHeadAttention
from ..norm import LayerNorm32
from .blocks import FeedForwardNet


class ModulatedTransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedTransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(
        self,
        x: torch.Tensor,
        mod: torch.Tensor,
        context: torch.Tensor,
        block_idx: Optional[int] = None,
        **kwargs,
    ):
        """Block forward with optional Base-Masked Cross-State Attention (BMCSA).

        When ``kwargs.get('bmcsa_flag', False)`` is True AND this block index
        is in ``kwargs['bmcsa_blocks']`` (or ``bmcsa_blocks`` is None meaning
        all blocks), the self-attention output is blended between a per-state
        path (normal self_attn) and a cross-state-shared path (self_attn with
        ``share_kv_across_batch=True``), gated per-token by a mask M.

        Two M-computation modes (controlled by ``M_compute_mode``):

        - ``"static"`` (v4.3 legacy): M = M_base, a precomputed (1, L, 1)
          geometric gate passed in via kwargs. Same M reused at every Euler
          step and every block.

        - ``"dynamic"`` (v3.3 default, method.md section 5.2): M is computed
          per block from the CURRENT modulated hidden state h via
          pairwise cross-state cosine agreement. This eliminates the
          "stale M_base" problem of static mode (M_base from Pass-1 end vs
          Pass-2 evolving hidden) at the cost of K^2 * L * D ops per block.

        kwargs schema (only consulted when bmcsa_flag is True):
            bmcsa_flag : bool
            bmcsa_blocks : Optional[Iterable[int]]  -- None means all
            bmcsa_strength : float, default 1.0
            M_compute_mode : str, "static" or "dynamic", default "static"
            M_base : (1, L, 1) -- required when M_compute_mode == "static"
            tau_M_dynamic : float, default 0.6 -- sigmoid center for dynamic-M
            kappa_M_dynamic : float, default 0.1 -- sigmoid sharpness
            M_attn : Optional (1, L, 1) -- optional semantic gate multiplier
            dynamic_M_log : Optional[Dict[int, torch.Tensor]] -- when given
                            (and dynamic mode), each visited block writes its
                            M into log[block_idx] for diagnostics / viz.

        When bmcsa_flag is False or this block is not in bmcsa_blocks, the
        forward is identical to the unmodified TRELLIS block.
        """
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        # --- BMCSA self-attention branch ---
        bmcsa_flag = bool(kwargs.get("bmcsa_flag", False))
        bmcsa_blocks = kwargs.get("bmcsa_blocks", None)
        bmcsa_active = bmcsa_flag and (bmcsa_blocks is None or (block_idx is not None and block_idx in bmcsa_blocks))
        if bmcsa_active and h.shape[0] > 1:
            y_self = self.self_attn(h)
            y_shared = self.self_attn(h, share_kv_across_batch=True)

            M_mode = str(kwargs.get("M_compute_mode", "static")).lower()

            if M_mode == "static":
                if "M_base" not in kwargs:
                    raise KeyError("BMCSA static mode requires kwargs['M_base'] (1, L, 1)")
                M = kwargs["M_base"].to(y_self.dtype)
            elif M_mode == "dynamic":
                # Compute M per block from CURRENT modulated hidden h.
                # h: (K, L, D). Pairwise cross-state cosine per token.
                # Defaults from method.md section 5.2 (tau_M=0.7, kappa_M=0.05).
                K_b, L_b, D_b = h.shape
                tau_M = float(kwargs.get("tau_M_dynamic", 0.7))
                kappa_M = float(kwargs.get("kappa_M_dynamic", 0.05))

                # L2-normalize over channel; fp32 cosine for numerical stability.
                h_fp32 = h.to(torch.float32)
                h_normed = h_fp32 / (h_fp32.norm(dim=-1, keepdim=True) + 1e-6)
                # Pairwise: (K, K, L)
                pairwise = torch.einsum("kld,jld->kjl", h_normed, h_normed)
                eye_K = torch.eye(K_b, device=h.device, dtype=torch.bool)
                pairwise = pairwise.masked_fill(eye_K.unsqueeze(-1), 0.0)
                # Off-diagonal mean per token (denominator K*(K-1) since diagonal zeroed)
                if K_b > 1:
                    agree = pairwise.sum(dim=(0, 1)) / (K_b * (K_b - 1))    # (L,)
                else:
                    agree = pairwise.sum(dim=(0, 1))                        # (L,) zeros

                M_dyn = torch.sigmoid((agree - tau_M) / max(kappa_M, 1e-8))  # (L,)
                M = M_dyn.view(1, -1, 1).to(y_self.dtype)

                # Optional logging hook for downstream viz / debug.
                dyn_M_log = kwargs.get("dynamic_M_log", None)
                if dyn_M_log is not None and block_idx is not None:
                    # Store fp16 CPU copy per block (overwrites previous step's
                    # entry, which is the convention: caller resets the dict
                    # before each sample_once call if per-step history is wanted).
                    dyn_M_log[int(block_idx)] = M_dyn.detach().to(torch.float16).cpu()
            else:
                raise ValueError(
                    f"BMCSA M_compute_mode must be 'static' or 'dynamic'; got {M_mode!r}"
                )

            # Optional semantic gate (v4.2: M_effective = M_geom * M_attn).
            M_attn_arg = kwargs.get("M_attn", None)
            if M_attn_arg is not None:
                M_attn_t = M_attn_arg.to(y_self.dtype)
                M = M * M_attn_t

            strength = float(kwargs.get("bmcsa_strength", 1.0))
            eff_M = torch.clamp(strength * M, 0.0, 1.0)
            h = (1.0 - eff_M) * y_self + eff_M * y_shared
        else:
            h = self.self_attn(h)
        # ------------------------------------

        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(
        self,
        x: torch.Tensor,
        mod: torch.Tensor,
        context: torch.Tensor,
        block_idx: Optional[int] = None,
        **kwargs,
    ):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, context, block_idx, use_reentrant=False, **kwargs,
            )
        else:
            return self._forward(x, mod, context, block_idx=block_idx, **kwargs)
        
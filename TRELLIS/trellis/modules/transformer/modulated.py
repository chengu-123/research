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

        # --- BMCSA self-attention branch (v3.3.4 motion-aware composition) ---
        # v3.3.4 design (replaces v3.3.2 multiplicative multi-gate + v3.3.3
        # single-gate saturated dynamic):
        #
        #   y_avg   = self_attn(h, share_kv_across_batch=True)   # K/V mean
        #   y_self  = self_attn(h)                                # per-state
        #
        #   M_move = clamp(
        #       max(M_motion_static, M_motion_dynamic) * (1 - M_base_geom),
        #       0, 1 - kv_floor
        #   )
        #   h = M_move * y_self + (1 - M_move) * y_avg
        #
        # SEMANTIC SHIFT: instead of computing M_eff = "K/V-share strength"
        # via multiplicative gate product (v3.3.2) which catastrophically
        # attenuated base edges, we compute M_move = "per-state preservation
        # strength" via two OR-composed motion signals and use M_base as a
        # SAFETY FLOOR (not a multiplicative term in the move-promoting chain).
        #
        # WHY THIS DESIGN (the discovery from v3.3.2 s2/s4 inspection):
        # In motion-corridor voxels, K/V averaging across K=6 states forces
        # the model to find a "common consensus" geometry. But TRELLIS was
        # trained on Objaverse static objects, so it doesn't encode
        # articulated-object physical constraints (e.g., a drawer should
        # sit ~1 voxel above the cabinet floor, not coincident with it).
        # When K/V averaged, the per-state image conds (each correctly
        # encoding the drawer's elevated position via DINOv2 patches)
        # get blended into a "lowest common denominator" geometry where
        # drawer sits on the floor. Single-state self_attn at corridor
        # voxels preserves each state's image-cond signal -> the model
        # outputs physically correct drawer placement.
        #
        # Two MOTION signals (OR composition for conservative preservation):
        #   M_motion_static  : Pass-1 footprint*(1-shared), passed via kwarg
        #                      M_motion_corridor. High at swept-volume voxels.
        #   M_motion_dynamic : per-block per-step from variance-based
        #                      cross-state divergence of prev_x_0_pred
        #                      (F1 fix, see Gate 4 below). High at voxels
        #                      where K-state Tweedie estimates disagree NOW.
        # Composed: M_move_raw = max(M_motion_static, M_motion_dynamic).
        #
        # SAFETY FLOOR via M_base_geom (Pass-1 P_base derived):
        #   M_move = M_move_raw * (1 - M_base_geom)
        # At confident base voxels (M_base_geom ~ 1), this forces M_move -> 0
        # regardless of (possibly spurious) motion signals. Specifically
        # protects base BACK FACE voxels (where v3.3.2's multi-gate had
        # M_attn low -> over-attenuation -> lost base voxels). M_attn is
        # NOT used in BMCSA composition; it stays at guide stage only
        # (v4.3 / v3.3.3 behavior).
        #
        # K-redundancy FLOOR (s3-loss mitigation):
        #   M_move = clamp(M_move, 0, 1 - kv_floor)
        # v3.3.2 had per-state branches that gave each state ~100% freedom
        # in corridor; single-state TRELLIS instability (e.g. s3 corruption)
        # had no K-redundancy to fall back on. kv_floor=0.2 default forces
        # all voxels to retain at least 20% K/V averaging contribution,
        # giving K-redundancy safety against single-state failures while
        # still preserving most of the per-state physical detail.
        #
        # Kwargs (all optional, missing defaults to safe identity):
        #   M_motion_corridor : (1, L, 1) static motion gate from Pass-1
        #   M_base            : (1, L, 1) static base gate for safety floor
        #   M_compute_mode    : 'static' | 'dynamic' (whether to compute M_dyn)
        #   M_dynamic_signal  : 'variance' (F1, default) | 'cosine' (legacy)
        #   var_percentile, var_eta : variance-path sigmoid params
        #   prev_x_0_pred     : (K, C, D, H, W) for dynamic computation
        #   kv_floor          : K-redundancy floor (default 0.2)
        #   bmcsa_strength    : overall move strength multiplier (default 1.0)
        #   M_attn            : (legacy) if provided, applied as soft floor
        #                      reducing M_move where semantic agreement is high
        #                      (off by default; M_attn typically at guide stage)
        #   dynamic_M_log     : optional dict to record per-block M_move
        bmcsa_flag = bool(kwargs.get("bmcsa_flag", False))
        bmcsa_blocks = kwargs.get("bmcsa_blocks", None)
        bmcsa_active = bmcsa_flag and (bmcsa_blocks is None or (block_idx is not None and block_idx in bmcsa_blocks))
        if bmcsa_active and h.shape[0] > 1:
            y_self = self.self_attn(h)
            y_shared = self.self_attn(h, share_kv_across_batch=True)
            K_b, L_b, D_b = h.shape
            device_M = h.device
            dtype_M = y_self.dtype

            # ---- Gate 1: M_base_geom (static, Pass-1 P_base derived) ----
            M_base_geom = kwargs.get("M_base", None)

            # ---- Gate 2: M_attn (static, semantic agreement on z_final) ----
            M_attn_t = kwargs.get("M_attn", None)

            # ---- Gate 3: NOT motion corridor (static, swept volume) ----
            M_motion = kwargs.get("M_motion_corridor", None)

            # ---- Gate 4: M_dynamic (optional attenuator) ----
            # Computed only when prev_x_0_pred (8-channel) is provided.
            #
            # v3.3.4 (root-cause fix for stageB.md Problem 2 saturation):
            # Old signal source was per-token cosine of L2-normalised 8-channel
            # prev_x_0_pred. L2-normalise discards the latent MAGNITUDE which
            # carries the strongest base/move occupancy discriminator (an
            # occupied voxel's SS latent norm is much larger than an empty
            # voxel's). At intermediate Pass-2 t in (0, 0.5], shared-noise
            # K-parallel Tweedie estimates are heavily correlated DIRECTIONALLY
            # (all K pulled toward the same DINOv2-conditioned manifold),
            # so cos saturates to >0.99 across all tokens regardless of
            # whether the voxel is base or move (empirically verified:
            # v3.3.2 (S=12, B=24) M_scalar all 0.998).
            #
            # New signal: cross-state VARIANCE in the un-normalised 8-channel
            # space. Move voxels have high cross-state variance (different
            # occupancy across K), base voxels have low variance (consistent
            # occupancy). Identical math to SCAR.compute_tweedie_variance_mask
            # (already empirically validated on Pass-1 mask) but applied per
            # block, per step.
            #
            # New kwargs:
            #   M_dynamic_signal: 'variance' (default, fixed) | 'cosine' (legacy ablation)
            #   var_percentile : in-batch log-variance quantile for sigmoid center
            #   var_eta        : sigmoid sharpness in log-variance space
            # tau_M_dynamic / kappa_M_dynamic remain effective only when
            # M_dynamic_signal='cosine' (legacy path, kept for ablation).
            M_dyn = None
            M_mode = str(kwargs.get("M_compute_mode", "dynamic")).lower()
            M_signal = str(kwargs.get("M_dynamic_signal", "variance")).lower()
            prev_x0 = kwargs.get("prev_x_0_pred", None)
            if M_mode == "dynamic" and prev_x0 is not None:
                # prev_x0: (K, C, D, H, W) where C=8 for TRELLIS SS latent.
                # Token order matches DiT patchify (row-major over D,H,W).
                K_x, C_x, D_x, H_x, W_x = prev_x0.shape
                if K_x != K_b:
                    raise ValueError(
                        f"prev_x_0_pred K={K_x} mismatches hidden batch K={K_b}"
                    )
                # Reshape to (K, L, C) per-token view, fp32 for numeric safety
                x_tok = prev_x0.view(K_x, C_x, -1).permute(0, 2, 1).to(torch.float32)  # (K, L, C)
                L_x = x_tok.shape[1]
                if L_x != L_b:
                    raise ValueError(
                        f"prev_x_0_pred token count {L_x} mismatches "
                        f"hidden token count {L_b}; ensure x_0 resolution matches DiT"
                    )
                if M_signal == "variance":
                    # Cross-state un-normalised variance per token.
                    # Magnitude IS the signal: empty vs occupied voxels differ
                    # primarily in latent norm, which L2-normalise discards.
                    x_mean = x_tok.mean(dim=0, keepdim=True)                              # (1, L, C)
                    div_per_token = (x_tok - x_mean).pow(2).sum(dim=(0, -1)) / max(K_x, 1)   # (L,)
                    log_div = torch.log(div_per_token + 1e-6)
                    # In-batch quantile -> auto-adaptive per sample (no magic threshold).
                    var_pct = float(kwargs.get("var_percentile", 0.65))
                    var_eta = float(kwargs.get("var_eta", 0.5))
                    tau_var = torch.quantile(log_div, var_pct)
                    # High M_dyn at LOW variance -> base; low at HIGH variance -> move.
                    M_dyn_flat = torch.sigmoid((tau_var - log_div) / max(var_eta, 1e-8))    # (L,)
                elif M_signal == "cosine":
                    # Legacy v3.3.x path: pairwise cosine on L2-normalised 8-d.
                    # Saturates to ~1 in production; kept ONLY for ablation.
                    x_norm = torch.nn.functional.normalize(x_tok, dim=-1, eps=1e-6)        # (K, L, C)
                    pairwise = torch.einsum("klc,jlc->kjl", x_norm, x_norm)                # (K, K, L)
                    eye_K = torch.eye(K_x, device=device_M, dtype=torch.bool)
                    pairwise = pairwise.masked_fill(eye_K.unsqueeze(-1), 0.0)
                    if K_x > 1:
                        agree = pairwise.sum(dim=(0, 1)) / (K_x * (K_x - 1))               # (L,)
                    else:
                        agree = pairwise.sum(dim=(0, 1))
                    tau_M = float(kwargs.get("tau_M_dynamic", 0.7))
                    kappa_M = float(kwargs.get("kappa_M_dynamic", 0.05))
                    M_dyn_flat = torch.sigmoid((agree - tau_M) / max(kappa_M, 1e-8))       # (L,)
                else:
                    raise ValueError(
                        f"M_dynamic_signal must be 'variance' or 'cosine'; got {M_signal!r}"
                    )
                M_dyn = M_dyn_flat.view(1, -1, 1).to(dtype_M)
            elif M_mode == "dynamic" and prev_x0 is None:
                # Dynamic mode requested but no prev_x_0_pred provided
                # (e.g. first step). Attenuator defaults to 1 -> no effect.
                M_dyn = None
            elif M_mode == "static":
                M_dyn = None    # static: no dynamic attenuator
            else:
                raise ValueError(
                    f"BMCSA M_compute_mode must be 'static' or 'dynamic'; got {M_mode!r}"
                )

            # ---- v3.3.6 FINAL composition (Stage B locked design) ----
            # After 4 iterations (v3.3.2 / v3.3.3 / v3.3.4 / v3.3.5), the empirically
            # validated default is v3.3.4-style motion-aware safety floor:
            #
            #   M_motion_combined = max(M_motion_static, M_motion_dynamic)
            #   M_move = M_motion_combined * (1 - M_base_geom)
            #
            # K-redundancy floor kv_floor=0.2 applied below caps M_move at 0.8.
            #
            # ABLATION SWITCHES (via kwargs.get('bmcsa_composition', 'v334')):
            #   'v334' (default): the formula above. Empirically validated as
            #          producing the cleanest base extraction + base/move
            #          separation across 30857.
            #   'v335': default-per-state with K-share only at confident base:
            #          M_move = 1 - alpha * M_base * (1 - M_motion_combined)
            #          Reverse-engineered from v3.3.2's accidental dynamic-M
            #          cosine saturation. Tested on 30857: no visible
            #          improvement vs v334 (drawer-above-cabinet physical
            #          effect was not reproduced). Kept for AAAI ablation.
            #
            # Inputs:
            #   M_motion: static motion corridor (Pass-1 derived, kwarg)
            #             Source can be 'occupancy_only' (default) or 'enhanced'
            #             (multi-source OR — Plan C+, ablation only)
            #   M_dyn: dynamic motion signal (F1 variance, per-step per-block)
            #          shared across blocks within a step
            #   M_base_geom: Pass-1 P_base_shared derived static prior
            #   M_attn_t: deliberately NOT used in BMCSA composition (lives at
            #             guide stage; v3.3.2 had base back-face attenuation
            #             when M_attn entered as multiplicative gate)
            gate_contributions = {}
            composition_mode = str(kwargs.get("bmcsa_composition", "v334")).lower()

            # ---- Step 1: OR-compose motion signals (shared by both modes) ----
            M_motion_combined = torch.zeros(1, L_b, 1, device=device_M, dtype=dtype_M)
            if M_motion is not None:
                m_static = M_motion.to(device=device_M, dtype=dtype_M).clamp(0.0, 1.0)
                M_motion_combined = torch.maximum(M_motion_combined, m_static)
                gate_contributions["M_motion_static"] = m_static
            if M_dyn is not None and M_signal == "variance":
                # M_dyn high = K consensus = base-like; invert for motion-like.
                m_dyn_motion = (1.0 - M_dyn).clamp(0.0, 1.0)
                M_motion_combined = torch.maximum(M_motion_combined, m_dyn_motion)
                gate_contributions["M_motion_dynamic"] = m_dyn_motion
            elif M_dyn is not None and M_signal == "cosine":
                # Legacy cosine path saturates to 1; do not use for motion derivation.
                gate_contributions["M_dynamic_cosine_legacy"] = M_dyn

            # ---- Step 2: composition selection ----
            if composition_mode == "v334":
                # v3.3.4 motion-aware safety floor (default)
                # M_move = motion_combined * (1 - M_base_geom)
                M_move_raw = M_motion_combined.clone()
                if M_base_geom is not None:
                    m_base = M_base_geom.to(device=device_M, dtype=dtype_M).clamp(0.0, 1.0)
                    M_move_raw = M_move_raw * (1.0 - m_base)
                    gate_contributions["M_base_floor"] = m_base
                # M_attn optional extra base safety (off by default)
                if M_attn_t is not None:
                    m_attn = M_attn_t.to(device=device_M, dtype=dtype_M).clamp(0.0, 1.0)
                    M_move_raw = M_move_raw * (1.0 - m_attn)
                    gate_contributions["M_attn_floor"] = m_attn
            elif composition_mode == "v335":
                # v3.3.5 default-per-state composition (ablation)
                # M_move = 1 - alpha * M_base * (1 - motion_combined)
                alpha = float(kwargs.get("base_kshare_alpha", 0.9))
                if M_base_geom is not None:
                    m_base = M_base_geom.to(device=device_M, dtype=dtype_M).clamp(0.0, 1.0)
                    P_definite_base = m_base * (1.0 - M_motion_combined)
                    M_move_raw = (1.0 - alpha * P_definite_base).clamp(0.0, 1.0)
                    gate_contributions["M_base_geom"] = m_base
                    gate_contributions["P_definite_base"] = P_definite_base
                else:
                    M_move_raw = torch.ones(1, L_b, 1, device=device_M, dtype=dtype_M)
                if M_attn_t is not None:
                    m_attn = M_attn_t.to(device=device_M, dtype=dtype_M).clamp(0.0, 1.0)
                    P_extra = m_attn * (
                        M_base_geom.to(device_M, dtype_M)
                        if M_base_geom is not None else 1.0
                    ) * (1.0 - M_motion_combined)
                    M_move_raw = torch.minimum(
                        M_move_raw, (1.0 - alpha * P_extra).clamp(0.0, 1.0),
                    )
                    gate_contributions["M_attn_extra"] = m_attn
            else:
                raise ValueError(
                    f"bmcsa_composition must be 'v334' or 'v335'; got {composition_mode!r}"
                )

            # ---- Step 4: K-redundancy floor (s3-loss mitigation) ----
            # v3.3.2's per-state branches in corridor had no K backup.
            # kv_floor keeps a minimum K/V averaging share so per-state DiT
            # instability (single-state corruption) is dampened by 5 sibling
            # states. Default 0.2 = 20% K-averaging always present.
            kv_floor = float(kwargs.get("kv_floor", 0.2))
            kv_floor = max(0.0, min(1.0, kv_floor))
            strength = float(kwargs.get("bmcsa_strength", 1.0))
            M_move = torch.clamp(strength * M_move_raw, 0.0, max(1.0 - kv_floor, 1e-6))

            # ---- Optional logging of per-block M_move for viz ----
            dyn_M_log = kwargs.get("dynamic_M_log", None)
            if dyn_M_log is not None and block_idx is not None:
                # Store fp16 CPU copy of FINAL M_move per block.
                dyn_M_log[int(block_idx)] = M_move.squeeze().detach().to(torch.float16).cpu()

            # ---- Final blend: motion-aware decisive composition ----
            # M_move high  -> per-state self_attn (physical detail preserved)
            # M_move low   -> K/V averaged (base consensus)
            h = M_move * y_self + (1.0 - M_move) * y_shared
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
        
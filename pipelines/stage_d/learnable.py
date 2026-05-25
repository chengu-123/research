"""Stage D learnable parameters.

All P1-stage learnable variables in one ``nn.Module`` so the optimizer sees
a clean parameter group structure and gradient checkpointing wraps a single
forward.

Convention:
  - Scalars and per-voxel logits get smaller LR (``lr_scalar``).
  - Adapter MLPs (inserted inside SS-DiT) get ``lr_adapter``.
  - Output heads (H_sup / H_part / H_joint) get ``lr_head``.
  - psi_param's seven content slots (axis 3, origin 3, type_logit 1) come
    from ``encode_joint(psi_0)``; the two ``softplus`` limit slots come
    from ``inverse_softplus(pi/2)`` and ``inverse_softplus(0.3)``.
  - ``psi_ema_buf`` is a non-learnable buffer used by ``stage_detach`` for
    the joint variable EMA during the gradient ramp-in (5%-15% of training).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .config import (
    PSI_PARAM_DIM,
    StageDConfig,
    TRELLIS_LATENT_GRID,
    TRELLIS_SS_IN_CH,
    TRELLIS_SS_MODEL_CH,
)


def _inverse_softplus(y: float) -> float:
    """Numerically stable inverse of ``softplus(x) = ln(1 + exp(x))``."""
    if y <= 0.0:
        raise ValueError(f"softplus output must be > 0, got {y}")
    # x = ln(exp(y) - 1); for large y this is approximately y.
    if y > 20.0:
        return y
    return math.log(math.expm1(y))


class ZeroInitAdapter(nn.Module):
    """Tiny residual MLP inserted after a SS-DiT block.

    Output projection is zero-initialized so that at iter 0 the wrapped
    SS-DiT is *exactly* the frozen TRELLIS backbone (the W-RFSDS gradient
    starts pushing the adapter outputs away from zero).

    Forward:
        h_in:  [B, L, C=1024]
        out:   [B, L, C]      (added back as residual by SS_DiT_WithAdapters)
    """

    def __init__(self, model_channels: int = TRELLIS_SS_MODEL_CH,
                 hidden_dim: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(model_channels)
        self.fc1 = nn.Linear(model_channels, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, model_channels, bias=True)

        # Standard init for fc1; zero init for fc2 (output projection).
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(h))))


class ZeroInitMLP(nn.Module):
    """Zero-init residual MLP used for H_sup / H_part / H_joint output heads.

    Final linear is zero-initialized so heads contribute 0 at iter 0
    (lambda ramp-ins in schedules.py also start at 0, providing extra safety).
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, out_dim, bias=True)
        self.act = nn.GELU()

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        return self.fc3(h)


def encode_joint_init(psi_0: Dict[str, torch.Tensor],
                      device: torch.device,
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Pack ``psi_0`` from Bootstrap into the flat ``psi_param`` of length 19.

    Bootstrap's ``stage_c_joint_init`` returns a dict with at least:
      axis        [3] unit-norm vector (world space)
      origin      [3] world-space anchor point in [-0.5, 0.5]
      type_logit  scalar (signed; sigmoid -> prismatic probability)
    The two ``*_limit_raw`` slots are pre-image of softplus so that
    ``softplus(theta_limit_raw) == pi/2`` and ``softplus(disp_limit_raw) == 0.3``
    at init.
    """
    axis = psi_0["axis"].to(device=device, dtype=dtype)
    origin = psi_0["origin"].to(device=device, dtype=dtype)
    type_logit = psi_0["type_logit"].to(device=device, dtype=dtype).reshape(())

    psi = torch.zeros(PSI_PARAM_DIM, device=device, dtype=dtype)
    psi[0:3] = axis
    psi[3:6] = origin
    psi[6] = type_logit
    psi[7] = float(_inverse_softplus(math.pi / 2.0))
    psi[8] = float(_inverse_softplus(0.3))
    # psi[9:19] left at zero (reserved for future use).
    return psi


def encode_delta_phi_init(phi_0: torch.Tensor,
                          device: torch.device,
                          dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Initialize ``delta_phi`` from Bootstrap's shifted ``phi_0``.

    phi_0 is a length-6 tensor already shifted so phi_0[CANONICAL_STATE_IDX]=0
    (see method.md NEW.1 and pipeline.md B7). Since delta_phi is the
    *increment* sequence (length 5) and shifts do not change differences,
    the encoding is independent of the canonical-state choice.

    Returns
    -------
    delta_phi : torch.Tensor [5]   inverse_softplus of phi_0 increments
    """
    if phi_0.shape != (6,):
        raise ValueError(f"phi_0 must be [6], got {tuple(phi_0.shape)}")

    diffs = (phi_0[1:] - phi_0[:-1]).to(device=device, dtype=dtype)
    # Bootstrap monotone phi yields positive diffs; clamp for safety
    # (joint_init may emit numerically tiny negatives near zero crossings).
    diffs = diffs.clamp_min(1.0e-4)
    # inverse_softplus: ln(exp(y) - 1) = ln(expm1(y))
    return torch.log(torch.expm1(diffs))


def init_alpha_m_from_M_attn(M_attn_at_U: torch.Tensor) -> torch.Tensor:
    """Initialize per-voxel move-gate logit from BMCSA cross-state agreement.

    M_attn high  -> base-like -> alpha_m should be low (logit negative)
    M_attn low   -> move-like -> alpha_m should be high (logit positive)

    We sample M_attn at U_object voxel coords (already done outside) and
    map (clamped) [0.05, 0.95] to ``logit(1 - M)``: M=0 -> logit(1) = +inf,
    M=1 -> logit(0) = -inf, M=0.5 -> 0. The 0.05/0.95 clamp keeps it finite.
    """
    if M_attn_at_U.ndim != 1:
        raise ValueError(f"M_attn_at_U must be [N], got {tuple(M_attn_at_U.shape)}")
    m_clamped = M_attn_at_U.clamp(0.05, 0.95)
    # alpha_m = logit(1 - M_attn) = log((1-M) / M)
    return torch.log((1.0 - m_clamped) / m_clamped)


class StageDLearnable(nn.Module):
    """Container for every learnable parameter in Stage D P1.

    The actual computation flow is in ``train.py`` (orchestration) and
    ``ss_dit_wrapper.py`` / ``joint_ops.py`` / ``render.py`` (sub-steps).
    This module only holds parameters and exposes them via attribute access.
    """

    def __init__(self,
                 cfg: StageDConfig,
                 n_obj: int,
                 psi_0: Dict[str, torch.Tensor],
                 phi_0: torch.Tensor,
                 M_attn_at_U: torch.Tensor,
                 device: torch.device,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_obj = int(n_obj)

        # ---- z_s_base residual (broadcast-friendly shape) ----
        # SS-DiT input shape is [B=1, 8, 16, 16, 16]. We store as [8, 16, 16, 16]
        # and unsqueeze(0) at forward time.
        self.Delta_z_s = nn.Parameter(
            torch.zeros(TRELLIS_SS_IN_CH, TRELLIS_LATENT_GRID,
                        TRELLIS_LATENT_GRID, TRELLIS_LATENT_GRID,
                        device=device, dtype=dtype)
        )

        # ---- Per-voxel gate logits ----
        # alpha_g zero-init: gate logit reduces to occ_at_U + lambda_sup * H_sup.
        self.alpha_g = nn.Parameter(
            torch.zeros(n_obj, device=device, dtype=dtype)
        )
        # alpha_m: BCE-anchored to BMCSA's M_attn in warmup_g0; logit(1 - M_clamped).
        self.alpha_m = nn.Parameter(
            init_alpha_m_from_M_attn(M_attn_at_U).to(device=device, dtype=dtype)
        )

        # ---- Joint parameters ----
        self.psi_param = nn.Parameter(
            encode_joint_init(psi_0, device=device, dtype=dtype)
        )
        self.delta_phi = nn.Parameter(
            encode_delta_phi_init(phi_0, device=device, dtype=dtype)
        )

        # ---- Adapters (one per insertion block) ----
        self.adapters = nn.ModuleDict({
            str(block_idx): ZeroInitAdapter(
                model_channels=TRELLIS_SS_MODEL_CH,
                hidden_dim=cfg.adapter_hidden_dim,
            )
            for block_idx in cfg.adapter_blocks
        })

        # ---- Output heads ----
        # H_sup, H_part: per-voxel scalar logit residual.
        # H_joint: 19-dim residual on psi_param (same layout).
        self.H_sup = ZeroInitMLP(
            in_dim=cfg.feat_dim, hidden_dim=cfg.head_hidden_dim, out_dim=1
        )
        self.H_part = ZeroInitMLP(
            in_dim=cfg.feat_dim, hidden_dim=cfg.head_hidden_dim, out_dim=1
        )
        self.H_joint = ZeroInitMLP(
            in_dim=cfg.feat_dim, hidden_dim=cfg.head_hidden_dim, out_dim=PSI_PARAM_DIM
        )

        # ---- Non-learnable EMA buffer for psi gradient ramp-in ----
        # Used by stage_detach(..., mode="joint"): the EMA tracks psi during
        # iters 5%-15%, providing a slow signal so heads can start outputting
        # meaningful deltas before the joint variable itself becomes free.
        self.register_buffer(
            "psi_ema_buf",
            self.psi_param.detach().clone(),
            persistent=False,
        )

    # ------------------------------------------------------------------ #
    # Parameter group builder for the AdamW optimizer.                    #
    # ------------------------------------------------------------------ #
    def make_param_groups(self) -> List[Dict[str, object]]:
        """Group parameters by category so AdamW can use category-specific LR.

        Categories:
          - "adapter":  the three adapter MLPs
          - "head":     H_sup, H_part, H_joint
          - "scalar":   Delta_z_s, alpha_g, alpha_m, psi_param, delta_phi
        """
        adapter_params: List[nn.Parameter] = []
        for adapter in self.adapters.values():
            adapter_params.extend(adapter.parameters())

        head_params: List[nn.Parameter] = (
            list(self.H_sup.parameters())
            + list(self.H_part.parameters())
            + list(self.H_joint.parameters())
        )

        scalar_params: List[nn.Parameter] = [
            self.Delta_z_s, self.alpha_g, self.alpha_m,
            self.psi_param, self.delta_phi,
        ]

        return [
            {"params": adapter_params, "lr": self.cfg.lr_adapter,
             "weight_decay": self.cfg.weight_decay, "name": "adapter"},
            {"params": head_params, "lr": self.cfg.lr_head,
             "weight_decay": self.cfg.weight_decay, "name": "head"},
            {"params": scalar_params, "lr": self.cfg.lr_scalar,
             "weight_decay": self.cfg.weight_decay, "name": "scalar"},
        ]


@dataclass
class TypeVoteCloneState:
    """Snapshot used during S3 dual-clone resolution.

    When P1 ends Main G1a (~50% iters) and the type vote returns confidence
    below threshold, train.py clones the entire learnable state into two
    copies (rev / pri) with frozen type_logit, runs each for the remainder
    of Main G1b, then commits whichever has lower final SDS + RGB loss.
    """
    learnable_rev: StageDLearnable
    learnable_pri: StageDLearnable
    forced_type_logit_rev: float = -10.0   # sigmoid -> ~0 (revolute)
    forced_type_logit_pri: float = +10.0   # sigmoid -> ~1 (prismatic)


__all__ = [
    "ZeroInitAdapter", "ZeroInitMLP",
    "encode_joint_init", "encode_delta_phi_init", "init_alpha_m_from_M_attn",
    "StageDLearnable", "TypeVoteCloneState",
]

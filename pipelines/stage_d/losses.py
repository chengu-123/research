"""Stage D loss components and aggregator.

method.md section 9.1 enumerates the full loss list:

    L_total = lambda_sds   * L_sds        (W-RFSDS through Wan2.2)
            + lambda_lat   * L_lat_rec    (auxiliary, off by default; C2)
            + lambda_rgb   * L_rgb_rec    (L1 + LPIPS vs Wan video target)
            + lambda_first * L_first      (frame 0 vs no-carpet s_0_pure)
            + lambda_last  * L_last       (frame F-1 vs no-carpet s_5_pure)
            + lambda_contact * L_contact  (axis-through-anchor-band prior)
            + lambda_gate  * L_gate       (encourages g, m -> {0, 1})
            + lambda_shell * L_shell      (sparsity on uncertain shell voxels)
            + lambda_m_prior * L_m_prior  (BCE alpha_m vs (1 - M_attn))
            + lambda_z     * L_z          (Delta_z_s L2 stability)

This module:
  - implements each individual component
  - provides ``LPIPSModule`` (a thin holder so we don't keep a global lpips
    instance, but only construct it once per Stage D run)
  - exposes ``aggregate_loss`` which composes all components according to
    the current phase / schedule.

The LPIPS package expects ``[N, 3, H, W]`` inputs in ``[-1, 1]``; we
rescale from our ``[0, 1]`` render output inline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F

from .w_rfsds import WanRFSDSContext, latent_recon_loss, w_rfsds_loss


# =============================================================================
# LPIPS holder (loaded once)
# =============================================================================

class LPIPSModule(nn.Module):
    """Wraps ``lpips.LPIPS(net=net_type)`` and exposes a fp32 forward.

    Stored as a non-trainable module: ``requires_grad_ = False`` on all
    params (lpips weights are frozen anyway, but we make it explicit).

    Forward expects ``[N, 3, H, W]`` in ``[0, 1]``; we rescale to ``[-1, 1]``
    internally to match lpips's convention.
    """

    def __init__(self, net_type: str = "vgg") -> None:
        super().__init__()
        self.lpips_net = lpips.LPIPS(net=net_type, verbose=False)
        for p in self.lpips_net.parameters():
            p.requires_grad_(False)
        self.lpips_net.eval()

    def forward(self, pred_01: torch.Tensor, target_01: torch.Tensor) -> torch.Tensor:
        """Mean LPIPS distance over the batch.

        Parameters
        ----------
        pred_01, target_01 : Tensor [N, 3, H, W] in [0, 1]

        Returns
        -------
        loss : Tensor scalar
        """
        if pred_01.shape != target_01.shape:
            raise ValueError(
                f"LPIPS shape mismatch: pred={tuple(pred_01.shape)}, "
                f"target={tuple(target_01.shape)}"
            )
        if pred_01.ndim != 4 or pred_01.shape[1] != 3:
            raise ValueError(
                f"LPIPS expects [N, 3, H, W]; got {tuple(pred_01.shape)}"
            )
        pred_neg11 = pred_01 * 2.0 - 1.0
        target_neg11 = target_01 * 2.0 - 1.0
        return self.lpips_net(pred_neg11, target_neg11).flatten().mean()


# =============================================================================
# Individual loss components
# =============================================================================

def loss_rgb_recon(
    rgb_T3HW: torch.Tensor,
    wan_target_T3HW: torch.Tensor,
    lpips_module: LPIPSModule,
    lpips_chunk: int = 7,
) -> torch.Tensor:
    """L1 + LPIPS reconstruction loss against the Wan video target.

    Parameters
    ----------
    rgb_T3HW : Tensor [F, 3, H, W] in [0, 1]
    wan_target_T3HW : Tensor [F, 3, H, W] in [0, 1]
    lpips_chunk : int
        Sub-batch size used to chunk LPIPS to avoid the VGG backbone going
        OOM on a 21-frame batch. 7 means 3 sub-batches of 7 each.

    Returns
    -------
    loss : Tensor scalar
    """
    if rgb_T3HW.shape != wan_target_T3HW.shape:
        raise ValueError(
            f"rgb / target shape mismatch: rgb={tuple(rgb_T3HW.shape)}, "
            f"target={tuple(wan_target_T3HW.shape)}"
        )
    l1 = F.l1_loss(rgb_T3HW, wan_target_T3HW)

    F_total = rgb_T3HW.shape[0]
    lpips_terms: List[torch.Tensor] = []
    for start in range(0, F_total, lpips_chunk):
        stop = min(start + lpips_chunk, F_total)
        lpips_terms.append(
            lpips_module(rgb_T3HW[start:stop], wan_target_T3HW[start:stop])
        )
    lp = torch.stack(lpips_terms).mean()
    return l1 + lp


def loss_first_frame_anchor(
    rgb_T3HW: torch.Tensor,
    s_0_pure_3HW: torch.Tensor,
    lpips_module: LPIPSModule,
) -> torch.Tensor:
    """Frame 0 of rendered video vs the no-carpet ``s_0_pure`` reference.

    With NEW.1 canonical-state shift (c=2), rendering frame 0 means the
    canonical asset is *back-warped* by SE(3)(phi[0]) (negative phi for
    revolute, negative-direction translation for prismatic) to reach the
    closed pose. This loss is the direct anchor to the frame-0 observation in
    the same H/W and color space as the Wan video target, so it is critical
    for resolving the canonical-position ambiguity.

    Parameters
    ----------
    rgb_T3HW : Tensor [F, 3, H, W] in [0, 1]
    s_0_pure_3HW : Tensor [3, H, W] in [0, 1]
    """
    if rgb_T3HW.ndim != 4 or rgb_T3HW.shape[1] != 3:
        raise ValueError(
            f"rgb_T3HW must be [F, 3, H, W]; got {tuple(rgb_T3HW.shape)}"
        )
    expected_s0 = (3, int(rgb_T3HW.shape[2]), int(rgb_T3HW.shape[3]))
    if s_0_pure_3HW.shape != expected_s0:
        raise ValueError(
            f"s_0_pure must be {expected_s0}; "
            f"got {tuple(s_0_pure_3HW.shape)}"
        )
    frame_0 = rgb_T3HW[0:1]                                          # [1, 3, H, W]
    target = s_0_pure_3HW.unsqueeze(0).to(frame_0.dtype)            # [1, 3, H, W]
    target = target.to(frame_0.device)
    l1 = F.l1_loss(frame_0, target)
    lp = lpips_module(frame_0, target)
    return l1 + lp


def loss_last_frame_anchor(
    rgb_T3HW: torch.Tensor,
    s_5_pure_3HW: torch.Tensor,
    lpips_module: LPIPSModule,
) -> torch.Tensor:
    """Final rendered frame vs the no-carpet end-state reference."""
    if rgb_T3HW.ndim != 4 or rgb_T3HW.shape[1] != 3:
        raise ValueError(
            f"rgb_T3HW must be [F, 3, H, W]; got {tuple(rgb_T3HW.shape)}"
        )
    expected_s5 = (3, int(rgb_T3HW.shape[2]), int(rgb_T3HW.shape[3]))
    if s_5_pure_3HW.shape != expected_s5:
        raise ValueError(
            f"s_5_pure must be {expected_s5}; "
            f"got {tuple(s_5_pure_3HW.shape)}"
        )
    frame_last = rgb_T3HW[-1:].contiguous()
    target = s_5_pure_3HW.unsqueeze(0).to(frame_last.dtype)
    target = target.to(frame_last.device)
    l1 = F.l1_loss(frame_last, target)
    lp = lpips_module(frame_last, target)
    return l1 + lp


def loss_contact_anchor(
    axis: torch.Tensor,
    origin: torch.Tensor,
    anchors_world: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Point-line distance from anchor voxels to the joint axis line.

    The base-move contact band (Stage B output) gives ~10-50 anchor voxels
    on or near the part interface. A revolute axis (or a prismatic axis,
    interpreted geometrically as the contact line) is physically required
    to pass through this band. We penalize the mean perpendicular distance
    from each anchor to the line ``{origin + t * axis : t in R}``.

    Parameters
    ----------
    axis : Tensor [3]    unit vector (caller normalizes via ``project_joint``)
    origin : Tensor [3]  world space
    anchors_world : Tensor [N_a, 3]  world space (converted from voxel via
                                     ``voxel_to_world``)

    Returns
    -------
    loss : Tensor scalar    mean perpendicular distance (world units)
    """
    if anchors_world.shape[-1] != 3 or anchors_world.ndim != 2:
        raise ValueError(
            f"anchors_world must be [N_a, 3]; got {tuple(anchors_world.shape)}"
        )
    diff = anchors_world - origin.unsqueeze(0)                        # [N_a, 3]
    along_axis = (diff * axis.unsqueeze(0)).sum(dim=-1, keepdim=True)  # [N_a, 1]
    proj = along_axis * axis.unsqueeze(0)                              # [N_a, 3]
    perp = diff - proj                                                 # [N_a, 3]
    perp_sq = (perp * perp).sum(dim=-1)                                # [N_a]
    # sqrt(sum + eps) for numerical stability at zero distance
    dist = torch.sqrt(perp_sq + eps)
    return dist.mean()


def loss_gate_sharpening(
    r: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """``mean(sigmoid(x) * (1 - sigmoid(x)))`` on both gate logits.

    Equivalent to (a constant times) the entropy of the Bernoulli with
    probability sigmoid(x). Encourages probabilities away from 0.5
    toward {0, 1}. The BinaryConcrete temperature schedule does the
    annealing on the *forward* path; this loss adds explicit pressure on
    the *gradient* path.
    """
    p_r = torch.sigmoid(r)
    p_b = torch.sigmoid(b)
    return ((p_r * (1.0 - p_r)).mean() + (p_b * (1.0 - p_b)).mean())


def loss_shell_sparsity(
    r: torch.Tensor,
    shell_mask: torch.Tensor,
) -> torch.Tensor:
    """``sigmoid(r[shell_mask]).mean()`` (encourages dying of uncertain shell).

    method.md D-v3.14. The shell voxels (boundary band 0.1 < O_init < 0.3,
    minus carpet, restricted to U_object) are "exploration" candidates;
    after the optimizer has had time to discover whether they are real
    geometry, the unselected ones should be pushed to die (g = 0).

    Returns 0 if ``shell_mask`` is all False (degenerate).
    """
    if r.shape != shell_mask.shape:
        raise ValueError(
            f"r / shell_mask shape mismatch: r={tuple(r.shape)}, "
            f"shell={tuple(shell_mask.shape)}"
        )
    n_shell = int(shell_mask.sum().item())
    if n_shell == 0:
        return torch.zeros((), device=r.device, dtype=r.dtype)
    return torch.sigmoid(r[shell_mask]).mean()


def loss_m_prior(
    alpha_m: torch.Tensor,
    m_attn_at_U: torch.Tensor,
    clamp_lo: float = 0.05,
    clamp_hi: float = 0.95,
) -> torch.Tensor:
    """BCE-with-logits driving ``alpha_m -> logit(1 - M_attn_clamped)``.

    BMCSA M_attn is high on base voxels (cross-state feature agreement)
    and low on move voxels. We want ``alpha_m`` (the *move* logit) to be
    high on move voxels and low on base, so the target is ``1 - M_attn``.

    This term is only active in warmup_g0; in main_g1 onward the SDS
    gradient drives alpha_m, and the prior is decayed to 0 (see
    ``schedules.schedule_lambda_m_prior``).
    """
    if alpha_m.shape != m_attn_at_U.shape:
        raise ValueError(
            f"alpha_m / m_attn shape mismatch: alpha_m={tuple(alpha_m.shape)}, "
            f"m_attn={tuple(m_attn_at_U.shape)}"
        )
    target = (1.0 - m_attn_at_U).clamp(clamp_lo, clamp_hi)
    return F.binary_cross_entropy_with_logits(alpha_m, target)


def loss_delta_z_stability(Delta_z_s: torch.Tensor) -> torch.Tensor:
    """L2 on the SS latent residual; keeps z_s_base close to z_s0."""
    return (Delta_z_s ** 2).mean()


# =============================================================================
# Aggregator
# =============================================================================

@dataclass
class LossInputs:
    """All quantities required by ``aggregate_loss``.

    Built per-iter by the training loop. Tensors that change every iter
    (rgb, r, b, joint params) come from the current forward; tensors that
    are static for the whole run (anchors_world, shell_mask, target video,
    s_0, M_attn) are loaded once from Bootstrap.
    """
    # Renderer output (current iter)
    rgb_T3HW: torch.Tensor                # [F, 3, H, W] in [0, 1] (grad-on)

    # Gate logits (current iter, before BinaryConcrete)
    r: torch.Tensor                       # [N_obj]   presence gate logit
    b: torch.Tensor                       # [N_obj]   move     gate logit

    # Joint params (current iter, post project_joint)
    axis: torch.Tensor                    # [3] unit
    origin: torch.Tensor                  # [3]

    # Latent residual
    Delta_z_s: torch.Tensor               # [8, 16, 16, 16] or [1, 8, 16, 16, 16]

    # alpha_m parameter (for L_m_prior)
    alpha_m: torch.Tensor                 # [N_obj]

    # Static Bootstrap-derived inputs
    wan_video_target_T3HW_01: torch.Tensor  # [F, 3, H, W] in [0, 1]
    s_0_pure_3HW_01: torch.Tensor           # [3, H, W] in [0, 1]
    s_5_pure_3HW_01: Optional[torch.Tensor] # [3, H, W] in [0, 1]
    z_wan_target: torch.Tensor              # [16, F_lat, H_lat, W_lat]
    anchors_world: torch.Tensor             # [N_a, 3]
    shell_mask: torch.Tensor                # [N_obj] bool
    m_attn_at_U: torch.Tensor               # [N_obj] in [0, 1]

    # W-RFSDS specifics
    wan_cond: Dict[str, Any]               # cached
    tau: float                              # this iter's sampled tau
    cfg_scale: float                        # this iter's CFG


def aggregate_loss(
    inp: LossInputs,
    ctx: WanRFSDSContext,
    lpips_module: LPIPSModule,
    cfg_lambdas_first: float,
    cfg_lambdas_last: float,
    cfg_lambdas_contact: float,
    cfg_lambdas_gate: float,
    cfg_lambdas_z: float,
    sched_lambdas_w_rfsds: Tuple[float, float, float],   # (sds, lat, rgb)
    sched_lambda_shell: float,
    sched_lambda_m_prior: float,
    seed_for_eps: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compose all loss components and return (total_loss, log_dict).

    The log dict contains the *unweighted* component values (so logs are
    interpretable across schedule changes). The total uses the per-iter
    schedule weights.

    Components that are skipped under the current schedule (weight == 0)
    are *still* evaluated to keep the autograd graph consistent across
    iterations — except for the expensive ones (L_sds, L_lat_rec) which
    are gated by ``lambda > 0`` to avoid an unnecessary Wan call.
    """
    rgb_T3HW = inp.rgb_T3HW
    # Permuted view for any consumer needing [3, F, H, W] (Wan VAE).
    rgb_3FHW = rgb_T3HW.permute(1, 0, 2, 3)                           # [3, F, H, W]

    log: Dict[str, float] = {}
    total = torch.zeros((), device=rgb_T3HW.device, dtype=rgb_T3HW.dtype)

    lambda_sds, lambda_lat, lambda_rgb = sched_lambdas_w_rfsds

    # ---- L_sds (W-RFSDS); only evaluate if nonzero weight ----
    # ★ S5 fix: when both lambda_sds > 0 and lambda_lat > 0, request the VAE
    # latent back so L_lat can reuse it (skips a second Wan VAE encode +
    # backward, the most expensive op per iter).
    z_theta_cached: Optional[torch.Tensor] = None
    if lambda_sds > 0.0:
        need_z = lambda_lat > 0.0
        result = w_rfsds_loss(
            rgb_3FHW, inp.wan_cond, ctx,
            tau=inp.tau, cfg_scale=inp.cfg_scale, seed=seed_for_eps,
            return_z_theta=need_z,
        )
        if need_z:
            L_sds, z_theta_cached = result
        else:
            L_sds = result
        total = total + lambda_sds * L_sds
        log["L_sds"] = float(L_sds.detach().item())
    else:
        log["L_sds"] = 0.0

    # ---- L_lat_rec (auxiliary; off by default; reuses z_theta when possible) ----
    if lambda_lat > 0.0:
        L_lat = latent_recon_loss(
            rgb_3FHW, inp.z_wan_target, ctx,
            z_render_cached=z_theta_cached,    # ★ S5 fix
        )
        total = total + lambda_lat * L_lat
        log["L_lat"] = float(L_lat.detach().item())
    else:
        log["L_lat"] = 0.0

    # ---- L_rgb_rec (pixel L1 + LPIPS) ----
    if lambda_rgb > 0.0:
        L_rgb = loss_rgb_recon(rgb_T3HW, inp.wan_video_target_T3HW_01, lpips_module)
        total = total + lambda_rgb * L_rgb
        log["L_rgb"] = float(L_rgb.detach().item())
    else:
        log["L_rgb"] = 0.0

    # ---- L_first (frame 0 vs no-carpet s_0_pure; ALWAYS on, fixed weight) ----
    L_first = loss_first_frame_anchor(rgb_T3HW, inp.s_0_pure_3HW_01, lpips_module)
    total = total + cfg_lambdas_first * L_first
    log["L_first"] = float(L_first.detach().item())

    # ---- L_last (frame F-1 vs no-carpet s_5_pure; used by InP only) ----
    if cfg_lambdas_last > 0.0:
        if inp.s_5_pure_3HW_01 is None:
            raise ValueError("cfg_lambdas_last > 0 requires s_5_pure_3HW_01")
        L_last = loss_last_frame_anchor(rgb_T3HW, inp.s_5_pure_3HW_01, lpips_module)
        total = total + cfg_lambdas_last * L_last
        log["L_last"] = float(L_last.detach().item())
    else:
        log["L_last"] = 0.0

    # ---- L_contact (axis-anchor band) ----
    L_contact = loss_contact_anchor(inp.axis, inp.origin, inp.anchors_world)
    total = total + cfg_lambdas_contact * L_contact
    log["L_contact"] = float(L_contact.detach().item())

    # ---- L_gate (rounds g, m to {0, 1}) ----
    L_gate = loss_gate_sharpening(inp.r, inp.b)
    total = total + cfg_lambdas_gate * L_gate
    log["L_gate"] = float(L_gate.detach().item())

    # ---- L_shell (sparsity on uncertain shell) ----
    if sched_lambda_shell > 0.0:
        L_shell = loss_shell_sparsity(inp.r, inp.shell_mask)
        total = total + sched_lambda_shell * L_shell
        log["L_shell"] = float(L_shell.detach().item())
    else:
        log["L_shell"] = 0.0

    # ---- L_m_prior (warmup BCE) ----
    if sched_lambda_m_prior > 0.0:
        L_mp = loss_m_prior(inp.alpha_m, inp.m_attn_at_U)
        total = total + sched_lambda_m_prior * L_mp
        log["L_m_prior"] = float(L_mp.detach().item())
    else:
        log["L_m_prior"] = 0.0

    # ---- L_z (Delta_z_s stability) ----
    L_z = loss_delta_z_stability(inp.Delta_z_s)
    total = total + cfg_lambdas_z * L_z
    log["L_z"] = float(L_z.detach().item())

    log["L_total"] = float(total.detach().item())
    return total, log


__all__ = [
    "LPIPSModule",
    "loss_rgb_recon", "loss_first_frame_anchor", "loss_last_frame_anchor",
    "loss_contact_anchor",
    "loss_gate_sharpening", "loss_shell_sparsity", "loss_m_prior",
    "loss_delta_z_stability",
    "LossInputs", "aggregate_loss",
]

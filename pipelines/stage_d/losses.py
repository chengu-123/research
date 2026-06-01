"""Stage D loss components and aggregator.

method.md section 9.1 enumerates the full loss list:

    L_total = lambda_sds   * L_sds        (W-RFSDS through Wan2.2)
            + lambda_lat   * L_lat_rec    (auxiliary, off by default; C2)
            + lambda_rgb   * L_pure_geom  (silhouette + luminance edges vs six pure states)
            + lambda_first * L_first      (frame 0 vs no-carpet s_0_pure)
            + lambda_last  * L_last       (frame F-1 vs no-carpet s_5_pure)
            + lambda_contact * L_contact  (axis-through-anchor-band prior)
            + lambda_gate  * L_gate       (encourages g, m -> {0, 1})
            + lambda_shell * L_shell      (sparsity on uncertain shell voxels)
            + lambda_m_prior * L_m_prior  (BCE alpha_m vs (1 - M_attn))
            + lambda_move_cover * L_move_cover
            + lambda_move_suppress * L_move_suppress
            + lambda_z     * L_z          (Delta_z_s L2 stability)

This module:
  - implements each individual component
  - provides ``LPIPSModule`` for callers that still construct the historical
    Stage D interface
  - exposes ``aggregate_loss`` which composes all components according to
    the current phase / schedule.

The LPIPS package expects ``[N, 3, H, W]`` inputs in ``[-1, 1]``; we
rescale from our ``[0, 1]`` render output inline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import STATE_INDICES
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

def _soft_silhouette(batch_N3HW: torch.Tensor) -> torch.Tensor:
    return batch_N3HW.abs().sum(dim=1, keepdim=True).clamp(0.0, 1.0)


def _luminance_edges(batch_N3HW: torch.Tensor) -> torch.Tensor:
    weights = batch_N3HW.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    luma = (batch_N3HW * weights).sum(dim=1, keepdim=True)
    kernel_x = batch_N3HW.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = batch_N3HW.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3) / 8.0
    grad_x = F.conv2d(luma, kernel_x, padding=1)
    grad_y = F.conv2d(luma, kernel_y, padding=1)
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1.0e-6)


def _luminance_1HW(frame_3HW: torch.Tensor) -> torch.Tensor:
    if frame_3HW.ndim != 3 or frame_3HW.shape[0] != 3:
        raise ValueError(f"expected [3, H, W], got {tuple(frame_3HW.shape)}")
    weights = frame_3HW.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    return (frame_3HW * weights).sum(dim=0, keepdim=True)


def dynamic_mask(
    ref_first_3HW: torch.Tensor,
    ref_last_3HW: torch.Tensor,
    dilate_px: int = 4,
    thresh: float = 0.04,
) -> torch.Tensor:
    """Binary [1, H, W] mask of pixels whose FOREGROUND (silhouette) changes across
    endpoints -- the moving part's swept region.

    SILHOUETTE diff, NOT luminance diff: M0/M5 are two INDEPENDENT TRELLIS keyframe
    recons (``build_per_state_keyframe_models``, per-state occupancy + conditioning)
    that shade the SHARED static cabinet differently. A luminance diff therefore lit
    up the static base even where geometry is identical -- VERIFIED: 65% of the old
    luminance D was appearance contamination on the shared foreground, so L_cover
    demanded covering the static cabinet -> move over-inflation + non-convergence
    (baseanchor_v1). The foreground/occupancy diff fires only where geometry appears
    or disappears = where the part actually moves. ``thresh`` is the foreground
    cutoff on ``sum_c |rgb|`` (black background -> 0).
    """
    if ref_first_3HW.shape != ref_last_3HW.shape:
        raise ValueError(
            f"endpoint ref shape mismatch: first={tuple(ref_first_3HW.shape)} "
            f"last={tuple(ref_last_3HW.shape)}"
        )
    first = ref_first_3HW.detach()
    last = ref_last_3HW.detach().to(device=first.device, dtype=first.dtype)
    fg_first = (first.abs().sum(dim=0, keepdim=True) > float(thresh)).to(dtype=first.dtype)
    fg_last = (last.abs().sum(dim=0, keepdim=True) > float(thresh)).to(dtype=first.dtype)
    mask = (fg_first - fg_last).abs()
    if int(dilate_px) > 0:
        radius = int(dilate_px)
        kernel = 2 * radius + 1
        mask = F.max_pool2d(
            mask.unsqueeze(0),
            kernel_size=kernel,
            stride=1,
            padding=radius,
        ).squeeze(0)
    return mask.clamp(0.0, 1.0)


def loss_motion_ownership(
    move_sil_K1HW: torch.Tensor,
    ref_first_3HW: torch.Tensor,
    ref_last_3HW: torch.Tensor,
    dilate_px: int = 4,
    thresh: float = 0.04,
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Move branch must cover dynamic pixels and avoid static foreground pixels."""
    if move_sil_K1HW.ndim == 3:
        move_sil = move_sil_K1HW.unsqueeze(1)
    else:
        move_sil = move_sil_K1HW
    if move_sil.ndim != 4 or move_sil.shape[1] != 1:
        raise ValueError(f"move_sil must be [K, 1, H, W], got {tuple(move_sil.shape)}")

    ref_first = ref_first_3HW.to(device=move_sil.device, dtype=move_sil.dtype)
    ref_last = ref_last_3HW.to(device=move_sil.device, dtype=move_sil.dtype)
    D = dynamic_mask(ref_first, ref_last, dilate_px=dilate_px, thresh=thresh)
    if tuple(D.shape[-2:]) != tuple(move_sil.shape[-2:]):
        raise ValueError(
            f"mask/render size mismatch: D={tuple(D.shape)} move={tuple(move_sil.shape)}"
        )
    fg_first = _soft_silhouette(ref_first.unsqueeze(0)).squeeze(0)
    fg_last = _soft_silhouette(ref_last.unsqueeze(0)).squeeze(0)
    fg = (fg_first + fg_last).clamp(0.0, 1.0)
    move_cov = move_sil.clamp(0.0, 1.0).amax(dim=0)
    static = fg * (1.0 - D)
    eps_t = move_sil.new_tensor(float(eps))
    L_cover = (F.relu(D - move_cov) * D).sum() / (D.sum() + eps_t)
    L_suppress = (move_cov * static).sum() / (static.sum() + eps_t)
    return L_cover, L_suppress


def _pure_geometry_loss(
    pred_N3HW: torch.Tensor,
    target_N3HW: torch.Tensor,
    edge_weight: float,
) -> torch.Tensor:
    pred_sil = _soft_silhouette(pred_N3HW)
    target_sil = _soft_silhouette(target_N3HW)
    sil = F.l1_loss(pred_sil, target_sil)
    pred_edges = _luminance_edges(pred_N3HW) * target_sil
    target_edges = _luminance_edges(target_N3HW) * target_sil
    edge = F.l1_loss(pred_edges, target_edges)
    return sil + float(edge_weight) * edge


def _pure_photometric_loss(
    pred_N3HW: torch.Tensor,
    target_N3HW: torch.Tensor,
    lpips_module: LPIPSModule,
    edge_weight: float,
) -> torch.Tensor:
    """Appearance (L1 + LPIPS) + geometric (silhouette + edge) endpoint anchor.

    The endpoint targets s_0_pure / s_5_pure are TRELLIS keyframe renders (M0
    closed / M5 open) in the SAME render + texture space as the StageD output
    (verified: silver-metal + JENN-AIR display, identical to the recon), so a
    direct FOREGROUND-MASKED RGB L1 is valid and far richer than silhouette-only.
    Stage D does not optimize texture, but it DOES optimize the gates + joint that
    move the textured Gaussians, so matching the rendered appearance of the
    closed/open endpoints directly supervises the articulation (range, canonical
    pose, move/base split). Composition (geometry-led, perceptual-tethered):
        silhouette(1.0) + edge_weight*edge + masked_L1(1.0) + 0.1*LPIPS
    silhouette + masked_L1 are co-primary geometric/appearance drivers; LPIPS is a
    weak, alignment-robust perceptual tether (NOT the driver, since StageD does not
    optimize texture); the background is excluded from L1 so it cannot dilute the
    foreground signal.
    """
    pred_c = pred_N3HW.clamp(0.0, 1.0)
    target_c = target_N3HW.clamp(0.0, 1.0)
    pred_sil = _soft_silhouette(pred_N3HW)
    target_sil = _soft_silhouette(target_N3HW)
    sil = F.l1_loss(pred_sil, target_sil)
    pred_edges = _luminance_edges(pred_N3HW) * target_sil
    target_edges = _luminance_edges(target_N3HW) * target_sil
    edge = F.l1_loss(pred_edges, target_edges)
    # foreground-masked RGB L1 (mask = target support; background excluded)
    fg = target_sil                                                   # [N, 1, H, W]
    masked_l1 = ((pred_c - target_c).abs() * fg).sum() / (fg.sum() * 3.0 + 1.0e-6)
    lpips_val = lpips_module(pred_c, target_c)
    return sil + float(edge_weight) * edge + masked_l1 + 0.1 * lpips_val


def loss_six_state_recon(
    rgb_T3HW: torch.Tensor,
    pure_state_targets_K3HW: torch.Tensor,
    lpips_module: LPIPSModule,
    silhouette_weight: float = 1.0,
    lpips_chunk: int = 7,
) -> torch.Tensor:
    """Geometry loss against the six pure observed states.

    Stage D does not update Gaussian texture. Direct RGB L1/LPIPS against
    PartNet gray targets would push texture mismatch through geometry, gates,
    and motion. This loss uses foreground silhouette and luminance edges only.

    Parameters
    ----------
    rgb_T3HW : Tensor [F, 3, H, W] in [0, 1]
    pure_state_targets_K3HW : Tensor [K=6, 3, H, W] in [0, 1]
    lpips_chunk : int
        Historical interface parameter. Geometry-only supervision does not
        call LPIPS because Stage D does not optimize texture.

    Returns
    -------
    loss : Tensor scalar
    """
    if rgb_T3HW.ndim != 4 or rgb_T3HW.shape[1] != 3:
        raise ValueError(f"rgb_T3HW must be [F, 3, H, W]; got {tuple(rgb_T3HW.shape)}")
    if pure_state_targets_K3HW.ndim != 4 or pure_state_targets_K3HW.shape[1] != 3:
        raise ValueError(
            f"pure_state_targets_K3HW must be [K, 3, H, W]; "
            f"got {tuple(pure_state_targets_K3HW.shape)}"
        )
    state_indices = torch.tensor(STATE_INDICES, device=rgb_T3HW.device, dtype=torch.long)
    if int(pure_state_targets_K3HW.shape[0]) != int(state_indices.numel()):
        raise ValueError(
            f"pure_state_targets_K3HW must have K={int(state_indices.numel())}; "
            f"got {int(pure_state_targets_K3HW.shape[0])}"
        )
    selected = rgb_T3HW.index_select(0, state_indices)
    target = pure_state_targets_K3HW.to(device=selected.device, dtype=selected.dtype)
    if selected.shape != target.shape:
        raise ValueError(
            f"six-state rgb / target shape mismatch: rgb={tuple(selected.shape)}, "
            f"target={tuple(target.shape)}"
        )
    return _pure_geometry_loss(
        selected,
        target,
        edge_weight=0.25 * float(silhouette_weight),
    )


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
    return _pure_photometric_loss(frame_0, target, lpips_module, edge_weight=0.25)


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
    return _pure_photometric_loss(frame_last, target, lpips_module, edge_weight=0.25)


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
    move_logits: torch.Tensor,
    m_attn_at_U: torch.Tensor,
    clamp_lo: float = 0.05,
    clamp_hi: float = 0.95,
) -> torch.Tensor:
    """BCE-with-logits driving effective move logits from a base-like prior.

    The input is high on base voxels and low on move voxels. We want
    the effective move logit ``b`` to be high on move voxels and low on base,
    so the target is ``1 - base_like_prior``.

    In main stages this must supervise ``b = alpha_m + lambda_part * H_part``,
    not just ``alpha_m``. Otherwise the part head can override the prior while
    the prior loss remains unchanged.
    """
    if move_logits.shape != m_attn_at_U.shape:
        raise ValueError(
            f"move_logits / m_attn shape mismatch: logits={tuple(move_logits.shape)}, "
            f"m_attn={tuple(m_attn_at_U.shape)}"
        )
    target = (1.0 - m_attn_at_U).clamp(clamp_lo, clamp_hi)
    return F.binary_cross_entropy_with_logits(move_logits, target)


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
    pure_state_targets_K3HW_01: torch.Tensor # [K, 3, H, W] in [0, 1]
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


def loss_move_floor(b: torch.Tensor, floor: float) -> torch.Tensor:
    """Anti-collapse hinge on the MEAN soft move fraction ``mean(sigmoid(b))``.

    The single-view video-SDS prefers a static render, so the move set drifts to
    base and the HARD binary_concrete gate commits there; the collapse is a
    *coordinated* local minimum (all voxels base) that no single-voxel gradient
    escapes, since flipping one tiny voxel does not change the render. This term
    pushes ALL b up together (a coordinated gradient on the smooth sigmoid, which
    does NOT vanish at the committed corner like the gate's STE) whenever the soft
    move fraction falls below ``floor``, lifting the move set back out of the
    collapse. One-sided: inactive once the move mass is healthy (> floor).
    """
    move_frac = torch.sigmoid(b).mean()
    return torch.relu(b.new_tensor(float(floor)) - move_frac)


def loss_move_ceiling(b: torch.Tensor, ceiling: float) -> torch.Tensor:
    """Anti-whole-object hinge on the mean soft move fraction."""
    move_frac = torch.sigmoid(b).mean()
    return torch.relu(move_frac - b.new_tensor(float(ceiling)))


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
    cfg_lambda_move_floor: float = 0.0,
    move_floor_frac: float = 0.0,
    cfg_lambda_move_ceiling: float = 0.0,
    move_ceiling_frac: float = 1.0,
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

    # ---- Pure-state geometry supervision (six selected frames) ----
    if lambda_rgb > 0.0:
        L_rgb = loss_six_state_recon(
            rgb_T3HW,
            inp.pure_state_targets_K3HW_01,
            lpips_module,
        )
        total = total + lambda_rgb * L_rgb
        log["L_rgb"] = float(L_rgb.detach().item())
    else:
        log["L_rgb"] = 0.0

    # ---- L_first (frame 0 vs no-carpet s_0_pure; pixel anchor, gated) ----
    if cfg_lambdas_first > 0.0:
        L_first = loss_first_frame_anchor(rgb_T3HW, inp.s_0_pure_3HW_01, lpips_module)
        total = total + cfg_lambdas_first * L_first
        log["L_first"] = float(L_first.detach().item())
    else:
        log["L_first"] = 0.0

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

    # ---- L_move_floor (anti-collapse: keep soft move mass above a floor) ----
    if cfg_lambda_move_floor > 0.0 and move_floor_frac > 0.0:
        L_move_floor = loss_move_floor(inp.b, move_floor_frac)
        total = total + cfg_lambda_move_floor * L_move_floor
        log["L_move_floor"] = float(L_move_floor.detach().item())
    else:
        log["L_move_floor"] = 0.0

    # ---- L_move_ceiling (anti-escape: prevent whole-object-as-move) ----
    if cfg_lambda_move_ceiling > 0.0 and move_ceiling_frac < 1.0:
        L_move_ceiling = loss_move_ceiling(inp.b, move_ceiling_frac)
        total = total + cfg_lambda_move_ceiling * L_move_ceiling
        log["L_move_ceiling"] = float(L_move_ceiling.detach().item())
    else:
        log["L_move_ceiling"] = 0.0

    # ---- L_shell (sparsity on uncertain shell) ----
    if sched_lambda_shell > 0.0:
        L_shell = loss_shell_sparsity(inp.r, inp.shell_mask)
        total = total + sched_lambda_shell * L_shell
        log["L_shell"] = float(L_shell.detach().item())
    else:
        log["L_shell"] = 0.0

    # ---- L_m_prior (warmup BCE) ----
    if sched_lambda_m_prior > 0.0:
        L_mp = loss_m_prior(inp.b, inp.m_attn_at_U)
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
    "loss_six_state_recon", "loss_first_frame_anchor", "loss_last_frame_anchor",
    "loss_contact_anchor",
    "loss_gate_sharpening", "loss_shell_sparsity", "loss_m_prior",
    "loss_move_floor", "loss_move_ceiling",
    "dynamic_mask", "loss_motion_ownership",
    "loss_delta_z_stability",
    "LossInputs", "aggregate_loss",
]

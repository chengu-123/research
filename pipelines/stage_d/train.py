"""Stage D P1 training loop.

Orchestrates everything: phase routing, per-iter forward / backward,
type vote, periodic silhouette check, logging, checkpointing.

Inputs (built by ``run_stage_d.py`` before calling ``train_stage_d_p1``):
  - ``bootstrap``      : ``BootstrapBundle`` (loaded artifacts on device)
  - ``trellis``        : ``TrellisModules`` (frozen SS-DiT / SS-VAE-dec / D_GS)
  - ``wan_ctx``        : ``WanRFSDSContext``     (Wan VAE / low / high noise DiT)
  - ``camera``         : ``LockedCamera``
  - ``lpips_module``   : ``LPIPSModule``
  - ``cfg``            : ``StageDConfig``
  - ``out_dir``        : str

Outputs (saved under ``out_dir/``):
  - ``learnable_p1_final.pt``   committed type, frozen psi/phi, final g/m/r/b
  - ``z_slat0.pt``              unchanged (P1 does not learn z_slat0)
  - ``viz/iter_NNNNNN/``        snapshots from ``viz.save_iter_snapshot``
  - ``logs.jsonl``              one JSON line per ``cfg.log_every`` iter
"""
from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

from trellis.modules import sparse as sp

from .config import (
    CANONICAL_STATE_IDX,
    F_FRAMES,
    K_STATES,
    PSI_PARAM_DIM,
    STATE_INDICES,
    StageDConfig,
    TRELLIS_DGS_N_GAUSS_PER_VOXEL,
    TRELLIS_OCC_RES,
)
from .feature_sample import (
    binary_concrete_ste,
    sample_hidden_at_U,
    voxel_to_world,
)
from .joint_ops import (
    JointParams,
    phi_rollout,
    project_joint,
    stage_detach,
    weighted_feature_pool,
)
from .learnable import StageDLearnable
from .losses import (
    LossInputs,
    LPIPSModule,
    aggregate_loss,
    dynamic_mask,
    loss_motion_ownership,
)
from .render import (
    DGSWithParent,
    LockedCamera,
    RenderInputs,
    StageDCameraConfig,
    build_locked_camera,
    render_move_silhouettes,
    render_21_with_warp,
    render_static_gaussians,
    sample_camera_for_iter,
)
from .schedules import (
    phase_of,
    sample_tau_chord_anneal,
    sample_t_ss,
    schedule_cfg,
    schedule_gate_temperature,
    schedule_head_lambdas,
    schedule_lambda_gate,
    schedule_lambda_m_prior,
    schedule_lambda_shell,
    schedule_snapshot,
    schedule_w_rfsds_weights,
)
from .silhouette_check import (
    SilhouetteCheckReport,
    default_check_state_indices,
    iter_0_camera_check,
    periodic_silhouette_check,
)
from .ss_dit_wrapper import SS_DiT_WithAdapters
from .type_vote import (
    DualCloneState,
    TypeVoteResult,
    commit_dual_clone,
    make_dual_clones,
    run_type_vote,
    zero_type_logit_grad,
)
from .viz import (
    save_3d_state_html,
    save_initial_render_contract_diagnostics,
    save_iter_snapshot,
    save_loss_curves_html,
    save_motion_ownership_debug,
    save_no_learning_gs_ablation,
    save_p1_final_summary,
    save_phi_curve,
    save_phi_curve_html,
)
from .w_rfsds import (
    WanRFSDSContext,
    _prepare_fun_inp_rfsds_condition,
    rebuild_fun_inp_y_for_keyframes,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Bootstrap bundle (everything Stage D loads from disk)
# =============================================================================

@dataclass
class BootstrapBundle:
    """All Bootstrap artifacts loaded onto the training device.

    Lifetimes:
      - ``z_s0`` and ``trellis_cond_can`` stay resident (cheap; SS-DiT cond).
      - ``z_slat0`` and ``U_object`` stay resident (drive D_GS every iter).
      - ``M_attn_at_U`` / ``shell_mask`` / ``anchors_world`` /
        ``pure_state_targets_K3HW_01`` / ``z_wan_target`` / ``wan_cond`` /
        ``slat_mean`` / ``slat_std`` stay resident (every iter).
      - ``dit_hidden_cache`` is diagnostic-only and dropped after loading
        (it was useful to past Stage C v8.1; Stage D learns adapters and
        does not need it).

    All tensors are placed on ``device`` and dtypes converted to fp32
    unless otherwise noted (the bf16 cast happens inside Wan's autocast
    context inside ``w_rfsds_loss``).
    """
    # SS / SLAT geometry
    z_s0: torch.Tensor                        # [1, 8, 16, 16, 16]
    z_slat0: torch.Tensor                     # [N_obj, 8]
    slat_mean: torch.Tensor                   # [8]
    slat_std: torch.Tensor                    # [8]
    slat_shell_mask: torch.Tensor             # [N_obj] bool
    U_object: torch.Tensor                    # [N_obj, 3] int
    U_object_with_batch: torch.Tensor         # [N_obj, 4] int32 (for SparseTensor)
    gaussian_parent_idx: torch.Tensor         # [N_obj * 32] long
    # Voxel index helpers
    n_obj: int

    # Joint init
    psi_0: Dict[str, torch.Tensor]            # {axis [3], origin [3], type_logit scalar}
    phi_0: torch.Tensor                       # [6] canonical-shifted
    anchors_object: torch.Tensor              # [N_a, 3] int voxel
    anchors_world: torch.Tensor               # [N_a, 3] float (voxel_to_world cached)

    # BMCSA M_attn at U_object (for alpha_m init + L_m_prior)
    M_attn_at_U: torch.Tensor                 # [N_obj] in [0, 1]
    base_anchor_at_U: torch.Tensor            # [N_obj] bool visible base forced to m=0

    # Conditioning + targets
    trellis_cond_can: torch.Tensor            # [1, N_dino, 1024]
    wan_cond: Dict[str, Any]                  # context / context_null / seq_len / y
    z_wan_target: torch.Tensor                # [16, F_lat, H_lat, W_lat]
    wan_video_target_T3HW_01: torch.Tensor    # [F, 3, H, W] in [0, 1]
    pure_state_targets_K3HW_01: torch.Tensor  # [K, 3, H, W] in [0, 1]
    s_0_pure_3HW_01: torch.Tensor             # [3, H, W] in [0, 1]
    s_5_pure_3HW_01: Optional[torch.Tensor]   # [3, H, W] in [0, 1]
    frame_num: int
    resolution_hw: Tuple[int, int]
    latent_hw: Tuple[int, int]


@dataclass
class TrellisModules:
    """Frozen TRELLIS modules used by Stage D.

    Loaded once by ``run_stage_d.py`` from the same pipeline that Bootstrap
    used. All ``requires_grad`` set to False.
    """
    ss_dit: Any              # SparseStructureFlowModel
    ss_vae_decoder: Any      # SparseStructureVAE.decoder (or whole .decoder() callable)
    d_gs: Any                # SLatGaussianDecoder
    slat_sampler: Any        # for U-expand (deferred)


# =============================================================================
# Helper: build SparseTensor for D_GS from (z_slat0, U_object)
# =============================================================================

def _build_sparse_in(
    z_slat: torch.Tensor,
    U_object_with_batch: torch.Tensor,
) -> sp.SparseTensor:
    """Bundle the (feats, coords) into a TRELLIS SparseTensor.

    z_slat : [N_obj, 8]
    U_object_with_batch : [N_obj, 4] int32 (first column = batch index 0)
    """
    return sp.SparseTensor(feats=z_slat, coords=U_object_with_batch)


def _apply_base_anchor_to_move_logits(
    b: torch.Tensor,
    bootstrap: BootstrapBundle,
    cfg: StageDConfig,
) -> torch.Tensor:
    if not bool(cfg.base_anchor_enable):
        return b
    anchor = bootstrap.base_anchor_at_U.to(device=b.device, dtype=torch.bool)
    return torch.where(anchor, b.new_full(b.shape, float(cfg.base_anchor_logit)), b)


# =============================================================================
# Helper: measure decoded canonical object bbox (FreeArt3D camera framing)
# =============================================================================

def measure_canonical_object_bbox(
    trellis: TrellisModules,
    bootstrap: BootstrapBundle,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """Measure the decoded canonical object's world-space bounding box.

    FreeArt3D's run_rendering (mine/pipelines/render.py:265-275) frames the
    camera by the object: camera_distance = 2.1 * max(bbox extent), aimed at
    the bbox center. The TRELLIS SLAT decodes geometry into only a SUBSET of
    the [-0.5, 0.5]^3 box (true max_extent < 1, centroid offset from origin),
    so Stage D must measure the actual decoded Gaussians instead of assuming a
    unit box centered at the origin. Returns the bbox ``center`` ([3] world
    space) and ``max_extent`` (largest bbox side length, world units).
    """
    d_gs_w = DGSWithParent(d_gs_frozen=trellis.d_gs).to(device)
    sparse_in = _build_sparse_in(bootstrap.z_slat0, bootstrap.U_object_with_batch)
    with torch.no_grad():
        gauss, _parent_idx = d_gs_w(sparse_in)
        xyz = gauss.get_xyz                       # [N, 3] world space, in [-0.5, 0.5]
        min_corner = xyz.amin(dim=0)
        max_corner = xyz.amax(dim=0)
    center = 0.5 * (min_corner + max_corner)
    max_extent = float((max_corner - min_corner).amax().item())
    return center, max_extent


# =============================================================================
# One regular-phase forward (warmup_g0 / main_g1a / main_g1b / transition)
# =============================================================================

def _regular_forward(
    *,
    learnable: StageDLearnable,
    ss_dit_w: SS_DiT_WithAdapters,
    ss_vae_decoder: Any,
    d_gs_w: DGSWithParent,
    bootstrap: BootstrapBundle,
    camera: LockedCamera,
    f_global: float,
    cfg: StageDConfig,
    committed_type: Optional[Literal["revolute", "prismatic"]],
    forced_type_logit: Optional[float] = None,
) -> Tuple[torch.Tensor, JointParams, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           Dict[str, torch.Tensor], RenderInputs, RenderInputs,
           torch.Tensor, torch.Tensor]:
    """Run one full forward for a non-warmup-G- phase.

    Returns
    -------
    rgb_T3HW : Tensor [F, 3, H, W] in [0, 1] (grad-on)
    joint    : JointParams         (projected, grad-on)
    r, b     : Tensor [N_obj]      (gate logits, pre-STE)
    g_obj    : Tensor [N_obj]      (post-STE forward; binary-ish)
    m_obj    : Tensor [N_obj]
    u_shifted: Tensor [K=6]        (for diagnostics)
    head_preact_l2 : Tensor scalar (mean square of the pre-tanh head outputs;
                     add ``cfg.lambda_head_leash * head_preact_l2`` to the loss so
                     the head pre-activation cannot run away and saturate tanh,
                     which would sever the gradient to the shared SS-DiT adapter)
    head_stats : dict of detached scalar tensors for saturation diagnostics
    render_inputs, phi_rev, phi_pri : tensors reused by move-only ownership render
    """
    # ---- 1) z_s_base = z_s0 + Delta_z_s ----
    z_s_base = bootstrap.z_s0 + learnable.Delta_z_s.unsqueeze(0)

    # ---- 2) Apply stage_detach so q_sample uses the right grad path ----
    z_for_q = stage_detach(z_s_base, f_global, mode="q_sample",
                           f_full_detach_end=cfg.f_warmup_g_minus_end,
                           f_ramp_end=cfg.f_warmup_g0_end + 0.05)

    # ---- 3) Sample t_ss and run SS-DiT one-step refiner ----
    t_ss = sample_t_ss(f_global, cfg, device=z_s_base.device)
    if t_ss is None:
        raise RuntimeError(
            "regular_forward called in warmup_g_minus; use _warmup_minus_forward"
        )
    sigma_min = 1.0e-5
    eps = torch.randn_like(z_for_q)
    z_t = (1.0 - t_ss) * z_for_q + (sigma_min + (1.0 - sigma_min) * t_ss) * eps

    _pred_v, captured = ss_dit_w.forward_capture(
        z_t, t_raw=float(t_ss), cond=bootstrap.trellis_cond_can,
    )

    # ---- 4) SS-VAE decoder occupancy logits ----
    occ_logits = ss_vae_decoder(z_s_base)                  # [1, 1, 64, 64, 64]

    # ---- 5) Per-voxel feature ----
    feat, occ_at_U = sample_hidden_at_U(
        captured, bootstrap.U_object, occ_logits,
        capture_blocks=cfg.adapter_blocks,
        fourier_num_freqs=cfg.fourier_num_freqs,
    )                                                       # [N, D], [N]

    # ---- 6) Per-voxel gate logits ----
    lambda_sup, lambda_part, lambda_joint = schedule_head_lambdas(f_global, cfg)
    # Pre-tanh head pre-activations, squashed into the gate residual by
    # bound * tanh(.). head_preact_l2 keeps the pre-activation away from hard
    # tanh saturation so gradients can still reach the heads and adapters.
    sup_pre = learnable.H_sup(feat).squeeze(-1)
    part_pre = learnable.H_part(feat).squeeze(-1)
    H_sup_out = cfg.head_logit_residual_bound * torch.tanh(sup_pre)
    H_part_out = cfg.head_logit_residual_bound * torch.tanh(part_pre)
    head_preact_l2 = sup_pre.pow(2).mean() + part_pre.pow(2).mean()
    head_abs = torch.cat(
        [sup_pre.detach().abs().reshape(-1), part_pre.detach().abs().reshape(-1)],
        dim=0,
    )
    head_stats = {
        "head_preact_abs_mean": head_abs.mean(),
        "head_preact_abs_max": head_abs.max(),
        "head_tanh_saturation_frac": (
            head_abs > float(cfg.head_saturation_abs)
        ).float().mean(),
    }
    r = learnable.alpha_g + lambda_sup * H_sup_out
    b = learnable.alpha_m + lambda_part * H_part_out
    b = _apply_base_anchor_to_move_logits(b, bootstrap, cfg)

    # ---- 7) BinaryConcrete STE gates ----
    T_g, T_m = schedule_gate_temperature(f_global, cfg)
    g_obj = binary_concrete_ste(r, T_g)
    m_obj = binary_concrete_ste(b, T_m)

    # ---- 8) Joint head pool + projection ----
    m_soft = torch.sigmoid(b)
    F_pool = weighted_feature_pool(feat, m_soft, eps=cfg.sgs_eps)
    delta_psi = learnable.H_joint(F_pool)                   # [19]
    psi_for_warp = stage_detach(
        learnable.psi_param, f_global, mode="joint",
        ema_buf=learnable.psi_ema_buf,
        f_full_detach_end=cfg.f_warmup_g_minus_end,
        f_ramp_end=cfg.f_warmup_g0_end + 0.05,
    )
    psi_combined = psi_for_warp + lambda_joint * delta_psi
    if cfg.freeze_joint_axis_only:
        # Pin ONLY the axis direction psi[0:3] at the Stage-C init: take it from
        # psi_for_warp DETACHED (no SDS grad) and drop the H_joint residual on
        # those dims, so neither the video-SDS nor the joint head can rotate the
        # hinge. origin[3:6] + type[6] + range[7] + disp[8] stay trainable (the
        # pivot/range still optimize). FreeArt3D opt_dir=False. See config.py.
        psi_combined = torch.cat(
            [psi_for_warp[0:3].detach(), psi_combined[3:]], dim=0
        )
    if cfg.freeze_joint_hinge:
        # Pin the revolute HINGE (axis[0:3] + origin[3:6]) at the Stage-C
        # geometric init: take [0:6] from psi_for_warp DETACHED (no grad) and
        # drop the H_joint residual there, so neither SDS-on-psi nor the joint
        # head can drift the pivot. range/type/disp [6:9] still refine. This is
        # the airtight channel-level freeze (grad-masking psi_param alone misses
        # the lambda_joint * H_joint residual). Covers the main loop AND the
        # dual-clone since both call _regular_forward.
        psi_combined = torch.cat(
            [psi_for_warp[0:6].detach(), psi_combined[6:]], dim=0
        )
    if cfg.freeze_joint_range:
        # Also pin the rotation RANGE theta (psi[7]) at the Stage-C geometric
        # estimate: with only the hinge frozen, the free SDS + H_joint range
        # residual blows theta up (90 -> 128 deg by iter 120). type[6] + disp[8]
        # stay from psi_combined (type committed by the vote; disp unused for
        # revolute). SDS then refines only the gates.
        psi_combined = torch.cat(
            [psi_combined[0:7], psi_for_warp[7:8].detach(), psi_combined[8:]], dim=0
        )
    # If we are inside a dual-clone branch the type_logit is forced.
    if forced_type_logit is not None:
        psi_combined = psi_combined.clone()
        psi_combined[6] = float(forced_type_logit)
    joint = project_joint(psi_combined, eps=cfg.sgs_eps)

    # ---- 9) Phi rollout with NEW.1 canonical shift ----
    u_shifted, _u_render, phi_rev, phi_pri = phi_rollout(
        learnable.delta_phi,
        joint.theta_max,
        joint.disp_max,
        canonical_idx=CANONICAL_STATE_IDX,
        n_out_frames=F_FRAMES,
        n_states=K_STATES,
    )

    # ---- 10) D_GS decode (canonical Gaussians) ----
    sparse_in = _build_sparse_in(bootstrap.z_slat0, bootstrap.U_object_with_batch)
    gauss_can, _parent_idx = d_gs_w(sparse_in)
    g_per_gauss = g_obj[bootstrap.gaussian_parent_idx]
    m_per_gauss = m_obj[bootstrap.gaussian_parent_idx]

    render_inputs = RenderInputs(
        xyz_canon=gauss_can.get_xyz,
        opacity_canon=gauss_can.get_opacity,
        rot_canon=gauss_can.get_rotation,
        scale_canon=gauss_can.get_scaling,
        sh_canon=gauss_can._features_dc,
        g_per_gauss=g_per_gauss,
        m_per_gauss=m_per_gauss,
    )
    ownership_render_inputs = RenderInputs(
        xyz_canon=gauss_can.get_xyz,
        opacity_canon=gauss_can.get_opacity,
        rot_canon=gauss_can.get_rotation,
        scale_canon=gauss_can.get_scaling,
        sh_canon=gauss_can._features_dc,
        g_per_gauss=g_per_gauss,
        m_per_gauss=m_soft[bootstrap.gaussian_parent_idx],
    )

    # ---- 11) Render 21 frames ----
    type_soft = (torch.sigmoid(joint.type_logit) if committed_type is None
                 else None)
    rgb_T3HW = render_21_with_warp(
        render_inputs, joint, phi_rev, phi_pri, camera,
        type_soft=type_soft, committed_type=committed_type, sh_degree=0,
    )

    return (
        rgb_T3HW, joint, r, b, g_obj, m_obj, u_shifted, head_preact_l2,
        head_stats, render_inputs, ownership_render_inputs, phi_rev, phi_pri,
    )


# =============================================================================
# Warmup G- forward (SS-DiT skipped; only Delta_z_s + alpha_g + alpha_m active)
# =============================================================================

def _warmup_minus_forward(
    *,
    learnable: StageDLearnable,
    ss_vae_decoder: Any,
    d_gs_w: DGSWithParent,
    bootstrap: BootstrapBundle,
    camera: LockedCamera,
    cfg: StageDConfig,
) -> Tuple[torch.Tensor, JointParams, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Cheap warmup pass: no SS-DiT call, no W-RFSDS, only the decoder path.

    pipeline.md section 9.6: in warmup_g_minus (0-5%) we only update
    ``Delta_z_s`` + ``alpha_g`` + ``alpha_m`` via the SS-VAE decoder
    (occ_logits) + L_first + L_gate + L_z (+ L_m_prior). Joint is frozen
    at Bootstrap's psi_0 / phi_0 (already canonical-shifted).
    """
    z_s_base = bootstrap.z_s0 + learnable.Delta_z_s.unsqueeze(0)
    occ_logits = ss_vae_decoder(z_s_base)                  # [1, 1, 64, 64, 64]

    U = bootstrap.U_object.long()
    flat = U[:, 0] * TRELLIS_OCC_RES * TRELLIS_OCC_RES + U[:, 1] * TRELLIS_OCC_RES + U[:, 2]
    occ_at_U = occ_logits.view(-1).index_select(0, flat)   # [N_obj]

    r = learnable.alpha_g
    b = learnable.alpha_m
    b = _apply_base_anchor_to_move_logits(b, bootstrap, cfg)

    T_g, T_m = schedule_gate_temperature(0.0, cfg)
    g_obj = binary_concrete_ste(r, T_g)
    m_obj = binary_concrete_ste(b, T_m)

    # Frozen joint from Bootstrap.
    axis = bootstrap.psi_0["axis"].to(z_s_base.device)
    origin = bootstrap.psi_0["origin"].to(z_s_base.device)
    type_logit = bootstrap.psi_0["type_logit"].to(z_s_base.device).reshape(())
    theta_max = torch.tensor(1.5708, device=z_s_base.device)   # pi / 2
    disp_max = torch.tensor(0.3, device=z_s_base.device)
    joint = JointParams(axis=axis, origin=origin, type_logit=type_logit,
                        theta_max=theta_max, disp_max=disp_max)

    # Compute phi_render from Bootstrap's phi_0 (already canonical-shifted).
    # We linearly interpolate phi_0 [6] to 21 frames.
    # ★ M3 fix: render BOTH branches and soft-blend by sigmoid(type_logit),
    # matching _regular_forward. The old code committed to a hard type based
    # on Bootstrap's stage_c_joint_init estimate; if that estimate was wrong
    # (e.g. revolute mis-classified as prismatic at init), warmup_g_minus
    # would anchor alpha_g / alpha_m / Delta_z_s toward the wrong rendering
    # for 5% of total iters and the model would need to undo that later.
    # ★ Fix SD-2: Stage C outputs phi_0 in normalized progress space
    # (renormalized to [0, 1] then canonical-shifted to [-u[c], 1-u[c]]),
    # NOT in physical units. Must scale by theta_max / disp_max here, exactly
    # matching the _regular_forward phi_rollout convention (see joint_ops.py
    # phi_rollout: phi_rev = u_render * theta_max, phi_pri = u_render * disp_max).
    # Without this scaling, warmup_g_minus rendered ~0.5 rad max rotation
    # (~29 deg) instead of pi/2 (~90 deg) for typical drawer/cabinet inputs,
    # silently producing visually-wrong warmup renderings and biasing the
    # alpha_g / alpha_m / Delta_z_s anchor toward the wrong joint motion.
    from .joint_ops import linear_interp_through
    phi_0 = bootstrap.phi_0.to(z_s_base.device).to(z_s_base.dtype)
    u_shifted = phi_0                                       # already canonical-shifted, normalized
    u_render = linear_interp_through(u_shifted, n_out=F_FRAMES)
    phi_rev = u_render * theta_max                          # scale to radians
    phi_pri = u_render * disp_max                           # scale to world units

    # D_GS decode + render
    sparse_in = _build_sparse_in(bootstrap.z_slat0, bootstrap.U_object_with_batch)
    gauss_can, _parent_idx = d_gs_w(sparse_in)
    g_per_gauss = g_obj[bootstrap.gaussian_parent_idx]
    m_per_gauss = m_obj[bootstrap.gaussian_parent_idx]

    render_inputs = RenderInputs(
        xyz_canon=gauss_can.get_xyz,
        opacity_canon=gauss_can.get_opacity,
        rot_canon=gauss_can.get_rotation,
        scale_canon=gauss_can.get_scaling,
        sh_canon=gauss_can._features_dc,
        g_per_gauss=g_per_gauss,
        m_per_gauss=m_per_gauss,
    )
    type_soft = torch.sigmoid(type_logit)
    rgb_T3HW = render_21_with_warp(
        render_inputs, joint, phi_rev, phi_pri, camera,
        type_soft=type_soft, committed_type=None, sh_degree=0,
    )
    return rgb_T3HW, joint, r, b, g_obj, m_obj


# =============================================================================
# Optimizer factory (used by dual-clone)
# =============================================================================

def build_optimizer(learnable: StageDLearnable, cfg: StageDConfig
                    ) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        learnable.make_param_groups(),
        betas=(0.9, 0.999), eps=1.0e-8,
    )


def _optimizer_grad_diagnostics(
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for group_idx, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group{group_idx}"))
        total_sq = 0.0
        max_abs = 0.0
        params_with_grad = 0
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if grad.numel() == 0:
                continue
            params_with_grad += 1
            grad_norm = torch.linalg.vector_norm(grad.float()).item()
            total_sq += grad_norm * grad_norm
            max_abs = max(max_abs, float(grad.abs().max().item()))
        stats[f"grad_norm_{name}"] = total_sq ** 0.5
        stats[f"grad_max_abs_{name}"] = max_abs
        stats[f"grad_params_{name}"] = float(params_with_grad)
    return stats


# =============================================================================
# Multi-view keyframe context + per-iter camera / target selection
# =============================================================================

@dataclass
class MultiViewKeyframeContext:
    """Everything the per-iter multi-view camera/anchor sampler needs.

    Built once in ``train_stage_d_p1`` when multi-view is active and shared by
    both the main loop and the dual-clone loop so they sample identical cameras
    and re-condition Wan-Fun-InP the same way (sharp edge #1: the dual-clone path
    must replicate the MV logic or main_g1b silently reverts to single-view +
    real-GT).
    """
    m0: Any                                # state-0 (closed) TRELLIS Gaussian
    m5: Any                                # state-5 (open)   TRELLIS Gaussian
    object_center: Tuple[float, float, float]
    object_scale: float
    image_h: int
    image_w: int
    canonical_camera_cfg: StageDCameraConfig
    canonical_locked_camera: LockedCamera
    ref_first_canon: torch.Tensor          # [3, H, W] detached canonical M0 render
    ref_last_canon: torch.Tensor           # [3, H, W] detached canonical M5 render
    wan_cond_canonical: Dict[str, Any]     # prepared fun_inp cond for the canonical view
    cam_rng: torch.Generator               # dedicated CPU rng (cfg.mv_seed)


def _mv_camera_and_targets(
    mv: MultiViewKeyframeContext,
    cfg: StageDConfig,
    wan_ctx: WanRFSDSContext,
    device: torch.device,
    force_canonical: bool,
) -> Tuple[LockedCamera, torch.Tensor, torch.Tensor, Dict[str, Any], bool]:
    """Pick this iter's camera + keyframe anchor targets + Wan condition.

    Returns ``(locked_camera, ref_first, ref_last, wan_cond_iter, is_canonical)``.

    * canonical draw (or ``force_canonical`` in warmup): reuse the pre-built
      canonical camera, refs and Wan condition (no re-render, no cond rebuild).
    * random draw: build a locked camera for the sampled view, render M0/M5 from
      it (``.detach()`` -- the anchor target side of ``_pure_geometry_loss`` is NOT
      detached internally, so a non-detached ref would leak grad into the frozen
      keyframe Gaussians), and rebuild ONLY the Fun-InP ``y_guidance`` for the
      new reference frames (reuses cached in_context + seq_len).
    """
    cam_cfg, is_canon = sample_camera_for_iter(
        mv.cam_rng, cfg, mv.object_center, mv.object_scale,
        mv.image_h, mv.image_w, mv.canonical_camera_cfg,
    )
    if force_canonical:
        is_canon = True
    if is_canon:
        return (
            mv.canonical_locked_camera,
            mv.ref_first_canon,
            mv.ref_last_canon,
            mv.wan_cond_canonical,
            True,
        )
    locked_cam = build_locked_camera(cam_cfg, device=device, dtype=torch.float32)
    with torch.no_grad():
        ref_first = render_static_gaussians(
            mv.m0.get_xyz, mv.m0.get_opacity, mv.m0.get_rotation,
            mv.m0.get_scaling, mv.m0._features_dc, locked_cam,
        ).detach()
        ref_last = render_static_gaussians(
            mv.m5.get_xyz, mv.m5.get_opacity, mv.m5.get_rotation,
            mv.m5.get_scaling, mv.m5._features_dc, locked_cam,
        ).detach()
    wan_cond_iter = rebuild_fun_inp_y_for_keyframes(
        mv.wan_cond_canonical, ref_first, ref_last, wan_ctx,
    )
    return locked_cam, ref_first, ref_last, wan_cond_iter, False


def _build_wan_cond_canonical(
    bootstrap: BootstrapBundle,
    ref_first_canon: torch.Tensor,
    ref_last_canon: torch.Tensor,
    wan_ctx: WanRFSDSContext,
) -> Dict[str, Any]:
    """Prepare the canonical Fun-InP condition once (first/last = canonical refs).

    Shallow-copies ``bootstrap.wan_cond``, swaps the first/last conditioning
    frames to the canonical keyframe renders, drops any stale ``_prepared_backend``
    marker, then runs the (one-time, T5-to-CPU) prepare. Subsequent random-view
    iters reuse this via ``rebuild_fun_inp_y_for_keyframes`` (in_context + seq_len
    are camera-independent).
    """
    wc = dict(bootstrap.wan_cond)
    fv = wc["fun_video"].clone()
    fv[:, :, 0] = ref_first_canon.to(device=fv.device, dtype=fv.dtype)
    fv[:, :, -1] = ref_last_canon.to(device=fv.device, dtype=fv.dtype)
    wc["fun_video"] = fv
    wc.pop("_prepared_backend", None)
    return _prepare_fun_inp_rfsds_condition(wc, wan_ctx)


def _add_motion_ownership_loss(
    *,
    total_loss: torch.Tensor,
    log: Dict[str, float],
    cfg: StageDConfig,
    render_inputs: RenderInputs,
    joint: JointParams,
    phi_rev: torch.Tensor,
    phi_pri: torch.Tensor,
    camera: LockedCamera,
    ref_first_3HW: torch.Tensor,
    ref_last_3HW: torch.Tensor,
    committed_type: Optional[Literal["revolute", "prismatic"]],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if cfg.lambda_move_cover <= 0.0 and cfg.lambda_move_suppress <= 0.0:
        log["L_move_cover"] = 0.0
        log["L_move_suppress"] = 0.0
        return total_loss, None, None, None
    if committed_type is None:
        raise RuntimeError(
            "motion ownership requires a committed joint type; set "
            "cfg.force_committed_type or enable it only after type commit"
        )

    move_sil = render_move_silhouettes(
        render_inputs,
        joint,
        phi_rev,
        phi_pri,
        camera,
        frame_indices=(0, F_FRAMES - 1),
        committed_type=committed_type,
        sh_degree=0,
    )
    L_cover, L_suppress = loss_motion_ownership(
        move_sil,
        ref_first_3HW,
        ref_last_3HW,
        dilate_px=4,
        thresh=0.04,
    )
    total_loss = (
        total_loss
        + float(cfg.lambda_move_cover) * L_cover
        + float(cfg.lambda_move_suppress) * L_suppress
    )
    log["L_move_cover"] = float(L_cover.detach().item())
    log["L_move_suppress"] = float(L_suppress.detach().item())
    log["L_total"] = float(total_loss.detach().item())

    with torch.no_grad():
        ref_first = ref_first_3HW.to(device=move_sil.device, dtype=move_sil.dtype)
        ref_last = ref_last_3HW.to(device=move_sil.device, dtype=move_sil.dtype)
        dyn = dynamic_mask(ref_first, ref_last, dilate_px=4, thresh=0.04)
        fg = (
            ref_first.abs().sum(dim=0, keepdim=True)
            + ref_last.abs().sum(dim=0, keepdim=True)
        ).clamp(0.0, 1.0)
        static = fg * (1.0 - dyn)
    return total_loss, move_sil.detach(), dyn.detach(), static.detach()


# =============================================================================
# Main training driver
# =============================================================================

def train_stage_d_p1(
    bootstrap: BootstrapBundle,
    trellis: TrellisModules,
    wan_ctx: WanRFSDSContext,
    camera: LockedCamera,
    lpips_module: LPIPSModule,
    cfg: StageDConfig,
    out_dir: str,
    device: torch.device,
    keyframe_models: Optional[Dict[int, Any]] = None,
    object_center: Optional[Tuple[float, float, float]] = None,
    object_scale: Optional[float] = None,
    canonical_camera_cfg: Optional[StageDCameraConfig] = None,
    ref_first_canon: Optional[torch.Tensor] = None,
    ref_last_canon: Optional[torch.Tensor] = None,
    image_hw: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Run the full Stage D P1 schedule. Returns a summary dict.

    Schedule (with default boundaries):
      iter 0 .. 0.05 * total                  warmup_g_minus  (no SS-DiT, no W-RFSDS)
      iter 0.05 .. 0.10 * total               warmup_g0       (fixed t_ss + lambda_m_prior)
      iter 0.10 .. 0.50 * total               main_g1a        (two-branch, free type_logit)
      [type vote at iter 0.50 * total]
        if committed: main_g1b runs single-branch
        else        : dual-clone, run two clones in alternation
      iter 0.65 .. 0.75 * total               transition      (slow W-RFSDS down)

    Multi-view keyframe anchors (when ``cfg.mv_enable`` and ``keyframe_models``
    is provided): each iter samples a camera (``cfg.mv_canonical_ratio`` chance
    of the canonical view, else a random azimuth/elevation), renders the M0/M5
    keyframe Gaussians from it as the first/last pixel-anchor targets, and
    re-conditions Wan-Fun-InP on those reference frames. Warmup phases force the
    canonical view (no SDS there; avoids jitter on the Delta_z_s anchor). When
    the keyframe pieces are absent (single-view fallback) the loop keeps the
    canonical camera and the bootstrap real-GT s_0/s_5/wan_cond targets.
    """
    os.makedirs(out_dir, exist_ok=True)
    logs_path = os.path.join(out_dir, "logs.jsonl")
    logs_file = open(logs_path, "a", encoding="utf-8")

    learnable = StageDLearnable(
        cfg=cfg,
        n_obj=bootstrap.n_obj,
        psi_0=bootstrap.psi_0,
        phi_0=bootstrap.phi_0,
        M_attn_at_U=bootstrap.M_attn_at_U,
        device=device,
    ).to(device)

    ss_dit_w = SS_DiT_WithAdapters(
        base_ss_dit=trellis.ss_dit,
        adapters=learnable.adapters,
        capture_blocks=cfg.adapter_blocks,
    ).to(device)

    d_gs_w = DGSWithParent(d_gs_frozen=trellis.d_gs).to(device)

    optimizer = build_optimizer(learnable, cfg)

    type_vote_iter = int(round(cfg.f_main_g1a_end * cfg.total_iters))
    if cfg.force_committed_type is not None and cfg.force_committed_type not in (
        "revolute", "prismatic"
    ):
        raise ValueError(
            "cfg.force_committed_type must be None/'revolute'/'prismatic'; "
            f"got {cfg.force_committed_type!r}"
        )
    # When force_committed_type is set, the type is committed from iter 0: the
    # render uses a single branch (no revolute/prismatic blend phantom) and the
    # `it == type_vote_iter and committed_type is None` guard skips the vote.
    committed_type: Optional[Literal["revolute", "prismatic"]] = cfg.force_committed_type
    dual_clone: Optional[DualCloneState] = None
    silhouette_state_indices = default_check_state_indices(cfg.silhouette_n_states)

    # Dense Wan target is used only by W-RFSDS/latent supervision. Pixel
    # reconstruction supervision is six-state pure target only.
    wan_target_T3HW = bootstrap.wan_video_target_T3HW_01     # [F, 3, H, W] in [0, 1]
    pure_targets_K3HW = bootstrap.pure_state_targets_K3HW_01
    lambda_last_active = cfg.lambda_last if str(cfg.wan_backend) == "fun_inp" else 0.0

    base_anchor_count = int(bootstrap.base_anchor_at_U.detach().sum().item())
    base_anchor_frac = float(base_anchor_count) / float(max(1, bootstrap.n_obj))
    summary: Dict[str, Any] = {
        "committed_type": None,
        "n_iters_run": 0,
        "base_anchor_count": base_anchor_count,
        "base_anchor_frac": base_anchor_frac,
    }

    # ------------------------------------------------------------------ #
    # Multi-view keyframe context (built once; shared with dual-clone).   #
    # Active only when cfg.mv_enable, keyframe_models are provided, and    #
    # the Fun-InP Wan teacher is loaded (the keyframe condition rebuild    #
    # is Fun-InP specific). Otherwise the loop falls back to single-view   #
    # with the bootstrap real-GT anchors / condition.                     #
    # ------------------------------------------------------------------ #
    mv_ctx: Optional[MultiViewKeyframeContext] = None
    mv_active = (
        bool(getattr(cfg, "mv_enable", False))
        and keyframe_models is not None
        and wan_ctx is not None
        and str(cfg.wan_backend) == "fun_inp"
    )
    if mv_active:
        if (object_center is None or object_scale is None
                or canonical_camera_cfg is None
                or ref_first_canon is None or ref_last_canon is None
                or image_hw is None):
            raise ValueError(
                "mv_enable=True requires object_center/object_scale/"
                "canonical_camera_cfg/ref_first_canon/ref_last_canon/image_hw"
            )
        if 0 not in keyframe_models or 5 not in keyframe_models:
            raise ValueError(
                f"keyframe_models must contain states 0 and 5; got "
                f"{sorted(keyframe_models.keys())}"
            )
        wan_cond_canonical = _build_wan_cond_canonical(
            bootstrap, ref_first_canon, ref_last_canon, wan_ctx,
        )
        cam_rng = torch.Generator(device="cpu").manual_seed(int(cfg.mv_seed))
        mv_ctx = MultiViewKeyframeContext(
            m0=keyframe_models[0],
            m5=keyframe_models[5],
            object_center=tuple(float(c) for c in object_center),
            object_scale=float(object_scale),
            image_h=int(image_hw[0]),
            image_w=int(image_hw[1]),
            canonical_camera_cfg=canonical_camera_cfg,
            canonical_locked_camera=camera,
            ref_first_canon=ref_first_canon.detach(),
            ref_last_canon=ref_last_canon.detach(),
            wan_cond_canonical=wan_cond_canonical,
            cam_rng=cam_rng,
        )
        logger.info(
            "[stage_d] multi-view keyframe anchors ENABLED "
            "(canonical_ratio=%.2f azi=[%.1f,%.1f] ele=[%.1f,%.1f] seed=%d)",
            cfg.mv_canonical_ratio, cfg.mv_azi_min_deg, cfg.mv_azi_max_deg,
            cfg.mv_ele_min_deg, cfg.mv_ele_max_deg, int(cfg.mv_seed),
        )
        summary["mv_enabled"] = True
        summary["mv_canon_iters"] = 0
        summary["mv_random_iters"] = 0
    else:
        summary["mv_enabled"] = False

    # ============================================================ #
    # Iter-0 camera sanity check (fail-loud on camera mismatch).    #
    # Render frame 0 of the current 3D (no learnable updates yet)   #
    # and compare its silhouette to Wan-canonical s_0. Mismatch ->  #
    # clear CameraMismatchError pointing at StageDCameraConfig.     #
    # ============================================================ #
    os.makedirs(os.path.join(out_dir, "viz"), exist_ok=True)
    with torch.no_grad():
        rgb_T3HW_init, _joint_init, _r_init, _b_init, _g_init, _m_init = (
            _warmup_minus_forward(
                learnable=learnable, ss_vae_decoder=trellis.ss_vae_decoder,
                d_gs_w=d_gs_w, bootstrap=bootstrap, camera=camera, cfg=cfg,
            )
        )
        sparse_in_init = _build_sparse_in(bootstrap.z_slat0, bootstrap.U_object_with_batch)
        gauss_init, parent_idx_init = d_gs_w(sparse_in_init)
        canonical_rgb = render_static_gaussians(
            gauss_init.get_xyz,
            gauss_init.get_opacity,
            gauss_init.get_rotation,
            gauss_init.get_scaling,
            gauss_init._features_dc,
            camera,
        )
        support_rgb = render_static_gaussians(
            gauss_init.get_xyz,
            gauss_init.get_opacity,
            gauss_init.get_rotation,
            gauss_init.get_scaling,
            gauss_init._features_dc,
            camera,
            opacity_scale=torch.sigmoid(learnable.alpha_g.detach())[parent_idx_init],
        )
        support_scale = torch.sigmoid(learnable.alpha_g.detach())[parent_idx_init]
        move_scale = torch.sigmoid(learnable.alpha_m.detach())[parent_idx_init]
        base_only_rgb = render_static_gaussians(
            gauss_init.get_xyz,
            gauss_init.get_opacity,
            gauss_init.get_rotation,
            gauss_init.get_scaling,
            gauss_init._features_dc,
            camera,
            opacity_scale=support_scale * (1.0 - move_scale),
        )
        move_only_rgb = render_static_gaussians(
            gauss_init.get_xyz,
            gauss_init.get_opacity,
            gauss_init.get_rotation,
            gauss_init.get_scaling,
            gauss_init._features_dc,
            camera,
            opacity_scale=support_scale * move_scale,
        )
        summary["no_learning_gs_ablation"] = save_no_learning_gs_ablation(
            out_root=out_dir,
            canonical_3HW_01=canonical_rgb,
            support_3HW_01=support_rgb,
            base_only_3HW_01=base_only_rgb,
            move_only_3HW_01=move_only_rgb,
            s0_pure_3HW_01=bootstrap.s_0_pure_3HW_01,
        )
        contract_metrics = save_initial_render_contract_diagnostics(
            out_root=out_dir,
            canonical_3HW_01=canonical_rgb,
            support_3HW_01=support_rgb,
            initial_warp_frame0_3HW_01=rgb_T3HW_init[0],
            s0_pure_3HW_01=bootstrap.s_0_pure_3HW_01,
            sc_pure_3HW_01=bootstrap.pure_state_targets_K3HW_01[CANONICAL_STATE_IDX],
            canonical_state_idx=CANONICAL_STATE_IDX,
        )
        summary["initial_render_contract"] = contract_metrics
        iter0_iou = iter_0_camera_check(
            rendered_frame_0_3HW=rgb_T3HW_init[0],
            s_0_pure_3HW=bootstrap.s_0_pure_3HW_01,
            iou_threshold=float(cfg.iter0_camera_iou_threshold),
            save_diag_to=os.path.join(out_dir, "viz", "iter_0_camera_diag.png"),
        )
        summary["iter_0_camera_iou"] = iter0_iou
        # Free init render before the real loop begins (cheap; ~34MB on 832x464x21).
        del rgb_T3HW_init

    if bool(cfg.render_contract_only):
        summary["render_contract_only"] = True
        logs_file.close()
        return summary

    p1_stop_iter = int(round(cfg.f_transition_end * cfg.total_iters))
    if p1_stop_iter <= 0:
        raise ValueError(
            f"f_transition_end * total_iters must be positive, got "
            f"{cfg.f_transition_end} * {cfg.total_iters}"
        )

    # ---- Iter-0 snapshot: pristine pre-optimization baseline for comparison ----
    # The main loop's viz is guarded against phase warmup_g_minus, so it==0 never
    # produced an iter_000000/. Mirror the loop's it==0 path EXACTLY:
    # warmup_g_minus uses _warmup_minus_forward (NOT _regular_forward, which
    # raises there since sample_t_ss returns None). At f_global=0 the head
    # lambdas are 0, so this is the identical initial state the run starts from.
    # Save/restore RNG around it so the snapshot does not perturb the training
    # RNG stream (keeps the run comparable to a no-snapshot baseline).
    _cpu_rng_state = torch.get_rng_state()
    _cuda_rng_state = torch.cuda.get_rng_state_all()
    with torch.no_grad():
        rgb0_T3HW, joint0, r0, b0, _g0_obj, _m0_obj = _warmup_minus_forward(
            learnable=learnable,
            ss_vae_decoder=trellis.ss_vae_decoder,
            d_gs_w=d_gs_w,
            bootstrap=bootstrap,
            camera=camera,
            cfg=cfg,
        )
        u0_shifted = bootstrap.phi_0.to(device)
        g0_soft = torch.sigmoid(r0).detach()
        m0_soft = torch.sigmoid(b0).detach()
        phi0_rev = (u0_shifted * joint0.theta_max).detach()
        phi0_pri = (u0_shifted * joint0.disp_max).detach()
        save_iter_snapshot(
            out_dir, 0,
            rgb_T3HW_01=rgb0_T3HW.detach(),
            pure_state_targets_K3HW_01=pure_targets_K3HW,
            metrics={"f_global": 0.0, "note": "iter0_pristine_pre_optimization"},
            U_object=bootstrap.U_object.cpu().numpy(),
            g_per_voxel=g0_soft,
            m_per_voxel=m0_soft,
        )
        save_phi_curve(
            out_dir, 0,
            u_shifted=u0_shifted.detach(),
            phi_rev=phi0_rev,
            phi_pri=phi0_pri,
        )
        type0_str = ("prismatic" if joint0.type_logit.item() > 0.0
                     else "revolute")
        save_3d_state_html(
            out_dir, 0,
            U_object_np=bootstrap.U_object.cpu().numpy(),
            g_per_voxel_np=g0_soft.cpu().numpy(),
            m_per_voxel_np=m0_soft.cpu().numpy(),
            joint_axis_world_np=joint0.axis.detach().cpu().numpy(),
            joint_origin_world_np=joint0.origin.detach().cpu().numpy(),
            joint_type=type0_str,
        )
        save_phi_curve_html(
            out_dir, 0,
            u_shifted_np=u0_shifted.detach().cpu().numpy(),
            phi_rev_np=phi0_rev.cpu().numpy(),
            phi_pri_np=phi0_pri.cpu().numpy(),
            type_soft=float(torch.sigmoid(joint0.type_logit).item()),
        )
    torch.set_rng_state(_cpu_rng_state)
    torch.cuda.set_rng_state_all(_cuda_rng_state)

    for it in range(p1_stop_iter):
        consumed_until_marker = summary.get("_dual_clone_consumed_until")
        if consumed_until_marker is not None and it < int(consumed_until_marker):
            continue

        f_global = it / max(1, cfg.total_iters - 1)
        phase = phase_of(f_global, cfg)

        # ---- Multi-view: pick this iter's camera + keyframe anchor targets ----
        # Warmup phases (warmup_g_minus / warmup_g0) force the canonical view:
        # no SDS runs there, and jittering the camera would perturb the
        # Delta_z_s anchor. Random views only kick in once the SDS path is live.
        is_warmup_phase = phase in ("warmup_g_minus", "warmup_g0")
        if mv_ctx is not None:
            (cam_iter, ref_first_iter, ref_last_iter,
             wan_cond_iter, is_canon) = _mv_camera_and_targets(
                mv_ctx, cfg, wan_ctx, device, force_canonical=is_warmup_phase,
            )
            if is_canon:
                summary["mv_canon_iters"] += 1
            else:
                summary["mv_random_iters"] += 1
        else:
            cam_iter = camera
            ref_first_iter = bootstrap.s_0_pure_3HW_01
            ref_last_iter = bootstrap.s_5_pure_3HW_01
            wan_cond_iter = bootstrap.wan_cond
            is_canon = True

        # ---- Forward ----
        if phase == "warmup_g_minus":
            rgb_T3HW, joint, r, b, g_obj, m_obj = _warmup_minus_forward(
                learnable=learnable, ss_vae_decoder=trellis.ss_vae_decoder,
                d_gs_w=d_gs_w, bootstrap=bootstrap, camera=cam_iter, cfg=cfg,
            )
            u_shifted = bootstrap.phi_0.to(device)
            # Warmup G- runs no heads / no SS-DiT, so there is no head pre-activation
            # to leash (the adapter is intentionally inactive here).
            head_preact_l2 = None
            head_stats = None
            render_inputs = None
            ownership_render_inputs = None
            phi_rev = None
            phi_pri = None
        else:
            (
                rgb_T3HW, joint, r, b, g_obj, m_obj, u_shifted, head_preact_l2,
                head_stats, render_inputs, ownership_render_inputs, phi_rev, phi_pri,
            ) = _regular_forward(
                learnable=learnable,
                ss_dit_w=ss_dit_w,
                ss_vae_decoder=trellis.ss_vae_decoder,
                d_gs_w=d_gs_w,
                bootstrap=bootstrap,
                camera=cam_iter,
                f_global=f_global,
                cfg=cfg,
                committed_type=committed_type,
            )

        # ---- Loss schedule sample ----
        if phase != "warmup_g_minus":
            tau = sample_tau_chord_anneal(
                iter_idx=it,
                total_iters=p1_stop_iter,
                mean=cfg.tau_logit_normal_mean_p1,
                std=cfg.tau_logit_normal_std_p1,
                device=device,
            )
            cfg_scale = schedule_cfg(f_global, cfg)
        else:
            # Warmup G- does not invoke W-RFSDS at all; tau / cfg_scale unused.
            tau = 0.5
            cfg_scale = 0.0

        # ---- Build LossInputs ----
        loss_inputs = LossInputs(
            rgb_T3HW=rgb_T3HW,
            r=r, b=b,
            axis=joint.axis, origin=joint.origin,
            Delta_z_s=learnable.Delta_z_s,
            alpha_m=learnable.alpha_m,
            pure_state_targets_K3HW_01=pure_targets_K3HW,
            s_0_pure_3HW_01=ref_first_iter,
            s_5_pure_3HW_01=ref_last_iter,
            z_wan_target=bootstrap.z_wan_target,
            anchors_world=bootstrap.anchors_world,
            shell_mask=bootstrap.slat_shell_mask,
            m_attn_at_U=bootstrap.M_attn_at_U,
            wan_cond=wan_cond_iter,
            tau=tau,
            cfg_scale=cfg_scale,
        )

        sched_lambdas = schedule_w_rfsds_weights(f_global, cfg)
        sched_lam_shell = schedule_lambda_shell(f_global, cfg)
        sched_lam_m_prior = schedule_lambda_m_prior(f_global, cfg)
        sched_lam_gate = schedule_lambda_gate(f_global, cfg)

        total_loss, log = aggregate_loss(
            loss_inputs, wan_ctx, lpips_module,
            cfg_lambdas_first=cfg.lambda_first,
            cfg_lambdas_last=lambda_last_active,
            cfg_lambdas_contact=cfg.lambda_contact,
            cfg_lambdas_gate=sched_lam_gate,
            cfg_lambdas_z=cfg.lambda_z,
            sched_lambdas_w_rfsds=sched_lambdas,
            sched_lambda_shell=sched_lam_shell,
            sched_lambda_m_prior=sched_lam_m_prior,
            cfg_lambda_move_floor=cfg.lambda_move_floor,
            move_floor_frac=cfg.move_floor_frac,
            cfg_lambda_move_ceiling=cfg.lambda_move_ceiling,
            move_ceiling_frac=cfg.move_ceiling_frac,
        )

        ownership_move_sil = None
        ownership_dyn = None
        ownership_static = None
        if phase != "warmup_g_minus":
            if ownership_render_inputs is None or phi_rev is None or phi_pri is None:
                raise RuntimeError("regular phase missing render inputs for ownership loss")
            (
                total_loss,
                ownership_move_sil,
                ownership_dyn,
                ownership_static,
            ) = _add_motion_ownership_loss(
                total_loss=total_loss,
                log=log,
                cfg=cfg,
                render_inputs=ownership_render_inputs,
                joint=joint,
                phi_rev=phi_rev,
                phi_pri=phi_pri,
                camera=cam_iter,
                ref_first_3HW=ref_first_iter,
                ref_last_3HW=ref_last_iter,
                committed_type=committed_type,
            )
        else:
            log["L_move_cover"] = 0.0
            log["L_move_suppress"] = 0.0

        # ---- Head pre-activation leash. ----
        if head_preact_l2 is not None and cfg.lambda_head_leash > 0.0:
            total_loss = total_loss + cfg.lambda_head_leash * head_preact_l2
            log["L_head_leash"] = float(head_preact_l2.detach().item())
        if head_stats is not None:
            for key, value in head_stats.items():
                log[key] = float(value.detach().item())

        # ---- Backward ----
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_log = _optimizer_grad_diagnostics(optimizer)
        if (
            cfg.require_live_head_adapter_grads
            and f_global >= float(cfg.live_grad_check_start_frac)
            and (
                grad_log.get("grad_norm_head", 0.0) <= float(cfg.live_grad_min_norm)
                or grad_log.get("grad_norm_adapter", 0.0) <= float(cfg.live_grad_min_norm)
            )
        ):
            raise RuntimeError(
                "dead head/adapter gradient path: "
                f"grad_norm_head={grad_log.get('grad_norm_head', 0.0):.3e}, "
                f"grad_norm_adapter={grad_log.get('grad_norm_adapter', 0.0):.3e}, "
                f"f_global={f_global:.3f}"
            )
        if cfg.grad_clip_norm > 0:
            clip_total_norm = torch.nn.utils.clip_grad_norm_(
                [p for grp in optimizer.param_groups for p in grp["params"]],
                cfg.grad_clip_norm,
            )
            grad_log["grad_clip_total_norm"] = float(clip_total_norm.detach().item())
        optimizer.step()
        log.update(grad_log)
        log["base_anchor_count"] = float(base_anchor_count)
        log["base_anchor_frac"] = base_anchor_frac
        if base_anchor_count > 0:
            anchor_mask = bootstrap.base_anchor_at_U.to(device=b.device, dtype=torch.bool)
            log["base_anchor_move_prob"] = float(
                torch.sigmoid(b[anchor_mask]).detach().mean().item()
            )
        else:
            log["base_anchor_move_prob"] = 0.0

        # ---- Periodic: log ----
        if it % cfg.log_every == 0:
            snap = schedule_snapshot(f_global, cfg)
            line = {
                "it": it, "f_global": f_global, "tau": float(tau),
                "mv_is_canon": bool(is_canon), "mv_active": bool(mv_ctx is not None),
                "axis": joint.axis.detach().cpu().tolist(),
                "origin": joint.origin.detach().cpu().tolist(),
                "theta_max_deg": float(joint.theta_max.detach().cpu()) * 57.29578,
                **snap, **log,
            }
            logs_file.write(json.dumps(line) + "\n")
            logs_file.flush()
            logger.info(
                "[stage_d it=%d phase=%s is_canon=%s tau=%.3f cfg=%.1f L_total=%.4f "
                "L_first=%.4f L_last=%.4f]",
                it, phase, is_canon, tau, cfg_scale, log["L_total"],
                log.get("L_first", 0.0), log.get("L_last", 0.0),
            )

        # ---- Periodic: viz ----
        if it % cfg.viz_every == 0 and phase != "warmup_g_minus":
            with torch.no_grad():
                g_soft = torch.sigmoid(r).detach()
                m_soft_viz = torch.sigmoid(b).detach()
                phi_rev_viz = (u_shifted * joint.theta_max).detach()
                phi_pri_viz = (u_shifted * joint.disp_max).detach()

                # 1) PNG strips + metrics.json + gates.npz
                save_iter_snapshot(
                    out_dir, it,
                    rgb_T3HW_01=rgb_T3HW.detach(),
                    pure_state_targets_K3HW_01=pure_targets_K3HW,
                    metrics={"f_global": f_global, **log},
                    U_object=bootstrap.U_object.cpu().numpy(),
                    g_per_voxel=g_soft,
                    m_per_voxel=m_soft_viz,
                    base_anchor_per_voxel=bootstrap.base_anchor_at_U,
                )
                if (
                    ownership_move_sil is not None
                    and ownership_dyn is not None
                    and ownership_static is not None
                ):
                    save_motion_ownership_debug(
                        out_dir,
                        it,
                        move_sil_K1HW=ownership_move_sil,
                        dynamic_mask_1HW=ownership_dyn,
                        static_mask_1HW=ownership_static,
                        ref_first_3HW=ref_first_iter,
                        ref_last_3HW=ref_last_iter,
                    )
                # 2) phi npz (raw)
                save_phi_curve(
                    out_dir, it,
                    u_shifted=u_shifted.detach(),
                    phi_rev=phi_rev_viz,
                    phi_pri=phi_pri_viz,
                )
                # 3) 3D interactive HTML: base/move overlay + joint axis
                # joint_type for axis viz: use committed type if available,
                # else pick by current type_logit (just for visualization).
                if committed_type is None:
                    type_str = ("prismatic" if joint.type_logit.item() > 0.0
                                else "revolute")
                else:
                    type_str = committed_type
                save_3d_state_html(
                    out_dir, it,
                    U_object_np=bootstrap.U_object.cpu().numpy(),
                    g_per_voxel_np=g_soft.cpu().numpy(),
                    m_per_voxel_np=m_soft_viz.cpu().numpy(),
                    joint_axis_world_np=joint.axis.detach().cpu().numpy(),
                    joint_origin_world_np=joint.origin.detach().cpu().numpy(),
                    joint_type=type_str,
                )
                # 4) phi rollout HTML (line plot)
                save_phi_curve_html(
                    out_dir, it,
                    u_shifted_np=u_shifted.detach().cpu().numpy(),
                    phi_rev_np=phi_rev_viz.cpu().numpy(),
                    phi_pri_np=phi_pri_viz.cpu().numpy(),
                    type_soft=float(torch.sigmoid(joint.type_logit).item()),
                )

        # ---- Periodic: silhouette check (raises on failure) ----
        # ★ Sharp edge #4: this compares the render to the 6 REAL pure states,
        # which were captured from the canonical camera. On a random-view iter
        # the render is from a different camera, so the comparison would
        # false-raise. Only run it on canonical iters.
        if (cfg.silhouette_check_every > 0
                and phase != "warmup_g_minus"
                and is_canon
                and it > 0 and it % cfg.silhouette_check_every == 0):
            with torch.no_grad():
                periodic_silhouette_check(
                    iter_idx=it,
                    cfg_check_every=cfg.silhouette_check_every,
                    cfg_iou_threshold=cfg.silhouette_iou_threshold,
                    cfg_n_states=cfg.silhouette_n_states,
                    sample_state_indices=silhouette_state_indices,
                    rendered_T3HW_01=rgb_T3HW.detach(),
                    pure_state_targets_K3HW_01=pure_targets_K3HW,
                )

        # ---- Type vote at G1a end ----
        if it == type_vote_iter and committed_type is None:
            # ★ M4 fix: the vote closure must replicate _regular_forward's
            # exact joint pathway using the SAME schedule-driven lambdas so
            # the eval predicts the same psi the regular forward would. The
            # old version hardcoded 0.3 / 0.5; that's off-schedule and gave
            # a biased vote (votes correlated with the hardcoded weights,
            # not with the current learned lambdas).
            f_vote = float(type_vote_iter) / max(1, cfg.total_iters - 1)
            l_sup_v, l_part_v, l_joint_v = schedule_head_lambdas(f_vote, cfg)

            def _single_pass(t_ss_val: float, seed: int) -> JointParams:
                gen = torch.Generator(device=device).manual_seed(int(seed))
                eps = torch.randn(bootstrap.z_s0.shape, generator=gen,
                                  device=device, dtype=bootstrap.z_s0.dtype)
                z_s_base = bootstrap.z_s0 + learnable.Delta_z_s.unsqueeze(0)
                z_t = ((1.0 - t_ss_val) * z_s_base
                       + (1.0e-5 + (1.0 - 1.0e-5) * t_ss_val) * eps)
                _pred_v, captured = ss_dit_w.forward_capture(
                    z_t, t_raw=t_ss_val, cond=bootstrap.trellis_cond_can,
                )
                occ_logits = trellis.ss_vae_decoder(z_s_base)
                feat, occ_at_U = sample_hidden_at_U(
                    captured, bootstrap.U_object, occ_logits,
                    capture_blocks=cfg.adapter_blocks,
                    fourier_num_freqs=cfg.fourier_num_freqs,
                )
                # Match _regular_forward's b = alpha_m + lambda_part * H_part(feat).
                H_part_vote = cfg.head_logit_residual_bound * torch.tanh(
                    learnable.H_part(feat).squeeze(-1)
                )
                b_logit = learnable.alpha_m + l_part_v * H_part_vote
                b_logit = _apply_base_anchor_to_move_logits(b_logit, bootstrap, cfg)
                m_soft = torch.sigmoid(b_logit)
                F_pool = weighted_feature_pool(feat, m_soft, eps=cfg.sgs_eps)
                delta_psi = learnable.H_joint(F_pool)
                # Match _regular_forward's psi_pred = project(psi_param + lambda_joint * H_joint).
                psi_combined = learnable.psi_param + l_joint_v * delta_psi
                return project_joint(psi_combined, eps=cfg.sgs_eps)

            vote = run_type_vote(learnable, cfg, _single_pass)
            if vote.committed:
                committed_type = vote.committed_type
                # Freeze psi_param[6] to the committed value (huge magnitude logit).
                with torch.no_grad():
                    learnable.psi_param[6].fill_(
                        +10.0 if committed_type == "prismatic" else -10.0
                    )
                summary["committed_type"] = committed_type
                summary["type_vote"] = {
                    "mean_logit": vote.mean_type_logit,
                    "confidence": vote.confidence, "via": "direct",
                }
            else:
                # Dual-clone path: build two clones and run them to G1b end.
                summary["type_vote"] = {
                    "mean_logit": vote.mean_type_logit,
                    "confidence": vote.confidence, "via": "dual_clone",
                }
                dual_clone = make_dual_clones(
                    learnable=learnable,
                    optimizer_state_dict=optimizer.state_dict(),
                    cfg=cfg,
                    optimizer_factory=lambda lr: build_optimizer(lr, cfg),
                )
                logger.info(
                    "[stage_d] entering dual-clone branches for "
                    "(%.0f -> %.0f) iters each",
                    type_vote_iter,
                    cfg.f_main_g1b_end * cfg.total_iters,
                )
                committed_type = _run_dual_clone_to_completion(
                    dual_clone=dual_clone,
                    start_iter=it + 1,
                    end_iter=int(round(cfg.f_main_g1b_end * cfg.total_iters)),
                    cfg=cfg,
                    trellis=trellis, ss_dit_w_factory=lambda lr: SS_DiT_WithAdapters(
                        base_ss_dit=trellis.ss_dit, adapters=lr.adapters,
                        capture_blocks=cfg.adapter_blocks,
                    ).to(device),
                    d_gs_w=d_gs_w,
                    bootstrap=bootstrap,
                    camera=camera,
                    wan_ctx=wan_ctx,
                    lpips_module=lpips_module,
                    device=device,
                    out_dir=out_dir,
                    summary=summary,
                    mv_ctx=mv_ctx,
                )
                summary["committed_type"] = committed_type
                # Replace main learnable + optimizer with the winning clone.
                learnable = (dual_clone.learnable_rev if committed_type == "revolute"
                             else dual_clone.learnable_pri)
                optimizer = (dual_clone.optimizer_rev if committed_type == "revolute"
                              else dual_clone.optimizer_pri)
                ss_dit_w = SS_DiT_WithAdapters(
                    base_ss_dit=trellis.ss_dit, adapters=learnable.adapters,
                    capture_blocks=cfg.adapter_blocks,
                ).to(device)
                # Fast-forward iter counter past G1b (dual-clone consumed it).
                # The outer for-loop will continue from it+1; main_g1b is done.
                # We jump iteration ahead to f_main_g1b_end:
                # (cannot directly modify the for var; we'll just let phase_of
                # naturally route subsequent iters to "transition" once it
                # > f_main_g1b_end * total_iters; given current it is
                # type_vote_iter, we manually break out of normal main_g1
                # below.)
                # We achieve "skip ahead" by manually advancing the iter loop.
                # (Python for-loop var assignment doesn't affect iteration;
                # instead we'll let _run_dual_clone_to_completion have done
                # iters [start_iter, end_iter), and the outer loop continues
                # from start_iter onwards but will skip them at the top of
                # the loop, before any forward/backward is run.)
                consumed_until = int(round(cfg.f_main_g1b_end * cfg.total_iters))
                summary["_dual_clone_consumed_until"] = consumed_until
                continue

        # ---- Periodic checkpoint ----
        if it > 0 and it % cfg.save_checkpoint_every == 0:
            torch.save(
                {"learnable_state": learnable.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "committed_type": committed_type, "iter": it},
                os.path.join(out_dir, f"ckpt_{it:06d}.pt"),
            )

        summary["n_iters_run"] = it + 1

    # ---- Final P1 artifacts ----
    summary["committed_type"] = committed_type
    final_path = os.path.join(out_dir, "learnable_p1_final.pt")
    torch.save(
        {"learnable_state": learnable.state_dict(),
         "committed_type": committed_type,
         "cfg": cfg.__dict__,
         "summary": summary},
        final_path,
    )

    # ---- Final P1 visualization (PNG + 3D HTML deliverables) ----
    with torch.no_grad():
        final_rgb, final_joint, final_r, final_b, _, _, _, _, _, _, _, _, _ = _regular_forward(
            learnable=learnable, ss_dit_w=ss_dit_w,
            ss_vae_decoder=trellis.ss_vae_decoder, d_gs_w=d_gs_w,
            bootstrap=bootstrap, camera=camera,
            f_global=1.0, cfg=cfg, committed_type=committed_type,
        )
        save_p1_final_summary(
            out_root=out_dir,
            final_rgb_T3HW_01=final_rgb.detach(),
            pure_state_targets_K3HW_01=pure_targets_K3HW,
            U_object_np=bootstrap.U_object.cpu().numpy(),
            g_per_voxel_np=torch.sigmoid(final_r).detach().cpu().numpy(),
            m_per_voxel_np=torch.sigmoid(final_b).detach().cpu().numpy(),
            joint_axis_world_np=final_joint.axis.detach().cpu().numpy(),
            joint_origin_world_np=final_joint.origin.detach().cpu().numpy(),
            committed_type=committed_type or "uncommitted",
            summary={
                k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in summary.items()
            },
        )

    # Close log file BEFORE reading it for loss curves (Windows-safe).
    logs_file.close()

    # ---- Loss / schedule curves HTML (from logs.jsonl) ----
    save_loss_curves_html(out_root=out_dir)

    logger.info("[stage_d] P1 done; wrote %s", final_path)
    return summary


# =============================================================================
# Dual-clone inner runner (alternates rev / pri updates over G1b range)
# =============================================================================

def _run_dual_clone_to_completion(
    *,
    dual_clone: DualCloneState,
    start_iter: int,
    end_iter: int,
    cfg: StageDConfig,
    trellis: TrellisModules,
    ss_dit_w_factory,
    d_gs_w: DGSWithParent,
    bootstrap: BootstrapBundle,
    camera: LockedCamera,
    wan_ctx: WanRFSDSContext,
    lpips_module: LPIPSModule,
    device: torch.device,
    out_dir: str,
    summary: Dict[str, Any],
    mv_ctx: Optional["MultiViewKeyframeContext"] = None,
) -> Literal["revolute", "prismatic"]:
    """Alternate-update both clones across ``[start_iter, end_iter)``.

    Each clone gets every-other iter; both run to ``end_iter`` so we
    compare apples-to-apples on the final L_sds + L_rgb_rec. The total
    wall-clock budget is the same as the regular G1b would have used
    (each clone runs at half rate, end at the same point).

    ★ Sharp edge #1: this is a SECOND render+loss path (main_g1b, ~15% of
    iters when the type vote is undecided). It MUST replicate the main loop's
    multi-view camera-sample + M0/M5 render + condition rebuild + LossInputs
    swap, or main_g1b silently reverts to single-view + real-GT targets. The
    dual-clone runs only in main_g1b (never warmup), so the camera is never
    force-canonical here.
    """
    ss_dit_w_rev = ss_dit_w_factory(dual_clone.learnable_rev)
    ss_dit_w_pri = ss_dit_w_factory(dual_clone.learnable_pri)

    for it in range(start_iter, end_iter):
        is_pri_iter = (it % 2 == 1)
        f_global = it / max(1, cfg.total_iters - 1)

        if is_pri_iter:
            learnable_cur = dual_clone.learnable_pri
            opt_cur = dual_clone.optimizer_pri
            ss_dit_w_cur = ss_dit_w_pri
            forced_logit = dual_clone.forced_type_logit_pri
            committed_type: Literal["revolute", "prismatic"] = "prismatic"
        else:
            learnable_cur = dual_clone.learnable_rev
            opt_cur = dual_clone.optimizer_rev
            ss_dit_w_cur = ss_dit_w_rev
            forced_logit = dual_clone.forced_type_logit_rev
            committed_type = "revolute"

        # ---- Multi-view: same per-iter camera/target selection as main loop ----
        if mv_ctx is not None:
            (cam_iter, ref_first_iter, ref_last_iter,
             wan_cond_iter, is_canon) = _mv_camera_and_targets(
                mv_ctx, cfg, wan_ctx, device, force_canonical=False,
            )
            if is_canon:
                summary["mv_canon_iters"] = summary.get("mv_canon_iters", 0) + 1
            else:
                summary["mv_random_iters"] = summary.get("mv_random_iters", 0) + 1
        else:
            cam_iter = camera
            ref_first_iter = bootstrap.s_0_pure_3HW_01
            ref_last_iter = bootstrap.s_5_pure_3HW_01
            wan_cond_iter = bootstrap.wan_cond

        (
            rgb_T3HW, joint, r, b, g_obj, m_obj, u_shifted, head_preact_l2,
            head_stats, render_inputs, ownership_render_inputs, phi_rev, phi_pri,
        ) = _regular_forward(
            learnable=learnable_cur,
            ss_dit_w=ss_dit_w_cur,
            ss_vae_decoder=trellis.ss_vae_decoder,
            d_gs_w=d_gs_w,
            bootstrap=bootstrap,
            camera=cam_iter,
            f_global=f_global,
            cfg=cfg,
            committed_type=committed_type,
            forced_type_logit=forced_logit,
        )

        tau = sample_tau_chord_anneal(
            iter_idx=it,
            total_iters=int(round(cfg.f_transition_end * cfg.total_iters)),
            mean=cfg.tau_logit_normal_mean_p1,
            std=cfg.tau_logit_normal_std_p1,
            device=device,
        )
        cfg_scale = schedule_cfg(f_global, cfg)
        sched_lambdas = schedule_w_rfsds_weights(f_global, cfg)
        sched_lam_shell = schedule_lambda_shell(f_global, cfg)
        sched_lam_m_prior = schedule_lambda_m_prior(f_global, cfg)
        sched_lam_gate = schedule_lambda_gate(f_global, cfg)

        loss_inputs = LossInputs(
            rgb_T3HW=rgb_T3HW, r=r, b=b,
            axis=joint.axis, origin=joint.origin,
            Delta_z_s=learnable_cur.Delta_z_s,
            alpha_m=learnable_cur.alpha_m,
            pure_state_targets_K3HW_01=bootstrap.pure_state_targets_K3HW_01,
            s_0_pure_3HW_01=ref_first_iter,
            s_5_pure_3HW_01=ref_last_iter,
            z_wan_target=bootstrap.z_wan_target,
            anchors_world=bootstrap.anchors_world,
            shell_mask=bootstrap.slat_shell_mask,
            m_attn_at_U=bootstrap.M_attn_at_U,
            wan_cond=wan_cond_iter,
            tau=tau, cfg_scale=cfg_scale,
        )
        total_loss, log = aggregate_loss(
            loss_inputs, wan_ctx, lpips_module,
            cfg_lambdas_first=cfg.lambda_first,
            cfg_lambdas_last=(cfg.lambda_last if str(cfg.wan_backend) == "fun_inp" else 0.0),
            cfg_lambdas_contact=cfg.lambda_contact,
            cfg_lambdas_gate=sched_lam_gate,
            cfg_lambdas_z=cfg.lambda_z,
            sched_lambdas_w_rfsds=sched_lambdas,
            sched_lambda_shell=sched_lam_shell,
            sched_lambda_m_prior=sched_lam_m_prior,
            cfg_lambda_move_floor=cfg.lambda_move_floor,
            move_floor_frac=cfg.move_floor_frac,
            cfg_lambda_move_ceiling=cfg.lambda_move_ceiling,
            move_ceiling_frac=cfg.move_ceiling_frac,
        )
        total_loss, _move_sil, _dyn, _static = _add_motion_ownership_loss(
            total_loss=total_loss,
            log=log,
            cfg=cfg,
            render_inputs=ownership_render_inputs,
            joint=joint,
            phi_rev=phi_rev,
            phi_pri=phi_pri,
            camera=cam_iter,
            ref_first_3HW=ref_first_iter,
            ref_last_3HW=ref_last_iter,
            committed_type=committed_type,
        )

        # Head pre-activation leash (same as the main loop; keeps the adapter alive).
        if head_preact_l2 is not None and cfg.lambda_head_leash > 0.0:
            total_loss = total_loss + cfg.lambda_head_leash * head_preact_l2

        opt_cur.zero_grad(set_to_none=True)
        total_loss.backward()
        # Keep type_logit frozen on this clone.
        zero_type_logit_grad(learnable_cur)
        if cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for grp in opt_cur.param_groups for p in grp["params"]],
                cfg.grad_clip_norm,
            )
        opt_cur.step()

        # Remember final L_sds + lambda_rgb * L_rgb for each clone's last iter.
        # L_rgb is six-state pure reconstruction, not the dense Wan video.
        score_value = float(log["L_sds"]) + float(sched_lambdas[2]) * float(log["L_rgb"])
        if committed_type == "revolute":
            dual_clone.final_loss_rev = score_value
        else:
            dual_clone.final_loss_pri = score_value

    _winner_learnable, _winner_opt, winner = commit_dual_clone(dual_clone)
    summary["dual_clone_final_loss"] = {
        "rev": dual_clone.final_loss_rev,
        "pri": dual_clone.final_loss_pri,
        "winner": winner,
    }
    return winner


__all__ = [
    "BootstrapBundle", "TrellisModules",
    "build_optimizer", "train_stage_d_p1",
]

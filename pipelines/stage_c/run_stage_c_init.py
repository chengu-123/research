"""Stage C top-level driver: production wire-in.

Composes the pre-existing algorithm modules into the JointInit contract:
    move_geometry.compute_per_state_move_geom -> PerStateMoveGeom
    joint_type_detect.detect_joint_type        -> JointTypeResult
    axis_fit.fit_axis_origin                   -> AxisResult
    phi_fit.fit_phi                            -> PhiResult
    anchor_extract.extract_anchors             -> anchors + diag
    confidence.aggregate_confidence            -> ConfidenceBundle

Per method.md sec 6 B6 the spec contract:
    psi_0, phi_0, anchors_object = stage_c_joint_init(
        z_final, M_attn_boot_64, O_init, is_carpet_mask, U_seed
    )

We extend the spec inputs with v3.3.6 Stage B v3.3.6 enriched signals
(O_base_canonical, O_move_per_state, P_base_canonical,
P_move_evidence_per_state, M_motion_corridor_64) via the typed
`StageCInputs` dataclass. When v3.3.6 signals are absent, Stage C raises
since the spec algorithm depends on per-state move evidence.

Per user direction "总是返回 + 加 confidence 字段": this driver never
raises NoArticulationError; degenerate inputs yield a low-confidence
JointInit that Stage D's dual-clone branch can route around.

OUTPUTS:
    JointInit with:
      psi.axis, psi.origin, psi.type_logit          from joint_type + axis_fit
      psi.theta_limit_raw, psi.disp_limit_raw       derived from observed extent
      psi.delta_u_init                              from phi_fit
      phi_0                                         c-shifted (phi_0[c] == 0)
      anchors_object                                voxel coords (N_a, 3) int32
      confidence                                    aggregated [0, 1]
      sub_confidence                                per-component breakdown
      diagnostics                                   all intermediate stats
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np
import torch

from pipelines.stage_c.anchor_extract import extract_anchors
from pipelines.stage_c.axis_fit import fit_axis_origin
from pipelines.stage_c.config import StageCConfig
from pipelines.stage_c.confidence import aggregate_confidence
from pipelines.stage_c.io_contract import JointInit, Psi, StageCInputs
from pipelines.stage_c.joint_type_detect import detect_joint_type
from pipelines.stage_c.move_geometry import (
    compute_per_state_move_geom,
    overall_move_extent,
)
from pipelines.stage_c.phi_fit import fit_phi


def _inverse_softplus_scalar(y: float, eps: float = 1e-6) -> float:
    """Stable inverse of softplus for a positive scalar."""
    y = max(float(y), eps)
    if y > 20.0:
        return y
    return float(math.log(math.expm1(y)))


def _encode_psi(
    axis: torch.Tensor,
    origin: torch.Tensor,
    type_logit: float,
    observed_max_angle: float,
    observed_max_disp: float,
    delta_u_init: torch.Tensor,
    cfg: StageCConfig,
) -> Psi:
    """Pack axis/origin/type + theta/disp/delta_u into the 19-element Psi.

    theta_limit_raw / disp_limit_raw use observed range * margin, with a
    floor (theta_limit_min / disp_limit_min) so Stage D's softplus never
    starts at a near-singular value.
    """
    theta_observed = max(float(observed_max_angle), cfg.theta_limit_min)
    disp_observed = max(float(observed_max_disp), cfg.disp_limit_min)
    theta_limit_softplus = max(theta_observed * cfg.theta_limit_margin, cfg.theta_limit_min)
    disp_limit_softplus = max(disp_observed * cfg.disp_limit_margin, cfg.disp_limit_min)
    theta_limit_raw = _inverse_softplus_scalar(theta_limit_softplus)
    disp_limit_raw = _inverse_softplus_scalar(disp_limit_softplus)

    return Psi(
        axis=axis.detach().to(torch.float32),
        origin=origin.detach().to(torch.float32),
        type_logit=float(type_logit),
        theta_limit_raw=float(theta_limit_raw),
        disp_limit_raw=float(disp_limit_raw),
        delta_u_init=delta_u_init.detach().to(torch.float32),
    )


def _pad_or_trim_delta_u(delta_u_init: torch.Tensor, target_len: int = 5) -> torch.Tensor:
    """Ensure delta_u_init is exactly target_len entries.

    Stage D's `learnable.delta_phi` is shape [5]; phi_fit returns
    delta_u_init of length (K - 1). For K=6 this is exactly 5. We pad with
    a small positive value (inverse_softplus of phi_min_gap) if K-1 < 5,
    or trim if K-1 > 5 (rare).
    """
    n = int(delta_u_init.shape[0])
    if n == target_len:
        return delta_u_init
    device = delta_u_init.device
    dtype = delta_u_init.dtype
    if n < target_len:
        # Pad with inverse_softplus(1e-3) -> small positive softplus output
        pad_val = _inverse_softplus_scalar(1e-3)
        pad = torch.full((target_len - n,), pad_val, device=device, dtype=dtype)
        return torch.cat([delta_u_init, pad], dim=0)
    # Trim: keep first target_len
    return delta_u_init[:target_len]


def run_stage_c_joint_init(
    inputs: StageCInputs,
    cfg: Optional[StageCConfig] = None,
    out_dir: Optional[str] = None,
) -> JointInit:
    """Stage C joint init driver (production).

    Algorithm flow (method.md sec 6 B6, leveraging v3.3.6 Stage B rich
    outputs per user direction):

      1. compute_per_state_move_geom: K per-state move centroids in world
         space, prefer soft (P_move_evidence_per_state) over hard.
      2. detect_joint_type: line-vs-arc fit on centroid trajectory
         (replaces v8.1 BIC, ~1 sec vs ~30 sec).
      3. fit_axis_origin: prefer arc fit for revolute, line PCA for
         prismatic; fall back to M_motion_corridor_64 PCA when degenerate.
      4. fit_phi: project centroids on axis -> u_raw -> normalise -> c-shift
         (NEW.1 canonical-state shift, phi_0[c]=0).
      5. extract_anchors: M_motion_corridor_64 ∩ O_base_canonical -> FPS.
      6. aggregate_confidence: weighted combination of 4 sub-confidences.

    Returns a JointInit with confidence in [0, 1] always (never raises on
    degenerate input -- low confidence signals Stage D to dual-clone).
    """
    if cfg is None:
        cfg = StageCConfig()

    # Requires v3.3.6 enriched signals for the data-driven algorithm.
    # The spec inputs (z_final, M_attn_boot_64, O_init, is_carpet_mask,
    # U_seed) are insufficient for the line-vs-arc fit and motion-corridor
    # anchor extraction; we need O_move_per_state or P_move_evidence_per_state.
    if not (
        inputs.O_move_per_state is not None
        or inputs.P_move_evidence_per_state is not None
    ):
        raise ValueError(
            "Stage C joint init requires either O_move_per_state or "
            "P_move_evidence_per_state from Stage B v3.3.6+. Got neither."
        )

    device = inputs.device()
    K = inputs.K()
    res = int(cfg.resolution)
    c_idx = int(cfg.canonical_state_idx)
    if not (0 <= c_idx < K):
        raise ValueError(
            f"canonical_state_idx={c_idx} out of range for K={K}"
        )

    # ---- Step 1: per-state move geometry ----
    geom = compute_per_state_move_geom(
        O_move_per_state=inputs.O_move_per_state,
        P_move_evidence_per_state=inputs.P_move_evidence_per_state,
        res=res,
        min_voxels=int(cfg.min_move_voxels_per_state),
        soft_threshold=float(cfg.soft_centroid_threshold),
        prefer_soft=True,
    )

    # ---- Step 2: joint type detection (line vs arc fit) ----
    type_result = detect_joint_type(
        geom=geom,
        type_decision_margin=float(cfg.type_decision_margin),
        arc_min_states=int(cfg.arc_min_states),
    )

    # ---- Step 3: axis + origin from the better-fitting primitive ----
    axis_result = fit_axis_origin(
        type_result=type_result,
        geom=geom,
        M_motion_corridor_64=inputs.M_motion_corridor_64,
        corridor_pca_threshold=float(cfg.corridor_pca_threshold),
        res=res,
        device=device,
        dtype=torch.float32,
    )

    # ---- Step 4: phi_0 from centroid projection (with c-shift) ----
    phi_result = fit_phi(
        geom=geom,
        joint_type_str=type_result.type_str,
        axis=axis_result.axis,
        origin=axis_result.origin,
        canonical_state_idx=c_idx,
        enforce_monotone=bool(cfg.phi_enforce_monotone),
        phi_min_gap=float(cfg.phi_min_gap),
    )

    # ---- Step 5: anchor extraction (M_motion_corridor ∩ O_base) ----
    anchors_object, anchor_diag = extract_anchors(
        M_motion_corridor_64=inputs.M_motion_corridor_64,
        O_base_canonical=inputs.O_base_canonical,
        P_base_canonical=inputs.P_base_canonical,
        is_carpet_mask_flat=inputs.is_carpet_mask.to(device).flatten().bool(),
        corridor_threshold=float(cfg.anchor_corridor_threshold),
        base_threshold=float(cfg.anchor_base_threshold),
        dilate_radius=int(cfg.anchor_dilate_radius),
        target_count=int(cfg.anchor_target_count),
        min_count=int(cfg.anchor_min_count),
        res=res,
        fps_seed=0,
    )

    # ---- Step 6: aggregated confidence ----
    conf_bundle = aggregate_confidence(
        type_result=type_result,
        axis_result=axis_result,
        geom=geom,
        phi_result=phi_result,
        O_base_canonical=inputs.O_base_canonical,
        P_base_canonical=inputs.P_base_canonical,
        w_type=float(cfg.conf_w_type),
        w_axis=float(cfg.conf_w_axis),
        w_centroid=float(cfg.conf_w_centroid),
        w_base=float(cfg.conf_w_base),
        res=res,
    )

    # ---- Encode Psi (19-element packed form) ----
    delta_u_init = _pad_or_trim_delta_u(phi_result.delta_u_init, target_len=5)
    psi = _encode_psi(
        axis=axis_result.axis,
        origin=axis_result.origin,
        type_logit=type_result.type_logit,
        observed_max_angle=phi_result.observed_max_angle,
        observed_max_disp=phi_result.observed_max_disp,
        delta_u_init=delta_u_init,
        cfg=cfg,
    )

    # ---- Final JointInit assembly ----
    overall_extent = overall_move_extent(geom)
    result = JointInit(
        psi=psi,
        phi_0=phi_result.phi_0_shifted.to(device=device, dtype=torch.float32),
        anchors_object=anchors_object.to(torch.int32),
        confidence=float(conf_bundle.overall),
        sub_confidence={
            "type": float(conf_bundle.type_conf),
            "axis": float(conf_bundle.axis_conf),
            "centroid": float(conf_bundle.centroid_conf),
            "base": float(conf_bundle.base_conf),
        },
        diagnostics={
            "stub": False,
            "K": K,
            "canonical_state_idx": c_idx,
            "joint_type": type_result.type_str,
            "type_logit": float(type_result.type_logit),
            "n_valid_states": int(type_result.n_valid_states),
            "residual_line": float(type_result.residual_line),
            "residual_arc": float(type_result.residual_arc),
            "axis_fit_source": axis_result.fit_source,
            "axis_fit_residual": float(axis_result.fit_residual),
            "phi_monotone_enforced": bool(phi_result.monotone_enforced),
            "observed_max_angle_rad": float(phi_result.observed_max_angle),
            "observed_max_disp_world": float(phi_result.observed_max_disp),
            "overall_move_extent_world": float(overall_extent),
            "n_anchors": int(anchors_object.shape[0]),
            "anchor_diag": anchor_diag,
            "confidence_notes": conf_bundle.notes,
            "has_v336_signals": bool(inputs.has_v336_signals()),
        },
    )

    # ---- Persist diagnostics JSON (optional) ----
    if out_dir is not None and cfg.save_diagnostics_json:
        os.makedirs(out_dir, exist_ok=True)
        meta = result.to_dict_serialisable()
        with open(os.path.join(out_dir, "stage_c_joint_init.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ---- Visualisation ----
    if out_dir is not None and cfg.save_viz:
        from pipelines.stage_c.viz import save_stage_c_viz
        viz_dir = os.path.join(out_dir, "viz")
        save_stage_c_viz(
            viz_dir,
            inputs=inputs,
            geom=geom,
            type_result=type_result,
            axis_result=axis_result,
            phi_result=phi_result,
            anchors_object=result.anchors_object,
            joint_init=result,
        )

    return result

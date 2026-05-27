"""Stage C v3 top-level driver: cardinal-cand + voxel-physical-scoring + dual.

v3 supersedes the centroid-only v2 implementation. Pipeline:

  Step 1: build_per_state_voxel_sets
            -> V_per_state_voxel, V_union_voxel, valid_state
  Step 2: extract_anchors
            -> anchors_object (contact band voxels, no FPS subsample)
  Step 3: detect_joint_type_v3
            -> enumerate 6 cardinal axes per type
            -> voxel reverse-warp score each candidate
            -> best_pris, best_rev, type_logit
  Step 4: fit_phi_from_candidate (twice: primary + secondary)
            -> normalized phi_0, delta_u_init for each type
  Step 5: build_psi (twice)
            -> 19-dim packed Psi for primary and secondary
  Step 6: aggregate_confidence
            -> overall + sub-confidence
  Step 7: assemble JointInit with .secondary field set

Output JSON contains BOTH candidates. Stage D dual-clone reads .secondary
when type_logit margin is small.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np
import torch

from pipelines.stage_c.anchor_extract import extract_anchors
from pipelines.stage_c.axis_fit import AxisResult, fit_axis_origin
from pipelines.stage_c.config import StageCConfig
from pipelines.stage_c.confidence import aggregate_confidence
from pipelines.stage_c.io_contract import JointInit, Psi, StageCInputs
from pipelines.stage_c.joint_type_detect import (
    CandidateResult,
    JointTypeResult,
    build_per_state_voxel_sets,
    detect_joint_type_v3,
)
from pipelines.stage_c.phi_fit import (
    PhiResult,
    fit_phi_from_candidate,
    _inverse_softplus_scalar,
)


def _encode_psi_from_candidate(
    candidate: CandidateResult,
    phi_result: PhiResult,
    cfg: StageCConfig,
) -> Psi:
    """Pack axis/origin/type_logit/theta_limit_raw/disp_limit_raw/delta_u_init.

    v3 critical: ALWAYS use the observed extent of the SELECTED type for the
    matching limit, plus a floor for the OTHER type's limit (so Stage D can
    dual-clone meaningfully). This kills the bug where revolute selection
    silently set disp_limit_raw to inverse_softplus(0.067) regardless of data.
    """
    if candidate.type_str == "prismatic":
        type_logit_value = +5.0   # strong commit to prismatic in this branch
        observed_disp = phi_result.observed_max_disp
        # For prismatic, set theta to floor (revolute extent unused for this branch)
        theta_limit_softplus = cfg.theta_limit_min
        disp_limit_softplus = max(
            observed_disp * cfg.disp_limit_margin, cfg.disp_limit_min
        )
    else:  # revolute
        type_logit_value = -5.0
        observed_angle = phi_result.observed_max_angle
        theta_limit_softplus = max(
            observed_angle * cfg.theta_limit_margin, cfg.theta_limit_min
        )
        disp_limit_softplus = cfg.disp_limit_min

    theta_limit_raw = _inverse_softplus_scalar(theta_limit_softplus)
    disp_limit_raw = _inverse_softplus_scalar(disp_limit_softplus)

    return Psi(
        axis=candidate.axis.detach().to(torch.float32),
        origin=candidate.origin.detach().to(torch.float32),
        type_logit=float(type_logit_value),
        theta_limit_raw=float(theta_limit_raw),
        disp_limit_raw=float(disp_limit_raw),
        delta_u_init=phi_result.delta_u_init.detach().to(torch.float32),
    )


def _pad_or_trim_delta_u(delta_u_init: torch.Tensor, target_len: int = 5) -> torch.Tensor:
    """Stage D's `learnable.delta_phi` is shape [5]; phi_fit returns (K-1)."""
    n = int(delta_u_init.shape[0])
    if n == target_len:
        return delta_u_init
    if n < target_len:
        pad_val = _inverse_softplus_scalar(1e-3)
        pad = torch.full(
            (target_len - n,), pad_val,
            device=delta_u_init.device, dtype=delta_u_init.dtype,
        )
        return torch.cat([delta_u_init, pad], dim=0)
    return delta_u_init[:target_len]


def _build_joint_init_from_candidate(
    candidate: CandidateResult,
    cfg: StageCConfig,
    anchors_object: torch.Tensor,
    type_logit_override: Optional[float] = None,
    confidence: float = 0.5,
    sub_confidence: Optional[dict] = None,
    diagnostics: Optional[dict] = None,
    secondary: Optional[JointInit] = None,
) -> JointInit:
    """Build a JointInit for one candidate (primary or secondary)."""
    phi_result = fit_phi_from_candidate(
        phi_raw_signed=candidate.phi_k,
        canonical_state_idx=int(cfg.canonical_state_idx),
        enforce_monotone=bool(cfg.phi_enforce_monotone),
        phi_min_gap=float(cfg.phi_min_gap),
        joint_type_str=candidate.type_str,
    )
    psi = _encode_psi_from_candidate(candidate, phi_result, cfg)
    if type_logit_override is not None:
        psi = Psi(
            axis=psi.axis, origin=psi.origin,
            type_logit=float(type_logit_override),
            theta_limit_raw=psi.theta_limit_raw,
            disp_limit_raw=psi.disp_limit_raw,
            delta_u_init=psi.delta_u_init,
        )
    psi = Psi(
        axis=psi.axis, origin=psi.origin, type_logit=psi.type_logit,
        theta_limit_raw=psi.theta_limit_raw,
        disp_limit_raw=psi.disp_limit_raw,
        delta_u_init=_pad_or_trim_delta_u(psi.delta_u_init, target_len=5),
    )
    return JointInit(
        psi=psi,
        phi_0=phi_result.phi_0_shifted.to(torch.float32),
        anchors_object=anchors_object.to(torch.int32),
        confidence=float(confidence),
        sub_confidence=dict(sub_confidence or {}),
        diagnostics=dict(diagnostics or {}),
        secondary=secondary,
    )


def run_stage_c_joint_init(
    inputs: StageCInputs,
    cfg: Optional[StageCConfig] = None,
    out_dir: Optional[str] = None,
) -> JointInit:
    """v3 Stage C joint init driver.

    Algorithm (method.md sec 6 B6 + GPT structural + cardinal-axis prior):
      1. build_per_state_voxel_sets (clean per-state evidence)
      2. extract_anchors (contact band, no FPS)
      3. detect_joint_type_v3 (6 cardinal x 2 types = 12 candidates, each scored
         by voxel reverse-warp + IoU + base-conflict + contact + monotone)
      4. fit_phi_from_candidate (twice: primary + secondary)
      5. assemble JointInit + .secondary

    Returns JointInit with both primary and secondary candidates set when
    available.
    """
    if cfg is None:
        cfg = StageCConfig()
    if not (
        inputs.O_move_per_state is not None
        or inputs.P_move_evidence_per_state is not None
    ):
        raise ValueError(
            "Stage C v3 requires O_move_per_state or P_move_evidence_per_state."
        )

    device = inputs.device()
    K = inputs.K()
    res = int(cfg.resolution)
    c_idx = int(cfg.canonical_state_idx)
    if not (0 <= c_idx < K):
        raise ValueError(f"canonical_state_idx={c_idx} out of range for K={K}")

    # ---- Step 1: per-state cleaned voxel sets ----
    V_per_state, V_union, valid_state = build_per_state_voxel_sets(
        O_move_per_state=inputs.O_move_per_state,
        P_move_evidence_per_state=inputs.P_move_evidence_per_state,
        O_base_canonical=inputs.O_base_canonical,
        is_carpet_mask_flat=inputs.is_carpet_mask.to(device).flatten().bool(),
        res=res,
        soft_threshold=float(cfg.soft_centroid_threshold),
        min_voxels=int(cfg.min_move_voxels_per_state),
        prefer_soft=True,
    )

    # ---- Step 2: anchor extraction (contact band, no FPS) ----
    anchors_object, anchor_diag = extract_anchors(
        M_motion_corridor_64=inputs.M_motion_corridor_64,
        O_base_canonical=inputs.O_base_canonical,
        P_base_canonical=inputs.P_base_canonical,
        is_carpet_mask_flat=inputs.is_carpet_mask.to(device).flatten().bool(),
        corridor_threshold=float(cfg.anchor_corridor_threshold),
        base_threshold=float(cfg.anchor_base_threshold),
        dilate_radius=int(cfg.anchor_dilate_radius),
        target_count=int(cfg.anchor_target_count),  # ignored in v3
        min_count=int(cfg.anchor_min_count),
        res=res,
        fps_seed=0,
    )

    # ---- Step 3: cardinal candidate enumeration + voxel scoring ----
    type_result: JointTypeResult = detect_joint_type_v3(
        V_per_state_voxel=V_per_state,
        V_union_voxel=V_union,
        valid_state=valid_state,
        O_base_canonical=inputs.O_base_canonical,
        anchors_voxel=anchors_object,
        canonical_state_idx=c_idx,
        res=res,
        type_margin=float(cfg.type_decision_margin),
        device=device,
        dtype=torch.float32,
    )

    axis_result: AxisResult = fit_axis_origin(
        type_result=type_result,
        res=res,
        device=device,
        dtype=torch.float32,
    )

    # ---- Step 6: aggregate confidence ----
    conf_bundle = aggregate_confidence(
        type_result=type_result,
        axis_result=axis_result,
        O_base_canonical=inputs.O_base_canonical,
        P_base_canonical=inputs.P_base_canonical,
        valid_state=valid_state,
        K=K,
        w_type=float(cfg.conf_w_type),
        w_axis=float(cfg.conf_w_axis),
        w_centroid=float(cfg.conf_w_centroid),
        w_base=float(cfg.conf_w_base),
        res=res,
    )

    # ---- Step 4+5+7: assemble primary + secondary JointInits ----
    # Pick primary and secondary based on type_str
    if type_result.type_str == "prismatic":
        primary_cand = type_result.best_pris
        secondary_cand = type_result.best_rev
    elif type_result.type_str == "revolute":
        primary_cand = type_result.best_rev
        secondary_cand = type_result.best_pris
    else:
        # uncertain: take the higher-scoring as primary
        if (
            type_result.best_pris is not None
            and type_result.best_rev is not None
        ):
            if type_result.best_pris.score.score >= type_result.best_rev.score.score:
                primary_cand = type_result.best_pris
                secondary_cand = type_result.best_rev
            else:
                primary_cand = type_result.best_rev
                secondary_cand = type_result.best_pris
        else:
            primary_cand = type_result.best_pris or type_result.best_rev
            secondary_cand = None

    if primary_cand is None:
        raise RuntimeError(
            "Stage C v3: no primary candidate produced (joint_type_detect_v3 "
            "returned empty best_pris and best_rev)."
        )

    common_diag = {
        "stage_c_version": "v3_cardinal_voxel_scoring",
        "K": K,
        "canonical_state_idx": c_idx,
        "joint_type": type_result.type_str,
        "type_logit": float(type_result.type_logit),
        "type_confidence": float(type_result.confidence),
        "n_valid_states": int(type_result.n_valid_states),
        "n_anchors": int(anchors_object.shape[0]),
        "axis_fit_source": axis_result.fit_source,
        "axis_fit_residual": float(axis_result.fit_residual),
        "anchor_diag": anchor_diag,
        "confidence_notes": conf_bundle.notes,
        "primary_score_breakdown": {
            "score": float(primary_cand.score.score),
            "consistency": float(primary_cand.score.consistency),
            "conflict": float(primary_cand.score.conflict),
            "coverage": float(primary_cand.score.coverage),
            "contact_compat": float(primary_cand.score.contact_compat),
            "monotone_quality": float(primary_cand.score.monotone_quality),
        },
    }
    if secondary_cand is not None:
        common_diag["secondary_score_breakdown"] = {
            "score": float(secondary_cand.score.score),
            "consistency": float(secondary_cand.score.consistency),
            "conflict": float(secondary_cand.score.conflict),
            "coverage": float(secondary_cand.score.coverage),
            "contact_compat": float(secondary_cand.score.contact_compat),
            "monotone_quality": float(secondary_cand.score.monotone_quality),
        }
    # FreeArt3D-style geometric scores (diagnostic)
    if type_result.pris_geom_scores is not None:
        common_diag["pris_geom_abs_proj"] = type_result.pris_geom_scores.tolist()
    if type_result.rev_geom_scores is not None:
        common_diag["rev_geom_abs_proj"] = type_result.rev_geom_scores.tolist()

    sub_conf = {
        "type": float(conf_bundle.type_conf),
        "axis": float(conf_bundle.axis_conf),
        "centroid": float(conf_bundle.centroid_conf),
        "base": float(conf_bundle.base_conf),
    }

    # Build secondary JointInit FIRST so we can nest it into primary
    secondary_init: Optional[JointInit] = None
    if secondary_cand is not None:
        secondary_init = _build_joint_init_from_candidate(
            candidate=secondary_cand,
            cfg=cfg,
            anchors_object=anchors_object,
            type_logit_override=-type_result.type_logit,   # mirror for secondary
            confidence=float(conf_bundle.overall * 0.7),    # lower than primary
            sub_confidence=sub_conf,
            diagnostics={"slot": "secondary", **common_diag},
            secondary=None,    # no recursion
        )

    primary_init = _build_joint_init_from_candidate(
        candidate=primary_cand,
        cfg=cfg,
        anchors_object=anchors_object,
        type_logit_override=float(type_result.type_logit),
        confidence=float(conf_bundle.overall),
        sub_confidence=sub_conf,
        diagnostics={"slot": "primary", **common_diag},
        secondary=secondary_init,
    )

    # ---- Persist diagnostics ----
    if out_dir is not None and cfg.save_diagnostics_json:
        os.makedirs(out_dir, exist_ok=True)
        meta = primary_init.to_dict_serialisable()
        with open(
            os.path.join(out_dir, "stage_c_joint_init.json"), "w", encoding="utf-8",
        ) as f:
            json.dump(meta, f, indent=2, ensure_ascii=True)

    # ---- Visualization (v3 viz module updates pending; current viz expects
    #      legacy geom/phi_result/axis_result so we skip if not adapted) ----
    if out_dir is not None and cfg.save_viz:
        viz_dir = os.path.join(out_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)
        _save_v3_diagnostics(viz_dir, type_result, primary_init, anchor_diag)

    return primary_init


def _save_v3_diagnostics(
    viz_dir: str,
    type_result: JointTypeResult,
    primary_init: JointInit,
    anchor_diag: dict,
) -> None:
    """Minimal text-based v3 diagnostics dump (HTML viz is a separate task)."""
    summary_lines = [
        "=== Stage C v3 diagnostics ===",
        f"Selected type: {type_result.type_str}",
        f"type_logit: {type_result.type_logit:.4f}",
        f"type_confidence: {type_result.confidence:.4f}",
        f"n_valid_states: {type_result.n_valid_states}",
        "",
        "--- All 12 candidates (6 cardinal x 2 types) ---",
    ]
    for c in type_result.all_candidates:
        axis_str = (
            f"[{c.axis[0].item():+.0f},{c.axis[1].item():+.0f},{c.axis[2].item():+.0f}]"
        )
        s = c.score
        summary_lines.append(
            f"  {c.type_str:<10s} axis={axis_str}  "
            f"score={s.score:.3f}  consistency={s.consistency:.3f}  "
            f"conflict={s.conflict:.3f}  coverage={s.coverage:.3f}  "
            f"contact={s.contact_compat:.3f}  monotone={s.monotone_quality:.3f}"
        )
    summary_lines.append("")
    summary_lines.append(f"--- FreeArt3D geom scores ---")
    if type_result.pris_geom_scores is not None:
        summary_lines.append(
            f"  pris (argmax wins): {type_result.pris_geom_scores.tolist()}"
        )
    if type_result.rev_geom_scores is not None:
        summary_lines.append(
            f"  rev  (argmin wins): {type_result.rev_geom_scores.tolist()}"
        )
    summary_lines.append("")
    summary_lines.append("--- Anchor diagnostics ---")
    for k, v in anchor_diag.items():
        summary_lines.append(f"  {k}: {v}")

    with open(os.path.join(viz_dir, "v3_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))


__all__ = ["run_stage_c_joint_init"]

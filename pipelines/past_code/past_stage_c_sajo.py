"""Stage C driver: Screw-Axis Joint Optimization with Contact Anchors (SAJO).

Consumes the ``(K, 64, 64, 64)`` occupancy stack from Stage B and returns
a single selected joint hypothesis (revolute or prismatic) with its
screw axis parameters, per-state magnitudes, and soft part masks. Every
intermediate (joint-free split, anchors, both EM traces, BIC scores) is
persisted under ``out_dir`` for diagnostics and paper figures.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from pipelines.sajo.anchors import extract_anchors, joint_free_split
from pipelines.sajo.bic import BICResult, run_dual_em_and_select
from pipelines.sajo.em import EMHParams, EMResult, em_optimize
from pipelines.sajo.init import init_prismatic, init_revolute
from pipelines.utils.voxel_io import save_json, save_voxel_grid
from pipelines.utils.voxel_viz import (
    save_anchors_html,
    save_axis_overlay_html,
    save_em_traces_html,
    save_soft_voxel_html,
    save_two_masks_html,
)


@dataclass
class SAJOResult:
    joint_type: str
    omega: torch.Tensor               # (3,) unit
    q: torch.Tensor                   # (3,) world coord — rotation origin for revolute
    v: torch.Tensor                   # (3,) — Plücker moment (rev) or v_hat (pris)
    phi_k: torch.Tensor               # (K,)
    M_base: torch.Tensor              # (64, 64, 64)
    M_move: torch.Tensor              # (64, 64, 64)
    T_k_list: torch.Tensor            # (K, 4, 4)
    p_base: torch.Tensor              # (64, 64, 64) — from joint-free split
    p_move: torch.Tensor              # (64, 64, 64)
    anchors_meta: Dict[str, Any] = field(default_factory=dict)
    bic: Dict[str, Any] = field(default_factory=dict)
    em_rev: Dict[str, Any] = field(default_factory=dict)
    em_pris: Dict[str, Any] = field(default_factory=dict)
    converged: bool = False
    n_iters: int = 0


def _build_em_hparams(cfg_sajo: Any) -> EMHParams:
    em = cfg_sajo.em
    return EMHParams(
        n_outer=int(em.n_outer),
        n_inner=int(em.n_inner),
        lr_S=float(em.lr_S),
        lr_phi=float(em.lr_phi),
        alpha=float(em.alpha),
        beta=float(em.beta),
        eta_r=float(em.eta_r),
        tol=float(em.tol),
        active_voxel_thresh=float(em.active_voxel_thresh),
    )


def _em_result_to_dict(em: EMResult) -> Dict[str, Any]:
    return dict(
        joint_type=em.joint_type,
        omega=em.omega.detach().cpu().tolist(),
        q=em.q.detach().cpu().tolist(),
        v=em.v.detach().cpu().tolist(),
        phi_k=em.phi_k.detach().cpu().tolist(),
        L_data_final=float(em.L_data_final),
        L_reg_trace=list(em.L_reg_trace),
        converged=bool(em.converged),
        n_iters=int(em.n_iters),
        meta=dict(em.meta),
    )


def run_sajo(
    O_stack: torch.Tensor,
    cfg_sajo: Any,
    out_dir: str,
    joint_type_override: Optional[str] = None,
) -> SAJOResult:
    """Run the full Stage C pipeline.

    Parameters
    ----------
    O_stack : torch.Tensor
        ``(K, 64, 64, 64)`` binary or soft occupancy grids from Stage B.
    cfg_sajo : OmegaConf
        The ``sajo:`` subtree of ``configs/v1.yaml``.
    out_dir : str
        Destination directory; created if missing.
    joint_type_override : str, optional
        ``'revolute'`` or ``'prismatic'`` to skip BIC and force that type.

    Returns
    -------
    SAJOResult
    """
    os.makedirs(out_dir, exist_ok=True)
    device = O_stack.device
    resolution = int(O_stack.shape[-1])

    # C1: joint-free split.
    split_cfg = cfg_sajo.split
    p_base, p_move = joint_free_split(
        O_stack,
        sigma_b=float(split_cfg.sigma_b),
        sigma_m=float(split_cfg.sigma_m),
        tau_b=float(split_cfg.tau_b),
        tau_m=float(split_cfg.tau_m),
    )
    save_voxel_grid(os.path.join(out_dir, "p_base.npy"),
                    p_base.detach().cpu().numpy().astype(np.float32))
    save_voxel_grid(os.path.join(out_dir, "p_move.npy"),
                    p_move.detach().cpu().numpy().astype(np.float32))

    # C2: contact anchors.
    anch_cfg = cfg_sajo.anchors
    anchor_coords, anchor_weights, anchors_meta = extract_anchors(
        p_base, p_move,
        min_component_size=int(anch_cfg.min_component_size),
        max_components=int(anch_cfg.max_components),
        binarize_threshold=float(anch_cfg.binarize_threshold),
    )
    save_json({
        "coords": anchor_coords.detach().cpu().tolist(),
        "weights": anchor_weights.detach().cpu().tolist(),
        "meta": anchors_meta,
    }, os.path.join(out_dir, "anchors.json"))
    save_voxel_grid(
        os.path.join(out_dir, "anchors.npy"),
        anchor_coords.detach().cpu().numpy().astype(np.int64),
    )

    # C3: dual initialization.
    rev_init = init_revolute(anchor_coords, anchor_weights, O_stack, p_move, resolution)
    pris_init = init_prismatic(O_stack, p_move, resolution)
    save_json({
        "omega": rev_init["omega"].cpu().tolist(),
        "q": rev_init["q"].cpu().tolist(),
        "v": rev_init["v"].cpu().tolist(),
        "phi_k": rev_init["phi_k"].cpu().tolist(),
    }, os.path.join(out_dir, "init_revolute.json"))
    save_json({
        "v_hat": pris_init["v_hat"].cpu().tolist(),
        "phi_k": pris_init["phi_k"].cpu().tolist(),
    }, os.path.join(out_dir, "init_prismatic.json"))

    # C4 + C5: dual EM + BIC selection, or single branch when overridden.
    hp = _build_em_hparams(cfg_sajo)

    if joint_type_override in ("revolute", "prismatic"):
        init = rev_init if joint_type_override == "revolute" else pris_init
        em_selected = em_optimize(
            joint_type_override,
            O_stack, p_base, p_move,
            anchor_coords, anchor_weights,
            init, hp, resolution,
        )
        bic_report: Dict[str, Any] = {
            "joint_type": joint_type_override,
            "override": True,
            "bic_rev": None,
            "bic_pris": None,
            "confidence": None,
            "N": None,
        }
        em_rev_dict = _em_result_to_dict(em_selected) if joint_type_override == "revolute" else {}
        em_pris_dict = _em_result_to_dict(em_selected) if joint_type_override == "prismatic" else {}
    else:
        bic_cfg = cfg_sajo.bic
        bic_result: BICResult = run_dual_em_and_select(
            O_stack, p_base, p_move,
            anchor_coords, anchor_weights,
            rev_init, pris_init, hp,
            k_rev_const=int(bic_cfg.k_rev),
            k_pris_const=int(bic_cfg.k_pris),
            min_N=int(bic_cfg.min_N),
            resolution=resolution,
        )
        em_selected = bic_result.selected
        em_rev_dict = _em_result_to_dict(bic_result.em_rev)
        em_pris_dict = _em_result_to_dict(bic_result.em_pris)
        bic_report = {
            "joint_type": bic_result.joint_type,
            "bic_rev": bic_result.bic_rev,
            "bic_pris": bic_result.bic_pris,
            "confidence": bic_result.confidence,
            "N": bic_result.N,
            "k_rev": bic_result.k_rev,
            "k_pris": bic_result.k_pris,
            "override": False,
        }

    save_json(em_rev_dict, os.path.join(out_dir, "em_revolute.json"))
    save_json(em_pris_dict, os.path.join(out_dir, "em_prismatic.json"))
    save_json(bic_report, os.path.join(out_dir, "bic.json"))

    # Final masks: M_base is the complement of M_move inside the
    # occupied-region prior. This gives a non-overlapping soft partition.
    M_move = em_selected.M_move
    mu_O = O_stack.mean(dim=0)                                # (D,H,W)
    M_base = (mu_O * (1.0 - M_move)).clamp(0.0, 1.0)

    save_voxel_grid(os.path.join(out_dir, "M_base.npy"),
                    M_base.detach().cpu().numpy().astype(np.float32))
    save_voxel_grid(os.path.join(out_dir, "M_move.npy"),
                    M_move.detach().cpu().numpy().astype(np.float32))
    save_voxel_grid(os.path.join(out_dir, "T_k.npy"),
                    em_selected.T_k_list.detach().cpu().numpy().astype(np.float32))

    # --- Visualization ---
    viz_dir = os.path.join(out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    p_base_np = p_base.detach().cpu().numpy().astype(np.float32)
    p_move_np = p_move.detach().cpu().numpy().astype(np.float32)
    M_base_np = M_base.detach().cpu().numpy().astype(np.float32)
    M_move_np = M_move.detach().cpu().numpy().astype(np.float32)
    mu_O_np = O_stack.mean(dim=0).detach().cpu().numpy().astype(np.float32)

    # Joint-free soft split
    save_soft_voxel_html(p_base_np, os.path.join(viz_dir, "p_base.html"),
                         title="SAJO: p_base (low cross-state variance)")
    save_soft_voxel_html(p_move_np, os.path.join(viz_dir, "p_move.html"),
                         title="SAJO: p_move (high cross-state variance)")

    # Final part masks (base vs move overlay)
    save_two_masks_html(M_base_np, M_move_np,
                        os.path.join(viz_dir, "M_base_vs_move.html"),
                        title="SAJO: M_base / M_move")

    # Contact anchors on top of mean occupancy
    save_anchors_html(
        mu_O_np,
        anchor_coords.detach().cpu().numpy() if anchor_coords.numel() > 0 else np.zeros((0, 3)),
        anchor_weights.detach().cpu().numpy() if anchor_weights.numel() > 0 else np.zeros((0,)),
        os.path.join(viz_dir, "anchors.html"),
        title="SAJO: contact anchors",
    )

    # Fitted axis overlay on M_move
    save_axis_overlay_html(
        M_move=M_move_np,
        joint_type=em_selected.joint_type,
        omega_world=em_selected.omega.detach().cpu().numpy(),
        q_world=em_selected.q.detach().cpu().numpy(),
        out_path=os.path.join(viz_dir, "axis_overlay.html"),
        title=f"SAJO: fitted {em_selected.joint_type} axis",
    )

    # EM traces + BIC bar chart
    rev_trace = em_rev_dict.get("L_reg_trace", []) if isinstance(em_rev_dict, dict) else []
    pris_trace = em_pris_dict.get("L_reg_trace", []) if isinstance(em_pris_dict, dict) else []
    save_em_traces_html(
        rev_trace=rev_trace,
        pris_trace=pris_trace,
        bic=bic_report,
        out_path=os.path.join(viz_dir, "em_traces.html"),
    )

    return SAJOResult(
        joint_type=em_selected.joint_type,
        omega=em_selected.omega,
        q=em_selected.q,
        v=em_selected.v,
        phi_k=em_selected.phi_k,
        M_base=M_base,
        M_move=M_move,
        T_k_list=em_selected.T_k_list,
        p_base=p_base,
        p_move=p_move,
        anchors_meta=dict(anchors_meta),
        bic=bic_report,
        em_rev=em_rev_dict,
        em_pris=em_pris_dict,
        converged=bool(em_selected.converged),
        n_iters=int(em_selected.n_iters),
    )

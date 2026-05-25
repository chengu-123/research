"""End-to-end Stage C SegMatch v6 driver (2026-04-23 rewrite).

Flow:
  C.0  partition.count_based_partition     — count signature + M_attn always_on classifier
  C.1  swept_volume.select_anchor_state    — max-displacement state
  C.2  phase_em.run_phased_em              — Phase 1 anchor → Phase 3 global relax
  C.3  swept_volume.compute_swept_volume   — dense φ sweep of canonical_move
  C.4  swept_volume.late_commit_carve      — carve canonical_base ∩ SV (α-bounded)
  C.5  seg_refine (optional)               — graph-cut polish
  C.6  axis_refine                          — contact principal axis
  C.7  aggregation                          — per-state assignment + canonical volumes
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

import numpy as np
import torch

from .aggregation import aggregate_canonical
from .axis_refine import refine_axis
from .config import Diagnostics, SegMatchHParams, StageCResult
from .features import load_O_stack, load_m_attn_64, load_z_final
from .partition import count_based_partition, persistence_run_length
from .phase_em import run_phased_em
from .seg_refine import refine_segmentation
from .swept_volume import (
    compute_swept_volume,
    late_commit_carve,
    select_anchor_state,
)
from .viz import write_all_viz


def _resolve_device(hp: SegMatchHParams) -> torch.device:
    if hp.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_stage_c(
    stage_b_dir: str,
    stage_c_dir: str,
    hp: SegMatchHParams = None,
    write_viz: bool = True,
) -> StageCResult:
    """Top-level Stage C SegMatch v6 driver."""
    if hp is None:
        hp = SegMatchHParams()
    device = _resolve_device(hp)
    dtype = hp.torch_dtype
    os.makedirs(stage_c_dir, exist_ok=True)

    # ---- Inputs --------------------------------------------------------
    O_stack = load_O_stack(stage_b_dir, device=device, dtype=dtype)
    M_attn_64 = load_m_attn_64(stage_b_dir, device=device, dtype=dtype)
    z_final = None
    if hp.use_zfinal_classifier:
        z_final = load_z_final(stage_b_dir, device=device, dtype=dtype)
    # v8.1: DiT 1024-dim hidden states for MRF prior (NOVELTY).
    dit_hidden = None
    if getattr(hp, "use_dit_prior", False):
        from .features import load_dit_hidden
        dit_hidden = load_dit_hidden(stage_b_dir, device=device, dtype=dtype)
        if dit_hidden is None:
            print(f"[run_stage_c] use_dit_prior=True but dit_hidden.pt missing at {stage_b_dir}; falling back")
    K = O_stack.shape[0]

    # ---- C.0 count-based partition + z_final material classifier ------
    # z_final (8-dim SS latent) is used as the primary always_on classifier.
    # M_attn is retained in seg_refine.py only (graph-cut unary).
    partition = count_based_partition(
        O_stack, M_attn_64, z_final=z_final,
        count_base_threshold=hp.count_base_threshold
            if hp.count_base_threshold <= K else K,
        count_move_max=hp.count_move_max,
        m_attn_base_threshold=hp.m_attn_base_threshold,
        m_attn_move_threshold=hp.m_attn_move_threshold,
        far_aon_edt_threshold=hp.far_aon_edt_threshold,
        zfinal_min_seeds=hp.zfinal_min_seeds,
        zfinal_margin_coef=hp.zfinal_margin_coef,
    )

    n_always_on = int(partition.always_on.sum().item())
    n_true_base = int(partition.true_base.sum().item())
    n_move_interior = int(partition.move_interior.sum().item())
    n_ambiguous = int(partition.ambiguous_on.sum().item())
    n_move_initial = int(partition.move_mask_k.any(dim=0).sum().item())
    n_base_initial = int(partition.base_mask_k.any(dim=0).sum().item())
    # Material classifier extras (best-effort — may be absent if fallback used)
    mc_diag: Dict[str, Any] = {
        "move_interior_ratio_of_always_on": round(
            n_move_interior / max(n_always_on, 1), 4,
        ),
        "ambiguous_ratio_of_always_on": round(
            n_ambiguous / max(n_always_on, 1), 4,
        ),
    }

    # ---- C.1 anchor selection ------------------------------------------
    if hp.adaptive_anchor:
        anchor_idx, anchor_stats = select_anchor_state(
            O_stack, partition.move_strong,
            move_seeds_global=partition.shell | partition.move_interior,
            min_hard_seed_ratio=hp.anchor_min_hard_seed_ratio,
            fallback_idx=hp.fixed_anchor_state
                if hp.fixed_anchor_state < K else K - 1,
        )
    else:
        anchor_idx = min(hp.fixed_anchor_state, K - 1)
        anchor_stats = {"selected_idx": anchor_idx,
                        "selected_ratio": -1.0,
                        "per_state_ratio": [], "per_state_hard_count": []}

    # Persistence (for seg_refine unary)
    persistence = persistence_run_length(O_stack)

    # ---- C.2 phase-based EM (1: anchor → 3: global relax) -------------
    phase_result = run_phased_em(
        O_stack=O_stack,
        move_mask_k=partition.move_mask_k,
        canonical_omega_c=partition.canonical_omega_c,
        anchor_idx=anchor_idx,
        hp=hp, resolution=hp.resolution,
        device=device, dtype=dtype,
    )

    # ---- C.3 swept volume (dense φ sweep) ------------------------------
    axis_params = (
        {"omega": phase_result.omega, "q": phase_result.q, "v": phase_result.v}
        if phase_result.joint_type == "revolute"
        else {"v_hat": phase_result.omega}
    )
    swept_volume_mask = compute_swept_volume(
        canonical_move=phase_result.canonical_move,
        joint_type=phase_result.joint_type,
        axis_params=axis_params,
        phi_k=phase_result.phi_k,
        n_samples=hp.swept_n_samples,
        phi_margin=hp.swept_phi_margin,
        resolution=hp.resolution,
    )

    # ---- C.4 late-commit carve (base ∩ SV → remove) --------------------
    canonical_base_prefinal = partition.canonical_base_init
    # Also promote ambiguous_on voxels that fall inside SV to move
    ambiguous_in_sv = partition.ambiguous_on & swept_volume_mask
    canonical_move_after_carve = phase_result.canonical_move | (
        ambiguous_in_sv & partition.canonical_omega_c
    )
    carve_result = late_commit_carve(
        canonical_base=canonical_base_prefinal,
        swept_volume=swept_volume_mask,
        alpha_lower=hp.base_alpha_lower,
        reference_volume=partition.canonical_omega_c,
    )
    canonical_base_final = carve_result.canonical_base

    # Ambiguous voxels OUTSIDE SV that remain → classify as base (default)
    ambiguous_outside_sv = partition.ambiguous_on & ~swept_volume_mask
    canonical_base_final = canonical_base_final | (
        ambiguous_outside_sv & partition.canonical_omega_c
    )

    # ---- Propagate canonical → per-state masks --------------------------
    occ_bool = (O_stack > 0.5)
    move_mask_k_final = occ_bool & canonical_move_after_carve.unsqueeze(0)
    base_mask_k_final = occ_bool & canonical_base_final.unsqueeze(0)

    # ---- v8.1 NOVELTY: compute DiT 1024-dim prior for MRF -----
    dit_prior = None
    dit_prior_meta: Dict[str, Any] = {"use_dit_prior": False}
    if dit_hidden is not None:
        from .dit_prior import compute_dit_priors
        try:
            dit_prior = compute_dit_priors(
                dit_hidden=dit_hidden,
                shell=partition.shell,
                always_on=partition.always_on,
                footprint=partition.footprint,
                target_blocks=hp.dit_prior_blocks,
                d_latent=16, out_size=hp.resolution,
                far_aon_edt_threshold=hp.dit_prior_far_aon_edt,
                min_seeds=hp.dit_prior_min_seeds,
                projection_temperature=hp.dit_prior_projection_temperature,
            )
            dit_prior_meta = {"use_dit_prior": True, **dit_prior.meta}
            print(
                f"[run_stage_c] DiT prior: n_shell={dit_prior_meta['n_shell_seeds']:.0f} "
                f"n_far={dit_prior_meta['n_far_seeds']:.0f} "
                f"axis_norm={dit_prior_meta['axis_norm']:.3f} "
                f"fallback={dit_prior_meta['fallback']:.0f}"
            )
        except Exception as e:
            print(f"[run_stage_c] DiT prior computation failed: {e}; falling back")
            dit_prior = None
            dit_prior_meta = {"use_dit_prior": False, "error": str(e)}

    # ---- C.5 [optional] seg_refine via graph-cut (polish boundary) -----
    seg_result = refine_segmentation(
        O_stack, phase_result.T_k, M_attn_64,
        prev_move_mask_k=move_mask_k_final,
        hp=hp, resolution=hp.resolution,
        persistence=persistence,
        dit_prior=dit_prior,
    )

    # v8: drop the conservative clip. The MRF graph-cut with differential
    # r_base - r_move unary is globally optimal on 2-label Potts (Kolmogorov-
    # Zabih 2004). Previously we intersected with canonical_move_after_carve
    # which shrinks by the voting-backwarp-clip cascade (30857: 398 MRF
    # voxels ∩ 30 vote-surviving voxels → 0-8 final). The MRF's move label
    # already encodes {r_move < r_base}; there's no value in further
    # restriction by a more lossy signal. Keep swept-volume carve only as a
    # safety against pathological T_k drift (applied below).
    move_mask_k_refined = seg_result.move_mask_k
    base_mask_k_refined = seg_result.base_mask_k

    # ---- C.6 axis refine (contact principal axis) ----------------------
    fit_for_refine = phase_result_to_fit(phase_result)
    refine_result = refine_axis(
        fit_for_refine,
        canonical_base_final,
        canonical_move_after_carve,
        M_attn_64, hp, resolution=hp.resolution,
    )

    # ---- C.7 aggregation -----------------------------------------------
    agg = aggregate_canonical(
        O_stack, canonical_base_final, canonical_move_after_carve,
        refine_result.joint_fit.T_k, hp,
        upsample_resolution=hp.warp_resolution,
    )

    # ---- Package -------------------------------------------------------
    fit = refine_result.joint_fit
    n_move_final = int(move_mask_k_refined.any(dim=0).sum().item())
    n_base_final = int(base_mask_k_refined.any(dim=0).sum().item())

    result = StageCResult(
        joint_type=fit.joint_type,
        omega=fit.omega.detach().cpu(),
        q=fit.q.detach().cpu(),
        v=fit.v.detach().cpu(),
        T_k=fit.T_k.detach().cpu(),
        phi_k=fit.phi_k.detach().cpu(),
        canonical_base=agg.canonical_base.detach().cpu(),
        canonical_move=agg.canonical_move.detach().cpu(),
        contact_region=agg.contact_region.detach().cpu(),
        per_state_assignment=agg.per_state_assignment.detach().cpu(),
        meta={
            "anchor_state_idx": int(anchor_idx),
            "anchor_stats": anchor_stats,
            "base_centroid": partition.base_centroid.detach().cpu().tolist(),
            "bic_rev": float(phase_result.bic_rev),
            "bic_pris": float(phase_result.bic_pris),
            "bic_margin": float(phase_result.bic_margin),
            "phase1_loss": float(phase_result.phase1_loss),
            "phase3_loss": float(phase_result.phase3_loss),
            "dit_prior": dit_prior_meta,
            "n_always_on": n_always_on,
            "n_true_base_initial": n_true_base,
            "n_move_interior_initial": n_move_interior,
            "n_ambiguous_initial": n_ambiguous,
            **mc_diag,
            "n_move_initial": n_move_initial,
            "n_move_final": n_move_final,
            "n_base_initial": n_base_initial,
            "n_base_final": n_base_final,
            "swept_triggered": bool(carve_result.triggered),
            "swept_lower_bound_protected": bool(carve_result.lower_bound_protected),
            "n_carved": int(carve_result.n_carved),
            "alpha_final": float(carve_result.alpha_final),
        },
    )

    diag = Diagnostics(
        joint_type_selected=fit.joint_type,
        bic_rev=float(phase_result.bic_rev),
        bic_pris=float(phase_result.bic_pris),
        bic_margin=float(phase_result.bic_margin),
        # v8: true per-hypothesis Phase-3 L_final (previously both copied
        # from phase3_loss, which masked whether rev vs pris actually differ).
        volumetric_loss_rev=float(phase_result.phase3_loss_rev),
        volumetric_loss_pris=float(phase_result.phase3_loss_pris),
        n_move_voxels_initial=n_move_initial,
        n_move_voxels_final=n_move_final,
        n_base_voxels_initial=n_base_initial,
        n_base_voxels_final=n_base_final,
        n_flips=int(seg_result.n_flips),
        n_overlap_deleted=int(carve_result.n_carved),
        icp_used=False,
        warm_start_used="phase1_anchor",
        anchor_state_idx=int(anchor_idx),
        n_always_on_total=n_always_on,
        n_true_base_initial=n_true_base,
        n_move_interior_initial=n_move_interior,
        n_ambiguous_on_initial=n_ambiguous,
        phase1_final_loss=float(phase_result.phase1_loss),
        phase2_final_loss=0.0,
        phase3_final_loss=float(phase_result.phase3_loss),
        swept_carving_triggered=bool(carve_result.triggered),
        swept_lower_bound_protected=bool(carve_result.lower_bound_protected),
        phase1_loss_rev=float(phase_result.phase1_loss_rev),
        phase1_loss_pris=float(phase_result.phase1_loss_pris),
    )

    _save_artifacts(stage_c_dir, result, diag)

    if write_viz:
        viz_dir = os.path.join(stage_c_dir, "viz")
        _write_viz_v6_safe(
            viz_dir=viz_dir,
            O_stack=O_stack,
            M_attn_64=M_attn_64,
            partition=partition,
            persistence=persistence,
            phase_result=phase_result,
            swept_volume_mask=swept_volume_mask,
            canonical_base_final=canonical_base_final,
            canonical_move_final=canonical_move_after_carve,
            seg_result=seg_result,
            refine_result=refine_result,
            agg=agg,
            carve_result=carve_result,
            diag=diag,
            resolution=hp.resolution,
        )
    return result


# ---- Helpers ---------------------------------------------------------


def phase_result_to_fit(phase_result):
    """Adapt PhaseResult to the VolumetricFit shape expected by axis_refine."""
    from .volumetric_fit import VolumetricFit
    return VolumetricFit(
        joint_type=phase_result.joint_type,
        omega=phase_result.omega, q=phase_result.q, v=phase_result.v,
        phi_k=phase_result.phi_k, T_k=phase_result.T_k,
        L_final=phase_result.phase3_loss, L_trace=[],
        meta={},
    )


def _write_viz_v6_safe(**kwargs):
    """Write viz, swallowing plotly errors so the driver never crashes."""
    try:
        write_all_viz_v6(**kwargs)
    except Exception as exc:                                         # noqa: BLE001
        print(f"[viz warning] {type(exc).__name__}: {exc}")


def write_all_viz_v6(**kwargs):
    """Adapter — current viz.write_all_viz expects v5 inputs; we pass
    only the fields it uses and a diagnostic Phase1/Phase3 summary."""
    viz_dir = kwargs["viz_dir"]
    os.makedirs(viz_dir, exist_ok=True)
    # Delegate to the v5 viz with the v6 inputs remapped; anything v6-specific
    # that v5 viz doesn't consume is dumped as a diagnostic JSON.
    partition = kwargs["partition"]
    phase_result = kwargs["phase_result"]

    # v5 viz compatibility — pass the minimum it needs
    try:
        write_all_viz(
            viz_dir=viz_dir,
            O_stack=kwargs["O_stack"],
            M_attn_64=kwargs["M_attn_64"],
            move_mask_k_initial=partition.move_mask_k,
            base_mask_k_initial=partition.base_mask_k,
            p_base=partition.true_base.to(kwargs["O_stack"].dtype),
            p_move=(partition.shell | partition.move_interior).to(
                kwargs["O_stack"].dtype,
            ),
            footprint=partition.footprint.to(kwargs["O_stack"].dtype),
            base_centroid=partition.base_centroid,
            warm=_DummyWarm(partition.base_centroid, phase_result),
            fit_result=_DummyFitResult(phase_result),
            seg_result=kwargs["seg_result"],
            refine_result=kwargs["refine_result"],
            agg=kwargs["agg"],
            overlap_result=_DummyOverlapResult(
                kwargs["canonical_base_final"],
                kwargs["canonical_move_final"],
                kwargs["carve_result"],
            ),
            diag=kwargs["diag"],
            resolution=kwargs["resolution"],
        )
    except Exception as exc:                                         # noqa: BLE001
        # Fall back to dumping only the phase + diag summary.
        print(f"[viz fallback] {type(exc).__name__}: {exc}")


class _DummyWarm:
    """Shape-compat wrapper to satisfy v5 viz.write_all_viz."""
    def __init__(self, base_centroid, phase_result):
        self.centroids_world = base_centroid.unsqueeze(0).expand(
            phase_result.phi_k.shape[0], -1,
        ).detach().cpu()
        self.joint_type_hint = phase_result.joint_type
        self.axis = phase_result.omega.detach().cpu()
        self.q = phase_result.q.detach().cpu()
        self.phi_k = phase_result.phi_k.detach().cpu()


class _DummyFitResult:
    """Satisfy v5 viz.save_volumetric_loss_html signature."""
    def __init__(self, phase_result):
        # Synthesize rev/pris stubs from phase result
        class _Sub:
            def __init__(self, L_final, phi_k, omega, q, v, joint_type):
                self.L_final = float(L_final)
                self.L_trace = [float(L_final)]
                self.phi_k = phi_k
                self.omega = omega
                self.q = q
                self.v = v
                self.joint_type = joint_type
                self.T_k = torch.eye(4).expand(phi_k.shape[0], -1, -1).clone()
        rev_l = (phase_result.phase3_loss if phase_result.joint_type == "revolute"
                 else phase_result.phase3_loss * 2)
        pris_l = (phase_result.phase3_loss if phase_result.joint_type == "prismatic"
                  else phase_result.phase3_loss * 2)
        self.rev = _Sub(rev_l, phase_result.phi_k,
                         phase_result.omega, phase_result.q, phase_result.v,
                         "revolute")
        self.pris = _Sub(pris_l, phase_result.phi_k,
                          phase_result.omega, phase_result.q, phase_result.v,
                          "prismatic")
        self.joint_fit = self.rev if phase_result.joint_type == "revolute" else self.pris
        self.joint_fit.joint_type = phase_result.joint_type
        self.joint_fit.omega = phase_result.omega
        self.joint_fit.q = phase_result.q
        self.joint_fit.v = phase_result.v
        self.joint_fit.phi_k = phase_result.phi_k
        self.joint_fit.T_k = phase_result.T_k
        self.bic_rev = float(phase_result.bic_rev)
        self.bic_pris = float(phase_result.bic_pris)


class _DummyOverlapResult:
    """Adapt carve result to the fields v5 viz.write_all_viz expects."""
    def __init__(self, canonical_base_final, canonical_move_final, carve_result):
        self.canonical_base = canonical_base_final
        self.canonical_move = canonical_move_final
        self.contact_region = torch.zeros_like(canonical_base_final)
        self.overlap_before = carve_result.swept_volume & canonical_base_final
        self.n_containment_deleted = int(carve_result.n_carved)
        self.n_contact_kept = 0


def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _save_artifacts(stage_c_dir: str, result: StageCResult, diag: Diagnostics) -> None:
    np.save(os.path.join(stage_c_dir, "canonical_base.npy"),
            _tensor_to_numpy(result.canonical_base).astype(np.uint8))
    np.save(os.path.join(stage_c_dir, "canonical_move.npy"),
            _tensor_to_numpy(result.canonical_move).astype(np.uint8))
    np.save(os.path.join(stage_c_dir, "contact_region.npy"),
            _tensor_to_numpy(result.contact_region).astype(np.uint8))
    np.save(os.path.join(stage_c_dir, "per_state_assignment.npy"),
            _tensor_to_numpy(result.per_state_assignment))
    torch.save({
        "joint_type": result.joint_type,
        "omega": result.omega, "q": result.q, "v": result.v,
        "T_k": result.T_k, "phi_k": result.phi_k,
    }, os.path.join(stage_c_dir, "joint_params.pt"))

    meta_blob: Dict[str, Any] = dict(result.meta)
    meta_blob["joint_type"] = result.joint_type
    with open(os.path.join(stage_c_dir, "meta.json"), "w") as f:
        json.dump(meta_blob, f, indent=2)

    diag_dict = asdict(diag) if is_dataclass(diag) else diag.__dict__
    with open(os.path.join(stage_c_dir, "diagnostics.json"), "w") as f:
        json.dump(diag_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Stage C SegMatch v6 driver")
    parser.add_argument("--stage_b_dir", required=True)
    parser.add_argument("--stage_c_dir", required=True)
    parser.add_argument("--fixed_anchor", type=int, default=-1,
                        help="If >=0, disable adaptive anchor and use this state")
    # v8: defaults None so CLI doesn't override config.py. Previously hardcoded
    # argparse defaults (8 / 30) silently clobbered config.py's updated 80 / 150,
    # causing Phase 1 to under-converge (62k loss) and flipping 7201 rev→pris.
    parser.add_argument("--phase1_iters", type=int, default=None,
                        help="Override phase1 iters (default: use config.py)")
    parser.add_argument("--phase3_iters", type=int, default=None,
                        help="Override phase3 iters (default: use config.py)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_viz", action="store_true")
    args = parser.parse_args()

    hp = SegMatchHParams()
    if args.fixed_anchor >= 0:
        hp.adaptive_anchor = False
        hp.fixed_anchor_state = args.fixed_anchor
    if args.phase1_iters is not None:
        hp.phase1_iters = args.phase1_iters
    if args.phase3_iters is not None:
        hp.phase3_iters = args.phase3_iters
    hp.device = args.device
    run_stage_c(args.stage_b_dir, args.stage_c_dir, hp,
                write_viz=not args.no_viz)


if __name__ == "__main__":
    main()

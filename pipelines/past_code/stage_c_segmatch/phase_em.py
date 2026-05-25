"""Phase-based EM orchestration (stagec_3.md).

Three phases:

* **Phase 1 — anchor**: Fix all T_k = I except the chosen anchor state,
  estimate T_anchor alone from a canonical_move initialized by
  backwarping the anchor's move seeds. Converges a single high-quality
  screw transform.

* **Phase 2 — sequential propagation**: With T_anchor fixed, linearly
  interpolate warm-starts for the other states along the screw axis,
  then refine each T_k locally. Direction k=anchor-1, anchor-2, ..., 0
  and k=anchor+1, ..., K-1.

* **Phase 3 — global relax**: All T_k and canonical_move jointly
  updated. Monotonicity prior on φ_k. Late-commit swept-volume
  carving applied at the end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from ..sajo.screw import (
    exp_prismatic,
    exp_se3,
    project_prismatic,
    project_revolute,
)
from ..sajo.warp import batch_trilinear_warp, trilinear_warp
from .moments import (
    _fit_prismatic,
    _fit_revolute,
    _fit_revolute_inertia,
    compute_move_centroids,
)
from .swept_volume import compute_canonical_move_vote
from .volumetric_fit import (
    VolumetricFit,
    fit_single_state_anchor,
    volumetric_fit_pipeline,
)


@dataclass
class PhaseResult:
    joint_type: str
    omega: torch.Tensor                 # (3,) axis direction / v_hat
    q: torch.Tensor                     # (3,) axis pivot
    v: torch.Tensor                     # (3,) Plücker moment / v_hat
    phi_k: torch.Tensor                 # (K,)
    T_k: torch.Tensor                   # (K, 4, 4)
    canonical_move: torch.Tensor        # (D, H, W) bool, clipped to Ω_c
    phase1_loss: float = float("inf")
    phase1_loss_rev: float = float("inf")   # v8: Phase-1 rev fit L_final
    phase1_loss_pris: float = float("inf")  # v8: Phase-1 pris fit L_final
    phase2_losses: List[float] = None
    phase3_loss: float = float("inf")
    phase3_loss_rev: float = float("inf")   # v8: Phase-3 rev fit L_final
    phase3_loss_pris: float = float("inf")  # v8: Phase-3 pris fit L_final
    bic_rev: float = float("inf")
    bic_pris: float = float("inf")
    bic_margin: float = 0.0

    def __post_init__(self):
        if self.phase2_losses is None:
            self.phase2_losses = []


# ---- Phase 1 bootstrap ------------------------------------------------


def _warm_start_single_state(
    O_stack: torch.Tensor,
    move_mask_k: torch.Tensor,
    anchor_idx: int,
    canonical_omega_c: torch.Tensor,
    resolution: int = 64,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Moment-match between canonical O_0 move region and anchor state.

    **v8 upgrade (2026-04-24)**: previously used a crude 2-point
    shell-centroid warm start where ``omega = cross(v_hat, ref_up)`` gave
    an arbitrary rotation axis unrelated to the true hinge direction. For
    a rotating door (7201 oven, 90° around bottom edge), this produced
    garbage warm start → Phase 3 could not recover → pris BIC won
    incorrectly. v8 uses the AOF-Path-A inertia-tensor trajectory
    (:func:`moments._fit_revolute_inertia`) which tracks the principal
    axis of the move volume across states — for elongated parts the
    principal axis swings clearly with rotation, giving a much better
    omega init than centroid-cross-product.

    Returns ``{"revolute": init_r, "prismatic": init_p}`` each containing
    a single-state anchor (``phi_anchor`` scalar).
    """
    device = O_stack.device
    dtype = O_stack.dtype
    K = O_stack.shape[0]

    # Canonical move init from state 0 seeds (canonical_move_init at time 0)
    # We approximate canonical_move as (move_mask_k[0] & Ω_c) — effectively
    # state 0's move voxels.
    canonical_move_0 = (move_mask_k[0] & canonical_omega_c).to(dtype)
    anchor_move = (move_mask_k[anchor_idx] & (O_stack[anchor_idx] > 0.5)).to(dtype)

    # ----- Prismatic warm start: 2-point centroid (unchanged, works OK) ---
    idx = torch.arange(canonical_move_0.shape[0], device=device, dtype=dtype)
    ii, jj, kk = torch.meshgrid(idx, idx, idx, indexing="ij")
    coord = torch.stack([ii, jj, kk], dim=-1) / float(resolution - 1) - 0.5

    def _centroid(mask: torch.Tensor) -> torch.Tensor:
        w = mask.clamp_min(0.0)
        w_sum = w.sum().clamp_min(1e-8)
        return (w.unsqueeze(-1) * coord).sum(dim=(0, 1, 2)) / w_sum

    c0 = _centroid(canonical_move_0)
    c_anchor = _centroid(anchor_move)
    delta = c_anchor - c0
    delta_norm = delta.norm().clamp_min(1e-8)
    v_hat = delta / delta_norm
    phi_pris = float(delta_norm.item())

    # ----- Revolute warm start: AOF-Path-A inertia-tensor (v8) -----------
    try:
        omega_try, q_try, phi_k_try, mse_try = _fit_revolute_inertia(
            O_stack, move_mask_k, resolution=resolution,
        )
        inertia_ok = bool(mse_try != float("inf")) and bool(
            torch.isfinite(phi_k_try).all().item()
        )
    except Exception:
        inertia_ok = False
        omega_try = None

    if inertia_ok and omega_try is not None:
        omega_init = omega_try.to(device=device, dtype=dtype).detach()
        q_init = q_try.to(device=device, dtype=dtype).detach()
        # v8: the inertia circle-fit can return phi ≈ π when centroid arc is
        # ambiguous (observed on 7201 oven door — phi=180° despite 90° true
        # rotation). We keep omega (very reliable direction from principal-
        # axis trajectory) but recompute phi via the 2-point centroid
        # radius formula, which is bounded to [0, π] and more conservative.
        # Phase 1 Adam can grow phi if needed — but starting at π traps us
        # in a bad local minimum.
        delta_perp = delta - (delta @ omega_init) * omega_init
        r_est = (c_anchor - q_init).norm().clamp_min(1e-3)
        phi_rev = float(2.0 * torch.arcsin(
            (delta_perp.norm() / (2.0 * r_est)).clamp(0.0, 1.0)
        ).item())
        # Cap the initial guess at π/2 — doors rarely rotate past 90° and
        # starting beyond 90° forces the optimizer to cross a high-loss ridge.
        phi_rev = min(phi_rev, 1.5708)
    else:
        # Legacy fallback: 2-point cross-product heuristic
        ref_up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        if abs(float((v_hat * ref_up).sum().item())) > 0.95:
            ref_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        omega_init = torch.linalg.cross(v_hat, ref_up)
        omega_init = omega_init / omega_init.norm().clamp_min(1e-8)
        q_init = 0.5 * (c0 + c_anchor)
        q_init = q_init - (q_init @ omega_init) * omega_init
        delta_perp = delta - (delta @ omega_init) * omega_init
        r_est = (c_anchor - q_init).norm().clamp_min(1e-3)
        phi_rev = float(2.0 * torch.arcsin(
            (delta_perp.norm() / (2.0 * r_est)).clamp(0.0, 1.0)
        ).item())

    return {
        "revolute": {
            "omega": omega_init.detach(),
            "q": q_init.detach(),
            "phi_anchor": phi_rev,
        },
        "prismatic": {
            "v_hat": v_hat.detach(),
            "phi_anchor": phi_pris,
        },
    }


def run_phase1_anchor(
    O_stack: torch.Tensor,
    move_mask_k: torch.Tensor,
    canonical_omega_c: torch.Tensor,
    anchor_idx: int,
    hp,
    resolution: int = 64,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[VolumetricFit, VolumetricFit]:
    """Fit T_anchor for both revolute and prismatic independently."""
    if device is None:
        device = O_stack.device

    warm = _warm_start_single_state(
        O_stack, move_mask_k, anchor_idx, canonical_omega_c,
        resolution=resolution,
    )

    # Canonical move init for Phase 1 = (state 0 move_mask clipped to Ω_c)
    canonical_move_init = (move_mask_k[0] & canonical_omega_c).to(dtype)
    O_anchor = O_stack[anchor_idx].to(device=device, dtype=dtype)
    # v8: anchor-state move mask — target for BCE is the MOVE REGION at
    # anchor state, not the full O_anchor. Removes cabinet noise floor.
    move_mask_anchor = move_mask_k[anchor_idx].to(device=device, dtype=dtype)

    rev = fit_single_state_anchor(
        O_anchor, O_stack, canonical_move_init,
        anchor_state_idx=anchor_idx,
        joint_type="revolute", init_params=warm["revolute"],
        n_inner_steps=hp.phase1_iters,
        lr_axis=hp.phase1_lr_axis, lr_phi=hp.phase1_lr_phi,
        move_mask_anchor=move_mask_anchor,
        resolution=resolution, device=device, dtype=dtype,
    )
    pris = fit_single_state_anchor(
        O_anchor, O_stack, canonical_move_init,
        anchor_state_idx=anchor_idx,
        joint_type="prismatic", init_params=warm["prismatic"],
        n_inner_steps=hp.phase1_iters,
        lr_axis=hp.phase1_lr_axis, lr_phi=hp.phase1_lr_phi,
        move_mask_anchor=move_mask_anchor,
        resolution=resolution, device=device, dtype=dtype,
    )
    return rev, pris


# ---- Phase 2 / 3 use the existing volumetric_fit_pipeline ------------


def run_phase3_global_relax(
    O_stack: torch.Tensor,
    move_mask_k: torch.Tensor,
    rev_init: Dict[str, torch.Tensor],
    pris_init: Dict[str, torch.Tensor],
    canonical_omega_c: torch.Tensor,
    hp,
    resolution: int = 64,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
):
    """All T_k jointly — reuses volumetric_fit_pipeline with updated input.

    The move_mask_k passed here should INCLUDE move_interior voxels so
    the variance loss has its true global minimum at the correct motion
    magnitude (see long-drawer analysis in stagec_3.md §数学验证).
    """
    # Respect hp for inner steps and lr — volumetric_fit_pipeline uses
    # hp.fit_inner_steps, hp.fit_lr_axis, hp.fit_lr_phi. Override via a
    # local copy so Phase 3 uses the phase3-specific lrs.
    class _HpView:
        pass
    view = _HpView()
    view.fit_inner_steps = hp.phase3_iters
    view.fit_lr_axis = hp.phase3_lr_axis
    view.fit_lr_phi = hp.phase3_lr_phi
    view.fit_weight_decay = hp.fit_weight_decay
    # v8: forward monotonicity_lambda for zigzag prevention on non-monotonic
    # TRELLIS SCAR samples (7128 door state 3 > state 4-5 XOR).
    view.monotonicity_lambda = float(getattr(hp, "monotonicity_lambda", 0.0))

    warm_dict = {"revolute": rev_init, "prismatic": pris_init}
    result = volumetric_fit_pipeline(
        O_stack, move_mask_k, warm_dict, view,
        resolution=resolution, device=device, dtype=dtype,
    )
    return result


# ---- Top-level phase driver ------------------------------------------


def run_phased_em(
    O_stack: torch.Tensor,
    move_mask_k: torch.Tensor,
    canonical_omega_c: torch.Tensor,
    anchor_idx: int,
    hp,
    resolution: int = 64,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> PhaseResult:
    """Orchestrate Phase 1 (anchor) → Phase 3 (global relax).

    Phase 2 (sequential propagation) is implicitly handled by the
    warm-started Phase 3 volumetric_fit_pipeline — given a good
    T_anchor, Adam on the full joint-constrained loss converges
    quickly. Explicit per-state sequential propagation is deferred
    to a future ablation (adds complexity without proven gain).
    """
    if device is None:
        device = O_stack.device

    # Phase 1: anchor (rev + pris tried independently)
    rev1, pris1 = run_phase1_anchor(
        O_stack, move_mask_k, canonical_omega_c,
        anchor_idx=anchor_idx, hp=hp,
        resolution=resolution, device=device, dtype=dtype,
    )

    # Select provisional type by Phase-1 loss (BIC will confirm in Phase 3)
    provisional = "revolute" if rev1.L_final < pris1.L_final else "prismatic"

    # Build Phase-3 warm-starts from Phase-1 anchors.
    # Propagate φ linearly from anchor: φ_k = (k / anchor_idx) * φ_anchor
    # (stagec_3.md §Phase 2 recipe; equivalent to linear interpolation)
    K = O_stack.shape[0]

    def _linear_propagate(phi_anchor_idx: int, phi_k_anchor: torch.Tensor) -> torch.Tensor:
        phi_k = torch.zeros(K, device=device, dtype=dtype)
        if phi_anchor_idx == 0:
            return phi_k
        for k in range(K):
            phi_k[k] = phi_k_anchor[phi_anchor_idx] * (float(k) / float(phi_anchor_idx))
        return phi_k

    rev_init = {
        "omega": rev1.omega.detach(),
        "q": rev1.q.detach(),
        "phi_k": _linear_propagate(anchor_idx, rev1.phi_k),
    }
    pris_init = {
        "v_hat": pris1.omega.detach(),
        "phi_k": _linear_propagate(anchor_idx, pris1.phi_k),
    }

    # Phase 3: global relax on all K transforms
    p3 = run_phase3_global_relax(
        O_stack, move_mask_k, rev_init, pris_init, canonical_omega_c,
        hp=hp, resolution=resolution, device=device, dtype=dtype,
    )

    # Canonical move from the final fit
    fit = p3.joint_fit
    vote_result_phase3 = compute_canonical_move_vote(
        O_stack, move_mask_k, fit.T_k, canonical_omega_c,
        vote_method=hp.vote_method,
        hard_vote_threshold=hp.hard_vote_threshold,
        resolution=resolution,
    )

    return PhaseResult(
        joint_type=fit.joint_type,
        omega=fit.omega, q=fit.q, v=fit.v,
        phi_k=fit.phi_k, T_k=fit.T_k,
        canonical_move=vote_result_phase3.canonical_move,
        phase1_loss=min(rev1.L_final, pris1.L_final),
        phase1_loss_rev=float(rev1.L_final),
        phase1_loss_pris=float(pris1.L_final),
        phase2_losses=[],
        phase3_loss=fit.L_final,
        phase3_loss_rev=float(p3.rev.L_final),
        phase3_loss_pris=float(p3.pris.L_final),
        bic_rev=p3.bic_rev,
        bic_pris=p3.bic_pris,
        bic_margin=p3.bic_margin,
    )

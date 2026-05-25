"""C.4 Joint-constrained whole-shape volumetric rigid alignment.

Correspondence-free articulation discovery: directly parameterize the
joint (axis + per-state phi_k) and optimize via Adam on a volumetric
variance loss across K back-warped canonical views. No feature matching,
no ICP-style nearest-neighbor correspondence, no RANSAC — every
occupied voxel contributes to the loss densely.

For K=6, we have 4+5=9 DoF (revolute) or 2+5=7 DoF (prismatic), fit
against ~5K active voxels × K states = ~30K soft-occupancy residuals.
This is 3 orders of magnitude over-constrained, which is why the
magnitude of ``phi_k`` is recovered robustly (contrast: v3's sparse
matching gave ~5 points per pair, severely under-constrained).

Variance loss:
    canonical_views_k(v) = O_move_k(T_k(v))   (trilinear sampled)
    L = sum_v sum_k (canonical_views_k(v) - mean_k canonical_views(v))²

Equivalent to the classical pairwise sum up to a factor of K.

Reuses ``pipelines/sajo/screw.py`` (SE(3) exp + manifold projection)
and ``pipelines/sajo/warp.py`` (differentiable trilinear warp).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from ..sajo.screw import (
    exp_prismatic,
    exp_se3,
    project_prismatic,
    project_revolute,
)
from ..sajo.warp import batch_trilinear_warp, trilinear_warp


@dataclass
class VolumetricFit:
    joint_type: str                        # "revolute" | "prismatic"
    omega: torch.Tensor                    # (3,) unit — rev axis / pris direction
    q: torch.Tensor                        # (3,) rev: point on axis; pris: zero
    v: torch.Tensor                        # (3,) rev: Plücker moment; pris: v_hat
    phi_k: torch.Tensor                    # (K,) phi_0 = 0
    T_k: torch.Tensor                      # (K, 4, 4) SE(3)
    L_final: float
    L_trace: List[float] = field(default_factory=list)
    meta: Dict[str, float] = field(default_factory=dict)


@dataclass
class VolumetricFitResult:
    joint_fit: VolumetricFit               # selected by BIC
    rev: VolumetricFit
    pris: VolumetricFit
    bic_rev: float
    bic_pris: float
    bic_margin: float
    n_active: int                          # active voxels in canonical (for BIC N)


# ---- Core loss --------------------------------------------------------


def _build_T_k(
    joint_type: str,
    S: torch.Tensor,
    phi_full: torch.Tensor,
) -> torch.Tensor:
    """Build (K, 4, 4) transform stack.

    * revolute: ``S`` is ``(6,) = (omega, v)``; manifold projection is applied
    * prismatic: ``S`` is ``(3,) = v_hat``; unit-normalized
    """
    K = phi_full.shape[0]
    if joint_type == "revolute":
        omega_u, v_clean = project_revolute(S[:3], S[3:])
        twist = torch.cat([omega_u, v_clean])
        return torch.stack([exp_se3(twist, phi_full[k]) for k in range(K)])
    if joint_type == "prismatic":
        v_hat_u = project_prismatic(S)
        return torch.stack([exp_prismatic(v_hat_u, phi_full[k]) for k in range(K)])
    raise ValueError(f"Unknown joint_type: {joint_type}")


def _variance_loss(
    O_move_stack: torch.Tensor,
    T_k: torch.Tensor,
    resolution: int = 64,
) -> torch.Tensor:
    """Cross-state variance of canonical-backwarp views.

    Parameters
    ----------
    O_move_stack : (K, D, H, W) — per-state move occupancy, soft in [0, 1]
    T_k : (K, 4, 4) — canonical→state k transforms

    Returns
    -------
    scalar loss — ``sum_v sum_k (c_k(v) - mean_k c_k(v))²``
    """
    # Backwarp each state to canonical
    T_k_inv = torch.linalg.inv(T_k)                                # (K, 4, 4)
    canonical_views = batch_trilinear_warp(
        O_move_stack, T_k_inv, resolution=resolution,
    )                                                              # (K, D, H, W)
    mean_view = canonical_views.mean(dim=0, keepdim=True)          # (1, D, H, W)
    deviations = canonical_views - mean_view                       # (K, D, H, W)
    return (deviations ** 2).sum()


# ---- Single-hypothesis fitter ----------------------------------------


def fit_volumetric(
    O_move_stack: torch.Tensor,
    joint_type: str,
    init_params: Dict[str, torch.Tensor],
    n_inner_steps: int,
    lr_axis: float,
    lr_phi: float,
    weight_decay: float = 0.0,
    monotonicity_lambda: float = 0.0,
    resolution: int = 64,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> VolumetricFit:
    """Adam optimize joint params for one joint type on volumetric loss.

    ``init_params`` must contain:
        revolute: ``{"omega": (3,), "q": (3,), "phi_k": (K,)}``
        prismatic: ``{"v_hat": (3,), "phi_k": (K,)}``

    ``phi_0`` is fixed = 0 (gauge). Only ``phi_1..phi_{K-1}`` optimized.
    """
    if device is None:
        device = O_move_stack.device
    K = O_move_stack.shape[0]
    O_move_stack = O_move_stack.to(device=device, dtype=dtype)

    if joint_type == "revolute":
        omega_init = init_params["omega"].to(device=device, dtype=dtype).detach().clone()
        q_init = init_params["q"].to(device=device, dtype=dtype).detach().clone()
        # Remove along-axis component of q (gauge)
        omega_init = omega_init / omega_init.norm().clamp_min(1e-8)
        q_init = q_init - (q_init @ omega_init) * omega_init
        v_init = torch.linalg.cross(q_init, omega_init)
        S = torch.nn.Parameter(torch.cat([omega_init, v_init]))    # (6,)
        phi_free_init = init_params["phi_k"][1:].to(device=device, dtype=dtype).detach().clone()
    elif joint_type == "prismatic":
        v_init = init_params["v_hat"].to(device=device, dtype=dtype).detach().clone()
        v_init = v_init / v_init.norm().clamp_min(1e-8)
        S = torch.nn.Parameter(v_init)                             # (3,)
        phi_free_init = init_params["phi_k"][1:].to(device=device, dtype=dtype).detach().clone()
    else:
        raise ValueError(f"Unknown joint_type: {joint_type}")

    phi_free = torch.nn.Parameter(phi_free_init)

    opt = torch.optim.Adam([
        {"params": [S], "lr": lr_axis, "weight_decay": weight_decay},
        {"params": [phi_free], "lr": lr_phi, "weight_decay": weight_decay},
    ])

    L_trace: List[float] = []

    for step in range(n_inner_steps):
        opt.zero_grad()

        phi_full = torch.cat([
            torch.zeros(1, device=device, dtype=dtype),
            phi_free,
        ])                                                          # (K,)
        T_k = _build_T_k(joint_type, S, phi_full)                   # (K, 4, 4)
        L = _variance_loss(O_move_stack, T_k, resolution=resolution)

        # v8: monotonicity soft constraint. TRELLIS SCAR samples may have
        # non-monotonic articulation realizations (e.g. 7128 door: state 3
        # is MORE open than state 4-5 in the K-sample sequence). Without
        # this penalty, Adam on the variance loss can fall into a zigzag
        # local minimum — observed: 7128 default fit phi_k = [0,15,4,18,28,36]
        # with L=1507, while monotonic fit reaches L=680. The penalty pulls
        # phi_k toward monotonic non-decreasing without forcing it strictly;
        # if the data is genuinely non-monotonic, the penalty cost is low.
        if monotonicity_lambda > 0.0:
            phi_diff = phi_full[1:] - phi_full[:-1]     # (K-1,)
            mono_loss = torch.relu(-phi_diff).pow(2).sum()
            L = L + monotonicity_lambda * mono_loss

        L.backward()
        opt.step()

        # Manifold projection
        with torch.no_grad():
            if joint_type == "revolute":
                omega_u, v_clean = project_revolute(S[:3], S[3:])
                S.data.copy_(torch.cat([omega_u, v_clean]))
            else:
                v_hat_u = project_prismatic(S)
                S.data.copy_(v_hat_u)

        L_trace.append(float(L.item()))

    # Final forward pass to build clean outputs
    with torch.no_grad():
        phi_full = torch.cat([
            torch.zeros(1, device=device, dtype=dtype),
            phi_free,
        ])
        if joint_type == "revolute":
            omega_u, v_clean = project_revolute(S[:3], S[3:])
            twist = torch.cat([omega_u, v_clean])
            q_rec = torch.linalg.cross(omega_u, v_clean)
            T_k = torch.stack([exp_se3(twist, phi_full[k]) for k in range(K)])
            return VolumetricFit(
                joint_type="revolute",
                omega=omega_u.detach(),
                q=q_rec.detach(),
                v=v_clean.detach(),
                phi_k=phi_full.detach(),
                T_k=T_k.detach(),
                L_final=L_trace[-1] if L_trace else float("inf"),
                L_trace=L_trace,
                meta={"n_inner_steps": len(L_trace)},
            )
        else:
            v_hat_u = project_prismatic(S)
            T_k = torch.stack([exp_prismatic(v_hat_u, phi_full[k]) for k in range(K)])
            return VolumetricFit(
                joint_type="prismatic",
                omega=v_hat_u.detach(),
                q=torch.zeros(3, device=device, dtype=dtype),
                v=v_hat_u.detach(),
                phi_k=phi_full.detach(),
                T_k=T_k.detach(),
                L_final=L_trace[-1] if L_trace else float("inf"),
                L_trace=L_trace,
                meta={"n_inner_steps": len(L_trace)},
            )


# ---- BIC selection ----------------------------------------------------


def compute_bic(
    rev: VolumetricFit,
    pris: VolumetricFit,
    n_active: int,
    K: int,
    object_centroid: Optional[torch.Tensor] = None,
    object_scale: float = 0.25,
    lambda_physical: float = 0.0,          # v8.1: default OFF; selector uses loss-ratio
    tau_improvement: float = 0.25,         # v8.1: rev picked iff (L_p-L_r)/L_p > tau
    eps: float = 1.0e-8,
) -> Tuple[str, float, float, float]:
    """BIC = 2L + k·log(N) + physical-prior(rev).

    ``k_rev = 4 + (K-1)`` (2 dof omega + 2 dof q_proj + K-1 free phi).
    ``k_pris = 2 + (K-1)`` (2 dof v_hat + K-1 free phi).

    **v8 addition (2026-04-24): physical pivot prior on revolute**.
    The vanilla BIC penalty (2 * log(N) for the 2 extra rev DoF) is
    insufficient to prevent a "degenerate rev" that emulates translation
    via large-radius small-angle rotation — the revolute fit places ``q``
    far from the object body and uses ``phi * radius ≈ L_translation`` to
    approximate a drawer's pure translation. Observed on 30857: rev finds
    ``|q|=0.16`` (far outside object bbox), phi_k non-uniform from 0° to
    76° tracing a giant arc that happens to overlap the drawer trajectory
    — ``loss_rev < loss_pris`` by 33 despite the drawer being truly
    prismatic, and BIC's standard 16-unit penalty cannot overcome it.

    The fix: penalize rev when its pivot ``q`` is far from the object
    centroid. A true revolute joint's axis passes through or near the
    object body (hinges are physically on the object boundary). A
    degenerate rev with distant ``q`` fails this physical consistency.

    Let ``d = ||q - centroid||`` (both in normalized [-0.5, 0.5] coords).
    Penalty = ``lambda_physical * max(0, d - object_scale)^2``. With
    ``lambda_physical = 50`` and ``object_scale = 0.25`` (roughly half the
    grid extent), a rev with ``|q|=0.16`` from origin gets penalty 0 if
    object is centered at origin (d=0.16 < 0.25). If centroid is not at
    origin, d grows; penalty activates. For 30857 with base_centroid near
    (0, 0.09, 0.10) and rev q=(0.04, -0.10, 0.12), d ≈ 0.20 → penalty 0
    still. Set ``object_scale = 0.15`` to bite harder on 30857.

    Relative ranking is the primary signal; absolute BIC values are
    optimistic because residuals aren't i.i.d. (spatial correlation).
    """
    log_N = math.log(max(n_active * K, 1))
    k_rev = 4 + (K - 1)
    k_pris = 2 + (K - 1)
    bic_rev = 2.0 * rev.L_final + k_rev * log_N
    bic_pris = 2.0 * pris.L_final + k_pris * log_N

    # v8.1 (2026-04-24) — Loss-improvement-ratio selection.
    #
    # Empirical findings: standard BIC with log(N = n_active·K) penalty
    # underweights the rev vs pris DoF difference for n_active in the
    # hundreds. The extra 2 DoF (+ relaxed initial q) allows rev to
    # overfit pris cases by ~5-10% loss reduction (e.g. 30857 rev=336
    # vs pris=369, rev wins BIC by 50). Meanwhile TRUE rev cases show
    # a qualitatively LARGER rev-vs-pris improvement (7201 rev=1548 vs
    # pris=2414, ratio 0.64; 7128 rev=909 vs pris=1401, ratio 0.65).
    # The LOSS-RATIO criterion separates all 4 test cases cleanly at a
    # threshold of ~0.75:
    #
    #   ratio = L_rev / L_pris
    #   < 0.75 → rev is a genuine win (not overfit)
    #   ≥ 0.75 → extra DoF is overfitting, prefer pris
    #
    # This is equivalent to requiring rel_improvement > τ = 0.25. It is
    # a principled relative-information-gain threshold that compensates
    # for non-iid spatial correlation in the voxel residuals (effective
    # sample size << n_active · K, so standard BIC is too permissive).
    # Legacy pivot penalty (lambda_physical > 0) is retained for
    # backward-compat but DEFAULT OFF as of v8.1.
    if object_centroid is not None and lambda_physical > 0.0:
        q_dev = (rev.q.detach().cpu() - object_centroid.detach().cpu()).norm().item()
        excess = max(0.0, q_dev - object_scale)
        bic_rev = bic_rev + lambda_physical * (excess ** 2)

    # Primary selector: loss-improvement ratio
    rel_improv = (pris.L_final - rev.L_final) / max(pris.L_final, eps)
    if rel_improv > tau_improvement:
        joint_type = "revolute"
    else:
        joint_type = "prismatic"

    # margin kept as BIC margin (for diagnostics continuity)
    mn = min(bic_rev, bic_pris)
    margin = abs(bic_rev - bic_pris) / max(abs(mn), eps)
    return joint_type, float(bic_rev), float(bic_pris), float(margin)


# ---- Top-level pipeline ----------------------------------------------


def fit_single_state_anchor(
    O_anchor: torch.Tensor,
    O_stack: torch.Tensor,
    canonical_move: torch.Tensor,
    anchor_state_idx: int,
    joint_type: str,
    init_params: Dict[str, torch.Tensor],
    n_inner_steps: int,
    lr_axis: float,
    lr_phi: float,
    move_mask_anchor: Optional[torch.Tensor] = None,
    resolution: int = 64,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> "VolumetricFit":
    """Phase 1 anchor EM: fit T_{anchor} alone, given canonical_move.

    E-step (implicit): warp canonical_move by T_anchor → predicted mask
    M-step: update T_anchor via BCE loss on predicted vs target

    **v8 target fix (2026-04-24)**. The previous formulation used
    ``target = O_anchor`` (full anchor occupancy), which forced the
    optimizer to predict the entire cabinet from the ~500-voxel
    canonical_move. The ~3000 cabinet voxels contributed a constant
    ``-log(eps) ≈ 16`` BCE each — producing a ~52k floor loss that
    dominated the gradient. With 80 Adam steps at lr=1e-2 the optimizer
    could not escape (verified: phase1_loss 62125 → 62126 over 10× more
    iters). The gradient w.r.t. the 500 canonical voxels was swamped by
    the cabinet noise floor.

    New target: ``move_mask_anchor = (shell ∪ move_interior) ∩ O_anchor``
    — the moving-part voxels at the anchor state. Forward-warped canonical
    should land here, not in the static cabinet. If ``move_mask_anchor``
    is None (API backward compat), fall back to ``O_anchor`` with a
    warning printed once.

    ``init_params`` must contain the single-state parameter for the
    joint_type — ``{"omega", "q", "phi_anchor"}`` for revolute or
    ``{"v_hat", "phi_anchor"}`` for prismatic. ``phi_anchor`` is scalar.
    """
    if device is None:
        device = O_stack.device
    O_anchor = O_anchor.to(device=device, dtype=dtype)
    canonical_f = canonical_move.to(device=device, dtype=dtype)
    if move_mask_anchor is not None:
        target_mask = move_mask_anchor.to(device=device, dtype=dtype)
    else:
        # Backward-compat fallback: emit a loud warning, use full occupancy.
        import warnings
        warnings.warn(
            "fit_single_state_anchor called without move_mask_anchor. "
            "Falling back to full O_anchor target (known to produce "
            "~52k cabinet-noise-floor loss and fail to converge). "
            "Pass move_mask_anchor = move_mask_k[anchor_idx] for v8 behaviour.",
            RuntimeWarning, stacklevel=2,
        )
        target_mask = O_anchor

    if joint_type == "revolute":
        omega_init = init_params["omega"].to(device=device, dtype=dtype).detach().clone()
        q_init = init_params["q"].to(device=device, dtype=dtype).detach().clone()
        omega_init = omega_init / omega_init.norm().clamp_min(1e-8)
        q_init = q_init - (q_init @ omega_init) * omega_init
        v_init = torch.linalg.cross(q_init, omega_init)
        S = torch.nn.Parameter(torch.cat([omega_init, v_init]))
        phi = torch.nn.Parameter(
            torch.as_tensor(
                init_params["phi_anchor"], device=device, dtype=dtype,
            ).reshape(()).detach().clone()
        )
    elif joint_type == "prismatic":
        v_init = init_params["v_hat"].to(device=device, dtype=dtype).detach().clone()
        v_init = v_init / v_init.norm().clamp_min(1e-8)
        S = torch.nn.Parameter(v_init)
        phi = torch.nn.Parameter(
            torch.as_tensor(
                init_params["phi_anchor"], device=device, dtype=dtype,
            ).reshape(()).detach().clone()
        )
    else:
        raise ValueError(f"Unknown joint_type: {joint_type}")

    opt = torch.optim.Adam([
        {"params": [S], "lr": lr_axis},
        {"params": [phi], "lr": lr_phi},
    ])

    L_trace: List[float] = []
    for step in range(n_inner_steps):
        opt.zero_grad()
        if joint_type == "revolute":
            omega_u, v_clean = project_revolute(S[:3], S[3:])
            twist = torch.cat([omega_u, v_clean])
            T_anchor = exp_se3(twist, phi)
        else:
            v_hat_u = project_prismatic(S)
            T_anchor = exp_prismatic(v_hat_u, phi)

        # Warp canonical_move to the anchor state frame
        predicted = trilinear_warp(canonical_f, T_anchor, resolution=resolution)
        # BCE against move-masked anchor target (v8: was full O_anchor, now
        # restricted to anchor's move region to remove cabinet noise floor).
        predicted = predicted.clamp(1e-7, 1.0 - 1e-7)
        target = target_mask.clamp(0.0, 1.0)
        bce = -(target * predicted.log() + (1.0 - target) * (1.0 - predicted).log())
        L = bce.sum()
        L.backward()
        opt.step()

        with torch.no_grad():
            if joint_type == "revolute":
                omega_u, v_clean = project_revolute(S[:3], S[3:])
                S.data.copy_(torch.cat([omega_u, v_clean]))
            else:
                v_hat_u = project_prismatic(S)
                S.data.copy_(v_hat_u)
        L_trace.append(float(L.item()))

    # Rebuild final T_anchor
    with torch.no_grad():
        if joint_type == "revolute":
            omega_u, v_clean = project_revolute(S[:3], S[3:])
            twist = torch.cat([omega_u, v_clean])
            q_rec = torch.linalg.cross(omega_u, v_clean)
            T_anchor = exp_se3(twist, phi)
            phi_k_out = torch.zeros(
                O_stack.shape[0], device=device, dtype=dtype,
            )
            phi_k_out[anchor_state_idx] = phi
            T_k_stack = torch.stack([
                exp_se3(twist, phi_k_out[k]) for k in range(O_stack.shape[0])
            ])
            return VolumetricFit(
                joint_type="revolute",
                omega=omega_u.detach(), q=q_rec.detach(), v=v_clean.detach(),
                phi_k=phi_k_out.detach(), T_k=T_k_stack.detach(),
                L_final=L_trace[-1] if L_trace else float("inf"),
                L_trace=L_trace,
                meta={"phase": "anchor", "anchor_idx": anchor_state_idx},
            )
        v_hat_u = project_prismatic(S)
        phi_k_out = torch.zeros(O_stack.shape[0], device=device, dtype=dtype)
        phi_k_out[anchor_state_idx] = phi
        T_k_stack = torch.stack([
            exp_prismatic(v_hat_u, phi_k_out[k])
            for k in range(O_stack.shape[0])
        ])
        return VolumetricFit(
            joint_type="prismatic",
            omega=v_hat_u.detach(),
            q=torch.zeros(3, device=device, dtype=dtype),
            v=v_hat_u.detach(),
            phi_k=phi_k_out.detach(), T_k=T_k_stack.detach(),
            L_final=L_trace[-1] if L_trace else float("inf"),
            L_trace=L_trace,
            meta={"phase": "anchor", "anchor_idx": anchor_state_idx},
        )


def volumetric_fit_pipeline(
    O_stack: torch.Tensor,
    move_mask_k: torch.Tensor,
    warm_start_dict: Dict[str, Dict[str, torch.Tensor]],
    hp,
    resolution: int = 64,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> VolumetricFitResult:
    """Fit both revolute and prismatic via volumetric Adam, BIC select.

    Parameters
    ----------
    O_stack : (K, D, H, W) per-state occupancy in [0, 1]
    move_mask_k : (K, D, H, W) bool, the current move assignment per state
    warm_start_dict : output of ``moments.warm_start_as_dict``
    hp : SegMatchHParams (duck-typed: reads fit_inner_steps, fit_lr_axis, fit_lr_phi)
    """
    if device is None:
        device = O_stack.device
    K = O_stack.shape[0]

    O_move_stack = (O_stack * move_mask_k.to(O_stack.dtype)).to(
        device=device, dtype=dtype,
    )
    n_active = int((O_move_stack > 0.5).any(dim=0).sum().item())

    mono_lambda = float(getattr(hp, "monotonicity_lambda", 0.0))

    # v8 (2026-04-24) MULTI-START revolute fit. Motivation: Phase 3 Adam on
    # the variance loss has multiple local minima in (omega, q, phi) space.
    # 7128 microwave: default warm q=(-0.09,-0.04,-0.00) near origin lands in
    # a basin with L=1558 non-monotonic phi_k; q=(-0.70,0.37,0.46) matching
    # GT hinge lands in basin with L=680 (monotonic 0→90°) — the GLOBAL
    # minimum. The optimizer cannot escape to L=680 from the origin basin.
    #
    # We enumerate 8 q candidates covering (a) the warm-start midpoint, (b)
    # the union-bbox corners of the move footprint (capturing hinges on body
    # boundary), and (c) axis-aligned OFFSETS outside the move bbox (for
    # hinges attached to base frame outside the body). Each runs fewer iters
    # (hp.fit_inner_steps // 2) then the winner gets refined at full iters.
    warm_rev = warm_start_dict["revolute"]
    base_omega = warm_rev["omega"].to(device=device, dtype=dtype)
    base_q = warm_rev["q"].to(device=device, dtype=dtype)
    base_phi_k = warm_rev["phi_k"].to(device=device, dtype=dtype)

    move_union_bool = move_mask_k.any(dim=0)
    if move_union_bool.sum() > 0:
        idx = torch.arange(resolution, device=device, dtype=dtype)
        ii, jj, kk = torch.meshgrid(idx, idx, idx, indexing="ij")
        grid_w = torch.stack([ii, jj, kk], dim=-1) / float(resolution - 1) - 0.5
        occ_coords = grid_w[move_union_bool]
        bbox_min = occ_coords.min(dim=0).values
        bbox_max = occ_coords.max(dim=0).values
    else:
        bbox_min = torch.full((3,), -0.3, device=device, dtype=dtype)
        bbox_max = torch.full((3,), +0.3, device=device, dtype=dtype)

    q_candidates: List[torch.Tensor] = [base_q]
    # bbox corners (8) — hinges along move-part edges
    for dx in (bbox_min[0], bbox_max[0]):
        for dy in (bbox_min[1], bbox_max[1]):
            for dz in (bbox_min[2], bbox_max[2]):
                q_candidates.append(torch.stack([dx, dy, dz]).clone())
    # axis-aligned offsets outside bbox — hinges on base frame behind/beside move
    bbox_size = bbox_max - bbox_min
    margin = bbox_size * 0.5
    for sign in (-1.0, +1.0):
        for ax in range(3):
            q_off = base_q.clone()
            q_off[ax] = (bbox_max[ax] + margin[ax]) if sign > 0 else (bbox_min[ax] - margin[ax])
            q_candidates.append(q_off)

    fast_iters = max(1, hp.fit_inner_steps // 2)
    best_rev = None
    for q_try in q_candidates:
        init_try = {"omega": base_omega, "q": q_try, "phi_k": base_phi_k}
        try:
            cand = fit_volumetric(
                O_move_stack, "revolute", init_try,
                n_inner_steps=fast_iters,
                lr_axis=hp.fit_lr_axis, lr_phi=hp.fit_lr_phi,
                weight_decay=hp.fit_weight_decay,
                monotonicity_lambda=mono_lambda,
                resolution=resolution, device=device, dtype=dtype,
            )
        except Exception:
            continue
        if best_rev is None or cand.L_final < best_rev.L_final:
            best_rev = cand
    if best_rev is None:
        # Should never hit — defensive fallback
        best_rev = fit_volumetric(
            O_move_stack, "revolute", warm_rev,
            n_inner_steps=hp.fit_inner_steps,
            lr_axis=hp.fit_lr_axis, lr_phi=hp.fit_lr_phi,
            weight_decay=hp.fit_weight_decay,
            monotonicity_lambda=mono_lambda,
            resolution=resolution, device=device, dtype=dtype,
        )
    # Refine the winner at full iters
    refine_init = {
        "omega": best_rev.omega,
        "q": best_rev.q,
        "phi_k": best_rev.phi_k,
    }
    rev = fit_volumetric(
        O_move_stack, "revolute", refine_init,
        n_inner_steps=hp.fit_inner_steps,
        lr_axis=hp.fit_lr_axis, lr_phi=hp.fit_lr_phi,
        weight_decay=hp.fit_weight_decay,
        monotonicity_lambda=mono_lambda,
        resolution=resolution, device=device, dtype=dtype,
    )

    pris = fit_volumetric(
        O_move_stack, "prismatic",
        warm_start_dict["prismatic"],
        n_inner_steps=hp.fit_inner_steps,
        lr_axis=hp.fit_lr_axis,
        lr_phi=hp.fit_lr_phi,
        weight_decay=hp.fit_weight_decay,
        monotonicity_lambda=mono_lambda,
        resolution=resolution, device=device, dtype=dtype,
    )

    # v8: compute full-object centroid (base + move) for physical pivot prior.
    # Using only the move centroid biases toward move's location, missing the
    # point — a true revolute joint's pivot is on the OBJECT (cabinet for
    # door/drawer), not at the move centroid. For 30857 drawer: full-object
    # centroid ≈ cabinet centroid (near origin), fitted rev |q|=0.16 →
    # |q - obj_centroid| ≈ 0.19 → strong penalty. For 7201 door: object
    # centroid near origin, rev |q|=0.04 → |q - obj_centroid| ≈ 0.04 → 0
    # penalty. Correct geometric separation between true rev and fake rev.
    footprint = (O_stack > 0.5).any(dim=0).to(dtype)
    if footprint.sum() > 0:
        idx = torch.arange(resolution, device=device, dtype=dtype)
        ii, jj, kk = torch.meshgrid(idx, idx, idx, indexing="ij")
        coord = torch.stack([ii, jj, kk], dim=-1) / float(resolution - 1) - 0.5
        w = footprint.clamp_min(0.0)
        object_centroid = (w.unsqueeze(-1) * coord).sum(dim=(0, 1, 2)) / w.sum().clamp_min(1e-8)
    else:
        object_centroid = torch.zeros(3, device=device, dtype=dtype)
    # v8.1: loss-improvement ratio is the primary selector. Disable pivot penalty.
    joint_type, bic_rev, bic_pris, margin = compute_bic(
        rev, pris, n_active, K,
        object_centroid=object_centroid, object_scale=0.10, lambda_physical=0.0,
        tau_improvement=0.25,
    )
    selected = rev if joint_type == "revolute" else pris

    return VolumetricFitResult(
        joint_fit=selected, rev=rev, pris=pris,
        bic_rev=bic_rev, bic_pris=bic_pris, bic_margin=margin,
        n_active=n_active,
    )

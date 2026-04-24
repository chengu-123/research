"""EM optimization for SAJO: alternates between soft ``M_move``
re-estimation (E-step) and Adam gradient updates on the screw
parameters ``(S, phi_k)`` (M-step), under a differentiable energy
``L_reg = L_data + alpha * L_anchor + beta * ||phi_k||^2``.

Two distinct hypotheses are supported:

* ``em_revolute`` parameterizes ``S = (omega, v) in R^6`` with constraints
  ``||omega|| = 1`` and ``v . omega = 0``, enforced by manifold
  retraction after every Adam step. The revolute transform is built via
  :func:`pipelines.sajo.screw.exp_se3` on the 4x4 twist matrix.
* ``em_prismatic`` parameterizes ``v_hat in S^2``; the transform is the
  closed-form ``T(phi) = [[I, phi * v_hat], [0, 1]]``.

Energy descent is per the standard EM proof (E-step is a soft
assignment that minimizes a regularized energy with an implicit
entropy term on ``M_move``; M-step is gradient descent on a bounded
energy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .anchors import anchor_coords_to_world
from .screw import (
    exp_prismatic,
    exp_se3,
    project_prismatic,
    project_revolute,
)
from .warp import batch_trilinear_warp


# ---- Hyperparameters and result containers ---------------------------


@dataclass
class EMHParams:
    n_outer: int = 20
    n_inner: int = 10
    lr_S: float = 1e-2
    lr_phi: float = 5e-3
    alpha: float = 0.1
    beta: float = 1e-4
    eta_r: float = 0.05
    tol: float = 1e-4
    active_voxel_thresh: float = 0.1
    anchor_pca_eps_ratio: float = 0.2  # (lambda2/lambda1) threshold for plane vs rail


@dataclass
class EMResult:
    joint_type: str
    omega: torch.Tensor           # (3,) unit
    q: torch.Tensor               # (3,) — point on axis (revolute); prismatic ignored
    v: torch.Tensor               # (3,) — Plücker moment (revolute) or v_hat (prismatic)
    phi_k: torch.Tensor           # (K,)
    M_move: torch.Tensor          # (D,H,W)
    T_k_list: torch.Tensor        # (K, 4, 4)
    L_data_final: float
    L_reg_trace: List[float] = field(default_factory=list)
    converged: bool = False
    n_iters: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


# ---- Energy components -----------------------------------------------


def _compute_T_k(
    joint_type: str,
    S_params: torch.Tensor,
    phi_k: torch.Tensor,
) -> torch.Tensor:
    """Build the full ``(K, 4, 4)`` stack of per-state transforms."""
    K = phi_k.shape[0]
    if joint_type == "revolute":
        # S_params: (6,) raw (omega, v)
        omega_u, v_clean = project_revolute(S_params[:3], S_params[3:])
        twist = torch.cat([omega_u, v_clean], dim=-1)  # autograd-safe
        T_list = []
        for k in range(K):
            T_list.append(exp_se3(twist, phi_k[k]))
        return torch.stack(T_list, dim=0)              # (K, 4, 4)
    elif joint_type == "prismatic":
        v_hat = project_prismatic(S_params[:3])         # (3,)
        T_list = []
        for k in range(K):
            T_list.append(exp_prismatic(v_hat, phi_k[k]))
        return torch.stack(T_list, dim=0)               # (K, 4, 4)
    else:
        raise ValueError(f"Unknown joint_type: {joint_type}")


def _L_data(
    T_k: torch.Tensor,
    O_stack: torch.Tensor,
    M_move: torch.Tensor,
) -> torch.Tensor:
    """Weighted warp consistency loss, summed across K states.

    ``L_data = sum_k sum_x M_move(x) * (warp(O_0, T_k)(x) - O_k(x))^2``

    Notes
    -----
    - ``warp(O_0, T_k)`` evaluates ``O_0`` at ``T_k^{-1}(x)``, which is
      the canonical-frame value of the move part after the inverse warp.
    - The data term uses the squared L2, following spec §2.5.
    """
    K = O_stack.shape[0]
    O_0 = O_stack[0]
    # Expand state 0 to batch dim for batched warp: we need T_0, ..., T_{K-1}.
    O_batch = O_0.unsqueeze(0).expand(K, -1, -1, -1).contiguous()
    warped = batch_trilinear_warp(O_batch, T_k)                  # (K,D,H,W)
    diff_sq = (warped - O_stack) ** 2                            # (K,D,H,W)
    loss = (M_move[None] * diff_sq).sum()
    return loss


def _L_data_unweighted_for_bic(
    T_k: torch.Tensor,
    O_stack: torch.Tensor,
    M_move: torch.Tensor,
) -> torch.Tensor:
    """Same as ``_L_data`` but returned as a detached float for BIC.
    We still weight by ``M_move`` since that's the proposer sample size."""
    with torch.no_grad():
        return _L_data(T_k, O_stack, M_move)


def _L_anchor_revolute(
    S_params: torch.Tensor,
    anchor_world: torch.Tensor,
    anchor_weights: torch.Tensor,
) -> torch.Tensor:
    """Squared perpendicular distance from anchor points to the rotation
    axis line (ω through q). Gauge-invariant under translating q along ω."""
    if anchor_world.shape[0] == 0:
        return torch.zeros((), device=S_params.device, dtype=S_params.dtype)
    omega_u, v_clean = project_revolute(S_params[:3], S_params[3:])
    # Point on axis closest to the origin: q = omega x v (for ||omega||=1, v.omega=0).
    q = torch.linalg.cross(omega_u, v_clean)
    diff = anchor_world - q                                      # (N,3)
    proj = (diff * omega_u).sum(dim=-1, keepdim=True) * omega_u  # (N,3)
    perp = diff - proj                                           # (N,3)
    d2 = (perp * perp).sum(dim=-1)                               # (N,)
    return (anchor_weights * d2).sum()


def _L_anchor_prismatic(
    S_params: torch.Tensor,
    n_hat: Optional[torch.Tensor],
    e_A: Optional[torch.Tensor],
    mode: str,
) -> torch.Tensor:
    """Prismatic anchor term. ``mode`` is either 'plane' (penalize out-of-plane
    direction; use ``n_hat``) or 'rail' (penalize deviation from rail; use
    ``e_A``)."""
    v_hat = project_prismatic(S_params[:3])
    if mode == "plane" and n_hat is not None:
        dot = (v_hat * n_hat).sum()
        return 1.0 - dot * dot
    elif mode == "rail" and e_A is not None:
        dot = (v_hat * e_A).sum()
        return 1.0 - dot * dot
    else:
        return torch.zeros((), device=v_hat.device, dtype=v_hat.dtype)


# ---- Shared EM body ---------------------------------------------------


def _e_step(
    O_stack: torch.Tensor,
    T_k: torch.Tensor,
    p_move: torch.Tensor,
    hp: EMHParams,
) -> torch.Tensor:
    """Soft re-estimation of ``M_move`` from the current warp residual.

    ``r(x) = sum_k |warp(O_0, T_k)(x) - O_k(x)|``
    ``M_move(x) = sigmoid((tau_r - r(x)) / eta_r) * p_move(x)``

    ``tau_r`` is the median of ``r`` over voxels with
    ``p_move > active_voxel_thresh`` (spec §2.6).
    """
    with torch.no_grad():
        K = O_stack.shape[0]
        O_0 = O_stack[0]
        O_batch = O_0.unsqueeze(0).expand(K, -1, -1, -1).contiguous()
        warped = batch_trilinear_warp(O_batch, T_k)
        residual = (warped - O_stack).abs().sum(dim=0)            # (D,H,W)

        active = p_move > hp.active_voxel_thresh
        if active.any():
            tau_r = torch.median(residual[active])
        else:
            tau_r = torch.median(residual)

        new_M = torch.sigmoid((tau_r - residual) / hp.eta_r) * p_move
    return new_M


def _anchor_pca(
    anchor_world: torch.Tensor,
    anchor_weights: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
           Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Cached PCA for the prismatic anchor term.

    Returns ``(eigenvalues, eigenvectors, n_hat, e_A)`` where ``n_hat`` is
    the smallest-eigenvalue eigenvector (plane normal), and ``e_A`` is the
    largest-eigenvalue eigenvector (rail direction).
    """
    if anchor_world.shape[0] < 3:
        return None, None, None, None
    w = anchor_weights.clamp_min(0.0)
    w_sum = w.sum().clamp_min(eps)
    mu = (w[:, None] * anchor_world).sum(dim=0) / w_sum
    centered = anchor_world - mu
    cov = (w[:, None] * centered).T @ centered / w_sum
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = torch.linalg.eigh(cov)                     # ascending
    n_hat = eigvecs[:, 0]                                          # smallest
    e_A = eigvecs[:, -1]                                           # largest
    return eigvals, eigvecs, n_hat, e_A


# ---- Revolute EM ------------------------------------------------------


def em_revolute(
    O_stack: torch.Tensor,
    p_base: torch.Tensor,
    p_move: torch.Tensor,
    anchor_coords: torch.Tensor,
    anchor_weights: torch.Tensor,
    init_params: Dict[str, torch.Tensor],
    hp: EMHParams,
    resolution: int = 64,
) -> EMResult:
    device = O_stack.device
    dtype = O_stack.dtype
    K = O_stack.shape[0]

    anchor_world = anchor_coords_to_world(anchor_coords, resolution).to(
        device=device, dtype=dtype
    ) if anchor_coords.numel() > 0 else torch.zeros((0, 3), device=device, dtype=dtype)
    anchor_w = anchor_weights.to(device=device, dtype=dtype) if anchor_weights.numel() > 0 else torch.zeros((0,), device=device, dtype=dtype)

    # Initialize optimizable parameters.
    omega_init = init_params["omega"].to(device=device, dtype=dtype).detach().clone()
    v_init = init_params.get("v")
    if v_init is None:
        q_init = init_params.get("q", torch.zeros(3, device=device, dtype=dtype))
        v_init = torch.linalg.cross(q_init.to(device=device, dtype=dtype), omega_init)
    v_init = v_init.to(device=device, dtype=dtype).detach().clone()

    omega_u, v_clean = project_revolute(omega_init, v_init)
    S = torch.nn.Parameter(torch.cat([omega_u, v_clean]).clone())
    phi_k = torch.nn.Parameter(
        init_params["phi_k"].to(device=device, dtype=dtype).detach().clone()
    )
    optimizer = torch.optim.Adam([
        {"params": [S], "lr": hp.lr_S},
        {"params": [phi_k], "lr": hp.lr_phi},
    ])

    M_move = p_move.detach().clone()

    L_reg_trace: List[float] = []
    converged = False
    prev_L_reg = None

    for outer in range(hp.n_outer):
        # E-step first (plan §2.6): refine M_move using current S, phi_k.
        with torch.no_grad():
            T_k_e = _compute_T_k("revolute", S, phi_k)
        M_move = _e_step(O_stack, T_k_e, p_move, hp)

        # M-step: update (S, phi_k) with M_move fixed.
        for _inner in range(hp.n_inner):
            optimizer.zero_grad()
            T_k = _compute_T_k("revolute", S, phi_k)
            L_data = _L_data(T_k, O_stack, M_move)
            L_anc = _L_anchor_revolute(S, anchor_world, anchor_w)
            L_prior = hp.beta * (phi_k * phi_k).sum()
            L_reg = L_data + hp.alpha * L_anc + L_prior
            L_reg.backward()
            optimizer.step()
            with torch.no_grad():
                omega_u, v_clean = project_revolute(S[:3], S[3:])
                S.data.copy_(torch.cat([omega_u, v_clean]))

        # Evaluate L_reg after M-step for convergence tracking.
        with torch.no_grad():
            T_k = _compute_T_k("revolute", S, phi_k)
            L_data = _L_data(T_k, O_stack, M_move)
            L_anc = _L_anchor_revolute(S, anchor_world, anchor_w)
            L_prior = hp.beta * (phi_k * phi_k).sum()
            L_reg = L_data + hp.alpha * L_anc + L_prior
            L_reg_trace.append(float(L_reg.item()))

        if prev_L_reg is not None:
            denom = abs(prev_L_reg) + 1e-12
            if abs(prev_L_reg - L_reg_trace[-1]) / denom < hp.tol:
                converged = True
                prev_L_reg = L_reg_trace[-1]
                break
        prev_L_reg = L_reg_trace[-1]

    with torch.no_grad():
        omega_u, v_clean = project_revolute(S[:3], S[3:])
        q_rec = torch.linalg.cross(omega_u, v_clean)
        T_k_final = _compute_T_k("revolute", S, phi_k)
        L_data_final = float(_L_data_unweighted_for_bic(T_k_final, O_stack, M_move).item())

    return EMResult(
        joint_type="revolute",
        omega=omega_u.detach(),
        q=q_rec.detach(),
        v=v_clean.detach(),
        phi_k=phi_k.detach(),
        M_move=M_move.detach(),
        T_k_list=T_k_final.detach(),
        L_data_final=L_data_final,
        L_reg_trace=L_reg_trace,
        converged=converged,
        n_iters=len(L_reg_trace),
        meta={"anchor_count": int(anchor_world.shape[0])},
    )


# ---- Prismatic EM -----------------------------------------------------


def em_prismatic(
    O_stack: torch.Tensor,
    p_base: torch.Tensor,
    p_move: torch.Tensor,
    anchor_coords: torch.Tensor,
    anchor_weights: torch.Tensor,
    init_params: Dict[str, torch.Tensor],
    hp: EMHParams,
    resolution: int = 64,
) -> EMResult:
    device = O_stack.device
    dtype = O_stack.dtype
    K = O_stack.shape[0]

    anchor_world = anchor_coords_to_world(anchor_coords, resolution).to(
        device=device, dtype=dtype
    ) if anchor_coords.numel() > 0 else torch.zeros((0, 3), device=device, dtype=dtype)
    anchor_w = anchor_weights.to(device=device, dtype=dtype) if anchor_weights.numel() > 0 else torch.zeros((0,), device=device, dtype=dtype)

    # Cache anchor PCA; pick 'plane' vs 'rail' mode once at init.
    eigvals, _eigvecs, n_hat, e_A = _anchor_pca(anchor_world, anchor_w)
    anchor_mode: str = "none"
    if eigvals is not None and eigvals.numel() >= 3:
        lam1 = float(eigvals[-1].item())
        lam2 = float(eigvals[-2].item())
        if lam1 > 1e-12:
            ratio = lam2 / lam1
            anchor_mode = "plane" if ratio > hp.anchor_pca_eps_ratio else "rail"
        else:
            anchor_mode = "plane"

    v_hat_init = init_params["v_hat"].to(device=device, dtype=dtype).detach().clone()
    v_hat_init = project_prismatic(v_hat_init)
    # Parameter vector of length 3 (v_hat). We still use _compute_T_k with the
    # same `S` first 3 entries to keep a uniform API.
    S = torch.nn.Parameter(v_hat_init.clone())
    phi_k = torch.nn.Parameter(
        init_params["phi_k"].to(device=device, dtype=dtype).detach().clone()
    )
    optimizer = torch.optim.Adam([
        {"params": [S], "lr": hp.lr_S},
        {"params": [phi_k], "lr": hp.lr_phi},
    ])

    M_move = p_move.detach().clone()
    L_reg_trace: List[float] = []
    converged = False
    prev_L_reg = None

    for outer in range(hp.n_outer):
        # E-step first (plan §2.6): refine M_move using current S, phi_k.
        with torch.no_grad():
            T_k_e = _compute_T_k("prismatic", S, phi_k)
        M_move = _e_step(O_stack, T_k_e, p_move, hp)

        # M-step: update (S, phi_k) with M_move fixed.
        for _inner in range(hp.n_inner):
            optimizer.zero_grad()
            T_k = _compute_T_k("prismatic", S, phi_k)
            L_data = _L_data(T_k, O_stack, M_move)
            L_anc = _L_anchor_prismatic(S, n_hat, e_A, anchor_mode)
            L_prior = hp.beta * (phi_k * phi_k).sum()
            L_reg = L_data + hp.alpha * L_anc + L_prior
            L_reg.backward()
            optimizer.step()
            with torch.no_grad():
                v_clean = project_prismatic(S[:3])
                S.data[:3].copy_(v_clean)

        # Evaluate L_reg after M-step for convergence tracking.
        with torch.no_grad():
            T_k = _compute_T_k("prismatic", S, phi_k)
            L_data = _L_data(T_k, O_stack, M_move)
            L_anc = _L_anchor_prismatic(S, n_hat, e_A, anchor_mode)
            L_prior = hp.beta * (phi_k * phi_k).sum()
            L_reg = L_data + hp.alpha * L_anc + L_prior
            L_reg_trace.append(float(L_reg.item()))

        if prev_L_reg is not None:
            denom = abs(prev_L_reg) + 1e-12
            if abs(prev_L_reg - L_reg_trace[-1]) / denom < hp.tol:
                converged = True
                prev_L_reg = L_reg_trace[-1]
                break
        prev_L_reg = L_reg_trace[-1]

    with torch.no_grad():
        v_hat_final = project_prismatic(S[:3])
        T_k_final = _compute_T_k("prismatic", S, phi_k)
        L_data_final = float(_L_data_unweighted_for_bic(T_k_final, O_stack, M_move).item())

    # Revolute-style omega/q reporting: for prismatic we set omega = v_hat
    # (direction) and q = origin (unused by URDF for prismatic, but we emit
    # the base centroid for consistency at the stage_f level).
    return EMResult(
        joint_type="prismatic",
        omega=v_hat_final.detach(),
        q=torch.zeros(3, device=device, dtype=dtype),
        v=v_hat_final.detach(),
        phi_k=phi_k.detach(),
        M_move=M_move.detach(),
        T_k_list=T_k_final.detach(),
        L_data_final=L_data_final,
        L_reg_trace=L_reg_trace,
        converged=converged,
        n_iters=len(L_reg_trace),
        meta={
            "anchor_count": int(anchor_world.shape[0]),
            "anchor_mode": anchor_mode,
        },
    )


# ---- Dispatch --------------------------------------------------------


def em_optimize(
    joint_type: str,
    O_stack: torch.Tensor,
    p_base: torch.Tensor,
    p_move: torch.Tensor,
    anchor_coords: torch.Tensor,
    anchor_weights: torch.Tensor,
    init_params: Dict[str, torch.Tensor],
    hp: EMHParams,
    resolution: int = 64,
) -> EMResult:
    """Run EM for a single joint hypothesis. Dispatches to
    :func:`em_revolute` or :func:`em_prismatic`."""
    if joint_type == "revolute":
        return em_revolute(
            O_stack, p_base, p_move,
            anchor_coords, anchor_weights,
            init_params, hp, resolution,
        )
    elif joint_type == "prismatic":
        return em_prismatic(
            O_stack, p_base, p_move,
            anchor_coords, anchor_weights,
            init_params, hp, resolution,
        )
    else:
        raise ValueError(f"Unknown joint_type: {joint_type}")

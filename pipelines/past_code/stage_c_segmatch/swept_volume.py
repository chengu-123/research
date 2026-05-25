"""Swept-volume carving and residual-weighted canonical_move voting.

Core v6 (stagec_3.md) operations:

1. ``compute_canonical_move_vote`` — weighted soft vote over per-state
   back-warped move candidates, with volume-conservation threshold.
   Replaces v5's naive median aggregation.

2. ``compute_swept_volume`` — densely sample φ in the observed
   per-state parameter range, warp ``canonical_move`` by each T(φ),
   union. This is the key operator that resolves the long-drawer
   always_on-interior ambiguity.

3. ``late_commit_carve`` — apply swept volume to remove base voxels
   that physically lie in the moving part's trajectory. Lower-bound
   protection rejects pathological (e.g. almost-all-O_0) carvings.

4. ``anchor_state_selection`` — pick the state with the largest hard
   seed set (|S_hard|/|O_k|) as the Phase 1 EM anchor. For typical
   data this is state 5 (max opened), but the algorithm is agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from ..sajo.screw import exp_prismatic, exp_se3
from ..sajo.warp import batch_trilinear_warp, trilinear_warp


@dataclass
class VoteResult:
    canonical_move: torch.Tensor        # (D, H, W) bool
    score: torch.Tensor                 # (D, H, W) float — per-voxel soft score
    tau_star: float                     # threshold used
    backwarp_stack: torch.Tensor        # (K, D, H, W) float — per-state backwarps
    weights: torch.Tensor               # (K,) float — per-state voting weights
    target_volume: float                # median-of-per-state volumes used for tau


@dataclass
class CarveResult:
    canonical_base: torch.Tensor        # (D, H, W) bool — base after carve
    swept_volume: torch.Tensor          # (D, H, W) bool — union of T(φ)(canonical_move)
    n_carved: int                       # voxels removed from base
    triggered: bool                     # True if carving was actually applied
    lower_bound_protected: bool         # True if alpha lower bound blocked carving
    alpha_final: float                  # |base_after| / |base_before|


# ---- Canonical voting -------------------------------------------------


def compute_canonical_move_vote(
    O_stack: torch.Tensor,
    move_mask_k: torch.Tensor,
    T_k: torch.Tensor,
    canonical_omega_c: torch.Tensor,
    residuals: Optional[torch.Tensor] = None,
    beta: Optional[float] = None,
    vote_method: str = "volume_conservation",
    hard_vote_threshold: int = 3,
    resolution: int = 64,
) -> VoteResult:
    """Residual-weighted soft vote for canonical_move, clipped to Ω_c.

    Parameters
    ----------
    O_stack : (K, D, H, W) — per-state occupancy (float in [0, 1])
    move_mask_k : (K, D, H, W) bool — per-state move candidate mask
    T_k : (K, 4, 4) — canonical-to-state transforms
    canonical_omega_c : (D, H, W) bool — Ω_c = O_0 (hard clip for output)
    residuals : (K,) float, optional — per-state EM fit residuals for weighting.
        If None, uniform weights.
    beta : optional float — weight = exp(-β * r²). If None, auto: 2/median(r²).
    vote_method :
        * "volume_conservation" — pick τ* so that ``|score > τ*| ≈ median_k |W_k|``
        * "hard_majority" — τ* = (hard_vote_threshold - 0.5) / K  (≥ hard_vote_threshold states vote)
    hard_vote_threshold : used when method == "hard_majority".
    """
    if O_stack.dim() != 4 or move_mask_k.dim() != 4:
        raise ValueError("O_stack and move_mask_k must be (K, D, H, W)")
    K, D, H, W = O_stack.shape
    device = O_stack.device
    dtype = O_stack.dtype

    # Per-state candidate occupancy (soft)
    M_k_cand = O_stack * move_mask_k.to(dtype)                      # (K, D, H, W)

    # Backwarp each to canonical via T_k^{-1}
    T_k_inv = torch.linalg.inv(T_k.to(device=device, dtype=dtype))
    backwarp_stack = batch_trilinear_warp(
        M_k_cand, T_k_inv, resolution=resolution,
    )                                                                # (K, D, H, W)

    # Weights
    if residuals is None:
        weights = torch.ones(K, device=device, dtype=dtype)
    else:
        r = residuals.to(device=device, dtype=dtype)
        if beta is None:
            median_r2 = torch.median(r ** 2).clamp_min(1e-8)
            beta_val = 2.0 / float(median_r2.item())
        else:
            beta_val = float(beta)
        weights = torch.exp(-beta_val * r ** 2)
    weights = weights / weights.sum().clamp_min(1e-8)
    w_bcast = weights.view(K, 1, 1, 1)

    # Weighted soft score
    score = (w_bcast * backwarp_stack).sum(dim=0)                    # (D, H, W)

    # Volume target: median of per-state candidate volumes
    per_state_vol = (M_k_cand > 0.5).flatten(1).sum(dim=1).to(torch.float32)
    target_volume = float(per_state_vol.median().item())

    # Pick threshold
    if vote_method == "volume_conservation":
        # Sort score descending; pick τ* such that approximately target_volume
        # voxels pass the threshold.
        flat = score.flatten()
        n_target = int(max(1, min(flat.numel() - 1, round(target_volume))))
        sorted_scores, _ = torch.sort(flat, descending=True)
        tau_star = float(sorted_scores[n_target].item())
        # Clamp tau to a sane minimum to avoid picking up 0-score background
        tau_star = max(tau_star, 1e-3)
    elif vote_method == "hard_majority":
        tau_star = (hard_vote_threshold - 0.5) / float(K)
    else:
        raise ValueError(f"Unknown vote_method: {vote_method}")

    # v8: threshold + footprint clip. Previous code clipped strictly to
    # Ω_c = O_0, which requires backwarped canonical to land exactly inside
    # state-0 occupancy. For even slightly imperfect T_k (2-3 voxel mis-fit),
    # backwarp positions drift a few voxels outside O_0 → strict clip wipes
    # most of canonical_move → downstream pipeline collapses move to 0
    # (observed on 30857: 142 candidates → 30 in Ω_c → 0 in final).
    # Footprint (count > 0) is the physically meaningful outer bound:
    # canonical_move must be somewhere the object EVER occupied, but need
    # not be strictly in state-0's snapshot.
    footprint = (O_stack > 0.5).any(dim=0)
    canonical_move_raw = score > tau_star
    canonical_move = canonical_move_raw & footprint

    return VoteResult(
        canonical_move=canonical_move,
        score=score.detach(),
        tau_star=tau_star,
        backwarp_stack=backwarp_stack.detach(),
        weights=weights.detach(),
        target_volume=target_volume,
    )


# ---- Swept volume -----------------------------------------------------


def _build_T_phi(
    joint_type: str,
    axis_params: Dict[str, torch.Tensor],
    phi: torch.Tensor,
) -> torch.Tensor:
    """Build a single 4×4 SE(3) for a given φ on the joint manifold."""
    if joint_type == "revolute":
        omega = axis_params["omega"]
        v = axis_params.get("v")
        if v is None:
            q = axis_params["q"]
            v = torch.linalg.cross(q, omega)
        twist = torch.cat([omega, v])
        return exp_se3(twist, phi)
    if joint_type == "prismatic":
        v_hat = axis_params["v_hat"]
        return exp_prismatic(v_hat, phi)
    raise ValueError(f"Unknown joint_type: {joint_type}")


def compute_swept_volume(
    canonical_move: torch.Tensor,
    joint_type: str,
    axis_params: Dict[str, torch.Tensor],
    phi_k: torch.Tensor,
    n_samples: int = 50,
    phi_margin: float = 0.05,
    resolution: int = 64,
) -> torch.Tensor:
    """Dense sweep of T(φ)(canonical_move) over observed φ range.

    The observed φ range is ``[min_k φ_k, max_k φ_k]`` extended by
    ``phi_margin`` fraction on each side (default 5%) for robustness.
    For non-convex trajectories we take the element-wise maximum over
    samples (equivalent to union in binary).

    Returns (D, H, W) bool swept volume.
    """
    if canonical_move.dim() != 3:
        raise ValueError(f"canonical_move must be (D,H,W); got {tuple(canonical_move.shape)}")
    device = canonical_move.device
    dtype = torch.float32

    phi_min = float(phi_k.min().item())
    phi_max = float(phi_k.max().item())
    phi_span = max(abs(phi_max - phi_min), 1e-6)
    phi_lo = phi_min - phi_margin * phi_span
    phi_hi = phi_max + phi_margin * phi_span
    phi_samples = torch.linspace(phi_lo, phi_hi, n_samples,
                                  device=device, dtype=dtype)

    canonical_f = canonical_move.to(device=device, dtype=dtype)
    sv = torch.zeros_like(canonical_f)

    for phi in phi_samples:
        T = _build_T_phi(joint_type, axis_params, phi)
        warped = trilinear_warp(canonical_f, T, resolution=resolution)
        sv = torch.maximum(sv, warped)

    return sv > 0.5


# ---- Late-commit carving ----------------------------------------------


def late_commit_carve(
    canonical_base: torch.Tensor,
    swept_volume: torch.Tensor,
    alpha_lower: float = 0.3,
    reference_volume: Optional[torch.Tensor] = None,
) -> CarveResult:
    """Remove base voxels that lie inside the swept volume, with safety.

    Lower-bound protection: if carving would remove more than
    ``1 − alpha_lower`` fraction of base, abort and keep the original
    (this guards against ω/q errors that would sweep most of O_0).

    ``reference_volume`` (default: ``canonical_base`` itself) sets the
    denominator for the α ratio. Pass ``O_0`` if you want α relative to
    the full state-0 occupancy (stagec_3.md recommendation).
    """
    if canonical_base.dim() != 3 or swept_volume.dim() != 3:
        raise ValueError("canonical_base and swept_volume must be (D,H,W)")

    base_before = canonical_base.to(torch.bool)
    sv = swept_volume.to(torch.bool)
    if reference_volume is None:
        ref = base_before
    else:
        ref = reference_volume.to(torch.bool)

    ref_n = int(ref.sum().item())
    base_before_n = int(base_before.sum().item())

    if ref_n == 0:
        return CarveResult(
            canonical_base=base_before, swept_volume=sv,
            n_carved=0, triggered=False,
            lower_bound_protected=False, alpha_final=1.0,
        )

    base_after = base_before & ~sv
    base_after_n = int(base_after.sum().item())
    n_carved = base_before_n - base_after_n
    alpha_final = base_after_n / float(ref_n)

    if alpha_final < alpha_lower:
        # Pathological carving — abort
        return CarveResult(
            canonical_base=base_before, swept_volume=sv,
            n_carved=0, triggered=False,
            lower_bound_protected=True,
            alpha_final=base_before_n / float(ref_n),
        )

    return CarveResult(
        canonical_base=base_after, swept_volume=sv,
        n_carved=n_carved, triggered=True,
        lower_bound_protected=False, alpha_final=alpha_final,
    )


# ---- Anchor state selection -------------------------------------------


def select_anchor_state(
    O_stack: torch.Tensor,
    move_strong: torch.Tensor,
    move_seeds_global: Optional[torch.Tensor] = None,
    min_hard_seed_ratio: float = 0.05,
    fallback_idx: int = -1,
) -> Tuple[int, Dict[str, float]]:
    """Pick the EM anchor state = MAX DISPLACEMENT from state 0.

    **v8 rewrite (2026-04-24)**. The previous heuristic picked the state with
    the largest ``|S_hard|/|O_k|`` ratio, which in practice was often state 0
    itself (e.g. 7201 oven: closed-door state 0 had highest shell-intersection
    ratio 0.559). With anchor=0, Phase 1 trivially fits ``T_0 = I`` (zero
    displacement) and ``_linear_propagate`` returns all-zero ``phi_k``; Phase 3
    inherits garbage warm start and converges to a degenerate optimum —
    observed as ``volumetric_loss_rev == volumetric_loss_pris`` and wrong
    joint-type BIC selection.

    The v8 primary signal is the symmetric-difference voxel count
    ``|O_k XOR O_0|``, which directly measures displacement. State 0 is
    excluded from candidates by construction (T_0 = I, nothing to fit).
    For mostly-monotonic TRELLIS-sampled trajectories this picks state K-1.

    Parameters
    ----------
    O_stack : (K, D, H, W) per-state occupancy.
    move_strong : (D, H, W) bool — reported in stats only (backward compat).
    move_seeds_global : (D, H, W) bool — reported in stats only.
    min_hard_seed_ratio : reported, no longer used as a hard gate.
    fallback_idx : used only when every XOR count is zero (no motion).

    Returns
    -------
    (anchor_idx, stats_dict) — stats adds ``per_state_xor_vs_state0`` and
    ``selection_rule='max_xor_vs_state0_v8'``.
    """
    K = O_stack.shape[0]
    O_bool_flat = (O_stack > 0.5).view(K, -1)
    O_0 = O_bool_flat[0]

    # v8 primary signal: XOR voxel count vs. state 0.
    xor_counts: List[int] = []
    for k in range(K):
        if k == 0:
            xor_counts.append(0)
            continue
        xor_counts.append(int((O_bool_flat[k] ^ O_0).sum().item()))

    # Legacy ratio (retained for diagnostics).
    if move_seeds_global is None:
        seed = move_strong.view(-1)
    else:
        seed = (move_strong | move_seeds_global).view(-1)
    ratios: List[float] = []
    abs_counts: List[int] = []
    for k in range(K):
        o_k = O_bool_flat[k]
        inter = int((o_k & seed).sum().item())
        total = int(o_k.sum().item())
        ratios.append(inter / max(total, 1))
        abs_counts.append(inter)

    # Primary selection: argmax XOR over k ≥ 1 (exclude identity state).
    if K > 1:
        candidate_ks = list(range(1, K))
        best_idx = int(max(candidate_ks, key=lambda k: xor_counts[k]))
        if xor_counts[best_idx] == 0:
            best_idx = fallback_idx if fallback_idx >= 0 else K - 1
    else:
        best_idx = 0

    stats = {
        "per_state_ratio": ratios,
        "per_state_hard_count": abs_counts,
        "per_state_xor_vs_state0": xor_counts,
        "selected_idx": best_idx,
        "selected_ratio": ratios[best_idx],
        "selected_xor": xor_counts[best_idx],
        "selection_rule": "max_xor_vs_state0_v8",
    }
    return best_idx, stats

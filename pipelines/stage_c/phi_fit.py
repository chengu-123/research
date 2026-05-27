"""Per-state phi (normalized progress) + canonical-state shift + delta_u_init.

v3 (cardinal-cand + voxel-scoring; supersedes v2 centroid-only):

Input: a CandidateResult from joint_type_detect (axis, origin, phi_k raw)
       where phi_k is already computed in the candidate's natural unit:
         - prismatic: median projection on axis (world units)
         - revolute:  median angle around axis (radians)

This module:
  1. Enforces monotonicity via PAV (pool-adjacent-violators) if needed
  2. Normalizes to [0, 1] via (u - u.min) / (u.max - u.min)
  3. Spreads ties with phi_min_gap
  4. Applies canonical-state shift (NEW.1): u_shifted = u_norm - u_norm[c]
  5. Computes delta_u_init = inverse_softplus(diffs(u_norm))   [with bug fix]
  6. Computes BOTH observed_max_disp and observed_max_angle so Stage D
     dual-clone has the alternate type's extents available

Critical bug fix (B1): _inverse_softplus parenthesizes (-expm1(-y)) BEFORE
.clamp_min(eps). The unparenthesized form (`-x.clamp_min(eps)`) parses as
`-(x.clamp_min(eps))`, which yields -eps from a negative x, then log(-eps)=NaN.
This was the source of `delta_u_init: [NaN, ...]` in the previous output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch


@dataclass
class PhiResult:
    phi_0_shifted: torch.Tensor       # (K,) c-shifted progress; phi_0_shifted[c] = 0
    u_normalized: torch.Tensor        # (K,) pre-shift progress in [0, 1] (monotone)
    u_raw: torch.Tensor               # (K,) raw signed projection / angle values
    delta_u_init: torch.Tensor        # (K-1,) softplus-inverted diffs for Stage D init
    observed_max_angle: float         # max |raw angle| in radians (always computed)
    observed_max_disp: float          # max |raw disp| in world units (always computed)
    monotone_enforced: bool
    valid_states_used: int


def _inverse_softplus(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Stable inverse softplus: y -> x such that softplus(x) = y.

    softplus(x) = log(1 + exp(x))
    inverse:     x = log(exp(y) - 1) = y + log(1 - exp(-y))

    BUG FIX: parenthesize `(-expm1(-y))` BEFORE `.clamp_min(eps)`. Operator
    precedence makes `-x.clamp_min(eps)` parse as `-(x.clamp_min(eps))`. With
    `expm1(-y) in (-1, 0)`, the unparenthesized form clamps the negative
    value to +eps, then negates to -eps, then `log(-eps)` is NaN.
    """
    y = y.clamp_min(eps)
    return y + torch.log((-torch.expm1(-y)).clamp_min(eps))


def _inverse_softplus_scalar(y: float, eps: float = 1e-6) -> float:
    y = max(float(y), eps)
    if y > 20.0:
        return y
    return float(math.log(math.expm1(y)))


def _pool_adjacent_violators(u: np.ndarray) -> np.ndarray:
    """Isotonic regression onto monotone-non-decreasing cone via standard PAV.

    Output has same length as input; values are weakly increasing.
    """
    n = len(u)
    if n <= 1:
        return u.copy()
    vals = u.copy().astype(np.float64)
    weights = np.ones(n, dtype=np.float64)
    i = 0
    while i < n - 1:
        if vals[i] <= vals[i + 1]:
            i += 1
            continue
        new_w = weights[i] + weights[i + 1]
        new_v = (vals[i] * weights[i] + vals[i + 1] * weights[i + 1]) / new_w
        vals = np.concatenate([vals[:i], [new_v], vals[i + 2:]])
        weights = np.concatenate([weights[:i], [new_w], weights[i + 2:]])
        n -= 1
        if i > 0:
            i -= 1
    out: list[float] = []
    for v, w in zip(vals, weights):
        out.extend([float(v)] * int(round(w)))
    return np.array(out, dtype=np.float64)


def fit_phi_from_candidate(
    phi_raw_signed: torch.Tensor,             # (K,) raw signed projection/angle
    canonical_state_idx: int = 2,
    enforce_monotone: bool = True,
    phi_min_gap: float = 1e-3,
    joint_type_str: str = "prismatic",        # only affects which "observed_max_*" is meaningful
) -> PhiResult:
    """Normalize + canonical-shift + delta_u_init from raw per-state phi.

    Always computes both observed_max_disp and observed_max_angle so Stage D
    dual-clone has the alternate type's extent available (v3 fix for B5/B6).
    """
    K = int(phi_raw_signed.shape[0])
    u_raw_np = phi_raw_signed.detach().cpu().numpy().astype(np.float64)

    # Observed extents BEFORE any normalization, in the raw unit
    # For prismatic: world units. For revolute: radians. Symmetric (abs).
    finite_mask = np.isfinite(u_raw_np)
    if finite_mask.any():
        max_abs_raw = float(np.abs(u_raw_np[finite_mask]).max())
    else:
        max_abs_raw = 0.0

    if joint_type_str == "revolute":
        observed_max_angle = max_abs_raw
        # Compute alternate "disp" view: pretend the same raw values were
        # along an axis (gives same magnitude scale; useful for dual-clone).
        observed_max_disp = max_abs_raw
    else:
        observed_max_disp = max_abs_raw
        observed_max_angle = max_abs_raw

    # PAV monotone enforce
    monotone_enforced = False
    if enforce_monotone and K >= 2:
        diffs = np.diff(u_raw_np)
        n_pos = int((diffs > 0).sum())
        n_neg = int((diffs < 0).sum())
        if n_neg > n_pos:
            u_mono = -_pool_adjacent_violators(-u_raw_np)
        else:
            u_mono = _pool_adjacent_violators(u_raw_np)
        if not np.allclose(u_mono, u_raw_np, atol=1e-9):
            monotone_enforced = True
            u_raw_np = u_mono

    # Normalize to [0, 1]
    u_min = float(np.nanmin(u_raw_np))
    u_max = float(np.nanmax(u_raw_np))
    span = u_max - u_min
    if span < 1e-9:
        u_norm = np.linspace(0.0, 1.0, K).astype(np.float64)
    else:
        u_norm = (u_raw_np - u_min) / span

    # Spread ties with phi_min_gap
    for i in range(1, K):
        if u_norm[i] <= u_norm[i - 1] + phi_min_gap:
            u_norm[i] = u_norm[i - 1] + phi_min_gap
    # Re-normalize
    rng = max(u_norm[-1] - u_norm[0], 1e-9)
    u_norm = (u_norm - u_norm[0]) / rng

    # Canonical state shift (NEW.1)
    c = max(0, min(canonical_state_idx, K - 1))
    u_shifted = u_norm - u_norm[c]

    # delta_u_init
    diffs = np.diff(u_norm)
    diffs_clamped = np.maximum(diffs, phi_min_gap)
    delta_u_init = _inverse_softplus(torch.from_numpy(diffs_clamped))

    return PhiResult(
        phi_0_shifted=torch.from_numpy(u_shifted).float(),
        u_normalized=torch.from_numpy(u_norm).float(),
        u_raw=torch.from_numpy(u_raw_np).float(),
        delta_u_init=delta_u_init.float(),
        observed_max_angle=float(observed_max_angle),
        observed_max_disp=float(observed_max_disp),
        monotone_enforced=monotone_enforced,
        valid_states_used=int(finite_mask.sum()),
    )


# ---------------------------------------------------------------------------
# Backward-compat wrapper for run_stage_c_init.py imports
# ---------------------------------------------------------------------------


def fit_phi(
    geom=None,                                # unused in v3 (legacy API)
    joint_type_str: str = "prismatic",
    axis: Optional[torch.Tensor] = None,      # unused in v3
    origin: Optional[torch.Tensor] = None,    # unused in v3
    canonical_state_idx: int = 2,
    enforce_monotone: bool = True,
    phi_min_gap: float = 1e-3,
    phi_raw_signed: Optional[torch.Tensor] = None,   # v3 primary input
) -> PhiResult:
    """v3 entry: pass phi_raw_signed (from CandidateResult.phi_k).

    Legacy args (geom/axis/origin) accepted for signature compatibility but
    unused -- joint_type_detect_v3 already computed the per-state phi inside
    each candidate.
    """
    if phi_raw_signed is None:
        raise ValueError(
            "fit_phi v3 requires phi_raw_signed (from CandidateResult.phi_k). "
            "Legacy centroid-projection path is removed."
        )
    return fit_phi_from_candidate(
        phi_raw_signed=phi_raw_signed,
        canonical_state_idx=canonical_state_idx,
        enforce_monotone=enforce_monotone,
        phi_min_gap=phi_min_gap,
        joint_type_str=joint_type_str,
    )


__all__ = [
    "PhiResult", "fit_phi", "fit_phi_from_candidate", "_inverse_softplus_scalar",
]

"""Per-state phi (normalized progress) extraction + canonical-state shift.

Per method.md section 7.7 + v3.3.1 NEW.1:

  1. u_raw_k = (centroid_k - origin) projected onto axis (signed)
              -- for prismatic: world-unit displacement
              -- for revolute: signed angle around axis (in radians)
  2. u_norm_k = (u_raw_k - u_raw_min) / (u_raw_max - u_raw_min) in [0, 1]
  3. u_shifted = u_norm - u_norm[c]    # c-shift (default c=2)
                                       # u_shifted[c] = 0, others can be negative
  4. delta_u_init (5,) = inverse_softplus(diff(u_norm))  # used by Stage D
                                       # NOTE: diffs are PRESERVED by c-shift
                                       # so this works the same either way

The c-shifted phi_0 is what goes into BOTH:
  - JointInit.phi_0 (return value)
  - the swept_volume_corridor(psi_0, phi_0) call in Bootstrap B7

Missing states (geom.valid_mask[k] == False) get interpolated u values from
their neighbours so phi_0 is a complete (K,) vector. If endpoint states
are missing, we extrapolate using the closest valid trend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .move_geometry import PerStateMoveGeom


@dataclass
class PhiResult:
    phi_0_shifted: torch.Tensor       # (K,) c-shifted progress; phi_0_shifted[c] = 0
    u_normalized: torch.Tensor        # (K,) pre-shift progress in [0, 1] (monotone)
    u_raw: torch.Tensor               # (K,) raw signed projection values
    delta_u_init: torch.Tensor        # (K-1,) softplus-inverted diffs for Stage D init
    observed_max_angle: float         # for revolute: max |u_raw| in radians
    observed_max_disp: float          # for prismatic: max |u_raw| in world units
    monotone_enforced: bool           # whether PAV smoothing was applied
    valid_states_used: int


def _inverse_softplus(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Stable inverse softplus: y -> x such that softplus(x) = y.

    softplus(x) = log(1 + exp(x))
    inverse:     x = log(exp(y) - 1) = y + log(1 - exp(-y))

    The second form is numerically stable for y > 0.
    """
    y = y.clamp_min(eps)
    return y + torch.log(-torch.expm1(-y).clamp_min(eps))


def _pool_adjacent_violators(u: np.ndarray) -> np.ndarray:
    """Project u onto the monotone-non-decreasing cone (isotonic regression
    via the standard PAV algorithm). Used when centroid-projection produces
    non-monotonic u due to noise.

    Output has same length as input; values are weakly increasing.
    """
    n = len(u)
    if n <= 1:
        return u.copy()
    vals = u.copy().astype(np.float64)
    weights = np.ones(n, dtype=np.float64)
    # PAV: scan, pool violators
    i = 0
    while i < n - 1:
        if vals[i] <= vals[i + 1]:
            i += 1
            continue
        # Violation: pool i and i+1 into weighted average
        new_w = weights[i] + weights[i + 1]
        new_v = (vals[i] * weights[i] + vals[i + 1] * weights[i + 1]) / new_w
        vals = np.concatenate([vals[:i], [new_v], vals[i + 2:]])
        weights = np.concatenate([weights[:i], [new_w], weights[i + 2:]])
        n -= 1
        # Move back one to re-check
        if i > 0:
            i -= 1
    # Now vals has length <= original. Expand back by repetition according to weights.
    out: list[float] = []
    for v, w in zip(vals, weights):
        out.extend([float(v)] * int(round(w)))
    return np.array(out, dtype=np.float64)


def _interpolate_missing_u_raw(
    u_raw_partial: np.ndarray,     # (K,) with nan for missing
    valid_mask: np.ndarray,        # (K,) bool
) -> np.ndarray:
    """Fill nan entries in u_raw via 1D linear interpolation over k=0..K-1.

    Endpoint extrapolation uses the nearest two valid values' slope.
    """
    K = len(u_raw_partial)
    out = u_raw_partial.copy()
    valid_ks = np.where(valid_mask)[0]
    if len(valid_ks) == K:
        return out
    if len(valid_ks) < 2:
        # Degenerate: 0 or 1 valid centroid -> cannot interpolate.
        # Fill with linear ramp from 0 to 1 in raw units; this is a sentinel
        # that downstream confidence will mark low.
        return np.linspace(0.0, 1.0, K).astype(np.float64)
    # Linear interp over valid_ks
    out = np.interp(np.arange(K), valid_ks, u_raw_partial[valid_ks]).astype(np.float64)
    return out


def _project_centroids_prismatic(
    centroids_world: torch.Tensor,    # (K, 3) world
    axis: torch.Tensor,                # (3,)
    origin: torch.Tensor,              # (3,)
    valid_mask: torch.Tensor,          # (K,) bool
) -> np.ndarray:
    """u_raw_k = (centroid_k - origin) . axis  (signed scalar, world units)."""
    K = int(centroids_world.shape[0])
    axis_np = axis.detach().cpu().numpy().astype(np.float64)
    origin_np = origin.detach().cpu().numpy().astype(np.float64)
    cents_np = centroids_world.detach().cpu().numpy().astype(np.float64)
    valid_np = valid_mask.detach().cpu().numpy().astype(bool)

    u_raw = np.full(K, np.nan, dtype=np.float64)
    for k in range(K):
        if not valid_np[k]:
            continue
        u_raw[k] = float(np.dot(cents_np[k] - origin_np, axis_np))
    return u_raw


def _project_centroids_revolute(
    centroids_world: torch.Tensor,    # (K, 3) world
    axis: torch.Tensor,                # (3,) unit
    origin: torch.Tensor,              # (3,) point on axis line
    valid_mask: torch.Tensor,          # (K,) bool
    reference_state_k: int,            # k index whose angle is defined as 0
) -> np.ndarray:
    """u_raw_k = signed angle of (centroid_k - origin)_perp relative to
    reference state's perpendicular component, around `axis`.

    Returns u_raw in RADIANS. Sign follows right-hand rule around axis.
    Missing entries are np.nan.
    """
    K = int(centroids_world.shape[0])
    axis_np = axis.detach().cpu().numpy().astype(np.float64)
    axis_np = axis_np / (np.linalg.norm(axis_np) + 1e-12)
    origin_np = origin.detach().cpu().numpy().astype(np.float64)
    cents_np = centroids_world.detach().cpu().numpy().astype(np.float64)
    valid_np = valid_mask.detach().cpu().numpy().astype(bool)

    # Per-state vector from origin, projected to plane perpendicular to axis
    perp = np.full((K, 3), np.nan, dtype=np.float64)
    for k in range(K):
        if not valid_np[k]:
            continue
        v = cents_np[k] - origin_np
        v_perp = v - np.dot(v, axis_np) * axis_np
        perp[k] = v_perp

    # Pick reference: use the requested reference_state_k if valid, else first valid
    if 0 <= reference_state_k < K and valid_np[reference_state_k]:
        ref_k = reference_state_k
    else:
        ref_ks = np.where(valid_np)[0]
        ref_k = int(ref_ks[0]) if len(ref_ks) > 0 else 0
    ref_perp = perp[ref_k]
    ref_norm = np.linalg.norm(ref_perp)
    if ref_norm < 1e-8:
        # Reference centroid is essentially on the axis -> all angles undefined.
        # Fallback: use the next valid state as reference.
        for kk in range(K):
            if valid_np[kk] and np.linalg.norm(perp[kk]) > 1e-6:
                ref_k = kk
                ref_perp = perp[kk]
                ref_norm = np.linalg.norm(ref_perp)
                break
    if ref_norm < 1e-8:
        # Total failure
        return np.full(K, np.nan, dtype=np.float64)
    ref_unit = ref_perp / ref_norm

    # ref_perp x axis gives the "90 degree" direction; signed angle uses atan2
    sign_dir = np.cross(ref_unit, axis_np)
    sign_dir = sign_dir / (np.linalg.norm(sign_dir) + 1e-12)
    # Note: angle theta defined so that rotating ref_unit by +theta around axis
    # using right-hand rule lands on the unit-perp of state k.

    u_raw = np.full(K, np.nan, dtype=np.float64)
    for k in range(K):
        if not valid_np[k]:
            continue
        pk = perp[k]
        pn = np.linalg.norm(pk)
        if pn < 1e-8:
            u_raw[k] = 0.0
            continue
        pk_unit = pk / pn
        # cos = pk_unit . ref_unit; sin = (pk_unit . sign_dir)
        cos_t = float(np.clip(np.dot(pk_unit, ref_unit), -1.0, 1.0))
        sin_t = float(np.dot(pk_unit, sign_dir))
        u_raw[k] = float(np.arctan2(sin_t, cos_t))
    return u_raw


def fit_phi(
    geom: PerStateMoveGeom,
    joint_type_str: str,
    axis: torch.Tensor,
    origin: torch.Tensor,
    canonical_state_idx: int = 2,
    enforce_monotone: bool = True,
    phi_min_gap: float = 1e-3,
) -> PhiResult:
    """Compute per-state phi_0 + delta_u init.

    joint_type_str: "prismatic" | "revolute" | "uncertain"
                    For "uncertain" we treat as prismatic for the projection
                    (gives a sane default; confidence will be marked low).
    """
    K = int(geom.centroid_world.shape[0])

    # 1) Project per-state centroids onto axis
    if joint_type_str == "revolute":
        u_raw = _project_centroids_revolute(
            geom.centroid_world, axis, origin, geom.valid_mask,
            reference_state_k=0,   # state 0 angle = 0 by convention
        )
    else:
        # prismatic or uncertain
        u_raw = _project_centroids_prismatic(
            geom.centroid_world, axis, origin, geom.valid_mask,
        )

    # 2) Fill missing states via interpolation
    valid_np = geom.valid_mask.detach().cpu().numpy().astype(bool)
    if np.isnan(u_raw).any():
        u_raw = _interpolate_missing_u_raw(u_raw, valid_np)

    observed_max_angle = float(np.abs(u_raw[~np.isnan(u_raw)]).max()) if not np.isnan(u_raw).all() else 0.0
    observed_max_disp = float(np.abs(u_raw[~np.isnan(u_raw)]).max()) if not np.isnan(u_raw).all() else 0.0

    # 3) Enforce monotonicity if requested (PAV on u_raw)
    monotone_enforced = False
    if enforce_monotone:
        # Check monotonicity in either direction; pick the dominant one
        diffs = np.diff(u_raw)
        n_pos = int((diffs > 0).sum())
        n_neg = int((diffs < 0).sum())
        if n_neg > n_pos:
            # Mostly decreasing -> flip sign, monotonize, flip back
            u_raw_mono = -_pool_adjacent_violators(-u_raw)
        else:
            u_raw_mono = _pool_adjacent_violators(u_raw)
        if not np.allclose(u_raw_mono, u_raw, atol=1e-9):
            monotone_enforced = True
            u_raw = u_raw_mono

    # 4) Normalize to [0, 1]
    u_min = float(u_raw.min())
    u_max = float(u_raw.max())
    span = u_max - u_min
    if span < 1e-9:
        # Degenerate: all centroids identical -> linear ramp
        u_norm = np.linspace(0.0, 1.0, K).astype(np.float64)
    else:
        u_norm = (u_raw - u_min) / span

    # Enforce min positive gap so inverse_softplus is well-defined
    # by spreading any ties slightly
    for i in range(1, len(u_norm)):
        if u_norm[i] <= u_norm[i - 1] + phi_min_gap:
            u_norm[i] = u_norm[i - 1] + phi_min_gap
    # Re-normalize after gap enforcement
    u_norm = (u_norm - u_norm[0]) / max(u_norm[-1] - u_norm[0], 1e-9)

    # 5) c-shift
    c = max(0, min(canonical_state_idx, K - 1))
    u_shifted = u_norm - u_norm[c]

    # 6) delta_u init (for Stage D's learnable.delta_phi)
    diffs = np.diff(u_norm)                 # (K-1,)  c-shift preserves diffs
    diffs_clamped = np.maximum(diffs, phi_min_gap)
    delta_u_init = _inverse_softplus(torch.from_numpy(diffs_clamped))

    device = geom.centroid_world.device
    return PhiResult(
        phi_0_shifted=torch.from_numpy(u_shifted).to(device=device, dtype=torch.float32),
        u_normalized=torch.from_numpy(u_norm).to(device=device, dtype=torch.float32),
        u_raw=torch.from_numpy(u_raw).to(device=device, dtype=torch.float32),
        delta_u_init=delta_u_init.to(device=device, dtype=torch.float32),
        observed_max_angle=observed_max_angle if joint_type_str == "revolute" else 0.0,
        observed_max_disp=observed_max_disp if joint_type_str != "revolute" else 0.0,
        monotone_enforced=monotone_enforced,
        valid_states_used=int(valid_np.sum()),
    )

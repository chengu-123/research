"""Joint type + axis detection via cardinal-axis enumeration + voxel scoring.

v3 (cardinal-cand + voxel-physical-scoring; supersedes centroid-only v2):

Algorithm:
  Step A. Build per-state move voxel sets V_k (in voxel int and world coords)
          and the swept union V_union.
  Step B. FreeArt3D-style cardinal axis invariant (estimate.py:387-401):
          - Compute pair-wise centroid displacement vectors
          - prismatic axis = cardinal with MAX |sum proj| (parallel to motion)
          - revolute axis  = cardinal with MIN |sum proj| (perpendicular)
          These give us geometric prior; physical scoring is the final arbiter.
  Step C. For BOTH types (prismatic, revolute), generate K_cardinal candidates
          on the 6 cardinal axes (sign convention: ensure phi monotone-increasing
          in k by flipping axis if needed). For revolute, origin = anchor band
          centroid projected onto axis line through swept centroid (physical
          constraint: hinge must be at base-move contact).
  Step D. For each candidate, run voxel reverse-warp scoring (voxel_scoring.py):
              score = consistency * (1 - conflict) * coverage *
                      contact_compat * monotone_quality
  Step E. best_pris = argmax score over 6 prismatic candidates
          best_rev  = argmax score over 6 revolute candidates
          type_logit = log(best_pris.score / best_rev.score)
          recommended = sign(type_logit) when margin > threshold

The result is a JointTypeResult that records BOTH best_pris and best_rev so
axis_fit / phi_fit / run_stage_c_init can construct primary AND secondary
JointInit (Stage D dual-clone gets two well-initialized branches).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from .voxel_scoring import (
    CandidateScore,
    angular_median_around_axis,
    envelope_advance_along_axis,
    freeart3d_axis_invariant,
    reverse_align_and_score,
    voxel_to_world,
)


# ---------------------------------------------------------------------------
# Cardinal set
# ---------------------------------------------------------------------------


def cardinal_axes(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Six cardinal axes as a (6, 3) tensor: +X, -X, +Y, -Y, +Z, -Z."""
    return torch.tensor([
        [+1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, +1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0, +1.0], [0.0, 0.0, -1.0],
    ], device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Per-state voxel set builder
# ---------------------------------------------------------------------------


def build_per_state_voxel_sets(
    O_move_per_state: Optional[torch.Tensor],         # (K, D, H, W) uint8/bool
    P_move_evidence_per_state: Optional[torch.Tensor],# (K, D, H, W) float
    O_base_canonical: Optional[torch.Tensor],         # (D, H, W) uint8/bool
    is_carpet_mask_flat: torch.Tensor,                # (res^3,) bool
    res: int = 64,
    soft_threshold: float = 0.1,
    min_voxels: int = 20,
    prefer_soft: bool = True,
) -> Tuple[List[torch.Tensor], torch.Tensor, List[bool]]:
    """Build per-state cleaned voxel coordinate lists.

    Returns
    -------
    V_per_state_voxel : list of K tensors (N_k, 3) int voxel coords
                        Empty tensor if state's evidence below min_voxels.
    V_union_voxel     : (N_union, 3) int voxel coords of OR_k(V_k)
    valid_mask        : list of K bool, True if state has >= min_voxels
    """
    device = is_carpet_mask_flat.device
    carpet_3d = is_carpet_mask_flat.reshape(res, res, res).to(device)
    base_3d = (
        O_base_canonical.to(device).bool()
        if O_base_canonical is not None
        else torch.zeros(res, res, res, dtype=torch.bool, device=device)
    )

    if prefer_soft and P_move_evidence_per_state is not None:
        src = P_move_evidence_per_state.to(device)
        K = int(src.shape[0])
        masks = [(src[k] >= soft_threshold) for k in range(K)]
    elif O_move_per_state is not None:
        src = O_move_per_state.to(device)
        K = int(src.shape[0])
        masks = [src[k].bool() for k in range(K)]
    elif P_move_evidence_per_state is not None:
        src = P_move_evidence_per_state.to(device)
        K = int(src.shape[0])
        masks = [(src[k] >= soft_threshold) for k in range(K)]
    else:
        raise ValueError(
            "build_per_state_voxel_sets: need at least one of "
            "O_move_per_state or P_move_evidence_per_state"
        )

    V_per_state: List[torch.Tensor] = []
    valid: List[bool] = []
    union_mask = torch.zeros(res, res, res, dtype=torch.bool, device=device)
    for k in range(K):
        m = masks[k] & (~base_3d) & (~carpet_3d)
        coords = torch.nonzero(m, as_tuple=False).to(torch.int32)
        if coords.shape[0] >= min_voxels:
            V_per_state.append(coords)
            valid.append(True)
            union_mask = union_mask | m
        else:
            V_per_state.append(torch.zeros(0, 3, dtype=torch.int32, device=device))
            valid.append(False)
    V_union = torch.nonzero(union_mask, as_tuple=False).to(torch.int32)
    return V_per_state, V_union, valid


# ---------------------------------------------------------------------------
# Candidate generation: phi_k per state given (type, axis, origin)
# ---------------------------------------------------------------------------


def _project_phi_prismatic(
    V_per_state_world: List[torch.Tensor],
    axis_unit: torch.Tensor,
    valid: List[bool],
    canonical_state_idx: int,
    percentile: float = 0.5,
) -> torch.Tensor:
    """Per-state phi from percentile projection of voxels on axis (signed).

    Uses median (percentile=0.5) which is robust to occlusion-revealed extra
    voxels; the user noted Stage B per-state count is NOT monotone for prismatic
    so the median per-state position along axis is more stable than mean or
    extreme percentile.

    Returns (K,) tensor; NaN for invalid states (filled later by interpolation).
    """
    K = len(V_per_state_world)
    out = torch.full((K,), float("nan"))
    for k in range(K):
        if not valid[k]:
            continue
        Vk = V_per_state_world[k]
        proj = (Vk * axis_unit.unsqueeze(0)).sum(dim=-1)
        out[k] = float(torch.quantile(proj, percentile).item())
    # Fill NaN by linear interpolation over valid k indices
    return _interp_nan_linear(out)


def _project_phi_revolute(
    V_per_state_world: List[torch.Tensor],
    axis_unit: torch.Tensor,
    origin: torch.Tensor,
    valid: List[bool],
    canonical_state_idx: int,
) -> torch.Tensor:
    """Per-state phi from angular median around axis. Returns (K,) radians.

    Reference perpendicular = canonical state's centroid - origin, projected
    perpendicular to axis. NaN-filled by interpolation.
    """
    K = len(V_per_state_world)
    a = axis_unit / axis_unit.norm().clamp_min(1e-12)
    # Build reference perp from canonical state
    ref_perp = None
    if 0 <= canonical_state_idx < K and valid[canonical_state_idx]:
        Vc = V_per_state_world[canonical_state_idx]
        cc = Vc.mean(dim=0)
        v = cc - origin
        v_perp = v - (v @ a) * a
        if v_perp.norm().item() > 1e-6:
            ref_perp = v_perp / v_perp.norm()
    if ref_perp is None:
        # Fallback: use first valid state
        for k in range(K):
            if not valid[k]:
                continue
            ck = V_per_state_world[k].mean(dim=0)
            v = ck - origin
            v_perp = v - (v @ a) * a
            if v_perp.norm().item() > 1e-6:
                ref_perp = v_perp / v_perp.norm()
                break
    if ref_perp is None:
        return torch.zeros(K)

    out = angular_median_around_axis(
        V_per_state_world, axis_unit, origin, ref_perp_unit=ref_perp,
    )
    return _interp_nan_linear(out)


def _interp_nan_linear(x: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate NaN entries in a 1D tensor."""
    arr = x.detach().cpu().numpy().astype(np.float64)
    K = len(arr)
    valid_idx = np.where(~np.isnan(arr))[0]
    if len(valid_idx) == K:
        return torch.from_numpy(arr).float()
    if len(valid_idx) < 2:
        # Degenerate: <= 1 valid -> linear ramp 0..1
        return torch.linspace(0.0, 1.0, K)
    arr_filled = np.interp(np.arange(K), valid_idx, arr[valid_idx])
    return torch.from_numpy(arr_filled).float()


def _resolve_revolute_origin(
    axis_unit: torch.Tensor,
    swept_centroid: torch.Tensor,
    anchors_world: Optional[torch.Tensor],
) -> torch.Tensor:
    """Compute origin = point on axis line {swept_centroid + t*axis} nearest to
    anchor centroid. If anchors absent, return swept_centroid.

    Physical motivation: hinge axis must pass through base-move contact band.
    """
    if anchors_world is None or anchors_world.shape[0] == 0:
        return swept_centroid.clone()
    anchor_centroid = anchors_world.mean(dim=0)
    t_star = float((anchor_centroid - swept_centroid) @ axis_unit)
    return swept_centroid + t_star * axis_unit


# ---------------------------------------------------------------------------
# Public result schema
# ---------------------------------------------------------------------------


@dataclass
class CandidateResult:
    """One (type, axis, origin, phi_k) candidate + its score."""

    type_str: str                        # "prismatic" | "revolute"
    axis: torch.Tensor                   # (3,) cardinal unit vector
    origin: torch.Tensor                 # (3,) world
    phi_k: torch.Tensor                  # (K,) per-state progress
    score: CandidateScore                # voxel-level physical score breakdown


@dataclass
class JointTypeResult:
    """v3 dual-candidate output (replaces v2 single line-vs-arc result)."""

    # Selected primary
    type_str: str                        # "prismatic" | "revolute" | "uncertain"
    type_logit: float                    # log(best_pris.score / best_rev.score)
    confidence: float                    # |type_logit| / margin_norm, clamped [0,1]

    # Best candidate per type (Stage D dual-clone uses both)
    best_pris: Optional[CandidateResult] = None
    best_rev: Optional[CandidateResult] = None

    # All candidates for diagnostics
    all_candidates: List[CandidateResult] = field(default_factory=list)

    # Voxel-statistical diagnostics
    n_valid_states: int = 0
    pris_geom_scores: Optional[torch.Tensor] = None  # (6,) FreeArt3D argmax values
    rev_geom_scores: Optional[torch.Tensor] = None   # (6,) FreeArt3D argmin values

    # Legacy fields for confidence.py / run_stage_c_init.py compatibility
    residual_line: float = float("inf")
    residual_arc: float = float("inf")
    line_origin: Optional[np.ndarray] = None
    line_direction: Optional[np.ndarray] = None
    arc_center: Optional[np.ndarray] = None
    arc_normal: Optional[np.ndarray] = None
    arc_radius: Optional[float] = None


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def detect_joint_type_v3(
    V_per_state_voxel: List[torch.Tensor],            # K tensors of (N_k, 3) int
    V_union_voxel: torch.Tensor,                       # (N_u, 3) int
    valid_state: List[bool],
    O_base_canonical: Optional[torch.Tensor],         # (res, res, res) for conflict check
    anchors_voxel: Optional[torch.Tensor],            # (N_a, 3) int, may be None
    canonical_state_idx: int = 2,
    res: int = 64,
    type_margin: float = 0.20,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> JointTypeResult:
    """v3 cardinal-cand + voxel-scoring joint type detection.

    Replaces v2's centroid line-vs-arc residual ratio.
    """
    if device is None:
        device = V_union_voxel.device
    K = len(V_per_state_voxel)
    n_valid = int(sum(valid_state))

    if n_valid < 2 or V_union_voxel.shape[0] < 10:
        # Degenerate input
        return JointTypeResult(
            type_str="uncertain", type_logit=0.0, confidence=0.0,
            n_valid_states=n_valid,
        )

    # ---- Step A: World coordinates ----
    V_per_state_world: List[torch.Tensor] = []
    for V_int in V_per_state_voxel:
        if V_int.shape[0] == 0:
            V_per_state_world.append(torch.zeros(0, 3, device=device, dtype=dtype))
        else:
            V_per_state_world.append(
                voxel_to_world(V_int, res=res).to(device=device, dtype=dtype)
            )
    V_union_world = voxel_to_world(V_union_voxel, res=res).to(device=device, dtype=dtype)
    swept_centroid = V_union_world.mean(dim=0)
    anchors_world = (
        voxel_to_world(anchors_voxel, res=res).to(device=device, dtype=dtype)
        if anchors_voxel is not None and anchors_voxel.shape[0] > 0
        else None
    )

    # ---- Step B: FreeArt3D-style axis invariant (geometric prior) ----
    pris_geom, rev_geom = freeart3d_axis_invariant(V_per_state_world)

    # ---- Step C+D: Enumerate cardinal candidates per type, score each ----
    cardinals = cardinal_axes(device, dtype)             # (6, 3)
    all_candidates: List[CandidateResult] = []
    pris_candidates: List[CandidateResult] = []
    rev_candidates: List[CandidateResult] = []

    for axis_idx in range(6):
        a = cardinals[axis_idx].clone()

        # ===== Prismatic candidate =====
        origin_pris = swept_centroid.clone()  # any point on axis OK for prismatic
        phi_pris = _project_phi_prismatic(
            V_per_state_world, a, valid_state, canonical_state_idx,
        )
        # Sign convention: ensure phi advances with k (else flip axis + phi sign)
        if int(K) >= 2:
            first_valid = next((k for k in range(K) if valid_state[k]), 0)
            last_valid = next((k for k in range(K - 1, -1, -1) if valid_state[k]), K - 1)
            if phi_pris[last_valid].item() < phi_pris[first_valid].item():
                a_signed = -a
                phi_pris = -phi_pris
            else:
                a_signed = a
        else:
            a_signed = a

        score_pris = reverse_align_and_score(
            joint_type="prismatic",
            axis=a_signed,
            origin=origin_pris,
            phi_k=phi_pris,
            canonical_state_idx=canonical_state_idx,
            V_per_state_voxel=V_per_state_voxel,
            V_union_voxel=V_union_voxel,
            O_base_canonical=O_base_canonical,
            anchors_world=anchors_world,
            res=res,
        )
        c_pris = CandidateResult(
            type_str="prismatic", axis=a_signed, origin=origin_pris,
            phi_k=phi_pris, score=score_pris,
        )
        pris_candidates.append(c_pris)
        all_candidates.append(c_pris)

        # ===== Revolute candidate =====
        origin_rev = _resolve_revolute_origin(a, swept_centroid, anchors_world)
        phi_rev = _project_phi_revolute(
            V_per_state_world, a, origin_rev, valid_state, canonical_state_idx,
        )
        if int(K) >= 2:
            first_valid = next((k for k in range(K) if valid_state[k]), 0)
            last_valid = next((k for k in range(K - 1, -1, -1) if valid_state[k]), K - 1)
            if phi_rev[last_valid].item() < phi_rev[first_valid].item():
                a_rev = -a
                phi_rev = -phi_rev
                origin_rev = _resolve_revolute_origin(a_rev, swept_centroid, anchors_world)
            else:
                a_rev = a
        else:
            a_rev = a

        score_rev = reverse_align_and_score(
            joint_type="revolute",
            axis=a_rev,
            origin=origin_rev,
            phi_k=phi_rev,
            canonical_state_idx=canonical_state_idx,
            V_per_state_voxel=V_per_state_voxel,
            V_union_voxel=V_union_voxel,
            O_base_canonical=O_base_canonical,
            anchors_world=anchors_world,
            res=res,
        )
        c_rev = CandidateResult(
            type_str="revolute", axis=a_rev, origin=origin_rev,
            phi_k=phi_rev, score=score_rev,
        )
        rev_candidates.append(c_rev)
        all_candidates.append(c_rev)

    # ---- Step E: Best per type + type decision ----
    best_pris = max(pris_candidates, key=lambda c: c.score.score)
    best_rev = max(rev_candidates, key=lambda c: c.score.score)

    eps = 1e-6
    s_pris = max(best_pris.score.score, eps)
    s_rev = max(best_rev.score.score, eps)
    type_logit = float(math.log(s_pris / s_rev))

    if type_logit > type_margin:
        type_str = "prismatic"
    elif type_logit < -type_margin:
        type_str = "revolute"
    else:
        type_str = "uncertain"

    # Confidence from margin
    confidence = float(min(abs(type_logit) / max(type_margin * 2.0, eps), 1.0))

    return JointTypeResult(
        type_str=type_str,
        type_logit=type_logit,
        confidence=confidence,
        best_pris=best_pris,
        best_rev=best_rev,
        all_candidates=all_candidates,
        n_valid_states=n_valid,
        pris_geom_scores=pris_geom.detach().cpu(),
        rev_geom_scores=rev_geom.detach().cpu(),
    )


# ---------------------------------------------------------------------------
# Backward-compat wrapper for run_stage_c_init.py + confidence.py
# ---------------------------------------------------------------------------


def detect_joint_type(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Legacy entry name kept for compatibility with existing imports.

    New code should call `detect_joint_type_v3` directly with explicit args.
    This wrapper is intentionally minimal -- callers pass the v3 signature.
    """
    return detect_joint_type_v3(*args, **kwargs)


__all__ = [
    "JointTypeResult", "CandidateResult", "detect_joint_type_v3", "detect_joint_type",
    "build_per_state_voxel_sets", "cardinal_axes",
]

"""C.0 Count-based signature partition with M_attn-classified always_on.

v6 rewrite (2026-04-23): reject statistics-based soft partition; use
the Bayesian-style count distribution directly. For each voxel ``v``
the cross-state occupancy count ``c(v) = Σ_k O_k(v) ∈ {0, ..., K}``
gives three high-confidence classes:

* ``c(v) = 0``      — air (never occupied)
* ``c(v) = K``      — always_on (shared across all K states)
* ``c(v) ≤ move_max`` — high-confidence move seed
* rest                — boundary (2 ≤ c ≤ K-1)

The always_on class is **ambiguous by signature alone** (cabinet
interior vs long-drawer body vs hinge-near-door look identical as
signature). We use the Stage B v4.3 ``M_attn_64`` prior as the
**sole** disambiguation signal, applied as a three-way classifier:

* always_on ∩ ``M_attn > τ_base``       → true_base (cabinet core)
* always_on ∩ ``M_attn < τ_move``       → move_interior (drawer / door interior)
* always_on ∩ ``τ_move ≤ M_attn ≤ τ_base`` → ambiguous (deferred to swept-volume carving)

This is the ONLY place M_attn contributes to segmentation.

Canonical (state-0 frame, per stagec_3.md) is strictly
``Ω_c = {v : O_0(v) = 1}``. All downstream canonical_base /
canonical_move must be subsets of Ω_c.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class CountPartition:
    """Result of count-based partition with M_attn classification."""

    count: torch.Tensor                 # (D, H, W) int — cross-state occupancy count
    footprint: torch.Tensor             # (D, H, W) bool — count > 0
    always_on: torch.Tensor             # (D, H, W) bool — count == K
    shell: torch.Tensor                 # (D, H, W) bool — 0 < count < K
    move_strong: torch.Tensor           # (D, H, W) bool — count ≤ count_move_max (high-conf move)
    true_base: torch.Tensor             # (D, H, W) bool — always_on ∩ (M_attn > τ_base)
    move_interior: torch.Tensor         # (D, H, W) bool — always_on ∩ (M_attn < τ_move)
    ambiguous_on: torch.Tensor          # (D, H, W) bool — always_on ∩ middle M_attn band
    # Per-state hard masks (initial, pre-EM)
    move_mask_k: torch.Tensor           # (K, D, H, W) bool — O_k ∩ (shell ∪ move_interior)
    base_mask_k: torch.Tensor           # (K, D, H, W) bool — O_k ∩ true_base
    # Canonical-frame (state-0 anchored) seeds
    canonical_base_init: torch.Tensor   # (D, H, W) bool — true_base ∩ O_0
    canonical_move_init: torch.Tensor   # (D, H, W) bool — (shell ∪ move_interior) ∩ O_0
    canonical_omega_c: torch.Tensor     # (D, H, W) bool — O_0
    base_centroid: torch.Tensor         # (3,) world coord, weighted centroid of true_base


def _voxel_coord_grid(resolution: int, device: torch.device,
                      dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(resolution, device=device, dtype=dtype)
    ii, jj, kk = torch.meshgrid(idx, idx, idx, indexing="ij")
    return torch.stack([ii, jj, kk], dim=-1) / float(resolution - 1) - 0.5


def persistence_run_length(O_stack: torch.Tensor) -> torch.Tensor:
    """Maximum consecutive-1s run length in each voxel's cross-state signature."""
    if O_stack.dim() != 4:
        raise ValueError(f"O_stack must be (K,D,H,W); got {tuple(O_stack.shape)}")
    K = O_stack.shape[0]
    occ = (O_stack > 0.5).to(torch.int32)
    max_run = torch.zeros(O_stack.shape[1:], dtype=torch.int32, device=O_stack.device)
    cur_run = torch.zeros_like(max_run)
    for k in range(K):
        cur_run = torch.where(occ[k] > 0, cur_run + 1,
                               torch.zeros_like(cur_run))
        max_run = torch.maximum(max_run, cur_run)
    return max_run


def count_based_partition(
    O_stack: torch.Tensor,
    M_attn_64: Optional[torch.Tensor],
    z_final: Optional[torch.Tensor] = None,
    count_base_threshold: int = 6,
    count_move_max: int = 1,
    m_attn_base_threshold: float = 0.7,
    m_attn_move_threshold: float = 0.3,
    occupancy_threshold: float = 0.5,
    far_aon_edt_threshold: float = 15.0,
    zfinal_min_seeds: int = 20,
    zfinal_margin_coef: float = 0.3,
    eps: float = 1e-8,
) -> CountPartition:
    """Partition voxels using count signature + M_attn classifier on always_on.

    Parameters
    ----------
    O_stack : (K, D, H, W) float or bool — Stage B per-state occupancy.
    M_attn_64 : (D, H, W) in [0, 1] — Stage B v4.3 semantic prior. If
        ``None``, always_on voxels are all classified as ambiguous and
        deferred to the late-commit swept-volume carving.
    count_base_threshold : treat ``count == this`` as always_on (default K).
    count_move_max : ``count <= this`` → high-confidence move seed.
    m_attn_base_threshold, m_attn_move_threshold : thresholds for classifying
        always_on voxels (τ_base and τ_move). Middle band is ambiguous.
    """
    if O_stack.dim() != 4:
        raise ValueError(f"O_stack must be (K,D,H,W); got {tuple(O_stack.shape)}")

    device = O_stack.device
    dtype = O_stack.dtype
    K, D, H, W = O_stack.shape

    # For default count_base_threshold=K, align it if caller passed a specific value.
    if count_base_threshold is None or count_base_threshold > K:
        count_base_threshold = K

    occ_bool = O_stack > occupancy_threshold
    count = occ_bool.to(torch.int32).sum(dim=0)                 # (D, H, W)
    footprint = count > 0
    always_on = count == count_base_threshold
    shell = footprint & ~always_on
    move_strong = footprint & (count <= count_move_max)         # excludes air

    # Classify always_on voxels. Priority order (2026-04-24 audit):
    #   1. z_final-based supervised classifier (strong empirical signal)
    #   2. M_attn classifier (fallback, weak on always_on)
    #   3. No signal → everything ambiguous, defer to swept-volume carving.
    if z_final is not None:
        from .material_classifier import classify_always_on_by_zfinal
        mc = classify_always_on_by_zfinal(
            z_final=z_final.to(device=device, dtype=dtype),
            O_stack=O_stack,
            always_on=always_on,
            shell=shell,
            far_aon_edt_threshold=far_aon_edt_threshold,
            min_seeds_shell=zfinal_min_seeds,
            min_seeds_far_aon=zfinal_min_seeds,
            margin_coef=zfinal_margin_coef,
        )
        if mc.classifier_applied:
            true_base = mc.true_base
            move_interior = mc.move_interior
            ambiguous_on = mc.ambiguous_on
        else:
            # Too few seeds — fall through to M_attn path
            true_base = torch.zeros_like(always_on)
            move_interior = torch.zeros_like(always_on)
            ambiguous_on = always_on.clone()
    elif M_attn_64 is not None:
        if M_attn_64.dim() != 3 or M_attn_64.shape != (D, H, W):
            raise ValueError(
                f"M_attn_64 must be ({D}, {H}, {W}); got {tuple(M_attn_64.shape)}"
            )
        M_attn_t = M_attn_64.to(device=device, dtype=dtype)
        true_base = always_on & (M_attn_t > m_attn_base_threshold)
        move_interior = always_on & (M_attn_t < m_attn_move_threshold)
        ambiguous_on = always_on & ~true_base & ~move_interior
    else:
        # No signal: all always_on deferred to swept-volume carving
        true_base = torch.zeros_like(always_on)
        move_interior = torch.zeros_like(always_on)
        ambiguous_on = always_on.clone()

    # Canonical-frame seeds (state-0 anchored, Ω_c = O_0)
    omega_c = occ_bool[0]                                        # O_0
    canonical_base_init = true_base & omega_c
    # Move seeds in canonical: shell voxels in O_0 (state-0 leading/trailing edges)
    # plus classified move_interior within O_0.
    canonical_move_init = (shell | move_interior) & omega_c

    # Per-state hard masks — the move mask NOW INCLUDES move_interior,
    # fixing the long-drawer / hinge-near-door biases of v5.
    move_seeds_global = shell | move_interior
    move_mask_k = occ_bool & move_seeds_global.unsqueeze(0)
    base_mask_k = occ_bool & true_base.unsqueeze(0)

    # Base centroid from true_base voxels (weighted by M_attn if available)
    coords_world = _voxel_coord_grid(D, device, dtype)
    if M_attn_64 is not None:
        base_weights = (true_base.to(dtype)
                         * M_attn_64.to(device=device, dtype=dtype))
    else:
        base_weights = true_base.to(dtype)
    w_sum = base_weights.sum().clamp_min(eps)
    if float(w_sum.item()) > eps:
        base_centroid = (base_weights.unsqueeze(-1) * coords_world).sum(
            dim=(0, 1, 2),
        ) / w_sum
    else:
        # Fallback: centroid of footprint
        footprint_weights = footprint.to(dtype)
        fw_sum = footprint_weights.sum().clamp_min(eps)
        base_centroid = (footprint_weights.unsqueeze(-1) * coords_world).sum(
            dim=(0, 1, 2),
        ) / fw_sum

    return CountPartition(
        count=count,
        footprint=footprint,
        always_on=always_on,
        shell=shell,
        move_strong=move_strong,
        true_base=true_base,
        move_interior=move_interior,
        ambiguous_on=ambiguous_on,
        move_mask_k=move_mask_k,
        base_mask_k=base_mask_k,
        canonical_base_init=canonical_base_init,
        canonical_move_init=canonical_move_init,
        canonical_omega_c=omega_c,
        base_centroid=base_centroid,
    )


def partition_candidates(
    O_stack: torch.Tensor,
    M_attn_64: Optional[torch.Tensor] = None,
    sigma_b: float = 0.25,
    sigma_m: float = 0.15,
    tau_b: float = 0.05,
    tau_m: float = 0.05,
    mode: str = "footprint",
    M_attn_threshold: float = 0.3,
    M_attn_tau: float = 0.05,
    p_threshold: float = 0.5,
    occupancy_threshold: float = 0.5,
    eps: float = 1e-8,
):
    """Legacy signature wrapper for backward compatibility.

    v6 replaces the soft-field formulation with the count-based one; this
    wrapper calls :func:`count_based_partition` and returns the same
    6-tuple as the v5 signature (move_mask_k, base_mask_k, p_base_stub,
    p_move_stub, base_centroid, footprint). The ``p_base_stub`` /
    ``p_move_stub`` are binary-castable for the viz layer.
    """
    if M_attn_64 is None:
        # Fall back to old joint_free_split for rare callers without M_attn.
        from ..sajo.anchors import joint_free_split
        p_base, p_move = joint_free_split(O_stack, sigma_b=sigma_b,
                                           sigma_m=sigma_m, tau_b=tau_b,
                                           tau_m=tau_m, mode=mode)
        occ_bin = O_stack > occupancy_threshold
        move_mask_k = occ_bin & (p_move > p_threshold).unsqueeze(0)
        base_mask_k = occ_bin & (p_base > p_threshold).unsqueeze(0)
        footprint = (O_stack > occupancy_threshold).any(dim=0)
        # Centroid
        K, D, H, W = O_stack.shape
        coords = _voxel_coord_grid(D, O_stack.device, O_stack.dtype)
        w_sum = p_base.sum().clamp_min(eps)
        centroid = (p_base.unsqueeze(-1) * coords).sum(dim=(0, 1, 2)) / w_sum
        return move_mask_k, base_mask_k, p_base, p_move, centroid, footprint.to(O_stack.dtype)

    result = count_based_partition(
        O_stack, M_attn_64,
        m_attn_base_threshold=1.0 - M_attn_threshold,
        m_attn_move_threshold=M_attn_threshold,
        occupancy_threshold=occupancy_threshold,
    )
    # Synthesize p_base / p_move for viz compatibility
    p_base = result.true_base.to(O_stack.dtype)
    p_move = (result.shell | result.move_interior).to(O_stack.dtype)
    footprint = result.footprint.to(O_stack.dtype)
    return (result.move_mask_k, result.base_mask_k,
            p_base, p_move, result.base_centroid, footprint)

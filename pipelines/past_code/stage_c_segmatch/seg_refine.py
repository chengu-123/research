"""C.5 Motion-consistency segmentation refinement.

Given the current ``T_k`` from volumetric fit, recompute a per-voxel
motion-consistency score ``c(v) = (1/K) Σ_k O_k(T_k(v))`` — this is high
when the rigid-body hypothesis places ``v`` onto occupied voxels in
every state, low when the warped trajectory misses occupancy. Combined
with the v4.3 ``M_attn_64`` semantic prior as a graph-cut unary, this
either confirms the initial (SAJO-based) segmentation or flips voxels
whose motion prediction disagrees with the observed occupancy.

The resulting per-state hard masks ``move_mask_k``, ``base_mask_k`` are
fed back into volumetric_fit for the next outer iteration.

Uses the same PyMaxflow 2-label α-expansion as v3's graph_cut.py but
with the **motion-consistency data term** replacing the forward-warp
agreement term (which required already-correct T_k, chicken-and-egg).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import maxflow


@dataclass
class SegRefineResult:
    move_mask_k: torch.Tensor           # (K, D, H, W) bool
    base_mask_k: torch.Tensor           # (K, D, H, W) bool
    canonical_labels: torch.Tensor      # (D, H, W) int8 {-1=air, 0=base, 1=move}
    data_term: torch.Tensor             # (D, H, W) float
    n_flips: int                        # voxels whose canonical label changed
    gc_energy: float                    # min-cut value


def _logit(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.log(x.clamp(0.0, 1.0) + eps) - torch.log(1.0 - x.clamp(0.0, 1.0) + eps)


def compute_motion_consistency(
    O_stack: torch.Tensor,
    T_k: torch.Tensor,
    resolution: int = 64,
) -> torch.Tensor:
    """``c(v) = (1/K) Σ_k O_k(T_k(v))`` via trilinear sampling.

    **Retained for backward compat / viz**. The v8 graph-cut uses
    :func:`compute_base_move_residuals` (differential) instead.

    High c(v) means: "under current rigid hypothesis, canonical voxel v
    lands on occupied voxels in all K states" → this voxel IS part of the
    moving body. Low c(v) means: "the rigid hypothesis takes v to empty
    voxels in most states" → v is NOT moving (or hypothesis is wrong).
    """
    K = O_stack.shape[0]
    device = O_stack.device
    dtype = O_stack.dtype
    idx = torch.arange(resolution, device=device, dtype=dtype)
    ii, jj, kk = torch.meshgrid(idx, idx, idx, indexing="ij")
    p = torch.stack([ii, jj, kk], dim=-1) / float(resolution - 1) - 0.5
    p_h = torch.cat([p, torch.ones_like(p[..., :1])], dim=-1)                   # (D,H,W,4)

    agreement = torch.zeros((resolution, resolution, resolution),
                            device=device, dtype=dtype)
    for k in range(K):
        p_src = p_h @ T_k[k].to(device=device, dtype=dtype).T                   # (D,H,W,4)
        xyz = p_src[..., :3]
        grid = 2.0 * (xyz + 0.5) - 1.0
        grid_xyz = grid[..., [2, 1, 0]].unsqueeze(0)
        O_k = O_stack[k].unsqueeze(0).unsqueeze(0)
        sampled = F.grid_sample(O_k, grid_xyz, mode="bilinear",
                                padding_mode="zeros", align_corners=True)
        agreement = agreement + sampled.squeeze(0).squeeze(0)
    return agreement / float(K)


def compute_base_move_residuals(
    O_stack: torch.Tensor,
    T_k: torch.Tensor,
    resolution: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """**v8 CORE**. Per-voxel residuals under {static, rigid-move} hypotheses.

    Let O_0 be the canonical reference (T_0 = I by construction). For
    each canonical voxel v:

        r_base(v) = Σ_k (O_k(v)        - O_0(v))²      — static assumption
        r_move(v) = Σ_k (O_k(T_k(v))   - O_0(v))²      — rigid-body assumption

    Interpretation:
        * Cabinet voxel: O_k(v) = O_0(v) = 1 for all k → r_base ≈ 0.
          O_k(T_k(v)) maps to wherever cabinet ISN'T at state k → r_move > 0.
          (r_base - r_move < 0) → base label is cheaper → correct.
        * Drawer shell voxel: O_k(v) alternates with state → r_base > 0.
          If T_k correct, O_k(T_k(v)) = O_0(v) → r_move ≈ 0.
          (r_base - r_move > 0) → move label is cheaper → correct.
        * Drawer always_on (count=K, inside cabinet): O_k(v) = O_0(v) = 1
          for all k → r_base = 0. O_k(T_k(v)) ≈ 1 for correct T_k → r_move
          also ≈ 0. **Tie — Potts smoothness + semantic prior decides.**

    Complexity: one backward-warp per state (K trilerp calls).
    """
    K = O_stack.shape[0]
    device = O_stack.device
    dtype = O_stack.dtype

    O_0 = O_stack[0].to(device=device, dtype=dtype)

    # Base residual: direct L2 of O_k(v) vs O_0(v), summed over k.
    r_base = torch.zeros_like(O_0)
    for k in range(K):
        r_base = r_base + (O_stack[k].to(device=device, dtype=dtype) - O_0) ** 2

    # Move residual: O_k sampled at T_k(v), compared to O_0(v).
    idx = torch.arange(resolution, device=device, dtype=dtype)
    ii, jj, kk = torch.meshgrid(idx, idx, idx, indexing="ij")
    p = torch.stack([ii, jj, kk], dim=-1) / float(resolution - 1) - 0.5
    p_h = torch.cat([p, torch.ones_like(p[..., :1])], dim=-1)                   # (D,H,W,4)

    r_move = torch.zeros_like(O_0)
    for k in range(K):
        p_src = p_h @ T_k[k].to(device=device, dtype=dtype).T                   # (D,H,W,4)
        xyz = p_src[..., :3]
        grid = 2.0 * (xyz + 0.5) - 1.0
        grid_xyz = grid[..., [2, 1, 0]].unsqueeze(0)
        O_k_vol = O_stack[k].unsqueeze(0).unsqueeze(0)
        sampled = F.grid_sample(O_k_vol, grid_xyz, mode="bilinear",
                                padding_mode="zeros", align_corners=True)
        sampled = sampled.squeeze(0).squeeze(0)
        r_move = r_move + (sampled - O_0) ** 2

    return r_base, r_move


def _sarle_bimodality(vals: np.ndarray) -> float:
    """Sarle's bimodality coefficient.

    ``b = (g² + 1) / (k_excess + 3·(N-1)²/((N-2)·(N-3)))``

    where ``g`` is skewness and ``k_excess = kurtosis - 3`` is excess
    kurtosis. ``b > 5/9 ≈ 0.555`` indicates bimodality; ``b → 1`` for
    strongly separated double-peak distributions.
    """
    N = vals.size
    if N < 4:
        return 5.0 / 9.0
    mean = vals.mean()
    std = vals.std() + 1e-8
    g = ((vals - mean) ** 3).mean() / (std ** 3)
    k_raw = ((vals - mean) ** 4).mean() / (std ** 4)
    k_excess = k_raw - 3.0
    denom = k_excess + 3.0 * ((N - 1) ** 2) / ((N - 2) * (N - 3))
    return float((g * g + 1.0) / denom)


def run_graph_cut(
    M_attn_64: torch.Tensor,
    motion_consistency: torch.Tensor,
    active_mask: torch.Tensor,
    hp,
    persistence: Optional[torch.Tensor] = None,
    K: Optional[int] = None,
    dit_prior: Optional[object] = None,
) -> Tuple[torch.Tensor, float]:
    """2-label α-expansion graph-cut: ``canonical label ∈ {base, move}``.

    Unary (nats-ish, non-negative after shift):
        U_base(v) = -λ_attn · logit(M_attn) + λ_mot · (data_term - 0.5)
                    - λ_persist · (persist(v)/K - 0.5)
        U_move(v) = +λ_attn · logit(M_attn) - λ_mot · (data_term - 0.5)
                    + λ_persist · (persist(v)/K - 0.5)

    (High M_attn → prefer base; high motion consistency → prefer move;
     long persistence run → prefer base.)

    Pairwise: 6-connectivity Potts with ``λ_smooth``.
    Air voxels (``active_mask == False``) are forced to label 0 (base) by
    a big unary, then re-labeled -1 (unassigned) in the output.

    ``persistence`` is the per-voxel max-run-length from
    ``partition.persistence_run_length``. If supplied (with ``K``), it
    contributes an additional unary that pulls long-run voxels into base.
    Skipped when ``persistence is None`` or ``hp.lambda_persistence == 0``.
    """
    D, H, W = M_attn_64.shape
    device = M_attn_64.device

    if hp.lambda_smooth_adaptive:
        active_vals = M_attn_64[active_mask].detach().cpu().numpy()
        rho = _sarle_bimodality(active_vals) if active_vals.size > 0 else 5.0 / 9.0
        denom = max(rho - 5.0 / 9.0, 1.0e-3)
        lambda_smooth = float(hp.lambda_smooth / denom)
    else:
        lambda_smooth = float(hp.lambda_smooth)

    logit_attn = _logit(M_attn_64, eps=hp.logit_eps)
    # v8: hard clip to bound single-voxel attention contribution. Without this,
    # saturated M_attn (≈0 or ≈1) voxels contribute ±log(1/eps) to the unary
    # and can dominate scale-balance against motion + Potts terms by 10×.
    clip_val = float(getattr(hp, "logit_attn_clip", 4.0))
    if clip_val > 0.0:
        logit_attn = logit_attn.clamp(-clip_val, clip_val)

    # v8 CORE data term: `motion_consistency` is interpreted as the tuple
    # (r_base, r_move) residuals per voxel (packed differently depending
    # on entry path — see refine_segmentation). When it comes in as a
    # single scalar field (legacy path), we fall back to the single-sided
    # formulation. When it's the (r_base, r_move) tensor we use the
    # differential form.
    if motion_consistency.dim() == 4 and motion_consistency.shape[0] == 2:
        # v8 differential path: [0]=r_base, [1]=r_move
        r_base = motion_consistency[0]
        r_move = motion_consistency[1]
        # Scale-normalise residuals to be in [0, 1] roughly — divide by K (each
        # state contributes residual ∈ [0, 1]).
        K_f = float(max(K or 1, 1))
        r_base_n = (r_base / K_f).clamp(0.0, 1.0)
        r_move_n = (r_move / K_f).clamp(0.0, 1.0)
        # Unary is the DIFFERENTIAL:  label-local residual + attn bias.
        # Lower r_base  → base cheaper. Lower r_move  → move cheaper.
        U_base = hp.lambda_motion * r_base_n - hp.lambda_attn * logit_attn
        U_move = hp.lambda_motion * r_move_n + hp.lambda_attn * logit_attn
    else:
        # Legacy single-sided formulation (kept for non-v8 callers).
        U_base = -hp.lambda_attn * logit_attn + hp.lambda_motion * (motion_consistency - 0.5)
        U_move = +hp.lambda_attn * logit_attn - hp.lambda_motion * (motion_consistency - 0.5)

    lambda_persistence = float(getattr(hp, "lambda_persistence", 0.0))
    if persistence is not None and lambda_persistence > 0.0 and K is not None and K > 0:
        # Normalised persistence in [-0.5, 0.5]: high → base, low → move
        persist_signal = (persistence.to(M_attn_64.dtype) / float(K)) - 0.5
        U_base = U_base - lambda_persistence * persist_signal
        U_move = U_move + lambda_persistence * persist_signal

    # v8.1 AAAI-NOVELTY (2026-04-24): DiT 1024-dim hidden-state MRF priors.
    # Two independent signals derived from the SS-DiT mid-late block hidden
    # states at t≈0.3, captured in Stage B via forward hooks (see
    # pipelines/stage_b_scar.py:capture_dit_hidden_states and
    # pipelines/stage_c_segmatch/dit_prior.py). Active when
    # hp.use_dit_prior=True AND dit_hidden.pt exists.
    if dit_prior is not None:
        lambda_dit_proto = float(getattr(hp, "lambda_dit_proto", 0.0))
        lambda_dit_boundary = float(getattr(hp, "lambda_dit_boundary", 0.0))
        dit_clip = float(getattr(hp, "logit_attn_clip", 4.0))
        if lambda_dit_proto > 0.0:
            # logit of p_move_dit; clip to prevent saturation dominance
            p_m = dit_prior.p_move.to(M_attn_64.dtype).clamp(hp.logit_eps, 1.0 - hp.logit_eps)
            logit_dit = torch.log(p_m / (1.0 - p_m)).clamp(-dit_clip, dit_clip)
            # High logit_dit → prefer MOVE; mirror of M_attn sign convention
            U_base = U_base + lambda_dit_proto * logit_dit
            U_move = U_move - lambda_dit_proto * logit_dit
        if lambda_dit_boundary > 0.0:
            # s_boundary ∈ [0, 1]; high → articulation boundary → move-likely
            sb = dit_prior.s_boundary.to(M_attn_64.dtype).clamp(0.0, 1.0)
            sb_centered = 2.0 * sb - 1.0                      # [-1, +1]
            U_base = U_base + lambda_dit_boundary * sb_centered
            U_move = U_move - lambda_dit_boundary * sb_centered

    U_base_np = U_base.detach().cpu().numpy().astype(np.float64)
    U_move_np = U_move.detach().cpu().numpy().astype(np.float64)
    active_np = active_mask.detach().cpu().numpy()

    huge = 1.0e6
    U_base_np[~active_np] = 0.0
    U_move_np[~active_np] = huge

    shift = min(U_base_np.min(), U_move_np.min())
    if shift < 0:
        U_base_np = U_base_np - shift
        U_move_np = U_move_np - shift

    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes((D, H, W))
    # sourcecap = cost of label 1 (move); sinkcap = cost of label 0 (base)
    g.add_grid_tedges(nodes, U_move_np, U_base_np)

    structure6 = np.zeros((3, 3, 3), dtype=np.int32)
    structure6[2, 1, 1] = 1
    structure6[1, 2, 1] = 1
    structure6[1, 1, 2] = 1
    g.add_grid_edges(nodes, weights=lambda_smooth, structure=structure6,
                     symmetric=True)

    flow = g.maxflow()
    move_np = g.get_grid_segments(nodes)   # True = sink = move

    labels_np = np.zeros((D, H, W), dtype=np.int8)
    labels_np[move_np] = 1
    labels_np[~active_np] = -1

    labels = torch.from_numpy(labels_np).to(device)
    return labels, float(flow)


def refine_segmentation(
    O_stack: torch.Tensor,
    T_k: torch.Tensor,
    M_attn_64: torch.Tensor,
    prev_move_mask_k: torch.Tensor,
    hp,
    resolution: int = 64,
    persistence: Optional[torch.Tensor] = None,
    dit_prior: Optional[object] = None,
) -> SegRefineResult:
    """One pass of motion-consistency segmentation refinement.

    ``prev_move_mask_k`` is only used to count label flips for diagnostics;
    the new labels are independently derived from M_attn + motion
    consistency (+ optional persistence run-length, AOF Path).
    """
    K = O_stack.shape[0]
    # v8: compute differential residuals instead of single-sided consistency.
    # Stack into (2, D, H, W) so run_graph_cut detects the differential path.
    r_base, r_move = compute_base_move_residuals(O_stack, T_k, resolution=resolution)
    data_term = torch.stack([r_base, r_move], dim=0)

    active_mask = (O_stack > 0.5).float().mean(dim=0) > hp.active_thresh        # (D,H,W)
    labels, gc_energy = run_graph_cut(
        M_attn_64, data_term, active_mask, hp,
        persistence=persistence, K=K, dit_prior=dit_prior,
    )

    canonical_base = (labels == 0)
    canonical_move = (labels == 1)

    # Per-state hard masks: intersect canonical assignment with per-state occupancy
    occ_bin = (O_stack > 0.5)
    base_mask_k = occ_bin & canonical_base.unsqueeze(0)
    move_mask_k = occ_bin & canonical_move.unsqueeze(0)

    # Count flips: voxels that changed canonical label (move ↔ base) compared
    # to the previous iteration's canonical-union of move_mask_k
    prev_canonical_move = prev_move_mask_k.any(dim=0)
    flip_mask = prev_canonical_move != canonical_move
    # Only count flips where canonical label is actually assigned (not air)
    flip_mask = flip_mask & active_mask
    n_flips = int(flip_mask.sum().item())

    return SegRefineResult(
        move_mask_k=move_mask_k,
        base_mask_k=base_mask_k,
        canonical_labels=labels,
        data_term=data_term.detach(),
        n_flips=n_flips,
        gc_energy=gc_energy,
    )

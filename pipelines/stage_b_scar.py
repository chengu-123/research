"""Stage B driver: SCAR v3 sampling + optional SDEdit Pass-2 refinement.

Replaces `stage_b_vgcf.py` as the main driver for base-consistent Stage-1
sampling. The VGCF driver is preserved in the tree as a legacy ablation.

Pipeline (v3 default, 2026-04-18):
  Pass 1
    1. Run SCARSampler with symmetric mix (no push) for K-parallel rough
       alignment.
    2. Decode latents to 64^3 occupancy via SS VAE decoder.
    3. Optional: remove grounding disk/carpet voxels.
  Pass 2 (SDEdit refinement for state 0)
    4. Build augmented intersection guide in 64^3 prob space: open
       states' soft intersection UNION state 0's exclusive voxels.
    5. Encode guide to SS latent via the frozen SparseStructureEncoder.
    6. Re-sample K=6 batch via SDEdit from t_star=0.5 with per-state
       guides (state 0 uses z_guide, 1..K-1 use their Pass-1 latents).
    7. Replace state 0's latent with refined output; re-decode all K.
    8. Apply remove_disk again on the refined O_stack.
    9. Post-hoc rigid ICP is DISABLED by default (icp_enabled=false).
   10. Persist O_stack, diagnostics, per-step Tweedie decodes,
       visualizations.

See record/2026-04-18-stageb-v3-sdedit-design.md for the full design.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from pipelines.sajo.anchors import joint_free_split
from pipelines.utils.icp import align_to_reference
from pipelines.utils.postprocessing import remove_disk
from pipelines.utils.voxel_io import save_voxel_grid
from pipelines.utils.voxel_viz import (
    save_diagnostics_curves_html,
    save_two_masks_html,
    save_voxel_html,
    save_voxel_stack_html,
)


@dataclass
class SCARResult:
    O_stack: torch.Tensor           # (K, 64, 64, 64) binary
    O_stack_soft: torch.Tensor      # (K, 64, 64, 64) sigmoid outputs
    z_final: torch.Tensor           # (K, 8, 16, 16, 16)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    icp_offsets: List[np.ndarray] = field(default_factory=list)
    icp_capped: List[bool] = field(default_factory=list)


# ---------------------------------------------------------------------------
# v8 DiT 1024-dim hidden state capture (2026-04-24).
#
# TRELLIS's 8-dim ``z_final`` is a VAE-compressed bottleneck that loses most
# per-voxel DINOv2-aligned semantic content (128:1 compression). For
# training-free articulated part segmentation (Stage C), the SS-DiT's
# internal 1024-dim hidden state at mid-late blocks carries richer signals:
#   (i)   absolute voxel position via sinusoidal APE retained in residuals;
#   (ii)  local VAE-latent-driven embedding;
#   (iii) image-aligned semantics from 24 cross-attention blocks against
#         DINOv2 ViT-L/14 patch tokens (1369 patches + CLS + 4 register).
# Block 14-18 are the DIFT-style "semantic sweet spot" (mid-late layers
# where content has crystallised but isn't yet specialised for velocity
# output). Timestep t≈0.3 likewise balances "structure formed, features not
# yet velocity-specific" (cf. DIFT's t=261/1000 for SD).
#
# Cost: one extra SS-DiT forward pass (~150ms on A100 at K=6). Storage:
# ~50MB fp16 per block of (K=6, 4096 tokens, 1024 dim).
# ---------------------------------------------------------------------------


@torch.no_grad()
def capture_dit_hidden_states(
    flow_model,
    z_final: torch.Tensor,
    cond,
    target_blocks: List[int] = (14, 16, 18),
    t_star: float = 0.3,
    sigma_min: float = 1.0e-5,
    seed: int = 0,
    save_dtype: torch.dtype = torch.float16,
) -> Dict[int, torch.Tensor]:
    """Capture SS-DiT hidden states at specified mid-late blocks at t=t_star.

    We construct a synthetic noisy latent ``x_t = (1-t)·z_final + σ(t)·ε``
    matching the flow-matching forward process, run a single forward pass,
    and read off per-block hidden states via forward hooks on
    ``flow_model.blocks[idx]``. Hooks are removed before returning.

    Parameters
    ----------
    flow_model : ``SparseStructureFlowModel`` (TRELLIS SS-DiT).
    z_final : ``(K, C, D, H, W)`` — the clean Pass-1 latent.
    cond : DINOv2 conditioning; same type as what the sampler feeds
        to ``_inference_model``. For TrellisImageTo3DPipeline this is
        the prenorm patch-token tensor ``(1, 1374, 1024)`` or similar.
    target_blocks : list[int] — block indices to hook. Default (14, 16, 18).
    t_star : float in (0, 1). t=0 is clean, t=1 is pure noise. 0.3 is
        the DIFT-analogous semantic sweet spot.
    sigma_min : forward-process noise floor (matches FlowEulerSampler).
    seed : deterministic noise seed.
    save_dtype : fp16 halves memory; default for storage.

    Returns
    -------
    dict ``{block_idx: (K, L, C)}`` on CPU in save_dtype, where L is the
    token-space length (flattened 16³ = 4096 for SS-DiT) and C is the
    model's hidden dim (1024 for the L-size config).
    """
    from collections import defaultdict

    device = z_final.device
    dtype = z_final.dtype

    store: Dict[int, List[torch.Tensor]] = defaultdict(list)
    handles = []

    for idx in target_blocks:
        if idx < 0 or idx >= len(flow_model.blocks):
            continue

        def make_hook(i):
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    tensor_out = output[0]
                else:
                    tensor_out = output
                store[i].append(tensor_out.detach().to(save_dtype).cpu())
            return hook

        handles.append(flow_model.blocks[idx].register_forward_hook(make_hook(idx)))

    # Construct synthetic noisy latent at t_star. Matches FlowEulerSampler's
    # forward diffusion: x_t = (1-t)·x_0 + (σ_min + (1-σ_min)·t)·ε.
    K = z_final.shape[0]
    noise_gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(z_final.shape, generator=noise_gen, device=device, dtype=dtype)
    sigma_t = sigma_min + (1.0 - sigma_min) * t_star
    x_t = (1.0 - t_star) * z_final + sigma_t * noise

    # Timestep tensor — SS-DiT expects t scaled by 1000 (see flow_euler.py:39).
    t_tensor = torch.full((K,), 1000.0 * t_star, device=device, dtype=torch.float32)

    try:
        _ = flow_model(x_t, t_tensor, cond)
    except TypeError:
        # Some flow model wrappers take kwargs; attempt common alternates.
        _ = flow_model(x_t, t=t_tensor, cond=cond)

    for h in handles:
        h.remove()

    result: Dict[int, torch.Tensor] = {}
    for idx in target_blocks:
        if idx not in store or not store[idx]:
            continue
        result[idx] = store[idx][0]  # (K, L, C) fp16 cpu
    return result


def _compute_intersection_mask(
    O_stack: torch.Tensor, vote_threshold: float = 0.83,
) -> np.ndarray:
    """Base mask = voxels occupied in >= vote_threshold fraction of states.

    0.83 with K=6 means "occupied in >= 5 of 6 states".
    """
    K = O_stack.shape[0]
    min_votes = int(np.ceil(vote_threshold * K))
    votes = (O_stack > 0.5).sum(dim=0)
    return (votes >= min_votes).cpu().numpy().astype(bool)


def _compute_P_base_shared(
    soft_p1: torch.Tensor,
    exclude_state_0: bool = False,
) -> torch.Tensor:
    """Compute the shared Pass-2 base probability field.

    v4 canonical form: ``P_base_shared = mean_{k=0..K-1} P^(k)`` — mean across
    ALL K states (including state 0). Even though state 0 has a TRELLIS
    closed-state shrink bias, with K=6 its contribution is 1/K of the mean, so
    at cabinet outer-shell voxels the mean stays well above 0.5 (e.g. state 0
    at 0.2 + five others at 0.9 → mean 0.78). Including state 0 also preserves
    its real-image contribution at voxels where it is correct.

    ``exclude_state_0=True`` reverts to v4-draft behaviour (mean over s1-5); kept
    as an ablation knob for the paper.
    """
    if soft_p1.ndim != 4:
        raise ValueError(f"soft_p1 must be (K, D, H, W); got {tuple(soft_p1.shape)}")
    K = soft_p1.shape[0]
    if exclude_state_0 and K < 2:
        raise ValueError(f"exclude_state_0 requires K >= 2, got K={K}")
    P_for_base = soft_p1[1:] if exclude_state_0 else soft_p1
    return P_for_base.mean(dim=0)                                       # (64, 64, 64)


def _compute_M_base_tokenspace(
    soft_p1: Optional[torch.Tensor] = None,
    tau_M: float = 0.05,
    token_resolution: int = 16,
    exclude_state_0: bool = False,
    P_base_shared: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build Stage B v4 BMCSA spatial gate mask in DiT token space.

    v4 canonical: use ``mean_{k=0..K-1} P^(k)`` (all K, with state 0). See
    :func:`_compute_P_base_shared`. The resulting 64^3 probability mask is
    sigmoid-softened around 0.5 with sharpness ``tau_M`` and avg-pooled to
    DiT token resolution (kernel = 64/token_resolution).

    Parameters
    ----------
    soft_p1 : torch.Tensor, optional
        Shape ``(K, 64, 64, 64)``. Used to compute ``P_base_shared`` when the
        latter is not passed directly. Ignored if ``P_base_shared`` is given.
    tau_M : float, default 0.05
        Sigmoid sharpness. Smaller = sharper transition around 0.5.
    token_resolution : int, default 16
        Target DiT token-spatial resolution per axis. For TRELLIS-image-large
        (res=16, patch=1): token_resolution=16, L=4096, pool_kernel=4.
    exclude_state_0 : bool, default False
        Passed through to :func:`_compute_P_base_shared` when computing from
        ``soft_p1``. ``False`` (canonical v4) uses all K states. Ignored if
        ``P_base_shared`` is supplied directly.
    P_base_shared : torch.Tensor, optional
        Precomputed shared base field, shape ``(64, 64, 64)``. When provided,
        this is used directly and ``soft_p1``/``exclude_state_0`` are ignored.
        Preferred in v4 so the same shared field is reused by the guide
        builder without recomputing.

    Returns
    -------
    torch.Tensor
        Shape ``(1, L, 1)`` where ``L = token_resolution^3``.
    """
    if P_base_shared is None:
        if soft_p1 is None:
            raise ValueError("must provide either soft_p1 or P_base_shared")
        P_base_shared = _compute_P_base_shared(soft_p1, exclude_state_0=exclude_state_0)

    # Soft mask around 0.5 via sigmoid((P - 0.5) / tau_M).
    M_base_64 = torch.sigmoid((P_base_shared - 0.5) / float(tau_M))     # (64, 64, 64)

    src_res = M_base_64.shape[-1]
    if src_res % token_resolution != 0:
        raise ValueError(
            f"token_resolution {token_resolution} does not divide source resolution {src_res}"
        )
    pool_kernel = src_res // token_resolution

    # (1, 1, 64, 64, 64) → avg_pool3d → (1, 1, 8, 8, 8)
    M_tok = torch.nn.functional.avg_pool3d(
        M_base_64.unsqueeze(0).unsqueeze(0),
        kernel_size=pool_kernel,
        stride=pool_kernel,
    )

    # Flatten to token order: DiT patchify then .view(...).permute(0, 2, 1)
    # results in tokens ordered as row-major flattening of (D, H, W). avg_pool3d
    # preserves this order, so .view(1, L, 1) directly matches.
    L = token_resolution ** 3
    M_flat = M_tok.view(1, L, 1)
    return M_flat


def _compute_M_attn_tokenspace(
    z_final: torch.Tensor,
    threshold: float = 0.5,
    tau: float = 0.1,
    return_raw_agreement: bool = False,
) -> torch.Tensor:
    """Plan 3 (Stage B v4.2): attention-map-inspired dynamic M_attn from
    per-voxel cross-state feature agreement.

    Distinguishes "cabinet shrunk shell" (all K states share similar semantic
    features at boundary voxels, high agreement) from "drawer trajectory
    middle" (state 0's air feature vs states 1-5's drawer feature, low
    agreement) using PURE latent-space similarity. The geometric M_base and
    this semantic M_attn are multiplied at the BMCSA blend site -- a voxel
    must pass BOTH geometric and semantic consensus to be treated as base.

    Algorithm (per token ℓ):
        1. Collect per-state latent vectors: h_k(ℓ) ∈ R^C where C = latent_channels
           (= 8 for TRELLIS-image-large SS flow).
        2. L2-normalise along channel dim: ĥ_k(ℓ) = h_k / ||h_k||.
        3. Pairwise cosine similarity, averaged over k ≠ k':
               agree(ℓ) = 1/(K·(K-1)) · Σ_{k,k' : k ≠ k'} <ĥ_k(ℓ), ĥ_{k'}(ℓ)>
        4. M_attn(ℓ) = sigmoid((agree(ℓ) - threshold) / tau).

    Parameters
    ----------
    z_final : torch.Tensor
        Shape ``(K, C, D, H, W)``, the Pass 1 SS latent at token resolution
        (TRELLIS-image-large: (K, 8, 16, 16, 16)).
    threshold : float, default 0.5
        Sigmoid center. Tokens with ``agree > threshold`` get M_attn > 0.5;
        below, M_attn < 0.5. 0.5 is a neutral choice; dataset-specific tuning
        may yield better results.
    tau : float, default 0.1
        Sigmoid sharpness. Smaller tau ⇒ sharper transition.
    return_raw_agreement : bool, default False
        If True, also returns the raw agreement score tensor shape (L,) for viz.

    Returns
    -------
    M_attn : torch.Tensor
        Shape ``(1, L, 1)`` where ``L = D × H × W``. Broadcasts against DiT
        self-attention outputs ``(K, L, D_feat)`` via the L axis.
    raw_agreement : torch.Tensor, optional
        Shape ``(L,)``, raw cosine-similarity agreement scores (before sigmoid).
    """
    if z_final.ndim != 5:
        raise ValueError(
            f"z_final must be (K, C, D, H, W); got shape {tuple(z_final.shape)}"
        )
    K, C, D, H, W = z_final.shape
    if K < 2:
        raise ValueError(f"M_attn requires K >= 2 for cross-state comparison; got K={K}")
    L = D * H * W

    # Reshape to per-token view: (K, L, C)
    z_tok = z_final.view(K, C, L).permute(0, 2, 1).contiguous()        # (K, L, C)

    # L2 normalise along channel.
    z_norm = torch.nn.functional.normalize(z_tok, dim=-1)              # (K, L, C)

    # Pairwise cosine similarity across states, averaged over off-diagonal pairs.
    # For efficiency: cat over token dim, compute (K, K) pairwise per token,
    # then average. Shape of sim: (L, K, K).
    # Use einsum: z_norm: (K, L, C), z_norm: (K, L, C) -> (K, K, L)
    sim = torch.einsum("klc,mlc->kml", z_norm, z_norm)                 # (K, K, L)
    sim_lkm = sim.permute(2, 0, 1)                                      # (L, K, K)

    # Off-diagonal mask (K, K): 1 for k != k'
    off_diag = (1.0 - torch.eye(K, device=z_norm.device, dtype=z_norm.dtype))
    n_pairs = K * (K - 1)                                              # denominator
    agree = (sim_lkm * off_diag).sum(dim=(-2, -1)) / n_pairs          # (L,), in [-1, 1]

    # Sigmoid to get M_attn in [0, 1].
    M_attn = torch.sigmoid((agree - float(threshold)) / float(tau))    # (L,)
    M_attn_flat = M_attn.view(1, L, 1)                                 # broadcast-ready

    if return_raw_agreement:
        return M_attn_flat, agree
    return M_attn_flat


def _build_augmented_intersection_guide(
    P_stack: torch.Tensor,
    encoder: torch.nn.Module,
    target_state: int = 0,
    guide_mode: str = "augmented_intersection",
    P_base_shared: Optional[torch.Tensor] = None,
    return_intermediates: bool = False,
) -> Any:
    """Construct the SDEdit guide latent for ``target_state`` refinement.

    Operates in 64^3 probability space to find voxels that the OTHER states
    (``k != target_state``) agree are occupied (``P_base``), plus the target
    state's voxels that no other state sees (``P_excl``). Their union is
    binarised and encoded back to SS latent via the frozen
    SparseStructureEncoder.

    This generalises the state-0 refinement to any endpoint. For
    ``target_state=0`` (closed), the guide enlarges the TRELLIS-shrunk
    cabinet while preserving the closed drawer. For ``target_state=K-1``
    (fully open), the reference pool symmetrically uses all non-K-1 states
    and preserves state K-1's extended-drawer voxels.

    Parameters
    ----------
    P_stack : torch.Tensor
        Shape ``(K, 64, 64, 64)``, sigmoid-decoded Pass-1 occupancies.
        Recommended to be post-``remove_disk`` so the guide does not inherit
        carpet voxels.
    encoder : torch.nn.Module
        ``pipe.models['sparse_structure_encoder']``, frozen.
    target_state : int
        Which state index to build the guide FOR. The remaining K-1 states
        serve as the reference pool (``P_open`` in the state-0 case). Negative
        indices are normalised (``-1 -> K-1``).
    guide_mode : str
        - ``"augmented_intersection"`` (default): union of non-target soft
          intersection and target-exclusive voxels. Preserves the target
          state's unique features (e.g. closed drawer for state 0, fully
          extended drawer for state K-1).
        - ``"pure_intersection"``: non-target intersection only. Target's
          unique voxels are dropped; SDEdit must regenerate them from c_k.

    Returns
    -------
    torch.Tensor
        Shape ``(1, 8, 16, 16, 16)``, the encoded guide latent.
    """
    if P_stack.ndim != 4:
        raise ValueError(f"P_stack must be (K, D, H, W); got shape {tuple(P_stack.shape)}")
    K = P_stack.shape[0]
    if K < 3:
        raise ValueError(f"guide builder requires K >= 3, got K={K}")

    # Normalise negative target indices.
    if target_state < 0:
        target_state = K + target_state
    if not (0 <= target_state < K):
        raise ValueError(f"target_state {target_state} out of range for K={K}")

    # Reference pool = all states except target (used for MAX in P_excl).
    ref_idx = [k for k in range(K) if k != target_state]
    P_ref = P_stack[ref_idx]                        # (K-1, 64, 64, 64)

    # P_base: if caller provides a shared base (v4 BMCSA mode), use it uniformly
    # across all targets. Otherwise fall back to per-target soft intersection
    # over the non-target reference pool (v3 legacy).
    if P_base_shared is not None:
        if P_base_shared.shape != P_stack.shape[1:]:
            raise ValueError(
                f"P_base_shared shape {tuple(P_base_shared.shape)} must match "
                f"occupancy spatial shape {tuple(P_stack.shape[1:])}"
            )
        P_base = P_base_shared.to(device=P_stack.device, dtype=P_stack.dtype)
    else:
        P_base = P_ref.min(dim=0).values            # v3 legacy: per-target MIN

    if guide_mode == "augmented_intersection":
        P_max_ref = P_ref.max(dim=0).values
        P_excl = torch.clamp(P_stack[target_state] - P_max_ref, min=0.0)
        P_guide = torch.maximum(P_base, P_excl)
    elif guide_mode == "pure_intersection":
        P_guide = P_base
    else:
        raise ValueError(
            f"unknown guide_mode {guide_mode!r}; expected 'augmented_intersection' or 'pure_intersection'"
        )

    # Binarise to match encoder training distribution (Objaverse binary
    # occupancy). Shape (64, 64, 64) -> (1, 1, 64, 64, 64) for encoder batch.
    O_guide_bin = (P_guide > 0.5).to(dtype=torch.float32)
    O_guide = O_guide_bin.unsqueeze(0).unsqueeze(0)

    # Encoder can be fp16; cast input to its parameter dtype.
    enc_dtype = next(encoder.parameters()).dtype
    O_guide = O_guide.to(device=next(encoder.parameters()).device, dtype=enc_dtype)

    with torch.no_grad():
        z_guide = encoder(O_guide)                  # mean branch, (1, 8, 16, 16, 16)

    if return_intermediates:
        # Return both the latent guide and the prob-space fields used to build
        # it. P_base here is either the shared one (v4) or the per-target MIN
        # (v3 legacy); in v4 every target returns an identical P_base snapshot.
        if guide_mode == "augmented_intersection":
            P_max_ref = P_ref.max(dim=0).values
            P_excl_viz = torch.clamp(P_stack[target_state] - P_max_ref, min=0.0)
        else:
            P_excl_viz = torch.zeros_like(P_base)
        intermediates = {
            "P_base": P_base.detach().cpu(),
            "P_excl": P_excl_viz.detach().cpu(),
            "P_guide": P_guide.detach().cpu(),
            "O_guide_bin": O_guide_bin.detach().cpu(),
            "target_state": int(target_state),
            "P_base_source": "shared" if P_base_shared is not None else "per_target_min",
        }
        return z_guide, intermediates
    return z_guide


def _sdedit_refine_k6(
    sampler: Any,
    flow_model: torch.nn.Module,
    guides: torch.Tensor,
    cond: torch.Tensor,
    neg_cond: torch.Tensor,
    t_star: float,
    total_steps: int,
    rescale_t: float,
    cfg_strength: float,
    cfg_interval: tuple,
    shared_noise: Optional[torch.Tensor] = None,
    verbose: bool = True,
    collect_per_step: bool = False,
) -> Any:
    """Run a K-parallel SDEdit refinement pass with optional shared-noise init.

    Given per-state guide latents ``guides`` (shape ``(K, 8, 16, 16, 16)``),
    add Gaussian noise at rectified-flow level ``t_star``, then denoise from
    ``t_star`` to 0 using the sampler's inherited ``sample_once`` (which the
    GuidanceIntervalSamplerMixin upgrades with CFG + interval).

    All per-state mechanics (mix / push) are bypassed: we call
    ``sampler.sample_once`` directly, inheriting only the vanilla Euler step +
    CFG-interval guidance.

    Parameters
    ----------
    sampler : SCARSampler (or any FlowEulerGuidanceIntervalSampler subclass)
        The same sampler instance used in Pass 1. Its ``sigma_min`` is reused;
        its SCAR-specific ``mix`` and ``push`` branches are not invoked because
        we call ``sample_once`` directly.
    flow_model : torch.nn.Module
        ``pipe.models['sparse_structure_flow_model']``.
    guides : torch.Tensor
        Shape ``(K, 8, 16, 16, 16)``; per-state guide latents. State 0 should
        carry the augmented intersection guide; states k>=1 may carry their
        Pass-1 final latent (identity-like).
    cond, neg_cond : torch.Tensor
        Shape ``(K, N_tok, d)``; DINOv2 features used by CFG.
    t_star : float
        SDEdit noise level in (0, 1). 0.5 balances guide geometry and
        c_k-driven refinement.
    total_steps : int
        Full schedule length (= Pass 1's step count). Pass 2 uses
        ``ceil(total_steps * t_star)`` steps from ``t_star`` to 0.
    rescale_t, cfg_strength, cfg_interval
        Inherited from Pass 1's sampler params.
    shared_noise : torch.Tensor, optional
        Shape ``(K, 8, 16, 16, 16)`` or broadcastable to that shape. When
        provided, reused as the SDEdit noise term instead of sampling fresh
        Gaussian. This is the recommended behavior for multi-state consistency:
        Pass 2 should share the K-parallel initialization noise that Pass 1
        used, so state 0's refined trajectory stays in the same "random family"
        as states 1..K-1 (which are kept from Pass 1). When None, fresh
        ``torch.randn_like(guides)`` is used (legacy SDEdit behavior).
    verbose : bool
        Whether to print a tqdm progress bar.
    collect_per_step : bool
        When True, the return value is an ``edict`` with keys ``samples``,
        ``pred_x_t`` (list of intermediate x_{t-dt}), ``pred_x_0`` (list of
        Tweedie estimates), and ``t_seq`` (the rescaled t schedule, length
        n_pass2+1). When False (default), returns only the final sample
        tensor for API compatibility with legacy callers.

    Returns
    -------
    torch.Tensor or edict
        Final refined latents ``(K, 8, 16, 16, 16)``, or a per-step packet
        when ``collect_per_step=True``. Downstream callers typically use
        only the refined entries (state 0 / state K-1); other entries are
        approximately identity of their guides (modulo SDEdit noise).
    """
    from easydict import EasyDict as edict
    if not (0.0 < t_star < 1.0):
        raise ValueError(f"t_star must be in (0, 1); got {t_star}")
    sigma_min = sampler.sigma_min

    # Forward-noise to t_star. Use shared noise when provided to keep Pass 2
    # on the same random-seed family as Pass 1's K-parallel trajectory.
    if shared_noise is None:
        noise_term = torch.randn_like(guides)
    else:
        if shared_noise.shape != guides.shape:
            # Allow broadcasting from (1, C, D, H, W) -> (K, C, D, H, W).
            noise_term = shared_noise.expand_as(guides).contiguous()
        else:
            noise_term = shared_noise
        noise_term = noise_term.to(device=guides.device, dtype=guides.dtype)
    x_t = (1.0 - t_star) * guides + (sigma_min + (1.0 - sigma_min) * t_star) * noise_term

    # Build t-schedule from t_star to 0, mirroring the parent sampler's
    # rescale_t transform.
    n_pass2 = max(1, int(np.ceil(total_steps * t_star)))
    t_seq = np.linspace(float(t_star), 0.0, n_pass2 + 1)
    t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
    t_pairs = list(zip(t_seq[:-1], t_seq[1:]))

    from tqdm import tqdm
    pbar = tqdm(t_pairs, desc=f"SDEdit refine (t*={t_star:.2f})", disable=not verbose)

    pred_x_0_list: List[torch.Tensor] = []
    pred_x_t_list: List[torch.Tensor] = []

    sample = x_t
    for t, t_prev in pbar:
        t = float(t)
        t_prev = float(t_prev)
        out = sampler.sample_once(
            flow_model, sample, t, t_prev,
            cond=cond, neg_cond=neg_cond,
            cfg_strength=cfg_strength, cfg_interval=cfg_interval,
        )
        sample = out.pred_x_prev
        if collect_per_step:
            pred_x_0_list.append(out.pred_x_0.detach())
            pred_x_t_list.append(sample.detach())

    if collect_per_step:
        return edict({
            "samples": sample,
            "pred_x_t": pred_x_t_list,
            "pred_x_0": pred_x_0_list,
            "t_seq": t_seq.tolist() if hasattr(t_seq, "tolist") else list(t_seq),
        })
    return sample


def _sdedit_refine_k6_bmcsa(
    sampler: Any,
    flow_model: torch.nn.Module,
    guides: torch.Tensor,
    cond: torch.Tensor,
    neg_cond: torch.Tensor,
    t_star: float,
    pass2_steps: int,
    rescale_t: float,
    cfg_strength: float,
    cfg_interval: tuple,
    M_base: Optional[torch.Tensor] = None,
    bmcsa_blocks: Optional[List[int]] = None,
    bmcsa_strength: float = 1.0,
    shared_noise: Optional[torch.Tensor] = None,
    M_attn: Optional[torch.Tensor] = None,
    verbose: bool = True,
    collect_per_step: bool = False,
    M_compute_mode: str = "static",
    tau_M_dynamic: float = 0.7,
    kappa_M_dynamic: float = 0.05,
    capture_dynamic_M: bool = False,
) -> Any:
    """Pass 2 SDEdit refinement with Base-Masked Cross-State Attention (BMCSA).

    Generalises ``_sdedit_refine_k6`` by threading BMCSA kwargs through each
    call to ``sampler.sample_once``, which in turn passes them into
    ``model(x_t, t, cond, **kwargs)``. The edited DiT block
    (``ModulatedTransformerCrossBlock._forward``) consumes ``bmcsa_flag``,
    ``bmcsa_blocks``, ``bmcsa_strength`` and the M-mask described below.

    Two M-computation modes:

    - ``M_compute_mode='static'`` (v4.3 legacy): pass in ``M_base`` of shape
      ``(1, L, 1)`` precomputed from Pass-1 ``P_base_shared``. Same M used at
      every Euler step and every block.

    - ``M_compute_mode='dynamic'`` (v3.3 default): each block computes its
      own M from the current modulated hidden state h via pairwise
      cross-state cosine agreement (per-token). ``M_base`` is ignored. Uses
      ``tau_M_dynamic`` and ``kappa_M_dynamic`` to control the sigmoid gate.

    When ``capture_dynamic_M=True`` (dynamic mode only) the per-step
    per-block M maps are returned in the result edict under ``dynamic_M``.

    Parameters
    ----------
    sampler, flow_model, guides, cond, neg_cond, rescale_t, cfg_strength,
    cfg_interval, shared_noise, verbose, collect_per_step
        Same meaning as in ``_sdedit_refine_k6``.
    t_star : float
        SDEdit noise level (0 < t_star < 1).
    pass2_steps : int
        Fixed Euler step count for Pass 2, independent of ``t_star``.
    M_base : torch.Tensor or None
        Required when ``M_compute_mode='static'``. Shape ``(1, L, 1)``.
    bmcsa_blocks : list of int, optional
        DiT block indices to apply BMCSA on. None means all blocks.
    bmcsa_strength : float, default 1.0
        Overall multiplier on the shared-attention branch.
    M_compute_mode : str
        'static' or 'dynamic'. See above.
    tau_M_dynamic : float, default 0.6
        Sigmoid center for dynamic-M (cosine agreement threshold).
    kappa_M_dynamic : float, default 0.1
        Sigmoid sharpness for dynamic-M.
    capture_dynamic_M : bool, default False
        When True (and dynamic mode), each step's per-block M map is captured
        for diagnostics and returned in the result edict.

    Returns
    -------
    torch.Tensor or edict
        ``(K, 8, 16, 16, 16)`` refined latents, or an edict with per-step
        trajectories (and ``dynamic_M`` log) when ``collect_per_step=True``.
    """
    if not (0.0 < t_star < 1.0):
        raise ValueError(f"t_star must be in (0, 1); got {t_star}")
    if pass2_steps < 1:
        raise ValueError(f"pass2_steps must be >= 1; got {pass2_steps}")
    if M_compute_mode not in ("static", "dynamic"):
        raise ValueError(
            f"M_compute_mode must be 'static' or 'dynamic'; got {M_compute_mode!r}"
        )
    if M_compute_mode == "static" and M_base is None:
        raise ValueError("M_compute_mode='static' requires M_base to be provided")
    sigma_min = sampler.sigma_min

    # Forward-noise to t_star.
    if shared_noise is None:
        noise_term = torch.randn_like(guides)
    else:
        if shared_noise.shape != guides.shape:
            noise_term = shared_noise.expand_as(guides).contiguous()
        else:
            noise_term = shared_noise
        noise_term = noise_term.to(device=guides.device, dtype=guides.dtype)
    x_t = (1.0 - t_star) * guides + (sigma_min + (1.0 - sigma_min) * t_star) * noise_term

    # Fixed-length schedule.
    t_seq = np.linspace(float(t_star), 0.0, pass2_steps + 1)
    t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
    t_pairs = list(zip(t_seq[:-1], t_seq[1:]))

    bmcsa_blocks_set: Optional[set] = None
    if bmcsa_blocks is not None:
        bmcsa_blocks_set = set(int(i) for i in bmcsa_blocks)

    from easydict import EasyDict as edict

    # Base BMCSA kwargs (constant across steps).
    base_kwargs = dict(
        bmcsa_flag=True,
        bmcsa_blocks=bmcsa_blocks_set,
        bmcsa_strength=float(bmcsa_strength),
        M_compute_mode=M_compute_mode,
    )
    if M_compute_mode == "static":
        base_kwargs["M_base"] = M_base.to(device=guides.device)
    else:
        base_kwargs["tau_M_dynamic"] = float(tau_M_dynamic)
        base_kwargs["kappa_M_dynamic"] = float(kappa_M_dynamic)
    if M_attn is not None:
        base_kwargs["M_attn"] = M_attn.to(device=guides.device)

    # Dynamic-M log: dict {step_idx: {block_idx: (L,) fp16}}; only populated when
    # M_compute_mode='dynamic' and capture_dynamic_M=True.
    dynamic_M_log: Dict[int, Dict[int, torch.Tensor]] = {}

    from tqdm import tqdm
    pbar = tqdm(
        t_pairs,
        desc=(
            f"SDEdit+BMCSA refine (t*={t_star:.2f}, steps={pass2_steps}, "
            f"M_mode={M_compute_mode})"
        ),
        disable=not verbose,
    )

    pred_x_0_list: List[torch.Tensor] = []
    pred_x_t_list: List[torch.Tensor] = []
    sample = x_t
    for step_idx, (t, t_prev) in enumerate(pbar):
        t = float(t)
        t_prev = float(t_prev)

        # Per-step capture hook for dynamic-M.
        step_kwargs = dict(base_kwargs)
        if M_compute_mode == "dynamic" and capture_dynamic_M:
            this_step_log: Dict[int, torch.Tensor] = {}
            step_kwargs["dynamic_M_log"] = this_step_log
        else:
            this_step_log = None

        out = sampler.sample_once(
            flow_model, sample, t, t_prev,
            cond=cond, neg_cond=neg_cond,
            cfg_strength=cfg_strength, cfg_interval=cfg_interval,
            **step_kwargs,
        )
        sample = out.pred_x_prev
        if collect_per_step:
            pred_x_0_list.append(out.pred_x_0.detach())
            pred_x_t_list.append(sample.detach())
        if this_step_log is not None and len(this_step_log) > 0:
            dynamic_M_log[step_idx] = this_step_log

    if collect_per_step:
        return edict({
            "samples": sample,
            "pred_x_t": pred_x_t_list,
            "pred_x_0": pred_x_0_list,
            "t_seq": t_seq.tolist() if hasattr(t_seq, "tolist") else list(t_seq),
            "dynamic_M": dynamic_M_log,
        })
    return sample


def run_scar(
    pipe: Any,
    cond: Dict[str, torch.Tensor],
    K: int,
    out_dir: str,
    cfg_scar: Dict[str, Any],
    seed: int = 0,
    remove_disk_flag: bool = True,
    device: Optional[str] = None,
    cfg_sdedit: Optional[Dict[str, Any]] = None,
) -> SCARResult:
    """Run Stage B v3: SCAR Pass-1 sampling + SDEdit Pass-2 state-0 refinement.

    Parameters
    ----------
    pipe : TrellisImageTo3DPipeline
        Must have ``pipe.sparse_structure_sampler`` set to a
        :class:`SCARSampler` instance. For Pass 2, it must also expose
        ``pipe.models['sparse_structure_encoder']``.
    cond : dict
        ``{'cond': (K, N_tok, d), 'neg_cond': (K, N_tok, d)}``.
    K : int
        Number of articulation states.
    out_dir : str
        Destination directory.
    cfg_scar : dict
        Pass-1 config (legacy SCAR keys: ``alpha_schedule``, ``icp_enabled``,
        etc.). Push is expected to be disabled via ``alpha_peak=0`` at this
        point in the pipeline.
    seed : int
        Shared-noise seed.
    remove_disk_flag : bool
        Whether to run ``remove_disk`` post-decoding (applied both after
        Pass 1 and after Pass 2).
    cfg_sdedit : dict or None
        Pass-2 SDEdit config. Keys:
          - ``enabled`` (bool, default True): when False, Pass 2 is skipped.
          - ``t_star`` (float, default 0.5): SDEdit noise level.
          - ``guide_mode`` (str, default 'augmented_intersection').
        When None, behaves as if disabled (Pass 1 only).

    Returns
    -------
    SCARResult
    """
    os.makedirs(out_dir, exist_ok=True)
    if device is None:
        device = cond["cond"].device
    device = torch.device(device)

    flow_model = pipe.models["sparse_structure_flow_model"]
    decoder = pipe.models["sparse_structure_decoder"]

    gen = torch.Generator(device=device).manual_seed(int(seed))
    eps = torch.randn(
        (1, 8, flow_model.resolution, flow_model.resolution, flow_model.resolution),
        device=device, generator=gen,
    )
    noise = eps.repeat(K, 1, 1, 1, 1)

    sampler_params: Dict[str, Any] = dict(pipe.sparse_structure_sampler_params)
    # Override step count from scar config (TRELLIS-image-large default is 25;
    # spec uses 12 per methods.md section Hyperparameter Summary).
    if "steps" in cfg_scar:
        sampler_params["steps"] = int(cfg_scar["steps"])
    sampler = pipe.sparse_structure_sampler
    out = sampler.sample(
        flow_model, noise,
        cond=cond["cond"], neg_cond=cond["neg_cond"],
        verbose=True,
        **sampler_params,
    )
    z_final = out.samples

    # ---- Pass 1 decode ----
    logits_p1 = decoder(z_final)
    soft_p1 = torch.sigmoid(logits_p1).squeeze(1)
    binary_p1 = (soft_p1 > 0.5).to(soft_p1.dtype)

    if remove_disk_flag:
        binary_np = binary_p1.detach().cpu().numpy()
        voxels_5d = binary_np[:, None].astype(np.float32)
        voxels_5d = remove_disk(voxels_5d)
        binary_p1 = torch.from_numpy(voxels_5d[:, 0]).to(
            device=soft_p1.device, dtype=soft_p1.dtype,
        )
        soft_p1 = soft_p1 * binary_p1

    # ---- Pass 2: SDEdit state-0 refinement ----
    # Defaults to disabled if cfg_sdedit is None or enabled=False.
    sdedit_enabled = (
        cfg_sdedit is not None
        and bool(cfg_sdedit.get("enabled", False))
        and K >= 3
    )
    sdedit_report: Dict[str, Any] = {"enabled": sdedit_enabled}

    if sdedit_enabled:
        if "sparse_structure_encoder" not in pipe.models:
            print(
                "[run_scar] WARNING: sparse_structure_encoder not found in pipe.models; "
                "skipping SDEdit Pass 2 and falling back to Pass 1 output."
            )
            sdedit_enabled = False
            sdedit_report["enabled"] = False
            sdedit_report["skip_reason"] = "encoder_missing"

    if sdedit_enabled:
        encoder = pipe.models["sparse_structure_encoder"]
        guide_mode = str(cfg_sdedit.get("guide_mode", "augmented_intersection"))
        t_star = float(cfg_sdedit.get("t_star", 0.5))
        # v4 BMCSA vs v3 legacy dispatch.
        mode = str(cfg_sdedit.get("mode", "bmcsa")).lower()
        if mode not in ("bmcsa", "v3"):
            raise ValueError(
                f"cfg_sdedit.mode must be 'bmcsa' or 'v3'; got {mode!r}"
            )

        # refine_states semantics:
        #   mode = "v3":    list of specific targets (default endpoints).
        #   mode = "bmcsa": all K states are refined (list is ignored except
        #                   for viz-only intermediates; default full range).
        if mode == "bmcsa":
            refine_states_list = list(range(K))
        else:
            raw_refine = cfg_sdedit.get("refine_states", [0])
            refine_states_list = []
            for s in raw_refine:
                s = int(s)
                if s < 0:
                    s = K + s
                if 0 <= s < K:
                    refine_states_list.append(s)
            refine_states_list = sorted(set(refine_states_list))
            if not refine_states_list:
                raise ValueError(f"refine_states resolved to empty list for K={K}; raw={raw_refine}")

        # In v4 BMCSA mode, compute the shared P_base ONCE (mean over all K
        # states by default) and reuse it for BOTH the guide builder and the
        # M_base token-space mask. This ensures every state's guide shares the
        # same base, and the BMCSA gate M_base spatially matches where the
        # guide says "this is base".
        # In v3 mode, guide builder falls back to per-target MIN (legacy).
        exclude_s0_for_base = bool(cfg_sdedit.get("exclude_state_0", False))
        P_base_shared_64_raw: Optional[torch.Tensor] = None
        P_base_shared_64: Optional[torch.Tensor] = None
        if mode == "bmcsa":
            P_base_shared_64_raw = _compute_P_base_shared(
                soft_p1=soft_p1,
                exclude_state_0=exclude_s0_for_base,
            )                                               # (64, 64, 64)
            P_base_shared_64 = P_base_shared_64_raw         # default: raw (no semantic filter)

        # v4.3: M_attn computation hoisted UP before guide builder, so the
        # semantic filter is applied at the SOURCE (P_base_shared) rather than
        # only at BMCSA blend time. This is the fix for the v4.2 failure where
        # a long-drawer prismatic joint (e.g. sample 26525) leaks drawer
        # trajectory voxels into P_base_shared → over-large guide → all states
        # start Pass 2 from nearly-identical geometry → s0-2 collapse.
        # See record/stageb_detail.md Problem 13.
        M_attn_16_flat: Optional[torch.Tensor] = None
        attn_agreement_16: Optional[torch.Tensor] = None
        M_attn_64_filter: Optional[torch.Tensor] = None
        attn_m_enabled = (
            mode == "bmcsa" and bool(cfg_sdedit.get("attn_m_enabled", False))
        )
        attn_m_apply_at = str(cfg_sdedit.get("attn_m_apply_at", "guide")).lower()
        if attn_m_apply_at not in ("guide", "bmcsa", "both"):
            raise ValueError(
                f"cfg_sdedit.attn_m_apply_at must be 'guide' | 'bmcsa' | 'both'; got {attn_m_apply_at!r}"
            )
        if attn_m_enabled:
            attn_threshold = float(cfg_sdedit.get("attn_m_threshold", 0.7))
            attn_tau = float(cfg_sdedit.get("attn_m_tau", 0.05))
            M_attn_16_flat, agree_raw = _compute_M_attn_tokenspace(
                z_final=z_final,
                threshold=attn_threshold,
                tau=attn_tau,
                return_raw_agreement=True,
            )                                               # (1, L=4096, 1), (4096,)
            token_res_attn = int(round(agree_raw.shape[0] ** (1.0 / 3.0)))
            attn_agreement_16 = agree_raw.view(
                token_res_attn, token_res_attn, token_res_attn,
            )
            if attn_m_apply_at in ("guide", "both"):
                # Upsample M_attn from 16^3 to 64^3 via trilinear interpolation,
                # then filter P_base at source.
                M_attn_16_spatial = (
                    M_attn_16_flat
                    .view(1, 1, token_res_attn, token_res_attn, token_res_attn)
                    .to(device=soft_p1.device, dtype=soft_p1.dtype)
                )
                src_res = soft_p1.shape[-1]                 # 64
                M_attn_64_filter = torch.nn.functional.interpolate(
                    M_attn_16_spatial,
                    size=(src_res, src_res, src_res),
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0).contiguous()        # (64, 64, 64)
                # Filter P_base_shared at source — everything downstream (guide
                # builder, M_base tokenspace) uses the filtered version.
                P_base_shared_64 = P_base_shared_64_raw * M_attn_64_filter

        # Per-state guides: start from Pass-1 latents (identity), then override
        # each target with its own augmented-intersection guide (guide build is
        # shared between v3 and bmcsa modes; only the Pass-2 sampler differs).
        guides = z_final.clone()                            # (K, 8, 16, 16, 16)
        guide_intermediates_by_target: Dict[int, Dict[str, Any]] = {}
        for tgt in refine_states_list:
            z_guide_t, inter_t = _build_augmented_intersection_guide(
                P_stack=soft_p1,
                encoder=encoder,
                target_state=tgt,
                guide_mode=guide_mode,
                return_intermediates=True,
                P_base_shared=P_base_shared_64,             # filtered when attn_m applies at guide
            )                                               # z_guide_t: (1, 8, 16, 16, 16)
            z_guide_t = z_guide_t.to(device=z_final.device, dtype=z_final.dtype)
            guides[tgt] = z_guide_t[0]
            guide_intermediates_by_target[tgt] = inter_t

        # Derive Pass 2 schedule params from the sampler params used for Pass 1.
        total_steps = int(sampler_params.get("steps", 25))
        rescale_t_p2 = float(sampler_params.get("rescale_t", 1.0))
        cfg_strength_p2 = float(sampler_params.get("cfg_strength", 7.5))
        cfg_interval_p2 = tuple(sampler_params.get("cfg_interval", (0.0, 1.0)))

        if mode == "bmcsa":
            # v4 path: fixed-length Pass 2 + BMCSA. Build M_base in token space
            # from the SAME P_base_shared used above, per spec §4.2 (v4.1).
            tau_M = float(cfg_sdedit.get("tau_M", 0.05))
            # Auto-derive token_resolution from flow_model if not explicitly set
            # in config, so edits to patch_size won't silently break this.
            patch_size = int(getattr(flow_model, "patch_size", 1))
            resolution = int(getattr(flow_model, "resolution", 16))
            auto_token_res = resolution // patch_size
            token_resolution = int(cfg_sdedit.get("token_resolution", auto_token_res))
            if token_resolution != auto_token_res:
                print(
                    f"[run_scar] WARNING: cfg token_resolution={token_resolution} "
                    f"does not match flow_model resolution/patch_size="
                    f"{resolution}/{patch_size}={auto_token_res}. BMCSA will fail."
                )
            # Reuse the already-computed P_base_shared_64 (same one used for
            # the guide builder). In v4.3 "guide" mode this is the M_attn-
            # filtered version, so M_base is semantically filtered already
            # (no need to also apply M_attn at BMCSA blend time).
            M_base_flat = _compute_M_base_tokenspace(
                P_base_shared=P_base_shared_64,
                tau_M=tau_M,
                token_resolution=token_resolution,
            )                                               # (1, L, 1) on soft_p1's device

            # v4.3 dispatch: M_attn is applied at "guide" (P_base filter,
            # baked in), "bmcsa" (BMCSA blend kwarg), or "both".
            # Note: M_attn_16_flat was computed up-front before the guide
            # builder; here we just decide whether to pass it downstream.
            M_attn_flat_for_bmcsa: Optional[torch.Tensor] = None
            if attn_m_enabled and attn_m_apply_at in ("bmcsa", "both"):
                M_attn_flat_for_bmcsa = M_attn_16_flat

            pass2_steps = int(cfg_sdedit.get("pass2_steps", 12))
            bmcsa_blocks_cfg = cfg_sdedit.get("bmcsa_blocks", "all")
            if bmcsa_blocks_cfg in ("all", None):
                bmcsa_blocks = None                         # all blocks
            else:
                bmcsa_blocks = [int(i) for i in bmcsa_blocks_cfg]
            bmcsa_strength_cfg = float(cfg_sdedit.get("bmcsa_strength", 1.0))

            # v3.3 default: dynamic-M (per-block, computed from current hidden).
            # v4.3 legacy: static M_base (precomputed from Pass-1 P_base_shared).
            # Defaults follow method.md section 5.2.
            M_compute_mode = str(cfg_sdedit.get("M_compute_mode", "dynamic")).lower()
            tau_M_dyn = float(cfg_sdedit.get("tau_M_dynamic", 0.7))
            kappa_M_dyn = float(cfg_sdedit.get("kappa_M_dynamic", 0.05))
            capture_dyn_M = bool(cfg_sdedit.get("capture_dynamic_M", True))

            sdedit_out = _sdedit_refine_k6_bmcsa(
                sampler=sampler,
                flow_model=flow_model,
                guides=guides,
                cond=cond["cond"],
                neg_cond=cond["neg_cond"],
                t_star=t_star,
                pass2_steps=pass2_steps,
                rescale_t=rescale_t_p2,
                cfg_strength=cfg_strength_p2,
                cfg_interval=cfg_interval_p2,
                M_base=M_base_flat if M_compute_mode == "static" else None,
                bmcsa_blocks=bmcsa_blocks,
                bmcsa_strength=bmcsa_strength_cfg,
                shared_noise=noise,
                M_attn=M_attn_flat_for_bmcsa,               # None when apply_at="guide"
                verbose=True,
                collect_per_step=True,
                M_compute_mode=M_compute_mode,
                tau_M_dynamic=tau_M_dyn,
                kappa_M_dynamic=kappa_M_dyn,
                capture_dynamic_M=capture_dyn_M,
            )
        else:
            # v3 legacy: variable-length Pass 2 by ceil(total_steps * t_star).
            sdedit_out = _sdedit_refine_k6(
                sampler=sampler,
                flow_model=flow_model,
                guides=guides,
                cond=cond["cond"],
                neg_cond=cond["neg_cond"],
                t_star=t_star,
                total_steps=total_steps,
                rescale_t=rescale_t_p2,
                cfg_strength=cfg_strength_p2,
                cfg_interval=cfg_interval_p2,
                shared_noise=noise,
                verbose=True,
                collect_per_step=True,
            )
        z_refined = sdedit_out.samples                      # (K, 8, 16, 16, 16)
        pass2_pred_x_0 = sdedit_out.pred_x_0                # list of (K, 8, 16, 16, 16)
        pass2_t_seq = sdedit_out.t_seq                      # length n_pass2 + 1

        # Replacement policy:
        #   mode = "bmcsa": replace ALL K states with Pass 2 output.
        #   mode = "v3":    replace only states in refine_states_list.
        z_final_v3 = z_final.clone()
        if mode == "bmcsa":
            z_final_v3 = z_refined.clone()
            replaced_states = list(range(K))
        else:
            for tgt in refine_states_list:
                z_final_v3[tgt] = z_refined[tgt]
            replaced_states = refine_states_list

        # Re-decode with the replaced latents.
        logits = decoder(z_final_v3)
        soft = torch.sigmoid(logits).squeeze(1)
        binary = (soft > 0.5).to(soft.dtype)

        if remove_disk_flag:
            binary_np = binary.detach().cpu().numpy()
            voxels_5d = binary_np[:, None].astype(np.float32)
            voxels_5d = remove_disk(voxels_5d)
            binary = torch.from_numpy(voxels_5d[:, 0]).to(
                device=soft.device, dtype=soft.dtype,
            )
            soft = soft * binary

        z_final = z_final_v3
        # Per-state voxel deltas vs Pass 1, for diagnostics.
        voxel_deltas = {
            f"voxel_delta_state{tgt}": int(
                ((binary[tgt] > 0.5) != (binary_p1[tgt] > 0.5)).sum().item()
            )
            for tgt in replaced_states
        }
        pass2_steps_effective = (
            int(cfg_sdedit.get("pass2_steps", 12))
            if mode == "bmcsa"
            else max(1, int(np.ceil(total_steps * t_star)))
        )
        sdedit_report.update({
            "mode": mode,
            "guide_mode": guide_mode,
            "t_star": t_star,
            "pass2_steps": pass2_steps_effective,
            "shared_noise": True,                           # Pass 1's K-parallel noise reused
            "refine_states": refine_states_list,
            "replaced_states": replaced_states,
            **voxel_deltas,
        })
        # v3.3 BMCSA M-compute mode
        if mode == "bmcsa":
            sdedit_report["M_compute_mode"] = str(cfg_sdedit.get("M_compute_mode", "dynamic"))
            if sdedit_report["M_compute_mode"] == "dynamic":
                sdedit_report["tau_M_dynamic"] = float(cfg_sdedit.get("tau_M_dynamic", 0.6))
                sdedit_report["kappa_M_dynamic"] = float(cfg_sdedit.get("kappa_M_dynamic", 0.1))
                # Store the captured dynamic_M log to disk for viz / debug.
                dyn_M_payload = getattr(sdedit_out, "dynamic_M", {})
                if dyn_M_payload:
                    n_steps_logged = len(dyn_M_payload)
                    n_blocks_logged = (
                        len(next(iter(dyn_M_payload.values()))) if n_steps_logged > 0 else 0
                    )
                    sdedit_report["dynamic_M_captured"] = {
                        "n_steps_logged": int(n_steps_logged),
                        "n_blocks_logged": int(n_blocks_logged),
                    }
        # v4.2 / v4.3 M_attn report
        if mode == "bmcsa" and bool(cfg_sdedit.get("attn_m_enabled", False)):
            sdedit_report["attn_m_enabled"] = True
            sdedit_report["attn_m_threshold"] = float(cfg_sdedit.get("attn_m_threshold", 0.7))
            sdedit_report["attn_m_tau"] = float(cfg_sdedit.get("attn_m_tau", 0.05))
            sdedit_report["attn_m_apply_at"] = attn_m_apply_at
    else:
        # Pass 1 only (legacy path).
        logits = logits_p1
        soft = soft_p1
        binary = binary_p1

    icp_offsets: List[np.ndarray] = []
    icp_capped_flags: List[bool] = []
    if cfg_scar.get("icp_enabled", True) and K >= 2:
        vote_thresh = float(cfg_scar.get("icp_vote_threshold", 0.83))
        base_mask = _compute_intersection_mask(binary, vote_threshold=vote_thresh)
        target = binary[0].cpu().numpy()
        aligned_stack = [target]
        icp_offsets.append(np.zeros(3, dtype=np.float32))
        icp_capped_flags.append(False)
        max_t = float(cfg_scar.get("icp_max_translation", 1.5))
        for k in range(1, K):
            source = binary[k].cpu().numpy()
            aligned, offset, capped = align_to_reference(
                source, target,
                base_mask=base_mask,
                max_translation=max_t,
            )
            aligned_stack.append(aligned)
            icp_offsets.append(offset)
            icp_capped_flags.append(capped)
        binary = torch.from_numpy(np.stack(aligned_stack, axis=0)).to(
            device=device, dtype=soft.dtype,
        )
    else:
        icp_offsets = [np.zeros(3, dtype=np.float32) for _ in range(K)]
        icp_capped_flags = [False] * K

    save_voxel_grid(os.path.join(out_dir, "O_stack.npy"),
                    binary.detach().cpu().numpy().astype(np.uint8))
    save_voxel_grid(os.path.join(out_dir, "O_stack_soft.npy"),
                    soft.detach().cpu().numpy().astype(np.float16))
    torch.save(z_final.detach().cpu(), os.path.join(out_dir, "z_final.pt"))

    # v8 (2026-04-24): optionally capture 1024-dim SS-DiT hidden states at
    # DIFT-analogous mid-late blocks × mid-t. Used by Stage C SegMatch v8 as
    # articulated-part discriminator (replaces the 8-dim z_final classifier
    # that collapsed to EDT-equivalent). Controlled by cfg_scar key:
    #   "capture_dit_hidden": true/false (default false to preserve legacy runs)
    #   "dit_hook_blocks": list[int] (default [14, 16, 18])
    #   "dit_hook_t_star": float (default 0.3)
    if bool(cfg_scar.get("capture_dit_hidden", False)):
        dit_blocks_cfg = cfg_scar.get("dit_hook_blocks", [14, 16, 18])
        dit_blocks_cfg = [int(b) for b in dit_blocks_cfg]
        dit_t_star_cfg = float(cfg_scar.get("dit_hook_t_star", 0.3))
        try:
            dit_hidden_store = capture_dit_hidden_states(
                flow_model=flow_model,
                z_final=z_final,
                cond=cond["cond"],
                target_blocks=dit_blocks_cfg,
                t_star=dit_t_star_cfg,
            )
            meta_blob = {
                "target_blocks": dit_blocks_cfg,
                "t_star": dit_t_star_cfg,
                "blocks_captured": sorted(list(dit_hidden_store.keys())),
                "shape_per_block": (
                    list(dit_hidden_store[dit_blocks_cfg[0]].shape)
                    if dit_blocks_cfg and dit_blocks_cfg[0] in dit_hidden_store
                    else None
                ),
                "dtype": "float16",
            }
            torch.save(
                {"hidden_states": dit_hidden_store, "meta": meta_blob},
                os.path.join(out_dir, "dit_hidden.pt"),
            )
            print(
                f"[run_scar] saved DiT hidden states: "
                f"blocks={meta_blob['blocks_captured']} t_star={dit_t_star_cfg} "
                f"shape_per_block={meta_blob['shape_per_block']}"
            )
        except Exception as e:
            print(f"[run_scar] WARNING: DiT hidden state capture failed: {e}")

    diag = list(getattr(out, "scar_diagnostics", None) or [])
    with open(os.path.join(out_dir, "scar_diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)

    icp_report = {
        "offsets": [o.tolist() for o in icp_offsets],
        "capped": icp_capped_flags,
        "max_translation": float(cfg_scar.get("icp_max_translation", 1.5)),
    }
    with open(os.path.join(out_dir, "icp_report.json"), "w") as f:
        json.dump(icp_report, f, indent=2)

    with open(os.path.join(out_dir, "sdedit_report.json"), "w") as f:
        json.dump(sdedit_report, f, indent=2)

    # Save Pass-1 occupancies alongside Pass-2 to enable before/after diffs.
    if sdedit_report.get("enabled", False):
        save_voxel_grid(os.path.join(out_dir, "O_stack_pass1.npy"),
                        binary_p1.detach().cpu().numpy().astype(np.uint8))
        save_voxel_grid(os.path.join(out_dir, "O_stack_pass1_soft.npy"),
                        soft_p1.detach().cpu().numpy().astype(np.float16))

    # ---- v3.3 Dynamic-M BMCSA: persist per-step per-block M log to disk ----
    # The log is {step_idx: {block_idx: (L,) fp16}} where L = token_resolution^3
    # (4096 for TRELLIS-image-large). Used by viz code below and downstream
    # diagnostic scripts.
    if (
        sdedit_report.get("enabled", False)
        and sdedit_report.get("mode") == "bmcsa"
        and sdedit_report.get("M_compute_mode") == "dynamic"
    ):
        dyn_M_payload = getattr(sdedit_out, "dynamic_M", {})
        if dyn_M_payload:
            # Convert nested dict to a plain torch tensor stack for compact storage.
            # Sort step indices and block indices for deterministic ordering.
            step_ids = sorted(dyn_M_payload.keys())
            block_ids = sorted(dyn_M_payload[step_ids[0]].keys())
            # Stack into (n_steps, n_blocks, L) fp16.
            M_stack = torch.stack([
                torch.stack([dyn_M_payload[s][b] for b in block_ids], dim=0)
                for s in step_ids
            ], dim=0)                                                    # (S, B, L)
            torch.save(
                {
                    "M_stack": M_stack,                                  # (S, B, L) fp16
                    "step_ids": [int(x) for x in step_ids],
                    "block_ids": [int(x) for x in block_ids],
                    "tau_M_dynamic": float(sdedit_report.get("tau_M_dynamic", 0.6)),
                    "kappa_M_dynamic": float(sdedit_report.get("kappa_M_dynamic", 0.1)),
                    "token_resolution": int(round(M_stack.shape[-1] ** (1.0 / 3.0))),
                },
                os.path.join(out_dir, "dynamic_M_log.pt"),
            )

    # ------------- Pass 2 Guide visualisations + s1..K-1 overlap -------------
    # Always emit the "s1..K-1 intersection" (state-0 reference overlap)
    # regardless of Pass 2 status, since it's useful for Pass 1 diagnostics.
    if K >= 2:
        viz_guide_dir = os.path.join(out_dir, "viz", "guide")
        os.makedirs(viz_guide_dir, exist_ok=True)

        # s1..K-1 soft intersection (the "overlap of open states" field used as
        # state 0's base reference). Emit as both soft and binary HTMLs.
        P_open = soft_p1[1:]                                            # (K-1, 64, 64, 64)
        P_overlap_1_to_Km1 = P_open.min(dim=0).values.detach().cpu().numpy().astype(np.float32)
        # soft prob viz
        from pipelines.utils.voxel_viz import save_soft_voxel_html
        save_soft_voxel_html(
            P_overlap_1_to_Km1,
            os.path.join(viz_guide_dir, "states_1_to_Km1_overlap_soft.html"),
            title="min_{k=1..K-1} P^(k)  (state-0 reference overlap, soft)",
            threshold=0.15,
        )
        save_voxel_html(
            (P_overlap_1_to_Km1 > 0.5).astype(np.float32),
            os.path.join(viz_guide_dir, "states_1_to_Km1_overlap_bin.html"),
            title="min_{k=1..K-1} P^(k) > 0.5  (binary)",
        )

        # Also: pairwise overlap counts for all-k pairs, as a single HTML with
        # a dropdown per pair (audit how much any two states agree).
        # (Dropped for brevity; per-pair IoU is already in Pass 1 per_step_stats.)

    # ---- v4 BMCSA: visualise M_base (the mean of s1..K-1, 64^3 and token 8^3) ----
    # This is the actual base gate used to blend cross-state attention during
    # Pass 2. Only emit when BMCSA is active.
    bmcsa_active_for_viz = (
        sdedit_report.get("enabled", False)
        and sdedit_report.get("mode", None) == "bmcsa"
        and K >= 2
    )
    if bmcsa_active_for_viz:
        viz_bmcsa_dir = os.path.join(out_dir, "viz", "bmcsa")
        os.makedirs(viz_bmcsa_dir, exist_ok=True)

        tau_M_used = float(cfg_sdedit.get("tau_M", 0.05))
        exclude_s0 = bool(cfg_sdedit.get("exclude_state_0", False))
        P_base_shared_64 = _compute_P_base_shared(soft_p1, exclude_state_0=exclude_s0)
        M_base_64 = torch.sigmoid((P_base_shared_64 - 0.5) / tau_M_used)
        M_base_64_np = M_base_64.detach().cpu().numpy().astype(np.float32)

        from pipelines.utils.voxel_viz import save_soft_voxel_html as _save_soft
        title_suffix = "s1..K-1 mean" if exclude_s0 else "all K mean"
        _save_soft(
            M_base_64_np,
            os.path.join(viz_bmcsa_dir, "M_base_64.html"),
            title=f"M_base (64³) = sigmoid((P_base_shared - 0.5)/tau_M), {title_suffix}, tau_M={tau_M_used}",
            threshold=0.15,
        )

        # Also save the P_base_shared probability field itself (pre-sigmoid).
        _save_soft(
            P_base_shared_64.detach().cpu().numpy().astype(np.float32),
            os.path.join(viz_bmcsa_dir, "P_base_shared_64.html"),
            title=f"P_base_shared (64³) = mean over {title_suffix}",
            threshold=0.15,
        )
        save_voxel_grid(
            os.path.join(viz_bmcsa_dir, "M_base_64.npy"),
            M_base_64_np.astype(np.float16),
        )

        # ---- v3.3 Dynamic-M visualisation ----
        # When M_compute_mode='dynamic', emit:
        #   1) Per-block mean M (token-space 16^3): how each block sees "base"
        #   2) Per-step mean M (across blocks)
        #   3) Step-by-block M_mean heatmap PNG
        #   4) Block-pair similarity (verifies per-block M is actually different)
        if sdedit_report.get("M_compute_mode") == "dynamic":
            dyn_M_payload = getattr(sdedit_out, "dynamic_M", {})
            if dyn_M_payload:
                step_ids = sorted(dyn_M_payload.keys())
                block_ids = sorted(dyn_M_payload[step_ids[0]].keys())
                M_stack_np = np.stack([
                    np.stack(
                        [dyn_M_payload[s][b].numpy().astype(np.float32) for b in block_ids],
                        axis=0,
                    )
                    for s in step_ids
                ], axis=0)                                              # (S, B, L)
                S, B, L = M_stack_np.shape
                token_res = int(round(L ** (1.0 / 3.0)))
                assert token_res ** 3 == L, (
                    f"dynamic_M L={L} is not a perfect cube; got token_res={token_res}"
                )

                # 1) Per-block mean M, reshape to (token_res, token_res, token_res)
                # and upsample to 64^3 for direct comparison with M_base_64.
                M_per_block_mean = M_stack_np.mean(axis=0)              # (B, L)
                M_per_block_mean_3d = M_per_block_mean.reshape(B, token_res, token_res, token_res)

                from pipelines.utils.voxel_viz import save_soft_voxel_html as _save_soft_v
                for idx, blk in enumerate(block_ids):
                    # Upsample 16^3 -> 64^3 for visual consistency with M_base_64.
                    m_blk_16 = torch.from_numpy(M_per_block_mean_3d[idx])
                    m_blk_64 = torch.nn.functional.interpolate(
                        m_blk_16.unsqueeze(0).unsqueeze(0),
                        size=(64, 64, 64),
                        mode="trilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0).numpy().astype(np.float32)
                    _save_soft_v(
                        m_blk_64,
                        os.path.join(
                            viz_bmcsa_dir,
                            f"M_dynamic_block_{blk:02d}_mean_64.html",
                        ),
                        title=(
                            f"M_dynamic block={blk}, mean over {S} steps "
                            f"(upsampled 16->64, base-likelihood gate)"
                        ),
                        threshold=0.15,
                    )

                # 2) Per-step mean M (across blocks), also upsampled.
                M_per_step_mean = M_stack_np.mean(axis=1)               # (S, L)
                M_per_step_mean_3d = M_per_step_mean.reshape(S, token_res, token_res, token_res)
                for idx, step in enumerate(step_ids):
                    m_step_16 = torch.from_numpy(M_per_step_mean_3d[idx])
                    m_step_64 = torch.nn.functional.interpolate(
                        m_step_16.unsqueeze(0).unsqueeze(0),
                        size=(64, 64, 64),
                        mode="trilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0).numpy().astype(np.float32)
                    _save_soft_v(
                        m_step_64,
                        os.path.join(
                            viz_bmcsa_dir,
                            f"M_dynamic_step_{step:02d}_mean_64.html",
                        ),
                        title=(
                            f"M_dynamic step={step}, mean over {B} blocks "
                            f"(upsampled 16->64)"
                        ),
                        threshold=0.15,
                    )

                # 3) (S, B) heatmap of M_mean — scalar per (step, block).
                M_scalar_sb = M_stack_np.mean(axis=-1)                  # (S, B)
                # 4) Block-pair similarity: cos(M_block_i_mean_over_step, M_block_j_mean_over_step)
                # Verifies that dynamic-M actually differs across blocks (not collapsed
                # to identical maps).
                M_block_l2 = M_per_block_mean / (
                    np.linalg.norm(M_per_block_mean, axis=-1, keepdims=True) + 1e-8
                )
                block_pairwise_cos = M_block_l2 @ M_block_l2.T          # (B, B) in [-1, 1]

                # Write summary plotly HTML (S-B heatmap + B-B cosine heatmap +
                # M_mean over steps per block line plot).
                from pipelines.utils.voxel_viz import _PLOTLY_OK
                if _PLOTLY_OK:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=(
                            f"Dynamic-M mean per (step, block) — token-space avg "
                            f"(S={S}, B={B})",
                            f"Pairwise cosine between block-M maps "
                            f"(M_dynamic block i vs block j, mean over steps)",
                            "Per-block M_mean evolution over steps",
                        ),
                        vertical_spacing=0.08,
                    )
                    fig.add_trace(
                        go.Heatmap(
                            z=M_scalar_sb,
                            x=[f"blk{b}" for b in block_ids],
                            y=[f"step{s}" for s in step_ids],
                            colorscale="Viridis",
                            zmin=0.0, zmax=1.0,
                            colorbar=dict(title="M_mean", y=0.85, len=0.25),
                        ),
                        row=1, col=1,
                    )
                    fig.add_trace(
                        go.Heatmap(
                            z=block_pairwise_cos,
                            x=[f"blk{b}" for b in block_ids],
                            y=[f"blk{b}" for b in block_ids],
                            colorscale="RdBu", reversescale=True,
                            zmin=-1.0, zmax=1.0,
                            colorbar=dict(title="cos(M_i, M_j)", y=0.5, len=0.25),
                        ),
                        row=2, col=1,
                    )
                    for idx, blk in enumerate(block_ids):
                        fig.add_trace(
                            go.Scatter(
                                x=step_ids,
                                y=M_scalar_sb[:, idx],
                                mode="lines+markers",
                                name=f"blk{blk}",
                            ),
                            row=3, col=1,
                        )
                    fig.update_xaxes(title_text="block", row=1, col=1)
                    fig.update_yaxes(title_text="step", row=1, col=1)
                    fig.update_xaxes(title_text="block j", row=2, col=1)
                    fig.update_yaxes(title_text="block i", row=2, col=1)
                    fig.update_xaxes(title_text="step", row=3, col=1)
                    fig.update_yaxes(title_text="M_mean", row=3, col=1)
                    fig.update_layout(
                        title=(
                            f"v3.3 Dynamic-M BMCSA diagnostics — "
                            f"tau_M={sdedit_report.get('tau_M_dynamic', 0.6)}, "
                            f"kappa_M={sdedit_report.get('kappa_M_dynamic', 0.1)}"
                        ),
                        height=1200,
                    )
                    fig.write_html(
                        os.path.join(viz_bmcsa_dir, "dynamic_M_diagnostics.html"),
                        include_plotlyjs="cdn",
                    )

                # Also save the (S, B) scalar heatmap as numpy for downstream
                # ablation / inspection.
                save_voxel_grid(
                    os.path.join(viz_bmcsa_dir, "M_dynamic_scalar_step_block.npy"),
                    M_scalar_sb.astype(np.float16),
                )

        # v4.2/v4.3: M_attn viz (only when attn_m_enabled)
        if sdedit_report.get("attn_m_enabled", False) and attn_agreement_16 is not None:
            agree_np = attn_agreement_16.detach().cpu().numpy().astype(np.float32)
            # Apply the same sigmoid as used for M_attn computation
            attn_thresh_used = float(sdedit_report.get("attn_m_threshold", 0.7))
            attn_tau_used = float(sdedit_report.get("attn_m_tau", 0.05))
            # NumPy-safe sigmoid
            M_attn_16_np = 1.0 / (1.0 + np.exp(-(agree_np - attn_thresh_used) / attn_tau_used))
            _save_soft(
                agree_np,
                os.path.join(viz_bmcsa_dir, "attn_agreement_16.html"),
                title=(
                    f"Cross-state latent-feature agreement (16³, raw cosine sim) - "
                    f"high = cabinet-like, low = move-like"
                ),
                threshold=0.15,
            )
            _save_soft(
                M_attn_16_np,
                os.path.join(viz_bmcsa_dir, "M_attn_16.html"),
                title=f"M_attn (16³) = sigmoid((agree - {attn_thresh_used})/{attn_tau_used})",
                threshold=0.15,
            )
            save_voxel_grid(
                os.path.join(viz_bmcsa_dir, "M_attn_16.npy"),
                M_attn_16_np.astype(np.float16),
            )

            # v4.3: M_attn upsampled to 64³ (the actual filter applied to P_base_shared)
            if M_attn_64_filter is not None:
                M_attn_64_np = M_attn_64_filter.detach().cpu().numpy().astype(np.float32)
                _save_soft(
                    M_attn_64_np,
                    os.path.join(viz_bmcsa_dir, "M_attn_64.html"),
                    title="M_attn (64³, trilinear upsample from 16³, applied as P_base filter)",
                    threshold=0.15,
                )
                save_voxel_grid(
                    os.path.join(viz_bmcsa_dir, "M_attn_64.npy"),
                    M_attn_64_np.astype(np.float16),
                )

            # v4.3: P_base_shared RAW vs FILTERED comparison
            # (only when filter is applied at guide stage)
            if P_base_shared_64_raw is not None and M_attn_64_filter is not None:
                P_raw_np = P_base_shared_64_raw.detach().cpu().numpy().astype(np.float32)
                P_filt_np = P_base_shared_64.detach().cpu().numpy().astype(np.float32)
                _save_soft(
                    P_raw_np,
                    os.path.join(viz_bmcsa_dir, "P_base_shared_64_raw.html"),
                    title="P_base_shared (64³, RAW before M_attn filter) = mean over K",
                    threshold=0.15,
                )
                _save_soft(
                    P_filt_np,
                    os.path.join(viz_bmcsa_dir, "P_base_shared_64_filtered.html"),
                    title="P_base_shared (64³, FILTERED by M_attn_64) — used by guide + M_base",
                    threshold=0.15,
                )
                # Diff for diagnosing where the filter removed voxels
                P_diff_np = P_raw_np - P_filt_np
                _save_soft(
                    P_diff_np,
                    os.path.join(viz_bmcsa_dir, "P_base_shared_diff.html"),
                    title="P_base diff (raw - filtered): bright = filtered OUT as move/noise",
                    threshold=0.15,
                )

    # Per-target guide viz (only when Pass 2 actually ran).
    if sdedit_report.get("enabled", False):
        viz_guide_dir = os.path.join(out_dir, "viz", "guide")
        os.makedirs(viz_guide_dir, exist_ok=True)
        for tgt, inter in guide_intermediates_by_target.items():
            tag = f"state{tgt}"
            P_base_np = inter["P_base"].numpy().astype(np.float32)
            P_excl_np = inter["P_excl"].numpy().astype(np.float32)
            P_guide_np = inter["P_guide"].numpy().astype(np.float32)
            O_bin_np = inter["O_guide_bin"].numpy().astype(np.float32)

            save_soft_voxel_html(
                P_base_np,
                os.path.join(viz_guide_dir, f"{tag}_P_base.html"),
                title=f"P_base  (ref intersection, target={tgt})",
                threshold=0.15,
            )
            save_soft_voxel_html(
                P_excl_np,
                os.path.join(viz_guide_dir, f"{tag}_P_excl.html"),
                title=f"P_excl  (target={tgt} exclusive, ReLU)",
                threshold=0.15,
            )
            save_soft_voxel_html(
                P_guide_np,
                os.path.join(viz_guide_dir, f"{tag}_P_guide.html"),
                title=f"P_guide = max(P_base, P_excl), target={tgt}",
                threshold=0.15,
            )
            save_voxel_html(
                O_bin_np,
                os.path.join(viz_guide_dir, f"{tag}_O_guide_bin.html"),
                title=f"O_guide_bin = (P_guide > 0.5), target={tgt}",
            )
            # Also save the numpy arrays for downstream ablation / inspection.
            save_voxel_grid(
                os.path.join(viz_guide_dir, f"{tag}_P_guide.npy"),
                P_guide_np.astype(np.float16),
            )

    # ------------- Pass 2 per-step visualisation -------------
    if sdedit_report.get("enabled", False):
        pass2_step_dir = os.path.join(out_dir, "pass2_per_step")
        os.makedirs(pass2_step_dir, exist_ok=True)
        pass2_stats = []

        for step_idx, x0_hat in enumerate(pass2_pred_x_0):
            # x0_hat: (K, 8, 16, 16, 16) -- Tweedie clean estimate at step_idx.
            with torch.no_grad():
                logits_step = decoder(x0_hat.to(device))
                occ_step = (torch.sigmoid(logits_step).squeeze(1) > 0.5).float()

            occ_np = occ_step.detach().cpu().numpy()

            counts = [int(occ_np[k].sum()) for k in range(occ_np.shape[0])]

            ious = []
            for i in range(occ_np.shape[0]):
                for j in range(i + 1, occ_np.shape[0]):
                    inter = ((occ_np[i] > 0.5) & (occ_np[j] > 0.5)).sum()
                    union = ((occ_np[i] > 0.5) | (occ_np[j] > 0.5)).sum()
                    ious.append(float(inter / max(union, 1)))

            all_occ = (occ_np > 0.5).all(axis=0)
            any_occ = (occ_np > 0.5).any(axis=0)

            t_now = float(pass2_t_seq[step_idx]) if step_idx < len(pass2_t_seq) else 0.0
            t_prev = float(pass2_t_seq[step_idx + 1]) if step_idx + 1 < len(pass2_t_seq) else 0.0

            step_stat = {
                "step": step_idx,
                "t": t_now,
                "t_prev": t_prev,
                "voxel_counts": counts,
                "pairwise_iou_mean": float(np.mean(ious)) if ious else 0.0,
                "pairwise_iou_min": float(np.min(ious)) if ious else 0.0,
                "pairwise_iou_max": float(np.max(ious)) if ious else 0.0,
                "always_occupied": int(all_occ.sum()),
                "ever_occupied": int(any_occ.sum()),
            }
            pass2_stats.append(step_stat)

            save_voxel_stack_html(
                occ_np.astype(np.float32),
                os.path.join(pass2_step_dir, f"O_step_{step_idx:02d}.html"),
                title=f"Pass2 step {step_idx}  t={t_now:.3f}",
            )

        with open(os.path.join(pass2_step_dir, "per_step_stats.json"), "w") as f:
            json.dump(pass2_stats, f, indent=2)

    # ---- Per-step Tweedie decode + cross-state consistency analysis ----
    # Decode each Euler step's Tweedie estimate (pred_x_0) through the VAE
    # decoder to see how 64^3 occupancy evolves. Matches the visualisation
    # style of diagnose_trellis_steps.py so SCAR runs can be compared to
    # baseline plain-TRELLIS runs step-by-step.
    pred_x0_list = getattr(out, "pred_x_0", [])
    if pred_x0_list:
        step_dir = os.path.join(out_dir, "per_step")
        os.makedirs(step_dir, exist_ok=True)
        per_step_stats = []

        for step_idx, x0_hat in enumerate(pred_x0_list):
            # x0_hat: (K, 8, 16, 16, 16) — Tweedie clean estimate at this step
            with torch.no_grad():
                logits_step = decoder(x0_hat.to(device))                 # (K, 1, 64, 64, 64)
                occ_step = (torch.sigmoid(logits_step).squeeze(1) > 0.5).float()

            occ_np = occ_step.detach().cpu().numpy()
            K_s = occ_np.shape[0]

            counts = [int(occ_np[k].sum()) for k in range(K_s)]

            ious = []
            for i in range(K_s):
                for j in range(i + 1, K_s):
                    inter = ((occ_np[i] > 0.5) & (occ_np[j] > 0.5)).sum()
                    union = ((occ_np[i] > 0.5) | (occ_np[j] > 0.5)).sum()
                    ious.append(float(inter / max(union, 1)))

            all_occ = (occ_np > 0.5).all(axis=0)
            any_occ = (occ_np > 0.5).any(axis=0)
            always = int(all_occ.sum())
            ever = int(any_occ.sum())

            voxel_std = occ_np.std(axis=0)
            mean_std = float(voxel_std.mean())
            max_std = float(voxel_std.max())

            x0_np = x0_hat.detach().cpu().float().numpy()
            latent_var_per_voxel = x0_np.var(axis=0).sum(axis=0)

            diag_entry = diag[step_idx] if step_idx < len(diag) else {}
            step_stat = {
                "step": step_idx,
                "t": float(diag_entry.get("t", 0.0)),
                "alpha": float(diag_entry.get("alpha", 0.0)),
                "mixed": bool(diag_entry.get("mixed", False)),
                "voxel_counts": counts,
                "pairwise_iou_mean": float(np.mean(ious)) if ious else 0.0,
                "pairwise_iou_min": float(np.min(ious)) if ious else 0.0,
                "pairwise_iou_max": float(np.max(ious)) if ious else 0.0,
                "always_occupied": always,
                "ever_occupied": ever,
                "always_ever_ratio": float(always / max(ever, 1)),
                "voxel_std_mean": mean_std,
                "voxel_std_max": max_std,
                "latent_var_mean": float(latent_var_per_voxel.mean()),
                "latent_var_max": float(latent_var_per_voxel.max()),
            }
            per_step_stats.append(step_stat)

            # Generate HTML directly from in-memory occupancy (no npy intermediate)
            phase_tag = "mix" if diag_entry.get("mixed", False) else (
                "push" if diag_entry.get("alpha", 0.0) > 0 else "plain"
            )
            save_voxel_stack_html(
                occ_np.astype(np.float32),
                os.path.join(step_dir, f"O_step_{step_idx:02d}.html"),
                title=f"Step {step_idx} [{phase_tag}] t={diag_entry.get('t', 0.0):.3f}",
            )

        with open(os.path.join(step_dir, "per_step_stats.json"), "w") as f:
            json.dump(per_step_stats, f, indent=2)

        # Summary curves: IoU / always-ever / latent variance across steps
        from pipelines.utils.voxel_viz import _PLOTLY_OK
        if _PLOTLY_OK:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                subplot_titles=(
                    "Pairwise IoU (min/mean/max)",
                    "Always-occupied / Ever-occupied ratio",
                    "Latent variance (mean/max)",
                ),
            )
            steps = [s["step"] for s in per_step_stats]
            mix_steps = [s["step"] for s in per_step_stats if s["mixed"]]
            push_steps = [s["step"] for s in per_step_stats if s["alpha"] > 0]

            fig.add_trace(
                go.Scatter(x=steps, y=[s["pairwise_iou_mean"] for s in per_step_stats],
                           mode="lines+markers", name="IoU mean"),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=steps, y=[s["pairwise_iou_min"] for s in per_step_stats],
                           mode="lines", name="IoU min", line=dict(dash="dot")),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=steps, y=[s["pairwise_iou_max"] for s in per_step_stats],
                           mode="lines", name="IoU max", line=dict(dash="dot")),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=steps, y=[s["always_ever_ratio"] for s in per_step_stats],
                           mode="lines+markers", name="always/ever"),
                row=2, col=1,
            )
            fig.add_trace(
                go.Scatter(x=steps, y=[s["latent_var_mean"] for s in per_step_stats],
                           mode="lines+markers", name="latent var mean"),
                row=3, col=1,
            )
            fig.add_trace(
                go.Scatter(x=steps, y=[s["latent_var_max"] for s in per_step_stats],
                           mode="lines", name="latent var max", line=dict(dash="dot")),
                row=3, col=1,
            )
            # Shade phase regions
            if mix_steps:
                fig.add_vrect(x0=min(mix_steps) - 0.5, x1=max(mix_steps) + 0.5,
                              fillcolor="LightBlue", opacity=0.2, line_width=0,
                              annotation_text="mix", annotation_position="top left")
            if push_steps:
                fig.add_vrect(x0=min(push_steps) - 0.5, x1=max(push_steps) + 0.5,
                              fillcolor="LightGreen", opacity=0.2, line_width=0,
                              annotation_text="push", annotation_position="top left")

            fig.update_layout(title="SCAR per-step cross-state consistency", height=900)
            fig.update_xaxes(title_text="step", row=3, col=1)
            fig.write_html(
                os.path.join(step_dir, "per_step_curves.html"),
                include_plotlyjs="cdn",
            )
    # ---- End per-step diagnostics ----

    viz_dir = os.path.join(out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    binary_np = binary.detach().cpu().numpy().astype(np.float32)
    save_voxel_stack_html(
        binary_np,
        os.path.join(viz_dir, "O_stack.html"),
        title="Stage B SCAR: O_k (post-ICP)",
    )
    for k in range(binary_np.shape[0]):
        save_voxel_html(
            binary_np[k],
            os.path.join(viz_dir, f"O_k_{k:02d}.html"),
            title=f"O_{k}",
        )
    save_diagnostics_curves_html(
        diag,
        os.path.join(viz_dir, "scar_diagnostics.html"),
        title="SCAR per-step diagnostics",
        keys=("alpha", "M_mean", "M_mean_active", "push_norm", "active_voxels"),
    )

    # ---- Base/Move classification visualization ----
    # Preview Stage C's initial partition (C.1) on our O_stack: pure
    # footprint formulation from pipelines/sajo/anchors.joint_free_split.
    # M_attn is intentionally NOT applied here (audit 2026-04-23): it was
    # found to misclassify hinge-near-door voxels (mu=1, std=0, but low
    # M_attn due to door-vs-cabinet semantic distinction) as move. The
    # long-prismatic-trajectory bug is already handled by the std-gate on
    # p_base alone. M_attn enters Stage C downstream as a graph-cut unary.
    bmp_base, bmp_move = joint_free_split(
        binary,
        sigma_b=0.25, sigma_m=0.15, tau_b=0.05, tau_m=0.05,
        mode="footprint",
    )
    save_voxel_grid(os.path.join(out_dir, "p_base_preview.npy"),
                    bmp_base.detach().cpu().numpy().astype(np.float32))
    save_voxel_grid(os.path.join(out_dir, "p_move_preview.npy"),
                    bmp_move.detach().cpu().numpy().astype(np.float32))
    save_two_masks_html(
        bmp_base.detach().cpu().numpy().astype(np.float32),
        bmp_move.detach().cpu().numpy().astype(np.float32),
        os.path.join(viz_dir, "base_move_preview.html"),
        title="Stage B base/move preview (footprint, joint-free split)",
    )

    return SCARResult(
        O_stack=binary,
        O_stack_soft=soft,
        z_final=z_final,
        diagnostics=diag,
        icp_offsets=icp_offsets,
        icp_capped=icp_capped_flags,
    )

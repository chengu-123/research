"""Per-state TRELLIS keyframe Gaussian models for multi-view W-RFSDS.

Multi-view W-RFSDS uses, as InP keyframes and pixel-anchor targets, 2D renders
of per-state TRELLIS 3D models for the FIRST (closed, state-0) and LAST (open,
state-5) articulation states. This module builds those per-state Gaussian models
reusably: given the bootstrap latents and per-state pure-image targets, it
samples each requested state's own SLAT and decodes it into a TRELLIS Gaussian.

The build path is the validated recipe from ``_verify_perstate_slat.py``
(Slurm job 5793 produced base-aligned states 0/2/5): per state ``k`` ->
``occ = sigmoid(ss_vae_decoder(z_final[k:k+1])) > 0.5`` -> coords via
``_flat_idx_to_xyz`` -> ``pipe.sample_slat`` with that state's own cond and a
fixed per-state noise seed -> ``DGSWithParent(slat_decoder_gs)`` -> Gaussian.

This module performs no rendering and constructs no camera. It only returns the
decoded per-state Gaussian objects, keyed by state index, so callers (the Stage D
training loop, a diagnostic, a test) can render or anchor them as they wish.

It does not import or modify any Stage D training entrypoint. It reuses the
frozen API from ``pipelines.bootstrap`` and ``pipelines.stage_d``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from pipelines.bootstrap import (
    _build_trellis_cond_from_float_states,
    _flat_idx_to_xyz,
)
from pipelines.stage_d.render import DGSWithParent
from pipelines.stage_d.train import _build_sparse_in

# SLAT sampling defaults from the validated recipe (_verify_perstate_slat.py).
OCC_RES = 64
SLAT_STEPS = 25
SLAT_CFG = 7.5
SLAT_THRESH = 0.5


def _occ_to_coords4(
    z_final_k: torch.Tensor,
    ss_vae_decoder: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Option-A occupancy -> (N, 4) int32 batched SLAT coords for one state.

    ``occ = sigmoid(ss_vae_decoder(z_final[k:k+1]))``; voxels with
    ``occ > SLAT_THRESH`` become SLAT coords. The first column is the batch
    index (0), matching the ``_build_sparse_in`` / ``DGSWithParent`` convention.

    Parameters
    ----------
    z_final_k : Tensor
        Single-state sparse-structure latent ``z_final[k:k+1]``.
    ss_vae_decoder : nn.Module
        TRELLIS ``sparse_structure_decoder`` (occupancy logits decoder).
    device : torch.device
        Device for the occupancy decode and the returned coords.

    Returns
    -------
    coords4 : Tensor [N, 4] int32
        ``[batch_idx=0, x, y, z]`` for each occupied voxel.
    """
    with torch.no_grad():
        occ_logit = ss_vae_decoder(z_final_k.to(device))          # [1, 1, R, R, R]
        occ = torch.sigmoid(occ_logit)
    flat = (occ.view(-1) > SLAT_THRESH).nonzero(as_tuple=False).squeeze(-1)
    coords3 = _flat_idx_to_xyz(flat, OCC_RES).to(dtype=torch.int32, device=device)
    n = int(coords3.shape[0])
    batch_col = torch.zeros((n, 1), dtype=torch.int32, device=device)
    coords4 = torch.cat([batch_col, coords3], dim=-1)             # [N, 4]
    return coords4


def _sample_slat_feats(
    pipe,
    coords4: torch.Tensor,
    cond_k: Dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Sample SLAT feats (N, 8) for one state via ``pipe.sample_slat``.

    Recipe from ``pipelines/bootstrap.py:_run_b8_slat_sampler``: the SLAT noise
    is a SparseTensor with ``feats = randn(N, flow_model.in_channels)`` drawn
    after a fixed per-state seed, so the only difference between states is the
    geometry / cond, not the latent noise draw.

    Parameters
    ----------
    pipe : TRELLIS image-to-3D pipeline.
    coords4 : Tensor [N, 4] int32
        Batched SLAT coords for this state.
    cond_k : dict
        This state's own conditioning, ``{"cond": ..., "neg_cond": ...}``.
    device : torch.device
        Device for the noise tensor.
    seed : int
        Deterministic seed for this state's noise draw.

    Returns
    -------
    z_slat : Tensor [N, 8]
        Sampled SLAT features (detached).
    """
    from trellis.modules import sparse as sp

    flow_model = pipe.models["slat_flow_model"]
    n = int(coords4.shape[0])
    torch.manual_seed(int(seed))
    noise = sp.SparseTensor(
        feats=torch.randn(
            n, flow_model.in_channels, device=device,
            dtype=next(flow_model.parameters()).dtype,
        ),
        coords=coords4,
    )
    z_slat_sparse = pipe.sample_slat(
        cond=cond_k,
        coords=coords4,
        sampler_params={"steps": SLAT_STEPS, "cfg_strength": SLAT_CFG},
        noise=noise,
    )
    return z_slat_sparse.feats.detach()


def _decode_gaussians(
    d_gs_w: DGSWithParent,
    z_slat: torch.Tensor,
    coords4: torch.Tensor,
):
    """Decode SLAT feats + coords into one TRELLIS Gaussian object.

    Parameters
    ----------
    d_gs_w : DGSWithParent
        Wrapped frozen TRELLIS ``slat_decoder_gs``.
    z_slat : Tensor [N, 8]
        Sampled SLAT features for this state.
    coords4 : Tensor [N, 4] int32
        Batched SLAT coords for this state.

    Returns
    -------
    gauss : Gaussian
        TRELLIS Gaussian object (single batch element). Access positions /
        opacity / rotation / scaling via ``gauss.get_xyz`` etc.
    """
    sparse_in = _build_sparse_in(z_slat, coords4)
    with torch.no_grad():
        gauss, _parent_idx = d_gs_w(sparse_in)
    return gauss


def _cond_for_state(
    cond_all: Dict[str, torch.Tensor],
    k: int,
) -> Dict[str, torch.Tensor]:
    """Slice the per-state conditioning for state ``k``.

    Uses each state's OWN cond token (``cond["cond"][k:k+1]``) and its own
    negative cond (``cond["neg_cond"][k:k+1]``), falling back to a zero
    negative token when the pipeline does not provide one.
    """
    cond_tok = cond_all["cond"]
    neg_tok = cond_all.get("neg_cond")
    neg = (
        neg_tok[k:k + 1] if neg_tok is not None
        else torch.zeros_like(cond_tok[k:k + 1])
    )
    return {"cond": cond_tok[k:k + 1], "neg_cond": neg}


def build_per_state_keyframe_models(
    pipe,
    z_final: torch.Tensor,
    pure_state_targets_K3HW: torch.Tensor,
    state_indices: Sequence[int] = (0, 5),
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[int, object]:
    """Build per-state TRELLIS keyframe Gaussian models, keyed by state index.

    For each ``k`` in ``state_indices`` this samples state ``k``'s own SLAT from
    its own conditioning and decodes it into a TRELLIS Gaussian, reusing the
    validated ``_verify_perstate_slat.py`` recipe. All states share the frozen
    TRELLIS weights, so the returned Gaussians live in the same world frame /
    scale (base-aligned); the geometry differs only by each state's occupancy
    and conditioning.

    No rendering and no camera here: the function returns only the decoded
    Gaussian objects so the caller decides how to render / anchor them.

    Parameters
    ----------
    pipe : TRELLIS image-to-3D pipeline with ``models`` and ``sample_slat``.
        Must expose ``models["sparse_structure_decoder"]``,
        ``models["slat_decoder_gs"]`` and ``models["slat_flow_model"]``.
    z_final : Tensor [K, ...]
        Per-state bootstrap sparse-structure latents; state ``k`` is sliced as
        ``z_final[k:k+1]``.
    pure_state_targets_K3HW : Tensor [K, 3, H, W]
        Per-state pure (background-free) RGB targets in ``[0, 1]``. Conditioning
        is built once via ``_build_trellis_cond_from_float_states`` and each
        state ``k`` uses its own row (``cond["cond"][k:k+1]``,
        ``cond["neg_cond"][k:k+1]``).
    state_indices : Sequence[int], default (0, 5)
        Which states to build. ``0`` = closed keyframe (M0), ``5`` = open
        keyframe (M5).
    seed : int, default 0
        Base seed. State ``k`` uses a fixed per-state seed ``seed + k`` for its
        SLAT noise draw, so each keyframe is deterministic and reproducible
        independently of the other states.
    device : torch.device, optional
        Compute device. Defaults to ``z_final.device`` (or CUDA when the latent
        is on CPU).

    Returns
    -------
    Dict[int, Gaussian]
        Mapping from state index to its TRELLIS Gaussian object, e.g.
        ``{0: M0, 5: M5}``.
    """
    if device is None:
        device = z_final.device if z_final.is_cuda else torch.device("cuda")

    ss_vae_decoder = pipe.models["sparse_structure_decoder"]
    d_gs = pipe.models["slat_decoder_gs"]
    d_gs_w = DGSWithParent(d_gs_frozen=d_gs).to(device)

    cond_all = _build_trellis_cond_from_float_states(pipe, pure_state_targets_K3HW)

    gauss_by_state: Dict[int, object] = {}
    for k in state_indices:
        k = int(k)
        coords4 = _occ_to_coords4(z_final[k:k + 1], ss_vae_decoder, device)
        cond_k = _cond_for_state(cond_all, k)
        z_slat = _sample_slat_feats(pipe, coords4, cond_k, device, seed=seed + k)
        gauss = _decode_gaussians(d_gs_w, z_slat, coords4)
        gauss_by_state[k] = gauss

    return gauss_by_state

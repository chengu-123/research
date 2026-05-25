"""C.7 Canonical aggregation with median warp.

Backwarps per-state move occupancy into the canonical frame via the
inverse of ``T_k`` and aggregates across K states with a robust
statistic (default: median) to suppress voxelization aliasing and
TRELLIS per-state noise. Base occupancy is aggregated in-place (base
does not move, so identity warp). Produces the canonical geometry
consumed by Stage D for visibility-aware texture completion.

Reuses ``pipelines/sajo/warp.py::batch_trilinear_warp`` for the
autograd-safe rigid warp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from ..sajo.warp import batch_trilinear_warp


@dataclass
class AggregationResult:
    canonical_base: torch.Tensor              # (D, H, W) bool
    canonical_move: torch.Tensor              # (D, H, W) bool
    contact_region: torch.Tensor              # (D, H, W) bool
    canonical_base_soft: torch.Tensor         # (D, H, W) float — pre-threshold
    canonical_move_soft: torch.Tensor         # (D, H, W) float
    per_state_assignment: torch.Tensor        # (K, D, H, W) int8


def _reduce(stack: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "median":
        vals, _ = torch.median(stack, dim=0)
        return vals
    if mode == "max":
        return stack.max(dim=0).values
    if mode == "or":
        return (stack > 0.5).any(dim=0).to(stack.dtype)
    raise ValueError(f"Unknown aggregator: {mode}")


def _invert_T(T: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(T)


def aggregate_canonical(
    O_stack: torch.Tensor,
    base_mask: torch.Tensor,
    move_mask: torch.Tensor,
    T_k: torch.Tensor,
    hp,
    upsample_resolution: Optional[int] = None,
) -> AggregationResult:
    """Produce canonical base/move volumes from assignments + T_k.

    Parameters
    ----------
    O_stack : (K, D, H, W) soft or binary occupancy (values in [0, 1]).
    base_mask, move_mask : (D, H, W) bool assignments from C.5/§5.4.
    T_k : (K, 4, 4) canonical-to-state transforms.
    hp : SegMatchHParams — uses ``warp_resolution``, ``aggregator``.
    upsample_resolution : optional override of hp.warp_resolution.

    Air voxels (not in base_mask or move_mask) get per_state_assignment -1
    whenever they are unoccupied in that state.
    """
    if O_stack.dim() != 4:
        raise ValueError(f"O_stack must be (K,D,H,W); got {tuple(O_stack.shape)}")
    K, D, H, W = O_stack.shape
    device = O_stack.device
    dtype = O_stack.dtype

    warp_res = upsample_resolution if upsample_resolution is not None else hp.warp_resolution
    do_upsample = warp_res != D

    base_f = base_mask.to(dtype)
    move_f = move_mask.to(dtype)

    # Per-state masked occupancy
    move_occ = O_stack * move_f.unsqueeze(0)                      # (K, D, H, W)
    base_occ = O_stack * base_f.unsqueeze(0)                      # (K, D, H, W)

    # Base: identity warp — aggregate directly
    canonical_base_soft = _reduce(base_occ, hp.aggregator)

    # Move: backwarp state k -> canonical via T_k^{-1}
    T_inv = torch.stack([_invert_T(T_k[k]) for k in range(K)])    # (K, 4, 4)

    if do_upsample:
        # Upsample move_occ to warp_res, warp at high res, downsample back.
        up = F.interpolate(move_occ.unsqueeze(1),
                           size=(warp_res, warp_res, warp_res),
                           mode="trilinear", align_corners=True).squeeze(1)
        warped_hires = batch_trilinear_warp(up, T_inv, resolution=warp_res)
        warped = F.avg_pool3d(warped_hires.unsqueeze(1),
                              kernel_size=warp_res // D, stride=warp_res // D).squeeze(1)
    else:
        warped = batch_trilinear_warp(move_occ, T_inv, resolution=D)

    canonical_move_soft = _reduce(warped, hp.aggregator)

    canonical_base = canonical_base_soft > 0.5
    canonical_move = canonical_move_soft > 0.5

    # Contact region = dilate(canonical_base, 1) ∩ canonical_move
    base_dil = F.max_pool3d(canonical_base.to(dtype)[None, None],
                             kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0) > 0.5
    contact_region = base_dil & canonical_move

    # Per-state assignment: 0=base, 1=move, -1=unassigned
    # v8.1 FIX (2026-04-24): the pre-v8.1 code did `per_state[k] = O_k ∩ canonical_move`
    # which is WRONG for state k > 0 — canonical_move lives in state-0 frame, so
    # intersecting directly with O_k gives only the CANONICAL-FRAME OVERLAP region
    # (e.g., only always-on move voxels). This explains the observed collapse
    # (7201 state 5: n_move=59, 7128 state 5: n_move=142, 30857 all states ~29)
    # in per_state_assignment even though canonical_move itself is ~400+ voxels.
    # Correct definition: per_state_assignment[k, w_voxel] = 1 iff
    #   (1) O_k(w_voxel) is occupied, AND
    #   (2) w_voxel corresponds to a canonical_move voxel under T_k
    #       — i.e., T_k^{-1}(w_voxel) lies in canonical_move.
    # We implement by FORWARD-warping canonical_move into state-k world frame.
    per_state = torch.full((K, D, H, W), -1, dtype=torch.int8, device=device)
    occ_bin = (O_stack > 0.5)
    # Base is static (T=identity): per_state[k] & base_mask intersected with O_k
    per_state[(occ_bin) & base_mask.unsqueeze(0)] = 0
    # Move: forward-warp canonical_move to state k frame for each k
    move_canonical = move_mask.to(dtype)                             # (D,H,W)
    # batch forward warp canonical mask through T_k for each k
    move_canonical_batch = move_canonical.unsqueeze(0).repeat(K, 1, 1, 1)  # (K,D,H,W)
    warped_move_per_state = batch_trilinear_warp(
        move_canonical_batch, T_k.to(device=device, dtype=dtype), resolution=D,
    ) > 0.5                                                          # (K, D, H, W) bool
    per_state[(occ_bin) & warped_move_per_state] = 1

    return AggregationResult(
        canonical_base=canonical_base,
        canonical_move=canonical_move,
        contact_region=contact_region,
        canonical_base_soft=canonical_base_soft,
        canonical_move_soft=canonical_move_soft,
        per_state_assignment=per_state,
    )

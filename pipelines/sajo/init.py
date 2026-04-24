"""Dual-hypothesis initialization (revolute + prismatic) for SAJO's EM.

Provides the starting point for both branches of the screw-axis
optimization: weighted PCA of contact anchors (revolute) and centroid
displacement of move voxels across states (prismatic). Both branches
produce per-state magnitudes ``phi_k`` consistent with the state-0
frame convention.
"""

from __future__ import annotations

from typing import Dict

import torch

from .anchors import anchor_coords_to_world


# ---- Helpers ---------------------------------------------------------


def _move_centroid(O_k: torch.Tensor, p_move: torch.Tensor,
                    resolution: int = 64) -> torch.Tensor:
    """Occupancy-weighted world-coord centroid of the move region in state k."""
    device = O_k.device
    dtype = O_k.dtype
    weight = O_k * p_move                                    # (D,H,W)
    total = weight.sum().clamp_min(1e-8)

    idx = torch.arange(resolution, device=device, dtype=dtype)
    ii, jj, kk = torch.meshgrid(idx, idx, idx, indexing="ij")
    cx = (weight * ii).sum() / total
    cy = (weight * jj).sum() / total
    cz = (weight * kk).sum() / total
    centroid_idx = torch.stack([cx, cy, cz])                 # (3,) voxel idx
    return centroid_idx / float(resolution - 1) - 0.5        # world coord


# ---- Revolute init ---------------------------------------------------


def init_revolute(
    anchor_coords: torch.Tensor,
    anchor_weights: torch.Tensor,
    O_stack: torch.Tensor,
    p_move: torch.Tensor,
    resolution: int = 64,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Initialize a revolute screw axis from weighted PCA of anchor positions.

    Returns a dict with ``omega``, ``q``, ``v`` (Plücker moment) and
    ``phi_k`` (signed angles of move-centroid rotation in the plane
    perpendicular to ``omega``).
    """
    device = O_stack.device
    dtype = O_stack.dtype
    K = O_stack.shape[0]

    if anchor_coords.numel() == 0:
        # Fallback: use the move-voxel principal axis as an imperfect omega_0.
        omega = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        q = torch.zeros(3, device=device, dtype=dtype)
    else:
        pts_world = anchor_coords_to_world(anchor_coords, resolution).to(device=device, dtype=dtype)
        w = anchor_weights.to(device=device, dtype=dtype).clamp_min(0.0)
        w_sum = w.sum().clamp_min(eps)
        mu = (w[:, None] * pts_world).sum(dim=0) / w_sum          # (3,)
        centered = pts_world - mu                                 # (N,3)
        cov = (w[:, None] * centered).T @ centered / w_sum         # (3,3)
        cov = 0.5 * (cov + cov.T)                                  # symmetrize
        eigvals, eigvecs = torch.linalg.eigh(cov)                  # ascending
        omega = eigvecs[:, -1]                                     # largest variance direction
        omega = omega / omega.norm().clamp_min(eps)
        q = mu

    # Per-state initial rotation angle via signed projection of
    # (c_k - q) rotated toward (c_0 - q) in the plane perpendicular to omega.
    c_0 = _move_centroid(O_stack[0], p_move, resolution)           # world (3,)
    phi_k = torch.zeros(K, device=device, dtype=dtype)
    for k in range(K):
        if k == 0:
            continue
        c_k = _move_centroid(O_stack[k], p_move, resolution)
        a = c_0 - q
        b = c_k - q
        a_perp = a - (a @ omega) * omega
        b_perp = b - (b @ omega) * omega
        cross = torch.linalg.cross(a_perp, b_perp)
        sin_theta = (cross @ omega)
        cos_theta = (a_perp * b_perp).sum()
        phi_k[k] = torch.atan2(sin_theta, cos_theta)

    v = torch.linalg.cross(q, omega)
    return {
        "omega": omega.detach(),
        "q": q.detach(),
        "v": v.detach(),
        "phi_k": phi_k.detach(),
    }


# ---- Prismatic init --------------------------------------------------


def init_prismatic(
    O_stack: torch.Tensor,
    p_move: torch.Tensor,
    resolution: int = 64,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Initialize a prismatic direction from move-centroid displacement."""
    device = O_stack.device
    dtype = O_stack.dtype
    K = O_stack.shape[0]

    c_0 = _move_centroid(O_stack[0], p_move, resolution)
    deltas = torch.zeros(K, 3, device=device, dtype=dtype)
    for k in range(K):
        c_k = _move_centroid(O_stack[k], p_move, resolution)
        deltas[k] = c_k - c_0

    mean_delta = deltas.mean(dim=0)
    norm = mean_delta.norm().clamp_min(eps)
    v_hat = mean_delta / norm

    phi_k = deltas @ v_hat                                    # (K,)
    return {
        "v_hat": v_hat.detach(),
        "phi_k": phi_k.detach(),
    }

"""SE(3) algebra for SAJO.

All functions operate on torch tensors and preserve autograd through
both the twist direction `S` and the magnitude `phi`. The revolute
exponential goes through `torch.linalg.matrix_exp` on the 4x4 twist
matrix; prismatic is closed-form; tiny-angle guard uses a truncated
series to keep gradients well-conditioned near `phi = 0`.

Conventions
-----------
* Revolute twist: `S = (omega, v) in R^6` with `||omega|| = 1`,
  `v . omega = 0`, `v = q cross omega` where `q` is any point on the
  rotation axis.
* Prismatic twist: `S = (0, v_hat)` with `||v_hat|| = 1`.
* Rigid transform: `T(phi) = exp([S] * phi) in SE(3)` as a 4x4 matrix.
"""

from __future__ import annotations

from typing import Tuple

import torch

# ---- Basic operators --------------------------------------------------


def hat_so3(omega: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric (3,) -> (3,3). Supports arbitrary leading batch dims."""
    if omega.shape[-1] != 3:
        raise ValueError(f"hat_so3 expects last dim 3, got {tuple(omega.shape)}")
    o1, o2, o3 = omega.unbind(-1)
    zero = torch.zeros_like(o1)
    row0 = torch.stack([zero, -o3, o2], dim=-1)
    row1 = torch.stack([o3, zero, -o1], dim=-1)
    row2 = torch.stack([-o2, o1, zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def twist_hat(twist: torch.Tensor) -> torch.Tensor:
    """Map a 6-vector `(omega, v)` to the 4x4 twist matrix `[S]`."""
    if twist.shape[-1] != 6:
        raise ValueError(f"twist_hat expects last dim 6, got {tuple(twist.shape)}")
    omega = twist[..., :3]
    v = twist[..., 3:]
    omega_hat = hat_so3(omega)  # (..., 3, 3)
    top = torch.cat([omega_hat, v.unsqueeze(-1)], dim=-1)  # (..., 3, 4)
    pad = torch.zeros(*twist.shape[:-1], 1, 4, device=twist.device, dtype=twist.dtype)
    return torch.cat([top, pad], dim=-2)  # (..., 4, 4)


def plucker(omega_unit: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Compute the Plücker moment `v = q x omega` for a unit axis through `q`."""
    return torch.linalg.cross(q, omega_unit, dim=-1)


# ---- Manifold projections --------------------------------------------


def project_revolute(omega: torch.Tensor, v: torch.Tensor,
                     eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project `(omega, v)` onto the revolute-twist manifold:
    `||omega|| = 1` and `v . omega = 0`."""
    norm = torch.linalg.norm(omega, dim=-1, keepdim=True).clamp_min(eps)
    omega_u = omega / norm
    dot = (v * omega_u).sum(dim=-1, keepdim=True)
    v_clean = v - dot * omega_u
    return omega_u, v_clean


def project_prismatic(v_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a prismatic direction to unit length."""
    norm = torch.linalg.norm(v_hat, dim=-1, keepdim=True).clamp_min(eps)
    return v_hat / norm


# ---- Exponential map -------------------------------------------------


_TINY_PHI = 1.0e-5


def exp_se3(twist: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """General SE(3) exponential.

    Args:
        twist: `(6,)` or `(..., 6)` tensor (unprojected; the caller is
            responsible for manifold constraints if needed).
        phi: scalar or broadcastable tensor of magnitudes.

    Returns:
        `(..., 4, 4)` homogeneous transform.

    Revolute-friendly `torch.linalg.matrix_exp` is used. Tiny-`phi`
    branch (`|phi| < 1e-5`) falls back to a second-order truncation
    `I + phi * [S] + 0.5 * (phi * [S])^2` to keep gradients smooth near
    the identity.
    """
    if twist.shape[-1] != 6:
        raise ValueError(f"exp_se3 expects twist last dim 6, got {tuple(twist.shape)}")
    S = twist_hat(twist)  # (..., 4, 4)

    phi = torch.as_tensor(phi, device=twist.device, dtype=twist.dtype)
    while phi.dim() < S.dim() - 2:
        phi = phi.unsqueeze(0)
    phi_scalar = phi[..., None, None]  # broadcastable to (..., 4, 4)

    # Eye tensor with the right batch dims
    eye = torch.eye(4, device=twist.device, dtype=twist.dtype)
    eye = eye.expand(*S.shape[:-2], 4, 4)

    # Tiny-angle fallback for stability near phi = 0
    abs_phi = phi.abs()
    if abs_phi.numel() == 1 and abs_phi.item() < _TINY_PHI:
        phi_S = phi_scalar * S
        return eye + phi_S + 0.5 * (phi_S @ phi_S)

    return torch.linalg.matrix_exp(phi_scalar * S)


def exp_revolute(omega_unit: torch.Tensor, v: torch.Tensor,
                 phi: torch.Tensor) -> torch.Tensor:
    """Build a revolute twist from projected `(omega_unit, v)` and apply
    `exp_se3`. Thin wrapper so callers don't need to concatenate."""
    twist = torch.cat([omega_unit, v], dim=-1)
    return exp_se3(twist, phi)


def exp_prismatic(v_hat: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Closed-form prismatic `T = [[I, phi * v_hat], [0, 1]]`.
    Fully autograd-friendly; does not invoke `matrix_exp`."""
    if v_hat.shape[-1] != 3:
        raise ValueError(f"exp_prismatic expects v_hat last dim 3, got {tuple(v_hat.shape)}")
    phi = torch.as_tensor(phi, device=v_hat.device, dtype=v_hat.dtype)
    while phi.dim() < v_hat.dim() - 1:
        phi = phi.unsqueeze(0)
    phi_bc = phi.unsqueeze(-1)  # (..., 1)
    t = phi_bc * v_hat  # (..., 3)

    eye = torch.eye(4, device=v_hat.device, dtype=v_hat.dtype)
    eye = eye.expand(*v_hat.shape[:-1], 4, 4).clone()
    eye[..., 0, 3] = t[..., 0]
    eye[..., 1, 3] = t[..., 1]
    eye[..., 2, 3] = t[..., 2]
    return eye


# ---- Logarithm (used for diagnostics / init sanity checks) ------------


def log_se3(T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of `exp_se3`: recover `(twist_unit, phi)` from a rigid transform.

    Assumes the input is close to a rigid transform; we do not re-project R.
    Returns (twist (6,), phi scalar).
    """
    if T.shape[-2:] != (4, 4):
        raise ValueError(f"log_se3 expects (...,4,4), got {tuple(T.shape)}")
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    # rotation angle via trace
    trace = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_phi = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    phi = torch.arccos(cos_phi)

    sin_phi = torch.sin(phi)
    safe_sin = torch.where(sin_phi.abs() < 1e-8, torch.ones_like(sin_phi), sin_phi)

    log_R = (R - R.transpose(-1, -2)) * (phi / (2.0 * safe_sin))[..., None, None]
    omega = torch.stack([log_R[..., 2, 1],
                         log_R[..., 0, 2],
                         log_R[..., 1, 0]], dim=-1)

    # phi ~ 0 -> translation-only (prismatic)
    tiny = phi.abs() < _TINY_PHI
    if torch.any(tiny):
        # for prismatic component, omega is zero and v = t
        if omega.dim() == 1:
            return torch.cat([torch.zeros_like(t), t], dim=-1), torch.zeros_like(phi)

    # V^{-1} t gives v for revolute
    eye3 = torch.eye(3, device=T.device, dtype=T.dtype).expand(*R.shape)
    half_phi = phi * 0.5
    half_cot = torch.where(
        phi.abs() < 1e-8,
        torch.ones_like(phi),
        half_phi / torch.tan(half_phi),
    )
    A = half_cot[..., None, None] * eye3
    B = -0.5 * hat_so3(omega)
    omega_outer = omega.unsqueeze(-1) * omega.unsqueeze(-2)
    C = ((1.0 - half_cot) / (phi.clamp_min(1e-8) ** 2))[..., None, None] * omega_outer
    V_inv = A + B + C
    v = (V_inv @ t.unsqueeze(-1)).squeeze(-1)

    twist = torch.cat([omega, v], dim=-1)
    return twist, phi

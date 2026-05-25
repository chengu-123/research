"""Joint kinematics: phi rollout, SE(3) build, quaternion ops, psi unpack.

All operations are differentiable so gradient flows from rendered RGB ->
SE(3) matrices -> (psi, phi) -> learnable scalars.

Quaternion convention: **wxyz** (real part first). This matches TRELLIS
``trellis/representations/gaussian/general_utils.py:build_rotation`` line
85-88 (``r = q[:, 0]`` = w, then x/y/z). Any conversion from a 3x3 rotation
matrix produces wxyz.

Canonical state shift (method.md NEW.1): the phi rollout subtracts
``u[CANONICAL_STATE_IDX]`` from the normalized progress so that
``u_shifted[c] = 0``. Render frame ``c`` uses identity SE(3); other states
use signed phi (positive past c, negative before c). For revolute this
yields opposite-direction rotation for states < c; for prismatic, reverse
translation. ``SE3_revolute`` and ``SE3_prismatic`` natively handle signed
phi via the Rodrigues / translation formulas.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from .config import CANONICAL_STATE_IDX, F_FRAMES, K_STATES


# =============================================================================
# psi_param unpack + axis projection
# =============================================================================

@dataclass
class JointParams:
    """Unpacked, differentiable joint parameters.

    axis        : Tensor [3]   unit-norm (after project_joint)
    origin      : Tensor [3]   world-space coordinate in (-0.5, 0.5)
    type_logit  : Tensor scalar  sigmoid -> prismatic probability
    theta_max   : Tensor scalar  softplus(theta_limit_raw) -> radians > 0
    disp_max    : Tensor scalar  softplus(disp_limit_raw)  -> world units > 0
    """
    axis: torch.Tensor
    origin: torch.Tensor
    type_logit: torch.Tensor
    theta_max: torch.Tensor
    disp_max: torch.Tensor


def project_joint(psi_param: torch.Tensor, eps: float = 1.0e-6) -> JointParams:
    """Unpack ``psi_param[19]`` into structured, projected joint params.

    The only projection is ``axis = normalize(axis_raw)``; the other fields
    are passed through (no clipping) so gradients reach H_joint cleanly.
    The L2 normalize on axis has well-defined gradient as long as
    ``||axis_raw|| > eps`` (Bootstrap ``stage_c_joint_init`` always returns a
    non-zero axis, and the gradient signal keeps it bounded away from zero).

    Parameters
    ----------
    psi_param : Tensor [19]
    eps : float
        Small value added to the norm denominator for numerical stability.

    Returns
    -------
    JointParams
    """
    if psi_param.shape != (19,):
        raise ValueError(f"psi_param must be [19], got {tuple(psi_param.shape)}")
    axis_raw = psi_param[0:3]
    origin = psi_param[3:6]
    type_logit = psi_param[6]
    theta_limit_raw = psi_param[7]
    disp_limit_raw = psi_param[8]
    # psi_param[9:19] reserved (zero-init, no effect here).

    axis_norm = axis_raw.norm(dim=-1).clamp_min(eps)
    axis = axis_raw / axis_norm

    theta_max = F.softplus(theta_limit_raw)
    disp_max = F.softplus(disp_limit_raw)

    return JointParams(
        axis=axis,
        origin=origin,
        type_logit=type_logit,
        theta_max=theta_max,
        disp_max=disp_max,
    )


# =============================================================================
# phi rollout (canonical-shifted progress -> per-frame radians / world units)
# =============================================================================

def linear_interp_through(values_K: torch.Tensor, n_out: int) -> torch.Tensor:
    """Piecewise-linear interpolation through ``K`` control points.

    Output ``out`` has length ``n_out``; ``out[0] = values_K[0]``,
    ``out[-1] = values_K[-1]``, and interior values are linearly
    interpolated. We use ``torch.lerp`` so gradients flow cleanly.

    Parameters
    ----------
    values_K : Tensor [K]
    n_out : int
    """
    if values_K.ndim != 1:
        raise ValueError(f"values_K must be 1D, got {tuple(values_K.shape)}")
    K = values_K.shape[0]
    if K < 2:
        raise ValueError(f"need K >= 2 control points, got K={K}")
    if n_out < K:
        raise ValueError(f"n_out={n_out} must be >= K={K}")

    device = values_K.device
    dtype = values_K.dtype

    # Sample positions t ∈ [0, K-1] (real), uniformly spaced over [0, K-1].
    t_continuous = torch.linspace(
        0.0, float(K - 1), n_out, device=device, dtype=dtype
    )                                              # [n_out]
    idx_lo = t_continuous.floor().long().clamp(0, K - 2)       # [n_out]
    idx_hi = (idx_lo + 1).clamp(0, K - 1)                       # [n_out]
    frac = (t_continuous - idx_lo.to(dtype)).clamp(0.0, 1.0)    # [n_out]
    v_lo = values_K[idx_lo]                                     # [n_out]
    v_hi = values_K[idx_hi]
    return torch.lerp(v_lo, v_hi, frac)


def phi_rollout(
    delta_phi: torch.Tensor,
    theta_max: torch.Tensor,
    disp_max: torch.Tensor,
    canonical_idx: int = CANONICAL_STATE_IDX,
    n_out_frames: int = F_FRAMES,
    n_states: int = K_STATES,
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-frame phi sequences with canonical-state shift.

    Steps (method.md section 7.7):
        1. ``delta_u_inc = softplus(delta_phi)``                    # [K-1] > 0
        2. ``u_raw = [0, cumsum(delta_u_inc)]``                     # [K] strictly increasing
        3. ``u = u_raw / u_raw[-1]``                                # [K] in [0, 1]
        4. ``u_shifted = u - u[canonical_idx]``                     # [K]; u_shifted[c] = 0
        5. ``u_render = linear_interp_through(u_shifted, n_out_frames)``  # [n_out] signed
        6. ``phi_render_rev = u_render * theta_max``                # [n_out] radians, signed
           ``phi_render_pri = u_render * disp_max``                 # [n_out] world units, signed

    Returns
    -------
    u_shifted : Tensor [n_states]
        Shifted normalized progress; ``u_shifted[canonical_idx] = 0``.
    u_render : Tensor [n_out_frames]
        Linearly interpolated u_shifted to per-frame resolution; signed.
    phi_render_rev : Tensor [n_out_frames]
        ``u_render * theta_max``; in radians, signed; ``phi_render_rev[c-mapped] = 0``.
    phi_render_pri : Tensor [n_out_frames]
        ``u_render * disp_max``;  in world units, signed; same property.
    """
    if delta_phi.shape != (n_states - 1,):
        raise ValueError(
            f"delta_phi must be [{n_states - 1}], got {tuple(delta_phi.shape)}"
        )
    if not (0 <= canonical_idx < n_states):
        raise ValueError(
            f"canonical_idx={canonical_idx} out of range [0, {n_states})"
        )

    delta_u_inc = F.softplus(delta_phi)                                  # [K-1] > 0
    zero = torch.zeros(1, device=delta_phi.device, dtype=delta_phi.dtype)
    u_raw = torch.cat([zero, torch.cumsum(delta_u_inc, dim=0)], dim=0)   # [K]
    u = u_raw / (u_raw[-1] + eps)                                         # [K] in [0, 1]
    u_shifted = u - u[canonical_idx]                                      # [K], u_shifted[c] = 0
    u_render = linear_interp_through(u_shifted, n_out=n_out_frames)       # [n_out] signed
    phi_render_rev = u_render * theta_max                                 # [n_out]
    phi_render_pri = u_render * disp_max                                  # [n_out]
    return u_shifted, u_render, phi_render_rev, phi_render_pri


# =============================================================================
# SE(3) build (revolute / prismatic) — accepts signed phi
# =============================================================================

def _skew(v: torch.Tensor) -> torch.Tensor:
    """3x3 skew-symmetric matrix from 3-vector ``v`` (Lie-algebra cross-prod).

    ★ S2 fix: build via ``torch.stack`` instead of in-place index assignment
    into a ``torch.zeros``. The old pattern (``K = zeros; K[i,j] = -v[k]``)
    relies on autograd's auto-promotion of the leaf K to a non-leaf when
    assigning a grad-requiring source. That behavior is fragile (varies
    across PyTorch versions and breaks under autocast). Stack-based
    construction is unambiguously differentiable.
    """
    if v.shape != (3,):
        raise ValueError(f"_skew expects [3], got {tuple(v.shape)}")
    # ★ Fix SD-5: use torch.zeros_like(v[0]) instead of torch.zeros(()) with
    # explicit dtype. Under autocast mixed precision, v may be promoted
    # (e.g. fp32 from a learnable param while autocast promotes intermediates
    # to fp16), and stacking 0 (fp32 scalar) with -vz (fp16 element view)
    # produces an undefined-promotion stack. zeros_like guarantees row
    # consistency with v's actual element type/device at call time.
    zero = torch.zeros_like(v[0])
    vx, vy, vz = v[0], v[1], v[2]
    row0 = torch.stack([zero, -vz, vy])
    row1 = torch.stack([vz, zero, -vx])
    row2 = torch.stack([-vy, vx, zero])
    return torch.stack([row0, row1, row2], dim=0)


def SE3_revolute(
    axis: torch.Tensor,
    origin: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """SE(3) rotation by signed angle ``phi`` around axis through ``origin``.

    Rodrigues formula:  R = I + sin(phi) * K + (1 - cos(phi)) * K^2
    Translation:        t = (I - R) @ origin   (so origin is a fixed point)
    Result:             T = [R | t; 0 0 0 1]

    Negative phi -> rotation in opposite direction (Rodrigues is naturally
    signed; sin(-phi) = -sin(phi), 1 - cos(-phi) = 1 - cos(phi)).

    Parameters
    ----------
    axis : Tensor [3]    unit vector (caller responsibility)
    origin : Tensor [3]  any point in world space
    phi : Tensor scalar  signed angle (radians)

    Returns
    -------
    T : Tensor [4, 4]
    """
    if axis.shape != (3,) or origin.shape != (3,):
        raise ValueError(
            f"axis/origin must be [3]; got {tuple(axis.shape)} / {tuple(origin.shape)}"
        )
    device = axis.device
    dtype = axis.dtype
    I3 = torch.eye(3, device=device, dtype=dtype)
    K = _skew(axis)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    R = I3 + sin_phi * K + (1.0 - cos_phi) * (K @ K)            # [3, 3]
    t = (I3 - R) @ origin                                       # [3]

    # ★ S2 fix: stack-based 4x4 assembly (no in-place index assignment).
    bottom_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)
    top = torch.cat([R, t.unsqueeze(-1)], dim=-1)               # [3, 4]
    T = torch.cat([top, bottom_row.unsqueeze(0)], dim=0)        # [4, 4]
    return T


def SE3_prismatic(axis: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """SE(3) translation by signed distance ``phi`` along ``axis``.

    Negative phi -> translation in opposite direction (no rotation).

    Parameters
    ----------
    axis : Tensor [3]    unit vector
    phi : Tensor scalar  signed displacement (world units)

    Returns
    -------
    T : Tensor [4, 4]
    """
    if axis.shape != (3,):
        raise ValueError(f"axis must be [3]; got {tuple(axis.shape)}")
    device = axis.device
    dtype = axis.dtype
    # ★ S2 fix: stack-based 4x4 assembly (no in-place index assignment).
    I3 = torch.eye(3, device=device, dtype=dtype)
    t = phi * axis                                              # [3]
    top = torch.cat([I3, t.unsqueeze(-1)], dim=-1)              # [3, 4]
    bottom_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)
    T = torch.cat([top, bottom_row.unsqueeze(0)], dim=0)        # [4, 4]
    return T


# =============================================================================
# Quaternion (wxyz convention; matches TRELLIS general_utils.build_rotation)
# =============================================================================

def quaternion_from_axis_angle(axis: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Axis-angle -> unit quaternion in wxyz order.

    q = [cos(phi/2), axis * sin(phi/2)]

    Negative phi yields the conjugate quaternion (equivalent rotation in
    the opposite direction), which is exactly what we want for canonical
    states before c.

    Parameters
    ----------
    axis : Tensor [3]    unit vector
    phi : Tensor scalar  signed angle (radians)

    Returns
    -------
    q : Tensor [4] in wxyz order
    """
    if axis.shape != (3,):
        raise ValueError(f"axis must be [3]; got {tuple(axis.shape)}")
    half = phi * 0.5
    w = torch.cos(half)
    s = torch.sin(half)
    return torch.stack([w, s * axis[0], s * axis[1], s * axis[2]], dim=0)


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``q1 * q2`` in wxyz order; supports batched ``q2``.

    Parameters
    ----------
    q1 : Tensor [4] or [B, 4]   (wxyz)
    q2 : Tensor [B, 4]           (wxyz)

    Returns
    -------
    q : Tensor [B, 4]            (wxyz)

    Notes
    -----
    Used to compose the joint's SE(3) rotation onto each move Gaussian's
    canonical quaternion before the differentiable rasterizer reads them.
    """
    if q2.ndim != 2 or q2.shape[-1] != 4:
        raise ValueError(f"q2 must be [B, 4], got {tuple(q2.shape)}")
    if q1.ndim == 1:
        if q1.shape != (4,):
            raise ValueError(f"q1 must be [4] or [B, 4], got {tuple(q1.shape)}")
        q1 = q1.unsqueeze(0).expand_as(q2)                           # [B, 4]
    elif q1.ndim == 2:
        if q1.shape != q2.shape:
            raise ValueError(
                f"q1/q2 shape mismatch: {tuple(q1.shape)} vs {tuple(q2.shape)}"
            )
    else:
        raise ValueError(f"q1 must be 1D or 2D, got {q1.ndim}D")

    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)


# =============================================================================
# Misc helpers
# =============================================================================

def weighted_feature_pool(feat: torch.Tensor, m_soft: torch.Tensor,
                           eps: float = 1.0e-6) -> torch.Tensor:
    """Move-soft-mass weighted pool over voxels.

    ``F_pool[d] = sum_i w[i] * feat[i, d]`` where ``w = m_soft / sum(m_soft)``.

    Used by the H_joint head: pooling features only at move-likely voxels
    gives the joint MLP a part-specific feature without committing to a
    hard binary mask.

    Parameters
    ----------
    feat : Tensor [N, D]
    m_soft : Tensor [N] in [0, 1]  (sigmoid of move logit)

    Returns
    -------
    F_pool : Tensor [D]
    """
    if feat.ndim != 2 or m_soft.ndim != 1 or feat.shape[0] != m_soft.shape[0]:
        raise ValueError(
            f"shape mismatch: feat={tuple(feat.shape)}, m_soft={tuple(m_soft.shape)}"
        )
    weights = m_soft / (m_soft.sum() + eps)                          # [N]
    return (weights.unsqueeze(-1) * feat).sum(dim=0)                  # [D]


def stage_detach(
    tensor: torch.Tensor,
    f_global: float,
    mode: str,
    ema_buf: torch.Tensor = None,
    f_full_detach_end: float = 0.05,
    f_ramp_end: float = 0.15,
    ema_alpha: float = 0.95,
) -> torch.Tensor:
    """Gradient-flow gate for psi during the warmup ramp-in.

    Three regions (default boundaries 5% / 15% of total iters):
      f_global < 0.05  :  ``return tensor.detach()`` (no gradient yet)
      0.05 <= f < 0.15 :  ramp / EMA mode
                          - mode='q_sample': linear blend
                              detach + rho * (tensor - detach)
                            where rho = (f - 0.05) / 0.10
                          - mode='joint':    EMA buffer pass-through
                              ema_buf = 0.95 * ema_buf + 0.05 * tensor.detach()
                              return ema_buf.clone()
      f >= 0.15        :  ``return tensor`` (full gradient)

    Parameters
    ----------
    tensor : Tensor
    f_global : float in [0, 1]   training progress
    mode : str   'q_sample' or 'joint'
    ema_buf : Tensor  required when mode='joint'
    """
    if f_global < f_full_detach_end:
        return tensor.detach()
    if f_global < f_ramp_end:
        if mode == "q_sample":
            rho = (f_global - f_full_detach_end) / (f_ramp_end - f_full_detach_end)
            return tensor.detach() + rho * (tensor - tensor.detach())
        if mode == "joint":
            if ema_buf is None:
                raise ValueError("stage_detach mode='joint' requires ema_buf")
            ema_buf.mul_(ema_alpha).add_((1.0 - ema_alpha) * tensor.detach())
            return ema_buf.clone()
        raise ValueError(f"unknown stage_detach mode: {mode!r}")
    return tensor


__all__ = [
    "JointParams", "project_joint",
    "linear_interp_through", "phi_rollout",
    "SE3_revolute", "SE3_prismatic",
    "quaternion_from_axis_angle", "quaternion_multiply",
    "weighted_feature_pool", "stage_detach",
]

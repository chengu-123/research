"""C.6 Axis refinement with contact-region principal-axis constraint.

Given the hard base/move assignment from C.5 and the joint fit from
C.4, this stage extracts the contact region (1-voxel dilation of base
intersected with move), computes its weighted PCA principal axis, and
refines the joint axis via Adam with a direction-alignment term plus a
point-through-contact-centroid term. The per-state ``phi_k`` and the
assignment are both frozen — this stage only cleans up the axis.

See spec §4.C.6. Implements the target.md G3 physical prior that "the
axis must pass through the base–move contact boundary near an anchor
set or its narrow neighborhood."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ..sajo.screw import (
    exp_prismatic,
    exp_se3,
    project_prismatic,
    project_revolute,
)
from .volumetric_fit import VolumetricFit as JointFit


@dataclass
class AxisRefineResult:
    joint_fit: JointFit
    contact_region: torch.Tensor              # (D, H, W) bool
    principal_dir: Optional[torch.Tensor]     # (3,) unit or None if contact empty
    principal_pos: Optional[torch.Tensor]     # (3,) world or None
    loss_trace: list


def extract_contact_region(
    base_mask: torch.Tensor,
    move_mask: torch.Tensor,
    dilation_kernel: int = 3,
) -> torch.Tensor:
    """``contact = dilate(base) ∩ move`` with 26-connectivity dilation."""
    if base_mask.dim() != 3 or move_mask.dim() != 3:
        raise ValueError("base_mask and move_mask must be (D,H,W)")
    pad = dilation_kernel // 2
    b = base_mask.to(torch.float32)[None, None]
    dilated = F.max_pool3d(b, kernel_size=dilation_kernel, stride=1, padding=pad).squeeze(0).squeeze(0) > 0.5
    return dilated & move_mask


def contact_principal_axis(
    contact: torch.Tensor,
    M_attn_64: torch.Tensor,
    resolution: int = 64,
    eps: float = 1e-8,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """Weighted PCA of contact voxels' world coords.

    Weights default to ``M_attn_64`` values clamped to ``[eps, 1]``.
    Returns (principal_dir, principal_pos, voxel_world_coords). Each is
    None if contact has < 3 voxels; ``voxel_world_coords`` is ``(N, 3)``.
    """
    coords_vox = contact.nonzero(as_tuple=False).to(torch.int64)   # (N, 3)
    if coords_vox.shape[0] < 3:
        return None, None, torch.zeros((0, 3), device=contact.device)

    device = contact.device
    dtype = M_attn_64.dtype
    coords_world = coords_vox.to(dtype=dtype) / float(resolution - 1) - 0.5
    weights = M_attn_64[coords_vox[:, 0], coords_vox[:, 1], coords_vox[:, 2]].clamp_min(eps)

    w_sum = weights.sum().clamp_min(eps)
    mu = (weights.unsqueeze(-1) * coords_world).sum(dim=0) / w_sum
    centered = coords_world - mu
    cov = (weights.unsqueeze(-1) * centered).T @ centered / w_sum
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    principal_dir = eigvecs[:, -1]
    principal_dir = principal_dir / principal_dir.norm().clamp_min(eps)
    return principal_dir, mu, coords_world


def _rebuild_T_k(joint_type: str,
                 omega_or_vhat: torch.Tensor,
                 v_or_unused: torch.Tensor,
                 phi_k: torch.Tensor) -> torch.Tensor:
    K = phi_k.shape[0]
    device = phi_k.device
    dtype = phi_k.dtype
    if joint_type == "revolute":
        twist = torch.cat([omega_or_vhat, v_or_unused])
        return torch.stack([exp_se3(twist, phi_k[k]) for k in range(K)])
    if joint_type == "prismatic":
        return torch.stack([exp_prismatic(omega_or_vhat, phi_k[k]) for k in range(K)])
    raise ValueError(f"Unknown joint_type: {joint_type}")


def refine_axis(
    fit: JointFit,
    base_mask: torch.Tensor,
    move_mask: torch.Tensor,
    M_attn_64: torch.Tensor,
    hp,
    resolution: int = 64,
) -> AxisRefineResult:
    """Adam refine axis (omega, q for revolute; v_hat for prismatic).

    Assignment (``base_mask``, ``move_mask``) and ``phi_k`` are frozen;
    only the axis parameters optimize. Each inner step re-projects onto
    the manifold (``||omega||=1, v.omega=0`` or ``||v_hat||=1``).
    """
    contact = extract_contact_region(base_mask, move_mask)
    principal_dir, principal_pos, contact_world = contact_principal_axis(
        contact, M_attn_64, resolution=resolution,
    )

    if principal_dir is None:
        return AxisRefineResult(
            joint_fit=fit, contact_region=contact,
            principal_dir=None, principal_pos=None, loss_trace=[],
        )

    device = fit.omega.device
    dtype = fit.omega.dtype
    principal_dir = principal_dir.to(device=device, dtype=dtype)
    principal_pos = principal_pos.to(device=device, dtype=dtype)
    contact_world = contact_world.to(device=device, dtype=dtype)
    contact_weights = M_attn_64[contact].clamp_min(1e-6).to(device=device, dtype=dtype)
    if contact_weights.sum() > 0:
        contact_weights = contact_weights / contact_weights.sum()

    loss_trace: list = []

    if fit.joint_type == "revolute":
        omega = torch.nn.Parameter(fit.omega.clone())
        q_init = fit.q.clone()
        # Remove along-axis component so we optimize free q in the plane perp to omega
        q_proj = q_init - (q_init @ omega.detach()) * omega.detach()
        q = torch.nn.Parameter(q_proj)
        opt = torch.optim.Adam([
            {"params": [omega], "lr": hp.fit_lr_axis},
            {"params": [q], "lr": hp.fit_lr_axis},
        ])
        for it in range(hp.axis_refine_iters):
            opt.zero_grad()
            omega_u = omega / omega.norm().clamp_min(1e-8)

            L_dir = hp.w_axis_dir * (1.0 - torch.abs((omega_u * principal_dir).sum()))

            diff = contact_world - q                                  # (N, 3)
            along = (diff * omega_u).sum(-1, keepdim=True) * omega_u   # (N, 3)
            perp = diff - along                                        # (N, 3)
            d2 = (perp ** 2).sum(-1)                                   # (N,)
            L_pass = hp.w_axis_pass * (contact_weights * d2).sum()

            L = hp.lambda_axis * (L_dir + L_pass)
            L.backward()
            opt.step()
            loss_trace.append(float(L.item()))

        with torch.no_grad():
            omega_u = omega / omega.norm().clamp_min(1e-8)
            q_final = q - (q @ omega_u) * omega_u
            v_final = torch.linalg.cross(q_final, omega_u)
        T_k_new = _rebuild_T_k("revolute", omega_u.detach(), v_final.detach(), fit.phi_k)
        new_fit = JointFit(
            joint_type="revolute",
            omega=omega_u.detach(), q=q_final.detach(), v=v_final.detach(),
            phi_k=fit.phi_k, T_k=T_k_new.detach(),
            L_final=fit.L_final, L_trace=fit.L_trace,
            meta={**fit.meta, "axis_refine_iters": len(loss_trace)},
        )
    else:  # prismatic
        v_hat = torch.nn.Parameter(fit.omega.clone())
        opt = torch.optim.Adam([v_hat], lr=hp.fit_lr_axis)
        for it in range(hp.axis_refine_iters):
            opt.zero_grad()
            v_hat_u = v_hat / v_hat.norm().clamp_min(1e-8)
            L_dir = 1.0 - torch.abs((v_hat_u * principal_dir).sum())
            L = hp.lambda_axis * hp.w_axis_dir * L_dir
            L.backward()
            opt.step()
            loss_trace.append(float(L.item()))
        with torch.no_grad():
            v_hat_u = project_prismatic(v_hat)
        T_k_new = _rebuild_T_k("prismatic", v_hat_u.detach(),
                                torch.zeros_like(v_hat_u), fit.phi_k)
        new_fit = JointFit(
            joint_type="prismatic",
            omega=v_hat_u.detach(),
            q=torch.zeros(3, device=device, dtype=dtype),
            v=v_hat_u.detach(),
            phi_k=fit.phi_k, T_k=T_k_new.detach(),
            L_final=fit.L_final, L_trace=fit.L_trace,
            meta={**fit.meta, "axis_refine_iters": len(loss_trace)},
        )

    return AxisRefineResult(
        joint_fit=new_fit, contact_region=contact,
        principal_dir=principal_dir.detach(),
        principal_pos=principal_pos.detach(),
        loss_trace=loss_trace,
    )

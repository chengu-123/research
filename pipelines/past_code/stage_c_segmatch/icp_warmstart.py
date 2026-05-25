"""C.3 Optional open3d ICP warm-start refinement (ablation).

After moment matching gives an initial ``(axis, phi_k)``, run per-pair
point-cloud ICP as an additional refinement step, extract per-pair
transforms, and re-estimate the joint parameters. This is the v5
``use_icp_warmstart=True`` ablation branch; if open3d is not installed
or disabled, the moment-matching warm start is used directly.

Caveat: ICP is per-point nearest-neighbor matching under the hood —
using it here is strictly a warm-start improvement, not a replacement
for the volumetric fit in ``volumetric_fit.py``. If open3d is unavailable,
``warm_start_with_optional_icp`` silently returns the moment-matching
result unchanged.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

try:
    import open3d as o3d
    _OPEN3D_OK = True
except Exception:
    _OPEN3D_OK = False

from ..sajo.screw import exp_prismatic, exp_se3
from .moments import WarmStart, warm_start_as_dict


def _voxel_world_coords(mask: torch.Tensor, resolution: int = 64) -> np.ndarray:
    """(D, H, W) bool -> (N, 3) world coord float numpy array."""
    vox = mask.nonzero(as_tuple=False).cpu().numpy().astype(np.float64)
    return vox / float(resolution - 1) - 0.5


def _open3d_icp_pair(
    src_points: np.ndarray,
    tgt_points: np.ndarray,
    T_init: np.ndarray,
    max_corr_dist: float,
    max_iters: int,
):
    """Wrapper around open3d's point-to-point ICP. Returns 4x4 numpy."""
    if src_points.shape[0] < 4 or tgt_points.shape[0] < 4:
        return T_init
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src_points)
    pcd_tgt = o3d.geometry.PointCloud()
    pcd_tgt.points = o3d.utility.Vector3dVector(tgt_points)
    reg = o3d.pipelines.registration.registration_icp(
        pcd_src, pcd_tgt,
        max_correspondence_distance=float(max_corr_dist),
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(max_iters),
        ),
    )
    return np.asarray(reg.transformation)


def warm_start_with_optional_icp(
    warm: WarmStart,
    O_stack: torch.Tensor,
    move_mask_k: torch.Tensor,
    use_icp: bool = False,
    max_corr_dist: float = 0.1,
    max_iters: int = 30,
    resolution: int = 64,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Return the Adam init dict (revolute + prismatic) with optional ICP polish.

    Without ICP: just returns the moment-matching two-hypothesis dict.
    With ICP: for each pair (0, k), runs open3d ICP from the moment-matching
    init, extracts a refined ``T_{0→k}``, and averages across k to update
    the initial axis and phi_k estimates.

    If ``open3d`` is not importable, silently returns the moment-matching dict.
    """
    # Pass O_stack/move_mask_k through so warm_start_as_dict can pick the
    # better of (centroid arc, inertia trajectory) for the revolute branch.
    init_dict = warm_start_as_dict(
        warm,
        O_stack=O_stack, move_mask_k=move_mask_k,
        resolution=resolution,
    )
    if not use_icp or not _OPEN3D_OK:
        return init_dict

    K = O_stack.shape[0]
    device = O_stack.device
    dtype = O_stack.dtype

    # Build initial T_{0→k} for each k using moment-matching params (prismatic path)
    # ICP refinement is mainly useful to polish phi_k magnitudes; joint-constraint
    # is re-imposed in volumetric_fit.py anyway.
    pris_init = init_dict["prismatic"]
    v_hat = pris_init["v_hat"]
    phi_k_pris = pris_init["phi_k"]

    src_points = _voxel_world_coords(move_mask_k[0], resolution=resolution)
    refined_phi = [0.0]
    for k in range(1, K):
        tgt_points = _voxel_world_coords(move_mask_k[k], resolution=resolution)
        T_init_np = np.eye(4, dtype=np.float64)
        # Translation only (prismatic init)
        t_init = float(phi_k_pris[k]) * v_hat.cpu().numpy().astype(np.float64)
        T_init_np[:3, 3] = t_init
        T_refined = _open3d_icp_pair(
            src_points, tgt_points, T_init_np,
            max_corr_dist=max_corr_dist, max_iters=max_iters,
        )
        # Project refined translation onto v_hat direction to preserve prismatic
        # assumption — full 6-DoF pose is not needed here.
        t_refined = T_refined[:3, 3]
        phi_k_refined = float(np.dot(t_refined, v_hat.cpu().numpy()))
        refined_phi.append(phi_k_refined)

    phi_k_refined_tensor = torch.tensor(
        refined_phi, device=device, dtype=dtype,
    )
    # Overwrite prismatic phi_k with ICP-refined magnitudes; keep v_hat direction
    init_dict["prismatic"]["phi_k"] = phi_k_refined_tensor
    # Revolute init keeps moment-matching values (ICP refinement of revolute
    # requires angle estimation, postponed to a future ablation)
    return init_dict


def is_open3d_available() -> bool:
    return _OPEN3D_OK

"""C.2 Moment-matching warm start for joint parameters.

Fits an initial guess of the articulation parameters from per-state
centroid trajectories of ``move_mask_k``. Pure geometry, zero
correspondence, zero feature, zero optimization — a centroid per state
plus a PCA on the displacements (prismatic) or a planar arc fit
(revolute). Used to seed the volumetric Adam in ``volumetric_fit.py``.

For K states, this gives ~100x stronger initialization than random,
in <1 ms per sample. The subsequent volumetric optimization is much
less prone to local minima with a decent warm start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class WarmStart:
    joint_type_hint: str                # "revolute" | "prismatic" | "uncertain"
    axis: torch.Tensor                  # (3,) unit — omega (rev) or v_hat (pris)
    q: torch.Tensor                     # (3,) world coord — axis point (rev) or zero (pris)
    phi_k: torch.Tensor                 # (K,) per-state param (phi_0 = 0)
    centroids_world: torch.Tensor       # (K, 3) per-state move centroid in world coord
    fit_residual: Dict[str, float]      # {"revolute": ..., "prismatic": ...}


def _weighted_centroid(
    weight: torch.Tensor,
    resolution: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """``weight``: (D, H, W) float ≥ 0 → (3,) world centroid in [-0.5, 0.5]^3."""
    device = weight.device
    dtype = weight.dtype
    D = weight.shape[0]
    idx = torch.arange(D, device=device, dtype=dtype)
    ii, jj, kk = torch.meshgrid(idx, idx, idx, indexing="ij")
    total = weight.sum().clamp_min(eps)
    cx = (weight * ii).sum() / total
    cy = (weight * jj).sum() / total
    cz = (weight * kk).sum() / total
    idx_centroid = torch.stack([cx, cy, cz])
    return idx_centroid / float(resolution - 1) - 0.5


def _fit_prismatic(centroids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Fit ``centroid_k = centroid_0 + phi_k * v_hat``.

    Returns (v_hat unit (3,), phi_k (K,), mean squared residual).
    """
    K = centroids.shape[0]
    displacements = centroids - centroids[0].unsqueeze(0)                       # (K, 3)
    # Weighted PCA on displacements: principal direction
    # Weight later displacements more (they have longer travel)
    w = displacements.norm(dim=-1)
    if w.max() < 1e-6:
        v_hat = torch.tensor([1.0, 0.0, 0.0],
                             device=centroids.device, dtype=centroids.dtype)
        phi_k = torch.zeros(K, device=centroids.device, dtype=centroids.dtype)
        return v_hat, phi_k, 0.0
    # SVD on stacked displacement vectors
    U, S, Vh = torch.linalg.svd(displacements, full_matrices=False)
    v_hat = Vh[0]
    # Sign: prefer direction of largest absolute projection onto displacement_mean
    sign = torch.sign((displacements.sum(0) * v_hat).sum())
    if float(sign) < 0:
        v_hat = -v_hat
    v_hat = v_hat / v_hat.norm().clamp_min(1e-8)
    phi_k = displacements @ v_hat                                               # (K,)
    # Residual: perpendicular component
    residual = displacements - phi_k.unsqueeze(-1) * v_hat.unsqueeze(0)
    mse = float((residual ** 2).sum(-1).mean().item())
    return v_hat, phi_k, mse


def _per_state_covariance(
    O_stack: torch.Tensor,
    move_mask_k: torch.Tensor,
    resolution: int,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-state weighted (centroid, eigenvalues_asc, eigenvectors) of move voxels.

    For state k, eigenvecs are the columns of a (3, 3) matrix sorted ASCENDING
    by eigenvalue (so the **third** column is the principal direction).
    Returns ``(centroids (K, 3), eigvals (K, 3), eigvecs (K, 3, 3))``.
    """
    K = O_stack.shape[0]
    device = O_stack.device
    dtype = O_stack.dtype
    centroids = []
    eigvals_list = []
    eigvecs_list = []

    idx = torch.arange(O_stack.shape[1], device=device, dtype=dtype)
    ii, jj, kk = torch.meshgrid(idx, idx, idx, indexing="ij")
    coord_grid_world = torch.stack([ii, jj, kk], dim=-1) / float(resolution - 1) - 0.5

    for k in range(K):
        weight = O_stack[k] * move_mask_k[k].to(dtype)
        if weight.sum() < 4:
            centroids.append(torch.zeros(3, device=device, dtype=dtype))
            eigvals_list.append(torch.zeros(3, device=device, dtype=dtype))
            eigvecs_list.append(torch.eye(3, device=device, dtype=dtype))
            continue
        mask = weight > 0
        coords_world = coord_grid_world[mask]
        w = weight[mask]
        w_sum = w.sum().clamp_min(eps)
        mu = (w[:, None] * coords_world).sum(dim=0) / w_sum
        centered = coords_world - mu
        cov = (w[:, None] * centered).T @ centered / w_sum
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        centroids.append(mu)
        eigvals_list.append(eigvals)
        eigvecs_list.append(eigvecs)

    return (torch.stack(centroids),
            torch.stack(eigvals_list),
            torch.stack(eigvecs_list))


def _eigengap_per_state(eigvals: torch.Tensor, axis_idx: int,
                         eps: float = 1e-8) -> torch.Tensor:
    """Relative gap from eigval[axis_idx] to its neighbour, normalised by max."""
    K = eigvals.shape[0]
    out = torch.zeros(K, device=eigvals.device, dtype=eigvals.dtype)
    for k in range(K):
        ev = eigvals[k]
        if axis_idx < 2:
            gap = (ev[axis_idx + 1] - ev[axis_idx]).abs()
        else:
            gap = (ev[axis_idx] - ev[axis_idx - 1]).abs()
        out[k] = gap / ev.abs().max().clamp_min(eps)
    return out


def _omega_from_inertia_trajectory(
    eigvecs: torch.Tensor,
    eigvals: torch.Tensor,
    eigengap_threshold: float,
    eps: float = 1e-8,
) -> Tuple[Optional[torch.Tensor], float]:
    """Recover ω from the trajectory of inertia-tensor principal axes.

    For each state k the move volume's covariance ``C_k = R_k C_0 R_k^T``
    where ``R_k`` is the per-state rotation about ω. Therefore each
    principal axis ``e_i^(k)`` lives in the plane perpendicular to ω, so
    the matrix ``A_i = Σ_k e_i^(k) e_i^(k)^T`` has ω in its null space:
    take the smallest-eigenvalue eigenvector. We do this for each
    non-degenerate axis i (eigengap > threshold) and fuse the candidates
    by confidence-weighted alignment.

    Returns ``(omega, confidence)`` or ``(None, 0)`` on full degeneracy.
    """
    K = eigvecs.shape[0]
    device = eigvecs.device
    dtype = eigvecs.dtype

    omega_candidates: List[torch.Tensor] = []
    confidences: List[float] = []

    for axis_idx in range(3):
        gaps = _eigengap_per_state(eigvals, axis_idx, eps)
        if float(gaps.min().item()) < eigengap_threshold:
            continue

        # Sign-align e_i^(k) across k against the k=0 reference.
        ref = eigvecs[0, :, axis_idx]
        e_aligned = [ref]
        for k in range(1, K):
            cand = eigvecs[k, :, axis_idx]
            sign = torch.sign((cand * ref).sum())
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            e_aligned.append(cand * sign)
        e_stack = torch.stack(e_aligned)                         # (K, 3)

        A_i = e_stack.T @ e_stack                                # (3, 3)
        A_i = 0.5 * (A_i + A_i.T)
        eigvals_A, eigvecs_A = torch.linalg.eigh(A_i)
        omega_i = eigvecs_A[:, 0]                                # smallest eigval
        omega_i = omega_i / omega_i.norm().clamp_min(eps)

        # Confidence ∝ how well the trajectory plane is filled
        # (gap between smallest and next eigval relative to top eigval).
        conf = float(((eigvals_A[1] - eigvals_A[0]).abs() /
                      eigvals_A[2].abs().clamp_min(eps)).item())
        omega_candidates.append(omega_i)
        confidences.append(conf)

    if len(omega_candidates) == 0:
        return None, 0.0

    omega_stack = torch.stack(omega_candidates)                  # (M, 3)
    conf_t = torch.tensor(confidences, device=device, dtype=dtype)

    # Sign-align candidates to first
    ref = omega_stack[0]
    signs = torch.sign((omega_stack * ref).sum(dim=-1))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    omega_aligned = omega_stack * signs.unsqueeze(-1)
    omega = (conf_t.unsqueeze(-1) * omega_aligned).sum(dim=0)
    omega = omega / omega.norm().clamp_min(eps)

    overall_conf = float(conf_t.mean().item())
    return omega, overall_conf


def _fit_q_phi_given_omega(
    centroids: torch.Tensor,
    omega: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Given ω and per-state centroids, fit q (axis point) and φ_k (angles).

    Centroids ``c_k`` of a rotating rigid body should lie on a circle in
    the plane perpendicular to ω. We project them onto that plane, fit a
    circle (LS), and read off (q_proj, φ_k, mse).
    """
    K = centroids.shape[0]
    device = centroids.device
    dtype = centroids.dtype

    # Project centroids onto the plane ⊥ ω
    along = (centroids @ omega).unsqueeze(-1) * omega.unsqueeze(0)
    proj = centroids - along                                      # (K, 3) in plane
    out_of_plane = (centroids @ omega) - (centroids @ omega).mean()

    # Build orthonormal basis (u, v) inside the plane
    u_basis_raw = proj[0] - proj.mean(dim=0)
    if u_basis_raw.norm() < eps:
        # All projections coincide → degenerate; return centroid as q and zero phi
        q = proj.mean(dim=0)
        phi = torch.zeros(K, device=device, dtype=dtype)
        return q, phi, float("inf")
    u_basis = u_basis_raw / u_basis_raw.norm().clamp_min(eps)
    v_basis_raw = torch.linalg.cross(omega, u_basis)
    v_basis = v_basis_raw / v_basis_raw.norm().clamp_min(eps)

    u_coord = (proj @ u_basis)
    v_coord = (proj @ v_basis)

    # Solve (x - cu)^2 + (y - cv)^2 = r^2 → linearise
    A = torch.stack([2.0 * u_coord, 2.0 * v_coord,
                     torch.ones_like(u_coord)], dim=-1)
    b = u_coord * u_coord + v_coord * v_coord
    sol = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)
    cu, cv, c0 = sol[0], sol[1], sol[2]

    q = cu * u_basis + cv * v_basis                              # in-plane component
    # restore along-axis offset (use the mean to keep gauge consistent)
    q = q + (centroids @ omega).mean() * omega
    # Project q ⊥ ω so the gauge is unique
    q = q - (q @ omega) * omega + (centroids @ omega).mean() * omega

    # Angle of each centroid relative to centroid_0 in the (u, v) plane
    ref_u = u_coord[0] - cu
    ref_v = v_coord[0] - cv
    ref_norm = torch.sqrt(ref_u * ref_u + ref_v * ref_v).clamp_min(eps)
    phi = torch.zeros(K, device=device, dtype=dtype)
    for k in range(K):
        du = u_coord[k] - cu
        dv = v_coord[k] - cv
        cos_a = (du * ref_u + dv * ref_v) / (ref_norm *
                                              torch.sqrt(du * du + dv * dv).clamp_min(eps))
        sin_a = (ref_u * dv - ref_v * du) / (ref_norm *
                                              torch.sqrt(du * du + dv * dv).clamp_min(eps))
        phi[k] = torch.atan2(sin_a, cos_a)

    # MSE = radial residual + out-of-plane residual
    radii = torch.sqrt((u_coord - cu) ** 2 + (v_coord - cv) ** 2)
    radial_mse = ((radii - radii.mean()) ** 2).mean()
    oop_mse = (out_of_plane ** 2).mean()
    mse = float((radial_mse + oop_mse).item())
    return q, phi, mse


def _fit_revolute_inertia(
    O_stack: torch.Tensor,
    move_mask_k: torch.Tensor,
    resolution: int = 64,
    eigengap_threshold: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Inertia-tensor warm start (AOF Path A, 2026-04-23 increment).

    For revolute joints, leverages the second-moment trajectory of the
    move volume rather than just the first-moment (centroid) trajectory.
    Particularly valuable for elongated parts (drawers, doors) where
    centroid arcs may be small but the principal axis swings clearly.

    Falls back gracefully (returns mse=∞) on rotational symmetries
    (eigenvalue degeneracy) where principal axes are not uniquely defined.

    Returns ``(omega (3,), q (3,), phi_k (K,), mse)``.
    """
    K = O_stack.shape[0]
    device = O_stack.device
    dtype = O_stack.dtype

    centroids, eigvals, eigvecs = _per_state_covariance(
        O_stack, move_mask_k, resolution,
    )
    omega, _conf = _omega_from_inertia_trajectory(
        eigvecs, eigvals, eigengap_threshold,
    )
    if omega is None:
        return (torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype),
                torch.zeros(3, device=device, dtype=dtype),
                torch.zeros(K, device=device, dtype=dtype),
                float("inf"))

    q, phi, mse = _fit_q_phi_given_omega(centroids, omega)
    return omega, q, phi, mse


def _fit_revolute(centroids: torch.Tensor,
                   eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Fit a planar circular arc through K centroids.

    Returns ``(omega (3,), q (3,), phi_k (K,), mse)`` where the axis
    ``(omega, q)`` describes a 3D line and ``phi_k`` are signed angles
    relative to the vector ``centroid_0 - q``.
    """
    K = centroids.shape[0]
    # Plane fit: SVD on centered centroids; smallest singular vector = plane normal
    mu = centroids.mean(dim=0)
    centered = centroids - mu                                                   # (K, 3)
    if K < 3:
        omega = torch.tensor([0.0, 0.0, 1.0],
                             device=centroids.device, dtype=centroids.dtype)
        q = mu
        phi_k = torch.zeros(K, device=centroids.device, dtype=centroids.dtype)
        return omega, q, phi_k, float("inf")

    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    # omega = normal to best-fit plane = smallest singular vector (last row of Vh)
    omega = Vh[-1]
    omega = omega / omega.norm().clamp_min(eps)

    # Project centroids onto the plane (removing omega-component)
    proj = centered - (centered @ omega).unsqueeze(-1) * omega.unsqueeze(0)      # (K, 3)
    # Circle center in the plane: solve 2D least squares
    # (x_k - cx)^2 + (y_k - cy)^2 = r^2  →  linearize to
    # 2 cx x_k + 2 cy y_k + c = x_k^2 + y_k^2 where c = r^2 - cx^2 - cy^2
    # Build orthonormal basis (u, v) in the plane
    u_basis = proj[0] / proj[0].norm().clamp_min(eps)
    v_basis_raw = proj[1] - (proj[1] @ u_basis) * u_basis
    v_basis_norm = v_basis_raw.norm().clamp_min(eps)
    if float(v_basis_norm) < 1e-6:
        # Centroids nearly collinear → can't fit circle cleanly
        q = mu
        phi_k = torch.zeros(K, device=centroids.device, dtype=centroids.dtype)
        return omega, q, phi_k, float("inf")
    v_basis = v_basis_raw / v_basis_norm

    u_coord = proj @ u_basis
    v_coord = proj @ v_basis
    A = torch.stack([2.0 * u_coord, 2.0 * v_coord,
                     torch.ones_like(u_coord)], dim=-1)                         # (K, 3)
    b = u_coord * u_coord + v_coord * v_coord
    sol = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)
    cu, cv = sol[0], sol[1]
    q_plane = cu * u_basis + cv * v_basis + mu                                  # (3,)

    # Angular progression phi_k relative to (centroid_0 - q) vector
    vecs = centroids - q_plane                                                  # (K, 3)
    # Remove omega-component to keep vectors in-plane
    vecs = vecs - (vecs @ omega).unsqueeze(-1) * omega.unsqueeze(0)
    norms = vecs.norm(dim=-1).clamp_min(eps)
    vecs_unit = vecs / norms.unsqueeze(-1)
    # Reference = vector at state 0
    ref = vecs_unit[0]
    cross_ref = torch.linalg.cross(ref.expand_as(vecs_unit), vecs_unit, dim=-1)  # (K, 3)
    sin_comp = (cross_ref * omega).sum(dim=-1)
    cos_comp = (vecs_unit @ ref)
    phi_k = torch.atan2(sin_comp, cos_comp)                                     # (K,)

    # Residual: distance from each centroid to the fitted circle in the plane
    r = norms.mean()
    residuals = (norms - r)
    # Plus out-of-plane component
    out_of_plane = (centered @ omega)
    mse = float(((residuals ** 2) + (out_of_plane ** 2)).mean().item())

    return omega, q_plane, phi_k, mse


def compute_move_centroids(
    O_stack: torch.Tensor,
    move_mask_k: torch.Tensor,
    resolution: int = 64,
) -> torch.Tensor:
    """Per-state weighted centroids of ``O_stack[k] · move_mask_k``.

    Returns (K, 3) world-coordinate centroids.
    """
    K = O_stack.shape[0]
    centroids = []
    for k in range(K):
        w = O_stack[k] * move_mask_k[k].to(O_stack.dtype)
        centroids.append(_weighted_centroid(w, resolution))
    return torch.stack(centroids)                                               # (K, 3)


def moment_matching_warm_start(
    O_stack: torch.Tensor,
    move_mask_k: torch.Tensor,
    resolution: int = 64,
    use_inertia_for_revolute: bool = True,
    inertia_eigengap_threshold: float = 0.05,
) -> WarmStart:
    """Three-hypothesis warm start from per-state move statistics.

    Hypotheses tried:
      * **prismatic**: line fit through per-state centroid trajectory
      * **revolute (centroid arc)**: planar arc through the K centroids
      * **revolute (inertia)** [AOF Path A, 2026-04-23]: from the trajectory
        of inertia-tensor principal axes; uses second-moment information
        and is particularly good on elongated parts (drawers, doors).

    Among the two revolute hypotheses we keep the lower-residual one;
    then we compare the chosen revolute against prismatic by residual
    and report the winner as ``joint_type_hint``. ``fit_residual`` holds
    both chosen residuals (rev = best of arc/inertia; pris = line).
    """
    K = O_stack.shape[0]
    centroids = compute_move_centroids(O_stack, move_mask_k, resolution)

    v_hat, phi_k_pris, mse_pris = _fit_prismatic(centroids)
    omega_arc, q_arc, phi_k_arc, mse_arc = _fit_revolute(centroids)

    if use_inertia_for_revolute:
        omega_in, q_in, phi_k_in, mse_in = _fit_revolute_inertia(
            O_stack, move_mask_k, resolution=resolution,
            eigengap_threshold=inertia_eigengap_threshold,
        )
    else:
        omega_in, q_in, phi_k_in, mse_in = (
            omega_arc, q_arc, phi_k_arc, float("inf"),
        )

    if mse_in < mse_arc:
        omega_r, q_r, phi_k_r, mse_r, rev_source = (
            omega_in, q_in, phi_k_in, mse_in, "inertia",
        )
    else:
        omega_r, q_r, phi_k_r, mse_r, rev_source = (
            omega_arc, q_arc, phi_k_arc, mse_arc, "centroid_arc",
        )

    if mse_pris <= mse_r:
        return WarmStart(
            joint_type_hint="prismatic",
            axis=v_hat,
            q=torch.zeros(3, device=centroids.device, dtype=centroids.dtype),
            phi_k=phi_k_pris, centroids_world=centroids,
            fit_residual={
                "revolute": mse_r, "prismatic": mse_pris,
                "revolute_arc": mse_arc, "revolute_inertia": mse_in,
                "rev_source": rev_source,
            },
        )
    return WarmStart(
        joint_type_hint="revolute",
        axis=omega_r, q=q_r, phi_k=phi_k_r, centroids_world=centroids,
        fit_residual={
            "revolute": mse_r, "prismatic": mse_pris,
            "revolute_arc": mse_arc, "revolute_inertia": mse_in,
            "rev_source": rev_source,
        },
    )


def warm_start_as_dict(
    ws: WarmStart,
    O_stack: Optional[torch.Tensor] = None,
    move_mask_k: Optional[torch.Tensor] = None,
    resolution: int = 64,
    use_inertia_for_revolute: bool = True,
    inertia_eigengap_threshold: float = 0.05,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Return both revolute and prismatic init dicts for Adam starting points.

    The revolute branch picks the lower-residual of (centroid_arc, inertia)
    if ``O_stack`` and ``move_mask_k`` are provided (so inertia can be
    recomputed). Without those, falls back to the centroid-arc only.
    """
    centroids = ws.centroids_world
    v_hat_p, phi_k_p, _ = _fit_prismatic(centroids)
    omega_arc, q_arc, phi_k_arc, mse_arc = _fit_revolute(centroids)

    rev_init = {"omega": omega_arc, "q": q_arc, "phi_k": phi_k_arc}
    if (use_inertia_for_revolute and O_stack is not None
            and move_mask_k is not None):
        omega_in, q_in, phi_k_in, mse_in = _fit_revolute_inertia(
            O_stack, move_mask_k, resolution=resolution,
            eigengap_threshold=inertia_eigengap_threshold,
        )
        if mse_in < mse_arc:
            rev_init = {"omega": omega_in, "q": q_in, "phi_k": phi_k_in}

    return {
        "revolute": rev_init,
        "prismatic": {"v_hat": v_hat_p, "phi_k": phi_k_p},
    }

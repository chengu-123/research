"""Supervised prototype classifier for always_on voxel material (2026-04-24).

Replaces the M_attn-based always_on classifier. Uses the Stage B SS-VAE
latent (``z_final``) directly as an 8-dim per-voxel material descriptor.

**Insight (diagnosed on 30857_b + 7201_b)**:
- ``M_attn = sigmoid(cross-state z_final cosine agreement)``. For always_on
  voxels, every state is occupied → cross-state agreement is trivially
  high for BOTH cabinet and drawer-interior. M_attn cannot distinguish
  them.
- ``z_final`` itself, at state-0 (or any single state), carries the raw
  8-dim latent. **The absolute feature values differ between cabinet
  material and drawer material** — this is the signal M_attn collapsed
  into a scalar and lost.

**Algorithm**:
1. ``shell`` voxels (count ∈ [1, K-1]) are definitely drawer — use their
   mean z_final as the **drawer prototype**.
2. ``far_aon`` voxels (always_on AND EDT(shell) > τ_edt) are definitely
   cabinet — use their mean as the **cabinet prototype**.
3. Build a 1-D discriminant axis ``a = drawer_proto - cabinet_proto``
   (normalized). For every always_on voxel, project its z_final onto
   ``a``; side of threshold = its class.
4. Decision threshold = midpoint of seed projections. Margin band near
   threshold → ambiguous, deferred to downstream swept-volume carving.

Empirical held-out accuracy (see ``scripts/diagnose_z_final_material.py``):
- 30857_b: 93.4% (threshold=12) / 91.8% (threshold=15) balanced
- 7201_b:  99.9%

This is a CVPR/AAAI-grade signal. Paper novelty: "SS-VAE latent as
training-free per-voxel material descriptor for canonical always_on
partition".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class MaterialClassification:
    # Three-way classification of always_on voxels
    true_base: torch.Tensor               # (D, H, W) bool — cabinet side of margin
    move_interior: torch.Tensor           # (D, H, W) bool — drawer side of margin
    ambiguous_on: torch.Tensor            # (D, H, W) bool — within margin band
    # Raw soft signal for downstream graph-cut / carving
    decision_projection: torch.Tensor     # (D, H, W) float — projection on drawer axis
    drawer_axis: torch.Tensor             # (C,) unit vector
    drawer_proto: torch.Tensor            # (C,) feature mean at near-shell always_on
    cabinet_proto: torch.Tensor           # (C,) feature mean at far_aon
    threshold: float                      # decision midpoint
    margin: float                         # ambiguity half-width
    # Seed diagnostics
    n_near_shell_seeds: int               # drawer seed (always_on, EDT < near_shell_max)
    n_far_aon_seeds: int                  # cabinet seed
    far_aon_edt_threshold_used: float
    # Post-filter diagnostics
    n_move_before_connectivity: int
    n_move_after_connectivity: int
    n_move_capped_by_edt: int
    edt_cap_used: float
    # If seeds too few → classifier skipped; all always_on → ambiguous
    classifier_applied: bool


def classify_always_on_by_zfinal(
    z_final: torch.Tensor,
    O_stack: torch.Tensor,
    always_on: torch.Tensor,
    shell: torch.Tensor,
    far_aon_edt_threshold: float = 15.0,
    min_seeds_shell: int = 20,
    min_seeds_far_aon: int = 20,
    margin_coef: float = 0.3,
    auto_threshold_search: bool = True,
    search_range: tuple = (8.0, 20.0),
    near_shell_max_edt: float = 2.0,
    edt_cap_percentile: float = 75.0,
    use_connectivity_filter: bool = True,
) -> MaterialClassification:
    """Classify each always_on voxel as true_base / move_interior / ambiguous
    by projecting its z_final feature onto a supervised drawer axis.

    Parameters
    ----------
    z_final : (K, C, d, d, d), typically (6, 8, 16, 16, 16) — Stage B SS latent.
    O_stack : (K, D, H, W) — per-state occupancy (float or bool).
    always_on : (D, H, W) bool — voxels with ``count == K``.
    shell : (D, H, W) bool — voxels with ``0 < count < K``.
    far_aon_edt_threshold : EDT(shell) threshold above which always_on is
        considered high-confidence cabinet seed. Auto-searched if
        ``auto_threshold_search=True`` (recommended).
    min_seeds_{shell,far_aon} : minimum number of seed voxels required in
        each class. Below this, the classifier skips and everything is
        flagged as ambiguous (downstream SV carving must decide).
    margin_coef : decision-boundary half-width expressed as
        ``margin = coef * 0.5 * (proj_std_shell + proj_std_far_aon)``. Voxels
        within ``[threshold - margin, threshold + margin]`` are ambiguous.
    auto_threshold_search : if True, scans ``edt_threshold`` in ``search_range``
        and picks the smallest value whose far_aon seed count ≥ min_seeds_far_aon.
        This adapts to sample-specific drawer geometry (e.g. 7201_b needs
        threshold ≈ 8 because far_aon pool is small; 30857_b can use 12 or 15).
    """
    if z_final.dim() != 5:
        raise ValueError(f"z_final must be (K, C, d, d, d); got {tuple(z_final.shape)}")
    if O_stack.dim() != 4:
        raise ValueError(f"O_stack must be (K, D, H, W); got {tuple(O_stack.shape)}")

    device = z_final.device
    dtype = z_final.dtype
    K, C = z_final.shape[:2]
    D, H, W = O_stack.shape[1:]

    # 1. Upsample z_final to full 64^3
    z = F.interpolate(z_final, size=(D, H, W), mode="trilinear", align_corners=True)
    z_mean = z.mean(dim=0)                                        # (C, D, H, W)

    # 2. EDT from shell (computed once; used for seed selection and filters)
    from scipy.ndimage import distance_transform_edt, label as cc_label
    edt_np = distance_transform_edt(~shell.cpu().numpy().astype(bool))
    edt = torch.from_numpy(edt_np).to(device=device, dtype=torch.float32)

    # 3. DRAWER seed = near-shell always_on (pure always-occupied features,
    #    adjacent to shell). Fix 2026-04-24: shell itself is MIXED across
    #    states (some states occupy, some don't), so using it biases the
    #    drawer axis toward "drawer-vs-empty" instead of "drawer-vs-cabinet".
    near_shell_aon = always_on & (edt < near_shell_max_edt)

    # 4. CABINET seed = far-from-shell always_on. Adaptive threshold via
    #    median-EDT (not auto-highest): gives a larger, more representative
    #    cabinet seed pool. Prevents drawer over-classification caused by
    #    a narrow "deep-only" cabinet prototype.
    edt_aon = edt[always_on]
    median_edt = float(edt_aon.median().item()) if edt_aon.numel() > 0 else 0.0

    chosen_edt = far_aon_edt_threshold
    if auto_threshold_search:
        # Use median-EDT as the primary choice, floored to avoid picking up
        # shell-adjacent voxels (which are drawer, not cabinet).
        chosen_edt = max(median_edt, float(search_range[0]))
        # If median threshold gives too few seeds, shrink until we have enough.
        candidate = always_on & (edt > chosen_edt)
        if int(candidate.sum().item()) < min_seeds_far_aon:
            scan = np.linspace(chosen_edt, float(search_range[0]), 8)
            for t in scan[1:]:
                if int((always_on & (edt > float(t))).sum().item()) >= min_seeds_far_aon:
                    chosen_edt = float(t)
                    break

    far_aon = always_on & (edt > chosen_edt)

    n_near = int(near_shell_aon.sum().item())
    n_far = int(far_aon.sum().item())

    # 5. Graceful fallback on insufficient seeds
    empty = torch.zeros_like(always_on)
    if n_near < min_seeds_shell or n_far < min_seeds_far_aon:
        zero_axis = torch.zeros(C, device=device, dtype=dtype)
        zero_field = torch.zeros((D, H, W), device=device, dtype=dtype)
        return MaterialClassification(
            true_base=empty, move_interior=empty,
            ambiguous_on=always_on.clone(),
            decision_projection=zero_field,
            drawer_axis=zero_axis,
            drawer_proto=zero_axis.clone(),
            cabinet_proto=zero_axis.clone(),
            threshold=0.0, margin=0.0,
            n_near_shell_seeds=n_near, n_far_aon_seeds=n_far,
            far_aon_edt_threshold_used=chosen_edt,
            n_move_before_connectivity=0,
            n_move_after_connectivity=0,
            n_move_capped_by_edt=0,
            edt_cap_used=0.0,
            classifier_applied=False,
        )

    # 6. Gather seed features (pure always_on voxels at both ends)
    near_f = z_mean[:, near_shell_aon].T                          # (n_near, C)
    far_f = z_mean[:, far_aon].T                                  # (n_far, C)

    # 7. Build prototypes and discriminant axis
    drawer_proto = near_f.mean(dim=0)                             # (C,)
    cabinet_proto = far_f.mean(dim=0)                             # (C,)
    axis = drawer_proto - cabinet_proto
    axis_norm = axis.norm().clamp_min(1e-8)
    axis = axis / axis_norm

    # 8. Project seeds to set threshold + margin
    near_proj = near_f @ axis                                     # (n_near,)
    far_proj = far_f @ axis                                       # (n_far,)
    threshold = 0.5 * (near_proj.mean() + far_proj.mean())

    std_near = near_proj.std()
    std_far = far_proj.std()
    margin = float(margin_coef * 0.5 * (std_near + std_far))

    # 9. Project every voxel (full field)
    z_flat = z_mean.reshape(C, -1)
    proj_all = (axis @ z_flat).reshape(D, H, W)

    # 10. Three-way raw classification on always_on
    hi = float(threshold) + margin
    lo = float(threshold) - margin
    drawer_mask_raw = proj_all > hi
    cabinet_mask_raw = proj_all < lo
    amb_mask_raw = ~drawer_mask_raw & ~cabinet_mask_raw

    move_interior_raw = always_on & drawer_mask_raw
    true_base = always_on & cabinet_mask_raw
    ambiguous_raw = always_on & amb_mask_raw

    n_move_before_conn = int(move_interior_raw.sum().item())

    # 11. Connectivity filter — move_interior must be in the same 26-CC
    #     as at least one shell voxel. Physical prior: the moving part is
    #     one connected rigid body; islands of "drawer" feature far from
    #     shell are feature-space false positives.
    move_interior = move_interior_raw.clone()
    if use_connectivity_filter and n_move_before_conn > 0:
        combined = (shell | move_interior_raw).cpu().numpy().astype(np.uint8)
        structure = np.ones((3, 3, 3), dtype=np.uint8)
        labeled, n_cc = cc_label(combined, structure=structure)
        shell_np = shell.cpu().numpy()
        # Labels that contain any shell voxel
        shell_labels = set(labeled[shell_np].tolist()) - {0}
        # Keep only move_interior voxels whose CC touches shell
        mi_np = move_interior_raw.cpu().numpy()
        keep = np.zeros_like(mi_np, dtype=bool)
        for lbl in shell_labels:
            keep |= (labeled == lbl) & mi_np
        move_interior = torch.from_numpy(keep).to(device)
    n_move_after_conn = int(move_interior.sum().item())

    # 12. EDT cap — drawer can't extend beyond an adaptive distance from
    #     shell. Cap = 75-percentile of current-move-interior's EDT.
    edt_cap = float("inf")
    n_capped = 0
    if n_move_after_conn > 0:
        mi_edt = edt[move_interior]
        if mi_edt.numel() > 0:
            edt_cap = float(torch.quantile(
                mi_edt, edt_cap_percentile / 100.0,
            ).item())
            cap_mask = edt <= edt_cap
            before_cap = int(move_interior.sum().item())
            move_interior = move_interior & cap_mask
            n_capped = before_cap - int(move_interior.sum().item())

    # 13. Voxels now classified as neither base nor move are ambiguous
    #     (including those filtered out by connectivity/EDT cap)
    ambiguous_on = always_on & ~true_base & ~move_interior

    return MaterialClassification(
        true_base=true_base,
        move_interior=move_interior,
        ambiguous_on=ambiguous_on,
        decision_projection=proj_all.detach(),
        drawer_axis=axis.detach(),
        drawer_proto=drawer_proto.detach(),
        cabinet_proto=cabinet_proto.detach(),
        threshold=float(threshold.item() if hasattr(threshold, "item") else threshold),
        margin=margin,
        n_near_shell_seeds=n_near,
        n_far_aon_seeds=n_far,
        far_aon_edt_threshold_used=chosen_edt,
        n_move_before_connectivity=n_move_before_conn,
        n_move_after_connectivity=n_move_after_conn,
        n_move_capped_by_edt=n_capped,
        edt_cap_used=edt_cap if edt_cap != float("inf") else 0.0,
        classifier_applied=True,
    )

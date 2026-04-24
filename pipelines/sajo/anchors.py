"""Joint-free soft base/move split and 26-connectivity contact anchors.

This module breaks SAJO's circular dependency between segmentation and
articulation: the split is a pure function of the per-state occupancy
grids from Stage B and does NOT depend on any joint parameter.

All tensor ops live on GPU (or wherever ``O_stack`` is); the only CPU
sync is for ``scipy.ndimage.label`` connected components, which is
cheap for 64^3 inputs.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---- Joint-free split -------------------------------------------------


def joint_free_split(
    O_stack: torch.Tensor,
    sigma_b: float = 0.25,
    sigma_m: float = 0.15,
    tau_b: float = 0.05,
    tau_m: float = 0.05,
    M_attn: torch.Tensor = None,
    M_attn_threshold: float = 0.3,
    M_attn_tau: float = 0.05,
    mode: str = "footprint",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Soft base/move probability maps from cross-state occupancy statistics.

    Two formulas are supported via the ``mode`` argument:

    * ``"footprint"`` (default, **bug-fixed 2026-04-22**) — the move mass is
      ``p_move = footprint - p_base`` where ``footprint(v) = max_k O_k(v)``
      is the "ever-occupied" set across states. Robust to arbitrary joint
      magnitudes: for large revolute angles or long prismatic displacements
      the move part at different states occupies disjoint voxels, so
      ``mean_k O_k`` is low and the legacy formula ``p_move = mu * sigmoid(std)``
      silently drops endpoint move voxels. The footprint complement preserves
      them. Long-prismatic-trajectory middle voxels (mu ≈ 5/6) still classify
      correctly because the std-based gate on ``p_base`` already handles them
      (std ≈ 0.37 is well above sigma_b=0.25).

    * ``"legacy"`` — the original ``p_move = mu * sigmoid((std - sigma_m)/tau_m)``
      formula, kept for ablation parity with prior publications (e.g. the SAJO
      EM baseline). Matches the pre-2026-04-22 behavior.

    M_attn parameter (DEPRECATED for joint_free_split, 2026-04-23 audit)
    -------------------------------------------------------------------
    The optional ``M_attn`` argument was historically multiplied into
    ``p_base`` (via a soft sigmoid gate) to use the Stage B v4.3 cross-stage
    semantic prior. Empirical analysis on 7201_b revealed that hinge-near-door
    voxels for revolute joints have ``mu = 1, std = 0`` (look like base by
    occupancy stats) but low ``M_attn`` (DiT semantics correctly identify
    them as door material). Multiplying by an M_attn-gate flips these voxels
    to move, which neither matches user expectation nor downstream needs (the
    voxels don't move spatially anyway, so labelling them as base is fine).
    More importantly, the original motivation — fixing long-prismatic
    trajectory middle — is **already handled by the footprint+std formula
    alone**, without M_attn.

    M_attn is therefore **kept as a parameter for backward compatibility but
    ignored by default**. The proper place for M_attn integration is
    downstream in Stage C's graph-cut segmentation refinement, where it
    combines cleanly with motion-consistency unaries.

    Parameters
    ----------
    O_stack : torch.Tensor
        ``(K, D, H, W)`` float (binary or soft).
    sigma_b, sigma_m : float
        Cross-state std thresholds for base (low std) and move (high std).
        ``sigma_m`` / ``tau_m`` are only used in ``mode='legacy'``.
    tau_b, tau_m : float
        Sigmoid temperatures (smaller = sharper transition).
    M_attn : torch.Tensor, optional
        Accepted for API stability but **NOT applied** as of 2026-04-23.
        See the docstring section above.
    M_attn_threshold, M_attn_tau : float
        Accepted for API stability; not used.
    mode : {"footprint", "legacy"}, default "footprint"
        Which formula to use. The footprint formulation fixes the revolute
        endpoint bug (documented 2026-04-22) and is the default.

    Returns
    -------
    p_base, p_move : torch.Tensor
        ``(D, H, W)`` in ``[0, 1]``.
    """
    if O_stack.dim() != 4:
        raise ValueError(f"O_stack must be (K,D,H,W), got {tuple(O_stack.shape)}")
    if mode not in ("footprint", "legacy"):
        raise ValueError(f"mode must be 'footprint' or 'legacy', got {mode!r}")
    if M_attn is not None:
        # Validate shape even though we no longer apply it, so a stale caller
        # passing the wrong shape still gets a clear error rather than silent
        # acceptance.
        if M_attn.dim() != 3:
            raise ValueError(f"M_attn must be (D,H,W), got {tuple(M_attn.shape)}")
        if M_attn.shape != O_stack.shape[1:]:
            raise ValueError(
                f"M_attn spatial shape {tuple(M_attn.shape)} must match "
                f"O_stack spatial {tuple(O_stack.shape[1:])}"
            )

    mu = O_stack.mean(dim=0)                                  # (D,H,W)
    var = ((O_stack - mu) ** 2).mean(dim=0)                   # (D,H,W)
    std = torch.sqrt(var.clamp_min(1e-12))                    # (D,H,W)

    if mode == "legacy":
        p_base = mu * (1.0 - torch.sigmoid((std - sigma_b) / tau_b))
        p_move = mu * torch.sigmoid((std - sigma_m) / tau_m)
        return p_base, p_move

    # mode == "footprint" — pure occupancy-stats formulation, no M_attn
    footprint = O_stack.max(dim=0).values                     # (D,H,W)
    p_base = mu * (1.0 - torch.sigmoid((std - sigma_b) / tau_b))
    p_move = (footprint - p_base).clamp_min(0.0)
    return p_base, p_move


# ---- 26-connectivity contact anchors ---------------------------------


def _neighborhood_max(x: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """26-neighborhood max pool on a ``(D, H, W)`` scalar volume."""
    x5 = x[None, None]
    return F.max_pool3d(x5, kernel_size=kernel, stride=1, padding=kernel // 2).squeeze(0).squeeze(0)


def extract_anchors(
    p_base: torch.Tensor,
    p_move: torch.Tensor,
    min_component_size: int = 8,
    max_components: int = 2,
    binarize_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Extract contact-anchor voxels from the soft split.

    Procedure
    ---------
    1. Binarize base/move at ``binarize_threshold``.
    2. Dilate the move mask by one voxel (26-connectivity) via 3x3x3 max pool.
    3. Contact = base AND dilated(move).
    4. Connected components (6-connectivity via ``scipy.ndimage.label``);
       drop any component with < ``min_component_size`` voxels and keep
       the top ``max_components`` by size.
    5. Weight ``w(x) = p_base(x) * max_{y in N26(x)} p_move(y)``.

    Returns
    -------
    coords : torch.Tensor
        ``(N, 3)`` int64 voxel indices into the 64^3 grid.
    weights : torch.Tensor
        ``(N,)`` float weights.
    meta : dict
        ``n_components``, ``component_sizes``, ``uncertain``
        (True when ``N < min_component_size``), ``anchor_count``.
    """
    try:
        from scipy.ndimage import label as scipy_label
    except ImportError as exc:
        raise ImportError(
            "extract_anchors requires scipy (scipy.ndimage.label)."
        ) from exc

    device = p_base.device
    dtype = p_base.dtype

    B_base = (p_base > binarize_threshold).to(dtype)
    B_move = (p_move > binarize_threshold).to(dtype)

    # Dilate move by one voxel using a 3x3x3 max pool.
    B_move_dil = (_neighborhood_max(B_move, 3) > 0.5).to(dtype)

    # Weighted max pool on p_move for the weight formula.
    p_move_max = _neighborhood_max(p_move, 3)

    B_contact = ((B_base > 0.5) & (B_move_dil > 0.5)).to(dtype)

    contact_np = B_contact.detach().cpu().numpy().astype(np.uint8)
    labels, n_labels = scipy_label(contact_np)                # 6-connectivity default

    sizes = np.bincount(labels.ravel())
    if len(sizes) <= 1:
        meta = dict(
            n_components=0,
            component_sizes=[],
            uncertain=True,
            anchor_count=0,
            note="no_contact_components",
        )
        empty_coords = torch.zeros((0, 3), dtype=torch.int64, device=device)
        empty_weights = torch.zeros((0,), dtype=dtype, device=device)
        return empty_coords, empty_weights, meta

    # Exclude background (label 0), sort descending by size, filter by min size.
    comp_sizes = sizes[1:]
    order = np.argsort(-comp_sizes)
    keep_labels: list = []
    kept_sizes: list = []
    for idx in order:
        if comp_sizes[idx] < min_component_size:
            break
        keep_labels.append(int(idx) + 1)
        kept_sizes.append(int(comp_sizes[idx]))
        if len(keep_labels) >= max_components:
            break

    if len(keep_labels) == 0:
        meta = dict(
            n_components=0,
            component_sizes=[int(s) for s in comp_sizes.tolist()],
            uncertain=True,
            anchor_count=0,
            note="no_component_above_min_size",
        )
        empty_coords = torch.zeros((0, 3), dtype=torch.int64, device=device)
        empty_weights = torch.zeros((0,), dtype=dtype, device=device)
        return empty_coords, empty_weights, meta

    mask = np.isin(labels, keep_labels)
    coords_np = np.argwhere(mask).astype(np.int64)            # (N, 3)

    coords = torch.from_numpy(coords_np).to(device=device)
    # Weight formula: p_base(x) * max_{y in N26(x)} p_move(y)
    weights = p_base[coords[:, 0], coords[:, 1], coords[:, 2]] * \
              p_move_max[coords[:, 0], coords[:, 1], coords[:, 2]]

    meta = dict(
        n_components=len(keep_labels),
        component_sizes=kept_sizes,
        uncertain=bool(coords_np.shape[0] < min_component_size),
        anchor_count=int(coords_np.shape[0]),
    )
    return coords, weights, meta


def anchor_coords_to_world(coords: torch.Tensor, resolution: int = 64) -> torch.Tensor:
    """Map integer voxel indices to world coords ``(i,j,k)/(R-1) - 0.5``."""
    return coords.to(torch.float32) / float(resolution - 1) - 0.5

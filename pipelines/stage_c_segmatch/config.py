"""Hyperparameters and result dataclasses for SegMatch v6.

v6 replaces the unbiased-fitting + overlap-cleanup flow with a
phase-based EM anchored at the maximum-displacement state, combined
with count-based partition and M_attn-classified always_on voxels,
plus late-commit swept-volume carving.

Reference: record/stageC/stagec_3.md (combined with M_attn as
always_on classifier, 2026-04-23 audit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class SegMatchHParams:
    # ---- C.0 count-based partition ------------------------------------
    count_base_threshold: int = 6                   # count(v) == K → true_base
    count_move_max: int = 1                         # count(v) ≤ count_move_max → pure move seed
    # z_final-based material classifier — DISABLED 2026-04-24 (v8 rollback audit).
    # Regression root cause: material_classifier over-classifies always_on
    # voxels as move_interior (2115/4699 = 45% for 7201 oven door), polluting
    # canonical_move → Phase 3 degenerates to identity T_k → BIC collapses to
    # DoF-penalty-only comparison → revolute mis-predicted as prismatic.
    # SAJO-style partition (used in experiment_b) correctly handles always_on
    # without per-voxel material classification. See record/stageC/v8_plan.
    use_zfinal_classifier: bool = False
    far_aon_edt_threshold: float = 15.0             # EDT dist above which always_on is cabinet seed
    zfinal_min_seeds: int = 20                      # minimum seed count (shell / far_aon)
    zfinal_margin_coef: float = 0.3                 # decision margin as fraction of seed std
    # M_attn fallback (legacy, only used if z_final unavailable)
    m_attn_base_threshold: float = 0.7              # always_on ∩ (M_attn > this) → true_base
    m_attn_move_threshold: float = 0.3              # always_on ∩ (M_attn < this) → move_interior

    # ---- C.1 anchor selection -----------------------------------------
    adaptive_anchor: bool = True                    # pick state with max exclusive voxels
    fixed_anchor_state: int = 5                     # fallback when adaptive=False
    anchor_min_hard_seed_ratio: float = 0.05        # |S_hard|/|O_k| must exceed this

    # ---- Phase 1 anchor EM --------------------------------------------
    # v8 (2026-04-24): bumped 8 → 80. Root-cause analysis of 30857/7201
    # found Phase 1 terminating at L=40k-70k (absolute garbage) because 8
    # Adam steps on a 4-6 DoF rigid-body problem cannot converge. At 80
    # iters with default lr=1e-2 and Adam momentum, convergence on
    # well-conditioned single-state anchor is reliable (tested empirically).
    phase1_iters: int = 80
    phase1_lr_axis: float = 1.0e-2
    phase1_lr_phi: float = 5.0e-3

    # ---- Phase 2 sequential propagation --------------------------------
    phase2_iters_per_state: int = 5
    phase2_lr_axis: float = 5.0e-3                  # slower, we already have good T_anchor
    phase2_lr_phi: float = 5.0e-3

    # ---- Phase 3 global relax -----------------------------------------
    # v8: 30 → 50 → 150. Pairs with higher phase1 convergence; prevents
    # Phase 3 from being the sole escape route from a bad Phase 1 attractor.
    # 150 iters gives revolute room to escape the pris-like local min
    # observed on 7201 (door stuck at ~0.2 rad instead of true π/2).
    phase3_iters: int = 150
    # v8: lr_axis 2e-3 → 5e-3 (moderate boost for direction learning)
    # lr_phi 2e-3 → 5e-3 (match axis lr; needed for large-angle rev fits)
    phase3_lr_axis: float = 5.0e-3
    phase3_lr_phi: float = 5.0e-3
    # v8: monotonicity soft penalty used in fit_volumetric. Was dead config;
    # now active as `λ·Σrelu(−Δφ)²`. λ=10 aggressively discourages zigzag
    # phi_k trajectories (root cause of 7128 Phase 3 rev collapse). Set to 0
    # to disable (e.g., for datasets with genuinely non-monotonic articulation).
    monotonicity_lambda: float = 10.0

    # ---- Symmetric hypothesis tracking --------------------------------
    symmetric_eigenvalue_ratio: float = 0.95        # λ_2/λ_1 > this → treat as symmetric
    symmetric_n_hypotheses: int = 8                 # candidates on symmetry axis

    # ---- Canonical move voting ----------------------------------------
    vote_method: str = "volume_conservation"        # "volume_conservation" | "hard_majority"
    hard_vote_threshold: int = 3                    # ceil(K/2) for K=6 is 3
    hard_vote_switch_after_epoch: int = 1           # switch soft after this many outer epochs
    soft_vote_beta: Optional[float] = None          # None = auto from median residual

    # ---- Swept-volume carving -----------------------------------------
    swept_n_samples: int = 50                       # φ samples for SV
    swept_phi_margin: float = 0.05                  # extend φ range by this fraction
    base_alpha_lower: float = 0.3                   # |base_final| ≥ α·|O_0|
    carve_warning_coef: float = 0.01                # soft penalty during Phase 1-2

    # ---- Segmentation refine ------------------------------------------
    lambda_attn: float = 1.0                        # graph-cut M_attn unary
    lambda_motion: float = 2.0                      # graph-cut motion-consistency unary
    # persistence term DISABLED 2026-04-24 (v8): for prismatic drawers with
    # K·Δ < L, drawer_interior_always_on has persistence=1 (count=K) but is
    # NOT base — it's the drawer's stay-inside-cabinet portion. The
    # persistence term pushed these voxels to base, which is the direct
    # cause of 30857's n_move_voxels_final=0 failure. Setting to 0 here
    # neutralises it without changing the downstream signature.
    lambda_persistence: float = 0.0
    lambda_smooth: float = 1.0
    lambda_smooth_adaptive: bool = False
    logit_eps: float = 1.0e-3
    # v8: 0.3 → 0.0. Old threshold required a voxel to be occupied in > 30%
    # of states (≥ 2 of K=6) to be graph-cut active. For drawers/doors with
    # large displacement, shell-endpoint voxels occupied in only 1 state were
    # forcibly assigned base (U_move = +inf), collapsing move to zero.
    # 0.0 means "any footprint voxel" = (count > 0), matching the physical
    # semantics of "all ever-occupied voxels participate in segmentation".
    active_thresh: float = 0.0
    # M_attn logit hard clip (v8): prevents saturated voxels (M_attn ≈ 0 or 1)
    # from contributing |logit| values so large they overwhelm motion + Potts
    # terms. ±4 gives ~e^-4 = 1.8% tolerance, matches Agent 2 recommendation.
    logit_attn_clip: float = 4.0

    # ---- v8.1 NOVELTY: DiT 1024-dim hidden-state MRF priors ----
    # Loaded from `stage_b_dir/dit_hidden.pt` (produced by Stage B
    # capture_dit_hidden_states with capture_dit_hidden=True in configs/v1.yaml).
    # Two signals:
    #   (1) Prototype-projection p_move_dit(v) via shell-vs-far-aon Fisher axis
    #   (2) Cross-seed variance s_boundary(v) as articulation-boundary cue
    # See pipelines/stage_c_segmatch/dit_prior.py for derivation and
    # seg_refine.py run_graph_cut() for MRF integration.
    # If dit_hidden.pt is missing or use_dit_prior=False, graph-cut falls back
    # to motion + M_attn + persistence unary (pre-v8.1 behaviour / ablation).
    use_dit_prior: bool = True
    dit_prior_blocks: Optional[List[int]] = None          # None → use all in dit_hidden.pt
    dit_prior_far_aon_edt: float = 3.0                    # EDT voxels for cabinet seeds
    dit_prior_min_seeds: int = 20                         # min voxel count per seed class
    dit_prior_projection_temperature: float = 0.1         # sigmoid sharpness on score
    lambda_dit_proto: float = 1.0                         # drawer-prototype logit unary weight
    lambda_dit_boundary: float = 0.5                      # cross-seed variance unary weight

    # ---- Axis refine --------------------------------------------------
    lambda_axis: float = 0.5
    w_axis_dir: float = 1.0
    w_axis_pass: float = 0.5
    axis_refine_iters: int = 20

    # ---- Canonical aggregation ----------------------------------------
    warp_resolution: int = 64
    aggregator: str = "median"

    # ---- Contact band detection (downgraded overlap_cleanup) -----------
    contact_band_radius: int = 2

    # ---- Generic ------------------------------------------------------
    resolution: int = 64
    device: str = "cuda"
    dtype_str: str = "float32"

    # Legacy v5 compatibility: volumetric_fit single-shot may be reused
    # by Phase 1/2/3 via fit_volumetric. Keep these names for backward-
    # compatible helper interfaces (reused internally).
    fit_inner_steps: int = 80
    fit_lr_axis: float = 1.0e-2
    fit_lr_phi: float = 5.0e-3
    fit_weight_decay: float = 0.0
    fit_outer_epochs: int = 1                       # driver does phases explicitly

    # Kept for signature-stability callers (ignored in v6)
    split_sigma_b: float = 0.25
    split_sigma_m: float = 0.15
    split_tau_b: float = 0.05
    split_tau_m: float = 0.05
    split_mode: str = "footprint"
    split_M_attn_threshold: float = 0.3
    split_M_attn_tau: float = 0.05
    p_threshold: float = 0.5
    dit_block: int = 18
    use_moment_warm_start: bool = True
    use_icp_warmstart: bool = False
    icp_max_correspondence_dist: float = 0.1
    icp_max_iters: int = 30

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.dtype_str == "float32":
            return torch.float32
        if self.dtype_str == "float64":
            return torch.float64
        raise ValueError(f"Unknown dtype_str: {self.dtype_str}")


@dataclass
class StageCResult:
    joint_type: str
    omega: torch.Tensor
    q: torch.Tensor
    v: torch.Tensor
    T_k: torch.Tensor
    phi_k: torch.Tensor
    canonical_base: torch.Tensor
    canonical_move: torch.Tensor
    contact_region: torch.Tensor
    per_state_assignment: torch.Tensor
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Diagnostics:
    joint_type_selected: str
    bic_rev: float
    bic_pris: float
    bic_margin: float
    volumetric_loss_rev: float
    volumetric_loss_pris: float
    n_move_voxels_initial: int
    n_move_voxels_final: int
    n_base_voxels_initial: int
    n_base_voxels_final: int
    n_flips: int
    n_overlap_deleted: int
    icp_used: bool
    warm_start_used: str
    # v6 additions
    anchor_state_idx: int = -1
    n_always_on_total: int = 0
    n_true_base_initial: int = 0
    n_move_interior_initial: int = 0
    n_ambiguous_on_initial: int = 0
    phase1_final_loss: float = 0.0
    phase2_final_loss: float = 0.0
    phase3_final_loss: float = 0.0
    swept_carving_triggered: bool = False
    swept_lower_bound_protected: bool = False
    # v8 additions: per-hypothesis Phase-1 losses (was hidden in min())
    phase1_loss_rev: float = 0.0
    phase1_loss_pris: float = 0.0

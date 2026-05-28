"""Stage C configuration dataclass.

All hyperparameters in one place. Defaults are tuned to the v3.3.6 Stage B
output statistics on sample 30857 (prismatic drawer) and 7201 (revolute
oven). Loaded from `configs/v1.yaml` under the `stage_c` block when used
through Bootstrap, or constructed directly for unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StageCConfig:
    # ---- General ----------------------------------------------------------
    resolution: int = 64                    # voxel grid side
    canonical_state_idx: int = 2            # c in method.md NEW.1 (default 2)

    # ---- Move centroid extraction (move_geometry.py) ----------------------
    min_move_voxels_per_state: int = 30     # below this, state's move treated as "missing"
    soft_centroid_threshold: float = 0.1    # for P_move_evidence_per_state weighted centroid

    # ---- Joint type detection (joint_type_detect.py) ----------------------
    type_decision_margin: float = 0.035     # abs log(best_pris / best_rev) threshold
                                            # below this, keep type uncertain but still emit both branches
    arc_min_states: int = 4                 # need >= 4 centroids to fit a circle reliably
    ransac_iters_line: int = 50             # RANSAC iterations for line fit
    ransac_inlier_threshold: float = 0.02   # world-unit distance for line inlier

    # ---- Axis warm start (axis_fit.py) -----------------------------------
    # When type is revolute and centroid fit fails -> fallback to motion corridor
    # principal axis (PCA of M_motion_corridor_64 > corridor_pca_threshold voxels).
    corridor_pca_threshold: float = 0.1
    axis_origin_fallback_to_centroid: bool = True  # if circle fit fails, use bbox-centroid + axis projection

    # ---- Anchor extraction (anchor_extract.py) ---------------------------
    anchor_corridor_threshold: float = 0.1       # M_motion_corridor_64 > this
    anchor_base_threshold: float = 0.5           # O_base_canonical / P_base_canonical > this
    anchor_dilate_radius: int = 1                # voxel dilation after intersection
    anchor_target_count: int = 48                # FPS down-sample to this many
    anchor_min_count: int = 8                    # if fewer survive, expand corridor threshold

    # ---- Phi fit (phi_fit.py) --------------------------------------------
    phi_enforce_monotone: bool = True            # if centroid projections non-monotone,
                                                 # smooth-monotone via PAV (pool-adjacent-violators)
    phi_min_gap: float = 1e-3                    # min positive gap between adjacent u values
                                                 # to keep softplus inverse well-defined

    # ---- Theta / disp limit init (encode_joint.py) -----------------------
    theta_limit_margin: float = 1.3              # observed_max_angle * 1.3 -> theta_max init
    disp_limit_margin: float = 1.3               # observed_max_disp * 1.3 -> disp_max init
    theta_limit_min: float = 0.3                 # rad, lower bound (~17 degrees)
    disp_limit_min: float = 0.05                 # world unit, lower bound

    # ---- Confidence aggregation (confidence.py) --------------------------
    conf_w_type: float = 0.30        # type fit margin contribution
    conf_w_axis: float = 0.30        # axis fit residual contribution
    conf_w_centroid: float = 0.20    # centroid stability (monotonicity + std) contribution
    conf_w_base: float = 0.20        # base completeness contribution

    # ---- IO + viz ---------------------------------------------------------
    save_viz: bool = True            # write HTML / npy under {out_dir}/viz/
    save_diagnostics_json: bool = True

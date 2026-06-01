"""Stage D hyperparameters and frozen constants.

All numeric values that govern Stage D's behaviour live here. Constants
that come from TRELLIS-image-large / Wan2.2-I2V-A14B model checkpoints are
in the upper section (do not change without re-checking the checkpoints).
Tunable training hyperparameters are in StageDConfig below.

The pipeline_v3 / method_v3.md descriptions in record/ are the source of
truth for *semantics*; this file is the source of truth for *numbers*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# =============================================================================
# Frozen constants (from TRELLIS / Wan2.2 checkpoint configs)
# =============================================================================

# --- TRELLIS-image-large: ss_flow_img_dit_L_16l8_fp16.json ---
TRELLIS_SS_RES: int = 16                  # SparseStructureFlowModel.resolution
TRELLIS_SS_PATCH: int = 1                 # patch_size; token grid = 16^3 = 4096
TRELLIS_SS_IN_CH: int = 8                 # SS-DiT in_channels (operates on SS-VAE latent)
TRELLIS_SS_MODEL_CH: int = 1024
TRELLIS_SS_NUM_BLOCKS: int = 24
TRELLIS_SS_DTYPE_FP16: bool = True
TRELLIS_SS_SIGMA_MIN: float = 1.0e-5
TRELLIS_SS_T_SCHEDULE_MEAN: float = 1.0   # logit-normal mean (training schedule)
TRELLIS_SS_T_SCHEDULE_STD: float = 1.0    # logit-normal std

# --- TRELLIS SS-VAE decoder ---
TRELLIS_OCC_RES: int = 64                 # SS-VAE decoded occupancy resolution
TRELLIS_LATENT_TOKENS: int = 4096         # 16^3
TRELLIS_LATENT_GRID: int = 16             # token grid side

# --- TRELLIS SLAT VAE decoder (D_GS) ---
TRELLIS_DGS_N_GAUSS_PER_VOXEL: int = 32
TRELLIS_DGS_AABB_MIN: float = -0.5        # world space lower bound
TRELLIS_DGS_AABB_MAX: float = 0.5         # world space upper bound

# --- TRELLIS DINOv2 cond (image_size=518) ---
TRELLIS_DINO_DIM: int = 1024
TRELLIS_DINO_N_TOKENS: int = 1374         # 518/14 = 37 patches/axis -> 37*37 = 1369 + cls/regs

# --- Wan2.2-I2V-A14B: wan_i2v_A14B.py ---
WAN_VAE_STRIDE: Tuple[int, int, int] = (4, 8, 8)   # (T, H, W) stride
WAN_PATCH_SIZE: Tuple[int, int, int] = (1, 2, 2)    # DiT patch
WAN_BOUNDARY_NORMALIZED: float = 0.900               # in [0, 1]; in [0, 1000) -> 900
WAN_NUM_TRAIN_TIMESTEPS: int = 1000
WAN_LATENT_CH: int = 16
WAN_Y_CH: int = 20                                    # 4-ch mask + 16-ch VAE latent
WAN_DIT_DIM: int = 5120                               # A14B hidden dim
WAN_DIT_NUM_LAYERS: int = 40
WAN_T5_DIM: int = 4096
WAN_TEXT_LEN: int = 512                               # padded T5 text length

# --- Stage A default output dimensions used by legacy callers/tests.
# Runtime Stage D loads the actual H/W from Bootstrap artifacts.
F_FRAMES: int = 21                                    # frame count; F % 4 == 1
H_PIXEL: int = 464
W_PIXEL: int = 832
H_LATENT: int = H_PIXEL // WAN_VAE_STRIDE[1]          # 58
W_LATENT: int = W_PIXEL // WAN_VAE_STRIDE[2]          # 104
F_LATENT: int = (F_FRAMES - 1) // WAN_VAE_STRIDE[0] + 1  # 6

# --- Bootstrap state sampling ---
K_STATES: int = 6                                      # BMCSA K
STATE_INDICES: Tuple[int, ...] = (0, 4, 8, 12, 16, 20) # 21 frames -> 6 states

# --- Canonical state (NEW.1) ---
# Phi sequence's zero point: phi[CANONICAL_STATE_IDX] = 0; other states reach
# their pose by SE(3)(axis, origin, phi[k] - phi[c]). c=2 picked because
# TRELLIS reconstructs move geometry more accurately when DINOv2 sees the
# part partially exposed (vs s_0 closed). See method.md NEW.1.
CANONICAL_STATE_IDX: int = 2

# --- Joint-head output unpacking (psi_param layout) ---
# psi_param[0:3]   axis (unit vector after project_joint)
# psi_param[3:6]   origin (world-space coordinate in [-0.5, 0.5])
# psi_param[6]     type_logit (sigmoid -> prismatic probability)
# psi_param[7]     theta_limit_raw (softplus -> radians, default ~pi/2)
# psi_param[8]     disp_limit_raw  (softplus -> world units, default ~0.3)
# psi_param[9:19]  reserved for future use (10 floats)
PSI_PARAM_DIM: int = 19


# =============================================================================
# Tunable training hyperparameters
# =============================================================================

@dataclass
class StageDConfig:
    """Stage D training hyperparameters.

    Field groups (matches method_v3.md sections):
      - Phase boundaries:     warmup_g_minus / warmup_g0 / main_g1a / main_g1b / transition
      - Optimizer:            AdamW, separate LR for adapter / heads / scalars
      - W-RFSDS:              tau sampling (inv-CDF logit-normal), CFG schedule
      - Loss weights:         per-phase lambdas
      - Gates (BinaryConcrete temperatures): T_g, T_m schedule
      - Periodic Stage C.5 silhouette check
      - Type vote (S3):       trigger iter, confidence threshold, dual-clone budget
    """

    # ---- Phase boundaries (fraction of total iters) ----
    f_warmup_g_minus_end: float = 0.05
    f_warmup_g0_end: float     = 0.10
    f_main_g1a_end: float      = 0.50    # type vote happens at this point
    f_main_g1b_end: float      = 0.65    # by here type is committed
    f_transition_end: float    = 0.75    # P1 stops; P2 takes over (Stage F handles P2)
    # Stage D itself runs from 0 to f_transition_end. Past that is Stage F (P2).
    total_iters: int = 10000

    # ---- Adapter / head architecture ----
    adapter_hidden_dim: int = 256        # zero-init MLP hidden dim
    adapter_blocks: Tuple[int, int, int] = (14, 16, 18)
    head_hidden_dim: int = 512
    # Head output-projection init gain (xavier). 0.0 = zero-init: heads output 0,
    # so d(head_out)/d(feat)=0 at init and the shared SS-DiT adapter starts starved.
    # Keep this small and nonzero so the adapter receives task gradient through
    # H_sup/H_part/H_joint from the first active head phase.
    head_out_init_gain: float = 0.1
    # Feature dim at each voxel: 3 * 1024 (block hidden) + 3*2*F (Fourier PE) + 1 (occ logit)
    fourier_num_freqs: int = 6           # ★ S4 fix: feat_dim is derived from this

    # ---- Optimizer ----
    lr_adapter:  float = 1.0e-3
    lr_head:     float = 1.0e-3
    lr_scalar:   float = 5.0e-4          # Delta_z_s only
    lr_gate:     float = 1.0e-3          # alpha_g / alpha_m
    lr_joint:    float = 1.0e-4          # psi_param / delta_phi
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    # Freeze the revolute HINGE (psi axis[0:3] + origin[3:6]) at the Stage-C
    # geometric init by zeroing its grad each step; SDS then refines only
    # range (psi[7]) + gates. Diagnosis: the axis DIRECTION already converges to
    # GT and range ~= 90, but the pivot drifts under high-variance video-SDS ->
    # wrong door arc + axis jitter. Pinning the hinge geometrically (codex-rec)
    # fixes both. type_logit[6] + range[7] + disp[8] stay free.
    freeze_joint_hinge: bool = False
    # Also pin the rotation RANGE (theta = psi[7]) at the Stage-C geometric
    # estimate. With ONLY the hinge frozen, the free SDS + H_joint range residual
    # blows theta up (90 -> 128 deg by iter 120, job 5822) -> the video-SDS is a
    # poor estimator for EVERY kinematic DoF. Freeze range too; SDS refines only
    # the gates (segmentation). The Stage-C closed->open angle is the geometric
    # range (= GT 90 for 7201).
    freeze_joint_range: bool = False
    # Freeze ONLY the axis DIRECTION (psi[0:3]) at the Stage-C geometric init,
    # leaving origin (psi[3:6]) + range (psi[7]) + qpos (delta_phi) + gates
    # trainable. This is the FreeArt3D practice (opt_dir=False): the hinge axis
    # direction comes from the reliable geometric init and is NOT optimized by
    # the high-variance video-SDS, which otherwise drags a correct-but-low-
    # confidence axis off toward diagonals (7201: Stage-C axis sub_confidence
    # =0.13 but its value [1,0,0] == GT; free-joint run 5851 wandered the axis to
    # [0.7-0.97, 0.2-0.58, *] and theta to 115deg). Unlike freeze_joint_hinge
    # (which also pins origin), this leaves the pivot free. The forward detaches
    # psi[0:3] AND drops the H_joint residual there via psi_for_warp[0:3].detach().
    freeze_joint_axis_only: bool = False

    # ---- W-RFSDS Wan ----
    wan_backend: str = "fun_inp"          # "fun_inp" is the active Stage D contract
    render_contract_only: bool = False    # diagnostics only; no Wan load or optimization
    fun_inp_config_path: Optional[str] = None
    wan_expert_mode: str = "both"         # "both" or "low_only"; low_only is for smoke runs
    wan_offload_dit: bool = False
    wan_teacher_device_id: int = -1       # dual-GPU: put Wan experts+T5 on cuda:{id}; -1 = same GPU as student
    wan_sds_resolution_hw: Optional[List[int]] = None

    # CFG schedule (CHORD A.1: 25 -> 12 linear decay over training).
    cfg_warmup_g0: float = 25.0
    cfg_main_g1_end: float = 20.0
    cfg_transition_end: float = 16.0
    cfg_p2_default: float = 12.0

    # tau sampling: inverse-CDF of logit-normal(mean, std). For P1 we use
    # the SS-DiT training schedule logit-normal(1, 1) (mode at sigmoid(1) ~= 0.73).
    # For P2 (texture, lower noise) we shift to logit-normal(0, 1) (mode 0.5).
    tau_logit_normal_mean_p1: float = 1.0
    tau_logit_normal_std_p1:  float = 1.0
    tau_logit_normal_mean_p2: float = 0.0
    tau_logit_normal_std_p2:  float = 1.0

    # SDS gradient stability
    sds_residual_clip_norm: Optional[float] = None  # None = no clip on residual

    # ---- t_ss (SS-DiT one-step refiner inner timestep) ----
    # Mid-flow region for adapter feature capture.
    t_ss_warmup_g0_fixed: float = 0.30
    t_ss_main_low:  float = 0.25
    t_ss_main_high: float = 0.55
    t_ss_transition_low:  float = 0.20
    t_ss_transition_high: float = 0.40

    # ---- BinaryConcrete temperatures (g, m gate sharpness) ----
    T_g_warmup: float = 1.5
    T_g_main_end: float = 0.6
    T_g_transition_end: float = 0.2
    T_m_warmup: float = 1.5
    T_m_main_end: float = 0.6
    T_m_transition_end: float = 0.2
    alpha_g_init_logit: float = 3.0

    # ---- Loss weights (phase-gated; see schedules.py) ----
    # Pixel anchors are RE-ENABLED. Targets are now per-state TRELLIS
    # keyframe-model renders (M0 closed / M5 open) rendered from the same camera
    # as StageD's recon -> same TRELLIS frame/scale/texture, so the
    # silhouette+luminance-edge anchor measures pure articulation error (valid),
    # unlike real-GT which injected a size mismatch.
    lambda_first: float = 0.1           # frame 0 anchor to keyframe-render M0 (closed)
    lambda_last: float = 0.1            # frame F-1 anchor to keyframe-render M5 (open)
    lambda_contact: float = 0.2         # axis-through-anchor band
    lambda_gate: float = 0.05           # rounds g, m to {0, 1}
    lambda_shell_main: float = 0.02     # shell sparsity (D-v3.14)
    lambda_z: float = 1.0e-3            # Delta_z_s L2 stability
    lambda_m_prior_warmup: float = 0.5  # BCE(alpha_m, M_attn_boot) in warmup_g0
    lambda_m_prior_main: float = 0.0    # optional weak Stage-B move-prior anchor after warmup
    # lambda_sds, lambda_lat, lambda_rgb come from schedules.schedule_w_rfsds_weights
    # Global multiplier on the scheduled W-RFSDS (SDS) weight. The single-view
    # video-SDS PREFERS a static render (collapsing the move set to empty), because
    # opening the door reveals interior the frozen canonical recon renders poorly.
    # The endpoint anchors (L_first/L_last) must out-vote that pull; scaling the SDS
    # down gives the reliable geometric anchors authority. 1.0 = unscaled.
    lambda_sds_scale: float = 1.0
    # Motion ownership from the first/last keyframe luminance difference. This
    # supervises WHICH pixels must be rendered by the move branch, instead of
    # only asking the final composite image to match the endpoints.
    lambda_move_cover: float = 0.0
    lambda_move_suppress: float = 0.0
    # Anti-collapse move-mass floor is disabled. It keeps mass alive but does not
    # identify the correct part, and previous runs showed it can preserve a wrong
    # ownership solution.
    lambda_move_floor: float = 0.0
    move_floor_frac: float = 0.08
    lambda_move_ceiling: float = 0.0
    move_ceiling_frac: float = 0.20
    # Visible-base anchor: six axis-aligned first-hit rays over the Stage-B
    # base_move_preview split. A ray anchors only when its first hit is preview
    # base; if move wins the first hit, that ray contributes no base anchor.
    # These voxels are hard-clamped to m=0 in Stage D, leaving all non-anchor
    # base/move ownership trainable.
    base_anchor_enable: bool = False
    base_anchor_logit: float = -20.0
    base_anchor_preview_dir: Optional[str] = None
    base_anchor_preview_base_thresh: float = 0.3
    base_anchor_preview_move_thresh: float = 0.3
    gate_hardening_start_frac: float = 0.45
    gate_hardening_ramp_frac: float = 0.15

    # ---- Lambda ramps (sup / part / joint heads) ----
    lambda_sup_max:   float = 0.3
    lambda_part_max:  float = 0.6
    lambda_joint_max: float = 0.5
    # Head gate-logit residual bound: H_*_out = bound * tanh(pre). The move logit is
    # b = alpha_m + lambda_part * H_part_out, so the head can move b by at most
    # +-(bound * lambda_part). The bootstrap seeds alpha_m[rod] ~= +1.9 (mislabeled
    # move); to let the head flip it to base (b < 0) the head must reach < -1.9, so
    # bound * lambda_part_max must exceed ~1.9. With bound=4, lambda_part_max=0.6 the
    # head authority is +-2.4 -> enough to overturn the wrong prior. (Was 2.0 x 0.3 =
    # +-0.6, far too weak; the rod could never flip.)
    head_logit_residual_bound: float = 4.0
    # L2 on the pre-tanh head pre-activations. This keeps tanh from saturating
    # into a frozen +/-bound residual, which otherwise kills gradients to both
    # head weights and the upstream SS-DiT adapters.
    lambda_head_leash: float = 1.0e-2
    head_saturation_abs: float = 4.0
    require_live_head_adapter_grads: bool = False
    live_grad_check_start_frac: float = 0.12
    live_grad_min_norm: float = 1.0e-12

    # ---- Multi-view supervision (keyframe-render anchors) ----
    mv_enable: bool = True
    mv_canonical_ratio: float = 0.5      # P(use canonical camera) per iter; rest are random
    mv_azi_min_deg: float = -37.5        # random-view azimuth lower bound (deg)
    mv_azi_max_deg: float = 52.5         # random-view azimuth upper bound (deg)
    mv_ele_min_deg: float = 30.0
    mv_ele_max_deg: float = 60.0
    mv_seed: int = 0                     # dedicated RNG seed for camera sampling

    # ---- Stage C.5 periodic silhouette check (S1) ----
    iter0_camera_iou_threshold: float = 0.4
    silhouette_check_every: int = 0     # disabled unless a no-carpet video target exists
    silhouette_iou_threshold: float = 0.85
    silhouette_n_states: int = 4        # how many states to check (subsampled from K)
    silhouette_expand_dilate: int = 2

    # ---- Type vote at P1 end (S3) ----
    type_vote_n_t_samples: int = 4      # t_ss samples
    type_vote_n_seed_samples: int = 2   # eps seeds -> total 8 forward passes
    type_vote_confidence_threshold: float = 0.7
    # If confidence < threshold: clone learnable into rev/pri branches, run each
    # for (f_main_g1b_end - f_main_g1a_end) * total_iters / 2 more iters, then
    # commit the lower-final-loss branch.
    # Commit the joint TYPE from iter 0 (skip the blend render AND the iter-500
    # vote) when Stage C is already confident. While committed_type is None the
    # renderer blends revolute+prismatic by sigmoid(type_logit); 7201 psi_0 has
    # type_logit=-0.229 (-> 44% prismatic) despite type_confidence=0.955, so
    # iter 0-500 render a revolute/prismatic DOUBLE-WARP phantom. Set to the
    # Stage-C joint_type ("revolute"/"prismatic") to render a single committed
    # branch from the start. None = keep the vote-at-iter-500 behavior.
    force_committed_type: Optional[str] = None

    # ---- D_GS forward dtype ----
    use_fp16_autocast: bool = True      # SS-DiT, D_GS forward under autocast

    # ---- Periodic diagnostics ----
    log_every: int = 50
    viz_every: int = 500
    save_checkpoint_every: int = 2000

    # ---- Misc ----
    sgs_eps: float = 1.0e-6             # small constant for numerical stability
    binary_concrete_eps: float = 1.0e-6

    # ----------------------------------------------------------------- #
    # Derived (★ S4 fix): keep in sync with fourier_num_freqs +          #
    # len(adapter_blocks); never read raw values from outside this class.#
    # ----------------------------------------------------------------- #
    @property
    def feat_dim(self) -> int:
        """Per-voxel feature dim: sum(captured hidden) + Fourier PE + occ logit.

        ``= len(adapter_blocks) * TRELLIS_SS_MODEL_CH + 3 * 2 * fourier_num_freqs + 1``
        For the defaults (3 blocks, 6 freqs): 3072 + 36 + 1 = 3109.
        """
        return (
            len(self.adapter_blocks) * TRELLIS_SS_MODEL_CH
            + 3 * 2 * int(self.fourier_num_freqs)
            + 1
        )


__all__ = [
    "TRELLIS_SS_RES", "TRELLIS_SS_PATCH", "TRELLIS_SS_IN_CH", "TRELLIS_SS_MODEL_CH",
    "TRELLIS_SS_NUM_BLOCKS", "TRELLIS_SS_DTYPE_FP16", "TRELLIS_SS_SIGMA_MIN",
    "TRELLIS_SS_T_SCHEDULE_MEAN", "TRELLIS_SS_T_SCHEDULE_STD",
    "TRELLIS_OCC_RES", "TRELLIS_LATENT_TOKENS", "TRELLIS_LATENT_GRID",
    "TRELLIS_DGS_N_GAUSS_PER_VOXEL", "TRELLIS_DGS_AABB_MIN", "TRELLIS_DGS_AABB_MAX",
    "TRELLIS_DINO_DIM", "TRELLIS_DINO_N_TOKENS",
    "WAN_VAE_STRIDE", "WAN_PATCH_SIZE", "WAN_BOUNDARY_NORMALIZED",
    "WAN_NUM_TRAIN_TIMESTEPS", "WAN_LATENT_CH", "WAN_Y_CH",
    "WAN_DIT_DIM", "WAN_DIT_NUM_LAYERS", "WAN_T5_DIM", "WAN_TEXT_LEN",
    "F_FRAMES", "H_PIXEL", "W_PIXEL", "H_LATENT", "W_LATENT", "F_LATENT",
    "K_STATES", "STATE_INDICES", "CANONICAL_STATE_IDX", "PSI_PARAM_DIM",
    "StageDConfig",
]

"""Stage D: W-RFSDS geometry refinement on canonical articulated 3D.

Inputs (from Stage B Bootstrap, cached on disk under <bootstrap_dir>/):
  z_s0, z_slat0, slat_mean, slat_std, slat_shell_mask,
  U_object, gaussian_parent_idx, psi_0, phi_0 (canonical-shifted),
  anchors_object, M_attn_boot_64, is_carpet_mask, trellis_cond_can,
  wan_cond_cached, z_wan_target, wan_video_target_3FHW, s_0_with_carpet.

Frozen modules:
  TRELLIS:  ss_dit (SparseStructureFlowModel), ss_vae_decoder, d_gs,
            slat_sampler (only used by silhouette_check for U-expand;
            U-expand path itself is deferred — see silhouette_check.py
            header), dinov2 (used at Bootstrap; cached cond reused here).
  Wan2.2:   low_noise_model, high_noise_model, vae, t5 text encoder
            (vae encoder runs grad-enabled inside W-RFSDS; everything else
             stays under torch.no_grad).

Learnable parameters (only):
  Delta_z_s [8, 16, 16, 16]            SS-latent residual (unsqueezed at use)
  alpha_g   [N_obj]                     per-voxel presence gate logit
  alpha_m   [N_obj]                     per-voxel move gate logit
  psi_param [19]                        joint axis/origin/type/limits
  delta_phi [5]                         phi increments (softplus -> positive)
  adapter_{14,16,18}                    zero-init residual MLPs inside SS-DiT
  H_sup, H_part, H_joint                output heads, zero-init

Output (saved to <out_dir>/):
  learnable_p1_final.pt   committed type + frozen joint + final psi/phi/g/m
  z_slat0.pt              unchanged (z_slat0 not learned in P1)
  rendered_p1_final.mp4   diagnostic 21-frame render at P1 end (TODO viz utility)
  viz/                    per-N-iter HTML/PNG snapshots
  summary.json            committed_type, n_iters_run, dual_clone info if any
  logs.jsonl              one JSON line per cfg.log_every iter

Entry point: ``run_stage_d_main(bootstrap_dir, out_dir, cfg, wan_ckpt_dir)``.
"""
from __future__ import annotations

# Public API ------------------------------------------------------------------

from .config import (
    CANONICAL_STATE_IDX,
    F_FRAMES,
    F_LATENT,
    H_LATENT,
    H_PIXEL,
    K_STATES,
    STATE_INDICES,
    StageDConfig,
    W_LATENT,
    W_PIXEL,
)
from .learnable import StageDLearnable
from .render import StageDCameraConfig
from .run_stage_d import (
    load_bootstrap_bundle,
    load_trellis_modules,
    run_stage_d_main,
)
from .silhouette_check import U_ExpandRequired
from .train import BootstrapBundle, TrellisModules, train_stage_d_p1


__all__ = [
    # Config + constants
    "StageDConfig", "StageDCameraConfig",
    "CANONICAL_STATE_IDX",
    "F_FRAMES", "F_LATENT", "H_LATENT", "W_LATENT", "H_PIXEL", "W_PIXEL",
    "K_STATES", "STATE_INDICES",
    # Core types
    "StageDLearnable", "BootstrapBundle", "TrellisModules",
    # Entry points
    "run_stage_d_main", "train_stage_d_p1",
    "load_bootstrap_bundle", "load_trellis_modules",
    # Errors
    "U_ExpandRequired",
]

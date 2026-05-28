"""Bootstrap orchestrator: B1-B12 per method.md section 6.

Composes Stage A (Wan I2V) -> Stage B (SCAR + BMCSA via run_scar) -> Stage C
(joint init) -> SLAT sampling -> Wan condition/VAE encoding -> comprehensive
artifact persistence.

PRINCIPLES:
  - This is a B-Wrap (orchestrator) layer; per-stage code remains in
    `pipelines/stage_a_wan.py`, `pipelines/stage_b_scar.py`,
    `pipelines/stage_c/`.
  - Heavy GPU/checkpoint dependencies (Stage A Wan I2V, SLAT sampler on
    U_object, Wan VAE encoding, Wan T5 encoding) are routed through
    explicit `skip_*` config flags + `NotImplementedError` stubs where
    integration is not yet wired. Each stub names exactly what it needs.
  - No try/except for compatibility/patch behaviour. Missing dependencies
    raise loudly per project conventions.

OUTPUT LAYOUT:
    {out_dir}/
      stage_a/                          # Stage A Wan I2V debug artifacts
      stage_b/                          # run_scar outputs (O_stack, viz/, etc.)
      stage_c/                          # stage_c_joint_init.json
      bootstrap/                        # B12 consolidated artifacts (see below)
        z_s0.pt
        z_final.pt
        z_slat0.pt + slat_mean.pt + slat_std.pt + slat_shell_mask.pt
        dit_hidden.pt
        O_init.npy
        M_attn_boot_64.npy
        is_carpet_mask.npy
        U_seed.npy
        U_object.npy
        gaussian_parent_idx.npy
        psi_0.json
        phi_0.npy
        anchors_object.npy
        trellis_cond_can.pt
        wan_cond_cached.pt
        z_wan_target.pt
        wan_video_target_3FHW.pt
        s_0_clean.pt
        s_0_pure.pt
        bootstrap_meta.json
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pipelines.stage_c import (
    JointInit,
    Psi,
    StageCConfig,
    StageCInputs,
    run_stage_c_joint_init,
)


# ---------------------------------------------------------------------------
# Config + Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BootstrapConfig:
    """All Bootstrap hyperparameters in one place.

    Designed to be constructed from configs/v1.yaml's `bootstrap` block, or
    directly for tests. Skip flags allow incremental wiring while the
    heavy-dependency steps are being built.
    """

    # ---- B1 (Stage A Wan I2V) --------------------------------------------
    skip_b1_stage_a: bool = False
    wan_ckpt_dir: Optional[str] = None
    stage_a_wan_size: str = "832*480"
    stage_a_resolution_hw: Optional[Tuple[int, int]] = None
    stage_a_frame_num: int = 21
    stage_a_seed: int = 42
    stage_a_lang: str = "zh"
    stage_a_sampling_steps: int = 50
    stage_a_guide_scale: float = 3.5
    stage_a_sample_shift: float = 5.0
    stage_a_sample_solver: str = "unipc"
    stage_a_offload_model: bool = False
    stage_a_convert_model_dtype: bool = True
    stage_a_t5_cpu: bool = False
    stage_a_device_id: int = 0
    bootstrap_input_mode: str = "stagea_video"
    stage_a_video_path: Optional[str] = None
    stage_image_dir: Optional[str] = None
    stage_image_pattern: str = "rendering_joint_00_state_{i:02d}.png"
    stage_image_paths: Tuple[str, ...] = ()
    s_0_pure_path: Optional[str] = None

    # ---- B2 (K=6 frame sampling) -----------------------------------------
    state_indices: Tuple[int, ...] = (0, 4, 8, 12, 16, 20)

    # ---- B3-B4 (Stage B Pass-1 + Pass-2 via run_scar) --------------------
    # These get forwarded directly into run_scar's cfg_scar/cfg_sdedit dicts.
    cfg_scar: Dict[str, Any] = field(default_factory=dict)
    cfg_sdedit: Dict[str, Any] = field(default_factory=dict)
    stage_b_remove_disk: bool = True

    # ---- B5 (carpet detect + U_seed) -------------------------------------
    # method.md sec 6 step B5: three-way union
    #   {O_mean > 0.3} ∪ {O_max > 0.5} ∪ boundary_band(0.1, 0.3)
    # then dilate by radius=2 (NEW.1 S1.b)
    u_seed_mean_threshold: float = 0.3
    u_seed_max_threshold: float = 0.5
    u_seed_boundary_low: float = 0.1
    u_seed_boundary_high: float = 0.3
    u_seed_dilate_radius: int = 2

    # Carpet detection method (axis convention from pipelines/utils/postprocessing.py
    # remove_disk: carpet lives along the LAST voxel axis, indices [0..disk_height]).
    #   'derive_from_diff' (default, RECOMMENDED): set carpet = voxels where
    #       O_init > 0.5 AND O_base_canonical == 0; relies on run_scar's
    #       remove_disk having already cleaned O_base_canonical. Lossless,
    #       no axis assumption.
    #   'slab_axis_minus1': simple heuristic — flag the bottom-3 slab along
    #       the LAST voxel axis if its density is >2x column average.
    #   'none': always return all-False mask (assume run_scar handled it
    #       and downstream consumers don't strictly need is_carpet_mask).
    carpet_method: str = "derive_from_diff"

    # ---- B7 sanity ------------------------------------------------------
    # pipeline.md sec 7: |U_object| should be in [10k, 30k]; outside this
    # range emit a warning (loose threshold for prismatic long drawers).
    u_object_min_voxels: int = 5_000
    u_object_max_voxels: int = 40_000

    # ---- B6 (Stage C joint init) -----------------------------------------
    stage_c: StageCConfig = field(default_factory=StageCConfig)

    # ---- B7 (corridor + anchor expansion -> U_object) --------------------
    corridor_n_samples: int = 50          # discrete phi values along trajectory
    anchor_dilate_radius: int = 2

    # ---- B8 (SLAT sampler on U_object) -----------------------------------
    skip_b8_slat: bool = False
    slat_steps: int = 25
    slat_cfg_strength: float = 7.5

    # ---- B10 (build_wan_i2v_cond) ----------------------------------------
    skip_b10_wan_cond: bool = False

    # ---- B11 (Wan VAE encoding) ------------------------------------------
    skip_b11_wan_vae: bool = False

    # ---- B12 persistence -------------------------------------------------
    save_bootstrap_artifacts: bool = True
    persist_dit_hidden: bool = True

    # ---- General ---------------------------------------------------------
    resolution: int = 64                  # voxel grid side
    device: str = "cuda"


@dataclass
class BootstrapResult:
    """Spec'd Bootstrap outputs (method.md section 6 B12).

    Tensors may be on the requested device (default CUDA). Numpy arrays are
    CPU-only. None values indicate `skip_*` was set or the step legitimately
    has no output (e.g. when Stage A artifact was loaded from disk).
    """

    # B3-B4
    z_s0: torch.Tensor                                # (1, 8, 16, 16, 16)
    z_final: torch.Tensor                             # (K, 8, 16, 16, 16)
    O_init: torch.Tensor                              # (1, 1, 64, 64, 64)
    M_attn_boot_64: torch.Tensor                      # (64, 64, 64)

    # B5
    is_carpet_mask: torch.Tensor                      # (64^3,) bool
    U_seed: torch.Tensor                              # (N_seed, 3) int32

    # B6 (Stage C joint init)
    joint_init: JointInit

    # B7
    U_object: torch.Tensor                            # (N_obj, 3) int32
    U_object_with_batch: torch.Tensor                 # (N_obj, 4) int32

    # B8
    z_slat0: Optional[torch.Tensor]                   # (N_obj, 8) post-norm
    slat_mean: Optional[torch.Tensor]                 # (8,)
    slat_std: Optional[torch.Tensor]                  # (8,)
    slat_shell_mask: torch.Tensor                     # (N_obj,) bool

    # B9
    gaussian_parent_idx: torch.Tensor                 # (N_gauss,) int32

    # B10 / B11
    wan_cond_cached: Optional[Dict[str, Any]]         # T5 + 4-ch mask + VAE first frame
    z_wan_target: Optional[torch.Tensor]              # (16, 6, 58, 104)

    # Inputs preserved + passthrough
    trellis_cond_can: torch.Tensor                    # (1, N_dino, 1024) DINOv2(s_0_carpet)
    wan_video_target_3FHW: torch.Tensor               # (3, F, H, W) uint8
    s_0_clean: torch.Tensor                           # (3, H, W) float [0,1]
    s_0_pure: torch.Tensor                            # (3, H, W) float [0,1], no carpet

    # Stage B v3.3.6 secondary outputs (passed through)
    O_base_canonical: torch.Tensor                    # (64, 64, 64) uint8
    O_move_per_state: torch.Tensor                    # (K, 64, 64, 64) uint8
    P_base_canonical: torch.Tensor                    # (64, 64, 64) float
    P_move_evidence_per_state: torch.Tensor           # (K, 64, 64, 64) float
    M_motion_corridor_64: Optional[torch.Tensor]      # (64, 64, 64) float

    # Optional dit_hidden capture (block 14/16/18)
    dit_hidden_cache: Optional[Dict[int, torch.Tensor]]

    # Meta
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BootstrapInputBundle:
    """Unified Bootstrap input after resolving Stage A video or six images."""

    wan_video_target_3FHW: torch.Tensor
    s_0_clean: torch.Tensor
    state_images: Optional[List[Image.Image]]
    source_meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# B1: Bootstrap input (Stage A video, six images, or optional Stage A run)
# ---------------------------------------------------------------------------


def _load_stage_a_video_tensor(
    video_path: str,
    expected_frame_num: int,
) -> torch.Tensor:
    """Load and validate a Stage A uint8 video tensor."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Stage A video tensor not found: {video_path}")
    video = torch.load(video_path, map_location="cpu")
    if video.dtype != torch.uint8:
        raise TypeError(
            f"loaded {video_path} dtype={video.dtype}, expected uint8"
        )
    if video.ndim != 4 or int(video.shape[0]) != 3:
        raise ValueError(
            f"loaded {video_path} shape={tuple(video.shape)}, expected [3, F, H, W]"
        )
    if int(video.shape[1]) != int(expected_frame_num):
        raise ValueError(
            f"loaded {video_path} frame count={int(video.shape[1])}, "
            f"expected {int(expected_frame_num)}"
        )
    H_loaded = int(video.shape[2])
    W_loaded = int(video.shape[3])
    if H_loaded % 16 != 0 or W_loaded % 16 != 0:
        raise ValueError(
            f"loaded {video_path} spatial shape=({H_loaded}, {W_loaded}) is not "
            "aligned to Wan VAE stride 8 and DiT patch size 2"
        )
    return video


def _resolve_stage_a_video_path(out_dir: str, cfg: BootstrapConfig) -> str:
    if cfg.stage_a_video_path is not None:
        return os.path.abspath(os.fspath(cfg.stage_a_video_path))
    stage_a_dir = os.path.join(out_dir, "stage_a")
    candidate_paths = [
        os.path.join(stage_a_dir, "wan_video_target_3FHW_uint8.pt"),
        os.path.join(out_dir, "wan_video_target_3FHW_uint8.pt"),
        os.path.join(out_dir, "bootstrap", "wan_video_target_3FHW.pt"),
    ]
    for p in candidate_paths:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "bootstrap_input_mode='stagea_video' but no pre-existing Stage A "
        f"video tensor was found. Searched: {candidate_paths}"
    )


def _load_state_images_from_config(cfg: BootstrapConfig) -> Tuple[List[Image.Image], List[str]]:
    K = len(cfg.state_indices)
    if len(cfg.stage_image_paths) > 0:
        paths = [os.path.abspath(os.fspath(p)) for p in cfg.stage_image_paths]
        if len(paths) != K:
            raise ValueError(
                f"stage_image_paths length {len(paths)} must equal K={K}"
            )
        for p in paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"missing state image: {p}")
        images = [Image.open(p).convert("RGBA") for p in paths]
        return images, paths

    if cfg.stage_image_dir is None:
        raise ValueError(
            "bootstrap_input_mode='six_images' requires stage_image_dir or "
            "stage_image_paths"
        )
    from pipelines.utils.state_input import load_K_state_images

    image_dir = os.path.abspath(os.fspath(cfg.stage_image_dir))
    images = load_K_state_images(
        image_dir,
        K=K,
        state_indices=cfg.state_indices,
        image_pattern=cfg.stage_image_pattern,
        out_mode="RGBA",
    )
    paths = [
        os.path.join(image_dir, cfg.stage_image_pattern.format(i=i))
        for i in range(K)
    ]
    return images, paths


def _pil_to_rgb_uint8_chw(image: Image.Image) -> torch.Tensor:
    """Convert PIL RGB/RGBA to uint8 [3, H, W], alpha-premultiplied on black."""
    rgba = np.array(image.convert("RGBA"), dtype=np.float32)
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4] / 255.0
    premultiplied = (rgb * alpha).round().clip(0.0, 255.0).astype(np.uint8)
    return torch.from_numpy(np.transpose(premultiplied, (2, 0, 1))).contiguous()


def _load_s0_pure_reference(
    image_path: str,
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    """Load the no-carpet frame-0 reference and resize to Stage A H/W."""
    p = os.path.abspath(os.fspath(image_path))
    if not os.path.isfile(p):
        raise FileNotFoundError(f"s_0_pure image not found: {p}")
    H_tgt, W_tgt = int(target_hw[0]), int(target_hw[1])
    if H_tgt <= 0 or W_tgt <= 0:
        raise ValueError(f"target_hw must be positive; got {target_hw}")

    image = Image.open(p).convert("RGBA")
    s0 = _pil_to_rgb_uint8_chw(image).float() / 255.0
    H_src, W_src = int(s0.shape[1]), int(s0.shape[2])
    src_aspect = float(W_src) / float(H_src)
    tgt_aspect = float(W_tgt) / float(H_tgt)
    rel_aspect_err = abs(src_aspect - tgt_aspect) / max(tgt_aspect, 1.0e-6)
    if rel_aspect_err > 0.02:
        raise ValueError(
            f"s_0_pure aspect mismatch: image shape=({H_src}, {W_src}) "
            f"but Stage A target shape=({H_tgt}, {W_tgt}); relative error "
            f"{rel_aspect_err:.4f} > 0.02. The pure image must come from the "
            "same camera and source folder as 00_seg."
        )
    if (H_src, W_src) != (H_tgt, W_tgt):
        s0 = F.interpolate(
            s0.unsqueeze(0),
            size=(H_tgt, W_tgt),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).clamp(0.0, 1.0)
    return s0.contiguous()


def _tensor_frame_to_rgb_pil(frame_3hw_uint8: torch.Tensor) -> Image.Image:
    if frame_3hw_uint8.dtype != torch.uint8:
        raise TypeError(f"frame dtype must be uint8; got {frame_3hw_uint8.dtype}")
    if frame_3hw_uint8.ndim != 3 or int(frame_3hw_uint8.shape[0]) != 3:
        raise ValueError(
            f"frame must be [3, H, W]; got {tuple(frame_3hw_uint8.shape)}"
        )
    arr = frame_3hw_uint8.cpu().numpy()
    return Image.fromarray(np.transpose(arr, (1, 2, 0)), mode="RGB")


def _validate_state_indices_for_video(
    state_indices: Sequence[int],
    frame_num: int,
) -> Tuple[int, ...]:
    indices = tuple(int(i) for i in state_indices)
    if len(indices) == 0:
        raise ValueError("state_indices must be non-empty")
    if min(indices) < 0 or max(indices) >= int(frame_num):
        raise ValueError(
            f"state_indices {list(indices)} out of range for frame_num={frame_num}"
        )
    if any(indices[i] >= indices[i + 1] for i in range(len(indices) - 1)):
        raise ValueError(f"state_indices must be strictly increasing; got {list(indices)}")
    return indices


def _state_images_to_video_tensor(
    images: Sequence[Image.Image],
    state_indices: Sequence[int],
    frame_num: int,
) -> torch.Tensor:
    """Create a 21-frame target by holding each observed state until the next."""
    indices = _validate_state_indices_for_video(state_indices, frame_num)
    if len(images) != len(indices):
        raise ValueError(
            f"len(images)={len(images)} must equal len(state_indices)={len(indices)}"
        )
    frames = [_pil_to_rgb_uint8_chw(img) for img in images]
    H, W = int(frames[0].shape[1]), int(frames[0].shape[2])
    if H % 16 != 0 or W % 16 != 0:
        raise ValueError(
            f"six-image spatial shape=({H}, {W}) is not aligned to Wan VAE "
            "stride 8 and DiT patch size 2"
        )
    for j, frame in enumerate(frames):
        if tuple(frame.shape) != (3, H, W):
            raise ValueError(
                f"state image {j} shape={tuple(frame.shape)} differs from "
                f"state image 0 shape={(3, H, W)}"
            )

    video = torch.empty((3, int(frame_num), H, W), dtype=torch.uint8)
    for f in range(int(frame_num)):
        src = 0
        for j, idx in enumerate(indices):
            if f >= idx:
                src = j
        video[:, f] = frames[src]
    return video


def _sample_state_images_from_video(
    video_3FHW: torch.Tensor,
    state_indices: Sequence[int],
) -> List[Image.Image]:
    indices = _validate_state_indices_for_video(state_indices, int(video_3FHW.shape[1]))
    return [_tensor_frame_to_rgb_pil(video_3FHW[:, idx]) for idx in indices]


def _run_b1_stage_a(
    s_0_with_carpet: Any,
    user_motion_prompt: str,
    cfg: BootstrapConfig,
    out_dir: str,
) -> BootstrapInputBundle:
    """Resolve Bootstrap input into a unified video tensor plus K state images.

    wan_video_target_3FHW: (3, F, H, W) uint8 [0, 255]
    s_0_clean: (3, H, W) float [0, 1] (first frame as float)
    """
    if cfg.skip_b1_stage_a:
        mode = str(cfg.bootstrap_input_mode)
        if mode == "stagea_video":
            loaded_path = _resolve_stage_a_video_path(out_dir, cfg)
            video = _load_stage_a_video_tensor(loaded_path, cfg.stage_a_frame_num)
            cfg.stage_a_resolution_hw = (int(video.shape[2]), int(video.shape[3]))
            s_0_clean = video[:, 0].float() / 255.0
            state_images = _sample_state_images_from_video(video, cfg.state_indices)
            return BootstrapInputBundle(
                wan_video_target_3FHW=video,
                s_0_clean=s_0_clean,
                state_images=state_images,
                source_meta={
                    "mode": mode,
                    "path": loaded_path,
                    "constructed_video": False,
                },
            )
        if mode == "six_images":
            images, paths = _load_state_images_from_config(cfg)
            video = _state_images_to_video_tensor(
                images,
                state_indices=cfg.state_indices,
                frame_num=cfg.stage_a_frame_num,
            )
            cfg.stage_a_resolution_hw = (int(video.shape[2]), int(video.shape[3]))
            s_0_clean = video[:, 0].float() / 255.0
            return BootstrapInputBundle(
                wan_video_target_3FHW=video,
                s_0_clean=s_0_clean,
                state_images=list(images),
                source_meta={
                    "mode": mode,
                    "image_paths": paths,
                    "image_pattern": cfg.stage_image_pattern,
                    "constructed_video": True,
                    "video_fill": "hold_previous_keyframe",
                },
            )
        raise ValueError(
            "bootstrap_input_mode must be 'stagea_video' or 'six_images' when "
            f"skip_b1_stage_a=True; got {mode!r}"
        )

    if cfg.wan_ckpt_dir is None:
        raise ValueError(
            "B1 requires cfg.wan_ckpt_dir when skip_b1_stage_a=False"
        )

    # Import here to avoid heavy Wan2.2 import for stub runs
    from pipelines.stage_a_wan import run_stage_a

    stage_a_dir = os.path.join(out_dir, "stage_a")
    stage_a_result = run_stage_a(
        image=s_0_with_carpet,
        user_motion_prompt=user_motion_prompt,
        wan_ckpt_dir=cfg.wan_ckpt_dir,
        out_dir=stage_a_dir,
        seed=cfg.stage_a_seed,
        frame_num=cfg.stage_a_frame_num,
        wan_size_label=cfg.stage_a_wan_size,
        sampling_steps=cfg.stage_a_sampling_steps,
        guide_scale=cfg.stage_a_guide_scale,
        sample_shift=cfg.stage_a_sample_shift,
        sample_solver=cfg.stage_a_sample_solver,
        lang=cfg.stage_a_lang,
        offload_model=cfg.stage_a_offload_model,
        convert_model_dtype=cfg.stage_a_convert_model_dtype,
        t5_cpu=cfg.stage_a_t5_cpu,
        device_id=cfg.stage_a_device_id,
    )
    cfg.stage_a_resolution_hw = tuple(int(v) for v in stage_a_result.resolution_hw)
    s_0_clean = stage_a_result.wan_video_target_3FHW[:, 0].float() / 255.0
    state_images = _sample_state_images_from_video(
        stage_a_result.wan_video_target_3FHW,
        cfg.state_indices,
    )
    return BootstrapInputBundle(
        wan_video_target_3FHW=stage_a_result.wan_video_target_3FHW,
        s_0_clean=s_0_clean,
        state_images=state_images,
        source_meta={
            "mode": "run_stage_a",
            "constructed_video": False,
        },
    )


# ---------------------------------------------------------------------------
# B3-B4: Stage B Pass-1 + Pass-2 via existing run_scar
# ---------------------------------------------------------------------------


def _run_b3_b4_stage_b(
    pipe: Any,
    wan_video_target_3FHW: torch.Tensor,
    state_images: Optional[Sequence[Image.Image]],
    cfg: BootstrapConfig,
    out_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Optional[Dict[int, torch.Tensor]]]:
    """Run Stage B (SCAR Pass-1 + BMCSA Pass-2) on the K=6 sampled frames.

    Returns:
        scar_payload: dict with keys
            'z_final' (K, 8, 16, 16, 16), 'cond' (DINOv2 features for K),
            'soft_p1' (K, 64, 64, 64), 'binary_p1' (K, 64, 64, 64)
        stage_b_artifacts: dict of numpy arrays loaded from stage_b/
            O_base_canonical, O_move_per_state, P_base_canonical,
            P_move_evidence_per_state, M_motion_corridor_64 (if available)
        dit_hidden_cache: {block: (K, L, 1024)} fp16 if cfg.scar.capture_dit_hidden
    """
    # Import here to avoid heavy TRELLIS load for stub runs
    from pipelines.stage_b_scar import run_scar

    K = len(cfg.state_indices)
    # B2: sample K frames from video, prepare for DINOv2
    K_frames_uint8 = torch.stack(
        [wan_video_target_3FHW[:, idx] for idx in cfg.state_indices], dim=0
    )  # (K, 3, H, W) uint8
    if state_images is None:
        state_images = _sample_state_images_from_video(
            wan_video_target_3FHW,
            cfg.state_indices,
        )
    if len(state_images) != K:
        raise ValueError(f"len(state_images)={len(state_images)} must equal K={K}")

    # DINOv2 preprocess: pipe should expose preprocess_image method or similar
    # Each pipeline implementation may differ; assume the canonical TRELLIS
    # ImageTo3D pipeline interface used by run_scar elsewhere in this repo.
    if not hasattr(pipe, "preprocess_image"):
        raise AttributeError(
            "pipe (TRELLIS pipeline) must expose preprocess_image() per the "
            "convention used in pipelines/recon.py and pipelines/stage_b_scar.py"
        )

    cond_list = [pipe.preprocess_image(img) for img in state_images]
    # Encode K parallel; pipe.get_cond should handle list/batch
    if not hasattr(pipe, "get_cond"):
        raise AttributeError(
            "pipe must expose get_cond(images) -> {'cond', 'neg_cond'} "
            "matching pipelines/stage_b_scar.run_scar's signature"
        )
    cond = pipe.get_cond(cond_list)  # dict with 'cond' and 'neg_cond' tensors

    # Run SCAR + Pass-2 BMCSA (writes to stage_b dir)
    stage_b_dir = os.path.join(out_dir, "stage_b")
    cfg_scar = dict(cfg.cfg_scar)
    cfg_sdedit = dict(cfg.cfg_sdedit)
    scar_result = run_scar(
        pipe=pipe,
        cond=cond,
        K=K,
        out_dir=stage_b_dir,
        cfg_scar=cfg_scar,
        seed=cfg.stage_a_seed,
        remove_disk_flag=cfg.stage_b_remove_disk,
        device=cfg.device,
        cfg_sdedit=cfg_sdedit,
    )

    # Load v3.3.6 enriched artifacts that run_scar persists
    def _load_npy(name: str) -> Optional[np.ndarray]:
        p = os.path.join(stage_b_dir, name)
        return np.load(p) if os.path.isfile(p) else None

    stage_b_artifacts: Dict[str, np.ndarray] = {}
    for key in [
        "O_base_canonical.npy",
        "O_move_per_state.npy",
        "P_base_canonical.npy",
        "P_move_evidence_per_state.npy",
    ]:
        arr = _load_npy(key)
        if arr is None:
            raise FileNotFoundError(
                f"Stage B did not produce {key}. Stage B v3.3.6+ is required."
            )
        stage_b_artifacts[key.replace(".npy", "")] = arr

    # Motion corridor: viz/bmcsa/M_motion_corridor_64.npy
    mcorr_path = os.path.join(
        stage_b_dir, "viz", "bmcsa", "M_motion_corridor_64.npy"
    )
    if os.path.isfile(mcorr_path):
        stage_b_artifacts["M_motion_corridor_64"] = np.load(mcorr_path)

    # dit_hidden capture
    dit_hidden_cache: Optional[Dict[int, torch.Tensor]] = None
    dit_path = os.path.join(stage_b_dir, "dit_hidden.pt")
    if os.path.isfile(dit_path):
        dit_blob = torch.load(dit_path, map_location="cpu")
        dit_hidden_cache = dit_blob.get("hidden_states", None)

    scar_payload = {
        "z_final": scar_result.z_final,
        "O_stack": scar_result.O_stack,
        "O_stack_soft": scar_result.O_stack_soft,
        "cond": cond,
        "K_frames_uint8": K_frames_uint8,
    }
    return scar_payload, stage_b_artifacts, dit_hidden_cache


# ---------------------------------------------------------------------------
# B4 helpers: z_s0, O_init, M_attn_boot_64
# ---------------------------------------------------------------------------


def _compute_b4_derived(
    z_final: torch.Tensor,
    ss_vae_decoder: torch.nn.Module,
    cfg: BootstrapConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """B4: z_s0 = mean(z_final), O_init = sigmoid(decoder(z_s0)), M_attn_boot_64.

    M_attn_boot_16: per-token pairwise cross-state cosine on the 8-channel
    z_final, trilinear upsampled to 64.
    """
    K = z_final.shape[0]
    z_s0 = z_final.mean(dim=0, keepdim=True)                        # (1, 8, 16, 16, 16)

    with torch.no_grad():
        O_init_logit = ss_vae_decoder(z_s0)                          # (1, 1, 64, 64, 64)
    O_init = torch.sigmoid(O_init_logit)

    # M_attn at 16 token resolution: pairwise cosine on per-token 8-d feature
    C = z_final.shape[1]
    z_tok = z_final.reshape(K, C, -1).permute(0, 2, 1)               # (K, L=4096, 8)
    L = z_tok.shape[1]
    z_norm = F.normalize(z_tok, dim=-1, eps=1e-6)
    # einsum: (K, L, C) x (K, L, C) -> (K, K, L)
    sim = torch.einsum("klc,jlc->kjl", z_norm, z_norm)
    eye_K = torch.eye(K, device=z_norm.device, dtype=torch.bool)
    sim = sim.masked_fill(eye_K.unsqueeze(-1), 0.0)
    if K > 1:
        agree = sim.sum(dim=(0, 1)) / (K * (K - 1))                  # (L,)
    else:
        agree = sim.sum(dim=(0, 1))
    # NOTE: method.md uses sigmoid((agree - tau)/kappa); we expose the raw
    # agreement here (downstream consumers can sigmoid themselves) to keep
    # Bootstrap's M_attn_boot_64 a single canonical map.
    M_attn_16 = agree.view(16, 16, 16).to(torch.float32)
    # ★ Fix Bootstrap WARN-1: align_corners=True per method.md B4 (line 673)
    # and pipeline.md sec 6.2 (line 663-664). align_corners=False introduces
    # a half-voxel offset relative to downstream consumers (Stage D
    # alpha_m init, BMCSA M_attn filter) that read M_attn_boot_64 at
    # voxel-grid coords.
    M_attn_64 = F.interpolate(
        M_attn_16.unsqueeze(0).unsqueeze(0),
        size=(cfg.resolution, cfg.resolution, cfg.resolution),
        mode="trilinear",
        align_corners=True,
    ).squeeze(0).squeeze(0)

    return z_s0, O_init, M_attn_64


# ---------------------------------------------------------------------------
# B5: Carpet detect + U_seed
# ---------------------------------------------------------------------------


def _detect_carpet(
    O_init: torch.Tensor,
    O_base_canonical: Optional[torch.Tensor],
    method: str,
    resolution: int,
) -> torch.Tensor:
    """Carpet voxel mask (flat int64-indexable bool (resolution^3,)).

    Axis convention follows pipelines/utils/postprocessing.py:remove_disk —
    carpet sits at low indices along the LAST voxel axis (W).

    Methods:
      'derive_from_diff' (default):  carpet = (O_init > 0.5) AND NOT
          (O_base_canonical > 0). Relies on run_scar's remove_disk having
          cleaned O_base_canonical. Returns the voxels removed by remove_disk
          that we still see in O_init (re-decoded from z_s0 = mean(z_final)).
      'slab_axis_minus1':  bottom-3 slab heuristic along axis -1 (W).
          Returns voxels in W in [0, disk_height] when its density is
          anomalously high.
      'none':  always all-False (downstream uses O_base_canonical directly).
    """
    if method == "none":
        return torch.zeros(resolution ** 3, dtype=torch.bool)

    if method == "derive_from_diff":
        if O_base_canonical is None:
            raise ValueError(
                "carpet_method='derive_from_diff' requires O_base_canonical; "
                "got None. Either provide it or switch to carpet_method='none' "
                "or 'slab_axis_minus1'."
            )
        O = O_init.detach().cpu()
        if O.dim() == 5:
            O = O[0, 0]
        elif O.dim() == 4:
            O = O[0]
        in_O_init = (O > 0.5).bool()
        Ob = O_base_canonical.detach().cpu()
        if Ob.dim() == 5:
            Ob = Ob[0, 0]
        elif Ob.dim() == 4:
            Ob = Ob[0]
        in_base = Ob.bool()
        # In O_init but not in base — most plausibly the carpet that
        # remove_disk peeled off the canonical base.
        carpet_3d = in_O_init & ~in_base
        return carpet_3d.contiguous().view(-1)

    if method == "slab_axis_minus1":
        O = O_init.detach().cpu()
        if O.dim() == 5:
            O = O[0, 0]
        elif O.dim() == 4:
            O = O[0]
        occ = (O > 0.5).float()                                        # (D, H, W)
        # Profile along W axis (the carpet axis per remove_disk).
        w_profile = occ.sum(dim=(0, 1))                                # (W,)
        if float(w_profile.sum()) < 10:
            return torch.zeros(resolution ** 3, dtype=torch.bool)
        avg = float(w_profile.mean())
        carpet_w_set: List[int] = []
        for w in range(min(8, resolution)):
            if float(w_profile[w]) > avg * 2.0:
                carpet_w_set.append(w)
        if not carpet_w_set:
            return torch.zeros(resolution ** 3, dtype=torch.bool)
        mask_3d = torch.zeros((resolution, resolution, resolution), dtype=torch.bool)
        for w in carpet_w_set:
            mask_3d[:, :, w] = True
        return mask_3d.view(-1)

    raise ValueError(
        f"carpet_method must be 'derive_from_diff' | 'slab_axis_minus1' | "
        f"'none'; got {method!r}"
    )


# Back-compat shim: keep the old symbol so test_bootstrap_unit.py and any
# import paths continue to work. Routes to 'slab_axis_minus1' on bare call.
def _detect_carpet_freeart3d(O_init: torch.Tensor, resolution: int) -> torch.Tensor:
    return _detect_carpet(O_init, O_base_canonical=None, method="slab_axis_minus1", resolution=resolution)


def _flat_idx_to_xyz(flat_idx: torch.Tensor, res: int) -> torch.Tensor:
    """Convert flat index in [0, res^3) to (N, 3) (x, y, z) coords."""
    z = flat_idx % res
    y = (flat_idx // res) % res
    x = flat_idx // (res * res)
    return torch.stack([x, y, z], dim=-1).to(torch.int32)


def _dilate_voxels(xyz: torch.Tensor, radius: int, res: int) -> torch.Tensor:
    """Morphological dilation of a voxel set, returns unique dilated coords."""
    if xyz.shape[0] == 0:
        return xyz
    if radius <= 0:
        return xyz
    # Build offsets within a (2r+1)^3 cube
    r = int(radius)
    offsets = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if dx * dx + dy * dy + dz * dz <= r * r * 3:  # cube neighbourhood
                    offsets.append([dx, dy, dz])
    offs = torch.tensor(offsets, dtype=torch.int32, device=xyz.device)
    expanded = xyz.unsqueeze(1) + offs.unsqueeze(0)                    # (N, M, 3)
    expanded = expanded.reshape(-1, 3)
    # Clamp into grid + unique
    expanded = expanded.clamp(0, res - 1)
    flat = expanded[:, 0].long() * res * res + expanded[:, 1].long() * res + expanded[:, 2].long()
    flat_unique = torch.unique(flat)
    return _flat_idx_to_xyz(flat_unique, res)


def _compute_b5_u_seed(
    O_init: torch.Tensor,
    z_final: torch.Tensor,
    is_carpet_mask: torch.Tensor,
    ss_vae_decoder: torch.nn.Module,
    cfg: BootstrapConfig,
) -> torch.Tensor:
    """B5: build U_seed = three-way union of high-mean, high-max, boundary band,
    then dilate radius=2 (NEW.1 S1.b)."""
    res = cfg.resolution
    O_mean_flat = O_init.view(-1)                                     # (res^3,)
    not_carpet = (~is_carpet_mask.to(O_mean_flat.device)).float()
    O_obj_flat = O_mean_flat * not_carpet

    boundary_band = (
        (O_mean_flat > cfg.u_seed_boundary_low)
        & (O_mean_flat < cfg.u_seed_boundary_high)
        & (~is_carpet_mask.to(O_mean_flat.device))
    )

    # O_max via per-state decode
    with torch.no_grad():
        O_per_state_logit = ss_vae_decoder(z_final)                    # (K, 1, R, R, R)
    O_per_state = torch.sigmoid(O_per_state_logit)
    O_max = O_per_state.max(dim=0, keepdim=False).values               # (1, R, R, R)
    O_max_flat = O_max.view(-1) * not_carpet

    seed_mask = (
        (O_obj_flat > cfg.u_seed_mean_threshold)
        | (O_max_flat > cfg.u_seed_max_threshold)
        | boundary_band
    )
    seed_flat_idx = torch.nonzero(seed_mask, as_tuple=False).squeeze(-1)
    if seed_flat_idx.numel() == 0:
        return torch.zeros((0, 3), dtype=torch.int32, device=O_init.device)
    raw_xyz = _flat_idx_to_xyz(seed_flat_idx, res).to(O_init.device)
    U_seed = _dilate_voxels(raw_xyz, cfg.u_seed_dilate_radius, res)
    return U_seed


# ---------------------------------------------------------------------------
# B7: Corridor + anchor band -> U_object
# ---------------------------------------------------------------------------


def _voxel_to_world(u_xyz: torch.Tensor, res: int) -> torch.Tensor:
    """method.md sec 4.11: (u + 0.5) / res - 0.5."""
    return (u_xyz.float() + 0.5) / res - 0.5


def _world_to_voxel(w_xyz: torch.Tensor, res: int) -> torch.Tensor:
    """Inverse of voxel_to_world; rounds + clamps."""
    return ((w_xyz + 0.5) * res - 0.5).round().long().clamp(0, res - 1)


def _se3_revolute(axis: torch.Tensor, origin: torch.Tensor, phi: float) -> torch.Tensor:
    """Build 4x4 SE(3) rotation around axis at origin by angle phi (rad)."""
    a = axis / (axis.norm() + 1e-8)
    K = torch.tensor(
        [[0, -a[2], a[1]],
         [a[2], 0, -a[0]],
         [-a[1], a[0], 0]],
        device=a.device, dtype=a.dtype,
    )
    R = torch.eye(3, device=a.device, dtype=a.dtype) + torch.sin(torch.tensor(phi, device=a.device)) * K \
        + (1 - torch.cos(torch.tensor(phi, device=a.device))) * (K @ K)
    t = origin - R @ origin
    T = torch.eye(4, device=a.device, dtype=a.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _se3_prismatic(axis: torch.Tensor, phi: float) -> torch.Tensor:
    """Build 4x4 SE(3) translation along axis by signed distance phi."""
    a = axis / (axis.norm() + 1e-8)
    T = torch.eye(4, device=a.device, dtype=a.dtype)
    T[:3, 3] = a * float(phi)
    return T


def _compute_swept_volume_corridor(
    psi: Psi,
    phi_0: torch.Tensor,
    U_seed: torch.Tensor,
    cfg: BootstrapConfig,
) -> torch.Tensor:
    """B7 corridor: sample N phi values between min(phi_0) and max(phi_0),
    warp U_seed by each, union voxels.

    For prismatic: translation along psi.axis by phi.
    For revolute: rotation around psi.axis at psi.origin by phi.

    NOTE: this is a coarse corridor — U_seed approximates the canonical
    geometry; warping gives the swept region. The intent is to ensure
    Stage D has support voxels covering the trajectory.
    """
    res = cfg.resolution
    if U_seed.numel() == 0:
        return U_seed
    phi_min = float(phi_0.min())
    phi_max = float(phi_0.max())
    if phi_max <= phi_min:
        return U_seed
    n_samples = max(2, cfg.corridor_n_samples)
    phi_grid = np.linspace(phi_min, phi_max, n_samples)

    # Convert U_seed to world for warp
    U_world = _voxel_to_world(U_seed, res)                              # (N, 3) world
    is_prismatic = bool(psi.type_logit >= 0.0)
    # Apply each phi
    swept_set: List[torch.Tensor] = []
    for phi_i in phi_grid:
        if is_prismatic:
            T = _se3_prismatic(psi.axis, float(phi_i))
        else:
            T = _se3_revolute(psi.axis, psi.origin, float(phi_i))
        # Apply T to U_world (homogeneous)
        U_h = torch.cat([U_world, torch.ones(U_world.shape[0], 1, device=U_world.device, dtype=U_world.dtype)], dim=-1)
        U_warped = U_h @ T.T
        U_warped_xyz = U_warped[:, :3]
        U_warped_voxel = _world_to_voxel(U_warped_xyz, res).to(torch.int32)
        # Filter to within [0, res)
        in_bounds = (
            (U_warped_voxel >= 0).all(dim=-1)
            & (U_warped_voxel < res).all(dim=-1)
        )
        swept_set.append(U_warped_voxel[in_bounds])

    swept = torch.cat(swept_set, dim=0) if swept_set else U_seed
    # Unique
    flat = swept[:, 0].long() * res * res + swept[:, 1].long() * res + swept[:, 2].long()
    flat_unique = torch.unique(flat)
    return _flat_idx_to_xyz(flat_unique, res).to(U_seed.device)


def _union_voxel_sets(
    *voxel_sets: torch.Tensor,
    res: int,
) -> torch.Tensor:
    """Union of multiple (N_i, 3) int32 voxel coord tensors, unique-sorted."""
    nonempty = [v for v in voxel_sets if v is not None and v.numel() > 0]
    if not nonempty:
        return torch.zeros((0, 3), dtype=torch.int32)
    cat = torch.cat([v.to(torch.long) for v in nonempty], dim=0)
    flat = cat[:, 0] * res * res + cat[:, 1] * res + cat[:, 2]
    flat = torch.unique(flat)
    return _flat_idx_to_xyz(flat, res).to(nonempty[0].device).to(torch.int32)


# ---------------------------------------------------------------------------
# B8: SLAT sampler on U_object
# ---------------------------------------------------------------------------


def _run_b8_slat_sampler(
    pipe: Any,
    U_object: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    cfg: BootstrapConfig,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Sample SLAT on the constructed U_object.

    Returns (z_slat0, slat_mean, slat_std). All None if cfg.skip_b8_slat.

    Implements method.md section 6 B8: build a (N_obj, 4) int32 batched coord
    tensor for U_object, wrap as SparseTensor noise, call pipe.sample_slat
    with cond on state 0 (s_0_carpet). pipe.sample_slat returns sparse
    samples that are ALREADY post-norm (`samples * std + mean` applied
    inside, see TRELLIS/trellis/pipelines/trellis_image_to_3d.py:410-412).
    We expose slat_mean / slat_std separately for Stage D's tanh
    reparameterization (method.md sec 4.3 v3.3.1 S4).
    """
    if cfg.skip_b8_slat:
        # ★ Fix BS-3: even when SLAT sampling is skipped, Stage D's tanh
        # reparam (method.md sec 4.3 v3.3.1 S4) requires slat_mean / slat_std
        # to be saved. Pull them from pipe.slat_normalization (the JSON
        # config block) without running the sampler. If pipe is missing,
        # raise loudly per CLAUDE.md "no compat/patch code" — caller must
        # provide a properly-loaded TRELLIS pipe.
        if not hasattr(pipe, "slat_normalization"):
            raise AttributeError(
                "skip_b8_slat=True but pipe.slat_normalization is missing; "
                "Stage D requires slat_mean/slat_std for tanh reparam. "
                "Provide a fully-loaded TRELLIS Image-to-3D pipeline."
            )
        slat_mean = torch.tensor(
            pipe.slat_normalization["mean"], device=cfg.device, dtype=torch.float32,
        )
        slat_std = torch.tensor(
            pipe.slat_normalization["std"], device=cfg.device, dtype=torch.float32,
        )
        return None, slat_mean, slat_std
    if not hasattr(pipe, "slat_sampler") or not hasattr(pipe, "models"):
        raise AttributeError(
            "B8 SLAT sampler requires pipe.slat_sampler + pipe.models["
            "'slat_flow_model'] from a fully-loaded TRELLIS Image-to-3D "
            "pipeline. Set skip_b8_slat=True or wire pipe properly."
        )
    if U_object.numel() == 0:
        raise ValueError("B8: U_object is empty; cannot sample SLAT.")

    # ★ Fix BS-1: build per-state cond for the CANONICAL state s_c (default c=2),
    # NOT state 0. Per method.md:722-733 v3.3.2 NEW.1-consistency fix:
    # "SLAT decoder lets xyz_canon converge to the geometry described by cond;
    # canonical=s_c needs cond=s_c to be self-consistent". Using s_0 cond would
    # make SLAT reconstruct s_0 geometry while phi[c]=0 expects s_c geometry,
    # leading to a phi reference / canonical geometry mismatch.
    # pipe.get_cond produced (K, N_tok, D); we take state c.
    c_idx = int(cfg.stage_c.canonical_state_idx)
    if isinstance(cond, dict) and "cond" in cond:
        cond_for_slat = {
            "cond": cond["cond"][c_idx:c_idx + 1],
            "neg_cond": cond["neg_cond"][c_idx:c_idx + 1]
                if "neg_cond" in cond and cond["neg_cond"] is not None
                else torch.zeros_like(cond["cond"][c_idx:c_idx + 1]),
        }
    else:
        raise TypeError(
            f"B8 expected cond dict with 'cond'/'neg_cond' tensors; got {type(cond)}"
        )

    # Build (N_obj, 4) int32 coords with batch col 0 first (TRELLIS convention).
    import torchsparse as _ts  # noqa: F401  (force kernel load before SparseTensor)
    from trellis.modules import sparse as sp

    N = int(U_object.shape[0])
    coords_int32 = U_object.to(dtype=torch.int32, device=cfg.device)
    batch_col = torch.zeros((N, 1), dtype=torch.int32, device=cfg.device)
    coords_4 = torch.cat([batch_col, coords_int32], dim=-1)

    flow_model = pipe.models["slat_flow_model"]
    noise = sp.SparseTensor(
        feats=torch.randn(
            N, flow_model.in_channels, device=cfg.device,
            dtype=next(flow_model.parameters()).dtype,
        ),
        coords=coords_4,
    )
    sampler_params = {
        "steps": int(cfg.slat_steps),
        "cfg_strength": float(cfg.slat_cfg_strength),
    }
    z_slat_sparse = pipe.sample_slat(
        cond=cond_for_slat,
        coords=coords_4,
        sampler_params=sampler_params,
        noise=noise,
    )
    # pipe.sample_slat applied post-norm internally; extract dense
    # per-voxel feats (N_obj, 8). The returned object's `.feats` is what
    # method.md calls `z_slat0`.
    z_slat0 = z_slat_sparse.feats.detach()

    # Expose slat_mean / slat_std for Stage D tanh reparam (manifold-aware
    # 3-sigma bound). Source: pipe.slat_normalization (method.md sec 4.3 cite).
    if not hasattr(pipe, "slat_normalization"):
        raise AttributeError(
            "pipe missing slat_normalization {'mean','std'}; cannot expose "
            "slat_mean / slat_std for Stage D."
        )
    slat_mean = torch.tensor(
        pipe.slat_normalization["mean"], device=cfg.device, dtype=torch.float32,
    )
    slat_std = torch.tensor(
        pipe.slat_normalization["std"], device=cfg.device, dtype=torch.float32,
    )
    return z_slat0, slat_mean, slat_std


# ---------------------------------------------------------------------------
# B10: build_wan_i2v_cond
# ---------------------------------------------------------------------------


def _run_b10_wan_cond(
    s_0_pure: torch.Tensor,
    user_motion_prompt: str,
    cfg: BootstrapConfig,
    wan_t5: Any = None,
    wan_vae: Any = None,
) -> Optional[Dict[str, Any]]:
    """Build cached Wan I2V condition dict.

    Per method.md section 5.4 + Wan2.2 image2video.py:259-323:
      1. Normalise s_0_pure to [-1, 1] and resize to (H, W) target.
      2. Build a (3, F, H, W) "fake video": first frame = s_0, rest = zeros.
      3. wan_vae.encode -> y_vae (16, F_lat, h_lat, w_lat).
      4. Build 4-channel mask: frame 0 visible, frames 1.. masked.
      5. Channel-concat mask + y_vae -> y (20, F_lat, h_lat, w_lat).
      6. wan_t5(pos), wan_t5(neg) -> context, context_null.
      7. seq_len = (F_lat * h_lat * w_lat) / (patch_size[1] * patch_size[2]).

    Skipped if cfg.skip_b10_wan_cond.
    """
    if cfg.skip_b10_wan_cond:
        return None
    if wan_t5 is None or wan_vae is None:
        raise AttributeError(
            "B10 requires wan_t5 (umt5-xxl encoder) and wan_vae (Wan2.1 VAE). "
            "Extract from Stage A WanI2V instance (wan.text_encoder + wan.vae) "
            "or set cfg.skip_b10_wan_cond=True."
        )

    from pipelines.wan_helpers.prompts import build_articulated_prompts

    device = torch.device(cfg.device)
    if cfg.stage_a_resolution_hw is None:
        raise ValueError(
            "cfg.stage_a_resolution_hw is unknown. Run B1 Stage A or load a "
            "Stage A artifact before building Wan I2V conditioning."
        )
    H, W = int(cfg.stage_a_resolution_hw[0]), int(cfg.stage_a_resolution_hw[1])
    F_count = int(cfg.stage_a_frame_num)
    vae_stride = (4, 8, 8)
    patch_size = (1, 2, 2)
    lat_h = H // vae_stride[1]
    lat_w = W // vae_stride[2]

    # Step 1+2: prepare s_0_pure as image-frame-1 of a zero-padded video.
    # s_0_pure is (3, H_in, W_in) in [0, 1]. Resize to target (H, W),
    # normalise to [-1, 1].
    img = s_0_pure.to(device=device, dtype=torch.float32)
    if img.shape[-2] != H or img.shape[-1] != W:
        img = F.interpolate(
            img.unsqueeze(0), size=(H, W), mode="bicubic", align_corners=False,
        ).squeeze(0).clamp(0.0, 1.0)
    img_neg11 = img.sub(0.5).mul(2.0)                                     # [-1, 1]

    fake_video_neg11 = torch.cat([
        img_neg11.unsqueeze(1),                                            # (3, 1, H, W)
        torch.zeros(3, F_count - 1, H, W, device=device),                  # (3, F-1, H, W)
    ], dim=1)                                                              # (3, F, H, W)

    # Step 3: Wan VAE encode (expects List[Tensor [3, T, H, W]])
    y_vae = wan_vae.encode([fake_video_neg11])[0]                          # (16, F_lat, h_lat, w_lat)

    # Sanity: confirm latent shape matches our analytic prediction.
    C_lat, F_lat, h_lat_v, w_lat_v = y_vae.shape
    if (h_lat_v, w_lat_v) != (lat_h, lat_w):
        raise RuntimeError(
            f"B10 Wan VAE latent (h, w) = ({h_lat_v}, {w_lat_v}) != "
            f"predicted ({lat_h}, {lat_w})"
        )

    # Step 4: 4-channel mask
    msk = torch.ones(1, F_count, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.concat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:],
    ], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
    # msk shape: (4, F_lat, lat_h, lat_w)

    # Step 5: channel-concat -> (20, F_lat, lat_h, lat_w)
    y = torch.cat([msk, y_vae], dim=0)

    # Step 6: T5 encoding (pos + neg)
    pos_prompt, neg_prompt = build_articulated_prompts(user_motion_prompt, lang=cfg.stage_a_lang)
    context = wan_t5([pos_prompt], device)
    context_null = wan_t5([neg_prompt], device)

    # Step 7: seq_len
    max_seq_len = (F_lat * lat_h * lat_w) // (patch_size[1] * patch_size[2])

    return {
        "context": context,                # List[Tensor [L_text, 4096]]
        "context_null": context_null,
        "seq_len": int(max_seq_len),
        "y": [y],                          # List[Tensor [20, F_lat, h_lat, w_lat]]
        "F_lat": int(F_lat),
        "h_lat": int(lat_h),
        "w_lat": int(lat_w),
        "vae_stride": vae_stride,
        "patch_size": patch_size,
        "pos_prompt": pos_prompt,
        "neg_prompt": neg_prompt,
    }


# ---------------------------------------------------------------------------
# B11: Wan VAE encoding of 21 clean frames
# ---------------------------------------------------------------------------


def _run_b11_wan_vae_encode(
    wan_video_target_3FHW: torch.Tensor,
    cfg: BootstrapConfig,
    wan_vae: Any = None,
) -> Optional[torch.Tensor]:
    """Encode the 21-frame Wan video to latent.

    method.md section 6 B11:
        z_wan_target = wan_vae.encode([(video * 2 - 1)])[0].detach()
    Expected shape is derived from the Stage A actual H/W at vae_stride=(4,8,8).
    Skipped if cfg.skip_b11_wan_vae.
    """
    if cfg.skip_b11_wan_vae:
        return None
    if wan_vae is None:
        raise AttributeError(
            "B11 requires wan_vae (Wan2.1 VAE). Extract from Stage A WanI2V "
            "instance (wan.vae) or set cfg.skip_b11_wan_vae=True."
        )
    # uint8 [0,255] -> float [0,1] -> [-1, 1]
    video_float01 = wan_video_target_3FHW.float() / 255.0
    video_neg11 = video_float01 * 2.0 - 1.0
    # wan_vae.encode expects List[Tensor [3, T, H, W]]
    z_wan_target = wan_vae.encode([video_neg11.to(cfg.device)])[0].detach()
    return z_wan_target


# ---------------------------------------------------------------------------
# B11.5: slat_shell_mask
# ---------------------------------------------------------------------------


def _compute_slat_shell_mask(
    U_object: torch.Tensor,
    O_init: torch.Tensor,
    is_carpet_mask: torch.Tensor,
    cfg: BootstrapConfig,
) -> torch.Tensor:
    """method.md B11.5: voxels in U_object where O_init in boundary band AND
    not carpet."""
    res = cfg.resolution
    if U_object.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool)
    flat_idx = (
        U_object[:, 0].long() * res * res
        + U_object[:, 1].long() * res
        + U_object[:, 2].long()
    )
    O_at_U = O_init.view(-1)[flat_idx]
    carpet_at_U = is_carpet_mask.to(flat_idx.device)[flat_idx]
    shell = (
        (O_at_U > cfg.u_seed_boundary_low)
        & (O_at_U < cfg.u_seed_boundary_high)
        & (~carpet_at_U)
    )
    return shell


# ---------------------------------------------------------------------------
# B12: comprehensive persistence
# ---------------------------------------------------------------------------


def _save_bootstrap_artifacts(result: BootstrapResult, out_dir: str, cfg: BootstrapConfig) -> List[str]:
    """B12: write the full artifact set to {out_dir}/bootstrap/."""
    boot_dir = os.path.join(out_dir, "bootstrap")
    os.makedirs(boot_dir, exist_ok=True)
    saved: List[str] = []

    def _save_tensor(name: str, t: Optional[torch.Tensor], pt: bool = True) -> None:
        if t is None:
            return
        p = os.path.join(boot_dir, name + (".pt" if pt else ".npy"))
        if pt:
            torch.save(t.detach().cpu(), p)
        else:
            np.save(p, t.detach().cpu().numpy() if torch.is_tensor(t) else t)
        saved.append(p)

    # Latents
    _save_tensor("z_s0", result.z_s0)
    _save_tensor("z_final", result.z_final)
    _save_tensor("z_slat0", result.z_slat0)
    _save_tensor("slat_mean", result.slat_mean)
    _save_tensor("slat_std", result.slat_std)
    _save_tensor("slat_shell_mask", result.slat_shell_mask)

    # Voxel + occupancy fields (npy)
    _save_tensor("O_init", result.O_init, pt=False)
    _save_tensor("M_attn_boot_64", result.M_attn_boot_64, pt=False)
    _save_tensor("is_carpet_mask", result.is_carpet_mask, pt=False)
    _save_tensor("U_seed", result.U_seed, pt=False)
    _save_tensor("U_object", result.U_object, pt=False)
    _save_tensor("U_object_with_batch", result.U_object_with_batch, pt=False)
    _save_tensor("gaussian_parent_idx", result.gaussian_parent_idx, pt=False)
    _save_tensor("anchors_object", result.joint_init.anchors_object, pt=False)
    _save_tensor("phi_0", result.joint_init.phi_0, pt=False)

    # Stage B v3.3.6 passthroughs
    _save_tensor("O_base_canonical", result.O_base_canonical, pt=False)
    _save_tensor("O_move_per_state", result.O_move_per_state, pt=False)
    _save_tensor("P_base_canonical", result.P_base_canonical, pt=False)
    _save_tensor("P_move_evidence_per_state", result.P_move_evidence_per_state, pt=False)
    if result.M_motion_corridor_64 is not None:
        _save_tensor("M_motion_corridor_64", result.M_motion_corridor_64, pt=False)

    # Inputs / passthroughs
    _save_tensor("trellis_cond_can", result.trellis_cond_can)
    _save_tensor("wan_video_target_3FHW", result.wan_video_target_3FHW)
    _save_tensor("s_0_clean", result.s_0_clean)
    _save_tensor("s_0_pure", result.s_0_pure)
    if result.z_wan_target is not None:
        _save_tensor("z_wan_target", result.z_wan_target)
    if result.wan_cond_cached is not None:
        # ★ Fix Bootstrap WARN-7: move every tensor in wan_cond_cached to
        # CPU before save. Otherwise the dict carries device tensors and
        # torch.load on a machine with a different device layout fails
        # silently (or remaps in unintended ways). Stage D's
        # load_bootstrap_bundle then calls .to(device) per field, so CPU
        # is the safe portable on-disk representation.
        wan_cond_cpu: Dict[str, Any] = {}
        for k, v in result.wan_cond_cached.items():
            if isinstance(v, torch.Tensor):
                wan_cond_cpu[k] = v.detach().cpu()
            elif isinstance(v, list):
                wan_cond_cpu[k] = [
                    (t.detach().cpu() if isinstance(t, torch.Tensor) else t)
                    for t in v
                ]
            else:
                wan_cond_cpu[k] = v
        p = os.path.join(boot_dir, "wan_cond_cached.pt")
        torch.save(wan_cond_cpu, p)
        saved.append(p)

    # dit_hidden if available
    if cfg.persist_dit_hidden and result.dit_hidden_cache is not None:
        p = os.path.join(boot_dir, "dit_hidden_cache.pt")
        torch.save(result.dit_hidden_cache, p)
        saved.append(p)

    # psi_0 as json (human-readable)
    p_psi = os.path.join(boot_dir, "psi_0.json")
    with open(p_psi, "w") as f:
        json.dump(result.joint_init.to_dict_serialisable(), f, indent=2)
    saved.append(p_psi)

    # meta
    p_meta = os.path.join(boot_dir, "bootstrap_meta.json")
    with open(p_meta, "w") as f:
        json.dump(result.meta, f, indent=2)
    saved.append(p_meta)

    return saved


# ---------------------------------------------------------------------------
# run_bootstrap: full B1-B12 orchestrator
# ---------------------------------------------------------------------------


def run_bootstrap(
    s_0_with_carpet: Optional[Any],
    user_motion_prompt: str,
    out_dir: str,
    pipe: Any,
    cfg: Optional[BootstrapConfig] = None,
    wan_t5: Any = None,
    wan_vae: Any = None,
) -> BootstrapResult:
    """Bootstrap: B1 -> B12, end-to-end.

    Parameters
    ----------
    s_0_with_carpet : PIL Image / np.ndarray / torch.Tensor / None
        Closed-state input image (with FreeArt3D grounding disk baked in).
        Pass None when cfg.skip_b1_stage_a=True (Stage A artifact loaded from disk).
    user_motion_prompt : str
        Per-object motion description (Chinese or English).
    out_dir : str
        Top-level output directory; subdirs `stage_a/`, `stage_b/`, `stage_c/`,
        `bootstrap/` will be created.
    pipe : TrellisImageTo3DPipeline
        Loaded TRELLIS pipeline (must expose get_cond, preprocess_image, .models,
        sparse_structure_sampler, sparse_structure_decoder, sparse_structure_encoder,
        slat_sampler [for B8]).
    cfg : BootstrapConfig, optional
        Defaults to BootstrapConfig() if None.
    wan_t5, wan_vae : optional
        Needed by B10 + B11. Skippable via cfg.skip_b10_wan_cond / skip_b11_wan_vae.
    """
    if cfg is None:
        cfg = BootstrapConfig()
    os.makedirs(out_dir, exist_ok=True)

    # ---- B1: Stage A -----------------------------------------------------
    print("[bootstrap] B1 Stage A Wan I2V")
    input_bundle = _run_b1_stage_a(
        s_0_with_carpet=s_0_with_carpet,
        user_motion_prompt=user_motion_prompt,
        cfg=cfg,
        out_dir=out_dir,
    )
    wan_video_target_3FHW = input_bundle.wan_video_target_3FHW
    s_0_clean = input_bundle.s_0_clean
    if cfg.s_0_pure_path is None:
        raise ValueError(
            "Bootstrap now requires cfg.s_0_pure_path. Stage A and Stage B "
            "consume the carpeted 00_seg/Stage A video, but Stage D frame-0 "
            "supervision and Wan I2V conditioning require the no-carpet "
            "00_pure image from the same source folder."
        )
    s_0_pure = _load_s0_pure_reference(
        cfg.s_0_pure_path,
        target_hw=(int(wan_video_target_3FHW.shape[2]), int(wan_video_target_3FHW.shape[3])),
    )

    # ---- B3 + B4: Stage B Pass-1 + Pass-2 --------------------------------
    print("[bootstrap] B3-B4 Stage B (SCAR + BMCSA)")
    scar_payload, stage_b_artifacts, dit_hidden_cache = _run_b3_b4_stage_b(
        pipe=pipe,
        wan_video_target_3FHW=wan_video_target_3FHW,
        state_images=input_bundle.state_images,
        cfg=cfg,
        out_dir=out_dir,
    )
    z_final = scar_payload["z_final"]
    cond = scar_payload["cond"]

    ss_vae_decoder = pipe.models["sparse_structure_decoder"]
    z_s0, O_init, M_attn_boot_64 = _compute_b4_derived(z_final, ss_vae_decoder, cfg)

    # ---- B5: Carpet + U_seed --------------------------------------------
    print("[bootstrap] B5 carpet + U_seed")
    # Load O_base_canonical early so 'derive_from_diff' carpet method works.
    O_base_canonical_for_carpet = torch.from_numpy(
        stage_b_artifacts["O_base_canonical"]
    )
    is_carpet_mask = _detect_carpet(
        O_init=O_init,
        O_base_canonical=O_base_canonical_for_carpet,
        method=cfg.carpet_method,
        resolution=cfg.resolution,
    ).to(cfg.device)
    U_seed = _compute_b5_u_seed(O_init, z_final, is_carpet_mask, ss_vae_decoder, cfg)
    print(
        f"  carpet_method={cfg.carpet_method}: "
        f"carpet voxels={int(is_carpet_mask.sum())}, "
        f"U_seed voxels={int(U_seed.shape[0])}"
    )

    # ---- B6: Stage C joint init -----------------------------------------
    print("[bootstrap] B6 Stage C joint init")
    O_base_canonical = torch.from_numpy(stage_b_artifacts["O_base_canonical"]).to(cfg.device)
    O_move_per_state = torch.from_numpy(stage_b_artifacts["O_move_per_state"]).to(cfg.device)
    P_base_canonical = torch.from_numpy(stage_b_artifacts["P_base_canonical"]).to(cfg.device)
    P_move_evidence_per_state = torch.from_numpy(stage_b_artifacts["P_move_evidence_per_state"]).to(cfg.device)
    M_motion_corridor_64 = (
        torch.from_numpy(stage_b_artifacts["M_motion_corridor_64"]).to(cfg.device)
        if "M_motion_corridor_64" in stage_b_artifacts else None
    )
    stage_c_inputs = StageCInputs(
        z_final=z_final,
        M_attn_boot_64=M_attn_boot_64,
        O_init=O_init,
        is_carpet_mask=is_carpet_mask,
        U_seed=U_seed,
        O_base_canonical=O_base_canonical,
        O_move_per_state=O_move_per_state,
        P_base_canonical=P_base_canonical,
        P_move_evidence_per_state=P_move_evidence_per_state,
        M_motion_corridor_64=M_motion_corridor_64,
        dit_hidden=dit_hidden_cache,
    )
    stage_c_dir = os.path.join(out_dir, "stage_c")
    joint_init: JointInit = run_stage_c_joint_init(stage_c_inputs, cfg.stage_c, stage_c_dir)

    # ★ Bootstrap contract assertion: Stage C MUST return phi_0 with the
    # canonical-state-shifted convention (phi_0[c] == 0). method.md sec 6
    # B6 applies the shift inside Bootstrap historically; we delegate this
    # to Stage C and verify here to avoid silent double-shift / no-shift bugs.
    c_idx = int(cfg.stage_c.canonical_state_idx)
    phi_at_c = float(joint_init.phi_0[c_idx])
    if abs(phi_at_c) > 1e-4:
        raise ValueError(
            f"Stage C contract violation: phi_0[c={c_idx}] = {phi_at_c:.6f} "
            f"!= 0. Stage C joint init MUST return canonical-state-shifted "
            f"phi_0 per method.md sec 6 NEW.1. Either Stage C forgot to "
            f"shift, or Bootstrap was upgraded to apply shift externally "
            f"(only one should)."
        )
    print(
        f"  joint_type={joint_init.joint_type()}, "
        f"confidence={joint_init.confidence:.3f}, "
        f"n_anchors={int(joint_init.anchors_object.shape[0])}, "
        f"phi_0[c={c_idx}]={phi_at_c:.4f} (must be 0)"
    )

    # ---- B7: U_seed -> U_object via corridor + anchor band --------------
    print("[bootstrap] B7 U_object expansion (corridor + anchors)")
    corridor = _compute_swept_volume_corridor(
        joint_init.psi, joint_init.phi_0, U_seed, cfg
    )
    anchor_band = _dilate_voxels(
        joint_init.anchors_object.to(cfg.device),
        cfg.anchor_dilate_radius,
        cfg.resolution,
    )
    U_object = _union_voxel_sets(U_seed, corridor, anchor_band, res=cfg.resolution)
    n_U_object = int(U_object.shape[0])
    print(
        f"  U_object voxels: {n_U_object} "
        f"(seed={int(U_seed.shape[0])}, corridor={int(corridor.shape[0])}, "
        f"anchor_band={int(anchor_band.shape[0])})"
    )
    # ★ Sanity check per pipeline.md sec 7: |U_object| in [10k, 30k]; loose
    # tolerance [5k, 40k] before warning. Outside [u_object_min, u_object_max]
    # signals a problem upstream (Bootstrap continues but flags it).
    if n_U_object < cfg.u_object_min_voxels:
        print(
            f"  WARNING: U_object={n_U_object} < u_object_min_voxels="
            f"{cfg.u_object_min_voxels}. SCAR may be missing voxels or "
            f"is_carpet_mask is over-aggressive."
        )
    elif n_U_object > cfg.u_object_max_voxels:
        print(
            f"  WARNING: U_object={n_U_object} > u_object_max_voxels="
            f"{cfg.u_object_max_voxels}. Corridor expansion may be over-"
            f"inflating; consider tightening cfg.corridor_n_samples or "
            f"reviewing joint_init psi.axis correctness."
        )

    # Build U_object_with_batch (B+coords)
    N_obj = int(U_object.shape[0])
    batch_col = torch.zeros((N_obj, 1), dtype=torch.int32, device=U_object.device)
    U_object_with_batch = torch.cat([batch_col, U_object.to(torch.int32)], dim=-1)

    # ---- B8: SLAT sampler on U_object ----------------------------------
    print("[bootstrap] B8 SLAT sampling on U_object")
    z_slat0, slat_mean, slat_std = _run_b8_slat_sampler(pipe, U_object, cond, cfg)

    # ---- B9: gaussian_parent_idx ---------------------------------------
    gaussian_parent_idx = torch.arange(
        N_obj, device=cfg.device, dtype=torch.int64,
    ).repeat_interleave(32).to(torch.int32)

    # ---- B10: build_wan_i2v_cond ---------------------------------------
    print("[bootstrap] B10 build_wan_i2v_cond")
    wan_cond_cached = _run_b10_wan_cond(s_0_pure, user_motion_prompt, cfg, wan_t5, wan_vae)

    # ---- B11: Wan VAE encoding -----------------------------------------
    print("[bootstrap] B11 Wan VAE encoding")
    z_wan_target = _run_b11_wan_vae_encode(wan_video_target_3FHW, cfg, wan_vae)

    # ---- B11.5: slat_shell_mask ----------------------------------------
    slat_shell_mask = _compute_slat_shell_mask(U_object, O_init, is_carpet_mask, cfg)

    # ---- trellis_cond_can: DINOv2 of CANONICAL state s_c (default c=2) -
    # ★ Fix BS-2: per method.md:722-733 v3.3.2 NEW.1-consistency fix,
    # Stage D's one-step SS-DiT canonical forward must use s_c cond, not
    # s_0 cond. Mismatch breaks the canonical-geometry / phi-reference
    # contract (canonical geom = s_c per NEW.1, phi[c]=0).
    c_idx = int(cfg.stage_c.canonical_state_idx)
    trellis_cond_can = (
        cond["cond"][c_idx:c_idx + 1].detach() if "cond" in cond
        else cond[c_idx:c_idx + 1].detach()
    )

    # ---- Assemble result -----------------------------------------------
    result = BootstrapResult(
        z_s0=z_s0,
        z_final=z_final,
        O_init=O_init,
        M_attn_boot_64=M_attn_boot_64,
        is_carpet_mask=is_carpet_mask,
        U_seed=U_seed,
        joint_init=joint_init,
        U_object=U_object,
        U_object_with_batch=U_object_with_batch,
        z_slat0=z_slat0,
        slat_mean=slat_mean,
        slat_std=slat_std,
        slat_shell_mask=slat_shell_mask,
        gaussian_parent_idx=gaussian_parent_idx,
        wan_cond_cached=wan_cond_cached,
        z_wan_target=z_wan_target,
        trellis_cond_can=trellis_cond_can,
        wan_video_target_3FHW=wan_video_target_3FHW,
        s_0_clean=s_0_clean,
        s_0_pure=s_0_pure,
        O_base_canonical=O_base_canonical,
        O_move_per_state=O_move_per_state,
        P_base_canonical=P_base_canonical,
        P_move_evidence_per_state=P_move_evidence_per_state,
        M_motion_corridor_64=M_motion_corridor_64,
        dit_hidden_cache=dit_hidden_cache,
        meta={
            "K": len(cfg.state_indices),
            "state_indices": list(cfg.state_indices),
            "resolution": cfg.resolution,
            "stage_a_wan_size": cfg.stage_a_wan_size,
            "stage_a_resolution_hw": list(cfg.stage_a_resolution_hw),
            "stage_a_skipped": cfg.skip_b1_stage_a,
            "bootstrap_input": input_bundle.source_meta,
            "s_0_pure_path": os.path.abspath(os.fspath(cfg.s_0_pure_path)),
            "wan_condition_source": "s_0_pure",
            "slat_skipped": cfg.skip_b8_slat,
            "wan_cond_skipped": cfg.skip_b10_wan_cond,
            "wan_vae_skipped": cfg.skip_b11_wan_vae,
            "joint_type": joint_init.joint_type(),
            "joint_confidence": joint_init.confidence,
            "n_voxels": {
                "U_seed": int(U_seed.shape[0]),
                "U_object": int(U_object.shape[0]),
                "anchors": int(joint_init.anchors_object.shape[0]),
                "carpet": int(is_carpet_mask.sum()),
            },
        },
    )

    # ---- B12: persist --------------------------------------------------
    if cfg.save_bootstrap_artifacts:
        print("[bootstrap] B12 persisting artifacts")
        saved = _save_bootstrap_artifacts(result, out_dir, cfg)
        result.meta["artifacts_written"] = saved

    print(f"[bootstrap] complete. joint_type={joint_init.joint_type()}, n_U_object={N_obj}")
    return result

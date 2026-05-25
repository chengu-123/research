"""Stage D entry point: load Bootstrap + TRELLIS + Wan2.2, call train.

Usage from ``run_v1.py``::

    from pipelines.stage_d import run_stage_d_main

    run_stage_d_main(
        bootstrap_dir=os.path.join(out_root, 'stage_b'),
        out_dir=os.path.join(out_root, 'stage_d'),
        cfg=stage_d_cfg,
        wan_ckpt_dir=cfg.wan_ckpt_dir,
        device='cuda',
    )

Bootstrap is expected to have written its outputs as individual files in
``bootstrap_dir`` per ``record/pipeline.md`` section 6.1 (the loader
``load_bootstrap_bundle`` checks for the file names listed there).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch

from .config import (
    CANONICAL_STATE_IDX,
    F_FRAMES,
    H_PIXEL,
    StageDConfig,
    TRELLIS_OCC_RES,
    W_PIXEL,
)
from .feature_sample import voxel_to_world
from .losses import LPIPSModule
from .render import StageDCameraConfig, build_locked_camera
from .train import BootstrapBundle, TrellisModules, train_stage_d_p1
from .w_rfsds import load_wan_for_rfsds


logger = logging.getLogger(__name__)


# =============================================================================
# Bootstrap loader
# =============================================================================

def _load_pt(path: str) -> Any:
    """Load a torch ``.pt`` file (CPU); raise with a clear path on miss."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Bootstrap artifact missing: {path}")
    return torch.load(path, map_location="cpu")


def _load_npy(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Bootstrap artifact missing: {path}")
    return np.load(path)


def _load_json(path: str) -> Any:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Bootstrap artifact missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_tensor(x: Any, device: torch.device,
                dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Coerce numpy / list / tensor to a torch tensor on the target device."""
    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, (list, tuple)):
        t = torch.tensor(x)
    else:
        raise TypeError(f"cannot coerce {type(x)!r} to tensor")
    t = t.to(device=device)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t


def load_bootstrap_bundle(
    bootstrap_dir: str,
    device: torch.device,
    n_gauss_per_voxel: int = 32,
) -> BootstrapBundle:
    """Read every Bootstrap artifact from ``bootstrap_dir`` into a bundle.

    File naming follows ``record/pipeline.md`` section 6.1.
    """
    if not os.path.isdir(bootstrap_dir):
        raise FileNotFoundError(f"bootstrap_dir not found: {bootstrap_dir!r}")

    # --- SS / SLAT geometry ---
    z_s0 = _load_pt(os.path.join(bootstrap_dir, "z_s0.pt")).to(device)
    z_slat0 = _load_pt(os.path.join(bootstrap_dir, "z_slat0.pt")).to(device)
    slat_mean = _as_tensor(
        _load_pt(os.path.join(bootstrap_dir, "slat_mean.pt")), device, torch.float32,
    )
    slat_std = _as_tensor(
        _load_pt(os.path.join(bootstrap_dir, "slat_std.pt")), device, torch.float32,
    )
    slat_shell_mask = _as_tensor(
        _load_pt(os.path.join(bootstrap_dir, "slat_shell_mask.pt")),
        device, torch.bool,
    )
    U_object_np = _load_npy(os.path.join(bootstrap_dir, "U_object.npy"))
    U_object = torch.from_numpy(U_object_np.astype(np.int64)).to(device)   # [N_obj, 3]
    n_obj = int(U_object.shape[0])

    # U_object_with_batch: [N_obj, 4] int32 with leading zero (single-batch)
    batch_col = torch.zeros(n_obj, 1, dtype=torch.int32, device=device)
    U_object_with_batch = torch.cat([
        batch_col, U_object.to(torch.int32),
    ], dim=-1)

    # gaussian_parent_idx: if Bootstrap saved it, use it; else compute trivially.
    gpi_path = os.path.join(bootstrap_dir, "gaussian_parent_idx.npy")
    if os.path.isfile(gpi_path):
        gaussian_parent_idx = torch.from_numpy(
            _load_npy(gpi_path).astype(np.int64)
        ).to(device)
    else:
        gaussian_parent_idx = torch.arange(
            n_obj, device=device, dtype=torch.long
        ).repeat_interleave(n_gauss_per_voxel)

    # --- Joint init (psi_0 may be saved as json or pt) ---
    psi_0_path_json = os.path.join(bootstrap_dir, "psi_0.json")
    psi_0_path_pt = os.path.join(bootstrap_dir, "psi_0.pt")
    if os.path.isfile(psi_0_path_json):
        psi_0_raw = _load_json(psi_0_path_json)
        psi_0 = {
            k: _as_tensor(v, device, torch.float32) for k, v in psi_0_raw.items()
        }
    elif os.path.isfile(psi_0_path_pt):
        psi_0_raw = _load_pt(psi_0_path_pt)
        psi_0 = {
            k: _as_tensor(v, device, torch.float32) for k, v in psi_0_raw.items()
        }
    else:
        raise FileNotFoundError(
            f"psi_0 missing: expected one of {psi_0_path_json!r} or {psi_0_path_pt!r}"
        )

    phi_0 = _as_tensor(
        _load_npy(os.path.join(bootstrap_dir, "phi_0.npy")),
        device, torch.float32,
    )
    anchors_object_np = _load_npy(os.path.join(bootstrap_dir, "anchors_object.npy"))
    anchors_object = torch.from_numpy(anchors_object_np.astype(np.int64)).to(device)
    anchors_world = voxel_to_world(anchors_object, res=TRELLIS_OCC_RES)

    # --- BMCSA M_attn at U_object ---
    M_attn_64 = torch.from_numpy(
        _load_npy(os.path.join(bootstrap_dir, "M_attn_boot_64.npy"))
    ).to(device).float()                                  # [64, 64, 64]
    # Index M_attn_64 at U_object's voxel coords using the flat index convention
    # i = d * 64*64 + h * 64 + w.
    R = TRELLIS_OCC_RES
    flat_idx = (
        U_object[:, 0] * R * R + U_object[:, 1] * R + U_object[:, 2]
    )
    M_attn_at_U = M_attn_64.view(-1)[flat_idx]            # [N_obj]

    # --- Conditioning ---
    trellis_cond_can = _load_pt(
        os.path.join(bootstrap_dir, "trellis_cond_can.pt")
    ).to(device)                                          # [1, N_dino, 1024]
    wan_cond = _load_pt(os.path.join(bootstrap_dir, "wan_cond_cached.pt"))
    # Move dict-of-list-of-tensor onto device.
    wan_cond_on_dev: Dict[str, Any] = {}
    for k, v in wan_cond.items():
        if isinstance(v, list):
            wan_cond_on_dev[k] = [
                (t.to(device) if isinstance(t, torch.Tensor) else t) for t in v
            ]
        elif isinstance(v, torch.Tensor):
            wan_cond_on_dev[k] = v.to(device)
        else:
            wan_cond_on_dev[k] = v
    z_wan_target = _load_pt(
        os.path.join(bootstrap_dir, "z_wan_target.pt")
    ).to(device).float()

    # --- Targets ---
    wan_video_target_uint8 = _load_pt(
        os.path.join(bootstrap_dir, "wan_video_target_3FHW.pt")
    )                                                      # [3, F, H, W] uint8
    if wan_video_target_uint8.dtype != torch.uint8:
        wan_video_target_uint8 = (
            wan_video_target_uint8.clamp(0.0, 1.0) * 255.0
        ).to(torch.uint8)
    wan_video_target_T3HW_01 = (
        wan_video_target_uint8.float().to(device) / 255.0
    ).permute(1, 0, 2, 3).contiguous()                     # [F, 3, H, W] in [0, 1]
    if wan_video_target_T3HW_01.shape != (F_FRAMES, 3, H_PIXEL, W_PIXEL):
        raise RuntimeError(
            f"Bootstrap target video shape mismatch: expected "
            f"({F_FRAMES}, 3, {H_PIXEL}, {W_PIXEL}); "
            f"got {tuple(wan_video_target_T3HW_01.shape)}"
        )

    # s_0_with_carpet may be saved as s_0_clean.pt (legacy name) or
    # s_0_with_carpet.pt (v3.3 name). Try the v3.3 name first.
    s0_path_v33 = os.path.join(bootstrap_dir, "s_0_with_carpet.pt")
    s0_path_legacy = os.path.join(bootstrap_dir, "s_0_clean.pt")
    if os.path.isfile(s0_path_v33):
        s_0 = _load_pt(s0_path_v33)
    elif os.path.isfile(s0_path_legacy):
        s_0 = _load_pt(s0_path_legacy)
    else:
        raise FileNotFoundError(
            f"s_0 input missing: expected {s0_path_v33!r} or {s0_path_legacy!r}"
        )
    s_0 = s_0.to(device)
    if s_0.dtype == torch.uint8:
        s_0 = s_0.float() / 255.0
    if s_0.shape != (3, H_PIXEL, W_PIXEL):
        raise RuntimeError(
            f"s_0_with_carpet shape mismatch: expected "
            f"(3, {H_PIXEL}, {W_PIXEL}); got {tuple(s_0.shape)}"
        )

    return BootstrapBundle(
        z_s0=z_s0,
        z_slat0=z_slat0,
        slat_mean=slat_mean, slat_std=slat_std,
        slat_shell_mask=slat_shell_mask,
        U_object=U_object,
        U_object_with_batch=U_object_with_batch,
        gaussian_parent_idx=gaussian_parent_idx,
        n_obj=n_obj,
        psi_0=psi_0, phi_0=phi_0,
        anchors_object=anchors_object, anchors_world=anchors_world,
        M_attn_at_U=M_attn_at_U,
        trellis_cond_can=trellis_cond_can,
        wan_cond=wan_cond_on_dev, z_wan_target=z_wan_target,
        wan_video_target_T3HW_01=wan_video_target_T3HW_01,
        s_0_with_carpet_3HW_01=s_0,
    )


# =============================================================================
# TRELLIS module loader
# =============================================================================

def load_trellis_modules(pretrained: str = "JeffreyXiang/TRELLIS-image-large",
                          device: torch.device = torch.device("cuda")) -> TrellisModules:
    """Load the four frozen TRELLIS modules Stage D needs.

    Reuses ``pipelines.recon.build_trellis_pipeline`` so we get the same
    SS-VAE encoder side-load that Stage B needed (harmless if absent in
    Stage D, but consistent with other stages).
    """
    sys.path.append("TRELLIS")
    from pipelines.recon import build_trellis_pipeline

    pipe = build_trellis_pipeline(device=str(device.type), pretrained=pretrained)

    ss_dit = pipe.models["sparse_structure_flow_model"]
    ss_vae_decoder = pipe.models["sparse_structure_decoder"]
    d_gs = pipe.models["slat_decoder_gs"]
    slat_sampler = pipe.slat_sampler

    for module in (ss_dit, ss_vae_decoder, d_gs):
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()

    return TrellisModules(
        ss_dit=ss_dit,
        ss_vae_decoder=ss_vae_decoder,
        d_gs=d_gs,
        slat_sampler=slat_sampler,
    )


# =============================================================================
# Top-level entry
# =============================================================================

def run_stage_d_main(
    bootstrap_dir: str,
    out_dir: str,
    cfg: StageDConfig,
    wan_ckpt_dir: str,
    repo_root: Optional[str] = None,
    device: str = "cuda",
    device_id: int = 0,
    trellis_pretrained: str = "JeffreyXiang/TRELLIS-image-large",
    camera: Optional[StageDCameraConfig] = None,
    lpips_net: str = "vgg",
) -> Dict[str, Any]:
    """Build everything, then run ``train_stage_d_p1``.

    Parameters
    ----------
    camera : Optional[StageDCameraConfig]
        If None, defaults to ``StageDCameraConfig.freeart3d_canonical()`` —
        the camera that FreeArt3D's ``pipelines/render.py`` uses to render
        PartNet inputs (fov=45 deg, azi=22.5 deg, ele=45 deg, dist=2.1,
        +Z up). TRELLIS canonical world up is verified +Z via
        ``trellis/utils/render_utils.py:33``. The iter-0 silhouette IoU
        sanity check inside ``train_stage_d_p1`` validates against
        ``s_0_with_carpet``; failure raises ``CameraMismatchError``.

    Returns the training summary dict (committed_type, n_iters_run,
    type_vote details, iter_0_camera_iou, etc).
    """
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(device)
    if repo_root is None:
        # Assume run_v1.py is the entry point and we are at <repo>/pipelines/...
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        )
        # repo_root should now be the dir containing pipelines/ and Wan2.2/

    logger.info("[stage_d] loading Bootstrap bundle from %s", bootstrap_dir)
    bootstrap = load_bootstrap_bundle(bootstrap_dir, dev)

    logger.info("[stage_d] loading TRELLIS modules")
    trellis = load_trellis_modules(pretrained=trellis_pretrained, device=dev)

    logger.info("[stage_d] loading Wan2.2 for W-RFSDS from %s", wan_ckpt_dir)
    wan_ctx = load_wan_for_rfsds(
        wan_ckpt_dir=wan_ckpt_dir, repo_root=repo_root, device=dev,
        convert_model_dtype=True, device_id=device_id,
    )

    logger.info("[stage_d] building camera + LPIPS")
    if camera is None:
        # ★ Default to FreeArt3D's hardcoded render camera (matches s_0).
        # +Z up is the verified TRELLIS canonical world convention; not a
        # tunable. Iter-0 silhouette IoU check inside train.py validates.
        camera = StageDCameraConfig.freeart3d_canonical()
        logger.info(
            "[stage_d] using freeart3d_canonical camera (TRELLIS +Z up); "
            "iter-0 sanity check will validate against s_0_with_carpet."
        )
    locked_camera = build_locked_camera(camera, device=dev, dtype=torch.float32)
    lpips_module = LPIPSModule(net_type=lpips_net).to(dev)

    logger.info("[stage_d] entering train_stage_d_p1 (total_iters=%d)",
                cfg.total_iters)
    summary = train_stage_d_p1(
        bootstrap=bootstrap,
        trellis=trellis,
        wan_ctx=wan_ctx,
        camera=locked_camera,
        lpips_module=lpips_module,
        cfg=cfg,
        out_dir=out_dir,
        device=dev,
    )
    # Persist summary as JSON for downstream inspection.
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    logger.info("[stage_d] done; committed_type=%s",
                summary.get("committed_type"))
    return summary


__all__ = [
    "load_bootstrap_bundle", "load_trellis_modules", "run_stage_d_main",
]

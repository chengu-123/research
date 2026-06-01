"""CAST-U Step 2: canonical SLAT texture refine (delta_z) with W-RFSDS + pixel anchor.

Geometry, part split, and joint trajectory are FROZEN (from Bootstrap / Stage C).
Only the per-voxel SLAT texture is optimized through a single learnable
``delta_z`` (method.md S4 tanh reparam):

    z = z_slat0 + 3.0 * slat_std * tanh(delta_z),   delta_z = Parameter(zeros)

The canonical SLAT is decoded by the frozen TRELLIS D_GS into Gaussians, warped
along the FROZEN joint trajectory (``phi_0`` from Bootstrap, scaled by the frozen
theta_max / disp_max) using the same two-branch base/move split the Stage D train
loop uses, and rasterized into a 21-frame locked-camera video. The six observed
articulation states (0..5) map to render frames [0, 4, 8, 12, 16, 20].

Supervision (a direct supervised fit; PSNR/SSIM vs the real images MUST rise):
  - L_anchor : masked-L1 between render_frame[k] and the REAL pure input image of
    state k (bootstrap/pure_state_targets_K3HW.pt). Never eval renders / gt_mesh.
  - L_wrfsds : CHORD Eq.3 surrogate from Wan2.2-Fun-A14B-InP on the full 21-frame
    video; endpoints conditioned on the real s_0 / s_5 (cached fun_video/fun_mask).
  - reg      : L2 on delta_z for stability.

  loss = L_anchor + lambda_sds * L_wrfsds + lambda_reg * reg(delta_z)

Two-GPU layout (no offload, no t5_cpu, per the user's tested H800 config):
  - TRELLIS pipe + SLAT decode + render + optimization + Wan VAE encode: cuda:0
  - Wan Fun-InP DiT experts + T5 text encoder: cuda:1 (teacher_device)
  Cross-device autograd flows the SDS residual grad from cuda:1 back to cuda:0.

Smoke mode (``--lambda_sds 0``): skips the Wan load entirely (single GPU) and
validates the render + anchor wiring + PSNR/SSIM rise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# --- repo path setup (mine/ is repo root; Wan2.2 + TRELLIS vendored under it) ---
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent                      # .../mine
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "TRELLIS") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "TRELLIS"))

from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

from trellis.modules import sparse as sp

from pipelines.recon import build_trellis_pipeline
from pipelines.stage_d.config import (
    F_FRAMES,
    K_STATES,
    STATE_INDICES,
    TRELLIS_OCC_RES,
)
from pipelines.stage_d.joint_ops import JointParams, linear_interp_through
from pipelines.stage_d.render import (
    DGSWithParent,
    RenderInputs,
    StageDCameraConfig,
    build_locked_camera,
    render_21_with_warp,
    render_static_gaussians,
)

# SH0 DC -> RGB constant (TRELLIS gaussian DC convention).
_SH_C0 = 0.28209479177387814


# =============================================================================
# Args
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CAST-U Step 2 SLAT texture refine.")
    p.add_argument("--bootstrap_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--agg_init", default=None,
                   help="aggregated_init.pt path (required with --use_agg_init)")
    p.add_argument("--trellis", default="/lustre/1230003454/hf_models/TRELLIS-image-large")
    p.add_argument("--wan_ckpt", default="/lustre/1230003454/hf_models/Wan2.2-Fun-A14B-InP")
    p.add_argument("--fun_config",
                   default="/lustre/1230003454/current/third_party/VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml")
    p.add_argument("--repo_root", default=str(_REPO_ROOT))
    # Optimization
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--lambda_sds", type=float, default=0.1)
    p.add_argument("--lambda_reg", type=float, default=1.0e-3)
    p.add_argument("--use_agg_init", action="store_true")
    p.add_argument("--agg_init_iters", type=int, default=100)
    p.add_argument("--agg_init_lr", type=float, default=0.05)
    # W-RFSDS sampling
    p.add_argument("--cfg_hi", type=float, default=25.0)
    p.add_argument("--cfg_lo", type=float, default=12.0)
    p.add_argument("--tau_min", type=float, default=0.02)
    p.add_argument("--tau_max", type=float, default=0.98)
    p.add_argument("--sample_shift", type=float, default=5.0)
    # Devices
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--teacher_device", default="cuda:1",
                   help="Wan DiT experts + T5 device; set == device for single GPU")
    # Logging / viz
    p.add_argument("--metric_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="few iters; forces lambda_sds=0 + single GPU")
    return p.parse_args()


# =============================================================================
# Loaders
# =============================================================================

def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_inputs(bdir: Path, device: torch.device) -> Dict:
    """Load exactly the artifacts the texture refine needs (geometry/joint frozen)."""
    z_slat0 = torch.load(bdir / "z_slat0.pt", map_location=device).float()        # [Nc, 8]
    coords = torch.from_numpy(
        np.load(bdir / "z_slat_coords.npy").astype(np.int32)
    ).to(device)                                                                  # [Nc, 4]
    Nc = int(z_slat0.shape[0])
    if tuple(coords.shape) != (Nc, 4):
        raise RuntimeError(f"z_slat_coords {tuple(coords.shape)} != ({Nc}, 4)")

    slat_std = torch.load(bdir / "slat_std.pt", map_location=device).float().reshape(-1)
    if slat_std.numel() != z_slat0.shape[1]:
        raise RuntimeError(f"slat_std {tuple(slat_std.shape)} != z_slat0 ch {z_slat0.shape[1]}")

    gpi = torch.from_numpy(
        np.load(bdir / "gaussian_parent_idx.npy").astype(np.int64)
    ).to(device)                                                                  # [Nc * 32]

    U_object = coords[:, 1:].to(torch.int64)                                      # [Nc, 3]

    # --- frozen per-voxel MOVE mask (binary) from move evidence over states ---
    O_move = np.load(bdir / "O_move_per_state.npy").astype(np.float32)            # [K,64,64,64]
    R = TRELLIS_OCC_RES
    O_move = O_move.reshape(O_move.shape[0], R * R * R)
    flat = (U_object[:, 0] * R * R + U_object[:, 1] * R + U_object[:, 2]).cpu().numpy()
    m_per_state_at_U = O_move[:, flat]                                            # [K, Nc]
    m_voxel = torch.from_numpy((m_per_state_at_U > 0.5).any(axis=0)).to(device).float()  # [Nc]

    # --- frozen joint (primary) ---
    psi_full = _load_json(bdir / "psi_0.json")
    psi = psi_full["psi"]
    axis = torch.tensor(psi["axis"], dtype=torch.float32, device=device)
    axis = axis / axis.norm().clamp_min(1.0e-8)
    origin = torch.tensor(psi["origin"], dtype=torch.float32, device=device)
    type_logit = torch.tensor(float(psi["type_logit"]), dtype=torch.float32, device=device)
    theta_max = F.softplus(torch.tensor(float(psi["theta_limit_raw"]), device=device))
    disp_max = F.softplus(torch.tensor(float(psi["disp_limit_raw"]), device=device))
    joint_type = str(psi_full.get("joint_type", "revolute"))
    committed_type = "prismatic" if joint_type.lower().startswith("pri") else "revolute"
    phi_u = torch.from_numpy(np.load(bdir / "phi_0.npy")).float().to(device)      # [K] u_shifted

    # --- real pure GT images (NEVER eval renders) ---
    pure_K = torch.load(bdir / "pure_state_targets_K3HW.pt", map_location=device).float()
    if pure_K.max() > 1.5:
        pure_K = pure_K / 255.0
    pure_K = pure_K.clamp(0.0, 1.0)                                               # [K,3,H,W]
    H, W = int(pure_K.shape[2]), int(pure_K.shape[3])

    # --- cached Wan Fun-InP condition (already encodes real s_0 / s_5 endpoints) ---
    wan_cond_raw = None
    wcp = bdir / "wan_cond_cached.pt"
    if wcp.is_file():
        wan_cond_raw = torch.load(wcp, map_location="cpu")

    return {
        "z_slat0": z_slat0, "coords": coords, "Nc": Nc, "U_object": U_object,
        "slat_std": slat_std, "gaussian_parent_idx": gpi, "m_voxel": m_voxel,
        "axis": axis, "origin": origin, "type_logit": type_logit,
        "theta_max": theta_max, "disp_max": disp_max,
        "committed_type": committed_type, "phi_u": phi_u,
        "pure_K": pure_K, "H": H, "W": W, "wan_cond_raw": wan_cond_raw,
    }


def load_trellis_dgs(trellis_path: str, device: torch.device) -> DGSWithParent:
    pipe = build_trellis_pipeline(device=str(device), pretrained=trellis_path)
    d_gs = pipe.models["slat_decoder_gs"]
    for p in d_gs.parameters():
        p.requires_grad_(False)
    d_gs.eval()
    return DGSWithParent(d_gs_frozen=d_gs).to(device)


# =============================================================================
# Texture math
# =============================================================================

def slat_z_from_delta(z_slat0: torch.Tensor, slat_std: torch.Tensor,
                      delta_z: torch.Tensor, scale: float = 3.0) -> torch.Tensor:
    """method.md S4: z = z_slat0 + scale * slat_std * tanh(delta_z)."""
    return z_slat0 + scale * slat_std.unsqueeze(0) * torch.tanh(delta_z)


def decode_voxel_color(d_gs_w: DGSWithParent, z: torch.Tensor,
                       coords: torch.Tensor, Nc: int) -> torch.Tensor:
    """Per-voxel RGB (mean over the 32 child Gaussians' SH0 DC), in [0, 1]."""
    sparse_in = sp.SparseTensor(feats=z, coords=coords)
    gauss, _parent = d_gs_w(sparse_in)
    dc = gauss._features_dc.reshape(gauss._features_dc.shape[0], -1)[:, :3]       # [Nc*32, 3]
    g_per = dc.shape[0] // Nc
    per_vox = dc.reshape(Nc, g_per, 3).mean(dim=1)
    return (0.5 + _SH_C0 * per_vox).clamp(0.0, 1.0)


def build_render_inputs(d_gs_w: DGSWithParent, z: torch.Tensor, coords: torch.Tensor,
                        gaussian_parent_idx: torch.Tensor,
                        m_voxel: torch.Tensor) -> RenderInputs:
    """Decode z -> canonical Gaussians; g=1 (whole object), m from frozen move mask."""
    sparse_in = sp.SparseTensor(feats=z, coords=coords)
    gauss, _parent = d_gs_w(sparse_in)
    n_gauss = gauss.get_xyz.shape[0]
    g_per_gauss = torch.ones(n_gauss, device=z.device, dtype=torch.float32)
    m_per_gauss = m_voxel[gaussian_parent_idx]
    return RenderInputs(
        xyz_canon=gauss.get_xyz,
        opacity_canon=gauss.get_opacity,
        rot_canon=gauss.get_rotation,
        scale_canon=gauss.get_scaling,
        sh_canon=gauss._features_dc,
        g_per_gauss=g_per_gauss,
        m_per_gauss=m_per_gauss,
    )


def make_frozen_joint(data: Dict) -> JointParams:
    return JointParams(
        axis=data["axis"], origin=data["origin"], type_logit=data["type_logit"],
        theta_max=data["theta_max"], disp_max=data["disp_max"],
    )


def frozen_phi(phi_u: torch.Tensor, theta_max: torch.Tensor,
               disp_max: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interpolate frozen canonical-shifted u (K states) to F frames, scale to units."""
    u_render = linear_interp_through(phi_u, n_out=F_FRAMES)
    return u_render * theta_max, u_render * disp_max


def render_video(d_gs_w: DGSWithParent, z: torch.Tensor, data: Dict,
                 camera, phi_rev: torch.Tensor, phi_pri: torch.Tensor) -> torch.Tensor:
    """-> [F, 3, H, W] in [0, 1] at the frozen joint trajectory."""
    render_inputs = build_render_inputs(
        d_gs_w, z, data["coords"], data["gaussian_parent_idx"], data["m_voxel"],
    )
    joint = make_frozen_joint(data)
    return render_21_with_warp(
        render_inputs, joint, phi_rev, phi_pri, camera,
        type_soft=None, committed_type=data["committed_type"], sh_degree=0,
    )


# =============================================================================
# Camera (FreeArt3D canonical, framed on the decoded object bbox)
# =============================================================================

def build_camera(d_gs_w: DGSWithParent, data: Dict, device: torch.device):
    z0 = data["z_slat0"]
    sparse_in = sp.SparseTensor(feats=z0, coords=data["coords"])
    with torch.no_grad():
        gauss, _ = d_gs_w(sparse_in)
        xyz = gauss.get_xyz
        mn, mx = xyz.amin(dim=0), xyz.amax(dim=0)
    center = 0.5 * (mn + mx)
    max_extent = float((mx - mn).amax().item())
    cam_cfg = StageDCameraConfig.freeart3d_canonical(
        image_h=int(data["H"]), image_w=int(data["W"]),
        object_scale=max_extent,
        object_center=tuple(float(c) for c in center.tolist()),
    )
    return build_locked_camera(cam_cfg, device=device, dtype=torch.float32)


# =============================================================================
# Anchor + metrics
# =============================================================================

def _observed_frame_indices() -> List[int]:
    return list(STATE_INDICES)                                                    # [0,4,8,12,16,20]


def _gt_mask(target_3HW: torch.Tensor, thresh: float = 0.02) -> torch.Tensor:
    """Foreground mask: where the real pure image is not background-black. [1,H,W]."""
    return (target_3HW.amax(dim=0, keepdim=True) > thresh).float()


def _resize_to(render_3HW: torch.Tensor, H: int, W: int) -> torch.Tensor:
    if render_3HW.shape[-2:] == (H, W):
        return render_3HW
    return F.interpolate(render_3HW.unsqueeze(0), size=(H, W),
                         mode="bilinear", align_corners=False).squeeze(0)


def anchor_loss(video_F3HW: torch.Tensor, pure_K: torch.Tensor) -> torch.Tensor:
    """Masked-L1 over the 6 observed frames vs the real pure images."""
    H, W = int(pure_K.shape[2]), int(pure_K.shape[3])
    frames = _observed_frame_indices()
    total = video_F3HW.new_zeros(())
    for k, fi in enumerate(frames):
        render = _resize_to(video_F3HW[fi], H, W)
        target = pure_K[k]
        mask = _gt_mask(target)
        diff = (render - target).abs() * mask
        total = total + diff.sum() / mask.sum().clamp_min(1.0) / 3.0
    return total / len(frames)


@torch.no_grad()
def compute_metrics(video_F3HW: torch.Tensor, pure_K: torch.Tensor
                    ) -> Tuple[float, float, List[float], List[float]]:
    """Mean PSNR / SSIM (skimage) render@observed vs real pure, on the masked union box."""
    H, W = int(pure_K.shape[2]), int(pure_K.shape[3])
    frames = _observed_frame_indices()
    psnrs, ssims = [], []
    for k, fi in enumerate(frames):
        render = _resize_to(video_F3HW[fi], H, W).clamp(0.0, 1.0)
        target = pure_K[k].clamp(0.0, 1.0)
        r = render.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        t = target.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        psnrs.append(float(psnr_fn(t, r, data_range=1.0)))
        ssims.append(float(ssim_fn(t, r, channel_axis=2, data_range=1.0)))
    return float(np.mean(psnrs)), float(np.mean(ssims)), psnrs, ssims


def save_side_by_side(video_F3HW: torch.Tensor, pure_K: torch.Tensor,
                      out_dir: Path, tag: str, states: Tuple[int, ...] = (0, 2, 5)) -> None:
    H, W = int(pure_K.shape[2]), int(pure_K.shape[3])
    frames = _observed_frame_indices()
    for k in states:
        fi = frames[k]
        render = _resize_to(video_F3HW[fi], H, W).clamp(0.0, 1.0)
        target = pure_K[k].clamp(0.0, 1.0)
        pair = torch.cat([render, target], dim=2)                                # [3,H,2W]
        arr = (pair.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"{tag}_state{k:02d}_render_vs_real.png")


# =============================================================================
# Wan W-RFSDS condition (reuse cached fun_video/fun_mask; keep T5 on teacher GPU)
# =============================================================================

def prepare_wan_cond_keep_t5(ctx, wan_cond_raw: Dict, data: Dict,
                             teacher_device: torch.device) -> Dict:
    """Build the prepared Fun-InP condition from the cached fun_video/fun_mask.

    Mirrors ``_prepare_fun_inp_rfsds_condition`` but encodes the prompt with T5 on
    ``teacher_device`` and does NOT move T5 to CPU afterward (user: no t5_cpu).
    Endpoints (real s_0 / s_5) are already baked into the cached fun_video frames
    0 / -1, so we consume them directly.
    """
    import math
    pipe = ctx.fun_pipeline
    H, W = int(ctx.resolution_hw[0]), int(ctx.resolution_hw[1])
    F_count = int(ctx.frame_num)

    video = wan_cond_raw["fun_video"].to(device=ctx.device, dtype=torch.float32)
    mask_video = wan_cond_raw["fun_mask"].to(device=ctx.device, dtype=torch.float32)
    if tuple(video.shape) != (1, 3, F_count, H, W):
        raise RuntimeError(f"fun_video {tuple(video.shape)} != (1,3,{F_count},{H},{W})")
    if tuple(mask_video.shape) != (1, 1, F_count, H, W):
        raise RuntimeError(f"fun_mask {tuple(mask_video.shape)} != (1,1,{F_count},{H},{W})")

    pos_prompt = wan_cond_raw["pos_prompt"]
    neg_prompt = wan_cond_raw["neg_prompt"]
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        pos_prompt, neg_prompt, True, num_videos_per_prompt=1,
        max_sequence_length=512, device=teacher_device,
    )
    if isinstance(prompt_embeds, torch.Tensor):
        in_context = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        in_context = in_context.to(device=teacher_device, dtype=ctx.dtype)
    else:
        in_context = [c.to(device=teacher_device, dtype=ctx.dtype)
                      for c in (negative_prompt_embeds + prompt_embeds)]

    flat_video = video.permute(0, 2, 1, 3, 4).reshape(F_count, 3, H, W)
    init_video = pipe.image_processor.preprocess(flat_video, height=H, width=W)
    init_video = init_video.to(device=ctx.device, dtype=torch.float32)
    init_video = init_video.reshape(1, F_count, 3, H, W).permute(0, 2, 1, 3, 4)

    flat_mask = mask_video.permute(0, 2, 1, 3, 4).reshape(F_count, 1, H, W)
    mask_condition = pipe.mask_processor.preprocess(flat_mask, height=H, width=W)
    mask_condition = mask_condition.to(device=ctx.device, dtype=torch.float32)
    mask_condition = mask_condition.reshape(1, F_count, 1, H, W).permute(0, 2, 1, 3, 4)

    masked_video = init_video * (torch.tile(mask_condition, [1, 3, 1, 1, 1]) < 0.5)
    from pipelines.stage_d.w_rfsds import _resize_mask_like_official
    _, masked_video_latents = pipe.prepare_mask_latents(
        None, masked_video, 1, H, W, ctx.dtype, ctx.device, None, True,
        noise_aug_strength=None,
    )
    mask_condition = torch.concat(
        [torch.repeat_interleave(mask_condition[:, :, 0:1], repeats=4, dim=2),
         mask_condition[:, :, 1:]],
        dim=2,
    )
    mask_condition = mask_condition.view(1, mask_condition.shape[2] // 4, 4, H, W).transpose(1, 2)
    mask_latents = _resize_mask_like_official(1 - mask_condition, masked_video_latents, True)
    mask_latents = mask_latents.to(device=ctx.device, dtype=ctx.dtype)
    masked_video_latents = masked_video_latents.to(device=ctx.device, dtype=ctx.dtype)

    y = torch.cat(
        [torch.cat([mask_latents, mask_latents], dim=0),
         torch.cat([masked_video_latents, masked_video_latents], dim=0)],
        dim=1,
    ).to(device=teacher_device, dtype=ctx.dtype)

    patch_size = tuple(int(v) for v in pipe.transformer.config.patch_size)
    seq_len = int(math.ceil(
        (ctx.h_latent * ctx.w_latent) / (patch_size[1] * patch_size[2]) * ctx.f_latent
    ))

    return {
        "backend": "fun_inp",
        "in_context": in_context,
        "y_guidance": y,
        "seq_len": int(seq_len),
        "_prepared_backend": "fun_inp",
    }


def gpu_mem(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    free, total = torch.cuda.mem_get_info(device)
    used = (total - free) / 1024 ** 3
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    return f"used={used:.1f}GB peak={peak:.1f}GB/{total / 1024 ** 3:.0f}GB"


# =============================================================================
# Aggregated-init pre-fit
# =============================================================================

def prefit_agg_init(d_gs_w: DGSWithParent, delta_z: torch.Tensor, data: Dict,
                    agg_path: Path, n_iters: int, lr: float, scale: float = 3.0) -> None:
    """Pre-fit delta_z so decoded per-voxel color matches aggregated_init target_color."""
    agg = torch.load(agg_path, map_location=data["z_slat0"].device)
    agg_coords = agg["coords"].to(torch.int64).to(data["z_slat0"].device)        # [Nc,3]
    if not torch.equal(agg_coords, data["U_object"]):
        raise RuntimeError("aggregated_init coords do not match z_slat_coords row-for-row")
    target_color = agg["target_color"].float().to(data["z_slat0"].device).clamp(0.0, 1.0)
    valid = agg["valid"].bool().to(data["z_slat0"].device)
    opt = torch.optim.Adam([delta_z], lr=lr)
    z0, std, coords, Nc = data["z_slat0"], data["slat_std"], data["coords"], data["Nc"]
    print(f"[agg_init] pre-fitting delta_z to target_color: "
          f"valid={int(valid.sum())}/{Nc} for {n_iters} iters", flush=True)
    for it in range(n_iters):
        z = slat_z_from_delta(z0, std, delta_z, scale)
        color = decode_voxel_color(d_gs_w, z, coords, Nc)
        loss = ((color[valid] - target_color[valid]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 25 == 0 or it == n_iters - 1:
            print(f"[agg_init] it={it} color_mse={float(loss.detach()):.5f}", flush=True)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    smoke = bool(args.smoke)
    lambda_sds = 0.0 if smoke else float(args.lambda_sds)
    device = torch.device(args.device)
    teacher_device = torch.device(args.teacher_device)
    use_wan = lambda_sds > 0.0
    if not use_wan:
        teacher_device = device

    bdir = Path(args.bootstrap_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "losses.jsonl"
    n_iters = 20 if smoke else int(args.iters)

    print(f"[setup] device={device} teacher={teacher_device} use_wan={use_wan} "
          f"lambda_sds={lambda_sds} iters={n_iters}", flush=True)

    data = load_inputs(bdir, device)
    print(f"[data] Nc={data['Nc']} HxW={data['H']}x{data['W']} "
          f"joint={data['committed_type']} theta_max={float(data['theta_max']):.4f} "
          f"disp_max={float(data['disp_max']):.4f} "
          f"n_move_voxel={int(data['m_voxel'].sum())} phi_u={[round(x,3) for x in data['phi_u'].tolist()]}",
          flush=True)

    d_gs_w = load_trellis_dgs(args.trellis, device)
    camera = build_camera(d_gs_w, data, device)
    phi_rev, phi_pri = frozen_phi(data["phi_u"], data["theta_max"], data["disp_max"])

    # --- learnable texture param ---
    delta_z = torch.zeros_like(data["z_slat0"], requires_grad=True)

    if args.use_agg_init:
        if args.agg_init is None:
            raise ValueError("--use_agg_init requires --agg_init <aggregated_init.pt>")
        prefit_agg_init(d_gs_w, delta_z, data, Path(args.agg_init).resolve(),
                        n_iters=int(args.agg_init_iters), lr=float(args.agg_init_lr))

    # --- Wan ctx (only if SDS on) ---
    ctx = None
    wan_cond = None
    if use_wan:
        from pipelines.stage_d.w_rfsds import load_wan_fun_inp_for_rfsds, w_rfsds_loss
        if data["wan_cond_raw"] is None:
            raise FileNotFoundError("wan_cond_cached.pt missing; required for W-RFSDS")
        print(f"[wan] loading Fun-InP: VAE on {device}, experts+T5 on {teacher_device}",
              flush=True)
        ctx = load_wan_fun_inp_for_rfsds(
            model_dir=args.wan_ckpt, repo_root=args.repo_root, device=device,
            fun_config_path=args.fun_config, sample_shift=float(args.sample_shift),
            frame_num=F_FRAMES, resolution_hw=(data["H"], data["W"]),
            teacher_device=teacher_device,
        )
        wan_cond = prepare_wan_cond_keep_t5(ctx, data["wan_cond_raw"], data, teacher_device)
        torch.cuda.reset_peak_memory_stats(device)
        print(f"[wan] ready (T5 kept on {teacher_device}). {gpu_mem(device)}", flush=True)

    optimizer = torch.optim.Adam([delta_z], lr=float(args.lr))
    gen = torch.Generator(device="cpu").manual_seed(int(args.seed))

    # --- iter-0 metrics + viz (before any step) ---
    with torch.no_grad():
        z0 = slat_z_from_delta(data["z_slat0"], data["slat_std"], delta_z)
        video0 = render_video(d_gs_w, z0, data, camera, phi_rev, phi_pri)
        psnr0, ssim0, p0_list, s0_list = compute_metrics(video0, data["pure_K"])
        save_side_by_side(video0, data["pure_K"], viz_dir, tag="iter0000")
    print(f"[iter 0] PSNR={psnr0:.4f} SSIM={ssim0:.4f} per_state_psnr="
          f"{[round(x,2) for x in p0_list]}", flush=True)

    best = {"iter0": {"psnr": psnr0, "ssim": ssim0}}
    log_f = log_path.open("w", encoding="utf-8")
    rec0 = {"iter": 0, "psnr": psnr0, "ssim": ssim0,
            "psnr_per_state": p0_list, "ssim_per_state": s0_list, "phase": "init"}
    log_f.write(json.dumps(rec0) + "\n")
    log_f.flush()

    last_psnr, last_ssim = psnr0, ssim0
    for it in range(1, n_iters + 1):
        z = slat_z_from_delta(data["z_slat0"], data["slat_std"], delta_z)
        video = render_video(d_gs_w, z, data, camera, phi_rev, phi_pri).clamp(0.0, 1.0)

        l_anchor = anchor_loss(video, data["pure_K"])
        reg = (delta_z ** 2).mean()
        loss = l_anchor + float(args.lambda_reg) * reg

        sds_val = 0.0
        if use_wan:
            tau = float(torch.empty(1).uniform_(args.tau_min, args.tau_max, generator=gen).item())
            frac = it / max(1, n_iters)
            cfg_scale = float(args.cfg_hi + (args.cfg_lo - args.cfg_hi) * frac)
            video_3FHW = video.permute(1, 0, 2, 3).contiguous()                   # [3,F,H,W]
            l_sds = w_rfsds_loss(video_3FHW, wan_cond, ctx, tau=tau, cfg_scale=cfg_scale)
            loss = loss + lambda_sds * l_sds
            sds_val = float(l_sds.detach().cpu())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it % int(args.metric_every) == 0 or it == n_iters:
            with torch.no_grad():
                z_eval = slat_z_from_delta(data["z_slat0"], data["slat_std"], delta_z)
                video_eval = render_video(d_gs_w, z_eval, data, camera, phi_rev, phi_pri)
                psnr, ssim, p_list, s_list = compute_metrics(video_eval, data["pure_K"])
            last_psnr, last_ssim = psnr, ssim
            rec = {"iter": it, "loss": float(loss.detach().cpu()),
                   "anchor": float(l_anchor.detach().cpu()), "sds": sds_val,
                   "reg": float(reg.detach().cpu()), "psnr": psnr, "ssim": ssim,
                   "psnr_per_state": p_list, "ssim_per_state": s_list,
                   "mem": gpu_mem(device)}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(f"[iter {it}] loss={rec['loss']:.4f} anchor={rec['anchor']:.4f} "
                  f"sds={sds_val:.4f} PSNR={psnr:.4f} SSIM={ssim:.4f}", flush=True)

    # --- final viz + save delta_z ---
    with torch.no_grad():
        z_final = slat_z_from_delta(data["z_slat0"], data["slat_std"], delta_z)
        video_final = render_video(d_gs_w, z_final, data, camera, phi_rev, phi_pri)
        psnr_f, ssim_f, pf_list, sf_list = compute_metrics(video_final, data["pure_K"])
        save_side_by_side(video_final, data["pure_K"], viz_dir, tag="final")
    log_f.close()

    torch.save({"delta_z": delta_z.detach().cpu(),
                "z_final": z_final.detach().cpu(),
                "coords": data["coords"].detach().cpu()},
               out_dir / "delta_z.pt")

    summary = {
        "psnr_iter0": psnr0, "ssim_iter0": ssim0,
        "psnr_final": psnr_f, "ssim_final": ssim_f,
        "psnr_rose": bool(psnr_f > psnr0), "ssim_rose": bool(ssim_f > ssim0),
        "n_iters": n_iters, "lambda_sds": lambda_sds,
        "use_agg_init": bool(args.use_agg_init), "use_wan": use_wan,
        "psnr_per_state_final": pf_list, "ssim_per_state_final": sf_list,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] PSNR {psnr0:.4f} -> {psnr_f:.4f} (rose={summary['psnr_rose']}) | "
          f"SSIM {ssim0:.4f} -> {ssim_f:.4f} (rose={summary['ssim_rose']}) | out={out_dir}",
          flush=True)


if __name__ == "__main__":
    main()

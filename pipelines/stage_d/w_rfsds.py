"""W-RFSDS distillation from Wan2.2 I2V into our rendered video.

The gradient signal that drives all Stage D learning (geometry, joint,
gates) flows from this single function. The math is CHORD Eq.(3)
(arxiv:2601.04194):

    L_W-RFSDS(theta; z, y) = E_{sigma ~ w_hat(sigma), eps}
                              [ (v_hat(z_sigma; sigma, y) - eps + z) . dz/dtheta ]

where:
    - z       = Wan_VAE.encode(rendered_video)              (grad-enabled)
    - z_sigma = (1 - sigma) * z + sigma * eps                (CHORD forward)
    - v_hat   = Wan2.2 DiT velocity prediction              (no-grad, dual-expert)
    - sigma   = tau ~ inverse-CDF of logit-normal(mean, std) (schedules.py)

Implementation uses ``residual = v_pred - eps + z.detach()`` and forms the
SDS loss as ``loss = (residual.detach() * z).sum() / z.numel()``. The
``.detach()`` chain ensures only z's gradient w.r.t. the renderer flows
backward — the residual is treated as a constant target direction.

Dual-expert switch (Wan2.2 wan_i2v_A14B.py:36 boundary=0.900):
    high_noise_model  if  tau * num_train_timesteps >= 900
    low_noise_model   otherwise

Memory:
    - Wan VAE encode + backward is the expensive step (CHORD §D
      Limitations explicitly acknowledges this).
    - Wan DiT forwards (cond + uncond) run under torch.no_grad so we don't
      retain activations for 40-layer DiT * 2 forwards.
    - The VAE latent H/W follows the Stage A actual output H/W. The
      bookkeeping is contained inside this module.
"""
from __future__ import annotations

import math
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import (
    WAN_VAE_STRIDE,
    WAN_BOUNDARY_NORMALIZED,
    WAN_LATENT_CH,
    WAN_NUM_TRAIN_TIMESTEPS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Wan2.2 component loader
# =============================================================================

def _ensure_wan_on_sys_path(repo_root: str) -> None:
    """Insert ``<repo_root>/Wan2.2`` on ``sys.path`` (same as Stage A)."""
    wan_root = os.path.join(repo_root, "Wan2.2")
    if not os.path.isdir(wan_root):
        raise RuntimeError(
            f"Stage D W-RFSDS requires vendored Wan2.2 at {wan_root!r}; "
            "directory not found."
        )
    if wan_root not in sys.path:
        sys.path.insert(0, wan_root)


@dataclass
class WanRFSDSContext:
    """Holds the frozen Wan2.2 components and a few invariants.

    Loaded once at Stage D init and passed by reference into every
    ``w_rfsds_loss`` call. All weights are frozen and on the same device.

    Note on ``sample_shift``: Wan's scheduler (fm_solvers_unipc.py:109-119)
    applies a shift transformation to the sigma schedule:
        sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)
    The W-RFSDS forward ``z_sigma = (1-sigma)*z + sigma*eps`` must use the
    SHIFTED sigma so the noisy latent matches Wan's training distribution.
    Default shift=5.0 for 480p I2V; see Stage A's ``sample_shift`` arg.
    """
    wan_vae: Any                 # Wan2_1_VAE; .encode([Tensor [3, T, H, W] in [-1,1]])
    low_noise_model: Any         # WanModel; tau < 0.9 expert
    high_noise_model: Optional[Any]  # WanModel; tau >= 0.9 expert
    boundary_in_t_units: float   # = WAN_BOUNDARY_NORMALIZED * (num_train_timesteps)
    num_train_timesteps: int
    sample_shift: float          # ★ C2 fix: Wan scheduler shift parameter (default 5.0)
    frame_num: int
    resolution_hw: Tuple[int, int]
    f_latent: int
    h_latent: int
    w_latent: int
    device: torch.device
    dtype: torch.dtype
    offload_dit: bool = False
    backend: str = "i2v"
    fun_pipeline: Any = None
    fun_boundary_normalized: Optional[float] = None
    teacher_device: Optional[torch.device] = None  # dual-GPU: experts/T5 device; None => same as `device`


def load_wan_for_rfsds(
    wan_ckpt_dir: str,
    repo_root: str,
    device: torch.device,
    convert_model_dtype: bool = True,
    device_id: int = 0,
    sample_shift: float = 5.0,
    frame_num: int = 21,
    resolution_hw: Tuple[int, int] = (464, 832),
    expert_mode: str = "both",
    offload_dit: bool = False,
) -> WanRFSDSContext:
    """Load the three Wan2.2 components we need for W-RFSDS.

    Uses the same ``wan.image2video.WanI2V`` class that Stage A uses, but
    freezes all parameters. DiT experts may be kept on CPU between W-RFSDS
    calls because they are frozen teachers and do not participate in backward.
    The T5 text encoder is NOT loaded here because the
    ``wan_cond_cached`` dict from Bootstrap already contains the encoded
    text embeddings (context / context_null) as tensors.

    Parameters
    ----------
    wan_ckpt_dir : str
        Local directory containing the Wan2.2-I2V-A14B weights (same as
        what Stage A uses).
    repo_root : str
        Absolute path to ``mine/`` (so we can locate ``mine/Wan2.2/``).
    device : torch.device
    convert_model_dtype : bool
        If True, casts the DiT and VAE to ``config.param_dtype`` (bf16 by
        default for A14B). Matches Stage A's setting.
    device_id : int
        CUDA device index (matches the ``device`` arg).
    """
    # Same offline guards as Stage A; must precede HF imports.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    _ensure_wan_on_sys_path(repo_root)

    if expert_mode not in ("both", "low_only"):
        raise ValueError(
            f"expert_mode must be 'both' or 'low_only', got {expert_mode!r}"
        )

    from wan.configs.wan_i2v_A14B import i2v_A14B as wan_cfg
    from wan.modules.model import WanModel
    from wan.modules.vae2_1 import Wan2_1_VAE
    load_kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
    if convert_model_dtype:
        load_kwargs["torch_dtype"] = wan_cfg.param_dtype

    logger.info("[stage_d][wan] loading VAE")
    vae = Wan2_1_VAE(
        vae_pth=os.path.join(wan_ckpt_dir, wan_cfg.vae_checkpoint),
        device=device,
    )
    logger.info("[stage_d][wan] loading low-noise DiT from %s",
                wan_cfg.low_noise_checkpoint)
    low_noise = WanModel.from_pretrained(
        wan_ckpt_dir, subfolder=wan_cfg.low_noise_checkpoint, **load_kwargs,
    )
    logger.info("[stage_d][wan] low-noise DiT loaded")
    high_noise = None
    if expert_mode == "both":
        logger.info("[stage_d][wan] loading high-noise DiT from %s",
                    wan_cfg.high_noise_checkpoint)
        high_noise = WanModel.from_pretrained(
            wan_ckpt_dir, subfolder=wan_cfg.high_noise_checkpoint, **load_kwargs,
        )
        logger.info("[stage_d][wan] high-noise DiT loaded")

    for model in [m for m in (low_noise, high_noise) if m is not None]:
        model.eval().requires_grad_(False)
        if convert_model_dtype and next(model.parameters()).dtype != wan_cfg.param_dtype:
            logger.info("[stage_d][wan] converting DiT to %s", wan_cfg.param_dtype)
            model.to(wan_cfg.param_dtype)
        if offload_dit:
            logger.info("[stage_d][wan] keeping DiT on CPU between W-RFSDS calls")
            model.to("cpu")
        else:
            logger.info("[stage_d][wan] moving DiT to %s", device)
            model.to(device)
    dit_resident = "cpu" if offload_dit else str(device)
    logger.info("[stage_d][wan] DiT expert mode=%s resident on %s",
                expert_mode, dit_resident)

    # Freeze everything.
    for m in [m for m in (low_noise, high_noise) if m is not None]:
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()

    # vae may be a Wan2_1_VAE wrapper holding ``vae.model`` + ``vae.scale``;
    # the inner nn.Module is what carries parameters.
    if hasattr(vae, "model"):
        for p in vae.model.parameters():
            p.requires_grad_(False)
        vae.model.eval()

    # ``boundary * num_train_timesteps`` gives the cutoff in [0, 1000) scale
    # (matches wan/image2video.py:341).
    num_train_ts = int(wan_cfg.num_train_timesteps)
    if num_train_ts != WAN_NUM_TRAIN_TIMESTEPS:
        raise RuntimeError(
            f"Wan config num_train_timesteps={num_train_ts} differs from "
            f"our WAN_NUM_TRAIN_TIMESTEPS={WAN_NUM_TRAIN_TIMESTEPS}; "
            "constants need to be re-checked."
        )
    boundary = float(getattr(wan_cfg, "boundary", WAN_BOUNDARY_NORMALIZED))
    if abs(boundary - WAN_BOUNDARY_NORMALIZED) > 1.0e-6:
        raise RuntimeError(
            f"Wan config boundary={boundary} differs from our "
            f"WAN_BOUNDARY_NORMALIZED={WAN_BOUNDARY_NORMALIZED}; "
            "constants need to be re-checked."
        )
    boundary_in_t_units = boundary * float(num_train_ts)

    H, W = int(resolution_hw[0]), int(resolution_hw[1])
    if H % 16 != 0 or W % 16 != 0:
        raise ValueError(
            f"resolution_hw=({H}, {W}) must align to Wan VAE stride 8 and "
            "DiT patch size 2"
        )
    F_count = int(frame_num)
    if F_count <= 1 or (F_count - 1) % int(WAN_VAE_STRIDE[0]) != 0:
        raise ValueError(
            f"frame_num must be of the form 4n+1 for Wan VAE stride 4; got {F_count}"
        )
    f_latent = (F_count - 1) // int(WAN_VAE_STRIDE[0]) + 1
    h_latent = H // int(WAN_VAE_STRIDE[1])
    w_latent = W // int(WAN_VAE_STRIDE[2])

    # Param dtype (bf16 for A14B by default after convert_model_dtype).
    sample_param = next(low_noise.parameters())
    param_dtype = sample_param.dtype

    return WanRFSDSContext(
        wan_vae=vae,
        low_noise_model=low_noise,
        high_noise_model=high_noise,
        boundary_in_t_units=boundary_in_t_units,
        num_train_timesteps=num_train_ts,
        sample_shift=float(sample_shift),          # ★ C2 fix
        frame_num=F_count,
        resolution_hw=(H, W),
        f_latent=int(f_latent),
        h_latent=int(h_latent),
        w_latent=int(w_latent),
        device=device,
        dtype=param_dtype,
        offload_dit=bool(offload_dit),
    )


def _resolve_fun_config_path(fun_config_path: Optional[str]) -> str:
    if fun_config_path is not None:
        path = os.path.abspath(os.fspath(fun_config_path))
    else:
        videox_root = os.environ.get("VIDEOX_FUN_ROOT")
        if videox_root is None:
            videox_root = os.path.abspath(os.path.expanduser("~/VideoX-Fun"))
        path = os.path.join(videox_root, "config", "wan2.2", "wan_civitai_i2v.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"VideoX-Fun config not found: {path}. Set fun_inp_config_path or VIDEOX_FUN_ROOT."
        )
    return path


def _ensure_videox_fun_on_sys_path(fun_config_path: str) -> None:
    videox_root = os.environ.get("VIDEOX_FUN_ROOT")
    if videox_root is None:
        root = os.path.abspath(os.path.join(os.path.dirname(fun_config_path), os.pardir, os.pardir))
    else:
        root = os.path.abspath(os.fspath(videox_root))
    if not os.path.isdir(os.path.join(root, "videox_fun")):
        raise FileNotFoundError(
            f"VideoX-Fun source tree not found at {root}. Set VIDEOX_FUN_ROOT."
        )
    if root not in sys.path:
        sys.path.insert(0, root)


def _filter_init_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    import inspect

    allowed = set(inspect.signature(cls.__init__).parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def load_wan_fun_inp_for_rfsds(
    model_dir: str,
    repo_root: str,
    device: torch.device,
    fun_config_path: Optional[str] = None,
    device_id: int = 0,
    sample_shift: float = 5.0,
    frame_num: int = 21,
    resolution_hw: Tuple[int, int] = (464, 832),
    expert_mode: str = "both",
    teacher_device: Optional[torch.device] = None,
    weight_dtype: torch.dtype = torch.bfloat16,
) -> WanRFSDSContext:
    """Load VideoX-Fun Wan2.2-Fun-A14B-InP for W-RFSDS.

    The returned context uses the same public ``w_rfsds_loss`` entry point as
    I2V, but the loss dispatches to the Fun-InP transformer API and its
    first/last-frame ``video``/``mask_video`` condition.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    _ = repo_root
    _ = device_id

    if expert_mode not in ("both", "low_only"):
        raise ValueError(
            f"expert_mode must be 'both' or 'low_only', got {expert_mode!r}"
        )

    model_dir = os.path.abspath(os.fspath(model_dir))
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Fun-InP model_dir not found: {model_dir}")
    config_path = _resolve_fun_config_path(fun_config_path)
    _ensure_videox_fun_on_sys_path(config_path)

    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    from videox_fun.models import (
        AutoencoderKLWan,
        AutoencoderKLWan3_8,
        AutoTokenizer,
        Wan2_2Transformer3DModel,
        WanT5EncoderModel,
    )
    from videox_fun.pipeline import Wan2_2FunInpaintPipeline
    from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
    from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    config = OmegaConf.load(config_path)
    transformer_kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"])

    low_subpath = transformer_kwargs.get("transformer_low_noise_model_subpath", "transformer")
    high_subpath = transformer_kwargs.get("transformer_high_noise_model_subpath", "transformer")
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(model_dir, low_subpath),
        transformer_additional_kwargs=transformer_kwargs,
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    is_moe = transformer_kwargs.get("transformer_combination_type", "single") == "moe"
    if is_moe and expert_mode == "both":
        transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(model_dir, high_subpath),
            transformer_additional_kwargs=transformer_kwargs,
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
    else:
        # expert_mode='low_only' (or non-MoE config): skip the high-noise
        # expert entirely. It only serves tau >= boundary (~top 12.5% of the
        # noise range); the low-noise expert covers the rest, including the SDS
        # refinement band. Freeing its ~28 GB is what lets the grad-enabled Wan
        # VAE encode of the 21-frame video fit -- both A14B experts (~56 GB)
        # plus that encode (~18 GB) exceed one 80 GB GPU. _select_fun_inp_
        # transformer falls back to the low-noise model when high is None.
        transformer_2 = None

    vae_kwargs = OmegaConf.to_container(config["vae_kwargs"])
    vae_cls = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[vae_kwargs.get("vae_type", "AutoencoderKLWan")]
    vae = vae_cls.from_pretrained(
        os.path.join(model_dir, vae_kwargs.get("vae_subpath", "vae")),
        additional_kwargs=vae_kwargs,
    ).to(weight_dtype)

    text_kwargs = OmegaConf.to_container(config["text_encoder_kwargs"])
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_dir, text_kwargs.get("tokenizer_subpath", "tokenizer"))
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_dir, text_kwargs.get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=text_kwargs,
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    scheduler_name = str(config.get("scheduler_name", "Flow"))
    scheduler_cls = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[scheduler_name]
    scheduler_kwargs = OmegaConf.to_container(config["scheduler_kwargs"])
    scheduler = scheduler_cls(**_filter_init_kwargs(scheduler_cls, scheduler_kwargs))

    pipe = Wan2_2FunInpaintPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
    # Device placement. Single GPU (default): everything on `device`. Dual-GPU
    # teacher-student split: the frozen Wan experts + T5 (the SDS teacher) go on
    # `teacher_device`, the differentiable Wan VAE stays on `device` with the
    # render/TRELLIS student. Each iter only a ~23 MB latent crosses GPUs (the
    # experts run no_grad and the SDS residual is detached), so both A14B
    # experts stay resident without the 21-frame grad VAE encode OOMing.
    tdev = teacher_device if teacher_device is not None else device
    pipe.vae.to(device)
    pipe.transformer.to(tdev)
    if pipe.transformer_2 is not None:
        pipe.transformer_2.to(tdev)
    pipe.text_encoder.to(tdev)

    for module in (pipe.transformer, pipe.transformer_2, pipe.text_encoder, pipe.vae):
        if module is None:
            continue
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()

    H, W = int(resolution_hw[0]), int(resolution_hw[1])
    F_count = int(frame_num)
    if H % 16 != 0 or W % 16 != 0:
        raise ValueError(
            f"resolution_hw=({H}, {W}) must align to Wan VAE stride 8 and DiT patch size 2"
        )
    if F_count <= 1 or (F_count - 1) % int(WAN_VAE_STRIDE[0]) != 0:
        raise ValueError(
            f"frame_num must be of the form 4n+1 for Wan VAE stride 4; got {F_count}"
        )

    num_train_ts = int(getattr(pipe.scheduler.config, "num_train_timesteps", WAN_NUM_TRAIN_TIMESTEPS))
    boundary = float(transformer_kwargs.get("boundary", 0.875))
    f_latent = (F_count - 1) // int(WAN_VAE_STRIDE[0]) + 1
    h_latent = H // int(WAN_VAE_STRIDE[1])
    w_latent = W // int(WAN_VAE_STRIDE[2])

    return WanRFSDSContext(
        wan_vae=pipe.vae,
        low_noise_model=pipe.transformer,
        high_noise_model=pipe.transformer_2,
        boundary_in_t_units=boundary * float(num_train_ts),
        num_train_timesteps=num_train_ts,
        sample_shift=float(sample_shift),
        frame_num=F_count,
        resolution_hw=(H, W),
        f_latent=int(f_latent),
        h_latent=int(h_latent),
        w_latent=int(w_latent),
        device=device,
        dtype=weight_dtype,
        backend="fun_inp",
        fun_pipeline=pipe,
        fun_boundary_normalized=boundary,
        teacher_device=tdev,
    )


def _shifted_sigma(tau: float, shift: float) -> float:
    """Apply Wan scheduler's sigma shift transformation.

    Wan2.2 ``fm_solvers_unipc.py:109-119``:
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    The model was trained on shifted sigmas, so the W-RFSDS forward sample
    ``z_sigma = (1-sigma)*z + sigma*eps`` must use the same shifted sigma
    to stay in-distribution. ``shift=1`` is the identity (no shift).
    """
    if shift == 1.0:
        return float(tau)
    return float(shift * tau / (1.0 + (shift - 1.0) * tau))


# =============================================================================
# Wan VAE encode helper ([-1, 1] normalization, list interface)
# =============================================================================

def _to_wan_vae_input(video_3FHW_float01: torch.Tensor) -> torch.Tensor:
    """Convert rendered RGB in [0, 1] to Wan VAE expected [-1, 1].

    Wan2.2's ``image2video.py:259`` documents this convention:
        ``img = TF.to_tensor(img).sub_(0.5).div_(0.5)``
    which is exactly ``x * 2 - 1`` after ``to_tensor`` (which divides by 255).
    """
    return video_3FHW_float01 * 2.0 - 1.0


def _resize_rgb_for_wan_sds(
    rgb_3FHW_float01: torch.Tensor,
    resolution_hw: Tuple[int, int],
) -> torch.Tensor:
    H = int(resolution_hw[0])
    W = int(resolution_hw[1])
    if rgb_3FHW_float01.shape[-2:] == (H, W):
        return rgb_3FHW_float01
    rgb_F3HW = rgb_3FHW_float01.permute(1, 0, 2, 3)
    resized = F.interpolate(
        rgb_F3HW, size=(H, W), mode="bilinear", align_corners=False,
    )
    return resized.permute(1, 0, 2, 3)


def _resize_i2v_y_for_wan_sds(
    y: torch.Tensor,
    ctx: WanRFSDSContext,
) -> torch.Tensor:
    target = (int(ctx.f_latent), int(ctx.h_latent), int(ctx.w_latent))
    if tuple(y.shape[-3:]) == target:
        return y
    if y.shape[0] != 20:
        raise ValueError(f"Wan I2V y must have 20 channels, got {tuple(y.shape)}")
    mask = F.interpolate(
        y[:4].unsqueeze(0).float(), size=target, mode="nearest",
    ).squeeze(0).to(dtype=y.dtype)
    latent = F.interpolate(
        y[4:].unsqueeze(0).float(), size=target, mode="trilinear",
        align_corners=False,
    ).squeeze(0).to(dtype=y.dtype)
    return torch.cat([mask, latent], dim=0)


def _i2v_seq_len_for_ctx(ctx: WanRFSDSContext) -> int:
    return int(ctx.f_latent) * int(ctx.h_latent) * int(ctx.w_latent) // 4


def wan_vae_encode_grad(
    rgb_3FHW_float01: torch.Tensor,
    ctx: WanRFSDSContext,
) -> torch.Tensor:
    """Encode the rendered 21-frame video through Wan VAE WITH gradient.

    Wan VAE expects List[Tensor [3, T, H, W] in [-1, 1]] and returns
    List[Tensor [16, T_lat, H_lat, W_lat]] (one element per video).
    We strip the list and add the batch axis for downstream DiT input
    convenience.

    Parameters
    ----------
    rgb_3FHW_float01 : Tensor [3, F, H, W] in [0, 1]
        Direct render output from ``render_21_with_warp`` (permuted from
        ``[F, 3, H, W]`` to ``[3, F, H, W]``).
    ctx : WanRFSDSContext

    Returns
    -------
    z : Tensor [1, 16, F_lat, H_lat, W_lat]
        Grad-enabled VAE latent. ``z.requires_grad`` is True iff the input
        rgb has ``requires_grad`` (true for our differentiable renderer).
    """
    if ctx.backend == "fun_inp":
        return _fun_inp_vae_encode_grad(rgb_3FHW_float01, ctx)

    expected_prefix = (3, int(ctx.frame_num))
    if rgb_3FHW_float01.shape[:2] != expected_prefix:
        raise ValueError(
            f"rgb_3FHW must start with {expected_prefix}, "
            f"got {tuple(rgb_3FHW_float01.shape)}"
        )
    rgb_sds = _resize_rgb_for_wan_sds(rgb_3FHW_float01, ctx.resolution_hw)
    rgb_neg11 = _to_wan_vae_input(rgb_sds).to(ctx.device)

    def _encode_one(video_3FHW: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(dtype=ctx.wan_vae.dtype):
            return (
                ctx.wan_vae.model.encode(
                    video_3FHW.unsqueeze(0), ctx.wan_vae.scale,
                )
                .float()
                .squeeze(0)
            )

    from torch.utils.checkpoint import checkpoint
    z = checkpoint(_encode_one, rgb_neg11, use_reentrant=False)
    # Sanity-check the latent shape.
    expected = (
        WAN_LATENT_CH,
        int(ctx.f_latent),
        int(ctx.h_latent),
        int(ctx.w_latent),
    )
    if tuple(z.shape) != expected:
        raise RuntimeError(
            f"Wan VAE latent shape mismatch: expected {expected}, "
            f"got {tuple(z.shape)}"
        )
    return z.unsqueeze(0)                              # [1, 16, F_lat, H_lat, W_lat]


def _fun_inp_vae_encode_grad(
    rgb_3FHW_float01: torch.Tensor,
    ctx: WanRFSDSContext,
) -> torch.Tensor:
    expected_rgb = (3, int(ctx.frame_num), int(ctx.resolution_hw[0]), int(ctx.resolution_hw[1]))
    if rgb_3FHW_float01.shape != expected_rgb:
        raise ValueError(
            f"rgb_3FHW must be {expected_rgb}, "
            f"got {tuple(rgb_3FHW_float01.shape)}"
        )
    video = _to_wan_vae_input(rgb_3FHW_float01).unsqueeze(0).to(
        device=ctx.device, dtype=ctx.fun_pipeline.vae.dtype
    )
    def _encode(v: torch.Tensor) -> torch.Tensor:
        return ctx.fun_pipeline.vae.encode(v)[0].mode()

    from torch.utils.checkpoint import checkpoint
    z = checkpoint(_encode, video, use_reentrant=False)
    expected = (
        1,
        WAN_LATENT_CH,
        int(ctx.f_latent),
        int(ctx.h_latent),
        int(ctx.w_latent),
    )
    if tuple(z.shape) != expected:
        raise RuntimeError(
            f"Fun-InP VAE latent shape mismatch: expected {expected}, got {tuple(z.shape)}"
        )
    return z


def _build_fun_inp_y_guidance(
    video: torch.Tensor,
    mask_video: torch.Tensor,
    ctx: WanRFSDSContext,
) -> Tuple[torch.Tensor, int]:
    """Build the Fun-InP y_guidance (mask+masked-video latents) and seq_len for the given
    keyframe video. video [1,3,F,H,W], mask_video [1,1,F,H,W]. Returns (y on ctx.teacher_device
    or ctx.device, seq_len). No prompt encode, no T5 move. Camera-dependent part of the cond."""
    pipe = ctx.fun_pipeline
    H, W = int(ctx.resolution_hw[0]), int(ctx.resolution_hw[1])
    F_count = int(ctx.frame_num)
    video = video.to(device=ctx.device, dtype=torch.float32)
    mask_video = mask_video.to(device=ctx.device, dtype=torch.float32)

    flat_video = video.permute(0, 2, 1, 3, 4).reshape(F_count, 3, H, W)
    init_video = pipe.image_processor.preprocess(flat_video, height=H, width=W)
    init_video = init_video.to(device=ctx.device, dtype=torch.float32)
    init_video = init_video.reshape(1, F_count, 3, H, W).permute(0, 2, 1, 3, 4)

    flat_mask = mask_video.permute(0, 2, 1, 3, 4).reshape(F_count, 1, H, W)
    mask_condition = pipe.mask_processor.preprocess(flat_mask, height=H, width=W)
    mask_condition = mask_condition.to(device=ctx.device, dtype=torch.float32)
    mask_condition = mask_condition.reshape(1, F_count, 1, H, W).permute(0, 2, 1, 3, 4)

    masked_video = init_video * (torch.tile(mask_condition, [1, 3, 1, 1, 1]) < 0.5)
    _, masked_video_latents = pipe.prepare_mask_latents(
        None,
        masked_video,
        1,
        H,
        W,
        ctx.dtype,
        ctx.device,
        None,
        True,
        noise_aug_strength=None,
    )
    mask_condition = torch.concat(
        [
            torch.repeat_interleave(mask_condition[:, :, 0:1], repeats=4, dim=2),
            mask_condition[:, :, 1:],
        ],
        dim=2,
    )
    mask_condition = mask_condition.view(1, mask_condition.shape[2] // 4, 4, H, W)
    mask_condition = mask_condition.transpose(1, 2)
    mask_latents = _resize_mask_like_official(1 - mask_condition, masked_video_latents, True)
    mask_latents = mask_latents.to(device=ctx.device, dtype=ctx.dtype)
    masked_video_latents = masked_video_latents.to(device=ctx.device, dtype=ctx.dtype)

    y = torch.cat(
        [torch.cat([mask_latents, mask_latents], dim=0),
         torch.cat([masked_video_latents, masked_video_latents], dim=0)],
        dim=1,
    ).to(device=ctx.teacher_device or ctx.device, dtype=ctx.dtype)

    patch_size = tuple(int(v) for v in pipe.transformer.config.patch_size)
    seq_len = int(math.ceil((ctx.h_latent * ctx.w_latent) / (patch_size[1] * patch_size[2]) * ctx.f_latent))
    return y, int(seq_len)


def rebuild_fun_inp_y_for_keyframes(
    base_wan_cond: Dict[str, Any],
    ref_first_3HW: torch.Tensor,
    ref_last_3HW: torch.Tensor,
    ctx: WanRFSDSContext,
) -> Dict[str, Any]:
    """Return a SHALLOW COPY of an already-prepared canonical wan_cond with a FRESH y_guidance
    for new first/last keyframe images, reusing the cached in_context + seq_len
    (camera-independent). Used by the multi-view path to re-condition Wan-Fun-InP on
    per-camera reference frames without re-encoding the prompt or moving T5."""
    if base_wan_cond.get("_prepared_backend") != "fun_inp":
        raise ValueError(
            "rebuild_fun_inp_y_for_keyframes requires an already-prepared "
            "base_wan_cond with _prepared_backend == 'fun_inp'"
        )
    base_video = base_wan_cond["fun_video"]
    video = base_video.clone()
    ref_first = ref_first_3HW.detach().to(device=base_video.device, dtype=base_video.dtype)
    ref_last = ref_last_3HW.detach().to(device=base_video.device, dtype=base_video.dtype)
    video[:, :, 0] = ref_first
    video[:, :, -1] = ref_last
    mask_video = base_wan_cond["fun_mask"]

    y, _seq = _build_fun_inp_y_guidance(video, mask_video, ctx)

    out = dict(base_wan_cond)
    out["y_guidance"] = y
    return out


def _prepare_fun_inp_rfsds_condition(
    wan_cond: Dict[str, Any],
    ctx: WanRFSDSContext,
) -> Dict[str, Any]:
    if wan_cond.get("_prepared_backend") == "fun_inp":
        return wan_cond
    if wan_cond.get("backend") != "fun_inp":
        raise ValueError("Fun-InP W-RFSDS requires wan_cond['backend'] == 'fun_inp'")
    pipe = ctx.fun_pipeline
    H, W = int(ctx.resolution_hw[0]), int(ctx.resolution_hw[1])
    F_count = int(ctx.frame_num)
    video = wan_cond["fun_video"].to(device=ctx.device, dtype=torch.float32)
    mask_video = wan_cond["fun_mask"].to(device=ctx.device, dtype=torch.float32)
    if tuple(video.shape) != (1, 3, F_count, H, W):
        raise ValueError(f"fun_video shape mismatch: got {tuple(video.shape)}")
    if tuple(mask_video.shape) != (1, 1, F_count, H, W):
        raise ValueError(f"fun_mask shape mismatch: got {tuple(mask_video.shape)}")

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        wan_cond["pos_prompt"],
        wan_cond["neg_prompt"],
        True,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=ctx.teacher_device or ctx.device,
    )
    if isinstance(prompt_embeds, torch.Tensor):
        in_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    else:
        in_prompt_embeds = negative_prompt_embeds + prompt_embeds

    y, seq_len = _build_fun_inp_y_guidance(video, mask_video, ctx)

    wan_cond["in_context"] = in_prompt_embeds
    wan_cond["y_guidance"] = y
    wan_cond["seq_len"] = int(seq_len)
    wan_cond["_prepared_backend"] = "fun_inp"

    # T5 is only needed for this one-time prompt encode (embeds cached in
    # wan_cond["in_context"] and reused every iter), so move it off the GPU to
    # free ~11 GB. Single-GPU: that headroom goes to the grad VAE encode.
    # Dual-GPU: it frees the teacher GPU, which holds both ~27 GB experts and is
    # tighter than it looks once the DiT forward activations are added. The
    # prompt is fixed, so this is a one-time move with no recurring transfer.
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    return wan_cond


def _resize_mask_like_official(
    mask: torch.Tensor,
    latent: torch.Tensor,
    process_first_frame_only: bool = True,
) -> torch.Tensor:
    latent_size = latent.size()
    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :], size=target_size, mode="trilinear", align_corners=False
        )
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :], size=target_size, mode="trilinear", align_corners=False
            )
            return torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        return first_frame_resized
    return F.interpolate(mask, size=list(latent_size[2:]), mode="trilinear", align_corners=False)


# =============================================================================
# W-RFSDS loss
# =============================================================================

def _select_expert(
    tau: float,
    ctx: WanRFSDSContext,
) -> Any:
    """Return high_noise_model when ``tau * (num_train_timesteps - 1) >= boundary``.

    The comparison matches ``wan/image2video.py:341, 388-391`` which uses
    ``t.item() >= boundary`` with ``boundary = self.boundary *
    num_train_timesteps``.

    ★ Fix SD-1: scaling must match the t_wan tensor passed to the model
    forward (line ~367 below uses ``tau * (num_train_timesteps - 1)``).
    Without this fix, at tau=0.9 the selector saw ``0.9 * 1000 = 900 >= 900``
    (high expert) but the model received ``0.9 * 999 = 899.1`` which in Wan's
    scheduler corresponds to the low-expert regime — i.e. high_noise weights
    were applied to an input the low_noise expert should have seen. The
    practical effect was small (1 timestep boundary jitter) but spec demands
    one consistent scaling.
    """
    t_in_train_units = float(tau) * (float(ctx.num_train_timesteps) - 1.0)
    if t_in_train_units >= ctx.boundary_in_t_units:
        if ctx.high_noise_model is None:
            return ctx.low_noise_model
        return ctx.high_noise_model
    return ctx.low_noise_model


def w_rfsds_loss(
    rgb_3FHW_float01: torch.Tensor,
    wan_cond: Dict[str, Any],
    ctx: WanRFSDSContext,
    tau: float,
    cfg_scale: float,
    seed: Optional[int] = None,
    return_z_theta: bool = False,
) -> "torch.Tensor | Tuple[torch.Tensor, torch.Tensor]":
    """Compute the W-RFSDS surrogate loss (CHORD Eq. 3) with Wan-shifted sigma.

    The loss is constructed so its gradient w.r.t. ``rgb_3FHW_float01``
    equals (in expectation over eps and tau) the W-RFSDS update direction.
    The two Wan DiT forwards run under ``torch.no_grad`` because the
    residual is detached before being multiplied by ``z``; only the VAE
    encode needs backward.

    ★ C2 fix: Wan's scheduler maps the integer timestep to a SHIFTED sigma
    (see ``_shifted_sigma``). The W-RFSDS forward ``z_sigma = (1-sigma)*z +
    sigma*eps`` must use the shifted sigma so the noisy latent is in the
    distribution Wan was trained on. The selector below uses the UN-shifted
    ``tau`` to pick the high/low-noise expert (matching Wan's boundary check
    on the raw timestep), and the shifted sigma for the noise mixture.

    Parameters
    ----------
    rgb_3FHW_float01 : Tensor [3, F, H, W] in [0, 1]
        Differentiable render output (caller should already have clamped to
        [0, 1] to keep VAE input in-distribution).
    wan_cond : dict   from Bootstrap B11
    ctx : WanRFSDSContext  (includes ``sample_shift``)
    tau : float in (0, 1)  raw quantile from inverse-CDF
    cfg_scale : float       CFG (CHORD: linear 25 -> 12 over training)
    seed : int or None      deterministic eps for type vote
    return_z_theta : bool
        If True, returns ``(loss, z_theta)`` so the caller can reuse
        ``z_theta`` for ``latent_recon_loss`` without a second VAE encode
        (S5 fix; saves the most expensive backward pass per iter).

    Returns
    -------
    loss : Tensor scalar  (or (loss, z_theta) if return_z_theta=True)
    """
    if ctx.backend == "fun_inp":
        return _w_rfsds_loss_fun_inp(
            rgb_3FHW_float01,
            wan_cond,
            ctx,
            tau=tau,
            cfg_scale=cfg_scale,
            seed=seed,
            return_z_theta=return_z_theta,
        )

    # ---- 1) Encode rendered video to VAE latent (grad-enabled) ----
    # For the default SDS path, compute the frozen-VAE VJP against a detached
    # RGB leaf, then replay that pixel gradient through the renderer with a
    # surrogate loss. This is first-order equivalent to backpropagating the
    # SDS surrogate through VAE and renderer together, but avoids keeping the
    # Wan VAE graph alive until the main optimizer backward.
    rgb_for_vae = rgb_3FHW_float01 if return_z_theta else (
        rgb_3FHW_float01.detach().requires_grad_(True)
    )
    z_theta = wan_vae_encode_grad(rgb_for_vae, ctx)
    # z_theta: [1, 16, F_lat, H_lat, W_lat]

    # ---- 2) Forward sample z_sigma = (1-sigma)*z + sigma*eps  (★ C2) ----
    # CHORD section 3.2 + Wan scheduler shift: sigma_shifted is what Wan saw
    # at training time when its UniPC scheduler used the given timestep.
    sigma = _shifted_sigma(float(tau), ctx.sample_shift)
    with torch.no_grad():
        if seed is not None:
            gen = torch.Generator(device=z_theta.device).manual_seed(int(seed))
            eps = torch.randn(z_theta.shape, generator=gen,
                              device=z_theta.device, dtype=z_theta.dtype)
        else:
            eps = torch.randn_like(z_theta)
        z_sigma = (1.0 - sigma) * z_theta.detach() + sigma * eps

    # ---- 3) Pick expert by raw tau; build t_wan in [0, 1000) ----
    # The expert switch uses the integer timestep, NOT the shifted sigma.
    # Match wan/image2video.py:341,388-391 which compares timestep against
    # boundary = boundary_normalized * num_train_timesteps.
    wan_model = _select_expert(tau, ctx)
    if next(wan_model.parameters()).device != ctx.device:
        logger.info("[stage_d][wan] moving selected DiT to %s", ctx.device)
        wan_model.to(ctx.device)
        if ctx.device.type == "cuda":
            torch.cuda.empty_cache()
    t_wan = torch.tensor(
        [float(tau) * (float(ctx.num_train_timesteps) - 1.0)],
        device=ctx.device, dtype=torch.float32,
    )

    # ---- 4) Two DiT forwards (cond + uncond), both under no_grad ----
    x_input = [z_sigma.squeeze(0).to(ctx.dtype)]       # List of [16, F_lat, H_lat, W_lat]
    y_input = [
        _resize_i2v_y_for_wan_sds(y, ctx).to(device=ctx.device, dtype=ctx.dtype)
        if isinstance(y, torch.Tensor) else y
        for y in wan_cond["y"]
    ]
    context = [
        c.to(device=ctx.device, dtype=ctx.dtype)
        if isinstance(c, torch.Tensor) else c
        for c in wan_cond["context"]
    ]
    context_null = [
        c.to(device=ctx.device, dtype=ctx.dtype)
        if isinstance(c, torch.Tensor) else c
        for c in wan_cond["context_null"]
    ]
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=ctx.dtype):
        seq_len = _i2v_seq_len_for_ctx(ctx)
        v_pred_cond = wan_model(
            x_input, t=t_wan,
            context=context,
            seq_len=seq_len,
            y=y_input,
        )[0].to(z_theta.dtype).unsqueeze(0)            # [1, 16, F_lat, H_lat, W_lat]

        v_pred_uncond = wan_model(
            x_input, t=t_wan,
            context=context_null,
            seq_len=seq_len,
            y=y_input,
        )[0].to(z_theta.dtype).unsqueeze(0)

        v_pred = v_pred_uncond + float(cfg_scale) * (v_pred_cond - v_pred_uncond)
        # CHORD Eq.3 residual; detached, treated as fixed target direction.
        residual = (v_pred - eps + z_theta.detach()).detach()

    if ctx.offload_dit:
        logger.info("[stage_d][wan] moving selected DiT to CPU before VAE backward")
        wan_model.to("cpu")
        if ctx.device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- 5) Form surrogate loss whose gradient w.r.t. theta equals SDS ----
    loss_z = (residual * z_theta).sum() / float(z_theta.numel())
    if return_z_theta:
        return loss_z, z_theta
    del x_input, y_input, context, context_null, z_sigma, eps
    del v_pred_cond, v_pred_uncond, v_pred
    if ctx.device.type == "cuda":
        torch.cuda.empty_cache()
    loss_value = loss_z.detach()
    grad_rgb = torch.autograd.grad(
        loss_z, rgb_for_vae, retain_graph=False, create_graph=False,
    )[0]
    surrogate = (grad_rgb.detach() * rgb_3FHW_float01).sum()
    loss = surrogate + (loss_value - surrogate.detach())
    return loss


def _select_fun_inp_transformer(tau: float, ctx: WanRFSDSContext) -> Any:
    t_in_train_units = float(tau) * (float(ctx.num_train_timesteps) - 1.0)
    if ctx.high_noise_model is not None and t_in_train_units >= ctx.boundary_in_t_units:
        return ctx.high_noise_model
    return ctx.low_noise_model


def _w_rfsds_loss_fun_inp(
    rgb_3FHW_float01: torch.Tensor,
    wan_cond: Dict[str, Any],
    ctx: WanRFSDSContext,
    tau: float,
    cfg_scale: float,
    seed: Optional[int] = None,
    return_z_theta: bool = False,
) -> "torch.Tensor | Tuple[torch.Tensor, torch.Tensor]":
    wan_cond = _prepare_fun_inp_rfsds_condition(wan_cond, ctx)
    z_theta = _fun_inp_vae_encode_grad(rgb_3FHW_float01, ctx)

    sigma = _shifted_sigma(float(tau), ctx.sample_shift)
    with torch.no_grad():
        if seed is not None:
            gen = torch.Generator(device=z_theta.device).manual_seed(int(seed))
            eps = torch.randn(z_theta.shape, generator=gen,
                              device=z_theta.device, dtype=z_theta.dtype)
        else:
            eps = torch.randn_like(z_theta)
        z_sigma = (1.0 - sigma) * z_theta.detach() + sigma * eps

    t_wan = torch.tensor(
        [float(tau) * (float(ctx.num_train_timesteps) - 1.0)],
        device=ctx.device,
        dtype=torch.float32,
    )
    latent_model_input = torch.cat([z_sigma, z_sigma], dim=0)
    if hasattr(ctx.fun_pipeline.scheduler, "scale_model_input"):
        latent_model_input = ctx.fun_pipeline.scheduler.scale_model_input(latent_model_input, t_wan)

    transformer = _select_fun_inp_transformer(float(tau), ctx)
    # Dual-GPU: the experts live on ctx.teacher_device. Move the (tiny, already
    # detached) latent + timestep there for the no_grad teacher forward, then
    # bring the score back to ctx.device. Single-GPU: teacher_device == device,
    # so every .to() is a no-op. in_context / y_guidance were cached on the
    # teacher device by _prepare_fun_inp_rfsds_condition.
    tdev = ctx.teacher_device or ctx.device
    timestep = t_wan.expand(latent_model_input.shape[0]).to(tdev)
    # The Wan DiT is bf16 and its time_embedding feeds the fp32 timestep
    # sinusoid into a bf16 Linear; the model is designed to run under autocast
    # (matching the in-house path's ``with torch.no_grad(), autocast(...)``).
    # Without it: "mat1 and mat2 must have the same dtype, Float vs BFloat16".
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=ctx.dtype):
        noise_pred = transformer(
            x=latent_model_input.to(device=tdev, dtype=ctx.dtype),
            context=wan_cond["in_context"],
            t=timestep,
            seq_len=int(wan_cond["seq_len"]),
            y=wan_cond["y_guidance"],
        ).to(ctx.device)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        v_pred = noise_pred_uncond + float(cfg_scale) * (noise_pred_text - noise_pred_uncond)
        residual = (v_pred.to(z_theta.dtype) - eps + z_theta.detach()).detach()

    loss = (residual * z_theta).sum() / float(z_theta.numel())
    if return_z_theta:
        return loss, z_theta
    return loss


# =============================================================================
# Latent-reconstruction auxiliary loss (NOT in CHORD; kept for optional use)
# =============================================================================

def latent_recon_loss(
    rgb_3FHW_float01: torch.Tensor,
    z_wan_target: torch.Tensor,
    ctx: WanRFSDSContext,
    z_render_cached: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L2 in Wan VAE latent space against Bootstrap's ``z_wan_target``.

    NOT in CHORD. method.md keeps this as an optional auxiliary loss
    (disabled by default in P1, enabled in P2). If enabled at large weight
    in P1, can bias optimization toward over-fitting to Wan video artifacts.

    ★ S5 fix: caller can pass ``z_render_cached`` from a prior
    ``w_rfsds_loss(..., return_z_theta=True)`` call to skip the (expensive)
    duplicate Wan VAE encode + backward. Halves the iter's peak VAE memory
    when both losses are active.

    Parameters
    ----------
    rgb_3FHW_float01 : Tensor [3, F, H, W] in [0, 1]
    z_wan_target : Tensor [16, F_lat, H_lat, W_lat]
    ctx : WanRFSDSContext
    z_render_cached : Optional[Tensor [1, 16, F_lat, H_lat, W_lat]]
        Reuse of z from a prior w_rfsds_loss call this iter.

    Returns
    -------
    loss : Tensor scalar
    """
    if z_render_cached is not None:
        if z_render_cached.shape[0] != 1:
            raise ValueError(
                f"z_render_cached must have batch=1; got {tuple(z_render_cached.shape)}"
            )
        z_render = z_render_cached.squeeze(0)
    else:
        z_render = wan_vae_encode_grad(rgb_3FHW_float01, ctx).squeeze(0)
    if z_render.shape != z_wan_target.shape:
        raise ValueError(
            f"shape mismatch: z_render={tuple(z_render.shape)}, "
            f"z_wan_target={tuple(z_wan_target.shape)}"
        )
    return ((z_render - z_wan_target.detach()) ** 2).mean()


__all__ = [
    "WanRFSDSContext", "load_wan_for_rfsds", "load_wan_fun_inp_for_rfsds",
    "wan_vae_encode_grad", "w_rfsds_loss", "latent_recon_loss",
]

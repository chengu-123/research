"""Stage A backend: Wan2.2-Fun-A14B-InP start/end-frame video generation.

This module keeps the same downstream artifact contract as ``stage_a_wan``:
``run_stage_a_fun_inp`` returns a ``StageAResult`` whose video tensor is
``[3, F, H, W]`` uint8. VideoX-Fun is imported only inside the heavy loader so
normal I2V usage and unit tests do not require the package to be installed.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from pipelines.stage_a_wan import (
    StageAResult,
    _DEFAULT_WAN_SIZE_LABEL,
    _SUPPORTED_SIZE_LABELS,
    _WAN_MAX_AREA_CONFIGS,
    _coerce_input_image,
    _predict_wan_output_hw,
)
from pipelines.utils.seeding import seed_everything
from pipelines.utils.visualization_a import save_all_stage_a_visualisations
from pipelines.wan_helpers import build_articulated_prompts


SUPPORTED_FUN_INP_SIZE_LABELS = tuple(sorted(_SUPPORTED_SIZE_LABELS))
_FUN_INP_BACKEND = "wan2_2_fun_a14b_inp"
_FUN_INP_VAE_STRIDE = (4, 8, 8)


def _resize_rgb(image: Image.Image, sample_hw: Tuple[int, int]) -> Image.Image:
    H, W = int(sample_hw[0]), int(sample_hw[1])
    return image.resize((W, H), Image.BICUBIC)


def _validate_start_end_aspect(
    start_image: Image.Image,
    end_image: Image.Image,
    max_rel_err: float = 0.02,
) -> None:
    start_aspect = float(start_image.height) / float(start_image.width)
    end_aspect = float(end_image.height) / float(end_image.width)
    rel_err = abs(start_aspect - end_aspect) / max(start_aspect, 1.0e-8)
    if rel_err > float(max_rel_err):
        raise ValueError(
            f"start/end image aspect mismatch: start={start_image.size}, "
            f"end={end_image.size}, rel_err={rel_err:.4f} > {max_rel_err:.4f}"
        )


def _prepare_fun_inp_condition(
    start_image: Union[Image.Image, np.ndarray, torch.Tensor],
    end_image: Union[Image.Image, np.ndarray, torch.Tensor],
    frame_num: int,
    sample_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, Image.Image]:
    """Build VideoX-Fun InP ``video`` and ``mask_video`` tensors.

    Mask polarity follows upstream ``get_image_to_video_latent``:
    known frames are 0 and unknown frames are 255. For our start/end-frame
    condition, only frame 0 and frame F-1 are known.
    """
    if frame_num <= 1 or (frame_num - 1) % int(_FUN_INP_VAE_STRIDE[0]) != 0:
        raise ValueError(
            f"frame_num must be of the form 4n+1 for Wan VAE stride 4; got {frame_num}"
        )
    H, W = int(sample_hw[0]), int(sample_hw[1])
    if H <= 0 or W <= 0 or H % 8 != 0 or W % 8 != 0:
        raise ValueError(f"sample_hw must be positive and divisible by 8; got {sample_hw}")

    start = _coerce_input_image(start_image)
    end = _coerce_input_image(end_image)
    _validate_start_end_aspect(start, end)

    start_r = _resize_rgb(start, (H, W))
    end_r = _resize_rgb(end, (H, W))
    start_t = torch.from_numpy(np.array(start_r, dtype=np.uint8, copy=True)).permute(2, 0, 1)
    end_t = torch.from_numpy(np.array(end_r, dtype=np.uint8, copy=True)).permute(2, 0, 1)

    video = torch.zeros((1, 3, int(frame_num), H, W), dtype=torch.float32)
    video[:, :, 0] = start_t.float() / 255.0
    video[:, :, -1] = end_t.float() / 255.0

    mask = torch.full((1, 1, int(frame_num), H, W), 255.0, dtype=torch.float32)
    mask[:, :, 0] = 0.0
    mask[:, :, -1] = 0.0
    return video, mask, start_r


def _fun_inp_video_to_float01_uint8(
    video: torch.Tensor,
    expected_F: int,
    expected_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert VideoX-Fun pipeline output to ``[3, F, H, W]`` float/uint8."""
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"expected torch.Tensor from VideoX-Fun, got {type(video)!r}")

    v = video.detach().cpu()
    if v.ndim == 5:
        if v.shape[0] != 1:
            raise ValueError(f"expected batch size 1; got shape={tuple(v.shape)}")
        if v.shape[1] == 3:
            v = v[0]
        elif v.shape[2] == 3:
            v = v[0].permute(1, 0, 2, 3)
        else:
            raise ValueError(f"cannot identify channel dimension in shape={tuple(video.shape)}")
    elif v.ndim == 4 and v.shape[0] == 3:
        pass
    else:
        raise ValueError(
            f"VideoX-Fun output must be [1,3,F,H,W], [1,F,3,H,W], or [3,F,H,W]; "
            f"got {tuple(video.shape)}"
        )

    expected = (3, int(expected_F), int(expected_hw[0]), int(expected_hw[1]))
    if tuple(v.shape) != expected:
        raise ValueError(f"Fun-InP output shape mismatch: expected {expected}, got {tuple(v.shape)}")

    v = v.to(dtype=torch.float32)
    if float(v.amin()) < -1.0e-4:
        v = (v.clamp(-1.0, 1.0) + 1.0) * 0.5
    else:
        v = v.clamp(0.0, 1.0)
    v_uint8 = (v * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)
    return v, v_uint8


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
            f"VideoX-Fun config not found: {path}. Set --fun_config or VIDEOX_FUN_ROOT."
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


def _filter_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    import inspect

    allowed = set(inspect.signature(cls.__init__).parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _load_fun_inp_pipeline(
    model_dir: str,
    fun_config_path: str,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> Tuple[Any, float]:
    _ensure_videox_fun_on_sys_path(fun_config_path)
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

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Fun-InP model_dir not found: {model_dir}")

    config = OmegaConf.load(fun_config_path)
    transformer_kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"])
    low_subpath = transformer_kwargs.get("transformer_low_noise_model_subpath", "transformer")
    high_subpath = transformer_kwargs.get("transformer_high_noise_model_subpath", "transformer")

    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(model_dir, low_subpath),
        transformer_additional_kwargs=transformer_kwargs,
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    if transformer_kwargs.get("transformer_combination_type", "single") == "moe":
        transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(model_dir, high_subpath),
            transformer_additional_kwargs=transformer_kwargs,
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
    else:
        transformer_2 = None

    vae_classes = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }
    vae_kwargs = OmegaConf.to_container(config["vae_kwargs"])
    vae_cls = vae_classes[vae_kwargs.get("vae_type", "AutoencoderKLWan")]
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
    scheduler_classes = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }
    scheduler_cls = scheduler_classes[scheduler_name]
    scheduler_kwargs = OmegaConf.to_container(config["scheduler_kwargs"])
    scheduler = scheduler_cls(**_filter_kwargs(scheduler_cls, scheduler_kwargs))

    pipe = Wan2_2FunInpaintPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )

    pipe.to(device=device)

    boundary = float(transformer_kwargs.get("boundary", 0.875))
    return pipe, boundary


def run_stage_a_fun_inp(
    start_image: Union[Image.Image, np.ndarray, torch.Tensor],
    end_image: Union[Image.Image, np.ndarray, torch.Tensor],
    user_motion_prompt: str,
    model_dir: str,
    out_dir: str,
    fun_config_path: Optional[str] = None,
    seed: int = 42,
    frame_num: int = 21,
    wan_size_label: str = _DEFAULT_WAN_SIZE_LABEL,
    sampling_steps: int = 50,
    guide_scale: float = 6.0,
    sample_shift: float = 5.0,
    lang: str = "zh",
    fps: int = 16,
    device_id: int = 0,
    weight_dtype: torch.dtype = torch.bfloat16,
) -> StageAResult:
    """Generate a Stage A video from start/end images using Fun-InP."""
    if not torch.cuda.is_available():
        raise RuntimeError("Stage A Fun-InP requires CUDA.")
    if frame_num <= 1 or (frame_num - 1) % int(_FUN_INP_VAE_STRIDE[0]) != 0:
        raise ValueError(
            f"frame_num must be of the form 4n+1 with n>=1; got {frame_num}"
        )
    wan_size_label = str(wan_size_label)
    if wan_size_label not in _SUPPORTED_SIZE_LABELS:
        raise ValueError(
            f"wan_size_label={wan_size_label!r} is not supported; "
            f"supported labels: {sorted(_SUPPORTED_SIZE_LABELS)}"
        )

    os.makedirs(out_dir, exist_ok=True)
    model_dir = os.path.abspath(os.fspath(model_dir))
    fun_config_path = _resolve_fun_config_path(fun_config_path)
    device = torch.device(f"cuda:{int(device_id)}")
    seed_everything(int(seed))

    start = _coerce_input_image(start_image)
    end = _coerce_input_image(end_image)
    _validate_start_end_aspect(start, end)
    input_hw = (int(start.height), int(start.width))
    expected_hw = _predict_wan_output_hw(input_hw, int(_WAN_MAX_AREA_CONFIGS[wan_size_label]))
    H, W = int(expected_hw[0]), int(expected_hw[1])

    start.save(os.path.join(out_dir, "input_s_0_with_carpet.png"))
    end.save(os.path.join(out_dir, "input_s_5_with_carpet.png"))

    pos_prompt, neg_prompt = build_articulated_prompts(user_motion_prompt, lang=lang)
    cond_video, cond_mask, _clip_image = _prepare_fun_inp_condition(
        start_image=start,
        end_image=end,
        frame_num=int(frame_num),
        sample_hw=(H, W),
    )

    pipe, boundary = _load_fun_inp_pipeline(
        model_dir=model_dir,
        fun_config_path=fun_config_path,
        device=device,
        weight_dtype=weight_dtype,
    )

    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.no_grad():
        sample = pipe(
            pos_prompt,
            num_frames=int(frame_num),
            negative_prompt=neg_prompt,
            height=H,
            width=W,
            generator=generator,
            guidance_scale=float(guide_scale),
            num_inference_steps=int(sampling_steps),
            boundary=float(boundary),
            video=cond_video,
            mask_video=cond_mask,
            shift=float(sample_shift),
        ).videos

    video_float01, video_uint8 = _fun_inp_video_to_float01_uint8(
        sample,
        expected_F=int(frame_num),
        expected_hw=(H, W),
    )

    artifact_paths = save_all_stage_a_visualisations(
        video_3fhw_float01=video_float01.numpy(),
        out_dir=out_dir,
        pos_prompt=pos_prompt,
        neg_prompt=neg_prompt,
        user_motion_prompt=user_motion_prompt,
        lang=lang,
        seed=int(seed),
        frame_num=int(frame_num),
        resolution_hw=(H, W),
        sampling_steps=int(sampling_steps),
        guide_scale=float(guide_scale),
        sample_shift=float(sample_shift),
        sample_solver="flow",
        wan_ckpt_dir=str(model_dir),
        fps=int(fps),
        extra={
            "wan_backend": _FUN_INP_BACKEND,
            "fun_config_path": str(fun_config_path),
            "fun_boundary": float(boundary),
            "wan_size_label": wan_size_label,
            "wan_max_area": int(_WAN_MAX_AREA_CONFIGS[wan_size_label]),
            "input_hw": [int(input_hw[0]), int(input_hw[1])],
            "aspect_preserved": True,
            "condition_mask": "first_last_known",
        },
    )

    target_uint8_path = os.path.join(out_dir, "wan_video_target_3FHW_uint8.pt")
    torch.save(video_uint8, target_uint8_path)
    artifact_paths.append(target_uint8_path)

    return StageAResult(
        wan_video_target_3FHW=video_uint8,
        wan_video_float01_3FHW=video_float01,
        pos_prompt=pos_prompt,
        neg_prompt=neg_prompt,
        user_motion_prompt=user_motion_prompt,
        lang=lang,
        seed=int(seed),
        frame_num=int(frame_num),
        resolution_hw=(H, W),
        sampling_steps=int(sampling_steps),
        guide_scale=float(guide_scale),
        sample_shift=float(sample_shift),
        sample_solver="flow",
        wan_ckpt_dir=str(model_dir),
        out_dir=str(out_dir),
        artifact_paths=artifact_paths,
    )


__all__ = [
    "SUPPORTED_FUN_INP_SIZE_LABELS",
    "_prepare_fun_inp_condition",
    "_fun_inp_video_to_float01_uint8",
    "run_stage_a_fun_inp",
]

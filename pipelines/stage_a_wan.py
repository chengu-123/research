"""Stage A: Wan2.2 I2V articulation video generation.

Single-shot, fixed-seed inference. Given a user input image with carpet
(grounding disk) baked in and a per-object motion prompt, generate a 21-frame
RGB video at 16 fps with a locked-off camera while the articulated part moves.
The output ``wan_video_target_3FHW`` becomes the universal target for Stage B
bootstrap (TRELLIS K=6 conditioning, Wan VAE latent target).

Hard constraints:
  - frame_num = 21              (4*5 + 1; gives F_lat = 6 for K=6 states)
  - size label = 832*480        (official Wan2.2 I2V 480P area profile)
  - input aspect is preserved   (Wan I2V derives actual H/W from input aspect)
  - seed = 42                   (no multi-candidate selection per CHORD A.1)
  - guide_scale = 3.5           (Wan2.2 I2V A14B sample_guide_scale)

Offline-only execution: Wan2.2 weights are loaded from a local checkpoint
directory; HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE are exported before any
HuggingFace-aware library is imported, so a missing model file will fail
loudly rather than silently fetching from the network.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Offline guard. Must run BEFORE any huggingface_hub / transformers import.
# Wan2.2's T5 tokenizer goes through transformers.AutoTokenizer; the DiT goes
# through diffusers.ModelMixin. Both libraries respect these env vars.
# -----------------------------------------------------------------------------
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from dataclasses import dataclass, field
from typing import Tuple, Union

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_WAN_ROOT = os.path.join(_REPO_ROOT, "Wan2.2")
if not os.path.isdir(_WAN_ROOT):
    raise RuntimeError(
        f"Stage A requires the vendored Wan2.2 sources at {_WAN_ROOT!r}; "
        "directory not found."
    )

from pipelines.utils.seeding import seed_everything
from pipelines.utils.visualization_a import save_all_stage_a_visualisations
from pipelines.wan_helpers import build_articulated_prompts


_WAN_TASK = "i2v-A14B"
_DEFAULT_WAN_SIZE_LABEL = "832*480"
_WAN_MAX_AREA_CONFIGS = {
    "720*1280": 720 * 1280,
    "1280*720": 1280 * 720,
    "480*832": 480 * 832,
    "832*480": 832 * 480,
}
_SUPPORTED_SIZE_LABELS = frozenset(_WAN_MAX_AREA_CONFIGS.keys())
SUPPORTED_WAN_SIZE_LABELS = tuple(sorted(_SUPPORTED_SIZE_LABELS))
_WAN_VAE_STRIDE = (4, 8, 8)
_WAN_PATCH_SIZE = (1, 2, 2)


@dataclass
class StageAResult:
    """Stage A return payload.

    The downstream Stage B contract (pipeline_v3 Section 6.1) reads
    ``wan_video_target_3FHW`` as uint8 in [0, 255]; ``.float() / 255.0`` and
    ``* 2 - 1`` push it back to Wan VAE's [-1, 1] input range.
    """
    wan_video_target_3FHW: torch.Tensor      # [3, F, H, W] uint8 [0,255]
    wan_video_float01_3FHW: torch.Tensor     # [3, F, H, W] float32 [0,1]
    pos_prompt: str
    neg_prompt: str
    user_motion_prompt: str
    lang: str
    seed: int
    frame_num: int
    resolution_hw: Tuple[int, int]
    sampling_steps: int
    guide_scale: Union[float, Tuple[float, float]]
    sample_shift: float
    sample_solver: str
    wan_ckpt_dir: str
    out_dir: str
    artifact_paths: list = field(default_factory=list)


def _coerce_input_image(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
    """Coerce the user's input into RGB PIL without changing aspect or size."""
    if isinstance(image, Image.Image):
        pil = image
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[-1] in (3, 4):
            arr = image
        elif image.ndim == 3 and image.shape[0] in (3, 4):
            arr = np.transpose(image, (1, 2, 0))
        else:
            raise ValueError(f"unexpected ndarray shape {image.shape} for input image")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).round().astype(np.uint8)
        pil = Image.fromarray(arr)
    elif isinstance(image, torch.Tensor):
        t = image.detach()
        if t.ndim == 3 and t.shape[0] in (3, 4):
            t = t.permute(1, 2, 0)
        elif not (t.ndim == 3 and t.shape[-1] in (3, 4)):
            raise ValueError(f"unexpected tensor shape {tuple(t.shape)} for input image")
        t = t.float().cpu().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
        pil = Image.fromarray(t)
    else:
        raise TypeError(f"unsupported image type {type(image)!r}")

    if pil.mode == "RGBA":
        pil = pil.convert("RGB")
    elif pil.mode != "RGB":
        pil = pil.convert("RGB")

    return pil


def _predict_wan_output_hw(input_hw: Tuple[int, int], max_area: int) -> Tuple[int, int]:
    """Mirror WanI2V.generate() lines 262-271 for shape contract checks."""
    input_h, input_w = int(input_hw[0]), int(input_hw[1])
    if input_h <= 0 or input_w <= 0:
        raise ValueError(f"input_hw must be positive; got {input_hw}")
    aspect_ratio = float(input_h) / float(input_w)
    lat_h = round(
        np.sqrt(float(max_area) * aspect_ratio)
        // _WAN_VAE_STRIDE[1]
        // _WAN_PATCH_SIZE[1]
        * _WAN_PATCH_SIZE[1]
    )
    lat_w = round(
        np.sqrt(float(max_area) / aspect_ratio)
        // _WAN_VAE_STRIDE[2]
        // _WAN_PATCH_SIZE[2]
        * _WAN_PATCH_SIZE[2]
    )
    return int(lat_h * _WAN_VAE_STRIDE[1]), int(lat_w * _WAN_VAE_STRIDE[2])


def _load_wan_i2v_components():
    """Import Wan only when Stage A actually runs generation."""
    import sys

    if _WAN_ROOT not in sys.path:
        sys.path.insert(0, _WAN_ROOT)
    from wan.configs.wan_i2v_A14B import i2v_A14B as wan_cfg
    from wan.image2video import WanI2V

    return wan_cfg, WanI2V


def _wan_video_to_float01_uint8(
    video_3FHW_neg11: torch.Tensor,
    expected_F: int,
    expected_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Wan2.2 raw output [3, F, H, W] in [-1, 1] to (float01, uint8).

    Wan's save_video uses ``value_range=(-1, 1)`` (utils.py:96), confirming
    the output range. We clamp before rescaling so any minor over/undershoot
    from the VAE decoder does not produce out-of-byte-range values.

    H/W are strict. ``resolution_hw`` is the actual Wan output contract used
    by Bootstrap and Stage D, while Wan's public ``size`` label is only the
    max-area profile for I2V.
    """
    if not isinstance(video_3FHW_neg11, torch.Tensor):
        raise TypeError(f"expected torch.Tensor from Wan, got {type(video_3FHW_neg11)!r}")
    if video_3FHW_neg11.ndim != 4 or video_3FHW_neg11.shape[0] != 3:
        raise ValueError(
            f"Wan output must be [3, F, H, W]; got {tuple(video_3FHW_neg11.shape)}"
        )
    _, F_out, H_out, W_out = video_3FHW_neg11.shape
    if F_out != expected_F:
        raise ValueError(
            f"Wan output frame count mismatch: expected F={expected_F}, "
            f"got F={F_out} (shape={tuple(video_3FHW_neg11.shape)})"
        )
    expected_H, expected_W = int(expected_hw[0]), int(expected_hw[1])
    if (H_out, W_out) != (expected_H, expected_W):
        raise ValueError(
            f"Wan output shape mismatch: expected "
            f"(3, {expected_F}, {expected_H}, {expected_W}), "
            f"got {tuple(video_3FHW_neg11.shape)}"
        )
    if H_out % 16 != 0 or W_out % 16 != 0:
        raise ValueError(
            f"Wan output (H, W) = ({H_out}, {W_out}) not aligned to vae_stride "
            f"(8) * patch_size (2) = 16; Wan internal lat alignment is broken"
        )

    v = video_3FHW_neg11.detach().to(dtype=torch.float32, device="cpu")
    v = v.clamp_(-1.0, 1.0)
    v_float01 = (v + 1.0) * 0.5                          # [-1, 1] -> [0, 1]
    v_uint8 = (v_float01 * 255.0).round().clamp_(0.0, 255.0).to(torch.uint8)
    return v_float01, v_uint8


def run_stage_a(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    user_motion_prompt: str,
    wan_ckpt_dir: str,
    out_dir: str,
    seed: int = 42,
    frame_num: int = 21,
    wan_size_label: str = _DEFAULT_WAN_SIZE_LABEL,
    sampling_steps: int = 50,
    guide_scale: Union[float, Tuple[float, float]] = 3.5,
    sample_shift: float = 5.0,
    sample_solver: str = "unipc",
    lang: str = "zh",
    fps: int = 16,
    offload_model: bool = False,
    convert_model_dtype: bool = True,
    device_id: int = 0,
) -> StageAResult:
    """Stage A end-to-end driver.

    Parameters
    ----------
    image : PIL.Image | np.ndarray | torch.Tensor
        User-provided closed-state image with carpet/grounding disk baked in.
        The aspect ratio is preserved; Wan I2V derives actual H/W from this
        aspect ratio and the official area profile.
    user_motion_prompt : str
        Per-object motion description (e.g. zh: "the drawer slowly slides
        outward in a continuous motion"). The universal camera-lock addon is
        appended automatically by ``build_articulated_prompts``.
    wan_ckpt_dir : str
        Local directory containing the Wan2.2-I2V-A14B weights laid out as
        produced by ``huggingface-cli download --local-dir <dir>``:
            <wan_ckpt_dir>/
                low_noise_model/
                high_noise_model/
                google/umt5-xxl/
                models_t5_umt5-xxl-enc-bf16.pth
                Wan2.1_VAE.pth
        No network access is permitted (HF_HUB_OFFLINE=1 has been exported at
        module load).
    out_dir : str
        Directory to receive the Stage A artifacts.
    seed : int, default 42
        Wan2.2 RNG seed. pipeline_v3 Section 5.3 fixes this; do not vary.
    frame_num : int, default 21
        Must satisfy ``frame_num % 4 == 1``. pipeline_v3 Section 1.3 fixes 21.
    wan_size_label : str, default "832*480"
        Official Wan2.2 I2V area profile. This is a max-area label, not fixed
        output H/W. Wan computes actual H/W from max_area and input aspect.
    sampling_steps : int, default 50
        UniPC steps. pipeline_v3 Section 5.3.
    guide_scale : float | (float, float), default 3.5
        CFG. Wan2.2 I2V A14B config uses sample_guide_scale=(3.5, 3.5).
        CHORD's 25->12 schedule belongs to W-RFSDS (Stage D/F), not here.
    sample_shift : float, default 5.0
        Wan flow scheduler shift. Wan i2v_A14B default.
    sample_solver : str, default "unipc"
        UniPC or DPM++.
    lang : {"zh", "en"}, default "zh"
        Prompt language for the camera-lock addon and negative prompt.
    fps : int, default 16
        MP4 writer frame rate. wan_shared_cfg.sample_fps default.
    offload_model, convert_model_dtype : bool
        VRAM controls forwarded to ``WanI2V``. ``offload_model`` defaults to
        False so the Wan components stay resident on GPU for H800-class runs.
    device_id : int, default 0
        CUDA device index. CPU execution is not supported (Wan VAE relies on
        CUDA kernels).
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Stage A requires CUDA; Wan2.2 VAE decode is not implemented on CPU."
        )
    if frame_num <= 1 or (frame_num - 1) % 4 != 0:
        raise ValueError(
            f"frame_num must be of the form 4n+1 with n>=1; got {frame_num}"
        )
    wan_size_label = str(wan_size_label)
    if wan_size_label not in _SUPPORTED_SIZE_LABELS:
        raise ValueError(
            f"wan_size_label={wan_size_label!r} is not supported for "
            f"Wan2.2 {_WAN_TASK}; supported labels: {sorted(_SUPPORTED_SIZE_LABELS)}"
        )
    if not os.path.isdir(wan_ckpt_dir):
        raise FileNotFoundError(
            f"wan_ckpt_dir not a directory: {wan_ckpt_dir!r}"
        )

    os.makedirs(out_dir, exist_ok=True)

    seed_everything(int(seed))

    pos_prompt, neg_prompt = build_articulated_prompts(user_motion_prompt, lang=lang)

    pil_image = _coerce_input_image(image)
    input_hw = (int(pil_image.height), int(pil_image.width))
    pil_image.save(os.path.join(out_dir, "input_s_0_with_carpet.png"))

    wan_cfg, WanI2V = _load_wan_i2v_components()
    wan_max_area = int(_WAN_MAX_AREA_CONFIGS[wan_size_label])
    expected_hw = _predict_wan_output_hw(input_hw, wan_max_area)
    init_on_cpu = bool(offload_model)

    wan = WanI2V(
        config=wan_cfg,
        checkpoint_dir=wan_ckpt_dir,
        device_id=int(device_id),
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        init_on_cpu=init_on_cpu,
        convert_model_dtype=bool(convert_model_dtype),
    )

    video_3FHW_neg11 = wan.generate(
        input_prompt=pos_prompt,
        img=pil_image,
        max_area=wan_max_area,
        frame_num=int(frame_num),
        shift=float(sample_shift),
        sample_solver=str(sample_solver),
        sampling_steps=int(sampling_steps),
        guide_scale=guide_scale,
        n_prompt=neg_prompt,
        seed=int(seed),
        offload_model=bool(offload_model),
    )

    video_float01, video_uint8 = _wan_video_to_float01_uint8(
        video_3FHW_neg11,
        expected_F=int(frame_num),
        expected_hw=expected_hw,
    )
    H, W = int(video_uint8.shape[2]), int(video_uint8.shape[3])
    print(
        f"[stage_a] output resolution_hw=({H}, {W}); "
        f"input_hw={input_hw}; Wan size_label={wan_size_label}; "
        f"max_area={wan_max_area}"
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
        guide_scale=guide_scale,
        sample_shift=float(sample_shift),
        sample_solver=str(sample_solver),
        wan_ckpt_dir=str(wan_ckpt_dir),
        fps=int(fps),
        extra={
            "wan_size_label": wan_size_label,
            "wan_max_area": int(wan_max_area),
            "input_hw": [int(input_hw[0]), int(input_hw[1])],
            "aspect_preserved": True,
            "offload_model": bool(offload_model),
            "init_on_cpu": init_on_cpu,
            "convert_model_dtype": bool(convert_model_dtype),
        },
    )

    target_uint8_path = os.path.join(out_dir, "wan_video_target_3FHW_uint8.pt")
    torch.save(video_uint8, target_uint8_path)
    artifact_paths.append(target_uint8_path)

    result = StageAResult(
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
        guide_scale=guide_scale,
        sample_shift=float(sample_shift),
        sample_solver=str(sample_solver),
        wan_ckpt_dir=str(wan_ckpt_dir),
        out_dir=str(out_dir),
        artifact_paths=artifact_paths,
    )

    return result

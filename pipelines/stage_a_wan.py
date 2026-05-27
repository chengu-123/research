"""Stage A: Wan2.2 I2V articulation video generation (pipeline_v3 Section 5).

Single-shot, fixed-seed inference. Given a user input image with carpet
(grounding disk) baked in and a per-object motion prompt, generate a 21-frame
832x464 RGB video at 16 fps with a locked-off camera while the articulated
part moves. The output ``wan_video_target_3FHW`` becomes the universal target
for Stage B bootstrap (TRELLIS K=6 conditioning, Wan VAE latent target).

Hard constraints (pipeline_v3.3.1 Section 1.3 / 5.3):
  - frame_num = 21              (4*5 + 1; gives F_lat = 6 for K=6 states)
  - resolution = (464, 832)     (H=464, W=832; H/8=58, W/8=104, both /2 OK)
                                This is the actual Wan2.2/CHORD 480P
                                landscape output for the official 832*480
                                area profile. Off-distribution sizes
                                invalidate W-RFSDS because Wan's DiT was
                                never trained on that area scale.
  - seed = 42                   (no multi-candidate selection per CHORD A.1)
  - guide_scale = 5.0           (pipeline_v3 Section 5.3; CHORD 25->12 is
                                 for SDS distillation, NOT video generation)

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

import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

# -----------------------------------------------------------------------------
# Make the vendored Wan2.2 package importable. mine/Wan2.2/wan/__init__.py
# defines ``wan.WanI2V``; we add mine/Wan2.2 to sys.path so ``import wan`` works
# regardless of how this module is invoked.
# -----------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_WAN_ROOT = os.path.join(_REPO_ROOT, "Wan2.2")
if not os.path.isdir(_WAN_ROOT):
    raise RuntimeError(
        f"Stage A requires the vendored Wan2.2 sources at {_WAN_ROOT!r}; "
        "directory not found."
    )
if _WAN_ROOT not in sys.path:
    sys.path.insert(0, _WAN_ROOT)

from wan.configs.wan_i2v_A14B import i2v_A14B as _WAN_I2V_A14B_CFG
from wan.configs import MAX_AREA_CONFIGS as _WAN_MAX_AREA_CONFIGS
from wan.configs import SUPPORTED_SIZES as _WAN_SUPPORTED_SIZES
from wan.image2video import WanI2V

from pipelines.utils.optical_flow import OpticalFlowReport, background_static_check
from pipelines.utils.seeding import seed_everything
from pipelines.utils.visualization_a import save_all_stage_a_visualisations
from pipelines.wan_helpers import build_articulated_prompts


# Official Wan2.2 I2V-A14B size labels are area profiles for I2V, not strict
# output H/W. Wan image2video.py recomputes latent H/W from max_area plus input
# aspect ratio. CHORD uses the actual Wan default landscape output 832x464, so
# this pipeline treats (H, W) as actual output H/W and maps it back to Wan's
# official area label.
_WAN_TASK = "i2v-A14B"
_WAN_OUTPUT_TO_SIZE_LABEL = {
    (464, 832): "832*480",
    (832, 480): "480*832",
    (720, 1280): "1280*720",
    (1280, 720): "720*1280",
}
_SUPPORTED_HW = frozenset(_WAN_OUTPUT_TO_SIZE_LABEL.keys())
_SUPPORTED_SIZE_LABELS = frozenset(_WAN_SUPPORTED_SIZES[_WAN_TASK])
_MISSING_SIZE_LABELS = frozenset(_WAN_OUTPUT_TO_SIZE_LABEL.values()) - _SUPPORTED_SIZE_LABELS
if _MISSING_SIZE_LABELS:
    raise RuntimeError(
        f"Stage A Wan size labels {sorted(_MISSING_SIZE_LABELS)} are not in "
        f"Wan2.2 {_WAN_TASK} SUPPORTED_SIZES={sorted(_SUPPORTED_SIZE_LABELS)}"
    )


class WanQualityError(RuntimeError):
    """Raised when Wan2.2 output fails the camera-static sanity check."""


@dataclass
class StageAResult:
    """Stage A return payload.

    The downstream Stage B contract (pipeline_v3 Section 6.1) reads
    ``wan_video_target_3FHW`` as uint8 in [0, 255]; ``.float() / 255.0`` and
    ``* 2 - 1`` push it back to Wan VAE's [-1, 1] input range.
    """
    wan_video_target_3FHW: torch.Tensor      # [3, 21, 464, 832] uint8 [0,255]
    wan_video_float01_3FHW: torch.Tensor     # [3, 21, 464, 832] float32 [0,1]
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
    sanity_report: OpticalFlowReport
    artifact_paths: list = field(default_factory=list)


def _resize_input_image(image: Union[Image.Image, np.ndarray, torch.Tensor],
                        target_hw: Tuple[int, int]) -> Image.Image:
    """Coerce the user's input image into a PIL Image at exactly ``target_hw``.

    Wan I2V's generate() respects the input image aspect ratio when computing
    its internal latent (h, w) under ``max_area`` (see image2video.py line
    262-271). To get the CHORD/Wan default 832x464 output, resize the input
    image to (W=832, H=464) and use the official 832*480 max-area profile.
    Carpet/grounding-disk geometry placed by the user is assumed to already
    be visually centred and reasonably scaled for this aspect ratio.
    """
    H, W = int(target_hw[0]), int(target_hw[1])

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

    if pil.size != (W, H):  # PIL .size is (W, H)
        pil = pil.resize((W, H), Image.Resampling.LANCZOS)
    return pil


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
    resolution_hw: Tuple[int, int] = (464, 832),
    sampling_steps: int = 50,
    guide_scale: Union[float, Tuple[float, float]] = 5.0,
    sample_shift: float = 5.0,
    sample_solver: str = "unipc",
    lang: str = "zh",
    fps: int = 16,
    offload_model: bool = True,
    convert_model_dtype: bool = True,
    t5_cpu: bool = False,
    device_id: int = 0,
    sanity_check: bool = True,
    sanity_threshold_ratio: float = 0.0015,
    sanity_max_moved_fraction: float = 0.10,
    raise_on_sanity_fail: bool = True,
) -> StageAResult:
    """Stage A end-to-end driver.

    Parameters
    ----------
    image : PIL.Image | np.ndarray | torch.Tensor
        User-provided closed-state image with carpet/grounding disk baked in
        at the input (pipeline_v3 Section 1.1). Any aspect ratio is accepted;
        the image is LANCZOS-resized to ``resolution_hw`` before Wan I2V.
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
        Directory to receive the five Stage A debug artifacts.
    seed : int, default 42
        Wan2.2 RNG seed. pipeline_v3 Section 5.3 fixes this; do not vary.
    frame_num : int, default 21
        Must satisfy ``frame_num % 4 == 1``. pipeline_v3 Section 1.3 fixes 21.
    resolution_hw : (int, int), default (464, 832)
        Actual output (H, W). MUST map to one of Wan2.2 I2V-A14B official
        area labels, expressed as actual (H, W): (464, 832), (832, 480),
        (720, 1280), (1280, 720).
        Off-distribution sizes (e.g. 288x512) make the W-RFSDS gradient
        direction unreliable because Wan's DiT was never trained at that
        area scale. The default (464, 832) follows the official 832*480
        480P area profile and the CHORD/Wan actual landscape output.
    sampling_steps : int, default 50
        UniPC steps. pipeline_v3 Section 5.3.
    guide_scale : float | (float, float), default 5.0
        CFG. pipeline_v3 Section 5.3 specifies 5.0 for Stage A video
        generation. CHORD's 25->12 schedule belongs to W-RFSDS (Stage D/F),
        not here.
    sample_shift : float, default 5.0
        Wan flow scheduler shift. Wan i2v_A14B default.
    sample_solver : str, default "unipc"
        UniPC or DPM++.
    lang : {"zh", "en"}, default "zh"
        Prompt language for the camera-lock addon and negative prompt.
    fps : int, default 16
        MP4 writer frame rate. wan_shared_cfg.sample_fps default.
    offload_model, convert_model_dtype, t5_cpu : bool
        VRAM controls forwarded to ``WanI2V``. Defaults keep peak VRAM under
        ~55 GB on a single H800/H100 80 GB card at (464, 832, F=21).
    device_id : int, default 0
        CUDA device index. CPU execution is not supported (Wan VAE relies on
        CUDA kernels).
    sanity_check : bool, default True
        Run optical-flow sanity check on the generated video.
    sanity_threshold_ratio : float, default 0.0015
        Pixels-per-bbox-diagonal threshold on per-transition mean flow.
    sanity_max_moved_fraction : float, default 0.10
        Fail if more than this fraction of transitions exceed threshold.
    raise_on_sanity_fail : bool, default True
        When True (recommended) raise ``WanQualityError`` on failed sanity
        check, leaving the caller to decide whether to retry with a new seed
        or rewrite the prompt. When False the result is returned anyway so
        downstream stages can run on a known-bad video for debugging.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Stage A requires CUDA; Wan2.2 VAE decode is not implemented on CPU."
        )
    if frame_num <= 1 or (frame_num - 1) % 4 != 0:
        raise ValueError(
            f"frame_num must be of the form 4n+1 with n>=1; got {frame_num}"
        )
    H, W = int(resolution_hw[0]), int(resolution_hw[1])
    if H % 8 != 0 or W % 8 != 0:
        raise ValueError(f"resolution_hw must be multiples of 8; got ({H}, {W})")
    if (H // 8) % 2 != 0 or (W // 8) % 2 != 0:
        raise ValueError(
            f"resolution_hw / 8 must be even for DiT patch_size=(1,2,2); "
            f"got latent ({H // 8}, {W // 8})"
        )
    if (H, W) not in _SUPPORTED_HW:
        raise ValueError(
            f"resolution_hw=({H}, {W}) is not a supported actual Wan I2V "
            f"output (H, W). Supported outputs are {sorted(_SUPPORTED_HW)}. "
            f"Use the CHORD/Wan default 480P landscape (464, 832), 480P "
            f"portrait (832, 480), 720P landscape (720, 1280), or 720P "
            f"portrait (1280, 720)."
        )
    if not os.path.isdir(wan_ckpt_dir):
        raise FileNotFoundError(
            f"wan_ckpt_dir not a directory: {wan_ckpt_dir!r}"
        )

    os.makedirs(out_dir, exist_ok=True)

    seed_everything(int(seed))

    pos_prompt, neg_prompt = build_articulated_prompts(user_motion_prompt, lang=lang)

    pil_image = _resize_input_image(image, target_hw=(H, W))
    pil_image.save(os.path.join(out_dir, "input_s_0_with_carpet.png"))

    wan = WanI2V(
        config=_WAN_I2V_A14B_CFG,
        checkpoint_dir=wan_ckpt_dir,
        device_id=int(device_id),
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=bool(t5_cpu),
        init_on_cpu=True,
        convert_model_dtype=bool(convert_model_dtype),
    )

    # Wan2.2 I2V uses `max_area` only as an area profile; the actual (h, w)
    # is determined inside image2video.py from `max_area` plus input aspect.
    # For the default CHORD/Wan output (464, 832), keep the input at 464x832
    # but pass the official 832*480 max area profile.
    wan_size_label = _WAN_OUTPUT_TO_SIZE_LABEL[(H, W)]
    wan_max_area = int(_WAN_MAX_AREA_CONFIGS[wan_size_label])
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
        expected_hw=(H, W),
    )
    actual_h, actual_w = video_3FHW_neg11.shape[2], video_3FHW_neg11.shape[3]
    print(
        f"[stage_a] output resolution_hw=({H}, {W}); "
        f"Wan size_label={wan_size_label}; max_area={wan_max_area}; "
        f"Wan actual output (H, W) = ({actual_h}, {actual_w})"
    )

    if sanity_check:
        report = background_static_check(
            video_float01.numpy(),
            threshold_ratio=float(sanity_threshold_ratio),
            max_moved_fraction=float(sanity_max_moved_fraction),
        )
    else:
        report = OpticalFlowReport(
            passed=True,
            moved_fraction=0.0,
            max_moved_fraction=float(sanity_max_moved_fraction),
            threshold_ratio=float(sanity_threshold_ratio),
            threshold_pixels=0.0,
            bbox_diagonal=float(np.sqrt(H * H + W * W)),
            per_transition_displacement=[0.0] * (int(frame_num) - 1),
        )

    artifact_paths = save_all_stage_a_visualisations(
        video_3fhw_float01=video_float01.numpy(),
        out_dir=out_dir,
        report=report,
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
        sanity_report=report,
        artifact_paths=artifact_paths,
    )

    if sanity_check and (not report.passed) and raise_on_sanity_fail:
        raise WanQualityError(
            f"Wan2.2 video failed background-static sanity check: "
            f"moved_fraction={report.moved_fraction:.3f} > "
            f"max_moved_fraction={report.max_moved_fraction:.3f} "
            f"(threshold_pixels={report.threshold_pixels:.2f}, "
            f"bbox_diagonal={report.bbox_diagonal:.2f}). "
            f"Inspect {os.path.join(out_dir, 'optical_flow_per_frame.png')} "
            f"and {os.path.join(out_dir, 'wan_video_target.mp4')}. "
            f"Try a new seed, lower guide_scale (5.0 -> 3.5), or rewrite "
            f"the motion prompt with more explicit 'locked camera' language."
        )

    return result

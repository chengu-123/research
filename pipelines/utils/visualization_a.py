"""Stage A (Wan2.2 video generation) debug visualisations.

Four debug artifacts are produced for every Stage A run:
  - ``wan_video_target.mp4``       full 16-fps video
  - ``wan_video_grid.png``         all F frames laid out as a grid
  - ``keyframes_6.png``            state_indices = [0, 4, 8, 12, 16, 20] frames
                                   (matches Stage B 6-state sampling)
  - ``meta.json``                  prompts, seed, resolution, Wan settings

All inputs are expected as the Stage A canonical
``video_3fhw_float01: np.ndarray [3, F, H, W]`` in [0, 1].
"""

from __future__ import annotations

import json
import os
from typing import Iterable, List, Optional

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

def _video_uint8_fhwc(video_3fhw_float01: np.ndarray) -> np.ndarray:
    """[3, F, H, W] float [0, 1] -> [F, H, W, 3] uint8."""
    v = np.clip(video_3fhw_float01, 0.0, 1.0)
    v = (v * 255.0).astype(np.uint8)
    return np.transpose(v, (1, 2, 3, 0))


def save_video_mp4(
    video_3fhw_float01: np.ndarray,
    out_path: str,
    fps: int = 16,
    quality: int = 8,
) -> None:
    frames_fhwc = _video_uint8_fhwc(video_3fhw_float01)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=quality)
    for i in range(frames_fhwc.shape[0]):
        writer.append_data(frames_fhwc[i])
    writer.close()


def save_frame_grid(
    video_3fhw_float01: np.ndarray,
    out_path: str,
    n_cols: int = 7,
    label_frames: bool = True,
) -> None:
    """Save all F frames in a grid; each subplot labelled with frame index."""
    frames_fhwc = _video_uint8_fhwc(video_3fhw_float01)
    F = frames_fhwc.shape[0]
    n_rows = (F + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            idx = r * n_cols + c
            ax.set_xticks([]); ax.set_yticks([])
            if idx < F:
                ax.imshow(frames_fhwc[idx])
                if label_frames:
                    ax.set_title(f"f={idx}", fontsize=9)
            else:
                ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_keyframes(
    video_3fhw_float01: np.ndarray,
    out_path: str,
    state_indices: Iterable[int] = (0, 4, 8, 12, 16, 20),
) -> None:
    """Save the K keyframes that Stage B samples for cross-state conditioning."""
    frames_fhwc = _video_uint8_fhwc(video_3fhw_float01)
    F = frames_fhwc.shape[0]
    state_indices = [int(i) for i in state_indices]
    if any(i < 0 or i >= F for i in state_indices):
        raise ValueError(
            f"state_indices {state_indices} out of range for F={F}"
        )
    fig, axes = plt.subplots(1, len(state_indices), figsize=(2.6 * len(state_indices), 2.6))
    axes = np.atleast_1d(axes)
    for j, idx in enumerate(state_indices):
        axes[j].imshow(frames_fhwc[idx])
        axes[j].set_title(f"state_{j} (f={idx})", fontsize=10)
        axes[j].set_xticks([]); axes[j].set_yticks([])
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_meta_json(
    out_path: str,
    pos_prompt: str,
    neg_prompt: str,
    user_motion_prompt: str,
    lang: str,
    seed: int,
    frame_num: int,
    resolution_hw: tuple,
    sampling_steps: int,
    guide_scale,
    sample_shift: float,
    sample_solver: str,
    wan_ckpt_dir: str,
    extra: Optional[dict] = None,
) -> None:
    payload = {
        "user_motion_prompt": user_motion_prompt,
        "pos_prompt": pos_prompt,
        "neg_prompt": neg_prompt,
        "lang": lang,
        "seed": int(seed),
        "frame_num": int(frame_num),
        "resolution_hw": [int(resolution_hw[0]), int(resolution_hw[1])],
        "sampling_steps": int(sampling_steps),
        "guide_scale": (list(guide_scale) if isinstance(guide_scale, (tuple, list)) else float(guide_scale)),
        "sample_shift": float(sample_shift),
        "sample_solver": str(sample_solver),
        "wan_ckpt_dir": str(wan_ckpt_dir),
    }
    if extra is not None:
        payload["extra"] = extra
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_all_stage_a_visualisations(
    video_3fhw_float01: np.ndarray,
    out_dir: str,
    pos_prompt: str,
    neg_prompt: str,
    user_motion_prompt: str,
    lang: str,
    seed: int,
    frame_num: int,
    resolution_hw: tuple,
    sampling_steps: int,
    guide_scale,
    sample_shift: float,
    sample_solver: str,
    wan_ckpt_dir: str,
    fps: int = 16,
    state_indices: Iterable[int] = (0, 4, 8, 12, 16, 20),
    extra: Optional[dict] = None,
) -> List[str]:
    """Write the Stage A debug artifacts under ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []

    p_mp4 = os.path.join(out_dir, "wan_video_target.mp4")
    save_video_mp4(video_3fhw_float01, p_mp4, fps=fps)
    paths.append(p_mp4)

    p_grid = os.path.join(out_dir, "wan_video_grid.png")
    save_frame_grid(video_3fhw_float01, p_grid)
    paths.append(p_grid)

    p_kf = os.path.join(out_dir, "keyframes_6.png")
    save_keyframes(video_3fhw_float01, p_kf, state_indices=state_indices)
    paths.append(p_kf)

    p_meta = os.path.join(out_dir, "meta.json")
    save_meta_json(
        out_path=p_meta,
        pos_prompt=pos_prompt,
        neg_prompt=neg_prompt,
        user_motion_prompt=user_motion_prompt,
        lang=lang,
        seed=seed,
        frame_num=frame_num,
        resolution_hw=resolution_hw,
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        sample_shift=sample_shift,
        sample_solver=sample_solver,
        wan_ckpt_dir=wan_ckpt_dir,
        extra=extra,
    )
    paths.append(p_meta)

    return paths

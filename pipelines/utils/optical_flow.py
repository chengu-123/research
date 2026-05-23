"""Background-static sanity check for Wan2.2-generated articulated video.

Wan2.2 sometimes drifts the camera (despite the camera-lock prompt) or shifts
the entire object across the frame. Such failures break downstream
assumptions: U_object reuses state-0 voxel geometry as anchor, and Stage D
renders at a single locked camera. We must detect them at Stage A and abort
with a clear error rather than letting them corrupt later stages.

The check follows pipeline_v3 Section 5.4 (citing ViPS arxiv 2604.17623
Section 0.F.3): a frame is "moved" if its mean optical-flow magnitude
exceeds ``threshold_ratio * bbox_diagonal``. The video passes if at most
``max_moved_fraction`` (default 10%) of inter-frame transitions exceed the
threshold.

Implementation uses cv2.calcOpticalFlowFarneback on grayscale uint8 frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np


@dataclass
class OpticalFlowReport:
    passed: bool
    moved_fraction: float
    max_moved_fraction: float
    threshold_ratio: float
    threshold_pixels: float
    bbox_diagonal: float
    per_transition_displacement: List[float] = field(default_factory=list)


def _video_to_uint8_hwc(video_3fhw_float01: np.ndarray) -> np.ndarray:
    """Convert [3, F, H, W] in [0, 1] to [F, H, W, 3] in uint8 [0, 255]."""
    if video_3fhw_float01.ndim != 4 or video_3fhw_float01.shape[0] != 3:
        raise ValueError(
            f"video_3fhw_float01 must be [3, F, H, W]; got {video_3fhw_float01.shape}"
        )
    v = np.clip(video_3fhw_float01, 0.0, 1.0)
    v = (v * 255.0).astype(np.uint8)
    return np.transpose(v, (1, 2, 3, 0))  # [F, H, W, 3]


def _farneback_mean_flow_magnitude(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> float:
    """Return the mean magnitude (pixels) of Farneback dense optical flow."""
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_gray,
        next=curr_gray,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(mag.mean())


def background_static_check(
    video_3fhw_float01: np.ndarray,
    threshold_ratio: float = 0.0015,
    max_moved_fraction: float = 0.10,
) -> OpticalFlowReport:
    """Verify that the camera and background stay nearly static across frames.

    Parameters
    ----------
    video_3fhw_float01 : np.ndarray
        Wan2.2 output in [3, F, H, W] float [0, 1].
    threshold_ratio : float, default 0.0015
        A frame transition is "moved" if its mean optical-flow magnitude
        exceeds ``threshold_ratio * bbox_diagonal`` pixels. Default from
        pipeline_v3 Section 5.4 (citing ViPS arxiv 2604.17623 0.F.3).
    max_moved_fraction : float, default 0.10
        Pass criterion: at most this fraction of frame transitions exceed
        the threshold (ViPS reports 0.71% reject rate at this threshold).

    Returns
    -------
    OpticalFlowReport
        ``passed`` indicates pass/fail. ``per_transition_displacement`` is
        the F-1 length list of mean flow magnitudes (pixels). Use this for
        visualisation and debugging.
    """
    frames_uint8 = _video_to_uint8_hwc(video_3fhw_float01)  # [F, H, W, 3]
    F, H, W = frames_uint8.shape[:3]
    if F < 2:
        raise ValueError(f"need at least 2 frames for optical flow; got F={F}")

    bbox_diagonal = float(np.sqrt(H * H + W * W))
    threshold_pixels = threshold_ratio * bbox_diagonal

    gray_frames = [cv2.cvtColor(frames_uint8[i], cv2.COLOR_RGB2GRAY) for i in range(F)]
    displacements: List[float] = []
    for i in range(F - 1):
        d = _farneback_mean_flow_magnitude(gray_frames[i], gray_frames[i + 1])
        displacements.append(d)

    moved_count = int(sum(d > threshold_pixels for d in displacements))
    moved_fraction = moved_count / float(len(displacements))
    passed = moved_fraction <= max_moved_fraction

    return OpticalFlowReport(
        passed=passed,
        moved_fraction=moved_fraction,
        max_moved_fraction=max_moved_fraction,
        threshold_ratio=threshold_ratio,
        threshold_pixels=threshold_pixels,
        bbox_diagonal=bbox_diagonal,
        per_transition_displacement=displacements,
    )

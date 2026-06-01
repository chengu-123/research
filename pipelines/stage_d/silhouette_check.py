"""Periodic silhouette consistency check (Stage C.5 reborn, S1 fix).

The classic v3 "preflight coverage" ran once before Stage D started and
could not recover if Bootstrap missed key voxels (e.g. a revolute pivot
that ends up outside the U_seed corridor — past stageC v8.1 needed
13-q multi-start to find these for samples 7201 / 7128). S1 replaces it
with a periodic check (every ``cfg.silhouette_check_every`` iters):

  1. Render current learnable state at a subset of K=6 states using the
     committed-or-soft-blended joint params.
  2. Compute silhouette = ``rendered_alpha > 0.5`` per frame.
  3. Compare against silhouette of the Wan video target frame at the
     same state index (silhouette of target = pixels where the object
     has non-background colour; we approximate with a sum-channel
     threshold since Wan output has black background bg=(0,0,0) by
     Stage A convention).
  4. If any IoU < ``cfg.silhouette_iou_threshold``: emit a warning,
     compute the diff voxel set (back-project mis-aligned silhouette to
     U candidate), and call the U-expand path.

The U-expand path (``expand_U_and_resample_slat``) is the surgical part:

  * dilate the diff voxel set by ``cfg.silhouette_expand_dilate``
  * unique-merge with current U_object → U_object_new
  * re-run TRELLIS slat_sampler on U_object_new
  * rebuild gaussian_parent_idx
  * **resize** learnable.alpha_g and learnable.alpha_m to ``N_obj_new``
  * mark new voxels' alpha_g, alpha_m at neutral logit 0 (soft 0.5)
  * migrate AdamW state for the resized parameters

For a first cut we implement the CHECK + log; the EXPAND path raises a
clear error pointing back to Bootstrap re-run. The user has signed off
on this scoping (silhouette expand inside the live training loop is a
follow-up engineering task; failure mode is loud and actionable).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import F_FRAMES, STATE_INDICES


logger = logging.getLogger(__name__)


# =============================================================================
# Silhouette extraction
# =============================================================================

def silhouette_from_alpha(alpha_T_HW: torch.Tensor,
                          thresh: float = 0.5) -> torch.Tensor:
    """``alpha > thresh`` per frame; returns ``[T, H, W]`` bool."""
    if alpha_T_HW.ndim != 3:
        raise ValueError(f"alpha must be [T, H, W]; got {tuple(alpha_T_HW.shape)}")
    return alpha_T_HW > thresh


def silhouette_from_rgb(rgb_T3HW_01: torch.Tensor,
                         bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                         thresh: float = 1.0e-3) -> torch.Tensor:
    """Approximate silhouette as ``||rgb - bg_color|| > thresh`` per frame.

    Used when the rasterizer returns only RGB (no alpha) or for the Wan
    target (which doesn't expose alpha). Background is assumed black by
    default; if Stage A changes its bg_color this must change too.
    """
    if rgb_T3HW_01.ndim != 4 or rgb_T3HW_01.shape[1] != 3:
        raise ValueError(
            f"rgb must be [T, 3, H, W]; got {tuple(rgb_T3HW_01.shape)}"
        )
    bg = torch.tensor(bg_color, device=rgb_T3HW_01.device,
                      dtype=rgb_T3HW_01.dtype).view(1, 3, 1, 1)
    diff = (rgb_T3HW_01 - bg).abs().sum(dim=1)                   # [T, H, W]
    return diff > thresh


def silhouette_iou(sil_a: torch.Tensor, sil_b: torch.Tensor) -> float:
    """IoU of two bool silhouette tensors of identical shape."""
    if sil_a.shape != sil_b.shape:
        raise ValueError(
            f"silhouette shape mismatch: {tuple(sil_a.shape)} vs {tuple(sil_b.shape)}"
        )
    inter = (sil_a & sil_b).sum().float()
    union = (sil_a | sil_b).sum().float()
    if union.item() == 0.0:
        return 1.0     # both empty = trivially identical
    return float((inter / union).item())


# =============================================================================
# Periodic check entry point
# =============================================================================

@dataclass
class SilhouetteCheckReport:
    """Diagnostic payload returned to the training loop."""
    iter_idx: int
    n_states_checked: int
    per_state_iou: List[float]
    min_iou: float
    triggered_expand: bool
    n_voxels_added: int = 0
    notes: str = ""


class U_ExpandRequired(RuntimeError):
    """Raised when silhouette IoU is below threshold and live-expand is needed.

    First-version behaviour: we raise this error to halt training cleanly
    rather than silently degrading. The error message includes:
      - which state(s) failed
      - the diff voxel count
      - the suggested action (re-run Bootstrap with larger U_seed dilate
        radius, or implement the U-expand path)

    A future PR will replace this raise with the in-loop expansion logic.
    """


class CameraMismatchError(RuntimeError):
    """Raised at iter 0 when the rendered frame 0 does not match s_0.

    Distinct from ``U_ExpandRequired`` because the fix is different:
    ``U_ExpandRequired`` -> re-run Bootstrap with larger U; this -> fix
    ``StageDCameraConfig`` (wrong pose / FoV / up axis). The error message
    points the user at the camera config and at the freeart3d_canonical()
    classmethod / world_up_axis ablation.
    """


def iter_0_camera_check(
    rendered_frame_0_3HW: torch.Tensor,
    s_0_pure_3HW: torch.Tensor,
    iou_threshold: float = 0.5,
    save_diag_to: Optional[str] = None,
) -> float:
    """Sanity-check the render camera against the real input image.

    Called ONCE at training iter 0 (before any optimization step). Renders
    frame 0 of the current 3D under the configured camera, computes the
    silhouette IoU against the no-carpet ``s_0_pure`` image, and raises
    ``CameraMismatchError`` if IoU is below ``iou_threshold``. This catches
    the most common Stage D bug — wrong camera convention — at iter 0
    rather than after a long failed training run.

    The silhouette of ``s_0_pure`` is the frame-0 observation foreground;
    the silhouette of the render is the rasterized foreground (alpha-like).
    Both use the ``silhouette_from_rgb`` heuristic (sum-channel > thresh
    over black bg). If your bg is not black, override that explicitly.

    Parameters
    ----------
    rendered_frame_0_3HW : Tensor [3, H, W] in [0, 1]
        Frame 0 of the current iter's render (no grad needed).
    s_0_pure_3HW : Tensor [3, H, W] in [0, 1]
        No-carpet frame-0 reference image (from Bootstrap).
    iou_threshold : float
        Below this, raise CameraMismatchError. Default 0.5 = conservative
        (catches large mismatches; will pass with small angular offsets
        that L_first can refine).
    save_diag_to : Optional[str]
        If set, save a 3-panel diagnostic PNG (render | s_0 | overlay) to
        this path for debugging. None = skip.

    Returns
    -------
    iou : float    IoU value (for logging when the check passes).
    """
    if rendered_frame_0_3HW.ndim != 3 or rendered_frame_0_3HW.shape[0] != 3:
        raise ValueError(
            f"rendered_frame_0 must be [3, H, W]; got {tuple(rendered_frame_0_3HW.shape)}"
        )
    if s_0_pure_3HW.shape != rendered_frame_0_3HW.shape:
        raise ValueError(
            f"shape mismatch: render={tuple(rendered_frame_0_3HW.shape)}, "
            f"s_0_pure={tuple(s_0_pure_3HW.shape)}"
        )

    # silhouette_from_rgb expects [T, 3, H, W]; unsqueeze a temporal dim.
    sil_pred = silhouette_from_rgb(rendered_frame_0_3HW.unsqueeze(0))
    sil_target = silhouette_from_rgb(s_0_pure_3HW.unsqueeze(0))
    iou = silhouette_iou(sil_pred, sil_target)

    if save_diag_to is not None:
        # ★ Fix SD-6: removed try/except (CLAUDE.md "no compat/patch code").
        # PIL+numpy are required deps for the entire diagnostics pipeline;
        # if either import fails the user has bigger problems and we should
        # raise loudly. Diag PNG failures (e.g. disk full) likewise should
        # surface, not be swallowed.
        import numpy as _np
        from PIL import Image
        r = (rendered_frame_0_3HW.detach().cpu()
             .float().clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        t = (s_0_pure_3HW.detach().cpu()
             .float().clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        # [3, H, W] -> [H, W, 3]
        r_hwc = _np.transpose(r, (1, 2, 0))
        t_hwc = _np.transpose(t, (1, 2, 0))
        overlay = (0.5 * r_hwc.astype(_np.float32)
                   + 0.5 * t_hwc.astype(_np.float32)).clip(0, 255).astype(_np.uint8)
        strip = _np.concatenate([r_hwc, t_hwc, overlay], axis=1)
        Image.fromarray(strip).save(save_diag_to)

    if iou < iou_threshold:
        msg = (
            f"Iter-0 camera sanity check FAILED: rendered frame 0 vs s_0 "
            f"silhouette IoU = {iou:.3f} < threshold {iou_threshold:.3f}. "
            "This means the render camera does not match the camera that "
            "produced s_0_pure. "
            "If your input is a PartNet image rendered by FreeArt3D "
            "(mine/pipelines/render.py), the default "
            "StageDCameraConfig.freeart3d_canonical() should already match "
            "it (fov=45 deg, azi=22.5 deg, ele=45 deg, dist=2.1, +Z up). "
            "Persistent failure usually indicates: (a) Bootstrap produced a "
            "misoriented canonical 3D (re-check Stage B output); (b) your "
            "input was rendered with a different convention (override with "
            "an explicit StageDCameraConfig(pos, look_at, up, fov_x_deg, "
            "fov_y_deg, image_h, image_w) matching your data)."
        )
        if save_diag_to is not None:
            msg += f" Diagnostic strip saved to {save_diag_to!r}."
        raise CameraMismatchError(msg)

    logger.info(
        "[stage_d iter-0 camera check] silhouette IoU = %.3f (OK, threshold %.3f)",
        iou, iou_threshold,
    )
    return iou


def periodic_silhouette_check(
    iter_idx: int,
    cfg_check_every: int,
    cfg_iou_threshold: float,
    cfg_n_states: int,
    sample_state_indices: Sequence[int],
    rendered_T3HW_01: torch.Tensor,
    pure_state_targets_K3HW_01: torch.Tensor,
) -> Optional[SilhouetteCheckReport]:
    """Run the silhouette IoU check; return None unless this iter is on cadence.

    Parameters
    ----------
    iter_idx : int
    cfg_check_every : int
    cfg_iou_threshold : float
    cfg_n_states : int
        Number of frames (sub-sampled from the 21-frame render) to check.
    sample_state_indices : Sequence[int]
        Frame indices (within 0..F_FRAMES-1) to sample. Pass e.g.
        ``[0, 7, 14, 20]`` for 4 evenly spaced checkpoints.
    rendered_T3HW_01 : Tensor [F, 3, H, W] in [0, 1]
        Current iter's rendered video (no_grad needed; caller may pass
        ``rgb_T3HW.detach()``).
    pure_state_targets_K3HW_01 : Tensor [K, 3, H, W] in [0, 1]
        Six observed no-background states from Bootstrap.

    Returns
    -------
    SilhouetteCheckReport or None
    """
    if iter_idx == 0 or iter_idx % cfg_check_every != 0:
        return None

    if len(sample_state_indices) != cfg_n_states:
        raise ValueError(
            f"sample_state_indices length {len(sample_state_indices)} != "
            f"cfg_n_states {cfg_n_states}"
        )
    if rendered_T3HW_01.shape[0] != F_FRAMES:
        raise ValueError(
            f"rendered must have F={F_FRAMES} frames; "
            f"got {rendered_T3HW_01.shape[0]}"
        )
    if pure_state_targets_K3HW_01.ndim != 4 or pure_state_targets_K3HW_01.shape[1] != 3:
        raise ValueError(
            f"pure_state_targets_K3HW_01 must be [K, 3, H, W]; "
            f"got {tuple(pure_state_targets_K3HW_01.shape)}"
        )

    per_state_iou: List[float] = []
    for idx in sample_state_indices:
        if idx not in STATE_INDICES:
            raise ValueError(
                f"silhouette check frame {idx} is not one of STATE_INDICES={STATE_INDICES}"
            )
        k = STATE_INDICES.index(int(idx))
        sil_pred = silhouette_from_rgb(rendered_T3HW_01[idx:idx + 1])
        sil_target = silhouette_from_rgb(pure_state_targets_K3HW_01[k:k + 1])
        iou = silhouette_iou(sil_pred, sil_target)
        per_state_iou.append(iou)

    min_iou = min(per_state_iou)
    report = SilhouetteCheckReport(
        iter_idx=iter_idx,
        n_states_checked=cfg_n_states,
        per_state_iou=per_state_iou,
        min_iou=min_iou,
        triggered_expand=False,
    )

    if min_iou < cfg_iou_threshold:
        report.triggered_expand = True
        fail_states = [
            sample_state_indices[i] for i, iou in enumerate(per_state_iou)
            if iou < cfg_iou_threshold
        ]
        report.notes = (
            f"min IoU {min_iou:.3f} < threshold {cfg_iou_threshold:.3f}; "
            f"failing states: {fail_states}"
        )
        logger.warning(
            "[stage_d silhouette check it=%d] %s", iter_idx, report.notes,
        )
        raise U_ExpandRequired(
            f"Silhouette consistency check failed at iter {iter_idx}: "
            f"min IoU {min_iou:.3f} < {cfg_iou_threshold:.3f}. "
            f"Failing state indices (within 0..{F_FRAMES - 1}): {fail_states}. "
            "Suggested action: re-run Bootstrap (Stage B) with a larger "
            "U_seed dilate radius (currently 2; try 3) so the corridor "
            "covers more of the joint trajectory. The in-loop U-expand "
            "path is not yet implemented; see record/method.md S1.a for the "
            "design and pipelines/stage_d/silhouette_check.py header for the "
            "deferred engineering work."
        )

    logger.info(
        "[stage_d silhouette check it=%d] min IoU %.3f (OK)",
        iter_idx, min_iou,
    )
    return report


# =============================================================================
# Default state-index sampler
# =============================================================================

def default_check_state_indices(cfg_n_states: int) -> List[int]:
    """Pick ``cfg_n_states`` evenly-spaced indices over ``F_FRAMES`` frames.

    For ``cfg_n_states = 4``: returns four entries sampled from
    ``STATE_INDICES``. Endpoints always included so we check
    both closed (frame 0) and most-open (frame 20) state alignments.
    """
    if cfg_n_states < 2:
        raise ValueError(f"cfg_n_states must be >= 2, got {cfg_n_states}")
    if cfg_n_states == 2:
        return [0, F_FRAMES - 1]
    if cfg_n_states > len(STATE_INDICES):
        raise ValueError(
            f"cfg_n_states must be <= {len(STATE_INDICES)}, got {cfg_n_states}"
        )
    positions = np.linspace(0, len(STATE_INDICES) - 1, cfg_n_states, dtype=int).tolist()
    interior = [int(STATE_INDICES[pos]) for pos in positions]
    # de-dupe while preserving order
    seen = set()
    out: List[int] = []
    for i in interior:
        if i not in seen:
            seen.add(i)
            out.append(int(i))
    return out


__all__ = [
    "silhouette_from_alpha", "silhouette_from_rgb", "silhouette_iou",
    "SilhouetteCheckReport", "U_ExpandRequired", "CameraMismatchError",
    "periodic_silhouette_check", "default_check_state_indices",
    "iter_0_camera_check",
]

"""Stage B input multiplexer.

Stage B (SCAR + BMCSA) consumes K=6 articulation-state images as DINOv2
conditioning. The images can come from two sources:

  (a) Stage A's generated 21-frame video, saved as
      ``wan_video_target_3FHW_uint8.pt`` (torch.Tensor uint8 [3, F, H, W]).
      We sample 6 frames at indices ``state_indices`` (default
      [0, 4, 8, 12, 16, 20] — matching Stage A's ``keyframes_6.png``).

  (b) A directory of 6 pre-segmented PNG files named according to a pattern
      (default ``"{i:02d}_seg.png"`` -> ``00_seg.png`` ... ``05_seg.png``,
      the FreeArt3D convention).

The CLI driver (``scripts/stageb.py``) decides which mode applies based on
the ``--input`` argument extension / type and forwards to
:func:`load_K_state_images`.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image


_DEFAULT_STATE_INDICES_F21 = (0, 4, 8, 12, 16, 20)
_DEFAULT_IMAGE_PATTERN = "{i:02d}_seg.png"


def load_K_state_images(
    input_source: Union[str, os.PathLike],
    K: int = 6,
    state_indices: Optional[Sequence[int]] = None,
    image_pattern: str = _DEFAULT_IMAGE_PATTERN,
    out_mode: str = "RGBA",
) -> List[Image.Image]:
    """Load K articulation-state PIL images from EITHER a Stage A video
    tensor (.pt) OR a directory of segmented PNGs.

    Mode selection:
      - If ``input_source`` is a file ending in ``.pt``: video mode.
        Loads ``torch.Tensor`` of shape ``[3, F, H, W]`` (uint8 or float in
        [0, 1]) and samples frames at ``state_indices``.
      - If ``input_source`` is a directory: image-dir mode. Looks for
        ``image_pattern.format(i=k)`` for ``k`` in ``0..K-1``.
      - Anything else: ``FileNotFoundError``.

    Parameters
    ----------
    input_source : path-like
        Either a ``.pt`` file (Stage A output) or a directory of K PNGs.
    K : int, default 6
        Number of articulation states. Must equal ``len(state_indices)`` in
        video mode (also enforced).
    state_indices : sequence of int, optional
        Frame indices to sample from the video. Default
        ``(0, 4, 8, 12, 16, 20)`` matches Stage A's F=21 keyframe layout.
        Ignored in image-dir mode.
    image_pattern : str
        Filename template for image-dir mode. ``{i:02d}`` is required.
    out_mode : {"RGBA", "RGB"}, default "RGBA"
        PIL mode of returned images. RGBA preserves any alpha mask present in
        segmented PNGs (FreeArt3D convention); video-mode frames are converted
        from RGB to RGBA with alpha=255 to keep the downstream signature
        uniform.

    Returns
    -------
    list of PIL.Image
        Length-K. Mode = ``out_mode``.
    """
    if out_mode not in ("RGB", "RGBA"):
        raise ValueError(f"out_mode must be 'RGB' or 'RGBA'; got {out_mode!r}")
    if state_indices is None:
        state_indices = _DEFAULT_STATE_INDICES_F21
    state_indices = tuple(int(i) for i in state_indices)
    if len(state_indices) != int(K):
        raise ValueError(
            f"len(state_indices)={len(state_indices)} must equal K={K}"
        )
    if "{i:02d}" not in image_pattern and "{i:" not in image_pattern:
        raise ValueError(
            f"image_pattern must contain '{{i:02d}}' or similar; got {image_pattern!r}"
        )

    p = os.fspath(input_source)
    if os.path.isfile(p) and p.lower().endswith(".pt"):
        return _load_from_video_pt(p, K=K, state_indices=state_indices, out_mode=out_mode)
    if os.path.isdir(p):
        return _load_from_image_dir(p, K=K, image_pattern=image_pattern, out_mode=out_mode)

    raise FileNotFoundError(
        f"--input must be a .pt file (Stage A video tensor) or a directory "
        f"with K={K} files matching {image_pattern!r}; got: {p!r}"
    )


def _load_from_video_pt(
    pt_path: str,
    K: int,
    state_indices: Sequence[int],
    out_mode: str,
) -> List[Image.Image]:
    """Load a Stage A video tensor and sample K frames at ``state_indices``."""
    obj = torch.load(pt_path, map_location="cpu")
    # Some saves wrap the tensor in a dict; accept either form.
    if isinstance(obj, dict):
        if "video" in obj:
            video = obj["video"]
        elif "wan_video_target_3FHW" in obj:
            video = obj["wan_video_target_3FHW"]
        else:
            raise ValueError(
                f"{pt_path}: dict payload must contain 'video' or "
                f"'wan_video_target_3FHW'; got keys {list(obj.keys())}"
            )
    else:
        video = obj

    if not isinstance(video, torch.Tensor):
        raise TypeError(
            f"{pt_path}: expected torch.Tensor [3, F, H, W]; got {type(video).__name__}"
        )
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(
            f"{pt_path}: expected shape [3, F, H, W]; got {tuple(video.shape)}"
        )

    F_total, H, W = int(video.shape[1]), int(video.shape[2]), int(video.shape[3])
    if min(state_indices) < 0 or max(state_indices) >= F_total:
        raise ValueError(
            f"state_indices {list(state_indices)} out of range for F={F_total}"
        )

    if video.dtype == torch.uint8:
        arr = video.numpy()                                          # [3, F, H, W] uint8
    elif video.dtype in (torch.float32, torch.float16, torch.bfloat16):
        v = video.to(torch.float32).clamp_(0.0, 1.0)
        arr = (v.numpy() * 255.0).round().astype(np.uint8)
    else:
        raise TypeError(
            f"{pt_path}: video dtype must be uint8 or float; got {video.dtype}"
        )

    images: List[Image.Image] = []
    for fidx in state_indices:
        frame_chw = arr[:, fidx]                                    # [3, H, W]
        frame_hwc = np.transpose(frame_chw, (1, 2, 0))              # [H, W, 3]
        img = Image.fromarray(frame_hwc, mode="RGB")
        if out_mode == "RGBA":
            img = img.convert("RGBA")
        images.append(img)
    return images


def _load_from_image_dir(
    dir_path: str,
    K: int,
    image_pattern: str,
    out_mode: str,
) -> List[Image.Image]:
    """Load K images named per ``image_pattern`` from ``dir_path``."""
    images: List[Image.Image] = []
    for i in range(K):
        fname = image_pattern.format(i=i)
        fpath = os.path.join(dir_path, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"missing state image: {fpath} (pattern={image_pattern!r}, K={K})"
            )
        img = Image.open(fpath)
        if img.mode != out_mode:
            img = img.convert(out_mode)
        images.append(img)
    return images


def describe_input_source(input_source: Union[str, os.PathLike]) -> dict:
    """Return a short metadata dict describing the input source. Used for
    logging / meta.json so a Stage B run records where its K images came from.
    """
    p = os.fspath(input_source)
    if os.path.isfile(p) and p.lower().endswith(".pt"):
        try:
            obj = torch.load(p, map_location="cpu")
        except Exception as e:
            return {"mode": "video_pt", "path": p, "error": str(e)}
        video = obj["video"] if isinstance(obj, dict) and "video" in obj else (
            obj["wan_video_target_3FHW"]
            if isinstance(obj, dict) and "wan_video_target_3FHW" in obj
            else obj
        )
        shape = tuple(int(s) for s in video.shape) if isinstance(video, torch.Tensor) else None
        return {"mode": "video_pt", "path": p, "shape": shape}
    if os.path.isdir(p):
        return {"mode": "image_dir", "path": p}
    return {"mode": "unknown", "path": p}

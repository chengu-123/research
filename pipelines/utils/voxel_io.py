"""Shared I/O helpers for the v1 pipeline: input image loading,
voxel grid persistence, and joint-info JSON assembly."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image


def load_seg_images(input_dir: str, K: int, pattern: str = "{i:02d}_seg.png") -> List[Image.Image]:
    """Load K per-state segmented images from `input_dir` following `pattern`.

    Example pattern: "{i:02d}_seg.png" -> 00_seg.png .. 05_seg.png.
    Raises FileNotFoundError if any file is missing; preserves the natural
    state order implied by i = 0 .. K-1.
    """
    images: List[Image.Image] = []
    for i in range(K):
        fname = pattern.format(i=i)
        fpath = os.path.join(input_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Expected input image not found: {fpath}")
        images.append(Image.open(fpath).convert("RGBA"))
    return images


def save_voxel_grid(path: str, grid: np.ndarray, dtype: Any = None) -> None:
    """Persist a voxel grid as `.npy`. If `dtype` is given, cast first."""
    arr = np.asarray(grid)
    if dtype is not None:
        arr = arr.astype(dtype)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, arr)


def load_voxel_grid(path: str) -> np.ndarray:
    return np.load(path)


def _to_python_list(x: Any) -> Any:
    """Convert numpy / torch tensors to plain Python lists for JSON dumps."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_python_list(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_python_list(v) for k, v in x.items()}
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def compose_joint_info_json(sajo_result: Any, out_path: str) -> Dict[str, Any]:
    """Write a joint_info.json file that is drop-in compatible with the
    existing [pipelines/urdf.py](pipelines/urdf.py):`write_urdf` reader and
    with `eval_utils/joints.py:Joint(..., method='ours')`.

    The expected schema is a list of per-joint entries (v1 ships exactly one):
    ``[{"type", "range", "axis": {"origin", "direction"}}]``

    `sajo_result` is the `SAJOResult` dataclass defined in
    `pipelines/stage_c_sajo.py`. We accept either the dataclass or an
    equivalent dict to keep unit-testing easy.
    """
    if is_dataclass(sajo_result):
        data = asdict(sajo_result)
    elif isinstance(sajo_result, dict):
        data = dict(sajo_result)
    else:
        data = {k: getattr(sajo_result, k) for k in (
            "joint_type", "omega", "q", "phi_k",
        )}

    joint_type = str(data["joint_type"])
    phi_k = _to_python_list(data["phi_k"])
    phi_min = float(min(phi_k))
    phi_max = float(max(phi_k))

    direction = _to_python_list(data["omega"])
    origin = _to_python_list(data["q"])

    entry = {
        "type": joint_type,
        "range": [phi_min, phi_max],
        "axis": {
            "origin": origin,
            "direction": direction,
        },
    }
    payload = [entry]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return entry


def save_json(obj: Any, path: str) -> None:
    """Dump any JSON-serializable object (converting tensors/arrays) to `path`."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_python_list(obj), f, indent=2)

"""C.2 Feature upsample and L2 normalization.

Upsamples the Stage B SS latent ``z_final`` from 16^3 to 64^3 and
optionally L2-normalizes along the channel axis. Also the loader for
the ``dit_hidden`` feature source used by the C.8 auto-switch gate
(when Stage B has saved a DiT block hook).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def upsample_features(
    z: torch.Tensor,
    out_size: int = 64,
    mode: str = "trilinear",
) -> torch.Tensor:
    """Upsample ``(K, C, d, d, d)`` features to ``(K, C, D, D, D)``.

    Uses ``align_corners=True`` to match the world-coord convention of
    ``pipelines/sajo/warp.py`` (voxel index ``i / (R-1)`` at corners).
    """
    if z.dim() != 5:
        raise ValueError(f"expected (K,C,d,d,d); got {tuple(z.shape)}")
    return F.interpolate(
        z,
        size=(out_size, out_size, out_size),
        mode=mode,
        align_corners=True if mode == "trilinear" else None,
    )


def l2_normalize(F_feat: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """L2-normalize features along the channel axis (dim=1)."""
    if F_feat.dim() != 5:
        raise ValueError(f"expected (K,C,D,H,W); got {tuple(F_feat.shape)}")
    norm = F_feat.norm(dim=1, keepdim=True).clamp_min(eps)
    return F_feat / norm


def load_feature_source(
    stage_b_dir: str,
    source: str = "z_final",
    dit_block: int = 18,
    out_size: int = 64,
    l2norm: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load a per-voxel 3D descriptor field from Stage B artifacts.

    ``source``:
      - ``"z_final"``: loads ``stage_b/z_final.pt`` (K, 8, 16, 16, 16) and
        trilinearly upsamples to (K, 8, 64, 64, 64).
      - ``"dit_hidden"``: loads ``stage_b/dit_hidden_block_{dit_block}.pt``
        (K, C_hidden, 64, 64, 64) if present — requires a Stage B hook.

    Returns
    -------
    torch.Tensor
        ``(K, C, out_size, out_size, out_size)`` optionally L2-normalized.
    """
    if source == "z_final":
        path = os.path.join(stage_b_dir, "z_final.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"z_final not found at {path}")
        z = torch.load(path, map_location="cpu")
        if isinstance(z, dict):
            if "z_final" in z:
                z = z["z_final"]
            else:
                raise KeyError(f"z_final.pt dict has no 'z_final' key; keys={list(z.keys())}")
        z = z.to(dtype=dtype)
        if z.dim() == 4:
            z = z.unsqueeze(0)
        feat = upsample_features(z, out_size=out_size, mode="trilinear")
    elif source == "dit_hidden":
        path = os.path.join(stage_b_dir, f"dit_hidden_block_{dit_block}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"DiT hidden features not found at {path}; "
                f"Stage B hook must be enabled."
            )
        feat = torch.load(path, map_location="cpu").to(dtype=dtype)
        if feat.shape[-1] != out_size:
            feat = upsample_features(feat, out_size=out_size, mode="trilinear")
    else:
        raise ValueError(f"Unknown feature source: {source}")

    if device is not None:
        feat = feat.to(device)
    if l2norm:
        feat = l2_normalize(feat)
    return feat


def load_m_attn_64(stage_b_dir: str,
                   device: Optional[torch.device] = None,
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Load ``M_attn_64.npy`` from ``stage_b/viz/bmcsa/``."""
    path = os.path.join(stage_b_dir, "viz", "bmcsa", "M_attn_64.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"M_attn_64 not found at {path}")
    arr = np.load(path)
    tensor = torch.from_numpy(arr).to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def load_z_final(stage_b_dir: str,
                  device: Optional[torch.device] = None,
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Load the raw Pass-1 SS latent ``z_final.pt`` — (K, C, d, d, d).

    This is the UN-COMPRESSED per-voxel 8-dim feature from the TRELLIS
    SS-VAE encoder. Used as the primary signal for always_on voxel
    material classification (replacing M_attn, which collapses this
    information into a scalar cross-state agreement).
    """
    path = os.path.join(stage_b_dir, "z_final.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"z_final.pt not found at {path}")
    z = torch.load(path, map_location="cpu")
    if isinstance(z, dict):
        if "z_final" in z:
            z = z["z_final"]
        else:
            raise KeyError(f"z_final.pt dict has no 'z_final' key; keys={list(z.keys())}")
    z = z.to(dtype=dtype)
    if z.dim() == 4:
        z = z.unsqueeze(0)
    if device is not None:
        z = z.to(device)
    return z


def load_O_stack(stage_b_dir: str,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Load ``O_stack.npy`` — per-state occupancy (K, D, H, W)."""
    path = os.path.join(stage_b_dir, "O_stack.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"O_stack not found at {path}")
    arr = np.load(path)
    tensor = torch.from_numpy(arr).to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def load_dit_hidden(
    stage_b_dir: str,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Optional[dict]:
    """Load SS-DiT mid-late block hidden states saved by Stage B (v8+).

    Returns dict ``{block_idx: (K, L, C) tensor}`` with spatial tokens L=4096
    (for 16^3 grid) and channel C=1024, in the requested dtype/device. The
    file also contains a ``"meta"`` key (target_blocks list, t_star, etc.).

    Returns ``None`` if the file is not present (older Stage B runs).
    """
    path = os.path.join(stage_b_dir, "dit_hidden.pt")
    if not os.path.exists(path):
        return None
    blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, dict) or "hidden_states" not in blob:
        raise ValueError(
            f"dit_hidden.pt missing 'hidden_states' key; got keys={list(blob.keys()) if isinstance(blob, dict) else type(blob)}"
        )
    raw = blob["hidden_states"]
    out: dict = {}
    for idx, tensor in raw.items():
        t = tensor.to(dtype=dtype)
        if device is not None:
            t = t.to(device)
        out[int(idx)] = t
    # Attach meta
    out["_meta"] = blob.get("meta", {})
    return out

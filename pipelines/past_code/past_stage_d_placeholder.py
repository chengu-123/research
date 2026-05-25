"""Stage D placeholder: one standard TRELLIS Stage-2 call on the
state-0 occupancy grid, followed by the default TRELLIS mesh decoder.

This file intentionally stays thin — when full ACTF lands in a later
iteration, the whole file will be replaced by ``pipelines/stage_d_actf.py``
and ``pipelines/stage_e_fusion.py``. The public return type is kept
minimal so downstream code (``run_v1.py``, ``stage_f_assemble.py``) can
be ported to the full pipeline with just an import swap.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass
class StageDResult:
    slat: Any                 # trellis sparse tensor
    mesh: Any                 # trimesh.Trimesh (GLB-bakeable)
    coords: torch.Tensor      # (N, 4) int voxel coordinates used for Stage-2


def run_stage_d_placeholder(
    pipe: Any,
    cond_0: Dict[str, torch.Tensor],
    O_0: torch.Tensor,
    out_dir: str,
    cfg_stage_d: Any,
) -> StageDResult:
    """Run one standard Stage-2 + mesh decode pass on state-0 occupancy.

    Parameters
    ----------
    pipe : TrellisImageTo3DPipeline
        TRELLIS pipeline instance.
    cond_0 : dict
        Single-state conditioning: ``{'cond': (1, 1369, 768),
        'neg_cond': (1, 1369, 768)}``. The caller is expected to slice
        the full K-state cond stack before calling us.
    O_0 : torch.Tensor
        ``(64, 64, 64)`` binary occupancy grid.
    out_dir : str
        Destination directory.
    cfg_stage_d : OmegaConf
        The ``stage_d:`` subtree of ``configs/v1.yaml``.

    Returns
    -------
    StageDResult
    """
    os.makedirs(out_dir, exist_ok=True)
    device = O_0.device if hasattr(O_0, "device") else torch.device("cuda")

    # Prepare voxel input for Stage-2 in the same shape recon.py used.
    voxel = O_0[None, None].to(device=device, dtype=torch.float32)    # (1,1,64,64,64)
    coords = torch.argwhere(voxel > 0)[:, [0, 2, 3, 4]].int()

    # Override the pipeline default Stage-2 sampler params with v1.yaml values.
    sampler_params = dict(pipe.slat_sampler_params)
    if hasattr(cfg_stage_d, "stage2_sampler_params"):
        s2 = cfg_stage_d.stage2_sampler_params
        if s2 is not None:
            for key in ("steps", "cfg_strength"):
                if hasattr(s2, key):
                    sampler_params[key] = float(getattr(s2, key))
            if hasattr(s2, "cfg_interval"):
                sampler_params["cfg_interval"] = tuple(s2.cfg_interval)

    slat = pipe.sample_slat(cond_0, coords, sampler_params, noise=None)

    # Decode to mesh via TRELLIS's default decoder.
    outputs = pipe.decode_slat(slat)

    # Bake to GLB using TRELLIS postprocessing utilities.
    from trellis.utils import postprocessing_utils
    mesh = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=float(cfg_stage_d.simplify),
        texture_size=int(cfg_stage_d.texture_size),
        baking_mode=str(cfg_stage_d.baking_mode),
    )

    # Persist outputs.
    torch.save({"coords": coords.cpu()}, os.path.join(out_dir, "slat_meta.pt"))
    np.save(os.path.join(out_dir, "coords_0.npy"), coords.cpu().numpy().astype(np.int32))
    mesh.export(os.path.join(out_dir, "mesh_full.glb"))

    return StageDResult(slat=slat, mesh=mesh, coords=coords)

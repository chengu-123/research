"""End-to-end smoke test: SCAR + ICP on real 30857 input.

This test is marked slow and is skipped if:
  - CUDA is not available
  - TRELLIS checkpoints are not cached
  - The 30857 input directory is missing

Intended to be run manually or in CI with the mine environment set up.
"""

import json
import os
import sys
import shutil

import numpy as np
import pytest


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "30857")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "30857_scar_smoke")

requires_cuda = pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="needs CUDA",
)
requires_data = pytest.mark.skipif(
    not os.path.isdir(DATA_DIR) or
    not os.path.exists(os.path.join(DATA_DIR, "rendering_joint_00_state_00.png")),
    reason=f"30857 data missing in {DATA_DIR}",
)


@requires_cuda
@requires_data
@pytest.mark.slow
def test_scar_on_30857_produces_consistent_base():
    """SCAR should produce K=6 grids with high pairwise IoU in the base region."""
    import torch
    from omegaconf import OmegaConf
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.pipelines.samplers import SCARSampler
    from PIL import Image

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pipelines.stage_b_scar import run_scar

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    pipe = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipe.cuda()

    K = 6
    images = [
        Image.open(os.path.join(DATA_DIR, f"rendering_joint_00_state_{i:02d}.png")).convert("RGBA")
        for i in range(K)
    ]
    preprocessed = [pipe.preprocess_image(im) for im in images]
    cond = pipe.get_cond(preprocessed)

    pipe.sparse_structure_sampler = SCARSampler(
        sigma_min=pipe.sparse_structure_sampler.sigma_min,
    )
    pipe.sparse_structure_sampler_params["steps"] = 12
    pipe.sparse_structure_sampler_params["cfg_strength"] = 7.5
    pipe.sparse_structure_sampler_params["cfg_interval"] = (0.0, 1.0)

    result = run_scar(
        pipe=pipe, cond=cond, K=K,
        out_dir=OUT_DIR,
        cfg_scar={
            "alpha_schedule": [0.7, 1.0, 1.0, 0.5],
            "icp_enabled": True,
            "icp_max_translation": 1.5,
            "icp_vote_threshold": 0.83,
        },
        seed=0, remove_disk_flag=True,
    )

    assert result.O_stack.shape == (K, 64, 64, 64)

    O_np = result.O_stack.detach().cpu().numpy().astype(bool)

    votes = O_np.sum(axis=0)
    base_mask = votes >= 5
    if base_mask.sum() > 0:
        ious = []
        for k in range(K):
            k_base = O_np[k] & base_mask
            iou = k_base.sum() / max(base_mask.sum(), 1)
            ious.append(iou)
        mean_iou = float(np.mean(ious))
        assert mean_iou > 0.9, f"base-region IoU only {mean_iou:.3f}; expected > 0.9"

    with open(os.path.join(OUT_DIR, "scar_diagnostics.json")) as f:
        diag = json.load(f)
    for s in range(4):
        assert diag[s]["alpha"] > 0, f"step {s} alpha=0 (SCAR not active)"
    for s in range(4, 12):
        assert diag[s]["alpha"] == 0.0, f"step {s} alpha != 0 (SCAR still active)"

    with open(os.path.join(OUT_DIR, "icp_report.json")) as f:
        icp_rep = json.load(f)
    for offs in icp_rep["offsets"]:
        assert np.linalg.norm(offs) <= 1.5 + 1e-4, f"ICP offset exceeds cap: {offs}"

    assert os.path.exists(os.path.join(OUT_DIR, "viz", "O_stack.html"))
    assert os.path.exists(os.path.join(OUT_DIR, "viz", "scar_diagnostics.html"))

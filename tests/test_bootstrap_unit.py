"""Unit tests for pipelines/bootstrap.py helpers.

Exercises the pure-Python/torch helpers (no GPU, no TRELLIS pipeline, no
Wan2.2 checkpoints). Validates:

  - _detect_carpet_freeart3d returns plausible mask
  - _compute_b5_u_seed produces non-empty voxel set with expected ranges
  - _compute_swept_volume_corridor warps U_seed without crashing
  - _union_voxel_sets dedups correctly
  - _se3_prismatic / _se3_revolute SE(3) matrices are valid
  - _save_bootstrap_artifacts writes the expected files

Heavy steps (B1 Stage A, B3-B4 run_scar, B8 SLAT, B10 Wan cond, B11 Wan VAE)
are not exercised here; they are tested separately as e2e on GPU.
"""

import os
import sys
import tempfile
import importlib.util

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.bootstrap import (
    BootstrapConfig,
    BootstrapResult,
    _compute_b4_derived,
    _compute_b5_u_seed,
    _compute_slat_shell_mask,
    _compute_swept_volume_corridor,
    _detect_carpet_freeart3d,
    _dilate_voxels,
    _flat_idx_to_xyz,
    _run_b1_stage_a,
    _save_bootstrap_artifacts,
    _se3_prismatic,
    _se3_revolute,
    _state_images_to_video_tensor,
    _union_voxel_sets,
    _voxel_to_world,
    _world_to_voxel,
)
from pipelines.stage_c import JointInit, Psi, StageCConfig, StageCInputs, run_stage_c_joint_init


def test_wan_resolution_contract():
    cfg = BootstrapConfig()
    assert cfg.stage_a_wan_size == "832*480"
    assert cfg.stage_a_resolution_hw is None

    from pipelines.stage_a_wan import _predict_wan_output_hw, _WAN_MAX_AREA_CONFIGS

    max_area = int(_WAN_MAX_AREA_CONFIGS[cfg.stage_a_wan_size])
    assert _predict_wan_output_hw((800, 800), max_area) == (624, 624)


def test_stage_d_default_contract_is_only_a_default():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "pipelines", "stage_d", "config.py"
    )
    spec = importlib.util.spec_from_file_location("stage_d_config_contract", config_path)
    stage_d_config = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = stage_d_config
    spec.loader.exec_module(stage_d_config)

    assert (stage_d_config.H_PIXEL, stage_d_config.W_PIXEL) == (464, 832)
    assert (stage_d_config.H_LATENT, stage_d_config.W_LATENT) == (58, 104)


def test_b1_accepts_dynamic_stage_a_resolution():
    cfg = BootstrapConfig(skip_b1_stage_a=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage_a_dir = os.path.join(tmp, "stage_a")
        os.makedirs(stage_a_dir, exist_ok=True)
        pt_path = os.path.join(stage_a_dir, "wan_video_target_3FHW_uint8.pt")
        torch.save(torch.zeros((3, 21, 624, 624), dtype=torch.uint8), pt_path)

        bundle = _run_b1_stage_a(None, "dummy motion", cfg, tmp)
        assert tuple(bundle.wan_video_target_3FHW.shape) == (3, 21, 624, 624)
        assert tuple(bundle.s_0_clean.shape) == (3, 624, 624)
        assert cfg.stage_a_resolution_hw == (624, 624)
        assert bundle.source_meta["mode"] == "stagea_video"


def test_b1_accepts_six_image_input_dir():
    from PIL import Image

    cfg = BootstrapConfig(skip_b1_stage_a=True)
    cfg.bootstrap_input_mode = "six_images"
    cfg.stage_image_pattern = "state_{i:02d}.png"
    with tempfile.TemporaryDirectory() as tmp:
        cfg.stage_image_dir = tmp
        for i in range(6):
            arr = np.zeros((64, 64, 4), dtype=np.uint8)
            arr[:, :, 0] = i * 30
            arr[:, :, 3] = 255
            Image.fromarray(arr, mode="RGBA").save(
                os.path.join(tmp, f"state_{i:02d}.png")
            )

        bundle = _run_b1_stage_a(None, "dummy motion", cfg, tmp)
        video = bundle.wan_video_target_3FHW
        assert tuple(video.shape) == (3, 21, 64, 64)
        assert int(video[0, 0, 0, 0]) == 0
        assert int(video[0, 4, 0, 0]) == 30
        assert int(video[0, 20, 0, 0]) == 150
        assert tuple(bundle.s_0_clean.shape) == (3, 64, 64)
        assert cfg.stage_a_resolution_hw == (64, 64)
        assert bundle.source_meta["mode"] == "six_images"
        assert bundle.source_meta["constructed_video"] is True


def test_state_images_to_video_tensor_holds_previous_keyframe():
    from PIL import Image

    images = []
    for i in range(6):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        arr[:, :, 1] = i * 20
        arr[:, :, 3] = 255
        images.append(Image.fromarray(arr, mode="RGBA"))
    video = _state_images_to_video_tensor(
        images,
        state_indices=(0, 4, 8, 12, 16, 20),
        frame_num=21,
    )
    assert tuple(video.shape) == (3, 21, 64, 64)
    assert int(video[1, 3, 0, 0]) == 0
    assert int(video[1, 4, 0, 0]) == 20
    assert int(video[1, 19, 0, 0]) == 80
    assert int(video[1, 20, 0, 0]) == 100


def test_voxel_world_roundtrip():
    res = 64
    src = torch.tensor([[0, 0, 0], [32, 32, 32], [63, 63, 63]], dtype=torch.int32)
    w = _voxel_to_world(src, res)
    back = _world_to_voxel(w, res)
    assert torch.equal(back.to(torch.int32), src), f"roundtrip failed: {back}"
    print("[ok] voxel<->world roundtrip")


def test_carpet_detect():
    res = 64
    O = torch.zeros(1, 1, res, res, res)
    # Plant a dense horizontal slab at y=0..2 (the "carpet")
    O[0, 0, :, 0:3, :] = 1.0
    # Plant some object voxels above
    O[0, 0, 20:40, 30:50, 20:40] = 1.0
    mask = _detect_carpet_freeart3d(O, res)
    assert mask.shape == (res ** 3,) and mask.dtype == torch.bool
    # Carpet voxels (y in [0,2]) should be flagged
    mask_3d = mask.view(res, res, res)
    n_carpet = int(mask_3d[:, 0:3, :].sum())
    assert n_carpet > 0, f"no carpet voxels detected, expected dense slab"
    print(f"[ok] carpet detect: {n_carpet} flagged in y<=2")


def test_u_seed():
    res = 64
    K = 6
    # Mock O_init: object in centre region
    O_init = torch.zeros(1, 1, res, res, res)
    O_init[0, 0, 20:40, 20:40, 20:40] = 0.8
    O_init[0, 0, 18:42, 18:42, 18:42] = torch.max(
        O_init[0, 0, 18:42, 18:42, 18:42], torch.full((24, 24, 24), 0.2)
    )

    # Mock z_final + decoder
    class _StubDecoder(torch.nn.Module):
        def forward(self, x):
            # Return logits matching x.shape's batch dim, shape (B, 1, res, res, res)
            B = x.shape[0]
            return torch.full((B, 1, res, res, res), -2.0)  # sigmoid -> low

    z_final = torch.randn(K, 8, 16, 16, 16)
    decoder = _StubDecoder()
    is_carpet = torch.zeros(res ** 3, dtype=torch.bool)
    cfg = BootstrapConfig()
    U_seed = _compute_b5_u_seed(O_init, z_final, is_carpet, decoder, cfg)
    assert U_seed.shape[1] == 3
    assert U_seed.shape[0] > 0, "U_seed empty even with planted object"
    print(f"[ok] U_seed: {U_seed.shape[0]} voxels")


def test_se3_matrices():
    axis = torch.tensor([0.0, 1.0, 0.0])
    origin = torch.tensor([0.0, 0.0, 0.0])
    # Prismatic: translate along axis by 0.3
    T = _se3_prismatic(axis, 0.3)
    assert T.shape == (4, 4)
    pt = torch.tensor([1.0, 1.0, 1.0, 1.0])
    pt2 = T @ pt
    assert torch.allclose(pt2[:3], torch.tensor([1.0, 1.3, 1.0]), atol=1e-5)
    # Revolute: rotate 90 deg around +Y at origin
    T = _se3_revolute(axis, origin, float(np.pi / 2))
    assert T.shape == (4, 4)
    pt = torch.tensor([1.0, 0.0, 0.0, 1.0])
    pt2 = T @ pt
    assert torch.allclose(pt2[:3], torch.tensor([0.0, 0.0, -1.0]), atol=1e-5)
    print("[ok] SE(3) matrices")


def test_swept_corridor_prismatic():
    res = 64
    # U_seed = a small box
    U_seed = torch.tensor(
        [[x, y, z] for x in range(28, 33) for y in range(28, 33) for z in range(28, 33)],
        dtype=torch.int32,
    )
    psi = Psi(
        axis=torch.tensor([1.0, 0.0, 0.0]),
        origin=torch.zeros(3),
        type_logit=10.0,  # prismatic-confident
        theta_limit_raw=0.0,
        disp_limit_raw=0.0,
        delta_u_init=torch.zeros(5),
    )
    phi_0 = torch.linspace(-0.2, 0.2, 6) - 0.0  # symmetric trajectory
    cfg = BootstrapConfig()
    cfg.corridor_n_samples = 10
    swept = _compute_swept_volume_corridor(psi, phi_0, U_seed, cfg)
    # Swept along +X should add voxels with x > 32
    assert swept.shape[0] > U_seed.shape[0], (
        f"swept ({swept.shape[0]}) not larger than U_seed ({U_seed.shape[0]})"
    )
    print(f"[ok] swept corridor (prismatic): U_seed={U_seed.shape[0]} -> swept={swept.shape[0]}")


def test_union_voxel_sets():
    res = 64
    a = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=torch.int32)
    b = torch.tensor([[1, 1, 1], [3, 3, 3]], dtype=torch.int32)
    u = _union_voxel_sets(a, b, res=res)
    assert u.shape[0] == 4, f"expected 4 unique voxels, got {u.shape[0]}"
    print(f"[ok] _union_voxel_sets: dedup -> {u.shape[0]}")


def test_slat_shell_mask():
    res = 64
    U_object = torch.tensor(
        [[10, 10, 10], [20, 20, 20], [30, 30, 30]], dtype=torch.int32
    )
    O_init = torch.zeros(1, 1, res, res, res)
    # Make voxel (20, 20, 20) in boundary band
    O_init[0, 0, 20, 20, 20] = 0.2
    # (10, 10, 10) above band, (30, 30, 30) below band
    O_init[0, 0, 10, 10, 10] = 0.5
    O_init[0, 0, 30, 30, 30] = 0.05
    is_carpet = torch.zeros(res ** 3, dtype=torch.bool)
    cfg = BootstrapConfig()
    shell = _compute_slat_shell_mask(U_object, O_init, is_carpet, cfg)
    assert shell.shape == (3,)
    assert shell.tolist() == [False, True, False], f"got {shell.tolist()}"
    print("[ok] slat_shell_mask")


def test_save_bootstrap_artifacts_smoke():
    res = 64
    K = 6
    psi = Psi(
        axis=torch.tensor([0.0, 0.0, 1.0]),
        origin=torch.zeros(3),
        type_logit=0.0,
        theta_limit_raw=0.5,
        disp_limit_raw=0.5,
        delta_u_init=torch.zeros(5),
    )
    joint_init = JointInit(
        psi=psi,
        phi_0=torch.linspace(-0.4, 0.6, K),
        anchors_object=torch.tensor([[10, 20, 30]], dtype=torch.int32),
        confidence=0.0,
        sub_confidence={"stub": 1.0},
        diagnostics={"stub": True},
    )

    N = 100
    result = BootstrapResult(
        z_s0=torch.randn(1, 8, 16, 16, 16),
        z_final=torch.randn(K, 8, 16, 16, 16),
        O_init=torch.rand(1, 1, res, res, res),
        M_attn_boot_64=torch.rand(res, res, res),
        is_carpet_mask=torch.zeros(res ** 3, dtype=torch.bool),
        U_seed=torch.randint(0, res, (N, 3), dtype=torch.int32),
        joint_init=joint_init,
        U_object=torch.randint(0, res, (N, 3), dtype=torch.int32),
        U_object_with_batch=torch.cat([
            torch.zeros((N, 1), dtype=torch.int32),
            torch.randint(0, res, (N, 3), dtype=torch.int32),
        ], dim=-1),
        z_slat0=None,
        slat_mean=None,
        slat_std=None,
        slat_shell_mask=torch.zeros((N,), dtype=torch.bool),
        gaussian_parent_idx=torch.arange(N).repeat_interleave(32).to(torch.int32),
        wan_cond_cached=None,
        z_wan_target=None,
        trellis_cond_can=torch.randn(1, 1374, 1024),
        wan_video_target_3FHW=torch.zeros((3, 21, 464, 832), dtype=torch.uint8),
        pure_state_targets_K3HW=torch.zeros((K, 3, 464, 832), dtype=torch.float32),
        s_0_clean=torch.zeros((3, 464, 832), dtype=torch.float32),
        s_0_pure=torch.zeros((3, 464, 832), dtype=torch.float32),
        s_5_pure=torch.ones((3, 464, 832), dtype=torch.float32),
        O_base_canonical=torch.zeros((res, res, res), dtype=torch.uint8),
        O_move_per_state=torch.zeros((K, res, res, res), dtype=torch.uint8),
        P_base_canonical=torch.zeros((res, res, res), dtype=torch.float32),
        P_move_evidence_per_state=torch.zeros((K, res, res, res), dtype=torch.float32),
        M_motion_corridor_64=None,
        dit_hidden_cache=None,
        meta={"K": K, "stage_a_skipped": True, "n_voxels": {"U_object": N}},
    )
    cfg = BootstrapConfig()
    with tempfile.TemporaryDirectory() as tmp:
        saved = _save_bootstrap_artifacts(result, tmp, cfg)
        assert len(saved) > 5, f"too few artifacts saved: {len(saved)}"
        # Check key artifacts exist
        boot_dir = os.path.join(tmp, "bootstrap")
        for name in ["z_s0.pt", "z_final.pt", "O_init.npy", "psi_0.json",
                     "U_object.npy", "phi_0.npy", "pure_state_targets_K3HW.pt",
                     "s_0_pure.pt", "s_5_pure.pt",
                     "bootstrap_meta.json"]:
            p = os.path.join(boot_dir, name)
            assert os.path.isfile(p), f"missing {name}"
    print(f"[ok] _save_bootstrap_artifacts wrote {len(saved)} files")


def test_stage_c_stub_integration():
    """Make sure Bootstrap can call Stage C with v3.3.6-rich inputs."""
    K = 6
    res = 64
    inputs = StageCInputs(
        z_final=torch.randn(K, 8, 16, 16, 16),
        M_attn_boot_64=torch.rand(res, res, res),
        O_init=torch.rand(1, 1, res, res, res),
        is_carpet_mask=torch.zeros(res ** 3, dtype=torch.bool),
        U_seed=torch.randint(0, res, (200, 3), dtype=torch.int32),
        O_base_canonical=torch.zeros((res, res, res), dtype=torch.uint8),
        O_move_per_state=torch.zeros((K, res, res, res), dtype=torch.uint8),
        P_base_canonical=torch.zeros((res, res, res), dtype=torch.float32),
        P_move_evidence_per_state=torch.zeros((K, res, res, res), dtype=torch.float32),
        M_motion_corridor_64=torch.zeros((res, res, res), dtype=torch.float32),
    )
    cfg = StageCConfig()
    out = run_stage_c_joint_init(inputs, cfg)
    assert out.phi_0.shape == (K,)
    assert torch.allclose(out.phi_0[cfg.canonical_state_idx], torch.tensor(0.0))
    print(f"[ok] Stage C stub via Bootstrap inputs: joint_type={out.joint_type()}")


def main():
    print("=" * 60)
    print("Bootstrap unit tests (pure-Python, no GPU/checkpoints)")
    print("=" * 60)
    test_voxel_world_roundtrip()
    test_carpet_detect()
    test_u_seed()
    test_se3_matrices()
    test_swept_corridor_prismatic()
    test_union_voxel_sets()
    test_slat_shell_mask()
    test_save_bootstrap_artifacts_smoke()
    test_stage_c_stub_integration()
    print()
    print("ALL Bootstrap unit tests pass.")


if __name__ == "__main__":
    main()

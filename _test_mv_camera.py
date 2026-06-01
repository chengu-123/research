"""Standalone CPU unit test for StageD multi-view camera sampling (Task 1).

Pure camera math: no GPU / Wan / TRELLIS forward needed. Verifies

  1. ``freeart3d_sampled(22.5, 45.0, ...)`` reproduces
     ``freeart3d_canonical(...)`` exactly (pos within 1e-6; up/fov/near/far/
     image/bg equal).
  2. ``sample_camera_for_iter`` draws ~half canonical (mv_canonical_ratio=0.5)
     and keeps random-view azimuth/elevation inside the configured ranges.

Run:
    source /lustre/1230003454/env/mine1/bin/activate
    cd /lustre/1230003454/current/mine
    PYTHONPATH=/lustre/1230003454/current/mine:/lustre/1230003454/current/mine/TRELLIS \
        python _test_mv_camera.py
"""
from __future__ import annotations

import torch

from pipelines.stage_d.config import StageDConfig
from pipelines.stage_d.render import (
    StageDCameraConfig,
    sample_camera_for_iter,
)


def _assert_close(a: float, b: float, tol: float, msg: str) -> None:
    if abs(float(a) - float(b)) > tol:
        raise AssertionError(f"{msg}: |{a} - {b}| = {abs(a - b)} > {tol}")


def test_sampled_equals_canonical() -> None:
    H, W = 512, 512
    scale = 0.49
    center = (0.0, 0.03, 0.02)

    canon = StageDCameraConfig.freeart3d_canonical(
        H, W, object_scale=scale, object_center=center,
    )
    sampled = StageDCameraConfig.freeart3d_sampled(
        22.5, 45.0, H, W, scale, center,
    )

    # pos componentwise within 1e-6
    for i, name in enumerate("xyz"):
        _assert_close(
            sampled.pos[i], canon.pos[i], 1.0e-6,
            f"pos.{name} mismatch",
        )

    # up / fov / near / far / image / bg / look_at exact equality
    assert sampled.up == canon.up, f"up mismatch: {sampled.up} != {canon.up}"
    assert sampled.look_at == canon.look_at, (
        f"look_at mismatch: {sampled.look_at} != {canon.look_at}"
    )
    assert sampled.fov_x_deg == canon.fov_x_deg, "fov_x mismatch"
    assert sampled.fov_y_deg == canon.fov_y_deg, "fov_y mismatch"
    assert sampled.near == canon.near, "near mismatch"
    assert sampled.far == canon.far, "far mismatch"
    assert sampled.image_h == canon.image_h, "image_h mismatch"
    assert sampled.image_w == canon.image_w, "image_w mismatch"
    assert sampled.bg_color == canon.bg_color, "bg_color mismatch"

    print(
        "[1] sampled(22.5,45) == canonical: PASS  "
        f"(pos={tuple(round(v, 9) for v in sampled.pos)})"
    )


def test_iter_sampler_distribution() -> None:
    H, W = 512, 512
    scale = 0.49
    center = (0.0, 0.03, 0.02)

    cfg = StageDConfig(
        mv_enable=True,
        mv_canonical_ratio=0.5,
        mv_azi_jitter_deg=30.0,
        mv_ele_min_deg=30.0,
        mv_ele_max_deg=60.0,
        mv_seed=0,
    )

    canonical_cfg = StageDCameraConfig.freeart3d_canonical(
        H, W, object_scale=scale, object_center=center,
    )

    rng = torch.Generator(device="cpu")
    rng.manual_seed(cfg.mv_seed)

    n = 200
    n_canon = 0
    azi_lo, azi_hi = 22.5 - cfg.mv_azi_jitter_deg, 22.5 + cfg.mv_azi_jitter_deg
    ele_lo, ele_hi = cfg.mv_ele_min_deg, cfg.mv_ele_max_deg

    for _ in range(n):
        cam, is_canon = sample_camera_for_iter(
            rng, cfg, center, scale, H, W, canonical_cfg,
        )
        if is_canon:
            n_canon += 1
            # canonical draws must return the exact pre-built object
            assert cam is canonical_cfg, "canonical draw must return canonical_cfg"
            continue
        # Random view: recover azi/ele from the position and bound-check.
        cx, cy, cz = center
        d = 2.1 * scale
        dx = cam.pos[0] - cx
        dy = cam.pos[1] - cy
        dz = cam.pos[2] - cz
        # x = cx + sin(ar)*d ; y = cy - cos(ar)*d -> ar = atan2(dx, -dy)
        azi_rad = torch.atan2(torch.tensor(dx), torch.tensor(-dy)).item()
        azi_deg = azi_rad * 180.0 / torch.pi
        # z = cz + sin(el)*d -> el = asin(dz/d)
        ele_rad = torch.asin(torch.tensor(dz / d).clamp(-1.0, 1.0)).item()
        ele_deg = ele_rad * 180.0 / torch.pi

        assert azi_lo - 1.0e-4 <= azi_deg <= azi_hi + 1.0e-4, (
            f"azi {azi_deg} out of [{azi_lo}, {azi_hi}]"
        )
        assert ele_lo - 1.0e-4 <= ele_deg <= ele_hi + 1.0e-4, (
            f"ele {ele_deg} out of [{ele_lo}, {ele_hi}]"
        )
        # random draws must NOT alias the canonical object
        assert cam is not canonical_cfg, "random draw aliased canonical_cfg"

    frac = n_canon / n
    # ~half canonical; allow a generous band for 200 Bernoulli(0.5) draws.
    assert 0.35 <= frac <= 0.65, f"canonical fraction {frac} not ~0.5"

    print(
        f"[2] iter sampler: {n_canon}/{n} canonical (frac={frac:.3f}); "
        f"random azi in [{azi_lo},{azi_hi}], ele in [{ele_lo},{ele_hi}]: PASS"
    )


if __name__ == "__main__":
    test_sampled_equals_canonical()
    test_iter_sampler_distribution()
    print("PASS")

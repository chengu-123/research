"""Verify Stage C driver returns valid JointInit shapes + sane data-driven
output on a planted prismatic-drawer pattern.

Standalone script, no pytest. Run with the project's `mine` conda env:
    conda activate mine && python tests/test_stage_c_stub_shapes.py
"""
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.stage_c import StageCConfig, StageCInputs, run_stage_c_joint_init


def _plant_prismatic_drawer(K: int, res: int, axis_dim: int = 0):
    """Create planted (O_base_canonical, O_move_per_state, P_*, M_motion_corridor_64).

    Per state k, drawer occupies a 6x6x6 patch centred at:
        centre[axis_dim] = 16 + 4*k    # slides along axis
        centre[other_dims] = 32        # fixed
    Base is a 16x10x16 slab at low axis_dim values (the cabinet body).
    """
    O_move = torch.zeros(K, res, res, res, dtype=torch.uint8)
    P_move = torch.zeros(K, res, res, res, dtype=torch.float32)
    O_base = torch.zeros(res, res, res, dtype=torch.uint8)
    P_base = torch.zeros(res, res, res, dtype=torch.float32)
    M_corr = torch.zeros(res, res, res, dtype=torch.float32)

    # Cabinet base: a 8x40x40 slab in axis_dim=[8, 15]
    if axis_dim == 0:
        O_base[8:16, 12:52, 12:52] = 1
        P_base[8:16, 12:52, 12:52] = 1.0
    else:
        # For non-x axis we'd need different slabs; not used in main test.
        pass

    for k in range(K):
        cx = 16 + 5 * k  # x slides 16..36 over K=6 (matches 5-step drawer trajectory)
        cy = 32
        cz = 32
        x_lo, x_hi = max(0, cx - 3), min(res, cx + 4)
        y_lo, y_hi = cy - 3, cy + 4
        z_lo, z_hi = cz - 3, cz + 4
        O_move[k, x_lo:x_hi, y_lo:y_hi, z_lo:z_hi] = 1
        P_move[k, x_lo:x_hi, y_lo:y_hi, z_lo:z_hi] = 0.8

    # Motion corridor: union of all per-state drawer positions
    footprint = (O_move.max(dim=0).values > 0).float()
    shared = O_move.float().mean(dim=0)
    M_corr = footprint * (1.0 - shared)

    return O_base, O_move, P_base, P_move, M_corr


def main() -> int:
    K = 6
    res = 64

    # Planted prismatic drawer along X axis
    O_base, O_move, P_base, P_move, M_corr = _plant_prismatic_drawer(K, res, axis_dim=0)

    inputs = StageCInputs(
        z_final=torch.randn(K, 8, 16, 16, 16),
        M_attn_boot_64=torch.rand(res, res, res),
        O_init=torch.rand(1, 1, res, res, res),
        is_carpet_mask=torch.zeros(res * res * res, dtype=torch.bool),
        U_seed=torch.randint(0, res, (500, 3), dtype=torch.int32),
        O_base_canonical=O_base,
        O_move_per_state=O_move,
        P_base_canonical=P_base,
        P_move_evidence_per_state=P_move,
        M_motion_corridor_64=M_corr,
    )

    cfg = StageCConfig()
    result = run_stage_c_joint_init(inputs, cfg)

    # ---- Shape contract assertions ----
    assert result.psi.axis.shape == (3,)
    assert result.psi.origin.shape == (3,)
    assert isinstance(result.psi.type_logit, float)
    assert result.psi.delta_u_init.shape == (5,)
    assert result.phi_0.shape == (K,)
    assert torch.allclose(result.phi_0[cfg.canonical_state_idx], torch.tensor(0.0))
    assert result.anchors_object.dim() == 2 and result.anchors_object.shape[1] == 3
    assert result.anchors_object.dtype == torch.int32
    assert result.psi.pack_19().shape == (19,)
    assert 0.0 <= result.confidence <= 1.0
    assert result.diagnostics["stage_c_version"] == "v4_branched_voxel_physics"

    # ---- Data-driven sanity (prismatic along +X) ----
    assert result.joint_type() == "prismatic", (
        f"Planted prismatic along X; got joint_type={result.joint_type()}"
    )
    axis_x = float(result.psi.axis[0].abs())
    axis_yz = float((result.psi.axis[1] ** 2 + result.psi.axis[2] ** 2).sqrt())
    assert axis_x > 0.85, (
        f"Axis should align with X (|axis_x|>0.85); got axis={result.psi.axis.tolist()}"
    )
    assert axis_yz < 0.5, (
        f"Axis Y/Z components should be small; got axis_y_z_mag={axis_yz}"
    )
    assert result.anchors_object.shape[0] >= 4, (
        f"Should extract at least 4 anchors from planted corridor; got "
        f"{result.anchors_object.shape[0]}"
    )

    print("OK Stage C production driver:")
    print(f"  joint_type            = {result.joint_type()}")
    print(f"  psi.axis              = {[round(float(x), 3) for x in result.psi.axis]}")
    print(f"  psi.origin            = {[round(float(x), 3) for x in result.psi.origin]}")
    print(f"  psi.type_logit        = {result.psi.type_logit:.4f}")
    print(f"  psi.theta_limit_raw   = {result.psi.theta_limit_raw:.4f}")
    print(f"  psi.disp_limit_raw    = {result.psi.disp_limit_raw:.4f}")
    print(f"  psi.delta_u_init      = {[round(float(x), 3) for x in result.psi.delta_u_init]}")
    print(f"  phi_0                 = {[round(float(x), 3) for x in result.phi_0]}")
    print(f"  phi_0[c=2]            = {float(result.phi_0[cfg.canonical_state_idx]):.4f}")
    print(f"  anchors_object.n      = {int(result.anchors_object.shape[0])}")
    print(f"  confidence            = {result.confidence:.3f}")
    print(f"  sub_confidence        = {result.sub_confidence}")
    print(f"  diagnostics.axis_fit_source = {result.diagnostics['axis_fit_source']}")
    print(f"  diagnostics.axis_fit_residual = {result.diagnostics['axis_fit_residual']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

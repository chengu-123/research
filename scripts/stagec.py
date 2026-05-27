"""Run Stage C joint init on a pre-existing stage_b/ output directory.

Bootstrap-free: only needs the artifacts that pipelines/stage_b_scar.run_scar
persists. No TRELLIS / Wan / GPU dependency — Stage C is pure geometry on
numpy/torch CPU.

Usage:
    conda activate mine
    python scripts/run_stage_c_from_stageb.py \\
        --stage_b_dir outputs/30857/stageb_v1/stage_b \\
        --out_dir outputs/30857/stage_c_test

What it does:
    1. Loads required v3.3.6 artifacts from stage_b_dir:
        O_base_canonical.npy, O_move_per_state.npy,
        P_base_canonical.npy, P_move_evidence_per_state.npy,
        viz/bmcsa/M_motion_corridor_64.npy,
        z_final.pt (only for K = shape[0])
    2. Fills the unused spec inputs (M_attn_boot_64, O_init, is_carpet_mask,
       U_seed) with sane placeholders.
    3. Calls run_stage_c_joint_init -> writes stage_c_joint_init.json + viz/
       under out_dir.
    4. Prints the joint init summary.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.stage_c import StageCConfig, StageCInputs, run_stage_c_joint_init


def _load_npy(p: str, name: str, required: bool = True):
    full = os.path.join(p, name)
    if not os.path.isfile(full):
        if required:
            raise FileNotFoundError(
                f"Missing required Stage B v3.3.6 artifact: {full}\n"
                f"Stage C requires Stage B v3.3.6+ outputs (O_base_canonical, "
                f"O_move_per_state, P_base_canonical, P_move_evidence_per_state, "
                f"viz/bmcsa/M_motion_corridor_64)."
            )
        return None
    return np.load(full)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage_b_dir", required=True,
        help="Path to a pre-existing stage_b/ output directory.",
    )
    p.add_argument(
        "--out_dir", required=True,
        help="Where to write stage_c_joint_init.json + viz/.",
    )
    p.add_argument(
        "--device", default="cpu",
        help="cpu or cuda (default cpu — Stage C is pure numpy/torch, no model).",
    )
    p.add_argument(
        "--no_viz", action="store_true",
        help="Skip writing viz HTMLs (saves ~5 sec).",
    )
    args = p.parse_args()

    sb = args.stage_b_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---- Load v3.3.6 enriched signals (REQUIRED) ----
    O_base = _load_npy(sb, "O_base_canonical.npy", required=True)
    O_move = _load_npy(sb, "O_move_per_state.npy", required=True)
    P_base = _load_npy(sb, "P_base_canonical.npy", required=True)
    P_move = _load_npy(sb, "P_move_evidence_per_state.npy", required=True)
    M_corr_path = os.path.join(sb, "viz", "bmcsa", "M_motion_corridor_64.npy")
    if not os.path.isfile(M_corr_path):
        raise FileNotFoundError(
            f"Missing {M_corr_path}. Stage B must run with use_motion_corridor=true."
        )
    M_corr = np.load(M_corr_path)

    # ---- z_final.pt: needed only for K = shape[0]; values not consumed by Stage C ----
    z_final_path = os.path.join(sb, "z_final.pt")
    if not os.path.isfile(z_final_path):
        raise FileNotFoundError(f"Missing {z_final_path}")
    # z_final.pt is our own checkpoint (saved by stage_b_scar.run_scar), so
    # weights_only=False is safe + explicit silences the PyTorch 2.4 FutureWarning.
    z_final = torch.load(z_final_path, map_location=device, weights_only=False)
    K = int(z_final.shape[0])
    res = int(O_base.shape[-1])
    print(f"[run_stage_c_from_stageb] loaded stage_b/ artifacts: K={K}, res={res}")

    # ---- Build StageCInputs (spec placeholders + v3.3.6 real data) ----
    inputs = StageCInputs(
        # Spec inputs — algorithm doesn't actually use these, placeholders OK:
        z_final=z_final.to(device),
        M_attn_boot_64=torch.zeros((res, res, res), device=device, dtype=torch.float32),
        O_init=torch.zeros((1, 1, res, res, res), device=device, dtype=torch.float32),
        is_carpet_mask=torch.zeros(res ** 3, dtype=torch.bool, device=device),
        U_seed=torch.zeros((0, 3), dtype=torch.int32, device=device),
        # v3.3.6 enriched signals — algorithm reads these:
        O_base_canonical=torch.from_numpy(O_base).to(device),
        O_move_per_state=torch.from_numpy(O_move).to(device),
        P_base_canonical=torch.from_numpy(P_base).to(device).float(),
        P_move_evidence_per_state=torch.from_numpy(P_move).to(device).float(),
        M_motion_corridor_64=torch.from_numpy(M_corr).to(device).float(),
        dit_hidden=None,
    )

    cfg = StageCConfig()
    cfg.save_viz = not args.no_viz
    cfg.save_diagnostics_json = True

    print("[run_stage_c_from_stageb] running Stage C joint init...")
    result = run_stage_c_joint_init(inputs, cfg, out_dir=out_dir)

    print()
    print("=" * 60)
    print(f"Stage C joint init result (out_dir={out_dir})")
    print("=" * 60)
    print(f"  joint_type            = {result.joint_type()}")
    print(f"  confidence (overall)  = {result.confidence:.4f}")
    print(f"  sub_confidence        = {result.sub_confidence}")
    print(f"  psi.axis              = {[round(float(x), 4) for x in result.psi.axis]}")
    print(f"  psi.origin            = {[round(float(x), 4) for x in result.psi.origin]}")
    print(f"  psi.type_logit        = {result.psi.type_logit:.4f}")
    print(f"  phi_0 (c-shifted)     = {[round(float(x), 4) for x in result.phi_0]}")
    print(f"  phi_0[c=2]            = {float(result.phi_0[cfg.canonical_state_idx]):.4f}  (must be 0)")
    print(f"  anchors_object.count  = {int(result.anchors_object.shape[0])}")
    print(f"  axis_fit_source       = {result.diagnostics['axis_fit_source']}")
    print(f"  axis_fit_residual     = {result.diagnostics['axis_fit_residual']:.6f}")
    print()
    print(f"Artifacts written to {out_dir}/:")
    print(f"  - stage_c_joint_init.json    (full diagnostics + Psi)")
    if cfg.save_viz:
        print(f"  - viz/summary.html           (text summary, no plotly needed)")
        print(f"  - viz/joint_overview_3d.html (3D scatter + axis + anchors)")
        print(f"  - viz/phi_progression.html   (u_raw / u_norm / phi_0 / delta_u)")
        print(f"  - viz/type_fit_diagnostics.html (residuals + sub-conf)")
        print(f"  - viz/M_motion_corridor_64.html")
        print(f"  - viz/anchors_overlay.html")
        print(f"  - viz/axis_overlay.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())

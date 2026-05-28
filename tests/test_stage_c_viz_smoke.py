"""Smoke test for pipelines/stage_c/viz.py.

Exercises save_stage_c_viz on planted prismatic-drawer data + verifies:
  - summary.html is always written (plotly-independent)
  - plotly HTMLs are written when plotly is available
  - viz module compiles cleanly

Run with:
    conda activate mine && python tests/test_stage_c_viz_smoke.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.stage_c import StageCConfig, StageCInputs, run_stage_c_joint_init
from pipelines.stage_c.viz import save_summary_html


def _plant_prismatic(K: int, res: int) -> tuple:
    """Reuse the planted prismatic drawer from test_stage_c_stub_shapes."""
    O_move = torch.zeros(K, res, res, res, dtype=torch.uint8)
    P_move = torch.zeros(K, res, res, res, dtype=torch.float32)
    O_base = torch.zeros(res, res, res, dtype=torch.uint8)
    P_base = torch.zeros(res, res, res, dtype=torch.float32)

    O_base[8:16, 12:52, 12:52] = 1
    P_base[8:16, 12:52, 12:52] = 1.0

    for k in range(K):
        cx = 16 + 5 * k
        x_lo, x_hi = max(0, cx - 3), min(res, cx + 4)
        O_move[k, x_lo:x_hi, 29:36, 29:36] = 1
        P_move[k, x_lo:x_hi, 29:36, 29:36] = 0.8

    footprint = (O_move.max(dim=0).values > 0).float()
    shared = O_move.float().mean(dim=0)
    M_corr = footprint * (1.0 - shared)
    return O_base, O_move, P_base, P_move, M_corr


def main() -> int:
    K = 6
    res = 64
    O_base, O_move, P_base, P_move, M_corr = _plant_prismatic(K, res)

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

    with tempfile.TemporaryDirectory() as tmp:
        # Drive Stage C end-to-end; this will internally call save_stage_c_viz.
        result = run_stage_c_joint_init(inputs, cfg, out_dir=tmp)
        viz_dir = os.path.join(tmp, "viz")

        # summary.html must always exist (plotly-independent path)
        summary_path = os.path.join(viz_dir, "summary.html")
        assert os.path.isfile(summary_path), f"missing {summary_path}"
        with open(summary_path, encoding="utf-8") as f:
            content = f.read()
        # Spot-check that summary contains expected fields
        assert "joint_type" in content
        assert "psi.axis" in content
        assert "phi_0" in content
        assert "anchors count" in content

        text_path = os.path.join(viz_dir, "v3_summary.txt")
        assert os.path.isfile(text_path), f"missing {text_path}"

        plotly_files = [
            "joint_overview_3d.html",
            "phi_progression.html",
            "type_fit_diagnostics.html",
            "M_motion_corridor_64.html",
            "anchors_overlay.html",
            "axis_overlay.html",
        ]
        try:
            import plotly  # noqa: F401
            plotly_ok = True
        except ImportError:
            plotly_ok = False

        if plotly_ok:
            for fn in plotly_files:
                p = os.path.join(viz_dir, fn)
                if not os.path.isfile(p):
                    raise AssertionError(
                        f"plotly available but missing viz {fn} (path={p})"
                    )
            print(f"OK Stage C viz wrote all {len(plotly_files) + 2} diagnostics files:")
            print("  - summary.html")
            print("  - v3_summary.txt")
            for fn in plotly_files:
                print(f"  - {fn}")
        else:
            print("OK Stage C viz wrote summary.html and v3_summary.txt")

        # Stand-alone summary writer test (works even without driver)
        with tempfile.TemporaryDirectory() as tmp2:
            standalone = os.path.join(tmp2, "stand_summary.html")

            class _FakeTR:
                type_str = "prismatic"
                type_logit = 1.2
                residual_line = 0.01
                residual_arc = float("inf")
                n_valid_states = 6

            class _FakeAR:
                fit_source = "centroid_line"
                fit_residual = 0.0009

            class _FakePR:
                monotone_enforced = False
                observed_max_angle = 0.0
                observed_max_disp = 0.25

            save_summary_html(
                standalone,
                joint_init=result,
                type_result=_FakeTR(),
                axis_result=_FakeAR(),
                phi_result=_FakePR(),
                title="standalone summary",
            )
            assert os.path.isfile(standalone)
            print("OK save_summary_html standalone call also works")

    return 0


if __name__ == "__main__":
    sys.exit(main())

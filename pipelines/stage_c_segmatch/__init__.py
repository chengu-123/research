"""Stage C SegMatch v5 package.

Training-free K-state articulated motion discovery via
correspondence-free whole-shape volumetric rigid alignment, with
cross-stage M_attn semantic prior (from Stage B v4.3). See
`record/stageC/2026-04-22-segmatch-v3-spec.md` and the 2026-04-23 v5
addendum for the full design.

Pipeline (run_stage_c.run_stage_c):
  C.1 partition            pipelines.sajo.anchors.joint_free_split (bug-fixed)
  C.2 moments              centroid-trajectory warm start
  C.3 icp_warmstart        optional open3d ICP (ablation)
  C.4 volumetric_fit       joint-constrained Adam on variance loss (CORE)
  C.5 seg_refine           motion-consistency graph-cut
  [iterate C.4 ↔ C.5]
  C.6 axis_refine          contact principal axis
  C.7 overlap_cleanup      canonical base-base overlap resolution
  C.8 aggregation          median backwarp + per-state assignment
"""

from __future__ import annotations

from .config import Diagnostics, SegMatchHParams, StageCResult

# v8: do NOT eagerly import run_stage_c — doing so triggers a double-import
# when the module is invoked via `python -m pipelines.stage_c_segmatch.run_stage_c`
# (runpy warning: "found in sys.modules after import of package ... prior to
# execution"), which subtly perturbs Adam initialization and flips the rev/pris
# BIC margin from 0.334 (correct) to 0.027 (wrong) on 7201. Import the
# function lazily via ``from pipelines.stage_c_segmatch.run_stage_c import
# run_stage_c`` instead.

__all__ = [
    "Diagnostics",
    "SegMatchHParams",
    "StageCResult",
]

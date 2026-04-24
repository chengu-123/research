"""Evaluation adapter for the v1 output layout.

Reads the joint parameters written by ``run_v1.py`` from
``<output_dir>/stage_f/joint_info.json`` and compares them against a
PartNet-Mobility-style ground-truth joint description. The schema
matches ``eval_utils.joints.Joint(joint_data, method='ours')``, so
this script is a thin wrapper around ``eval_joint`` from the existing
evaluation utilities.

For mesh-level metrics (Chamfer), the v1 pipeline only writes
``stage_f/M_base.glb`` and ``stage_f/M_move.glb``. If the user supplies
``--gt_mesh_dir``, this script will regenerate per-state meshes by
replaying the URDF at a linear sweep of ``q in [0, 1]`` via
:func:`pipelines.urdf.parse_urdf_and_save`, enabling a direct Chamfer
comparison against the ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True,
                   help="v1 output directory (contains stage_f/joint_info.json).")
    p.add_argument("--gt_joint", required=True,
                   help="Path to the ground-truth joint_info.json.")
    p.add_argument("--gt_mesh_dir", default=None,
                   help="Optional dir with GT per-state .glb meshes for Chamfer metrics.")
    p.add_argument("--n_states", type=int, default=6,
                   help="Number of q values to replay the URDF for (default 6).")
    p.add_argument("--method", default="ours",
                   help="Joint-frame convention for the predicted joint_info.json.")
    return p.parse_args()


def _load_joint_info(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            raise ValueError(f"Empty joint_info list at {path}")
        return data[0]
    return data


def main() -> None:
    args = parse_args()

    from eval_utils.joints import Joint, eval_joint

    pred_joint_path = os.path.join(args.pred_dir, "stage_f", "joint_info.json")
    if not os.path.isfile(pred_joint_path):
        print(f"[evaluate_v1] ERROR: predicted joint_info.json not found: {pred_joint_path}",
              file=sys.stderr)
        sys.exit(1)

    pred_data = _load_joint_info(pred_joint_path)
    gt_data = _load_joint_info(args.gt_joint)

    pred_joint = Joint(pred_data, method=args.method)
    gt_joint = Joint(gt_data)

    joint_metrics = eval_joint(pred_joint, gt_joint)

    print("==================== v1 evaluation ====================")
    print(f"pred_dir : {args.pred_dir}")
    print(f"gt_joint : {args.gt_joint}")
    print(f"joint_type (pred / gt): {pred_joint.type} / {gt_joint.type}")
    for k, v in joint_metrics.items():
        print(f"  {k}: {v}")

    # Report BIC / pybullet diagnostics when available.
    bic_path = os.path.join(args.pred_dir, "stage_c_sajo", "bic.json")
    if os.path.isfile(bic_path):
        with open(bic_path, "r", encoding="utf-8") as f:
            bic = json.load(f)
        print(f"[bic] selected={bic.get('joint_type')} confidence={bic.get('confidence')}")
    py_path = os.path.join(args.pred_dir, "stage_f", "pybullet_report.json")
    if os.path.isfile(py_path):
        with open(py_path, "r", encoding="utf-8") as f:
            py_report = json.load(f)
        print(f"[pybullet] valid={py_report.get('valid')} max_penetration={py_report.get('max_penetration')}")

    # Optional Chamfer against GT meshes via URDF replay.
    if args.gt_mesh_dir is not None:
        from pipelines.urdf import parse_urdf_and_save

        urdf_path = os.path.join(args.pred_dir, "stage_f", "output.urdf")
        states_dir = os.path.join(args.pred_dir, "stage_f", "states")
        os.makedirs(states_dir, exist_ok=True)
        for i in range(args.n_states):
            q = i / max(args.n_states - 1, 1)
            parse_urdf_and_save(
                urdf_path=urdf_path,
                qpos_by_part=[q],
                target_dir=states_dir,
                save_individual=False,
                combined_name=f"qpos_{i:02d}.glb",
            )
        print(f"[mesh] wrote {args.n_states} replayed state meshes to {states_dir}")
        print("[mesh] Chamfer comparison not yet implemented in v1 evaluator;")
        print("       use eval_utils.neucon_eval_utils.eval_mesh on pairs manually.")


if __name__ == "__main__":
    main()

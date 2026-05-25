"""Stage F driver: URDF assembly + pybullet self-intersection validation.

Takes the per-part meshes from :mod:`pipelines.mesh_split` plus the
joint parameters from :class:`pipelines.stage_c_sajo.SAJOResult`, emits
``M_base.glb``, ``M_move.glb``, ``joint_info.json``, ``output.urdf``,
and a ``pybullet_report.json`` with the self-intersection sweep result.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from pipelines.urdf import validate_urdf, write_urdf
from pipelines.utils.voxel_io import compose_joint_info_json, save_json


@dataclass
class StageFResult:
    base_mesh_path: str
    move_mesh_path: str
    urdf_path: str
    joint_info_path: str
    pybullet_report_path: str
    pybullet_report: Dict[str, Any]


def run_stage_f(
    base_mesh: "trimesh.Trimesh",
    move_mesh: "trimesh.Trimesh",
    sajo_result: Any,
    out_dir: str,
    cfg_py: Any,
    robot_name: str = "v1_obj",
) -> StageFResult:
    """Assemble the final URDF and run pybullet validation.

    Parameters
    ----------
    base_mesh, move_mesh : trimesh.Trimesh
        Disjoint per-part meshes in the TRELLIS canonical frame.
    sajo_result : SAJOResult
        Output of :func:`pipelines.stage_c_sajo.run_sajo`.
    out_dir : str
        Destination directory.
    cfg_py : OmegaConf
        The ``pybullet:`` subtree of ``configs/v1.yaml``.
    robot_name : str, default "v1_obj"
        Robot name for the URDF.

    Returns
    -------
    StageFResult
    """
    os.makedirs(out_dir, exist_ok=True)

    base_mesh_path = os.path.join(out_dir, "M_base.glb")
    move_mesh_path = os.path.join(out_dir, "M_move.glb")
    base_mesh.export(base_mesh_path)
    move_mesh.export(move_mesh_path)

    # For prismatic joints, set the URDF origin to the split base-mesh
    # centroid rather than SAJO's zero-origin placeholder. Revolute
    # joints use the SAJO-estimated rotation origin as-is.
    joint_type = str(sajo_result.joint_type)
    if joint_type == "prismatic":
        centroid = np.asarray(base_mesh.centroid, dtype=np.float64).tolist()
        sajo_result_for_json = dict(
            joint_type=joint_type,
            omega=sajo_result.v.detach().cpu().tolist(),
            q=centroid,
            phi_k=sajo_result.phi_k.detach().cpu().tolist(),
        )
    else:
        sajo_result_for_json = sajo_result

    joint_info_path = os.path.join(out_dir, "joint_info.json")
    compose_joint_info_json(sajo_result_for_json, joint_info_path)

    urdf_path = os.path.join(out_dir, "output.urdf")
    write_urdf(
        base_mesh=base_mesh_path,
        part_meshes=[move_mesh_path],
        joint_config=joint_info_path,
        urdf_path=urdf_path,
        robot_name=robot_name,
    )

    # pybullet validation pass.
    report = validate_urdf(
        urdf_path=urdf_path,
        n_samples=int(cfg_py.n_samples),
        epsilon=float(cfg_py.epsilon),
    )
    pybullet_report_path = os.path.join(out_dir, "pybullet_report.json")
    save_json(report, pybullet_report_path)

    return StageFResult(
        base_mesh_path=base_mesh_path,
        move_mesh_path=move_mesh_path,
        urdf_path=urdf_path,
        joint_info_path=joint_info_path,
        pybullet_report_path=pybullet_report_path,
        pybullet_report=report,
    )

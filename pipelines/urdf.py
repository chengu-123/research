import os
import json 
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from typing import Dict, List, Sequence, Union
from trimesh.transformations import rotation_matrix, translation_matrix

def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print XML in-place."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def _safe_normalize(v: Sequence[float]) -> List[float]:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return [0.0, 0.0, 1.0]
    return list(v / n)

def write_urdf(
    base_mesh: str,
    part_meshes: Sequence[str],
    joint_config: str,
    urdf_path: str,
    robot_name: str = "object",
) -> None:
    """
    Create a minimal URDF with a fixed base link and one jointed link per part.

    Parameters
    ----------
    base_mesh : str
        Path to base mesh (e.g., 'fixed.glb').
    part_meshes : Sequence[str]
        Paths to articulated part meshes: ['articulated_00.glb', 'articulated_01.glb', ...]
        The basename (without extension) will be used as the link/joint name suffix.
    joint_configs : Sequence[str]
        Paths to JSON files with fields:
            {
              "joint_type": "prismatic" | "revolute",
              "joint_axis": [x, y, z],
              "joint_position": [px, py, pz],
              "transform_scale": float
            }
        The sign and magnitude of transform_scale define the joint limit range:
            lower = min(0, transform_scale)
            upper = max(0, transform_scale)
        For prismatic: meters along axis. For revolute: radians about axis.
    urdf_path : str
        Output URDF file path.
    robot_name : str
        URDF robot name.
    """

    robot = ET.Element("robot", attrib={"name": robot_name})

    # Base link
    base_link_name = "base_link"
    link_base = ET.SubElement(robot, "link", attrib={"name": base_link_name})
    # Attach a visual to the base so exporters/viewers show it
    vis_base = ET.SubElement(link_base, "visual")
    geom_base = ET.SubElement(vis_base, "geometry")
    ET.SubElement(geom_base, "mesh", attrib={"filename": os.path.relpath(base_mesh, os.path.dirname(urdf_path))})
    col_base = ET.SubElement(link_base, "collision")
    geomc_base = ET.SubElement(col_base, "geometry")
    ET.SubElement(geomc_base, "mesh", attrib={"filename": os.path.relpath(base_mesh, os.path.dirname(urdf_path))})

    with open(joint_config, "r") as f:
        joint_config = json.load(f)

    # One link+joint per part
    for mesh_path, cfg in zip(part_meshes, joint_config):

        jtype = cfg["type"].lower()
        if jtype not in ("revolute", "prismatic"):
            raise ValueError(f"Unsupported joint_type: {jtype}")

        axis_cfg = cfg["axis"]

        # Axis / origin are emitted in TRELLIS voxel-world frame (Y-up);
        # transform to pybullet Z-up: (x, y, z) -> (x, z, -y).
        axis = list(axis_cfg.get("direction", [0.0, 0.0, 1.0]))
        axis[1], axis[2] = axis[2], -axis[1]

        pos = list(axis_cfg.get("origin", [0.0, 0.0, 0.0]))
        pos[1], pos[2] = pos[2], -pos[1]

        # Standard URDF limit: lower <= upper. SAJO emits `range = [phi_min, phi_max]`
        # from the observed phi_k trajectory.
        r = cfg.get("range", [0.0, 0.0])
        lower = float(min(r[0], r[1]))
        upper = float(max(r[0], r[1]))

        name_suffix = os.path.splitext(os.path.basename(mesh_path))[0]
        link_name = f"link_{name_suffix}"
        joint_name = f"joint_{name_suffix}"

        # Child link
        link = ET.SubElement(robot, "link", attrib={"name": link_name})
        vis = ET.SubElement(link, "visual")
        geom = ET.SubElement(vis, "geometry")
        ET.SubElement(geom, "mesh", attrib={"filename": os.path.relpath(mesh_path, os.path.dirname(urdf_path))})
        col = ET.SubElement(link, "collision")
        geomc = ET.SubElement(col, "geometry")
        ET.SubElement(geomc, "mesh", attrib={"filename": os.path.relpath(mesh_path, os.path.dirname(urdf_path))})

        # Joint connecting base -> part
        joint = ET.SubElement(robot, "joint", attrib={"name": joint_name, "type": jtype})
        ET.SubElement(joint, "parent", attrib={"link": base_link_name})
        ET.SubElement(joint, "child", attrib={"link": link_name})
        # origin: joint frame relative to parent
        ET.SubElement(
            joint,
            "origin",
            attrib={"xyz": f"{pos[0]} {pos[1]} {pos[2]}", "rpy": "0 0 0"},
        )
        ET.SubElement(joint, "axis", attrib={"xyz": f"{axis[0]} {axis[1]} {axis[2]}"})
        # Limits encode transform_scale range
        # Effort/velocity are arbitrary but required by some parsers; use generous defaults.
        ET.SubElement(
            joint,
            "limit",
            attrib={
                "lower": f"{lower}",
                "upper": f"{upper}",
                "effort": "1000",
                "velocity": "1000",
            },
        )

    # Pretty print
    _indent_xml(robot)
    tree = ET.ElementTree(robot)
    os.makedirs(os.path.dirname(urdf_path), exist_ok=True)
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)


def parse_urdf_and_save(
    urdf_path: str,
    qpos_by_part: Union[Sequence[float], Dict[str, float]],
    target_dir: str,
    save_individual: bool = True,
    combined_name: str = "assembled.glb",
) -> None:
    """
    Parse the URDF produced by `write_urdf`, apply joint configurations defined by qpos in [0,1],
    and save transformed meshes to target_dir. Also exports a combined GLB scene.

    Parameters
    ----------
    urdf_path : str
        Path to URDF file created by write_urdf.
    qpos_by_part : Sequence[float] | Dict[str, float]
        If a dict: keys are part names matching the suffix used in write_urdf
        (i.e., basename of the mesh without extension). Example:
            {"articulated_00": 0.3, "articulated_01": 1.0}
        If a list/tuple: it is assigned in the order joints appear in the URDF.
        Values must be in [0, 1].
    target_dir : str
        Output directory. Saves per-part meshes if save_individual is True, and a combined GLB.
    save_individual : bool
        If True, writes each transformed part mesh and base mesh (as OBJ).
    combined_name : str
        Filename for the combined scene GLB.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    base_dir = os.path.dirname(urdf_path)

    # Collect links' mesh files
    link_to_mesh = {}
    for link in root.findall("link"):
        name = link.get("name")
        # prefer visual mesh; fall back to collision if needed
        mesh_el = link.find("./visual/geometry/mesh")
        if mesh_el is None:
            mesh_el = link.find("./collision/geometry/mesh")
        if mesh_el is not None and mesh_el.get("filename"):
            mesh_file = os.path.normpath(os.path.join(base_dir, mesh_el.get("filename")))
            link_to_mesh[name] = mesh_file

    # Gather joints: parent -> child, type, axis, origin, limits
    joints = []
    for joint in root.findall("joint"):
        jname = joint.get("name")
        jtype = joint.get("type")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")

        origin_el = joint.find("origin")
        if origin_el is not None:
            xyz = [float(v) for v in origin_el.get("xyz", "0 0 0").split()]
        else:
            xyz = [0.0, 0.0, 0.0]

        axis_el = joint.find("axis")
        if axis_el is not None:
            axis = [float(v) for v in axis_el.get("xyz", "0 0 1").split()]
        else:
            axis = [0.0, 0.0, 1.0]
        axis = _safe_normalize(axis)

        limit_el = joint.find("limit")
        if limit_el is None:
            lower, upper = 0.0, 0.0
        else:
            lower = float(limit_el.get("lower", "0.0"))
            upper = float(limit_el.get("upper", "0.0"))

        name_suffix = child.replace("link_", "", 1) if child.startswith("link_") else child

        joints.append(
            dict(
                joint_name=jname,
                joint_type=jtype,
                parent=parent,
                child=child,
                name_suffix=name_suffix,
                origin=np.array(xyz, dtype=float),
                axis=np.array(axis, dtype=float),
                lower=lower,
                upper=upper,
            )
        )

    # Map qpos
    if isinstance(qpos_by_part, dict):
        def q_of(suffix: str) -> float:
            # allow suffix like 'articulated_00' or with extension removed
            key = suffix
            if key not in qpos_by_part and "." in key:
                key = os.path.splitext(key)[0]
            if key not in qpos_by_part:
                raise KeyError(f"qpos missing for part '{suffix}'.")
            q = float(qpos_by_part[key])
            if not (0.0 <= q <= 1.0):
                raise ValueError(f"qpos for '{suffix}' must be in [0,1], got {q}.")
            return q
        q_list = [q_of(j["name_suffix"]) for j in joints]
    else:
        q_list = list(map(float, qpos_by_part))
        if len(q_list) != len(joints):
            raise ValueError(f"Expected {len(joints)} qpos values, got {len(q_list)}.")
        for i, q in enumerate(q_list):
            if not (0.0 <= q <= 1.0):
                raise ValueError(f"qpos index {i} must be in [0,1], got {q}.")

    # Load base & parts; assemble scene
    os.makedirs(target_dir, exist_ok=True)

    # Load base (kept rigid)
    base_link = "base_link"
    if base_link not in link_to_mesh:
        # try a fallback: the first link with a mesh is base
        if len(link_to_mesh) == 0:
            raise RuntimeError("No meshes found in URDF.")
        base_link = list(link_to_mesh.keys())[0]

    base_mesh_path = link_to_mesh[base_link]
    base_geom = trimesh.load(base_mesh_path, force="mesh")

    scene = trimesh.Scene()
    scene.add_geometry(base_geom, geom_name=base_link)

    if save_individual:
        base_out = os.path.join(target_dir, f"{base_link}.obj")
        base_geom.export(base_out)

    # For each jointed child, compute transform and add to scene
    for j, q in zip(joints, q_list):
        child_link = j["child"]
        mesh_path = link_to_mesh.get(child_link)
        if mesh_path is None:
            continue  # skip if no geometry
        mesh = trimesh.load(mesh_path, force="mesh")

        # Map q in [0,1] to joint value using encoded limits
        val = j["lower"] + q * (j["upper"] - j["lower"])

        if j["joint_type"] == "prismatic":
            tvec = j["axis"] * val
            T = translation_matrix(tvec)
            mesh.apply_transform(T)
        elif j["joint_type"] == "revolute":
            angle = float(val)  # radians
            # Rotate about axis passing through origin point
            R = rotation_matrix(angle, j["axis"], point=j["origin"])
            mesh.apply_transform(R)
        else:
            # Unsupported joint types are treated as fixed
            pass

        scene.add_geometry(mesh, geom_name=child_link)

        if save_individual:
            out_name = os.path.join(target_dir, f"{child_link}.obj")
            mesh.export(out_name)

    # Export combined scene
    combined_path = os.path.join(target_dir, combined_name)
    # Trimesh can export scenes to GLB (requires pygltflib)
    scene.export(combined_path)


def validate_urdf(
    urdf_path: str,
    n_samples: int = 20,
    epsilon: float = 1e-3,
) -> Dict[str, object]:
    """Sweep each joint across its limit range and report self-collisions.

    Uses pybullet in DIRECT mode. For every joint, samples `n_samples` poses
    between `lower` and `upper`, resets the joint to each pose, runs
    collision detection, and records any contact with penetration depth
    exceeding `epsilon` as a violation.

    Returns a dict: {valid, max_penetration, n_joints, n_collisions_per_sample}.
    """
    import pybullet as p

    cid = p.connect(p.DIRECT)
    try:
        obj = p.loadURDF(urdf_path, useFixedBase=True, physicsClientId=cid)
        n_joints = p.getNumJoints(obj, physicsClientId=cid)
        report: Dict[str, object] = {
            "valid": True,
            "max_penetration": 0.0,
            "n_joints": n_joints,
            "n_collisions_per_sample": [],
        }
        for j_idx in range(n_joints):
            info = p.getJointInfo(obj, j_idx, physicsClientId=cid)
            lower, upper = info[8], info[9]
            if upper < lower:
                lower, upper = upper, lower
            if lower == upper:
                continue
            values = np.linspace(lower, upper, n_samples)
            for q in values:
                p.resetJointState(obj, j_idx, q, physicsClientId=cid)
                p.performCollisionDetection(physicsClientId=cid)
                contacts = p.getContactPoints(bodyA=obj, bodyB=obj, physicsClientId=cid)
                penetrations = [c[8] for c in contacts if c[8] < -epsilon]
                if penetrations:
                    report["valid"] = False
                    report["max_penetration"] = float(
                        min(report["max_penetration"], min(penetrations))
                    )
                report["n_collisions_per_sample"].append(len(penetrations))
        return report
    finally:
        p.disconnect(cid)


if __name__ == "__main__":

    # Unit test
    debug_dir = "debug/urdfs"
    os.makedirs(debug_dir, exist_ok=True)
     
    for model_id in ['100214']:
        
        write_urdf(
            base_mesh=f"outputs/{model_id}/sds_output/part_meshes/fixed.glb",
            part_meshes=[f"outputs/{model_id}/sds_output/part_meshes/articulated_00.glb"],
            joint_config=f"outputs/{model_id}/sds_output/joint_info.json",
            urdf_path=f"outputs/{model_id}/output.urdf",
            robot_name=model_id,
        )
        
        parse_urdf_and_save(
            urdf_path=f"outputs/{model_id}/output.urdf",
            qpos_by_part=[0.0],
            target_dir=debug_dir,
            save_individual=True,
            combined_name=f"{model_id}_qpos0.glb",
        )

        parse_urdf_and_save(
            urdf_path=f"outputs/{model_id}/output.urdf",
            qpos_by_part=[1.0],
            target_dir=debug_dir,
            save_individual=True,
            combined_name=f"{model_id}_qpos1.glb",
        )
    
    # Multi parts
    for model_id in ['46653']:
        write_urdf(
            base_mesh=f"outputs/{model_id}/sds_output/part_meshes/fixed.glb",
            part_meshes=[f"outputs/{model_id}/sds_output/part_meshes/articulated_{i:02d}.glb" for i in range(3)],
            joint_config=f"outputs/{model_id}/sds_output/joint_info.json",
            urdf_path=f"outputs/{model_id}/output.urdf",
            robot_name=model_id,
        )

        parse_urdf_and_save(
            urdf_path=f"outputs/{model_id}/output.urdf",
            qpos_by_part=[0.7, 0.5, 0.3],
            target_dir=debug_dir,
            save_individual=True,
            combined_name=f"{model_id}_qpos0.glb",
        ) 


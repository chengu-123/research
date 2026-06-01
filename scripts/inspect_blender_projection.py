import blenderproc as bproc

"""Inspect Blender camera projection for a FreeArt3D state GLB."""

import argparse
import json
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Blender screen projection.")
    parser.add_argument("--mesh_path", required=True)
    return parser.parse_args()


def look_at(obj, target: Vector) -> None:
    direction = obj.location - target
    rot_quat = direction.to_track_quat("Z", "Y")
    obj.rotation_euler = rot_quat.to_euler()


def setup_camera(source: Vector, target: Vector, fov: float):
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    cam_data.angle = np.radians(float(fov))
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    cam_obj.location = source
    look_at(cam_obj, target)
    return cam_obj


def imported_vertices(mesh_path: Path) -> np.ndarray:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    vertices = []
    for obj in bpy.context.view_layer.objects:
        if obj.type != "MESH":
            continue
        matrix_world = obj.matrix_world.copy()
        for vertex in obj.data.vertices:
            co = matrix_world @ vertex.co
            vertices.append([float(co.x), float(co.y), float(co.z)])
    if not vertices:
        raise ValueError(f"no vertices imported from {mesh_path}")
    return np.asarray(vertices, dtype=np.float32)


def main() -> None:
    args = parse_args()
    mesh_path = Path(args.mesh_path).resolve()
    vertices = imported_vertices(mesh_path)
    min_corner = vertices.min(axis=0)
    max_corner = vertices.max(axis=0)
    center = (min_corner + max_corner) * 0.5
    extent = max_corner - min_corner
    camera_distance = 2.1 * float(extent.max())
    azi = np.pi / 8.0
    height = np.sin(np.deg2rad(45.0)) * camera_distance
    source = np.asarray(
        [
            center[0] + np.sin(azi) * camera_distance,
            center[1] - np.cos(azi) * camera_distance,
            center[2] + height,
        ],
        dtype=np.float32,
    )
    camera = setup_camera(Vector(source.tolist()), Vector(center.tolist()), 45.0)
    bpy.context.scene.render.resolution_x = 800
    bpy.context.scene.render.resolution_y = 800
    bpy.context.view_layer.update()
    corners = np.asarray(
        [
            [min_corner[0], min_corner[1], min_corner[2]],
            [min_corner[0], min_corner[1], max_corner[2]],
            [min_corner[0], max_corner[1], min_corner[2]],
            [min_corner[0], max_corner[1], max_corner[2]],
            [max_corner[0], min_corner[1], min_corner[2]],
            [max_corner[0], min_corner[1], max_corner[2]],
            [max_corner[0], max_corner[1], min_corner[2]],
            [max_corner[0], max_corner[1], max_corner[2]],
        ],
        dtype=np.float32,
    )
    projected = []
    for point in corners:
        coord = world_to_camera_view(bpy.context.scene, camera, Vector(point.tolist()))
        projected.append([float(coord.x), float(coord.y), float(coord.z)])
    record = {
        "min": min_corner.tolist(),
        "max": max_corner.tolist(),
        "center": center.tolist(),
        "source": source.tolist(),
        "camera_matrix_world": [[float(v) for v in row] for row in camera.matrix_world],
        "corners": corners.tolist(),
        "projected_blender": projected,
    }
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

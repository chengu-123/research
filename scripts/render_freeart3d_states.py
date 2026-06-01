import blenderproc as bproc

"""Render FreeArt3D state GLBs with the repository Blender camera."""

import argparse
import gc
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-render FreeArt3D state GLBs.")
    parser.add_argument("--origin_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--states", default="0,1,2,3,4,5")
    parser.add_argument("--joint_idx", type=int, default=0)
    return parser.parse_args()


def parse_states(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def seed_blender(seed: int = 0, denoising: bool = False) -> None:
    bpy.context.scene.cycles.seed = int(seed)
    bpy.context.scene.cycles.use_animated_seed = False
    bpy.context.scene.cycles.use_adaptive_sampling = False
    bpy.context.scene.cycles.sampling_pattern = "TABULATED_SOBOL"
    bpy.context.view_layer.cycles.use_denoising = bool(denoising)
    bpy.context.scene.cycles.use_denoising = bool(denoising)


def look_at(obj, target: Vector) -> None:
    direction = obj.location - target
    rot_quat = direction.to_track_quat("Z", "Y")
    obj.rotation_euler = rot_quat.to_euler()


def clear_objects() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    gc.collect()


def clean_mesh_objects() -> None:
    gc.collect()
    for obj in list(bpy.data.objects):
        if obj.type == "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)
    for image in list(bpy.data.images):
        if image.users == 0:
            bpy.data.images.remove(image)


def setup_camera(source: Vector, target: Vector, camera=None, fov: float | None = None):
    if fov is not None:
        cam_data = bpy.data.cameras.new("Camera")
        cam_obj = bpy.data.objects.new("Camera", cam_data)
        cam_data.angle = np.radians(float(fov))
        bpy.context.scene.collection.objects.link(cam_obj)
        bpy.context.scene.camera = cam_obj
        camera = cam_obj
    camera.location = source
    look_at(camera, target)
    return camera


def setup_envmap() -> None:
    world = bpy.context.scene.world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    nodes.clear()
    env_texture = nodes.new(type="ShaderNodeTexEnvironment")
    env_texture.image = bpy.data.images.load("assets/sunrise_sky_dome_4k.exr")
    background_node = nodes.new(type="ShaderNodeBackground")
    node_tree.links.new(background_node.inputs["Color"], env_texture.outputs["Color"])
    output_node = nodes.new(type="ShaderNodeOutputWorld")
    node_tree.links.new(output_node.inputs["Surface"], background_node.outputs["Background"])
    mapping_node = nodes.get("Mapping") or nodes.new(type="ShaderNodeMapping")
    tex_coord_node = nodes.get("Texture Coordinate") or nodes.new(type="ShaderNodeTexCoord")
    node_tree.links.new(tex_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
    node_tree.links.new(mapping_node.outputs["Vector"], env_texture.inputs["Vector"])
    mapping_node.inputs["Rotation"].default_value[2] = np.radians(270)


def setup_rendering(render_size: int) -> None:
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.resolution_x = int(render_size)
    bpy.context.scene.render.resolution_y = int(render_size)
    bpy.context.view_layer.use_pass_diffuse_color = False
    bpy.context.scene.cycles.samples = 256


def hide_background(current_obj, index: int) -> None:
    current_obj.pass_index = int(index)
    bpy.context.view_layer.use_pass_object_index = True
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.use_nodes = True
    node_tree = bpy.context.scene.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()
    render_layer_node = nodes.new(type="CompositorNodeRLayers")
    id_mask_node = nodes.new(type="CompositorNodeIDMask")
    id_mask_node.index = int(index)
    links.new(render_layer_node.outputs["IndexOB"], id_mask_node.inputs["ID value"])
    alpha_over_node = nodes.new(type="CompositorNodeAlphaOver")
    alpha_over_node.inputs[1].default_value = (1, 1, 1, 0)
    links.new(id_mask_node.outputs["Alpha"], alpha_over_node.inputs[0])
    links.new(render_layer_node.outputs["Image"], alpha_over_node.inputs[2])
    composite_node = nodes.new(type="CompositorNodeComposite")
    links.new(alpha_over_node.outputs["Image"], composite_node.inputs["Image"])


def generate_camera_source(camera_distance: float, center: Vector) -> Vector:
    azi = np.pi / 8.0
    height = np.sin(np.deg2rad(45.0)) * camera_distance
    x_pos = np.sin(azi) * camera_distance
    y_pos = -np.cos(azi) * camera_distance
    return Vector((x_pos + center.x, y_pos + center.y, height + center.z))


def imported_bounds(mesh_path: str) -> tuple[Vector, Vector]:
    clear_objects()
    bpy.ops.import_scene.gltf(filepath=mesh_path)
    imported_meshes = [obj for obj in bpy.context.view_layer.objects if obj.select_get() and obj.type == "MESH"]
    bpy.context.view_layer.update()
    world_verts = []
    for obj in imported_meshes:
        matrix_world = obj.matrix_world.copy()
        for vertex in obj.data.vertices:
            world_verts.append(matrix_world @ vertex.co)
    if not world_verts:
        raise ValueError(f"no vertices imported from {mesh_path}")
    min_corner = Vector((min(v.x for v in world_verts), min(v.y for v in world_verts), min(v.z for v in world_verts)))
    max_corner = Vector((max(v.x for v in world_verts), max(v.y for v in world_verts), max(v.z for v in world_verts)))
    return min_corner, max_corner


def render_states(mesh_paths: list[str], output_dir: Path, joint_idx: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    render_size = 800
    fov = 45
    min_corner, max_corner = imported_bounds(mesh_paths[-1])
    center = Vector(((min_corner.x + max_corner.x) / 2.0, (min_corner.y + max_corner.y) / 2.0, (min_corner.z + max_corner.z) / 2.0))
    extent = max_corner - min_corner
    object_scale = max(extent.x, extent.y, extent.z)
    camera_distance = 2.1 * object_scale
    disk_radius = 1.0 * object_scale
    min_z = min_corner.z
    clear_objects()
    source = generate_camera_source(camera_distance, center)
    camera = setup_camera(fov=fov, source=source, target=center)
    setup_envmap()
    setup_rendering(render_size)

    for state_id, mesh_path in enumerate(mesh_paths):
        clean_mesh_objects()
        bpy.ops.import_scene.gltf(filepath=mesh_path)
        bpy.ops.mesh.primitive_cylinder_add(radius=disk_radius, depth=1e-5, location=(center.x, center.y, min_z - 1e-4))
        disk = bpy.context.view_layer.objects.active
        blue_mat = bpy.data.materials.new(name="BlueMaterial")
        blue_mat.diffuse_color = (0.3, 0.6, 1.0, 1.0)
        blue_mat.use_nodes = False
        disk.data.materials.append(blue_mat)
        output_render_path = output_dir / f"rendering_joint_{joint_idx:02d}_state_{state_id:02d}.png"
        setup_camera(source=source, target=center, camera=camera)
        bpy.context.scene.frame_set(state_id)
        bpy.context.scene.render.filepath = str(output_render_path)
        for obj in bpy.data.objects:
            hide_background(obj, 233)
        seed_blender()
        bpy.ops.render.render(write_still=True)

    for state_id, mesh_path in enumerate(mesh_paths):
        clean_mesh_objects()
        bpy.ops.import_scene.gltf(filepath=mesh_path)
        output_render_path = output_dir / f"rendering_pure_joint_{joint_idx:02d}_state_{state_id:02d}.png"
        setup_camera(source=source, target=center, camera=camera)
        bpy.context.scene.frame_set(state_id)
        bpy.context.scene.render.filepath = str(output_render_path)
        for obj in bpy.data.objects:
            hide_background(obj, 233)
        seed_blender(denoising=True)
        bpy.ops.render.render(write_still=True)


def main() -> None:
    args = parse_args()
    origin_dir = Path(args.origin_dir).resolve()
    states = parse_states(args.states)
    mesh_paths = [
        str(origin_dir / "sds_output" / "states" / f"qpos_{state:02d}.glb")
        for state in states
    ]
    render_states(mesh_paths, Path(args.out_dir).resolve(), int(args.joint_idx))


if __name__ == "__main__":
    main()

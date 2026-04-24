import os 
import os.path as osp

import bpy
import numpy as np
import mathutils
import trimesh

from eval_utils.utils_3d import transform_points

def look_at(obj, target, objaverse=False):
    direction = obj.location - target
    rot_quat = direction.to_track_quat('Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()

def clear_objects():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def setup_camera(source, fov=45):
    
    # Set up the camera
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    camera.data.angle = np.radians(fov)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
        
    camera.location = source
    look_at(camera, mathutils.Vector([0, 0, 0]))
    return camera  

def generate_opengl_view_matrices_on_sphere(n_cameras, radius=1.0):
    """
    Generate OpenGL-style world-to-camera matrices from points on a sphere.
    :param n_cameras: Number of cameras.
    :param radius: Radius of the sphere.
    :return: (n_cameras, 4, 4) array of view matrices.
    """
    def normalize(v):
        """Normalize a vector."""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def look_at_opengl(eye, target=np.array([0, 0, 0]), up=np.array([0, 1, 0])):
        """
        Create a world-to-camera (view) matrix in OpenGL coordinate system.
        :param eye: Camera position.
        :param target: Look-at target (default: origin).
        :param up: Up direction (default: +Y).
        :return: 4x4 view matrix (world-to-camera).
        """
        f = normalize(target - eye)           # forward
        s = normalize(np.cross(f, up))        # right
        u = np.cross(s, f)                    # recalculated up

        M = np.eye(4)
        M[0, :3] = s
        M[1, :3] = u
        M[2, :3] = -f
        M[:3, 3] = -M[:3, :3] @ eye
        return M

    def sample_sphere_points(n, radius=1.0):
        """
        Uniformly sample points on a sphere using spherical coordinates.
        :param n: Number of points.
        :param radius: Radius of the sphere.
        :return: Array of shape (n, 3) with sampled 3D points.
        """
        phi = np.arccos(1 - 2 * np.random.rand(n))
        theta = 2 * np.pi * np.random.rand(n)

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        return np.stack([x, y, z], axis=1)

    view_matrices = []
    eye_positions = sample_sphere_points(n_cameras, radius)
    for eye in eye_positions:
        view = look_at_opengl(eye)
        view_matrices.append(view)
    return np.array(view_matrices)

def eval_rendering_bpy(camera_sources, res_path, method, qpos_id, transformed_mesh_path):

    # Clear any existing objects in the scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Create a new scene
    scene = bpy.context.scene
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 8
    scene.render.film_transparent = True
     
    if 'ours' in method:
        # Set ambient environment color
        world = bpy.context.scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        world.use_nodes = True
        bg_node = world.node_tree.nodes.get("Background") 
        bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0) 
        scene = bpy.context.scene
        scene.view_settings.exposure = 1.0
    else:
        # Set envmap since original input has an envmap
        world = bpy.context.scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        world.use_nodes = True
        node_tree = world.node_tree
        nodes = node_tree.nodes
        nodes.clear()
        env_texture = nodes.new(type='ShaderNodeTexEnvironment')
        env_texture.image = bpy.data.images.load("assets/sunrise_sky_dome_4k.exr")
        background_node = nodes.new(type='ShaderNodeBackground')
        background_node.inputs['Strength'].default_value
        node_tree.links.new(background_node.inputs['Color'], env_texture.outputs['Color'])
        output_node = nodes.new(type='ShaderNodeOutputWorld')
        node_tree.links.new(output_node.inputs['Surface'], background_node.outputs['Background'])

        mapping_node = nodes.get("Mapping") or nodes.new(type="ShaderNodeMapping")
        tex_coord_node = nodes.get("Texture Coordinate") or nodes.new(type="ShaderNodeTexCoord")
        node_tree.links.new(tex_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
        node_tree.links.new(mapping_node.outputs["Vector"], env_texture.inputs["Vector"])
        rotation_degrees = 270  # Adjust this angle as needed
        mapping_node.inputs["Rotation"].default_value[2] = np.radians(rotation_degrees)

        scene = bpy.context.scene
        scene.view_settings.exposure = 0.0
 
    bpy.ops.import_scene.gltf(filepath=transformed_mesh_path) 
    res_folder = osp.join(res_path, method, f'qpos_{qpos_id:02d}')
    os.makedirs(res_folder, exist_ok=True)
    
    # Render from each camera position 
    for i, source in enumerate(camera_sources):

        setup_camera(source)
        image_path = osp.join(res_folder, f"cam_{i:02d}.png")
        scene.render.filepath = image_path
        if not osp.exists(image_path): # Skip if already rendered
            bpy.ops.render.render(write_still=True)
        
    # Clean up
    for obj in bpy.context.scene.objects:
        bpy.data.objects.remove(obj) 

def transform_bpy_mesh(pred_mesh_path, transformed_mesh_path, pred_mesh_extend, pred_mesh_centroid, rotz=None, 
    final_tsfm=None, scale=None, gt=False):
 
    # Clear existing objects
    bpy.ops.wm.read_factory_settings(use_empty=True) 
    bpy.ops.import_scene.gltf(filepath=pred_mesh_path) 
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    for obj in mesh_objects:

        # Get world matrix to transform local coordinates to world coordinates
        world_matrix = obj.matrix_world
        mesh = obj.data
        # Apply world transformation to each vertex
        vertices = []
        for vertex in mesh.vertices:
            # Transform vertex from local to world coordinates
            world_vertex = world_matrix @ vertex.co
            vertices.append(np.array([world_vertex.x, world_vertex.z, -world_vertex.y]))

        vertices = np.array(vertices)
        mesh_tri = trimesh.PointCloud(vertices)
        mesh_tri.apply_scale(pred_mesh_extend)
        mesh_tri.apply_translation(-pred_mesh_centroid)
        
        if not gt:
            mesh_tri.vertices = transform_points(mesh_tri.vertices, rotz)
            mesh_tri.apply_scale(scale)
            mesh_tri.vertices = transform_points(mesh_tri.vertices, final_tsfm) 
        
        # Apply world transformation to each vertex
        for i, vertex in enumerate(mesh.vertices):
            world_vertex = np.array([mesh_tri.vertices[i][0], -mesh_tri.vertices[i][2], mesh_tri.vertices[i][1]])
            vertex.co = mathutils.Vector(world_vertex)
        obj.matrix_world = mathutils.Matrix.Identity(4) 

    # Export the modified mesh back to GLB, preserving texture
    os.makedirs(osp.dirname(transformed_mesh_path), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=transformed_mesh_path,
        export_format='GLB', 
        use_selection=False
    )

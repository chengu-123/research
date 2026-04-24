import numpy as np 
import bpy
import os  
import gc 
import random

from mathutils import Vector
from pipelines.utils.seeding import seed_blender

def disable_all_denoiser():
    """ Disables all denoiser.

    At the moment this includes the cycles and the intel denoiser.
    """
    # Disable cycles denoiser
    bpy.context.view_layer.cycles.use_denoising = False
    bpy.context.scene.cycles.use_denoising = False

def look_at(obj, target):
    direction = obj.location - target
    rot_quat = direction.to_track_quat('Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()

def clean():
    
    gc.collect()
    bpy.ops.outliner.orphans_purge()
    if bpy.data.images.get("Render Result"):
        bpy.data.images["Render Result"].user_clear()
        bpy.data.images.remove(bpy.data.images["Render Result"])
    
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            bpy.data.objects.remove(obj)
                
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        
    for material in bpy.data.materials:
        if material.users == 0:
            bpy.data.materials.remove(material)
        
    for image in bpy.data.images:
        if image.users == 0:
            bpy.data.images.remove(image)

    for texture in bpy.data.textures:
        if texture.users == 0:
            bpy.data.textures.remove(texture)

    for node_group in bpy.data.node_groups:
        if node_group.users == 0:
            bpy.data.node_groups.remove(node_group)

def clear_objects():
    
    # Remove all objects using the data API (context-free)
    for obj in list(bpy.data.objects):
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass

    # Manually purge orphaned datablocks without context-dependent operators
    # Collections first (to reduce users on datablocks)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            try:
                bpy.data.collections.remove(collection)
            except Exception:
                pass

    datablock_groups = (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.textures,
        bpy.data.node_groups,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.curves,
    )
    for group in datablock_groups:
        for datablock in list(group):
            if datablock.users == 0:
                try:
                    group.remove(datablock)
                except Exception:
                    pass
 
def setup_camera(source, target, camera=None, fov=None):
     
    if fov is not None:
        cam_data = bpy.data.cameras.new("Camera")
        cam_obj = bpy.data.objects.new("Camera", cam_data)
        cam_data.angle = np.radians(fov)
        bpy.context.scene.collection.objects.link(cam_obj)
        bpy.context.scene.camera = cam_obj
        camera = cam_obj 
        
    camera.location = source 
    look_at(camera, target)
    return camera

def generate_camera_sources(camera_distance, total_views, azi, min_azi, max_azi,
    x_offset=0, y_offset=0, z_offset=0):
    
    if azi is not None:
        height = np.sin(np.deg2rad(45)) * camera_distance
        x_pos = np.sin(azi) * camera_distance
        y_pos = -np.cos(azi) * camera_distance
        sources = [Vector((x_pos + x_offset, y_pos + y_offset, height + z_offset))]
    else:
        sources = []  
        for i in range(total_views):        
            azi = min_azi + (max_azi - min_azi) * i / (total_views - 1)
            height = np.sin(np.deg2rad(45)) * camera_distance
            x_pos = np.sin(azi) * camera_distance
            y_pos = -np.cos(azi) * camera_distance
            sources.append(Vector((x_pos + x_offset, y_pos + y_offset, height + z_offset)))

    return sources

def setup_envmap():

    world = bpy.context.scene.world
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

def setup_ambient_light(color=(1.0, 1.0, 1.0, 1.0)): 

    world = bpy.context.scene.world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    nodes.clear()

    background_node = nodes.new(type='ShaderNodeBackground')
    background_node.inputs['Color'].default_value = color 
    background_node.inputs['Strength'].default_value = 1.0

    output_node = nodes.new(type='ShaderNodeOutputWorld')
    node_tree.links.new(output_node.inputs['Surface'], background_node.outputs['Background'])

def setup_rendering(render_size):
     
    bpy.context.scene.render.engine = 'CYCLES' 
    bpy.context.scene.render.resolution_x = render_size
    bpy.context.scene.render.resolution_y = render_size 
    bpy.context.view_layer.use_pass_diffuse_color = False
    bpy.context.scene.cycles.samples = 256

def setup_transparent_compositor(pass_index=233):
    # Make sure the compositor is actually used during render
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree
    nodes, links = tree.nodes, tree.links
    nodes.clear()

    rl = nodes.new("CompositorNodeRLayers")
    idmask = nodes.new("CompositorNodeIDMask"); idmask.index = pass_index
    over = nodes.new("CompositorNodeAlphaOver"); over.inputs[1].default_value = (1, 1, 1, 0)
    comp = nodes.new("CompositorNodeComposite")

    links.new(rl.outputs["IndexOB"], idmask.inputs["ID value"])
    links.new(idmask.outputs["Alpha"], over.inputs[0])
    links.new(rl.outputs["Image"], over.inputs[2])
    links.new(over.outputs["Image"], comp.inputs["Image"])

def hide_background(current_obj, index):
    
    current_obj.pass_index = index
    bpy.context.view_layer.use_pass_object_index = True
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.use_nodes = True
    node_tree = bpy.context.scene.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()
    
    render_layer_node = nodes.new(type="CompositorNodeRLayers")
    id_mask_node = nodes.new(type="CompositorNodeIDMask")
    id_mask_node.index = index
    links.new(render_layer_node.outputs["IndexOB"], id_mask_node.inputs["ID value"])
    alpha_over_node = nodes.new(type="CompositorNodeAlphaOver")
    alpha_over_node.inputs[1].default_value = (1, 1, 1, 0)  # Set transparent background
    links.new(id_mask_node.outputs["Alpha"], alpha_over_node.inputs[0])
    links.new(render_layer_node.outputs["Image"], alpha_over_node.inputs[2])
    composite_node = nodes.new(type="CompositorNodeComposite")
    links.new(alpha_over_node.outputs["Image"], composite_node.inputs["Image"])

def run_rendering(mesh_paths, output_dir, min_azi=np.pi / 8, max_azi=3 * np.pi / 8, recon=False, joint_idx=0):

    os.makedirs(output_dir, exist_ok=True)

    render_size = 800
    fov = 45 
    total_states = len(mesh_paths)
    if 'joint_-1' in mesh_paths[-1] and joint_idx != -1: # Multi-joint case, the last mesh is a reference
        total_states -= 1

    output_filepaths = []

    if recon:
        total_views = 3
        azi = None
    else:
        total_views = 1
        azi = np.pi / 8

    # Approach using bmesh to get accurate bounding box
    clear_objects()

    bpy.ops.import_scene.gltf(filepath=mesh_paths[-1]) 
    imported_meshes = [obj for obj in bpy.context.view_layer.objects if obj.select_get() and obj.type == 'MESH']
    bpy.context.view_layer.update()

    # Now process each mesh with correct world space transformations
    all_world_verts = []
    for obj in imported_meshes: 
        mesh = obj.data 
        matrix_world = obj.matrix_world.copy() 
        for v in mesh.vertices:
            world_co = matrix_world @ v.co
            all_world_verts.append(world_co)

    # Calculate bounding box from all world vertices
    min_corner = Vector((
        min(v.x for v in all_world_verts),
        min(v.y for v in all_world_verts),
        min(v.z for v in all_world_verts),
    ))
    max_corner = Vector((
        max(v.x for v in all_world_verts),
        max(v.y for v in all_world_verts),
        max(v.z for v in all_world_verts),
    ))

    # Now we have the correct world space min and max corners
    min_z = min_corner.z
    center_x = (min_corner.x + max_corner.x) / 2
    center_y = (min_corner.y + max_corner.y) / 2
    center_z = (min_corner.z + max_corner.z) / 2

    # Calculate scales
    extent = max_corner - min_corner
    object_scale = max(extent.x, extent.y, extent.z) 
    if recon:
        camera_distance = 1.2 * object_scale
    else:
        camera_distance = 2.1 * object_scale
        disk_radius = 1.0 * object_scale 

    clear_objects()
    target = Vector((center_x, center_y, center_z))
    source = Vector((center_x, center_y - camera_distance, center_z))
    camera = setup_camera(fov=fov, source=source, target=target)  
    
    if recon:
        setup_ambient_light()
    else:
        setup_envmap()

    setup_rendering(render_size)
    sources = generate_camera_sources(camera_distance, total_views, azi, min_azi, max_azi,
        x_offset=center_x, y_offset=center_y, z_offset=center_z)

    output_filepaths = [] 

    for i in range(total_states): 
        
        clean() 
        bpy.ops.import_scene.gltf(filepath=mesh_paths[i]) 
        imported_meshes = [obj for obj in bpy.context.view_layer.objects if obj.select_get() and obj.type == 'MESH']
        bpy.context.view_layer.update()
         
        if recon: # Align different states to the same height
            model_obj = imported_meshes[0]
            bbox_world = [model_obj.matrix_world @ Vector(corner) for corner in model_obj.bound_box]
            current_min_z = min(v.z for v in bbox_world)
            z_offset = min_z - current_min_z
            model_obj.location.z += z_offset  
        else: # Add blue disk 
            bpy.ops.mesh.primitive_cylinder_add(
                radius=disk_radius,
                depth=1e-5,
                location=(center_x, center_y, min_z - 1e-4),
            )
            disk = bpy.context.view_layer.objects.active 
            blue_mat = bpy.data.materials.new(name="BlueMaterial")
            blue_mat.diffuse_color = (0.3, 0.6, 1.0, 1.0)
            blue_mat.use_nodes = False 
            disk.data.materials.append(blue_mat)

        # Render
        for j in range(total_views):

            if recon:
                output_render_path = f'{output_dir}/rendering_recon_joint_{joint_idx:02d}_state_{i:02d}_view_{j:02d}.png'
            else:
                output_render_path = f'{output_dir}/rendering_joint_{joint_idx:02d}_state_{i:02d}.png'

            camera = setup_camera(source=sources[j], target=target, camera=camera) 
            bpy.context.scene.frame_set(i)
            bpy.context.scene.render.filepath = output_render_path
            # setup_transparent_compositor()
            for k, obj in enumerate(bpy.data.objects):
                hide_background(obj, 233)

            if recon:
                # Render position map & depth map
                bpy.context.view_layer.use_pass_position = True  # Enable Position pass
                bpy.context.view_layer.use_pass_z = True         # Enable Z pass (depth)
                bpy.context.scene.use_nodes = True
                tree = bpy.context.scene.node_tree
                tree.nodes.clear()

                render_layers = tree.nodes.new('CompositorNodeRLayers')

                # Setup position map output
                position_output = tree.nodes.new('CompositorNodeOutputFile')
                position_output.label = 'WorldPositionOutput'
                position_output.base_path = output_dir
                position_output.file_slots[0].path = f'rendering_recon_joint_{joint_idx:02d}_state_{i:02d}_view_{j:02d}_position'
                position_output.format.file_format = 'OPEN_EXR'
                position_output.format.color_depth = '32'
                position_output.format.color_mode = 'RGB'

                # Setup depth map output
                depth_output = tree.nodes.new('CompositorNodeOutputFile')
                depth_output.label = 'DepthOutput'
                depth_output.base_path = output_dir
                depth_output.file_slots[0].path = f'rendering_recon_joint_{joint_idx:02d}_state_{i:02d}_view_{j:02d}_depth'
                depth_output.format.file_format = 'OPEN_EXR'
                depth_output.format.color_depth = '32'
                depth_output.format.color_mode = 'RGB'

                # Optional: Add normalization for better depth visualization
                normalize = tree.nodes.new('CompositorNodeNormalize')
                tree.links.new(render_layers.outputs['Depth'], normalize.inputs[0])
                tree.links.new(normalize.outputs[0], depth_output.inputs[0])

                # Connect position pass
                tree.links.new(render_layers.outputs['Position'], position_output.inputs[0])

            if not os.path.exists(output_render_path):
                seed_blender()  
                bpy.ops.render.render(write_still=True)
            output_filepaths.append(output_render_path)   
        
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                bpy.data.objects.remove(obj)

    # Render images without blue disk
    if not recon: 

        for i in range(total_states):
            
            clean()
            bpy.ops.import_scene.gltf(filepath=mesh_paths[i]) 
            
            # Render
            output_render_path = f'{output_dir}/rendering_pure_joint_{joint_idx:02d}_state_{i:02d}.png' 

            setup_camera(source=sources[0], target=target, camera=camera)
            bpy.context.scene.frame_set(i)
            bpy.context.scene.render.filepath = output_render_path
            for k, obj in enumerate(bpy.data.objects):
                hide_background(obj, 233)

            if not os.path.exists(output_render_path):
                seed_blender(denoising=True)
                bpy.ops.render.render(write_still=True)
                
            for obj in bpy.data.objects:
                if obj.type == 'MESH':
                    bpy.data.objects.remove(obj)
        
    return output_filepaths 

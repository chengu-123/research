import blenderproc as bproc
import os
import bpy
import xml.etree.ElementTree as ET
import json
import argparse

from mathutils import Matrix, Vector
from pathlib import Path

def bpy_cleanup_mesh(obj):
    assert obj.type == 'MESH'
    # remove duplicate vertices
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.remove_doubles(threshold=1e-06)
    bpy.ops.object.mode_set(mode='OBJECT')
    # disable auto-smoothing
    # obj.data.use_auto_smooth = False
    # split edges with an angle above 70 degrees (1.22 radians)
    m = obj.modifiers.new("EdgeSplit", "EDGE_SPLIT")
    m.split_angle = 1.22173
    bpy.ops.object.modifier_apply(modifier="EdgeSplit")
    # move every face an epsilon in the direction of its normal, to reduce clipping artifacts
    m = obj.modifiers.new("Displace", "DISPLACE")
    m.strength = 0.00001
    bpy.ops.object.modifier_apply(modifier="Displace")

def parse_origin(element):
    """Parse the <origin> tag to extract translation and rotation."""
    translation = [0.0, 0.0, 0.0]
    rotation = [0.0, 0.0, 0.0]

    if element is not None:
        if 'xyz' in element.attrib:
            translation = [float(x) for x in element.attrib['xyz'].split()]
        if 'rpy' in element.attrib:
            rotation = [float(r) for r in element.attrib['rpy'].split()]

    return translation, rotation

def read_joints_and_meshes(urdf_path):
    """Parse the URDF file to extract joint and mesh information."""
    joints = {}
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    link_to_meshes = {}
    for link in root.findall('link'):
        link_name = link.get('name')
        visuals = link.findall('visual')
        mesh_data = []
        for visual in visuals:
            geometry = visual.find('geometry')
            origin = visual.find('origin')
            translation, rotation = parse_origin(origin)
            if geometry is not None:
                mesh = geometry.find('mesh')
                if mesh is not None:
                    mesh_path = mesh.get('filename')
                    if mesh_path:
                        mesh_data.append({
                            'path': mesh_path,
                            'translation': translation,
                            'rotation': rotation
                        })
        if mesh_data:
            link_to_meshes[link_name] = mesh_data

    continuous_info = []

    for joint in root.findall('joint'):
        joint_name = joint.get('name')
        joint_info = {
            'type': joint.get('type'),
            'axis': None,
            'limit': None,
            'child': [],
            'parent': None,
            'origin_translation': [0.0, 0.0, 0.0],
            'origin_rotation': [0.0, 0.0, 0.0],
            'mesh_data': []
        }

        origin = joint.find('origin')
        joint_info['origin_translation'], joint_info['origin_rotation'] = parse_origin(origin)

        axis = joint.find('axis')
        if axis is not None:
            joint_info['axis'] = [float(i) if i != 'None' else 0.0 for i in axis.get('xyz').split()]

        limit = joint.find('limit')
        if limit is not None:
            joint_info['limit'] = {
                'lower': float(limit.get('lower', 0)),
                'upper': float(limit.get('upper', 0))
            }

        parent = joint.find('parent')
        if parent is not None:
            joint_info['parent'] = parent.get('link')

        # TODO: Find all childs
        child = joint.find('child')
        parent = joint.find('parent')
        if child is not None:
            child_link = child.get('link')
            joint_info['child'].append(child_link)
            for child_name in joint_info['child']:
                joint_info['mesh_data'] += link_to_meshes.get(child_name, [])
        joints[joint_name] = joint_info

    return joints

def remove_materials_without_textures():
    """Remove materials without texture maps and delete the corresponding faces."""
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')

            # Get materials without textures
            materials_to_remove = []
            check_materials = False
            
            for i, material in enumerate(obj.data.materials):
                if material and has_texture_map(material):
                    check_materials = True
                
            if check_materials:
                for i, material in enumerate(obj.data.materials):
                    if not material or not has_texture_map(material):
                        materials_to_remove.append(i)

            # Remove faces using those materials
            bpy.ops.mesh.select_all(action='DESELECT')
            for mat_index in materials_to_remove:
                obj.active_material_index = mat_index
                bpy.ops.object.material_slot_select()
                bpy.ops.mesh.delete(type='FACE')

            bpy.ops.object.mode_set(mode='OBJECT')

            # Remove unused material slots
            for mat_index in sorted(materials_to_remove, reverse=True):
                obj.active_material_index = mat_index
                bpy.ops.object.material_slot_remove()

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            
            # Check geometry and normals
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode='OBJECT')
            
            # Check and assign materials
            for face in obj.data.polygons:
                if face.material_index >= len(obj.material_slots):
                    print(f"Face {face.index} has no valid material.")
                    face.material_index = 0  # Assign default material
            
            # Check UV mapping
            if not obj.data.uv_layers:
                print("UV mapping missing. Generating UVs...")
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.uv.smart_project()
                bpy.ops.object.mode_set(mode='OBJECT')

def has_texture_map(material):
    """Check if a material has a texture map (image texture)."""
    if material.use_nodes:
        for node in material.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                return True
    return False

def normalize_scene(center, scale):
    bpy.ops.transform.translate(value=center)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.transform.resize(value=(scale, scale, scale))
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

def process_state(joints, mesh_dir, state_idx, num_states, joint_type, joint_list):
    
    max_bounds = [-float('inf')] * 3
    min_bounds = [float('inf')] * 3
    objs = []
    
    """Process a single state, applying transformations to each mesh."""
    for joint_name, joint_info in joints.items():
        for mesh_entry in joint_info['mesh_data']:
            mesh_path = os.path.join(mesh_dir, mesh_entry['path'])
            if not os.path.exists(mesh_path):
                print(f"Mesh not found: {mesh_path}")
                continue
            objs += bproc.loader.load_obj(mesh_path)
    
    for obj in objs:
        obj_b = obj.blender_obj
        bpy_cleanup_mesh(obj_b)
    for obj in objs:
        obj_b = obj.blender_obj
        for i in range(3):
            max_bounds[i] = max(max_bounds[i], obj_b.bound_box[0][i])
            min_bounds[i] = min(min_bounds[i], obj_b.bound_box[0][i])
    for mat in bpy.data.materials:
        mat.use_backface_culling = True
    
    # Calculate joint-specific transformation
    for joint_name, joint_info in joints.items():
        if joint_name == joint_list[0]:
            if joint_type == 'revolute':
                lower = joint_info['limit']['lower'] if joint_info['limit'] else -1.57079
                upper = joint_info['limit']['upper'] if joint_info['limit'] else 1.57079
                angle = lower + (upper - lower) * (state_idx / (num_states - 1))
                origin_translation = [joint_info['origin_translation'][i] for i in range(3)]
                origin_translation = Matrix.Translation(origin_translation)
                joint_rotation = Matrix.Rotation(angle, 4, joint_info['axis'])
                joint_transformation = origin_translation @ joint_rotation @ origin_translation.inverted()
            elif joint_type == 'prismatic':
                lower = joint_info['limit']['lower'] if joint_info['limit'] else 0
                upper = joint_info['limit']['upper'] if joint_info['limit'] else 0.5
                translation = lower + (upper - lower) * (state_idx / (num_states - 1))
                origin_translation = [joint_info['origin_translation'][i] for i in range(3)]
                origin_translation = Matrix.Translation(origin_translation)
                joint_translation = Matrix.Translation([joint_info['axis'][i] * translation for i in range(3)])
                joint_transformation = origin_translation @ joint_translation @ origin_translation.inverted()

    # Apply joint-specific transformation
    obj_idx = 0
    for joint_name, joint_info in joints.items():
        for mesh_entry in joint_info['mesh_data']:
            mesh_path = os.path.join(mesh_dir, mesh_entry['path'])
            if not os.path.exists(mesh_path):
                print(f"Mesh not found: {mesh_path}")
                continue
            if joint_name in joint_list:
                objs[obj_idx].blender_obj.matrix_world @= joint_transformation
            obj_idx += 1
    
def export_state(output_path):
    """Export the current Blender scene as a GLB file."""
    bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB', use_selection=False)
    print(f"Exported: {output_path}")

def process_states(mesh_dir, urdf_path, output_dir, num_states=6, joint_type="revolute", joint_list=[], joint_idx=0):
    """Process all states and export them as GLB files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    joints = read_joints_and_meshes(urdf_path) 

    for state_idx in range(num_states):
        # Clear scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        # Process and export the state
        process_state(joints, mesh_dir, state_idx, num_states, joint_type, joint_list)
        export_state(f'{output_dir}/{state_idx:02d}.glb')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_list', type=str, default='configs/partnet.json')  
    parser.add_argument('--input_dir', type=str, default='datasets/PartNet_raw') 
    parser.add_argument('--output_dir', type=str, default='datasets/PartNet')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=1) 
    parser.add_argument('--num_states', type=int, default=6)
    args = parser.parse_args()
    
    with open(args.data_list) as f:
        data_info = json.load(f)
      
    model_ids = data_info['total_obj_ids']
    
    for model_id in model_ids[args.start_idx:min(args.end_idx, len(model_ids))]:

        mesh_dir = f"{args.input_dir}/{model_id}"
        urdf_path = f"{args.output_dir}/{model_id}/mobility.urdf"
        joints_info_path = f"{args.output_dir}/{model_id}/joints.json"
        output_dir = f"{args.output_dir}/{model_id}/gt_mesh"
        joint_info_output_path = f"{args.output_dir}/{model_id}/joint_info.json"

        with open(joints_info_path, "r") as f:
            joints_info = json.load(f)

        joint_type = "revolute" if model_id in data_info['revolute']['obj_ids'] else "prismatic" 
        joints = read_joints_and_meshes(urdf_path) 
        target_joint = joints_info['joints'][0] # Default to only choose the first joint as the target joint 

        # Output joint info for evaluation
        joint_info = []
        for joint_name, joint_cfg in joints.items():
            if joint_name == target_joint:
                current_joint_info = {}
                current_joint_info['type'] = joint_type
                current_joint_info['range'] = [joint_cfg['limit']['lower'], joint_cfg['limit']['upper']]
                current_joint_info['axis'] = {}
                current_joint_info['axis']['origin'] = joint_cfg['origin_translation']
                current_joint_info['axis']['direction'] = joint_cfg['axis']
                joint_info.append(current_joint_info)
        with open(joint_info_output_path, "w") as f:
            json.dump(joint_info, f, indent=4)
  
        print(f"Processing {model_id} with joint type {joint_type}")
        joint_list = [target_joint]        
        joint_list += joints_info['descendants'][target_joint]
        process_states(mesh_dir, urdf_path, output_dir, args.num_states, joint_type, joint_list, 0)

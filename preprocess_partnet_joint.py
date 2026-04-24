import xml.etree.ElementTree as ET
import json 
import os
import argparse

from tqdm import tqdm

def parse_urdf(file_path):
    """
    Parse a URDF file and extract information about joints of a specific type.
    
    Args:
        file_path (str): Path to the URDF file
        target_joint_type (str): Joint type to look for ('revolute', 'prismatic', etc.)
    
    Returns:
        tuple: (list of joint names of the target type, dict of descendants for each joint)
    """
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Find all joints
    joints = root.findall('.//joint')
    
    # Create dictionaries to store joint information
    joint_types = {}      # joint_name -> joint_type
    child_links = {}      # joint_name -> child_link_name
    parent_links = {}     # joint_name -> parent_link_name
    
    # Extract information from joints
    for joint in joints:
        joint_name = joint.get('name')
        joint_type = joint.get('type')
        
        child_elem = joint.find('child')
        parent_elem = joint.find('parent')
        
        if child_elem is not None and parent_elem is not None:
            child_link = child_elem.get('link')
            parent_link = parent_elem.get('link')
            
            joint_types[joint_name] = joint_type
            child_links[joint_name] = child_link
            parent_links[joint_name] = parent_link
    
    # Create reverse mapping from link to joint
    link_to_joint = {}    # link_name -> joint_name (where link is the child of the joint)
    for joint_name, child_link in child_links.items():
        link_to_joint[child_link] = joint_name
    
    # Find joints with the target type
    target_joints = [(j, t) for j, t in joint_types.items()]
    
    # Build descendant chains for each joint
    descendants = {joint_name: [] for joint_name in joint_types.keys()}
    
    # For each joint
    for joint_name in joint_types.keys():
        # Get its child link
        child_link = child_links.get(joint_name)
        
        # Look for joints that have this link as parent
        for other_joint, parent in parent_links.items():
            if parent == child_link:
                # Add to descendants
                descendants[joint_name].append(other_joint)
                
                # Recursively find all descendants
                to_process = [other_joint]
                while to_process:
                    current = to_process.pop(0)
                    current_child_link = child_links.get(current)
                    
                    for j, p in parent_links.items():
                        if p == current_child_link and j not in descendants[joint_name]:
                            descendants[joint_name].append(j)
                            to_process.append(j)
    
    return target_joints, descendants

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_list', type=str, default='configs/partnet.json')  
    parser.add_argument('--target_info', type=str, default='configs/partnet_target.json')  
    parser.add_argument('--input_dir', type=str, default='datasets/PartNet_raw')  
    parser.add_argument('--output_dir', type=str, default='datasets/PartNet')  
    args = parser.parse_args()
     
    print(f"Processing PartNet joints...")
    
    # Preprocess all PartNet models
    for model_id in tqdm(os.listdir(args.input_dir)):

        urdf_path = f'{args.input_dir}/{model_id}/mobility.urdf'
        meta_path = f'{args.input_dir}/{model_id}/meta.json'
        joint_meta_path = f'{args.input_dir}/{model_id}/mobility_v2.json'
        output_dir = f'{args.input_dir}/{model_id}'   
        joints_raw, descendants = parse_urdf(urdf_path)
        
        joints = []
        with open(meta_path, 'r') as f:
            meta_info = json.load(f)
        with open(joint_meta_path, 'r') as f:
            joint_meta_info = json.load(f) 

        for joint_name, joint_type in joints_raw:
            id = joint_name.split('_')[-1]
            for joint_meta in joint_meta_info:
                if joint_meta['id'] == int(id):
                    joint_tag = joint_meta['name']
            joints.append((joint_name, joint_type, joint_tag))

        output = { 
            "category": meta_info['model_cat'],
            "joints": joints,
            "descendants": descendants
        }

        with open(f'{output_dir}/joints.json', 'w') as f:
            json.dump(output, f, indent=4)
    
    # Preprocess target models
    with open(args.data_list) as f:
        data_info = json.load(f)
    with open(args.target_info) as f:
        target_info = json.load(f)
      
    model_ids = data_info['total_obj_ids']

    for model_id in tqdm(model_ids):

        with open(f'{args.input_dir}/{model_id}/joints.json', 'r') as f:
            joints_info = json.load(f)

        category = joints_info['category']
        target_joint_type = target_info[category]['joint_type']
        target_joint_tags = target_info[category]['joint_tags']

        valid_joints = []
        for joint in joints_info['joints']:
            joint_name, joint_type, joint_tag = joint
            if joint_type == target_joint_type and joint_tag in target_joint_tags:
                valid_joints.append(joint_name)

        joints_info['joints'] = valid_joints
        os.makedirs(f'{args.output_dir}/{model_id}', exist_ok=True)
        with open(f'{args.output_dir}/{model_id}/joints.json', 'w') as f:
            json.dump(joints_info, f, indent=4)
        os.system(f"cp {args.input_dir}/{model_id}/mobility.urdf {args.output_dir}/{model_id}/mobility.urdf")
        
import json
import numpy as np
import argparse

from pipelines.render import run_rendering

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_list', type=str, default='configs/partnet.json')  
    parser.add_argument('--input_dir', type=str, default='datasets/PartNet') 
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
        mesh_paths = [f"{mesh_dir}/gt_mesh/{i:02d}.glb" for i in range(args.num_states)]
        output_dir = f"{args.output_dir}/{model_id}" 
        run_rendering(mesh_paths, output_dir)

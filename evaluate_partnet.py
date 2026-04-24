import argparse
import tqdm
import os
import numpy as np
import os.path as osp
import json 
import random
import torch
import trimesh 
import mathutils

from eval_utils.neucon_eval_utils import eval_mesh 
from eval_utils.utils_3d import rotz_np, transform_points, Rt_to_pose, open3d_icp_api 
from eval_utils.render import generate_opengl_view_matrices_on_sphere, transform_bpy_mesh, eval_rendering_bpy
from eval_utils.clip import eval_clip_similarity
from eval_utils.joints import Joint, eval_joint

def auto_align_mesh_with_scale_and_rotz(gt_mesh, pred_mesh, name, dbg=False, ICP_POINTS=2000,
                                        scale_min=0.7,scale_max=1.3,evaltime=False):

    rot_angles = np.linspace(0, 360, 12)
    scales = np.linspace(scale_min, scale_max, 10)

    cfgs = []

    results, results2 = [], []
    for rot_angle in tqdm.tqdm(rot_angles, disable=True):
        for scale in tqdm.tqdm(scales, leave=False, disable=True):
            cfgs.append((rot_angle, scale))
            rotz = Rt_to_pose(rotz_np(np.deg2rad(rot_angle))[0])
            tmp_mesh = trimesh.Trimesh(pred_mesh.vertices, pred_mesh.faces)
            tmp_mesh.vertices = transform_points(tmp_mesh.vertices, rotz)
            tmp_mesh.apply_scale(gt_mesh.bounding_box.extents.max() / tmp_mesh.bounding_box.extents.max())
            tmp_mesh.vertices *= scale
            tmp_mesh_pts = tmp_mesh.sample(ICP_POINTS)
            gt_mesh_pts = gt_mesh.sample(ICP_POINTS)
            result = open3d_icp_api(tmp_mesh_pts, gt_mesh_pts, thresh=0.08 * gt_mesh.bounding_box.extents.max(),
                                    return_tsfm_only=False)
            result = open3d_icp_api(tmp_mesh_pts, gt_mesh_pts, thresh=0.05 * gt_mesh.bounding_box.extents.max(),
                                    init_Rt=result.transformation,
                                    return_tsfm_only=False)
            result2 = open3d_icp_api(gt_mesh_pts, tmp_mesh_pts, thresh=0.08 * gt_mesh.bounding_box.extents.max(),
                                     return_tsfm_only=False)
            result2 = open3d_icp_api(gt_mesh_pts, tmp_mesh_pts, thresh=0.05 * gt_mesh.bounding_box.extents.max(),
                                     init_Rt=result2.transformation, return_tsfm_only=False)
           
            results.append(result)
            results2.append(result2)

    fitnesses = [r.fitness for r in results]
    fitnesses2 = [r.fitness for r in results2]
    fitnesses_all = np.array(fitnesses) + np.array(fitnesses2)
    best_idx = np.argmax(fitnesses_all)

    final_tsfm = results[best_idx].transformation
    final_rotz, final_scale = cfgs[best_idx]
    
    rotz = Rt_to_pose(rotz_np(np.deg2rad(final_rotz))[0])
    pred_mesh.vertices = transform_points(pred_mesh.vertices, rotz)
    scale1 = gt_mesh.bounding_box.extents.max() / pred_mesh.bounding_box.extents.max()
    pred_mesh.apply_scale(scale1)
    pred_mesh.vertices *= final_scale
    scale = final_scale * scale1
    pred_mesh.vertices = transform_points(pred_mesh.vertices, final_tsfm)
    
    return pred_mesh, scale, final_rotz, final_tsfm

if __name__ == '__main__':
 
    args = argparse.ArgumentParser()
    args.add_argument('--dbg', action='store_true', help='Debug mode')
    args.add_argument('--dataset_folder', type=str, default='datasets/PartNet')
    args.add_argument('--pred_folder', type=str, default='outputs')
    args.add_argument('--res_folder', type=str, default='evaluations/PartNet')
    args.add_argument('--test_id', type=str, default='100214')
    args.add_argument('--num_states', type=int, default=6, help='Number of states')
    args.add_argument('--num_cams', type=int, default=5, help='Number of cameras')
    args = args.parse_args()
     
    dataset_folder = args.dataset_folder
    pred_folder = args.pred_folder
    test_id = args.test_id
    
    methods = ['ours']
    metrics = ['fscore', 'dist1', 'dist2', 'prec', 'recal']

    TOTAL_QPOS_NUM = args.num_states
    EACH_QPOS_CAM_NUM = args.num_cams
     
    os.makedirs(args.res_folder, exist_ok=True)
     
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    has_aligned = False

    # Sample cameras for rendering
    camera_distance = 1.7
    camera_poses = generate_opengl_view_matrices_on_sphere(TOTAL_QPOS_NUM * EACH_QPOS_CAM_NUM, camera_distance) 
    camera_sources = []
    for i, camera_pose in enumerate(camera_poses):
        inv_view = np.linalg.inv(camera_pose)
        camera_pos = inv_view[:3, 3]
        if i % EACH_QPOS_CAM_NUM == 0:
            azi = np.deg2rad(15)
            height = np.sin(np.deg2rad(20)) * camera_distance
            x_pos = np.sin(azi) * camera_distance
            y_pos = -np.cos(azi) * camera_distance
            camera_pos = [x_pos, y_pos, height]
        camera_sources.append(mathutils.Vector((camera_pos[0], camera_pos[1], camera_pos[2])))
    
    res = {}
    res_path = osp.join(args.res_folder, f'{test_id}.json') 
    align_path = f'{args.res_folder}/aligns/{test_id}.json'
    render_path = f'{args.res_folder}/renderings/{test_id}'
    os.makedirs(osp.dirname(align_path), exist_ok=True)
    os.makedirs(osp.dirname(render_path), exist_ok=True)

    if os.path.exists(align_path):
        with open(align_path, 'r') as f:
            align_json = json.load(f)
        has_aligned = True
    else:
        align_json = {}
    
    if not has_aligned:
        align_json['gt'] = []

    for method in methods:

        print(f"Evaluating {test_id} with {method}")

        item_res = {}

        if not has_aligned:
            align_json[method] = [] 

        # Load joint info
        gt_joint_info_path = osp.join(dataset_folder, test_id, 'joint_info.json')
        with open(gt_joint_info_path, 'r') as f:
            gt_joint_info = json.load(f)[0]
        gt_joint = Joint(gt_joint_info, method='gt')
        
        if method == 'ours':
            pred_joint_path = osp.join(pred_folder, test_id, 'sds_output', 'joint_info.json')
        else:
            pred_joint_path = osp.join(pred_folder, test_id, method, 'joint_info.json')
        with open(pred_joint_path, 'r') as f:
            pred_joint_info = json.load(f)[0]
        pred_joint = Joint(pred_joint_info, method=method)

        for qpos_id in tqdm.tqdm(range(TOTAL_QPOS_NUM)):

            # Load GT mesh & joint info
            gt_mesh_path = osp.join(dataset_folder, test_id, 'gt_mesh', f'{TOTAL_QPOS_NUM - 1 - qpos_id:02d}.glb')
            gt_mesh: trimesh.Trimesh = trimesh.load(gt_mesh_path, force='mesh')
            gt_mesh_extents = 1 / gt_mesh.extents.max()
            gt_mesh.apply_scale(1 / gt_mesh.extents.max())
            gt_mesh_centroid = gt_mesh.centroid
            gt_mesh.apply_translation(-gt_mesh.centroid) 
            transform_bpy_mesh(gt_mesh_path, f'{render_path}/gt/gt_mesh_aligned_{qpos_id:02d}.glb', 
                np.array(gt_mesh_extents), np.array(gt_mesh_centroid), gt=True)  
            
            if qpos_id == 0: # Use the qpos=1 state to align the joints
                # print(f"[GT Joint] Before alignment: {gt_joint.axis_orig}, {gt_joint.axis_dir}")
                gt_joint.apply_scale(gt_mesh_extents)
                gt_joint.apply_translation(-gt_mesh_centroid)
                # print(f"[GT Joint] After alignment: {gt_joint.axis_orig}, {gt_joint.axis_dir}")

            # Render GT mesh 
            if not os.path.exists(f'{render_path}/gt/qpos_{qpos_id:02d}/cam_04.png'):
                gt_images = eval_rendering_bpy(camera_sources[qpos_id*EACH_QPOS_CAM_NUM:(qpos_id+1)*EACH_QPOS_CAM_NUM],
                    render_path, 'gt', qpos_id, f'{render_path}/gt/gt_mesh_aligned_{qpos_id:02d}.glb')   
                gt_rendered_flag = True

            if not has_aligned and len(align_json['gt']) < TOTAL_QPOS_NUM:
                align_json['gt'].append({'scale1': gt_mesh_extents.tolist(),
                                            'translation1': gt_mesh_centroid.tolist()})
            
            if method == 'ours': # Default to fetch fromoutput folder
                pred_mesh_path = osp.join(pred_folder, test_id, 
                    'sds_output', 'states', f'qpos_{TOTAL_QPOS_NUM - 1 - qpos_id:02d}.glb')
            else:
                pred_mesh_path = osp.join(pred_folder, test_id, method, 
                    'sds_output', 'states', f'qpos_{TOTAL_QPOS_NUM - 1 - qpos_id:02d}.glb')

            if not osp.exists(pred_mesh_path):
                print(f"Prediction mesh {pred_mesh_path} does not exist")
                continue
                
            pred_mesh: trimesh.Trimesh = trimesh.load(pred_mesh_path, force='mesh')
            pred_mesh_extents = 1 / pred_mesh.extents.max()
            pred_mesh.apply_scale(1 / pred_mesh.extents.max())
            pred_mesh_centroid = pred_mesh.centroid
            pred_mesh.apply_translation(-pred_mesh.centroid) 

            if not has_aligned: 
                pred_mesh, scale, final_rotz, final_tsfm = auto_align_mesh_with_scale_and_rotz(gt_mesh, pred_mesh, name=f"mesh_ours_{qpos_id:02d}",
                                                                        scale_min=0.5, scale_max=1.5)
                rotz = Rt_to_pose(rotz_np(np.deg2rad(final_rotz))[0])
            else:
                scale = align_json[method][qpos_id]['scale2']
                rotz = align_json[method][qpos_id]['rotz']
                final_tsfm = align_json[method][qpos_id]['final_tsfm'] 
                pred_mesh.apply_transform(rotz)
                pred_mesh.apply_scale(scale)
                pred_mesh.apply_transform(final_tsfm) 

            if not has_aligned:
                align_json[method].append({'scale1': pred_mesh_extents.tolist(), 
                                            'translation1': pred_mesh_centroid.tolist(),
                                            'scale2': scale.tolist(), 
                                            'rotz': rotz.tolist(), 
                                            'final_tsfm': final_tsfm.tolist()})

            # Render pred mesh
            transform_bpy_mesh(pred_mesh_path, f'{render_path}/{method}/{method}_mesh_aligned_{qpos_id:02d}.glb', 
                np.array(pred_mesh_extents), np.array(pred_mesh_centroid), rotz=np.array(rotz),
                final_tsfm=np.array(final_tsfm), scale=np.array(scale), gt=False)
            pred_images = eval_rendering_bpy(camera_sources[qpos_id*EACH_QPOS_CAM_NUM:(qpos_id+1)*EACH_QPOS_CAM_NUM],
                render_path, method, qpos_id, f'{render_path}/{method}/{method}_mesh_aligned_{qpos_id:02d}.glb')   
 
            if qpos_id == 0:
                # print(f"[Pred Joint] Before alignment: {pred_joint.axis_orig}, {pred_joint.axis_dir}")
                pred_joint.apply_scale(pred_mesh_extents)
                pred_joint.apply_translation(-pred_mesh_centroid) 
                pred_joint.apply_transform(np.array(rotz))
                pred_joint.apply_scale(np.array(scale))
                pred_joint.apply_transform(np.array(final_tsfm))
                # print(f"[Pred Joint] After alignment: {pred_joint.axis_orig}, {pred_joint.axis_dir}")

            # Evaluate geometric metrics                                
            results = eval_mesh(pred_mesh, gt_mesh, threshold=.05 * 1.0, down_sample=None)    

            for metric in metrics:
                if metric not in item_res:
                    item_res[metric] = []
                item_res[metric].append(results[metric].item())
        
        # Compute mean geometric metrics
        fail_flag = True if len(item_res) == 0 else False
        for metric in metrics:
            item_res[metric] = np.mean(item_res[metric]) if not fail_flag else np.nan

        # Evaluate clip similarity 
        mean_clip_sim = eval_clip_similarity(render_path, method, TOTAL_QPOS_NUM, EACH_QPOS_CAM_NUM)
        item_res['clip_sim'] = mean_clip_sim

        # Evaluate joint metrics
        joint_res = eval_joint(pred_joint, gt_joint)
        item_res['joint_axis_err'] = joint_res['joint_axis_err']
        item_res['joint_orig_err'] = joint_res['joint_orig_err']

        # Update results
        res[test_id] = {method: item_res} if test_id not in res else {**res[test_id], **{method: item_res}} 
                
    # Save alignments
    os.makedirs(osp.dirname(align_path), exist_ok=True)
    with open(align_path, 'w') as f:
        json.dump(align_json, f, indent=4)

    # Save results as json
    with open(res_path, "w") as f:
        json.dump(res, f, indent=4) 

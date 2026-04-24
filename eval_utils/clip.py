import numpy as np
import clip
import torch

from PIL import Image

def eval_clip_similarity(base_dir, method, num_states, num_cams):

    perceptor, preprocess = clip.load('ViT-L/14@336px', jit=False) 
    perceptor = perceptor.eval().requires_grad_(False).cuda()

    clip_sims = []

    for qpos_id in range(num_states):
        for cam_pos in range(num_cams):

            pred_path = f'{base_dir}/{method}/qpos_{qpos_id:02d}/cam_{cam_pos:02d}.png'
            gt_path = f'{base_dir}/gt/qpos_{qpos_id:02d}/cam_{cam_pos:02d}.png'
            img_pred = Image.open(pred_path)
            img_gt = Image.open(gt_path) 
            img_pred = preprocess(img_pred).unsqueeze(0).cuda()
            encoded_pred = perceptor.encode_image(img_pred)
            img_gt = preprocess(img_gt).unsqueeze(0).cuda()
            encoded_gt = perceptor.encode_image(img_gt)
            cosine = torch.cosine_similarity(torch.mean(encoded_pred, dim=0),
                                            torch.mean(encoded_gt, dim=0), dim=0) 
            clip_sims.append(cosine.item())

    return np.mean(clip_sims)
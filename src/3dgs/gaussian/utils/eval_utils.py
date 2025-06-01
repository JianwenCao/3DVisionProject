import json
import os

import cv2
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from util.utils import Log
from gaussian.renderer import render
from gaussian.utils.loss_utils import ssim, psnr
from gaussian.utils.camera_utils import Camera
import shutil


def eval_rendering(
    gtimages,
    gtdepthdir,
    traj,
    gaussians,
    save_dir,
    background,
    projection_matrix,
    K,
    kf_idx,
    iteration="final",
):
    gtdepths = sorted(os.listdir(gtdepthdir)) if gtdepthdir is not None else None
    psnr_array, ssim_array, lpips_array, l1_array = [], [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")
    
    image_save_dir = f'{save_dir}/renders/image_{iteration}'
    depth_save_dir = f'{save_dir}/renders/depth_{iteration}'
    # vis_save_dir = f'{save_dir}/renders/vis_{iteration}'  

    if os.path.exists(image_save_dir):
        shutil.rmtree(image_save_dir)
    if os.path.exists(depth_save_dir):
        shutil.rmtree(depth_save_dir)
    # if os.path.exists(vis_save_dir):
    #     shutil.rmtree(vis_save_dir)

    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(depth_save_dir, exist_ok=True)
    # os.makedirs(vis_save_dir, exist_ok=True)
    
    for i, (idx, image) in enumerate(gtimages.items()):
        if idx % 5 != 0 and idx not in kf_idx and i != len(gtimages) - 1:
            continue
        frame = Camera.init_from_tracking(image.squeeze()/255.0, None, None, traj[idx], idx, projection_matrix, K)
        gtimage = frame.original_image.cuda()

        rendering = render(frame, gaussians, background)
        image = torch.clamp(rendering["render"], 0.0, 1.0)
        depth = rendering["depth"].detach().squeeze().cpu().numpy()

        if gtdepthdir is not None:
            gtdepth = cv2.imread(os.path.join(gtdepthdir, gtdepths[idx]), cv2.IMREAD_ANYDEPTH) / 6553.5 # 1000.
            gtdepth = cv2.resize(gtdepth, (depth.shape[-1], depth.shape[-2]), interpolation=cv2.INTER_NEAREST)
            invalid = gtdepth <= 0
            depth[invalid] = 0

        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{image_save_dir}/{idx:06d}.jpg', pred)
        cv2.imwrite(f'{depth_save_dir}/{idx:06d}.png', np.clip(depth*3276.75, 0, 65535).astype(np.uint16)) # 3276.75 mapping 20m to uint16
        # vis = np.concatenate((pred, cv2.imread(f'{save_dir}/renders/depth_{iteration}/{idx:06d}.png')), axis=0)
        # cv2.imwrite(f'{vis_save_dir}/{idx:06d}.jpg', vis)

        if gtdepthdir is not None and idx in kf_idx:
            l1_array.append(np.abs(gtdepth[depth > 0] - depth[depth > 0]).mean().item()) 

        # if idx in kf_idx:
        #     continue
        mask = gtimage > 0
        psnr_score = psnr((image[mask]).unsqueeze(0), (gtimage[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gtimage).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gtimage).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    output["mean_l1"] = float(np.mean(l1_array)) if l1_array else 0

    Log(f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}, depth l1: {output["mean_l1"]}', tag="Eval")

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    os.makedirs(psnr_save_dir, exist_ok=True)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output

def eval_rendering_kf(
    viewpoints,
    gaussians,
    save_dir,
    background,
    iteration="final",
):
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")
    for frame in viewpoints.values():
        gtimage = frame.original_image.cuda()

        rendering = render(frame, gaussians, background)
        image = (torch.exp(frame.exposure_a)) * rendering["render"] + frame.exposure_b
        image = torch.clamp(image, 0.0, 1.0)

        mask = gtimage > 0
        psnr_score = psnr((image[mask]).unsqueeze(0), (gtimage[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gtimage).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gtimage).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(f'kf mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}', tag="Eval")

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    os.makedirs(psnr_save_dir, exist_ok=True)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result_kf.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output

def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    print('saved to ', point_cloud_path)

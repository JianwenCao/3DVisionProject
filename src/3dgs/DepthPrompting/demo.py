from collections import OrderedDict
from model.ours.depth_prompt_main import depthprompting
from config import args as args_config
import torchvision
import numpy as np
np.set_printoptions(threshold = np.inf) 
np.set_printoptions(suppress = True)
import torch
torch.set_printoptions(sci_mode=False)
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from lietorch import SE3
from matplotlib import pyplot as plt
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np


def project_lidar_to_depth(points, pose, intrinsic, H=512, W=640, max_depth = 80):
    """TODO: Distortion is not done yet"""
    # Project LiDAR points in world coordinates to depth map
    # LiDAR coordinate system: x down (depth), y right, z up, right-handed
    # Mapping: camera X = LiDAR y, Y = -LiDAR z, Z = LiDAR x
    fx, fy, cx, cy = intrinsic

    # Adjust LiDAR points: x=depth(down), y=right, z=up -> camera X=right, Y=down, Z=forward
    points_xyz = points[:, :3]
    points_homo = torch.cat([points_xyz, torch.ones(points_xyz.shape[0], 1, device=points.device)], dim=1)

    # Transform from world to camera coordinates
    def pose_to_extrinsic(tx, ty, tz, qx, qy, qz, qw):
        def quaternion_to_rotation_matrix(qx, qy, qz, qw):
            norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
            qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
            R = np.array([
                [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
                [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
            ])
            return R
        R_A = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        t_A = np.array([tx, ty, tz])
        R_B = R_A
        t_B = t_A
        '''1670047 with'''
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_B
        extrinsic[:3, 3] = t_B
        return torch.tensor(extrinsic, dtype=torch.float32)

    # w2c = SE3(pose).matrix()
    tx, ty, tz, qx, qy, qz, qw = pose.tolist()
    w2c = pose_to_extrinsic(tx, ty, tz, qx, qy, qz, qw)
    cam_points = torch.matmul(points_homo, torch.inverse(w2c).T)
    cam_points_adj = torch.zeros_like(cam_points)
    cam_points_adj[:, 0] = -cam_points[:, 1]  # Camera X = -LiDAR y
    cam_points_adj[:, 1] = -cam_points[:, 2]  # Camera Y = -LiDAR z
    cam_points_adj[:, 2] = cam_points[:, 0]  # Camera Z = LiDAR x
    cam_points = cam_points_adj

    # Filter out negative depths
    def print_tensor_density(tensor, num_bins=100):
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        hist = torch.histc(tensor, bins=num_bins, min=min_val, max=max_val)
        for i in range(num_bins):
            bin_start = min_val + (max_val - min_val) * i / num_bins
            bin_end = min_val + (max_val - min_val) * (i + 1) / num_bins
            print(f"Range [{bin_start:.2f}, {bin_end:.2f}): count = {int(hist[i])}")
    depths = cam_points[:, 2]
    # print_tensor_density(depths)

    valid = (depths > 0.1) & (depths < max_depth)
    cam_points = cam_points[valid]
    depths = depths[valid]
    print(f"Valid points because of depth: {valid.sum().item()}/{points.shape[0]}")

    # Normalized coordinates
    x_norm = cam_points[:, 0] / depths
    y_norm = cam_points[:, 1] / depths

    # Temporarily disable distortion
    x_dist = x_norm
    y_dist = y_norm

    # Pixel coordinates
    u = fx * x_dist + cx
    v = fy * y_dist + cy


    # Filter valid points
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid].long()
    v = v[valid].long()
    depths = depths[valid]
    cam_points = cam_points[valid]
    print(f"Valid points because of uv: {valid.sum().item()}/{points.shape[0]}")
    
    if len(depths) == 0:
        print("No points with positive depth after transformation!")
        return torch.zeros(H, W, device="cpu")
    else:
        print(f"z range: [{depths.min().item()}, {depths.max().item()}]")
    # Generate depth map
    depth_map = torch.zeros(H, W, device=points.device)
    if valid.sum() > 0:
        sorted_indices = torch.argsort(depths, descending=True)
        u = u[sorted_indices]
        v = v[sorted_indices]
        depths = depths[sorted_indices]
        depth_map[v, u] = depths
    return depth_map, valid.sum().item()


if __name__ == '__main__':
    args = args_config
    args.height, args.width = 512, 640
    args.max_depth = 30
    args.prop_kernel = 9
    args.prop_time = 18
    args.conf_prop = True
    args.pretrain = "pretrained/Depthprompting_depthformer_kitti.tar"

    t_rgb = T.Compose([
        T.Resize(min(args.height, args.width), Image.BICUBIC), 
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    t_dep = T.Compose([
        T.Resize(min(args.height, args.width), Image.BICUBIC), 
    ])

    model = depthprompting(args=args).cuda()
    checkpoint = torch.load(args.pretrain)
    try:
        loaded_state_dict = checkpoint['state_dict']
    except:
        loaded_state_dict = checkpoint
    new_state_dict = OrderedDict()
    for n, v in loaded_state_dict.items():
        name = n.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    total_valid_num = 0
    for i in range(187):
        print(f"Processing frame {i}...")
        image = Image.open(f"../red_sculpture/frame{i}/image.png")
        points = torch.tensor(np.load(f"../red_sculpture/frame{i}/points.npy"))
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        print(f"x range: [{x.min().item()}, {x.max().item()}]")
        print(f"y range: [{y.min().item()}, {y.max().item()}]")
        print(f"z range: [{z.min().item()}, {z.max().item()}]")
        pose = torch.load(f"../red_sculpture/frame{i}/pose.pt")
        intrinsic = torch.tensor(np.loadtxt("../red_sculpture/intrinsic.txt"))
        # print(f"pose:\n{pose}")
        # print(f"points:\n{points[:5]}")
        # print(f"intrinsic:\n{intrinsic}")

        depth_map, valid_num = project_lidar_to_depth(points, pose, intrinsic, max_depth=args.max_depth)
        total_valid_num += valid_num
        torchvision.utils.save_image(depth_map, f"output/depth_map{i}.png")

        sample = {"rgb": t_rgb(image).unsqueeze(0).cuda(), "dep":t_dep(depth_map.unsqueeze(0).unsqueeze(0)).cuda()}
        depth_map_pred = model(sample)
        print(depth_map_pred.keys())
        pred = depth_map_pred['pred']
        pred[pred<0.1] = 0.1
        print(f"depth map pred shape: {pred.shape}, min: {pred.min()}, max: {pred.max()}")
        torch.save(pred, f"output/depthmap{i}.pt")
        torchvision.utils.save_image((pred - pred.min()) / (pred.max() - pred.min()), f"output/depth_map_pred{i}.png")
        print()
    print(f"Total valid points: {total_valid_num}")

    

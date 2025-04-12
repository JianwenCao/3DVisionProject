import multiprocessing as mp
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from gs_backend import GSBackEnd
from util.utils import load_config
from tqdm import tqdm
from transformers import pipeline
from PIL import Image
import torchvision
import logging
import time
from scipy.interpolate import griddata

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
    T = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    R_B = T @ R_A @ T.T
    t_B = T @ t_A
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R_B
    extrinsic[:3, 3] = t_B
    return torch.tensor(extrinsic, dtype=torch.float32)

def extrinsic_to_pose(T):
    t = T[:3, 3]
    R_mat = T[:3, :3]
    r = R.from_matrix(R_mat)
    q = r.as_quat()
    return torch.tensor([t[0], t[1], t[2], q[0], q[1], q[2], q[3]])

def project_lidar_to_depth(points, pose, intrinsic, H=512, W=640):
    """将 LiDAR 点云投影到相机深度图，适应室外场景"""
    fx, fy, cx, cy = intrinsic
    points_xyz = points[:, :3]
    points_homo = torch.cat([points_xyz, torch.ones(points_xyz.shape[0], 1, device=points.device)], dim=1)

    tx, ty, tz, qx, qy, qz, qw = pose.tolist()
    w2c = pose_to_extrinsic(tx, ty, tz, qx, qy, qz, qw)
    cam_points = torch.matmul(points_homo, torch.inverse(w2c).T)

    cam_points_adj = torch.zeros_like(cam_points)
    cam_points_adj[:, 0] = -cam_points[:, 1]  # 相机 X = -LiDAR y
    cam_points_adj[:, 1] = -cam_points[:, 2]  # 相机 Y = -LiDAR z
    cam_points_adj[:, 2] = cam_points[:, 0]   # 相机 Z = LiDAR x
    cam_points = cam_points_adj

    depths = cam_points[:, 2]
    valid = (depths > 0.1)  # 移除 max_depth 限制
    cam_points = cam_points[valid]
    depths = depths[valid]
    print(f"有效点数 (深度过滤): {valid.sum().item()}/{points.shape[0]}")

    x_norm = cam_points[:, 0] / depths
    y_norm = cam_points[:, 1] / depths
    u = fx * x_norm + cx
    v = fy * y_norm + cy

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid].long()
    v = v[valid].long()
    depths = depths[valid]
    print(f"有效点数 (UV 过滤): {valid.sum().item()}/{points.shape[0]}")

    if len(depths) == 0:
        print("没有有效点生成深度图！")
        return torch.zeros(H, W, device="cpu")

    depth_map = torch.zeros(H, W, device=points.device)
    if valid.sum() > 0:
        sorted_indices = torch.argsort(depths, descending=True)
        u = u[sorted_indices]
        v = v[sorted_indices]
        depths = depths[sorted_indices]
        depth_map[v, u] = depths
    return depth_map

def compute_scale_and_absolute_depth(disparity_map, lidar_depth_map, H=512, W=640):
    """从视差图转换为绝对深度图，适应室外场景"""
    # 处理负值和 0：设为小正值，避免除以 0
    disparity_map = np.where(disparity_map <= 0, 1e-6, disparity_map)

    # 获取 LiDAR 深度图的有效点
    valid_mask = lidar_depth_map > 0
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    u_valid = u_grid[valid_mask].flatten()
    v_valid = v_grid[valid_mask].flatten()
    Z_true = lidar_depth_map[valid_mask].flatten()

    # 从视差图中插值对应值
    D_values = griddata(
        (u_grid.flatten(), v_grid.flatten()),
        disparity_map.flatten(),
        (u_valid, v_valid),
        method='linear'
    )

    # 计算尺度因子 (Z_true = scale / D)
    valid_mask = ~np.isnan(D_values) & (D_values > 1e-6)
    scale = np.median(Z_true[valid_mask] * D_values[valid_mask])  # scale = Z_true * D
    print(f"尺度因子: {scale}")

    # 生成绝对深度图
    absolute_depth_map = scale / disparity_map  # Z = scale / D
    absolute_depth_map[disparity_map <= 1e-6] = np.inf  # 无效区域设为无穷远
    return scale, absolute_depth_map

def data_loader(queue, num_frames, dataset_path, config_path):
    pipe = pipeline(task="depth-estimation", model="xingyang1/Distill-Any-Depth-Large-hf", use_fast=True)
    batch_size = 10
    batch_data = {"images": [], "poses": [], "intrinsics": [], "depths": [], "tstamp": [], "scales": []}

    for i in tqdm(range(num_frames), desc="Loading data"):
        image_pil = Image.open(f"{dataset_path}/frame{i}/image.png")
        image = torch.tensor(np.array(image_pil)).permute(2, 0, 1).cpu()

        res = pipe(image_pil)
        depth_image, disparity_map = res["depth"], res["predicted_depth"]
        depth_image.save(f"{dataset_path}/frame{i}/depth.png")
        disparity_map = (disparity_map + 20) / 20
        disparity_map = disparity_map.numpy()

        tx, ty, tz, qx, qy, qz, qw = torch.load(f"{dataset_path}/frame{i}/pose.pt").tolist()
        extrinsic = pose_to_extrinsic(tx, ty, tz, qx, qy, qz, qw)
        pose = extrinsic_to_pose(extrinsic)
        intrinsics = torch.tensor(np.loadtxt(f"{dataset_path}/intrinsics.txt"))
        lidar_points = torch.tensor(np.load(f"{dataset_path}/frame{i}/points.npy"))
        lidar_depth_map = project_lidar_to_depth(lidar_points, pose, intrinsics)

        scale, absolute_depth_map = compute_scale_and_absolute_depth(disparity_map, lidar_depth_map.numpy())
        absolute_depth_map = torch.tensor(absolute_depth_map, dtype=torch.float32)

        # np.save(f"{dataset_path}/frame{i}/absolute_depth.npy", absolute_depth_map.numpy())

        batch_data["images"].append(image)
        batch_data["poses"].append(pose)
        batch_data["intrinsics"].append(intrinsics)
        batch_data["depths"].append(absolute_depth_map.cpu())
        batch_data["tstamp"].append(i)
        batch_data["scales"].append(scale)

        if len(batch_data["tstamp"]) == batch_size or i == num_frames - 1:
            packet = {
                'viz_idx': torch.tensor(batch_data["tstamp"]),
                'tstamp': torch.tensor(batch_data["tstamp"]),
                'poses': torch.stack(batch_data["poses"]),
                'images': torch.stack(batch_data["images"]),
                'depths': torch.stack(batch_data["depths"]),
                'intrinsics': torch.stack(batch_data["intrinsics"]),
                'pose_updates': None,
                'scale_updates': torch.tensor(batch_data["scales"]),
                'is_last': (i == num_frames - 1)
            }
            queue.put(packet)
            batch_data = {"images": [], "poses": [], "intrinsics": [], "depths": [], "tstamp": [], "scales": []}

    while not queue.empty():
        time.sleep(1)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    torchvision.disable_beta_transforms_warning()

    # num_frames = 187
    # dataset_path = "../../dataset/red_sculpture"
    # config_path = "../../config/config.yaml"

    # Elliot testing
    num_frames = 100 # normally 1013
    dataset_path = "../HI-SLAM2/data/red_sculpture_dense_fixed"
    config_path = "../../config/config_lidar.yaml"

    queue = mp.Queue(maxsize=8)
    config = load_config(config_path)
    gs = GSBackEnd(config, save_dir="./output", use_gui=True)

    loader_process = mp.Process(target=data_loader, args=(queue, num_frames, dataset_path, config_path))
    loader_process.start()

    pbar = tqdm(total=num_frames, desc="Processing frames")

    processed_frames = 0
    while processed_frames < num_frames:
        packet = queue.get()
        gs.process_track_data(packet)
        batch_size = len(packet['tstamp'])
        processed_frames += batch_size
        pbar.update(batch_size)

        if packet['is_last']:
            updated_poses = gs.finalize()
            break

    pbar.close()
    loader_process.join()
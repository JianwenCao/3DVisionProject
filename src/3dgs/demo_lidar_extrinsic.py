import multiprocessing as mp
import cv2
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from gs_backend_lidar import GSBackEnd
from util.utils import load_config
from tqdm import tqdm
from PIL import Image
import torchvision
import time
import argparse

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    R = np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy]
    ])
    return R

def quat_to_transform(tx, ty, tz, qx, qy, qz, qw, T_ab):
    """
    Convert coordinate system A pose (tx, ty, tz, qx, qy, qz, qw) to transform in B frame.
    Input:
    - tx, ty, tz, qx, qy, qz, qw: translation /quaternion in original frame
    - T_ab: transformation matrix from A to B
    """
    assert type(T_ab) == torch.Tensor, "Transformation matrix must be a torch tensor"
    assert T_ab.shape == (4, 4), "Transformation matrix must be 4x4"

    R_A = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    T_A = torch.eye(4, dtype=T_ab.dtype, device=T_ab.device)
    T_A[:3, :3] = torch.tensor(R_A, dtype=T_ab.dtype, device=T_ab.device)
    T_A[:3, 3] = torch.tensor([tx, ty, tz], dtype=T_ab.dtype, device=T_ab.device)
    T_B = T_ab @ T_A @ torch.inverse(T_ab)
    return T_B

def transform_to_tensor(T):
    assert type(T) == torch.Tensor, "Transformation matrix must be a torch tensor"
    assert T.shape == (4, 4), "Transformation matrix must be 4x4"

    t = T[:3, 3]
    R_mat = T[:3, :3]
    R_mat = R_mat / np.linalg.norm(R_mat, axis=0)
    r = R.from_matrix(R_mat)
    q = r.as_quat()
    return torch.tensor([t[0], t[1], t[2], q[0], q[1], q[2], q[3]])

def transform_points(points, T_ab):
    """
    Transform XYZRGB points from frames A to B.
    Points dimension (N, 6), rows are points, columns 0-5 are x, y, z, r, g, b.
    Input:
    - points: (N, 6) tensor of points in frame A.
    - T_ab: (4, 4) transformation matrix tensor from A to B.
    """
    assert type(points) == torch.Tensor, "Points must be a torch tensor"
    assert points.shape[1] == 6, "Points must have 6 columns (x, y, z, r, g, b)"
    assert type(T_ab) == torch.Tensor, "Transformation matrix must be a torch tensor"
    assert T_ab.shape == (4, 4), "Transformation matrix must be 4x4"

    T = torch.eye(6, dtype=torch.float32)
    T[:4, :4] = T_ab
    transformed_points = (T @ points.T).T
    return transformed_points

def data_loader(queue, num_frames, dataset_path, coordinate_transform, intrinsics, K):
    # LiDAR to IMU
    extR = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    extT = torch.tensor([0.04165, 0.02326, -0.0284], dtype=torch.float32)

    # IMU to LiDAR
    R_li = extR.T  # R_li = R_il^T
    P_li = -R_li @ extT  # P_li = -R_il^T * P_il

    # LiDAR to Camera
    R_cl = torch.tensor([
        [0.00610193, -0.999863, -0.0154172],
        [-0.00615449, 0.0153796, -0.999863],
        [0.999962, 0.00619598, -0.0060598]
    ], dtype=torch.float32)
    P_cl = torch.tensor([0.0194384, 0.104689, -0.0251952], dtype=torch.float32)

    # IMU to Camera
    R_ci = R_cl @ R_li
    P_ci = R_cl @ P_li + P_cl
    T_ci = torch.eye(4, dtype=torch.float32)
    T_ci[:3, :3] = R_ci
    T_ci[:3, 3] = P_ci

    for i in tqdm(range(num_frames), desc="Loading data"):
        image_pil = Image.open(f"{dataset_path}/frame{i}/image.png")
        image = np.array(image_pil)
        if len(intrinsics) > 4:
            dist_coeffs = np.array(intrinsics[4:])
            image = cv2.undistort(image, K, dist_coeffs)
        image = torch.tensor(image).permute(2, 0, 1).cpu()

        # Load pose（IMU to World）in FAST-LIVO2 coordinate system
        tx, ty, tz, qx, qy, qz, qw = torch.load(f"{dataset_path}/frame{i}/pose.pt").tolist()
        # Transform pose's coordinate system into Hi-SLAM2's coordinate system
        T_wi = quat_to_transform(tx, ty, tz, qx, qy, qz, qw, coordinate_transform)

        # Compute camera to World
        T_wc = T_wi @ torch.inverse(T_ci)
        pose = transform_to_tensor(T_wc)

        # Load lidar data in FAST-LIVO2 coordinate system
        lidar_points = torch.tensor(np.load(f"{dataset_path}/frame{i}/points.npy"))
        # Transform points' coordinate system into Hi-SLAM2's coordinate system
        transformed_lidar_points = transform_points(lidar_points, coordinate_transform)

        packet = {
            'viz_idx': torch.tensor([i]),
            'tstamp': torch.tensor([i]),
            'poses': pose.unsqueeze(0),
            'images': image.unsqueeze(0),
            'points': [transformed_lidar_points],
            'intrinsics': intrinsics.unsqueeze(0),
            'pose_updates': None,
            'is_last': (i == num_frames - 1)
        }
        queue.put(packet)

    while not queue.empty():
        time.sleep(1)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    torchvision.disable_beta_transforms_warning()

    parser = argparse.ArgumentParser(
        description="Runs 3DGS pipeline on preprocessed synced sensor data."
    )
    parser.add_argument(
        "--dataset", "-d",
        default="../../dataset/CBD_Building_01_hi",
        help="Path to local dataset w.r.t to the current working directory."
    )
    parser.add_argument(
        "--config", "-c",
        default='../../config/config_lidar.yaml',
        help="Path to local config_lidar.yaml w.r.t to the current working directory."
    )
    parser.add_argument(
        "--num_frames", "-n", type=int,
        default=105,
        help="Number of frames to process."
    )
    args = parser.parse_args()

    config_path = args.config
    dataset_path = args.dataset
    num_frames = args.num_frames

    queue = mp.Queue(maxsize=8)
    config = load_config(config_path)
    gs = GSBackEnd(config, save_dir="./output", use_gui=True)

    # Fast-LIVO coordinate system to Hi-SLAM2 coordinate system
    T = torch.tensor([
        [0, -1, 0, 0],  # -y_fast = x_gs
        [0, 0, -1, 0],  # -z_fast = y_gs
        [1, 0, 0, 0],  # x_fast = z_gs
        [0, 0, 0, 1]
    ], dtype=torch.float32)
    intrinsics = torch.tensor(np.loadtxt(f"{dataset_path}/intrinsics.txt"))
    K = np.array([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]])

    loader_process = mp.Process(target=data_loader, args=(queue, num_frames, dataset_path, T, intrinsics, K))
    loader_process.start()

    pbar = tqdm(total=num_frames, desc="Processing frames")

    processed_frames = 0
    frame_counter = 0
    step = 1

    while processed_frames < num_frames:
        packet = queue.get()
        frame_counter += len(packet['tstamp'])

        if frame_counter % step == 0 or packet['is_last']:
            gs.process_track_data(packet)
            batch_size = len(packet['tstamp'])
            processed_frames += batch_size
            pbar.update(batch_size)

            if packet['is_last']:
                updated_poses = gs.finalize()
                gtimages, trajs = [], []
                for i in range(num_frames):
                    image_pil = Image.open(f"{dataset_path}/frame{i}/image.png")
                    image = np.array(image_pil)
                    if len(intrinsics) > 4:
                        dist_coeffs = np.array(intrinsics[4:])
                        image = cv2.undistort(image, K, dist_coeffs)
                    gtimages.append(torch.tensor(image).permute(2, 0, 1).cpu())
                    tx, ty, tz, qx, qy, qz, qw = torch.load(f"{dataset_path}/frame{i}/pose.pt").tolist()
                    # 计算相机到世界的变换
                    T_wi = quat_to_transform(tx, ty, tz, qx, qy, qz, qw, T)
                    extR = torch.tensor([
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]
                    ], dtype=torch.float32)
                    extT = torch.tensor([0.04165, 0.02326, -0.0284], dtype=torch.float32)
                    R_li = extR.T
                    P_li = -R_li @ extT
                    R_cl = torch.tensor([
                        [0.00610193, -0.999863, -0.0154172],
                        [-0.00615449, 0.0153796, -0.999863],
                        [0.999962, 0.00619598, -0.0060598]
                    ], dtype=torch.float32)
                    P_cl = torch.tensor([0.0194384, 0.104689, -0.0251952], dtype=torch.float32)
                    R_ci = R_cl @ R_li
                    P_ci = R_cl @ P_li + P_cl
                    T_ci = torch.eye(4, dtype=torch.float32)
                    T_ci[:3, :3] = R_ci
                    T_ci[:3, 3] = P_ci
                    T_wc = T_wi @ torch.inverse(T_ci)
                    trajs.append(T_wc.cuda())
                gs.eval_rendering({index: tensor for index, tensor in enumerate(gtimages)}, None, trajs,
                                  torch.arange(0, num_frames))
                break
        else:
            batch_size = len(packet['tstamp'])
            processed_frames += batch_size
            pbar.update(batch_size)

    pbar.close()
    loader_process.join()
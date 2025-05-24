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
import argparse
import time

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy]
    ])

def quat_to_transform(tx, ty, tz, qx, qy, qz, qw, T_ab):
    assert isinstance(T_ab, torch.Tensor), "Transformation matrix must be a torch tensor"
    assert T_ab.shape == (4, 4), "Transformation matrix must be 4x4"

    R_A = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    T_A = torch.eye(4, dtype=T_ab.dtype, device=T_ab.device)
    T_A[:3, :3] = torch.tensor(R_A, dtype=T_ab.dtype, device=T_ab.device)
    T_A[:3, 3] = torch.tensor([tx, ty, tz], dtype=T_ab.dtype, device=T_ab.device)
    return T_ab @ T_A @ torch.inverse(T_ab)

def transform_to_tensor(T):
    assert isinstance(T, torch.Tensor), "Transformation matrix must be a torch tensor"
    assert T.shape == (4, 4), "Transformation matrix must be 4x4"

    t = T[:3, 3]
    R_mat = T[:3, :3]
    R_mat = R_mat / np.linalg.norm(R_mat, axis=0)
    r = R.from_matrix(R_mat)
    q = r.as_quat()
    return torch.tensor([t[0], t[1], t[2], q[0], q[1], q[2], q[3]])

def transform_points(points, T_ab):
    assert isinstance(points, torch.Tensor), "Points must be a torch tensor"
    assert points.shape[1] == 6, "Points must have 6 columns (x, y, z, r, g, b)"
    assert isinstance(T_ab, torch.Tensor), "Transformation matrix must be a torch tensor"
    assert T_ab.shape == (4, 4), "Transformation matrix must be 4x4"

    T = torch.eye(6, dtype=torch.float32)
    T[:4, :4] = T_ab
    return (T @ points.T).T

def compute_imu_to_camera_transform(coordinate_transform):
    """Compute IMU to Camera transformation matrix."""
    # LiDAR to IMU
    extR = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    extT = torch.tensor([0.04165, 0.02326, -0.0284], dtype=torch.float32)
    # IMU to LiDAR
    R_li = extR.T
    P_li = -R_li @ extT
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
    return coordinate_transform @ T_ci @ torch.inverse(coordinate_transform)

def data_loader(queue, num_frames, dataset_path, coordinate_transform, intrinsics, K, T_ci):
    batch_size = 10
    batch_data = {"images": [], "poses": [], "intrinsics": [], "points": [], "tstamp": []}
    for i in tqdm(range(num_frames), desc="Loading data"):
        with Image.open(f"{dataset_path}/frame{i}/image.png") as image_pil:
            image = np.array(image_pil)
        if len(intrinsics) > 4:
            dist_coeffs = np.array(intrinsics[4:])
            image = cv2.undistort(image, K, dist_coeffs)
        image = torch.tensor(image).permute(2, 0, 1).cpu()
        tx, ty, tz, qx, qy, qz, qw = torch.load(f"{dataset_path}/frame{i}/pose.pt").tolist()
        T_wi = quat_to_transform(tx, ty, tz, qx, qy, qz, qw, coordinate_transform)
        T_wc = T_wi @ torch.inverse(T_ci)
        T_cw = torch.inverse(T_wc)
        pose = transform_to_tensor(T_cw)

        lidar_points = torch.tensor(np.load(f"{dataset_path}/frame{i}/points.npy"))
        transformed_lidar_points = transform_points(lidar_points, coordinate_transform)

        batch_data["images"].append(image)
        batch_data["poses"].append(pose)
        batch_data["intrinsics"].append(intrinsics)
        batch_data["points"].append(transformed_lidar_points)
        batch_data["tstamp"].append(i)

        if len(batch_data["tstamp"]) == batch_size or i == num_frames - 1:
            packet = {
                'viz_idx': torch.tensor(batch_data["tstamp"]),
                'tstamp': torch.tensor(batch_data["tstamp"]),
                'poses': torch.stack(batch_data["poses"]),
                'images': torch.stack(batch_data["images"]),
                'points': batch_data["points"],  # feed in list
                'intrinsics': torch.stack(batch_data["intrinsics"]),
                'pose_updates': None,
                'is_last': (i == num_frames - 1)
            }
            queue.put(packet)
            batch_data = {"images": [], "poses": [], "intrinsics": [], "points": [], "tstamp": []}
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
        default="../../dataset/CBD_Building_01_hi_LIO",
        help="Path to local dataset."
    )
    parser.add_argument(
        "--config", "-c",
        default='../../config/config_lidar.yaml',
        help="Path to local config_lidar.yaml."
    )
    parser.add_argument(
        "--num_frames", "-n",
        type=int, default=105,
        help="Number of frames to process."
    )
    args = parser.parse_args()

    config_path = args.config
    dataset_path = args.dataset
    num_frames = args.num_frames

    queue = mp.Queue(maxsize=8)
    config = load_config(config_path)
    gs = GSBackEnd(config, save_dir="./output", use_gui=False)

    coordinate_transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
    intrinsics = torch.tensor(np.loadtxt(f"{dataset_path}/intrinsics.txt"))
    K = np.array([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]])

    T_ci = compute_imu_to_camera_transform(coordinate_transform)

    loader_process = mp.Process(target=data_loader, args=(queue, num_frames, dataset_path, coordinate_transform, intrinsics, K, T_ci))
    loader_process.start()

    pbar = tqdm(total=num_frames, desc="Processing frames")
    processed_frames = 0

    while processed_frames < num_frames:
        packet = queue.get()
        batch_size = len(packet['tstamp'])
        gs.process_track_data(packet)
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
                T_wi = quat_to_transform(tx, ty, tz, qx, qy, qz, qw, coordinate_transform)
                T_wc = T_wi @ torch.inverse(T_ci)
                T_cw = torch.inverse(T_wc)
                trajs.append(T_cw.cuda())
            gs.eval_rendering({index: tensor for index, tensor in enumerate(gtimages)}, None, trajs,
                              torch.arange(0, num_frames))
            break

    pbar.close()
    loader_process.join()
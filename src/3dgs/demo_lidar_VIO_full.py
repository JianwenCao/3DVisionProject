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
    assert type(points) == torch.Tensor, "Points must be a torch tensor"
    assert points.shape[1] == 6, "Points must have 6 columns (x, y, z, r, g, b)"
    assert type(T_ab) == torch.Tensor, "Transformation matrix must be a torch tensor"
    assert T_ab.shape == (4, 4), "Transformation matrix must be 4x4"

    T = torch.eye(6, dtype=torch.float32)
    T[:4, :4] = T_ab
    transformed_points = (T @ points.T).T
    return transformed_points


def data_loader(queue, num_frames, dataset_path, coordinate_transform, intrinsics, K, frame_mask):
    batch_size = 10
    batch_data = {"images": [], "poses": [], "intrinsics": [], "points": [], "tstamp": []}

    # Sort frame_mask to ensure sequential loading
    sorted_frame_mask = sorted(frame_mask)
    # Create a mapping from original frame index to new index (0 to len(frame_mask)-1)
    frame_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_frame_mask)}

    for i in tqdm(sorted_frame_mask, desc="Loading data"):
        if i >= num_frames:
            continue  # Skip if frame index exceeds total number of frames
        with Image.open(f"{dataset_path}/frame{i}/image.png") as image_pil:
            image = np.array(image_pil)
        if len(intrinsics) > 4:
            dist_coeffs = np.array(intrinsics[4:])
            image = cv2.undistort(image, K, dist_coeffs)
        image = torch.tensor(image).permute(2, 0, 1).cpu()
        tx, ty, tz, qx, qy, qz, qw = torch.load(f"{dataset_path}/frame{i}/pose.pt").tolist()
        c2w = quat_to_transform(tx, ty, tz, qx, qy, qz, qw, coordinate_transform)
        w2c = torch.inverse(c2w)
        pose = transform_to_tensor(w2c)

        lidar_points = torch.tensor(np.load(f"{dataset_path}/frame{i}/points.npy"))
        transformed_lidar_points = transform_points(lidar_points, coordinate_transform)

        batch_data["images"].append(image)
        batch_data["poses"].append(pose)
        batch_data["intrinsics"].append(intrinsics)
        batch_data["points"].append(transformed_lidar_points)
        # Use the new index for tstamp and viz_idx
        batch_data["tstamp"].append(frame_index_map[i])

        if len(batch_data["tstamp"]) == batch_size or i == sorted_frame_mask[-1]:
            packet = {
                'viz_idx': torch.tensor(batch_data["tstamp"]),
                'tstamp': torch.tensor(batch_data["tstamp"]),
                'poses': torch.stack(batch_data["poses"]),
                'images': torch.stack(batch_data["images"]),
                'points': batch_data["points"],  # feed in list
                'intrinsics': torch.stack(batch_data["intrinsics"]),
                'pose_updates': None,
                'is_last': (i == sorted_frame_mask[-1])
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
        default="../../dataset/CBD_01_full_VIO",
        help="Path to local dataset w.r.t to the current working directory."
    )
    parser.add_argument(
        "--config", "-c",
        default='../../config/config_lidar.yaml',
        help="Path to local config_lidar.yaml w.r.t to the current working directory."
    )
    parser.add_argument(
        "--num_frames", "-n", type=int,
        default=1180,
        help="Number of frames to process."
    )
    args = parser.parse_args()

    config_path = args.config
    dataset_path = args.dataset
    num_frames = args.num_frames

    # Define the frame mask
    frame_mask = [0, 121, 118, 116, 113, 111, 109, 108, 145, 160, 166, 173, 178, 183, 194, 207, 229, 257, 273, 290, 302,
                  314, 325, 335, 341, 350, 358, 360, 364, 368, 375, 382, 404, 411, 433, 440, 442, 457, 464, 471, 477,
                  487, 499, 509, 519, 530, 541, 556, 562, 579, 601, 618, 636, 643, 647, 653, 665, 675, 685, 695, 709,
                  730, 747, 756, 772, 781, 789, 808, 825, 844, 861, 882, 896, 904, 909, 914, 917, 925, 934, 944, 955,
                  960, 973, 981, 986, 997, 1008, 1026, 1030, 1033, 1037, 1038, 1046, 1058, 1068, 1078, 1084, 1090, 1097,
                  1100, 1108, 1114, 1118, 1124, 1175, 1130]

    queue = mp.Queue(maxsize=8)
    config = load_config(config_path)
    gs = GSBackEnd(config, save_dir="./output", use_gui=True)

    coordinate_transform = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

    intrinsics = torch.tensor(np.loadtxt(f"{dataset_path}/intrinsics.txt"))
    K = np.array([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]])

    loader_process = mp.Process(target=data_loader,
                                args=(queue, num_frames, dataset_path, coordinate_transform, intrinsics, K, frame_mask))
    loader_process.start()

    pbar = tqdm(total=len(frame_mask), desc="Processing frames")
    processed_frames = 0

    while processed_frames < len(frame_mask):
        packet = queue.get()
        batch_size = len(packet['tstamp'])
        gs.process_track_data(packet)
        processed_frames += batch_size
        pbar.update(batch_size)

        if packet['is_last']:
            updated_poses = gs.finalize()
            gtimages, trajs = [], []
            # Evaluate all frames for the entire trajectory
            for i in range(num_frames):
                image_pil = Image.open(f"{dataset_path}/frame{i}/image.png")
                image = np.array(image_pil)
                if len(intrinsics) > 4:
                    dist_coeffs = np.array(intrinsics[4:])
                    image = cv2.undistort(image, K, dist_coeffs)
                gtimages.append(torch.tensor(image).permute(2, 0, 1).cpu())
                tx, ty, tz, qx, qy, qz, qw = torch.load(f"{dataset_path}/frame{i}/pose.pt").tolist()
                extrinsic = torch.inverse(quat_to_transform(tx, ty, tz, qx, qy, qz, qw, coordinate_transform))
                trajs.append(extrinsic.cuda())
            gs.eval_rendering({index: tensor for index, tensor in enumerate(gtimages)}, None, trajs,
                              torch.tensor(frame_mask))
            break

    pbar.close()
    loader_process.join()
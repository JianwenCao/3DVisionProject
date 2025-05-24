import logging
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


def data_loader(queue, num_frames, dataset_path, coordinate_transform, intrinsics, K):
    batch_size = 10
    batch_data = {"images": [], "poses": [], "intrinsics": [], "points": [], "tstamp": []}
    keyframes = {"images": [], "poses": [], "intrinsics": [], "points": [], "tstamp": [], "c2w": []}
    whole_data = {"images": [], "poses": [], "intrinsics": [], "points": [], "tstamp": [], "w2c": []}

    for i in tqdm(range(num_frames), desc="Loading data"):
        try:
            image_pil = Image.open(f"{dataset_path}/frame{i}/image.png")
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
        except Exception as e:
            logging.error(e)

        is_key = False
        if i == 0 or i == num_frames - 1:
            keyframes["images"].append(image)
            keyframes["poses"].append(pose)
            keyframes["intrinsics"].append(intrinsics)
            keyframes["points"].append(transformed_lidar_points)
            keyframes["tstamp"].append(i)
            keyframes["c2w"].append(c2w)
            is_key = True
        else:
            last_keyframe_t = keyframes["c2w"][-1][:3, 3]
            last_keyframe_R = keyframes["c2w"][-1][:3, :3]
            distance = torch.norm(c2w[:3, 3] - last_keyframe_t)
            R_rel = torch.matmul(last_keyframe_R.T, c2w[:3, :3])
            lidar_in_camera = (w2c[:3, :3] @ transformed_lidar_points[:, :3].T + w2c[:3, 3:]).T
            theta = torch.acos((torch.trace(R_rel) - 1) / 2)
            ratio = distance / lidar_in_camera[:, -1].mean()
            if theta > 0.02 or ratio > 0.02:
                keyframes["images"].append(image)
                keyframes["poses"].append(pose)
                keyframes["intrinsics"].append(intrinsics)
                keyframes["points"].append(transformed_lidar_points)
                keyframes["tstamp"].append(i)
                keyframes["c2w"].append(c2w)
                is_key = True

        whole_data["images"].append(image)
        whole_data["poses"].append(pose)
        whole_data["intrinsics"].append(intrinsics)
        whole_data["points"].append(transformed_lidar_points)
        whole_data["tstamp"].append(i)
        whole_data["w2c"].append(w2c)

        if is_key:
            print(f"...adding frame {i} to the queue")
            batch_data["images"].append(image)
            batch_data["poses"].append(pose)
            batch_data["intrinsics"].append(intrinsics)
            batch_data["points"].append(transformed_lidar_points)
            batch_data["tstamp"].append(i)

        if len(batch_data["tstamp"]) == batch_size or i == (num_frames - 1):
            if i == (num_frames - 1):
                print("Number of frames in whole_data:", len(whole_data["tstamp"]))
                print("Number of frames in keyframes:", len(keyframes["tstamp"]))
                whole_data["images"] = torch.stack(whole_data["images"])
                whole_data["poses"] = torch.stack(whole_data["poses"])
                whole_data["intrinsics"] = torch.stack(whole_data["intrinsics"])
                whole_data["points"] = whole_data["points"]  # feed in list
                whole_data["tstamp"] = torch.tensor(whole_data["tstamp"])
                whole_data["w2c"] = torch.stack(whole_data["w2c"])
                keyframes["images"] = torch.stack(keyframes["images"])
                keyframes["poses"] = torch.stack(keyframes["poses"])
                keyframes["intrinsics"] = torch.stack(keyframes["intrinsics"])
                keyframes["points"] = keyframes["points"]  # feed in list
                keyframes["tstamp"] = torch.tensor(keyframes["tstamp"])
                keyframes["c2w"] = torch.stack(keyframes["c2w"])
                keyframes['viz_idx'] = keyframes["tstamp"].clone().detach()
                torch.save(whole_data, "whole_data.pt")
                torch.save(keyframes, "keyframes.pt")

            packet = {
                'viz_idx': torch.tensor(batch_data["tstamp"]),
                'tstamp': torch.tensor(batch_data["tstamp"]),
                'poses': torch.stack(batch_data["poses"]),
                'images': torch.stack(batch_data["images"]),
                'points': batch_data["points"],  # feed in list
                'intrinsics': torch.stack(batch_data["intrinsics"]),
                'pose_updates': None,
                'is_last': (i == num_frames - 1),
            }
            try:
                queue.put(packet)
            except Exception as e:
                logging.error(e)
            batch_data = {"images": [], "poses": [], "intrinsics": [], "points": [], "tstamp": []}

    while not queue.empty():
        time.sleep(1)
        logging.info("Waiting for queue to be consumed")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    torchvision.disable_beta_transforms_warning()

    parser = argparse.ArgumentParser(
        description="Runs 3DGS pipeline on preprocessed synced sensor data."
    )
    parser.add_argument(
        "--dataset", "-d",
        default="/data/storage/jianwen/CBD_01_VIO",
        help="Path to local dataset w.r.t to the current working directory."
    )
    parser.add_argument(
        "--config", "-c",
        default='../../config/config_lidar.yaml',
        help="Path to local config_lidar.yaml w.r.t to the current working directory."
    )
    parser.add_argument(
        "--num_frames", "-n", type=int,
        default=1181,
        help="Number of frames to process."
    )
    args = parser.parse_args()

    config_path = args.config
    dataset_path = args.dataset
    num_frames = args.num_frames

    queue = mp.Queue(maxsize=8)
    config = load_config(config_path)
    gs = GSBackEnd(config, save_dir="./output", use_gui=False)

    coordinate_transform = torch.tensor([
        [1, 0, 0, 0],  # -y_fast = x_gs
        [0, 1, 0, 0],  # -z_fast = y_gs
        [0, 0, 1, 0],  # x_fast = z_gs
        [0, 0, 0, 1]
    ], dtype=torch.float32)

    intrinsics = torch.tensor(np.loadtxt(f"{dataset_path}/intrinsics.txt"))
    K = np.array([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]])

    loader_process = mp.Process(target=data_loader, args=(queue, num_frames, dataset_path, coordinate_transform, intrinsics, K))
    loader_process.start()

    pbar = tqdm(total=num_frames, desc="Processing frames")
    processed_frames = 0

    while processed_frames < num_frames:
        packet = queue.get()
        gs.process_track_data(packet)
        pbar.update(packet['tstamp'][-1].item() - processed_frames)
        processed_frames = packet['tstamp'][-1].item()

        if packet['is_last']:
            whole_data = torch.load("whole_data.pt", weights_only=True, map_location="cpu")
            print("Number of frames in whole_data:", len(whole_data["tstamp"]))
            keyframes = torch.load("keyframes.pt", weights_only=True, map_location="cpu")
            print("Number of frames in keyframes:", len(keyframes["tstamp"]))
            print(keyframes["tstamp"])

            gs.process_track_data(keyframes)
            updated_poses = gs.finalize()
            
            gs.eval_rendering(gtimages={index: tensor for index, tensor in enumerate(whole_data["images"].cuda())}, gtdepthdir=None, traj=whole_data["w2c"].cuda(), kf_idx=keyframes["tstamp"])
            break

    pbar.close()
    loader_process.join()
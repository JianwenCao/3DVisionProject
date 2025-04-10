import multiprocessing as mp
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from gs_backend_lidar import GSBackEnd
from util.utils import load_config
from tqdm import tqdm
from PIL import Image
import torchvision
import time

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

def data_loader(queue, num_frames, dataset_path):
    batch_size = 10
    batch_data = {"images": [], "poses": [], "intrinsics": [], "points": [], "tstamp": []}

    for i in tqdm(range(num_frames), desc="Loading data"):
        image_pil = Image.open(f"{dataset_path}/frame{i}/image.png")
        image = torch.tensor(np.array(image_pil)).permute(2, 0, 1).cpu()

        tx, ty, tz, qx, qy, qz, qw = torch.load(f"{dataset_path}/frame{i}/pose.pt").tolist()
        extrinsic = pose_to_extrinsic(tx, ty, tz, qx, qy, qz, qw)
        pose = extrinsic_to_pose(extrinsic)
        intrinsics = torch.tensor(np.loadtxt(f"{dataset_path}/intrinsics.txt"))
        lidar_points = torch.tensor(np.load(f"{dataset_path}/frame{i}/points.npy"))


        batch_data["images"].append(image)
        batch_data["poses"].append(pose)
        batch_data["intrinsics"].append(intrinsics)
        batch_data["points"].append(lidar_points)
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

    num_frames = 1013
    dataset_path = "../../dataset/red_sculpture_dense"
    config_path = "../../config/config_lidar.yaml"

    queue = mp.Queue(maxsize=8)
    config = load_config(config_path)
    gs = GSBackEnd(config, save_dir="./output", use_gui=True)

    loader_process = mp.Process(target=data_loader, args=(queue, num_frames, dataset_path))
    loader_process.start()

    pbar = tqdm(total=num_frames, desc="Processing frames")

    processed_frames = 0
    frame_counter = 0
    step = 20  # step for keyframe selection

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
                break
        else:
            batch_size = len(packet['tstamp'])
            processed_frames += batch_size
            pbar.update(batch_size)

    pbar.close()
    loader_process.join()
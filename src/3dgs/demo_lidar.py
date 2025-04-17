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

def fastlivo_to_gs_extrinsic(tx, ty, tz, qx, qy, qz, qw):
    """
    Convert FASTLIVO coordinate system pose (tx, ty, tz, qx, qy, qz, qw) to extrinsic in GS frame.
    From camera's POV:
    - FASTLIVO coordinate system: x-forward, y-left, z-up
    - GS / Lietorch: x-right, y-down, z-forward
    """
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
    T = np.array([[0, -1, 0], 
                  [0, 0, -1], 
                  [1, 0, 0]])
    R_B = T @ R_A @ T.T
    t_B = T @ t_A
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R_B
    extrinsic[:3, 3] = t_B
    return torch.tensor(extrinsic, dtype=torch.float32)

def extrinsic_to_tensor(T):
    t = T[:3, 3]
    R_mat = T[:3, :3]
    r = R.from_matrix(R_mat)
    q = r.as_quat()
    return torch.tensor([t[0], t[1], t[2], q[0], q[1], q[2], q[3]])


def transform_points(points):
    """
    Transform points from FAST-LIVO2 to GS world frame.
    Points dimension (N, 6), rows are points, columns 0-5 are x, y, z, r, g, b.
    FAST-LIVO2: x-forward, y-left, z-up
    GS-LIVO: x-right, y-down, z-forward
    """
    T = torch.eye(6, dtype=torch.float32)
    T[:3, :3] = torch.tensor([[0, 1, 0],   # y -> x
                              [0, 0, -1],  # z -> -y
                              [1, 0, 0]],  # x -> z
                             dtype=torch.float32)
    transformed_points = (T @ points.T).T
    return transformed_points

def project_lidar_to_depth(points, pose, intrinsic, H=512, W=640, max_depth=80):
    """Project LiDAR points to depth map in GS coordinate system."""
    fx, fy, cx, cy = intrinsic
    points_xyz = points[:, :3]  # Extract x, y, z
    points_homo = torch.cat([points_xyz, torch.ones(points_xyz.shape[0], 1, device=points.device)], dim=1)

    # Convert pose to extrinsic (c2w in GS frame)
    tx, ty, tz, qx, qy, qz, qw = pose.tolist()
    w2c = torch.inverse(fastlivo_to_gs_extrinsic(tx, ty, tz, qx, qy, qz, qw))  # GS uses c2w, so invert to w2c

    # Transform points to camera frame
    cam_points = torch.matmul(points_homo, w2c.T)
    depths = cam_points[:, 2]
    valid = (depths > 0.1) & (depths < max_depth)
    cam_points = cam_points[valid]
    depths = depths[valid]

    if len(depths) == 0:
        return torch.zeros(H, W, device=points.device)

    x_norm = cam_points[:, 0] / depths
    y_norm = cam_points[:, 1] / depths
    u = fx * x_norm + cx
    v = fy * y_norm + cy

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid].long()
    v = v[valid].long()
    depths = depths[valid]

    depth_map = torch.zeros(H, W, device=points.device)
    if valid.sum() > 0:
        sorted_indices = torch.argsort(depths, descending=True)
        u = u[sorted_indices]
        v = v[sorted_indices]
        depths = depths[sorted_indices]
        depth_map[v, u] = depths
    return depth_map

def data_loader(queue, num_frames, dataset_path):
    batch_size = 10
    batch_data = {"images": [], "poses": [], "intrinsics": [], "points": [], "tstamp": []}

    for i in tqdm(range(num_frames), desc="Loading data"):
        image_pil = Image.open(f"{dataset_path}/frame{i}/image.png")
        image = torch.tensor(np.array(image_pil)).permute(2, 0, 1).cpu()

        tx, ty, tz, qx, qy, qz, qw = torch.load(f"{dataset_path}/frame{i}/pose.pt").tolist()
        extrinsic = fastlivo_to_gs_extrinsic(tx, ty, tz, qx, qy, qz, qw)
        c2w_gs = torch.inverse(extrinsic)
        pose = extrinsic_to_tensor(c2w_gs)
        intrinsics = torch.tensor(np.loadtxt(f"{dataset_path}/intrinsics.txt"))
        lidar_points = torch.tensor(np.load(f"{dataset_path}/frame{i}/points.npy"))
        transformed_lidar_points = transform_points(lidar_points)

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

    # num_frames = 1013
    # dataset_path = "../../dataset/red_sculpture_dense"
    # config_path = "../../config/config_lidar.yaml"

    # Elliot testing
    # num_frames = 500 # normally 1013
    # dataset_path = "../HI-SLAM2/data/red_sculpture_dense_fixed"
    num_frames = 78
    dataset_path = "../HI-SLAM2/data/CBD_Building_01"
    config_path = "../../config/config_lidar.yaml"

    queue = mp.Queue(maxsize=8)
    config = load_config(config_path)
    gs = GSBackEnd(config, save_dir="./output", use_gui=True)

    loader_process = mp.Process(target=data_loader, args=(queue, num_frames, dataset_path))
    loader_process.start()

    pbar = tqdm(total=num_frames, desc="Processing frames")

    processed_frames = 0
    frame_counter = 0
    step = 5  # step for keyframe selection

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
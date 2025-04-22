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

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
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
    assert T_ab.shape == (4,4), "Transformation matrix must be 4x4"

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

def project_lidar_to_depth(points, pose, intrinsic, H=512, W=640, max_depth=80):
    """Project LiDAR points to depth map in GS coordinate system."""
    fx, fy, cx, cy = intrinsic
    points_xyz = points[:, :3]  # Extract x, y, z
    points_homo = torch.cat([points_xyz, torch.ones(points_xyz.shape[0], 1, device=points.device)], dim=1)

    # Convert pose to extrinsic (c2w in GS frame)
    # Define transfrom FAST-LIVO2 to GS coordinate system
    T = torch.tensor([
            [ 0, -1,  0, 0],    # -y_fast = x_gs
            [ 0,  0, -1, 0],    # -z_fast = y_gs
            [ 1,  0,  0, 0],    # x_fast = z_gs
            [ 0,  0,  0, 1]   
        ], dtype=torch.float32)
    tx, ty, tz, qx, qy, qz, qw = pose.tolist() # c2w matrix
    w2c = torch.inverse(quat_to_transform(tx, ty, tz, qx, qy, qz, qw, T))
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

        # Define transfrom FAST-LIVO2 to GS coordinate system
        T = torch.tensor([
            [ 0, -1,  0, 0],    # -y_fast = x_gs
            [ 0,  0, -1, 0],    # -z_fast = y_gs
            [ 1,  0,  0, 0],    # x_fast = z_gs
            [ 0,  0,  0, 1]   
        ], dtype=torch.float32)
        
        tx, ty, tz, qx, qy, qz, qw = torch.load(f"{dataset_path}/frame{i}/pose.pt").tolist()
        c2w = quat_to_transform(tx, ty, tz, qx, qy, qz, qw, T)
        w2c = torch.inverse(c2w)
        pose   = transform_to_tensor(w2c)
        intrinsics = torch.tensor(np.loadtxt(f"{dataset_path}/intrinsics.txt"))
        lidar_points = torch.tensor(np.load(f"{dataset_path}/frame{i}/points.npy"))
        transformed_lidar_points = transform_points(lidar_points, T)

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
    step = 1  # step for keyframe selection

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
                     gtimages.append(torch.tensor(np.array(Image.open(f"{dataset_path}/frame{i}/image.png"))).permute(2, 0, 1))
                     tx, ty, tz, qx, qy, qz, qw = torch.load(f"{dataset_path}/frame{i}/pose.pt").tolist()
                     extrinsic = quat_to_transform(tx, ty, tz, qx, qy, qz, qw)
                     trajs.append(extrinsic.cuda())
                gs.eval_rendering({index: tensor for index, tensor in enumerate(gtimages)}, None, trajs, torch.arange(0, num_frames))
                break
        else:
            batch_size = len(packet['tstamp'])
            processed_frames += batch_size
            pbar.update(batch_size)

    pbar.close()
    loader_process.join()
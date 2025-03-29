import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from munch import munchify
from gs_backend import GSBackEnd
from lietorch import SE3
import yaml
import os
from PIL import Image
from torch.multiprocessing import Process, Queue
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import cv2

def rotation_matrix_to_quaternion(rot_mat):
    """
    Convert a 3x3 rotation matrix to a quaternion [qx, qy, qz, qw].
    Input: rot_mat - torch tensor [3, 3]
    Output: quat - torch tensor [4]
    """
    # Ensure the matrix is on the correct device
    device = rot_mat.device

    # Initialize quaternion
    quat = torch.zeros(4, device=device)

    # Compute quaternion components (based on standard algorithm)
    trace = rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2]

    if trace > 0:
        S = torch.sqrt(trace + 1.0) * 2
        quat[3] = 0.25 * S  # qw
        quat[0] = (rot_mat[2, 1] - rot_mat[1, 2]) / S  # qx
        quat[1] = (rot_mat[0, 2] - rot_mat[2, 0]) / S  # qy
        quat[2] = (rot_mat[1, 0] - rot_mat[0, 1]) / S  # qz
    elif (rot_mat[0, 0] > rot_mat[1, 1]) and (rot_mat[0, 0] > rot_mat[2, 2]):
        S = torch.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2]) * 2
        quat[3] = (rot_mat[2, 1] - rot_mat[1, 2]) / S  # qw
        quat[0] = 0.25 * S  # qx
        quat[1] = (rot_mat[0, 1] + rot_mat[1, 0]) / S  # qy
        quat[2] = (rot_mat[0, 2] + rot_mat[2, 0]) / S  # qz
    elif rot_mat[1, 1] > rot_mat[2, 2]:
        S = torch.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2]) * 2
        quat[3] = (rot_mat[0, 2] - rot_mat[2, 0]) / S  # qw
        quat[0] = (rot_mat[0, 1] + rot_mat[1, 0]) / S  # qx
        quat[1] = 0.25 * S  # qy
        quat[2] = (rot_mat[1, 2] + rot_mat[2, 1]) / S  # qz
    else:
        S = torch.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1]) * 2
        quat[3] = (rot_mat[1, 0] - rot_mat[0, 1]) / S  # qw
        quat[0] = (rot_mat[0, 2] + rot_mat[2, 0]) / S  # qx
        quat[1] = (rot_mat[1, 2] + rot_mat[2, 1]) / S  # qy
        quat[2] = 0.25 * S  # qz

    # Normalize quaternion
    quat = quat / torch.norm(quat)
    return quat


def transform_pose_to_camera_frame(pose):
    """
    Transform pose from LiDAR coordinate system to world-to-camera coordinate system.
    Input: pose - torch tensor [tx, ty, tz, qx, qy, qz, qw] (LiDAR-to-world)
    Output: new_pose - torch tensor [tx', ty', tz', qx', qy', qz', qw'] (world-to-camera)
    """
    # Extract translation and quaternion
    translation = pose[:3]  # [tx, ty, tz]
    quaternion = pose[3:]  # [qx, qy, qz, qw]

    # Define coordinate transformation rule: LiDAR (x, y, z) -> Camera (z, -x, -y)
    new_translation = torch.zeros_like(translation)
    new_translation[0] = -translation[1]  # tx' = -ty
    new_translation[1] = -translation[2]  # ty' = -tz
    new_translation[2] = translation[0]  # tz' = tx

    # Normalize quaternion if not already normalized
    quat_norm = torch.norm(quaternion)
    if quat_norm > 0:
        quaternion = quaternion / quat_norm

    # Create SE3 from original pose (LiDAR-to-world)
    se3_pose = SE3.InitFromVec(pose.unsqueeze(0))  # Add batch dimension

    # Define rotation transform from LiDAR to camera: (x, y, z) -> (z, -x, -y)
    rotation_transform = torch.tensor([
        [0, -1, 0],  # x' = -y
        [0, 0, -1],  # y' = -z
        [1, 0, 0]   # z' = x
    ], dtype=torch.float32, device=pose.device)

    # Convert rotation matrix to quaternion
    transform_quat = rotation_matrix_to_quaternion(rotation_transform)

    # Create SE3 for the coordinate transform (rotation only, no translation)
    transform_vec = torch.zeros(7, device=pose.device)
    transform_vec[3:] = transform_quat  # Set quaternion
    transform_se3 = SE3.InitFromVec(transform_vec.unsqueeze(0))

    # Apply transformation: camera-to-world pose = transform * original_pose
    cam_to_world_se3 = transform_se3 * se3_pose

    # Invert to get world-to-camera pose
    world_to_cam_se3 = cam_to_world_se3.inv()

    # Extract translation and quaternion from SE3 and return as 7D vector
    world_to_cam_vec = world_to_cam_se3.vec().squeeze(0)  # [tx', ty', tz', qx', qy', qz', qw']

    return world_to_cam_vec

def bilateral_interpolation(sparse_depth, sigma_spatial=5, sigma_depth=0.5, kernel_size=5):
    """
    Apply bilateral interpolation to fill sparse depth map.
    Input: sparse_depth - torch tensor [H, W] with sparse depth values
    Output: dense_depth - torch tensor [H, W] with interpolated depth values
    """
    sparse_depth = sparse_depth.float()
    H, W = sparse_depth.shape
    device = sparse_depth.device

    # Create coordinate grid
    y, x = torch.meshgrid(torch.arange(H, device=device),
                         torch.arange(W, device=device), indexing='ij')
    coords = torch.stack([x, y], dim=-1).float()  # [H, W, 2]

    # Get valid depth points
    valid_mask = sparse_depth > 0
    if not valid_mask.any():
        return sparse_depth.clone()

    valid_coords = coords[valid_mask]  # [N, 2]
    valid_depths = sparse_depth[valid_mask]  # [N]

    # Initialize output depth map and weight sum
    dense_depth = torch.zeros_like(sparse_depth)
    weight_sum = torch.zeros_like(sparse_depth)

    # Process each valid point
    half_kernel = kernel_size // 2
    for i in range(len(valid_coords)):
        u, v = valid_coords[i].long()  # Pixel coordinates (u=x, v=y)
        depth = valid_depths[i]

        # Define local window, ensuring it stays within bounds
        v_min = max(0, v - half_kernel)
        v_max = min(H, v + half_kernel + 1)
        u_min = max(0, u - half_kernel)
        u_max = min(W, u + half_kernel + 1)

        # Extract local coordinates and compute distances
        local_coords = coords[v_min:v_max, u_min:u_max]  # [kh, kw, 2]
        center_coord = valid_coords[i].float()  # Use the exact valid coordinate [2]

        spatial_dist = torch.norm(local_coords - center_coord, dim=-1)  # [kh, kw]
        spatial_weight = torch.exp(-spatial_dist**2 / (2 * sigma_spatial**2))

        # Compute depth weights
        local_depth = sparse_depth[v_min:v_max, u_min:u_max]  # [kh, kw]
        depth_diff = torch.where(local_depth > 0,
                               torch.abs(local_depth - depth),
                               torch.tensor(float('inf'), device=device))
        depth_weight = torch.exp(-depth_diff**2 / (2 * sigma_depth**2))

        # Combine weights and apply to depth
        weights = spatial_weight * depth_weight  # [kh, kw]
        weighted_depth = depth * weights

        # Accumulate results
        dense_depth[v_min:v_max, u_min:u_max] += weighted_depth
        weight_sum[v_min:v_max, u_min:u_max] += weights

    # Normalize by weight sum, preserve original valid points
    dense_depth = torch.where(weight_sum > 1e-6, dense_depth / (weight_sum + 1e-6), dense_depth)
    dense_depth = torch.where(valid_mask, sparse_depth, dense_depth)  # Keep original valid depths

    return dense_depth

def project_lidar_to_depth(points, pose, intrinsics, distortion, frame_idx, H=512, W=640, max_depth=100.0):
    points = points.to("cuda")
    pose = pose.to("cuda")
    intrinsics = intrinsics.to("cuda")
    distortion = distortion.to("cuda")
    fx, fy, cx, cy = intrinsics[0]
    k1, k2, p1, p2 = distortion

    # Extract points in world frame
    points_xyz = points[:, :3]

    # Step 1: Convert world points to LiDAR frame using inverse of pose (l2w -> w2l)
    points_homo = torch.cat([points_xyz, torch.ones(points_xyz.shape[0], 1, device=points.device)], dim=1)
    w2l = SE3(pose).inv().matrix()  # World to LiDAR transformation
    lidar_points = torch.matmul(w2l, points_homo.T).T[:, :3]  # LiDAR frame points

    # Step 2: Adjust LiDAR coordinate system to camera coordinate system
    cam_points = torch.zeros_like(lidar_points)
    cam_points[:, 0] = -lidar_points[:, 1]  # image u = -LiDAR y
    cam_points[:, 1] = -lidar_points[:, 2]  # image Y = -LiDAR z
    cam_points[:, 2] = lidar_points[:, 0]  # image Z = LiDAR x

    # Step 3: Filter invalid depths
    depths = cam_points[:, 2]
    valid_depth = (depths > 0.1) & (depths < max_depth)  # Min and max depth threshold
    cam_points = cam_points[valid_depth]
    depths = depths[valid_depth]

    if len(depths) == 0:
        print("No points with valid depth after filtering!")
        return torch.zeros(H, W, device="cpu")

    # Step 4: Projection to image plane (Perspective projection)
    x_norm = cam_points[:, 0] / depths
    y_norm = cam_points[:, 1] / depths
    u = fx * x_norm + cx
    v = fy * y_norm + cy

    # Step 5: Filter points outside image bounds
    valid_pixels = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid_pixels]
    v = v[valid_pixels]
    depths = depths[valid_pixels]

    if len(depths) == 0:
        print("No points within image bounds after filtering!")
        return torch.zeros(H, W, device="cpu")

    # Step 6: Generate sparse depth map (nearest point wins)
    u = u.long()  # Convert to integer indices
    v = v.long()
    sorted_indices = torch.argsort(depths)  # Ascending order (nearest first)
    u = u[sorted_indices]
    v = v[sorted_indices]
    depths = depths[sorted_indices]

    sparse_depth_map = torch.zeros(H, W, device="cuda")
    sparse_depth_map[v, u] = depths  # Assign depths to pixels

    # Step 7: Convert to numpy for inpainting
    sparse_depth_np = sparse_depth_map.cpu().numpy()
    mask = (sparse_depth_np > 0).astype(np.uint8) * 255  # Mask where depth exists
    empty_mask = (sparse_depth_np == 0).astype(np.uint8) * 255  # Mask where depth is missing

    # Step 8: Perform inpainting to fill holes
    dense_depth_np = cv2.inpaint(
        sparse_depth_np,          # Input sparse depth map
        empty_mask,              # Mask of missing areas
        inpaintRadius=3,         # Radius for inpainting (adjust based on sparsity)
        flags=cv2.INPAINT_TELEA  # Use Telea algorithm (faster, good for small gaps)
    )

    # Step 9: Return dense depth map as torch tensor
    dense_depth_map = torch.from_numpy(dense_depth_np).to("cpu")
    return dense_depth_map

def load_camera_params(file_path):
    # Load camera parameters from YAML file
    with open(file_path, 'r') as f:
        params = yaml.safe_load(f)
    scale = params['scale']
    fx = params['cam_fx'] * scale
    fy = params['cam_fy'] * scale
    cx = params['cam_cx'] * scale
    cy = params['cam_cy'] * scale
    k1 = params['cam_d0']
    k2 = params['cam_d1']
    p1 = params['cam_d2']
    p2 = params['cam_d3']
    intrinsics = torch.tensor([[fx, fy, cx, cy]], device="cpu")
    distortion = torch.tensor([k1, k2, p1, p2], device="cpu")
    return intrinsics, distortion


def load_and_reformat_data(data_dir, frame_indices):
    # Load and reformat data for processing
    batch_size = len(frame_indices)
    H, W = 512, 640
    images = torch.zeros(batch_size, 3, H, W, device="cpu")
    depths = torch.zeros(batch_size, H, W, device="cpu")
    normals = torch.zeros(batch_size, 3, H, W, device="cpu")
    poses = torch.zeros(batch_size, 7, device="cpu")
    tstamps = torch.zeros(batch_size, device="cpu")
    intrinsics_file = os.path.join(data_dir, "camera_pinhole.yaml")
    intrinsics, distortion = load_camera_params(intrinsics_file)
    intrinsics = intrinsics.repeat(batch_size, 1)
    for i, idx in enumerate(frame_indices):
        frame_dir = os.path.join(data_dir, f"frame{idx}")
        img = Image.open(os.path.join(frame_dir, "image.png")).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)
        images[i] = img_tensor

        points = torch.from_numpy(np.load(os.path.join(frame_dir, "points.npy"))).to("cpu")
        pose_orig = torch.load(os.path.join(frame_dir, "pose.pt")).to("cpu")
        quat = pose_orig[3:7]
        quat_norm = torch.norm(quat)
        if quat_norm > 0:
            quat = quat / quat_norm
        pose = torch.zeros(7, device="cpu")
        pose[0:3] = pose_orig[0:3]
        pose[3:7] = quat

        depths[i] = project_lidar_to_depth(points, pose, intrinsics, distortion, idx)
        pose_image = transform_pose_to_camera_frame(pose_orig) # world-to-camera pose
        poses[i] = pose_image

        depth_grad_y, depth_grad_x = torch.gradient(depths[i])
        normal = torch.stack([-depth_grad_x, -depth_grad_y, torch.ones_like(depth_grad_x)], dim=0)
        normal = normal / (torch.norm(normal, dim=0, keepdim=True) + 1e-6)
        normals[i] = normal
        tstamps[i] = float(idx)

    return {
        "images": images,
        "depths": depths,
        "normals": normals,
        "poses": poses,
        "tstamp": tstamps,
        "viz_idx": torch.tensor(frame_indices, device="cpu"),
        "intrinsics": intrinsics,
        "pose_updates": None,
        "scale_updates": None
    }


def load_config(config_path):
    # Load configuration from YAML file
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return munchify(config_dict)


class GSProcessorWrapper:
    def __init__(self, config, save_dir, use_gui=False):
        self.gs_backend = GSBackEnd(config, save_dir, use_gui=use_gui)
        self.save_dir = save_dir
        self.gs_backend.start()
        self.counter = 0  # 跟踪处理的帧数，模仿 hi2 的 counter

    def process_data(self, data_packet):
        # 处理数据包并更新计数器
        self.gs_backend.process_track_data(data_packet)
        self.counter += len(data_packet['viz_idx'])  # 更新已处理的关键帧数

    def finalize(self):
        result = self.gs_backend.finalize()
        self.gs_backend.terminate()
        return result

    def call_gs(self, data_packet):
        # 模仿 hi2 的 call_gs，直接调用 process_data
        self.process_data(data_packet)

def data_stream(queue, data_dir, total_frames, frame_step, batch_size=8, start=0):
    """ 数据生成器，模仿 mono_stream，每次放入 batch_size 帧 """
    all_frame_indices = list(range(start, total_frames, frame_step))
    batches = [all_frame_indices[i:i + batch_size] for i in range(0, len(all_frame_indices), batch_size)]

    for t, frame_indices in enumerate(batches):
        data_packet = load_and_reformat_data(data_dir, frame_indices)
        is_last = (t == len(batches) - 1)
        queue.put((t, data_packet, is_last))

    # 等待队列处理完成
    import time
    time.sleep(10)

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.multiprocessing.set_start_method('spawn')

    # 加载配置和初始化处理器
    config = load_config("../../config/config.yaml")
    processor = GSProcessorWrapper(config, "output", use_gui=False)

    # 定义总帧数、步长和每批关键帧数
    total_frames = 180
    frame_step = 1
    batch_size = 4  # 每次处理 8 个关键帧
    data_dir = "../../dataset/red_sculpture"

    # 创建队列和数据读取进程
    queue = Queue(maxsize=8)
    reader = Process(target=data_stream, args=(queue, data_dir, total_frames, frame_step, batch_size))
    reader.start()

    # 计算总批次数
    N = total_frames // frame_step
    num_batches = (N + batch_size - 1) // batch_size  # 向上取整
    pbar = tqdm(range(N), desc="Processing frames")

    # 主循环，模仿 hi2 的流式处理，但每次处理 8 帧
    processed_frames = 0
    while True:
        (t, data_packet, is_last) = queue.get()
        num_frames_in_batch = len(data_packet['viz_idx'])
        pbar.update(num_frames_in_batch)
        processed_frames += num_frames_in_batch

        # 检查是否有关键帧需要处理
        viz_idx = data_packet['viz_idx']  # 当前批次的帧索引
        if len(viz_idx) > 0:
            processor.call_gs(data_packet)  # 调用 GS 处理当前批次（8 帧）

        # 更新进度条描述
        pbar.set_description(f"Processing frame {processed_frames}/{total_frames} (batch {t+1}/{num_batches})")

        if is_last:
            pbar.close()
            break

    # 等待读取进程结束
    reader.join()

    # 完成处理并获取结果
    poses = processor.finalize()
    print("Processing complete! Final poses:", poses)
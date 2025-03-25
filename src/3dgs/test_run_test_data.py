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

def bilateral_filter_depth(depth, sigma_spatial=5, sigma_depth=0.1, kernel_size=5):
    H, W = depth.shape
    device = depth.device
    grid = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device)
    x, y = torch.meshgrid(grid, grid, indexing='ij')
    spatial_kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma_spatial ** 2))
    spatial_kernel = spatial_kernel / spatial_kernel.sum()
    spatial_kernel = spatial_kernel.view(1, 1, kernel_size, kernel_size)
    valid_mask = (depth > 0).float()
    depth_input = depth[None, None, :, :] * valid_mask[None, None, :, :]
    weights_input = valid_mask[None, None, :, :]
    padding = kernel_size // 2
    depth_smooth = F.conv2d(depth_input, spatial_kernel, padding=padding)[0, 0]
    weights_smooth = F.conv2d(weights_input, spatial_kernel, padding=padding)[0, 0]
    weights_smooth = weights_smooth.clamp(min=1e-6)
    depth_smooth = depth_smooth / weights_smooth
    depth_diff = (depth - depth_smooth).abs()
    range_weights = torch.exp(-depth_diff ** 2 / (2 * sigma_depth ** 2))
    combined_weights = range_weights * valid_mask
    depth_input = depth_smooth[None, None, :, :] * combined_weights[None, None, :, :]
    weights_input = combined_weights[None, None, :, :]
    depth_refined = F.conv2d(depth_input, spatial_kernel, padding=padding)[0, 0]
    weights_refined = F.conv2d(weights_input, spatial_kernel, padding=padding)[0, 0]
    weights_refined = weights_refined.clamp(min=1e-6)
    depth_refined = depth_refined / weights_refined
    output = torch.where(valid_mask > 0, depth, depth_refined)
    return output

def project_lidar_to_depth(points, pose, intrinsics, distortion, H=512, W=640):
    # Project LiDAR points in world coordinates to depth map
    # LiDAR coordinate system: x down (depth), y right, z up, right-handed
    # Mapping: camera X = LiDAR y, Y = -LiDAR z, Z = LiDAR x
    points = points.to("cuda")
    pose = pose.to("cuda")
    intrinsics = intrinsics.to("cuda")
    distortion = distortion.to("cuda")
    fx, fy, cx, cy = intrinsics[0]
    k1, k2, p1, p2 = distortion

    # Calculate actual FOV
    fov_h = 2 * torch.atan(torch.tensor(W / (2 * fx))).item() * 180 / 3.14159
    fov_v = 2 * torch.atan(torch.tensor(H / (2 * fy))).item() * 180 / 3.14159
    norm_x_max = torch.tan(torch.tensor(fov_h * 3.14159 / 180 / 2)).item()
    norm_y_max = torch.tan(torch.tensor(fov_v * 3.14159 / 180 / 2)).item()
    print(f"Calculated FOV: Horizontal {fov_h:.2f}°, Vertical {fov_v:.2f}°")
    print(f"Expected norm range: x [{-norm_x_max:.3f}, {norm_x_max:.3f}], y [{-norm_y_max:.3f}, {norm_y_max:.3f}]")

    # Adjust LiDAR points: x=depth(down), y=right, z=up -> camera X=right, Y=down, Z=forward
    points_xyz = points[:, :3]
    points_xyz_adjusted = torch.zeros_like(points_xyz)
    points_xyz_adjusted[:, 0] = points_xyz[:, 1]   # Camera X = LiDAR y (right)
    points_xyz_adjusted[:, 1] = -points_xyz[:, 2]  # Camera Y = -LiDAR z (up -> down)
    points_xyz_adjusted[:, 2] = points_xyz[:, 0]   # Camera Z = LiDAR x (depth)
    points_homo = torch.cat([points_xyz_adjusted, torch.ones(points_xyz.shape[0], 1, device=points.device)], dim=1)

    # Transform from world to camera coordinates
    w2c = SE3(pose).matrix()
    cam_points = torch.matmul(w2c, points_homo.T).T

    # Filter out negative depths
    depths = cam_points[:, 2]
    valid_depth = depths > 0.1
    cam_points = cam_points[valid_depth]
    depths = depths[valid_depth]

    if len(depths) == 0:
        print("No points with positive depth (> 0.1) after transformation!")
        return torch.zeros(H, W, device="cpu")

    # Normalized coordinates
    x_norm = cam_points[:, 0] / depths
    y_norm = cam_points[:, 1] / depths

    # Temporarily disable distortion
    x_dist = x_norm
    y_dist = y_norm

    # Pixel coordinates
    u = fx * x_dist + cx
    v = fy * y_dist + cy

    # Debug output
    print(f"World points range (original): x(depth) [{points_xyz[:, 0].min().item()}, {points_xyz[:, 0].max().item()}] "
          f"y(right) [{points_xyz[:, 1].min().item()}, {points_xyz[:, 1].max().item()}] "
          f"z(up) [{points_xyz[:, 2].min().item()}, {points_xyz[:, 2].max().item()}]")
    print(f"World points range (adjusted): x(right) [{points_xyz_adjusted[:, 0].min().item()}, {points_xyz_adjusted[:, 0].max().item()}] "
          f"y(down) [{points_xyz_adjusted[:, 1].min().item()}, {points_xyz_adjusted[:, 1].max().item()}] "
          f"z(depth) [{points_xyz_adjusted[:, 2].min().item()}, {points_xyz_adjusted[:, 2].max().item()}]")
    print(f"Cam points range: x [{cam_points[:, 0].min().item()}, {cam_points[:, 0].max().item()}] "
          f"y [{cam_points[:, 1].min().item()}, {cam_points[:, 1].max().item()}] "
          f"z [{depths.min().item()}, {depths.max().item()}]")
    print(f"Normalized range: x [{x_norm.min().item()}, {x_norm.max().item()}] "
          f"y [{y_norm.min().item()}, {y_norm.max().item()}]")
    print(f"Pixel u range: [{u.min().item()}, {u.max().item()}]")
    print(f"Pixel v range: [{v.min().item()}, {v.max().item()}]")

    # Filter valid points
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid].long()
    v = v[valid].long()
    depths = depths[valid]

    # Generate depth map
    depth_map = torch.zeros(H, W, device=points.device)
    if valid.sum() > 0:
        sorted_indices = torch.argsort(depths, descending=True)
        u = u[sorted_indices]
        v = v[sorted_indices]
        depths = depths[sorted_indices]
        depth_map[v, u] = depths
        print(f"Assigned depths range: [{depths.min().item()}, {depths.max().item()}]")
        print(f"Depth map non-zero count: {(depth_map > 0).sum().item()}")

    print(f"Valid points: {valid.sum().item()}/{points.shape[0]}")
    print(f"Depth range: {depths.min().item() if valid.sum() > 0 else 0} to {depths.max().item() if valid.sum() > 0 else 0}")

    return depth_map.cpu()

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
    # Load and reformat data for processing, reading every 10th frame
    batch_size = len(frame_indices)
    H, W = 512, 640
    images = torch.zeros(batch_size, 3, H, W, device="cpu")
    depths = torch.zeros(batch_size, H, W, device="cpu")
    normals = torch.zeros(batch_size, H, W, 3, device="cpu")  # Required by GSBackEnd
    poses = torch.zeros(batch_size, 7, device="cpu")
    tstamps = torch.zeros(batch_size, device="cpu")
    intrinsics_file = os.path.join(data_dir, "camera_pinhole.yaml")
    intrinsics, distortion = load_camera_params(intrinsics_file)

    for i, idx in enumerate(frame_indices):
        frame_dir = os.path.join(data_dir, f"frame{idx}")
        img = Image.open(os.path.join(frame_dir, "image.png")).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1) / 255.0
        if img_tensor.shape[1:] != (H, W):
            img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
        images[i] = img_tensor

        points = torch.from_numpy(np.load(os.path.join(frame_dir, "points.npy"))).to("cpu")
        print(f"Frame {idx} points.npy dimension: {points.shape}")

        pose_orig = torch.load(os.path.join(frame_dir, "pose.pt")).to("cpu")
        print(f"Frame {idx} pose: {pose_orig}")
        quat = pose_orig[3:7]
        quat_norm = torch.norm(quat)
        if quat_norm > 0:
            quat = quat / quat_norm
        pose = torch.zeros(7, device="cpu")
        pose[0:3] = pose_orig[0:3]
        pose[3:7] = quat
        poses[i] = pose

        depths[i] = project_lidar_to_depth(points, pose, intrinsics, distortion)
        # GSBackEnd requires normals, using simple gradient estimation here (can be replaced with actual data)
        depth_grad_y, depth_grad_x = torch.gradient(depths[i])
        normal = torch.stack([-depth_grad_x, -depth_grad_y, torch.ones_like(depth_grad_x)], dim=-1)
        normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
        normals[i] = normal
        tstamps[i] = float(idx)

    return {
        "images": images,
        "depths": depths,
        "normals": normals,  # Add normals
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
    def __init__(self, config, save_dir, use_gui=False):  # Enable GUI
        self.gs_backend = GSBackEnd(config, save_dir, use_gui=use_gui)
        self.save_dir = save_dir
        self.gs_backend.start()  # Start the process

    def process_data(self, data_packet):
        self.gs_backend.process_track_data(data_packet)

    def finalize(self):
        result = self.gs_backend.finalize()
        self.gs_backend.terminate()  # Terminate the process
        return result

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    config = load_config("../../config/config.yaml")
    processor = GSProcessorWrapper(config, "output", use_gui=True)  # Enable GUI
    # Modified to read every 10th frame
    total_frames = 18  # Adjust this based on your total number of frames
    frame_indices = list(range(0, total_frames, 10))  # Read every 10th frame
    data_packet = load_and_reformat_data("../../dataset/red_sculpture", frame_indices)
    processor.process_data(data_packet)
    poses = processor.finalize()
    print("Processing complete! Final poses:", poses)
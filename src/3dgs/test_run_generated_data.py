import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from munch import munchify
from gs_backend import GSBackEnd
from lietorch import SE3
import yaml


# [Keep your bilateral_filter_depth function unchanged]
def bilateral_filter_depth(depth, sigma_spatial=5, sigma_depth=0.1, kernel_size=5):
    """
    Memory-efficient depth completion using Gaussian smoothing with edge preservation.
    """
    H, W = depth.shape
    device = depth.device

    # Create Gaussian spatial kernel
    grid = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device)
    x, y = torch.meshgrid(grid, grid, indexing='ij')
    spatial_kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma_spatial ** 2))  # [k, k]
    spatial_kernel = spatial_kernel / spatial_kernel.sum()  # Normalize
    spatial_kernel = spatial_kernel.view(1, 1, kernel_size, kernel_size)  # [1, 1, k, k]

    # Mask for valid depth points
    valid_mask = (depth > 0).float()  # [H, W]

    # Step 1: Initial Gaussian smoothing of sparse depth
    depth_input = depth[None, None, :, :] * valid_mask[None, None, :, :]  # [1, 1, H, W]
    weights_input = valid_mask[None, None, :, :]  # [1, 1, H, W]

    # Pad and convolve
    padding = kernel_size // 2
    depth_smooth = F.conv2d(depth_input, spatial_kernel, padding=padding)[0, 0]  # [H, W]
    weights_smooth = F.conv2d(weights_input, spatial_kernel, padding=padding)[0, 0]  # [H, W]

    # Normalize where we have contributions
    weights_smooth = weights_smooth.clamp(min=1e-6)
    depth_smooth = depth_smooth / weights_smooth  # [H, W]

    # Step 2: Simple edge-preserving refinement
    depth_diff = (depth - depth_smooth).abs()  # [H, W]
    range_weights = torch.exp(-depth_diff ** 2 / (2 * sigma_depth ** 2))  # [H, W]
    combined_weights = range_weights * valid_mask  # Only refine where we have data

    # Final smoothing with range weights
    depth_input = depth_smooth[None, None, :, :] * combined_weights[None, None, :, :]  # [1, 1, H, W]
    weights_input = combined_weights[None, None, :, :]  # [1, 1, H, W]

    depth_refined = F.conv2d(depth_input, spatial_kernel, padding=padding)[0, 0]  # [H, W]
    weights_refined = F.conv2d(weights_input, spatial_kernel, padding=padding)[0, 0]  # [H, W]

    weights_refined = weights_refined.clamp(min=1e-6)
    depth_refined = depth_refined / weights_refined  # [H, W]

    # Preserve original valid depths
    output = torch.where(valid_mask > 0, depth, depth_refined)

    return output


# [Keep your generate_synthetic_data function unchanged]
def generate_synthetic_data(frame_indices):
    """
    Generate synthetic point cloud data and project to 2D depth maps for multiple frames.
    """
    batch_size = len(frame_indices)
    H, W = 480, 640
    num_points = 1000

    point_cloud = torch.rand(batch_size, num_points, 3, device="cpu") * 10.0 - 5.0
    z_values = point_cloud[:, :, 2]
    xy_distances = torch.cdist(point_cloud[:, :, :2], point_cloud[:, :, :2])
    depth_diff = torch.abs(z_values[:, :, None] - z_values[:, None, :])
    weights = (torch.exp(-xy_distances ** 2 / (2 * 5.0 ** 2)) *
               torch.exp(-depth_diff ** 2 / (2 * 0.1 ** 2)))
    smoothed_z = (z_values[:, None, :] * weights).sum(dim=2) / weights.sum(dim=2)
    point_cloud[:, :, 2] = smoothed_z

    poses = torch.zeros(batch_size, 7, device="cpu")
    poses[:, 0] = 1.0
    poses[:, 6] = frame_indices * 0.1

    w2c = SE3(poses).matrix()
    points_homo = torch.cat([point_cloud, torch.ones(batch_size, num_points, 1, device="cpu")], dim=2)
    cam_points = torch.bmm(w2c, points_homo.transpose(1, 2)).transpose(1, 2)[:, :, :3]

    fx, fy, cx, cy = 500.0, 500.0, W / 2, H / 2
    intrinsics = torch.tensor([[fx, fy, cx, cy]], device="cpu")

    u = (fx * cam_points[:, :, 0] / cam_points[:, :, 2] + cx).long()
    v = (fy * cam_points[:, :, 1] / cam_points[:, :, 2] + cy).long()
    depths = cam_points[:, :, 2]

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depths > 0)

    depth_maps = torch.zeros(batch_size, H, W, device="cpu")
    images = torch.zeros(batch_size, 3, H, W, device="cpu")

    for b in range(batch_size):
        u_b, v_b, depths_b = u[b][valid[b]], v[b][valid[b]], depths[b][valid[b]]
        depth_maps[b, v_b, u_b] = depths_b
        depth_maps[b] = bilateral_filter_depth(depth_maps[b])
        depth_norm = (depth_maps[b] - depth_maps[b].min()) / (depth_maps[b].max() - depth_maps[b].min() + 1e-6)
        images[b, 0] = depth_norm.cpu()
        images[b, 1] = depth_norm.cpu()
        images[b, 2] = depth_norm.cpu()

    return {
        "images": images,   # torch.Tensor, [batch_size, 3, H, W] - RGB images (CPU)
        "depths": depth_maps.cpu(), # torch.Tensor, [batch_size, H, W] - Depth maps (CPU)
        "poses": poses, # torch.Tensor, [batch_size, 7] - Camera poses as quaternion + translation (CPU)
        "tstamp": frame_indices.float(),    # torch.Tensor, [batch_size] - Timestamps (CPU)
        "viz_idx": frame_indices,   # torch.Tensor, [batch_size] - Visualization indices (CPU)
        "intrinsics": intrinsics,   # torch.Tensor, [1, 4] - Camera intrinsics fx, fy, cx, cy (CPU)
        "pose_updates": None,
        "scale_updates": None
    }


# Function to load config from YAML
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return munchify(config_dict)


class GSProcessorWrapper:
    def __init__(self, config, save_dir, use_gui=False):
        self.gs_backend = GSBackEnd(config, save_dir, use_gui=use_gui)
        self.save_dir = save_dir

    def process_data(self, data_packet):
        self.gs_backend.process_track_data(data_packet)

    def finalize(self):
        return self.gs_backend.finalize()


if __name__ == "__main__":
    # Load configuration from YAML file
    config = load_config("../../config/config.yaml")

    processor = GSProcessorWrapper(config, "output", use_gui=False)
    num_frames = 2
    batch_size = num_frames

    frame_indices = torch.arange(num_frames, device="cpu")
    data_packet = generate_synthetic_data(frame_indices)
    processor.process_data(data_packet)

    poses = processor.finalize()
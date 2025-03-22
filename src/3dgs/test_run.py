import torch
import numpy as np
from tqdm import trange
from munch import munchify

# Import the GSBackEnd class
from gs_backend import GSBackEnd


def bilateral_filter_depth(depth, sigma_spatial=5, sigma_depth=0.1, kernel_size=5):
    """Apply bilateral filtering to a depth map to smooth while preserving edges."""
    H, W = depth.shape
    padded_depth = torch.nn.functional.pad(depth[None, None, :, :],
                                           (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2),
                                           mode='reflect')[0, 0]
    output = torch.zeros_like(depth)

    for i in range(H):
        for j in range(W):
            window = padded_depth[i:i + kernel_size, j:j + kernel_size]
            center = depth[i, j]
            spatial_weights = torch.exp(
                -torch.arange(-(kernel_size // 2), kernel_size // 2 + 1) ** 2 / (2 * sigma_spatial ** 2)).cuda()
            spatial_weights = spatial_weights[:, None] * spatial_weights[None, :]
            depth_diff = window - center
            depth_weights = torch.exp(-depth_diff ** 2 / (2 * sigma_depth ** 2))
            weights = spatial_weights * depth_weights
            output[i, j] = (window * weights).sum() / weights.sum()

    return output


def generate_synthetic_data(frame_idx):
    """Generate synthetic point cloud data and project to 2D depth map."""
    H, W = 480, 640
    num_points = 10000  # Number of points in the point cloud

    # Generate random 3D point cloud (x, y, z) in world coordinates
    point_cloud = torch.rand(num_points, 3, device="cuda") * 10.0 - 5.0  # Range [-5, 5]

    # Step 1: Bilateral filtering on 3D point cloud (smooth z-coordinate)
    z_values = point_cloud[:, 2]
    xy_distances = torch.cdist(point_cloud[:, :2], point_cloud[:, :2])
    depth_diff = torch.abs(z_values[:, None] - z_values[None, :])
    weights = torch.exp(-xy_distances ** 2 / (2 * 5.0 ** 2)) * torch.exp(-depth_diff ** 2 / (2 * 0.1 ** 2))
    smoothed_z = (z_values[None, :] * weights).sum(dim=1) / weights.sum(dim=1)
    point_cloud[:, 2] = smoothed_z

    # Step 2: Define w2c matrix (pose as [N, 7] quaternion + translation)
    pose = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(frame_idx * 0.1)]],
                        dtype=torch.float32, device="cuda")  # Move along z-axis

    # Convert pose to 4x4 w2c matrix using SE3
    from lietorch import SE3
    w2c = SE3(pose).matrix()[0]  # Shape: [4, 4]

    # Project 3D points to camera coordinates
    points_homo = torch.cat([point_cloud, torch.ones(num_points, 1, device="cuda")], dim=1)  # [N, 4]
    cam_points = (w2c @ points_homo.T).T  # [N, 4], apply w2c transformation
    cam_points = cam_points[:, :3]  # [N, 3], drop homogeneous coord

    # Intrinsic parameters
    fx, fy, cx, cy = 500.0, 500.0, W / 2, H / 2
    intrinsics = torch.tensor([[fx, fy, cx, cy]], device="cpu")

    # Project to 2D image plane
    u = (fx * cam_points[:, 0] / cam_points[:, 2] + cx).long()
    v = (fy * cam_points[:, 1] / cam_points[:, 2] + cy).long()
    depths = cam_points[:, 2]

    # Filter points within image bounds and in front of camera
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depths > 0)
    u, v, depths = u[valid], v[valid], depths[valid]

    # Create sparse depth map
    depth_map = torch.zeros(H, W, device="cuda")
    depth_map[v, u] = depths

    # Step 3: Bilateral filtering to complete depth map
    depth_map = bilateral_filter_depth(depth_map, sigma_spatial=5, sigma_depth=0.1, kernel_size=5)

    # Generate synthetic RGB image (e.g., based on depth)
    image = torch.zeros(1, 3, H, W, device="cpu")
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    image[0, 0, :, :] = depth_norm.cpu()  # Red channel from depth
    image[0, 1, :, :] = depth_norm.cpu()  # Green channel from depth
    image[0, 2, :, :] = depth_norm.cpu()  # Blue channel from depth

    return {
        "images": image,  # CPU tensor [1, 3, H, W]
        "depths": depth_map[None, :, :].cpu(),  # CPU tensor [1, H, W]
        "poses": pose,  # CUDA tensor [1, 7]
        "tstamp": torch.tensor([float(frame_idx)], device="cuda"),  # [1]
        "viz_idx": torch.tensor([frame_idx], device="cuda"),  # [1]
        "intrinsics": intrinsics,  # CPU tensor [1, 4]
        "pose_updates": None,
        "scale_updates": None
    }


# Configuration with necessary 3DGS components
config = {
    "opt_params": {
        "pose_lr": 0.0001,
        "position_lr_init": 0.00016,
        "position_lr_final": 0.0000016,
        "position_lr_max_steps": 2000,
        "feature_lr": 0.0025,
        "opacity_lr": 0.05,
        "scaling_lr": 0.001,
        "rotation_lr": 0.001,
        "percent_dense": 0.01,
        "lambda_dssim": 0.2,
        "densify_grad_threshold": 0.0002
    },
    "Training": {
        "init_itr_num": 1050,
        "init_gaussian_update": 100,
        "init_gaussian_reset": 500,
        "init_gaussian_th": 0.005,
        "init_gaussian_extent": 30,
        "gaussian_update_every": 150,
        "gaussian_update_offset": 50,
        "gaussian_th": 0.7,
        "gaussian_extent": 1.0,
        "gaussian_reset": 2001,
        "size_threshold": 20,
        "window_size": 10,
        "rgb_boundary_threshold": 0.01,
        "compensate_exposure": False,
        "monocular": False
    },
    "Dataset": {
        "pcd_downsample": 64,
        "pcd_downsample_init": 32,
        "point_size": 0.05,
        "adaptive_pointsize": True
    }
}


class GSProcessorWrapper:
    def __init__(self, config, save_dir, use_gui=False):
        self.gs_backend = GSBackEnd(config, save_dir, use_gui=use_gui)
        self.save_dir = save_dir

    def process_data(self, data_packet):
        self.gs_backend.process_track_data(data_packet)

    def finalize(self):
        return self.gs_backend.finalize()


# Main execution
if __name__ == "__main__":
    processor = GSProcessorWrapper(config, "output", use_gui=False)

    # Process 5 frames with point cloud data
    for i in trange(5, desc="Processing point cloud frames"):
        data_packet = generate_synthetic_data(i)
        processor.process_data(data_packet)

    poses = processor.finalize()
    print(f"Processed {len(poses)} frames")
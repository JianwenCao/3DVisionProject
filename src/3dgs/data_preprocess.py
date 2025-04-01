import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import yaml
import torchvision
from lietorch import SE3
from utils_new import transform_pose_to_camera_frame, rotation_matrix_to_quaternion

def load_camera_params(file_path):
    """Load camera intrinsics and distortion from YAML."""
    with open(file_path) as f:
        params = yaml.safe_load(f)
    scale = params['scale']
    intrinsics = torch.tensor([[params['cam_fx'] * scale, params['cam_fy'] * scale,
                                params['cam_cx'] * scale, params['cam_cy'] * scale]])
    distortion = torch.tensor([params['cam_d0'], params['cam_d1'],
                               params['cam_d2'], params['cam_d3']])
    return intrinsics, distortion

def project_lidar_to_depth(points, pose, intrinsics, distortion, H=512, W=640, max_depth=400.0):
    """Project LiDAR points to a sparse depth map with distortion correction."""
    points, pose, intrinsics, distortion = (points.cuda(), pose.cuda(),
                                            intrinsics.cuda(), distortion.cuda())
    fx, fy, cx, cy = intrinsics[0]
    k1, k2, p1, p2 = distortion

    points_homo = torch.cat([points[:, :3], torch.ones(len(points), 1, device=points.device)], dim=1)
    lidar_points = torch.matmul(SE3(pose).inv().matrix(), points_homo.T).T[:, :3]

    cam_points = torch.stack([-lidar_points[:, 1], -lidar_points[:, 2], lidar_points[:, 0]], dim=1)
    depths = cam_points[:, 2]
    valid_depth = (depths > 0.01)
    cam_points, depths = cam_points[valid_depth], depths[valid_depth]

    x_norm = cam_points[:, 0] / depths
    y_norm = cam_points[:, 1] / depths

    r2 = x_norm ** 2 + y_norm ** 2
    radial = 1 + k1 * r2 + k2 * r2 ** 2
    x_dist = x_norm * radial + 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm ** 2)
    y_dist = y_norm * radial + p1 * (r2 + 2 * y_norm ** 2) + 2 * p2 * x_norm * y_norm

    u = (fx * x_dist + cx).long()
    v = (fy * y_dist + cy).long()
    valid_pixels = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, depths = u[valid_pixels], v[valid_pixels], depths[valid_pixels]

    sparse_depth = torch.zeros(H, W, device="cuda")
    if len(depths) > 0:
        sorted_indices = torch.argsort(depths)
        sparse_depth[v[sorted_indices], u[sorted_indices]] = depths[sorted_indices]
    return sparse_depth

def complete_depth(sparse_depth, image, model, t_rgb, t_dep, frame_idx, output_dir="output",
                  downsample_size=(256, 320), original_size=(512, 640)):
    """Complete sparse depth map with downsampling and upsampling, storing results."""
    # Downsample sparse depth
    sparse_depth_down = F.interpolate(sparse_depth.unsqueeze(0).unsqueeze(0),
                                    size=downsample_size, mode='nearest').squeeze(0)
    image_down = t_dep(image)  # Downsample image

    # Prepare sample for model
    sample = {
        "rgb": t_rgb(image_down).unsqueeze(0).cuda(),
        "dep": sparse_depth_down.unsqueeze(0).cuda()
    }

    # Run depth completion model
    depth_map_pred = model(sample)
    pred_down = depth_map_pred['pred']
    pred_down[pred_down < 0.1] = 0.1

    # Upsample to original size
    pred = F.interpolate(pred_down, size=original_size, mode='bilinear', align_corners=False)

    # Save results in frame-specific folder
    frame_folder = os.path.join(output_dir, f"frame{frame_idx}")
    os.makedirs(frame_folder, exist_ok=True)

    torchvision.utils.save_image(sparse_depth / (sparse_depth.max() + 1e-6),
                                 os.path.join(frame_folder, "sparse_depth.png"))
    torch.save(sparse_depth, os.path.join(frame_folder, "sparse_depth.pt"))
    torch.save(pred, os.path.join(frame_folder, "dense_depth.pt"))
    torchvision.utils.save_image((pred - pred.min()) / (pred.max() - pred.min() + 1e-6),
                                 os.path.join(frame_folder, "dense_depth.png"))

    return pred.squeeze(0)

def load_and_reformat_data(data_dir, frame_indices, depth_model, t_rgb, t_dep, output_dir="output"):
    """Load and preprocess data for a batch of frames with depth completion."""
    batch_size, H, W = len(frame_indices), 512, 640
    images = torch.zeros(batch_size, 3, H, W)
    depths = torch.zeros(batch_size, H, W)
    normals = torch.zeros(batch_size, 3, H, W)
    poses = torch.zeros(batch_size, 7)
    tstamps = torch.tensor(frame_indices, dtype=torch.float32)

    intrinsics, distortion = load_camera_params(os.path.join(data_dir, "camera_pinhole.yaml"))
    intrinsics = intrinsics.repeat(batch_size, 1)

    for i, idx in enumerate(frame_indices):
        frame_dir = os.path.join(data_dir, f"frame{idx}")
        img = Image.open(os.path.join(frame_dir, "image.png")).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)
        images[i] = img_tensor

        points = torch.from_numpy(np.load(os.path.join(frame_dir, "points.npy")))
        pose = torch.load(os.path.join(frame_dir, "pose.pt"))
        pose[3:] = pose[3:] / (torch.norm(pose[3:]) + 1e-6)

        sparse_depth = project_lidar_to_depth(points, pose, intrinsics, distortion)
        depths[i] = complete_depth(sparse_depth, img, depth_model, t_rgb, t_dep, idx, output_dir).cpu()
        poses[i] = transform_pose_to_camera_frame(pose)

        grad_y, grad_x = torch.gradient(depths[i])
        normal = torch.stack([-grad_x, -grad_y, torch.ones_like(grad_x)], dim=0)
        normals[i] = normal / (torch.norm(normal, dim=0, keepdim=True) + 1e-6)

    return {
        "images": images, "depths": depths, "normals": normals, "poses": poses,
        "tstamp": tstamps, "viz_idx": torch.tensor(frame_indices), "intrinsics": intrinsics
    }

def data_stream(queue, data_dir, total_frames, frame_step, batch_size, depth_model, t_rgb, t_dep, start=0):
    """Stream batches of frames into a queue with depth completion."""
    torch.cuda.init()
    frame_indices = list(range(start, total_frames, frame_step))
    batches = [frame_indices[i:i + batch_size] for i in range(0, len(frame_indices), batch_size)]

    for t, batch in enumerate(batches):
        queue.put((t, load_and_reformat_data(data_dir, batch, depth_model, t_rgb, t_dep, output_dir="output"),
                  t == len(batches) - 1))
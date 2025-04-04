import torch
import numpy as np
from PIL import Image
import os
import yaml
import matplotlib.pyplot as plt
from utils_new import pose_to_extrinsic, extrinsic_to_pose
import torch.cuda

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

# def project_lidar_to_depth(points, pose, intrinsics, distortion, H=512, W=640, max_depth=400.0):
#     """Project LiDAR points to a sparse depth map with distortion correction."""
#     points, pose, intrinsics, distortion = (points.cuda(), pose.cuda(),
#                                             intrinsics.cuda(), distortion.cuda())
#     fx, fy, cx, cy = intrinsics[0]
#     k1, k2, p1, p2 = distortion
#
#     points_homo = torch.cat([points[:, :3], torch.ones(len(points), 1, device=points.device)], dim=1)
#     lidar_points = torch.matmul(SE3(pose).inv().matrix(), points_homo.T).T[:, :3]
#
#     cam_points = torch.stack([-lidar_points[:, 1], -lidar_points[:, 2], lidar_points[:, 0]], dim=1)
#     depths = cam_points[:, 2]
#     valid_depth = (depths > 0.01)
#     cam_points, depths = cam_points[valid_depth], depths[valid_depth]
#
#     x_norm = cam_points[:, 0] / depths
#     y_norm = cam_points[:, 1] / depths
#
#     r2 = x_norm ** 2 + y_norm ** 2
#     radial = 1 + k1 * r2 + k2 * r2 ** 2
#     x_dist = x_norm * radial + 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm ** 2)
#     y_dist = y_norm * radial + p1 * (r2 + 2 * y_norm ** 2) + 2 * p2 * x_norm * y_norm
#
#     u = (fx * x_dist + cx).long()
#     v = (fy * y_dist + cy).long()
#     valid_pixels = (u >= 0) & (u < W) & (v >= 0) & (v < H)
#     u, v, depths = u[valid_pixels], v[valid_pixels], depths[valid_pixels]
#
#     sparse_depth = torch.zeros(H, W, device="cuda")
#     if len(depths) > 0:
#         sorted_indices = torch.argsort(depths)
#         sparse_depth[v[sorted_indices], u[sorted_indices]] = depths[sorted_indices]
#     return sparse_depth

def compute_wasserstein_distance(hist1, hist2):
    """Compute approximate 1D Wasserstein distance between two histograms."""
    cdf1 = torch.cumsum(hist1, dim=0)
    cdf2 = torch.cumsum(hist2, dim=0)
    w_dist = torch.abs(cdf1 - cdf2).sum()
    return w_dist


def select_keyframes(points_batch, poses, images, prev_ref_points=None, prev_ref_image=None,
                     w_threshold=0.5, w_lidar_weight=0.7):
    """Select keyframes based on Wasserstein distance, comparing with last keyframe."""
    batch_size = len(poses)
    all_points_xyz = [points[:, :3].cuda() for points in points_batch]  # Shape: [N, 3]
    all_images = images.cuda()  # Shape: [batch_size, 3, H, W]

    keyframe_indices = []
    bins = 50
    max_dist = 100.0  # Adjust based on your LiDAR range

    # Set initial reference: use previous batch's last keyframe if provided, else first frame
    if prev_ref_points is not None and prev_ref_image is not None:
        ref_points = prev_ref_points
        ref_image = prev_ref_image
        # print(f"Using previous batch's last keyframe as reference: {len(ref_points)} LiDAR points")
    else:
        ref_points = all_points_xyz[0]
        ref_image = all_images[0]
        keyframe_indices.append(0)  # Only add first frame for the very first batch
        # print(f"Initial reference frame: {len(ref_points)} LiDAR points")

    # Compare with last keyframe for all frames (skip 0 if no prev_ref)
    start_idx = 0 if prev_ref_points is not None else 1
    for i in range(start_idx, batch_size):
        curr_points = all_points_xyz[i]
        curr_image = all_images[i]
        # print(f"Frame {i}: {len(curr_points)} LiDAR points")

        # Compute LiDAR distribution (histogram on CPU)
        ref_dist = torch.norm(ref_points, dim=1).cpu()
        curr_dist = torch.norm(curr_points, dim=1).cpu()
        ref_hist_lidar, _ = torch.histogram(ref_dist, bins=bins, range=(0, max_dist), density=True)
        curr_hist_lidar, _ = torch.histogram(curr_dist, bins=bins, range=(0, max_dist), density=True)
        w_lidar = compute_wasserstein_distance(ref_hist_lidar, curr_hist_lidar).cuda()

        # Compute image distribution (histogram on CPU)
        ref_hist_img, _ = torch.histogram(ref_image.flatten().cpu(), bins=256, range=(0, 255), density=True)
        curr_hist_img, _ = torch.histogram(curr_image.flatten().cpu(), bins=256, range=(0, 255), density=True)
        w_image = compute_wasserstein_distance(ref_hist_img, curr_hist_img).cuda()

        # Combine distances with weights
        w_total = w_lidar_weight * w_lidar + (1 - w_lidar_weight) * w_image
        # print(
        #     f"Frame {i}: W_lidar = {w_lidar.item():.3f}, W_image = {w_image.item():.3f}, W_total = {w_total.item():.3f}")

        # Select as keyframe if total distance exceeds threshold
        if w_total > w_threshold:
            keyframe_indices.append(i)
            ref_points = curr_points  # Update reference to current keyframe
            ref_image = curr_image
            # print(f"Selected as keyframe: Frame {i}")
        # else:
        #     print(f"Frame {i} not selected (W_total below threshold)")

    # Return last keyframe data for next batch
    last_keyframe_points = ref_points
    last_keyframe_image = ref_image
    # print(f"Total keyframes selected: {len(keyframe_indices)} out of {batch_size}")
    return keyframe_indices, last_keyframe_points, last_keyframe_image


def load_and_reformat_data(data_dir, frame_indices, pipe, prev_ref_points=None, prev_ref_image=None, output_dir="output"):
    """Load data, select keyframes, and process only keyframes in batch with pipe."""
    os.makedirs(output_dir, exist_ok=True)

    batch_size, H, W = len(frame_indices), 512, 640
    images = torch.zeros(batch_size, 3, H, W)
    poses = torch.zeros(batch_size, 7)
    tstamps = torch.tensor(frame_indices, dtype=torch.float32)
    points_batch = []

    intrinsics, distortion = load_camera_params(os.path.join(data_dir, "camera_pinhole.yaml"))
    intrinsics_batch = intrinsics.repeat(batch_size, 1)

    for i, idx in enumerate(frame_indices):
        frame_dir = os.path.join(data_dir, f"frame{idx}")
        img = Image.open(os.path.join(frame_dir, "image.png")).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)
        images[i] = img_tensor

        points = torch.from_numpy(np.load(os.path.join(frame_dir, "points.npy")))
        points_batch.append(points)

        tx, ty, tz, qx, qy, qz, qw  = torch.load(os.path.join(frame_dir, "pose.pt")).tolist()
        extrinsic = pose_to_extrinsic(tx, ty, tz, qx, qy, qz, qw)
        poses[i] = extrinsic_to_pose(extrinsic)

    keyframe_indices, last_keyframe_points, last_keyframe_image = select_keyframes(
        points_batch, poses, images, prev_ref_points, prev_ref_image
    )

    # If no keyframes, return None for keyframe_data to indicate skipping
    if not keyframe_indices:
        # print(f"No keyframes selected for batch with frames {frame_indices}")
        return None, last_keyframe_points, last_keyframe_image

    # Process only keyframes in batch with pipe
    keyframe_images = images[keyframe_indices]
    keyframe_poses = poses[keyframe_indices]
    keyframe_tstamps = tstamps[keyframe_indices]
    keyframe_intrinsics = intrinsics_batch[keyframe_indices]

    pil_images = [Image.fromarray(img.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                  for img in keyframe_images]
    mono_depth_res = pipe(pil_images)
    depths = torch.stack([res["predicted_depth"] for res in mono_depth_res])
    depths = (depths + 20) / 20
    # depths = 1 / depths
    normals = torch.zeros(len(keyframe_indices), 3, H, W)
    for i in range(len(keyframe_indices)):
        grad_y, grad_x = torch.gradient(depths[i])
        normal = torch.stack([-grad_x, -grad_y, torch.ones_like(grad_x)], dim=0)
        normals[i] = normal / (torch.norm(normal, dim=0, keepdim=True) + 1e-6)

    keyframe_data = {
        "images": keyframe_images,
        "depths": depths,
        "normals": normals,
        "poses": keyframe_poses,
        "tstamp": keyframe_tstamps,
        "viz_idx": torch.tensor([frame_indices[i] for i in keyframe_indices]),
        "intrinsics": keyframe_intrinsics,
    }

    for i, idx in enumerate(keyframe_data["viz_idx"]):
        plt.figure(figsize=(10, 8))
        plt.imshow(keyframe_data["depths"][i].cpu().numpy(), cmap='viridis')
        plt.colorbar(label='Depth')
        plt.title(f'Depth Map - Keyframe {idx}')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"depth_keyframe{idx}.png"), bbox_inches='tight')
        plt.close()

    return keyframe_data, last_keyframe_points, last_keyframe_image


def data_stream(queue, data_dir, total_frames, batch_size, pipe, start=0):
    """Stream data into a queue, loading batches incrementally, ensuring at least 2 keyframes."""
    torch.cuda.init()
    frame_indices = list(range(start, total_frames))
    batches = [frame_indices[i:i + batch_size] for i in range(0, len(frame_indices), batch_size)]

    prev_ref_points = None
    prev_ref_image = None
    pending_keyframe_data = None
    pending_batch_idx = None

    for t, batch in enumerate(batches):
        try:
            keyframe_data, last_keyframe_points, last_keyframe_image = load_and_reformat_data(
                data_dir, batch, pipe, prev_ref_points, prev_ref_image, output_dir="output"
            )

            # Handle pending batch
            if pending_keyframe_data is not None:
                if keyframe_data is not None:
                    # Merge current batch with pending batch
                    for key in pending_keyframe_data:
                        if key != "viz_idx" and key != "tstamp":
                            pending_keyframe_data[key] = torch.cat(
                                [pending_keyframe_data[key], keyframe_data[key]], dim=0
                            )
                        else:
                            pending_keyframe_data[key] = torch.cat(
                                [pending_keyframe_data[key], keyframe_data[key]]
                            )
                    # Check if merged batch has at least 2 keyframes
                    if len(pending_keyframe_data["viz_idx"]) >= 2:
                        queue.put((pending_batch_idx, pending_keyframe_data, False))
                        pending_keyframe_data = None
                        pending_batch_idx = None
                # If no new keyframes, just update references and continue
                prev_ref_points = last_keyframe_points
                prev_ref_image = last_keyframe_image
                continue

            # Process current batch
            if keyframe_data is None:
                pass  # Skip silently
            elif len(keyframe_data["viz_idx"]) < 2 and t < len(batches) - 1:
                # Hold as pending if less than 2 keyframes and not the last batch
                pending_keyframe_data = keyframe_data
                pending_batch_idx = t
            else:
                # Enqueue if 2 or more keyframes, or if it's the last batch
                queue.put((t, keyframe_data, t == len(batches) - 1))

            prev_ref_points = last_keyframe_points
            prev_ref_image = last_keyframe_image

        except Exception as e:
            print(f"Error in batch {t} (frames {batch}): {e}")
            raise

    # Enqueue any remaining pending batch (even if < 2 keyframes at the end)
    if pending_keyframe_data is not None:
        queue.put((pending_batch_idx, pending_keyframe_data, True))
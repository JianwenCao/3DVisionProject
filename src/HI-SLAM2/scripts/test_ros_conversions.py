#!/usr/bin/env python3
"""
Visualize Offline Preprocessed Data

This script reads from a data folder (with structure: data/<folder_name>/frameX/)
and visualizes one of the following based on the --mode argument:
  - image: Display the image sequence.
  - lidar: Visualize the lidar point clouds in 3D.
    - Use the left/right arrow keys to step backward and forward through the frames.
  - pose: Visualize the camera pose trajectory in 3D.

Usage:
    python scripts/test_ros_conversions.py --mode [image|lidar|pose] --folder <folder_name>
"""

import argparse
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3d plotting

def get_sorted_frame_dirs(data_folder):
    """
    Returns a list of frame folder names (e.g. frame0, frame1, ...) sorted by frame number.
    """
    frame_dirs = [d for d in os.listdir(data_folder)
                  if os.path.isdir(os.path.join(data_folder, d)) and d.startswith("frame")]
    # Sort using the numeric part after "frame"
    frame_dirs.sort(key=lambda x: int(x.replace("frame", "")))
    return frame_dirs

def visualize_images(data_folder):
    """Display the image sequence."""
    frame_dirs = get_sorted_frame_dirs(data_folder)
    if not frame_dirs:
        print("No frame folders found in", data_folder)
        return

    plt.ion()  # interactive mode on
    fig, ax = plt.subplots()
    for frame in frame_dirs:
        img_path = os.path.join(data_folder, frame, "image.png")
        if os.path.exists(img_path):
            # cv2.imread returns BGR, convert to RGB.
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print("Failed to read image from", img_path)
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            ax.clear()
            ax.imshow(img_rgb)
            ax.set_title(frame)
            ax.axis('off')
            plt.pause(0.05)
    plt.ioff()
    plt.show()

def visualize_lidar(data_folder, xlim=(-50, 50), ylim=(-50, 50), zlim=(-10, 30)):
    """
    Visualize lidar points for each frame in 3D. Use the left/right arrow keys
    to step backward and forward through the frames.
    
    Args:
        data_folder (str): Path to the directory containing frame subfolders.
        xlim (tuple): (min_x, max_x) axis limits for X.
        ylim (tuple): (min_y, max_y) axis limits for Y.
        zlim (tuple): (min_z, max_z) axis limits for Z.
    """
    frame_dirs = get_sorted_frame_dirs(data_folder)
    if not frame_dirs:
        print("No frame folders found in", data_folder)
        return
    
    # Pre-load all lidar point clouds so that stepping through frames is quick
    all_points = []
    for frame in frame_dirs:
        points_path = os.path.join(data_folder, frame, "points.npy")
        if os.path.exists(points_path):
            try:
                points = np.load(points_path)  # shape: (N, 6) -> [x, y, z, r, g, b]
            except Exception as e:
                print("Error loading", points_path, ":", e)
                points = None
        else:
            print("File not found:", points_path)
            points = None
        all_points.append(points)
    
    # Create figure
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    current_idx = [0]
    
    def update_plot():
        """Clears the axes and plots the point cloud for the current frame."""
        ax.clear()
        
        points = all_points[current_idx[0]]
        if points is not None:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                       s=1, c=points[:, 3:6] / 255.0)
        
        frame_label = frame_dirs[current_idx[0]] if current_idx[0] < len(frame_dirs) else "N/A"
        ax.set_title(frame_label)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        # Optionally fix the axis limits. Uncomment if you want a fixed view:
        ax.set_xlim3d(*xlim)
        ax.set_ylim3d(*ylim)
        ax.set_zlim3d(*zlim)
        
        plt.draw()
    
    def on_key(event):
        """Keyboard event handler to move forward/backward through the frames."""
        if event.key == 'right':
            # Go forward one frame
            current_idx[0] = (current_idx[0] + 1) % len(all_points)
            update_plot()
        elif event.key == 'left':
            # Go backward one frame
            current_idx[0] = (current_idx[0] - 1) % len(all_points)
            update_plot()
    
    # Connect the key press event to our handler
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    update_plot()
    
    plt.ioff()
    plt.show()


def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion [qx, qy, qz, qw] to a 3x3 rotation matrix.
    """
    qx, qy, qz, qw = q
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm == 0:
        return np.eye(3)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
    ])
    return R

def plot_coordinate_frame(ax, translation, quaternion, scale=1):
    """
    Draw a coordinate frame (using quiver arrows) at a given translation and orientation.
    
    - x-axis: red
    - y-axis: green
    - z-axis: blue
    """
    R = quaternion_to_rotation_matrix(quaternion)
    origin = translation
    x_axis = R[:, 0] * scale
    y_axis = R[:, 1] * scale
    z_axis = R[:, 2] * scale
    ax.quiver(*origin, *x_axis, color='r', length=scale, normalize=True)
    ax.quiver(*origin, *y_axis, color='g', length=scale, normalize=True)
    ax.quiver(*origin, *z_axis, color='b', length=scale, normalize=True)

def visualize_pose(data_folder):
    """
    Visualize the pose trajectory with a coordinate frame drawn at every pose.
    Each pose file is expected to be a torch tensor of 7 elements: [tx, ty, tz, qx, qy, qz, qw].
    The trajectory is plotted as a 3D line, and a small coordinate frame (with arrows)
    is drawn at every step.
    """
    frame_dirs = get_sorted_frame_dirs(data_folder)
    if not frame_dirs:
        print("No frame folders found in", data_folder)
        return

    positions = []
    quaternions = []
    for frame in frame_dirs:
        pose_path = os.path.join(data_folder, frame, "pose.pt")
        if os.path.exists(pose_path):
            try:
                pose = torch.load(pose_path)
                pos = pose[:3].cpu().numpy()
                quat = pose[3:7].cpu().numpy()
                positions.append(pos)
                quaternions.append(quat)
            except Exception as e:
                print("Error loading pose from", pose_path, ":", e)
    if not positions:
        print("No poses loaded.")
        return

    positions = np.array(positions)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'k-', label="Trajectory")
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', s=50, label="Start")
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', s=50, label="End")
    for pos, quat in zip(positions, quaternions):
        plot_coordinate_frame(ax, pos, quat, scale=1)
    
    # Make x, y, z scales equal
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
    max_range = max((x_max - x_min), (y_max - y_min), (z_max - z_min)) * 0.5
    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_title("Camera Trajectory with Coordinate Frames at Every Step")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    save_path = os.path.join(data_folder, "pose_trajectory.png")
    plt.savefig(save_path)
    print("Pose trajectory plot saved to:", save_path)
    plt.close()

def visualize_sync_errors(data_folder):
    """
    Visualize the synchronization errors.

    Expects a file named 'sync_errors.csv' in the data_folder with CSV-formatted lines:
    frame_index,camera_timestamp,error_pose,error_lidar

    This function creates three plots:
      1. A time series plot of both pose and LiDAR errors (with frame index on the x-axis).
      2. A histogram for the distribution of pose errors.
      3. A histogram for the distribution of LiDAR errors.
    """
    sync_errors_file = os.path.join(data_folder, "sync_errors.csv")
    if not os.path.exists(sync_errors_file):
        print("Sync errors file not found in", data_folder)
        return
    
    frame_indices = []
    camera_timestamps = []
    error_poses = []
    error_lidars = []
    
    with open(sync_errors_file, 'r') as f:
        for line in f:
            if line.strip() == "" or line.startswith('#'):
                continue
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            try:
                frame_idx = int(parts[0])
                cam_time = float(parts[1])
                err_pose = float(parts[2])
                err_lidar = float(parts[3])
            except Exception as e:
                print("Error parsing line:", line, ":", e)
                continue
            
            frame_indices.append(frame_idx)
            camera_timestamps.append(cam_time)
            error_poses.append(err_pose)
            error_lidars.append(err_lidar)
    
    if not frame_indices:
        print("No valid sync error records found.")
        return

    # Time Series Plot: Errors vs. Frame Index
    plt.figure(figsize=(10, 5))
    plt.plot(frame_indices, error_poses, marker='o', linestyle='-', label='Pose Error')
    plt.plot(frame_indices, error_lidars, marker='x', linestyle='-', label='LiDAR Error')
    plt.xlabel("Frame Index")
    plt.ylabel("Sync Error (seconds)")
    plt.title("Time Synchronization Errors Over Sequence (Camera Time as Anchor)")
    plt.legend()
    plt.grid(True)
    timeseries_path = os.path.join(data_folder, "sync_errors_timeseries.png")
    plt.savefig(timeseries_path)
    plt.close()
    print("Time series plot saved to:", timeseries_path)

    # Histogram for Pose Error Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(error_poses, bins=30, alpha=0.7, label="Pose Error")
    plt.xlabel("Pose Sync Error (seconds)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Pose Time Sync Error")
    plt.legend()
    plt.grid(True)
    pose_hist_path = os.path.join(data_folder, "sync_errors_pose_hist.png")
    plt.savefig(pose_hist_path)
    plt.close()
    print("Pose error histogram saved to:", pose_hist_path)

    # Histogram for LiDAR Error Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(error_lidars, bins=30, alpha=0.7, label="LiDAR Error", color='orange')
    plt.xlabel("LiDAR Sync Error (seconds)")
    plt.ylabel("Frequency")
    plt.title("Distribution of LiDAR Time Sync Error")
    plt.legend()
    plt.grid(True)
    lidar_hist_path = os.path.join(data_folder, "sync_errors_lidar_hist.png")
    plt.savefig(lidar_hist_path)
    plt.close()
    print("LiDAR error histogram saved to:", lidar_hist_path)

    plt.show()

def visualize_reprojection(data_folder):
    """
    Visualize the reprojection of LiDAR points into the corresponding camera image.

    For each frame, the function:
      - Loads the camera image.
      - Loads the pose (assumed to be a 7-element tensor [tx, ty, tz, qx, qy, qz, qw] stored in pose.pt).
      - Loads the LiDAR points (assumed to be an (N,6) numpy array: columns [x, y, z, r, g, b]).
      - Constructs the world-to-camera transformation from the pose (the pose is in FASTLIVO camera-to-world form).
      - Converts the resulting camera coordinates from FASTLIVO 
        (x: forward, y: left, z: up) to the standard projection system (GS: x right, y down, z forward)
        using a conversion matrix T_conv.
      - Projects the LiDAR points to the image plane using the intrinsic matrix.
      - Computes two error metrics:
            * The percentage of valid points that fall outside the image.
            * The total count of valid points that are out-of-frame.
      - Displays a single window with four subplots:
           Top-left: the image with overlaid projected points.
           Top-right: a 3D view (in the camera frame) with fixed axis limits, seen from above and slightly behind.
           Bottom-left: plot of out-of-frame percentage vs. frame index.
           Bottom-right: plot of total out-of-frame count vs. frame index.
      - Allows navigation between frames using the left/right arrow keys.
    """
    frame_dirs = get_sorted_frame_dirs(data_folder)
    if not frame_dirs:
        print("No frame directories found in", data_folder)
        return

    # Load camera intrinsics from intrinsics.txt (assumed to be in data_folder)
    intrinsics_path = os.path.join(data_folder, "intrinsics.txt")
    if os.path.exists(intrinsics_path):
        K_vals = np.loadtxt(intrinsics_path)
        fx, fy, cx, cy = K_vals[0], K_vals[1], K_vals[2], K_vals[3]
        if len(K_vals) >= 6:
            W, H = int(K_vals[4]), int(K_vals[5])
        else:
            sample_img_path = os.path.join(data_folder, frame_dirs[0], "image.png")
            sample_img = cv2.imread(sample_img_path)
            if sample_img is None:
                print("Could not load sample image to infer dimensions.")
                return
            H, W, _ = sample_img.shape
    else:
        print("intrinsics.txt not found in", data_folder)
        return

    intrinsic_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0,  0,  1]])

    # Conversion matrix: from FASTLIVO (camera native, x: forward, y: left, z: up)
    # to standard projection (GS: x right, y down, z forward)
    T_conv = np.array([[0, -1, 0],
                       [0,  0, -1],
                       [1,  0,  0]])

    n_frames = len(frame_dirs)
    errors_pct = [np.nan] * n_frames       # percentage of valid points out-of-frame
    errors_total = [np.nan] * n_frames       # total count out-of-frame
    current_idx = [0]  # mutable container for current frame index

    # Create a figure with a 2x2 grid:
    # Top-left: image with reprojected LiDAR points.
    # Top-right: 3D view.
    # Bottom-left: error percentage plot.
    # Bottom-right: error total count plot.
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_3d = fig.add_subplot(gs[0, 1], projection='3d')
    ax_err_pct = fig.add_subplot(gs[1, 0])
    ax_err_total = fig.add_subplot(gs[1, 1])

    def update_display():
        frame = frame_dirs[current_idx[0]]
        img_path = os.path.join(data_folder, frame, "image.png")
        pose_path = os.path.join(data_folder, frame, "pose.pt")
        lidar_path = os.path.join(data_folder, frame, "points.npy")
        if not (os.path.exists(img_path) and os.path.exists(pose_path) and os.path.exists(lidar_path)):
            print(f"Missing data in {frame}, skipping.")
            return

        # Load image (BGR -> RGB)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Could not load image {img_path}")
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Load pose and LiDAR points
        pose = torch.load(pose_path).cpu().numpy()   # shape: (7,)
        points = np.load(lidar_path)                   # shape: (N,6)

        t = pose[:3]
        q = pose[3:]
        # Compute rotation matrix from quaternion (FASTLIVO convention)
        R_fast = quaternion_to_rotation_matrix(q)
        # Construct camera-to-world transform in FASTLIVO coordinates.
        T_cam_to_world = np.eye(4)
        T_cam_to_world[:3, :3] = R_fast
        T_cam_to_world[:3, 3] = t
        # Invert to get world-to-camera transform.
        R_world_to_cam = R_fast.T
        t_world_to_cam = -R_fast.T @ t
        T_world_to_cam = np.eye(4)
        T_world_to_cam[:3, :3] = R_world_to_cam
        T_world_to_cam[:3, 3] = t_world_to_cam

        # Reproject LiDAR points.
        # LiDAR points (in world coordinates, FASTLIVO) are in an (N,6) array.
        lidar_xyz = points[:, :3]
        colors = points[:, 3:6] / 255.0

        ones = np.ones((lidar_xyz.shape[0], 1))
        lidar_hom = np.hstack((lidar_xyz, ones))  # (N,4)
        # Transform to camera coordinates (FASTLIVO)
        cam_coords_fast = (T_world_to_cam @ lidar_hom.T).T  # (N,4)

        # For reprojection, convert from FASTLIVO to standard (GS) coordinates.
        cam_coords_gs = (T_conv @ cam_coords_fast[:, :3].T).T  # (N,3)
        valid_proj = cam_coords_gs[:, 2] > 0.1  # positive depth in GS coordinates
        cam_coords_gs_valid = cam_coords_gs[valid_proj]
        colors_valid = colors[valid_proj]

        # Pin-hole projection:
        X, Y, Z = cam_coords_gs_valid[:, 0], cam_coords_gs_valid[:, 1], cam_coords_gs_valid[:, 2]
        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy

        # Compute error metrics:
        out_of_bounds = ((u < 0) | (u > W) | (v < 0) | (v > H))
        total_valid = len(u)
        total_out = np.sum(out_of_bounds) if total_valid > 0 else 0
        err_pct = 100.0 * total_out / total_valid if total_valid > 0 else np.nan
        errors_pct[current_idx[0]] = err_pct
        errors_total[current_idx[0]] = total_out

        # Update top-left subplot: Image with projected points.
        ax_img.clear()
        ax_img.imshow(img_rgb)
        ax_img.scatter(u, v, s=2, c=colors_valid, marker='o', edgecolors='none')
        ax_img.set_title(f"Frame {frame} | Out-of-Bounds: {err_pct:.1f}% ({total_out} pts)")
        ax_img.axis('off')

        # Update top-right subplot: 3D view in camera coordinates (FASTLIVO).
        ax_3d.clear()
        # For the 3D view, show points in cam_coords_fast (without homogeneous coordinate)
        valid_3d = cam_coords_fast[:, 0] > 0.1  # show points in front of the camera (x positive)
        pts_3d = cam_coords_fast[valid_3d][:, :3]
        colors_3d = colors[valid_3d]
        if pts_3d.shape[0] > 0:
            ax_3d.scatter(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2],
                          c=colors_3d, s=2, marker='o')
        ax_3d.set_title("3D View (Camera Frame)")
        ax_3d.set_xlabel("X (forward)")
        ax_3d.set_ylabel("Y (left)")
        ax_3d.set_zlabel("Z (up)")
        # Set fixed limits for easy comparison (adjust these as needed).
        ax_3d.set_xlim([0, 70])
        ax_3d.set_ylim([-10, 10])
        ax_3d.set_zlim([-10, 10])
        # Change view: looking from above and slightly behind the camera.
        ax_3d.view_init(elev=30, azim=-125)

        # Update bottom-left subplot: Error percentage vs. frame index.
        x_vals = np.arange(n_frames)
        ax_err_pct.clear()
        ax_err_pct.plot(x_vals, errors_pct, marker='o', linestyle='-', color='b')
        ax_err_pct.set_xlabel("Frame Index")
        ax_err_pct.set_ylabel("Out-of-Bounds (%)", color='b')
        ax_err_pct.set_title("Reprojection Error (%) vs. Frame")
        ax_err_pct.set_xlim(0, n_frames-1)
        ax_err_pct.grid(True)

        # Update bottom-right subplot: Total out-of-frame count vs. frame index.
        ax_err_total.clear()
        ax_err_total.plot(x_vals, errors_total, marker='x', linestyle='--', color='r')
        ax_err_total.set_xlabel("Frame Index")
        ax_err_total.set_ylabel("Total Out-of-Frame", color='r')
        ax_err_total.set_title("Total Out-of-Frame Count vs. Frame")
        ax_err_total.set_xlim(0, n_frames-1)
        ax_err_total.grid(True)

        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'right':
            current_idx[0] = (current_idx[0] + 1) % n_frames
            update_display()
        elif event.key == 'left':
            current_idx[0] = (current_idx[0] - 1) % n_frames
            update_display()

    fig.canvas.mpl_connect('key_press_event', on_key)
    update_display()
    plt.show()



def main():
    parser = argparse.ArgumentParser(description="Visualize offline preprocessed data.")
    parser.add_argument('--mode', choices=['image', 'lidar', 'pose', 'sync_errors', 'reproj'], required=True,
                        help="Visualization mode: 'image', 'lidar', 'pose', 'sync_errors', or 'reproj'")
    parser.add_argument('--folder', required=True,
                        help="Folder name (inside data/) to load the frames from")
    args = parser.parse_args()

    data_folder = os.path.join("data", args.folder)
    if not os.path.exists(data_folder):
        print(f"Data folder HI-SLAM2/{data_folder} does not exist.")
        return

    if args.mode == "image":
        visualize_images(data_folder)
    elif args.mode == "lidar":
        visualize_lidar(data_folder)
    elif args.mode == "pose":
        visualize_pose(data_folder)
    elif args.mode == "sync_errors":
        visualize_sync_errors(data_folder)
    elif args.mode == "reproj":
        visualize_reprojection(data_folder)

if __name__ == '__main__':
    main()

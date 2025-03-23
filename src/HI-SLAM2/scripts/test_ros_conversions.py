#!/usr/bin/env python3
"""
Visualize Offline Preprocessed Data

This script reads from a data folder (with structure: data/<folder_name>/frameX/)
and visualizes one of the following based on the --mode argument:
  - image: Display the image sequence.
  - lidar: Visualize the lidar point clouds in 3D.
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
            plt.pause(0.1)
    plt.ioff()
    plt.show()

def visualize_lidar(data_folder, xlim=(-50, 50), ylim=(-50, 50), zlim=(-10, 30)):
    """
    Visualize lidar points for each frame in 3D with a fixed bounding box,
    so the axes do not change between frames.
    
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

    plt.ion()  # Enable interactive mode
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for frame in frame_dirs:
        points_path = os.path.join(data_folder, frame, "points.npy")
        if os.path.exists(points_path):
            try:
                # points shape (N,6): [x, y, z, r, g, b]
                points = np.load(points_path)
            except Exception as e:
                print("Error loading", points_path, ":", e)
                continue

            ax.clear()
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                       s=1, c=points[:, 3:6]/255.0)

            ax.set_title(frame)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            # Fix the axis limits so the viewpoint stays the same
            ax.set_xlim3d(*xlim)
            ax.set_ylim3d(*ylim)
            ax.set_zlim3d(*zlim)

            plt.draw()
            plt.pause(0.1)
        else:
            print("File not found:", points_path)

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
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize offline preprocessed data.")
    parser.add_argument('--mode', choices=['image', 'lidar', 'pose'], required=True,
                        help="Visualization mode: 'image', 'lidar', or 'pose'")
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

if __name__ == '__main__':
    main()

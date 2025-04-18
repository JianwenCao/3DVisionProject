# viz_gauss_normals.py
# ---------------------------------------
# Visualize Gaussian point cloud with false-color mapping of Gaussian normals in Open3D
# Usage:
#   python viz_gauss_normals.py --ply path/to/gaussian.ply
# This maps the normal vectors to a scalar field (e.g. normal_z) and applies a colormap.

import argparse
import numpy as np
import open3d as o3d
from plyfile import PlyData
import matplotlib.cm as cm


def load_points_quats(path):
    """Load XYZ positions and rotation quaternions from a PLY file."""
    ply = PlyData.read(path)
    v = ply['vertex'].data
    pts = np.vstack([v['x'], v['y'], v['z']]).T
    quats = np.vstack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']]).T
    return pts, quats


def quats_to_normals(quats):
    """Convert quaternions (x,y,z,w) to unit +Z axis normals."""
    x,y,z,w = quats[:,0], quats[:,1], quats[:,2], quats[:,3]
    normals = np.stack([
        2*(x*z + w*y),
        2*(y*z - w*x),
        w*w - (x*x + y*y - z*z)
    ], axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals


def main():
    parser = argparse.ArgumentParser(
        description="Open3D viewer for Gaussian point cloud normals false-color map"
    )
    parser.add_argument(
        "--ply", "-p", required=True,
        help="Path to Gaussian PLY (with x,y,z and rot_0..rot_3 fields)"
    )
    parser.add_argument(
        "--component", "-c", choices=["x","y","z"], default="z",
        help="Which normal component to map for coloring (x, y, or z)"
    )
    parser.add_argument(
        "--hist", action="store_true",
        help="If set, plot a histogram of the chosen normal component and exit"
    )
    args = parser.parse_args()

    # Load data
    pts, quats = load_points_quats(args.ply)
    # --- normalize quaternions to ensure unit length ---
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.maximum(norms, 1e-8)
    normals = quats_to_normals(quats)
    print("quat stats:", 
        np.min(quats, axis=0), 
        np.max(quats, axis=0), 
        np.mean(quats, axis=0))

    if args.hist:
        import matplotlib.pyplot as plt
        comp_idx = {"x":0, "y":1, "z":2}[args.component]
        values = normals[:, comp_idx]
        plt.hist(values, bins=100)
        plt.title(f"Histogram of Gaussian normal component '{args.component}'")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()
        return

    # Select scalar field from normals
    comp_idx = {"x":0, "y":1, "z":2}[args.component]
    scalar = normals[:, comp_idx] * 0.5 + 0.5    # map [-1,1] -> [0,1]

    # Apply a matplotlib colormap (jet) for false-color mapping
    cmap = cm.get_cmap("jet")
    colors = cmap(scalar)[:, :3]                  # RGB cols in [0,1]

    # Build Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    # attach normals if you want arrow view later
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize point cloud with false-color
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"Gaussian Normals False-Color ({args.component})",
        point_show_normal=False,
        width=800,
        height=600
    )

    parser = argparse.ArgumentParser(
        description="Open3D viewer for Gaussian point cloud normals false-color map"
    )
    parser.add_argument(
        "--ply", "-p", required=True,
        help="Path to Gaussian PLY (with x,y,z and rot_0..rot_3 fields)"
    )
    parser.add_argument(
        "--component", "-c", choices=["x","y","z"], default="z",
        help="Which normal component to map for coloring (x, y, or z)"
    )
    args = parser.parse_args()

    # Load data
    pts, quats = load_points_quats(args.ply)
    normals = quats_to_normals(quats)

    # Select scalar field from normals
    comp_idx = {"x":0, "y":1, "z":2}[args.component]
    scalar = normals[:, comp_idx] * 0.5 + 0.5    # map [-1,1] -> [0,1]

    # Apply a matplotlib colormap (jet) for false-color mapping
    cmap = cm.get_cmap("jet")
    colors = cmap(scalar)[:, :3]                  # RGB cols in [0,1]

    # Build Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    # attach normals if you want arrow view later
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize point cloud with false-color
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"Gaussian Normals False-Color ({args.component})",
        point_show_normal=False,
        width=800,
        height=600
    )

if __name__ == "__main__":
    main()

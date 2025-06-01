import os
import argparse
import numpy as np
import open3d as o3d
import cv2
import time
from glob import glob
from tqdm import trange
from scipy.spatial.transform import Rotation as R
import torch


def to_se3_matrix(pvec):
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat(pvec[4:]).as_matrix()
    pose[:3, 3] = pvec[1:4]
    return pose


def load_intrinsic_extrinsic(result, stamps):
    txt_path = os.path.join(result, 'intrinsics.txt')
    fx, fy, cx, cy, k1, k2, p1, p2 = np.loadtxt(txt_path)
    intrinsic = o3d.core.Tensor([[fx, 0,  cx],
                      [0,  fy, cy],
                      [0,   0,  1]], dtype=o3d.core.Dtype.Float64)
    poses = np.loadtxt(f'{result}/tsdf_integration/traj_full.txt')
    poses = [to_se3_matrix(poses[int(s)]) for s in stamps]
    poses = list(map(lambda x: o3d.core.Tensor(x, dtype=o3d.core.Dtype.Float64), poses))
    return intrinsic, poses


def integrate(depth_file_names, color_file_names, intrinsic, extrinsic, args):
    n_files = len(depth_file_names)
    device = o3d.core.Device('cpu:0')

    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=args.voxel_size,
        block_count=10000,
        device=device
    )

    start = time.time()

    pbar = trange(n_files, desc="Integration progress")
    for i in pbar:
        pbar.set_description(f"Integration progress, frame {i+1}/{n_files}")
        depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
        color = o3d.t.io.read_image(color_file_names[i]).to(device)
        pose = extrinsic[i]
        dep = cv2.imread(depth_file_names[i], cv2.IMREAD_ANYDEPTH) / args.depth_scale
        if dep.min() >= args.depth_max:
            continue

        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth, intrinsic, pose, args.depth_scale, args.depth_max)
    
        vbg.integrate(frustum_block_coords, depth, color, intrinsic, pose, args.depth_scale, args.depth_max)

    dt = time.time() - start
    print(f"Integration took {dt:.2f} seconds")
    return vbg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrate depth maps into TSDF')
    parser.add_argument('--result', type=str, required=True, help='Path to the result folder')
    parser.add_argument("--traj", type=str, default="whole_data.pt", help="Path to whole_data.pt for converting trajectory")
    parser.add_argument('--voxel_size', type=float, default=0.03, help='Voxel size')
    parser.add_argument('--depth_scale', type=float, default=3276.75, help='Depth scale')
    parser.add_argument('--depth_max', type=float, default=20.0, help='Maximum depth')
    parser.add_argument('--weight', type=float, default=[1], nargs='+', help='Weight threshold')
    args = parser.parse_args()

    save_dir = f"{args.result}/tsdf_integration"
    os.makedirs(save_dir, exist_ok=True)

    try:
        if not os.path.exists(args.traj):
            raise FileNotFoundError(f"Can't find trajectory file at {args.traj} for conversion.")
        whole_data = torch.load(args.traj, map_location="cpu")
        print(f"Loaded whole_data from {args.traj} with trajectory of length {len(whole_data['tstamp'])}, ")
        tstamps_full = whole_data["tstamp"].cpu().numpy()            # (N,)
        poses_full   = whole_data["poses"].cpu().numpy()            # (N,7)
        traj_full = np.hstack([
            tstamps_full[:, None],
            poses_full
        ])
        np.savetxt(
            os.path.join(save_dir, "traj_full.txt"),
            traj_full,
            fmt="%.6f",
            header="stamp tx ty tz qx qy qz qw"
        )
        print(f"Wrote full trajectory to {save_dir}/traj_full.txt\n")
    except Exception as e:
        print(f"Error during trajectory conversion: {e}")
        exit(1)

    depth_file_names = sorted(glob(f'{args.result}/renders/depth_after_opt/*'))
    color_file_names = sorted(glob(f'{args.result}/renders/image_after_opt/*'))
    stamps = [float(os.path.basename(i)[:-4]) for i in color_file_names]
    print(f"Found {len(depth_file_names)} depth maps and {len(color_file_names)} color images")

    intrinsic, extrinsic = load_intrinsic_extrinsic(args.result, stamps)
    vbg = integrate(depth_file_names, color_file_names, intrinsic, extrinsic, args)

    for w in args.weight:
        mesh = vbg.extract_triangle_mesh(weight_threshold=w)
        mesh = mesh.to_legacy()
        out = f'{save_dir}/tsdf_mesh_w{w:.1f}.ply'
        o3d.io.write_triangle_mesh(out, mesh)
        print(f"TSDF saved to {out}")

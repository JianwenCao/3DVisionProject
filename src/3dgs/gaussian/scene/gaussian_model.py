#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import time

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from gaussian.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian.utils.graphics_utils import BasicPointCloud, getWorld2View2
from gaussian.utils.sh_utils import RGB2SH
from gaussian.utils.voxel_map_utils import buildVoxelMap, updateVoxelMap, voxel_down_sample, VOXEL_SIZE, VOXEL_LOC
from dataclasses import dataclass
from typing import Dict, List, Tuple
import psutil  # For dynamic memory-based size adjustment

# import uuid
# import json

@dataclass
class BasicPointCloud:
    """Represents a basic point cloud with positions, colors, and normals."""
    points: np.ndarray  # (N, 3) array of 3D points
    colors: np.ndarray  # (N, 3) array of RGB colors
    normals: np.ndarray  # (N, 3) array of surface normals

@dataclass
class GaussianParams:
    """Stores parameters for a Gaussian voxel, including position, features, and properties."""
    xyz: torch.Tensor  # (3,) position of the Gaussian
    features: torch.Tensor  # (3, sh_dim) spherical harmonics features
    scales: torch.Tensor  # (3,) scaling factors for each axis
    rots: torch.Tensor  # (4,) quaternion rotation
    opacity: torch.Tensor  # (1,) opacity value
    normal: torch.Tensor  # (3,) surface normal

class SpatialHashTable:
    """Manages a global map of Gaussian voxels with spatial hashing for efficient lookups."""
    def __init__(self, max_global_size: int = 500000):
        """
        Initialize the SpatialHashTable.

        Args:
            max_global_size (int): Maximum number of voxels to store in global_map. Defaults to 500,000.
        """
        self.global_map: Dict[VOXEL_LOC, GaussianParams] = {}  # Maps voxel locations to their parameters (on CPU)
        self.hash_table: Dict[VOXEL_LOC, int] = {}  # Maps voxel locations to indices in the sliding window
        self.next_index = 0  # Incremental index for new voxels
        self.max_global_size = max_global_size  # Limit for global_map size

    def add(self, loc: VOXEL_LOC, params: GaussianParams) -> int:
        """
        Add a new voxel to the global map, ensuring it doesn't exceed max_global_size.

        Args:
            loc (VOXEL_LOC): The voxel location.
            params (GaussianParams): Parameters of the Gaussian voxel.

        Returns:
            int: Index assigned to the voxel.
        """
        # If global_map is full, remove the oldest voxel
        if len(self.global_map) >= self.max_global_size:
            oldest_loc = next(iter(sorted(self.global_map.keys(), key=lambda k: self.hash_table[k], reverse=True)))
            del self.global_map[oldest_loc]
            del self.hash_table[oldest_loc]
        # Store parameters on CPU to save GPU memory
        self.global_map[loc] = GaussianParams(
            xyz=params.xyz.cpu(),
            features=params.features.cpu(),
            scales=params.scales.cpu(),
            rots=params.rots.cpu(),
            opacity=params.opacity.cpu(),
            normal=params.normal.cpu()
        )
        self.hash_table[loc] = self.next_index
        self.next_index += 1
        return self.next_index - 1

    def get(self, loc: VOXEL_LOC) -> Tuple[GaussianParams, int]:
        """
        Retrieve a voxel's parameters from the global map, moving them to GPU if needed.

        Args:
            loc (VOXEL_LOC): The voxel location to query.

        Returns:
            Tuple[GaussianParams, int]: The voxel parameters (on GPU) and its index, or (None, -1) if not found.
        """
        if loc in self.global_map:
            params = self.global_map[loc]
            return GaussianParams(
                xyz=params.xyz.cuda(),
                features=params.features.cuda(),
                scales=params.scales.cuda(),
                rots=params.rots.cuda(),
                opacity=params.opacity.cuda(),
                normal=params.normal.cuda()
            ), self.hash_table.get(loc, -1)
        return None, -1

    def update(self, loc: VOXEL_LOC, params: GaussianParams):
        """
        Update the parameters of a voxel in the global map, storing on CPU.

        Args:
            loc (VOXEL_LOC): The voxel location.
            params (GaussianParams): Updated parameters.
        """
        self.global_map[loc] = GaussianParams(
            xyz=params.xyz.cpu(),
            features=params.features.cpu(),
            scales=params.scales.cpu(),
            rots=params.rots.cpu(),
            opacity=params.opacity.cpu(),
            normal=params.normal.cpu()
        )

    def remove_from_window(self, loc: VOXEL_LOC):
        """
        Remove a voxel from the sliding window's hash table.

        Args:
            loc (VOXEL_LOC): The voxel location to remove.
        """
        if loc in self.hash_table:
            del self.hash_table[loc]

class GaussianModel:
    def __init__(self, sh_degree: int, config=None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0, device="cuda")
        self._features_dc = torch.empty(0, device="cuda")
        self._features_rest = torch.empty(0, device="cuda")
        self._scaling = torch.empty(0, device="cuda")
        self._rotation = torch.empty(0, device="cuda")
        self._opacity = torch.empty(0, device="cuda")
        self.max_radii2D = torch.empty(0, device="cuda")
        self.xyz_gradient_accum = torch.empty(0, device="cuda")

        self.unique_kfIDs = torch.empty(0).int()
        self.n_obs = torch.empty(0).int()

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.config = config
        self.ply_input = None

        self.isotropic = False

        self.normals = torch.empty(0, 3, device="cuda")

        self.octree = None
        self.octree_size = None
        self.octree_initialized = False

        self.voxel_map = None  # Voxel map for downsampling point clouds

        total_memory = torch.cuda.get_device_properties(0).total_memory  # Total GPU memory in bytes
        allocated_memory = torch.cuda.memory_allocated(0)  # Currently allocated memory in bytes
        free_memory = total_memory - allocated_memory
        max_window_size = int(free_memory / (248 * 1.5))  # 248 bytes per voxel, 50% margin

        # Dynamically set max_window_size based on GPU memory (248 bytes per voxel, 50% margin)
        self.max_window_size = min(200000, max_window_size)  # Cap at 150,000 as per previous recommendation
        # Dynamically set max_global_size based on CPU memory (248 bytes per voxel, 50% margin)
        self.max_global_size = min(2000000, int(psutil.virtual_memory().available / (248 * 1.5)))
        self.sht = SpatialHashTable(max_global_size=self.max_global_size)  # Spatial hash table for global map
        self.cgb: List[Tuple[VOXEL_LOC, GaussianParams]] = []  # Current sliding window of Gaussian voxels
        self.ggb = None  # GPU buffer for rendering (xyzs, features, scales, rots, opacities, normals)
        self.prev_fov = None  # Previous field of view bounds
        self.current_fov = None  # Current field of view bounds
        print(f"Initialized LidarProcessor with max_window_size={self.max_window_size}, "
              f"max_global_size={self.max_global_size}")

    def _compute_fov_bounds(self, camera) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Axis-Aligned Bounding Box (AABB) of the camera's field of view (FoV).

        Args:
            camera: Camera object with FoVx, FoVy, R, and T attributes.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Minimum and maximum bounds of the FoV AABB.
        """
        h_fov = camera.FoVx  # Horizontal field of view in radians
        v_fov = camera.FoVy  # Vertical field of view in radians
        cam_pos = camera.T.cpu().numpy()  # Camera position
        cam_rot = camera.R.cpu().numpy()  # Camera rotation matrix

        near = 0.1  # Near plane distance
        far = 100.0  # Far plane distance

        # Compute the 8 corners of the FoV frustum
        tan_h_fov_2 = np.tan(h_fov / 2)
        tan_v_fov_2 = np.tan(v_fov / 2)
        corners = np.array([
            [-tan_h_fov_2, -tan_v_fov_2, near],
            [tan_h_fov_2, -tan_v_fov_2, near],
            [-tan_h_fov_2, tan_v_fov_2, near],
            [tan_h_fov_2, tan_v_fov_2, near],
            [-tan_h_fov_2, -tan_v_fov_2, far],
            [tan_h_fov_2, -tan_v_fov_2, far],
            [-tan_h_fov_2, tan_v_fov_2, far],
            [tan_h_fov_2, tan_v_fov_2, far],
        ])

        # Transform corners to world coordinates
        corners_world = (cam_rot @ corners.T).T + cam_pos
        min_bounds = np.min(corners_world, axis=0)
        max_bounds = np.max(corners_world, axis=0)

        return min_bounds, max_bounds


    def _is_in_fov(self, locs: List[VOXEL_LOC], min_bounds: np.ndarray, max_bounds: np.ndarray) -> np.ndarray:
        """
        Vectorized check to determine if voxels are within the camera's field of view.

        Args:
            locs (List[VOXEL_LOC]): List of voxel locations to check.
            min_bounds (np.ndarray): Minimum bounds of the FoV AABB.
            max_bounds (np.ndarray): Maximum bounds of the FoV AABB.

        Returns:
            np.ndarray: Boolean array indicating which voxels are within the FoV.
        """
        # Compute the center of each voxel
        centers = np.array([(np.array([loc.x, loc.y, loc.z]) + 0.5) * VOXEL_SIZE for loc in locs])
        # Check if centers are within the AABB bounds
        in_fov = np.all(centers >= min_bounds, axis=1) & np.all(centers <= max_bounds, axis=1)
        return in_fov

    def _update_global_map(self, camera):
        """
        Update the global map by identifying and moving voxels outside the current FoV.

        Step 1: Identify voxels from the previous sliding window that are still in the current FoV (OVERLAP).
        Step 2: Mark voxels not in the current FoV as DELETE, store them in the global map, and compact the window.

        Args:
            camera: Camera object to compute the current FoV.
        """
        if self.prev_fov is None or self.current_fov is None:
            return

        min_bounds, max_bounds = self.current_fov
        locs = [loc for loc, _ in self.cgb]
        if not locs:
            return

        # Step 1: Identify which voxels are still in the FoV
        in_fov = self._is_in_fov(locs, min_bounds, max_bounds)
        to_delete = [i for i, keep in enumerate(in_fov) if not keep]

        # Step 2: Delete and compact the sliding window
        if to_delete:
            to_delete = sorted(to_delete, reverse=True)
            for i in to_delete:
                loc, params = self.cgb[i]
                self.sht.update(loc, params)  # Store in global map (on CPU)
                if i < len(self.cgb) - 1:
                    # Swap with the last element to maintain contiguous memory
                    self.cgb[i], self.cgb[-1] = self.cgb[-1], self.cgb[i]
                    loc, _ = self.cgb[i]
                    self.sht.hash_table[loc] = i
                self.cgb.pop()
                self.sht.remove_from_window(loc)

    # def _append_new_voxels(self, new_xyz: np.ndarray, new_rgb: np.ndarray, new_normals: np.ndarray):
    #     """
    #     Append new voxels to the sliding window after identifying overlaps with the previous window.

    #     Step 3: Identify new voxels (ADD) that don't overlap with existing ones in the window.
    #     Step 4: Append new voxels to the sliding window (CGB) and update the spatial hash table.

    #     Args:
    #         new_xyz (np.ndarray): (N, 3) array of new point positions.
    #         new_rgb (np.ndarray): (N, 3) array of new point colors.
    #         new_normals (np.ndarray): (N, 3) array of new point normals.
    #     """
    #     device = "cuda"
    #     sh_dim = (self.max_sh_degree + 1) ** 2
    #     # Step 3: Compute voxel locations and identify new voxels (ADD)
    #     locs = set()
    #     for point in new_xyz:
    #         loc_xyz = point / VOXEL_SIZE
    #         loc_xyz[loc_xyz < 0] -= 1.0
    #         loc = VOXEL_LOC(int(loc_xyz[0]), int(loc_xyz[1]), int(loc_xyz[2]))
    #         locs.add(loc)
    #     add_indices = []
    #     add_locs = []
    #     for i, point in enumerate(new_xyz):
    #         loc_xyz = point / VOXEL_SIZE
    #         loc_xyz[loc_xyz < 0] -= 1.0
    #         loc = VOXEL_LOC(int(loc_xyz[0]), int(loc_xyz[1]), int(loc_xyz[2]))
    #         if loc not in self.sht.hash_table and loc in locs:
    #             add_indices.append(i)
    #             add_locs.append(loc)
    #     if not add_indices:
    #         return

    #     # Step 4: Create Gaussian parameters for new voxels
    #     add_xyz = torch.from_numpy(new_xyz[add_indices]).float().to(device).requires_grad_(False)
    #     add_rgb = torch.from_numpy(new_rgb[add_indices]).float().to(device).requires_grad_(False)
    #     add_normals = torch.from_numpy(new_normals[add_indices]).float().to(device).requires_grad_(False)

    #     # Initialize scales (log-space for numerical stability)
    #     s_xy = torch.full((len(add_indices),), VOXEL_SIZE, device=device)
    #     s_delta = torch.full_like(s_xy, 0.01)
    #     scales = torch.log(torch.stack([s_xy, s_xy, s_delta], dim=-1)).requires_grad_(False)

    #     # Compute rotations based on normals
    #     ex = torch.tensor([1.0, 0.0, 0.0], device=device).float().expand(len(add_indices), -1)
    #     cross_ex_n = torch.cross(ex, add_normals)
    #     cross_norm = torch.norm(cross_ex_n, dim=1, keepdim=True)
    #     u = cross_ex_n / (cross_norm + 1e-6)
    #     v = torch.cross(add_normals, u)
    #     rotation_matrix = torch.stack([u, v, add_normals], dim=-1)
    #     rots = self.batch_matrix_to_quaternion(rotation_matrix).requires_grad_(False)

    #     # Initialize features and opacities
    #     features = torch.zeros((len(add_indices), 3, sh_dim), device=device, dtype=torch.float32).requires_grad_(False)
    #     features[:, :3, 0] = RGB2SH(add_rgb)
    #     opacities = inverse_sigmoid(
    #         0.5 * torch.ones((len(add_indices), 1), device=device, dtype=torch.float32)).requires_grad_(False)

    #     # Append to CGB and update SHT
    #     print(f"Add {len(add_locs)} number of new voxel to CGB, current CGB size: {len(self.cgb)}")
    #     for i, (loc, xyz, feature, scale, rot, opacity, normal) in enumerate(
    #             zip(add_locs, add_xyz, features, scales, rots, opacities, add_normals)
    #     ):
    #         params = GaussianParams(
    #             xyz=xyz,
    #             features=feature,
    #             scales=scale,
    #             rots=rot,
    #             opacity=opacity,
    #             normal=normal
    #         )
    #         if len(self.cgb) < self.max_window_size:
    #             self.sht.add(loc, params)
    #             self.cgb.append((loc, params))
    #         else:
    #             print(f"Warning：CGB is full（Maximum capacity {self.max_window_size}），ignore new voxel {loc}")
    def _append_new_voxels(
            self,
            new_xyz: np.ndarray,
            new_rgb: np.ndarray,
            new_normals: np.ndarray
    ):
        """
        Vectorised replacement of the original `_append_new_voxels`.
        Functional output and side effects are **identical** for all valid inputs.
        """
        if new_xyz.size == 0:          # early‑out – nothing to do
            return

        # ------------------------------------------------------------------ #
        # 1.  Determine the voxel‑grid coordinates for every incoming point   #
        # ------------------------------------------------------------------ #
        #       ⌊ p / V ⌋   (true floor, handles negatives correctly)
        loc_int   = np.floor(new_xyz / VOXEL_SIZE).astype(np.int64)   # (N,3)

        # convert to VOXEL_LOC objects in one shot
        loc_objs  = [VOXEL_LOC(int(x), int(y), int(z)) for x, y, z in loc_int]

        # ------------------------------------------------------------------ #
        # 2.  Mask the points whose voxels are already present in the SHT     #
        # ------------------------------------------------------------------ #
        existing  = set(self.sht.hash_table.keys())        # O(hash‑table size)
        to_add_m  = np.fromiter(
            (loc not in existing for loc in loc_objs),
            dtype=bool,
            count=len(loc_objs)
        )                                                  # (N,)

        add_indices = np.nonzero(to_add_m)[0]              # 1‑D array of idx
        if add_indices.size == 0:                          # nothing new
            return

        add_locs    = [loc_objs[i] for i in add_indices]   # list[VOXEL_LOC]

        # ------------------------------------------------------------------ #
        # 3.  Build Gaussian parameters for the new voxels – all batched      #
        # ------------------------------------------------------------------ #
        device      = "cuda"
        sh_dim      = (self.max_sh_degree + 1) ** 2

        # (Nᴀ,3)
        add_xyz     = torch.as_tensor(new_xyz[add_indices],
                                    dtype=torch.float32,
                                    device=device,).requires_grad_(False)
        add_rgb     = torch.as_tensor(new_rgb[add_indices],
                                    dtype=torch.float32,
                                    device=device,).requires_grad_(False)
        add_normals = torch.as_tensor(new_normals[add_indices],
                                    dtype=torch.float32,
                                    device=device,).requires_grad_(False)

        # ── scales (log‑space) ──────────────────────────────────────────────
        s_xy        = torch.full((add_indices.size,), VOXEL_SIZE,
                                dtype=torch.float32, device=device)
        s_delta     = torch.full_like(s_xy, 0.01)
        scales      = torch.log(torch.stack([s_xy, s_xy, s_delta], dim=-1))

        # ── rotations from normals (batched) ────────────────────────────────
        ex          = torch.tensor([1.0, 0.0, 0.0], device=device)\
                        .expand(add_indices.size, -1)
        cross_ex_n  = torch.cross(ex, add_normals)
        u           = cross_ex_n / (torch.norm(cross_ex_n, dim=1, keepdim=True) + 1e-6)
        v           = torch.cross(add_normals, u)
        rot_mats    = torch.stack([u, v, add_normals], dim=-1)               # (Nᴀ,3,3)
        rots        = self.batch_matrix_to_quaternion(rot_mats)

        # ── SH features & log‑opacity ───────────────────────────────────────
        features    = torch.zeros((add_indices.size, 3, sh_dim), dtype=torch.float32, device=device)
        features[:, :3, 0] = RGB2SH(add_rgb)

        opacities   = inverse_sigmoid(
            0.5 * torch.ones((add_indices.size, 1),
                            dtype=torch.float32, device=device)
        )

        # ------------------------------------------------------------------ #
        # 4.  Append to the sliding‑window (CGB) and the spatial hash table   #
        # ------------------------------------------------------------------ #
        print(f"Add {len(add_locs)} new voxels to CGB, "
            f"current CGB size: {len(self.cgb)}")

        for loc, xyz, feat, sc, rt, op, nrm in zip(
                add_locs, add_xyz, features, scales, rots, opacities, add_normals
        ):
            params = GaussianParams(
                xyz=xyz,
                features=feat,
                scales=sc,
                rots=rt,
                opacity=op,
                normal=nrm
            )
            if len(self.cgb) < self.max_window_size:
                self.sht.add(loc, params)
                self.cgb.append((loc, params))
            else:
                print(f"Warning: CGB is full (max {self.max_window_size}); "
                    f"ignoring new voxel {loc}")


    def _sync_to_ggb(self):
        """
        Synchronize the sliding window (CGB) to the GPU buffer (GGB) for rendering.

        Step 5: Transfer CGB data to GGB, which is used by the renderer.
        """
        if not self.cgb:
            self.ggb = None
            return

        device = "cuda"

        # Concatenate parameters into contiguous tensors for efficient rendering
        xyzs = torch.cat([params.xyz.unsqueeze(0) for _, params in self.cgb], dim=0).to(device).requires_grad_(False)
        features = torch.cat([params.features.unsqueeze(0) for _, params in self.cgb], dim=0).to(device).requires_grad_(False)
        scales = torch.cat([params.scales.unsqueeze(0) for _, params in self.cgb], dim=0).to(device).requires_grad_(False)
        rots = torch.cat([params.rots.unsqueeze(0) for _, params in self.cgb], dim=0).to(device).requires_grad_(False)
        opacities = torch.cat([params.opacity.unsqueeze(0) for _, params in self.cgb], dim=0).to(device).requires_grad_(False)
        normals = torch.cat([params.normal.unsqueeze(0) for _, params in self.cgb], dim=0).to(device).requires_grad_(False)

        self.ggb = (xyzs, features, scales, rots, opacities, normals)

        # Log GPU memory usage for monitoring
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
            print(f"GPU memory: {allocated:.2f} MiB is allocated, {max_allocated:.2f} MiB peak, {reserved:.2f} MiB researved")

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None):
        cam = cam_info
        rgb_raw = (cam.original_image * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
        depth = o3d.geometry.Image(depthmap.astype(np.float32))

        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init)

    # Keyframe based
    # def create_pcd_from_lidar_points(self, cam_info, xyz, rgb, init=False):
    #
    #     # voxel_size = self.config["Dataset"].get(
    #     #     "voxel_size_init" if init else "voxel_size", 2.0
    #     # )
    #
    #     if init:
    #         downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
    #     else:
    #         downsample_factor = self.config["Dataset"]["pcd_downsample"]
    #     point_size = self.config["Dataset"]["point_size"]
    #     if "adaptive_pointsize" in self.config["Dataset"]:
    #         if self.config["Dataset"]["adaptive_pointsize"]:
    #             distances = np.sqrt(np.sum(xyz ** 2, axis=1))
    #             median_distance = np.median(distances)
    #             point_size = min(0.1, point_size * median_distance)
    #
    #     voxel_size = 0.4
    #
    #     # Build global voxel grid, xyz already in world coords, compute normals
    #     pcd_world = o3d.geometry.PointCloud()
    #     pcd_world.points = o3d.utility.Vector3dVector(xyz)
    #     pcd_world.colors = o3d.utility.Vector3dVector(rgb)
    #     pcd_ds = pcd_world.voxel_down_sample(voxel_size=voxel_size)
    #     # pcd_ds = pcd_world.random_down_sample(1.0 / downsample_factor)
    #     pcd_ds.estimate_normals(
    #         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=60)
    #     )  # TODO tune search params. Radius 1.0, max_nn 60 also works well.
    #
    #     pcd_ds.orient_normals_consistent_tangent_plane(k=10)
    #     cam_pose = -(cam_info.R.T @ cam_info.T).cpu().numpy()
    #     pcd_ds.orient_normals_towards_camera_location(cam_pose)
    #
    #     new_xyz = np.asarray(pcd_ds.points)
    #     new_rgb = np.asarray(pcd_ds.colors)
    #     new_normals = np.asarray(pcd_ds.normals)
    #
    #     pcd = BasicPointCloud(points=new_xyz, colors=new_rgb, normals=new_normals)
    #     self.ply_input = pcd
    #
    #     fused_point_cloud = torch.from_numpy(new_xyz).float().cuda()
    #     fused_color = RGB2SH(torch.from_numpy(new_rgb).float().cuda())
    #     features = torch.zeros(
    #         (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),
    #         device="cuda", dtype=torch.float32
    #     )
    #     features[:, :3, 0] = fused_color
    #     features[:, 3:, 1:] = 0.0
    #
    #     dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 1e-7) * point_size
    #     scales = torch.log(torch.sqrt(dist2))[..., None]
    #     if not self.isotropic:
    #         scales = scales.repeat(1, 3)
    #
    #     if self.config["Training"]["rotation_init"]:
    #         normal_rot_mats = self.batch_gaussian_rotation(torch.from_numpy(new_normals).float().cuda())
    #         normal_quats = self.batch_matrix_to_quaternion(normal_rot_mats)
    #         rots = normal_quats.clone().detach().requires_grad_(True).to("cuda")
    #         # For normal supervision:
    #         normals_cuda = torch.from_numpy(new_normals).float().cuda()
    #     else:
    #         rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    #         rots[:, 0] = 1
    #         normals_cuda = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
    #
    #     num_points = fused_point_cloud.shape[0]
    #     opacities = inverse_sigmoid(
    #         0.5 * torch.ones((num_points, 1), device="cuda", dtype=torch.float32)
    #     )
    #
    #     # Count Gaussians (number of points in fused_point_cloud)
    #     gaussian_count = num_points
    #
    #     # Save Gaussian count for this frame to a file
    #     frame_id = str(uuid.uuid4())  # Generate unique frame ID
    #     gaussian_log = {
    #         "frame_id": frame_id,
    #         "gaussian_count": int(gaussian_count),
    #         "timestamp": str(torch.cuda.Event().record())
    #     }
    #
    #     # Define output file path
    #     output_file = "gaussian_counts_keyframe_new.json"
    #
    #     # Load existing data or create new
    #     if os.path.exists(output_file):
    #         with open(output_file, 'r') as f:
    #             try:
    #                 data = json.load(f)
    #                 if not isinstance(data, list):
    #                     data = [data]
    #             except json.JSONDecodeError:
    #                 data = []
    #     else:
    #         data = []
    #
    #     # Append new entry
    #     data.append(gaussian_log)
    #
    #     # Save to file
    #     with open(output_file, 'w') as f:
    #         json.dump(data, f, indent=4)
    #
    #     return fused_point_cloud, features, scales, rots, opacities, normals_cuda

    # Old voxel based
    # def create_pcd_from_lidar_points(self, cam_info, xyz, rgb, init=False):
    #     # Load octree parameters from config
    #     voxel_size =  self.config["Dataset"].get("voxel_size", 1.0) # Root voxel size in meters, fixed to 0.5m
    #     max_layer = self.config["Dataset"].get("max_layer", 2)  # Maximum octree depth, fixed to 2 layers
    #     max_points_num = self.config["Dataset"].get("max_points_per_voxel", 50)  # Max points per voxel
    #
    #     # Build octree-based point cloud, xyz already in world coords, compute normals
    #     pcd_world = o3d.geometry.PointCloud()
    #     pcd_world.points = o3d.utility.Vector3dVector(xyz)
    #     pcd_world.colors = o3d.utility.Vector3dVector(rgb)
    #
    #     # Compute bounding box to determine size_expand
    #     bbox = pcd_world.get_axis_aligned_bounding_box()
    #     max_bound = np.max(bbox.get_extent())  # Maximum extent of the bounding box
    #     size_expand = min(max(voxel_size / max_bound, 0.01),
    #                       1.0)  # Convert voxel_size to relative scale, clamp to [0.01, 1.0]
    #
    #     # Create octree with specified max depth and computed size_expand
    #     octree = o3d.geometry.Octree(max_depth=max_layer)
    #     octree.convert_from_point_cloud(pcd_world, size_expand=size_expand)
    #
    #     # Down-sample by selecting points from leaf nodes, respecting max_points_num, and track depths for scales
    #     def extract_leaf_points(octree, max_points_num):
    #         points = []
    #         colors = []
    #         depths = []  # Track depth of each point for scale initialization
    #
    #         def traverse(node, node_info):
    #             if isinstance(node, o3d.geometry.OctreeLeafNode):
    #                 if node.indices:
    #                     # Select up to max_points_num points from the node
    #                     selected_indices = node.indices[:min(len(node.indices), max_points_num)]
    #                     points.extend(np.asarray(pcd_world.points)[selected_indices])
    #                     colors.extend(np.asarray(pcd_world.colors)[selected_indices])
    #                     depths.extend([node_info.depth] * len(selected_indices))  # Record depth for each point
    #
    #         octree.traverse(traverse)
    #         return np.array(points), np.array(colors), np.array(depths)
    #
    #     new_xyz, new_rgb, point_depths = extract_leaf_points(octree, max_points_num)
    #
    #     # Create down-sampled point cloud
    #     pcd_ds = o3d.geometry.PointCloud()
    #     pcd_ds.points = o3d.utility.Vector3dVector(new_xyz)
    #     pcd_ds.colors = o3d.utility.Vector3dVector(new_rgb)
    #
    #     # Estimate normals
    #     pcd_ds.estimate_normals(
    #         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=60)
    #     )  # TODO tune search params. Radius 1.0, max_nn 60 also works well.
    #
    #     pcd_ds.orient_normals_consistent_tangent_plane(k=10)
    #     cam_pose = -(cam_info.R.T @ cam_info.T).cpu().numpy()
    #     pcd_ds.orient_normals_towards_camera_location(cam_pose)
    #
    #     new_normals = np.asarray(pcd_ds.normals)
    #
    #     pcd = BasicPointCloud(points=new_xyz, colors=new_rgb, normals=new_normals)
    #     self.ply_input = pcd
    #
    #     fused_point_cloud = torch.from_numpy(new_xyz).float().cuda()
    #     fused_color = RGB2SH(torch.from_numpy(new_rgb).float().cuda())
    #     features = torch.zeros(
    #         (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),
    #         device="cuda", dtype=torch.float32
    #     )
    #     features[:, :3, 0] = fused_color
    #     features[:, 3:, 1:] = 0.0
    #
    #     # Initialize scales using LiDAR-Camera joint method: S = diag(s_delta, s_y, s_z)
    #     slice_thickness = 0.01  # Hyper-parameter for s_delta (plane thickness in meters)
    #     depths_tensor = torch.from_numpy(point_depths).float().cuda()
    #     # Compute s_y, s_z based on voxel size at each point's depth
    #     voxel_sizes = voxel_size / (2 ** depths_tensor)  # Voxel size at each depth (e.g., 0.5, 0.25, 0.125)
    #     # s_y = s_z = voxel_sizes  # s_y and s_z are equal, based on voxel size
    #     # s_delta = torch.full_like(s_y, slice_thickness)  # s_delta is constant
    #     s_xy = voxel_sizes  # s_x and s_y are equal, based on voxel size
    #     s_delta = torch.full_like(s_xy, slice_thickness)  # s_delta is constant
    #
    #     # # Construct scales as (s_delta, s_y, s_z) in the local frame
    #     # scales = torch.stack([s_delta, s_y, s_z], dim=-1)  # Shape: (N, 3)
    #     # Construct scales as (s_x, s_y, s_delta) in the local frame
    #     scales = torch.stack([s_xy, s_xy, s_delta], dim=-1)  # Shape: (N, 3)
    #
    #     # Apply log transformation as in original code
    #     scales = torch.log(scales)
    #
    #     if self.config["Training"]["rotation_init"]:
    #         normal_rot_mats = self.batch_gaussian_rotation(torch.from_numpy(new_normals).float().cuda())
    #         normal_quats = self.batch_matrix_to_quaternion(normal_rot_mats)
    #         rots = normal_quats.clone().detach().requires_grad_(True).to("cuda")
    #         # For normal supervision:
    #         normals_cuda = torch.from_numpy(new_normals).float().cuda()
    #     else:
    #         rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    #         rots[:, 0] = 1
    #         normals_cuda = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
    #
    #     num_points = fused_point_cloud.shape[0]
    #     opacities = inverse_sigmoid(
    #         0.5 * torch.ones((num_points, 1), device="cuda", dtype=torch.float32)
    #     )
    #     a = 1
    #     # # Count Gaussians (number of points in fused_point_cloud)
    #     # gaussian_count = num_points
    #     #
    #     # # Save Gaussian count for this frame to a file
    #     # frame_id = str(uuid.uuid4())  # Generate unique frame ID
    #     # gaussian_log = {
    #     #     "frame_id": frame_id,
    #     #     "gaussian_count": int(gaussian_count),
    #     #     "timestamp": str(torch.cuda.Event().record())
    #     # }
    #     #
    #     # # Define output file path
    #     # output_file = "gaussian_counts.json"
    #     #
    #     # # Load existing data or create new
    #     # if os.path.exists(output_file):
    #     #     with open(output_file, 'r') as f:
    #     #         try:
    #     #             data = json.load(f)
    #     #             if not isinstance(data, list):
    #     #                 data = [data]
    #     #         except json.JSONDecodeError:
    #     #             data = []
    #     # else:
    #     #     data = []
    #     #
    #     # # Append new entry
    #     # data.append(gaussian_log)
    #     #
    #     # # Save to file
    #     # with open(output_file, 'w') as f:
    #     #     json.dump(data, f, indent=4)
    #
    #     return fused_point_cloud, features, scales, rots, opacities, normals_cuda

    def create_pcd_from_lidar_points(self, camera, xyz: np.ndarray, rgb: np.ndarray, init: bool = False):
        # Update the field of view
        self.prev_fov = self.current_fov
        self.current_fov = self._compute_fov_bounds(camera)

        # Build or update the voxel map
        if init or self.voxel_map is None:
            self.voxel_map = buildVoxelMap(xyz, rgb)
            print(f"Building voxel map，including {len(self.voxel_map)} number of voxel")
        else:
            updateVoxelMap(xyz, self.voxel_map, rgb)
            print(f"Updating voxel map，including {len(self.voxel_map)} number of voxel")

        # Downsample the point cloud to create voxels
        new_xyz, new_rgb, new_normals = voxel_down_sample(self.voxel_map)

        # Step 1 & 2: Update the global map, remove and compact voxels outside FoV
        self._update_global_map(camera)

        # Step 3 & 4: Identify and append new voxels to the sliding window
        t0 = time.time()
        self._append_new_voxels(new_xyz, new_rgb, new_normals)
        t1 = time.time()
        print(t1-t0)

        # Step 5: Sync the sliding window to the GPU buffer
        self._sync_to_ggb()

        if self.ggb is None:
            device = "cuda"
            sh_dim = (self.max_sh_degree + 1) ** 2
            empty = torch.zeros((0, 3), device=device)
            return (empty, torch.zeros((0, 3, sh_dim), device=device), torch.zeros((0, 3), device=device),
                    torch.zeros((0, 4), device=device), torch.zeros((0, 1), device=device), empty)

        return self.ggb


    @staticmethod
    def batch_gaussian_rotation(normals: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of normals (shape: [N, 3]) in the Gaussian coordinate system,
        return a batch of 3x3 rotation matrices (shape: [N, 3, 3]) that define
        Gaussian plane.
        """
        # Following GS-LIVO Gaussian rotation initalization
        # In FAST-LIVO / GS-LIVO, camera optical axis was +x vector
        # In transformed (GS) coords, optical axis is +z axis

        device = normals.device
        cam_opt_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
        cam_opt_axis_batch = cam_opt_axis.unsqueeze(0).expand_as(normals)  # (N,3)

        # Basis vector v1: optical axis X normals
        v1 = torch.cross(cam_opt_axis_batch, normals, dim=1)  # (N,3)
        v1_norm = v1.norm(dim=1, keepdim=True)           # (N,1)

        # Handle near-parallel normals where cross product ~0.
        threshold = 1e-6
        use_alt = v1_norm < threshold
        e_alt = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0)
        v1_alt = torch.cross(e_alt, normals, dim=1)
        v1 = torch.where(use_alt, v1_alt, v1)  # (N,3)
        v1 = v1 / (v1.norm(dim=1, keepdim=True) + 1e-6)

        # Basis vector v2: normals x v1
        v2 = torch.cross(normals, v1, dim=1)
        v2 = v2 / (v2.norm(dim=1, keepdim=True) + 1e-6)

        # Basis vector v3: normals
        v3 = normals / (normals.norm(dim=1, keepdim=True) + 1e-6)

        R = torch.stack([v1, v2, v3], dim=2)  # (N,3,3)

        return R

    @staticmethod
    @torch.no_grad()
    def batch_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
        """
        Vectorized conversion from rotation matrices [N,3,3]
        to quaternions [N,4], in (x, y, z, w) format.
        """
        N = R.shape[0]
        quat = torch.empty((N, 4), device=R.device, dtype=R.dtype)

        m00 = R[:, 0, 0]
        m11 = R[:, 1, 1]
        m22 = R[:, 2, 2]
        trace = m00 + m11 + m22

        # Case 1: trace > 0
        mask_pos = trace > 0.0
        if mask_pos.any():
            s = torch.sqrt(trace[mask_pos] + 1.0) * 2.0
            quat[mask_pos, 3] = 0.25 * s  # w
            quat[mask_pos, 0] = (R[mask_pos, 2, 1] - R[mask_pos, 1, 2]) / s  # x
            quat[mask_pos, 1] = (R[mask_pos, 0, 2] - R[mask_pos, 2, 0]) / s  # y
            quat[mask_pos, 2] = (R[mask_pos, 1, 0] - R[mask_pos, 0, 1]) / s  # z

        # Case 2: m00 is largest
        mask_0 = (R[:, 0, 0] >= R[:, 1, 1]) & (R[:, 0, 0] >= R[:, 2, 2]) & (~mask_pos)
        if mask_0.any():
            s = torch.sqrt(1.0 + R[mask_0, 0, 0] - R[mask_0, 1, 1] - R[mask_0, 2, 2]) * 2.0
            quat[mask_0, 0] = 0.25 * s  # x
            quat[mask_0, 3] = (R[mask_0, 2, 1] - R[mask_0, 1, 2]) / s
            quat[mask_0, 1] = (R[mask_0, 0, 1] + R[mask_0, 1, 0]) / s
            quat[mask_0, 2] = (R[mask_0, 0, 2] + R[mask_0, 2, 0]) / s

        # Case 3: m11 is largest
        mask_1 = (R[:, 1, 1] >= R[:, 0, 0]) & (R[:, 1, 1] >= R[:, 2, 2]) & (~mask_pos) & (~mask_0)
        if mask_1.any():
            s = torch.sqrt(1.0 + R[mask_1, 1, 1] - R[mask_1, 0, 0] - R[mask_1, 2, 2]) * 2.0
            quat[mask_1, 1] = 0.25 * s  # y
            quat[mask_1, 3] = (R[mask_1, 0, 2] - R[mask_1, 2, 0]) / s
            quat[mask_1, 0] = (R[mask_1, 0, 1] + R[mask_1, 1, 0]) / s
            quat[mask_1, 2] = (R[mask_1, 1, 2] + R[mask_1, 2, 1]) / s

        # Case 4: m22 is largest
        mask_2 = ~(mask_pos | mask_0 | mask_1)
        if mask_2.any():
            s = torch.sqrt(1.0 + R[mask_2, 2, 2] - R[mask_2, 0, 0] - R[mask_2, 1, 1]) * 2.0
            quat[mask_2, 2] = 0.25 * s  # z
            quat[mask_2, 3] = (R[mask_2, 1, 0] - R[mask_2, 0, 1]) / s
            quat[mask_2, 0] = (R[mask_2, 0, 2] + R[mask_2, 2, 0]) / s
            quat[mask_2, 1] = (R[mask_2, 1, 2] + R[mask_2, 2, 1]) / s

        return quat

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def extend_from_pcd(
        self, fused_point_cloud, features, scales, rots, opacities, normals, kf_id
    ):
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))
        new_normals = normals.detach()        # TODO check plain Tensor, no Autograd

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_normals,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
        )

    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None
    ):
        fused_point_cloud, features, scales, rots, opacities = (
            self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap)
        )
        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, kf_id
        )

    def extend_from_lidar_seq(
            self, cam_info, kf_id=-1, init=False, pcd=None
    ):
        xyz = pcd[:, :3]
        rgb = pcd[:, 3:6] / 255.0

        fused_point_cloud, features, scales, rots, opacities, normals = (
            self.create_pcd_from_lidar_points(cam_info, xyz, rgb, init)
        )

        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, normals, kf_id
        )

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.max_steps = training_args.position_lr_max_steps

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    max_steps=self.max_steps+1000,
                )

                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(
        self, visibility_filters
    ):  ##Reset opacity for only non-visible gaussians
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)

        for filter in visibility_filters:
            opacities_new[filter] = self.get_opacity[filter]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.normals   = self.normals[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_normals=None,
        new_kf_ids=None,
        new_n_obs=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if new_normals is not None:
            self.normals = torch.cat((self.normals,
                                      new_normals.detach()), dim=0)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_normals = self.normals[selected_pts_mask].repeat(N, 1).detach()

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_normals,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )

        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_normals = self.normals[selected_pts_mask]

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_normals,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = torch.logical_and((self.get_opacity < min_opacity).squeeze(), (self.unique_kfIDs != self.unique_kfIDs.max()).cuda())
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

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
from scipy.spatial.transform import Rotation as R


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

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.config = config
        self.ply_input = None

        self.isotropic = False

        # self.normals = torch.empty(0, device="cuda")

    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

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

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None):
        cam = cam_info
        rgb_raw = (cam.original_image * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
        depth = o3d.geometry.Image(depthmap.astype(np.float32))

        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init)
    
    # def create_pcd_from_lidar_points(self, cam, xyz, rgb, init=False):
    #     if init:
    #         voxel_size = self.config["Dataset"].get("voxel_size_init", 2.0)
    #     else:
    #         voxel_size = self.config["Dataset"].get("voxel_size", 2.0)
    #     point_size = self.config["Dataset"]["point_size"]
    #     s_delta = self.config["Dataset"].get("s_delta", 0.01)

    #     # Transform lidar points in world frame to camera frame
    #     W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
    #     points_homo = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
    #     cam_points = points_homo @ W2C.T
    #     xyz_cam = cam_points[:, :3]

    #     pcd_tmp = o3d.geometry.PointCloud()
    #     pcd_tmp.points = o3d.utility.Vector3dVector(xyz_cam)
    #     pcd_tmp.colors = o3d.utility.Vector3dVector(rgb)

    #     # Estimate normals for point cloud
    #     pcd_tmp.estimate_normals(
    #         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    #     )

    #     # Create octree
    #     max_depth = 8
    #     octree = o3d.geometry.Octree(max_depth)
    #     octree.convert_from_point_cloud(pcd_tmp, size_expand=0.01)

    #     new_xyz = []
    #     new_rgb = []
    #     new_normals = []

    #     def traverse_octree(node, node_info):
    #         if isinstance(node, o3d.geometry.OctreeLeafNode):
    #             if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
    #                 indices = node.indices
    #                 if len(indices) > 0:
    #                     points = np.asarray(pcd_tmp.points)[indices]
    #                     colors = np.asarray(pcd_tmp.colors)[indices]
    #                     node_normals = np.asarray(pcd_tmp.normals)[indices]
    #                     centroid = np.mean(points, axis=0)
    #                     centroid_color = np.mean(colors, axis=0)
    #                     centroid_normal = np.mean(node_normals, axis=0) / (
    #                             np.linalg.norm(np.mean(node_normals, axis=0)) + 1e-6
    #                     )
    #                     new_xyz.append(centroid)
    #                     new_rgb.append(centroid_color)
    #                     new_normals.append(centroid_normal)

    #     octree.traverse(traverse_octree)
    #     new_xyz = np.array(new_xyz)
    #     new_rgb = np.array(new_rgb)
    #     new_normals = np.array(new_normals)

    #     if len(new_xyz) == 0:
    #         print("No points found in octree, falling back to voxel downsampling.")
    #         pcd_tmp = pcd_tmp.voxel_down_sample(voxel_size=voxel_size)
    #         new_xyz = np.asarray(pcd_tmp.points)
    #         new_rgb = np.asarray(pcd_tmp.colors)
    #         pcd_tmp.estimate_normals(
    #             search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    #         )
    #         new_normals = np.asarray(pcd_tmp.normals)

    #     pcd = BasicPointCloud(
    #         points=new_xyz, colors=new_rgb, normals=new_normals
    #     )
    #     self.ply_input = pcd

    #     fused_point_cloud = torch.from_numpy(new_xyz).float().cuda()
    #     fused_color = RGB2SH(torch.from_numpy(new_rgb).float().cuda())
    #     features = (
    #         torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
    #         .float()
    #         .cuda()
    #     )
    #     features[:, :3, 0] = fused_color
    #     features[:, 3:, 1:] = 0.0

    #     # GS-LIVO 初始化：缩放和旋转
    #     num_points = fused_point_cloud.shape[0]
    #     scales = torch.zeros((num_points, 3), device="cuda")
    #     rots = torch.zeros((num_points, 4), device="cuda")


    #     R = self.batch_gaussian_rotation(torch.from_numpy(new_normals).float().cuda())
    #     quats = self.batch_matrix_to_quaternion(R)
    #     rots = torch.tensor(quats, device="cuda")
    #     # self.normals = torch.from_numpy(new_normals).float().cuda()

    #     dist2 = (
    #             torch.clamp_min(
    #                 distCUDA2(fused_point_cloud),
    #                 0.0000001,
    #             )
    #             * point_size
    #     )
    #     scales = torch.log(torch.sqrt(dist2))[..., None]
    #     if not self.isotropic:
    #         scales = scales.repeat(1, 3)

    #     # debug sanity checks
    #     assert not torch.isinf(scales).any(),  "some scales are -inf!"
    #     assert (scales.exp()>0).all(),         "some Gaussians still zero‐sized"
    #     assert torch.isfinite(rots).all(),     "invalid quaternion values"

    #     opacities = inverse_sigmoid(
    #         0.5 * torch.ones((num_points, 1), dtype=torch.float, device="cuda")
    #     )

    #     return fused_point_cloud, features, scales, rots, opacities
    
    def create_pcd_from_lidar_points(self, cam, xyz, rgb, init=False):
        # pick voxel size
        voxel_size = self.config["Dataset"].get(
            "voxel_size_init" if init else "voxel_size", 2.0
        )

        # ----------------------------------------------------------------------------
        # 1) Build a world‐space pointcloud directly (skip camera‐frame octree entirely)
        # ----------------------------------------------------------------------------

        # xyz is already in world coords
        pcd_world = o3d.geometry.PointCloud()
        pcd_world.points = o3d.utility.Vector3dVector(xyz)
        pcd_world.colors = o3d.utility.Vector3dVector(rgb)

        # downsample on a fixed grid in world‐space
        pcd_ds = pcd_world.voxel_down_sample(voxel_size=0.1)

        # estimate normals once (in world frame)
        pcd_ds.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        new_xyz     = np.asarray(pcd_ds.points)
        new_rgb     = np.asarray(pcd_ds.colors)
        new_normals = np.asarray(pcd_ds.normals)

        # ----------------------------------------------------------------------------
        # 2) package into your GS pipeline exactly as before
        # ----------------------------------------------------------------------------
        pcd = BasicPointCloud(points=new_xyz, colors=new_rgb, normals=new_normals)
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(new_xyz).float().cuda()
        fused_color       = RGB2SH(torch.from_numpy(new_rgb).float().cuda())

        # build your feature tensor
        features = torch.zeros(
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),
            device="cuda", dtype=torch.float32
        )
        features[:, :3, 0] = fused_color
        # rest already zero

        # initialize scales & rotations exactly as before
        num_points = fused_point_cloud.shape[0]
        # rotation from normals
        Rmats = self.batch_gaussian_rotation(torch.from_numpy(new_normals).float().cuda())
        quats = self.batch_matrix_to_quaternion(Rmats)
        rots  = torch.tensor(quats, device="cuda")

        # scale from point‐spacing
        point_size = self.config["Dataset"]["point_size"]
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 1e-7) * point_size
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        # opacities
        opacities = inverse_sigmoid(
            0.5 * torch.ones((num_points, 1), device="cuda", dtype=torch.float32)
        )

        return fused_point_cloud, features, scales, rots, opacities


    @staticmethod
    def batch_gaussian_rotation(normals: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of normals (shape: [N, 3]) in the *new* coordinate system,
        return a batch of 3x3 rotation matrices (shape: [N, 3, 3]) that orient
        the 'Gaussian plane' as in FAST-LIVO (Equation 3), but adapted to your
        new coordinate axes.
        """

        # 1) The old FAST-LIVO 'x-axis' was "into screen". In your new coords,
        #    that direction is your +Z axis. So define e_x_old accordingly:
        device = normals.device
        e_x_old = torch.tensor([0.0, 0.0, 1.0], device=device)

        # Expand to match shape [N,3] so we can cross in batch.
        e_x_old_batch = e_x_old.unsqueeze(0).expand_as(normals)  # (N,3)

        # 2) Cross e_x_old with each normal to get the first basis vector, v1.
        v1 = torch.cross(e_x_old_batch, normals, dim=1)  # (N,3)
        v1_norm = v1.norm(dim=1, keepdim=True)           # (N,1)

        # 3) Some normals may be nearly parallel to e_x_old -> cross product ~ 0.
        #    For those, pick a different "backup" reference (e.g., [0,1,0]).
        threshold = 1e-6
        use_alt = v1_norm < threshold
        e_alt = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0)
        v1_alt = torch.cross(e_alt, normals, dim=1)
        v1 = torch.where(use_alt, v1_alt, v1)  # shape still (N,3)

        # Normalize v1
        v1 = v1 / (v1.norm(dim=1, keepdim=True) + 1e-6)

        # 4) Second basis vector v2 = n_i x v1
        v2 = torch.cross(normals, v1, dim=1)
        v2 = v2 / (v2.norm(dim=1, keepdim=True) + 1e-6)

        # 5) Third basis vector v3 = n_i (ensure normalized)
        v3 = normals / (normals.norm(dim=1, keepdim=True) + 1e-6)

        # 6) Stack into 3×3 rotation matrices: R = [v1, v2, v3]
        #    with each vector as a column. That yields shape (N,3,3).
        R = torch.stack([v1, v2, v3], dim=2)  # shape: (N,3,3)

        return R

    @staticmethod
    @torch.no_grad()
    def batch_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
        """
        Vectorized conversion from rotation matrices [N,3,3]
        to quaternions [N,4], in (x, y, z, w) format.
        """
        # Implementation is the same as shown previously or any standard
        # 'matrix to quaternion' PyTorch snippet, ensuring it’s fully batched.
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
            quat[mask_pos, 3] = 0.25 * s              # w
            quat[mask_pos, 0] = (R[mask_pos, 2, 1] - R[mask_pos, 1, 2]) / s  # x
            quat[mask_pos, 1] = (R[mask_pos, 0, 2] - R[mask_pos, 2, 0]) / s  # y
            quat[mask_pos, 2] = (R[mask_pos, 1, 0] - R[mask_pos, 0, 1]) / s  # z

        # Case 2: m00 is largest
        mask_0 = (R[:, 0, 0] >= R[:, 1, 1]) & (R[:, 0, 0] >= R[:, 2, 2]) & (~mask_pos)
        if mask_0.any():
            s = torch.sqrt(1.0 + R[mask_0, 0, 0] - R[mask_0, 1, 1] - R[mask_0, 2, 2]) * 2.0
            quat[mask_0, 0] = 0.25 * s               # x
            quat[mask_0, 3] = (R[mask_0, 2, 1] - R[mask_0, 1, 2]) / s
            quat[mask_0, 1] = (R[mask_0, 0, 1] + R[mask_0, 1, 0]) / s
            quat[mask_0, 2] = (R[mask_0, 0, 2] + R[mask_0, 2, 0]) / s

        # Case 3: m11 is largest
        mask_1 = (R[:, 1, 1] >= R[:, 0, 0]) & (R[:, 1, 1] >= R[:, 2, 2]) & (~mask_pos) & (~mask_0)
        if mask_1.any():
            s = torch.sqrt(1.0 + R[mask_1, 1, 1] - R[mask_1, 0, 0] - R[mask_1, 2, 2]) * 2.0
            quat[mask_1, 1] = 0.25 * s               # y
            quat[mask_1, 3] = (R[mask_1, 0, 2] - R[mask_1, 2, 0]) / s
            quat[mask_1, 0] = (R[mask_1, 0, 1] + R[mask_1, 1, 0]) / s
            quat[mask_1, 2] = (R[mask_1, 1, 2] + R[mask_1, 2, 1]) / s

        # Case 4: m22 is largest
        mask_2 = ~(mask_pos | mask_0 | mask_1)
        if mask_2.any():
            s = torch.sqrt(1.0 + R[mask_2, 2, 2] - R[mask_2, 0, 0] - R[mask_2, 1, 1]) * 2.0
            quat[mask_2, 2] = 0.25 * s               # z
            quat[mask_2, 3] = (R[mask_2, 1, 0] - R[mask_2, 0, 1]) / s
            quat[mask_2, 0] = (R[mask_2, 0, 2] + R[mask_2, 2, 0]) / s
            quat[mask_2, 1] = (R[mask_2, 1, 2] + R[mask_2, 2, 1]) / s

        return quat

    # def create_pcd_from_lidar_points(self, cam, xyz, rgb, init=False):
    #     if init:
    #         voxel_size = self.config["Dataset"].get("voxel_size_init", 2.0)  # initial voxel size
    #     else:
    #         voxel_size = self.config["Dataset"].get("voxel_size", 2.0)  # voxel size
    #     point_size = self.config["Dataset"]["point_size"]

    #     # transform lidar point in world frame to camera frame
    #     W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
    #     points_homo = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
    #     cam_points = points_homo @ W2C.T
    #     xyz_cam = cam_points[:, :3]

    #     pcd_tmp = o3d.geometry.PointCloud()
    #     pcd_tmp.points = o3d.utility.Vector3dVector(xyz_cam)
    #     pcd_tmp.colors = o3d.utility.Vector3dVector(rgb)

    #     # Create octree
    #     max_depth = 8  # Max depth for octree
    #     octree = o3d.geometry.Octree(max_depth)
    #     octree.convert_from_point_cloud(pcd_tmp, size_expand=0.01)

    #     new_xyz = []
    #     new_rgb = []

    #     def traverse_octree(node, node_info):
    #         if isinstance(node, o3d.geometry.OctreeLeafNode):
    #             if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
    #                 indices = node.indices
    #                 if len(indices) > 0:
    #                     points = np.asarray(pcd_tmp.points)[indices]
    #                     colors = np.asarray(pcd_tmp.colors)[indices]
    #                     centroid = np.mean(points, axis=0)
    #                     centroid_color = np.mean(colors, axis=0)
    #                     new_xyz.append(centroid)
    #                     new_rgb.append(centroid_color)

    #     octree.traverse(traverse_octree)
    #     new_xyz = np.array(new_xyz)
    #     new_rgb = np.array(new_rgb)

    #     if len(new_xyz) == 0:
    #         print("No points found in octree, falling back to voxel downsampling.")
    #         pcd_tmp = pcd_tmp.voxel_down_sample(voxel_size=voxel_size)
    #         new_xyz = np.asarray(pcd_tmp.points)
    #         new_rgb = np.asarray(pcd_tmp.colors)

    #     pcd = BasicPointCloud(
    #         points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
    #     )
    #     self.ply_input = pcd

    #     fused_point_cloud = torch.from_numpy(new_xyz).float().cuda()
    #     fused_color = RGB2SH(torch.from_numpy(new_rgb).float().cuda())
    #     features = (
    #         torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
    #         .float()
    #         .cuda()
    #     )
    #     features[:, :3, 0] = fused_color
    #     features[:, 3:, 1:] = 0.0

    #     dist2 = (
    #             torch.clamp_min(
    #                 distCUDA2(fused_point_cloud),
    #                 0.0000001,
    #             )
    #             * point_size
    #     )
    #     scales = torch.log(torch.sqrt(dist2))[..., None]
    #     if not self.isotropic:
    #         scales = scales.repeat(1, 3)

    #     rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    #     rots[:, 0] = 1
    #     opacities = inverse_sigmoid(
    #         0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
    #     )

    #     return fused_point_cloud, features, scales, rots, opacities

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def extend_from_pcd(
        self, fused_point_cloud, features, scales, rots, opacities, kf_id
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

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
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

        fused_point_cloud, features, scales, rots, opacities = (
            self.create_pcd_from_lidar_points(cam_info, xyz, rgb, init)
        )

        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, kf_id
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

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
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

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
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

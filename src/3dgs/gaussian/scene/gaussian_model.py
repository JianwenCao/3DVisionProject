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

import uuid
import json

import meshlib.mrmeshnumpy as mrnp
import meshlib.mrmeshpy as mr

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

        self.voxel_map = None
        self.init_map = False

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

    # Open3d based down_sample and normal estimation
    def create_pcd_from_lidar_points(self, cam_info, xyz, rgb, init=False):
        if init:
            voxel_size = self.config["Dataset"].get("voxel_size_init", 0.4)  # Root voxel size in meters
        else:
            voxel_size = self.config["Dataset"].get("voxel_size", 0.5)  # Root voxel size in meters

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        # Voxel downsampling
        pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)

        # Compute normals
        pcd_ds.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=60)
        )
        cam_pose = -(cam_info.R.T @ cam_info.T).cpu().numpy()
        pcd_ds.orient_normals_towards_camera_location(cam_pose)

        # Extract downsampled points, colors, and normals
        new_xyz = np.asarray(pcd_ds.points)
        new_rgb = np.asarray(pcd_ds.colors)
        new_normals = np.asarray(pcd_ds.normals)

        # Create BasicPointCloud
        pcd = BasicPointCloud(points=new_xyz, colors=new_rgb, normals=new_normals)
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(new_xyz).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(new_rgb).float().cuda())

        features = torch.zeros(
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),
            device="cuda", dtype=torch.float32
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        # Initialize scales using voxel size and thin s_delta
        M = new_xyz.shape[0]  # Number of points (downsampled)
        s_xy = torch.full((M,), voxel_size, device="cuda")
        s_delta = torch.full_like(s_xy, 0.01)  # 1 cm slice
        scales = torch.stack([s_xy, s_xy, s_delta], dim=-1)  # (M,3)
        scales = torch.log(scales)

        if self.config["Training"]["rotation_init"]:
            normals_cv = torch.from_numpy(new_normals).float().cuda()
            normals_gl = normals_cv * torch.tensor([1., -1., 1.],
                                                   device=normals_cv.device)

            # cos(θ) = Z · n = n_z
            dot = torch.clamp(normals_gl[:, 2], -1.0, 1.0)
            half_ang = 0.5 * torch.acos(dot)  # (N,)

            # Rotation axis = cross(Z, n) = (-n_y, n_x, 0)
            axis = torch.stack([
                -normals_gl[:, 1],
                normals_gl[:, 0],
                torch.zeros_like(dot)
            ], dim=1)
            axis = axis / (axis.norm(dim=1, keepdim=True) + 1e-8)

            # Local -> world quaternion [w, x, y, z]
            sin_h = torch.sin(half_ang).unsqueeze(1)
            cos_h = torch.cos(half_ang).unsqueeze(1)
            q_l2w = torch.cat([cos_h, axis * sin_h], dim=1)

            # Handle edge cases, parallel with optical axis
            eps = 1e-6
            mask_id = dot > (1 - eps)  # zero rotation
            mask_pi = dot < (-1 + eps)  # 180° rotation
            if mask_id.any():
                q_l2w[mask_id] = torch.tensor([1., 0., 0., 0.], device=q_l2w.device)
            if mask_pi.any():
                q_l2w[mask_pi] = torch.tensor([0., 1., 0., 0.], device=q_l2w.device)

            # World -> local and pack [w, x, y, z]
            rots = torch.cat([q_l2w[:, :1], -q_l2w[:, 1:]], dim=1)  # (N, 4)
            rots = rots.clone().detach().requires_grad_(True).cuda()
        else:
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1
            normals_gl = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")

        num_points = fused_point_cloud.shape[0]
        opacities = inverse_sigmoid(
            0.5 * torch.ones((num_points, 1), device="cuda", dtype=torch.float32)
        )

        return fused_point_cloud, features, scales, rots, opacities, normals_gl

    # MeshLib based down_sample and normal estimation
    # def create_pcd_from_lidar_points(self, cam_info, xyz, rgb, init=False):
    #     if init:
    #         voxel_size = self.config["Dataset"].get("voxel_size_init", 0.03)  # Root voxel size in meters
    #     else:
    #         voxel_size = self.config["Dataset"].get("voxel_size", 0.05)  # Root voxel size in meters
    #
    #     # Create MeshLib point cloud
    #     cloud = mrnp.pointCloudFromPoints(xyz)
    #     settings = mr.TriangulationHelpersSettings()
    #     settings.numNeis = 16  # TODO can tune mesh/search params
    #     allLocal = mr.buildUnitedLocalTriangulations(cloud, settings)
    #     cloud.normals = mr.makeUnorientedNormals(
    #         cloud,
    #         allLocal,
    #         orient=mr.OrientNormals.TowardOrigin)
    #
    #     new_xyz = xyz
    #     new_rgb = rgb
    #     new_normals = mrnp.toNumpyArray(cloud.normals)
    #
    #     # Voxel-based downsampling using MeshLib
    #     keys, inv = np.unique(
    #         np.floor(new_xyz / voxel_size).astype(int),
    #         axis=0, return_inverse=True)
    #     M = keys.shape[0]
    #     # Counts per voxel
    #     counts = np.bincount(inv, minlength=M).astype(np.float32)  # (M,)
    #
    #     # Sum up xyz, rgb, normals per voxel
    #     centroids = np.zeros((M, 3), dtype=np.float32)
    #     colors = np.zeros((M, 3), dtype=np.float32)
    #     norms = np.zeros((M, 3), dtype=np.float32)
    #     np.add.at(centroids, inv, new_xyz)
    #     np.add.at(colors, inv, new_rgb)
    #     np.add.at(norms, inv, new_normals)
    #
    #     # Normalize to get mean & unit normals
    #     centroids /= counts[:, None]
    #     colors /= counts[:, None]
    #     norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-8)
    #
    #     # Create BasicPointCloud
    #     pcd = BasicPointCloud(points=centroids, colors=colors, normals=norms)
    #     self.ply_input = pcd
    #
    #     fused_point_cloud = torch.from_numpy(centroids).float().cuda()
    #     fused_color = RGB2SH(torch.from_numpy(colors).float().cuda())
    #
    #     features = torch.zeros(
    #         (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),
    #         device="cuda", dtype=torch.float32
    #     )
    #     features[:, :3, 0] = fused_color
    #     features[:, 3:, 1:] = 0.0
    #
    #     # Initialize scales using voxel size and thin s_delta
    #     M = centroids.shape[0]  # Number of points (downsampled)
    #     s_xy = torch.full((M,), voxel_size, device="cuda")
    #     s_delta = torch.full_like(s_xy, 0.01)  # 1 cm slice
    #     scales = torch.stack([s_xy, s_xy, s_delta], dim=-1)  # (M,3)
    #     scales = torch.log(scales)
    #
    #     if self.config["Training"]["rotation_init"]:
    #         normals_cv = torch.from_numpy(norms).float().cuda()
    #         normals_gl = normals_cv * torch.tensor([1., -1., 1.],
    #                                                device=normals_cv.device)
    #
    #         # cos(θ) = Z · n = n_z
    #         dot = torch.clamp(normals_gl[:, 2], -1.0, 1.0)
    #         half_ang = 0.5 * torch.acos(dot)  # (N,)
    #
    #         # Rotation axis = cross(Z, n) = (-n_y, n_x, 0)
    #         axis = torch.stack([
    #             -normals_gl[:, 1],
    #             normals_gl[:, 0],
    #             torch.zeros_like(dot)
    #         ], dim=1)
    #         axis = axis / (axis.norm(dim=1, keepdim=True) + 1e-8)
    #
    #         # Local -> world quaternion [w, x, y, z]
    #         sin_h = torch.sin(half_ang).unsqueeze(1)
    #         cos_h = torch.cos(half_ang).unsqueeze(1)
    #         q_l2w = torch.cat([cos_h, axis * sin_h], dim=1)
    #
    #         # Handle edge cases, parallel with optical axis
    #         eps = 1e-6
    #         mask_id = dot > (1 - eps)  # zero rotation
    #         mask_pi = dot < (-1 + eps)  # 180° rotation
    #         if mask_id.any():
    #             q_l2w[mask_id] = torch.tensor([1., 0., 0., 0.], device=q_l2w.device)
    #         if mask_pi.any():
    #             q_l2w[mask_pi] = torch.tensor([0., 1., 0., 0.], device=q_l2w.device)
    #
    #         # World -> local and pack [w, x, y, z]
    #         rots = torch.cat([q_l2w[:, :1], -q_l2w[:, 1:]], dim=1)  # (N, 4)
    #         rots = rots.clone().detach().requires_grad_(True).cuda()
    #     else:
    #         rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    #         rots[:, 0] = 1
    #         normals_gl = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
    #
    #     num_points = fused_point_cloud.shape[0]
    #     opacities = inverse_sigmoid(
    #         0.5 * torch.ones((num_points, 1), device="cuda", dtype=torch.float32)
    #     )
    #
    #     return fused_point_cloud, features, scales, rots, opacities, normals_gl

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

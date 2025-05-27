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
import meshlib.mrmeshnumpy as mrnp
import meshlib.mrmeshpy as mr
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
from gaussian.utils.graphics_utils import BasicPointCloud
from gaussian.utils.sh_utils import RGB2SH
from scipy.spatial.transform import Rotation as R

from gaussian.scene.global_voxel_map import GlobalVoxelMap, GlobalVoxelSlot
import time

def print_gpu_mem(tag=""):
    torch.cuda.synchronize()
    alloc   = torch.cuda.memory_allocated()  / 1024**2   # MB
    reserv  = torch.cuda.memory_reserved()  / 1024**2
    peak    = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[{tag}] GPU   alloc {alloc:7.1f} MB | reserved {reserv:7.1f} MB | peak {peak:7.1f} MB")
    torch.cuda.reset_peak_memory_stats()

def print_gaussian_counts(gaussians, gvm, tag=""):
    print(f"[{tag}]   active_GPU = {gaussians._xyz.shape[0]:,}"
          f"   | active_keys = {len(gvm.active_keys):,}"
          f"   | total_voxels = {len(gvm.map):,}")

def assert_indices_consistent(gaussians, gvm):
    bad = [k for k in gvm.active_keys
           if gvm.map[k].gpu_idx < 0
           or gvm.map[k].gpu_idx >= gaussians._xyz.shape[0]]
    assert not bad, f"Found stale gpu_idx in {len(bad)} voxel-slots"

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

        self.normals = torch.empty(0, 3, device="cuda")

        self.global_voxel_map = GlobalVoxelMap(config)

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

    def create_pcd_from_lidar_points(self, cam_info, xyz, rgb, init=False):
        # Spatial hash root voxel size in meters
        if init:
            voxel_size = self.config["Mapping"].get("voxel_size_init", 0.4)  # Root voxel size in meters
        else:
            voxel_size = self.config["Mapping"].get("voxel_size", 0.5)  # Root voxel size in meters

        cloud = mrnp.pointCloudFromPoints(xyz)
        settings = mr.TriangulationHelpersSettings()
        settings.numNeis = 16 # TODO can tune mesh/search params
        allLocal = mr.buildUnitedLocalTriangulations(cloud, settings)
        cloud.normals = mr.makeUnorientedNormals(
            cloud, 
            allLocal,
            orient=mr.OrientNormals.TowardOrigin)

        new_xyz = xyz
        new_rgb = rgb
        new_normals = mrnp.toNumpyArray(cloud.normals)

        # Incoming point keys
        keys, inv = np.unique(
            np.floor(new_xyz / voxel_size).astype(int),
            axis=0, return_inverse=True)
        
        M = keys.shape[0]
        counts = np.bincount(inv, minlength=M).astype(np.float32) # (M,)

        # Sum up xyz, rgb, normals per voxel
        centroids = np.zeros((M,3), dtype=np.float32)
        colors = np.zeros((M,3), dtype=np.float32)
        norms = np.zeros((M,3), dtype=np.float32)
        np.add.at(centroids, inv, new_xyz)
        np.add.at(colors, inv, new_rgb)
        np.add.at(norms, inv, new_normals)

        # Normalize to get mean & unit normals
        centroids /= counts[:,None]
        colors /= counts[:,None]
        norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-8)
       
        pcd = BasicPointCloud(points=centroids, colors=colors, normals=norms)
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(centroids).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(colors).float().cuda())

        features = torch.zeros(
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),
            device="cuda", dtype=torch.float32
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        # Initialize scales using voxel size and thin s_delta
        s_xy = torch.full((M,), voxel_size,  device="cuda") # tangent extents
        s_delta = torch.full_like(s_xy, 0.01) # 1 cm slice
        scales = torch.stack([s_xy, s_xy, s_delta], dim=-1) # (M,3)
        scales = torch.log(scales)

        if self.config["Training"]["rotation_init"]:
            normals_cv = torch.from_numpy(norms).float().cuda()
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

        return fused_point_cloud, features, scales, rots, opacities, normals_gl, keys

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
        new_normals = normals.detach() # TODO check plain Tensor, no Autograd

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

    def move_gvm_to_gpu(self, kf_id):
        self._gpu_add_active_gaussians(self.global_voxel_map, list(self.global_voxel_map.map.keys()), kf_id)
        
    def extend_from_lidar_seq(
            self, cam_info, kf_id=-1, init=False, pcd=None
    ):
        # print("\n[DEBUG] New frame, called extend_from_lidar_seq")
        xyz = pcd[:, :3]
        rgb = pcd[:, 3:6] / 255.0
        gvm = self.global_voxel_map

        t0 = time.perf_counter()

        fused_point_cloud, features, scales, rots, opacities, normals, incoming_keys = (
            self.create_pcd_from_lidar_points(cam_info, xyz, rgb, init)
        )

        t1 = time.perf_counter()

        gvm.insert_gaussians(fused_point_cloud, features, scales, rots, opacities, normals, incoming_keys) # TODO Set gaussian params per point here?
        
        t2 = time.perf_counter()
        
        # Get keys to add/remove from GPU
        to_add, to_remove = gvm.cull_and_diff_active_voxels(cam_info, incoming_keys)
        print(f"current gaussians: {len(self._xyz)}, compared to last frame: + {len(to_add)} / - {len(to_remove)}")
        t3 = time.perf_counter()

        if to_remove:
            if init:
                raise ValueError("Should not be removing voxels on first frame")
            # print(f"[DEBUG] Removing {len(to_remove)} voxels from GPU")
            self._gpu_remove_inactive_gaussians(gvm, to_remove)

        t4 = time.perf_counter()

        if to_add:
            # print(f"[DEBUG] Adding {len(to_add)} voxels to GPU")
            self._gpu_add_active_gaussians(gvm, to_add, kf_id)

        t5 = time.perf_counter()

        # TODO remove later, assert gaussians live either in cuda 
        # tensor AND active_keys or in CPU map w/ gpu_idx=-1
        assert all(
                (slot.gpu_idx == -1) == (key not in gvm.active_keys)
                for key, slot in gvm.map.items()
            ), "Inconsistent gpu_idx in slots"
        # assert self._xyz.shape[0] == len(gvm.active_keys), \
        #     f"GPU size {self._xyz.shape[0]} != active_keys size {len(gvm.active_keys)}"
        
        t6 = time.perf_counter()

        print(f"Timing (ms): gen={(t1-t0)*1e3:.1f}, "
                f"insert={(t2-t1)*1e3:.1f}, "
                f"cull={(t3-t2)*1e3:.1f}, "
                f"gpu_remove={(t4-t3)*1e3:.1f}, "
                f"gpu_add={(t5-t4)*1e3:.1f}, "
                f"assert={(t6-t5)*1e3:.1f}")

    @torch.no_grad()
    def _gpu_add_active_gaussians(self, gvm, keys, kf_id):
        """
        Add voxel keys to GPU, append to optimization tensors, update active_keys.
        """
        slots = [gvm.map[k] for k in keys]
        if not slots:
            # print("No new voxels to activate on GPU")
            return
        
        def cat(field, requires_grad=True):
            arr = []
            for slot in slots:
                if len(arr) > 0 and len(slot.cpu_params[field]) != len(arr[-1]):
                    print(field, len(slot.cpu_params[field]), len(arr[-1]))
                    assert False
                arr.append(slot.cpu_params[field])
            arr = np.stack(arr, axis=0)  
            t = torch.from_numpy(arr).to("cuda").requires_grad_(requires_grad)
            return t
        
        new_xyz = nn.Parameter(cat("xyz"))
        new_f_dc = nn.Parameter(cat("f_dc").transpose(1, 2).contiguous()) # (N,1,3)
        new_f_rest = nn.Parameter(cat("f_rest").transpose(1, 2).contiguous()) # (N,SH-1,3)
        new_opacity = nn.Parameter(cat("opacity"))
        new_scaling = nn.Parameter(cat("scaling"))
        new_rot = nn.Parameter(cat("rotation"))
        new_normals = torch.stack([torch.as_tensor(s.cpu_params["normals"], 
                                                   dtype=torch.float32, device="cuda") 
                                                   for s in slots], dim=0)
        
        # Push onto optimizer
        self.densification_postfix(
            new_xyz,
            new_f_dc,
            new_f_rest,
            new_opacity,
            new_scaling,
            new_rot,
            new_normals,
            new_kf_ids=torch.full((len(slots),), kf_id, dtype=torch.int32),
            new_n_obs=torch.zeros((len(slots),), dtype=torch.int32),
        )

        # Set GPU indices in the voxel map slots
        start = self._xyz.shape[0] - len(slots)
        for i, s in enumerate(slots):
            # assert s.gpu_idx == -1, f"Slot {i} already has a gpu_idx"
            assert s.needs_init is False, f"Slot {i} needs init before pushing params to GPU"
            s.gpu_idx = start + i

        gvm.active_keys.update(keys)

    @torch.no_grad()
    def _gpu_remove_inactive_gaussians(self, gvm, keys_to_remove):
        """
        Copy optimized params from GPU back into CPU voxel, compact, update active_keys.
        """
        N = self._xyz.shape[0] # Total rows currently on GPU

        # Gather candidate drop indices, clamp to valid range
        raw_idx = [gvm.map[k].gpu_idx for k in keys_to_remove]
        drop_idx = [i for i in raw_idx if 0 <= i < N]
        # stale = set(raw_idx) - set(drop_idx)
        # if stale:
            # print(f"[DEBUG] Found {len(stale)} stale gpu_idx in {len(raw_idx)} slots")
        
        keep_mask = torch.ones(N, dtype=torch.bool, device=self._xyz.device)
        keep_mask[drop_idx] = False

        idx2key = {gvm.map[k].gpu_idx: k for k in keys_to_remove
                   if 0 <= gvm.map[k].gpu_idx < N}
        
        drop_idx_tensor = torch.tensor(drop_idx, dtype=torch.long, device=self._xyz.device)
        xyz = self._xyz[drop_idx_tensor].cpu().numpy() # (D,3)
        f_dc = self._features_dc[drop_idx_tensor, :1, :].transpose(1,2).contiguous().cpu().numpy() # (D,C)
        f_rest = self._features_rest[drop_idx_tensor].transpose(1,2).contiguous().cpu().numpy() # (D,…)
        opacity = self._opacity[drop_idx_tensor].cpu().numpy() # (D,1)
        scaling = self._scaling[drop_idx_tensor].cpu().numpy() # (D,3)
        rot = self._rotation[drop_idx_tensor].cpu().numpy() # (D,4)
        normals = self.normals[drop_idx_tensor].cpu().numpy() # (D,3)

        for i, idx in enumerate(drop_idx):
            key = idx2key[idx]
            # Write params to CPU and mark "not on GPU"
            gvm.cuda_params_to_voxel(key, {
                "xyz": xyz[i],
                "f_dc": f_dc[i],
                "f_rest": f_rest[i],
                "opacity": opacity[i],
                "scaling": scaling[i],
                "rotation": rot[i],
                "normals": normals[i],
            })
            gvm.map[key].gpu_idx = -1
        
        # Drop rows from GPU tensors, can maybe be optimized w/ swap+pop
        self.prune_points(~keep_mask)

        # Re‐assign gpu_idx for the survivors
        survivors = set(gvm.active_keys) - set(keys_to_remove)

        # Compute new row indices for all survivors
        keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1).cpu().numpy()
        remap = {old: new for new, old in enumerate(keep_idx)}

        stale_count = 0
        for key in survivors:
            slot = gvm.map[key]
            old = slot.gpu_idx
            new = remap.get(old, -1)
            slot.gpu_idx = new
            if new < 0:
                stale_count += 1 
        if stale_count > 0:
            raise ValueError(
                f"Found {stale_count} stale gpu_idx in {len(gvm.active_keys)} slots"
            )
            
        gvm.active_keys.difference_update(keys_to_remove)


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

    def handle_final_frame(self):
        return self._gpu_remove_inactive_gaussians(self.global_voxel_map, self.global_voxel_map.active_keys)

    def reset_parameters(self):
        """
        Empties all Gaussian parameters, optimizer state, and related attributes.
        Resets the model to an initial empty state while preserving configuration.
        """
        # Reset core parameter tensors to empty
        self._xyz = torch.empty(0, device="cuda")
        self._features_dc = torch.empty(0, device="cuda")
        self._features_rest = torch.empty(0, device="cuda")
        self._scaling = torch.empty(0, device="cuda")
        self._rotation = torch.empty(0, device="cuda")
        self._opacity = torch.empty(0, device="cuda")
        self.normals = torch.empty(0, 3, device="cuda")

        # Reset auxiliary tensors
        self.xyz_gradient_accum = torch.empty(0, device="cuda")
        self.denom = torch.empty(0, device="cuda")
        self.max_radii2D = torch.empty(0, device="cuda")
        self.unique_kfIDs = torch.empty(0, device="cuda").int()
        self.n_obs = torch.empty(0, device="cuda").int()

        # Clear optimizer state
        if self.optimizer is not None:
            self.optimizer.state.clear()
            self.optimizer.param_groups.clear()
            self.optimizer = None

        # Reset global voxel map (if desired)
        self.global_voxel_map = GlobalVoxelMap(self.config)

        # Reset active SH degree (optional, depending on your use case)
        self.active_sh_degree = 0
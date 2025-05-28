import numpy as np
from typing import List, Dict, Set, Tuple, Optional
import torch
from torch import nn

from gaussian.utils.camera_utils import Camera

GAUSS_FIELDS = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation", "normals")

class GlobalVoxelSlot:
    """
    Persistent storage for a single voxel's Gaussian parameters and occupancy.
    """
    __slots__ = ('cpu_params', 'cuda_tensors', 'count', 'needs_init', 'gpu_idx')

    def __init__(self):
        self.count = 0
        self.needs_init = True
        self.gpu_idx = -1

        self.cpu_params = {k: None for k in GAUSS_FIELDS}
        self.cuda_tensors = {k: torch.empty(0, device="cuda") for k in GAUSS_FIELDS}

    def to_cuda_tuple(self):
        """
        Convert CPU gaussian params to tensor tuple for GPU.
        """
        out = []
        for k in GAUSS_FIELDS:
            if self.cpu_params[k] is None:
                raise RuntimeError(f"GlobalVoxelSlot missing field {k} in cpu_params.")
            t = torch.as_tensor(self.cpu_params[k], dtype=torch.float32, device="cuda")
            t.requires_grad_(k != "normals") # normals not optimized, used for supervision
            self.cuda_tensors[k] = t
            out.append(t)
        return tuple(out)

class GlobalVoxelMap:
    """
    Spatial-hash based global voxel map with sliding-window active set.
    """

    def __init__(self, config: dict):
        self.voxel_size: float = config["Mapping"].get("voxel_size", 0.1)
        self.max_points_per_voxel: int = config["Mapping"].get("max_points_per_voxel", 1)
        self.max_active_gaussians: int = config["Mapping"].get("max_active_gaussians", 20000) # TODO implement

        self.map: Dict[Tuple[int, int, int], GlobalVoxelSlot] = {}
        self.active_keys: Set[Tuple[int, int, int]] = set()

    def cuda_params_to_voxel(self, key: Tuple[int,int,int], updated: dict) -> None:
        """
        Move optimized Gaussian parameters from GPU to map.
        """
        slot = self.map.get(key)
        if slot is None:
            raise ValueError("GPU trying to place optimized parameters in voxel slot that does not exist.")
        try:
            for k, v in updated.items():
                if k not in GAUSS_FIELDS:
                    raise ValueError(f"Invalid field {k} in updated Gaussian parameters.")
                if v is None:
                    raise ValueError(f"Field {k} in updated Gaussian parameters is None.")
                # TODO figure out why mix of tensor and ndarray are passed
                if torch.is_tensor(v):
                    slot.cpu_params[k] = v.detach().cpu().numpy()
                else:
                    slot.cpu_params[k] = np.asarray(v, dtype=np.float32, order="C")
        except (ValueError, AttributeError) as e:
            raise RuntimeError(f"Error updating voxel parameters: {e}")

    def insert_gaussians(self, fused_point_cloud, features, scales, rots, opacities, normals, keys):
        """
        For incoming point cloud and initialized parameters, insert into voxel
        map iff the voxel is empty or underfilled.

        Return: list of all incoming scan voxels for frustum culling, after underfilled voxels initialized.
        """
        # Find voxels that are not full (mask_np)
        default_slot = GlobalVoxelSlot()
        counts = np.fromiter(
            (self.map.get(tuple(k), default_slot).count
            for k in keys ),
            dtype=np.int32,
            count=keys.shape[0]
        )
        mask_np = counts < self.max_points_per_voxel

        xyz_cpu     = fused_point_cloud.detach().cpu().numpy()  # (M,3)
        feats_cpu   = features.detach().cpu().numpy()  # (M,3,C)
        scales_cpu  = scales.detach().cpu().numpy()  # (M,3)
        rots_cpu    = rots.detach().cpu().numpy()  # (M,4) or (M,3,3)
        opac_cpu    = opacities.detach().cpu().numpy()  # (M,1)
        norms_cpu   = normals.detach().cpu().numpy()  # (M,3)

        for i, keep in enumerate(mask_np):
            if not keep:
                continue
            key = tuple(keys[i])
            slot = self.map.get(key)
            if slot is None:
                slot = GlobalVoxelSlot()
                self.map[key] = slot
            
            if slot.count >= self.max_points_per_voxel:
                print(f"[WARN] insert_gaussians: voxel {key} already full (count={slot.count}). Redundant insert ignored.")
                continue
            
            slot.cpu_params["xyz"] = xyz_cpu[i]
            slot.cpu_params["f_dc"] = feats_cpu[i, :, 0:1]
            slot.cpu_params["f_rest"] = feats_cpu[i, :, 1:]
            slot.cpu_params["scaling"] = scales_cpu[i]
            slot.cpu_params["rotation"] = rots_cpu[i]
            slot.cpu_params["opacity"] = opac_cpu[i]
            slot.cpu_params["normals"] = norms_cpu[i]

            slot.count += 1
            slot.needs_init = False
        return

    def cull_and_diff_active_voxels(
        self,
        cam_info: Camera,
        new_keys: Optional[List[Tuple[int, int, int]]] = None
    ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
        """
        Update the active voxel set by culling in camera space.

        cam_info: current frame Camera obj
        new_keys: list of all voxel‐keys of incoming scan
        Returns (to_add, to_remove)
        """
        vs, he = self.voxel_size, self.voxel_size * 0.5
        old_active = set(self.active_keys)

        # helper: world‐space centers of a set of keys
        def compute_centers(keys):
            arr = np.array(list(keys), dtype=np.int32)
            return arr.astype(np.float32) * vs + he

        # extract camera parameters
        R = cam_info.R.cpu().numpy()            # (3×3) world→camera rotation
        T = cam_info.T.cpu().numpy()            # (3,)   world→camera translation
        fx = float(cam_info.fx);  fy = float(cam_info.fy)
        cx = float(cam_info.cx);  cy = float(cam_info.cy)
        W  = int(cam_info.image_width)
        H  = int(cam_info.image_height)

        # camera‐space cull: positive z, and 0 ≤ u < W, 0 ≤ v < H
        def camera_cull(centers_w):
            # 1) transform to camera coords
            #    p_cam = R @ p_world + T  → via (centers · R^T + T)
            p_cam = centers_w @ R.T + T[None, :]
            x, y, z = p_cam[:,0], p_cam[:,1], p_cam[:,2]

            # 2) project into pixel coords
            u = x * fx / z + cx
            v = y * fy / z + cy

            return (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        # 1) Cull existing active voxels
        if old_active:
            old_centers = compute_centers(old_active)
            mask_old = camera_cull(old_centers)
            visible_active = {k for k, v in zip(old_active, mask_old) if v}
        else:
            visible_active = set()

        # 2) Cull newly inserted voxels
        new_set = {tuple(v) for v in new_keys} if (len(new_keys)) else set()
        if new_set:
            new_centers = compute_centers(new_set)
            mask_new = camera_cull(new_centers)
            visible_new = {k for k, v in zip(new_set, mask_new) if v}
        else:
            visible_new = set()

        # 3) Fallback on very first frame: if nothing would be visible, just add all
        if not old_active and new_set and not visible_new:
            visible_new = new_set

        # 4) Figure out exactly which to add / remove
        to_remove = list(old_active - visible_active)
        to_add = list(visible_new - old_active)

        # print(f"[CAM-CULL] visible_old: {len(visible_active)}/{len(old_active)}, "
        #       f"visible_new: {len(visible_new)}/{len(new_keys or [])}")
        # print(f"[CAM-CULL] will add {len(to_add)}, remove {len(to_remove)}")

        return to_add, to_remove
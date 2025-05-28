import time
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
import torch

from gaussian.utils.camera_utils import Camera

class GaussianParam:
    def __init__(self, xyz=None, f_dc=None, f_rest=None, opacity=None, scaling=None, rotation=None, normals=None):
        self.xyz = xyz
        self.f_dc = f_dc
        self.f_rest = f_rest
        self.opacity = opacity
        self.scaling = scaling
        self.rotation = rotation
        self.normals = normals

        self.gpu_idx = -1

class GlobalVoxelMap:
    """
    Spatial-hash based global voxel map with sliding-window active set.
    """

    def __init__(self, config: dict):
        self.max_points_per_voxel = config.get("max_points_per_voxel", 1)
        self.map: Dict[Tuple[int, int, int], List[GaussianParam]] = {}
        self.active_keys: set = set()

    def update_map(self, xyz, features, scales, rots, opacities, normals, incoming_keys):
        xyz, features, scales, rots, opacities, normals, incoming_keys = xyz.cpu(), features.cpu(), scales.cpu(), rots.cpu(), opacities.cpu(), normals.cpu(), incoming_keys
        f_dcs, f_rests = features[:, :, :1], features[:, :, 1:]
        for i, key in enumerate(incoming_keys):
            key = tuple(key)
            if key in self.map.keys():
                if len(self.map[key]) < self.max_points_per_voxel:
                    self.map[key].extend(GaussianParam(xyz[i], f_dcs[i], f_rests[i], opacities[i], scales[i], rots[i], normals[i]))
                else:
                    pass
                    # print(f"[WARN] map full, rejecting voxel {key}")
            else:
                self.map[key] = [GaussianParam(xyz[i], f_dcs[i], f_rests[i], opacities[i], scales[i], rots[i], normals[i])]

    def update_active_keys(self, cam_info):
        global_keys = np.array(list(self.map.keys()), dtype=np.float32)

        R = cam_info.R.cpu().numpy()
        T = cam_info.T.cpu().numpy()
        fx = float(cam_info.fx);  fy = float(cam_info.fy)
        cx = float(cam_info.cx);  cy = float(cam_info.cy)
        W  = int(cam_info.image_width)
        H  = int(cam_info.image_height)
        p_cam = global_keys @ R.T + T[None, :]

        z = p_cam[:, 2]
        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        p_cam = p_cam @ intrinsics.T  # Apply intrinsic matrix to convert to pixel coordinates
        p_cam = p_cam[:, :2] / p_cam[:, 2:3]  # Normalize by depth to get pixel coordinates
        u, v = p_cam[:, 0], p_cam[:, 1]
        visiable_mask = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        
        current_active_keys = global_keys[visiable_mask]
        current_active_keys = set([tuple(key) for key in current_active_keys])
        global_keys = set([tuple(key) for key in global_keys])

        previous_active_keys = set([tuple(key) for key in self.active_keys])
        to_add = list(current_active_keys - previous_active_keys)
        to_remove = list(previous_active_keys - current_active_keys)
        self.active_keys = current_active_keys

        i = 0
        for key in self.active_keys:
            for param in self.map[key]:
                param.gpu_idx = i
                i += 1
        return to_add, to_remove

    def cuda_params_to_voxel(self, key: Tuple[int,int,int], updated: dict) -> None:
        xyz = updated["xyz"]
        f_dcs = updated["f_dc"]
        f_rests = updated["f_rest"]
        opacities = updated["opacity"]
        scales = updated["scaling"]
        rots = updated["rotation"]
        normals = updated["normals"]
        self.map[key][0] = GaussianParam(xyz, f_dcs, f_rests, opacities, scales, rots, normals)
    
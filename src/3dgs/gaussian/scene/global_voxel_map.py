# voxel_map.py
# ------------------------------------------
# Module for maintaining a global spatial-hash voxel map
# ------------------------------------------

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
import torch

def voxel_key_from_xyz(xyz: np.ndarray, voxel_size: float) -> tuple:
    return tuple((xyz // voxel_size).astype(int))

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

        # self.gaussians_per_voxel = None # TODO later, dynamically allocate this based on texture
        self.cpu_params = {k: None for k in GAUSS_FIELDS}
        self.cuda_tensors = {k: torch.empty(0, device="cuda") for k in GAUSS_FIELDS}

    def to_cuda_tuple(self):
        """
        Convert the stored voxel parameters to a tuple of CUDA tensors.
        For GPU processing.
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
    A spatial-hash based global voxel map with sliding-window active set.

    Public API:
      insert_points(points: np.ndarray)
      update_active(frustum_planes: np.ndarray)
    """

    def __init__(self, config: dict):
        self.voxel_size: float = config["Mapping"].get("voxel_size", 0.1)
        self.max_points_per_voxel: int = config["Mapping"].get("max_points_per_voxel", 1)
        self.max_active_gaussians: int = config["Mapping"].get("max_active_gaussians", 10000)

        # hash map: voxel_key -> GlobalVoxelSlot
        self.map: Dict[Tuple[int, int, int], GlobalVoxelSlot] = {}
        # currently visible (active) voxel keys, THESE SHOULD GO TO GPU
        self.active_keys: Set[Tuple[int, int, int]] = set()

    def cuda_params_to_voxel(self, key: Tuple[int,int,int], updated: dict) -> None:
        """
        Incoming, optimized Gaussian parameters from GPU to map.
        """
        slot = self.map.get(key)
        if slot is None:
            raise ValueError("GPU trying to place optimized parameters in voxel slot that does not exist.")
        # detach, move to CPU, convert to numpy
        try:
            for k, v in updated.items():
                if k not in GAUSS_FIELDS:
                    raise ValueError(f"Invalid field {k} in updated Gaussian parameters.")
                if v is None:
                    raise ValueError(f"Field {k} in updated Gaussian parameters is None.")
                # Accept torch.Tensor *or* numpy.ndarray
                # TODO figure out why mix of both are passed
                if torch.is_tensor(v):
                    slot.cpu_params[k] = v.detach().cpu().numpy()
                else:
                    slot.cpu_params[k] = np.asarray(v, dtype=np.float32, order="C")
        except (ValueError, AttributeError) as e:
            raise RuntimeError(f"Error updating voxel parameters: {e}")


    def insert_gaussians(self, fused_point_cloud, features, scales, rots, opacities, normals):
        """
        For each point, compute the voxel key and check if the voxel exists / is full.
        Insert raw gaussian param data into voxel slots iff the voxel is empty/underfilled
        Return list of voxel-keys that were flagged for init this frame.
        """
        # Find voxels that are not full (mask_np)
        pts_np = fused_point_cloud.cpu().numpy() # (M,3)

        # Compute incoming voxel spatial keys
        keys = np.floor(pts_np / self.voxel_size).astype(np.int32) # (M,3)
        default_slot = GlobalVoxelSlot()
        counts = np.fromiter(
            (self.map.get(tuple(k), default_slot).count
            for k in keys ),
            dtype=np.int32,
            count=keys.shape[0]
        )
        mask_np = counts < self.max_points_per_voxel
        mask = torch.from_numpy(mask_np).to(device=fused_point_cloud.device)
        
        keys_to_init: List[Tuple[int,int,int]] = []
        # TODO vectorize this or do in batch
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
            
            # TODO Maybe don't detach if initializing first frames
            slot.cpu_params["xyz"] = fused_point_cloud[i].detach().cpu().numpy()
            slot.cpu_params["f_dc"] = features[i, :, 0].detach().cpu().numpy()
            slot.cpu_params["f_rest"] = features[i, :, 1:].detach().cpu().numpy()
            slot.cpu_params["scaling"] = scales[i].detach().cpu().numpy()
            slot.cpu_params["rotation"] = rots[i].detach().cpu().numpy()
            slot.cpu_params["opacity"] = opacities[i].detach().cpu().numpy()
            slot.cpu_params["normals"] = normals[i].detach().cpu().numpy()

            slot.count += 1
            slot.needs_init = False
            keys_to_init.append(key)
        return mask, keys_to_init

    def update_active_gaussians(
        self,
        frustum_planes: np.ndarray,
        new_keys: Optional[List[Tuple[int, int, int]]] = None
    ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
        """
        Update the active voxel set based on frustum visibility.
        Retain visible active voxels, add new visible voxels, and remove non-visible ones.

        frustum_planes: (6,4) array of [a,b,c,d]
        new_keys: list of voxel-keys inserted this frame
        Returns: (to_add, to_remove) - lists of voxel keys to add/remove from GPU
        """
        vs = self.voxel_size
        he = vs * 0.5
        to_add = []
        to_remove = []

        # Step 1: Cull existing active voxels
        if self.active_keys:
            active_list = list(self.active_keys)
            arr = np.array(active_list, dtype=np.int32)
            centers = arr.astype(np.float32) * vs + he
            mask = self._cull_centers(centers, frustum_planes, he)
            visible_active = {active_list[i] for i, v in enumerate(mask) if v}
            print(f"[DEBUG] {len(visible_active)}/{len(active_list)} active voxels visible")
            to_remove = list(self.active_keys - visible_active)
        else:
            visible_active = set()
            to_remove = []

        # Step 2: Cull new keys and add visible ones
        if new_keys:
            nk_arr = np.array(new_keys, dtype=np.int32)
            nk_centers = nk_arr.astype(np.float32) * vs + he
            print(f"[DEBUG] Voxel centers range: min={nk_centers.min(axis=0)}, max={nk_centers.max(axis=0)}")
            nk_mask = self._cull_centers(nk_centers, frustum_planes, he)
            visible_new = [new_keys[i] for i, v in enumerate(nk_mask) if v]
            
            # First‐frame fallback: if we haven’t yet populated the GPU at all,
            # send *all* new_keys once so the renderer gets a non‐empty tensor.
            if not self.active_keys and new_keys and not visible_new:
                visible_new = new_keys

            print(f"[DEBUG] {len(visible_new)}/{len(new_keys)} new voxels visible")
            
            # Add only new keys that are visible and not already active
            for key in visible_new:
                if len(self.active_keys) >= self.max_active_gaussians:
                    print(f"[DEBUG] Max active Gaussians ({self.max_active_gaussians}) reached")
                    break
                if key not in self.active_keys:
                    slot = self.map.get(key)
                    if slot.needs_init and slot.cpu_params["xyz"] is None:
                        slot.cpu_params = self._initialize_gaussian_for_voxel(key)
                        slot.needs_init = False
                    to_add.append(key)
                    self.active_keys.add(key)
        else:
            visible_new = []

        print(f"[DEBUG] Added {len(to_add)} keys, removed {len(to_remove)} keys, total active_keys: {len(self.active_keys)}")
        return to_add, to_remove

    def _cull_centers(
        self,
        centers: np.ndarray,
        planes: np.ndarray,
        half_extent: float
    ) -> np.ndarray:
        """
        Vectorized AABB-frustum test for an array of centers.
        centers: (N,3), planes: (6,4)
        Returns boolean mask of length N.
        """
        n   = planes[:, :3]                  # (6,3)
        d   = planes[:, 3]                   # (6,)
        dist = centers @ n.T + d             # (N,6)
        r    = half_extent * np.abs(n).sum(1)         # (6,)
        outside = dist > r                   # (N,6)
        return ~outside.any(1)               # (N,)

    def _initialize_gaussian_for_voxel(self, key: Tuple[int, int, int]) -> dict:
        """
        Default Gaussian init: center at voxel centroid, scale = log(voxel_size).
        Returns a dict of init parameters for integration.
        """
        center = np.array(key) * self.voxel_size + self.voxel_size * 0.5
        init_scale = np.log(self.voxel_size)
        return {'xyz': center, 'scale': init_scale}
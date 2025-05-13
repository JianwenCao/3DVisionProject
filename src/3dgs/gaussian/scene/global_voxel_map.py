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
    Attributes:
      params: Optional[dict]     # cached init params or updated stats
      count: int                 # number of points accumulated
      needs_init: bool           # whether to init Gaussian for this voxel
      cgb_idx: int               # index in CPU Gaussian Buffer, -1 if inactive
    """
    __slots__ = ('params', 'count', 'needs_init', 'cgb_idx')

    def __init__(self, init_params: Optional[dict] = None):
        self.params = init_params
        self.count = 0
        self.needs_init = True
        self.cgb_idx = -1

        self._xyz = torch.empty(0, device="cuda")
        self._features_dc = torch.empty(0, device="cuda")
        self._features_rest = torch.empty(0, device="cuda")
        self._scaling = torch.empty(0, device="cuda")
        self._rotation = torch.empty(0, device="cuda")
        self._opacity = torch.empty(0, device="cuda")
        self._normals = torch.empty(0, 3, device="cuda")
        # self.gaussians_per_voxel = None # TODO later, dynamically allocate this based on texture

    def to_cuda_tuple(self):
        """
        Convert the stored voxel parameters to a tuple of CUDA tensors.
        For GPU processing.
        """
        # TODO: Implement the conversion logic
        pass

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

    def cuda_params_to_voxel(self, key: Tuple[int,int,int]) -> dict:
        """
        Incoming, optimized Gaussian parameters from GPU to map.
        """
        slot = self.map.get(key)
        if slot is None:
            raise ValueError("GPU trying to place optimized parameters in voxel slot that does not exist.")
        # TODO finish this logic, somehow get the GPU params and store in the slot

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
        # TODO replace this with something that fills the voxel slots with gaussian data, marks needs_init=False
        # for p in points:
        #     key = voxel_key_from_xyz(p, self.voxel_size)
        #     slot = self.map.get(key)
        #     if slot is None:
        #         slot = GlobalVoxelSlot()
        #         self.map[key] = slot

        #     if slot.count < self.max_points_per_voxel:
        #         slot.count += 1
        #         slot.needs_init = True
        #         keys_to_init.append(key)
        #     else:
        #         print(f"[WARN] insert_points: voxel {key} already full (count={slot.count}). Redundant insert ignored.")
        return mask, keys_to_init

    def update_active_gaussians(
        self,
        frustum_planes: np.ndarray,
        new_keys: Optional[List[Tuple[int,int,int]]] = None
    ) -> None:
        """
        Candidate-only culling: first prune existing active_keys,
        then cull only the new_keys for addition.

        frustum_planes: (6,4) array of [a,b,c,d]
        new_keys: list of voxel-keys inserted this frame
        """
        vs = self.voxel_size
        he = vs * 0.5
        print(f"[DEBUG] update_active_gaussians got frustum planes: {frustum_planes}")
        if self.active_keys:
            active_list = list(self.active_keys)
            arr = np.array(active_list, dtype=np.int32)
            centers = arr.astype(np.float32) * vs + he
            mask = self._cull_centers(centers, frustum_planes, he)
            visible_active = {active_list[i] for i, v in enumerate(mask) if v}
            print(f"[DEBUG] {len(visible_active)}/{len(active_list)} active voxels visible")
        else:
            visible_active = set()
        to_remove = self.active_keys - visible_active
        for key in to_remove:
            slot = self.map[key]
            if slot.cgb_idx >= 0:
                slot.cgb_idx = -1
            self.active_keys.remove(key)
        if new_keys:
            nk_arr = np.array(new_keys, dtype=np.int32)
            nk_centers = nk_arr.astype(np.float32) * vs + he
            print(f"[DEBUG] Voxel centers range: min={nk_centers.min(axis=0)}, max={nk_centers.max(axis=0)}")
            nk_mask = self._cull_centers(nk_centers, frustum_planes, he)
            visible_new = [new_keys[i] for i, v in enumerate(nk_mask) if v]
            print(f"[DEBUG] {len(visible_new)}/{len(new_keys)} new voxels visible")
        else:
            visible_new = []
        to_add = []
        # Add all new_keys during initialization, or visible_new otherwise
        keys_to_process = new_keys if new_keys and (len(visible_new) == 0 or len(self.active_keys) == 0) else visible_new
        for key in keys_to_process:
            if len(self.active_keys) >= self.max_active_gaussians:
                break
            slot = self.map[key]
            if slot.needs_init and slot.params is None:
                slot.params = self._initialize_gaussian_for_voxel(key)
                slot.needs_init = False
            if key not in self.active_keys:  # Avoid duplicates
                self.active_keys.add(key)
                to_add.append(key)
        print(f"[DEBUG] Added {len(to_add)} keys to active_keys, total active_keys: {len(self.active_keys)}")
        return to_add, list(to_remove)

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
        signs = np.sign(planes[:, :3]) # (6,3)
        corners = centers[None, :, :] + half_extent * signs[:, None, :] # (6,N,3)
        # dot with normals + d-term
        normals = planes[:, :3][:, None, :] # (6,1,3)
        dists = np.sum(corners * normals, axis=2) # (6,N)
        dists += planes[:, 3:4] # (6,N)
        return np.all(dists >= 0, axis=0) # (N,)

    def _initialize_gaussian_for_voxel(self, key: Tuple[int, int, int]) -> dict:
        """
        Default Gaussian init: center at voxel centroid, scale = log(voxel_size).
        Returns a dict of init parameters for integration.
        """
        center = np.array(key) * self.voxel_size + self.voxel_size * 0.5
        init_scale = np.log(self.voxel_size)
        return {'xyz': center, 'scale': init_scale}
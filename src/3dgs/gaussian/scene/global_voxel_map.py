# voxel_map.py
# ------------------------------------------
# Module for maintaining a global spatial-hash voxel map
# ------------------------------------------

import numpy as np
from typing import List, Dict, Set, Tuple, Optional

def voxel_key_from_xyz(xyz: np.ndarray, voxel_size: float) -> tuple:
    return tuple((xyz // voxel_size).astype(int))


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
        # currently visible (active) voxel keys
        self.active_keys: Set[Tuple[int, int, int]] = set()

    def insert_points(self, points: np.ndarray) -> List[Tuple[int,int,int]]:
        """
        Insert raw 3D points, increment per-voxel counts.
        Return list of voxel-keys that were flagged for init this frame.
        """
        new_keys: List[Tuple[int,int,int]] = []
        for p in points:
            key = voxel_key_from_xyz(p, self.voxel_size)
            slot = self.map.get(key)
            if slot is None:
                slot = GlobalVoxelSlot()
                self.map[key] = slot

            if slot.count < self.max_points_per_voxel:
                slot.count += 1
                slot.needs_init = True
                new_keys.append(key)
            else:
                print(f"[WARN] insert_points: voxel {key} already full (count={slot.count}). Redundant insert ignored.")
        return new_keys

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

        # Cull current active_keys
        if self.active_keys:
            active_list = list(self.active_keys)
            arr = np.array(active_list, dtype=np.int32)              # (A,3)
            centers = arr.astype(np.float32) * vs + he               # (A,3)
            mask = self._cull_centers(centers, frustum_planes, he)
            visible_active = { active_list[i] for i, v in enumerate(mask) if v }
        else:
            visible_active = set()

        # Remove invisible active keys
        to_remove = self.active_keys - visible_active
        for key in to_remove:
            slot = self.map[key]
            if slot.cgb_idx >= 0:
                slot.cgb_idx = -1
            self.active_keys.remove(key)

        # Cull and add new_keys
        if new_keys:
            nk_arr = np.array(new_keys, dtype=np.int32)             # (N,3)
            nk_centers = nk_arr.astype(np.float32) * vs + he         # (N,3)
            nk_mask = self._cull_centers(nk_centers, frustum_planes, he)
            visible_new = [ new_keys[i] for i, v in enumerate(nk_mask) if v ]
        else:
            visible_new = []

        # Add in visible order up to capacity
        for key in visible_new:
            if len(self.active_keys) >= self.max_active_gaussians:
                break
            slot = self.map[key]
            if slot.needs_init:
                slot.params = self._initialize_gaussian_for_voxel(key)
                slot.needs_init = False
            self.active_keys.add(key)

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
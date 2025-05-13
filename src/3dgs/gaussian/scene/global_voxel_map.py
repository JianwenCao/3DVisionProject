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
        vs, he = self.voxel_size, self.voxel_size * 0.5

        old_active = set(self.active_keys)

        # Compute world‐space centers
        def centers_of(keys):
            arr = np.array(list(keys), dtype=np.int32)
            return arr.astype(np.float32) * vs + he

        # 1) Cull existing actives, if any
        if old_active:
            old_centers    = centers_of(old_active)
            mask_old       = self._cull_centers(old_centers, frustum_planes, he)
            visible_active = {
                k for k, v in zip(old_active, mask_old) if v
            }
        else:
            visible_active = set()

        # 2) Which of the new insertions actually fall in view?
        if new_keys:
            new_set = set(new_keys)
            new_centers = centers_of(new_set)
            mask_new    = self._cull_centers(new_centers, frustum_planes, he)
            visible_new = {
                k for k, v in zip(new_set, mask_new) if v
            }
        else:
            visible_new = set()

        # 3) First‐frame fallback: if we're totally empty, just seed with all new
        if not old_active and new_keys and not visible_new:
            visible_new = set(new_keys)

        # 4) Compute diffs
        to_remove = list(old_active - visible_active)
        to_add    = list(visible_new - old_active)

        print(f"[DEBUG] Visible old: {len(visible_active)}/{len(old_active)}, "
            f"visible new: {len(visible_new)}/{len(new_keys or [])}")
        print(f"[DEBUG] Will add {len(to_add)}, remove {len(to_remove)}")

        return to_add, to_remove

    def _cull_centers(
        self,
        centers: np.ndarray,   # (N,3)
        planes: np.ndarray,    # (6,4) – [nx,ny,nz,d], plane eq: n⋅p + d ≤ 0 → inside
        half_extent: float
    ) -> np.ndarray:
        """
        Vectorized AABB↔frustum test: keep a voxel iff *all* of its 8 corners
        lie on the *inside* side of *every* plane.
        We do this by selecting, for each plane, the AABB corner that is
        farthest along that plane’s normal; if *that* corner is still inside
        (n·corner + d ≤ 0), the entire box intersects the frustum.
        """
        # 1) For each plane, pick the “farthest” corner
        signs   = np.sign(planes[:, :3])                   # (6,3)
        corners = centers[None,:,:] + half_extent*signs[:,None,:]  # (6,N,3)

        # 2) Compute signed distance of those corners to each plane
        normals = planes[:,:3][:,None,:]                   # (6,1,3)
        dists   = np.sum(corners * normals, axis=2)        # (6,N)
        dists  += planes[:,3:4]                            # (6,N)

        # 3) A box is “inside” iff *all* its farthest‐corners satisfy n·p + d ≤ 0
        return np.all(dists <= 0, axis=0)                  # (N,)  True = keep

    def _initialize_gaussian_for_voxel(self, key: Tuple[int, int, int]) -> dict:
        """
        Default Gaussian init: center at voxel centroid, scale = log(voxel_size).
        Returns a dict of init parameters for integration.
        """
        center = np.array(key) * self.voxel_size + self.voxel_size * 0.5
        init_scale = np.log(self.voxel_size)
        return {'xyz': center, 'scale': init_scale}
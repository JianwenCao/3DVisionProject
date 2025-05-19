import numpy as np
from typing import List, Dict, Set, Tuple, Optional

def voxel_key_from_xyz_array(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """Return integer (N,3) voxel indices for point array `xyz`."""
    return np.floor(xyz / voxel_size).astype(np.int32)

class GlobalVoxelSlot:
    __slots__ = ("count", "needs_init")

    def __init__(self):
        self.count: int = 0
        self.needs_init: bool = True


class GlobalVoxelMap:
    """Spatial‑hash voxel map storing Gaussian occupancy counts."""

    def __init__(self, config: dict):
        self.voxel_size: float = config["Mapping"].get("voxel_size", 0.1)
        self.max_points_per_voxel: int = config["Mapping"].get("max_points_per_voxel", 1)
        self.map: Dict[Tuple[int, int, int], GlobalVoxelSlot] = {}

    def get_scan_points_to_init(self, incoming_xyz: np.ndarray) -> np.ndarray:
        """
        Compute a boolean mask of length *N* that marks **only** the first
        `remaining_capacity` points per voxel (in original scan order),
        where `remaining_capacity = max_points_per_voxel − existing_count`.

        The implementation is completely NumPy‑vectorised except for the
        unavoidable O(U) dictionary look‑ups to fetch *existing_count* for
        each of the *U* unique voxels touched by this scan.
        """
        # 1. Voxelise all points -------------------------------------------------
        keys = voxel_key_from_xyz_array(incoming_xyz, self.voxel_size)  # (N,3)

        # 2. Group identical voxels in C using np.unique ------------------------
        #    uniq : (U,3) unique voxel indices (ordered by first appearance)
        #    inv  : (N,)  maps each point -> its voxel id  (0 … U‑1)
        #    scan_counts: (U,) #points in *this scan* for each voxel
        uniq, inv, scan_counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True
        )
        U = len(uniq)

        # 3. Fetch *existing* occupancy counts for each unique voxel -----------
        existing = np.empty(U, dtype=np.int32)
        for i, uk in enumerate(uniq):
            existing[i] = self.map.get(tuple(uk), GlobalVoxelSlot()).count

        # 4. Remaining capacity per voxel for this scan ------------------------
        capacity = self.max_points_per_voxel - existing  # (U,)
        np.clip(capacity, 0, None, out=capacity)         # negative → 0

        # 5. Build mask: accept first `capacity[v]` pts of each voxel ----------
        #    (a) stable‑sort point indices by voxel id so that points of the
        #        same voxel are contiguous *in original scan order*.
        order = np.argsort(inv, kind="stable")           # (N,)
        inv_sorted = inv[order]                          # (N,)

        #    (b) For each voxel v, build its starting index in the sorted list.
        #        Because inv_sorted is grouped, we can create a prefix‑sum of
        #        scan_counts: offset[v] = first index of voxel v in inv_sorted.
        offsets = np.concatenate(([0], np.cumsum(scan_counts[:-1])))  # (U,)

        #    (c) Point rank within its voxel group = position − offsets[voxel].
        ranks_sorted = np.arange(len(inv_sorted), dtype=np.int32) - offsets[inv_sorted]

        #    (d) Accept a point iff rank < capacity[voxel].
        accepted_sorted = ranks_sorted < capacity[inv_sorted]

        #    (e) Scatter back to original ordering.
        mask = np.zeros(len(incoming_xyz), dtype=bool)
        mask[order] = accepted_sorted
        return mask

    def update_voxel_occupancy(self, accepted_xyz: np.ndarray):
        """Increment voxel counters for the *already‑accepted* points."""
        if accepted_xyz.size == 0:
            return  # nothing to do

        keys = voxel_key_from_xyz_array(accepted_xyz, self.voxel_size)

        uniq, counts = np.unique(keys, axis=0, return_counts=True)
        for uk, cnt in zip(uniq, counts):
            key = tuple(uk)
            slot = self.map.get(key)
            if slot is None:
                slot = GlobalVoxelSlot()
                self.map[key] = slot
            allowable = self.max_points_per_voxel - slot.count
            if allowable > 0:
                slot.count += int(min(allowable, cnt))
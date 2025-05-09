import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import time

HASH_P = 116101
MAX_N = 10000000000
DEG2RAD = np.pi / 180.0

RANGE_INC = 0.02  # dept_err
DEGREE_INC = 0.05  # beam_err
PLANER_THRESHOLD = 0.0025  # min_eigen_value
VOXEL_SIZE = 0.03
MAX_LAYER = 2
MAX_POINTS_SIZE = 50
MAX_COV_POINTS_SIZE = 50
LAYER_POINT_SIZE = [5, 5, 5, 5, 5][:MAX_LAYER]

@dataclass
class VOXEL_LOC:
    x: int
    y: int
    z: int

    def __hash__(self):
        return ((((self.z * HASH_P) % MAX_N + self.y) * HASH_P) % MAX_N + self.x)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

@dataclass
class pointWithCov:
    point: np.ndarray  # (3,)
    cov: np.ndarray    # (3, 3)
    rgb: Optional[np.ndarray] = None  # (3,),

@dataclass
class Plane:
    center: np.ndarray
    normal: np.ndarray
    y_normal: np.ndarray
    x_normal: np.ndarray
    covariance: np.ndarray
    plane_cov: np.ndarray
    radius: float = 0.0
    min_eigen_value: float = 1.0
    mid_eigen_value: float = 1.0
    max_eigen_value: float = 1.0
    d: float = 0.0
    points_size: int = 0
    is_plane: bool = False
    is_init: bool = False
    id: int = 0
    is_update: bool = False
    last_update_points_size: int = 0
    update_enable: bool = True

def calcBodyCov(pb: np.ndarray, range_inc: float = RANGE_INC, degree_inc: float = DEGREE_INC) -> np.ndarray:
    range_ = np.sqrt(np.sum(pb ** 2))
    range_var = range_inc ** 2
    direction_var = np.array([
        [np.sin(DEG2RAD * degree_inc) ** 2, 0],
        [0, np.sin(DEG2RAD * degree_inc) ** 2]
    ])
    direction = pb / (range_ + 1e-6)
    direction_hat = np.array([
        [0, -direction[2], direction[1]],
        [direction[2], 0, -direction[0]],
        [-direction[1], direction[0], 0]
    ])
    base_vector1 = np.array([1, 1, -(direction[0] + direction[1]) / (direction[2] + 1e-6)])
    base_vector1 /= np.linalg.norm(base_vector1) + 1e-6
    base_vector2 = np.cross(base_vector1, direction)
    base_vector2 /= np.linalg.norm(base_vector2) + 1e-6
    N = np.vstack([base_vector1, base_vector2]).T
    A = range_ * direction_hat @ N
    cov = (direction[:, None] @ direction[None, :] * range_var +
           A @ direction_var @ A.T)
    return cov

def calcBodyCov_batch(
    pb: np.ndarray,
    range_inc: float  = RANGE_INC,
    degree_inc: float = DEGREE_INC,
) -> np.ndarray:
    """
    Vectorised version of `calcBodyCov`.
    Parameters
    ----------
    pb : (N, 3) array
        3‑D points expressed in the body frame.
    Returns
    -------
    cov : (N, 3, 3) array
        One 3 × 3 covariance matrix per input point.
    """
    pb         = np.asarray(pb)
    N          = pb.shape[0]

    # --- common scalars/arrays ------------------------------------------------
    rng        = np.linalg.norm(pb, axis=1)                        # (N,)
    rng_var    = range_inc ** 2
    sin_sq     = np.sin(DEG2RAD * degree_inc) ** 2                 # scalar

    # --- normalised directions ----------------------------------------------
    d          = pb / (rng[:, None] + 1e-6)                        # (N,3)

    # --- skew‑symmetric matrices  (N,3,3) ------------------------------------
    d_hat      = np.zeros((N, 3, 3))
    d_hat[:, 0, 1] = -d[:, 2]
    d_hat[:, 0, 2] =  d[:, 1]
    d_hat[:, 1, 0] =  d[:, 2]
    d_hat[:, 1, 2] = -d[:, 0]
    d_hat[:, 2, 0] = -d[:, 1]
    d_hat[:, 2, 1] =  d[:, 0]

    # --- two orthogonal basis vectors for the tangent plane ------------------
    b1         = np.stack([ np.ones(N),
                            np.ones(N),
                           -(d[:, 0] + d[:, 1]) / (d[:, 2] + 1e-6)], axis=1)
    b1        /= np.linalg.norm(b1, axis=1, keepdims=True) + 1e-6

    b2         = np.cross(b1, d)
    b2        /= np.linalg.norm(b2, axis=1, keepdims=True) + 1e-6

    N_mat      = np.stack([b1, b2], axis=2)                        # (N,3,2)

    # --- A = range * (d̂ · N) -----------------------------------------------
    A          = rng[:, None, None] * (d_hat @ N_mat)              # (N,3,2)

    # --- final covariance ----------------------------------------------------
    dir_outer  = rng_var * d[:, :, None] * d[:, None, :]           # (N,3,3)
    B          = A * sin_sq                                        # (N,3,2)
    cov2       = B @ A.transpose(0, 2, 1)                          # (N,3,3)

    return dir_outer + cov2

# def generate_pv_list(points: np.ndarray, rgb: Optional[np.ndarray] = None) -> List[pointWithCov]:
#     pv_list = []
#     for i, point in enumerate(points):
#         cov = calcBodyCov(point)
#         rgb_i = rgb[i] if rgb is not None and i < len(rgb) else None
#         pv = pointWithCov(point=point, cov=cov, rgb=rgb_i)
#         pv_list.append(pv)
#     return pv_list

def generate_pv_list(
    points: np.ndarray,
    rgb:    Optional[np.ndarray] = None
) -> List[pointWithCov]:
    """
    Vectorised replacement of the original `generate_pv_list`.
    Works with or without an RGB array of shape (N, 3).
    """
    points = np.asarray(points)
    covs   = calcBodyCov_batch(points)

    if rgb is not None:
        rgb = np.asarray(rgb)

    return [
        pointWithCov(
            point=pt,
            cov=cov,
            rgb=(rgb[i] if rgb is not None and i < len(rgb) else None)
        )
        for i, (pt, cov) in enumerate(zip(points, covs))
    ]

class OctoTree:
    def __init__(self, layer: int):
        self.temp_points: List[pointWithCov] = []
        self.new_points: List[pointWithCov] = []
        self.plane_ptr = Plane(
            center=np.zeros(3),
            normal=np.zeros(3),
            y_normal=np.zeros(3),
            x_normal=np.zeros(3),
            covariance=np.zeros((3, 3)),
            plane_cov=np.zeros((6, 6))
        )
        self.max_layer = MAX_LAYER
        self.layer = layer
        self.octo_state = 0
        self.leaves = [None] * 8
        self.voxel_center = np.zeros(3)
        self.quater_length = 0.0
        self.planer_threshold = PLANER_THRESHOLD
        self.max_plane_update_threshold = LAYER_POINT_SIZE[layer]
        self.update_size_threshold = 5
        self.all_points_num = 0
        self.new_points_num = 0
        self.max_points_size = MAX_POINTS_SIZE
        self.max_cov_points_size = MAX_COV_POINTS_SIZE
        self.init_octo = False
        self.update_cov_enable = True
        self.update_enable = True
        self.plane_id_counter = 0

    def init_plane(self, points: List[pointWithCov], plane: Plane):
        plane.points_size = len(points)
        plane.center = np.zeros(3)
        plane.covariance = np.zeros((3, 3))
        for pv in points:
            plane.center += pv.point
            plane.covariance += pv.point[:, None] @ pv.point[None, :]
        plane.center /= plane.points_size
        plane.covariance = plane.covariance / plane.points_size - plane.center[:, None] @ plane.center[None, :]
        eigenvalues, eigenvectors = np.linalg.eigh(plane.covariance)
        idx = np.argsort(eigenvalues)
        evalsMin, evalsMid, evalsMax = idx[0], idx[1], idx[2]
        plane.min_eigen_value = eigenvalues[evalsMin]
        plane.mid_eigen_value = eigenvalues[evalsMid]
        plane.max_eigen_value = eigenvalues[evalsMax]
        plane.normal = eigenvectors[:, evalsMin]
        plane.y_normal = eigenvectors[:, evalsMid]
        plane.x_normal = eigenvectors[:, evalsMax]
        plane.radius = np.sqrt(eigenvalues[evalsMax])
        plane.d = -plane.normal.dot(plane.center)
        if plane.min_eigen_value < self.planer_threshold:
            plane.is_plane = True
            plane.plane_cov = np.zeros((6, 6))
        else:
            plane.is_plane = False
        if not plane.is_init:
            plane.id = self.plane_id_counter
            self.plane_id_counter += 1
            plane.is_init = True
        if plane.last_update_points_size == 0:
            plane.last_update_points_size = plane.points_size
            plane.is_update = True
        elif plane.points_size - plane.last_update_points_size > 100:
            plane.last_update_points_size = plane.points_size
            plane.is_update = True

    def update_plane(self, points: List[pointWithCov], plane: Plane):
        sum_ppt = (plane.covariance + plane.center[:, None] @ plane.center[None, :]) * plane.points_size
        sum_p = plane.center * plane.points_size
        for pv in points:
            sum_ppt += pv.point[:, None] @ pv.point[None, :]
            sum_p += pv.point
        plane.points_size += len(points)
        plane.center = sum_p / plane.points_size
        plane.covariance = sum_ppt / plane.points_size - plane.center[:, None] @ plane.center[None, :]
        eigenvalues, eigenvectors = np.linalg.eigh(plane.covariance)
        idx = np.argsort(eigenvalues)
        evalsMin, evalsMid, evalsMax = idx[0], idx[1], idx[2]
        plane.min_eigen_value = eigenvalues[evalsMin]
        plane.mid_eigen_value = eigenvalues[evalsMid]
        plane.max_eigen_value = eigenvalues[evalsMax]
        plane.normal = eigenvectors[:, evalsMin]
        plane.y_normal = eigenvectors[:, evalsMid]
        plane.x_normal = eigenvectors[:, evalsMax]
        plane.radius = np.sqrt(eigenvalues[evalsMax])
        plane.d = -plane.normal.dot(plane.center)
        plane.is_plane = plane.min_eigen_value < self.planer_threshold
        plane.is_update = True

    def init_octo_tree(self):
        if len(self.temp_points) > self.max_plane_update_threshold:
            self.init_plane(self.temp_points, self.plane_ptr)
            if self.plane_ptr.is_plane:
                self.octo_state = 0
                if len(self.temp_points) > self.max_cov_points_size:
                    self.update_cov_enable = False
                if len(self.temp_points) > self.max_points_size:
                    self.update_enable = False
            else:
                self.octo_state = 1
                self.cut_octo_tree()
            self.init_octo = True
            self.new_points_num = 0

    def cut_octo_tree(self):
        if self.layer >= self.max_layer:
            self.octo_state = 0
            return
        for pv in self.temp_points:
            xyz = [0, 0, 0]
            if pv.point[0] > self.voxel_center[0]:
                xyz[0] = 1
            if pv.point[1] > self.voxel_center[1]:
                xyz[1] = 1
            if pv.point[2] > self.voxel_center[2]:
                xyz[2] = 1
            leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2]
            if self.leaves[leafnum] is None:
                self.leaves[leafnum] = OctoTree(self.layer + 1)
                self.leaves[leafnum].voxel_center = self.voxel_center + (2 * np.array(xyz) - 1) * self.quater_length
                self.leaves[leafnum].quater_length = self.quater_length / 2
            self.leaves[leafnum].temp_points.append(pv)
            self.leaves[leafnum].new_points_num += 1
        for i in range(8):
            if self.leaves[i] is not None and len(self.leaves[i].temp_points) > self.leaves[i].max_plane_update_threshold:
                self.leaves[i].init_plane(self.leaves[i].temp_points, self.leaves[i].plane_ptr)
                if self.leaves[i].plane_ptr.is_plane:
                    self.leaves[i].octo_state = 0
                else:
                    self.leaves[i].octo_state = 1
                    self.leaves[i].cut_octo_tree()
                self.leaves[i].init_octo = True
                self.leaves[i].new_points_num = 0

    def UpdateOctoTree(self, pv: pointWithCov):
        if not self.init_octo:
            self.new_points_num += 1
            self.all_points_num += 1
            self.temp_points.append(pv)
            if len(self.temp_points) > self.max_plane_update_threshold:
                self.init_octo_tree()
        else:
            if self.plane_ptr.is_plane:
                if self.update_enable:
                    self.new_points_num += 1
                    self.all_points_num += 1
                    if self.update_cov_enable:
                        self.temp_points.append(pv)
                    else:
                        self.new_points.append(pv)
                    if self.new_points_num > self.update_size_threshold:
                        if self.update_cov_enable:
                            self.init_plane(self.temp_points, self.plane_ptr)
                        else:
                            self.update_plane(self.new_points, self.plane_ptr)
                            self.new_points.clear()
                        self.new_points_num = 0
                    if self.all_points_num >= self.max_cov_points_size:
                        self.update_cov_enable = False
                        self.temp_points.clear()
                    if self.all_points_num >= self.max_points_size:
                        self.update_enable = False
                        self.plane_ptr.update_enable = False
                        self.new_points.clear()
            else:
                if self.layer < self.max_layer:
                    self.temp_points.clear()
                    self.new_points.clear()
                    xyz = [0, 0, 0]
                    if pv.point[0] > self.voxel_center[0]:
                        xyz[0] = 1
                    if pv.point[1] > self.voxel_center[1]:
                        xyz[1] = 1
                    if pv.point[2] > self.voxel_center[2]:
                        xyz[2] = 1
                    leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2]
                    if self.leaves[leafnum] is None:
                        self.leaves[leafnum] = OctoTree(self.layer + 1)
                        self.leaves[leafnum].voxel_center = self.voxel_center + (2 * np.array(xyz) - 1) * self.quater_length
                        self.leaves[leafnum].quater_length = self.quater_length / 2
                    self.leaves[leafnum].UpdateOctoTree(pv)
                else:
                    if self.update_enable:
                        self.new_points_num += 1
                        self.all_points_num += 1
                        if self.update_cov_enable:
                            self.temp_points.append(pv)
                        else:
                            self.new_points.append(pv)
                        if self.new_points_num > self.update_size_threshold:
                            if self.update_cov_enable:
                                self.init_plane(self.temp_points, self.plane_ptr)
                            else:
                                self.update_plane(self.new_points, self.plane_ptr)
                                self.new_points.clear()
                            self.new_points_num = 0
                        if self.all_points_num >= self.max_cov_points_size:
                            self.update_cov_enable = False
                            self.temp_points.clear()
                        if self.all_points_num >= self.max_points_size:
                            self.update_enable = False
                            self.plane_ptr.update_enable = False
                            self.new_points.clear()

def buildVoxelMap(input_points: np.ndarray, rgb: Optional[np.ndarray] = None) -> Dict[VOXEL_LOC, OctoTree]:
    pv_list = generate_pv_list(input_points, rgb)
    voxel_map = {}
    for pv in pv_list:
        loc_xyz = pv.point / VOXEL_SIZE
        loc_xyz[loc_xyz < 0] -= 1.0
        position = VOXEL_LOC(int(loc_xyz[0]), int(loc_xyz[1]), int(loc_xyz[2]))
        if position in voxel_map:
            voxel_map[position].temp_points.append(pv)
            voxel_map[position].new_points_num += 1
        else:
            octo_tree = OctoTree(0)
            octo_tree.quater_length = VOXEL_SIZE / 4
            octo_tree.voxel_center = (np.array([position.x, position.y, position.z]) + 0.5) * VOXEL_SIZE
            octo_tree.temp_points.append(pv)
            octo_tree.new_points_num += 1
            voxel_map[position] = octo_tree
    for octo_tree in voxel_map.values():
        octo_tree.init_octo_tree()
    return voxel_map

def updateVoxelMap(input_points: np.ndarray, voxel_map: Dict[VOXEL_LOC, OctoTree], rgb: Optional[np.ndarray] = None):
    t0 = time.time()
    pv_list = generate_pv_list(input_points, rgb)

    for pv in pv_list:
        loc_xyz = pv.point / VOXEL_SIZE
        loc_xyz[loc_xyz < 0] -= 1.0
        position = VOXEL_LOC(int(loc_xyz[0]), int(loc_xyz[1]), int(loc_xyz[2]))
        if position in voxel_map:
            voxel_map[position].UpdateOctoTree(pv)
        else:
            octo_tree = OctoTree(0)
            octo_tree.quater_length = VOXEL_SIZE / 4
            octo_tree.voxel_center = (np.array([position.x, position.y, position.z]) + 0.5) * VOXEL_SIZE
            octo_tree.UpdateOctoTree(pv)
            voxel_map[position] = octo_tree
    t1 = time.time()
    print(t1-t0)

def voxel_down_sample(voxel_map: Dict[VOXEL_LOC, OctoTree]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = []
    colors = []
    normals = []

    def process_leaf_node(octo_tree: OctoTree):
        if octo_tree.octo_state == 0 or all(leaf is None for leaf in octo_tree.leaves):
            if octo_tree.plane_ptr.is_plane and octo_tree.plane_ptr.points_size > 0:
                point = octo_tree.plane_ptr.center
                normal = octo_tree.plane_ptr.normal
                rgb_points = [pv.rgb for pv in octo_tree.temp_points if pv.rgb is not None]
                color = np.mean(rgb_points, axis=0) if rgb_points else np.zeros(3)
            else:
                if not octo_tree.temp_points:
                    return
                points_array = np.array([pv.point for pv in octo_tree.temp_points])
                point = np.mean(points_array, axis=0)
                cov = np.cov(points_array.T, bias=True)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                idx = np.argsort(eigenvalues)
                normal = eigenvectors[:, idx[0]]
                if np.dot(normal, point - octo_tree.voxel_center) < 0:
                    normal = -normal
                rgb_points = [pv.rgb for pv in octo_tree.temp_points if pv.rgb is not None]
                color = np.mean(rgb_points, axis=0) if rgb_points else np.zeros(3)
            points.append(point)
            colors.append(color)
            normals.append(normal)
        else:
            for leaf in octo_tree.leaves:
                if leaf is not None:
                    process_leaf_node(leaf)

    for octo_tree in voxel_map.values():
        process_leaf_node(octo_tree)

    return (np.array(points), np.array(colors), np.array(normals))
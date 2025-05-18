import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from scipy.spatial import cKDTree

HASH_P = 116101
MAX_N = 10000000000
DEG2RAD = np.pi / 180.0

RANGE_INC = 0.02  # dept_err
DEGREE_INC = 0.05  # beam_err
PLANER_THRESHOLD = 0.0025  # min_eigen_value
VOXEL_SIZE = 0.05
MAX_LAYER = 2
MAX_POINTS_SIZE = 50
MAX_COV_POINTS_SIZE = 30
LAYER_POINT_SIZE = [3, 5, 8][:MAX_LAYER]


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
    cov: np.ndarray  # (3, 3)
    rgb: Optional[np.ndarray] = None  # (3,)


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


def generate_pv_list(points: np.ndarray, rgb: Optional[np.ndarray] = None) -> List[pointWithCov]:
    pv_list = []
    for i, point in enumerate(points):
        cov = calcBodyCov(point)
        rgb_i = rgb[i] if rgb is not None and i < len(rgb) else None
        pv = pointWithCov(point=point, cov=cov, rgb=rgb_i)
        pv_list.append(pv)
    return pv_list


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
        if plane.points_size < 3:  # 提高到 3
            plane.is_plane = False
            return

        plane.center = np.zeros(3)
        plane.covariance = np.zeros((3, 3))
        for pv in points:
            plane.center += pv.point
            plane.covariance += pv.point[:, None] @ pv.point[None, :]
        plane.center /= plane.points_size
        plane.covariance = plane.covariance / plane.points_size - plane.center[:, None] @ plane.center[None, :]

        eigenvalues, eigenvectors = np.linalg.eigh(plane.covariance)
        idx = np.argsort(eigenvalues)
        plane.min_eigen_value = eigenvalues[idx[0]]
        plane.mid_eigen_value = eigenvalues[idx[1]]
        plane.max_eigen_value = eigenvalues[idx[2]]
        plane.normal = eigenvectors[:, idx[0]]
        plane.y_normal = eigenvectors[:, idx[1]]
        plane.x_normal = eigenvectors[:, idx[2]]
        # 标准化法向量方向
        if plane.normal[2] < 0:
            plane.normal = -plane.normal
            plane.y_normal = -plane.y_normal
            plane.x_normal = -plane.x_normal
        plane.radius = np.sqrt(plane.max_eigen_value)
        plane.d = -plane.normal.dot(plane.center)

        plane.is_plane = plane.min_eigen_value < self.planer_threshold

        if plane.is_plane:
            plane.plane_cov = np.zeros((6, 6))
            for pv in points:
                J = np.zeros((1, 6))
                J[0, :3] = pv.point - plane.center
                J[0, 3:6] = -plane.normal
                point_cov = pv.cov
                plane.plane_cov += J.T @ J * np.dot(plane.normal, point_cov @ plane.normal)
            plane.plane_cov /= plane.points_size

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
        if plane.points_size + len(points) < 3:  # 提高到 3
            plane.is_plane = False
            return

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
        plane.min_eigen_value = eigenvalues[idx[0]]
        plane.mid_eigen_value = eigenvalues[idx[1]]
        plane.max_eigen_value = eigenvalues[idx[2]]
        plane.normal = eigenvectors[:, idx[0]]
        plane.y_normal = eigenvectors[:, idx[1]]
        plane.x_normal = eigenvectors[:, idx[2]]
        # 标准化法向量方向
        if plane.normal[2] < 0:
            plane.normal = -plane.normal
            plane.y_normal = -plane.y_normal
            plane.x_normal = -plane.x_normal
        plane.radius = np.sqrt(plane.max_eigen_value)
        plane.d = -plane.normal.dot(plane.center)

        plane.is_plane = plane.min_eigen_value < self.planer_threshold
        if plane.is_plane:
            plane.plane_cov = np.zeros((6, 6))
            for pv in points:
                J = np.zeros((1, 6))
                J[0, :3] = pv.point - plane.center
                J[0, 3:6] = -plane.normal
                point_cov = pv.cov
                plane.plane_cov += J.T @ J * np.dot(plane.normal, point_cov @ plane.normal)
            plane.plane_cov /= plane.points_size
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
            if self.leaves[i] is not None and len(self.leaves[i].temp_points) > self.leaves[
                i].max_plane_update_threshold:
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
                        self.leaves[leafnum].voxel_center = self.voxel_center + (
                                2 * np.array(xyz) - 1) * self.quater_length
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
        if position not in voxel_map:
            octo_tree = OctoTree(0)
            octo_tree.quater_length = VOXEL_SIZE / 4
            octo_tree.voxel_center = (np.array([position.x, position.y, position.z]) + 0.5) * VOXEL_SIZE
            voxel_map[position] = octo_tree
        voxel_map[position].temp_points.append(pv)
        voxel_map[position].new_points_num += 1
    for octo_tree in voxel_map.values():
        octo_tree.init_octo_tree()
    return voxel_map


def updateVoxelMap(input_points: np.ndarray, voxel_map: Dict[VOXEL_LOC, OctoTree], rgb: Optional[np.ndarray] = None):
    pv_list = generate_pv_list(input_points, rgb)
    for pv in pv_list:
        loc_xyz = pv.point / VOXEL_SIZE
        loc_xyz[loc_xyz < 0] -= 1.0
        position = VOXEL_LOC(int(loc_xyz[0]), int(loc_xyz[1]), int(loc_xyz[2]))
        if position not in voxel_map:
            octo_tree = OctoTree(0)
            octo_tree.quater_length = VOXEL_SIZE / 4
            octo_tree.voxel_center = (np.array([position.x, position.y, position.z]) + 0.5) * VOXEL_SIZE
            voxel_map[position] = octo_tree
        voxel_map[position].UpdateOctoTree(pv)


def voxel_down_sample(
        input_points: np.ndarray,
        voxel_map: Dict[VOXEL_LOC, OctoTree],
        rgb: Optional[np.ndarray] = None,
        voxel_size: float = VOXEL_SIZE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对当前帧点云进行体视素下采样，基于八叉树叶节点采样物体表面，使用现有体视素地图。

    Args:
        input_points: 输入点云，形状为 (N, 3)
        voxel_map: 体视素地图，键为 VOXEL_LOC，值为 OctoTree
        rgb: 可选的RGB颜色，形状为 (N, 3)
        voxel_size: 体视素尺寸，默认为 VOXEL_SIZE

    Returns:
        points: 下采样后的点云，形状为 (M, 3)
        colors: 对应的颜色，形状为 (M, 3)
        normals: 对应的法向量，形状为 (M, 3)
    """
    if len(input_points) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))

    # 初始化 voxel_map（如果为空）
    if voxel_map is None:
        voxel_map = {}

    # 1. 将当前帧点云分配到体视素（更新 temp_points）
    for i, point in enumerate(input_points):
        loc_xyz = point / voxel_size
        loc_xyz[loc_xyz < 0] -= 1.0
        position = VOXEL_LOC(int(loc_xyz[0]), int(loc_xyz[1]), int(loc_xyz[2]))
        if position not in voxel_map:
            octo_tree = OctoTree(0)
            octo_tree.quater_length = voxel_size / 4
            octo_tree.voxel_center = (np.array([position.x, position.y, position.z]) + 0.5) * voxel_size
            voxel_map[position] = octo_tree
        cov = calcBodyCov(point)
        rgb_i = rgb[i] if rgb is not None and i < len(rgb) else None
        pv = pointWithCov(point=point, cov=cov, rgb=rgb_i)
        voxel_map[position].temp_points.append(pv)
        voxel_map[position].new_points_num += 1

    # 2. 初始化八叉树（仅对有新点的体视素）
    for octo_tree in voxel_map.values():
        if octo_tree.new_points_num > 0:
            octo_tree.init_octo_tree()

    # 3. 从叶节点采样
    points = []
    colors = []
    normals = []

    def process_leaf_node(octo_tree: OctoTree):
        if octo_tree.octo_state == 0 or all(leaf is None for leaf in octo_tree.leaves):
            if not octo_tree.temp_points or len(octo_tree.temp_points) < 3:
                return
            # 体视素内点的平均位置
            points_array = np.array([pv.point for pv in octo_tree.temp_points])
            point = np.mean(points_array, axis=0)

            # 法向量：优先使用 plane_ptr.normal
            if octo_tree.plane_ptr.is_plane and octo_tree.plane_ptr.points_size >= 3:
                normal = octo_tree.plane_ptr.normal
            else:
                cov = np.cov(points_array.T, bias=True)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                idx = np.argsort(eigenvalues)
                normal = eigenvectors[:, idx[0]]

            # 标准化法向量
            normal = normal / (np.linalg.norm(normal) + 1e-6)
            if normal[2] < 0:
                normal = -normal

            # 颜色：平均RGB
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

    # 4. 转换为NumPy数组
    points = np.array(points) if points else np.zeros((0, 3))
    colors = np.array(colors) if colors else np.zeros((0, 3))
    normals = np.array(normals) if normals else np.zeros((0, 3))

    # 5. 清空 temp_points，准备下一次处理
    for octo_tree in voxel_map.values():
        octo_tree.temp_points.clear()
        octo_tree.new_points_num = 0

    return points, colors, normals 78ip[=*/
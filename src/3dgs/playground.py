import sys
import os

import torchvision.transforms.functional
sys.path.append(os.path.join(os.path.dirname(__file__), 'hislam2'))

import torch
import cv2
import numpy as np
import numpy as np
from scipy.spatial.transform import Rotation as R

from hislam2.gs_backend import GSBackEnd
from hislam2.util.utils import load_config
from tqdm import tqdm

from transformers import pipeline
from PIL import Image
import torchvision

def pose_to_extrinsic(tx, ty, tz, qx, qy, qz, qw):
    def quaternion_to_rotation_matrix(qx, qy, qz, qw):
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ])
        return R
    R_A = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    t_A = np.array([tx, ty, tz])
    T = np.array([  [0, -1, 0],
                    [0, 0, -1],
                    [1, 0, 0]])
    R_B = T @ R_A @ T.T
    t_B = t_A
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R_B
    extrinsic[:3, 3] = t_B
    return torch.tensor(extrinsic, dtype=torch.float32)

def extrinsic_to_pose(T):
    """
    Convert a 4x4 extrinsic transformation matrix to translation and quaternion.
    
    Parameters
    ----------
    T : numpy.ndarray
        A 4x4 extrinsic transformation matrix with the following form:
            [ R (3x3)   t (3x1) ]
            [ 0  0  0     1     ]
    
    Returns
    -------
    tx, ty, tz, qx, qy, qz, qw : float
        The translation components (tx, ty, tz) and the quaternion (qx, qy, qz, qw).
        Note: The quaternion is returned in the (x, y, z, w) format.
    """
    # Extract the translation vector from the last column
    t = T[:3, 3]
    
    # Extract the rotation matrix from the top-left 3x3 block
    R_mat = T[:3, :3]
    
    # Convert the rotation matrix to a quaternion using SciPy.
    # The result is in the (x, y, z, w) order.
    r = R.from_matrix(R_mat)
    q = r.as_quat()
    
    return torch.tensor([t[0], t[1], t[2], q[0], q[1], q[2], q[3]])

num_frames = 187
video = {"images": [], "poses": [], "intrinsics": [], "depths": [], "normals": [], "tstamp": []}
config = load_config("./config/replica_config.yaml")
gs = GSBackEnd(config, save_dir="./output", use_gui=False)
pipe = pipeline(task="depth-estimation", model="xingyang1/Distill-Any-Depth-Large-hf", use_fast=True)
for i in tqdm(range(num_frames)):
    image_pil = Image.open(f"../red_sculpture/frame{i}/image.png")
    image = torch.tensor(np.array(image_pil)).permute(2, 0, 1)

    res = pipe(image_pil)
    depth_image, depth_map = res["depth"], res["predicted_depth"]
    depth_image.save(f"../red_sculpture/frame{i}/depth.png")
    depth_map = (depth_map + 20) / 20
    print(f"image: {image.shape}, max: {image.max()}, min: {image.min()}")
    print(f"depth_map: {depth_map.shape}, max: {depth_map.max()}, min: {depth_map.min()}")

    tx, ty, tz, qx, qy, qz, qw = torch.load(f"../red_sculpture/frame{i}/pose.pt").tolist()
    extrinsic = pose_to_extrinsic(tx, ty, tz, qx, qy, qz, qw)
    pose = extrinsic_to_pose(extrinsic)
    intrinsic = torch.tensor(np.loadtxt(f"../red_sculpture/intrinsic.txt"))

    video["images"].append(image)
    video["poses"].append(pose)
    video["intrinsics"].append(intrinsic)
    video["depths"].append(depth_map)
    video["tstamp"].append(i)

    if i >=11:
        viz_idx = torch.arange(i-11, i+1)
        
        data = {'viz_idx':  viz_idx,
                'tstamp':   torch.tensor(video["tstamp"])[viz_idx],
                'poses':    torch.stack(video["poses"])[viz_idx],
                'images':   torch.stack(video["images"])[viz_idx],
                'depths':   torch.stack(video["depths"])[viz_idx],
                'intrinsics':   torch.stack(video["intrinsics"])[viz_idx],
                'pose_updates':  None,
                'scale_updates': None}
        gs.process_track_data(data)

viz_idx = torch.arange(0, num_frames)
data = {'viz_idx':  viz_idx,
        'tstamp':   torch.tensor(video["tstamp"])[viz_idx],
        'poses':    torch.stack(video["poses"])[viz_idx],
        'images':   torch.stack(video["images"])[viz_idx],
        'depths':   torch.stack(video["depths"])[viz_idx],
        'intrinsics':   torch.stack(video["intrinsics"])[viz_idx],
        'pose_updates':  None,
        'scale_updates': None}
gs.process_track_data(data)
updated_poses = gs.finalize()

    
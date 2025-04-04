import torch
from munch import munchify
import yaml
from gs_backend import GSBackEnd
import numpy as np
from scipy.spatial.transform import Rotation as R


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

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return munchify(yaml.safe_load(f))

class GSProcessorWrapper:
    """Wrapper for GSBackEnd processing."""
    def __init__(self, config, save_dir, use_gui=False):
        self.gs_backend = GSBackEnd(config, save_dir, use_gui=use_gui)
        self.save_dir = save_dir
        self.gs_backend.start()
        self.counter = 0

    def process_data(self, data_packet):
        """Process a batch of data with GSBackEnd."""
        self.gs_backend.process_track_data(data_packet)
        self.counter += len(data_packet['viz_idx'])

    def finalize(self):
        """Finalize processing and terminate GSBackEnd."""
        result = self.gs_backend.finalize()
        self.gs_backend.terminate()
        return result
import torch
from munch import munchify
import yaml
from gs_backend import GSBackEnd
from lietorch import SE3

def rotation_matrix_to_quaternion(rot_mat):
    """Convert a 3x3 rotation matrix to a quaternion [qx, qy, qz, qw]."""
    device = rot_mat.device
    quat = torch.zeros(4, device=device)
    trace = torch.trace(rot_mat)

    if trace > 0:
        S = torch.sqrt(trace + 1.0) * 2
        quat[3] = 0.25 * S
        quat[0] = (rot_mat[2, 1] - rot_mat[1, 2]) / S
        quat[1] = (rot_mat[0, 2] - rot_mat[2, 0]) / S
        quat[2] = (rot_mat[1, 0] - rot_mat[0, 1]) / S
    elif rot_mat[0, 0] > max(rot_mat[1, 1], rot_mat[2, 2]):
        S = torch.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2]) * 2
        quat[3] = (rot_mat[2, 1] - rot_mat[1, 2]) / S
        quat[0] = 0.25 * S
        quat[1] = (rot_mat[0, 1] + rot_mat[1, 0]) / S
        quat[2] = (rot_mat[0, 2] + rot_mat[2, 0]) / S
    elif rot_mat[1, 1] > rot_mat[2, 2]:
        S = torch.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2]) * 2
        quat[3] = (rot_mat[0, 2] - rot_mat[2, 0]) / S
        quat[0] = (rot_mat[0, 1] + rot_mat[1, 0]) / S
        quat[1] = 0.25 * S
        quat[2] = (rot_mat[1, 2] + rot_mat[2, 1]) / S
    else:
        S = torch.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1]) * 2
        quat[3] = (rot_mat[1, 0] - rot_mat[0, 1]) / S
        quat[0] = (rot_mat[0, 2] + rot_mat[2, 0]) / S
        quat[1] = (rot_mat[1, 2] + rot_mat[2, 1]) / S
        quat[2] = 0.25 * S

    return quat / torch.norm(quat)

def transform_pose_to_camera_frame(pose):
    """Transform LiDAR-to-world pose to world-to-camera pose."""
    translation = pose[:3]
    quaternion = pose[3:] / torch.norm(pose[3:] + 1e-6)

    new_translation = torch.tensor([translation[2], -translation[0], -translation[1]],
                                   device=pose.device)

    rotation_transform = torch.tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]],
                                      dtype=torch.float32, device=pose.device)
    transform_quat = rotation_matrix_to_quaternion(rotation_transform)

    se3_pose = SE3.InitFromVec(pose.unsqueeze(0))
    transform_vec = torch.cat([torch.zeros(3, device=pose.device), transform_quat])
    transform_se3 = SE3.InitFromVec(transform_vec.unsqueeze(0))

    return (transform_se3 * se3_pose).inv().vec().squeeze(0)

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
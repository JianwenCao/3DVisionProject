import os
import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation as R


def to_se3_vec(pose_mat):
    quat = R.from_matrix(pose_mat[:3, :3]).as_quat()
    return np.hstack((pose_mat[:3, 3], quat))


nrs = ['0000', '0054', '0059', '0106', '0169', '0181', '0207', '0233']
seqs = [s for s in sorted(glob("data/ScanNet/scene*")) if any(n in s for n in nrs)]

for scan_path in seqs:
    print("preprocesing", os.path.basename(scan_path))

    K = np.loadtxt(f'{scan_path}/intrinsic/intrinsic_color.txt')
    K = [K[0,0], K[1,1], K[0,2], K[1,2]]
    np.savetxt(f'{scan_path}/calib.txt', K)

    traj = []
    for i, p in enumerate(sorted(glob(scan_path+'/pose/*'))):
        p = np.loadtxt(p)
        p = to_se3_vec(p)
        if np.isnan(p).any():
            print(f"skip {i} {p} due to NaN values")
            p = np.zeros(7)
        traj.append([i] + list(p))
    traj = np.stack(traj)
    np.savetxt(scan_path+'/traj.txt', traj)

import os
import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation as R


def to_se3_vec(pose):
    quat = R.from_matrix(pose[:3, :3]).as_quat()
    return np.hstack((pose[:3, 3], quat))


parent = os.path.dirname(__file__)
seqs = sorted(glob('data/Replica/room*')) + sorted(glob('data/Replica/office*'))
for seq in seqs:
    print(seq)

    os.system(f'rm -rf {seq}/colors')
    os.system(f'rm -rf {seq}/depths')
    os.makedirs(f'{seq}/colors', exist_ok=True)
    os.makedirs(f'{seq}/depths', exist_ok=True)
    
    for color, depth in zip(glob(f'{seq}/results/frame*'), glob(f'{seq}/results/depth*')):
        # print(color)
        os.symlink(f'../results/{os.path.basename(color)}', color.replace('results', 'colors'))
        os.symlink(f'../results/{os.path.basename(depth)}', depth.replace('results', 'depths'))

    traj = np.loadtxt(f"{seq}/traj.txt").reshape(-1, 4, 4)
    traj_tum = [np.hstack(([i], to_se3_vec(p))) for i, p in enumerate(traj)]
    np.savetxt(f'{seq}/traj_tum.txt', traj_tum)

import os    # nopep8
import sys   # nopep8
sys.path.append(os.path.join(os.path.dirname(__file__), 'hislam2'))   # nopep8
import time
import torch
import cv2
import re
import os
import argparse
import numpy as np
import lietorch
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

from tqdm import tqdm
from torch.multiprocessing import Process, Queue
from hi2 import Hi2

from interfaces.ros_listener import HISLAM2Data, HISLAM2Node



def show_image(image, depth_prior, depth, normal):
    from util.utils import colorize_np
    image = image[[2,1,0]].permute(1, 2, 0).cpu().numpy()
    depth = colorize_np(np.concatenate((depth_prior.cpu().numpy(), depth.cpu().numpy()), axis=1), range=(0, 4))
    normal = normal.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('rgb / prior normal / aligned prior depth / JDSA depth', np.concatenate((image / 255.0, (normal[...,[2,1,0]]+1.)/2., depth), axis=1)[::2,::2])
    cv2.waitKey(1)


def save_trajectory(hi2, traj_full, imagedir, output, start=0):
    t = hi2.video.counter.value
    tstamps = hi2.video.tstamp[:t]
    poses_wc = lietorch.SE3(hi2.video.poses[:t]).inv().data
    np.save("{}/intrinsics.npy".format(output), hi2.video.intrinsics[0].cpu().numpy()*8)

    tstamps_full = np.array([float(re.findall(r"[+]?(?:\d*\.\d+|\d+)", x)[-1]) for x in sorted(os.listdir(imagedir))[start:]])[..., np.newaxis]
    tstamps_kf = tstamps_full[tstamps.cpu().numpy().astype(int)]
    ttraj_kf = np.concatenate([tstamps_kf, poses_wc.cpu().numpy()], axis=1)
    np.savetxt(f"{output}/traj_kf.txt", ttraj_kf)                     #  for evo evaluation 
    if traj_full is not None:
        ttraj_full = np.concatenate([tstamps_full[:len(traj_full)], traj_full], axis=1)
        np.savetxt(f"{output}/traj_full.txt", ttraj_full)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--config", type=str, help="path to configuration file")
    parser.add_argument("--output", default='outputs/demo', help="path to save output")
    parser.add_argument("--gtdepthdir", type=str, default=None, help="optional for evaluation, assumes 16-bit depth scaled by 6553.5")

    parser.add_argument("--weights", default=os.path.join(os.path.dirname(__file__), "pretrained_models/droid.pth"))
    parser.add_argument("--buffer", type=int, default=-1, help="number of keyframes to buffer (default: 1/10 of total frames)")
    parser.add_argument("--undistort", action="store_true", help="undistort images if calib file contains distortion parameters")
    parser.add_argument("--cropborder", type=int, default=0, help="crop images to remove black border")

    parser.add_argument("--droidvis", action="store_true")
    parser.add_argument("--gsvis", action="store_true")

    parser.add_argument("--start", type=int, default=0, help="start frame")
    parser.add_argument("--length", type=int, default=100000, help="number of frames to process")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    torch.multiprocessing.set_start_method('spawn')

    # Initialize HISLAM2Data and HISLAM2Node
    data_store = HISLAM2Data()
    node = HISLAM2Node(data_store)
    node_process = Process(target=node.run)
    node_process.start()

    pbar = tqdm(range(args.length), desc="Processing keyframes")
    while True:
        image = data_store.get_latest_camera()
        pose = data_store.get_latest_pose()
        lidar = data_store.get_latest_lidar()

        if image is None or pose is None or lidar is None:
            time.sleep(0.01)
            continue

        t = time.time()
        intrinsics = torch.tensor(args.calib[:4])
        is_last = False  # Modify this as needed

        if hi2 is None:
            args.image_size = [image.shape[1], image.shape[2]]
            hi2 = Hi2(args)

        hi2.track(t, image[None], intrinsics[None], is_last)

        if args.droidvis and hi2.video.tstamp[hi2.video.counter.value-1] == t:
            from geom.ba import get_prior_depth_aligned
            index = hi2.video.counter.value-2
            depth_prior, _ = get_prior_depth_aligned(hi2.video.disps_prior_up[index][None].cuda(), hi2.video.dscales[index][None])
            show_image(image, 1./depth_prior.squeeze(), 1./hi2.video.disps_up[index], hi2.video.normals[index])
        pbar.set_description(f"Processing keyframe {hi2.video.counter.value} gs {hi2.gs.gaussians._xyz.shape[0]}")

        if is_last:
            pbar.close()
            break

    node_process.join()

    traj = hi2.terminate()
    save_trajectory(hi2, traj, args.imagedir, args.output, start=args.start)

    print("Done")

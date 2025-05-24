import cv2
import os
import sys
import numpy as np
from tqdm import tqdm

def extract_frames(input_video_path, output, output_colmap):
    os.makedirs(output, exist_ok=True)
    os.makedirs(output_colmap, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error: Couldn't open video file.")
        return

    frame_number = 0

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Extracting frames")
    while True:
        # Read frame from video
        ret, frame = cap.read()

        # Break if no more frames are available
        if not ret:
            break
        
        # Save the current frame as an image
        cv2.imwrite(os.path.join(output, f"{frame_number:06d}.jpg"), frame)

        # Save every 10th frame for colmap, not more than 100 images for faster processing
        if frame_number % 10 == 0 and frame_number < 1000:
            cv2.imwrite(os.path.join(output_colmap, f"{frame_number:06d}.jpg"), frame)

        frame_number += 1
        pbar.update(1)

    # Release the video capture object
    pbar.close()
    cap.release()


def run_colmap(output):
    colmap_binary = "colmap"

    db = f"{output}/colmap.db"
    images = f"{output}/images_colmap"

    os.system(f"{colmap_binary} feature_extractor --ImageReader.camera_model OPENCV --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --database_path {db} --image_path {images}")

    os.system(f"{colmap_binary} sequential_matcher --SiftMatching.guided_matching=true --database_path {db}")

    os.system(f"mkdir {output}/sparse")
    os.system(f"{colmap_binary} mapper --database_path {db} --image_path {images} --output_path {output}/sparse")

    os.system(f"{colmap_binary} bundle_adjuster --input_path {output}/sparse/0 --output_path {output}/sparse/0 --BundleAdjustment.refine_principal_point 1")

    os.system(f"mkdir {output}/sparse_txt")
    os.system(f"{colmap_binary} model_converter --input_path {output}/sparse/0 --output_path {output}/sparse_txt --output_type TXT")


# Example usage:
input_video_path = sys.argv[1]       # Path to the input video file
output = sys.argv[2]          # Path to the output folder
extract_frames(input_video_path, output + "/images", output + "/images_colmap")

# Run colmap
run_colmap(output)

# dump calib file
calib = open(f'{output}/sparse_txt/cameras.txt').readlines()[-1].split()[4:]
open(f'{output}/calib.txt', 'w').write(" ".join(calib))

#!/usr/bin/env python3
"""
hi_slam2_service.py

This script runs entirely in your Conda environment and uses roslibpy to connect
to a rosbridge server (running on your ROS environment). It subscribes to sensor topics,
decodes the incoming messages into efficient data structures (using NumPy and PyTorch),
and either processes them in real time for mapping (online mode) or saves them to disk
for offline analysis (preprocess mode).

Usage:
    python hi_slam2_service.py --mode [online|preprocess] --output-name [str] [--log-level [INFO|DEBUG]]
"""

import argparse
import roslibpy
import torch
import cv2
import numpy as np
import base64
import time
import threading
import queue
import os
import logging

# ----------------------------
# Queues for asynchronous processing
# ----------------------------
image_queue = queue.Queue(maxsize=100)
pose_queue = queue.Queue(maxsize=100)
lidar_queue = queue.Queue(maxsize=100)

# ----------------------------
# Helper Functions for Message Decoding
# ----------------------------
def decode_ros_image(msg):
    """
    Decode a ROS Image message (as a dict) into an RGB NumPy array.
    Handles raw image data for common encodings such as 'rgb8', 'bgr8', or 'mono8'.
    If the encoding is not one of these, falls back to cv2.imdecode.
    """
    try:
        encoding = msg.get('encoding', '').lower()
        height = int(msg.get('height', 0))
        width = int(msg.get('width', 0))
        if height == 0 or width == 0:
            raise ValueError("Invalid image dimensions")
        # Decode the raw data from base64.
        img_data = base64.b64decode(msg.get('data', ''))
        if encoding in ['rgb8', 'bgr8', 'mono8']:
            channels = 3 if encoding in ['rgb8', 'bgr8'] else 1
            expected_length = height * width * channels
            if len(img_data) != expected_length:
                raise ValueError(f"Decoded data length ({len(img_data)}) does not match expected ({expected_length}).")
            np_arr = np.frombuffer(img_data, dtype=np.uint8)
            img = np_arr.reshape((height, width, channels))
            if encoding == 'bgr8':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if encoding == 'mono8':
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return img
        else:
            np_arr = np.frombuffer(img_data, dtype=np.uint8)
            np_arr_copy = np_arr.copy()  # Ensure the array is writable.
            img = cv2.imdecode(np_arr_copy, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image data with imdecode")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print("Image decode error:", e)
        return None


point_dtype_32 = np.dtype([
    ('x', np.float32),            # offset 0
    ('y', np.float32),            # offset 4
    ('z', np.float32),            # offset 8
    ('intensity', np.float32),    # offset 12
    ('rgb', np.float32),          # offset 16
    ('_padding', 'V12')           # offset 20..31
])

def decode_rgb_pointcloud(msg):
    """
    Simplified decoder for a sensor_msgs/PointCloud2 with:
      - point_step = 32 bytes
      - fields: x, y, z, intensity, rgb (packed as float), plus 12 bytes padding
      - possibly multi-row (height>1), each row has row_step bytes
    Returns:
      A NumPy array of shape (N,6) -> columns: [x,y,z,r,g,b].
    """
    try:
        # 1) Extract base64-encoded or raw bytes from msg['data'].
        raw_data = msg.get('data', '')
        if isinstance(raw_data, dict):
            raw_data = raw_data.get('data', '')
        
        if isinstance(raw_data, str):
            data_bytes = base64.b64decode(raw_data)
            logger.debug("Decoded pointcloud data from base64 string.")
        elif isinstance(raw_data, (bytes, bytearray)):
            data_bytes = raw_data
        else:
            # If it's a list of floats, convert to bytes.
            if isinstance(raw_data, list):
                data_array = np.array(raw_data, dtype=np.float32)
                data_bytes = data_array.tobytes()
            else:
                raise ValueError(f"Unexpected type for pointcloud data: {type(raw_data)}")
        
        # 2) Parse metadata from the message (height, width, row_step, is_bigendian).
        height = msg.get('height', 1)
        width = msg.get('width', 0)
        point_step = msg.get('point_step', 32)
        row_step = msg.get('row_step', point_step * width)
        is_bigendian = msg.get('is_bigendian', False)
        
        if width == 0:
            # If the user didn't set width, try inferring from total bytes
            width = len(data_bytes) // point_step
        
        # 3) If big-endian, swap bytes after reading them.
        #    We'll do it after extracting each row, see below.
        
        # 4) Read each row in a loop, to handle any row padding properly.
        all_points = []
        for row_i in range(height):
            start = row_i * row_step
            end = start + row_step
            row_data = data_bytes[start:end]
            
            # Convert row_data to a structured array with our 32-byte dtype.
            row_points = np.frombuffer(row_data, dtype=point_dtype_32)
            
            # If big-endian, byteswap in place:
            if is_bigendian:
                row_points = row_points.byteswap().newbyteorder()
            
            all_points.append(row_points)
        
        # Concatenate all rows into one array of shape (height*width,).
        points = np.concatenate(all_points, axis=0)
        
        # 5) Extract x,y,z.
        xyz = np.stack((points['x'], points['y'], points['z']), axis=1)
        
        # 6) Unpack rgb. It's stored as a float, but we treat the bits as a uint32.
        rgb_float = points['rgb']
        rgb_uint = rgb_float.view(np.uint32)
        # Extract channels (assuming 0x00RRGGBB).
        r = ((rgb_uint >> 16) & 0xFF).astype(np.float32)
        g = ((rgb_uint >> 8) & 0xFF).astype(np.float32)
        b = (rgb_uint & 0xFF).astype(np.float32)
        
        # 7) Combine xyz and rgb => shape (N,6).
        rgb_arr = np.stack((r, g, b), axis=1)
        result = np.concatenate((xyz, rgb_arr), axis=1)
        
        return result
    except Exception as e:
        logger.error("Error decoding colorized pointcloud: %s", e)
        return None
    
def get_full_timestamp(msg):
    header = msg.get('header', {})
    stamp = header.get('stamp', {})
    secs = stamp.get('secs', None)
    nsecs = stamp.get('nsecs', 0)
    if secs is None:
        return time.time()
    return secs + nsecs * 1e-9


# ----------------------------
# Worker Functions with Timing
# ----------------------------
def process_image_queue(data_store):
    while True:
        msg = image_queue.get()
        process_start = time.perf_counter()
        img = decode_ros_image(msg)
        if img is not None:
            tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            stamp = get_full_timestamp(msg)
            data_store.store_camera(stamp, tensor)
            process_end = time.perf_counter()
            logger.debug(f"[Camera] Processed image @ timestamp {stamp:.9f} | "
                         f"Took {(process_end - process_start)*1000:.2f} ms")
        image_queue.task_done()

def process_pose_queue(data_store):
    while True:
        msg = pose_queue.get()
        process_start = time.perf_counter()
        try:
            stamp = get_full_timestamp(msg)
            pos = msg.get('pose', {}).get('pose', {}).get('position', {})
            ori = msg.get('pose', {}).get('pose', {}).get('orientation', {})
            tx = pos.get('x', 0.0)
            ty = pos.get('y', 0.0)
            tz = pos.get('z', 0.0)
            qx = ori.get('x', 0.0)
            qy = ori.get('y', 0.0)
            qz = ori.get('z', 0.0)
            qw = ori.get('w', 1.0)
            pose_tensor = torch.tensor([tx, ty, tz, qx, qy, qz, qw], dtype=torch.float32)
            data_store.store_pose(stamp, pose_tensor)
            process_end = time.perf_counter()
            logger.debug(f"[Pose] Processed pose @ timestamp {stamp:.9f} | "
                         f"Took {(process_end - process_start)*1000:.2f} ms")
        except Exception as e:
            logger.error("Pose processing error: %s", e)
        pose_queue.task_done()

def process_lidar_queue(data_store):
    while True:
        msg = lidar_queue.get()
        process_start = time.perf_counter()
        pc_decoded = decode_rgb_pointcloud(msg)
        if pc_decoded is not None:
            pc_tensor = torch.from_numpy(pc_decoded).float()
            stamp = get_full_timestamp(msg)
            data_store.store_lidar(stamp, pc_tensor)
            process_end = time.perf_counter()
            logger.debug(f"[LiDAR] Processed {pc_tensor.shape[0]} points @ timestamp {stamp:.9f} | "
                         f"Took {(process_end - process_start)*1000:.2f} ms")
        lidar_queue.task_done()


# ----------------------------
# Data Storage Class with Synchronization
# ----------------------------
class HISLAM2Data:
    """
    Thread-safe container to store incoming sensor data.
    """
    def __init__(self):
        self.camera_data = {}  # {timestamp: torch.Tensor [C, H, W]}
        self.pose_data = {}    # {timestamp: torch.Tensor [tx, ty, tz, qx, qy, qz, qw]}
        self.lidar_data = {}   # {timestamp: torch.Tensor [N, 6], columns: [x, y, z, r, g, b]}
        self.lock = threading.Lock()
        self.last_synchronized_time = -1 # Init -1 to indicate no sync yet

    def store_camera(self, stamp, img_tensor):
        with self.lock:
            self.camera_data[stamp] = img_tensor

    def store_pose(self, stamp, pose_tensor):
        with self.lock:
            self.pose_data[stamp] = pose_tensor

    def store_lidar(self, stamp, points):
        with self.lock:
            self.lidar_data[stamp] = points

    def pop_synced_data(self, exposure_time=0.0, tolerance=0.15):
        """
        Synchronize camera, pose, and LiDAR data using the first camera frame as the anchor.
        The camera timestamp is adjusted by an exposure_time offset.
        NOTE: Tolerance must be approximately 150-200ms as FAST-LIVO2 synced stream is at 10Hz.
        - Future work to tightly integrate with FAST-LIVO2 sync_packages().
        
        Mimics FAST-LIVO's approach:
        - If last_synchronized_time is negative, initialize it with the earliest LiDAR timestamp.
        - Compute img_capture_time = first camera time + exposure_time.
        - If img_capture_time is earlier than last_synchronized_time (within a tolerance), discard that camera frame.
        - If img_capture_time is later than the newest LiDAR or pose data, return None (i.e. wait for more data).
        - Otherwise, select the camera frame as the anchor and find the closest pose and LiDAR times.
        If both differences are within tolerance, pop and return the synchronized package.

        Returns:
        (sync_time, camera_tensor, pose_tensor, lidar_tensor) or None if no sync data exists.
        """
        with self.lock:
            # Ensure that all three sensor dictionaries have data.
            if not (self.camera_data and self.pose_data and self.lidar_data):
                logger.debug("One or more sensor dictionaries are empty.")
                return None

            # Use the earliest camera timestamp as the reference.
            cam_keys = sorted(self.camera_data.keys())
            cam_time = cam_keys[0]
            # Adjust camera timestamp with the exposure offset.
            img_capture_time = cam_time + exposure_time

            # If last_synchronized_time is not initialized, set it to the earliest LiDAR timestamp.
            if self.last_synchronized_time < 0:
                lidar_keys = sorted(self.lidar_data.keys())
                self.last_synchronized_time = lidar_keys[0]

            # Get the newest LiDAR and pose timestamps.
            lidar_newest = max(self.lidar_data.keys())
            pose_newest = max(self.pose_data.keys())

            # If the adjusted image capture time is earlier than the last synchronized time, discard this camera frame.
            if img_capture_time < self.last_synchronized_time + 1e-5:
                logger.debug("Image capture time is earlier than last synchronized time, discarding camera frame.")
                self.camera_data.pop(cam_time)
                return None

            # If the image capture time is later than the newest available LiDAR or pose data, wait.
            if img_capture_time > lidar_newest or img_capture_time > pose_newest:
                logger.debug("Image capture time is later than newest LiDAR/pose data, waiting for more data.")
                return None

            # Find the closest pose timestamp to img_capture_time.
            closest_pose = min(self.pose_data.keys(), key=lambda t: abs(t - img_capture_time))
            # Find the closest LiDAR timestamp to img_capture_time.
            closest_lidar = min(self.lidar_data.keys(), key=lambda t: abs(t - img_capture_time))

            # Check if the differences are within tolerance.
            if abs(img_capture_time - closest_pose) > tolerance or abs(img_capture_time - closest_lidar) > tolerance:
                logger.debug("Closest sensor data not within tolerance.")
                return None

            # Update last_synchronized_time to the synchronized time.
            self.last_synchronized_time = img_capture_time

            # Pop the synchronized data.
            cam = self.camera_data.pop(cam_time)
            pose = self.pose_data.pop(closest_pose)
            lidar = self.lidar_data.pop(closest_lidar)
            logger.debug(f"Popped synchronized data: camera {cam_time:.9f}, pose {closest_pose:.9f}, "
                        f"lidar {closest_lidar:.9f}; sync_time = {img_capture_time:.9f}")
            logger.info(f"Popped synchronized data: camera {cam_time:.9f}, pose {closest_pose:.9f}, "
                        f"lidar {closest_lidar:.9f}; sync_time = {img_capture_time:.9f}")
            return img_capture_time, cam, pose, lidar



    # Optional get functions for online mode
    def get_latest_camera(self):
        with self.lock:
            if self.camera_data:
                return self.camera_data[max(self.camera_data.keys())]
            return None

    def get_latest_pose(self):
        with self.lock:
            if self.pose_data:
                return self.pose_data[max(self.pose_data.keys())]
            return None

    def get_latest_lidar(self):
        with self.lock:
            if self.lidar_data:
                return self.lidar_data[max(self.lidar_data.keys())]
            return None

# ----------------------------
# Preprocess and Save (offline)
# ----------------------------
def preprocess_data(data_store, output_dir):
    """
    Continuously checks for synchronized data and saves each frame's data into a new folder.
    
    Each frame folder (e.g. frame0, frame1, ...) will contain:
      - image.png : the camera image
      - points.npy: the lidar points stored as a NumPy array (shape: [N,6])
      - pose.pt   : the pose stored as a torch tensor
    """
    import cv2  # local import to ensure cv2 is available for image saving
    index = 0
    while True:
        synced = data_store.pop_synced_data()
        if synced is not None:
            stamp, cam, pose, lidar = synced
            # Create a folder for the current frame.
            frame_folder = os.path.join(output_dir, f"frame{index}")
            os.makedirs(frame_folder, exist_ok=True)
            
            # --- Save Camera Image ---
            try:
                # Assume cam is a torch tensor with shape (C, H, W) in RGB.
                # Convert to (H, W, C) and then to uint8.
                img_np = cam.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                # Convert from RGB to BGR for OpenCV.
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                image_path = os.path.join(frame_folder, "image.png")
                cv2.imwrite(image_path, img_bgr)
            except Exception as e:
                print("Error saving image:", e)
            
            # --- Save Lidar Points ---
            try:
                # Convert lidar (torch tensor) to a NumPy array.
                points = lidar.cpu().numpy()
                points_path = os.path.join(frame_folder, "points.npy")
                np.save(points_path, points)
            except Exception as e:
                print("Error saving lidar points:", e)
            
            # --- Save Pose ---
            try:
                pose_path = os.path.join(frame_folder, "pose.pt")
                torch.save(pose, pose_path)
            except Exception as e:
                print("Error saving pose:", e)
            
            print(f"Saved synchronized data for stamp {stamp} as frame {index}.")
            index += 1
        else:
            time.sleep(0.01)


# ----------------------------
# Bridge Client Class
# ----------------------------
class HISLAM2BridgeClient:
    def __init__(self, ros_host='localhost', ros_port=9090, mode='online', output_dir=None, retries=3, retry_delay=1):
        self.host = ros_host
        self.port = ros_port
        self.mode = mode
        self.output_dir = output_dir
        self.ros = roslibpy.Ros(host=ros_host, port=ros_port)
        self.data_store = HISLAM2Data()
        self.retries = retries
        self.retry_delay = retry_delay

    def connect(self):
        for attempt in range(1, self.retries + 1):
            try:
                print(f"Attempt {attempt} to connect to rosbridge at {self.host}:{self.port}...")
                self.ros.run()
                print("Connected successfully to rosbridge.")
                return
            except roslibpy.core.RosTimeoutError:
                print(f"Connection attempt {attempt} failed. Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        raise roslibpy.core.RosTimeoutError("Failed to connect to rosbridge after multiple attempts.")

    def disconnect(self):
        self.ros.terminate()

    def subscribe_to_topics(self):
        """Subscribe to the synced FAST-LIVO2 image, pose, and LiDAR topics."""
        self.image_topic = roslibpy.Topic(self.ros, '/rgb_img', 'sensor_msgs/Image')
        self.image_topic.subscribe(lambda msg: image_queue.put(msg))

        self.pose_topic = roslibpy.Topic(self.ros, '/aft_mapped_to_init', 'nav_msgs/Odometry')
        self.pose_topic.subscribe(lambda msg: pose_queue.put(msg))

        self.lidar_topic = roslibpy.Topic(self.ros, '/cloud_registered', 'sensor_msgs/PointCloud2')
        self.lidar_topic.subscribe(lambda msg: lidar_queue.put(msg))

    def run(self):
        self.connect()
        self.subscribe_to_topics()
        # Start worker threads for processing each queue
        threading.Thread(target=process_image_queue, args=(self.data_store,), daemon=True).start()
        threading.Thread(target=process_pose_queue, args=(self.data_store,), daemon=True).start()
        threading.Thread(target=process_lidar_queue, args=(self.data_store,), daemon=True).start()

        # In preprocess mode, save data to HI-SLAM2/data/<output_name>
        if self.mode == 'preprocess':
            if self.output_dir is None:
                raise ValueError("Output folder name must be specified in preprocess mode.")
            # Create the output folder under HI-SLAM2/data/<output_name>
            out_folder = os.path.join("data", self.output_dir)
            os.makedirs(out_folder, exist_ok=True)
            threading.Thread(target=preprocess_data, args=(self.data_store, out_folder), daemon=True).start()

        print("HI-SLAM2 Bridge Client running and processing incoming data...")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Bridge client interrupted by user. Shutting down...")
        finally:
            self.disconnect()

# ----------------------------
# Main Entry Point
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HI-SLAM2 Bridge Client")
    parser.add_argument('--mode', choices=['online', 'preprocess'], default='online',
                        help="Operation mode: 'online' for real-time mapping, 'preprocess' to save data to disk.")
    parser.add_argument('--ros-host', default='localhost', help="ROS host address")
    parser.add_argument('--ros-port', type=int, default=9090, help="ROS port number, default 9090")
    parser.add_argument('--output-name', default='ros_preprocessed',
                        help="Specify folder name for preprocessed data output, stored in HI-SLAM2/data/<folder_name>")
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    help="Set the logging level (default: INFO)")
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    logging.basicConfig(level=numeric_level, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)

    client = HISLAM2BridgeClient(ros_host=args.ros_host,
                                 ros_port=args.ros_port,
                                 mode=args.mode,
                                 output_dir=args.output_name)
    client.run()

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
image_queue = queue.Queue()
pose_queue = queue.Queue()
lidar_queue = queue.Queue()
sync_queue = queue.Queue()

# ----------------------------
# Helper Functions for Message Decoding
# ----------------------------
def decode_ros_image(msg):
    """
    Decode a ROS Image message from base64 into an RGB NumPy array.
    Handles raw image data for common encodings such as 'rgb8', 'bgr8', or 'mono8'.
    If the encoding is not one of these, falls back to cv2.imdecode.
    """
    try:
        encoding = msg.get('encoding', '').lower()
        height = int(msg.get('height', 0))
        width = int(msg.get('width', 0))
        if height == 0 or width == 0:
            raise ValueError("Invalid image dimensions")
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
            np_arr = np.frombuffer(img_data, dtype=np.uint8) # Not directly writable
            np_arr_copy = np_arr.copy()
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
    Decodes FAST-LIVO2 PointCloudXYZRGB (sensor_msgs/PointCloud2) with:
      - point_step = 32 bytes
      - fields: x, y, z, intensity, rgb (packed as float), plus 12 bytes padding
      - possibly multi-row (height>1), each row has row_step bytes
    Returns:
      A NumPy array of shape (N,6) -> columns: [x,y,z,r,g,b].
    """
    try:
        raw_data = msg.get('data', '')
        if isinstance(raw_data, dict):
            raw_data = raw_data.get('data', '')
        
        if isinstance(raw_data, str):
            data_bytes = base64.b64decode(raw_data)
            # logger.debug("Decoded pointcloud data from base64 string.")
        elif isinstance(raw_data, (bytes, bytearray)):
            data_bytes = raw_data
        else:
            # If it's a list of floats, convert to bytes.
            if isinstance(raw_data, list):
                data_array = np.array(raw_data, dtype=np.float32)
                data_bytes = data_array.tobytes()
            else:
                raise ValueError(f"Unexpected type for pointcloud data: {type(raw_data)}")
        
        # Parse metadata
        height = msg.get('height', 1)
        width = msg.get('width', 0)
        point_step = msg.get('point_step', 32)
        row_step = msg.get('row_step', point_step * width)
        is_bigendian = msg.get('is_bigendian', False)
        
        if width == 0:
            width = len(data_bytes) // point_step
        
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
        
        # Extract x,y,z.
        xyz = np.stack((points['x'], points['y'], points['z']), axis=1)
        
        # Unpack rgb
        rgb_float = points['rgb']
        rgb_uint = rgb_float.view(np.uint32)
        # Extract channels (assuming 0x00RRGGBB).
        r = ((rgb_uint >> 16) & 0xFF).astype(np.float32)
        g = ((rgb_uint >> 8) & 0xFF).astype(np.float32)
        b = (rgb_uint & 0xFF).astype(np.float32)
        
        # xyz and rgb => shape (N,6).
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

    def store_camera(self, stamp, img_tensor):
        with self.lock:
            self.camera_data[stamp] = img_tensor

    def store_pose(self, stamp, pose_tensor):
        with self.lock:
            self.pose_data[stamp] = pose_tensor

    def store_lidar(self, stamp, points):
        with self.lock:
            self.lidar_data[stamp] = points

    def pop_synced_data(self, tolerance = None, wait_for_sensors=1):
        """
        For the earliest camera frame, wait up to `wait_for_sensors` seconds for both a 
        pose and a LiDAR frame whose timestamps are within `tolerance` of the camera timestamp.
        
        Once a synchronized package is found, pop the corresponding camera, pose, and LiDAR 
        entries from their buffers and prune any older entries (with timestamps earlier than 
        the camera frame). If, after waiting, either sensor still does not have a match within 
        tolerance, discard the camera frame.
        
        Returns:
        (sync_time, camera_tensor, pose_tensor, lidar_tensor) if a match is found,
        or None if no match is found.
        """
        if tolerance is None:
            raise ValueError("pop_synced_data() requires tolerance.")
        # Get the earliest camera frame as anchor.
        with self.lock:
            if not (self.camera_data and self.pose_data and self.lidar_data):
                # logger.debug("One or more sensor dictionaries are empty.")
                return None
            cam_keys = sorted(self.camera_data.keys())
            cam_time = cam_keys[0]
        
        start_wait = time.time()
        best_pose = None
        best_lidar = None
        diff_pose = float('inf')
        diff_lidar = float('inf')
        sync_errors = [] # [pose_error, lidar_error] w.r.t cam_time (anchor)
        
        # Wait for both pose and LiDAR to be within tolerance.
        while time.time() - start_wait < wait_for_sensors:
            with self.lock:
                if self.pose_data:
                    pose_keys = sorted(self.pose_data.keys())
                    best_pose = min(pose_keys, key=lambda t: abs(t - cam_time))
                    diff_pose = abs(cam_time - best_pose)
                if self.lidar_data:
                    lidar_keys = sorted(self.lidar_data.keys())
                    best_lidar = min(lidar_keys, key=lambda t: abs(t - cam_time))
                    diff_lidar = abs(cam_time - best_lidar)
            if diff_pose <= tolerance and diff_lidar <= tolerance:
                break
            time.sleep(0.005)  # Wait before checking again
        
        if diff_pose > tolerance or diff_lidar > tolerance:
            logger.debug(f"[SYNC] Discarding camera frame at {cam_time:.9f}: pose diff = {diff_pose:.9f}, lidar diff = {diff_lidar:.9f}")
            with self.lock:
                if cam_time in self.camera_data:
                    del self.camera_data[cam_time]
            return None
        
        # Both within tolerance, pop synchronized data.
        with self.lock:
            # Re-read in case new data arrived.
            pose_keys = sorted(self.pose_data.keys())
            best_pose = min(pose_keys, key=lambda t: abs(t - cam_time))
            # best_pose = min(pose_keys, key=lambda t: abs(t - best_lidar)) # align pose to lidar
            lidar_keys = sorted(self.lidar_data.keys())
            best_lidar = min(lidar_keys, key=lambda t: abs(t - cam_time))
            
            cam = self.camera_data.pop(cam_time)
            pose = self.pose_data.pop(best_pose)
            lidar = self.lidar_data.pop(best_lidar)
            # Prune old pose and lidar entries
            for key in list(self.pose_data.keys()):
                if key < cam_time:
                    del self.pose_data[key]
            for key in list(self.lidar_data.keys()):
                if key < cam_time:
                    del self.lidar_data[key]
        
        pose_error = abs(cam_time - best_pose)
        lidar_error = abs(cam_time - best_lidar)
        sync_time = cam_time  # Camera timestamp as anchor
        sync_errors = [pose_error, lidar_error]
        logger.info(f"[SYNC] Synchronized data: camera {cam_time:.9f}, pose {best_pose:.9f}, "
                    f"lidar {best_lidar:.9f}; sync_time = {sync_time:.9f}")
        return sync_time, cam, pose, lidar, sync_errors


    # (Placeholder) Optional get functions for online mode
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
# Sync Sensor Streams
# ----------------------------
def sync_thread(data_store, tolerance=0.05, wait=2):
    """
    Continuously attempt to synchronize incoming sensor data.
    If a synchronized package is found, put it into the sync_queue.
    """
    while True:
        synced = data_store.pop_synced_data(tolerance=tolerance, wait_for_sensors=wait)
        if synced is not None:
            sync_queue.put(synced)
        else:
            time.sleep(0.005)

# ----------------------------
# Preprocess and Save (offline)
# ----------------------------
def saving_thread(output_dir):
    """
    Continuously get synchronized data packages from the sync_queue and save them.
    Each frame folder (HI-SLAM2/data/folder/frameX) contains:
      - image.png:  RGB image (C, H, W)
      - points.npy: lidar points in world frame (shape: [N,6] -> columns: [x, y, z, r, g, b])
      - pose.pt:    pose torch tensor [tx, ty, tz, qx, qy, qz, qw]
    """
    index = 0
    sync_errors_file = os.path.join(output_dir, "sync_errors.csv")
    if not os.path.exists(sync_errors_file):
        with open(sync_errors_file, 'w') as f:
            f.write("frame_index,camera_timestamp,error_pose,error_lidar\n")
    
    while True:
        try:
            # Wait for a synchronized package; timeout if none.
            synced = sync_queue.get(timeout=1)
        except queue.Empty:
            continue

        stamp, cam, pose, lidar, sync_errors = synced
        error_pose, error_lidar = sync_errors

        # Create a folder for this frame.
        frame_folder = os.path.join(output_dir, f"frame{index}")
        os.makedirs(frame_folder, exist_ok=True)
        
        try:
            # Cam is torch tensor in (C, H, W) format in RGB.
            img_np = cam.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(frame_folder, "image.png"), img_bgr)
        except Exception as e:
            print("Error saving image:", e)
        
        try:
            points = lidar.cpu().numpy()
            np.save(os.path.join(frame_folder, "points.npy"), points)
        except Exception as e:
            print("Error saving lidar points:", e)
        
        try:
            torch.save(pose, os.path.join(frame_folder, "pose.pt"))
        except Exception as e:
            print("Error saving pose:", e)

        try:
            with open(sync_errors_file, 'a') as f:
                # frame_index, camera_timestamp, error_pose, error_lidar
                f.write(f"{index},{stamp},{error_pose},{error_lidar}\n")
        except Exception as e:
            print("Error writing sync errors:", e)
        
        print(f"[SAVE] Saved synchronized data for stamp {stamp} as frame {index}.")
        index += 1
        sync_queue.task_done()

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
        """
        Connects to rosbridge, subscribes to sensor topics, and starts the
        sensor processing, synchronization, and (if in preprocess mode) saving threads.
        """
        self.connect()
        self.subscribe_to_topics()

        threading.Thread(target=process_image_queue, args=(self.data_store,), daemon=True).start()
        threading.Thread(target=process_pose_queue, args=(self.data_store,), daemon=True).start()
        threading.Thread(target=process_lidar_queue, args=(self.data_store,), daemon=True).start()
        threading.Thread(target=sync_thread, args=(self.data_store,), daemon=True).start()

        if self.mode == 'preprocess':
            out_folder = os.path.join("data", self.output_dir)
            os.makedirs(out_folder, exist_ok=True)
            threading.Thread(target=saving_thread, args=(out_folder,), daemon=True).start()

        print("HI-SLAM2 Bridge Client running, waiting for incoming data...")
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

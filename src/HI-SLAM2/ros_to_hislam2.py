#!/usr/bin/env python3
"""
hi_slam2_service.py

This script runs entirely in your Conda environment and uses roslibpy to connect
to a rosbridge server (running on your ROS environment). It subscribes to sensor topics,
decodes the incoming messages into efficient data structures (using NumPy and PyTorch),
and places them into a thread‑safe shared data store for downstream real‑time processing.
"""

import roslibpy
import torch
import cv2
import numpy as np
import base64
import time
import threading
import queue

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
        print(f"Decoding image: encoding={encoding}, height={height}, width={width}")
        if height == 0 or width == 0:
            raise ValueError("Invalid image dimensions")

        # Decode the raw data from base64.
        img_data = base64.b64decode(msg.get('data', ''))
        # Get the decoded data length.
        decoded_length = len(img_data)
        print(f"Decoded data length: {decoded_length} bytes")

        # If the encoding is one of the raw formats, compute expected length.
        if encoding in ['rgb8', 'bgr8', 'mono8']:
            channels = 3 if encoding in ['rgb8', 'bgr8'] else 1
            expected_length = height * width * channels
            print(f"Expected data length for raw image: {expected_length} bytes")
            if decoded_length != expected_length:
                raise ValueError(f"Decoded data length ({decoded_length}) does not match expected ({expected_length}).")
            # Create a NumPy array from the raw data.
            np_arr = np.frombuffer(img_data, dtype=np.uint8)
            # Reshape into the proper dimensions.
            img = np_arr.reshape((height, width, channels))
            # If the encoding is 'bgr8', convert to RGB.
            if encoding == 'bgr8':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # If 'mono8', optionally convert to 3-channel RGB.
            if encoding == 'mono8':
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return img
        else:
            # If the encoding is not one of the raw types, assume it's compressed.
            # Note: imdecode requires a writable array; if needed, use .copy() here.
            np_arr = np.frombuffer(img_data, dtype=np.uint8)
            np_arr_copy = np_arr.copy()  # Ensure the array is writable.
            img = cv2.imdecode(np_arr_copy, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image data with imdecode")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print("Image decode error:", e)
        return None


def decode_ros_pointcloud2(msg):
    """
    Decode a simplified ROS PointCloud2 message.
    This example assumes the message 'data' field is a flat list of floats (x,y,z).
    In a real system, you’d parse the full PointCloud2 structure.
    """
    try:
        data = msg.get('data', [])
        pc_array = np.array(data, dtype=np.float32).reshape(-1, 3)
        return pc_array
    except Exception as e:
        print("Point cloud decode error:", e)
        return None

# ----------------------------
# Worker Functions
# ----------------------------
def process_image_queue(data_store):
    while True:
        msg = image_queue.get()  # Blocks until an item is available
        img = decode_ros_image(msg)
        if img is not None:
            # Zero-copy conversion: torch.from_numpy shares memory with the numpy array
            tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            # Use header stamp if available; otherwise, fallback to current time.
            stamp = msg.get('header', {}).get('stamp', {}).get('secs', time.time())
            data_store.store_camera(stamp, tensor)
        image_queue.task_done()

def process_pose_queue(data_store):
    while True:
        msg = pose_queue.get()
        try:
            stamp = msg.get('header', {}).get('stamp', {}).get('secs', time.time())
            pos = msg.get('pose', {}).get('pose', {}).get('position', {})
            tx = pos.get('x', 0.0)
            ty = pos.get('y', 0.0)
            tz = pos.get('z', 0.0)
            pose_mat = np.eye(4, dtype=np.float32)
            pose_mat[0, 3] = tx
            pose_mat[1, 3] = ty
            pose_mat[2, 3] = tz
            data_store.store_pose(stamp, pose_mat)
        except Exception as e:
            print("Pose processing error:", e)
        pose_queue.task_done()

def process_lidar_queue(data_store):
    while True:
        msg = lidar_queue.get()
        pc_array = decode_ros_pointcloud2(msg)
        if pc_array is not None:
            pc_tensor = torch.from_numpy(pc_array).float()
            stamp = msg.get('header', {}).get('stamp', {}).get('secs', time.time())
            data_store.store_lidar(stamp, pc_tensor)
        lidar_queue.task_done()

# ----------------------------
# Data Storage Class
# ----------------------------
class HISLAM2Data:
    """
    Thread-safe container to store incoming sensor data.
    """
    def __init__(self):
        self.camera_data = {}  # {timestamp: torch.Tensor}
        self.pose_data = {}    # {timestamp: np.ndarray (4x4)}
        self.lidar_data = {}   # {timestamp: torch.Tensor}
        self.lock = threading.Lock()

    def store_camera(self, stamp, tensor):
        with self.lock:
            self.camera_data[stamp] = tensor

    def store_pose(self, stamp, pose_mat):
        with self.lock:
            self.pose_data[stamp] = pose_mat

    def store_lidar(self, stamp, points):
        with self.lock:
            self.lidar_data[stamp] = points

    # (Optional) retrieval methods for downstream processing:
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
# Bridge Client Class
# ----------------------------
class HISLAM2BridgeClient:
    def __init__(self, ros_host='localhost', ros_port=9090, retries=3, retry_delay=1):
        self.host = ros_host         # Store host
        self.port = ros_port         # Store port
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
        # Subscribe to the image topic.
        self.image_topic = roslibpy.Topic(self.ros, '/left_camera/image', 'sensor_msgs/Image')
        self.image_topic.subscribe(lambda msg: image_queue.put(msg))
        # Subscribe to the odometry (pose) topic.
        self.pose_topic = roslibpy.Topic(self.ros, '/aft_mapped_to_init', 'nav_msgs/Odometry')
        self.pose_topic.subscribe(lambda msg: pose_queue.put(msg))
        # Subscribe to the point cloud topic.
        self.lidar_topic = roslibpy.Topic(self.ros, '/cloud_registered', 'sensor_msgs/PointCloud2')
        self.lidar_topic.subscribe(lambda msg: lidar_queue.put(msg))

    def run(self):
        self.connect()
        self.subscribe_to_topics()
        # Start worker threads for processing each queue.
        threading.Thread(target=process_image_queue, args=(self.data_store,), daemon=True).start()
        threading.Thread(target=process_pose_queue, args=(self.data_store,), daemon=True).start()
        threading.Thread(target=process_lidar_queue, args=(self.data_store,), daemon=True).start()
        print("HI-SLAM2 Bridge Client is now running and processing incoming data...")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Bridge client interrupted by user. Shutting down...")
        finally:
            self.disconnect()

if __name__ == '__main__':
    # If needed, replace 'localhost' with the explicit IP address of your ROS host.
    client = HISLAM2BridgeClient(ros_host='localhost', ros_port=9091)
    client.run()

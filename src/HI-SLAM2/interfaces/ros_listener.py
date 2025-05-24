#!/usr/bin/env python3
import rospy
import torch
import numpy as np
import cv2
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
import ros_numpy

class HISLAM2Data:
    """Holds camera, pose, lidar data for integration."""
    def __init__(self):
        self.camera_data = {}
        self.pose_data = {}
        self.lidar_data = {}

    def store_camera(self, stamp, tensor):
        self.camera_data[stamp] = tensor

    def store_pose(self, stamp, pose_mat):
        self.pose_data[stamp] = pose_mat

    def store_lidar(self, stamp, points):
        self.lidar_data[stamp] = points

class HISLAM2Node:
    def __init__(self):
        rospy.init_node("hi_slam2_node", anonymous=True)
        self.data_store = HISLAM2Data()
        rospy.Subscriber("/left_camera/image", Image, self.image_callback)
        # rospy.Subscriber("/rgb_img", Image, self.image_callback)
        rospy.Subscriber("/aft_mapped_to_init", Odometry, self.pose_callback)
        rospy.Subscriber("/cloud_registered", PointCloud2, self.lidar_callback)

    def image_callback(self, msg):
        try:
            rospy.loginfo("Image callback triggered")
            img_bgr = ros_numpy.numpify(msg)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
            stamp = msg.header.stamp.to_sec()
            self.data_store.store_camera(stamp, tensor)
            rospy.logdebug("Received image at stamp=%.6f, size=(%d, %d)", 
                        stamp, img_rgb.shape[0], img_rgb.shape[1])
        except Exception as e:
            rospy.logerr("image_callback error: %s", str(e))

    def pose_callback(self, msg):
        try:
            rospy.loginfo("Pose callback triggered")
            tx = msg.pose.pose.position.x
            ty = msg.pose.pose.position.y
            tz = msg.pose.pose.position.z
            rx = msg.pose.pose.orientation.x
            ry = msg.pose.pose.orientation.y
            rz = msg.pose.pose.orientation.z
            rw = msg.pose.pose.orientation.w
            pose_mat = np.eye(4, dtype=np.float32)
            pose_mat[0, 3] = tx
            pose_mat[1, 3] = ty
            pose_mat[2, 3] = tz
            stamp = msg.header.stamp.to_sec()
            self.data_store.store_pose(stamp, pose_mat)
            rospy.logdebug("Received pose at stamp=%.6f, translation=(%.3f, %.3f, %.3f)", 
                           stamp, tx, ty, tz)
        except Exception as e:
            rospy.logerr("pose_callback error: %s", str(e))

    def lidar_callback(self, msg):
        try:
            rospy.loginfo("Lidar callback triggered")
            pc_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
            pc_tensor = torch.from_numpy(pc_array).float()
            stamp = msg.header.stamp.to_sec()
            self.data_store.store_lidar(stamp, pc_tensor)
            rospy.logdebug("Received lidar at stamp=%.6f, points=%d", 
                           stamp, pc_array.shape[0])
        except Exception as e:
            rospy.logerr("lidar_callback error: %s", str(e))

    def run(self):
        rospy.loginfo("HI-SLAM2 ros_listener.py node is now spinning...")
        rospy.spin()

if __name__ == "__main__":
    node = HISLAM2Node()
    node.run()
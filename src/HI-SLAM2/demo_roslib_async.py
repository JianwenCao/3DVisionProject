import time
from hi2 import Hi2  # Your HI-SLAM2 processing library

def realtime_processing_loop(data_store, hi2_instance):
    """
    Continuously poll the shared data store for the latest sensor data
    and pass it to the HI-SLAM2 tracking pipeline.
    """
    last_processed_stamp = 0
    while True:
        # Retrieve the latest camera image and pose.
        # You may also retrieve lidar data if required.
        image = data_store.get_latest_camera()
        pose = data_store.get_latest_pose()
        
        # Check if new data is available.
        # (You might also compare timestamps or use a dedicated queue.)
        if image is not None and pose is not None:
            # For example, use the timestamp from the image data.
            current_stamp = max(data_store.camera_data.keys())
            if current_stamp > last_processed_stamp:
                # Call your tracking function with the latest data.
                hi2_instance.track(frame_index=current_stamp, image=image, intrinsics=None, is_last=False)
                last_processed_stamp = current_stamp
                print(f"Processed frame at stamp: {current_stamp}")
        time.sleep(0.05)  # Adjust the sleep time based on your processing speed

if __name__ == '__main__':
    # First, start the bridge client in a separate thread.
    from threading import Thread

    # Create the bridge client to subscribe and decode ROS topics.
    client = HISLAM2BridgeClient(ros_host='localhost', ros_port=9090)
    # Run the bridge client in a background thread.
    bridge_thread = Thread(target=client.run, daemon=True)
    bridge_thread.start()

    # Initialize your HI-SLAM2 processing instance.
    hi2 = Hi2()  # Initialize with any required configuration.
    
    # Start the real-time processing loop.
    realtime_processing_loop(client.data_store, hi2)

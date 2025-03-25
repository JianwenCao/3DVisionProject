# 3D Vision Project - ETH 2025

## Prerequisites
### Install Docker Engine
> Warning! Docker Desktop not recommended, you will run into issues running GUI apps in container and visualizing on host.
- Ensure [Docker Engine](https://docs.docker.com/engine/install/) (CLI version) is installed.


## Dockerized Environment
- ROS Noetic Base: osrf/ros:noetic-desktop-full (Ubuntu 20.04).
- Preinstalled Dependencies: ROS libraries, OpenCV, Eigen, PCL, Ceres, and Python tools.
- Non-root User: devuser
- Livox SDK
- Sophus (Patched): Applies a custom patch and builds the non-templated/double-only version for rpg_vikit.

## Installation Instructions

### 1. Open the Project Directory
On your **host machine**, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/JianwenCao/3DVisionProject.git
cd 3DVisionProject/
```

### 2. Download ROS Bags
Move your downloaded `.bag` files into the following directory **on your host machine**:

```bash
~/dataset_fastlivo2/
```

> **Note:** This folder will be **mounted** and accessible inside the container.

### 3. Open the DevContainer
- Reopen `3DVisionProject/` in **DevContainer**. Select **With GPU** or **No GPU**. If this is a new, unbuilt container, the `devcontainer.json + Dockerfile` pair will be built.

### 4. Initialize Catkin Workspace
Then, initialize `catkin`:

```bash
catkin init
```

### 5. Build All Packages
```bash
catkin build
```

---

### 6. Test **FAST-LIVO2** 
**Launch**
```bash
# Terminal 1
roscore

# Terminal 2
. devel/setup.bash
roslaunch fast_livo mapping_avia.launch

# Terminal 3
rosbag play ~/dataset_fastlivo2/YOUR_FILE.bag
```


## HI-SLAM2 in ROS environment:
- Build `HISLAM2 With GPU` devcontainer
- Postcreatecommand runs `setup.py` automatically, installing CUDA extensions.
> Warning: the postcreatecommand takes a long time, if you "rebuild" container instead of "reopen", this long step will take a while.
### Activate HI-SLAM2 Conda Environment:
>Same as HI-SLAM2 instructions
```bash
act_hi2
cd src/HI-SLAM2
wget https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt -P pretrained_models
wget https://zenodo.org/records/10447888/files/omnidata_dpt_depth_v2.ckpt -P pretrained_models
bash scripts/download_replica.sh
python scripts/preprocess_replica.py
```
Run demo:
```bash
python demo.py \
--imagedir data/Replica/room0/colors \
--calib calib/replica.txt \
--config config/replica_config.yaml \
--output outputs/room0 \
--gsvis \
--droidvis
```

### Convert FAST-LIVO2 ROS messages to HI-SLAM2 data structure (with bridge)
Container publishes ROS message on localhost and forwards websocket to 9090 (example). HISLAM2BridgeClient listens to websocket, parses, and stores data in torch tensors.
```bash
# Terminal 1
roslaunch rosbridge_server rosbridge_websocket.launch port:=9090

# Terminal 2: Decode ROS msgs and sync sensor data. Optionally save to folder.
act_hi2
cd /catkin_ws/src/HI-SLAM2
python scripts/run_ros_conversions.py 
            [-h] [--mode {online,preprocess}]   # Online for real-time pipeline, preprocess (save frames to folder)
            [--ros-host ROS_HOST]               # default 'localhost'
            [--ros-port ROS_PORT]               # default 9090
            [--output-name OUTPUT_NAME]         # folder NAME, data saved in Hi_SLAM2/data/<output-name>
            [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

# Terminal 3:
rosbag play path/to/dataset.bag
```

### Test Output of Offline Preprocessing
```bash
act_hi2
cd /catkin_ws/src/HI-SLAM2
python3 scripts/test_ros_conversions.py --mode {image, lidar, pose} --folder <folder name inside HI-SLAM2/data/>
```
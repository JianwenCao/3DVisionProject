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
Once inside the container, open a new terminal (**bash**) and ensure you are in:

```bash
devuser@docker-desktop:/catkin_ws$
```

Then, initialize `catkin`:

```bash
catkin init
```

### 5. Build All Packages

```bash
cd /catkin_ws && . /opt/ros/noetic/setup.bash
catkin build
```

---

### 6. Test **FAST-LIVO2** 

**Launch**
```bash
# Terminal 1
. /opt/ros/noetic/setup.bash
roscore

# Terminal 2
. /devel/setup.bash
roslaunch fast_livo mapping_avia.launch

# Terminal 3
rosbag play ~/dataset_fastlivo2/YOUR_FILE.bag
```

--- 
Testing MacOS commands:
```bash
sudo netstat -tulpn | grep 590
tigervncserver -kill :1
tigervncserver :1 -geometry 1280x800 -depth 24 -xstartup /usr/bin/xterm
```

Mac host:
brew install --cask tigervnc-viewer

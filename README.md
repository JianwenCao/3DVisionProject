# 3D Vision Project - ETH 2025

## Installation Instructions

### 1. Open the Project Directory
On your **host machine**, navigate to the project directory:

```bash
cd 3DVisionProject/
```

### 2. Download ROS Bags
Move your downloaded `.bag` files into the following directory **on your host machine**:

```bash
~/dataset_fastlivo2/
```

> **Note:** This folder will be **mounted** and accessible inside the container.

### 3. Open the DevContainer
- Reopen `3DVisionProject/` in the **DevContainer** (this will trigger the `.devcontainer/` Dockerfile build process).

### 4. Initialize Catkin Workspace
Once inside the container, open a new terminal (**bash**) and ensure you are in:

```bash
devuser@docker-desktop:/catkin_ws$
```

Then, initialize `catkin`:

```bash
catkin init
```

---

### 5. Initialize the submodules
Github has not resolved the various packages we are using (FAST-LIVO2, livox_ros_dirver, etc.).
```bash
git submodule update --init --recursive
```

### 6. Pre-Build **Sophus** Inside the Container

> **Why?**  
> If you **skip** this step, `catkin build` will fail to find `Sophus` when building `rpg_vikit`.
Both Sophus and rpg_vikit are NOT git submodules. They are baked into our project in a way that they will compile. Run the following inside the container:

```bash
cd src/Sophus
mkdir build && cd build
cmake ..
make
sudo make install
```

---

### 7. Build **FAST-LIVO2**

```bash
cd /catkin_ws && . /opt/ros/noetic/setup.bash
catkin build
```

### 8. Test **FAST-LIVO2** 

> **Warning: There is a known issue where RViz may fail to launch.  
> Still trying to debug and need help!**

**Launch**
```bash
. /devel/setup.bash
roslaunch fast_livo mapping_avia.launch
rosbag play PATH/TO/YOUR_DOWNLOADED.bag
```

**Error Output:**
```bash
devuser@docker-desktop:/catkin_ws$ rviz
qt.qpa.xcb: could not connect to display :0
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.


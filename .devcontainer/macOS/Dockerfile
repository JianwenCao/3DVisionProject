# Base Image: ROS Noetic (ARM64)
FROM arm64v8/ros:noetic-ros-core

# Set non-interactive mode for apt
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND}

# Set the display environment variable for VNC
ENV DISPLAY=:1

# Create matching non-root user
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update && apt-get install -y \
    sudo \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && rm -rf /var/lib/apt/lists/*

# Install ROS, system dependencies, VNC, and GUI packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    zip \
    unzip \
    tar \
    apt-transport-https \
    gnupg \
    lsb-release \
    python3-catkin-tools \
    libopencv-dev \
    libeigen3-dev \
    libpcl-dev \
    libceres-dev \
    ros-noetic-rviz \
    ros-noetic-pcl-ros \
    # VNC / GUI
    tigervnc-standalone-server \
    gnome-flashback \
    gnome-terminal \
    gnome-session \
    gnome-control-center \
    gnome-panel \
    gnome-settings-daemon \
    metacity \
    nautilus \
    terminator \
    dbus-x11 \
    x11-utils \
    x11-xserver-utils \
    xfce4 \
    xterm \
    net-tools \
    python3-websockify \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/noetic/setup.bash" >> /etc/bash.bashrc

# Clone noVNC into /opt/novnc
RUN git clone https://github.com/novnc/noVNC.git /opt/novnc

# -------------------------------
# Build Livox SDK from Source
# -------------------------------
WORKDIR /opt
RUN git clone https://github.com/Livox-SDK/Livox-SDK.git && \
    cd Livox-SDK && \
    mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    sudo make install

# -------------------------------
# Build and install Sophus for rpg_vikit
# -------------------------------
USER root
WORKDIR /tmp
COPY thirdparty/sophus.patch /tmp/
RUN git clone https://github.com/strasdat/Sophus.git && \
    cd Sophus && \
    git checkout a621ff && \
    git apply /tmp/sophus.patch && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig
USER devuser

# Expose ports for noVNC and TigerVNC
EXPOSE 8080 5901

# Copy and set permissions for entrypoint script
COPY --chmod=755 .devcontainer/macOS/entrypoint.sh /entrypoint.sh

# Run entrypoint script when the container starts
CMD ["/bin/bash", "-c", "/entrypoint.sh"]

# Use official ROS Noetic base image (Ubuntu 20.04)
# FROM osrf/ros:noetic-desktop-full
FROM --platform=linux/amd64 osrf/ros:noetic-desktop-full
# Set noninteractive mode
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND}

# Install System Dependencies
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
    ros-noetic-pcl-ros \
    ros-noetic-sophus \
    libx11-xcb1 \
    libxcb-xinerama0 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    mesa-utils \
    libgl1-mesa-dev \
    libglapi-mesa \
    mesa-vdpau-drivers \
    sudo && \
    rm -rf /var/lib/apt/lists/*

# Create a Non-root User matching your host (optional but recommended)
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=1000
RUN apt-get update && \
    apt-get install -y sudo && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Port forwading
# RUN apt-get update && apt-get install -y x11-xserver-utils xauth libxcb-xinerama0 libx11-xcb1 libglu1-mesa

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
# (non-templated/double-only version)
# Build with patch
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

# Switch to devuser by default
USER $USERNAME
WORKDIR /home/$USERNAME

CMD ["/bin/bash"]

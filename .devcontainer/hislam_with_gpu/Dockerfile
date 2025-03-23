# Use official ROS Noetic base image (Ubuntu 20.04)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
# FROM pandaman666/ubuntu:ros-noetic-runtime-ubuntu20.04-cuda11.8.0

# Set noninteractive mode
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND}

# -------------------------------
# Setup ROS Repository
# -------------------------------
RUN apt-get update && apt-get install -y \
    lsb-release \
    gnupg2 \
    curl && \
    curl -sSL "http://packages.ros.org/ros.key" | apt-key add - && \
    echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list && \
    apt-get update

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
    ros-noetic-desktop-full \
    python3-rosdep \
    python3-catkin-tools \
    libopencv-dev \
    libeigen3-dev \
    libpcl-dev \
    libceres-dev \
    ros-noetic-pcl-ros \
    ros-noetic-sophus \
    ros-noetic-rosbridge-suite \
    python3-pip \
    sudo && \
    rm -rf /var/lib/apt/lists/*

RUN rosdep init && rosdep update

# Install additional Python dependencies required for rosbridge
RUN /usr/bin/python3 -m pip install --no-cache-dir \
    twisted \
    pyOpenSSL \
    autobahn \
    tornado \
    pymongo \
    service-identity


# Create a Non-root User matching your host (optional but recommended)
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=1000
RUN apt-get update && \
    apt-get install -y sudo && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

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

# -------------------------------
# Install Miniconda and Set Up HI-SLAM2 Environment
# -------------------------------
# Download and install Miniconda for the devuser
WORKDIR /home/$USERNAME
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /home/$USERNAME/miniconda && \
    rm /tmp/miniconda.sh
# DO NOT PUT in default path

# -------------------------------
# Configure the Shell to Use System Python by Default
# -------------------------------
# Add to .bashrc to force system Python (/usr/bin) to take precedence.
RUN echo 'export PATH=/usr/bin:$PATH' >> /home/$USERNAME/.bashrc

# activate_conda to manually activate Conda when needed.
RUN echo 'if [ -f "/home/devuser/miniconda/etc/profile.d/conda.sh" ]; then' >> /home/$USERNAME/.bashrc && \
    echo '    alias activate_conda="source /home/devuser/miniconda/etc/profile.d/conda.sh && conda activate"' >> /home/$USERNAME/.bashrc && \
    echo 'fi' >> /home/$USERNAME/.bashrc

# Switch to devuser by default
USER $USERNAME
WORKDIR /home/$USERNAME
RUN echo "source /opt/ros/noetic/setup.bash" >> /home/$USERNAME/.bashrc

CMD ["/bin/bash"]

#!/bin/bash
set -e

# Change ownership of the workspace
sudo chown -R devuser:devuser /catkin_ws

# Append the alias to .bashrc
source /home/devuser/miniconda/etc/profile.d/conda.sh
echo "alias act_hi2='source /home/devuser/miniconda/bin/activate hislam2'" >> /home/devuser/.bashrc

# Change to the HI-SLAM2 folder and create the conda environment and run setup.py
cd /catkin_ws/src/
git clone --recursive https://github.com/Willyzw/HI-SLAM2 HI-SLAM2-original
cd HI-SLAM2-original

conda env create -f environment.yaml -vv
conda activate hislam2
pip install \
    PyOpenGL \
    PyOpenGL_accelerate \
    roslibpy \
    torch-cluster==1.6.3 \
    transformers==4.51.3
python setup.py install


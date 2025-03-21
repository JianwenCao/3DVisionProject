#!/bin/bash
set -e

# Change ownership of the workspace
sudo chown -R devuser:devuser /catkin_ws

# Append the alias to .bashrc
source /home/devuser/miniconda/etc/profile.d/conda.sh
echo "alias act_hi2='source /home/devuser/miniconda/bin/activate hislam2'" >> /home/devuser/.bashrc

# Change to the HI-SLAM2 folder and create the conda environment and run setup.py
cd /catkin_ws/src/HI-SLAM2
conda env create -f environment.yaml
conda run -n hislam2 python setup.py install


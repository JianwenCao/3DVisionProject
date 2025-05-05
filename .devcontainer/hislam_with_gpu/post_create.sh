#!/bin/bash
set -e

# Change ownership of the workspace
sudo chown -R devuser:devuser /catkin_ws

# Append the alias to .bashrc
source /home/devuser/miniconda/etc/profile.d/conda.sh
echo "alias act_hi2='source /home/devuser/miniconda/bin/activate hislam2'" >> /home/devuser/.bashrc

# Change to the HI-SLAM2 folder and create the conda environment and run setup.py
cd /catkin_ws/src/HI-SLAM2
rm -rf thirdparty/eigen
rm -rf thirdparty/lietorch
rm -rf thirdparty/simple-knn
rm -rf thirdparty/diff-gaussian-rasterization

git clone https://gitlab.com/libeigen/eigen.git thirdparty/eigen
git clone https://github.com/princeton-vl/lietorch.git thirdparty/lietorch
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git thirdparty/simple-knn
git clone -b main https://github.com/graphdeco-inria/diff-gaussian-rasterization.git thirdparty/diff-gaussian-rasterization
git clone https://github.com/g-truc/glm.git thirdparty/diff-gaussian-rasterization/third_party/glm
git submodule update --init --recursive

# conda env create -f environment.yaml -vv
conda activate hislam2
pip install -e thirdparty/simple-knn
pip install -e thirdparty/diff-gaussian-rasterization
python setup.py install


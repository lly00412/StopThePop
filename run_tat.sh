#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export DATA_DIR=/mnt/Data2/nerf_datasets/tandt_db/tandt/

python train.py -s $DATA_DIR/train --splatting_config configs/hierarchical.json -m output/tant/train --eval
#python render_w_Vcam.py -m output/m360/kitchen --render_depth
#python render_w_Vcam.py -m output/nerf_synthetic/ship2 --render_depth

# To view the results：
# 2D-GS-Viser-Viewer
# forward the output to a target port and do ssh mapping
# ssh -L 8080:127.0.0.1:8080 lchen39@155.246.81.47
# conda activate splatam
# python viewer.py /mnt/Data2/liyan/2d-gaussian-splatting/output/nerf_synthetic/ship/point_cloud/iteration_30000/point_cloud.ply -s /mnt/Data2/nerf_datasets/nerf_synthetic/ship/
# http://localhost:8080/
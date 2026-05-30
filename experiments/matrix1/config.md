# Experiment Configuration

## Hardware

GPU: NVIDIA GeForce RTX 4060 Laptop GPU  
GPU Memory: 8 GB  
Driver Version: 576.52  
CUDA Version: 12.9  
System: WSL Ubuntu

## Software

Environment: conda (ttt3r)

## Model

src/cut3r_512_dpt_4_64.pth

## Input

Video: /home/chen/Desktop/video.mp4  
Resolution: 512

## Command Template

python demo.py \
--model_path src/cut3r_512_dpt_4_64.pth \
--size 512 \
--seq_path /home/chen/Desktop/video.mp4 \
--output_dir ./experiments/matrix1/<experiment_name> \
--port 8080 \
--model_update_type <cut3r | ttt3r> \
--frame_interval <2 | 5> \
--reset_interval 100
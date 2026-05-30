# Experiment Matrix 1 – CUT3R vs TTT3R

Video: /home/chen/Desktop/video.mp4  
Resolution: 512  
Reset Interval: 100  

## Experiment Configuration

Model: src/cut3r_512_dpt_4_64.pth  
Input: video sequence  
GPU: CUDA (WSL environment)

## Results

| Method | Frame Interval | Reset Interval | FPS |
|------|------|------|------|
| CUT3R | 2 | 100 | 3.33 |
| CUT3R | 5 | 100 | 2.66 |
| TTT3R | 2 | 100 | 4.72 |
| TTT3R | 5 | 100 | 3.40 |

## Observation

TTT3R achieves consistently higher inference speed than CUT3R while producing visually similar reconstruction quality.

The fastest configuration is:

TTT3R + frame_interval=2 → 4.72 FPS
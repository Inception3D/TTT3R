#!/bin/bash

set -e

workdir='.'
model_names=('ttt3r') # ttt3r cut3r
beta_reduce_modes=('mean' 'entropy') # mean max variance entropy
ckpt_name='cut3r_512_dpt_4_64'
model_weights="${workdir}/src/${ckpt_name}.pth"

for model_name in "${model_names[@]}"; do
for beta_reduce_mode in "${beta_reduce_modes[@]}"; do

# for max_frames in 50 100 150 200 250 300 350 400
for max_frames in 200

do

    output_dir="${workdir}/eval_results/video_recon/7scenes_${max_frames}/${model_name}_${beta_reduce_mode}"
    echo "$output_dir"
    NCCL_TIMEOUT=360000 accelerate launch --num_processes 1 --main_process_port 29502 eval/mv_recon/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --model_name "$model_name" \
        --model_update_type "$model_name" \
        --beta_reduce_mode "$beta_reduce_mode" \
        --max_frames $max_frames

done
done
done

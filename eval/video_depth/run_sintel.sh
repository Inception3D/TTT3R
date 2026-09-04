#!/bin/bash

set -e

workdir='.'
model_names=('ttt3r') # ttt3r cut3r
beta_reduce_modes=('mean' 'entropy') # mean max variance entropy

ckpt_name='cut3r_512_dpt_4_64'
model_weights="${workdir}/src/${ckpt_name}.pth"
datasets=('sintel')

for model_name in "${model_names[@]}"; do
for beta_reduce_mode in "${beta_reduce_modes[@]}"; do
for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}/${model_name}_${beta_reduce_mode}"
    echo "$output_dir"

    accelerate launch --num_processes 1 --main_process_port 29555 eval/video_depth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512 \
        --model_update_type "$model_name" \
        --beta_reduce_mode "$beta_reduce_mode"

    # scale&shift scale metric
    python eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "metric"

    python eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale"

    python eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale&shift"
done
done
done

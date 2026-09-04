#!/bin/bash

set -e

workdir='.'
model_names=('ttt3r') # ttt3r cut3r
beta_reduce_modes=('mean' 'entropy') # mean max variance entropy

ckpt_name='cut3r_512_dpt_4_64'
model_weights="${workdir}/src/${ckpt_name}.pth"


# datasets=('tum_s1_50' 'tum_s1_100' 'tum_s1_150' 'tum_s1_200' 'tum_s1_300' 'tum_s1_400' 'tum_s1_500' 'tum_s1_600' 'tum_s1_700' 'tum_s1_800' 'tum_s1_900' 'tum_s1_1000')
datasets=('tum_s1_1000')

for model_name in "${model_names[@]}"; do
for beta_reduce_mode in "${beta_reduce_modes[@]}"; do
for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/relpose/${data}/${model_name}_${beta_reduce_mode}"
    echo "$output_dir"
    accelerate launch --num_processes 2 --main_process_port 29551 eval/relpose/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512 \
        --model_update_type "$model_name" \
        --beta_reduce_mode "$beta_reduce_mode"
done
done
done



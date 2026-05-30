#!/bin/bash
# Evaluate relpose on KITTI odometry (OOD generalization experiment)
# Configs: cut3r / ttt3r / aconst / brake / ortho
# Lengths: 200f (short) and 1000f (long)
#
# Usage:
#   bash eval/relpose/run_kitti_odo.sh
#   CUDA_VISIBLE_DEVICES=1 bash eval/relpose/run_kitti_odo.sh

set -e

workdir='.'
model_weights="${workdir}/src/cut3r_512_dpt_4_64.pth"

model_names=(
    'cut3r'
    'ttt3r'
    'ttt3r_random'    # aconst: constant dampening (p=0.33)
    'ttt3r_momentum_inv_t1' # brake: stability brake (best variant on zjc)
)

datasets=('kitti_odo_s1_200' 'kitti_odo_s1_1000')

for model_name in "${model_names[@]}"; do
for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/relpose/${data}/${model_name}"
    echo "=== ${data} / ${model_name} ==="
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} accelerate launch \
        --num_processes 1 \
        --main_process_port 29570 \
        eval/relpose/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512 \
        --model_update_type "$model_name"
done
done

echo "=== All done. Results in eval_results/relpose/kitti_odo_s1_*/ ==="

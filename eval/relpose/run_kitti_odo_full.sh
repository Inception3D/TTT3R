#!/bin/bash
# Evaluate relpose on KITTI odometry — FULL sequences (00-10)
# Methods: cut3r / ttt3r / ttt3r_random / ttt3r_momentum / ttt3r_ortho
#
# Usage:
#   bash eval/relpose/run_kitti_odo_full.sh
#   CUDA_VISIBLE_DEVICES=0 bash eval/relpose/run_kitti_odo_full.sh

set -e

workdir='.'
model_weights="${workdir}/src/cut3r_512_dpt_4_64.pth"

model_names=(
    'cut3r'
    'ttt3r'
    'ttt3r_random'
    'ttt3r_momentum'
    'ttt3r_ortho'
)

dataset='kitti_odo_full'
total=${#model_names[@]}
current=0

for model_name in "${model_names[@]}"; do
    current=$((current + 1))
    output_dir="${workdir}/eval_results/relpose/${dataset}/${model_name}"
    echo ""
    echo "========================================"
    echo "  [${current}/${total}] ${dataset} / ${model_name}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} accelerate launch \
        --num_processes 1 \
        --main_process_port 29570 \
        eval/relpose/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$dataset" \
        --size 512 \
        --model_update_type "$model_name"
    echo "[DONE] ${model_name} finished at $(date '+%H:%M:%S')"
done

echo ""
echo "=== All done at $(date). Results in eval_results/relpose/${dataset}/ ==="

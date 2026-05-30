#!/bin/bash

set -e

workdir='.'
if [ -n "${NUM_PROCESSES:-}" ]; then
  num_processes="${NUM_PROCESSES}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
  if [ -n "$gpu_count" ] && [ "$gpu_count" -gt 0 ]; then
    num_processes="$gpu_count"
  else
    num_processes="1"
  fi
else
  num_processes="1"
fi
amp_dtype="${AMP_DTYPE:-bf16}"
tf32="${TF32:-1}"
cudnn_benchmark="${CUDNN_BENCHMARK:-1}"
inference_mode="${INFERENCE_MODE:-1}"
# format: output_tag:model_update_type:alpha_drift
experiments=(
  'ttt3r:ttt3r:0.15'
  'ttt3r_momentum_inv_t1:ttt3r_momentum_inv_t1:0.15'
  'ttt3r_momentum_inv_t1_drift0:ttt3r_momentum_inv_t1:0.0'
)
ckpt_name='cut3r_512_dpt_4_64'
model_weights="${workdir}/src/${ckpt_name}.pth"
if [ ! -f "$model_weights" ]; then
    model_weights="${workdir}/model/${ckpt_name}.pth"
fi
if [ ! -f "$model_weights" ]; then
    echo "missing checkpoint: ${ckpt_name}.pth (checked src/ and model/)"
    exit 1
fi
# datasets=('kitti_s1_50' 'kitti_s1_100' 'kitti_s1_110' 'kitti_s1_150' 'kitti_s1_200' 'kitti_s1_250' 'kitti_s1_300' 'kitti_s1_350' 'kitti_s1_400' 'kitti_s1_450' 'kitti_s1_500')
datasets=('kitti_s1_500')


for exp in "${experiments[@]}"; do
IFS=':' read -r output_tag model_name alpha_drift <<< "$exp"
for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}/${output_tag}"
    echo "$output_dir"

    accelerate launch --num_processes "$num_processes" --main_process_port 29555 eval/video_depth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512 \
        --model_update_type "$model_name" \
        --alpha_drift "$alpha_drift" \
        --amp_dtype "$amp_dtype" \
        --tf32 "$tf32" \
        --cudnn_benchmark "$cudnn_benchmark" \
        --inference_mode "$inference_mode"

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

#!/bin/bash
# Stability brake tau sensitivity sweep.

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src

GPU=${1:-0}
WEIGHTS="model/cut3r_512_dpt_4_64.pth"
PORT=29610

echo "=========================================="
echo "S3 Stability brake tau sensitivity on GPU $GPU"
echo "=========================================="

for tau in 0.5 1.0 1.5 2.0 3.0; do
    TAG="ttt3r_momentum_inv_t${tau}"
    OUT="eval_results/relpose/scannet_s3_1000/${TAG}"
    if [ -f "$OUT/_error_log.txt" ] && [ "$(grep -c '| ATE' "$OUT/_error_log.txt" 2>/dev/null)" -ge 60 ]; then
        echo "[SKIP] $TAG already done"
        continue
    fi

    echo ""
    echo "[$(date)] $TAG"
    CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes 1 --main_process_port $PORT \
        eval/relpose/launch.py \
        --weights $WEIGHTS \
        --output_dir "$OUT" \
        --eval_dataset scannet_s3_1000 \
        --size 512 \
        --model_update_type ttt3r_momentum \
        --momentum_tau "$tau"
    PORT=$((PORT + 1))
done

echo ""
echo "=========================================="
echo "[$(date)] S3 complete"
echo "=========================================="

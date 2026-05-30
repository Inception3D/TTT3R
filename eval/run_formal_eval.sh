#!/bin/bash
# =============================================================================
# TTT3R 正式评测脚本
# 对比 cut3r / ttt3r / ttt3r_joint (L23+ttt3r) 在 ScanNet + TUM 上的位姿估计
# =============================================================================
# 前置条件:
#   1. 已运行 datasets_preprocess/prepare_scannet_local.py
#   2. 已运行 datasets_preprocess/prepare_tum_local.py
#   生成 data/long_scannet_s3/ 和 data/long_tum_s1/
# =============================================================================

set -e

cd "$(dirname "$0")/.."

MODEL_WEIGHTS="model/cut3r_512_dpt_4_64.pth"

# 三个对比配置
CONFIGS=("cut3r" "ttt3r" "ttt3r_joint")

# 频域超参 (仅 ttt3r_joint 使用, 但传给所有配置无副作用)
SPECTRAL_TAU=1.0
GEO_TAU=2.0
GEO_CUTOFF=4

# ========== 1. Relpose: ScanNet (stride 3, 1000 frames) ==========
echo "=========================================="
echo "[1/2] Relpose: ScanNet (scannet_s3_1000)"
echo "=========================================="

for config in "${CONFIGS[@]}"; do
    output_dir="eval_results/relpose/scannet_s3_1000/${config}"
    echo "  → ${config} => ${output_dir}"

    accelerate launch --num_processes 2 --main_process_port 29550 \
        eval/relpose/launch.py \
        --weights "$MODEL_WEIGHTS" \
        --output_dir "$output_dir" \
        --eval_dataset scannet_s3_1000 \
        --size 512 \
        --model_update_type "$config" \
        --spectral_temperature $SPECTRAL_TAU \
        --geo_gate_tau $GEO_TAU \
        --geo_gate_freq_cutoff $GEO_CUTOFF

    echo "  ✓ ${config} done"
done

# ========== 2. Relpose: TUM (stride 1, 1000 frames) ==========
echo "=========================================="
echo "[2/2] Relpose: TUM (tum_s1_1000)"
echo "=========================================="

for config in "${CONFIGS[@]}"; do
    output_dir="eval_results/relpose/tum_s1_1000/${config}"
    echo "  → ${config} => ${output_dir}"

    accelerate launch --num_processes 2 --main_process_port 29551 \
        eval/relpose/launch.py \
        --weights "$MODEL_WEIGHTS" \
        --output_dir "$output_dir" \
        --eval_dataset tum_s1_1000 \
        --size 512 \
        --model_update_type "$config" \
        --spectral_temperature $SPECTRAL_TAU \
        --geo_gate_tau $GEO_TAU \
        --geo_gate_freq_cutoff $GEO_CUTOFF

    echo "  ✓ ${config} done"
done

# ========== Summary ==========
echo ""
echo "=========================================="
echo "All formal evaluations complete."
echo "=========================================="
echo "Results:"
for config in "${CONFIGS[@]}"; do
    echo "  ScanNet: eval_results/relpose/scannet_s3_1000/${config}/_error_log.txt"
    echo "  TUM:     eval_results/relpose/tum_s1_1000/${config}/_error_log.txt"
done

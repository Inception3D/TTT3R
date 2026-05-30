#!/bin/bash
# =============================================================================
# TTT3R 正式评测 — 双卡并行
# GPU0: ScanNet relpose (cut3r → ttt3r → ttt3r_joint)
# GPU1: TUM relpose     (cut3r → ttt3r → ttt3r_joint)
# =============================================================================
set -e
cd "$(dirname "$0")/.."

MODEL_WEIGHTS="model/cut3r_512_dpt_4_64.pth"
CONFIGS=("cut3r" "ttt3r" "ttt3r_joint")

# 频域超参 (仅 ttt3r_joint 使用)
SPECTRAL_TAU=1.0
GEO_TAU=2.0
GEO_CUTOFF=4

mkdir -p eval_results/relpose

# ── GPU 0: ScanNet relpose ──
run_scannet() {
    for config in "${CONFIGS[@]}"; do
        output_dir="eval_results/relpose/scannet_s3_1000/${config}"
        echo "[GPU0/ScanNet] ${config} => ${output_dir}"
        CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29560 \
            eval/relpose/launch.py \
            --weights "$MODEL_WEIGHTS" \
            --output_dir "$output_dir" \
            --eval_dataset scannet_s3_1000 \
            --size 512 \
            --model_update_type "$config" \
            --spectral_temperature $SPECTRAL_TAU \
            --geo_gate_tau $GEO_TAU \
            --geo_gate_freq_cutoff $GEO_CUTOFF
        echo "[GPU0/ScanNet] ✓ ${config} done"
    done
    echo "[GPU0] All ScanNet relpose done."
}

# ── GPU 1: TUM relpose ──
run_tum() {
    for config in "${CONFIGS[@]}"; do
        output_dir="eval_results/relpose/tum_s1_1000/${config}"
        echo "[GPU1/TUM] ${config} => ${output_dir}"
        CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 --main_process_port 29561 \
            eval/relpose/launch.py \
            --weights "$MODEL_WEIGHTS" \
            --output_dir "$output_dir" \
            --eval_dataset tum_s1_1000 \
            --size 512 \
            --model_update_type "$config" \
            --spectral_temperature $SPECTRAL_TAU \
            --geo_gate_tau $GEO_TAU \
            --geo_gate_freq_cutoff $GEO_CUTOFF
        echo "[GPU1/TUM] ✓ ${config} done"
    done
    echo "[GPU1] All TUM relpose done."
}

# 并行启动
run_scannet &
PID_SCANNET=$!

run_tum &
PID_TUM=$!

echo "=========================================="
echo "Parallel eval launched:"
echo "  GPU0 (ScanNet): PID ${PID_SCANNET}"
echo "  GPU1 (TUM):     PID ${PID_TUM}"
echo "=========================================="

# 等待两个任务完成
wait $PID_SCANNET
STATUS_SCANNET=$?
wait $PID_TUM
STATUS_TUM=$?

echo ""
echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
echo "  ScanNet: exit code ${STATUS_SCANNET}"
echo "  TUM:     exit code ${STATUS_TUM}"
echo ""
echo "Results:"
for config in "${CONFIGS[@]}"; do
    echo "  ScanNet/${config}: eval_results/relpose/scannet_s3_1000/${config}/_error_log.txt"
    echo "  TUM/${config}:     eval_results/relpose/tum_s1_1000/${config}/_error_log.txt"
done

#!/bin/bash
# =============================================================================
# TTT3R 频域引导实验运行文档
# =============================================================================
# 在服务器上执行，工作目录: ~/code/TTT3R (或项目根目录)
# 前置条件: conda 环境已激活，GPU 可用
# =============================================================================

set -e

# ── 公共变量 ──
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="model/cut3r_512_dpt_4_64.pth"
SCANNET_ROOT="/mnt/sda/szy/research/dataset/scannetv2"
TUM_ROOT="/mnt/sda/szy/research/dataset/tum"
GPU_ID=0
NUM_SCANNET=10
SEED=42
MAX_FRAMES=200

# =============================================================================
# Step 1: B2 Memory Gate 消融实验 (优先级最高)
# =============================================================================
# 对比 7 个配置:
#   cut3r (baseline), ttt3r (baseline)
#   cut3r_memgate x3 组超参 (tau=2/3/5, skip_ratio=0.3/0.5)
#   ttt3r_memgate
# 预估时间: ~40 分钟 (7 configs × 18 scenes × ~20s/scene)
# 输出: analysis_results/memgate_ablation/

echo "============================================"
echo "[Step 1] B2 Memory Gate Ablation"
echo "============================================"

CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH=src python analysis/memgate_ablation.py \
    --model_path "$MODEL_PATH" \
    --scannet_root "$SCANNET_ROOT" \
    --tum_root "$TUM_ROOT" \
    --output_dir analysis_results/memgate_ablation \
    --num_scannet $NUM_SCANNET \
    --seed $SEED \
    --max_frames $MAX_FRAMES \
    2>&1 | tee analysis_results/memgate_ablation/run.log

echo "[Step 1] Done. Results: analysis_results/memgate_ablation/memgate_ablation_summary.txt"

# =============================================================================
# Step 2: SIASU 消融实验 (warm-start 修复后重跑)
# =============================================================================
# 对比 8 个配置:
#   cut3r, ttt3r (baseline)
#   cut3r_spectral x3 温度 (tau=1/2/4)
#   ttt3r_spectral x3 温度 (tau=1/2/4)
# 预估时间: ~50 分钟 (8 configs × 18 scenes × ~20s/scene)
# 输出: analysis_results/spectral_ablation/

echo "============================================"
echo "[Step 2] SIASU Spectral Ablation (warm-start fix)"
echo "============================================"

CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH=src python analysis/spectral_ablation.py \
    --model_path "$MODEL_PATH" \
    --scannet_root "$SCANNET_ROOT" \
    --tum_root "$TUM_ROOT" \
    --output_dir analysis_results/spectral_ablation \
    --num_scannet $NUM_SCANNET \
    --seed $SEED \
    --max_frames $MAX_FRAMES \
    2>&1 | tee analysis_results/spectral_ablation/run.log

echo "[Step 2] Done. Results: analysis_results/spectral_ablation/ablation_summary.txt"

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "============================================"
echo "All experiments completed."
echo "============================================"
echo "Results:"
echo "  Step 1 (B2 MemGate):  analysis_results/memgate_ablation/memgate_ablation_summary.txt"
echo "  Step 2 (SIASU):       analysis_results/spectral_ablation/ablation_summary.txt"
echo ""
echo "Plots:"
echo "  analysis_results/memgate_ablation/memgate_ablation.png"
echo "  analysis_results/spectral_ablation/ablation_comparison.png"

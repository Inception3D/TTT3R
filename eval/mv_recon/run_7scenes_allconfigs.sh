#!/bin/bash
# =============================================================================
# run_7scenes_allconfigs.sh — Run all DDD3R configs on 7scenes 3D Reconstruction
#
# Metrics: Acc / Comp / NC  (consistent with Spann3R NeurIPS'24, CUT3R, TTT3R)
#
# Dataset: 7scenes test split (18 sequences, 7 scenes)
#   chess/seq-03, chess/seq-05, fire/seq-03, fire/seq-04, heads/seq-01,
#   office/seq-02, office/seq-06, office/seq-07, office/seq-09,
#   pumpkin/seq-01, pumpkin/seq-07, redkitchen/seq-03, redkitchen/seq-04,
#   redkitchen/seq-06, redkitchen/seq-12, redkitchen/seq-14,
#   stairs/seq-01, stairs/seq-04
#
# Key parameters:
#   - kf_every=2 (every 2nd frame → ~200-500 views per sequence)
#   - max_frames=200 (cap at 200 views, consistent with TTT3R baseline)
#   - resolution=512x384
#
# Configs (14 total):
#   1. cut3r              (baseline, mask1=1.0)
#   2. ttt3r              (baseline, mask1=sigmoid gate)
#   3. constant           (ttt3r_random, alpha=0.5)
#   4. brake              (ttt3r_momentum, tau=2.0)
#   5. ortho              (ddd3r, gamma=0)
#   6. ddd3r_g1           (ddd3r, gamma=1)
#   7. ddd3r_g2           (ddd3r, gamma=2)
#   8. ddd3r_g3           (ddd3r, gamma=3)
#   9. ddd3r_g4           (ddd3r, gamma=4)
#  10. ddd3r_g5           (ddd3r, gamma=5)
#  11. auto_steep_clamp   (ddd3r, auto_gamma=steep_clamp)
#  12. auto_steep_sigmoid (ddd3r, auto_gamma=steep_sigmoid)
#  13. auto_warmup_linear (ddd3r, auto_gamma=warmup_linear)
#  14. auto_warmup_threshold (ddd3r, auto_gamma=warmup_threshold)
#
# Usage:
#   bash eval/mv_recon/run_7scenes_allconfigs.sh
#
# Assumes:
#   - 7scenes data at ~/TTT3R/data/7scenes/
#   - Model weights at ~/TTT3R/model/cut3r_512_dpt_4_64.pth
# =============================================================================

set -euo pipefail

export PATH="/root/miniconda3/bin:$PATH"
export NCCL_TIMEOUT=360000

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG="${WORKDIR}/7scenes_allconfigs.log"
MODEL_WEIGHTS="${WORKDIR}/model/cut3r_512_dpt_4_64.pth"
MAX_FRAMES=200
PORT=29590
# Match original TTT3R output path: eval_results/video_recon/7scenes_200/
RESULTS_NS="video_recon/7scenes_200"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "${LOG}"; }

cd "${WORKDIR}"
log "=== 7scenes evaluation (all DDD3R configs) started ==="

if [ ! -f "${MODEL_WEIGHTS}" ]; then
    log "ERROR: ${MODEL_WEIGHTS} not found"; exit 1
fi

if [ ! -d "${WORKDIR}/data/7scenes" ]; then
    log "ERROR: ${WORKDIR}/data/7scenes not found"; exit 1
fi

log "7scenes data: ${WORKDIR}/data/7scenes"
log "weights: $(du -h ${MODEL_WEIGHTS} | cut -f1)"
log "max_frames: ${MAX_FRAMES}"

run_eval() {
    local tag="$1"
    shift
    local update_type="$1"
    shift
    local extra_args=("$@")
    local out_dir="${WORKDIR}/eval_results/${RESULTS_NS}/${tag}"
    log "--- [run] ${tag} -> ${out_dir}"
    # Match original TTT3R run.sh: NCCL_TIMEOUT, --model_name, no --size (default 512)
    # PYTHONPATH=src needed because weights are in model/ not src/
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${WORKDIR}/src" \
    accelerate launch \
        --num_processes 1 \
        --main_process_port ${PORT} \
        eval/mv_recon/launch.py \
        --weights "${MODEL_WEIGHTS}" \
        --output_dir "${out_dir}" \
        --model_name "${update_type}" \
        --model_update_type "${update_type}" \
        --max_frames ${MAX_FRAMES} \
        "${extra_args[@]}" \
        2>&1 | tee -a "${LOG}"
    PORT=$((PORT + 1))
    log "--- [done] ${tag}"
}

# Baselines
run_eval "cut3r"    "cut3r"
run_eval "ttt3r"    "ttt3r"

# DDD3R spectrum: constant dampening (α⊥=α∥=0.5)
run_eval "constant" "ttt3r_random"

# Temporal brake baseline
run_eval "brake"    "ttt3r_momentum"

# DDD3R fixed ortho (γ=0)
run_eval "ortho"    "ddd3r"      --gamma 0.0

# DDD3R drift-adaptive spectrum
run_eval "ddd3r_g1" "ddd3r"      --gamma 1
run_eval "ddd3r_g2" "ddd3r"      --gamma 2
run_eval "ddd3r_g3" "ddd3r"      --gamma 3
run_eval "ddd3r_g4" "ddd3r"      --gamma 4
run_eval "ddd3r_g5" "ddd3r"      --gamma 5

# Auto-gamma variants
run_eval "auto_steep_clamp"      "ddd3r" --auto_gamma steep_clamp
run_eval "auto_steep_sigmoid"    "ddd3r" --auto_gamma steep_sigmoid
run_eval "auto_warmup_linear"    "ddd3r" --auto_gamma warmup_linear
run_eval "auto_warmup_threshold" "ddd3r" --auto_gamma warmup_threshold

log "=== All 14 configs complete ==="
log "Generate report: python3 generate_7scenes_report.py"

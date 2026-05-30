#!/bin/bash
# =============================================================================
# run_dtu_allconfigs.sh  —  Run all DDD3R configs on DTU 15-scene test set
#
# Metrics: Acc / Comp / NC  (consistent with DUSt3R CVPR'24, Spann3R NeurIPS'24,
#          MASt3R ECCV'24 — all use this same Acc/Comp/NC framework on DTU)
#
# Configs (14 total):
#   1. cut3r              (baseline)
#   2. ttt3r              (baseline)
#   3. constant           (ttt3r_random, constant dampening)
#   4. brake              (ttt3r_momentum)
#   5. ortho              (ddd3r, gamma=0)
#   6. ddd3r_g1           (ddd3r, gamma=1, expected cross-dataset sweet spot)
#   7. ddd3r_g2           (ddd3r, gamma=2)
#   8. ddd3r_g3           (ddd3r, gamma=3)
#   9. ddd3r_g4           (ddd3r, gamma=4)
#  10. ddd3r_g5           (ddd3r, gamma=5)
#  11. auto_steep_clamp   (ddd3r, auto_gamma=steep_clamp)
#  12. auto_steep_sigmoid (ddd3r, auto_gamma=steep_sigmoid)
#  13. auto_warmup_linear (ddd3r, auto_gamma=warmup_linear)
#  14. auto_warmup_threshold (ddd3r, auto_gamma=warmup_threshold)
#
# Assumes:
#   - DTU preprocessed data at ~/TTT3R/data/dtu/  (see setup_dtu_server.sh)
#   - Model weights at ~/TTT3R/model/cut3r_512_dpt_4_64.pth
#   - Run from ~/TTT3R: bash eval/mv_recon/run_dtu_allconfigs.sh
# =============================================================================

set -euo pipefail

export PATH="/root/miniconda3/bin:$PATH"

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG="${WORKDIR}/dtu_allconfigs.log"
MODEL_WEIGHTS="${WORKDIR}/model/cut3r_512_dpt_4_64.pth"
DTU_ROOT="${WORKDIR}/data/dtu"
SIZE=512
PORT=29570

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "${LOG}"; }

cd "${WORKDIR}"
log "=== DTU evaluation (all DDD3R configs) started ==="

if [ ! -f "${MODEL_WEIGHTS}" ]; then
    log "ERROR: ${MODEL_WEIGHTS} not found"; exit 1
fi

if [ ! -d "${DTU_ROOT}" ]; then
    log "ERROR: ${DTU_ROOT} not found. Run setup_dtu_server.sh first."; exit 1
fi

# Verify at least one test scene exists
if [ ! -d "${DTU_ROOT}/scan8" ]; then
    log "ERROR: ${DTU_ROOT}/scan8 not found. Check DTU data structure."; exit 1
fi

log "DTU data: ${DTU_ROOT}"
log "weights: $(du -h ${MODEL_WEIGHTS} | cut -f1)"

run_eval() {
    local tag="$1"
    local update_type="$2"
    local extra_args="${3:-}"
    local out_dir="${WORKDIR}/eval_results/mv_recon/dtu/${tag}"
    log "--- [run] ${tag} -> ${out_dir}"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${WORKDIR}/src" \
    accelerate launch \
        --num_processes 1 \
        --main_process_port ${PORT} \
        eval/mv_recon/launch.py \
        --weights "${MODEL_WEIGHTS}" \
        --output_dir "${out_dir}" \
        --eval_dataset dtu \
        --dtu_root "${DTU_ROOT}" \
        --size ${SIZE} \
        --model_update_type "${update_type}" \
        ${extra_args} \
        2>&1 | tee -a "${LOG}"
    PORT=$((PORT + 1))
    log "--- [done] ${tag}"
}

run_eval "cut3r"    "cut3r"
run_eval "ttt3r"    "ttt3r"
run_eval "constant" "ttt3r_random"
run_eval "brake"    "ttt3r_momentum"
run_eval "ortho"    "ddd3r"      "--gamma 0.0"
run_eval "ddd3r_g1" "ddd3r"      "--gamma 1"
run_eval "ddd3r_g2" "ddd3r"      "--gamma 2"
run_eval "ddd3r_g3" "ddd3r"      "--gamma 3"
run_eval "ddd3r_g4" "ddd3r"      "--gamma 4"
run_eval "ddd3r_g5" "ddd3r"      "--gamma 5"

# Auto-gamma variants
run_eval "auto_steep_clamp"      "ddd3r" "--auto_gamma steep_clamp"
run_eval "auto_steep_sigmoid"    "ddd3r" "--auto_gamma steep_sigmoid"
run_eval "auto_warmup_linear"    "ddd3r" "--auto_gamma warmup_linear"
run_eval "auto_warmup_threshold" "ddd3r" "--auto_gamma warmup_threshold"

# Pack results (skipped when called from run_recon_after_kitti.sh)
if [ "${SKIP_PACK:-0}" != "1" ]; then
    log "=== packing results ==="
    PACK="${WORKDIR}/dtu_results_$(date '+%Y%m%d_%H%M').tar.gz"
    tar czf "${PACK}" eval_results/mv_recon/dtu 2>/dev/null || true
    log "packed: ${PACK}"
    log "=== Done. Fetch with: ==="
    log "  scp -P 46355 root@region-9.autodl.pro:${PACK} ."
fi

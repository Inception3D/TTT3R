#!/bin/bash
# =============================================================================
# run_7scenes_parallel.sh — Run 14 DDD3R configs on 7scenes, 2 at a time
#
# Two-lane parallel execution on single GPU (A100 40GB, ~6GB per process).
# Each lane runs configs sequentially; two lanes run concurrently.
#
# Lane A (7 configs): cut3r, constant, ortho, ddd3r_g2, ddd3r_g4, auto_steep_clamp, auto_warmup_linear
# Lane B (7 configs): ttt3r, brake, ddd3r_g1, ddd3r_g3, ddd3r_g5, auto_steep_sigmoid, auto_warmup_threshold
#
# Usage:
#   bash eval/mv_recon/run_7scenes_parallel.sh
# =============================================================================

set -euo pipefail

export PATH="/root/miniconda3/bin:$PATH"
export NCCL_TIMEOUT=360000

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_WEIGHTS="${WORKDIR}/model/cut3r_512_dpt_4_64.pth"
MAX_FRAMES=200
RESULTS_NS="video_recon/7scenes_200"

cd "${WORKDIR}"

if [ ! -f "${MODEL_WEIGHTS}" ]; then
    echo "ERROR: ${MODEL_WEIGHTS} not found"; exit 1
fi
if [ ! -d "${WORKDIR}/data/7scenes" ]; then
    echo "ERROR: ${WORKDIR}/data/7scenes not found"; exit 1
fi

run_one() {
    local tag="$1"
    local update_type="$2"
    local port="$3"
    local logfile="$4"
    shift 4
    local extra_args=("$@")
    local out_dir="${WORKDIR}/eval_results/${RESULTS_NS}/${tag}"

    echo "[$(date '+%H:%M:%S')] START ${tag}" | tee -a "${logfile}"

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${WORKDIR}/src" \
    accelerate launch \
        --num_processes 1 \
        --main_process_port ${port} \
        eval/mv_recon/launch.py \
        --weights "${MODEL_WEIGHTS}" \
        --output_dir "${out_dir}" \
        --model_name "${update_type}" \
        --model_update_type "${update_type}" \
        --max_frames ${MAX_FRAMES} \
        "${extra_args[@]}" \
        >> "${logfile}" 2>&1

    echo "[$(date '+%H:%M:%S')] DONE  ${tag}" | tee -a "${logfile}"
}

run_lane() {
    local lane="$1"
    local port="$2"
    local logfile="${WORKDIR}/7scenes_lane_${lane}.log"
    > "${logfile}"  # truncate

    echo "[$(date '+%H:%M:%S')] === Lane ${lane} started ===" | tee -a "${logfile}"

    if [ "${lane}" = "A" ]; then
        run_one "cut3r"              "cut3r"          ${port} "${logfile}"
        run_one "constant"           "ttt3r_random"   ${port} "${logfile}"
        run_one "ortho"              "ddd3r"          ${port} "${logfile}" --gamma 0.0
        run_one "ddd3r_g2"           "ddd3r"          ${port} "${logfile}" --gamma 2
        run_one "ddd3r_g4"           "ddd3r"          ${port} "${logfile}" --gamma 4
        run_one "auto_steep_clamp"   "ddd3r"          ${port} "${logfile}" --auto_gamma steep_clamp
        run_one "auto_warmup_linear" "ddd3r"          ${port} "${logfile}" --auto_gamma warmup_linear
    else
        run_one "ttt3r"                  "ttt3r"          ${port} "${logfile}"
        run_one "brake"                  "ttt3r_momentum" ${port} "${logfile}"
        run_one "ddd3r_g1"               "ddd3r"          ${port} "${logfile}" --gamma 1
        run_one "ddd3r_g3"               "ddd3r"          ${port} "${logfile}" --gamma 3
        run_one "ddd3r_g5"               "ddd3r"          ${port} "${logfile}" --gamma 5
        run_one "auto_steep_sigmoid"     "ddd3r"          ${port} "${logfile}" --auto_gamma steep_sigmoid
        run_one "auto_warmup_threshold"  "ddd3r"          ${port} "${logfile}" --auto_gamma warmup_threshold
    fi

    echo "[$(date '+%H:%M:%S')] === Lane ${lane} finished ===" | tee -a "${logfile}"
}

echo "[$(date '+%H:%M:%S')] === 7scenes parallel eval (2 lanes x 7 configs) ==="
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Model: $(du -h ${MODEL_WEIGHTS} | cut -f1)"
echo "  Max frames: ${MAX_FRAMES}"

# Launch two lanes with different accelerate ports
run_lane "A" 29590 &
PID_A=$!

run_lane "B" 29591 &
PID_B=$!

echo "[$(date '+%H:%M:%S')] Lane A PID=${PID_A}, Lane B PID=${PID_B}"
echo "  Logs: 7scenes_lane_A.log, 7scenes_lane_B.log"
echo "  Monitor: tail -f ~/TTT3R/7scenes_lane_A.log ~/TTT3R/7scenes_lane_B.log"

# Wait for both
wait ${PID_A}
RC_A=$?
wait ${PID_B}
RC_B=$?

echo "[$(date '+%H:%M:%S')] === All done. Lane A exit=${RC_A}, Lane B exit=${RC_B} ==="
echo "Generate report: python3 eval/mv_recon/generate_7scenes_report.py"

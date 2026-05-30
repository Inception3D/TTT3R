#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
WEIGHTS_PATH="${WEIGHTS_PATH:-$REPO_ROOT/src/cut3r_512_dpt_4_64.pth}"

WAYMO_TFRECORD_GLOB="${WAYMO_TFRECORD_GLOB:?set WAYMO_TFRECORD_GLOB to your TFRecord glob}"
WAYMO_CAMERA="${WAYMO_CAMERA:-FRONT}"
WAYMO_OUTPUT_ROOT="${WAYMO_OUTPUT_ROOT:-$REPO_ROOT/data/waymo_relpose}"
MAX_SEGMENTS="${MAX_SEGMENTS:-8}"
MAX_FRAMES="${MAX_FRAMES:-500}"
STRIDE="${STRIDE:-1}"
MIN_FRAMES="${MIN_FRAMES:-30}"
OVERWRITE_DATA="${OVERWRITE_DATA:-0}"

NUM_PROCESSES="${NUM_PROCESSES:-}"
SIZE="${SIZE:-512}"
MAIN_PORT="${MAIN_PORT:-29563}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
TF32="${TF32:-1}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-1}"
INFERENCE_MODE="${INFERENCE_MODE:-1}"

TF_PACKAGE="${TF_PACKAGE:-tensorflow==2.12.0}"
WAYMO_PACKAGE="${WAYMO_PACKAGE:-waymo-open-dataset-tf-2-12-0}"

echo "[pipeline] repo root: $REPO_ROOT"
echo "[pipeline] venv: $VENV_DIR"
echo "[pipeline] tfrecord glob: $WAYMO_TFRECORD_GLOB"
echo "[pipeline] output root: $WAYMO_OUTPUT_ROOT"

if [ ! -d "$VENV_DIR" ]; then
  echo "[pipeline] creating virtual environment"
  bash "$REPO_ROOT/scripts/server/setup_remote_env.sh"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
pip install "$TF_PACKAGE" "$WAYMO_PACKAGE"

if [ ! -f "$WEIGHTS_PATH" ]; then
  echo "[pipeline] missing weights at: $WEIGHTS_PATH"
  exit 1
fi

overwrite_flag=()
if [ "$OVERWRITE_DATA" = "1" ]; then
  overwrite_flag+=(--overwrite)
fi

python "$REPO_ROOT/datasets_preprocess/prepare_waymo_relpose.py" \
  --tfrecord_glob "$WAYMO_TFRECORD_GLOB" \
  --output_root "$WAYMO_OUTPUT_ROOT" \
  --camera "$WAYMO_CAMERA" \
  --max_segments "$MAX_SEGMENTS" \
  --max_frames "$MAX_FRAMES" \
  --stride "$STRIDE" \
  --min_frames "$MIN_FRAMES" \
  "${overwrite_flag[@]}"

pushd "$REPO_ROOT" >/dev/null
DATASET="waymo_relpose" \
DATASET_ROOT="$WAYMO_OUTPUT_ROOT" \
WAYMO_RELPOSE_ROOT="$WAYMO_OUTPUT_ROOT" \
MODEL_WEIGHTS="$WEIGHTS_PATH" \
NUM_PROCESSES="$NUM_PROCESSES" \
SIZE="$SIZE" \
MAIN_PORT="$MAIN_PORT" \
AMP_DTYPE="$AMP_DTYPE" \
TF32="$TF32" \
CUDNN_BENCHMARK="$CUDNN_BENCHMARK" \
INFERENCE_MODE="$INFERENCE_MODE" \
bash eval/relpose/run_waymo_relpose.sh
popd >/dev/null

python "$REPO_ROOT/scripts/server/export_relpose_summary.py" \
  --eval_root "$REPO_ROOT/eval_results/relpose/waymo_relpose"

echo "[pipeline] done"
echo "[pipeline] summary: $REPO_ROOT/eval_results/relpose/waymo_relpose/summary.csv"
echo "[pipeline] per-seq: $REPO_ROOT/eval_results/relpose/waymo_relpose/per_sequence_results.csv"

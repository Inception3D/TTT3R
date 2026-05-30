#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
WEIGHTS_PATH="${WEIGHTS_PATH:-$REPO_ROOT/src/cut3r_512_dpt_4_64.pth}"

NUSCENES_DATAROOT="${NUSCENES_DATAROOT:?set NUSCENES_DATAROOT to your nuScenes data root}"
NUSCENES_VERSION="${NUSCENES_VERSION:-v1.0-mini}"
NUSCENES_CAMERA="${NUSCENES_CAMERA:-CAM_FRONT}"
NUSCENES_OUTPUT_ROOT="${NUSCENES_OUTPUT_ROOT:-$REPO_ROOT/data/nuscenes_relpose}"
MAX_SCENES="${MAX_SCENES:-10}"
MAX_FRAMES="${MAX_FRAMES:-500}"
STRIDE="${STRIDE:-1}"
MIN_FRAMES="${MIN_FRAMES:-30}"
COPY_MODE="${COPY_MODE:-copy}"
OVERWRITE_DATA="${OVERWRITE_DATA:-0}"

NUM_PROCESSES="${NUM_PROCESSES:-}"
SIZE="${SIZE:-512}"
MAIN_PORT="${MAIN_PORT:-29562}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
TF32="${TF32:-1}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-1}"
INFERENCE_MODE="${INFERENCE_MODE:-1}"

echo "[pipeline] repo root: $REPO_ROOT"
echo "[pipeline] venv: $VENV_DIR"
echo "[pipeline] dataset root: $NUSCENES_DATAROOT"
echo "[pipeline] output root: $NUSCENES_OUTPUT_ROOT"

if [ ! -d "$VENV_DIR" ]; then
  echo "[pipeline] creating virtual environment"
  bash "$REPO_ROOT/scripts/server/setup_remote_env.sh"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
pip install nuscenes-devkit pyquaternion

if [ ! -f "$WEIGHTS_PATH" ]; then
  echo "[pipeline] missing weights at: $WEIGHTS_PATH"
  exit 1
fi

overwrite_flag=()
if [ "$OVERWRITE_DATA" = "1" ]; then
  overwrite_flag+=(--overwrite)
fi

python "$REPO_ROOT/datasets_preprocess/prepare_nuscenes_relpose.py" \
  --dataroot "$NUSCENES_DATAROOT" \
  --version "$NUSCENES_VERSION" \
  --output_root "$NUSCENES_OUTPUT_ROOT" \
  --camera "$NUSCENES_CAMERA" \
  --max_scenes "$MAX_SCENES" \
  --max_frames "$MAX_FRAMES" \
  --stride "$STRIDE" \
  --min_frames "$MIN_FRAMES" \
  --copy_mode "$COPY_MODE" \
  "${overwrite_flag[@]}"

pushd "$REPO_ROOT" >/dev/null
DATASET="nuscenes_relpose" \
DATASET_ROOT="$NUSCENES_OUTPUT_ROOT" \
NUSCENES_RELPOSE_ROOT="$NUSCENES_OUTPUT_ROOT" \
MODEL_WEIGHTS="$WEIGHTS_PATH" \
NUM_PROCESSES="$NUM_PROCESSES" \
SIZE="$SIZE" \
MAIN_PORT="$MAIN_PORT" \
AMP_DTYPE="$AMP_DTYPE" \
TF32="$TF32" \
CUDNN_BENCHMARK="$CUDNN_BENCHMARK" \
INFERENCE_MODE="$INFERENCE_MODE" \
bash eval/relpose/run_nuscenes_relpose.sh
popd >/dev/null

python "$REPO_ROOT/scripts/server/export_relpose_summary.py" \
  --eval_root "$REPO_ROOT/eval_results/relpose/nuscenes_relpose"

echo "[pipeline] done"
echo "[pipeline] summary: $REPO_ROOT/eval_results/relpose/nuscenes_relpose/summary.csv"
echo "[pipeline] per-seq: $REPO_ROOT/eval_results/relpose/nuscenes_relpose/per_sequence_results.csv"

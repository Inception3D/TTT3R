#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
KITTI_VAL_ROOT="${KITTI_VAL_ROOT:-$REPO_ROOT/data/kitti/val}"
KITTI_LONG_ROOT="${KITTI_LONG_ROOT:-$REPO_ROOT/data/long_kitti_s1}"
WEIGHTS_PATH="${WEIGHTS_PATH:-$REPO_ROOT/src/cut3r_512_dpt_4_64.pth}"
TARGET_FRAMES="${TARGET_FRAMES:-500}"

if [ ! -d "$KITTI_VAL_ROOT" ]; then
    echo "[run] missing KITTI val root: $KITTI_VAL_ROOT"
    echo "[run] expected directories like */proj_depth/groundtruth/image_02 under that path"
    exit 1
fi

if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "[run] missing weights: $WEIGHTS_PATH"
    exit 1
fi

source "$VENV_DIR/bin/activate"

echo "[run] preparing long KITTI subset"
python "$REPO_ROOT/datasets_preprocess/long_prepare_kitti.py" \
  --source_root "$KITTI_VAL_ROOT" \
  --output_root "$KITTI_LONG_ROOT" \
  --target_frames "$TARGET_FRAMES"

echo "[run] verifying gathered subset"
find "$KITTI_LONG_ROOT" -path "*image_gathered_${TARGET_FRAMES}*" | head

echo "[run] launching brake KITTI evaluation"
pushd "$REPO_ROOT" >/dev/null
bash eval/video_depth/run_kitti.sh
popd >/dev/null

echo "[run] result files"
find "$REPO_ROOT/eval_results/video_depth/kitti_s1_${TARGET_FRAMES}" -name "result_*.json" -o -name "*.txt"

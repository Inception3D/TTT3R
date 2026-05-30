#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[setup] repo root: $REPO_ROOT"
echo "[setup] creating venv at: $VENV_DIR"

$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools
pip install -r "$REPO_ROOT/requirements.txt"
pip install evo open3d gdown scikit-image

echo "[setup] compiling RoPE CUDA extension"
pushd "$REPO_ROOT/src/croco/models/curope" >/dev/null
python setup.py build_ext --inplace || true
popd >/dev/null

echo "[setup] verifying torch/cuda"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "[setup] done"

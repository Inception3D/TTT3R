#!/bin/bash
# =============================================================================
# One-click setup script for KITTI Odometry full-sequence evaluation
# Run on fresh GPU server (PyTorch 2.1.0 + CUDA 12.1)
#
# Usage:
#   bash eval/relpose/setup_kitti_full.sh
#
# Prerequisites:
#   - Server has git, wget, unzip, pip
#   - Model weight file at src/cut3r_512_dpt_4_64.pth (or will be transferred)
# =============================================================================

set -e

WORKDIR=$(pwd)
KITTI_ROOT="${WORKDIR}/kitti_raw"
DATA_DIR="${WORKDIR}/data/long_kitti_odo_s1"

echo "============================================"
echo "  KITTI Odometry Full-Sequence Setup"
echo "  $(date)"
echo "============================================"

# --------------------------------------------------
# Step 1: Install Python dependencies
# --------------------------------------------------
echo ""
echo "[1/4] Installing Python dependencies..."
pip install numpy==1.26.4 2>/dev/null
pip install scipy evo transformers==4.38.2 accelerate 2>/dev/null
echo "[OK] Dependencies installed."

# --------------------------------------------------
# Step 2: Download KITTI data (sequences 00-10 only)
# --------------------------------------------------
echo ""
echo "[2/4] Downloading KITTI odometry data..."
bash eval/relpose/download_kitti.sh "${KITTI_ROOT}"

# --------------------------------------------------
# Step 3: Prepare data (symlinks + TUM poses)
# --------------------------------------------------
echo ""
echo "[3/4] Preparing evaluation data..."

python eval/relpose/prepare_kitti_odometry.py \
    --kitti_root "${KITTI_ROOT}" \
    --output_root "${DATA_DIR}" \
    --seqs 00 01 02 03 04 05 06 07 08 09 10 \
    --full

echo "[OK] Data prepared at ${DATA_DIR}"

# --------------------------------------------------
# Step 4: Verify setup
# --------------------------------------------------
echo ""
echo "[4/4] Verifying setup..."

# Check model weights
if [ -f "src/cut3r_512_dpt_4_64.pth" ]; then
    echo "  [OK] Model weights found."
else
    echo "  [WARN] Model weights NOT found at src/cut3r_512_dpt_4_64.pth"
    echo "         Please transfer them before running experiments."
fi

# Check sequences
ok=0
for seq in 00 01 02 03 04 05 06 07 08 09 10; do
    img_dir="${DATA_DIR}/${seq}/image_full"
    pose_file="${DATA_DIR}/${seq}/pose_full.txt"
    if [ -d "$img_dir" ] && [ -f "$pose_file" ]; then
        n_imgs=$(ls "$img_dir" | wc -l)
        n_poses=$(wc -l < "$pose_file")
        echo "  [OK] seq ${seq}: ${n_imgs} images, ${n_poses} poses"
        ok=$((ok + 1))
    else
        echo "  [FAIL] seq ${seq}: missing data"
    fi
done

echo ""
if [ $ok -eq 11 ]; then
    echo "============================================"
    echo "  Setup complete! All 11 sequences ready."
    echo "  Run experiments with:"
    echo "    bash eval/relpose/run_kitti_odo_full.sh"
    echo "============================================"
else
    echo "[ERROR] Only ${ok}/11 sequences ready. Check above for issues."
fi

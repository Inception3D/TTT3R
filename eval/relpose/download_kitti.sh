#!/bin/bash
# =============================================================================
# Download KITTI Odometry data (sequences 00-10) + poses
# Supports resume on interruption (wget -c)
#
# KITTI official only provides full zip (65GB color images for all 22 seqs).
# Strategy: download full zip → extract only 00-10 → delete zip to save space.
# Net disk usage: ~38GB (11 sequences) instead of 65GB.
#
# Usage:
#   bash eval/relpose/download_kitti.sh [KITTI_ROOT]
#   bash eval/relpose/download_kitti.sh /root/kitti_raw
# =============================================================================

set -e

KITTI_ROOT="${1:-$(pwd)/kitti_raw}"
SEQS="00 01 02 03 04 05 06 07 08 09 10"

echo "============================================"
echo "  KITTI Odometry Downloader"
echo "  Target: ${KITTI_ROOT}"
echo "  Sequences: ${SEQS}"
echo "  $(date)"
echo "============================================"

mkdir -p "${KITTI_ROOT}/sequences" "${KITTI_ROOT}/poses"

# --------------------------------------------------
# 1. Download & extract poses (<1MB, instant)
# --------------------------------------------------
if [ -f "${KITTI_ROOT}/poses/00.txt" ] && [ -f "${KITTI_ROOT}/poses/10.txt" ]; then
    echo "[1/3] Poses already exist. Skipping."
else
    echo "[1/3] Downloading poses..."
    cd "${KITTI_ROOT}"
    wget -c -q --show-progress -O data_odometry_poses.zip \
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip"
    unzip -q -o data_odometry_poses.zip
    if [ -d "dataset/poses" ]; then
        mv dataset/poses/*.txt poses/
        rm -rf dataset
    fi
    rm -f data_odometry_poses.zip
    cd - > /dev/null
    echo "[OK] Poses ready. (11 files)"
fi

# --------------------------------------------------
# 2. Download color images (65GB, resumable)
# --------------------------------------------------
COLOR_ZIP="${KITTI_ROOT}/data_odometry_color.zip"
ALL_EXIST=true
for seq in $SEQS; do
    if [ ! -d "${KITTI_ROOT}/sequences/${seq}/image_2" ]; then
        ALL_EXIST=false
        break
    fi
done

if $ALL_EXIST; then
    echo "[2/3] All 11 sequence image dirs exist. Skipping download."
else
    echo "[2/3] Downloading color images (65GB)..."
    echo "  This will take 20-60 minutes depending on bandwidth."
    echo "  Download supports resume (-c) if interrupted."
    echo "  Started: $(date '+%H:%M:%S')"

    cd "${KITTI_ROOT}"
    wget -c --show-progress -O "${COLOR_ZIP}" \
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip"

    # Verify download integrity
    echo "  Verifying zip integrity..."
    if python3 -c "
import zipfile, sys
try:
    z = zipfile.ZipFile('${COLOR_ZIP}')
    bad = z.testzip()
    if bad: print(f'Corrupt file: {bad}'); sys.exit(1)
    print(f'ZIP OK: {len(z.namelist())} files')
except Exception as e:
    print(f'ZIP error: {e}'); sys.exit(1)
"; then
        echo "  [OK] Zip verified."
    else
        echo "  [ERROR] Zip is corrupted! Delete and re-download:"
        echo "    rm ${COLOR_ZIP}"
        echo "    bash eval/relpose/download_kitti.sh ${KITTI_ROOT}"
        exit 1
    fi

    echo "  Finished download: $(date '+%H:%M:%S')"
    cd - > /dev/null
fi

# --------------------------------------------------
# 3. Extract only sequences 00-10 (save ~27GB)
# --------------------------------------------------
if ! $ALL_EXIST; then
    echo "[3/3] Extracting sequences 00-10 from zip..."
    cd "${KITTI_ROOT}"

    for seq in $SEQS; do
        if [ -d "sequences/${seq}/image_2" ]; then
            n=$(ls "sequences/${seq}/image_2" | wc -l)
            echo "  [SKIP] seq ${seq} already extracted (${n} frames)"
            continue
        fi
        echo -n "  Extracting seq ${seq}..."
        unzip -q -o "${COLOR_ZIP}" "dataset/sequences/${seq}/image_2/*" -d . 2>/dev/null
        if [ -d "dataset/sequences/${seq}" ]; then
            mkdir -p "sequences/${seq}"
            mv "dataset/sequences/${seq}/image_2" "sequences/${seq}/image_2"
        fi
        n=$(ls "sequences/${seq}/image_2" 2>/dev/null | wc -l)
        echo " ${n} frames"
    done

    # Cleanup
    rm -rf dataset
    echo ""

    # Delete zip to save 65GB disk space
    echo "  Deleting zip to free 65GB..."
    rm -f "${COLOR_ZIP}"
    echo "  [OK] Zip removed."

    cd - > /dev/null
else
    echo "[3/3] All sequences already extracted. Skipping."
fi

# --------------------------------------------------
# Summary
# --------------------------------------------------
echo ""
echo "============================================"
echo "  Download Complete!"
echo "============================================"
echo ""
echo "  Sequence summary:"
total_frames=0
for seq in $SEQS; do
    img_dir="${KITTI_ROOT}/sequences/${seq}/image_2"
    pose_file="${KITTI_ROOT}/poses/${seq}.txt"
    if [ -d "$img_dir" ] && [ -f "$pose_file" ]; then
        n=$(ls "$img_dir" | wc -l)
        total_frames=$((total_frames + n))
        printf "    seq %s: %5d frames  ✓\n" "$seq" "$n"
    else
        printf "    seq %s: MISSING  ✗\n" "$seq"
    fi
done
echo ""
echo "  Total: ${total_frames} frames"
echo "  Disk usage: $(du -sh ${KITTI_ROOT} | cut -f1)"
echo ""
echo "  Next step: prepare data with"
echo "    python eval/relpose/prepare_kitti_odometry.py \\"
echo "      --kitti_root ${KITTI_ROOT} --seqs ${SEQS} --full"

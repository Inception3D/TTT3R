#!/bin/bash
# =============================================================================
# setup_7scenes.sh — Unzip, preprocess, and verify 7scenes data
#
# Steps:
#   1. Unzip all scene zips to data/7scenes/
#   2. Run SimpleRecon depth preprocessing (depth.proj.png)
#   3. Verify all test sequences are complete
#
# Usage:
#   bash setup_7scenes.sh
# =============================================================================

set -euo pipefail

export PATH="/root/miniconda3/bin:$PATH"

WORKDIR="/root/TTT3R"
DOWNLOAD_DIR="/root/autodl-tmp/7scenes_download"
DATA_DIR="/root/autodl-tmp/7scenes_data"
LINK_DIR="${WORKDIR}/data/7scenes"

SCENES=(chess fire heads office pumpkin redkitchen stairs)

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# Step 1: Unzip
log "=== Step 1: Unzip ==="
mkdir -p "${DATA_DIR}"
for scene in "${SCENES[@]}"; do
    zip="${DOWNLOAD_DIR}/${scene}.zip"
    if [ ! -f "$zip" ]; then
        log "MISSING: $zip"
        exit 1
    fi
    if [ -d "${DATA_DIR}/${scene}" ]; then
        log "  ${scene}: already extracted"
    else
        log "  Extracting ${scene}..."
        unzip -q "$zip" -d "${DATA_DIR}/"
        log "  ${scene}: done"
    fi
done

# Step 2: Symlink to TTT3R/data/7scenes
log "=== Step 2: Create symlink ==="
rm -f "${LINK_DIR}"
ln -sf "${DATA_DIR}" "${LINK_DIR}"
log "  ${LINK_DIR} -> ${DATA_DIR}"

# Step 3: Depth preprocessing (SimpleRecon)
log "=== Step 3: Depth preprocessing ==="
pip install -q scikit-image 2>/dev/null

python3 << 'PYEOF'
import os
import sys
import numpy as np
from pathlib import Path

try:
    from skimage.io import imread as skimage_imread
    from skimage.io import imsave
except ImportError:
    print("ERROR: scikit-image not installed. Run: pip install scikit-image")
    sys.exit(1)

DATA_DIR = "/root/autodl-tmp/7scenes_data"
SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]

# Depth camera intrinsics (from SimpleRecon)
depth_focal = 585.0
rgb_focal = 525.0
img_w, img_h = 640, 480

# Calibration matrix: depth sensor to RGB sensor (from SimpleRecon)
# This is the standard 7scenes depth-to-color registration
calib_matrix = np.array([
    [0.99998, 0.00147, 0.00613, 0.02520],
    [-0.00153, 0.99993, 0.01187, 0.00064],
    [-0.00611, -0.01188, 0.99991, -0.00709],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float64)

processed = 0
skipped = 0
errors = 0

for scene in SCENES:
    scene_dir = Path(DATA_DIR) / scene
    if not scene_dir.exists():
        print(f"  MISSING: {scene_dir}")
        continue

    # Find all sequence directories
    for seq_dir in sorted(scene_dir.iterdir()):
        if not seq_dir.is_dir() or not seq_dir.name.startswith("seq-"):
            continue

        # Find all depth files
        depth_files = sorted(seq_dir.glob("frame-*.depth.png"))

        for depth_path in depth_files:
            proj_path = depth_path.parent / depth_path.name.replace(".depth.png", ".depth.proj.png")

            if proj_path.exists():
                skipped += 1
                continue

            try:
                # Read depth (uint16, mm)
                depth_raw = skimage_imread(str(depth_path)).astype(np.float64)
                depth_m = depth_raw / 1000.0  # to meters

                # Create pixel coordinates for depth image
                u_depth = np.arange(img_w).reshape(1, -1).repeat(img_h, axis=0)
                v_depth = np.arange(img_h).reshape(-1, 1).repeat(img_w, axis=1)

                # Unproject depth pixels to 3D (depth camera frame)
                z = depth_m
                x = (u_depth - img_w / 2.0) * z / depth_focal
                y = (v_depth - img_h / 2.0) * z / depth_focal

                # Stack to Nx4 homogeneous
                points = np.stack([x, y, z, np.ones_like(z)], axis=-1)
                shape = points.shape[:2]
                points_flat = points.reshape(-1, 4)

                # Transform to RGB camera frame
                points_rgb = (calib_matrix @ points_flat.T).T
                points_rgb = points_rgb.reshape(*shape, 4)

                # Project to RGB image plane
                z_rgb = points_rgb[..., 2]
                valid = z_rgb > 0
                u_rgb = (points_rgb[..., 0] / z_rgb * rgb_focal + img_w / 2.0)
                v_rgb = (points_rgb[..., 1] / z_rgb * rgb_focal + img_h / 2.0)

                # Create projected depth map
                proj_depth = np.zeros((img_h, img_w), dtype=np.uint16)

                u_int = np.round(u_rgb).astype(np.int32)
                v_int = np.round(v_rgb).astype(np.int32)

                mask = valid & (u_int >= 0) & (u_int < img_w) & (v_int >= 0) & (v_int < img_h)
                mask = mask & (z_rgb > 0) & (z_rgb < 100)

                depth_mm = (z_rgb * 1000).astype(np.uint16)

                valid_mask = mask.ravel()
                target_u = u_int.ravel()[valid_mask]
                target_v = v_int.ravel()[valid_mask]
                target_d = depth_mm.ravel()[valid_mask]

                # Use simple assignment (last write wins, close enough for evaluation)
                proj_depth[target_v, target_u] = target_d

                imsave(str(proj_path), proj_depth)
                processed += 1

            except Exception as e:
                print(f"  ERROR: {depth_path}: {e}")
                errors += 1

        if processed % 100 == 0 and processed > 0:
            print(f"  Processed: {processed}, Skipped: {skipped}")

print(f"\nDepth preprocessing complete:")
print(f"  Processed: {processed}")
print(f"  Skipped (already exist): {skipped}")
print(f"  Errors: {errors}")
PYEOF

# Step 4: Verify
log "=== Step 4: Verify test sequences ==="
TEST_SEQS=(
    "chess/seq-03" "chess/seq-05"
    "fire/seq-03" "fire/seq-04"
    "heads/seq-01"
    "office/seq-02" "office/seq-06" "office/seq-07" "office/seq-09"
    "pumpkin/seq-01" "pumpkin/seq-07"
    "redkitchen/seq-03" "redkitchen/seq-04" "redkitchen/seq-06" "redkitchen/seq-12" "redkitchen/seq-14"
    "stairs/seq-01" "stairs/seq-04"
)

ok=0
fail=0
for seq in "${TEST_SEQS[@]}"; do
    seq_dir="${DATA_DIR}/${seq}"
    if [ -d "$seq_dir" ]; then
        n_color=$(ls "${seq_dir}"/frame-*.color.png 2>/dev/null | wc -l)
        n_depth=$(ls "${seq_dir}"/frame-*.depth.proj.png 2>/dev/null | wc -l)
        n_pose=$(ls "${seq_dir}"/frame-*.pose.txt 2>/dev/null | wc -l)
        if [ "$n_color" -gt 0 ] && [ "$n_depth" -gt 0 ] && [ "$n_pose" -gt 0 ]; then
            log "  OK: ${seq} (${n_color} color, ${n_depth} depth.proj, ${n_pose} pose)"
            ok=$((ok + 1))
        else
            log "  INCOMPLETE: ${seq} (color=${n_color}, depth.proj=${n_depth}, pose=${n_pose})"
            fail=$((fail + 1))
        fi
    else
        log "  MISSING: ${seq}"
        fail=$((fail + 1))
    fi
done

log ""
log "=== Verification: ${ok}/18 OK, ${fail} issues ==="
log "Data: ${DATA_DIR}"
log "Symlink: ${LINK_DIR}"
log "Disk: $(du -sh ${DATA_DIR} | cut -f1)"
log ""
if [ "$fail" -eq 0 ]; then
    log "Ready to run: bash eval/mv_recon/run_7scenes_allconfigs.sh"
else
    log "WARNING: ${fail} sequences have issues"
fi

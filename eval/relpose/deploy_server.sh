#!/bin/bash
# =============================================================================
# Full deployment script for KITTI Odometry evaluation on a fresh GPU server
#
# This script handles EVERYTHING from scratch:
#   1. Clone repo (zjc branch)
#   2. Replace model.py with the version supporting all 5 methods
#   3. Download & prepare KITTI data (sequences 00-10, full length)
#   4. Transfer model weights (requires SCP from local)
#   5. Run all 55 experiments (11 seqs × 5 methods)
#   6. Pack results for download
#
# Usage on server:
#   # First: clone repo and run this script
#   git clone -b zjc git@github.com:szy233/TTT3R.git
#   cd TTT3R
#   bash eval/relpose/deploy_server.sh
#
# Or one-liner after SSH:
#   git clone -b zjc git@github.com:szy233/TTT3R.git && cd TTT3R && bash eval/relpose/deploy_server.sh
# =============================================================================

set -e

WORKDIR=$(pwd)
LOG="${WORKDIR}/kitti_full_deploy.log"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"
}

log "=== KITTI Full-Sequence Deployment Started ==="
log "Working directory: ${WORKDIR}"

# --------------------------------------------------
# Step 1: Replace model.py with all-methods version
# --------------------------------------------------
log "[Step 1] Setting up model.py with all 5 update methods..."
if [ -f "eval/relpose/model_all_methods.py.bak" ]; then
    cp src/dust3r/model.py src/dust3r/model.py.zjc_backup
    cp eval/relpose/model_all_methods.py.bak src/dust3r/model.py
    log "[OK] model.py replaced (backup at model.py.zjc_backup)"
else
    log "[WARN] model_all_methods.py.bak not found, using current model.py"
    log "       ttt3r_ortho and ttt3r_random may not be available!"
fi

# --------------------------------------------------
# Step 2: Install dependencies
# --------------------------------------------------
log "[Step 2] Installing Python dependencies..."
pip install numpy==1.26.4 scipy evo transformers==4.38.2 accelerate 2>&1 | tail -5 | tee -a "$LOG"
log "[OK] Dependencies installed."

# --------------------------------------------------
# Step 3: Check model weights
# --------------------------------------------------
log "[Step 3] Checking model weights..."
if [ ! -f "src/cut3r_512_dpt_4_64.pth" ]; then
    log "[WAIT] Model weights not found. Please transfer them:"
    log "  From local machine run:"
    log "    scp -P <PORT> /path/to/cut3r_512_dpt_4_64.pth root@<SERVER_IP>:${WORKDIR}/src/"
    log ""
    log "  Waiting for model weights to appear..."
    while [ ! -f "src/cut3r_512_dpt_4_64.pth" ]; do
        sleep 10
    done
    log "[OK] Model weights detected!"
else
    log "[OK] Model weights found."
fi

# --------------------------------------------------
# Step 4: Download & prepare KITTI data
# --------------------------------------------------
log "[Step 4] Setting up KITTI data..."
bash eval/relpose/setup_kitti_full.sh 2>&1 | tee -a "$LOG"

# --------------------------------------------------
# Step 5: Run all experiments
# --------------------------------------------------
log "[Step 5] Starting experiments (5 methods × 11 sequences)..."
log "  Methods: cut3r, ttt3r, ttt3r_random, ttt3r_momentum, ttt3r_ortho"
log "  Sequences: 00-10 (full length)"
log ""

bash eval/relpose/run_kitti_odo_full.sh 2>&1 | tee -a "$LOG"

# --------------------------------------------------
# Step 6: Pack results
# --------------------------------------------------
log "[Step 6] Packing results..."
RESULT_DIR="eval_results/relpose/kitti_odo_full"
PACK_NAME="kitti_odo_full_results_$(date '+%Y%m%d_%H%M').tar.gz"

if [ -d "$RESULT_DIR" ]; then
    tar czf "$PACK_NAME" "$RESULT_DIR"
    log "[OK] Results packed: ${PACK_NAME} ($(du -h ${PACK_NAME} | cut -f1))"
    log ""
    log "  Download with:"
    log "    scp -P <PORT> root@<SERVER_IP>:${WORKDIR}/${PACK_NAME} ."
else
    log "[ERROR] Result directory not found!"
fi

log ""
log "=== Deployment Complete ==="
log "Total time: started at first log entry, ended at $(date)"

#!/bin/bash

MODEL_PATH="src/cut3r_512_dpt_4_64.pth"
SEQ_PATH="/home/chen/Desktop/video.mp4"
BASE_OUT="./experiments/matrix1"
PORT=8080
SIZE=512

for UPDATE_TYPE in cut3r ttt3r
do
  for FRAME_INTERVAL in  2 5
  do
    for RESET_INTERVAL in 100
    do
      OUT_DIR="${BASE_OUT}/${UPDATE_TYPE}_fi${FRAME_INTERVAL}_rs${RESET_INTERVAL}"
      LOG_FILE="${OUT_DIR}/run.log"

      mkdir -p "$OUT_DIR"

      echo "========================================"
      echo "Running: $UPDATE_TYPE fi=$FRAME_INTERVAL rs=$RESET_INTERVAL"
      echo "Output: $OUT_DIR"
      echo "========================================"

      python -u demo.py \
        --model_path "$MODEL_PATH" \
        --size "$SIZE" \
        --seq_path "$SEQ_PATH" \
        --output_dir "$OUT_DIR" \
        --port "$PORT" \
        --model_update_type "$UPDATE_TYPE" \
        --frame_interval "$FRAME_INTERVAL" \
        --reset_interval "$RESET_INTERVAL" \
        > "$LOG_FILE" 2>&1 &

      PID=$!

      while kill -0 "$PID" 2>/dev/null; do
        if grep -q "Launching point cloud viewer..." "$LOG_FILE"; then
          echo "Inference finished for ${UPDATE_TYPE}_fi${FRAME_INTERVAL}_rs${RESET_INTERVAL}, stopping viewer..."
          kill -INT "$PID"
          wait "$PID" 2>/dev/null
          break
        fi
        sleep 2
      done

      echo "Finished: ${UPDATE_TYPE}_fi${FRAME_INTERVAL}_rs${RESET_INTERVAL}"
      echo


    done
  done
done

#!/bin/bash

set -euo pipefail

export DATASET="${DATASET:-nuscenes_relpose}"
bash eval/relpose/run_relpose_dataset.sh

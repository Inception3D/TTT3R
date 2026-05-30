#!/bin/bash

set -euo pipefail

export DATASET="${DATASET:-waymo_relpose}"
bash eval/relpose/run_relpose_dataset.sh

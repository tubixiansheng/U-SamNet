#!/usr/bin/env bash
set -euo pipefail

weights="${1:?Usage: scripts/infer_example.sh /path/to/best.pth /path/to/images [output_dir] [gpu_id]}"
images="${2:?Usage: scripts/infer_example.sh /path/to/best.pth /path/to/images [output_dir] [gpu_id]}"
output_dir="${3:-outputs/BUSI}"
gpu_id="${4:-0}"

python test.py \
  --dataset BUSI \
  --weights "$weights" \
  --input "$images" \
  --output-dir "$output_dir" \
  --gpu-id "$gpu_id"

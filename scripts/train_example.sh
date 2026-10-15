#!/usr/bin/env bash
set -euo pipefail

dataset_root="${1:?Usage: scripts/train_example.sh /path/to/BUSI [gpu_id]}"
gpu_id="${2:-0}"

python train.py \
  --dataset BUSI \
  --dataset-root "$dataset_root" \
  --gpu-id "$gpu_id" \
  --epochs 140 \
  --batch-size 1

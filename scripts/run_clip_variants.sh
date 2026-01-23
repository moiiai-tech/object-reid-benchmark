#!/bin/bash
# Benchmark different CLIP model variants
# Usage: ./scripts/run_clip_variants.sh [gpu_id]

set -e

# Change to project root
cd "$(dirname "$0")/.."

GPU_ID=${1:-7}

echo "Benchmarking CLIP model variants..."
echo "Using GPU: $GPU_ID"

python run_benchmark.py --config benchmark/configs/clip_variants.yaml gpu_id=$GPU_ID

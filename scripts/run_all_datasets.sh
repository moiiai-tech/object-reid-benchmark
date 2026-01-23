#!/bin/bash
# Run benchmark on all datasets in your collection
# Usage: ./scripts/run_all_datasets.sh [gpu_id]

set -e

# Change to project root
cd "$(dirname "$0")/.."

GPU_ID=${1:-7}

echo "Running comprehensive benchmark on all datasets..."
echo "Using GPU: $GPU_ID"

python run_benchmark.py --config benchmark/configs/all_my_datasets.yaml gpu_id=$GPU_ID

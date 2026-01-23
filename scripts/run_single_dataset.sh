#!/bin/bash
# Run benchmark on a single dataset
# Usage: ./scripts/run_single_dataset.sh <dataset_name> [gpu_id]
#
# Example: ./scripts/run_single_dataset.sh market1501 7

set -e

# Change to project root
cd "$(dirname "$0")/.."

if [ -z "$1" ]; then
    echo "Error: Dataset name required"
    echo "Usage: $0 <dataset_name> [gpu_id]"
    echo ""
    echo "Available datasets:"
    echo "  - market1501"
    echo "  - dukemtmcreid"
    echo "  - cuhk03"
    echo "  - msmt17"
    echo "  - celebreid"
    echo "  - pku"
    echo "  - lastid"
    echo "  - entireid"
    echo "  - grid"
    echo "  - viper"
    echo "  - ilids"
    echo "  - prid2011"
    echo "  ... and more (see benchmark/datasets/registry.py)"
    exit 1
fi

DATASET=$1
GPU_ID=${2:-7}

echo "Running benchmark on $DATASET..."
echo "Using GPU: $GPU_ID"

python run_benchmark.py \
    gpu_id=$GPU_ID \
    "datasets=[$DATASET]" \
    "output.csv_filename=${DATASET}_benchmark.csv"

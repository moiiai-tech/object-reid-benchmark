#!/bin/bash
# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
# Run cross-domain person re-identification benchmark

set -e

echo "=========================================="
echo "Cross-Domain Person Re-ID Benchmark"
echo "=========================================="
echo ""
echo "This evaluates models trained on one dataset"
echo "across multiple target datasets to test"
echo "domain generalization capabilities."
echo ""

# Check if config name is provided (without path or .yaml extension)
CONFIG_NAME=${1:-cross_domain}

echo "Using configuration: $CONFIG_NAME (from benchmark/configs/)"
echo ""

# Run the benchmark
# Hydra uses --config-name for the filename (without .yaml extension)
python run_benchmark.py --config-name "$CONFIG_NAME" gpu_id=0

echo ""
echo "=========================================="
echo "Cross-domain benchmark complete!"
echo "Check the results directory for outputs."
echo "=========================================="

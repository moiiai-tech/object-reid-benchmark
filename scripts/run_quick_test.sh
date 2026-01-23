#!/bin/bash
# Quick test benchmark on Market1501 and DukeMTMC-reID
# Usage: ./scripts/run_quick_test.sh

set -e

# Change to project root
cd "$(dirname "$0")/.."

echo "Running quick benchmark test..."
python run_benchmark.py --config benchmark/configs/quick_test.yaml

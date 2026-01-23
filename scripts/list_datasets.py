#!/usr/bin/env python
# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""List all available datasets in the registry."""

import os
import sys
from pathlib import Path

# Add project root to path and change directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

# Import after setting up paths - noqa: E402
from benchmark.datasets.registry import DATASET_REGISTRY  # noqa: E402


def main():
    print("Available datasets in registry:")
    print("=" * 80)

    custom_datasets = []
    standard_datasets = []

    for name, config in DATASET_REGISTRY.items():
        if config.is_custom:
            custom_datasets.append((name, config))
        else:
            standard_datasets.append((name, config))

    if standard_datasets:
        print("\nStandard datasets (using torchreid):")
        print("-" * 80)
        for name, config in sorted(standard_datasets):
            print(f"  {name:25s} - {config.height}x{config.width}")

    if custom_datasets:
        print("\nCustom datasets (using custom wrappers):")
        print("-" * 80)
        for name, config in sorted(custom_datasets):
            wrapper = config.custom_class.__name__ if config.custom_class else "Generic"
            print(f"  {name:25s} - {config.height}x{config.width} (wrapper: {wrapper})")

    print("\n" + "=" * 80)
    print(f"Total: {len(DATASET_REGISTRY)} datasets")

    print("\nUsage examples:")
    print("  python run_benchmark.py datasets=[market1501]")
    print("  python run_benchmark.py datasets=[market1501,dukemtmcreid]")
    print("  ./scripts/run_single_dataset.sh celebreid")
    print("  ./scripts/run_all_datasets.sh")


if __name__ == "__main__":
    main()

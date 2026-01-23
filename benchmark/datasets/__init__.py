# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Dataset management for person re-identification benchmarking."""

from .registry import DATASET_REGISTRY, DatasetConfig, load_dataset

__all__ = ["DatasetConfig", "DATASET_REGISTRY", "load_dataset"]

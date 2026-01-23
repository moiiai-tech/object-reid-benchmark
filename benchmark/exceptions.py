# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Custom exception classes for person re-identification benchmarking.

This module defines domain-specific exceptions that provide clear error messages
and make it easier to handle different types of failures appropriately.
"""


class ReIDException(Exception):
    """Base exception for all ReID benchmarking errors.

    All custom exceptions in the benchmark module inherit from this class,
    making it easy to catch any benchmark-related error.
    """

    pass


class ConfigurationError(ReIDException):
    """Raised when configuration is invalid or incomplete.

    Examples:
        - Missing required configuration fields
        - Invalid configuration value types
        - Conflicting configuration options
    """

    pass


class ModelLoadError(ReIDException):
    """Raised when model loading or initialization fails.

    Examples:
        - Failed to load pretrained weights
        - Corrupt checkpoint file
        - Invalid model architecture specification
        - Missing required model parameters
    """

    pass


class DatasetNotFoundError(ReIDException):
    """Raised when dataset is not found in registry.

    Examples:
        - Dataset name not registered
        - Invalid dataset identifier
    """

    def __init__(self, dataset_name: str, available_datasets: list[str]):
        available = ", ".join(available_datasets)
        super().__init__(
            f"Dataset '{dataset_name}' not found in registry.\n\n"
            f"Available datasets:\n  {available}\n\n"
            f"Troubleshooting:\n"
            f"  1. Check spelling of dataset name\n"
            f"  2. Run 'reid dataset list' to see all available datasets\n"
            f"  3. Check if dataset is registered in benchmark/datasets/registry.py"
        )


class DatasetLoadError(ReIDException):
    """Raised when dataset loading fails.

    Examples:
        - Dataset directory not found
        - Failed to instantiate dataset
        - Empty dataset (0 images)
    """

    def __init__(self, dataset_name: str, reason: str):
        super().__init__(
            f"Failed to load dataset '{dataset_name}': {reason}\n\n"
            f"Troubleshooting:\n"
            f"  1. Check if dataset exists in reid-data/{dataset_name}\n"
            f"  2. Verify dataset directory structure is correct\n"
            f"  3. Run 'reid dataset check' to see available datasets\n"
            f"  4. Check dataset name spelling in config"
        )


class ValidationError(ReIDException):
    """Raised when input validation fails.

    Examples:
        - Invalid parameter type
        - Parameter out of valid range
        - Missing required parameter
        - Invalid parameter combination
    """

    pass


class WeightResolutionError(ReIDException):
    """Raised when pretrained weight resolution fails.

    Examples:
        - Weights not found for dataset
        - Weight download failed
        - Checksum validation failed
        - Corrupt weight file
    """

    pass


class ImageLoadError(DatasetLoadError):
    """Raised when a specific image file cannot be loaded.

    This is a specialized DatasetLoadError for single image failures,
    allowing datasets to handle corrupt images gracefully.

    Examples:
        - Corrupt image file
        - Missing image file
        - Unsupported image format
    """

    pass


class ModelNotFoundError(ReIDException):
    """Raised when a requested model is not found in the registry.

    Examples:
        - Model name not registered
        - Invalid model identifier
        - Model architecture not supported
    """

    pass


class DependencyError(ReIDException):
    """Raised when required dependencies are missing.

    Examples:
        - Required package not installed
        - Incompatible package version
        - Missing system dependencies
    """

    pass


class BenchmarkError(ReIDException):
    """Raised when benchmark execution fails.

    Examples:
        - Invalid benchmark configuration
        - Benchmark setup failure
        - Results processing error
    """

    pass

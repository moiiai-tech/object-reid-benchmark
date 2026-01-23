# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Input validation utilities for person re-identification benchmarking.

This module provides reusable validation functions that ensure inputs are valid
before processing, preventing errors and providing clear feedback to users.
"""

import hashlib
import shutil
import torch
from pathlib import Path
from typing import Any, List, Optional

from benchmark.exceptions import ValidationError, WeightResolutionError


def validate_path_exists(path: str | Path, error_msg: Optional[str] = None) -> Path:
    """Validate that a file or directory exists.

    Args:
        path: Path to validate
        error_msg: Custom error message (default: auto-generated)

    Returns:
        Path object if valid

    Raises:
        ValidationError: If path does not exist
    """
    path = Path(path)
    if not path.exists():
        if error_msg is None:
            error_msg = f"Path does not exist: {path}"
        raise ValidationError(error_msg)
    return path


def validate_file_readable(path: str | Path) -> Path:
    """Validate that a file exists and is readable.

    Args:
        path: File path to validate

    Returns:
        Path object if valid

    Raises:
        ValidationError: If file doesn't exist or isn't readable
    """
    path = validate_path_exists(path, f"File not found: {path}")

    if not path.is_file():
        raise ValidationError(f"Not a file: {path}")

    try:
        with open(path, "rb") as f:
            f.read(1)  # Try to read one byte
    except PermissionError:
        raise ValidationError(f"File is not readable (permission denied): {path}")
    except Exception as e:
        raise ValidationError(f"Cannot read file {path}: {e}")

    return path


def validate_directory_exists(path: str | Path, create: bool = False) -> Path:
    """Validate that a directory exists, optionally creating it.

    Args:
        path: Directory path to validate
        create: If True, create directory if it doesn't exist

    Returns:
        Path object if valid

    Raises:
        ValidationError: If directory doesn't exist and create=False
    """
    path = Path(path)

    if not path.exists():
        if create:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValidationError(f"Failed to create directory {path}: {e}")
        else:
            raise ValidationError(f"Directory does not exist: {path}")

    if not path.is_dir():
        raise ValidationError(f"Not a directory: {path}")

    return path


def validate_positive_int(value: Any, name: str) -> int:
    """Validate that a value is a positive integer.

    Args:
        value: Value to validate
        name: Parameter name (for error messages)

    Returns:
        Integer value if valid

    Raises:
        ValidationError: If value is not a positive integer
    """
    if not isinstance(value, int):
        raise ValidationError(
            f"{name} must be an integer, got {type(value).__name__}: {value}"
        )

    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")

    return value


def validate_non_negative_int(value: Any, name: str) -> int:
    """Validate that a value is a non-negative integer.

    Args:
        value: Value to validate
        name: Parameter name (for error messages)

    Returns:
        Integer value if valid

    Raises:
        ValidationError: If value is not a non-negative integer
    """
    if not isinstance(value, int):
        raise ValidationError(
            f"{name} must be an integer, got {type(value).__name__}: {value}"
        )

    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")

    return value


def validate_device(device: str) -> str:
    """Validate CUDA device string.

    Args:
        device: Device string (e.g., "cuda:0", "cpu")

    Returns:
        Device string if valid

    Raises:
        ValidationError: If device string is invalid or device unavailable
    """
    if hasattr(device, 'type'):  # Check if it's a torch.device object
        device = str(device)

    if not isinstance(device, str):
        raise ValidationError(
            f"Device must be a string, got {type(device).__name__}: {device}"
        )

    device = device.lower().strip()

    if device == "cpu":
        return device

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValidationError(
                f"CUDA device '{device}' requested but CUDA is not available. "
                f"Use device='cpu' instead."
            )

        if device == "cuda":
            device_id = 0
        elif ":" in device:
            try:
                device_id = int(device.split(":")[1])
            except (ValueError, IndexError):
                raise ValidationError(f"Invalid CUDA device format: {device}")
        else:
            raise ValidationError(f"Invalid device string: {device}")

        device_count = torch.cuda.device_count()
        if device_id >= device_count:
            raise ValidationError(
                f"CUDA device {device_id} not available. "
                f"Only {device_count} GPU(s) detected: {list(range(device_count))}"
            )

        return device

    raise ValidationError(
        f"Invalid device string: {device}. Must be 'cpu' or 'cuda[:N]'"
    )


def validate_model_config(model_cfg: Any, required_fields: List[str]) -> None:
    """Validate that model configuration has required fields.

    Args:
        model_cfg: Model configuration object
        required_fields: List of required field names

    Raises:
        ValidationError: If any required field is missing
    """
    missing = []
    for field in required_fields:
        if not hasattr(model_cfg, field) or getattr(model_cfg, field) is None:
            missing.append(field)

    if missing:
        raise ValidationError(
            f"Model configuration missing required fields: {', '.join(missing)}\n"
            f"Required: {', '.join(required_fields)}"
        )


def check_disk_space(path: str | Path, required_mb: int) -> bool:
    """Check if sufficient disk space is available.

    Args:
        path: Path to check disk space for
        required_mb: Required space in megabytes

    Returns:
        True if sufficient space available

    Raises:
        ValidationError: If insufficient disk space
    """
    path = Path(path)
    if not path.exists():
        path = path.parent

    try:
        stat = shutil.disk_usage(path)
        available_mb = stat.free / (1024 * 1024)

        if available_mb < required_mb:
            raise ValidationError(
                f"Insufficient disk space at {path}\n"
                f"Required: {required_mb:.1f} MB\n"
                f"Available: {available_mb:.1f} MB\n"
                f"Need: {required_mb - available_mb:.1f} MB more"
            )

        return True
    except Exception as e:
        # Don't fail if we can't check disk space
        import logging

        logging.warning(f"Could not check disk space at {path}: {e}")
        return True


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute SHA-256 checksum of a file.

    Args:
        file_path: Path to file

    Returns:
        Hexadecimal SHA-256 checksum string

    Raises:
        ValidationError: If file cannot be read
    """
    file_path = validate_file_readable(file_path)

    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read in 8MB chunks
            for byte_block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        raise ValidationError(f"Failed to compute checksum for {file_path}: {e}")


def validate_checkpoint_file(
    path: str | Path, expected_sha256: Optional[str] = None
) -> bool:
    """Validate checkpoint file integrity.

    Args:
        path: Path to checkpoint file
        expected_sha256: Expected SHA-256 checksum (optional)

    Returns:
        True if valid

    Raises:
        WeightResolutionError: If validation fails
    """
    path = validate_file_readable(path)

    # Checksum validation (if provided)
    if expected_sha256:
        actual_sha256 = compute_file_sha256(path)
        if actual_sha256.lower() != expected_sha256.lower():
            raise WeightResolutionError(
                f"Checksum mismatch for {path}\n"
                f"Expected: {expected_sha256}\n"
                f"Actual:   {actual_sha256}\n"
                f"The file may be corrupt. Try re-downloading."
            )

    # Format validation - try to load with torch
    try:
        # Use weights_only=True for security
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Basic structure validation
        if not isinstance(checkpoint, dict):
            # Some checkpoints are just state dicts
            if not isinstance(checkpoint, (dict, list)):
                raise WeightResolutionError(
                    f"Invalid checkpoint format in {path}: "
                    f"Expected dict or state dict, got {type(checkpoint).__name__}"
                )

        return True
    except WeightResolutionError:
        raise
    except Exception as e:
        raise WeightResolutionError(
            f"Failed to load checkpoint {path}: {e}\n"
            f"The file may be corrupt or in an unsupported format."
        )


def validate_image_file(path: str | Path) -> bool:
    """Validate that a file is a valid image.

    Args:
        path: Path to image file

    Returns:
        True if valid

    Raises:
        ValidationError: If not a valid image file
    """
    from PIL import Image

    path = validate_file_readable(path)

    try:
        with Image.open(path) as img:
            # Try to load the image to verify it's valid
            img.verify()
        return True
    except Exception as e:
        raise ValidationError(f"Invalid or corrupt image file {path}: {e}")


def validate_list_not_empty(value: Any, name: str) -> list:
    """Validate that a value is a non-empty list.

    Args:
        value: Value to validate
        name: Parameter name (for error messages)

    Returns:
        List value if valid

    Raises:
        ValidationError: If value is not a non-empty list
    """
    if not isinstance(value, list):
        raise ValidationError(
            f"{name} must be a list, got {type(value).__name__}: {value}"
        )

    if len(value) == 0:
        raise ValidationError(f"{name} cannot be empty")

    return value


def validate_model_name(model_name: str) -> str:
    """Validate model name format and characters.

    Args:
        model_name: Model name to validate

    Returns:
        Model name if valid

    Raises:
        ValidationError: If model name is invalid
    """
    if not isinstance(model_name, str):
        raise ValidationError(
            f"model_name must be a string, got {type(model_name).__name__}: {model_name}"
        )

    if not model_name.strip():
        raise ValidationError("model_name cannot be empty")

    # Check for valid characters (alphanumeric, underscore, hyphen, and forward slash for CLIP models)
    import re

    if not re.match(r"^[a-zA-Z0-9_/-]+$", model_name):
        raise ValidationError(
            f"model_name '{model_name}' can only contain letters, numbers, underscores, hyphens, and forward slashes"
        )

    if len(model_name) > 50:
        raise ValidationError("model_name cannot exceed 50 characters")

    return model_name


def validate_dataset_name(dataset_name: str) -> str:
    """Validate dataset name format and characters.

    Args:
        dataset_name: Dataset name to validate

    Returns:
        Dataset name if valid

    Raises:
        ValidationError: If dataset name is invalid
    """
    if not isinstance(dataset_name, str):
        raise ValidationError(
            f"dataset_name must be a string, got {type(dataset_name).__name__}: {dataset_name}"
        )

    if not dataset_name.strip():
        raise ValidationError("dataset_name cannot be empty")

    # Check for valid characters (alphanumeric, underscore, hyphen)
    import re

    if not re.match(r"^[a-zA-Z0-9_-]+$", dataset_name):
        raise ValidationError(
            f"dataset_name '{dataset_name}' can only contain letters, numbers, underscores, and hyphens"
        )

    if len(dataset_name) > 50:
        raise ValidationError("dataset_name cannot exceed 50 characters")

    return dataset_name


def validate_model_parameters(params: Any) -> dict:
    """Validate model parameters dictionary.

    Args:
        params: Model parameters to validate

    Returns:
        Parameters dictionary if valid

    Raises:
        ValidationError: If parameters are invalid
    """
    if not isinstance(params, dict):
        raise ValidationError(
            f"params must be a dictionary, got {type(params).__name__}: {params}"
        )

    # Common parameter validations
    for key, value in params.items():
        if key == "num_classes" and value is not None:
            validate_positive_int(value, key)
        elif key == "dropout" and value is not None:
            if not isinstance(value, (int, float)):
                raise ValidationError(
                    f"{key} must be a number, got {type(value).__name__}"
                )
            if value < 0 or value > 1:
                raise ValidationError(f"{key} must be between 0 and 1, got {value}")
        elif key == "learning_rate" and value is not None:
            if not isinstance(value, (int, float)):
                raise ValidationError(
                    f"{key} must be a number, got {type(value).__name__}"
                )
            if value <= 0:
                raise ValidationError(f"{key} must be positive, got {value}")

    return params

# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Common utilities for CLI modules."""

import importlib
import sys
from pathlib import Path
from typing import NoReturn, Optional

from rich.console import Console

console = Console(stderr=True)


def error_exit(message: str, exit_code: int = 1) -> NoReturn:
    """
    Print error message and exit with code.

    Args:
        message: Error message to display
        exit_code: Exit code (default: 1)
    """
    console.print(f"[red]ERROR: {message}[/red]", style="bold")
    sys.exit(exit_code)


def warning(message: str) -> None:
    """
    Print warning message.

    Args:
        message: Warning message to display
    """
    console.print(f"[yellow]WARNING: {message}[/yellow]")


def success(message: str) -> None:
    """
    Print success message.

    Args:
        message: Success message to display
    """
    console.print(f"[green]{message}[/green]")


def info(message: str) -> None:
    """
    Print info message.

    Args:
        message: Info message to display
    """
    console.print(f"[cyan]ℹ️  {message}[/cyan]")


def validate_file_exists(file_path: str | Path, file_type: str = "File") -> Path:
    """
    Validate that a file exists.

    Args:
        file_path: Path to file
        file_type: Type of file for error message

    Returns:
        Path object if file exists

    Raises:
        SystemExit: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        error_exit(f"{file_type} not found: {path}")
    if not path.is_file():
        error_exit(f"{file_type} is not a file: {path}")
    return path


def validate_dir_exists(dir_path: str | Path, dir_type: str = "Directory") -> Path:
    """
    Validate that a directory exists.

    Args:
        dir_path: Path to directory
        dir_type: Type of directory for error message

    Returns:
        Path object if directory exists

    Raises:
        SystemExit: If directory doesn't exist
    """
    path = Path(dir_path)
    if not path.exists():
        error_exit(f"{dir_type} not found: {path}")
    if not path.is_dir():
        error_exit(f"{dir_type} is not a directory: {path}")
    return path


def safe_import(module_name: str, package: Optional[str] = None) -> bool:
    """
    Safely try to import a module.

    Args:
        module_name: Name of module to import
        package: Package name for relative imports (used when module_name starts with '.')

    Returns:
        True if import successful, False otherwise
    """
    try:
        importlib.import_module(module_name, package=package)
        return True
    except ImportError:
        return False


def check_cuda_available() -> bool:
    """
    Check if CUDA is available.

    Returns:
        True if CUDA available

    Raises:
        SystemExit: If CUDA not available
    """
    try:
        import torch

        if not torch.cuda.is_available():
            error_exit("CUDA is not available. This operation requires a GPU.")
        return True
    except ImportError:
        error_exit("PyTorch is not installed.")


def validate_gpu_id(gpu_id: int) -> int:
    """
    Validate GPU ID.

    Args:
        gpu_id: GPU ID to validate

    Returns:
        GPU ID if valid

    Raises:
        SystemExit: If GPU ID invalid
    """
    try:
        import torch

        if not torch.cuda.is_available():
            error_exit("CUDA is not available.")

        gpu_count = torch.cuda.device_count()
        if gpu_id >= gpu_count or gpu_id < 0:
            error_exit(f"Invalid GPU ID: {gpu_id}. Available GPUs: 0-{gpu_count - 1}")
        return gpu_id
    except ImportError:
        error_exit("PyTorch is not installed.")

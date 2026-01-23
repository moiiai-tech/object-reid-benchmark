# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Logging utilities for benchmarking with Rich formatting.

This module provides a centralized logging configuration using Rich library
for beautiful console output and file-based logging for debugging.
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Global console instance with custom theme
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "model": "bold magenta",
        "dataset": "bold blue",
        "metric": "bold yellow",
    }
)

console = Console(theme=custom_theme)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup a logger with Rich formatting.

    Args:
        name: Logger name (typically __name__ of the module)
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)

    Returns:
        Configured logger instance with Rich handler
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)

        # Use RichHandler for beautiful output
        handler = RichHandler(
            console=console,
            show_time=False,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
        )
        handler.setLevel(level)

        if format_string is None:
            format_string = "%(message)s"

        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False

    return logger


def get_model_logger(model_name: str) -> logging.Logger:
    """
    Get a logger configured for model wrappers.

    Args:
        model_name: Name of the model (e.g., 'CLIPReIDWrapper')

    Returns:
        Configured logger instance
    """
    return setup_logger(f"benchmark.models.{model_name}")


def get_dataset_logger(dataset_name: str) -> logging.Logger:
    """
    Get a logger configured for dataset wrappers.

    Args:
        dataset_name: Name of the dataset (e.g., 'MSMT17')

    Returns:
        Configured logger instance
    """
    return setup_logger(f"benchmark.datasets.{dataset_name}")


def setup_experiment_logging(
    experiment_name: str,
    results_dir: str = "results",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Setup logging for a benchmark experiment with both console and file output.

    This creates experiment-specific log files in results/{experiment_name}/logs/
    directory with automatic rotation. File logs contain more detailed information
    (DEBUG level) than console output (INFO level).

    Args:
        experiment_name: Name of the experiment (used for log directory)
        results_dir: Base directory for results (default: "results")
        console_level: Logging level for console output (default: INFO)
        file_level: Logging level for file output (default: DEBUG)

    Returns:
        Configured root logger with both console and file handlers

    Example:
        logger = setup_experiment_logging("market1501_osnet")
        logger.info("Starting benchmark...")  # Shows in console and file
        logger.debug("Detailed info...")      # Only in file
    """
    # Create log directory
    log_dir = Path(results_dir) / experiment_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"benchmark_{timestamp}.log"

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture everything

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler (Rich formatted)
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    # File handler (detailed, with rotation)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB per file
        backupCount=5,  # Keep 5 backup files
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log startup message
    logger.info(f"Experiment logging initialized: {experiment_name}")
    logger.info(f"Log file: {log_file}")

    return logger


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: dict,
) -> None:
    """
    Log an error with structured context information.

    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Dictionary of context information (dataset, model, etc.)

    Example:
        try:
            model = create_model(...)
        except Exception as e:
            log_error_with_context(
                logger, e,
                {"model_type": "clipreid", "dataset": "market1501"}
            )
    """
    logger.error(f"Error: {error}")
    if context:
        logger.error("Context:")
        for key, value in context.items():
            logger.error(f"  {key}: {value}")

    # Log stack trace at debug level
    logger.debug("Stack trace:", exc_info=True)


def log_validation_failure(
    logger: logging.Logger,
    field: str,
    value: any,
    expected: str,
) -> None:
    """
    Log a validation failure with details.

    Args:
        logger: Logger instance
        field: Field name that failed validation
        value: Invalid value
        expected: Description of expected value

    Example:
        log_validation_failure(
            logger,
            "num_classes",
            -1,
            "positive integer"
        )
    """
    logger.error(f"Validation failed for '{field}'")
    logger.error(f"  Value: {value} ({type(value).__name__})")
    logger.error(f"  Expected: {expected}")

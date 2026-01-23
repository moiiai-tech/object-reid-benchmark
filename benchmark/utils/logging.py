"""
Enhanced logging infrastructure for the object-reid benchmark system.

This module provides centralized logging configuration with file logging,
structured output, and proper error handling.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from ..exceptions import ConfigurationError


class BenchmarkLogger:
    """Centralized logger for benchmark operations."""

    _loggers: Dict[str, logging.Logger] = {}
    _configured = False

    @classmethod
    def configure_logging(
        cls,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        log_dir: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_console: bool = True,
        format_string: Optional[str] = None,
    ) -> None:
        """Configure logging for the benchmark system.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Specific log file name (if None, uses default)
            log_dir: Directory for log files (if None, uses ./logs)
            max_file_size: Maximum log file size in bytes
            backup_count: Number of backup files to keep
            enable_console: Whether to enable console logging
            format_string: Custom log format string

        Raises:
            ConfigurationError: If logging configuration fails
        """
        if cls._configured:
            return  # Already configured

        try:
            # Validate log level
            numeric_level = getattr(logging, log_level.upper(), None)
            if not isinstance(numeric_level, int):
                raise ConfigurationError(
                    "log_level", log_level, "must be a valid logging level"
                )

            # Set up log directory
            if log_dir:
                log_dir_path = Path(log_dir)
                log_dir_path.mkdir(parents=True, exist_ok=True)
            else:
                log_dir_path = Path("./logs")
                log_dir_path.mkdir(exist_ok=True)

            # Default format
            if format_string is None:
                format_string = (
                    "%(asctime)s - %(name)s - %(levelname)s - "
                    "%(filename)s:%(lineno)d - %(message)s"
                )

            formatter = logging.Formatter(format_string)

            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(numeric_level)

            root_logger.handlers.clear()

            if enable_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(numeric_level)
                console_handler.setFormatter(formatter)
                root_logger.addHandler(console_handler)

            if log_file:
                log_file_path = log_dir_path / log_file
            else:
                log_file_path = log_dir_path / "benchmark.log"

            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

            cls._configured = True

            # Log successful configuration
            logger = cls.get_logger("logging")
            logger.info(
                f"Logging configured - Level: {log_level}, File: {log_file_path}"
            )

        except Exception as e:
            raise ConfigurationError(
                "logging", "setup", f"Failed to configure logging: {e}"
            )

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger instance with the specified name.

        Args:
            name: Logger name (typically module name)

        Returns:
            Logger instance
        """
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)

            # Ensure logging is configured
            if not cls._configured:
                cls.configure_logging()

        return cls._loggers[name]

    @classmethod
    def log_exception(
        cls, logger: logging.Logger, message: str, exception: Exception
    ) -> None:
        """Log an exception with full context.

        Args:
            logger: Logger instance
            message: Error message
            exception: Exception that occurred
        """
        logger.error(
            f"{message}: {type(exception).__name__}: {exception}", exc_info=True
        )

    @classmethod
    def log_model_creation(
        cls,
        logger: logging.Logger,
        model_name: str,
        success: bool,
        details: Optional[str] = None,
    ) -> None:
        """Log model creation attempts.

        Args:
            logger: Logger instance
            model_name: Name of the model being created
            success: Whether model creation succeeded
            details: Additional details about the operation
        """
        status = "SUCCESS" if success else "FAILED"
        message = f"Model creation {status}: {model_name}"
        if details:
            message += f" - {details}"

        if success:
            logger.info(message)
        else:
            logger.error(message)

    @classmethod
    def log_dataset_loading(
        cls,
        logger: logging.Logger,
        dataset_name: str,
        success: bool,
        details: Optional[str] = None,
    ) -> None:
        """Log dataset loading attempts.

        Args:
            logger: Logger instance
            dataset_name: Name of the dataset being loaded
            success: Whether dataset loading succeeded
            details: Additional details about the operation
        """
        status = "SUCCESS" if success else "FAILED"
        message = f"Dataset loading {status}: {dataset_name}"
        if details:
            message += f" - {details}"

        if success:
            logger.info(message)
        else:
            logger.error(message)

    @classmethod
    def log_weight_resolution(
        cls,
        logger: logging.Logger,
        weight_path: str,
        success: bool,
        details: Optional[str] = None,
    ) -> None:
        """Log weight resolution attempts.

        Args:
            logger: Logger instance
            weight_path: Path to weight file
            success: Whether weight resolution succeeded
            details: Additional details about the operation
        """
        status = "SUCCESS" if success else "FAILED"
        message = f"Weight resolution {status}: {weight_path}"
        if details:
            message += f" - {details}"

        if success:
            logger.info(message)
        else:
            logger.error(message)


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return BenchmarkLogger.get_logger(name)


def setup_logging(**kwargs) -> None:
    """Convenience function to set up logging.

    Args:
        **kwargs: Arguments passed to BenchmarkLogger.configure_logging
    """
    BenchmarkLogger.configure_logging(**kwargs)

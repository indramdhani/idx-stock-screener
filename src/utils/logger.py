# -*- coding: utf-8 -*-
"""
Logger Utility for Indonesian Stock Screener
===========================================

Centralized logging configuration and utilities for the Indonesian Stock Screener.
Provides structured logging with file rotation, colored console output, and
configurable log levels.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


class LoggerConfig:
    """Logger configuration management."""

    def __init__(
        self,
        log_level: str = "INFO",
        log_dir: Optional[Path] = None,
        console_output: bool = True,
        file_output: bool = True,
        rotation: str = "1 day",
        retention: str = "30 days"
    ):
        """
        Initialize logger configuration.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_dir: Directory for log files (defaults to logs/ in project root)
            console_output: Enable console logging
            file_output: Enable file logging
            rotation: Log rotation policy
            retention: Log retention policy
        """
        self.log_level = log_level
        self.log_dir = log_dir or Path(__file__).parent.parent.parent / "logs"
        self.console_output = console_output
        self.file_output = file_output
        self.rotation = rotation
        self.retention = retention

        # Ensure log directory exists
        self.log_dir.mkdir(exist_ok=True)

    def setup_logger(self) -> None:
        """Setup logger with configured parameters."""
        # Remove default logger
        logger.remove()

        # Console logging
        if self.console_output:
            logger.add(
                sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level=self.log_level,
                colorize=True
            )

        # File logging
        if self.file_output:
            # Main application log
            logger.add(
                self.log_dir / "screener_{time:YYYY-MM-DD}.log",
                rotation=self.rotation,
                retention=self.retention,
                level="DEBUG",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                enqueue=True  # Thread-safe logging
            )

            # Error-only log
            logger.add(
                self.log_dir / "errors_{time:YYYY-MM-DD}.log",
                rotation=self.rotation,
                retention=self.retention,
                level="ERROR",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
                enqueue=True
            )

    def get_logger(self, name: str):
        """
        Get a logger instance for a specific module.

        Args:
            name: Logger name (typically __name__)

        Returns:
            Logger instance
        """
        return logger.bind(name=name)


# Global logger configuration
_logger_config: Optional[LoggerConfig] = None


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    console_output: bool = True,
    file_output: bool = True,
    rotation: str = "1 day",
    retention: str = "30 days"
) -> None:
    """
    Setup global logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        console_output: Enable console logging
        file_output: Enable file logging
        rotation: Log rotation policy
        retention: Log retention policy
    """
    global _logger_config

    _logger_config = LoggerConfig(
        log_level=log_level,
        log_dir=log_dir,
        console_output=console_output,
        file_output=file_output,
        rotation=rotation,
        retention=retention
    )

    _logger_config.setup_logger()


def get_logger(name: str = "screener"):
    """
    Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if _logger_config is None:
        setup_logging()

    return logger.bind(name=name)


class LogContext:
    """Context manager for structured logging."""

    def __init__(self, operation: str, **kwargs):
        """
        Initialize log context.

        Args:
            operation: Operation name
            **kwargs: Additional context variables
        """
        self.operation = operation
        self.context = kwargs
        self.start_time = None

    def __enter__(self):
        """Enter context."""
        import time
        self.start_time = time.time()

        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
        logger.info(f"Starting {self.operation}" + (f" ({context_str})" if context_str else ""))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        import time
        elapsed = time.time() - self.start_time if self.start_time else 0

        if exc_type is None:
            logger.info(f"Completed {self.operation} in {elapsed:.2f}s")
        else:
            logger.error(f"Failed {self.operation} after {elapsed:.2f}s: {exc_val}")

        return False  # Don't suppress exceptions


def log_function_call(func):
    """
    Decorator to log function calls with execution time.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        import time

        func_name = f"{func.__module__}.{func.__name__}" if hasattr(func, '__module__') else func.__name__
        logger.debug(f"Calling {func_name}")

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"Completed {func_name} in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed {func_name} after {elapsed:.3f}s: {e}")
            raise

    return wrapper


def log_async_function_call(func):
    """
    Decorator to log async function calls with execution time.

    Args:
        func: Async function to decorate

    Returns:
        Decorated async function
    """
    from functools import wraps

    @wraps(func)
    async def wrapper(*args, **kwargs):
        import time

        func_name = f"{func.__module__}.{func.__name__}" if hasattr(func, '__module__') else func.__name__
        logger.debug(f"Calling {func_name}")

        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"Completed {func_name} in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed {func_name} after {elapsed:.3f}s: {e}")
            raise

    return wrapper


# Convenience functions
def log_data_quality(symbol: str, quality_score: float, issues: list = None):
    """Log data quality information."""
    issues_str = f" (issues: {', '.join(issues)})" if issues else ""
    logger.info(f"Data quality for {symbol}: {quality_score:.2f}{issues_str}")


def log_signal_generated(symbol: str, signal_type: str, confidence: float):
    """Log signal generation."""
    logger.info(f"Signal generated: {symbol} [{signal_type}] confidence={confidence:.1%}")


def log_screening_results(total_stocks: int, signals_found: int, execution_time: float):
    """Log screening results summary."""
    logger.info(f"Screening completed: {signals_found} signals from {total_stocks} stocks in {execution_time:.2f}s")


def log_market_status(is_open: bool, next_event: str, minutes_to_event: int):
    """Log market status information."""
    status = "OPEN" if is_open else "CLOSED"
    logger.info(f"Market status: {status} - Next {next_event} in {minutes_to_event} minutes")


# Initialize default logging on import
if _logger_config is None:
    setup_logging()

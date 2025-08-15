# -*- coding: utf-8 -*-
"""
Utilities Package for Indonesian Stock Screener
==============================================

This package provides utility functions and helpers for the Indonesian Stock Screener,
including logging configuration, data processing helpers, and common utilities.
"""

from .logger import (
    setup_logging,
    get_logger,
    LogContext,
    LoggerConfig,
    log_function_call,
    log_async_function_call,
    log_data_quality,
    log_signal_generated,
    log_screening_results,
    log_market_status,
)

__all__ = [
    # Logger utilities
    "setup_logging",
    "get_logger",
    "LogContext",
    "LoggerConfig",
    "log_function_call",
    "log_async_function_call",

    # Logging convenience functions
    "log_data_quality",
    "log_signal_generated",
    "log_screening_results",
    "log_market_status",
]

__version__ = "1.0.0"
__author__ = "Indonesian Stock Screener Team"
__description__ = "Utility functions and helpers for Indonesian stock screening"

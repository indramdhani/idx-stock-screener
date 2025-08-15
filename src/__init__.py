# -*- coding: utf-8 -*-
"""
Indonesian Stock Screener - Main Package
========================================

This package provides a comprehensive Indonesian stock screening system
with modular architecture, automated data collection, technical analysis,
and risk management capabilities.

Modules:
    config: Configuration management with Pydantic validation
    data: Data collection, models, and validation
    analysis: Technical analysis and screening logic
    notifications: Telegram and other notification systems
    scheduler: Automated workflow scheduling
    utils: Utility functions and helpers
"""

__version__ = "1.0.0"
__author__ = "Indonesian Stock Screener Team"
__description__ = "Automated Indonesian stock screening and analysis system"
__license__ = "MIT"

# Core imports for easy access
from .config import TradingConfig, load_config
from .data import IDXDataCollector, StockData, TradingSignal

__all__ = [
    "TradingConfig",
    "load_config",
    "IDXDataCollector",
    "StockData",
    "TradingSignal",
]

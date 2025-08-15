# -*- coding: utf-8 -*-
"""
Data Collectors Package for Indonesian Stock Screener
====================================================

This package contains data collection utilities for fetching Indonesian stock
market data from various sources, with built-in validation and quality checks.
"""

from .idx_collector import IDXDataCollector, DataCollectionError
from .data_validator import (
    StockDataValidator,
    ValidationRule,
    DataCompletenessRule,
    DataFreshnessRule,
    PriceValidityRule,
    VolumeValidityRule,
    PriceAnomalyRule,
    GapDetectionRule,
    DataQualityRule,
)

__all__ = [
    # Main data collector
    "IDXDataCollector",
    "DataCollectionError",

    # Data validator and rules
    "StockDataValidator",
    "ValidationRule",
    "DataCompletenessRule",
    "DataFreshnessRule",
    "PriceValidityRule",
    "VolumeValidityRule",
    "PriceAnomalyRule",
    "GapDetectionRule",
    "DataQualityRule",
]

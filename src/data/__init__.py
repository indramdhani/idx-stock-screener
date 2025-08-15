# -*- coding: utf-8 -*-
"""
Data Package for Indonesian Stock Screener
==========================================

This package provides data collection, validation, and modeling functionality
for the Indonesian Stock Screener. It includes modules for:

- Stock data models and structures
- Data collection from various sources (Yahoo Finance, IDX API)
- Data validation and quality checks
- Trading signal models and structures
"""

from .models import (
    # Stock data models
    StockInfo,
    StockPriceData,
    TechnicalIndicators,
    StockData,
    StockScreeningResult,

    # Signal models
    SignalType,
    SignalStatus,
    RiskLevel,
    TakeProfitLevel,
    RiskParameters,
    PositionSizing,
    SignalContext,
    TradingSignal,
    SignalPerformanceMetrics,
)

from .collectors import (
    # Data collector
    IDXDataCollector,
    DataCollectionError,

    # Data validator
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
    # Stock data models
    "StockInfo",
    "StockPriceData",
    "TechnicalIndicators",
    "StockData",
    "StockScreeningResult",

    # Signal models
    "SignalType",
    "SignalStatus",
    "RiskLevel",
    "TakeProfitLevel",
    "RiskParameters",
    "PositionSizing",
    "SignalContext",
    "TradingSignal",
    "SignalPerformanceMetrics",

    # Data collection
    "IDXDataCollector",
    "DataCollectionError",

    # Data validation
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

__version__ = "1.0.0"
__author__ = "Indonesian Stock Screener Team"
__description__ = "Data management package for Indonesian stock screening and analysis"

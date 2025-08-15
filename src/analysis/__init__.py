# -*- coding: utf-8 -*-
"""
Analysis Package for Indonesian Stock Screener
==============================================

This package provides technical analysis, stock screening, and risk management
functionality for Indonesian stock market analysis. It combines multiple
technical indicators to generate high-quality trading signals with proper
risk management.

Modules:
    indicators: Technical indicator implementations (VWAP, ATR, RSI, EMA)
    screener: Main stock screening engine with strategy implementations
    risk_calculator: Position sizing and risk management calculations
"""

from .indicators import (
    VWAP,
    VWAPAnalyzer,
    ATR,
    ATRAnalyzer,
    RSI,
    RSIAnalyzer,
    EMA,
    EMAAnalyzer,
    calculate_all_indicators,
    get_trading_signals,
)

from .screener import StockScreener, ScreeningCriteria
from .risk_calculator import RiskCalculator

__all__ = [
    # Technical Indicators
    "VWAP",
    "VWAPAnalyzer",
    "ATR",
    "ATRAnalyzer",
    "RSI",
    "RSIAnalyzer",
    "EMA",
    "EMAAnalyzer",
    "calculate_all_indicators",
    "get_trading_signals",

    # Stock Screening
    "StockScreener",
    "ScreeningCriteria",

    # Risk Management
    "RiskCalculator",
]

__version__ = "1.0.0"
__author__ = "Indonesian Stock Screener Team"
__description__ = "Technical analysis and stock screening for Indonesian stock market"

# Convenience function for quick analysis
def analyze_stock(stock_data, config=None):
    """
    Perform comprehensive analysis on a single stock.

    Args:
        stock_data: StockData object with OHLCV data
        config: Optional configuration for analysis parameters

    Returns:
        Dictionary with complete analysis results including indicators and signals
    """
    if not stock_data or stock_data.daily_data.empty:
        return {'error': 'No stock data provided or data is empty'}

    try:
        # Get all technical indicators
        indicators = calculate_all_indicators(stock_data.daily_data, config)

        # Get consolidated trading signals
        signals = get_trading_signals(stock_data.daily_data, config)

        # Calculate basic stock metrics
        stock_metrics = {
            'symbol': stock_data.symbol,
            'current_price': stock_data.current_price,
            'daily_change': stock_data.daily_change,
            'daily_change_pct': stock_data.daily_change_pct,
            'volume': stock_data.daily_volume,
            'volume_ratio': stock_data.volume_ratio,
            'market_cap': stock_data.market_cap,
            'data_quality_score': stock_data.get_data_quality_score()
        }

        return {
            'stock_metrics': stock_metrics,
            'technical_indicators': indicators,
            'trading_signals': signals,
            'analysis_timestamp': stock_data.last_updated.isoformat()
        }

    except Exception as e:
        return {'error': f'Analysis failed: {str(e)}'}


def screen_stocks(stocks_data, config, strategy='intraday', account_balance=100_000_000):
    """
    Screen multiple stocks for trading opportunities.

    Args:
        stocks_data: Dictionary of StockData objects
        config: TradingConfig object
        strategy: Screening strategy ('intraday' or 'overnight')
        account_balance: Account balance for position sizing

    Returns:
        List of TradingSignal objects
    """
    if not stocks_data:
        return []

    try:
        screener = StockScreener(config)

        if strategy == 'intraday':
            return screener.screen_intraday_rebounds(stocks_data, account_balance)
        elif strategy == 'overnight':
            return screener.screen_overnight_setups(stocks_data, account_balance)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    except Exception as e:
        from loguru import logger
        logger.error(f"Stock screening failed: {e}")
        return []


def calculate_portfolio_risk(signals, account_balance, config):
    """
    Calculate risk metrics for a portfolio of signals.

    Args:
        signals: List of TradingSignal objects
        account_balance: Account balance
        config: TradingConfig object

    Returns:
        Dictionary with portfolio risk analysis
    """
    if not signals:
        return {'error': 'No signals provided'}

    try:
        risk_calculator = RiskCalculator(config)
        return risk_calculator.calculate_risk_metrics(signals, account_balance)

    except Exception as e:
        return {'error': f'Risk calculation failed: {str(e)}'}

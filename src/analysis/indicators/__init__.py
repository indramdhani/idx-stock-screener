# -*- coding: utf-8 -*-
"""
Technical Indicators Package for Indonesian Stock Screener
=========================================================

This package provides technical indicator implementations for stock analysis,
including trend indicators, momentum oscillators, volatility measures, and volume analysis.

All indicators are designed to work with Indonesian stock data and provide
consistent interfaces for calculation and analysis.
"""

from .vwap import VWAP, VWAPAnalyzer
from .atr import ATR, ATRAnalyzer
from .rsi import RSI, RSIAnalyzer
from .ema import EMA, EMAAnalyzer

__all__ = [
    # VWAP (Volume Weighted Average Price)
    "VWAP",
    "VWAPAnalyzer",

    # ATR (Average True Range)
    "ATR",
    "ATRAnalyzer",

    # RSI (Relative Strength Index)
    "RSI",
    "RSIAnalyzer",

    # EMA (Exponential Moving Average)
    "EMA",
    "EMAAnalyzer",
]

__version__ = "1.0.0"
__author__ = "Indonesian Stock Screener Team"
__description__ = "Technical indicators for Indonesian stock market analysis"

# Convenience functions for quick indicator calculations
def calculate_all_indicators(data, config=None):
    """
    Calculate all major indicators for a stock dataset.

    Args:
        data: DataFrame with OHLCV data
        config: Optional configuration for indicator parameters

    Returns:
        Dictionary with all calculated indicators
    """
    if data.empty:
        return {}

    # Default configuration
    if config is None:
        config = {
            'rsi_period': 14,
            'atr_period': 14,
            'ema_periods': [5, 13, 21],
            'vwap_enabled': True
        }

    results = {}

    try:
        # RSI
        rsi_analyzer = RSIAnalyzer(period=config.get('rsi_period', 14))
        results['rsi'] = rsi_analyzer.analyze_stock(data)

        # ATR
        atr_analyzer = ATRAnalyzer(period=config.get('atr_period', 14))
        results['atr'] = atr_analyzer.analyze_stock(data)

        # EMA
        ema_analyzer = EMAAnalyzer(periods=config.get('ema_periods', [5, 13, 21]))
        results['ema'] = ema_analyzer.analyze_stock(data)

        # VWAP (if intraday data available)
        if config.get('vwap_enabled', True) and 'Volume' in data.columns:
            vwap_analyzer = VWAPAnalyzer()
            results['vwap'] = vwap_analyzer.analyze_stock(data)

    except Exception as e:
        results['error'] = f"Error calculating indicators: {str(e)}"

    return results

def get_trading_signals(data, config=None):
    """
    Get consolidated trading signals from all indicators.

    Args:
        data: DataFrame with OHLCV data
        config: Optional configuration for indicator parameters

    Returns:
        Dictionary with consolidated signals and recommendations
    """
    indicators = calculate_all_indicators(data, config)

    if 'error' in indicators:
        return {'error': indicators['error']}

    signals = {
        'overall_signal': 'neutral',
        'confidence': 0.0,
        'individual_signals': {},
        'recommendations': {
            'entry': 'wait',
            'exit': 'hold',
            'stop_loss': None,
            'take_profit': None
        }
    }

    # Extract individual signals
    signal_scores = []

    if 'rsi' in indicators and 'signal' in indicators['rsi']:
        rsi_signal = indicators['rsi']['signal']
        signals['individual_signals']['rsi'] = rsi_signal

        # Convert RSI signal to score
        if rsi_signal in ['strong_buy']:
            signal_scores.append(2)
        elif rsi_signal in ['buy', 'bullish']:
            signal_scores.append(1)
        elif rsi_signal in ['strong_sell']:
            signal_scores.append(-2)
        elif rsi_signal in ['sell', 'bearish']:
            signal_scores.append(-1)
        else:
            signal_scores.append(0)

    if 'ema' in indicators and 'signal' in indicators['ema']:
        ema_signal = indicators['ema']['signal']
        signals['individual_signals']['ema'] = ema_signal

        # Convert EMA signal to score
        if ema_signal in ['strong_bullish']:
            signal_scores.append(2)
        elif ema_signal in ['bullish', 'weak_bullish']:
            signal_scores.append(1)
        elif ema_signal in ['strong_bearish']:
            signal_scores.append(-2)
        elif ema_signal in ['bearish', 'weak_bearish']:
            signal_scores.append(-1)
        else:
            signal_scores.append(0)

    if 'vwap' in indicators and 'signal' in indicators['vwap']:
        vwap_signal = indicators['vwap']['signal']
        signals['individual_signals']['vwap'] = vwap_signal

        # Convert VWAP signal to score
        if vwap_signal in ['strong_bullish']:
            signal_scores.append(1)
        elif vwap_signal in ['bullish']:
            signal_scores.append(0.5)
        elif vwap_signal in ['strong_bearish']:
            signal_scores.append(-1)
        elif vwap_signal in ['bearish']:
            signal_scores.append(-0.5)
        else:
            signal_scores.append(0)

    # Calculate overall signal
    if signal_scores:
        avg_score = sum(signal_scores) / len(signal_scores)

        if avg_score >= 1.5:
            signals['overall_signal'] = 'strong_buy'
        elif avg_score >= 0.5:
            signals['overall_signal'] = 'buy'
        elif avg_score <= -1.5:
            signals['overall_signal'] = 'strong_sell'
        elif avg_score <= -0.5:
            signals['overall_signal'] = 'sell'
        else:
            signals['overall_signal'] = 'neutral'

        # Calculate confidence based on signal agreement
        signal_agreement = 1 - (np.std(signal_scores) / 2 if len(signal_scores) > 1 else 0)
        signals['confidence'] = min(max(signal_agreement, 0), 1)

    # Generate recommendations
    if signals['overall_signal'] in ['strong_buy', 'buy']:
        signals['recommendations']['entry'] = 'buy'
    elif signals['overall_signal'] in ['strong_sell', 'sell']:
        signals['recommendations']['entry'] = 'sell'

    # Add stop loss suggestions from ATR
    if 'atr' in indicators and 'stop_loss_long' in indicators['atr']:
        signals['recommendations']['stop_loss'] = indicators['atr']['stop_loss_long']
        signals['recommendations']['take_profit'] = indicators['atr']['take_profit_long']

    return signals

# Import numpy for calculations
import numpy as np

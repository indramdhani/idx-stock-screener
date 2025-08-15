# -*- coding: utf-8 -*-
"""
ATR (Average True Range) Indicator
==================================

Implementation of Average True Range calculation for Indonesian Stock Screener.
ATR measures volatility by calculating the average of true ranges over a specified period.
It's particularly useful for setting stop-losses and take-profits based on market volatility.
"""

from __future__ import annotations

from typing import Optional, Union
import pandas as pd
import numpy as np


class ATR:
    """
    Average True Range (ATR) calculator.

    ATR is a volatility indicator that measures the degree of price volatility.
    It calculates the average of true ranges over a specified period.
    """

    @staticmethod
    def calculate_true_range(data: pd.DataFrame) -> pd.Series:
        """
        Calculate True Range for each period.

        True Range is the maximum of:
        1. Current High - Current Low
        2. |Current High - Previous Close|
        3. |Current Low - Previous Close|

        Args:
            data: DataFrame with OHLC data

        Returns:
            Series with True Range values

        Raises:
            ValueError: If required columns are missing
        """
        if data.empty:
            return pd.Series(dtype=float)

        # Validate required columns
        required_columns = ['High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Calculate the three components of True Range
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift(1)).abs()
        low_close = (data['Low'] - data['Close'].shift(1)).abs()

        # True Range is the maximum of the three
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        return true_range

    @staticmethod
    def calculate_atr(
        data: pd.DataFrame,
        period: int = 14,
        method: str = 'sma'
    ) -> pd.Series:
        """
        Calculate Average True Range.

        Args:
            data: DataFrame with OHLC data
            period: Period for ATR calculation (default: 14)
            method: Averaging method ('sma', 'ema', 'wilder') (default: 'sma')

        Returns:
            Series with ATR values

        Raises:
            ValueError: If invalid parameters or data
        """
        if period <= 0:
            raise ValueError("Period must be positive")

        if method not in ['sma', 'ema', 'wilder']:
            raise ValueError("Method must be 'sma', 'ema', or 'wilder'")

        # Calculate True Range
        true_range = ATR.calculate_true_range(data)

        if true_range.empty:
            return pd.Series(dtype=float)

        # Calculate ATR based on method
        if method == 'sma':
            # Simple Moving Average
            atr = true_range.rolling(window=period, min_periods=1).mean()

        elif method == 'ema':
            # Exponential Moving Average
            atr = true_range.ewm(span=period, adjust=False).mean()

        elif method == 'wilder':
            # Wilder's smoothing method (traditional ATR)
            atr = pd.Series(index=data.index, dtype=float)

            # Initialize first ATR value as SMA
            first_atr_idx = period - 1
            if len(true_range) > first_atr_idx:
                atr.iloc[first_atr_idx] = true_range.iloc[:period].mean()

                # Apply Wilder's smoothing for subsequent values
                for i in range(first_atr_idx + 1, len(true_range)):
                    prev_atr = atr.iloc[i-1]
                    current_tr = true_range.iloc[i]
                    atr.iloc[i] = (prev_atr * (period - 1) + current_tr) / period

        return atr

    @staticmethod
    def calculate_atr_bands(
        data: pd.DataFrame,
        atr_period: int = 14,
        multiplier: float = 2.0,
        price_column: str = 'Close'
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculate ATR-based bands around price.

        Args:
            data: DataFrame with OHLC data
            atr_period: Period for ATR calculation
            multiplier: ATR multiplier for bands
            price_column: Price column to use as center (default: 'Close')

        Returns:
            Tuple of (upper_band, lower_band) series
        """
        if price_column not in data.columns:
            raise ValueError(f"Price column '{price_column}' not found")

        # Calculate ATR
        atr = ATR.calculate_atr(data, period=atr_period, method='wilder')

        if atr.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        # Calculate bands
        price = data[price_column]
        upper_band = price + (atr * multiplier)
        lower_band = price - (atr * multiplier)

        return upper_band, lower_band

    @staticmethod
    def calculate_atr_stop_loss(
        entry_price: float,
        atr_value: float,
        multiplier: float = 2.0,
        is_long: bool = True
    ) -> float:
        """
        Calculate ATR-based stop loss.

        Args:
            entry_price: Entry price for the trade
            atr_value: Current ATR value
            multiplier: ATR multiplier for stop loss distance
            is_long: True for long positions, False for short

        Returns:
            Stop loss price
        """
        if entry_price <= 0 or atr_value <= 0:
            return entry_price

        if is_long:
            # For long positions, stop loss below entry
            return entry_price - (atr_value * multiplier)
        else:
            # For short positions, stop loss above entry
            return entry_price + (atr_value * multiplier)

    @staticmethod
    def calculate_atr_take_profit(
        entry_price: float,
        atr_value: float,
        multiplier: float = 3.0,
        is_long: bool = True
    ) -> float:
        """
        Calculate ATR-based take profit.

        Args:
            entry_price: Entry price for the trade
            atr_value: Current ATR value
            multiplier: ATR multiplier for take profit distance
            is_long: True for long positions, False for short

        Returns:
            Take profit price
        """
        if entry_price <= 0 or atr_value <= 0:
            return entry_price

        if is_long:
            # For long positions, take profit above entry
            return entry_price + (atr_value * multiplier)
        else:
            # For short positions, take profit below entry
            return entry_price - (atr_value * multiplier)

    @staticmethod
    def calculate_position_size_atr(
        account_balance: float,
        risk_percentage: float,
        entry_price: float,
        atr_value: float,
        atr_multiplier: float = 2.0
    ) -> int:
        """
        Calculate position size based on ATR risk.

        Args:
            account_balance: Total account balance
            risk_percentage: Percentage of account to risk (e.g., 0.02 for 2%)
            entry_price: Entry price per share
            atr_value: Current ATR value
            atr_multiplier: ATR multiplier for stop loss distance

        Returns:
            Number of shares to trade (rounded to lot size for IDX)
        """
        if account_balance <= 0 or risk_percentage <= 0 or entry_price <= 0 or atr_value <= 0:
            return 0

        # Calculate maximum risk amount
        max_risk_amount = account_balance * risk_percentage

        # Calculate risk per share (ATR-based stop loss distance)
        risk_per_share = atr_value * atr_multiplier

        if risk_per_share <= 0:
            return 0

        # Calculate raw position size
        raw_position_size = max_risk_amount / risk_per_share

        # Round down to nearest lot (100 shares for IDX)
        lot_size = 100
        position_lots = int(raw_position_size / lot_size)
        position_size = position_lots * lot_size

        return position_size

    @staticmethod
    def analyze_atr_trend(atr: pd.Series, lookback_periods: int = 5) -> dict:
        """
        Analyze ATR trend to determine volatility conditions.

        Args:
            atr: ATR values series
            lookback_periods: Number of periods to analyze

        Returns:
            Dictionary with trend analysis
        """
        if atr.empty or len(atr) < lookback_periods:
            return {'trend': 'unknown', 'volatility_state': 'unknown'}

        recent_atr = atr.tail(lookback_periods)

        # Calculate trend
        atr_slope = (recent_atr.iloc[-1] - recent_atr.iloc[0]) / (lookback_periods - 1)

        # Determine trend
        if atr_slope > 0.001:  # Arbitrary threshold for increasing volatility
            trend = 'increasing'
        elif atr_slope < -0.001:
            trend = 'decreasing'
        else:
            trend = 'stable'

        # Determine volatility state compared to recent average
        recent_avg = recent_atr.mean()
        current_atr = recent_atr.iloc[-1]

        if current_atr > recent_avg * 1.2:
            volatility_state = 'high'
        elif current_atr < recent_avg * 0.8:
            volatility_state = 'low'
        else:
            volatility_state = 'normal'

        return {
            'trend': trend,
            'volatility_state': volatility_state,
            'current_atr': float(current_atr),
            'recent_avg_atr': float(recent_avg),
            'atr_slope': float(atr_slope)
        }

    @staticmethod
    def get_atr_summary(data: pd.DataFrame, period: int = 14) -> dict:
        """
        Get comprehensive ATR summary statistics.

        Args:
            data: DataFrame with OHLC data
            period: ATR calculation period

        Returns:
            Dictionary with ATR summary
        """
        if data.empty:
            return {}

        try:
            # Calculate ATR and True Range
            atr = ATR.calculate_atr(data, period=period, method='wilder')
            true_range = ATR.calculate_true_range(data)

            if atr.empty:
                return {'error': 'Could not calculate ATR'}

            current_atr = atr.iloc[-1]
            current_price = data['Close'].iloc[-1]

            # ATR as percentage of price
            atr_percentage = (current_atr / current_price) * 100

            # ATR statistics
            atr_stats = {
                'current_atr': float(current_atr),
                'atr_percentage': float(atr_percentage),
                'atr_min': float(atr.min()),
                'atr_max': float(atr.max()),
                'atr_mean': float(atr.mean()),
                'atr_std': float(atr.std()),
            }

            # True Range statistics
            current_tr = true_range.iloc[-1]
            tr_stats = {
                'current_true_range': float(current_tr),
                'tr_mean': float(true_range.mean()),
                'tr_max': float(true_range.max())
            }

            # Trend analysis
            trend_analysis = ATR.analyze_atr_trend(atr)

            # Suggested levels
            stop_loss_long = ATR.calculate_atr_stop_loss(current_price, current_atr, 2.0, True)
            take_profit_long = ATR.calculate_atr_take_profit(current_price, current_atr, 3.0, True)

            suggested_levels = {
                'stop_loss_long': float(stop_loss_long),
                'take_profit_long': float(take_profit_long),
                'risk_reward_ratio': float((take_profit_long - current_price) / (current_price - stop_loss_long))
            }

            return {
                **atr_stats,
                **tr_stats,
                **trend_analysis,
                **suggested_levels
            }

        except Exception as e:
            return {'error': f'ATR analysis failed: {str(e)}'}


class ATRAnalyzer:
    """
    Advanced ATR analysis and signal generation.
    """

    def __init__(self, period: int = 14, method: str = 'wilder'):
        """
        Initialize ATR analyzer.

        Args:
            period: ATR calculation period
            method: ATR calculation method
        """
        self.period = period
        self.method = method

    def analyze_stock(self, data: pd.DataFrame) -> dict:
        """
        Perform comprehensive ATR analysis on stock data.

        Args:
            data: OHLC DataFrame

        Returns:
            Dictionary with analysis results
        """
        if data.empty:
            return {'error': 'No data provided'}

        try:
            # Basic ATR calculation
            atr = ATR.calculate_atr(data, period=self.period, method=self.method)

            if atr.empty:
                return {'error': 'Could not calculate ATR'}

            # Get comprehensive summary
            summary = ATR.get_atr_summary(data, period=self.period)

            # Additional analysis
            upper_band, lower_band = ATR.calculate_atr_bands(data, self.period, 2.0)

            current_price = data['Close'].iloc[-1]
            current_atr = atr.iloc[-1]

            # Volatility classification
            volatility_class = self._classify_volatility(summary.get('atr_percentage', 0))

            # Trading signals
            signals = self._generate_atr_signals(data, atr, current_price, current_atr)

            return {
                **summary,
                'volatility_classification': volatility_class,
                'upper_band': float(upper_band.iloc[-1]) if not upper_band.empty else None,
                'lower_band': float(lower_band.iloc[-1]) if not lower_band.empty else None,
                'signals': signals,
                'period': self.period,
                'method': self.method
            }

        except Exception as e:
            return {'error': f'ATR analysis failed: {str(e)}'}

    def _classify_volatility(self, atr_percentage: float) -> str:
        """Classify volatility based on ATR percentage."""
        if atr_percentage < 1.0:
            return 'very_low'
        elif atr_percentage < 2.0:
            return 'low'
        elif atr_percentage < 4.0:
            return 'normal'
        elif atr_percentage < 6.0:
            return 'high'
        else:
            return 'very_high'

    def _generate_atr_signals(
        self,
        data: pd.DataFrame,
        atr: pd.Series,
        current_price: float,
        current_atr: float
    ) -> dict:
        """Generate trading signals based on ATR analysis."""

        signals = {}

        # Volatility breakout signal
        if len(atr) >= self.period:
            avg_atr = atr.tail(self.period).mean()
            if current_atr > avg_atr * 1.5:
                signals['volatility_breakout'] = 'high_volatility'
            elif current_atr < avg_atr * 0.7:
                signals['volatility_breakout'] = 'low_volatility'
            else:
                signals['volatility_breakout'] = 'normal'

        # Position sizing recommendation
        for risk_pct in [0.01, 0.02, 0.03]:  # 1%, 2%, 3% risk
            account_balance = 100_000_000  # 100M IDR example
            pos_size = ATR.calculate_position_size_atr(
                account_balance, risk_pct, current_price, current_atr, 2.0
            )
            signals[f'position_size_{int(risk_pct*100)}pct'] = pos_size

        # Entry timing based on ATR
        trend_analysis = ATR.analyze_atr_trend(atr)
        if trend_analysis['volatility_state'] == 'low' and trend_analysis['trend'] == 'increasing':
            signals['entry_timing'] = 'favorable'  # Low vol increasing = good entry
        elif trend_analysis['volatility_state'] == 'high' and trend_analysis['trend'] == 'decreasing':
            signals['entry_timing'] = 'wait'  # High vol decreasing = wait
        else:
            signals['entry_timing'] = 'neutral'

        return signals

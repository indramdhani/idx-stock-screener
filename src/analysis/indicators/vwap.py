# -*- coding: utf-8 -*-
"""
VWAP (Volume Weighted Average Price) Indicator
==============================================

Implementation of Volume Weighted Average Price calculation for Indonesian Stock Screener.
VWAP is a trading benchmark that provides the average price a security has traded at
throughout the day, based on both volume and price.
"""

from __future__ import annotations

from typing import Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, time


class VWAP:
    """
    Volume Weighted Average Price (VWAP) calculator.

    VWAP is calculated as the cumulative sum of (price * volume) divided by
    the cumulative sum of volume over a specified period.
    """

    @staticmethod
    def calculate_vwap(
        data: pd.DataFrame,
        price_column: str = 'Close',
        volume_column: str = 'Volume',
        period: Optional[int] = None,
        use_typical_price: bool = True
    ) -> pd.Series:
        """
        Calculate VWAP for the given data.

        Args:
            data: DataFrame with OHLCV data
            price_column: Column name for price (default: 'Close')
            volume_column: Column name for volume (default: 'Volume')
            period: Rolling period for VWAP calculation (None for session VWAP)
            use_typical_price: Use (H+L+C)/3 instead of close price

        Returns:
            Series with VWAP values

        Raises:
            ValueError: If required columns are missing
        """
        if data.empty:
            return pd.Series(dtype=float)

        # Validate required columns
        required_columns = [volume_column]
        if use_typical_price:
            required_columns.extend(['High', 'Low', 'Close'])
        else:
            required_columns.append(price_column)

        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Calculate price to use
        if use_typical_price:
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        else:
            typical_price = data[price_column]

        # Handle zero or negative volumes
        volume = data[volume_column].copy()
        volume = volume.where(volume > 0, np.nan)

        # Calculate price * volume
        price_volume = typical_price * volume

        if period is None:
            # Session VWAP (cumulative from start)
            cumulative_pv = price_volume.expanding().sum()
            cumulative_volume = volume.expanding().sum()
        else:
            # Rolling VWAP
            cumulative_pv = price_volume.rolling(window=period, min_periods=1).sum()
            cumulative_volume = volume.rolling(window=period, min_periods=1).sum()

        # Calculate VWAP, avoiding division by zero
        vwap = cumulative_pv / cumulative_volume

        return vwap

    @staticmethod
    def calculate_session_vwap(
        intraday_data: pd.DataFrame,
        session_start: Optional[time] = None,
        session_end: Optional[time] = None
    ) -> float:
        """
        Calculate VWAP for a trading session from intraday data.

        Args:
            intraday_data: Intraday OHLCV data with timestamp index
            session_start: Session start time (default: market open)
            session_end: Session end time (default: market close)

        Returns:
            Single VWAP value for the session

        Raises:
            ValueError: If data is empty or invalid
        """
        if intraday_data.empty:
            raise ValueError("Intraday data cannot be empty")

        # Default IDX trading hours
        if session_start is None:
            session_start = time(9, 0)  # 9:00 AM
        if session_end is None:
            session_end = time(15, 0)   # 3:00 PM

        # Filter data for session hours
        if hasattr(intraday_data.index, 'time'):
            mask = (intraday_data.index.time >= session_start) & \
                   (intraday_data.index.time <= session_end)
            session_data = intraday_data[mask]
        else:
            # Use all data if no time filtering possible
            session_data = intraday_data

        if session_data.empty:
            raise ValueError("No data available for specified session")

        # Calculate session VWAP
        vwap_series = VWAP.calculate_vwap(session_data)

        # Return the final VWAP value
        return float(vwap_series.iloc[-1]) if not vwap_series.empty else 0.0

    @staticmethod
    def calculate_vwap_deviation(
        current_price: float,
        vwap_value: float
    ) -> float:
        """
        Calculate percentage deviation from VWAP.

        Args:
            current_price: Current stock price
            vwap_value: VWAP value

        Returns:
            Percentage deviation (positive = above VWAP, negative = below VWAP)
        """
        if vwap_value == 0:
            return 0.0

        return ((current_price - vwap_value) / vwap_value) * 100

    @staticmethod
    def calculate_vwap_bands(
        vwap: pd.Series,
        multiplier: float = 1.0,
        std_periods: int = 20
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculate VWAP bands using standard deviation.

        Args:
            vwap: VWAP values series
            multiplier: Standard deviation multiplier
            std_periods: Periods for standard deviation calculation

        Returns:
            Tuple of (upper_band, lower_band) series
        """
        if vwap.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        # Calculate rolling standard deviation of VWAP
        vwap_std = vwap.rolling(window=std_periods, min_periods=1).std()

        # Calculate bands
        upper_band = vwap + (vwap_std * multiplier)
        lower_band = vwap - (vwap_std * multiplier)

        return upper_band, lower_band

    @staticmethod
    def is_above_vwap(current_price: float, vwap_value: float, threshold: float = 0.0) -> bool:
        """
        Check if current price is above VWAP by a threshold.

        Args:
            current_price: Current stock price
            vwap_value: VWAP value
            threshold: Minimum percentage above VWAP (default: 0%)

        Returns:
            True if price is above VWAP + threshold
        """
        if vwap_value == 0:
            return False

        deviation = VWAP.calculate_vwap_deviation(current_price, vwap_value)
        return deviation >= threshold

    @staticmethod
    def is_below_vwap(current_price: float, vwap_value: float, threshold: float = 0.0) -> bool:
        """
        Check if current price is below VWAP by a threshold.

        Args:
            current_price: Current stock price
            vwap_value: VWAP value
            threshold: Minimum percentage below VWAP (default: 0%)

        Returns:
            True if price is below VWAP - threshold
        """
        if vwap_value == 0:
            return False

        deviation = VWAP.calculate_vwap_deviation(current_price, vwap_value)
        return deviation <= -threshold

    @staticmethod
    def calculate_multi_timeframe_vwap(
        data: pd.DataFrame,
        timeframes: list[str] = ['1H', '4H', '1D']
    ) -> pd.DataFrame:
        """
        Calculate VWAP for multiple timeframes.

        Args:
            data: Intraday OHLCV data
            timeframes: List of timeframe strings (pandas frequency strings)

        Returns:
            DataFrame with VWAP for each timeframe
        """
        if data.empty:
            return pd.DataFrame()

        result = pd.DataFrame(index=data.index)

        for timeframe in timeframes:
            try:
                # Resample data to timeframe
                resampled = data.resample(timeframe).agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()

                if not resampled.empty:
                    # Calculate VWAP for this timeframe
                    vwap_values = VWAP.calculate_vwap(resampled)

                    # Forward fill to match original index
                    vwap_reindexed = vwap_values.reindex(
                        data.index,
                        method='ffill'
                    )

                    result[f'VWAP_{timeframe}'] = vwap_reindexed

            except Exception:
                # Skip invalid timeframes
                continue

        return result

    @staticmethod
    def calculate_anchored_vwap(
        data: pd.DataFrame,
        anchor_date: Union[str, datetime, pd.Timestamp]
    ) -> pd.Series:
        """
        Calculate anchored VWAP starting from a specific date/time.

        Args:
            data: OHLCV data with datetime index
            anchor_date: Starting point for VWAP calculation

        Returns:
            Series with anchored VWAP values
        """
        if data.empty:
            return pd.Series(dtype=float)

        # Convert anchor_date to pandas timestamp
        if isinstance(anchor_date, str):
            anchor_date = pd.to_datetime(anchor_date)
        elif isinstance(anchor_date, datetime):
            anchor_date = pd.Timestamp(anchor_date)

        # Filter data from anchor date onwards
        anchored_data = data[data.index >= anchor_date]

        if anchored_data.empty:
            return pd.Series(dtype=float, index=data.index)

        # Calculate VWAP from anchor point
        anchored_vwap = VWAP.calculate_vwap(anchored_data, period=None)

        # Reindex to match original data
        result = pd.Series(index=data.index, dtype=float)
        result[anchored_data.index] = anchored_vwap

        return result

    @staticmethod
    def analyze_vwap_crossover(
        price: pd.Series,
        vwap: pd.Series
    ) -> pd.DataFrame:
        """
        Analyze price crossovers with VWAP.

        Args:
            price: Price series (typically Close prices)
            vwap: VWAP series

        Returns:
            DataFrame with crossover analysis
        """
        if price.empty or vwap.empty:
            return pd.DataFrame()

        # Align series
        aligned_data = pd.DataFrame({
            'price': price,
            'vwap': vwap
        }).dropna()

        if aligned_data.empty:
            return pd.DataFrame()

        # Calculate relative position
        aligned_data['above_vwap'] = aligned_data['price'] > aligned_data['vwap']
        aligned_data['deviation_pct'] = (
            (aligned_data['price'] - aligned_data['vwap']) / aligned_data['vwap']
        ) * 100

        # Detect crossovers
        aligned_data['crossover_up'] = (
            ~aligned_data['above_vwap'].shift(1) &
            aligned_data['above_vwap']
        )

        aligned_data['crossover_down'] = (
            aligned_data['above_vwap'].shift(1) &
            ~aligned_data['above_vwap']
        )

        # Calculate time above/below VWAP
        aligned_data['position_change'] = aligned_data['above_vwap'] != aligned_data['above_vwap'].shift(1)
        aligned_data['position_group'] = aligned_data['position_change'].cumsum()

        return aligned_data

    @staticmethod
    def get_vwap_summary(data: pd.DataFrame) -> dict:
        """
        Get summary statistics for VWAP analysis.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with VWAP summary statistics
        """
        if data.empty:
            return {}

        try:
            # Calculate VWAP
            vwap = VWAP.calculate_vwap(data)
            current_price = data['Close'].iloc[-1]
            current_vwap = vwap.iloc[-1] if not vwap.empty else 0

            # Calculate statistics
            deviation = VWAP.calculate_vwap_deviation(current_price, current_vwap)

            # Volume profile
            total_volume = data['Volume'].sum()
            avg_volume = data['Volume'].mean()

            # Price vs VWAP statistics
            above_vwap_pct = (data['Close'] > vwap).mean() * 100

            return {
                'current_price': float(current_price),
                'current_vwap': float(current_vwap),
                'deviation_pct': float(deviation),
                'is_above_vwap': deviation > 0,
                'total_volume': int(total_volume),
                'avg_volume': float(avg_volume),
                'time_above_vwap_pct': float(above_vwap_pct),
                'vwap_trend': 'bullish' if deviation > 1 else 'bearish' if deviation < -1 else 'neutral'
            }

        except Exception as e:
            return {'error': str(e)}


class VWAPAnalyzer:
    """
    Advanced VWAP analysis and signal generation.
    """

    def __init__(self, deviation_threshold: float = 2.0):
        """
        Initialize VWAP analyzer.

        Args:
            deviation_threshold: Threshold for significant VWAP deviation (%)
        """
        self.deviation_threshold = deviation_threshold

    def analyze_stock(self, data: pd.DataFrame) -> dict:
        """
        Perform comprehensive VWAP analysis on stock data.

        Args:
            data: OHLCV DataFrame

        Returns:
            Dictionary with analysis results
        """
        if data.empty:
            return {'error': 'No data provided'}

        try:
            # Basic VWAP calculation
            vwap = VWAP.calculate_vwap(data)
            current_price = data['Close'].iloc[-1]
            current_vwap = vwap.iloc[-1]

            # Deviation analysis
            deviation = VWAP.calculate_vwap_deviation(current_price, current_vwap)

            # VWAP bands
            upper_band, lower_band = VWAP.calculate_vwap_bands(vwap)

            # Crossover analysis
            crossover_analysis = VWAP.analyze_vwap_crossover(data['Close'], vwap)

            # Signal generation
            signal = self._generate_vwap_signal(
                current_price, current_vwap, deviation,
                upper_band.iloc[-1] if not upper_band.empty else 0,
                lower_band.iloc[-1] if not lower_band.empty else 0
            )

            return {
                'vwap_value': float(current_vwap),
                'deviation_pct': float(deviation),
                'signal': signal,
                'is_significant_deviation': abs(deviation) >= self.deviation_threshold,
                'upper_band': float(upper_band.iloc[-1]) if not upper_band.empty else None,
                'lower_band': float(lower_band.iloc[-1]) if not lower_band.empty else None,
                'recent_crossovers': self._count_recent_crossovers(crossover_analysis),
                'volume_profile': self._analyze_volume_profile(data, vwap)
            }

        except Exception as e:
            return {'error': f'VWAP analysis failed: {str(e)}'}

    def _generate_vwap_signal(
        self,
        price: float,
        vwap: float,
        deviation: float,
        upper_band: float,
        lower_band: float
    ) -> str:
        """Generate trading signal based on VWAP analysis."""

        if abs(deviation) < 0.5:  # Within 0.5% of VWAP
            return 'neutral'

        if deviation >= self.deviation_threshold:
            if price > upper_band:
                return 'strong_bearish'  # Price too far above VWAP
            else:
                return 'bearish'  # Above VWAP but not extreme

        if deviation <= -self.deviation_threshold:
            if price < lower_band:
                return 'strong_bullish'  # Price too far below VWAP
            else:
                return 'bullish'  # Below VWAP but not extreme

        return 'neutral'

    def _count_recent_crossovers(self, crossover_data: pd.DataFrame, periods: int = 5) -> dict:
        """Count recent VWAP crossovers."""
        if crossover_data.empty:
            return {'up': 0, 'down': 0}

        recent_data = crossover_data.tail(periods)

        return {
            'up': int(recent_data['crossover_up'].sum()),
            'down': int(recent_data['crossover_down'].sum())
        }

    def _analyze_volume_profile(self, data: pd.DataFrame, vwap: pd.Series) -> dict:
        """Analyze volume distribution relative to VWAP."""
        if data.empty or vwap.empty:
            return {}

        above_vwap_mask = data['Close'] > vwap

        volume_above = data[above_vwap_mask]['Volume'].sum()
        volume_below = data[~above_vwap_mask]['Volume'].sum()
        total_volume = volume_above + volume_below

        if total_volume == 0:
            return {}

        return {
            'volume_above_vwap_pct': (volume_above / total_volume) * 100,
            'volume_below_vwap_pct': (volume_below / total_volume) * 100,
            'volume_imbalance': 'bullish' if volume_below > volume_above else 'bearish'
        }

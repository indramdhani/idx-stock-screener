# -*- coding: utf-8 -*-
"""
RSI (Relative Strength Index) Indicator
=======================================

Implementation of Relative Strength Index calculation for Indonesian Stock Screener.
RSI is a momentum oscillator that measures the speed and magnitude of recent price changes
to evaluate overbought or oversold conditions in the price of a stock.
"""

from __future__ import annotations

from typing import Optional, Union
import pandas as pd
import numpy as np


class RSI:
    """
    Relative Strength Index (RSI) calculator.

    RSI is calculated as RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over a specified period.
    """

    @staticmethod
    def calculate_rsi(
        data: Union[pd.Series, pd.DataFrame],
        period: int = 14,
        method: str = 'wilder',
        price_column: str = 'Close'
    ) -> pd.Series:
        """
        Calculate Relative Strength Index.

        Args:
            data: Price data (Series or DataFrame with price column)
            period: Period for RSI calculation (default: 14)
            method: Smoothing method ('wilder', 'sma', 'ema') (default: 'wilder')
            price_column: Column name for price if DataFrame (default: 'Close')

        Returns:
            Series with RSI values

        Raises:
            ValueError: If invalid parameters or data
        """
        if period <= 0:
            raise ValueError("Period must be positive")

        if method not in ['wilder', 'sma', 'ema']:
            raise ValueError("Method must be 'wilder', 'sma', or 'ema'")

        # Extract price series
        if isinstance(data, pd.DataFrame):
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found")
            prices = data[price_column]
        else:
            prices = data

        if prices.empty or len(prices) < period + 1:
            return pd.Series(dtype=float, index=prices.index if hasattr(prices, 'index') else None)

        # Calculate price changes
        price_changes = prices.diff()

        # Separate gains and losses
        gains = price_changes.where(price_changes > 0, 0)
        losses = (-price_changes).where(price_changes < 0, 0)

        # Calculate average gains and losses based on method
        if method == 'wilder':
            # Wilder's smoothing method (traditional RSI)
            avg_gains = RSI._wilder_smoothing(gains, period)
            avg_losses = RSI._wilder_smoothing(losses, period)

        elif method == 'sma':
            # Simple Moving Average
            avg_gains = gains.rolling(window=period, min_periods=period).mean()
            avg_losses = losses.rolling(window=period, min_periods=period).mean()

        elif method == 'ema':
            # Exponential Moving Average
            avg_gains = gains.ewm(span=period, adjust=False).mean()
            avg_losses = losses.ewm(span=period, adjust=False).mean()

        # Calculate Relative Strength (RS)
        rs = avg_gains / avg_losses

        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))

        # Handle division by zero (when avg_losses is 0)
        rsi = rsi.fillna(100)  # If no losses, RSI = 100

        return rsi

    @staticmethod
    def _wilder_smoothing(data: pd.Series, period: int) -> pd.Series:
        """
        Apply Wilder's smoothing method.

        Args:
            data: Input data series
            period: Smoothing period

        Returns:
            Smoothed series
        """
        result = pd.Series(index=data.index, dtype=float)

        # Calculate initial average using SMA
        first_avg_idx = period
        if len(data) > first_avg_idx:
            result.iloc[first_avg_idx] = data.iloc[1:first_avg_idx + 1].mean()

            # Apply Wilder's smoothing for subsequent values
            for i in range(first_avg_idx + 1, len(data)):
                prev_avg = result.iloc[i - 1]
                current_value = data.iloc[i]
                result.iloc[i] = (prev_avg * (period - 1) + current_value) / period

        return result

    @staticmethod
    def calculate_rsi_divergence(
        prices: pd.Series,
        rsi: pd.Series,
        lookback_periods: int = 20
    ) -> dict:
        """
        Detect RSI divergences with price.

        Args:
            prices: Price series
            rsi: RSI series
            lookback_periods: Periods to look back for divergence

        Returns:
            Dictionary with divergence analysis
        """
        if len(prices) < lookback_periods or len(rsi) < lookback_periods:
            return {'bullish_divergence': False, 'bearish_divergence': False}

        recent_prices = prices.tail(lookback_periods)
        recent_rsi = rsi.tail(lookback_periods)

        # Find local highs and lows
        price_highs = recent_prices.rolling(window=3, center=True).max() == recent_prices
        price_lows = recent_prices.rolling(window=3, center=True).min() == recent_prices
        rsi_highs = recent_rsi.rolling(window=3, center=True).max() == recent_rsi
        rsi_lows = recent_rsi.rolling(window=3, center=True).min() == recent_rsi

        # Get actual high/low values
        price_high_values = recent_prices[price_highs].dropna()
        price_low_values = recent_prices[price_lows].dropna()
        rsi_high_values = recent_rsi[rsi_highs].dropna()
        rsi_low_values = recent_rsi[rsi_lows].dropna()

        bullish_divergence = False
        bearish_divergence = False

        # Check for bullish divergence (price making lower lows, RSI making higher lows)
        if len(price_low_values) >= 2 and len(rsi_low_values) >= 2:
            if (price_low_values.iloc[-1] < price_low_values.iloc[-2] and
                rsi_low_values.iloc[-1] > rsi_low_values.iloc[-2]):
                bullish_divergence = True

        # Check for bearish divergence (price making higher highs, RSI making lower highs)
        if len(price_high_values) >= 2 and len(rsi_high_values) >= 2:
            if (price_high_values.iloc[-1] > price_high_values.iloc[-2] and
                rsi_high_values.iloc[-1] < rsi_high_values.iloc[-2]):
                bearish_divergence = True

        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'price_trend': 'up' if recent_prices.iloc[-1] > recent_prices.iloc[0] else 'down',
            'rsi_trend': 'up' if recent_rsi.iloc[-1] > recent_rsi.iloc[0] else 'down'
        }

    @staticmethod
    def identify_rsi_levels(
        rsi: pd.Series,
        overbought: float = 70,
        oversold: float = 30
    ) -> dict:
        """
        Identify current RSI level conditions.

        Args:
            rsi: RSI series
            overbought: Overbought threshold (default: 70)
            oversold: Oversold threshold (default: 30)

        Returns:
            Dictionary with RSI level analysis
        """
        if rsi.empty:
            return {}

        current_rsi = rsi.iloc[-1]

        # Determine current condition
        if current_rsi >= overbought:
            condition = 'overbought'
        elif current_rsi <= oversold:
            condition = 'oversold'
        else:
            condition = 'neutral'

        # Calculate time in current condition
        condition_periods = 0
        for i in range(len(rsi) - 1, -1, -1):
            if condition == 'overbought' and rsi.iloc[i] >= overbought:
                condition_periods += 1
            elif condition == 'oversold' and rsi.iloc[i] <= oversold:
                condition_periods += 1
            elif condition == 'neutral' and oversold < rsi.iloc[i] < overbought:
                condition_periods += 1
            else:
                break

        # Recent crossovers
        recent_data = rsi.tail(10)  # Last 10 periods
        overbought_crosses = ((recent_data.shift(1) < overbought) & (recent_data >= overbought)).sum()
        oversold_crosses = ((recent_data.shift(1) > oversold) & (recent_data <= oversold)).sum()

        return {
            'current_rsi': float(current_rsi),
            'condition': condition,
            'periods_in_condition': condition_periods,
            'recent_overbought_crosses': int(overbought_crosses),
            'recent_oversold_crosses': int(oversold_crosses),
            'strength': 'strong' if current_rsi > 80 or current_rsi < 20 else 'moderate'
        }

    @staticmethod
    def calculate_rsi_momentum(rsi: pd.Series, periods: int = 3) -> dict:
        """
        Calculate RSI momentum and rate of change.

        Args:
            rsi: RSI series
            periods: Periods for momentum calculation

        Returns:
            Dictionary with momentum analysis
        """
        if len(rsi) < periods + 1:
            return {}

        current_rsi = rsi.iloc[-1]
        previous_rsi = rsi.iloc[-(periods + 1)]

        # RSI change
        rsi_change = current_rsi - previous_rsi
        rsi_change_pct = (rsi_change / previous_rsi) * 100 if previous_rsi != 0 else 0

        # RSI velocity (rate of change)
        rsi_velocity = rsi_change / periods

        # Momentum classification
        if abs(rsi_velocity) > 3:  # More than 3 points per period
            momentum = 'strong'
        elif abs(rsi_velocity) > 1:  # 1-3 points per period
            momentum = 'moderate'
        else:
            momentum = 'weak'

        direction = 'bullish' if rsi_change > 0 else 'bearish' if rsi_change < 0 else 'neutral'

        return {
            'rsi_change': float(rsi_change),
            'rsi_change_pct': float(rsi_change_pct),
            'rsi_velocity': float(rsi_velocity),
            'momentum_strength': momentum,
            'momentum_direction': direction
        }

    @staticmethod
    def get_rsi_summary(
        data: pd.DataFrame,
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
        price_column: str = 'Close'
    ) -> dict:
        """
        Get comprehensive RSI summary statistics.

        Args:
            data: DataFrame with price data
            period: RSI calculation period
            overbought: Overbought threshold
            oversold: Oversold threshold
            price_column: Price column name

        Returns:
            Dictionary with RSI summary
        """
        if data.empty:
            return {}

        try:
            # Calculate RSI
            rsi = RSI.calculate_rsi(data, period=period, price_column=price_column)

            if rsi.empty:
                return {'error': 'Could not calculate RSI'}

            # Basic statistics
            current_rsi = rsi.iloc[-1]
            rsi_stats = {
                'current_rsi': float(current_rsi),
                'rsi_min': float(rsi.min()),
                'rsi_max': float(rsi.max()),
                'rsi_mean': float(rsi.mean()),
                'rsi_std': float(rsi.std())
            }

            # Level analysis
            level_analysis = RSI.identify_rsi_levels(rsi, overbought, oversold)

            # Momentum analysis
            momentum_analysis = RSI.calculate_rsi_momentum(rsi)

            # Divergence analysis
            divergence_analysis = RSI.calculate_rsi_divergence(
                data[price_column], rsi
            )

            # Signal generation
            signal = RSI._generate_rsi_signal(current_rsi, overbought, oversold)

            return {
                **rsi_stats,
                **level_analysis,
                **momentum_analysis,
                **divergence_analysis,
                'signal': signal,
                'period': period,
                'thresholds': {'overbought': overbought, 'oversold': oversold}
            }

        except Exception as e:
            return {'error': f'RSI analysis failed: {str(e)}'}

    @staticmethod
    def _generate_rsi_signal(rsi_value: float, overbought: float, oversold: float) -> str:
        """Generate trading signal based on RSI value."""
        if rsi_value >= overbought:
            if rsi_value >= 80:
                return 'strong_sell'
            else:
                return 'sell'
        elif rsi_value <= oversold:
            if rsi_value <= 20:
                return 'strong_buy'
            else:
                return 'buy'
        elif rsi_value > 50:
            return 'bullish'
        elif rsi_value < 50:
            return 'bearish'
        else:
            return 'neutral'


class RSIAnalyzer:
    """
    Advanced RSI analysis and signal generation.
    """

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
        method: str = 'wilder'
    ):
        """
        Initialize RSI analyzer.

        Args:
            period: RSI calculation period
            overbought: Overbought threshold
            oversold: Oversold threshold
            method: RSI calculation method
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.method = method

    def analyze_stock(self, data: pd.DataFrame, price_column: str = 'Close') -> dict:
        """
        Perform comprehensive RSI analysis on stock data.

        Args:
            data: OHLC DataFrame
            price_column: Price column to use

        Returns:
            Dictionary with analysis results
        """
        if data.empty:
            return {'error': 'No data provided'}

        try:
            # Calculate RSI
            rsi = RSI.calculate_rsi(
                data,
                period=self.period,
                method=self.method,
                price_column=price_column
            )

            if rsi.empty:
                return {'error': 'Could not calculate RSI'}

            # Get comprehensive summary
            summary = RSI.get_rsi_summary(
                data,
                period=self.period,
                overbought=self.overbought,
                oversold=self.oversold,
                price_column=price_column
            )

            # Additional analysis
            multi_timeframe_rsi = self._calculate_multi_timeframe_rsi(data, price_column)

            # RSI patterns
            patterns = self._identify_rsi_patterns(rsi)

            # Trading recommendations
            recommendations = self._generate_trading_recommendations(rsi, data[price_column])

            return {
                **summary,
                'multi_timeframe': multi_timeframe_rsi,
                'patterns': patterns,
                'recommendations': recommendations,
                'analyzer_config': {
                    'period': self.period,
                    'overbought': self.overbought,
                    'oversold': self.oversold,
                    'method': self.method
                }
            }

        except Exception as e:
            return {'error': f'RSI analysis failed: {str(e)}'}

    def _calculate_multi_timeframe_rsi(self, data: pd.DataFrame, price_column: str) -> dict:
        """Calculate RSI for multiple periods."""
        timeframes = {
            'short': self.period // 2,    # Half period
            'medium': self.period,        # Standard period
            'long': self.period * 2       # Double period
        }

        results = {}

        for name, period in timeframes.items():
            if len(data) >= period + 1:
                try:
                    rsi = RSI.calculate_rsi(
                        data,
                        period=period,
                        method=self.method,
                        price_column=price_column
                    )

                    if not rsi.empty:
                        current_rsi = rsi.iloc[-1]
                        results[name] = {
                            'period': period,
                            'current_rsi': float(current_rsi),
                            'signal': RSI._generate_rsi_signal(
                                current_rsi, self.overbought, self.oversold
                            )
                        }
                except Exception:
                    continue

        return results

    def _identify_rsi_patterns(self, rsi: pd.Series) -> dict:
        """Identify common RSI patterns."""
        patterns = {}

        if len(rsi) < 10:
            return patterns

        recent_rsi = rsi.tail(10)

        # Double bottom/top patterns
        patterns['double_bottom'] = self._detect_double_bottom(recent_rsi)
        patterns['double_top'] = self._detect_double_top(recent_rsi)

        # RSI failure swings
        patterns['failure_swing_bull'] = self._detect_bullish_failure_swing(recent_rsi)
        patterns['failure_swing_bear'] = self._detect_bearish_failure_swing(recent_rsi)

        # Trend patterns
        patterns['rsi_uptrend'] = self._detect_rsi_uptrend(recent_rsi)
        patterns['rsi_downtrend'] = self._detect_rsi_downtrend(recent_rsi)

        return patterns

    def _detect_double_bottom(self, rsi: pd.Series) -> bool:
        """Detect RSI double bottom pattern."""
        if len(rsi) < 5:
            return False

        # Find local minima
        minima_mask = (rsi.shift(1) > rsi) & (rsi.shift(-1) > rsi)
        minima = rsi[minima_mask]

        # Check for two similar lows below oversold level
        if len(minima) >= 2:
            last_two_lows = minima.tail(2)
            if all(low <= self.oversold + 5 for low in last_two_lows):  # Within 5 points of oversold
                return abs(last_two_lows.iloc[0] - last_two_lows.iloc[1]) <= 5

        return False

    def _detect_double_top(self, rsi: pd.Series) -> bool:
        """Detect RSI double top pattern."""
        if len(rsi) < 5:
            return False

        # Find local maxima
        maxima_mask = (rsi.shift(1) < rsi) & (rsi.shift(-1) < rsi)
        maxima = rsi[maxima_mask]

        # Check for two similar highs above overbought level
        if len(maxima) >= 2:
            last_two_highs = maxima.tail(2)
            if all(high >= self.overbought - 5 for high in last_two_highs):  # Within 5 points of overbought
                return abs(last_two_highs.iloc[0] - last_two_highs.iloc[1]) <= 5

        return False

    def _detect_bullish_failure_swing(self, rsi: pd.Series) -> bool:
        """Detect bullish RSI failure swing."""
        # Simplified detection: RSI fails to reach oversold on pullback
        if len(rsi) < 5:
            return False

        recent_min = rsi.tail(5).min()
        return self.oversold < recent_min < 40  # Failed to reach oversold

    def _detect_bearish_failure_swing(self, rsi: pd.Series) -> bool:
        """Detect bearish RSI failure swing."""
        # Simplified detection: RSI fails to reach overbought on rally
        if len(rsi) < 5:
            return False

        recent_max = rsi.tail(5).max()
        return 60 < recent_max < self.overbought  # Failed to reach overbought

    def _detect_rsi_uptrend(self, rsi: pd.Series) -> bool:
        """Detect RSI uptrend."""
        if len(rsi) < 5:
            return False

        # Simple linear trend detection
        x = np.arange(len(rsi))
        y = rsi.values
        slope = np.polyfit(x, y, 1)[0]

        return slope > 1  # RSI increasing by more than 1 point per period on average

    def _detect_rsi_downtrend(self, rsi: pd.Series) -> bool:
        """Detect RSI downtrend."""
        if len(rsi) < 5:
            return False

        # Simple linear trend detection
        x = np.arange(len(rsi))
        y = rsi.values
        slope = np.polyfit(x, y, 1)[0]

        return slope < -1  # RSI decreasing by more than 1 point per period on average

    def _generate_trading_recommendations(self, rsi: pd.Series, prices: pd.Series) -> dict:
        """Generate trading recommendations based on RSI analysis."""
        if rsi.empty:
            return {}

        current_rsi = rsi.iloc[-1]
        recommendations = {}

        # Entry recommendations
        if current_rsi <= self.oversold:
            recommendations['entry'] = {
                'action': 'buy',
                'strength': 'strong' if current_rsi <= 20 else 'moderate',
                'reasoning': f'RSI oversold at {current_rsi:.1f}'
            }
        elif current_rsi >= self.overbought:
            recommendations['entry'] = {
                'action': 'sell',
                'strength': 'strong' if current_rsi >= 80 else 'moderate',
                'reasoning': f'RSI overbought at {current_rsi:.1f}'
            }
        else:
            recommendations['entry'] = {
                'action': 'wait',
                'strength': 'neutral',
                'reasoning': f'RSI neutral at {current_rsi:.1f}'
            }

        # Exit recommendations
        if len(rsi) >= 2:
            rsi_direction = 'up' if rsi.iloc[-1] > rsi.iloc[-2] else 'down'

            if current_rsi >= self.overbought and rsi_direction == 'down':
                recommendations['exit'] = {
                    'action': 'sell',
                    'reasoning': 'RSI turning down from overbought'
                }
            elif current_rsi <= self.oversold and rsi_direction == 'up':
                recommendations['exit'] = {
                    'action': 'cover',
                    'reasoning': 'RSI turning up from oversold'
                }

        # Risk management
        recommendations['risk_management'] = {
            'stop_loss_adjustment': 'tight' if current_rsi > 70 or current_rsi < 30 else 'normal',
            'position_sizing': 'reduce' if 20 < current_rsi < 80 else 'normal'
        }

        return recommendations

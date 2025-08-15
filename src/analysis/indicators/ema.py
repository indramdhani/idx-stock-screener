# -*- coding: utf-8 -*-
"""
EMA (Exponential Moving Average) Indicator
==========================================

Implementation of Exponential Moving Average calculation for Indonesian Stock Screener.
EMA gives more weight to recent prices and responds more quickly to price changes
compared to Simple Moving Average (SMA).
"""

from __future__ import annotations

from typing import Optional, Union, List
import pandas as pd
import numpy as np


class EMA:
    """
    Exponential Moving Average (EMA) calculator.

    EMA is calculated using the formula:
    EMA_today = (Price_today * Smoothing_Factor) + (EMA_yesterday * (1 - Smoothing_Factor))
    where Smoothing_Factor = 2 / (Period + 1)
    """

    @staticmethod
    def calculate_ema(
        data: Union[pd.Series, pd.DataFrame],
        period: int,
        price_column: str = 'Close',
        adjust: bool = True
    ) -> pd.Series:
        """
        Calculate Exponential Moving Average.

        Args:
            data: Price data (Series or DataFrame with price column)
            period: Period for EMA calculation
            price_column: Column name for price if DataFrame (default: 'Close')
            adjust: Use pandas adjustment method (default: True)

        Returns:
            Series with EMA values

        Raises:
            ValueError: If invalid parameters or data
        """
        if period <= 0:
            raise ValueError("Period must be positive")

        # Extract price series
        if isinstance(data, pd.DataFrame):
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found")
            prices = data[price_column]
        else:
            prices = data

        if prices.empty:
            return pd.Series(dtype=float, index=prices.index if hasattr(prices, 'index') else None)

        # Calculate EMA using pandas ewm function
        if adjust:
            ema = prices.ewm(span=period, adjust=True).mean()
        else:
            ema = prices.ewm(span=period, adjust=False).mean()

        return ema

    @staticmethod
    def calculate_multiple_emas(
        data: Union[pd.Series, pd.DataFrame],
        periods: List[int],
        price_column: str = 'Close'
    ) -> pd.DataFrame:
        """
        Calculate multiple EMAs with different periods.

        Args:
            data: Price data
            periods: List of periods for EMA calculation
            price_column: Price column name

        Returns:
            DataFrame with EMA columns
        """
        if isinstance(data, pd.DataFrame):
            prices = data[price_column]
        else:
            prices = data

        result = pd.DataFrame(index=prices.index)

        for period in periods:
            if period > 0:
                ema = EMA.calculate_ema(prices, period)
                result[f'EMA_{period}'] = ema

        return result

    @staticmethod
    def calculate_ema_crossover(
        fast_ema: pd.Series,
        slow_ema: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate EMA crossover signals.

        Args:
            fast_ema: Fast EMA series
            slow_ema: Slow EMA series

        Returns:
            DataFrame with crossover analysis
        """
        if fast_ema.empty or slow_ema.empty:
            return pd.DataFrame()

        # Align series
        aligned_data = pd.DataFrame({
            'fast_ema': fast_ema,
            'slow_ema': slow_ema
        }).dropna()

        if aligned_data.empty:
            return pd.DataFrame()

        # Calculate relative position
        aligned_data['fast_above_slow'] = aligned_data['fast_ema'] > aligned_data['slow_ema']

        # Detect crossovers
        aligned_data['golden_cross'] = (
            ~aligned_data['fast_above_slow'].shift(1) &
            aligned_data['fast_above_slow']
        )

        aligned_data['death_cross'] = (
            aligned_data['fast_above_slow'].shift(1) &
            ~aligned_data['fast_above_slow']
        )

        # Calculate EMA spread
        aligned_data['ema_spread'] = aligned_data['fast_ema'] - aligned_data['slow_ema']
        aligned_data['ema_spread_pct'] = (
            aligned_data['ema_spread'] / aligned_data['slow_ema']
        ) * 100

        return aligned_data

    @staticmethod
    def calculate_ema_slope(ema: pd.Series, periods: int = 3) -> pd.Series:
        """
        Calculate EMA slope (rate of change).

        Args:
            ema: EMA series
            periods: Number of periods for slope calculation

        Returns:
            Series with EMA slope values
        """
        if ema.empty or len(ema) < periods:
            return pd.Series(dtype=float, index=ema.index if hasattr(ema, 'index') else None)

        # Calculate slope using linear regression over specified periods
        slopes = pd.Series(index=ema.index, dtype=float)

        for i in range(periods - 1, len(ema)):
            y_values = ema.iloc[i - periods + 1:i + 1].values
            x_values = np.arange(len(y_values))

            if len(y_values) == periods:
                slope = np.polyfit(x_values, y_values, 1)[0]
                slopes.iloc[i] = slope

        return slopes

    @staticmethod
    def analyze_ema_trend(
        emas: pd.DataFrame,
        current_price: float
    ) -> dict:
        """
        Analyze trend based on multiple EMAs.

        Args:
            emas: DataFrame with multiple EMA columns
            current_price: Current stock price

        Returns:
            Dictionary with trend analysis
        """
        if emas.empty:
            return {}

        # Get current EMA values
        current_emas = emas.iloc[-1]

        # Sort EMAs by period (assuming column names like 'EMA_5', 'EMA_13', etc.)
        ema_columns = sorted(emas.columns, key=lambda x: int(x.split('_')[1]))

        trend_analysis = {
            'current_emas': current_emas.to_dict(),
            'price_vs_emas': {},
            'ema_alignment': 'unknown',
            'trend_strength': 'unknown'
        }

        # Compare price with each EMA
        for col in ema_columns:
            ema_value = current_emas[col]
            trend_analysis['price_vs_emas'][col] = {
                'value': float(ema_value),
                'price_above': current_price > ema_value,
                'distance_pct': ((current_price - ema_value) / ema_value) * 100 if ema_value > 0 else 0
            }

        # Check EMA alignment
        ema_values = [current_emas[col] for col in ema_columns]

        if len(ema_values) >= 2:
            # Bullish alignment: shorter EMAs above longer EMAs
            bullish_aligned = all(ema_values[i] >= ema_values[i+1] for i in range(len(ema_values)-1))
            # Bearish alignment: shorter EMAs below longer EMAs
            bearish_aligned = all(ema_values[i] <= ema_values[i+1] for i in range(len(ema_values)-1))

            if bullish_aligned:
                trend_analysis['ema_alignment'] = 'bullish'
            elif bearish_aligned:
                trend_analysis['ema_alignment'] = 'bearish'
            else:
                trend_analysis['ema_alignment'] = 'mixed'

        # Calculate trend strength based on EMA slopes
        if len(emas) >= 3:
            shortest_ema = ema_columns[0]
            slope = EMA.calculate_ema_slope(emas[shortest_ema], 3)

            if not slope.empty:
                current_slope = slope.iloc[-1]

                if abs(current_slope) > ema_values[0] * 0.01:  # 1% of EMA value
                    trend_analysis['trend_strength'] = 'strong'
                elif abs(current_slope) > ema_values[0] * 0.005:  # 0.5% of EMA value
                    trend_analysis['trend_strength'] = 'moderate'
                else:
                    trend_analysis['trend_strength'] = 'weak'

        return trend_analysis

    @staticmethod
    def get_ema_summary(
        data: pd.DataFrame,
        periods: List[int] = [5, 13, 21],
        price_column: str = 'Close'
    ) -> dict:
        """
        Get comprehensive EMA summary statistics.

        Args:
            data: DataFrame with price data
            periods: List of EMA periods
            price_column: Price column name

        Returns:
            Dictionary with EMA summary
        """
        if data.empty:
            return {}

        try:
            # Calculate multiple EMAs
            emas = EMA.calculate_multiple_emas(data, periods, price_column)

            if emas.empty:
                return {'error': 'Could not calculate EMAs'}

            current_price = data[price_column].iloc[-1]

            # Trend analysis
            trend_analysis = EMA.analyze_ema_trend(emas, current_price)

            # Crossover analysis (using first two EMAs)
            crossover_analysis = {}
            if len(periods) >= 2:
                fast_col = f'EMA_{periods[0]}'
                slow_col = f'EMA_{periods[1]}'

                if fast_col in emas.columns and slow_col in emas.columns:
                    crossover_data = EMA.calculate_ema_crossover(
                        emas[fast_col], emas[slow_col]
                    )

                    if not crossover_data.empty:
                        # Count recent crossovers
                        recent_data = crossover_data.tail(10)
                        crossover_analysis = {
                            'golden_crosses': int(recent_data['golden_cross'].sum()),
                            'death_crosses': int(recent_data['death_cross'].sum()),
                            'current_spread_pct': float(crossover_data['ema_spread_pct'].iloc[-1]),
                            'fast_above_slow': bool(crossover_data['fast_above_slow'].iloc[-1])
                        }

            # Signal generation
            signal = EMA._generate_ema_signal(trend_analysis, crossover_analysis)

            return {
                'periods': periods,
                'trend_analysis': trend_analysis,
                'crossover_analysis': crossover_analysis,
                'signal': signal,
                'current_price': float(current_price)
            }

        except Exception as e:
            return {'error': f'EMA analysis failed: {str(e)}'}

    @staticmethod
    def _generate_ema_signal(trend_analysis: dict, crossover_analysis: dict) -> str:
        """Generate trading signal based on EMA analysis."""

        # Default to neutral
        signal = 'neutral'

        # Check EMA alignment
        alignment = trend_analysis.get('ema_alignment', 'unknown')
        strength = trend_analysis.get('trend_strength', 'unknown')

        # Check crossover signals
        fast_above_slow = crossover_analysis.get('fast_above_slow', None)
        golden_crosses = crossover_analysis.get('golden_crosses', 0)
        death_crosses = crossover_analysis.get('death_crosses', 0)

        # Generate signal based on multiple factors
        if alignment == 'bullish':
            if strength == 'strong':
                signal = 'strong_bullish'
            elif fast_above_slow and golden_crosses > 0:
                signal = 'bullish'
            else:
                signal = 'weak_bullish'

        elif alignment == 'bearish':
            if strength == 'strong':
                signal = 'strong_bearish'
            elif not fast_above_slow and death_crosses > 0:
                signal = 'bearish'
            else:
                signal = 'weak_bearish'

        elif alignment == 'mixed':
            # Look at recent crossovers for direction
            if golden_crosses > death_crosses:
                signal = 'weak_bullish'
            elif death_crosses > golden_crosses:
                signal = 'weak_bearish'

        return signal


class EMAAnalyzer:
    """
    Advanced EMA analysis and signal generation.
    """

    def __init__(self, periods: List[int] = [5, 13, 21]):
        """
        Initialize EMA analyzer.

        Args:
            periods: List of EMA periods to analyze
        """
        self.periods = sorted(periods)

    def analyze_stock(self, data: pd.DataFrame, price_column: str = 'Close') -> dict:
        """
        Perform comprehensive EMA analysis on stock data.

        Args:
            data: OHLC DataFrame
            price_column: Price column to use

        Returns:
            Dictionary with analysis results
        """
        if data.empty:
            return {'error': 'No data provided'}

        try:
            # Calculate EMAs
            emas = EMA.calculate_multiple_emas(data, self.periods, price_column)

            if emas.empty:
                return {'error': 'Could not calculate EMAs'}

            # Get comprehensive summary
            summary = EMA.get_ema_summary(data, self.periods, price_column)

            # Additional analysis
            confluence_analysis = self._analyze_ema_confluence(data, emas, price_column)
            support_resistance = self._identify_ema_support_resistance(data, emas, price_column)
            entry_exit_signals = self._generate_entry_exit_signals(data, emas, price_column)

            return {
                **summary,
                'confluence_analysis': confluence_analysis,
                'support_resistance': support_resistance,
                'entry_exit_signals': entry_exit_signals,
                'analyzer_config': {
                    'periods': self.periods
                }
            }

        except Exception as e:
            return {'error': f'EMA analysis failed: {str(e)}'}

    def _analyze_ema_confluence(
        self,
        data: pd.DataFrame,
        emas: pd.DataFrame,
        price_column: str
    ) -> dict:
        """Analyze EMA confluence zones."""

        if emas.empty or len(self.periods) < 2:
            return {}

        current_price = data[price_column].iloc[-1]
        current_emas = emas.iloc[-1]

        # Find EMAs close to each other (confluence zones)
        ema_values = [current_emas[f'EMA_{period}'] for period in self.periods]

        confluence_zones = []

        for i in range(len(ema_values)):
            for j in range(i + 1, len(ema_values)):
                distance_pct = abs(ema_values[i] - ema_values[j]) / ema_values[i] * 100

                if distance_pct < 2.0:  # Within 2% of each other
                    confluence_zones.append({
                        'ema1': f'EMA_{self.periods[i]}',
                        'ema2': f'EMA_{self.periods[j]}',
                        'value1': float(ema_values[i]),
                        'value2': float(ema_values[j]),
                        'distance_pct': float(distance_pct),
                        'avg_value': float((ema_values[i] + ema_values[j]) / 2)
                    })

        # Analyze price position relative to confluence zones
        price_near_confluence = False
        nearest_confluence = None

        for zone in confluence_zones:
            distance_to_zone = abs(current_price - zone['avg_value']) / zone['avg_value'] * 100
            if distance_to_zone < 1.0:  # Within 1% of confluence zone
                price_near_confluence = True
                if nearest_confluence is None or distance_to_zone < nearest_confluence['distance']:
                    nearest_confluence = {
                        **zone,
                        'distance': distance_to_zone
                    }

        return {
            'confluence_zones': confluence_zones,
            'price_near_confluence': price_near_confluence,
            'nearest_confluence': nearest_confluence
        }

    def _identify_ema_support_resistance(
        self,
        data: pd.DataFrame,
        emas: pd.DataFrame,
        price_column: str
    ) -> dict:
        """Identify EMA-based support and resistance levels."""

        if emas.empty:
            return {}

        current_price = data[price_column].iloc[-1]

        # Look at recent price action relative to EMAs
        recent_data = data.tail(10)  # Last 10 periods

        support_levels = []
        resistance_levels = []

        for period in self.periods:
            ema_col = f'EMA_{period}'
            if ema_col not in emas.columns:
                continue

            recent_ema = emas[ema_col].tail(10)
            recent_prices = recent_data[price_column]

            # Check if EMA has acted as support
            touches_from_above = 0
            bounces_up = 0

            for i in range(1, len(recent_prices)):
                if (recent_prices.iloc[i-1] > recent_ema.iloc[i-1] and
                    recent_prices.iloc[i] <= recent_ema.iloc[i]):
                    touches_from_above += 1

                    # Check if price bounced back up
                    if (i < len(recent_prices) - 1 and
                        recent_prices.iloc[i+1] > recent_ema.iloc[i+1]):
                        bounces_up += 1

            # Check if EMA has acted as resistance
            touches_from_below = 0
            rejections_down = 0

            for i in range(1, len(recent_prices)):
                if (recent_prices.iloc[i-1] < recent_ema.iloc[i-1] and
                    recent_prices.iloc[i] >= recent_ema.iloc[i]):
                    touches_from_below += 1

                    # Check if price was rejected back down
                    if (i < len(recent_prices) - 1 and
                        recent_prices.iloc[i+1] < recent_ema.iloc[i+1]):
                        rejections_down += 1

            current_ema = recent_ema.iloc[-1]

            # Classify as support if multiple bounces
            if touches_from_above >= 2 and bounces_up >= 1:
                support_levels.append({
                    'level': float(current_ema),
                    'type': ema_col,
                    'strength': 'strong' if bounces_up >= 2 else 'moderate',
                    'touches': touches_from_above
                })

            # Classify as resistance if multiple rejections
            if touches_from_below >= 2 and rejections_down >= 1:
                resistance_levels.append({
                    'level': float(current_ema),
                    'type': ema_col,
                    'strength': 'strong' if rejections_down >= 2 else 'moderate',
                    'touches': touches_from_below
                })

        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'current_price': float(current_price)
        }

    def _generate_entry_exit_signals(
        self,
        data: pd.DataFrame,
        emas: pd.DataFrame,
        price_column: str
    ) -> dict:
        """Generate specific entry and exit signals based on EMA analysis."""

        if emas.empty or len(self.periods) < 2:
            return {}

        current_price = data[price_column].iloc[-1]

        signals = {
            'entry_signals': [],
            'exit_signals': [],
            'stop_loss_suggestions': [],
            'take_profit_suggestions': []
        }

        # Get trend analysis
        trend_analysis = EMA.analyze_ema_trend(emas, current_price)
        alignment = trend_analysis.get('ema_alignment', 'unknown')

        # Entry signals
        if alignment == 'bullish':
            # Price pullback to EMA in uptrend
            fast_ema = emas[f'EMA_{self.periods[0]}'].iloc[-1]
            if abs(current_price - fast_ema) / fast_ema < 0.02:  # Within 2%
                signals['entry_signals'].append({
                    'type': 'pullback_entry',
                    'direction': 'long',
                    'reasoning': f'Price near {self.periods[0]}-EMA in bullish trend',
                    'confidence': 'high'
                })

        elif alignment == 'bearish':
            # Price bounce to EMA in downtrend
            fast_ema = emas[f'EMA_{self.periods[0]}'].iloc[-1]
            if abs(current_price - fast_ema) / fast_ema < 0.02:  # Within 2%
                signals['entry_signals'].append({
                    'type': 'bounce_entry',
                    'direction': 'short',
                    'reasoning': f'Price near {self.periods[0]}-EMA in bearish trend',
                    'confidence': 'high'
                })

        # Crossover signals
        if len(self.periods) >= 2:
            fast_ema = emas[f'EMA_{self.periods[0]}'].iloc[-2:]
            slow_ema = emas[f'EMA_{self.periods[1]}'].iloc[-2:]

            # Golden cross
            if fast_ema.iloc[0] <= slow_ema.iloc[0] and fast_ema.iloc[1] > slow_ema.iloc[1]:
                signals['entry_signals'].append({
                    'type': 'golden_cross',
                    'direction': 'long',
                    'reasoning': f'{self.periods[0]}-EMA crossed above {self.periods[1]}-EMA',
                    'confidence': 'moderate'
                })

            # Death cross
            if fast_ema.iloc[0] >= slow_ema.iloc[0] and fast_ema.iloc[1] < slow_ema.iloc[1]:
                signals['entry_signals'].append({
                    'type': 'death_cross',
                    'direction': 'short',
                    'reasoning': f'{self.periods[0]}-EMA crossed below {self.periods[1]}-EMA',
                    'confidence': 'moderate'
                })

        # Stop loss suggestions based on EMAs
        if alignment == 'bullish':
            stop_loss_ema = emas[f'EMA_{self.periods[1]}'].iloc[-1] if len(self.periods) > 1 else emas[f'EMA_{self.periods[0]}'].iloc[-1]
            signals['stop_loss_suggestions'].append({
                'level': float(stop_loss_ema * 0.98),  # 2% below EMA
                'type': 'ema_based',
                'reasoning': f'Below {self.periods[1] if len(self.periods) > 1 else self.periods[0]}-EMA'
            })

        elif alignment == 'bearish':
            stop_loss_ema = emas[f'EMA_{self.periods[1]}'].iloc[-1] if len(self.periods) > 1 else emas[f'EMA_{self.periods[0]}'].iloc[-1]
            signals['stop_loss_suggestions'].append({
                'level': float(stop_loss_ema * 1.02),  # 2% above EMA
                'type': 'ema_based',
                'reasoning': f'Above {self.periods[1] if len(self.periods) > 1 else self.periods[0]}-EMA'
            })

        return signals

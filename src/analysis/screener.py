# -*- coding: utf-8 -*-
"""
Stock Screener Engine for Indonesian Stock Screener
===================================================

Main screening engine that combines technical indicators to identify trading opportunities.
Implements both intraday breakout and overnight rebound strategies using configurable criteria.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

from ..config.settings import TradingConfig
from ..data.models import (
    StockData, TradingSignal, SignalType, SignalStatus, RiskLevel,
    TakeProfitLevel, RiskParameters, SignalContext, PositionSizing
)
from .indicators import RSIAnalyzer, ATRAnalyzer, EMAAnalyzer, VWAPAnalyzer, get_trading_signals
from .risk_calculator import RiskCalculator


class ScreeningCriteria:
    """Container for screening criteria and thresholds."""

    def __init__(self, config: TradingConfig):
        """Initialize screening criteria from configuration."""
        self.config = config

        # Basic filters
        self.min_price = config.screening_criteria.min_price
        self.max_price = config.screening_criteria.max_price
        self.min_volume = config.screening_criteria.min_volume
        self.min_market_cap = config.screening_criteria.min_market_cap
        self.exclude_sectors = config.screening_criteria.exclude_sectors
        self.exclude_tickers = config.screening_criteria.exclude_tickers

        # Technical criteria
        self.rsi_oversold = config.indicators.rsi_oversold
        self.rsi_overbought = config.indicators.rsi_overbought
        self.ema_periods = config.indicators.ema_periods
        self.vwap_deviation_threshold = config.indicators.vwap_deviation_threshold
        self.volume_spike_threshold = config.indicators.volume_spike_threshold

        # Risk management
        self.min_rr_ratio = config.risk_management.min_rr_ratio
        self.max_risk_per_trade = config.risk_management.max_risk_per_trade


class StockScreener:
    """
    Main stock screening engine for identifying trading opportunities.

    Combines multiple technical indicators to generate high-quality trading signals
    with proper risk management parameters.
    """

    def __init__(self, config: TradingConfig):
        """
        Initialize stock screener.

        Args:
            config: Trading configuration with screening parameters
        """
        self.config = config
        self.criteria = ScreeningCriteria(config)
        self.risk_calculator = RiskCalculator(config)

        # Initialize analyzers
        self.rsi_analyzer = RSIAnalyzer(
            period=config.indicators.rsi_period,
            overbought=config.indicators.rsi_overbought,
            oversold=config.indicators.rsi_oversold
        )

        self.atr_analyzer = ATRAnalyzer(
            period=config.indicators.atr_period
        )

        self.ema_analyzer = EMAAnalyzer(
            periods=config.indicators.ema_periods
        )

        self.vwap_analyzer = VWAPAnalyzer()

        logger.info(f"StockScreener initialized with {len(config.default_tickers)} default tickers")

    def screen_intraday_rebounds(
        self,
        stocks_data: Dict[str, StockData],
        account_balance: float = 100_000_000
    ) -> List[TradingSignal]:
        """
        Screen for intraday rebound opportunities.

        Strategy: Buy oversold stocks with bullish momentum for intraday recovery.

        Args:
            stocks_data: Dictionary of stock data
            account_balance: Account balance for position sizing

        Returns:
            List of trading signals sorted by confidence
        """
        logger.info(f"Screening {len(stocks_data)} stocks for intraday rebound opportunities")

        signals = []

        for symbol, stock_data in stocks_data.items():
            try:
                signal = self._analyze_intraday_rebound(symbol, stock_data, account_balance)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.debug(f"Error analyzing {symbol} for intraday rebound: {e}")

        # Sort by confidence score (highest first)
        signals.sort(key=lambda s: s.confidence_score, reverse=True)

        logger.info(f"Found {len(signals)} intraday rebound signals")
        return signals

    def screen_overnight_setups(
        self,
        stocks_data: Dict[str, StockData],
        account_balance: float = 100_000_000
    ) -> List[TradingSignal]:
        """
        Screen for overnight setup opportunities.

        Strategy: Buy oversold quality stocks late in session for overnight recovery.

        Args:
            stocks_data: Dictionary of stock data
            account_balance: Account balance for position sizing

        Returns:
            List of trading signals sorted by confidence
        """
        logger.info(f"Screening {len(stocks_data)} stocks for overnight setup opportunities")

        signals = []

        for symbol, stock_data in stocks_data.items():
            try:
                signal = self._analyze_overnight_setup(symbol, stock_data, account_balance)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.debug(f"Error analyzing {symbol} for overnight setup: {e}")

        # Sort by confidence score (highest first)
        signals.sort(key=lambda s: s.confidence_score, reverse=True)

        logger.info(f"Found {len(signals)} overnight setup signals")
        return signals

    def _analyze_intraday_rebound(
        self,
        symbol: str,
        stock_data: StockData,
        account_balance: float
    ) -> Optional[TradingSignal]:
        """Analyze individual stock for intraday rebound opportunity."""

        # Basic filters
        if not self._passes_basic_filters(stock_data):
            return None

        # Get technical analysis
        indicators = get_trading_signals(stock_data.daily_data, {
            'rsi_period': self.config.indicators.rsi_period,
            'atr_period': self.config.indicators.atr_period,
            'ema_periods': self.config.indicators.ema_periods,
            'vwap_enabled': self.config.enable_vwap_filter
        })

        if 'error' in indicators:
            return None

        # Screening conditions for intraday rebound
        conditions = self._evaluate_intraday_conditions(stock_data, indicators)

        # Must meet minimum criteria
        required_conditions = ['price_range', 'volume_adequate', 'not_excluded']
        if not all(conditions.get(cond, False) for cond in required_conditions):
            return None

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(conditions)

        # Minimum confidence threshold for intraday signals
        if confidence_score < 0.6:
            return None

        # Generate signal
        return self._create_trading_signal(
            symbol=symbol,
            stock_data=stock_data,
            signal_type=SignalType.INTRADAY_REBOUND,
            conditions=conditions,
            confidence_score=confidence_score,
            indicators=indicators,
            account_balance=account_balance
        )

    def _analyze_overnight_setup(
        self,
        symbol: str,
        stock_data: StockData,
        account_balance: float
    ) -> Optional[TradingSignal]:
        """Analyze individual stock for overnight setup opportunity."""

        # Basic filters
        if not self._passes_basic_filters(stock_data):
            return None

        # Get technical analysis
        indicators = get_trading_signals(stock_data.daily_data, {
            'rsi_period': self.config.indicators.rsi_period,
            'atr_period': self.config.indicators.atr_period,
            'ema_periods': self.config.indicators.ema_periods,
            'vwap_enabled': self.config.enable_vwap_filter
        })

        if 'error' in indicators:
            return None

        # Screening conditions for overnight setup
        conditions = self._evaluate_overnight_conditions(stock_data, indicators)

        # Must meet minimum criteria
        required_conditions = ['price_range', 'volume_adequate', 'not_excluded', 'oversold_rsi']
        if not all(conditions.get(cond, False) for cond in required_conditions):
            return None

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(conditions)

        # Minimum confidence threshold for overnight signals
        if confidence_score < 0.7:  # Higher threshold for overnight holds
            return None

        # Generate signal
        return self._create_trading_signal(
            symbol=symbol,
            stock_data=stock_data,
            signal_type=SignalType.OVERNIGHT_SETUP,
            conditions=conditions,
            confidence_score=confidence_score,
            indicators=indicators,
            account_balance=account_balance
        )

    def _passes_basic_filters(self, stock_data: StockData) -> bool:
        """Check if stock passes basic filtering criteria."""

        # Price range filter
        current_price = stock_data.current_price
        if not (self.criteria.min_price <= current_price <= self.criteria.max_price):
            return False

        # Volume filter
        if stock_data.daily_volume < self.criteria.min_volume:
            return False

        # Market cap filter (if specified)
        if self.criteria.min_market_cap and stock_data.market_cap:
            if stock_data.market_cap < self.criteria.min_market_cap:
                return False

        # Sector exclusion
        if (self.criteria.exclude_sectors and
            stock_data.info.sector in self.criteria.exclude_sectors):
            return False

        # Ticker exclusion
        if stock_data.symbol in self.criteria.exclude_tickers:
            return False

        return True

    def _evaluate_intraday_conditions(
        self,
        stock_data: StockData,
        indicators: Dict
    ) -> Dict[str, bool]:
        """Evaluate conditions for intraday rebound strategy."""

        conditions = {}
        current_price = stock_data.current_price

        # Basic conditions
        conditions['price_range'] = (
            self.criteria.min_price <= current_price <= self.criteria.max_price
        )
        conditions['volume_adequate'] = (
            stock_data.daily_volume >= self.criteria.min_volume
        )
        conditions['not_excluded'] = stock_data.symbol not in self.criteria.exclude_tickers

        # Volume spike condition
        avg_volume = stock_data.average_volume(20)
        conditions['volume_spike'] = (
            stock_data.daily_volume >= avg_volume * self.criteria.volume_spike_threshold
        )

        # RSI conditions
        rsi_data = indicators.get('individual_signals', {}).get('rsi')
        if rsi_data:
            conditions['oversold_rsi'] = rsi_data in ['buy', 'strong_buy']
            conditions['rsi_bullish'] = rsi_data in ['bullish', 'buy', 'strong_buy']
        else:
            conditions['oversold_rsi'] = False
            conditions['rsi_bullish'] = False

        # EMA trend conditions
        ema_data = indicators.get('individual_signals', {}).get('ema')
        if ema_data:
            conditions['ema_uptrend'] = ema_data in ['bullish', 'strong_bullish', 'weak_bullish']
            conditions['ema_strong'] = ema_data in ['strong_bullish']
        else:
            conditions['ema_uptrend'] = False
            conditions['ema_strong'] = False

        # VWAP conditions (if enabled)
        if self.config.enable_vwap_filter:
            vwap_data = indicators.get('individual_signals', {}).get('vwap')
            if vwap_data:
                conditions['near_vwap'] = vwap_data in ['bullish', 'neutral']
            else:
                conditions['near_vwap'] = True  # Default to true if no VWAP data
        else:
            conditions['near_vwap'] = True

        # Momentum conditions
        daily_change_pct = stock_data.daily_change_pct
        conditions['positive_momentum'] = daily_change_pct > 0
        conditions['moderate_decline'] = -3.0 <= daily_change_pct <= 0

        return conditions

    def _evaluate_overnight_conditions(
        self,
        stock_data: StockData,
        indicators: Dict
    ) -> Dict[str, bool]:
        """Evaluate conditions for overnight setup strategy."""

        conditions = {}
        current_price = stock_data.current_price

        # Basic conditions
        conditions['price_range'] = (
            self.criteria.min_price <= current_price <= self.criteria.max_price
        )
        conditions['volume_adequate'] = (
            stock_data.daily_volume >= self.criteria.min_volume
        )
        conditions['not_excluded'] = stock_data.symbol not in self.criteria.exclude_tickers

        # Volume condition (higher volume on decline preferred)
        avg_volume = stock_data.average_volume(20)
        conditions['high_volume'] = (
            stock_data.daily_volume >= avg_volume * 1.2  # 20% above average
        )

        # RSI oversold condition (critical for overnight)
        rsi_data = indicators.get('individual_signals', {}).get('rsi')
        if rsi_data:
            conditions['oversold_rsi'] = rsi_data in ['buy', 'strong_buy']
            conditions['extremely_oversold'] = rsi_data == 'strong_buy'
        else:
            conditions['oversold_rsi'] = False
            conditions['extremely_oversold'] = False

        # Decline conditions
        daily_change_pct = stock_data.daily_change_pct
        conditions['significant_decline'] = daily_change_pct <= -2.0
        conditions['severe_decline'] = daily_change_pct <= -4.0

        # EMA support levels
        ema_data = indicators.get('individual_signals', {}).get('ema')
        if ema_data:
            conditions['near_ema_support'] = ema_data in ['weak_bullish', 'neutral']
        else:
            conditions['near_ema_support'] = False

        # Quality stock indicators
        conditions['quality_stock'] = (
            stock_data.market_cap and stock_data.market_cap > 1_000_000_000_000  # 1T IDR
        ) if stock_data.market_cap else False

        # VWAP deviation (should be below for oversold)
        if self.config.enable_vwap_filter:
            vwap_data = indicators.get('individual_signals', {}).get('vwap')
            if vwap_data:
                conditions['below_vwap'] = vwap_data in ['bearish', 'strong_bearish']
            else:
                conditions['below_vwap'] = False
        else:
            conditions['below_vwap'] = True

        return conditions

    def _calculate_confidence_score(self, conditions: Dict[str, bool]) -> float:
        """Calculate confidence score based on met conditions."""

        # Define condition weights
        weights = {
            'price_range': 0.1,
            'volume_adequate': 0.1,
            'not_excluded': 0.1,
            'volume_spike': 0.15,
            'high_volume': 0.15,
            'oversold_rsi': 0.2,
            'extremely_oversold': 0.1,
            'rsi_bullish': 0.1,
            'ema_uptrend': 0.15,
            'ema_strong': 0.1,
            'near_vwap': 0.05,
            'below_vwap': 0.05,
            'positive_momentum': 0.1,
            'moderate_decline': 0.05,
            'significant_decline': 0.15,
            'severe_decline': 0.1,
            'near_ema_support': 0.1,
            'quality_stock': 0.1
        }

        # Calculate weighted score
        total_weight = 0
        achieved_weight = 0

        for condition, met in conditions.items():
            if condition in weights:
                total_weight += weights[condition]
                if met:
                    achieved_weight += weights[condition]

        # Normalize to 0-1 range
        confidence = achieved_weight / total_weight if total_weight > 0 else 0

        # Apply bonuses for multiple strong conditions
        if conditions.get('oversold_rsi') and conditions.get('ema_uptrend'):
            confidence += 0.1

        if conditions.get('volume_spike') and conditions.get('significant_decline'):
            confidence += 0.1

        # Cap at 1.0
        return min(confidence, 1.0)

    def _create_trading_signal(
        self,
        symbol: str,
        stock_data: StockData,
        signal_type: SignalType,
        conditions: Dict[str, bool],
        confidence_score: float,
        indicators: Dict,
        account_balance: float
    ) -> TradingSignal:
        """Create a complete trading signal with risk parameters."""

        current_price = stock_data.current_price

        # Generate signal ID
        signal_id = f"{symbol}_{signal_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Calculate entry reasoning
        reasoning = self._generate_reasoning(conditions, indicators)

        # Calculate risk parameters
        risk_params = self._calculate_risk_parameters(
            stock_data, signal_type, indicators
        )

        # Create signal context
        context = self._create_signal_context(stock_data, indicators, conditions)

        # Determine risk level
        risk_level = self._determine_risk_level(confidence_score, conditions)

        # Create base signal
        signal = TradingSignal(
            signal_id=signal_id,
            symbol=symbol,
            signal_type=signal_type,
            timestamp=datetime.now(),
            entry_price=current_price,
            entry_reasoning=reasoning,
            risk_params=risk_params,
            confidence_score=confidence_score,
            risk_level=risk_level,
            context=context,
            status=SignalStatus.ACTIVE
        )

        # Add position sizing
        signal.update_position_sizing(
            account_balance,
            self.config.risk_management.max_risk_per_trade
        )

        # Add tags based on conditions
        signal = self._add_signal_tags(signal, conditions, indicators)

        # Set expiration (intraday signals expire at market close)
        if signal_type == SignalType.INTRADAY_REBOUND:
            signal.expires_at = datetime.now().replace(hour=15, minute=0, second=0, microsecond=0)
        else:
            signal.expires_at = datetime.now() + timedelta(hours=18)  # Next morning

        return signal

    def _calculate_risk_parameters(
        self,
        stock_data: StockData,
        signal_type: SignalType,
        indicators: Dict
    ) -> RiskParameters:
        """Calculate risk parameters including stop loss and take profit levels."""

        current_price = stock_data.current_price

        # Get ATR for dynamic risk calculation
        atr_value = 0.0
        if 'atr' in indicators and 'current_atr' in indicators['atr']:
            atr_value = indicators['atr']['current_atr']

        # Calculate stop loss and take profit based on strategy
        if self.config.enable_atr_tp_sl and atr_value > 0:
            # ATR-based calculation
            if signal_type == SignalType.INTRADAY_REBOUND:
                stop_loss = current_price - (atr_value * 1.5)
                tp1_price = current_price + (atr_value * 2.0)
                tp2_price = current_price + (atr_value * 3.0)
            else:  # Overnight setup
                stop_loss = current_price - (atr_value * 2.0)
                tp1_price = current_price + (atr_value * 3.0)
                tp2_price = current_price + (atr_value * 4.5)
        else:
            # Percentage-based calculation
            if signal_type == SignalType.INTRADAY_REBOUND:
                stop_loss = current_price * 0.993  # -0.7%
                tp1_price = current_price * 1.015  # +1.5%
                tp2_price = current_price * 1.025  # +2.5%
            else:  # Overnight setup
                stop_loss = current_price * 0.98   # -2%
                tp1_price = current_price * 1.025  # +2.5%
                tp2_price = current_price * 1.04   # +4%

        # Create take profit levels
        tp_levels = [
            TakeProfitLevel(
                price=tp1_price,
                percentage=60.0,
                reasoning="First target - partial profit taking"
            ),
            TakeProfitLevel(
                price=tp2_price,
                percentage=40.0,
                reasoning="Second target - remaining position"
            )
        ]

        # Calculate risk metrics
        risk_amount = abs(current_price - stop_loss)
        potential_reward = tp1_price - current_price  # Based on first TP
        risk_reward_ratio = potential_reward / risk_amount if risk_amount > 0 else 0

        return RiskParameters(
            stop_loss=stop_loss,
            take_profit_levels=tp_levels,
            risk_amount=risk_amount,
            potential_reward=potential_reward,
            risk_reward_ratio=risk_reward_ratio,
            atr_multiplier=1.5 if signal_type == SignalType.INTRADAY_REBOUND else 2.0
        )

    def _create_signal_context(
        self,
        stock_data: StockData,
        indicators: Dict,
        conditions: Dict[str, bool]
    ) -> SignalContext:
        """Create signal context with market and technical information."""

        # Determine market condition
        daily_change = stock_data.daily_change_pct
        if abs(daily_change) > 3:
            market_condition = "volatile"
        elif abs(daily_change) > 1:
            market_condition = "normal"
        else:
            market_condition = "quiet"

        # Volume analysis
        volume_ratio = stock_data.volume_ratio
        if volume_ratio > 2.0:
            volume_analysis = "high"
        elif volume_ratio > 1.2:
            volume_analysis = "above_average"
        else:
            volume_analysis = "normal"

        # Get technical values
        rsi_value = None
        if 'rsi' in indicators and 'current_rsi' in indicators['rsi']:
            rsi_value = indicators['rsi']['current_rsi']

        ema_alignment = None
        if 'ema' in indicators and 'trend_analysis' in indicators['ema']:
            ema_alignment = indicators['ema']['trend_analysis'].get('ema_alignment') == 'bullish'

        return SignalContext(
            market_condition=market_condition,
            volume_analysis=volume_analysis,
            rsi=rsi_value,
            ema_alignment=ema_alignment,
            technical_setup=self._generate_technical_setup_description(conditions)
        )

    def _generate_reasoning(self, conditions: Dict[str, bool], indicators: Dict) -> str:
        """Generate human-readable reasoning for the signal."""

        reasons = []

        # RSI conditions
        if conditions.get('oversold_rsi'):
            rsi_val = indicators.get('rsi', {}).get('current_rsi', 'N/A')
            reasons.append(f"RSI oversold ({rsi_val})")

        # Volume conditions
        if conditions.get('volume_spike'):
            reasons.append("Volume spike above average")
        elif conditions.get('high_volume'):
            reasons.append("High volume")

        # Price movement
        if conditions.get('significant_decline'):
            reasons.append("Significant price decline")
        elif conditions.get('positive_momentum'):
            reasons.append("Positive momentum")

        # EMA conditions
        if conditions.get('ema_uptrend'):
            reasons.append("EMA uptrend alignment")
        elif conditions.get('near_ema_support'):
            reasons.append("Near EMA support")

        # Quality indicators
        if conditions.get('quality_stock'):
            reasons.append("Quality large-cap stock")

        return "; ".join(reasons) if reasons else "Technical setup criteria met"

    def _generate_technical_setup_description(self, conditions: Dict[str, bool]) -> str:
        """Generate technical setup description."""

        if conditions.get('oversold_rsi') and conditions.get('ema_uptrend'):
            return "Oversold bounce in uptrend"
        elif conditions.get('oversold_rsi') and conditions.get('significant_decline'):
            return "Oversold reversal setup"
        elif conditions.get('volume_spike') and conditions.get('positive_momentum'):
            return "Volume breakout"
        else:
            return "Multi-factor technical setup"

    def _determine_risk_level(self, confidence_score: float, conditions: Dict[str, bool]) -> RiskLevel:
        """Determine risk level based on confidence and conditions."""

        if confidence_score >= 0.8:
            return RiskLevel.LOW
        elif confidence_score >= 0.7:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH

    def _add_signal_tags(
        self,
        signal: TradingSignal,
        conditions: Dict[str, bool],
        indicators: Dict
    ) -> TradingSignal:
        """Add relevant tags to the signal."""

        tags = []

        # RSI tags
        if conditions.get('extremely_oversold'):
            tags.append("extremely_oversold")
        elif conditions.get('oversold_rsi'):
            tags.append("oversold")

        # Volume tags
        if conditions.get('volume_spike'):
            tags.append("volume_spike")

        # Quality tags
        if conditions.get('quality_stock'):
            tags.append("large_cap")

        # Technical tags
        if conditions.get('ema_strong'):
            tags.append("strong_trend")

        # Risk tags
        if signal.risk_params.risk_reward_ratio >= 3.0:
            tags.append("high_rr")

        for tag in tags:
            signal.add_tag(tag)

        return signal

    def get_screening_summary(self, signals: List[TradingSignal]) -> Dict:
        """Generate summary statistics for screening results."""

        if not signals:
            return {
                'total_signals': 0,
                'avg_confidence': 0,
                'risk_distribution': {},
                'signal_types': {}
            }

        # Basic statistics
        total_signals = len(signals)
        avg_confidence = sum(s.confidence_score for s in signals) / total_signals

        # Risk distribution
        risk_distribution = {}
        for signal in signals:
            risk_level = signal.risk_level.value
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1

        # Signal type distribution
        signal_types = {}
        for signal in signals:
            signal_type = signal.signal_type.value
            signal_types[signal_type] = signal_types.get(signal_type, 0) + 1

        # Average risk-reward ratio
        avg_rr_ratio = sum(s.risk_params.risk_reward_ratio for s in signals) / total_signals

        return {
            'total_signals': total_signals,
            'avg_confidence': round(avg_confidence, 3),
            'avg_risk_reward_ratio': round(avg_rr_ratio, 2),
            'risk_distribution': risk_distribution,
            'signal_types': signal_types,
            'top_signals': [
                {
                    'symbol': s.symbol,
                    'confidence': s.confidence_score,
                    'risk_reward': s.risk_params.risk_reward_ratio
                }
                for s in signals[:5]  # Top 5 signals
            ]
        }

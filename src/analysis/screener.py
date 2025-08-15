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
                logger.info(f"Analyzing {symbol} for intraday rebound opportunities...")
                signal = self._analyze_intraday_rebound(symbol, stock_data, account_balance)
                if signal:
                    logger.info(f"Found intraday signal for {symbol} with confidence {signal.confidence_score:.2f}")
                    signals.append(signal)
                else:
                    logger.debug(f"No intraday signal generated for {symbol}")
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
                logger.info(f"Analyzing {symbol} for overnight setup opportunities...")
                signal = self._analyze_overnight_setup(symbol, stock_data, account_balance)
                if signal:
                    logger.info(f"Found overnight signal for {symbol} with confidence {signal.confidence_score:.2f}")
                    signals.append(signal)
                else:
                    logger.debug(f"No overnight signal generated for {symbol} - Conditions not met")
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
        try:
            logger.debug(f"Calculating indicators for {symbol}...")
            indicators = get_trading_signals(stock_data.daily_data, {
                'rsi_period': self.config.indicators.rsi_period,
                'atr_period': self.config.indicators.atr_period,
                'ema_periods': self.config.indicators.ema_periods,
                'vwap_enabled': self.config.enable_vwap_filter
            })
            logger.debug(f"Indicators result for {symbol}: {indicators}")
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return None

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
        try:
            logger.debug(f"Calculating indicators for {symbol}...")
            indicators = get_trading_signals(stock_data.daily_data, {
                'rsi_period': self.config.indicators.rsi_period,
                'atr_period': self.config.indicators.atr_period,
                'ema_periods': self.config.indicators.ema_periods,
                'vwap_enabled': self.config.enable_vwap_filter
            })
            logger.debug(f"Indicators result for {symbol}: {indicators}")
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return None

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
            logger.debug(f"Stock {stock_data.symbol} failed price range filter: {current_price} not in [{self.criteria.min_price}, {self.criteria.max_price}]")
            return False

        # Volume filter
        if stock_data.daily_volume < self.criteria.min_volume:
            logger.debug(f"Stock {stock_data.symbol} failed volume filter: {stock_data.daily_volume:,} < {self.criteria.min_volume:,}")
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
            logger.debug(f"Stock {stock_data.symbol} is in exclusion list")
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
        symbol = stock_data.symbol
        logger.debug(f"Evaluating intraday conditions for {symbol}")

        # Basic conditions
        conditions['price_range'] = (
            self.criteria.min_price <= current_price <= self.criteria.max_price
        )
        conditions['volume_adequate'] = (
            stock_data.daily_volume >= self.criteria.min_volume
        )
        conditions['not_excluded'] = symbol not in self.criteria.exclude_tickers

        logger.debug(f"{symbol} basic conditions: price_range={conditions['price_range']}, "
                    f"volume_adequate={conditions['volume_adequate']}, "
                    f"not_excluded={conditions['not_excluded']}")

        # Volume spike condition
        avg_volume = stock_data.average_volume(20)  # Calculate 20-period average volume
        conditions['volume_spike'] = (
            stock_data.daily_volume >= avg_volume * self.criteria.volume_spike_threshold
        )

        # RSI conditions
        rsi_signal = indicators.get('individual_signals', {}).get('rsi')
        logger.debug(f"RSI signal for {symbol}: {rsi_signal}")
        conditions['oversold_rsi'] = rsi_signal in ['buy', 'strong_buy']
        conditions['extremely_oversold'] = rsi_signal == 'strong_buy'
        conditions['rsi_bullish'] = rsi_signal in ['bullish', 'buy', 'strong_buy']

        logger.debug(f"{symbol} RSI conditions: oversold={conditions['oversold_rsi']}, "
                    f"extremely_oversold={conditions['extremely_oversold']}, "
                    f"bullish={conditions['rsi_bullish']}")

        # EMA trend conditions
        ema_signal = indicators.get('individual_signals', {}).get('ema')
        logger.debug(f"EMA signal for {stock_data.symbol}: {ema_signal}")
        conditions['ema_uptrend'] = ema_signal in ['bullish', 'strong_bullish', 'weak_bullish']
        conditions['ema_strong'] = ema_signal == 'strong_bullish'

        # VWAP conditions (if enabled)
        if self.config.enable_vwap_filter:
            vwap_signal = indicators.get('individual_signals', {}).get('vwap')
            logger.debug(f"VWAP signal for {stock_data.symbol}: {vwap_signal}")
            conditions['near_vwap'] = vwap_signal in ['bullish', 'neutral'] if vwap_signal else True
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
        symbol = stock_data.symbol
        logger.debug(f"Evaluating overnight conditions for {symbol}")

        # Basic conditions
        conditions['price_range'] = (
            self.criteria.min_price <= current_price <= self.criteria.max_price
        )
        conditions['volume_adequate'] = (
            stock_data.daily_volume >= self.criteria.min_volume
        )
        conditions['not_excluded'] = symbol not in self.criteria.exclude_tickers

        logger.debug(f"{symbol} basic conditions: price_range={conditions['price_range']}, "
                    f"volume_adequate={conditions['volume_adequate']}, "
                    f"not_excluded={conditions['not_excluded']}")

        # Volume condition (higher volume on decline preferred)
        avg_volume = stock_data.average_volume(20)  # Calculate 20-period average volume
        conditions['high_volume'] = (
            stock_data.daily_volume >= avg_volume * 1.2  # 20% above average
        )

        # RSI oversold condition (critical for overnight)
        rsi_signal = indicators.get('individual_signals', {}).get('rsi')
        logger.debug(f"RSI signal for {symbol}: {rsi_signal}")
        conditions['oversold_rsi'] = rsi_signal in ['buy', 'strong_buy']
        conditions['extremely_oversold'] = rsi_signal == 'strong_buy'

        logger.debug(f"{symbol} RSI conditions: oversold={conditions['oversold_rsi']}, "
                    f"extremely_oversold={conditions['extremely_oversold']}")

        # Decline conditions
        daily_change_pct = stock_data.daily_change_pct
        conditions['significant_decline'] = daily_change_pct <= -2.0
        conditions['severe_decline'] = daily_change_pct <= -4.0

        logger.debug(f"{symbol} decline conditions: daily_change={daily_change_pct:.2f}%, "
                    f"significant={conditions['significant_decline']}, "
                    f"severe={conditions['severe_decline']}")

        # EMA support levels
        ema_signal = indicators.get('individual_signals', {}).get('ema')
        logger.debug(f"EMA signal for {stock_data.symbol}: {ema_signal}")
        conditions['near_ema_support'] = ema_signal in ['weak_bullish', 'neutral']

        # Quality stock indicators
        conditions['quality_stock'] = (
            stock_data.market_cap and stock_data.market_cap > 1_000_000_000_000  # 1T IDR
        ) if stock_data.market_cap else False

        # VWAP deviation (should be below for oversold)
        if self.config.enable_vwap_filter:
            vwap_signal = indicators.get('individual_signals', {}).get('vwap')
            logger.debug(f"VWAP signal for {stock_data.symbol}: {vwap_signal}")
            conditions['below_vwap'] = vwap_signal in ['bearish', 'strong_bearish'] if vwap_signal else False
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

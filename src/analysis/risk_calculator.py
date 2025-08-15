# -*- coding: utf-8 -*-
"""
Risk Calculator for Indonesian Stock Screener
=============================================

Risk management and position sizing calculator that implements proper risk controls
for Indonesian stock trading. Includes position sizing based on account balance,
risk percentage, and ATR-based stop losses.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

from ..config.settings import TradingConfig
from ..data.models import StockData, TradingSignal, PositionSizing


class RiskCalculator:
    """
    Risk management calculator for position sizing and risk assessment.

    Implements various position sizing methods and risk management rules
    for Indonesian stock trading.
    """

    def __init__(self, config: TradingConfig):
        """
        Initialize risk calculator.

        Args:
            config: Trading configuration with risk management parameters
        """
        self.config = config
        self.max_risk_per_trade = config.risk_management.max_risk_per_trade
        self.max_portfolio_risk = config.risk_management.max_portfolio_risk
        self.default_stop_loss_atr = config.risk_management.default_stop_loss_atr
        self.min_rr_ratio = config.risk_management.min_rr_ratio

        # IDX specific parameters
        self.lot_size = 100  # Standard lot size for IDX
        self.min_lot = 1     # Minimum number of lots
        self.max_lot = 10000 # Maximum number of lots (practical limit)

        logger.info("RiskCalculator initialized with max risk per trade: {:.1%}".format(self.max_risk_per_trade))

    def calculate_position_size(
        self,
        signal: TradingSignal,
        account_balance: float,
        existing_risk: float = 0.0
    ) -> PositionSizing:
        """
        Calculate optimal position size for a trading signal.

        Args:
            signal: Trading signal with entry price and risk parameters
            account_balance: Current account balance
            existing_risk: Current portfolio risk amount

        Returns:
            PositionSizing object with calculated parameters

        Raises:
            ValueError: If invalid parameters provided
        """
        if account_balance <= 0:
            raise ValueError("Account balance must be positive")

        if signal.risk_params.risk_amount <= 0:
            logger.warning(f"Signal {signal.symbol} has zero risk amount")
            return PositionSizing(
                shares=0,
                position_value=0.0,
                risk_amount=0.0,
                risk_percentage=0.0
            )

        # Calculate maximum risk allowed for this trade
        max_trade_risk = account_balance * self.max_risk_per_trade

        # Check portfolio risk limit
        max_portfolio_risk_amount = account_balance * self.max_portfolio_risk
        remaining_risk_capacity = max_portfolio_risk_amount - existing_risk

        # Use the smaller of trade risk limit and remaining portfolio capacity
        available_risk = min(max_trade_risk, remaining_risk_capacity)

        if available_risk <= 0:
            logger.info(f"No risk capacity available for {signal.symbol}")
            return PositionSizing(
                shares=0,
                position_value=0.0,
                risk_amount=0.0,
                risk_percentage=0.0
            )

        # Calculate raw position size based on risk
        risk_per_share = signal.risk_params.risk_amount
        raw_shares = available_risk / risk_per_share

        # Convert to lots (round down to ensure we don't exceed risk)
        lots = int(raw_shares // self.lot_size)
        shares = lots * self.lot_size

        # Ensure minimum and maximum lot constraints
        if lots < self.min_lot:
            shares = 0
        elif lots > self.max_lot:
            shares = self.max_lot * self.lot_size

        # Calculate actual position metrics
        position_value = shares * signal.entry_price
        actual_risk = shares * risk_per_share
        risk_percentage = (actual_risk / account_balance) * 100 if account_balance > 0 else 0

        return PositionSizing(
            shares=shares,
            position_value=position_value,
            risk_amount=actual_risk,
            risk_percentage=risk_percentage,
            lot_size=self.lot_size
        )

    def calculate_atr_position_size(
        self,
        entry_price: float,
        atr_value: float,
        account_balance: float,
        atr_multiplier: float = 2.0,
        risk_percentage: float = None
    ) -> PositionSizing:
        """
        Calculate position size based on ATR risk.

        Args:
            entry_price: Entry price per share
            atr_value: Current ATR value
            account_balance: Account balance
            atr_multiplier: ATR multiplier for stop loss
            risk_percentage: Risk percentage (uses config default if None)

        Returns:
            PositionSizing object
        """
        if risk_percentage is None:
            risk_percentage = self.max_risk_per_trade

        if atr_value <= 0 or entry_price <= 0:
            return PositionSizing(0, 0.0, 0.0, 0.0)

        # Calculate risk per share using ATR
        risk_per_share = atr_value * atr_multiplier

        # Calculate maximum risk amount
        max_risk = account_balance * risk_percentage

        # Calculate position size
        raw_shares = max_risk / risk_per_share
        lots = int(raw_shares // self.lot_size)
        shares = max(0, lots * self.lot_size)

        # Calculate metrics
        position_value = shares * entry_price
        actual_risk = shares * risk_per_share
        actual_risk_pct = (actual_risk / account_balance) * 100 if account_balance > 0 else 0

        return PositionSizing(
            shares=shares,
            position_value=position_value,
            risk_amount=actual_risk,
            risk_percentage=actual_risk_pct
        )

    def validate_risk_reward_ratio(self, signal: TradingSignal) -> bool:
        """
        Validate that signal meets minimum risk-reward ratio.

        Args:
            signal: Trading signal to validate

        Returns:
            True if signal meets minimum RR ratio
        """
        return signal.risk_params.risk_reward_ratio >= self.min_rr_ratio

    def calculate_portfolio_risk(self, positions: List[Dict]) -> Dict[str, float]:
        """
        Calculate total portfolio risk from active positions.

        Args:
            positions: List of position dictionaries with risk_amount

        Returns:
            Dictionary with portfolio risk metrics
        """
        if not positions:
            return {
                'total_risk_amount': 0.0,
                'total_risk_percentage': 0.0,
                'position_count': 0,
                'avg_risk_per_position': 0.0
            }

        total_risk = sum(pos.get('risk_amount', 0) for pos in positions)
        position_count = len(positions)
        avg_risk = total_risk / position_count if position_count > 0 else 0

        # Calculate total portfolio value for percentage
        total_value = sum(pos.get('position_value', 0) for pos in positions)
        risk_percentage = (total_risk / total_value) * 100 if total_value > 0 else 0

        return {
            'total_risk_amount': total_risk,
            'total_risk_percentage': risk_percentage,
            'position_count': position_count,
            'avg_risk_per_position': avg_risk,
            'total_position_value': total_value
        }

    def suggest_position_adjustments(
        self,
        signal: TradingSignal,
        account_balance: float,
        existing_positions: List[Dict]
    ) -> Dict[str, any]:
        """
        Suggest position size adjustments based on current portfolio.

        Args:
            signal: New trading signal
            account_balance: Current account balance
            existing_positions: List of existing positions

        Returns:
            Dictionary with recommendations
        """
        # Calculate current portfolio risk
        portfolio_risk = self.calculate_portfolio_risk(existing_positions)
        current_risk_amount = portfolio_risk['total_risk_amount']

        # Calculate suggested position size
        suggested_position = self.calculate_position_size(
            signal, account_balance, current_risk_amount
        )

        # Generate recommendations
        recommendations = {
            'suggested_shares': suggested_position.shares,
            'suggested_lots': suggested_position.lots,
            'position_value': suggested_position.position_value,
            'risk_amount': suggested_position.risk_amount,
            'risk_percentage': suggested_position.risk_percentage
        }

        # Add warnings and adjustments
        warnings = []

        if suggested_position.shares == 0:
            warnings.append("Cannot open position - insufficient risk capacity")

        if current_risk_amount > account_balance * self.max_portfolio_risk * 0.8:
            warnings.append("Portfolio approaching maximum risk limit")

        if signal.risk_params.risk_reward_ratio < self.min_rr_ratio:
            warnings.append(f"Risk-reward ratio {signal.risk_params.risk_reward_ratio:.1f} below minimum {self.min_rr_ratio}")

        # Suggest risk reduction if needed
        if current_risk_amount > account_balance * self.max_portfolio_risk:
            risk_reduction_needed = current_risk_amount - (account_balance * self.max_portfolio_risk)
            recommendations['reduce_existing_risk_by'] = risk_reduction_needed

        recommendations['warnings'] = warnings
        recommendations['portfolio_risk'] = portfolio_risk

        return recommendations

    def calculate_stop_loss_levels(
        self,
        entry_price: float,
        atr_value: float,
        stock_data: StockData,
        method: str = 'atr'
    ) -> Dict[str, float]:
        """
        Calculate various stop loss levels.

        Args:
            entry_price: Entry price
            atr_value: ATR value
            stock_data: Stock data for support/resistance levels
            method: Method to use ('atr', 'percentage', 'support')

        Returns:
            Dictionary with stop loss levels
        """
        stop_levels = {}

        if method == 'atr' and atr_value > 0:
            # ATR-based stops
            stop_levels['atr_1x'] = entry_price - (atr_value * 1.0)
            stop_levels['atr_1_5x'] = entry_price - (atr_value * 1.5)
            stop_levels['atr_2x'] = entry_price - (atr_value * 2.0)

        elif method == 'percentage':
            # Percentage-based stops
            stop_levels['stop_1pct'] = entry_price * 0.99    # -1%
            stop_levels['stop_2pct'] = entry_price * 0.98    # -2%
            stop_levels['stop_3pct'] = entry_price * 0.97    # -3%

        elif method == 'support' and stock_data:
            # Support-based stops
            support, resistance = stock_data.calculate_support_resistance()
            if support:
                stop_levels['support_level'] = support * 0.995  # Slightly below support
                stop_levels['support_break'] = support * 0.98   # Clear break of support

        # Add trailing stop suggestions
        stop_levels['trailing_atr'] = entry_price - (atr_value * 2.0) if atr_value > 0 else entry_price * 0.98

        return stop_levels

    def calculate_take_profit_levels(
        self,
        entry_price: float,
        atr_value: float,
        stock_data: StockData,
        risk_amount: float
    ) -> List[Dict[str, any]]:
        """
        Calculate multiple take profit levels.

        Args:
            entry_price: Entry price
            atr_value: ATR value
            stock_data: Stock data for resistance levels
            risk_amount: Risk amount per share

        Returns:
            List of take profit level dictionaries
        """
        tp_levels = []

        if atr_value > 0:
            # ATR-based take profits
            tp_levels.extend([
                {
                    'level': entry_price + (atr_value * 1.5),
                    'percentage': 50,
                    'method': 'ATR 1.5x',
                    'rr_ratio': (atr_value * 1.5) / risk_amount if risk_amount > 0 else 0
                },
                {
                    'level': entry_price + (atr_value * 2.5),
                    'percentage': 30,
                    'method': 'ATR 2.5x',
                    'rr_ratio': (atr_value * 2.5) / risk_amount if risk_amount > 0 else 0
                },
                {
                    'level': entry_price + (atr_value * 4.0),
                    'percentage': 20,
                    'method': 'ATR 4x',
                    'rr_ratio': (atr_value * 4.0) / risk_amount if risk_amount > 0 else 0
                }
            ])

        # Add resistance-based targets if available
        if stock_data:
            support, resistance = stock_data.calculate_support_resistance()
            if resistance and resistance > entry_price:
                tp_levels.append({
                    'level': resistance * 0.995,  # Just below resistance
                    'percentage': 40,
                    'method': 'Resistance level',
                    'rr_ratio': (resistance * 0.995 - entry_price) / risk_amount if risk_amount > 0 else 0
                })

        # Sort by level (ascending)
        tp_levels.sort(key=lambda x: x['level'])

        return tp_levels

    def calculate_risk_metrics(self, signals: List[TradingSignal], account_balance: float) -> Dict:
        """
        Calculate comprehensive risk metrics for a list of signals.

        Args:
            signals: List of trading signals
            account_balance: Account balance

        Returns:
            Dictionary with risk analysis
        """
        if not signals:
            return {'error': 'No signals provided'}

        # Calculate individual position sizes
        total_position_value = 0
        total_risk_amount = 0
        valid_signals = 0
        risk_reward_ratios = []

        for signal in signals:
            position = self.calculate_position_size(signal, account_balance)

            if position.shares > 0:
                total_position_value += position.position_value
                total_risk_amount += position.risk_amount
                valid_signals += 1
                risk_reward_ratios.append(signal.risk_params.risk_reward_ratio)

        if valid_signals == 0:
            return {'error': 'No valid signals after position sizing'}

        # Calculate metrics
        avg_rr_ratio = np.mean(risk_reward_ratios)
        total_risk_pct = (total_risk_amount / account_balance) * 100
        avg_risk_per_trade = total_risk_amount / valid_signals
        portfolio_utilization = (total_position_value / account_balance) * 100

        # Risk concentration analysis
        risk_distribution = {}
        for signal in signals:
            position = self.calculate_position_size(signal, account_balance)
            if position.shares > 0:
                risk_bucket = self._get_risk_bucket(position.risk_percentage)
                risk_distribution[risk_bucket] = risk_distribution.get(risk_bucket, 0) + 1

        return {
            'total_signals': len(signals),
            'valid_signals': valid_signals,
            'total_position_value': total_position_value,
            'total_risk_amount': total_risk_amount,
            'total_risk_percentage': total_risk_pct,
            'avg_risk_reward_ratio': avg_rr_ratio,
            'avg_risk_per_trade': avg_risk_per_trade,
            'portfolio_utilization': portfolio_utilization,
            'risk_distribution': risk_distribution,
            'within_risk_limits': total_risk_pct <= (self.max_portfolio_risk * 100),
            'recommendations': self._generate_risk_recommendations(total_risk_pct, avg_rr_ratio)
        }

    def _get_risk_bucket(self, risk_percentage: float) -> str:
        """Categorize risk percentage into buckets."""
        if risk_percentage < 1.0:
            return 'low_risk'
        elif risk_percentage < 2.0:
            return 'medium_risk'
        else:
            return 'high_risk'

    def _generate_risk_recommendations(self, total_risk_pct: float, avg_rr_ratio: float) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []

        if total_risk_pct > (self.max_portfolio_risk * 100):
            recommendations.append(f"Total portfolio risk ({total_risk_pct:.1f}%) exceeds limit ({self.max_portfolio_risk:.1%})")

        if avg_rr_ratio < self.min_rr_ratio:
            recommendations.append(f"Average risk-reward ratio ({avg_rr_ratio:.1f}) below minimum ({self.min_rr_ratio})")

        if total_risk_pct < (self.max_portfolio_risk * 50):  # Using less than 50% of allowed risk
            recommendations.append("Portfolio risk utilization is conservative - consider larger positions")

        if not recommendations:
            recommendations.append("Risk parameters are within acceptable limits")

        return recommendations

    def get_risk_summary(self, account_balance: float, existing_positions: List[Dict] = None) -> Dict:
        """
        Get current risk summary for the portfolio.

        Args:
            account_balance: Current account balance
            existing_positions: List of existing positions

        Returns:
            Dictionary with risk summary
        """
        if existing_positions is None:
            existing_positions = []

        portfolio_risk = self.calculate_portfolio_risk(existing_positions)

        # Calculate available risk capacity
        max_portfolio_risk_amount = account_balance * self.max_portfolio_risk
        used_risk = portfolio_risk['total_risk_amount']
        available_risk = max_portfolio_risk_amount - used_risk
        risk_utilization = (used_risk / max_portfolio_risk_amount) * 100 if max_portfolio_risk_amount > 0 else 0

        return {
            'account_balance': account_balance,
            'max_portfolio_risk_amount': max_portfolio_risk_amount,
            'max_portfolio_risk_pct': self.max_portfolio_risk * 100,
            'used_risk_amount': used_risk,
            'available_risk_capacity': max(0, available_risk),
            'risk_utilization_pct': risk_utilization,
            'max_risk_per_trade_amount': account_balance * self.max_risk_per_trade,
            'max_risk_per_trade_pct': self.max_risk_per_trade * 100,
            'current_position_count': portfolio_risk['position_count'],
            'avg_risk_per_position': portfolio_risk['avg_risk_per_position'],
            'total_position_value': portfolio_risk['total_position_value'],
            'status': 'healthy' if risk_utilization < 80 else 'warning' if risk_utilization < 100 else 'exceeded'
        }

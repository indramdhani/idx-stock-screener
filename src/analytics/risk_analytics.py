"""
Risk Analytics Module for Indonesian Stock Screener

This module provides comprehensive risk analysis tools and metrics for trading
strategies and portfolio management, including VaR, drawdown analysis, and
position sizing calculations.

Author: IDX Stock Screener Team
Version: 1.0.0
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class RiskMetrics:
    """Collection of risk metrics for analysis"""

    # Value at Risk metrics
    var_95: float
    var_99: float
    expected_shortfall: float

    # Drawdown metrics
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: int  # days

    # Position metrics
    max_position_size: float
    portfolio_beta: float
    sector_exposure: Dict[str, float]

    # Concentration metrics
    top_holdings_pct: float
    sector_concentration: float

    def __post_init__(self):
        """Validate risk metrics"""
        if not 0 <= self.var_95 <= 1:
            raise ValueError("95% VaR must be between 0 and 1")
        if not 0 <= self.var_99 <= 1:
            raise ValueError("99% VaR must be between 0 and 1")
        if not 0 <= self.max_drawdown <= 1:
            raise ValueError("Maximum drawdown must be between 0 and 1")


class RiskAnalyzer:
    """Analyzer for portfolio and strategy risk metrics"""

    def __init__(self, portfolio_value: float = 0):
        """
        Initialize risk analyzer.

        Args:
            portfolio_value: Current portfolio value in IDR
        """
        self.portfolio_value = portfolio_value
        self._returns_data = None
        self._positions_data = None
        logger.info(f"Risk analyzer initialized with portfolio value: {portfolio_value:,.0f} IDR")

    def set_returns_data(self, returns: Union[List[float], np.ndarray]) -> None:
        """
        Set historical returns data for analysis.

        Args:
            returns: Array of historical returns
        """
        self._returns_data = np.array(returns)
        logger.info(f"Returns data set with {len(returns)} data points")

    def set_positions_data(self, positions: Dict[str, Dict]) -> None:
        """
        Set current portfolio positions for analysis.

        Args:
            positions: Dictionary of position data by symbol
        """
        self._positions_data = positions
        logger.info(f"Positions data set with {len(positions)} positions")

    def calculate_metrics(self) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.

        Returns:
            RiskMetrics containing all calculated metrics
        """
        if self._returns_data is None:
            raise ValueError("Returns data must be set before calculation")

        try:
            # Calculate all risk metrics
            var_95 = self._calculate_var(0.95)
            var_99 = self._calculate_var(0.99)
            es = self._calculate_expected_shortfall(0.95)

            drawdown_metrics = self._calculate_drawdown_metrics()
            position_metrics = self._calculate_position_metrics()
            concentration_metrics = self._calculate_concentration_metrics()

            metrics = RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=es,
                max_drawdown=drawdown_metrics['max_drawdown'],
                avg_drawdown=drawdown_metrics['avg_drawdown'],
                drawdown_duration=drawdown_metrics['duration'],
                max_position_size=position_metrics['max_size'],
                portfolio_beta=position_metrics['portfolio_beta'],
                sector_exposure=position_metrics['sector_exposure'],
                top_holdings_pct=concentration_metrics['top_holdings'],
                sector_concentration=concentration_metrics['sector_concentration']
            )

            return metrics

        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            raise

    def _calculate_var(self, confidence_level: float) -> float:
        """Calculate Value at Risk at given confidence level"""
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")

        try:
            # Historical VaR calculation
            if len(self._returns_data) < 100:
                logger.warning("Limited data points for VaR calculation")

            sorted_returns = np.sort(self._returns_data)
            index = int(len(sorted_returns) * (1 - confidence_level))
            var = -sorted_returns[index]

            return var

        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            raise

    def _calculate_expected_shortfall(self, confidence_level: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            var = self._calculate_var(confidence_level)
            losses = self._returns_data[self._returns_data <= -var]
            return -np.mean(losses) if len(losses) > 0 else var

        except Exception as e:
            logger.error(f"Expected Shortfall calculation failed: {e}")
            raise

    def _calculate_drawdown_metrics(self) -> Dict:
        """Calculate drawdown-related metrics"""
        try:
            cumulative_returns = np.cumprod(1 + self._returns_data)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max

            return {
                'max_drawdown': -np.min(drawdowns),
                'avg_drawdown': -np.mean(drawdowns[drawdowns < 0]),
                'duration': self._calculate_drawdown_duration(drawdowns)
            }

        except Exception as e:
            logger.error(f"Drawdown metrics calculation failed: {e}")
            raise

    def _calculate_drawdown_duration(self, drawdowns: np.ndarray) -> int:
        """Calculate average drawdown duration in days"""
        try:
            is_drawdown = drawdowns < 0
            duration_counts = []
            current_duration = 0

            for is_down in is_drawdown:
                if is_down:
                    current_duration += 1
                elif current_duration > 0:
                    duration_counts.append(current_duration)
                    current_duration = 0

            if current_duration > 0:
                duration_counts.append(current_duration)

            return int(np.mean(duration_counts)) if duration_counts else 0

        except Exception as e:
            logger.error(f"Drawdown duration calculation failed: {e}")
            raise

    def _calculate_position_metrics(self) -> Dict:
        """Calculate position-related risk metrics"""
        if not self._positions_data:
            return {
                'max_size': 0.0,
                'portfolio_beta': 1.0,
                'sector_exposure': {}
            }

        try:
            position_sizes = [
                pos['market_value'] / self.portfolio_value
                for pos in self._positions_data.values()
            ]

            sector_exposure = {}
            for pos in self._positions_data.values():
                sector = pos.get('sector', 'Unknown')
                sector_exposure[sector] = sector_exposure.get(sector, 0) + \
                    pos['market_value'] / self.portfolio_value

            # Calculate portfolio beta (placeholder)
            portfolio_beta = 1.0

            return {
                'max_size': max(position_sizes) if position_sizes else 0.0,
                'portfolio_beta': portfolio_beta,
                'sector_exposure': sector_exposure
            }

        except Exception as e:
            logger.error(f"Position metrics calculation failed: {e}")
            raise

    def _calculate_concentration_metrics(self) -> Dict:
        """Calculate portfolio concentration metrics"""
        if not self._positions_data:
            return {
                'top_holdings': 0.0,
                'sector_concentration': 0.0
            }

        try:
            # Calculate top holdings concentration
            position_sizes = [
                pos['market_value'] / self.portfolio_value
                for pos in self._positions_data.values()
            ]
            position_sizes.sort(reverse=True)
            top_holdings_pct = sum(position_sizes[:5])  # Top 5 holdings

            # Calculate sector concentration (Herfindahl Index)
            sector_exposure = {}
            for pos in self._positions_data.values():
                sector = pos.get('sector', 'Unknown')
                sector_exposure[sector] = sector_exposure.get(sector, 0) + \
                    pos['market_value'] / self.portfolio_value

            sector_concentration = sum(x * x for x in sector_exposure.values())

            return {
                'top_holdings': top_holdings_pct,
                'sector_concentration': sector_concentration
            }

        except Exception as e:
            logger.error(f"Concentration metrics calculation failed: {e}")
            raise

    def calculate_position_size(self,
                              price: float,
                              stop_loss: float,
                              risk_per_trade: float) -> int:
        """
        Calculate recommended position size based on risk parameters.

        Args:
            price: Current price
            stop_loss: Stop loss price
            risk_per_trade: Maximum risk per trade as decimal

        Returns:
            Number of shares to trade
        """
        try:
            if price <= 0 or stop_loss <= 0:
                raise ValueError("Price and stop loss must be positive")

            if not 0 < risk_per_trade < 1:
                raise ValueError("Risk per trade must be between 0 and 1")

            # Calculate position size based on fixed fractional position sizing
            risk_amount = self.portfolio_value * risk_per_trade
            risk_per_share = abs(price - stop_loss)

            if risk_per_share == 0:
                return 0

            shares = int(risk_amount / risk_per_share)

            # Round down to nearest lot size (100 shares)
            shares = (shares // 100) * 100

            return shares

        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            raise

"""
Performance Analyzer Module for Indonesian Stock Screener

This module provides comprehensive performance analysis capabilities including:
- Portfolio performance metrics
- Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)
- Drawdown analysis
- Trade statistics and analysis
- Rolling performance metrics
- Benchmark comparisons

Author: IDX Stock Screener Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PerformanceMetricType(Enum):
    """Performance metric types for categorization"""
    RETURNS = "returns"
    RISK = "risk"
    RISK_ADJUSTED = "risk_adjusted"
    DRAWDOWN = "drawdown"
    TRADE_STATS = "trade_stats"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics container"""
    # Basic Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    compound_annual_growth_rate: float = 0.0

    # Risk Metrics
    volatility: float = 0.0
    annualized_volatility: float = 0.0
    downside_deviation: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Risk-Adjusted Returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0

    # Drawdown Analysis
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    current_drawdown: float = 0.0
    drawdown_periods: int = 0
    recovery_factor: float = 0.0

    # Trade Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    # Additional Metrics
    skewness: float = 0.0
    kurtosis: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    tracking_error: float = 0.0

    # Time-based metrics
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    trading_days: int = 0

    # Rolling metrics
    rolling_sharpe_12m: List[float] = field(default_factory=list)
    rolling_return_12m: List[float] = field(default_factory=list)
    rolling_volatility_12m: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary format"""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, (list, np.ndarray)):
                result[field_name] = list(field_value) if field_value else []
            elif isinstance(field_value, datetime):
                result[field_name] = field_value.isoformat() if field_value else None
            else:
                result[field_name] = field_value
        return result


@dataclass
class TradeAnalysis:
    """Individual trade analysis results"""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    return_pct: Optional[float] = None
    return_amount: Optional[float] = None
    duration_days: Optional[int] = None
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    strategy: str = "unknown"
    confidence_score: Optional[float] = None


class PerformanceAnalyzer:
    """Advanced performance analyzer for trading strategies"""

    def __init__(self, risk_free_rate: float = 0.035):
        """
        Initialize performance analyzer

        Args:
            risk_free_rate: Annual risk-free rate (default: 3.5% for Indonesia)
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)

    def analyze_portfolio_performance(
        self,
        returns_series: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        trades: Optional[List[TradeAnalysis]] = None
    ) -> PerformanceMetrics:
        """
        Analyze comprehensive portfolio performance

        Args:
            returns_series: Portfolio daily returns
            benchmark_returns: Benchmark daily returns for comparison
            trades: List of individual trades for trade analysis

        Returns:
            PerformanceMetrics object with comprehensive metrics
        """
        try:
            if returns_series.empty:
                self.logger.warning("Empty returns series provided")
                return PerformanceMetrics()

            metrics = PerformanceMetrics()

            # Basic setup
            metrics.start_date = returns_series.index[0].to_pydatetime()
            metrics.end_date = returns_series.index[-1].to_pydatetime()
            metrics.trading_days = len(returns_series)

            # Calculate basic returns
            metrics = self._calculate_return_metrics(returns_series, metrics)

            # Calculate risk metrics
            metrics = self._calculate_risk_metrics(returns_series, metrics)

            # Calculate risk-adjusted returns
            metrics = self._calculate_risk_adjusted_metrics(returns_series, metrics)

            # Calculate drawdown analysis
            metrics = self._calculate_drawdown_metrics(returns_series, metrics)

            # Calculate benchmark comparison if provided
            if benchmark_returns is not None:
                metrics = self._calculate_benchmark_metrics(
                    returns_series, benchmark_returns, metrics
                )

            # Calculate trade statistics if provided
            if trades:
                metrics = self._calculate_trade_metrics(trades, metrics)

            # Calculate rolling metrics
            metrics = self._calculate_rolling_metrics(returns_series, metrics)

            self.logger.info(f"Performance analysis completed for {metrics.trading_days} days")
            return metrics

        except Exception as e:
            self.logger.error(f"Error in portfolio performance analysis: {e}")
            return PerformanceMetrics()

    def _calculate_return_metrics(
        self,
        returns: pd.Series,
        metrics: PerformanceMetrics
    ) -> PerformanceMetrics:
        """Calculate basic return metrics"""
        try:
            # Total return
            cumulative_returns = (1 + returns).cumprod()
            metrics.total_return = cumulative_returns.iloc[-1] - 1

            # Annualized return
            years = len(returns) / 252  # Assuming 252 trading days per year
            if years > 0:
                metrics.annualized_return = (1 + metrics.total_return) ** (1/years) - 1
                metrics.compound_annual_growth_rate = metrics.annualized_return

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating return metrics: {e}")
            return metrics

    def _calculate_risk_metrics(
        self,
        returns: pd.Series,
        metrics: PerformanceMetrics
    ) -> PerformanceMetrics:
        """Calculate risk metrics"""
        try:
            # Basic volatility
            metrics.volatility = returns.std()
            metrics.annualized_volatility = metrics.volatility * np.sqrt(252)

            # Downside deviation (for Sortino ratio)
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                metrics.downside_deviation = negative_returns.std() * np.sqrt(252)

            # Value at Risk and Conditional VaR
            metrics.var_95 = returns.quantile(0.05)
            cvar_returns = returns[returns <= metrics.var_95]
            if len(cvar_returns) > 0:
                metrics.cvar_95 = cvar_returns.mean()

            # Distribution metrics
            metrics.skewness = returns.skew()
            metrics.kurtosis = returns.kurtosis()

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return metrics

    def _calculate_risk_adjusted_metrics(
        self,
        returns: pd.Series,
        metrics: PerformanceMetrics
    ) -> PerformanceMetrics:
        """Calculate risk-adjusted return metrics"""
        try:
            daily_rf_rate = self.risk_free_rate / 252
            excess_returns = returns - daily_rf_rate

            # Sharpe Ratio
            if metrics.annualized_volatility > 0:
                metrics.sharpe_ratio = (
                    metrics.annualized_return - self.risk_free_rate
                ) / metrics.annualized_volatility

            # Sortino Ratio
            if metrics.downside_deviation > 0:
                metrics.sortino_ratio = (
                    metrics.annualized_return - self.risk_free_rate
                ) / metrics.downside_deviation

            # Calmar Ratio
            if abs(metrics.max_drawdown) > 0:
                metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown)

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted metrics: {e}")
            return metrics

    def _calculate_drawdown_metrics(
        self,
        returns: pd.Series,
        metrics: PerformanceMetrics
    ) -> PerformanceMetrics:
        """Calculate comprehensive drawdown analysis"""
        try:
            # Calculate cumulative returns
            cumulative_returns = (1 + returns).cumprod()

            # Calculate rolling maximum (peak)
            rolling_max = cumulative_returns.expanding().max()

            # Calculate drawdown
            drawdown = (cumulative_returns - rolling_max) / rolling_max

            # Max drawdown
            metrics.max_drawdown = drawdown.min()

            # Current drawdown
            metrics.current_drawdown = drawdown.iloc[-1]

            # Drawdown duration analysis
            drawdown_periods = (drawdown < -0.01)  # Periods with >1% drawdown
            if drawdown_periods.any():
                # Find longest drawdown period
                drawdown_groups = (drawdown_periods != drawdown_periods.shift()).cumsum()
                drawdown_lengths = drawdown_periods.groupby(drawdown_groups).sum()
                metrics.max_drawdown_duration = drawdown_lengths.max()
                metrics.drawdown_periods = (drawdown_periods).sum()

            # Recovery factor
            if metrics.max_drawdown < 0:
                metrics.recovery_factor = metrics.total_return / abs(metrics.max_drawdown)

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating drawdown metrics: {e}")
            return metrics

    def _calculate_benchmark_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        metrics: PerformanceMetrics
    ) -> PerformanceMetrics:
        """Calculate benchmark comparison metrics"""
        try:
            # Align series
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns)

            # Calculate beta
            covariance = np.cov(aligned_returns.dropna(), aligned_benchmark.dropna())[0][1]
            benchmark_variance = aligned_benchmark.var()
            if benchmark_variance > 0:
                metrics.beta = covariance / benchmark_variance

            # Calculate alpha
            daily_rf_rate = self.risk_free_rate / 252
            benchmark_return = (1 + aligned_benchmark).prod() ** (252/len(aligned_benchmark)) - 1
            expected_return = daily_rf_rate + metrics.beta * (benchmark_return - self.risk_free_rate)
            metrics.alpha = metrics.annualized_return - expected_return

            # Tracking error
            active_returns = aligned_returns - aligned_benchmark
            metrics.tracking_error = active_returns.std() * np.sqrt(252)

            # Information ratio
            if metrics.tracking_error > 0:
                metrics.information_ratio = metrics.alpha / metrics.tracking_error

            # Treynor ratio
            if metrics.beta > 0:
                metrics.treynor_ratio = (
                    metrics.annualized_return - self.risk_free_rate
                ) / metrics.beta

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating benchmark metrics: {e}")
            return metrics

    def _calculate_trade_metrics(
        self,
        trades: List[TradeAnalysis],
        metrics: PerformanceMetrics
    ) -> PerformanceMetrics:
        """Calculate trade-level statistics"""
        try:
            completed_trades = [t for t in trades if t.return_pct is not None]

            metrics.total_trades = len(completed_trades)

            if completed_trades:
                returns = [t.return_pct for t in completed_trades]
                winning_trades = [r for r in returns if r > 0]
                losing_trades = [r for r in returns if r < 0]

                metrics.winning_trades = len(winning_trades)
                metrics.losing_trades = len(losing_trades)
                metrics.win_rate = metrics.winning_trades / metrics.total_trades

                if winning_trades:
                    metrics.avg_win = np.mean(winning_trades)
                if losing_trades:
                    metrics.avg_loss = np.mean(losing_trades)

                # Profit factor
                total_wins = sum(winning_trades) if winning_trades else 0
                total_losses = abs(sum(losing_trades)) if losing_trades else 0
                if total_losses > 0:
                    metrics.profit_factor = total_wins / total_losses

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating trade metrics: {e}")
            return metrics

    def _calculate_rolling_metrics(
        self,
        returns: pd.Series,
        metrics: PerformanceMetrics
    ) -> PerformanceMetrics:
        """Calculate rolling performance metrics"""
        try:
            if len(returns) < 252:  # Need at least 1 year of data
                return metrics

            # 12-month rolling metrics
            window = 252
            daily_rf_rate = self.risk_free_rate / 252

            # Rolling returns
            rolling_returns = returns.rolling(window=window).apply(
                lambda x: (1 + x).prod() ** (252/len(x)) - 1
            )
            metrics.rolling_return_12m = rolling_returns.dropna().tolist()

            # Rolling volatility
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
            metrics.rolling_volatility_12m = rolling_vol.dropna().tolist()

            # Rolling Sharpe ratio
            rolling_sharpe = (rolling_returns - self.risk_free_rate) / rolling_vol
            metrics.rolling_sharpe_12m = rolling_sharpe.dropna().tolist()

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating rolling metrics: {e}")
            return metrics

    def generate_performance_summary(self, metrics: PerformanceMetrics) -> str:
        """Generate a formatted performance summary"""
        try:
            summary = []
            summary.append("=" * 60)
            summary.append("PORTFOLIO PERFORMANCE ANALYSIS")
            summary.append("=" * 60)

            # Period information
            if metrics.start_date and metrics.end_date:
                summary.append(f"Period: {metrics.start_date.strftime('%Y-%m-%d')} to {metrics.end_date.strftime('%Y-%m-%d')}")
                summary.append(f"Trading Days: {metrics.trading_days}")

            # Returns
            summary.append("\nRETURNS:")
            summary.append(f"  Total Return: {metrics.total_return:.2%}")
            summary.append(f"  Annualized Return: {metrics.annualized_return:.2%}")

            # Risk Metrics
            summary.append("\nRISK METRICS:")
            summary.append(f"  Volatility (Annual): {metrics.annualized_volatility:.2%}")
            summary.append(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
            summary.append(f"  VaR (95%): {metrics.var_95:.2%}")

            # Risk-Adjusted Returns
            summary.append("\nRISK-ADJUSTED RETURNS:")
            summary.append(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
            summary.append(f"  Sortino Ratio: {metrics.sortino_ratio:.3f}")
            summary.append(f"  Calmar Ratio: {metrics.calmar_ratio:.3f}")

            # Trade Statistics
            if metrics.total_trades > 0:
                summary.append("\nTRADE STATISTICS:")
                summary.append(f"  Total Trades: {metrics.total_trades}")
                summary.append(f"  Win Rate: {metrics.win_rate:.2%}")
                summary.append(f"  Avg Win: {metrics.avg_win:.2%}")
                summary.append(f"  Avg Loss: {metrics.avg_loss:.2%}")
                summary.append(f"  Profit Factor: {metrics.profit_factor:.2f}")

            summary.append("=" * 60)

            return "\n".join(summary)

        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
            return f"Error generating summary: {e}"

    def export_metrics(self, metrics: PerformanceMetrics, filepath: Path) -> bool:
        """Export performance metrics to file"""
        try:
            import json

            metrics_dict = metrics.to_dict()

            with open(filepath, 'w') as f:
                json.dump(metrics_dict, f, indent=2, default=str)

            self.logger.info(f"Performance metrics exported to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            return False

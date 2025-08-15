# -*- coding: utf-8 -*-
"""
Backtesting Engine for Indonesian Stock Screener
===============================================

Comprehensive backtesting system for validating and optimizing screening strategies.
Includes performance metrics, risk analysis, and strategy comparison capabilities.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import pandas as pd
import numpy as np
from loguru import logger

from ..config.settings import TradingConfig
from ..data.models import TradingSignal, SignalType, SignalStatus, StockData
from .screener import StockScreener
from .risk_calculator import RiskCalculator


@dataclass
class BacktestTrade:
    """Represents a single trade in the backtest."""

    signal_id: str
    symbol: str
    signal_type: SignalType
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = "open"  # "tp", "sl", "time", "open"

    # Position details
    shares: int = 0
    position_value: float = 0.0

    # Risk parameters
    stop_loss: float = 0.0
    take_profit: float = 0.0

    # Performance metrics
    pnl: float = 0.0
    pnl_pct: float = 0.0
    hold_days: float = 0.0

    # Market context
    rsi_entry: Optional[float] = None
    confidence_score: float = 0.0

    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_date is not None

    def update_exit(self, exit_date: datetime, exit_price: float, exit_reason: str):
        """Update trade exit information."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = exit_reason

        # Calculate performance
        if self.shares > 0:  # Long position
            self.pnl = (exit_price - self.entry_price) * self.shares
        else:  # Short position
            self.pnl = (self.entry_price - exit_price) * abs(self.shares)

        self.pnl_pct = (self.pnl / self.position_value) * 100 if self.position_value > 0 else 0
        self.hold_days = (exit_date - self.entry_date).total_seconds() / (24 * 3600)


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""

    start_date: datetime
    end_date: datetime
    initial_capital: float = 100_000_000  # 100M IDR

    # Exit rules
    max_hold_days_intraday: int = 1
    max_hold_days_overnight: int = 5

    # Risk management
    use_stop_loss: bool = True
    use_take_profit: bool = True
    position_sizing_method: str = "risk_based"  # "fixed", "risk_based"

    # Market conditions
    exclude_weekends: bool = True
    market_open_hour: int = 9
    market_close_hour: int = 15

    # Performance settings
    benchmark_symbol: str = "^JKSE"  # Jakarta Composite Index
    risk_free_rate: float = 0.06  # 6% annual risk-free rate

    # Data requirements
    min_trading_days: int = 5
    require_volume_data: bool = True


@dataclass
class BacktestResults:
    """Comprehensive backtesting results."""

    config: BacktestConfig
    trades: List[BacktestTrade] = field(default_factory=list)

    # Portfolio performance
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_hold_time: float = 0.0

    # Risk metrics
    max_portfolio_exposure: float = 0.0
    avg_portfolio_exposure: float = 0.0
    largest_loss: float = 0.0
    largest_win: float = 0.0

    # Strategy breakdown
    strategy_performance: Dict[str, Dict] = field(default_factory=dict)
    monthly_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Benchmark comparison
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0

    def calculate_summary_stats(self):
        """Calculate summary statistics from trades."""
        if not self.trades:
            return

        self.total_trades = len(self.trades)
        closed_trades = [t for t in self.trades if t.is_closed()]

        if not closed_trades:
            return

        # Win/loss statistics
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]

        self.winning_trades = len(winning_trades)
        self.losing_trades = len(losing_trades)
        self.win_rate = (self.winning_trades / len(closed_trades)) * 100

        if winning_trades:
            self.avg_win = np.mean([t.pnl for t in winning_trades])
            self.largest_win = max(t.pnl for t in winning_trades)

        if losing_trades:
            self.avg_loss = np.mean([t.pnl for t in losing_trades])
            self.largest_loss = min(t.pnl for t in losing_trades)

        # Hold time
        self.avg_hold_time = np.mean([t.hold_days for t in closed_trades])

        # Total return
        total_pnl = sum(t.pnl for t in closed_trades)
        self.total_return = total_pnl
        self.total_return_pct = (total_pnl / self.config.initial_capital) * 100

        # Calculate strategy-specific performance
        self._calculate_strategy_breakdown(closed_trades)

    def _calculate_strategy_breakdown(self, trades: List[BacktestTrade]):
        """Calculate performance breakdown by strategy."""
        strategy_stats = {}

        for strategy in [SignalType.INTRADAY_REBOUND, SignalType.OVERNIGHT_SETUP]:
            strategy_trades = [t for t in trades if t.signal_type == strategy]

            if strategy_trades:
                total_pnl = sum(t.pnl for t in strategy_trades)
                winners = len([t for t in strategy_trades if t.pnl > 0])

                strategy_stats[strategy.value] = {
                    'total_trades': len(strategy_trades),
                    'total_pnl': total_pnl,
                    'win_rate': (winners / len(strategy_trades)) * 100,
                    'avg_pnl': total_pnl / len(strategy_trades),
                    'avg_hold_time': np.mean([t.hold_days for t in strategy_trades])
                }

        self.strategy_performance = strategy_stats


class BacktestEngine:
    """
    Main backtesting engine for strategy validation.

    Simulates trading strategies over historical data to evaluate performance,
    calculate risk metrics, and optimize parameters.
    """

    def __init__(self, config: TradingConfig, backtest_config: BacktestConfig):
        """
        Initialize backtesting engine.

        Args:
            config: Trading configuration
            backtest_config: Backtesting parameters
        """
        self.config = config
        self.backtest_config = backtest_config

        # Core components
        self.screener = StockScreener(config)
        self.risk_calculator = RiskCalculator(config)

        # Backtest state
        self.current_date = backtest_config.start_date
        self.portfolio_value = backtest_config.initial_capital
        self.cash = backtest_config.initial_capital
        self.open_trades: List[BacktestTrade] = []
        self.closed_trades: List[BacktestTrade] = []

        # Performance tracking
        self.daily_portfolio_values = []
        self.daily_dates = []

        logger.info(f"Backtesting engine initialized for {backtest_config.start_date.date()} to {backtest_config.end_date.date()}")

    async def run_backtest(self, historical_data: Dict[str, pd.DataFrame]) -> BacktestResults:
        """
        Run complete backtest simulation.

        Args:
            historical_data: Dictionary mapping symbols to historical OHLCV data

        Returns:
            BacktestResults with comprehensive performance metrics
        """
        logger.info("Starting backtest simulation...")

        try:
            # Validate historical data
            validated_data = self._validate_historical_data(historical_data)

            if not validated_data:
                raise ValueError("No valid historical data available for backtesting")

            logger.info(f"Backtesting with {len(validated_data)} symbols")

            # Main backtest loop
            current_date = self.backtest_config.start_date

            while current_date <= self.backtest_config.end_date:
                # Skip weekends if configured
                if self.backtest_config.exclude_weekends and current_date.weekday() >= 5:
                    current_date += timedelta(days=1)
                    continue

                # Process trading day
                await self._process_trading_day(current_date, validated_data)

                # Record daily portfolio value
                self._update_daily_performance(current_date)

                current_date += timedelta(days=1)

            # Close any remaining open trades
            self._close_remaining_trades(self.backtest_config.end_date, validated_data)

            # Generate results
            results = self._generate_results()

            logger.info(f"Backtest completed: {results.total_trades} trades, {results.win_rate:.1f}% win rate")
            return results

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise

    def _validate_historical_data(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Validate and filter historical data for backtesting."""
        validated_data = {}

        for symbol, data in historical_data.items():
            if data.empty:
                continue

            # Check required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                logger.warning(f"Missing required columns for {symbol}")
                continue

            # Filter date range
            mask = (data.index >= self.backtest_config.start_date) & (data.index <= self.backtest_config.end_date)
            filtered_data = data[mask]

            # Check minimum trading days
            if len(filtered_data) < self.backtest_config.min_trading_days:
                logger.warning(f"Insufficient data for {symbol}: {len(filtered_data)} days")
                continue

            # Check for data quality issues
            if filtered_data.isnull().sum().sum() > len(filtered_data) * 0.1:  # More than 10% missing
                logger.warning(f"Poor data quality for {symbol}")
                continue

            validated_data[symbol] = filtered_data

        return validated_data

    async def _process_trading_day(self, current_date: datetime, historical_data: Dict[str, pd.DataFrame]):
        """Process a single trading day."""

        # Update current date for components
        self.current_date = current_date

        # Check for exits on existing positions
        await self._check_position_exits(current_date, historical_data)

        # Generate new signals (only during market hours)
        if self._is_market_hours(current_date):
            await self._generate_signals(current_date, historical_data)

    def _is_market_hours(self, date: datetime) -> bool:
        """Check if current time is during market hours."""
        hour = date.hour
        return self.backtest_config.market_open_hour <= hour < self.backtest_config.market_close_hour

    async def _check_position_exits(self, current_date: datetime, historical_data: Dict[str, pd.DataFrame]):
        """Check and execute position exits based on rules."""

        trades_to_close = []

        for trade in self.open_trades:
            symbol_data = historical_data.get(trade.symbol)
            if symbol_data is None:
                continue

            # Get current price data
            try:
                current_data = symbol_data.loc[current_date]
                current_price = current_data['Close']
                high_price = current_data['High']
                low_price = current_data['Low']
            except KeyError:
                continue  # No data for this date

            exit_reason = None
            exit_price = current_price

            # Check stop loss
            if self.backtest_config.use_stop_loss and trade.stop_loss > 0:
                if trade.shares > 0:  # Long position
                    if low_price <= trade.stop_loss:
                        exit_reason = "sl"
                        exit_price = trade.stop_loss
                else:  # Short position
                    if high_price >= trade.stop_loss:
                        exit_reason = "sl"
                        exit_price = trade.stop_loss

            # Check take profit
            if self.backtest_config.use_take_profit and trade.take_profit > 0 and exit_reason is None:
                if trade.shares > 0:  # Long position
                    if high_price >= trade.take_profit:
                        exit_reason = "tp"
                        exit_price = trade.take_profit
                else:  # Short position
                    if low_price <= trade.take_profit:
                        exit_reason = "tp"
                        exit_price = trade.take_profit

            # Check time-based exit
            if exit_reason is None:
                hold_days = (current_date - trade.entry_date).days
                max_hold = (self.backtest_config.max_hold_days_intraday
                           if trade.signal_type == SignalType.INTRADAY_REBOUND
                           else self.backtest_config.max_hold_days_overnight)

                if hold_days >= max_hold:
                    exit_reason = "time"
                    exit_price = current_price

            # Execute exit if triggered
            if exit_reason:
                trade.update_exit(current_date, exit_price, exit_reason)
                trades_to_close.append(trade)

        # Close trades and update cash
        for trade in trades_to_close:
            self.open_trades.remove(trade)
            self.closed_trades.append(trade)
            self.cash += trade.position_value + trade.pnl

    async def _generate_signals(self, current_date: datetime, historical_data: Dict[str, pd.DataFrame]):
        """Generate trading signals for current date."""

        # Prepare stock data for screening
        stocks_data = {}

        for symbol, data in historical_data.items():
            try:
                # Get historical data up to current date
                historical_subset = data[data.index <= current_date].tail(30)  # Last 30 days

                if len(historical_subset) < 20:  # Need minimum history
                    continue

                # Create StockData object (simplified for backtesting)
                from ..data.models import StockInfo
                stock_info = StockInfo(symbol=symbol, company_name=f"Company {symbol}")

                stock_data = StockData(
                    symbol=symbol,
                    info=stock_info,
                    daily_data=historical_subset,
                    intraday_data=pd.DataFrame()  # Not available in backtest
                )

                stocks_data[symbol] = stock_data

            except Exception as e:
                logger.debug(f"Error preparing data for {symbol}: {e}")
                continue

        if not stocks_data:
            return

        # Generate signals based on time of day
        signals = []

        # Morning/intraday signals (9 AM - 12 PM)
        if 9 <= current_date.hour < 12:
            intraday_signals = self.screener.screen_intraday_rebounds(stocks_data, self.cash)
            signals.extend(intraday_signals)

        # Afternoon/overnight signals (2 PM - 3 PM)
        elif 14 <= current_date.hour < 15:
            overnight_signals = self.screener.screen_overnight_setups(stocks_data, self.cash)
            signals.extend(overnight_signals)

        # Execute signals
        for signal in signals[:5]:  # Limit to top 5 signals
            await self._execute_signal(signal, current_date, historical_data)

    async def _execute_signal(self, signal: TradingSignal, current_date: datetime, historical_data: Dict[str, pd.DataFrame]):
        """Execute a trading signal."""

        symbol_data = historical_data.get(signal.symbol)
        if symbol_data is None:
            return

        try:
            current_data = symbol_data.loc[current_date]
            entry_price = current_data['Close']
        except KeyError:
            return

        # Calculate position size
        position_sizing = self.risk_calculator.calculate_position_size(signal, self.cash)

        if position_sizing.shares == 0:
            return  # Can't afford position

        # Check if we have enough cash
        required_cash = position_sizing.position_value
        if required_cash > self.cash:
            return

        # Create backtest trade
        trade = BacktestTrade(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            signal_type=signal.signal_type,
            entry_date=current_date,
            entry_price=entry_price,
            shares=position_sizing.shares,
            position_value=position_sizing.position_value,
            stop_loss=signal.risk_params.stop_loss,
            take_profit=signal.risk_params.primary_take_profit,
            confidence_score=signal.confidence_score,
            rsi_entry=signal.context.rsi if signal.context else None
        )

        # Execute trade
        self.open_trades.append(trade)
        self.cash -= required_cash

        logger.debug(f"Executed {signal.signal_type.value} signal for {signal.symbol} at {entry_price}")

    def _close_remaining_trades(self, end_date: datetime, historical_data: Dict[str, pd.DataFrame]):
        """Close any remaining open trades at the end of backtest period."""

        for trade in self.open_trades[:]:  # Copy list to avoid modification during iteration
            symbol_data = historical_data.get(trade.symbol)
            if symbol_data is None:
                continue

            try:
                final_data = symbol_data.loc[end_date]
                exit_price = final_data['Close']

                trade.update_exit(end_date, exit_price, "backtest_end")
                self.closed_trades.append(trade)
                self.cash += trade.position_value + trade.pnl

            except KeyError:
                # Use last available price
                last_date = symbol_data.index[-1]
                exit_price = symbol_data.loc[last_date, 'Close']

                trade.update_exit(last_date, exit_price, "backtest_end")
                self.closed_trades.append(trade)
                self.cash += trade.position_value + trade.pnl

        self.open_trades.clear()

    def _update_daily_performance(self, current_date: datetime):
        """Update daily portfolio performance tracking."""

        # Calculate current portfolio value
        open_positions_value = sum(trade.position_value for trade in self.open_trades)
        total_portfolio_value = self.cash + open_positions_value

        self.daily_dates.append(current_date)
        self.daily_portfolio_values.append(total_portfolio_value)

    def _generate_results(self) -> BacktestResults:
        """Generate comprehensive backtest results."""

        results = BacktestResults(config=self.backtest_config, trades=self.closed_trades)

        # Calculate basic statistics
        results.calculate_summary_stats()

        # Create equity curve
        if self.daily_dates and self.daily_portfolio_values:
            results.equity_curve = pd.Series(
                self.daily_portfolio_values,
                index=pd.DatetimeIndex(self.daily_dates)
            )

            # Calculate additional metrics
            self._calculate_advanced_metrics(results)

        return results

    def _calculate_advanced_metrics(self, results: BacktestResults):
        """Calculate advanced performance metrics."""

        if results.equity_curve.empty:
            return

        # Returns calculation
        daily_returns = results.equity_curve.pct_change().dropna()

        if len(daily_returns) == 0:
            return

        # Annualized return
        total_days = (self.backtest_config.end_date - self.backtest_config.start_date).days
        total_return_decimal = results.total_return_pct / 100
        results.annualized_return = ((1 + total_return_decimal) ** (365 / total_days) - 1) * 100

        # Max drawdown
        rolling_max = results.equity_curve.expanding().max()
        drawdown = (results.equity_curve - rolling_max) / rolling_max
        results.max_drawdown = drawdown.min() * 100

        # Sharpe ratio
        excess_returns = daily_returns - (self.backtest_config.risk_free_rate / 365)
        if excess_returns.std() != 0:
            results.sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(365)

        # Sortino ratio
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() != 0:
            results.sortino_ratio = (excess_returns.mean() / downside_returns.std()) * np.sqrt(365)

        # Monthly returns
        monthly_equity = results.equity_curve.resample('M').last()
        results.monthly_returns = monthly_equity.pct_change().dropna() * 100


class StrategyOptimizer:
    """
    Strategy parameter optimization using backtesting.

    Tests multiple parameter combinations to find optimal settings
    for the screening strategies.
    """

    def __init__(self, config: TradingConfig):
        """Initialize strategy optimizer."""
        self.config = config

    async def optimize_parameters(
        self,
        historical_data: Dict[str, pd.DataFrame],
        parameter_ranges: Dict[str, List],
        optimization_metric: str = "sharpe_ratio"
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search.

        Args:
            historical_data: Historical OHLCV data
            parameter_ranges: Dictionary of parameter names to test ranges
            optimization_metric: Metric to optimize for

        Returns:
            Dictionary with optimal parameters and results
        """
        logger.info("Starting parameter optimization...")

        best_metric = float('-inf')
        best_params = {}
        all_results = []

        # Generate parameter combinations
        import itertools
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())

        combinations = list(itertools.product(*param_values))
        logger.info(f"Testing {len(combinations)} parameter combinations")

        for i, combination in enumerate(combinations):
            # Create parameter set
            params = dict(zip(param_names, combination))

            try:
                # Update configuration with new parameters
                test_config = self._update_config_with_params(params)

                # Create backtest configuration
                backtest_config = BacktestConfig(
                    start_date=datetime.now() - timedelta(days=365),
                    end_date=datetime.now() - timedelta(days=30),
                    initial_capital=100_000_000
                )

                # Run backtest
                engine = BacktestEngine(test_config, backtest_config)
                results = await engine.run_backtest(historical_data)

                # Extract optimization metric
                metric_value = getattr(results, optimization_metric, 0)

                # Store results
                result_data = {
                    'parameters': params,
                    'metric_value': metric_value,
                    'total_trades': results.total_trades,
                    'win_rate': results.win_rate,
                    'total_return_pct': results.total_return_pct
                }
                all_results.append(result_data)

                # Check if this is the best so far
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params.copy()

                logger.info(f"Combination {i+1}/{len(combinations)}: {optimization_metric}={metric_value:.3f}")

            except Exception as e:
                logger.error(f"Error testing combination {params}: {e}")
                continue

        return {
            'best_parameters': best_params,
            'best_metric_value': best_metric,
            'optimization_metric': optimization_metric,
            'all_results': all_results
        }

    def _update_config_with_params(self, params: Dict[str, Any]) -> TradingConfig:
        """Update configuration with new parameters."""
        config = TradingConfig()  # Start with default

        # Map parameter names to configuration attributes
        param_mapping = {
            'rsi_oversold': ('indicators', 'rsi_oversold'),
            'rsi_overbought': ('indicators', 'rsi_overbought'),
            'min_volume': ('screening_criteria', 'min_volume'),
            'max_risk_per_trade': ('risk_management', 'max_risk_per_trade'),
            'atr_period': ('indicators', 'atr_period'),
        }

        for param_name, param_value in params.items():
            if param_name in param_mapping:
                section, attr = param_mapping[param_name]
                section_obj = getattr(config, section)
                setattr(section_obj, attr, param_value)

        return config


# Utility functions for backtesting analysis
def create_sample_historical_data(symbols: List[str], days: int = 365) -> Dict[str, pd.DataFrame]:
    """Create sample historical data for testing backtesting functionality."""

    historical_data = {}
    start_date = datetime.now() - timedelta(days=days)

    for symbol in symbols:
        # Generate sample OHLCV data
        dates = pd.date_range(start=start_date, periods=days, freq='D')

        # Simulate price movement
        np.random.seed(hash(symbol) % 1000)  # Consistent seed per symbol
        returns = np.random.normal(0.001, 0.02, days)  # ~0.1% daily return, 2% volatility

        prices = [1000 + hash(symbol) % 5000]  # Starting price based on symbol
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # Generate OHLC from close prices
        data = []
        for i, close in enumerate(prices):
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.005)) if i > 0 else close

            # Ensure OHLC consistency
            high = max(high, open_price, close)
            low = min(low, open_price, close)

            volume = int(np.random.normal(1_500_000, 500_000))  # Random volume

            data.append({
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close, 2),
                'Volume': max(volume, 100_000)
            })

        df = pd.DataFrame(data, index=dates)
        historical_data[symbol] = df

    return historical_data


def generate_backtest_report(results: BacktestResults) -> str:
    """Generate a formatted backtest report."""

    report_lines = [
        "=" * 60,
        "BACKTESTING REPORT",
        "=" * 60,
        "",
        "CONFIGURATION:",
        f"Period: {results.config.start_date.date()} to {results.config.end_date.date()}",
        f"Initial Capital: IDR {results.config.initial_capital:,.0f}",
        f"Position Sizing: {results.config.position_sizing_method}",
        "",
        "PERFORMANCE SUMMARY:",
        f"Total Return: IDR {results.total_return:,.0f} ({results.total_return_pct:.2f}%)",
        f"Annualized Return: {results.annualized_return:.2f}%",
        f"Max Drawdown: {results.max_drawdown:.2f}%",
        f"Sharpe Ratio: {results.sharpe_ratio:.3f}",
        f"Sortino Ratio: {results.sortino_ratio:.3f}",
        "",
        "TRADE STATISTICS:",
        f"Total Trades: {results.total_trades}",
        f"Winning Trades: {results.winning_trades}",
        f"Losing Trades: {results.losing_trades}",
        f"Win Rate: {results.win_rate:.1f}%",
        f"Average Win: IDR {results.avg_win:,.0f}",
        f"Average Loss: IDR {results.avg_loss:,.0f}",
        f"Average Hold Time: {results.avg_hold_time:.1f} days",
        f"Largest Win: IDR {results.largest_win:,.0f}",
        f"Largest Loss: IDR {results.largest_loss:,.0f}",
        "",
        "STRATEGY BREAKDOWN:",
    ]

    for strategy, stats in results.strategy_performance.items():
        report_lines.extend([
            f"{strategy.replace

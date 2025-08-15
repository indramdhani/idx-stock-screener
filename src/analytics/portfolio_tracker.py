"""
Portfolio Tracker Module for Indonesian Stock Screener

This module provides real-time portfolio management and position tracking capabilities:
- Real-time position tracking
- Portfolio state management
- Risk monitoring
- Performance tracking
- Position sizing and allocation
- Trade execution tracking

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
import json
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    """Position status enumeration"""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"
    CANCELLED = "cancelled"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


@dataclass
class Position:
    """Individual position data structure"""
    symbol: str
    entry_date: datetime
    entry_price: float
    quantity: int
    strategy: str
    confidence_score: float

    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_amount: float = 0.0

    # Current status
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    # Exit information
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    realized_pnl_pct: Optional[float] = None

    # Position metadata
    position_id: str = field(default_factory=lambda: f"pos_{datetime.now().timestamp()}")
    status: PositionStatus = PositionStatus.OPEN
    notes: str = ""

    # Performance tracking
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    days_held: int = 0

    def update_current_price(self, price: float):
        """Update current price and calculate unrealized P&L"""
        self.current_price = price
        if self.status == PositionStatus.OPEN:
            price_change = price - self.entry_price
            self.unrealized_pnl = price_change * self.quantity
            self.unrealized_pnl_pct = price_change / self.entry_price

            # Update MFE and MAE
            if self.unrealized_pnl_pct > self.max_favorable_excursion:
                self.max_favorable_excursion = self.unrealized_pnl_pct
            if self.unrealized_pnl_pct < self.max_adverse_excursion:
                self.max_adverse_excursion = self.unrealized_pnl_pct

    def close_position(self, exit_price: float, exit_date: Optional[datetime] = None):
        """Close the position and calculate realized P&L"""
        self.exit_price = exit_price
        self.exit_date = exit_date or datetime.now()
        self.status = PositionStatus.CLOSED

        price_change = exit_price - self.entry_price
        self.realized_pnl = price_change * self.quantity
        self.realized_pnl_pct = price_change / self.entry_price
        self.days_held = (self.exit_date - self.entry_date).days

    def get_position_value(self) -> float:
        """Get current position value"""
        if self.current_price:
            return self.current_price * self.quantity
        return self.entry_price * self.quantity

    def to_dict(self) -> Dict:
        """Convert position to dictionary"""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'entry_date': self.entry_date.isoformat(),
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'strategy': self.strategy,
            'confidence_score': self.confidence_score,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_amount': self.risk_amount,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'exit_date': self.exit_date.isoformat() if self.exit_date else None,
            'exit_price': self.exit_price,
            'realized_pnl': self.realized_pnl,
            'realized_pnl_pct': self.realized_pnl_pct,
            'status': self.status.value,
            'notes': self.notes,
            'max_favorable_excursion': self.max_favorable_excursion,
            'max_adverse_excursion': self.max_adverse_excursion,
            'days_held': self.days_held
        }


@dataclass
class PortfolioState:
    """Current portfolio state snapshot"""
    timestamp: datetime = field(default_factory=datetime.now)

    # Portfolio values
    total_capital: float = 0.0
    cash_available: float = 0.0
    invested_capital: float = 0.0
    total_portfolio_value: float = 0.0

    # Performance metrics
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    day_pnl: float = 0.0
    day_pnl_pct: float = 0.0

    # Position metrics
    total_positions: int = 0
    open_positions: int = 0
    closed_positions: int = 0

    # Risk metrics
    total_risk_amount: float = 0.0
    risk_utilization_pct: float = 0.0
    max_position_size_pct: float = 0.0

    # Strategy breakdown
    strategy_allocation: Dict[str, float] = field(default_factory=dict)
    strategy_performance: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert portfolio state to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_capital': self.total_capital,
            'cash_available': self.cash_available,
            'invested_capital': self.invested_capital,
            'total_portfolio_value': self.total_portfolio_value,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl_pct,
            'day_pnl': self.day_pnl,
            'day_pnl_pct': self.day_pnl_pct,
            'total_positions': self.total_positions,
            'open_positions': self.open_positions,
            'closed_positions': self.closed_positions,
            'total_risk_amount': self.total_risk_amount,
            'risk_utilization_pct': self.risk_utilization_pct,
            'max_position_size_pct': self.max_position_size_pct,
            'strategy_allocation': self.strategy_allocation,
            'strategy_performance': self.strategy_performance
        }


class PortfolioTracker:
    """Real-time portfolio tracker and manager"""

    def __init__(
        self,
        initial_capital: float,
        max_positions: int = 10,
        max_risk_per_trade: float = 0.02,
        max_total_risk: float = 0.06
    ):
        """
        Initialize portfolio tracker

        Args:
            initial_capital: Starting capital amount (IDR)
            max_positions: Maximum number of concurrent positions
            max_risk_per_trade: Maximum risk per trade (as fraction)
            max_total_risk: Maximum total portfolio risk (as fraction)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_positions = max_positions
        self.max_risk_per_trade = max_risk_per_trade
        self.max_total_risk = max_total_risk

        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Position] = []

        # State tracking
        self.current_state = PortfolioState()
        self.state_history: List[PortfolioState] = []

        # Performance tracking
        self.daily_returns: List[float] = []
        self.daily_values: List[float] = [initial_capital]

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Portfolio tracker initialized with capital: {initial_capital:,.0f} IDR")

    async def add_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: int,
        strategy: str,
        confidence_score: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[str]:
        """
        Add a new position to the portfolio

        Returns:
            Position ID if successful, None if failed
        """
        try:
            # Check position limits
            if len(self.get_open_positions()) >= self.max_positions:
                self.logger.warning(f"Maximum positions ({self.max_positions}) reached")
                return None

            # Check if already have position in this symbol
            existing_positions = [p for p in self.positions.values()
                                if p.symbol == symbol and p.status == PositionStatus.OPEN]
            if existing_positions:
                self.logger.warning(f"Already have open position in {symbol}")
                return None

            # Calculate position value and risk
            position_value = entry_price * quantity
            risk_amount = self._calculate_risk_amount(entry_price, stop_loss, quantity)

            # Check risk limits
            if not self._validate_risk_limits(risk_amount):
                self.logger.warning(f"Position would exceed risk limits: {risk_amount:,.0f} IDR")
                return None

            # Check capital availability
            if position_value > self.current_state.cash_available:
                self.logger.warning(f"Insufficient cash for position: {position_value:,.0f} IDR")
                return None

            # Create position
            position = Position(
                symbol=symbol,
                entry_date=datetime.now(),
                entry_price=entry_price,
                quantity=quantity,
                strategy=strategy,
                confidence_score=confidence_score,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_amount=risk_amount
            )

            # Add to portfolio
            self.positions[position.position_id] = position
            self.logger.info(f"Added position: {symbol} ({quantity} shares) at {entry_price}")

            # Update portfolio state
            await self.update_portfolio_state()

            return position.position_id

        except Exception as e:
            self.logger.error(f"Error adding position: {e}")
            return None

    async def close_position(
        self,
        position_id: str,
        exit_price: float,
        notes: str = ""
    ) -> bool:
        """Close a position"""
        try:
            if position_id not in self.positions:
                self.logger.warning(f"Position {position_id} not found")
                return False

            position = self.positions[position_id]
            if position.status != PositionStatus.OPEN:
                self.logger.warning(f"Position {position_id} is not open")
                return False

            # Close the position
            position.close_position(exit_price)
            position.notes = notes

            # Move to history
            self.position_history.append(position)

            self.logger.info(
                f"Closed position: {position.symbol} "
                f"P&L: {position.realized_pnl:,.0f} IDR ({position.realized_pnl_pct:.2%})"
            )

            # Update portfolio state
            await self.update_portfolio_state()

            return True

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return False

    async def update_positions(self, price_data: Dict[str, float]):
        """Update all open positions with current prices"""
        try:
            updated_count = 0

            for position in self.get_open_positions():
                if position.symbol in price_data:
                    new_price = price_data[position.symbol]
                    position.update_current_price(new_price)
                    updated_count += 1

            if updated_count > 0:
                await self.update_portfolio_state()
                self.logger.debug(f"Updated {updated_count} positions")

        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")

    async def check_exit_conditions(self) -> List[Tuple[str, str]]:
        """
        Check exit conditions for all open positions

        Returns:
            List of (position_id, reason) tuples for positions that should be closed
        """
        exit_signals = []

        try:
            for position in self.get_open_positions():
                if not position.current_price:
                    continue

                # Check stop loss
                if position.stop_loss and position.current_price <= position.stop_loss:
                    exit_signals.append((position.position_id, "stop_loss"))

                # Check take profit
                elif position.take_profit and position.current_price >= position.take_profit:
                    exit_signals.append((position.position_id, "take_profit"))

                # Check max adverse excursion (emergency stop)
                elif position.max_adverse_excursion < -0.10:  # -10% emergency stop
                    exit_signals.append((position.position_id, "emergency_stop"))

        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")

        return exit_signals

    async def update_portfolio_state(self):
        """Update current portfolio state"""
        try:
            state = PortfolioState()

            open_positions = self.get_open_positions()
            closed_positions = self.get_closed_positions()

            # Calculate portfolio values
            invested_value = sum(p.get_position_value() for p in open_positions)
            total_unrealized_pnl = sum(p.unrealized_pnl for p in open_positions)
            total_realized_pnl = sum(p.realized_pnl or 0 for p in closed_positions)

            state.total_capital = self.current_capital
            state.invested_capital = invested_value
            state.cash_available = self.current_capital - invested_value
            state.total_portfolio_value = self.current_capital + total_unrealized_pnl + total_realized_pnl

            # Performance metrics
            state.total_pnl = total_unrealized_pnl + total_realized_pnl
            state.total_pnl_pct = state.total_pnl / self.initial_capital

            # Calculate day P&L
            if self.daily_values:
                previous_value = self.daily_values[-1]
                state.day_pnl = state.total_portfolio_value - previous_value
                state.day_pnl_pct = state.day_pnl / previous_value if previous_value > 0 else 0

            # Position metrics
            state.total_positions = len(self.positions) + len(self.position_history)
            state.open_positions = len(open_positions)
            state.closed_positions = len(closed_positions)

            # Risk metrics
            state.total_risk_amount = sum(p.risk_amount for p in open_positions)
            state.risk_utilization_pct = state.total_risk_amount / (self.current_capital * self.max_total_risk)

            if open_positions:
                max_position_value = max(p.get_position_value() for p in open_positions)
                state.max_position_size_pct = max_position_value / self.current_capital

            # Strategy breakdown
            strategy_allocation = {}
            strategy_performance = {}

            for position in open_positions:
                strategy = position.strategy
                value = position.get_position_value()
                pnl = position.unrealized_pnl

                strategy_allocation[strategy] = strategy_allocation.get(strategy, 0) + value
                strategy_performance[strategy] = strategy_performance.get(strategy, 0) + pnl

            state.strategy_allocation = strategy_allocation
            state.strategy_performance = strategy_performance

            # Update state
            self.current_state = state
            self.state_history.append(state)

            # Update daily tracking
            if not self.daily_values or self.daily_values[-1] != state.total_portfolio_value:
                self.daily_values.append(state.total_portfolio_value)
                if len(self.daily_values) > 1:
                    daily_return = (self.daily_values[-1] - self.daily_values[-2]) / self.daily_values[-2]
                    self.daily_returns.append(daily_return)

        except Exception as e:
            self.logger.error(f"Error updating portfolio state: {e}")

    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        return [p for p in self.positions.values() if p.status == PositionStatus.OPEN]

    def get_closed_positions(self) -> List[Position]:
        """Get all closed positions"""
        return [p for p in self.position_history if p.status == PositionStatus.CLOSED]

    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """Get open position by symbol"""
        for position in self.get_open_positions():
            if position.symbol == symbol:
                return position
        return None

    def _calculate_risk_amount(
        self,
        entry_price: float,
        stop_loss: Optional[float],
        quantity: int
    ) -> float:
        """Calculate risk amount for position"""
        if stop_loss:
            risk_per_share = entry_price - stop_loss
            return max(0, risk_per_share * quantity)
        else:
            # Default 2% risk if no stop loss
            return entry_price * quantity * 0.02

    def _validate_risk_limits(self, new_risk_amount: float) -> bool:
        """Validate if new position respects risk limits"""
        current_total_risk = sum(p.risk_amount for p in self.get_open_positions())
        total_risk_after = current_total_risk + new_risk_amount

        # Check individual trade risk limit
        if new_risk_amount > self.current_capital * self.max_risk_per_trade:
            return False

        # Check total risk limit
        if total_risk_after > self.current_capital * self.max_total_risk:
            return False

        return True

    def get_portfolio_summary(self) -> str:
        """Generate portfolio summary string"""
        try:
            summary = []
            summary.append("=" * 50)
            summary.append("PORTFOLIO SUMMARY")
            summary.append("=" * 50)

            state = self.current_state

            summary.append(f"Total Capital: {state.total_capital:,.0f} IDR")
            summary.append(f"Cash Available: {state.cash_available:,.0f} IDR")
            summary.append(f"Invested Capital: {state.invested_capital:,.0f} IDR")
            summary.append(f"Total Portfolio Value: {state.total_portfolio_value:,.0f} IDR")
            summary.append("")
            summary.append(f"Total P&L: {state.total_pnl:,.0f} IDR ({state.total_pnl_pct:.2%})")
            summary.append(f"Day P&L: {state.day_pnl:,.0f} IDR ({state.day_pnl_pct:.2%})")
            summary.append("")
            summary.append(f"Open Positions: {state.open_positions}")
            summary.append(f"Total Positions: {state.total_positions}")
            summary.append(f"Risk Utilization: {state.risk_utilization_pct:.1%}")

            if self.get_open_positions():
                summary.append("\nOPEN POSITIONS:")
                for position in self.get_open_positions():
                    summary.append(
                        f"  {position.symbol}: {position.quantity} @ {position.entry_price:,.0f} "
                        f"({position.unrealized_pnl:+,.0f} IDR, {position.unrealized_pnl_pct:+.2%})"
                    )

            summary.append("=" * 50)
            return "\n".join(summary)

        except Exception as e:
            self.logger.error(f"Error generating portfolio summary: {e}")
            return f"Error generating summary: {e}"

    def export_portfolio_state(self, filepath: Path) -> bool:
        """Export current portfolio state to file"""
        try:
            export_data = {
                'current_state': self.current_state.to_dict(),
                'open_positions': [p.to_dict() for p in self.get_open_positions()],
                'closed_positions': [p.to_dict() for p in self.get_closed_positions()],
                'daily_returns': self.daily_returns,
                'daily_values': self.daily_values
            }

            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            self.logger.info(f"Portfolio state exported to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting portfolio state: {e}")
            return False

    def get_performance_series(self) -> pd.Series:
        """Get portfolio performance as pandas Series"""
        try:
            if len(self.daily_values) < 2:
                return pd.Series(dtype=float)

            dates = pd.date_range(
                start=datetime.now() - timedelta(days=len(self.daily_returns)),
                periods=len(self.daily_returns),
                freq='D'
            )

            return pd.Series(self.daily_returns, index=dates)

        except Exception as e:
            self.logger.error(f"Error getting performance series: {e}")
            return pd.Series(dtype=float)

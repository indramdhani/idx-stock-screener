# -*- coding: utf-8 -*-
"""
Trading Signal Models for Indonesian Stock Screener
==================================================

Data models for representing trading signals, including entry/exit points,
risk parameters, and signal metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class SignalType(Enum):
    """Types of trading signals"""

    INTRADAY_REBOUND = "intraday_rebound"
    OVERNIGHT_SETUP = "overnight_setup"
    BREAKOUT = "breakout"
    PULLBACK = "pullback"
    REVERSAL = "reversal"


class SignalStatus(Enum):
    """Signal status tracking"""

    ACTIVE = "active"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    STOPPED_OUT = "stopped_out"
    PROFIT_TAKEN = "profit_taken"


class RiskLevel(Enum):
    """Risk level classification"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class TakeProfitLevel:
    """Individual take profit target"""

    price: float
    percentage: float  # Percentage of position to close
    reasoning: str = ""

    def __post_init__(self):
        """Validate take profit level"""
        if self.price <= 0:
            raise ValueError("Take profit price must be positive")

        if not 0 < self.percentage <= 100:
            raise ValueError("Take profit percentage must be between 0 and 100")


@dataclass
class RiskParameters:
    """Risk management parameters for a signal"""

    stop_loss: float
    take_profit_levels: List[TakeProfitLevel]
    risk_amount: float
    potential_reward: float
    risk_reward_ratio: float
    atr_multiplier: Optional[float] = None

    def __post_init__(self):
        """Validate risk parameters"""
        if self.stop_loss <= 0:
            raise ValueError("Stop loss must be positive")

        if not self.take_profit_levels:
            raise ValueError("At least one take profit level is required")

        if self.risk_amount <= 0:
            raise ValueError("Risk amount must be positive")

        if self.potential_reward <= 0:
            raise ValueError("Potential reward must be positive")

        if self.risk_reward_ratio <= 0:
            raise ValueError("Risk reward ratio must be positive")

        # Validate take profit percentages sum to 100%
        total_percentage = sum(tp.percentage for tp in self.take_profit_levels)
        if abs(total_percentage - 100.0) > 0.01:
            raise ValueError(f"Take profit percentages must sum to 100%, got {total_percentage}")

    @property
    def primary_take_profit(self) -> float:
        """Get the first/primary take profit price"""
        return self.take_profit_levels[0].price

    @property
    def max_take_profit(self) -> float:
        """Get the highest take profit price"""
        return max(tp.price for tp in self.take_profit_levels)


@dataclass
class PositionSizing:
    """Position sizing calculation"""

    shares: int
    position_value: float
    risk_amount: float
    risk_percentage: float
    lot_size: int = 100  # IDX standard lot size

    def __post_init__(self):
        """Validate position sizing"""
        if self.shares < 0:
            raise ValueError("Shares cannot be negative")

        if self.position_value < 0:
            raise ValueError("Position value cannot be negative")

        if self.risk_amount < 0:
            raise ValueError("Risk amount cannot be negative")

        if not 0 <= self.risk_percentage <= 100:
            raise ValueError("Risk percentage must be between 0 and 100")

    @property
    def lots(self) -> int:
        """Calculate number of lots"""
        return self.shares // self.lot_size

    @property
    def avg_price_per_share(self) -> float:
        """Calculate average price per share"""
        if self.shares == 0:
            return 0.0
        return self.position_value / self.shares


@dataclass
class SignalContext:
    """Context information for signal generation"""

    market_condition: str = "normal"  # normal, volatile, trending
    sector_sentiment: str = "neutral"  # bullish, bearish, neutral
    volume_analysis: str = "normal"    # high, normal, low
    technical_setup: str = ""          # Description of technical setup
    fundamental_notes: str = ""        # Any fundamental considerations

    # Technical indicator values at signal time
    rsi: Optional[float] = None
    ema_alignment: Optional[bool] = None
    vwap_position: Optional[str] = None  # above, below, at
    support_distance: Optional[float] = None
    resistance_distance: Optional[float] = None


@dataclass
class TradingSignal:
    """Complete trading signal with all parameters"""

    # Basic identification
    signal_id: str
    symbol: str
    signal_type: SignalType
    timestamp: datetime

    # Entry parameters
    entry_price: float
    entry_reasoning: str

    # Risk management
    risk_params: RiskParameters
    confidence_score: float  # 0.0 to 1.0

    # Optional parameters with defaults
    position_sizing: Optional[PositionSizing] = None
    risk_level: RiskLevel = RiskLevel.MEDIUM

    # Context and metadata
    context: SignalContext = field(default_factory=SignalContext)
    tags: List[str] = field(default_factory=list)

    # Tracking
    status: SignalStatus = SignalStatus.ACTIVE
    expires_at: Optional[datetime] = None

    # Performance tracking (filled after signal execution)
    actual_entry_price: Optional[float] = None
    actual_exit_price: Optional[float] = None
    actual_pnl: Optional[float] = None
    actual_pnl_pct: Optional[float] = None
    execution_notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate trading signal"""
        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")

        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")

        if not self.entry_reasoning.strip():
            raise ValueError("Entry reasoning cannot be empty")

        # Validate signal ID format
        if not self.signal_id or len(self.signal_id) < 5:
            raise ValueError("Signal ID must be at least 5 characters")

    @property
    def is_long_signal(self) -> bool:
        """Check if this is a long (buy) signal"""
        return self.risk_params.primary_take_profit > self.entry_price

    @property
    def is_short_signal(self) -> bool:
        """Check if this is a short (sell) signal"""
        return self.risk_params.primary_take_profit < self.entry_price

    @property
    def risk_amount_per_share(self) -> float:
        """Calculate risk amount per share"""
        return abs(self.entry_price - self.risk_params.stop_loss)

    @property
    def potential_reward_per_share(self) -> float:
        """Calculate potential reward per share (primary TP)"""
        return abs(self.risk_params.primary_take_profit - self.entry_price)

    @property
    def is_expired(self) -> bool:
        """Check if signal has expired"""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at

    @property
    def age_minutes(self) -> float:
        """Get signal age in minutes"""
        return (datetime.now() - self.timestamp).total_seconds() / 60

    def add_tag(self, tag: str) -> None:
        """Add a tag to the signal"""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the signal"""
        if tag in self.tags:
            self.tags.remove(tag)

    def add_execution_note(self, note: str) -> None:
        """Add an execution note"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.execution_notes.append(f"[{timestamp}] {note}")

    def update_position_sizing(self, account_balance: float, risk_per_trade: float) -> None:
        """Calculate and update position sizing"""
        max_risk_amount = account_balance * risk_per_trade
        risk_per_share = self.risk_amount_per_share

        if risk_per_share <= 0:
            shares = 0
        else:
            shares = int(max_risk_amount / risk_per_share)
            # Ensure we can afford at least one lot
            if shares < 100:  # IDX lot size
                shares = 0

        position_value = shares * self.entry_price
        actual_risk = shares * risk_per_share
        risk_percentage = (actual_risk / account_balance) * 100 if account_balance > 0 else 0

        self.position_sizing = PositionSizing(
            shares=shares,
            position_value=position_value,
            risk_amount=actual_risk,
            risk_percentage=risk_percentage
        )

    def calculate_pnl(self, exit_price: float) -> tuple[float, float]:
        """Calculate P&L for a given exit price"""
        if self.position_sizing is None:
            return 0.0, 0.0

        if self.is_long_signal:
            pnl = (exit_price - self.entry_price) * self.position_sizing.shares
        else:
            pnl = (self.entry_price - exit_price) * self.position_sizing.shares

        pnl_pct = (pnl / self.position_sizing.position_value) * 100 if self.position_sizing.position_value > 0 else 0.0

        return pnl, pnl_pct

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary representation"""
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'timestamp': self.timestamp.isoformat(),
            'entry_price': self.entry_price,
            'entry_reasoning': self.entry_reasoning,
            'stop_loss': self.risk_params.stop_loss,
            'take_profit_levels': [
                {
                    'price': tp.price,
                    'percentage': tp.percentage,
                    'reasoning': tp.reasoning
                } for tp in self.risk_params.take_profit_levels
            ],
            'risk_reward_ratio': self.risk_params.risk_reward_ratio,
            'confidence_score': self.confidence_score,
            'risk_level': self.risk_level.value,
            'status': self.status.value,
            'tags': self.tags,
            'position_sizing': {
                'shares': self.position_sizing.shares,
                'position_value': self.position_sizing.position_value,
                'risk_amount': self.position_sizing.risk_amount,
                'risk_percentage': self.position_sizing.risk_percentage,
            } if self.position_sizing else None,
            'context': {
                'market_condition': self.context.market_condition,
                'sector_sentiment': self.context.sector_sentiment,
                'volume_analysis': self.context.volume_analysis,
                'technical_setup': self.context.technical_setup,
                'rsi': self.context.rsi,
                'ema_alignment': self.context.ema_alignment,
            },
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'age_minutes': self.age_minutes,
        }

    def to_telegram_format(self) -> str:
        """Format signal for Telegram notification"""
        signal_emoji = "🚀" if self.is_long_signal else "📉"
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}[self.risk_level.value]

        message_lines = [
            f"{signal_emoji} <b>TRADING SIGNAL</b> {signal_emoji}",
            "",
            f"📊 Symbol: <b>{self.symbol}</b>",
            f"🎯 Type: {self.signal_type.value.replace('_', ' ').title()}",
            f"{risk_emoji} Risk Level: {self.risk_level.value.title()}",
            "",
            f"💰 Entry: <b>IDR {self.entry_price:,.0f}</b>",
            f"🛑 Stop Loss: IDR {self.risk_params.stop_loss:,.0f}",
        ]

        # Add take profit levels
        for i, tp in enumerate(self.risk_params.take_profit_levels, 1):
            percentage_text = f" ({tp.percentage:.0f}%)" if len(self.risk_params.take_profit_levels) > 1 else ""
            message_lines.append(f"🎯 Take Profit {i}: IDR {tp.price:,.0f}{percentage_text}")

        message_lines.extend([
            "",
            f"📈 Risk/Reward: <b>{self.risk_params.risk_reward_ratio:.1f}:1</b>",
            f"🎲 Confidence: {self.confidence_score:.0%}",
        ])

        # Add position sizing if available
        if self.position_sizing:
            message_lines.extend([
                "",
                f"💼 Position Size: {self.position_sizing.shares:,} shares ({self.position_sizing.lots} lots)",
                f"💵 Position Value: IDR {self.position_sizing.position_value:,.0f}",
                f"⚠️ Risk Amount: IDR {self.position_sizing.risk_amount:,.0f} ({self.position_sizing.risk_percentage:.1f}%)",
            ])

        message_lines.extend([
            "",
            f"📋 Reasoning: {self.entry_reasoning}",
        ])

        # Add context information
        if self.context.technical_setup:
            message_lines.append(f"🔧 Setup: {self.context.technical_setup}")

        # Add tags if any
        if self.tags:
            tag_text = " ".join(f"#{tag}" for tag in self.tags)
            message_lines.append(f"🏷️ {tag_text}")

        message_lines.extend([
            "",
            f"⏰ {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"🆔 {self.signal_id}",
        ])

        return "\n".join(message_lines)


@dataclass
class SignalPerformanceMetrics:
    """Performance metrics for signal tracking"""

    total_signals: int = 0
    winning_signals: int = 0
    losing_signals: int = 0
    total_pnl: float = 0.0

    # Signal type breakdown
    signal_type_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage"""
        if self.total_signals == 0:
            return 0.0
        return (self.winning_signals / self.total_signals) * 100

    @property
    def loss_rate(self) -> float:
        """Calculate loss rate percentage"""
        if self.total_signals == 0:
            return 0.0
        return (self.losing_signals / self.total_signals) * 100

    @property
    def average_pnl(self) -> float:
        """Calculate average P&L per signal"""
        if self.total_signals == 0:
            return 0.0
        return self.total_pnl / self.total_signals

    def add_signal_result(self, signal: TradingSignal, pnl: float) -> None:
        """Add a signal result to performance metrics"""
        self.total_signals += 1
        self.total_pnl += pnl

        if pnl > 0:
            self.winning_signals += 1
        elif pnl < 0:
            self.losing_signals += 1

        # Update signal type performance
        signal_type = signal.signal_type.value
        if signal_type not in self.signal_type_performance:
            self.signal_type_performance[signal_type] = {
                'count': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0.0
            }

        self.signal_type_performance[signal_type]['count'] += 1
        self.signal_type_performance[signal_type]['total_pnl'] += pnl

        if pnl > 0:
            self.signal_type_performance[signal_type]['wins'] += 1
        elif pnl < 0:
            self.signal_type_performance[signal_type]['losses'] += 1

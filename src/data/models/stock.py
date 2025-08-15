# -*- coding: utf-8 -*-
"""
Stock Data Models for Indonesian Stock Screener
==============================================

Data models for representing stock information, including OHLCV data,
market information, and derived metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class StockInfo:
    """Basic stock information from market data provider"""

    symbol: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    shares_outstanding: Optional[int] = None
    currency: str = "IDR"
    exchange: str = "IDX"

    # Additional metadata
    last_updated: Optional[datetime] = None
    data_source: str = "yahoo"

    def __post_init__(self):
        """Post-initialization processing"""
        if self.last_updated is None:
            self.last_updated = datetime.now()

    @property
    def is_valid(self) -> bool:
        """Check if stock info contains minimum required data"""
        return bool(self.symbol and self.company_name)


@dataclass
class StockPriceData:
    """OHLCV price data for a stock"""

    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    adjusted_close: Optional[float] = None

    def __post_init__(self):
        """Validate price data consistency"""
        if self.high_price < max(self.open_price, self.close_price):
            raise ValueError(f"High price ({self.high_price}) cannot be lower than open ({self.open_price}) or close ({self.close_price})")

        if self.low_price > min(self.open_price, self.close_price):
            raise ValueError(f"Low price ({self.low_price}) cannot be higher than open ({self.open_price}) or close ({self.close_price})")

        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative: {self.volume}")

    @property
    def typical_price(self) -> float:
        """Calculate typical price (HLC/3)"""
        return (self.high_price + self.low_price + self.close_price) / 3

    @property
    def price_range(self) -> float:
        """Calculate price range (High - Low)"""
        return self.high_price - self.low_price

    @property
    def body_size(self) -> float:
        """Calculate candle body size"""
        return abs(self.close_price - self.open_price)

    @property
    def is_green(self) -> bool:
        """Check if candle is green (close > open)"""
        return self.close_price > self.open_price

    @property
    def price_change_pct(self) -> float:
        """Calculate percentage change from open to close"""
        if self.open_price == 0:
            return 0.0
        return ((self.close_price - self.open_price) / self.open_price) * 100


@dataclass
class TechnicalIndicators:
    """Container for technical indicator values"""

    symbol: str
    timestamp: datetime

    # Trend indicators
    ema_5: Optional[float] = None
    ema_13: Optional[float] = None
    ema_21: Optional[float] = None
    sma_20: Optional[float] = None

    # Momentum indicators
    rsi: Optional[float] = None
    rsi_period: int = 14

    # Volatility indicators
    atr: Optional[float] = None
    atr_period: int = 14
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    bollinger_middle: Optional[float] = None

    # Volume indicators
    vwap: Optional[float] = None
    volume_sma: Optional[float] = None
    volume_ratio: Optional[float] = None

    # Custom indicators
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None

    @property
    def is_ema_uptrend(self) -> bool:
        """Check if EMAs are in uptrend alignment"""
        if not all([self.ema_5, self.ema_13, self.ema_21]):
            return False
        return self.ema_5 > self.ema_13 > self.ema_21

    @property
    def is_rsi_oversold(self, threshold: float = 30) -> bool:
        """Check if RSI indicates oversold condition"""
        return self.rsi is not None and self.rsi < threshold

    @property
    def is_rsi_overbought(self, threshold: float = 70) -> bool:
        """Check if RSI indicates overbought condition"""
        return self.rsi is not None and self.rsi > threshold


@dataclass
class StockData:
    """Complete stock data container"""

    symbol: str
    info: StockInfo
    daily_data: pd.DataFrame
    intraday_data: pd.DataFrame
    indicators: Optional[TechnicalIndicators] = None
    last_updated: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate and process stock data"""
        if self.daily_data.empty:
            raise ValueError(f"Daily data cannot be empty for {self.symbol}")

        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in self.daily_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    @property
    def current_price(self) -> float:
        """Get the most recent closing price"""
        if self.daily_data.empty:
            raise ValueError("No daily data available")
        return float(self.daily_data['Close'].iloc[-1])

    @property
    def previous_close(self) -> float:
        """Get the previous day's closing price"""
        if len(self.daily_data) < 2:
            raise ValueError("Insufficient data for previous close")
        return float(self.daily_data['Close'].iloc[-2])

    @property
    def daily_change(self) -> float:
        """Calculate daily price change"""
        return self.current_price - self.previous_close

    @property
    def daily_change_pct(self) -> float:
        """Calculate daily percentage change"""
        if self.previous_close == 0:
            return 0.0
        return (self.daily_change / self.previous_close) * 100

    @property
    def daily_volume(self) -> int:
        """Get current day's volume"""
        if self.daily_data.empty:
            raise ValueError("No daily data available")
        return int(self.daily_data['Volume'].iloc[-1])

    @property
    def average_volume(self, periods: int = 20) -> float:
        """Calculate average volume over specified periods"""
        if len(self.daily_data) < periods:
            periods = len(self.daily_data)
        return float(self.daily_data['Volume'].tail(periods).mean())

    @property
    def volume_ratio(self) -> float:
        """Calculate volume ratio vs average"""
        avg_vol = self.average_volume()
        if avg_vol == 0:
            return 0.0
        return self.daily_volume / avg_vol

    @property
    def market_cap(self) -> Optional[float]:
        """Get market capitalization if available"""
        return self.info.market_cap

    @property
    def intraday_high(self) -> Optional[float]:
        """Get intraday high if intraday data available"""
        if self.intraday_data.empty:
            return None
        return float(self.intraday_data['High'].max())

    @property
    def intraday_low(self) -> Optional[float]:
        """Get intraday low if intraday data available"""
        if self.intraday_data.empty:
            return None
        return float(self.intraday_data['Low'].min())

    @property
    def intraday_range_pct(self) -> Optional[float]:
        """Calculate intraday range as percentage"""
        if self.intraday_data.empty:
            return None

        high = self.intraday_high
        low = self.intraday_low

        if high is None or low is None or low == 0:
            return None

        return ((high - low) / low) * 100

    def get_price_at_time(self, target_time: datetime) -> Optional[float]:
        """Get price closest to specified time from intraday data"""
        if self.intraday_data.empty:
            return None

        # Find closest timestamp
        time_diff = abs(self.intraday_data.index - target_time)
        closest_idx = time_diff.idxmin()

        return float(self.intraday_data.loc[closest_idx, 'Close'])

    def calculate_support_resistance(self, lookback_periods: int = 20) -> tuple[Optional[float], Optional[float]]:
        """Calculate support and resistance levels"""
        if len(self.daily_data) < lookback_periods:
            return None, None

        recent_data = self.daily_data.tail(lookback_periods)

        # Simple support/resistance based on recent highs and lows
        resistance = float(recent_data['High'].max())
        support = float(recent_data['Low'].min())

        return support, resistance

    def is_near_level(self, price: float, level: float, tolerance_pct: float = 2.0) -> bool:
        """Check if price is near a specific level within tolerance"""
        if level == 0:
            return False

        deviation_pct = abs((price - level) / level) * 100
        return deviation_pct <= tolerance_pct

    def get_data_quality_score(self) -> float:
        """Calculate data quality score (0-1)"""
        score = 0.0
        max_score = 5.0

        # Check data completeness
        if not self.daily_data.empty:
            score += 1.0

        if not self.intraday_data.empty:
            score += 1.0

        # Check data freshness (within last 2 days)
        if not self.daily_data.empty:
            last_date = self.daily_data.index[-1]
            days_old = (datetime.now() - last_date.to_pydatetime()).days
            if days_old <= 2:
                score += 1.0

        # Check for missing values
        if not self.daily_data.empty:
            missing_pct = self.daily_data.isnull().sum().sum() / (len(self.daily_data) * len(self.daily_data.columns))
            if missing_pct < 0.1:  # Less than 10% missing
                score += 1.0

        # Check volume data quality
        if not self.daily_data.empty and 'Volume' in self.daily_data.columns:
            zero_volume_pct = (self.daily_data['Volume'] == 0).sum() / len(self.daily_data)
            if zero_volume_pct < 0.1:  # Less than 10% zero volume days
                score += 1.0

        return score / max_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert stock data to dictionary representation"""
        return {
            'symbol': self.symbol,
            'current_price': self.current_price,
            'daily_change': self.daily_change,
            'daily_change_pct': self.daily_change_pct,
            'volume': self.daily_volume,
            'average_volume': self.average_volume(),
            'volume_ratio': self.volume_ratio,
            'market_cap': self.market_cap,
            'data_quality_score': self.get_data_quality_score(),
            'last_updated': self.last_updated.isoformat(),
            'company_name': self.info.company_name,
            'sector': self.info.sector,
        }


@dataclass
class StockScreeningResult:
    """Result from stock screening process"""

    stock_data: StockData
    meets_criteria: bool
    screening_scores: Dict[str, float]
    reasons: List[str]
    confidence_score: float

    @property
    def symbol(self) -> str:
        """Get stock symbol"""
        return self.stock_data.symbol

    @property
    def current_price(self) -> float:
        """Get current stock price"""
        return self.stock_data.current_price

    def add_reason(self, reason: str) -> None:
        """Add screening reason"""
        if reason not in self.reasons:
            self.reasons.append(reason)

    def get_summary(self) -> Dict[str, Any]:
        """Get screening result summary"""
        return {
            'symbol': self.symbol,
            'price': self.current_price,
            'meets_criteria': self.meets_criteria,
            'confidence_score': self.confidence_score,
            'reasons': self.reasons,
            'scores': self.screening_scores,
        }

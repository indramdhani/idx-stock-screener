# -*- coding: utf-8 -*-
"""
Indonesian Stock Intraday Screener – v1.0 (Refactored)
======================================================

Modernized version of the Indonesian stock screener with modular architecture,
enhanced configuration management, and improved data validation.

Features:
* Modular design with separate concerns
* Pydantic-based configuration validation
* Enhanced data quality checks
* Improved error handling and logging
* Compatibility with new architecture components

⚠️  Reminder: Yahoo Finance quotes for BEI are delayed ±10‑15 m.
Confirm real‑time prices before trading.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import sys
from pathlib import Path
from typing import List, Optional
import warnings

import pandas as pd
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import load_config, TradingConfig
from src.data import IDXDataCollector, StockDataValidator, StockData
from src.data.models import SignalType, TradingSignal, RiskParameters, TakeProfitLevel

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Create logs directory
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configuration file path
CONFIG_PATH = Path(__file__).parent / "src" / "config" / "trading_config.yaml"


class LegacyScreener:
    """
    Legacy screener implementation maintaining backward compatibility
    while leveraging new modular architecture.
    """

    def __init__(self, config: TradingConfig):
        """Initialize legacy screener with modern components."""
        self.config = config
        self.data_collector = IDXDataCollector(config)
        self.data_validator = StockDataValidator(config.dict())

        # Legacy parameters for backward compatibility
        self.capital_limit_idr = 500_000
        self.volume_min = 1_000_000
        self.move_pct_intraday = 0.8
        self.move_pct_overnight = -2.0
        self.tp_pct_intraday = 1.0
        self.sl_pct_intraday = -0.7
        self.tp_pct_overnight = 2.0
        self.sl_pct_overnight = -2.0

        # Feature flags from config
        self.enable_vwap_filter = config.enable_vwap_filter
        self.enable_atr_tp_sl = config.enable_atr_tp_sl
        self.vwap_max_dev_pct = 1.0
        self.atr_period = config.indicators.atr_period

        logger.info("Legacy screener initialized with modern architecture")

    def _calc_lot_afford(self, price: float, capital: int, lot_size: int = 100) -> int:
        """Calculate affordable lots for given price and capital."""
        return int(capital // (price * lot_size))

    def _calc_vwap(self, intraday_data: pd.DataFrame) -> float:
        """Calculate VWAP from intraday data."""
        if intraday_data.empty:
            return 0.0

        typical_price = (intraday_data['High'] + intraday_data['Low'] + intraday_data['Close']) / 3
        volume_price = typical_price * intraday_data['Volume']

        total_volume = intraday_data['Volume'].sum()
        if total_volume == 0:
            return intraday_data['Close'].mean()

        return volume_price.sum() / total_volume

    def _calc_atr(self, daily_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR from daily data."""
        if len(daily_data) < period:
            return 0.0

        high_low = daily_data['High'] - daily_data['Low']
        high_close = (daily_data['High'] - daily_data['Close'].shift()).abs()
        low_close = (daily_data['Low'] - daily_data['Close'].shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]

        return atr if not pd.isna(atr) else 0.0

    def _screen_candidates(
        self,
        stocks_data: dict[str, StockData],
        move_condition,
        tp_pct: float,
        sl_pct: float,
        signal_type: SignalType
    ) -> pd.DataFrame:
        """Screen stock candidates based on criteria."""
        candidates = []

        for symbol, stock_data in stocks_data.items():
            try:
                daily_data = stock_data.daily_data
                intraday_data = stock_data.intraday_data

                if daily_data.empty:
                    continue

                # Calculate basic metrics
                open_price = daily_data['Open'].iloc[0]
                current_price = stock_data.current_price
                volume = stock_data.daily_volume

                pct_change = ((current_price - open_price) / open_price) * 100

                # Apply movement condition
                if not move_condition(pct_change):
                    continue

                # Volume filter
                if volume < self.volume_min:
                    continue

                # Capital limit filter (per lot)
                if (current_price * 100) > self.capital_limit_idr:
                    continue

                # Optional VWAP filter
                if self.enable_vwap_filter and not intraday_data.empty:
                    vwap = self._calc_vwap(intraday_data)
                    if vwap > 0:
                        vwap_dev_pct = abs((current_price - vwap) / vwap) * 100
                        if vwap_dev_pct > self.vwap_max_dev_pct:
                            continue

                # Calculate TP/SL
                if self.enable_atr_tp_sl:
                    atr = self._calc_atr(daily_data, self.atr_period)
                    if atr > 0:
                        if signal_type == SignalType.INTRADAY_REBOUND:
                            tp = current_price + (atr * 1.5)  # ATR TP multiplier
                            sl = current_price - (atr * 1.0)  # ATR SL multiplier
                        else:  # Overnight
                            tp = current_price + (atr * 2.5)
                            sl = current_price - (atr * 1.5)
                    else:
                        # Fallback to percentage
                        tp = current_price * (1 + tp_pct / 100)
                        sl = current_price * (1 + sl_pct / 100)
                else:
                    # Use percentage TP/SL
                    tp = current_price * (1 + tp_pct / 100)
                    sl = current_price * (1 + sl_pct / 100)

                # Calculate affordable lots
                lots_affordable = self._calc_lot_afford(current_price, self.capital_limit_idr)

                candidates.append({
                    'symbol': symbol,
                    'price': round(current_price, 2),
                    'pct_change': round(pct_change, 2),
                    'volume': int(volume),
                    'tp': round(tp, 2),
                    'sl': round(sl, 2),
                    'lots_affordable': lots_affordable
                })

            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue

        # Convert to DataFrame and sort
        if not candidates:
            return pd.DataFrame()

        df = pd.DataFrame(candidates)

        # Sort based on signal type
        if signal_type == SignalType.INTRADAY_REBOUND:
            return df.sort_values('pct_change', ascending=False)  # Highest gain first
        else:
            return df.sort_values('pct_change', ascending=True)   # Most negative first

    async def morning_breakout(self, stocks_data: dict[str, StockData]) -> pd.DataFrame:
        """Screen for morning breakout opportunities."""
        logger.info("Screening for morning breakout opportunities...")

        return self._screen_candidates(
            stocks_data,
            move_condition=lambda pct: pct >= self.move_pct_intraday,
            tp_pct=self.tp_pct_intraday,
            sl_pct=self.sl_pct_intraday,
            signal_type=SignalType.INTRADAY_REBOUND
        )

    async def afternoon_rebound(self, stocks_data: dict[str, StockData]) -> pd.DataFrame:
        """Screen for afternoon rebound opportunities (overnight holds)."""
        logger.info("Screening for afternoon rebound opportunities...")

        # For afternoon screening, we can use the last 20 minutes average
        # This is a simplified version - could be enhanced with intraday analysis
        return self._screen_candidates(
            stocks_data,
            move_condition=lambda pct: pct <= self.move_pct_overnight,
            tp_pct=self.tp_pct_overnight,
            sl_pct=self.sl_pct_overnight,
            signal_type=SignalType.OVERNIGHT_SETUP
        )


async def main(
    tickers: Optional[List[str]] = None,
    capital: int = 500_000,
    config_path: Optional[Path] = None
) -> None:
    """
    Main screening function with enhanced error handling and logging.

    Args:
        tickers: List of stock symbols to screen
        capital: Capital limit in IDR
        config_path: Path to configuration file
    """
    # Setup
    tz = dt.timezone(dt.timedelta(hours=7))
    now = dt.datetime.now(tz)

    logger.info("="*70)
    logger.info(f"Indonesian Stock Screener v1.0 - {now.strftime('%Y-%m-%d %H:%M:%S')} WIB")
    logger.info(f"Capital limit: Rp {capital:,}")
    logger.info("="*70)

    try:
        # Load configuration
        config = load_config(config_path or CONFIG_PATH)
        logger.info(f"Configuration loaded: VWAP filter={config.enable_vwap_filter}, ATR TP/SL={config.enable_atr_tp_sl}")

        # Initialize screener
        screener = LegacyScreener(config)

        # Update capital limit
        screener.capital_limit_idr = capital

        # Use provided tickers or default from config
        if tickers is None:
            tickers = config.default_tickers

        logger.info(f"Screening {len(tickers)} stocks...")

        # Fetch stock data
        stocks_data = await screener.data_collector.fetch_realtime_data(tickers)

        if not stocks_data:
            logger.error("No stock data could be fetched. Please check your internet connection.")
            return

        logger.info(f"Successfully fetched data for {len(stocks_data)} stocks")

        # Validate data quality
        validation_summary = screener.data_validator.get_validation_summary(
            screener.data_validator.validate_multiple_stocks(stocks_data)
        )

        logger.info(f"Data validation: {validation_summary['valid_stocks']}/{validation_summary['total_stocks']} stocks passed validation")

        # Filter to valid stocks only
        valid_stocks = screener.data_validator.filter_valid_stocks(stocks_data, min_quality_score=0.7)

        if not valid_stocks:
            logger.error("No stocks passed data validation. Please try again later.")
            return

        # Screen for opportunities
        morning_results = await screener.morning_breakout(valid_stocks)
        afternoon_results = await screener.afternoon_rebound(valid_stocks)

        # Display results
        pd.set_option("display.float_format", "{:.2f}".format)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        print("\n" + "="*50)
        print("🚀 Morning Breakout Opportunities (Buy AM - Sell PM)")
        print("="*50)
        if not morning_results.empty:
            print(morning_results[['symbol', 'price', 'pct_change', 'tp', 'sl', 'lots_affordable']].head(10))
        else:
            print("No morning breakout opportunities found.")

        print("\n" + "="*50)
        print("🌙 Afternoon Rebound Opportunities (Buy PM - Sell AM)")
        print("="*50)
        if not afternoon_results.empty:
            print(afternoon_results[['symbol', 'price', 'pct_change', 'tp', 'sl', 'lots_affordable']].head(10))
        else:
            print("No afternoon rebound opportunities found.")

        # Save results to CSV
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        if not morning_results.empty:
            morning_file = LOGS_DIR / f"morning_breakout_{timestamp}.csv"
            morning_results.to_csv(morning_file, index=False)
            logger.info(f"Morning results saved to {morning_file}")

        if not afternoon_results.empty:
            afternoon_file = LOGS_DIR / f"afternoon_rebound_{timestamp}.csv"
            afternoon_results.to_csv(afternoon_file, index=False)
            logger.info(f"Afternoon results saved to {afternoon_file}")

        # Summary statistics
        total_opportunities = len(morning_results) + len(afternoon_results)
        logger.info(f"Screening completed: {total_opportunities} total opportunities found")

        if validation_summary.get('error_summary'):
            logger.warning(f"Data issues detected: {validation_summary['error_summary']}")

    except KeyboardInterrupt:
        logger.info("Screening interrupted by user")
    except Exception as e:
        logger.error(f"Screening failed: {e}")
        raise
    finally:
        logger.info("Screening session ended")


def run_quick_screen(symbols: Optional[List[str]] = None, capital: int = 500_000) -> None:
    """
    Quick screening function for immediate use.

    Args:
        symbols: Stock symbols to screen (uses defaults if None)
        capital: Capital limit in IDR
    """
    try:
        asyncio.run(main(symbols, capital))
    except Exception as e:
        logger.error(f"Quick screen failed: {e}")


if __name__ == "__main__":
    # Default run with backward compatibility
    run_quick_screen()

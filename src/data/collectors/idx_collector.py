# -*- coding: utf-8 -*-
"""
IDX Data Collector for Indonesian Stock Screener
===============================================

Collects Indonesian stock data from various sources with data validation
and quality checks. Supports Yahoo Finance as primary source with fallback
options for reliable data collection.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import pandas as pd
import yfinance as yf
import requests
from loguru import logger

from ..models.stock import StockData, StockInfo, StockPriceData
from ...config.settings import TradingConfig


class DataCollectionError(Exception):
    """Custom exception for data collection errors"""
    pass


class IDXDataCollector:
    """
    Collects Indonesian stock data from various sources with validation
    and quality checks.
    """

    def __init__(self, config: TradingConfig):
        """
        Initialize IDX data collector.

        Args:
            config: Trading configuration containing data collection parameters
        """
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Suppress yfinance warnings
        warnings.filterwarnings('ignore', category=FutureWarning)

        # IDX stock symbols - Top liquid stocks for initial implementation
        self._idx_symbols = self._load_idx_symbols()

        # Data validation cache
        self._validation_cache = {}
        self._failed_symbols = set()

        logger.info(f"IDX Data Collector initialized with {len(self._idx_symbols)} symbols")

    def _load_idx_symbols(self) -> List[str]:
        """
        Load IDX stock symbols with .JK suffix for Yahoo Finance.

        Returns:
            List of IDX stock symbols
        """
        # Start with config default tickers
        symbols = self.config.default_tickers.copy()

        # Add more liquid IDX stocks
        # # Latest LQ45 stocks (mid-2025)
        additional_symbols = [
            "ACES.JK", "ADMR.JK", "ADRO.JK", "AKRA.JK", "AMRT.JK",
            "ANTM.JK", "ARTO.JK", "ASII.JK", "BBCA.JK", "BBNI.JK",
            "BBRI.JK", "BBTN.JK", "BMRI.JK", "BRIS.JK", "BRPT.JK",
            "CPIN.JK", "CTRA.JK", "ESSA.JK", "EXCL.JK", "GOTO.JK",
            "ICBP.JK", "INCO.JK", "INDF.JK", "INKP.JK", "ISAT.JK",
            "ITMG.JK", "JPFA.JK", "JSMR.JK", "KLBF.JK", "MAPA.JK",
            "MAPI.JK", "MBMA.JK", "MDKA.JK", "MEDC.JK", "PGAS.JK",
            "PGEO.JK", "PTBA.JK", "SIDO.JK", "SMGR.JK", "SMRA.JK",
            "TLKM.JK", "TOWR.JK", "UNTR.JK", "UNVR.JK"
        ]

        # Combine and remove duplicates
        all_symbols = list(set(symbols + additional_symbols))

        logger.info(f"Loaded {len(all_symbols)} IDX symbols")
        return all_symbols

    async def fetch_realtime_data(
        self,
        symbols: Optional[List[str]] = None,
        max_workers: int = 10
    ) -> Dict[str, StockData]:
        """
        Fetch real-time stock data asynchronously.

        Args:
            symbols: List of stock symbols to fetch. If None, uses default symbols.
            max_workers: Maximum number of concurrent workers

        Returns:
            Dictionary mapping symbol to StockData

        Raises:
            DataCollectionError: If data collection fails critically
        """
        if symbols is None:
            symbols = self._idx_symbols

        logger.info(f"Fetching real-time data for {len(symbols)} symbols")
        start_time = time.time()

        # Filter out previously failed symbols (with some retry logic)
        symbols_to_fetch = [s for s in symbols if s not in self._failed_symbols]

        if len(symbols_to_fetch) != len(symbols):
            logger.warning(f"Skipping {len(symbols) - len(symbols_to_fetch)} previously failed symbols")

        stock_data = {}
        failed_symbols = []

        # Use ThreadPoolExecutor for concurrent data fetching
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._fetch_single_stock, symbol): symbol
                for symbol in symbols_to_fetch
            }

            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per stock
                    if result:
                        stock_data[symbol] = result
                    else:
                        failed_symbols.append(symbol)
                except Exception as e:
                    logger.error(f"Failed to fetch data for {symbol}: {e}")
                    failed_symbols.append(symbol)

        # Update failed symbols cache
        self._failed_symbols.update(failed_symbols)

        elapsed_time = time.time() - start_time
        success_rate = len(stock_data) / len(symbols_to_fetch) * 100 if symbols_to_fetch else 0

        logger.info(
            f"Data collection completed in {elapsed_time:.2f}s. "
            f"Success rate: {success_rate:.1f}% ({len(stock_data)}/{len(symbols_to_fetch)})"
        )

        if failed_symbols:
            logger.warning(f"Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols[:10]}...")

        if not stock_data:
            raise DataCollectionError("No stock data could be fetched from any source")

        return stock_data

    def _fetch_single_stock(self, symbol: str) -> Optional[StockData]:
        """
        Fetch data for a single stock symbol.

        Args:
            symbol: Stock symbol to fetch

        Returns:
            StockData object if successful, None otherwise
        """
        try:
            ticker = yf.Ticker(symbol)

            # Get stock info
            try:
                info = ticker.info
            except Exception:
                info = {}

            # Get historical daily data (30 days for technical analysis)
            daily_data = ticker.history(
                period=self.config.data.historical_period,
                interval="1d",
                auto_adjust=False,
                prepost=False,
                repair=True
            )

            if daily_data.empty:
                logger.debug(f"No daily data available for {symbol}")
                return None

            # Get intraday data (current day)
            intraday_data = pd.DataFrame()  # Start with empty DataFrame
            try:
                intraday_data = ticker.history(
                    period="1d",
                    interval=self.config.data.intraday_interval,
                    auto_adjust=False,
                    prepost=False,
                    repair=True
                )
            except Exception as e:
                logger.debug(f"Could not fetch intraday data for {symbol}: {e}")

            # Create StockInfo
            stock_info = StockInfo(
                symbol=symbol,
                company_name=info.get('longName', symbol.replace('.JK', '')),
                sector=info.get('sector'),
                industry=info.get('industry'),
                market_cap=info.get('marketCap'),
                shares_outstanding=info.get('sharesOutstanding'),
                currency=info.get('currency', 'IDR'),
                exchange=info.get('exchange', 'IDX'),
                last_updated=datetime.now(tz=datetime.now().astimezone().tzinfo),
                data_source='yahoo'
            )

            # Create StockData object
            stock_data = StockData(
                symbol=symbol,
                info=stock_info,
                daily_data=daily_data,
                intraday_data=intraday_data,
                last_updated=datetime.now(tz=datetime.now().astimezone().tzinfo)
            )

            # Debug log the data before validation
            logger.debug(f"Data timestamps for {symbol}:")
            if not daily_data.empty:
                logger.debug(f"Daily data index timezone: {daily_data.index[0].tzinfo}")
            if not intraday_data.empty:
                logger.debug(f"Intraday data index timezone: {intraday_data.index[0].tzinfo}")
            logger.debug(f"Stock info last_updated timezone: {stock_info.last_updated.tzinfo}")
            logger.debug(f"Stock data for {symbol} before validation:")
            logger.debug(f"Daily data shape: {daily_data.shape}, columns: {daily_data.columns.tolist()}")
            logger.debug(f"Intraday data shape: {intraday_data.shape}, columns: {intraday_data.columns.tolist()}")
            logger.debug(f"Stock info: {stock_info}")

            # Log last few rows of price data
            if not daily_data.empty:
                logger.debug(f"Last 5 rows of daily data:\n{daily_data.tail(5)}")
            if not intraday_data.empty:
                logger.debug(f"Last 5 rows of intraday data:\n{intraday_data.tail(5)}")

            # Validate data quality
            if not self._validate_stock_data(stock_data):
                logger.debug(f"Data quality validation failed for {symbol}")
                return None

            return stock_data

        except Exception as e:
            logger.debug(f"Error fetching data for {symbol}: {e}")
            return None

    def _validate_stock_data(self, stock_data: StockData) -> bool:
        """
        Validate stock data quality and completeness.

        Args:
            stock_data: StockData object to validate

        Returns:
            True if data passes validation, False otherwise
        """
        try:
            symbol = stock_data.symbol

            # Check basic data availability
            if stock_data.daily_data.empty:
                logger.debug(f"No daily data for {symbol}")
                return False

            # Check required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in stock_data.daily_data.columns]
            if missing_columns:
                logger.debug(f"Missing columns for {symbol}: {missing_columns}")
                return False

            # Check data freshness (within config max age)
            last_date = stock_data.daily_data.index[-1]
            now = datetime.now(tz=last_date.tzinfo if last_date.tzinfo else None)
            data_age = (now - last_date.to_pydatetime()).total_seconds() / 3600
            logger.debug(f"Last date timezone: {last_date.tzinfo}, Current timezone: {now.tzinfo}")
            if data_age > self.config.data.max_data_age_hours:
                logger.debug(f"Data too old for {symbol}: {data_age:.1f} hours")
                return False

            # Check minimum data points
            if len(stock_data.daily_data) < self.config.data.min_data_points:
                logger.debug(f"Insufficient data points for {symbol}: {len(stock_data.daily_data)}")
                return False

            # Check for reasonable price values (IDX stocks typically > 50 IDR)
            current_price = stock_data.current_price
            if current_price <= 0 or current_price < 50:
                logger.debug(f"Unrealistic price for {symbol}: {current_price}")
                return False

            # Check for excessive missing values
            missing_pct = stock_data.daily_data.isnull().sum().sum() / (
                len(stock_data.daily_data) * len(stock_data.daily_data.columns)
            )
            if missing_pct > 0.2:  # More than 20% missing
                logger.debug(f"Too many missing values for {symbol}: {missing_pct:.1%}")
                return False

            # Check volume data quality
            zero_volume_days = (stock_data.daily_data['Volume'] == 0).sum()
            if zero_volume_days > len(stock_data.daily_data) * 0.3:  # More than 30% zero volume
                logger.debug(f"Too many zero volume days for {symbol}: {zero_volume_days}")
                return False

            # Check for price anomalies (e.g., extreme price jumps)
            price_changes = stock_data.daily_data['Close'].pct_change().abs()
            extreme_changes = (price_changes > 0.5).sum()  # More than 50% change
            if extreme_changes > 2:  # More than 2 extreme changes in recent data
                logger.debug(f"Too many extreme price changes for {symbol}: {extreme_changes}")
                return False

            return True

        except Exception as e:
            logger.debug(f"Validation error for {stock_data.symbol}: {e}")
            return False

    def get_validation_summary(self, stocks_data: Dict[str, StockData]) -> Dict[str, Any]:
        """
        Get validation summary for collected stock data.

        Args:
            stocks_data: Dictionary of stock data

        Returns:
            Validation summary statistics
        """
        if not stocks_data:
            return {
                'total_stocks': 0,
                'valid_stocks': 0,
                'validation_rate': 0.0,
                'quality_scores': {},
                'issues': []
            }

        total_stocks = len(stocks_data)
        quality_scores = {}
        issues = []

        for symbol, stock_data in stocks_data.items():
            quality_score = stock_data.get_data_quality_score()
            quality_scores[symbol] = quality_score

            # Identify specific issues
            if quality_score < 0.8:
                data_age = (datetime.now() - stock_data.daily_data.index[-1].to_pydatetime()).days
                if data_age > 2:
                    issues.append(f"{symbol}: Stale data ({data_age} days old)")

                missing_pct = stock_data.daily_data.isnull().sum().sum() / (
                    len(stock_data.daily_data) * len(stock_data.daily_data.columns)
                )
                if missing_pct > 0.1:
                    issues.append(f"{symbol}: High missing data ({missing_pct:.1%})")

        valid_stocks = sum(1 for score in quality_scores.values() if score >= 0.7)
        validation_rate = valid_stocks / total_stocks * 100 if total_stocks > 0 else 0

        return {
            'total_stocks': total_stocks,
            'valid_stocks': valid_stocks,
            'validation_rate': validation_rate,
            'quality_scores': quality_scores,
            'issues': issues,
            'average_quality': sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0,
            'failed_symbols': list(self._failed_symbols)
        }

    def clear_failed_symbols_cache(self) -> None:
        """Clear the cache of failed symbols to retry fetching."""
        logger.info(f"Clearing cache of {len(self._failed_symbols)} failed symbols")
        self._failed_symbols.clear()

    def get_market_status(self) -> Dict[str, Any]:
        """
        Get IDX market status information.

        Returns:
            Market status information
        """
        now = datetime.now()

        # IDX trading hours: 9:00 AM - 3:00 PM WIB (Monday to Friday)
        market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=0, second=0, microsecond=0)

        is_weekday = now.weekday() < 5  # Monday = 0, Friday = 4
        is_market_hours = market_open <= now <= market_close
        is_market_open = is_weekday and is_market_hours

        # Calculate time to next market open/close
        if is_market_open:
            time_to_close = (market_close - now).total_seconds() / 60  # minutes
            next_event = "market_close"
            minutes_to_event = time_to_close
        else:
            if now > market_close or not is_weekday:
                # Calculate next trading day
                days_ahead = 1
                if now.weekday() >= 4:  # Friday or later
                    days_ahead = 7 - now.weekday()  # Days until Monday

                next_open = (now + timedelta(days=days_ahead)).replace(
                    hour=9, minute=0, second=0, microsecond=0
                )
            else:
                # Same day, before market open
                next_open = market_open

            time_to_open = (next_open - now).total_seconds() / 60  # minutes
            next_event = "market_open"
            minutes_to_event = time_to_open

        return {
            'is_market_open': is_market_open,
            'is_weekday': is_weekday,
            'is_market_hours': is_market_hours,
            'current_time': now.strftime('%Y-%m-%d %H:%M:%S'),
            'market_open_time': market_open.strftime('%H:%M'),
            'market_close_time': market_close.strftime('%H:%M'),
            'next_event': next_event,
            'minutes_to_next_event': int(minutes_to_event),
            'timezone': 'WIB'
        }

    async def fetch_filtered_data(
        self,
        min_price: float = 1000,
        max_price: float = 50000,
        min_volume: int = 1000000,
        exclude_sectors: Optional[List[str]] = None
    ) -> Dict[str, StockData]:
        """
        Fetch data with pre-filtering based on criteria.

        Args:
            min_price: Minimum stock price filter
            max_price: Maximum stock price filter
            min_volume: Minimum volume filter
            exclude_sectors: Sectors to exclude

        Returns:
            Filtered stock data dictionary
        """
        logger.info(f"Fetching filtered data (price: {min_price}-{max_price}, volume: {min_volume:,})")

        # Get all data first
        all_data = await self.fetch_realtime_data()

        # Apply filters
        filtered_data = {}
        excluded_count = 0

        for symbol, stock_data in all_data.items():
            try:
                # Price filter
                current_price = stock_data.current_price
                if not (min_price <= current_price <= max_price):
                    excluded_count += 1
                    continue

                # Volume filter
                if stock_data.daily_volume < min_volume:
                    excluded_count += 1
                    continue

                # Sector filter
                if exclude_sectors and stock_data.info.sector in exclude_sectors:
                    excluded_count += 1
                    continue

                filtered_data[symbol] = stock_data

            except Exception as e:
                logger.debug(f"Error filtering {symbol}: {e}")
                excluded_count += 1

        logger.info(
            f"Filtering completed: {len(filtered_data)} stocks passed filters, "
            f"{excluded_count} excluded"
        )

        return filtered_data

# -*- coding: utf-8 -*-
"""
Data Validator for Indonesian Stock Screener
==========================================

Utility for validating stock data quality, detecting anomalies,
and ensuring data meets requirements for technical analysis.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from loguru import logger

from ..models.stock import StockData, StockPriceData


class ValidationRule:
    """Base class for data validation rules"""

    def __init__(self, name: str, severity: str = "warning"):
        """
        Initialize validation rule.

        Args:
            name: Name of the validation rule
            severity: Severity level (info, warning, error)
        """
        self.name = name
        self.severity = severity

    def validate(self, stock_data: StockData) -> Tuple[bool, str]:
        """
        Validate stock data against this rule.

        Args:
            stock_data: StockData to validate

        Returns:
            Tuple of (is_valid, message)
        """
        raise NotImplementedError("Subclasses must implement validate method")


class DataCompletenessRule(ValidationRule):
    """Validate data completeness and required columns"""

    def __init__(self, required_columns: List[str], min_data_points: int = 20):
        super().__init__("data_completeness", "error")
        self.required_columns = required_columns
        self.min_data_points = min_data_points

    def validate(self, stock_data: StockData) -> Tuple[bool, str]:
        """Validate data completeness"""
        # Check if data exists
        if stock_data.daily_data.empty:
            return False, "No daily data available"

        # Check required columns
        missing_columns = [col for col in self.required_columns
                          if col not in stock_data.daily_data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"

        # Check minimum data points
        if len(stock_data.daily_data) < self.min_data_points:
            return False, f"Insufficient data points: {len(stock_data.daily_data)} < {self.min_data_points}"

        return True, "Data completeness check passed"


class DataFreshnessRule(ValidationRule):
    """Validate data freshness"""

    def __init__(self, max_age_hours: int = 24):
        super().__init__("data_freshness", "warning")
        self.max_age_hours = max_age_hours

    def validate(self, stock_data: StockData) -> Tuple[bool, str]:
        """Validate data freshness"""
        if stock_data.daily_data.empty:
            return False, "No data to check freshness"

        last_date = stock_data.daily_data.index[-1]
        now = datetime.now(tz=last_date.tzinfo if last_date.tzinfo else None)

        logger.debug(f"Data freshness check for {stock_data.symbol}:")
        logger.debug(f"Last date: {last_date} (tz: {last_date.tzinfo})")
        logger.debug(f"Current time: {now} (tz: {now.tzinfo})")

        data_age = (now - last_date.to_pydatetime()).total_seconds() / 3600

        if data_age > self.max_age_hours:
            logger.debug(f"Data age {data_age:.1f} hours exceeds maximum {self.max_age_hours} hours")
            return False, f"Data is {data_age:.1f} hours old (max: {self.max_age_hours})"

        return True, f"Data is fresh ({data_age:.1f} hours old)"


class PriceValidityRule(ValidationRule):
    """Validate price data for reasonableness"""

    def __init__(self, min_price: float = 50, max_price: float = 1000000):
        super().__init__("price_validity", "error")
        self.min_price = min_price
        self.max_price = max_price

    def validate(self, stock_data: StockData) -> Tuple[bool, str]:
        """Validate price reasonableness"""
        try:
            current_price = stock_data.current_price

            # Check price range
            if not (self.min_price <= current_price <= self.max_price):
                return False, f"Price {current_price} outside valid range [{self.min_price}, {self.max_price}]"

            # Check for zero or negative prices
            zero_prices = (stock_data.daily_data[['Open', 'High', 'Low', 'Close']] <= 0).any().any()
            if zero_prices:
                return False, "Contains zero or negative prices"

            # Check OHLC consistency
            for idx, row in stock_data.daily_data.iterrows():
                if row['High'] < max(row['Open'], row['Close']) or row['Low'] > min(row['Open'], row['Close']):
                    return False, f"OHLC inconsistency on {idx.date()}"

            return True, "Price validity check passed"

        except Exception as e:
            return False, f"Error validating prices: {e}"


class VolumeValidityRule(ValidationRule):
    """Validate volume data"""

    def __init__(self, min_volume: int = 100000, max_zero_volume_pct: float = 0.1):
        super().__init__("volume_validity", "warning")
        self.min_volume = min_volume
        self.max_zero_volume_pct = max_zero_volume_pct

    def validate(self, stock_data: StockData) -> Tuple[bool, str]:
        """Validate volume data"""
        try:
            # Check current volume
            current_volume = stock_data.daily_volume
            if current_volume < self.min_volume:
                return False, f"Current volume {current_volume:,} below minimum {self.min_volume:,}"

            # Check for excessive zero volume days
            zero_volume_days = (stock_data.daily_data['Volume'] == 0).sum()
            zero_volume_pct = zero_volume_days / len(stock_data.daily_data)

            if zero_volume_pct > self.max_zero_volume_pct:
                return False, f"Too many zero volume days: {zero_volume_pct:.1%} > {self.max_zero_volume_pct:.1%}"

            # Check for negative volumes
            negative_volumes = (stock_data.daily_data['Volume'] < 0).sum()
            if negative_volumes > 0:
                return False, f"Contains {negative_volumes} negative volume entries"

            return True, "Volume validity check passed"

        except Exception as e:
            return False, f"Error validating volume: {e}"


class PriceAnomalyRule(ValidationRule):
    """Detect price anomalies and extreme movements"""

    def __init__(self, max_daily_change_pct: float = 35.0, max_anomaly_count: int = 2):
        super().__init__("price_anomaly", "warning")
        self.max_daily_change_pct = max_daily_change_pct
        self.max_anomaly_count = max_anomaly_count

    def validate(self, stock_data: StockData) -> Tuple[bool, str]:
        """Detect price anomalies"""
        try:
            # Calculate daily percentage changes
            daily_changes = stock_data.daily_data['Close'].pct_change().abs() * 100
            daily_changes = daily_changes.dropna()

            # Find extreme changes
            extreme_changes = daily_changes[daily_changes > self.max_daily_change_pct]

            if len(extreme_changes) > self.max_anomaly_count:
                return False, f"Too many extreme price changes: {len(extreme_changes)} > {self.max_anomaly_count}"

            # Check for consecutive limit up/down (35% IDX limit)
            consecutive_limits = 0
            for change in daily_changes:
                if abs(change) >= 34.0:  # Close to limit
                    consecutive_limits += 1
                    if consecutive_limits > 2:
                        return False, "Multiple consecutive limit moves detected"
                else:
                    consecutive_limits = 0

            return True, "Price anomaly check passed"

        except Exception as e:
            return False, f"Error detecting anomalies: {e}"


class GapDetectionRule(ValidationRule):
    """Detect and validate price gaps"""

    def __init__(self, max_gap_pct: float = 20.0):
        super().__init__("gap_detection", "info")
        self.max_gap_pct = max_gap_pct

    def validate(self, stock_data: StockData) -> Tuple[bool, str]:
        """Detect significant price gaps"""
        try:
            daily_data = stock_data.daily_data
            if len(daily_data) < 2:
                return True, "Insufficient data for gap detection"

            gaps = []
            for i in range(1, len(daily_data)):
                prev_close = daily_data.iloc[i-1]['Close']
                current_open = daily_data.iloc[i]['Open']

                gap_pct = abs((current_open - prev_close) / prev_close) * 100
                if gap_pct > self.max_gap_pct:
                    date = daily_data.index[i].date()
                    gaps.append(f"{date}: {gap_pct:.1f}%")

            if gaps:
                gap_info = ", ".join(gaps[:3])  # Show first 3 gaps
                return True, f"Significant gaps detected: {gap_info}"

            return True, "No significant gaps detected"

        except Exception as e:
            return False, f"Error detecting gaps: {e}"


class DataQualityRule(ValidationRule):
    """Overall data quality assessment"""

    def __init__(self, max_missing_pct: float = 0.05):
        super().__init__("data_quality", "warning")
        self.max_missing_pct = max_missing_pct

    def validate(self, stock_data: StockData) -> Tuple[bool, str]:
        """Assess overall data quality"""
        try:
            # Check missing values
            total_cells = len(stock_data.daily_data) * len(stock_data.daily_data.columns)
            missing_cells = stock_data.daily_data.isnull().sum().sum()
            missing_pct = missing_cells / total_cells

            if missing_pct > self.max_missing_pct:
                return False, f"High missing data: {missing_pct:.1%} > {self.max_missing_pct:.1%}"

            # Check data variance (avoid flat-line data)
            price_variance = stock_data.daily_data['Close'].var()
            if price_variance == 0:
                return False, "No price variance (flat-line data)"

            # Calculate quality score
            quality_score = stock_data.get_data_quality_score()
            if quality_score < 0.7:
                return False, f"Low quality score: {quality_score:.2f}"

            return True, f"Data quality acceptable (score: {quality_score:.2f})"

        except Exception as e:
            return False, f"Error assessing data quality: {e}"


class StockDataValidator:
    """Main stock data validator with configurable rules"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize stock data validator.

        Args:
            config: Validation configuration parameters
        """
        self.config = config or {}
        self.rules = self._initialize_rules()
        self.validation_cache = {}

    def _initialize_rules(self) -> List[ValidationRule]:
        """Initialize validation rules based on configuration"""
        rules = []

        # Data completeness (critical)
        rules.append(DataCompletenessRule(
            required_columns=['Open', 'High', 'Low', 'Close', 'Volume'],
            min_data_points=self.config.get('min_data_points', 20)
        ))

        # Data freshness
        rules.append(DataFreshnessRule(
            max_age_hours=self.config.get('max_data_age_hours', 24)
        ))

        # Price validity
        rules.append(PriceValidityRule(
            min_price=self.config.get('min_price', 50),
            max_price=self.config.get('max_price', 1000000)
        ))

        # Volume validity
        rules.append(VolumeValidityRule(
            min_volume=self.config.get('min_volume', 100000),
            max_zero_volume_pct=self.config.get('max_zero_volume_pct', 0.1)
        ))

        # Price anomaly detection
        rules.append(PriceAnomalyRule(
            max_daily_change_pct=self.config.get('max_daily_change_pct', 35.0),
            max_anomaly_count=self.config.get('max_anomaly_count', 2)
        ))

        # Gap detection
        rules.append(GapDetectionRule(
            max_gap_pct=self.config.get('max_gap_pct', 20.0)
        ))

        # Overall data quality
        rules.append(DataQualityRule(
            max_missing_pct=self.config.get('max_missing_pct', 0.05)
        ))

        return rules

    def validate_stock(self, stock_data: StockData, cache_results: bool = True) -> Dict[str, Any]:
        """
        Validate a single stock's data.

        Args:
            stock_data: StockData to validate
            cache_results: Whether to cache validation results

        Returns:
            Validation results dictionary
        """
        symbol = stock_data.symbol

        # Check cache if enabled
        if cache_results and symbol in self.validation_cache:
            cache_entry = self.validation_cache[symbol]
            if (datetime.now() - cache_entry['timestamp']).seconds < 3600:  # 1 hour cache
                return cache_entry['results']

        results = {
            'symbol': symbol,
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': [],
            'quality_score': 0.0,
            'validation_timestamp': datetime.now()
        }

        passed_rules = 0
        total_rules = len(self.rules)

        # Run all validation rules
        for rule in self.rules:
            try:
                is_valid, message = rule.validate(stock_data)

                if is_valid:
                    passed_rules += 1
                    if rule.severity == 'info':
                        results['info'].append(f"{rule.name}: {message}")
                else:
                    if rule.severity == 'error':
                        results['errors'].append(f"{rule.name}: {message}")
                        results['is_valid'] = False
                    elif rule.severity == 'warning':
                        results['warnings'].append(f"{rule.name}: {message}")
                    else:
                        results['info'].append(f"{rule.name}: {message}")

            except Exception as e:
                logger.error(f"Error running validation rule {rule.name} for {symbol}: {e}")
                results['errors'].append(f"{rule.name}: Validation failed - {e}")
                results['is_valid'] = False

        # Calculate quality score
        results['quality_score'] = passed_rules / total_rules if total_rules > 0 else 0.0

        # Cache results if enabled
        if cache_results:
            self.validation_cache[symbol] = {
                'timestamp': datetime.now(),
                'results': results
            }

        return results

    def validate_multiple_stocks(
        self,
        stocks_data: Dict[str, StockData],
        parallel: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate multiple stocks' data.

        Args:
            stocks_data: Dictionary of stock data to validate
            parallel: Whether to use parallel processing (future enhancement)

        Returns:
            Dictionary of validation results for each stock
        """
        logger.info(f"Validating {len(stocks_data)} stocks")

        validation_results = {}

        for symbol, stock_data in stocks_data.items():
            try:
                validation_results[symbol] = self.validate_stock(stock_data)
            except Exception as e:
                logger.error(f"Failed to validate {symbol}: {e}")
                validation_results[symbol] = {
                    'symbol': symbol,
                    'is_valid': False,
                    'errors': [f"Validation failed: {e}"],
                    'warnings': [],
                    'info': [],
                    'quality_score': 0.0,
                    'validation_timestamp': datetime.now()
                }

        return validation_results

    def get_validation_summary(
        self,
        validation_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate summary statistics from validation results.

        Args:
            validation_results: Results from validate_multiple_stocks

        Returns:
            Validation summary statistics
        """
        if not validation_results:
            return {
                'total_stocks': 0,
                'valid_stocks': 0,
                'validation_rate': 0.0,
                'average_quality_score': 0.0,
                'error_summary': {},
                'warning_summary': {}
            }

        total_stocks = len(validation_results)
        valid_stocks = sum(1 for r in validation_results.values() if r['is_valid'])
        quality_scores = [r['quality_score'] for r in validation_results.values()]

        # Count errors and warnings by type
        error_counts = {}
        warning_counts = {}

        for result in validation_results.values():
            for error in result['errors']:
                rule_name = error.split(':')[0]
                error_counts[rule_name] = error_counts.get(rule_name, 0) + 1

            for warning in result['warnings']:
                rule_name = warning.split(':')[0]
                warning_counts[rule_name] = warning_counts.get(rule_name, 0) + 1

        return {
            'total_stocks': total_stocks,
            'valid_stocks': valid_stocks,
            'validation_rate': (valid_stocks / total_stocks * 100) if total_stocks > 0 else 0.0,
            'average_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            'error_summary': error_counts,
            'warning_summary': warning_counts,
            'quality_distribution': {
                'high_quality': sum(1 for s in quality_scores if s >= 0.9),
                'medium_quality': sum(1 for s in quality_scores if 0.7 <= s < 0.9),
                'low_quality': sum(1 for s in quality_scores if s < 0.7)
            }
        }

    def filter_valid_stocks(
        self,
        stocks_data: Dict[str, StockData],
        min_quality_score: float = 0.7
    ) -> Dict[str, StockData]:
        """
        Filter stocks to only include those meeting validation criteria.

        Args:
            stocks_data: Dictionary of stock data
            min_quality_score: Minimum quality score threshold

        Returns:
            Filtered dictionary containing only valid stocks
        """
        validation_results = self.validate_multiple_stocks(stocks_data)

        valid_stocks = {}
        excluded_count = 0

        for symbol, stock_data in stocks_data.items():
            validation_result = validation_results.get(symbol, {})

            if (validation_result.get('is_valid', False) and
                validation_result.get('quality_score', 0) >= min_quality_score):
                valid_stocks[symbol] = stock_data
            else:
                excluded_count += 1

        logger.info(
            f"Filtered stocks: {len(valid_stocks)} valid, {excluded_count} excluded "
            f"(min quality score: {min_quality_score})"
        )

        return valid_stocks

    def clear_cache(self) -> None:
        """Clear validation cache"""
        self.validation_cache.clear()
        logger.info("Validation cache cleared")

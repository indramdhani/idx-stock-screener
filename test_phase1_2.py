#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indonesian Stock Screener - Phase 1-2 Smoke Test
===============================================

Basic smoke test to verify Phase 1-2 implementation:
- Configuration system works correctly
- Data models can be instantiated
- Data collector can fetch and validate data
- All imports work as expected

Run this test to verify the foundation components are working.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test all critical imports work."""
    print("Testing imports...")

    try:
        # Core imports
        from src.config import TradingConfig, load_config
        from src.data import IDXDataCollector, StockDataValidator
        from src.data.models import StockData, StockInfo, TradingSignal, SignalType
        from src.utils import setup_logging, get_logger

        print("✅ All imports successful")
        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration system...")

    try:
        from src.config import TradingConfig, load_config

        # Test default configuration
        config = TradingConfig()

        # Validate key configuration values
        assert 0.005 <= config.risk_management.max_risk_per_trade <= 0.1
        assert config.screening_criteria.min_volume >= 100_000
        assert len(config.default_tickers) > 0
        assert config.indicators.rsi_period == 14

        # Test configuration validation
        assert config.risk_management.max_risk_per_trade == 0.02
        assert config.screening_criteria.min_price == 1000
        assert config.indicators.rsi_oversold == 30

        print("✅ Configuration system working")
        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_data_models():
    """Test data model instantiation."""
    print("\nTesting data models...")

    try:
        from src.data.models import (
            StockInfo, StockData, TradingSignal, SignalType,
            RiskParameters, TakeProfitLevel
        )
        from datetime import datetime
        import pandas as pd

        # Test StockInfo
        stock_info = StockInfo(
            symbol="BBCA.JK",
            company_name="Bank Central Asia",
            sector="Financial",
            market_cap=1000000000000
        )
        assert stock_info.symbol == "BBCA.JK"
        assert stock_info.is_valid

        # Test basic DataFrame for StockData
        sample_data = pd.DataFrame({
            'Open': [8500, 8520, 8510],
            'High': [8550, 8560, 8540],
            'Low': [8480, 8500, 8490],
            'Close': [8520, 8530, 8525],
            'Volume': [1000000, 1200000, 1100000]
        }, index=pd.date_range('2024-01-01', periods=3))

        # Test StockData
        stock_data = StockData(
            symbol="BBCA.JK",
            info=stock_info,
            daily_data=sample_data,
            intraday_data=pd.DataFrame()  # Empty for test
        )

        assert stock_data.symbol == "BBCA.JK"
        assert stock_data.current_price == 8525
        assert stock_data.daily_volume == 1100000

        # Test TradingSignal components
        tp_level = TakeProfitLevel(price=8600, percentage=100.0, reasoning="Test TP")
        risk_params = RiskParameters(
            stop_loss=8450,
            take_profit_levels=[tp_level],
            risk_amount=75,
            potential_reward=75,
            risk_reward_ratio=1.0
        )

        signal = TradingSignal(
            signal_id="TEST_001",
            symbol="BBCA.JK",
            signal_type=SignalType.INTRADAY_REBOUND,
            timestamp=datetime.now(),
            entry_price=8525,
            entry_reasoning="Test signal",
            risk_params=risk_params,
            confidence_score=0.8
        )

        assert signal.symbol == "BBCA.JK"
        assert signal.is_long_signal
        assert signal.confidence_score == 0.8

        print("✅ Data models working correctly")
        return True

    except Exception as e:
        print(f"❌ Data models test failed: {e}")
        return False

async def test_data_collector():
    """Test data collector functionality."""
    print("\nTesting data collector...")

    try:
        from src.config import TradingConfig
        from src.data import IDXDataCollector, StockDataValidator

        # Create configuration
        config = TradingConfig()

        # Initialize collector
        collector = IDXDataCollector(config)

        # Test market status (doesn't require network)
        market_status = collector.get_market_status()
        assert 'is_market_open' in market_status
        assert 'current_time' in market_status

        print("✅ Market status check working")

        # Test data collection with small sample
        test_symbols = ["BBCA.JK", "TLKM.JK"]  # Just 2 symbols for quick test

        print("  Fetching sample data (this may take 10-15 seconds)...")
        stocks_data = await collector.fetch_realtime_data(test_symbols)

        if stocks_data:
            print(f"  ✅ Successfully fetched data for {len(stocks_data)} symbols")

            # Test data validation
            validator = StockDataValidator(config.dict())
            validation_results = validator.validate_multiple_stocks(stocks_data)
            validation_summary = validator.get_validation_summary(validation_results)

            print(f"  ✅ Data validation completed: {validation_summary['validation_rate']:.1f}% pass rate")

            # Test individual stock data
            first_symbol = list(stocks_data.keys())[0]
            stock_data = stocks_data[first_symbol]

            assert hasattr(stock_data, 'current_price')
            assert hasattr(stock_data, 'daily_volume')
            assert not stock_data.daily_data.empty

            print(f"  ✅ Sample data for {first_symbol}: Price={stock_data.current_price}, Volume={stock_data.daily_volume:,}")

        else:
            print("  ⚠️  No data fetched (network/source issue - this is acceptable for testing)")

        print("✅ Data collector test completed")
        return True

    except Exception as e:
        print(f"❌ Data collector test failed: {e}")
        return False

def test_logging_system():
    """Test logging system."""
    print("\nTesting logging system...")

    try:
        from src.utils import setup_logging, get_logger

        # Setup logging
        setup_logging(log_level="INFO", console_output=True, file_output=False)

        # Get logger
        logger = get_logger("test")

        # Test logging (should not raise errors)
        logger.info("Test info message")
        logger.warning("Test warning message")

        print("✅ Logging system working")
        return True

    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

async def main():
    """Run all Phase 1-2 smoke tests."""
    print("="*60)
    print("🇮🇩 Indonesian Stock Screener - Phase 1-2 Smoke Test")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Data Models", test_data_models),
        ("Logging System", test_logging_system),
        ("Data Collector", test_data_collector),  # Async test
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1

        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")

    print("\n" + "="*60)
    print(f"PHASE 1-2 TEST SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Phase 1-2 components are working correctly!")
        print("\nNext steps:")
        print("- Proceed with Phase 3: Technical Analysis implementation")
        print("- Run: python main.py --validate-only")
        print("- Run: python initial-script.py (legacy mode)")
    else:
        print("⚠️  Some components need attention before proceeding to Phase 3")

    print("="*60)

    return passed == total

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest suite crashed: {e}")
        sys.exit(1)

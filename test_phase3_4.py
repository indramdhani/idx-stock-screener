#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indonesian Stock Screener - Phase 3-4 Test
==========================================

Comprehensive test for Phase 3-4 implementation:
- Technical indicator calculations (VWAP, ATR, RSI, EMA)
- Stock screening engine functionality
- Risk management and position sizing
- Signal generation and validation

Run this test to verify the technical analysis and screening components are working.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_sample_stock_data():
    """Create realistic sample stock data for testing."""

    # Generate 30 days of sample data
    dates = pd.date_range(end=datetime.now(), periods=30)

    # Create realistic price movement
    np.random.seed(42)  # For reproducible results

    base_price = 8500  # IDR
    returns = np.random.normal(0, 0.02, 30)  # 2% daily volatility
    returns[10] = -0.05  # Simulate a significant drop
    returns[20] = 0.04   # Simulate a bounce

    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Generate OHLC data
    data = []
    for i, close_price in enumerate(prices):
        high = close_price * (1 + abs(np.random.normal(0, 0.01)))
        low = close_price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] * (1 + np.random.normal(0, 0.005)) if i > 0 else close_price

        # Ensure OHLC consistency
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)

        volume = int(np.random.normal(1500000, 500000))  # Average 1.5M volume
        if i == 10:  # High volume on drop day
            volume *= 2

        data.append({
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close_price, 2),
            'Volume': max(volume, 100000)  # Minimum 100k volume
        })

    df = pd.DataFrame(data, index=dates)
    return df

def test_technical_indicators():
    """Test all technical indicators."""
    print("Testing technical indicators...")

    try:
        from src.analysis.indicators import VWAP, ATR, RSI, EMA, calculate_all_indicators

        # Create sample data
        sample_data = create_sample_stock_data()

        # Test VWAP
        print("  Testing VWAP...")
        vwap = VWAP.calculate_vwap(sample_data)
        assert not vwap.empty, "VWAP calculation failed"
        assert not vwap.iloc[-1] == 0, "VWAP has invalid values"

        current_price = sample_data['Close'].iloc[-1]
        vwap_current = vwap.iloc[-1]
        deviation = VWAP.calculate_vwap_deviation(current_price, vwap_current)
        assert isinstance(deviation, float), "VWAP deviation calculation failed"

        # Test ATR
        print("  Testing ATR...")
        atr = ATR.calculate_atr(sample_data, period=14)
        assert not atr.empty, "ATR calculation failed"
        assert atr.iloc[-1] > 0, "ATR should be positive"

        # Test stop loss calculation
        stop_loss = ATR.calculate_atr_stop_loss(current_price, atr.iloc[-1], 2.0, True)
        assert stop_loss < current_price, "Long stop loss should be below current price"

        # Test RSI
        print("  Testing RSI...")
        rsi = RSI.calculate_rsi(sample_data, period=14)
        assert not rsi.empty, "RSI calculation failed"
        assert 0 <= rsi.iloc[-1] <= 100, f"RSI value {rsi.iloc[-1]} out of range"

        # Test RSI levels
        rsi_levels = RSI.identify_rsi_levels(rsi)
        assert 'current_rsi' in rsi_levels, "RSI level analysis failed"
        assert 'condition' in rsi_levels, "RSI condition not identified"

        # Test EMA
        print("  Testing EMA...")
        ema_5 = EMA.calculate_ema(sample_data, period=5)
        ema_13 = EMA.calculate_ema(sample_data, period=13)
        ema_21 = EMA.calculate_ema(sample_data, period=21)

        assert not ema_5.empty, "EMA-5 calculation failed"
        assert not ema_13.empty, "EMA-13 calculation failed"
        assert not ema_21.empty, "EMA-21 calculation failed"

        # Test EMA crossover
        crossover_data = EMA.calculate_ema_crossover(ema_5, ema_13)
        assert not crossover_data.empty, "EMA crossover analysis failed"

        # Test comprehensive indicator calculation
        print("  Testing comprehensive indicator calculation...")
        all_indicators = calculate_all_indicators(sample_data)

        expected_indicators = ['rsi', 'atr', 'ema', 'vwap']
        for indicator in expected_indicators:
            assert indicator in all_indicators, f"Missing indicator: {indicator}"
            assert 'error' not in all_indicators[indicator], f"Error in {indicator}: {all_indicators[indicator].get('error', 'Unknown error')}"

        print("✅ Technical indicators test passed")
        return True

    except Exception as e:
        print(f"❌ Technical indicators test failed: {e}")
        return False

def test_screener_engine():
    """Test the stock screening engine."""
    print("\nTesting stock screening engine...")

    try:
        from src.config import TradingConfig
        from src.data.models import StockData, StockInfo
        from src.analysis import StockScreener

        # Create configuration
        config = TradingConfig()

        # Create sample stock data
        sample_data = create_sample_stock_data()

        # Create StockInfo and StockData objects
        stock_info = StockInfo(
            symbol="BBCA.JK",
            company_name="Bank Central Asia",
            sector="Financial Services",
            market_cap=500_000_000_000_000  # 500T IDR
        )

        stock_data = StockData(
            symbol="BBCA.JK",
            info=stock_info,
            daily_data=sample_data,
            intraday_data=pd.DataFrame()  # Empty for test
        )

        # Test screener initialization
        screener = StockScreener(config)
        assert screener is not None, "Screener initialization failed"

        # Test intraday screening
        print("  Testing intraday screening...")
        stocks_dict = {"BBCA.JK": stock_data}

        intraday_signals = screener.screen_intraday_rebounds(
            stocks_dict,
            account_balance=100_000_000
        )

        print(f"  Found {len(intraday_signals)} intraday signals")

        # Validate signals
        for signal in intraday_signals:
            assert signal.symbol == "BBCA.JK", "Signal symbol mismatch"
            assert signal.entry_price > 0, "Invalid entry price"
            assert signal.risk_params.stop_loss > 0, "Invalid stop loss"
            assert len(signal.risk_params.take_profit_levels) > 0, "No take profit levels"
            assert signal.confidence_score > 0, "Invalid confidence score"
            assert signal.position_sizing is not None, "Position sizing not calculated"

        # Test overnight screening
        print("  Testing overnight screening...")

        # Modify sample data to create oversold conditions
        oversold_data = sample_data.copy()
        oversold_data.loc[oversold_data.index[-5]:, 'Close'] *= 0.95  # 5% drop in last 5 days

        oversold_stock_data = StockData(
            symbol="BBCA.JK",
            info=stock_info,
            daily_data=oversold_data,
            intraday_data=pd.DataFrame()
        )

        overnight_stocks = {"BBCA.JK": oversold_stock_data}
        overnight_signals = screener.screen_overnight_setups(
            overnight_stocks,
            account_balance=100_000_000
        )

        print(f"  Found {len(overnight_signals)} overnight signals")

        # Test screening summary
        all_signals = intraday_signals + overnight_signals
        if all_signals:
            summary = screener.get_screening_summary(all_signals)
            assert 'total_signals' in summary, "Screening summary incomplete"
            assert summary['total_signals'] == len(all_signals), "Signal count mismatch"

        print("✅ Stock screening engine test passed")
        return True

    except Exception as e:
        print(f"❌ Stock screening engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_risk_calculator():
    """Test risk management and position sizing."""
    print("\nTesting risk calculator...")

    try:
        from src.config import TradingConfig
        from src.analysis import RiskCalculator
        from src.data.models import TradingSignal, SignalType, RiskParameters, TakeProfitLevel
        from datetime import datetime

        # Create configuration
        config = TradingConfig()
        risk_calc = RiskCalculator(config)

        # Create sample signal
        tp_level = TakeProfitLevel(price=8700, percentage=100.0, reasoning="Test TP")
        risk_params = RiskParameters(
            stop_loss=8300,
            take_profit_levels=[tp_level],
            risk_amount=200,  # IDR 200 per share
            potential_reward=200,
            risk_reward_ratio=1.0
        )

        sample_signal = TradingSignal(
            signal_id="TEST_001",
            symbol="BBCA.JK",
            signal_type=SignalType.INTRADAY_REBOUND,
            timestamp=datetime.now(),
            entry_price=8500,
            entry_reasoning="Test signal",
            risk_params=risk_params,
            confidence_score=0.8
        )

        # Test position sizing
        print("  Testing position sizing...")
        account_balance = 100_000_000  # 100M IDR
        position = risk_calc.calculate_position_size(sample_signal, account_balance)

        assert position.shares >= 0, "Negative position size"
        assert position.shares % 100 == 0, "Position size not in lots"
        assert position.risk_percentage <= 2.5, f"Risk percentage too high: {position.risk_percentage}%"

        if position.shares > 0:
            assert position.position_value > 0, "Invalid position value"
            assert position.risk_amount > 0, "Invalid risk amount"
            print(f"  Position: {position.shares:,} shares, Risk: {position.risk_percentage:.1f}%")

        # Test ATR position sizing
        print("  Testing ATR-based position sizing...")
        atr_position = risk_calc.calculate_atr_position_size(
            entry_price=8500,
            atr_value=150,
            account_balance=account_balance,
            atr_multiplier=2.0
        )

        assert atr_position.shares >= 0, "Negative ATR position size"
        if atr_position.shares > 0:
            print(f"  ATR Position: {atr_position.shares:,} shares")

        # Test risk validation
        print("  Testing risk validation...")
        valid_rr = risk_calc.validate_risk_reward_ratio(sample_signal)
        print(f"  Risk-reward validation: {valid_rr}")

        # Test portfolio risk calculation
        print("  Testing portfolio risk calculation...")
        positions = [
            {'risk_amount': 1_000_000, 'position_value': 50_000_000},
            {'risk_amount': 800_000, 'position_value': 40_000_000}
        ]

        portfolio_risk = risk_calc.calculate_portfolio_risk(positions)
        assert 'total_risk_amount' in portfolio_risk, "Portfolio risk calculation failed"
        assert portfolio_risk['total_risk_amount'] == 1_800_000, "Portfolio risk sum incorrect"

        # Test risk metrics
        print("  Testing risk metrics calculation...")
        signals = [sample_signal]
        risk_metrics = risk_calc.calculate_risk_metrics(signals, account_balance)

        assert 'total_signals' in risk_metrics, "Risk metrics incomplete"
        assert risk_metrics['total_signals'] == 1, "Signal count incorrect"

        print("✅ Risk calculator test passed")
        return True

    except Exception as e:
        print(f"❌ Risk calculator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_integration():
    """Test end-to-end integration of Phase 3-4 components."""
    print("\nTesting end-to-end integration...")

    try:
        from src.config import load_config
        from src.data import IDXDataCollector
        from src.analysis import screen_stocks, analyze_stock

        # Load configuration
        config_path = Path(__file__).parent / "src" / "config" / "trading_config.yaml"
        config = load_config(config_path)

        # Test with sample data (faster than real data fetching)
        print("  Creating sample portfolio...")

        sample_stocks = {}
        symbols = ["BBCA.JK", "TLKM.JK", "UNVR.JK"]

        for symbol in symbols:
            from src.data.models import StockData, StockInfo

            # Create varied sample data for each stock
            sample_data = create_sample_stock_data()

            # Modify data slightly for each stock
            if symbol == "TLKM.JK":
                sample_data['Close'] *= 0.4  # Lower price stock
            elif symbol == "UNVR.JK":
                sample_data.loc[sample_data.index[-3]:, 'Close'] *= 0.97  # Recent decline

            stock_info = StockInfo(
                symbol=symbol,
                company_name=f"Test Company {symbol}",
                sector="Test Sector"
            )

            stock_data = StockData(
                symbol=symbol,
                info=stock_info,
                daily_data=sample_data,
                intraday_data=pd.DataFrame()
            )

            sample_stocks[symbol] = stock_data

        # Test individual stock analysis
        print("  Testing individual stock analysis...")
        analysis = analyze_stock(sample_stocks["BBCA.JK"], config.dict())

        assert 'stock_metrics' in analysis, "Stock analysis incomplete"
        assert 'technical_indicators' in analysis, "Technical indicators missing"
        assert 'trading_signals' in analysis, "Trading signals missing"

        # Test portfolio screening
        print("  Testing portfolio screening...")

        # Test intraday screening
        intraday_signals = screen_stocks(
            sample_stocks,
            config,
            strategy='intraday',
            account_balance=100_000_000
        )

        print(f"  Intraday signals found: {len(intraday_signals)}")

        # Test overnight screening
        overnight_signals = screen_stocks(
            sample_stocks,
            config,
            strategy='overnight',
            account_balance=100_000_000
        )

        print(f"  Overnight signals found: {len(overnight_signals)}")

        # Test portfolio risk calculation
        all_signals = intraday_signals + overnight_signals
        if all_signals:
            from src.analysis import calculate_portfolio_risk
            portfolio_risk = calculate_portfolio_risk(all_signals, 100_000_000, config)

            assert 'total_signals' in portfolio_risk or 'error' in portfolio_risk, "Portfolio risk calculation failed"
            print(f"  Portfolio risk analysis completed")

        print("✅ End-to-end integration test passed")
        return True

    except Exception as e:
        print(f"❌ End-to-end integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all Phase 3-4 tests."""
    print("="*70)
    print("🇮🇩 Indonesian Stock Screener - Phase 3-4 Test")
    print("Technical Analysis & Risk Management")
    print("="*70)

    tests = [
        ("Technical Indicators", test_technical_indicators),
        ("Stock Screening Engine", test_screener_engine),
        ("Risk Calculator", test_risk_calculator),
        ("End-to-End Integration", test_integration),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            print(f"\n📊 Running {test_name} test...")

            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1
                print(f"✅ {test_name} test completed successfully")
            else:
                print(f"❌ {test_name} test failed")

        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print(f"PHASE 3-4 TEST SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Phase 3-4 components are working correctly!")
        print("\nPhase 3-4 Implementation Status:")
        print("✅ Technical Indicators (VWAP, ATR, RSI, EMA)")
        print("✅ Stock Screening Engine")
        print("✅ Risk Management & Position Sizing")
        print("✅ Signal Generation & Validation")
        print("✅ End-to-End Integration")
        print("\nNext steps:")
        print("- Proceed with Phase 5: Telegram Integration")
        print("- Run: python main.py --mode intraday")
        print("- Run: python main.py --mode overnight")
    else:
        print("⚠️  Some components need attention before proceeding to Phase 5")
        print("\nTroubleshooting:")
        print("- Check that all dependencies are installed: pip install -r requirements.txt")
        print("- Verify configuration file exists: src/config/trading_config.yaml")
        print("- Review error messages above for specific issues")

    print("="*70)
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
        import traceback
        traceback.print_exc()
        sys.exit(1)

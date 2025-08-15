# -*- coding: utf-8 -*-
"""
Indonesian Stock Screener - Main Application Entry Point
=======================================================

Main application entry point that orchestrates the Indonesian stock screening
system. Supports both direct execution and scheduled automation modes with
Telegram integration and enhanced workflow management.

Usage:
    python main.py                    # Run interactive screening
    python main.py --mode intraday    # Run intraday screening only
    python main.py --mode overnight   # Run overnight screening only
    python main.py --mode both        # Run both strategies
    python main.py --config custom.yaml  # Use custom configuration
    python main.py --validate-only    # Only validate data quality
    python main.py --telegram-test    # Test Telegram integration
    python main.py --scheduled        # Start scheduled automation
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import load_config, TradingConfig
from src.data import IDXDataCollector, StockDataValidator
from src.notifications import notify_signal, notify_market_update
from src.scheduler import run_screening_workflow, start_automated_screening


class MainApplication:
    """Main application orchestrator for the Indonesian Stock Screener."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the main application.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or Path(__file__).parent / "src" / "config" / "trading_config.yaml"
        self.config = load_config(self.config_path)

        # Initialize core components
        self.data_collector = IDXDataCollector(self.config)
        self.data_validator = StockDataValidator(self.config.dict())

        # Create logs directory
        self.logs_dir = Path(__file__).parent / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        logger.info(f"Application initialized with config: {self.config_path}")

    async def run_data_validation(self) -> bool:
        """
        Run comprehensive data validation check.

        Returns:
            True if validation passes, False otherwise
        """
        logger.info("Starting comprehensive data validation...")

        try:
            # Fetch sample data for validation
            sample_symbols = self.config.default_tickers[:20]  # Test with first 20 symbols
            stocks_data = await self.data_collector.fetch_realtime_data(sample_symbols)

            if not stocks_data:
                logger.error("No data available for validation")
                return False

            # Run validation
            validation_results = self.data_validator.validate_multiple_stocks(stocks_data)
            validation_summary = self.data_validator.get_validation_summary(validation_results)

            # Log validation results
            logger.info(f"Validation completed:")
            logger.info(f"  Total stocks: {validation_summary['total_stocks']}")
            logger.info(f"  Valid stocks: {validation_summary['valid_stocks']}")
            logger.info(f"  Validation rate: {validation_summary['validation_rate']:.1f}%")
            logger.info(f"  Average quality score: {validation_summary['average_quality_score']:.2f}")

            if validation_summary['error_summary']:
                logger.warning(f"Common errors: {validation_summary['error_summary']}")

            if validation_summary['warning_summary']:
                logger.warning(f"Common warnings: {validation_summary['warning_summary']}")

            # Save validation report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.logs_dir / f"validation_report_{timestamp}.json"

            import json
            with open(report_file, 'w') as f:
                json.dump(validation_summary, f, indent=2, default=str)

            logger.info(f"Validation report saved to {report_file}")

            return validation_summary['validation_rate'] >= 70.0  # 70% minimum pass rate

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False

    async def run_intraday_screening(self) -> Optional[dict]:
        """
        Run intraday screening workflow.

        Returns:
            Screening results or None if failed
        """
        logger.info("Starting intraday screening workflow...")

        try:
            # Check market status
            market_status = self.data_collector.get_market_status()
            logger.info(f"Market status: {'OPEN' if market_status['is_market_open'] else 'CLOSED'}")

            if not market_status['is_market_open']:
                logger.warning("Market is currently closed - results may be stale")

            # Fetch filtered data based on screening criteria
            stocks_data = await self.data_collector.fetch_filtered_data(
                min_price=self.config.screening_criteria.min_price,
                max_price=self.config.screening_criteria.max_price,
                min_volume=self.config.screening_criteria.min_volume,
                exclude_sectors=self.config.screening_criteria.exclude_sectors
            )

            if not stocks_data:
                logger.error("No stock data available for screening")
                return None

            logger.info(f"Screening {len(stocks_data)} stocks for intraday opportunities")

            try:
                # Import screening logic
                from src.analysis.screener import StockScreener
                screener = StockScreener(self.config)

                # Run intraday screening
                signals = screener.screen_intraday_rebounds(stocks_data, self.config.default_capital_idr)

                logger.info(f"Found {len(signals)} intraday signals")

                # Send high-confidence signals via notifications
                high_confidence_signals = [s for s in signals if s.confidence_score >= 0.8]
                for signal in high_confidence_signals[:5]:  # Max 5 signals
                    try:
                        await notify_signal(signal, self.config, priority="normal")
                    except Exception as e:
                        logger.debug(f"Failed to send signal notification: {e}")

                # Display results in a user-friendly format
                if signals:
                    print("\n" + "="*60)
                    print("🚀 INTRADAY REBOUND OPPORTUNITIES")
                    print("="*60)

                    for i, signal in enumerate(signals[:10], 1):  # Show top 10
                        print(f"\n{i}. {signal.symbol}")
                        print(f"   Entry: IDR {signal.entry_price:,.0f}")
                        print(f"   Stop Loss: IDR {signal.risk_params.stop_loss:,.0f}")
                        print(f"   Take Profit: IDR {signal.risk_params.primary_take_profit:,.0f}")
                        print(f"   Risk/Reward: {signal.risk_params.risk_reward_ratio:.1f}:1")
                        print(f"   Confidence: {signal.confidence_score:.1%}")
                        if signal.position_sizing:
                            print(f"   Position: {signal.position_sizing.shares:,} shares ({signal.position_sizing.lots} lots)")
                        print(f"   Reasoning: {signal.entry_reasoning}")
                else:
                    print("\n🔍 No intraday rebound opportunities found at this time")

                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results = {
                    'timestamp': timestamp,
                    'market_status': market_status,
                    'total_stocks_screened': len(stocks_data),
                    'signals_found': len(signals),
                    'signals': [signal.to_dict() for signal in signals]
                }

                # Save to JSON file
                results_file = self.logs_dir / f"intraday_screening_{timestamp}.json"
                import json
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)

                # Also save simplified CSV for Excel compatibility
                if signals:
                    import pandas as pd
                    signal_data = []
                    for signal in signals:
                        signal_data.append({
                            'Symbol': signal.symbol,
                            'Entry_Price': signal.entry_price,
                            'Stop_Loss': signal.risk_params.stop_loss,
                            'Take_Profit': signal.risk_params.primary_take_profit,
                            'Risk_Reward': signal.risk_params.risk_reward_ratio,
                            'Confidence': signal.confidence_score,
                            'Shares': signal.position_sizing.shares if signal.position_sizing else 0,
                            'Position_Value': signal.position_sizing.position_value if signal.position_sizing else 0,
                            'Reasoning': signal.entry_reasoning
                        })

                    df = pd.DataFrame(signal_data)
                    csv_file = self.logs_dir / f"intraday_signals_{timestamp}.csv"
                    df.to_csv(csv_file, index=False)
                    logger.info(f"CSV results saved to {csv_file}")

                logger.info(f"Intraday screening results saved to {results_file}")
                return results

            except Exception as e:
                logger.error(f"Intraday screening failed: {e}")
                return None
        except Exception as e:
            logger.error(f"Failed to check market status: {e}")
            return None

    async def run_overnight_screening(self) -> Optional[dict]:
        """
        Run overnight screening workflow.

        Returns:
            Screening results or None if failed
        """
        logger.info("Starting overnight screening workflow...")

        try:
            # Similar to intraday but with different criteria
            stocks_data = await self.data_collector.fetch_filtered_data(
                min_price=self.config.screening_criteria.min_price,
                max_price=self.config.screening_criteria.max_price,
                min_volume=self.config.screening_criteria.min_volume,
                exclude_sectors=self.config.screening_criteria.exclude_sectors
            )

            if not stocks_data:
                logger.error("No stock data available for screening")
                return None

            logger.info(f"Screening {len(stocks_data)} stocks for overnight opportunities")

            # Import screening logic
            from src.analysis.screener import StockScreener
            screener = StockScreener(self.config)

            # Run overnight screening
            signals = screener.screen_overnight_setups(stocks_data, self.config.default_capital_idr)

            logger.info(f"Found {len(signals)} overnight signals")

            # Send overnight signals via notifications (they're rarer, so send all)
            for signal in signals[:3]:  # Max 3 overnight signals
                try:
                    await notify_signal(signal, self.config, priority="normal")
                except Exception as e:
                    logger.debug(f"Failed to send signal notification: {e}")

            # Send summary if signals found
            if signals:
                try:
                    summary_text = f"🌙 Found {len(signals)} overnight opportunities for tomorrow's session."
                    await notify_market_update(
                        "Overnight Screening Complete",
                        summary_text,
                        self.config,
                        priority="normal"
                    )
                except Exception as e:
                    logger.debug(f"Failed to send summary notification: {e}")

            # Display results in a user-friendly format
            if signals:
                print("\n" + "="*60)
                print("🌙 OVERNIGHT SETUP OPPORTUNITIES")
                print("="*60)

                for i, signal in enumerate(signals[:10], 1):  # Show top 10
                    print(f"\n{i}. {signal.symbol}")
                    print(f"   Entry: IDR {signal.entry_price:,.0f}")
                    print(f"   Stop Loss: IDR {signal.risk_params.stop_loss:,.0f}")
                    print(f"   Take Profit: IDR {signal.risk_params.primary_take_profit:,.0f}")
                    print(f"   Risk/Reward: {signal.risk_params.risk_reward_ratio:.1f}:1")
                    print(f"   Confidence: {signal.confidence_score:.1%}")
                    if signal.position_sizing:
                        print(f"   Position: {signal.position_sizing.shares:,} shares ({signal.position_sizing.lots} lots)")
                    print(f"   Reasoning: {signal.entry_reasoning}")
            else:
                print("\n🔍 No overnight setup opportunities found at this time")

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {
                'timestamp': timestamp,
                'total_stocks_screened': len(stocks_data),
                'signals_found': len(signals),
                'signals': [signal.to_dict() for signal in signals]
            }

            # Save to JSON file
            results_file = self.logs_dir / f"overnight_screening_{timestamp}.json"
            import json
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            # Also save simplified CSV for Excel compatibility
            if signals:
                import pandas as pd
                signal_data = []
                for signal in signals:
                    signal_data.append({
                        'Symbol': signal.symbol,
                        'Entry_Price': signal.entry_price,
                        'Stop_Loss': signal.risk_params.stop_loss,
                        'Take_Profit': signal.risk_params.primary_take_profit,
                        'Risk_Reward': signal.risk_params.risk_reward_ratio,
                        'Confidence': signal.confidence_score,
                        'Shares': signal.position_sizing.shares if signal.position_sizing else 0,
                        'Position_Value': signal.position_sizing.position_value if signal.position_sizing else 0,
                        'Reasoning': signal.entry_reasoning
                    })

                df = pd.DataFrame(signal_data)
                csv_file = self.logs_dir / f"overnight_signals_{timestamp}.csv"
                df.to_csv(csv_file, index=False)
                logger.info(f"CSV results saved to {csv_file}")

            logger.info(f"Overnight screening results saved to {results_file}")
            return results

        except Exception as e:
            logger.error(f"Overnight screening failed: {e}")
            return None

    async def run_interactive_mode(self):
        """Run interactive screening mode."""
        logger.info("Starting interactive screening mode...")

        try:
            print("\n" + "="*70)
            print("🇮🇩 Indonesian Stock Screener - Interactive Mode")
            print("="*70)

            # Display current configuration
            print(f"Configuration: {self.config_path.name}")
            print(f"Default tickers: {len(self.config.default_tickers)} stocks")
            print(f"VWAP filter: {'✓' if self.config.enable_vwap_filter else '✗'}")
            print(f"ATR TP/SL: {'✓' if self.config.enable_atr_tp_sl else '✗'}")
            print(f"Telegram notifications: {'✓' if self.config.enable_telegram_notifications else '✗'}")

            while True:
                print("\n" + "-"*50)
                print("Select an option:")
                print("1. Run intraday screening")
                print("2. Run overnight screening")
                print("3. Run both screenings")
                print("4. Validate data quality")
                print("5. Check market status")
                print("6. View configuration")
                print("7. Test Telegram integration")
                print("8. Start automated screening")
                print("0. Exit")
                print("-"*50)

                try:
                    choice = input("Enter your choice (0-8): ").strip()

                    if choice == '0':
                        print("Exiting interactive mode...")
                        break
                    elif choice == '1':
                        await self.run_intraday_screening()
                    elif choice == '2':
                        await self.run_overnight_screening()
                    elif choice == '3':
                        await self.run_intraday_screening()
                        await self.run_overnight_screening()
                    elif choice == '4':
                        validation_passed = await self.run_data_validation()
                        if validation_passed:
                            print("✅ Data validation passed")
                        else:
                            print("❌ Data validation failed")
                    elif choice == '5':
                        market_status = self.data_collector.get_market_status()
                        print(f"Market Status: {'🟢 OPEN' if market_status['is_market_open'] else '🔴 CLOSED'}")
                        print(f"Current time: {market_status['current_time']}")
                        if not market_status['is_market_open']:
                            print(f"Next {market_status['next_event']} in {market_status['minutes_to_next_event']} minutes")
                    elif choice == '6':
                        print(f"\nCurrent Configuration ({self.config_path}):")
                        print(f"  Risk per trade: {self.config.risk_management.max_risk_per_trade:.1%}")
                        print(f"  Min volume: {self.config.screening_criteria.min_volume:,}")
                        print(f"  Price range: {self.config.screening_criteria.min_price:,} - {self.config.screening_criteria.max_price:,} IDR")
                        print(f"  RSI thresholds: {self.config.indicators.rsi_oversold} - {self.config.indicators.rsi_overbought}")
                    elif choice == '7':
                        await self.test_telegram_integration()
                    elif choice == '8':
                        await self.start_automated_mode()
                    else:
                        print("Invalid choice. Please try again.")

                except KeyboardInterrupt:
                    print("\nExiting interactive mode...")
                    break
                except Exception as e:
                    logger.error(f"Interactive mode error: {e}")
                    print(f"Error: {e}")

        except Exception as e:
            logger.error(f"Interactive mode failed: {e}")

    async def test_telegram_integration(self):
        """Test Telegram integration."""
        print("\n🤖 Testing Telegram Integration...")

        if not self.config.notifications.telegram_bot_token:
            print("❌ Telegram bot token not configured")
            return

        try:
            test_message = f"🧪 Test message from Indonesian Stock Screener\n\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S WIB')}\nStatus: System operational ✅"

            result = await notify_market_update(
                "System Test",
                test_message,
                self.config,
                priority="low"
            )

            if result.get('telegram', {}).get('success'):
                print("✅ Telegram integration working correctly")
            else:
                print("❌ Telegram integration failed")
                error = result.get('telegram', {}).get('error')
                if error:
                    print(f"   Error: {error}")

        except Exception as e:
            print(f"❌ Telegram test failed: {e}")

    async def start_automated_mode(self):
        """Start automated screening mode."""
        print("\n🤖 Starting Automated Screening Mode...")
        print("This will run continuous screening based on your schedule configuration.")
        print("Press Ctrl+C to stop.")

        try:
            # Send startup notification
            await notify_market_update(
                "Automated Screening Started",
                "Indonesian Stock Screener is now running in automated mode.",
                self.config,
                priority="normal"
            )

            # Start orchestrator
            orchestrator = await start_automated_screening(self.config)

            print("✅ Automated screening started successfully")
            print("📅 Schedule:")
            for job in orchestrator.scheduler.get_jobs():
                print(f"   • {job.name}: {job.next_run_time}")

            # Keep running until interrupted
            try:
                while orchestrator.is_running:
                    await asyncio.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                print("\n⏹️  Stopping automated screening...")
                await orchestrator.shutdown()
                print("✅ Automated screening stopped")

        except Exception as e:
            logger.error(f"Automated mode failed: {e}")
            print(f"❌ Failed to start automated mode: {e}")


def setup_logging(log_level: str = "INFO"):
    """Setup application logging."""
    logger.remove()

    # Console logging
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )

    # File logging
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    logger.add(
        logs_dir / "screener_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Indonesian Stock Screener - Automated stock screening system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Interactive mode
  python main.py --mode intraday         # Run intraday screening
  python main.py --mode overnight        # Run overnight screening
  python main.py --validate-only         # Data validation only
  python main.py --config custom.yaml   # Use custom configuration
  python main.py --log-level DEBUG      # Enable debug logging
        """
    )

    parser.add_argument(
        '--mode',
        choices=['interactive', 'intraday', 'overnight', 'both'],
        default='interactive',
        help='Screening mode (default: interactive)'
    )

    parser.add_argument(
        '--config',
        type=Path,
        help='Path to configuration file'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Run data validation only'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--telegram-test',
        action='store_true',
        help='Test Telegram integration'
    )

    parser.add_argument(
        '--scheduled',
        action='store_true',
        help='Start scheduled automation mode'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Indonesian Stock Screener v1.0.0'
    )

    return parser.parse_args()


async def main():
    """Main application entry point."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)

    try:
        # Initialize application
        app = MainApplication(args.config)

        logger.info("="*70)
        logger.info("🇮🇩 Indonesian Stock Screener v1.0.0")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Config: {app.config_path}")
        logger.info("="*70)

        # Run based on mode
        if args.validate_only:
            validation_passed = await app.run_data_validation()
            sys.exit(0 if validation_passed else 1)

        elif args.telegram_test:
            await app.test_telegram_integration()

        elif args.scheduled:
            await app.start_automated_mode()

        elif args.mode == 'interactive':
            await app.run_interactive_mode()

        elif args.mode == 'intraday':
            results = await app.run_intraday_screening()
            if results is None:
                sys.exit(1)

        elif args.mode == 'overnight':
            results = await app.run_overnight_screening()
            if results is None:
                sys.exit(1)

        elif args.mode == 'both':
            # Use enhanced workflow for both mode
            try:
                workflow_results = await run_screening_workflow('both', app.config)
                if 'error' in workflow_results:
                    logger.error(f"Workflow failed: {workflow_results['error']}")
                    sys.exit(1)
            except Exception as e:
                logger.error(f"Enhanced workflow failed, falling back to sequential: {e}")
                intraday_results = await app.run_intraday_screening()
                overnight_results = await app.run_overnight_screening()
                if intraday_results is None and overnight_results is None:
                    sys.exit(1)

        logger.info("Application completed successfully")

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Application failed: {e}")
        if args.log_level == 'DEBUG':
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

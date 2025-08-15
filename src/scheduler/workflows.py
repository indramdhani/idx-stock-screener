# -*- coding: utf-8 -*-
"""
Workflow Orchestrator for Indonesian Stock Screener
===================================================

Orchestrates different screening workflows including intraday screening,
overnight analysis, and market monitoring. Handles scheduling, error recovery,
and result management for automated operation.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback

from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

from ..config.settings import TradingConfig
from ..data import IDXDataCollector, StockDataValidator
from ..analysis import StockScreener, calculate_portfolio_risk
from ..notifications import notify_signal, notify_market_update
from ..utils.logger import LogContext


class WorkflowOrchestrator:
    """
    Main workflow orchestrator for the Indonesian Stock Screener.

    Manages scheduled execution of different screening strategies,
    handles errors and recovery, and coordinates between different
    system components.
    """

    def __init__(self, config: TradingConfig):
        """
        Initialize workflow orchestrator.

        Args:
            config: Trading configuration with scheduling parameters
        """
        self.config = config
        self.scheduler = AsyncIOScheduler(timezone='Asia/Jakarta')
        self.is_running = False

        # Core components
        self.data_collector = IDXDataCollector(config)
        self.data_validator = StockDataValidator(config.dict())
        self.screener = StockScreener(config)

        # Workflow state
        self.last_intraday_run = None
        self.last_overnight_run = None
        self.daily_signal_count = 0
        self.workflow_stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'signals_generated': 0,
            'last_reset_date': datetime.now().date()
        }

        # Results storage
        self.logs_dir = Path(__file__).parent.parent.parent / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        # Setup scheduler event listeners
        self.scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)

        logger.info("Workflow orchestrator initialized")

    async def initialize(self) -> bool:
        """
        Initialize the workflow orchestrator.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Setup scheduled jobs
            self._setup_scheduled_jobs()

            # Start scheduler
            self.scheduler.start()
            self.is_running = True

            logger.info("Workflow orchestrator started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize workflow orchestrator: {e}")
            return False

    def _setup_scheduled_jobs(self):
        """Setup scheduled jobs based on configuration."""

        # Intraday screening job (every 30 minutes during market hours)
        self.scheduler.add_job(
            func=self.run_intraday_screening,
            trigger=CronTrigger.from_crontab(
                self.config.scheduling.intraday_screening_cron
            ),
            id='intraday_screening',
            name='Intraday Stock Screening',
            max_instances=1,
            replace_existing=True
        )

        # Overnight screening job (daily at market close)
        self.scheduler.add_job(
            func=self.run_overnight_screening,
            trigger=CronTrigger.from_crontab(
                self.config.scheduling.overnight_screening_cron
            ),
            id='overnight_screening',
            name='Overnight Setup Screening',
            max_instances=1,
            replace_existing=True
        )

        # Daily risk review job
        self.scheduler.add_job(
            func=self.run_risk_review,
            trigger=CronTrigger.from_crontab(
                self.config.scheduling.risk_review_cron
            ),
            id='risk_review',
            name='Daily Risk Review',
            max_instances=1,
            replace_existing=True
        )

        # Market status check job (before market open)
        self.scheduler.add_job(
            func=self.run_market_status_check,
            trigger=CronTrigger(hour=8, minute=0),  # 8 AM WIB
            id='market_status_check',
            name='Market Status Check',
            max_instances=1,
            replace_existing=True
        )

        # Daily statistics reset
        self.scheduler.add_job(
            func=self.reset_daily_stats,
            trigger=CronTrigger(hour=0, minute=0),  # Midnight
            id='daily_stats_reset',
            name='Daily Statistics Reset',
            max_instances=1,
            replace_existing=True
        )

    async def run_intraday_screening(self) -> Dict[str, Any]:
        """
        Execute intraday screening workflow.

        Returns:
            Dictionary with workflow results
        """
        workflow_id = f"intraday_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with LogContext("intraday_screening", workflow_id=workflow_id):
            start_time = datetime.now()
            result = {
                'workflow_id': workflow_id,
                'start_time': start_time.isoformat(),
                'success': False,
                'signals': [],
                'stats': {},
                'errors': []
            }

            try:
                logger.info("Starting intraday screening workflow")

                # Check if market is open
                market_status = self.data_collector.get_market_status()
                if not market_status['is_market_open']:
                    logger.info("Market is closed, skipping intraday screening")
                    result['success'] = True
                    result['skip_reason'] = 'market_closed'
                    return result

                # Check daily signal limit
                if not self._check_daily_limit():
                    logger.warning("Daily signal limit reached, skipping screening")
                    result['skip_reason'] = 'daily_limit_reached'
                    return result

                # Collect stock data
                stocks_data = await self.data_collector.fetch_filtered_data(
                    min_price=self.config.screening_criteria.min_price,
                    max_price=self.config.screening_criteria.max_price,
                    min_volume=self.config.screening_criteria.min_volume,
                    exclude_sectors=self.config.screening_criteria.exclude_sectors
                )

                if not stocks_data:
                    logger.warning("No stock data available for screening")
                    result['success'] = True
                    result['skip_reason'] = 'no_data'
                    return result

                logger.info(f"Collected data for {len(stocks_data)} stocks")

                # Validate data quality
                validation_summary = self.data_validator.get_validation_summary(
                    self.data_validator.validate_multiple_stocks(stocks_data)
                )

                if validation_summary['validation_rate'] < 50:
                    logger.error("Data quality too poor for screening")
                    result['errors'].append("Poor data quality")
                    return result

                # Run screening
                signals = self.screener.screen_intraday_rebounds(
                    stocks_data,
                    self.config.default_capital_idr
                )

                # Update tracking
                self.last_intraday_run = datetime.now()
                self.daily_signal_count += len(signals)
                self.workflow_stats['signals_generated'] += len(signals)

                # Send notifications for high-confidence signals
                notification_results = []
                high_confidence_signals = [s for s in signals if s.confidence_score >= 0.8]

                for signal in high_confidence_signals[:5]:  # Max 5 signals
                    try:
                        notify_result = await notify_signal(
                            signal,
                            self.config,
                            priority="normal"
                        )
                        notification_results.append(notify_result)
                    except Exception as e:
                        logger.error(f"Failed to send signal notification: {e}")

                # Save results
                await self._save_workflow_results(workflow_id, 'intraday', signals)

                # Prepare result
                result.update({
                    'success': True,
                    'signals': [s.to_dict() for s in signals],
                    'stats': {
                        'total_screened': len(stocks_data),
                        'signals_found': len(signals),
                        'high_confidence_signals': len(high_confidence_signals),
                        'data_quality_rate': validation_summary['validation_rate'],
                        'execution_time': (datetime.now() - start_time).total_seconds()
                    },
                    'notifications_sent': len(notification_results)
                })

                logger.info(f"Intraday screening completed: {len(signals)} signals found")
                return result

            except Exception as e:
                logger.error(f"Intraday screening workflow failed: {e}")
                result['errors'].append(str(e))
                result['traceback'] = traceback.format_exc()

                # Send error notification
                try:
                    await notify_market_update(
                        "Screening Error",
                        f"Intraday screening failed: {str(e)}",
                        self.config,
                        priority="high"
                    )
                except Exception:
                    pass  # Don't fail if notification fails

                return result

    async def run_overnight_screening(self) -> Dict[str, Any]:
        """
        Execute overnight screening workflow.

        Returns:
            Dictionary with workflow results
        """
        workflow_id = f"overnight_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with LogContext("overnight_screening", workflow_id=workflow_id):
            start_time = datetime.now()
            result = {
                'workflow_id': workflow_id,
                'start_time': start_time.isoformat(),
                'success': False,
                'signals': [],
                'stats': {},
                'errors': []
            }

            try:
                logger.info("Starting overnight screening workflow")

                # Check daily signal limit
                if not self._check_daily_limit():
                    logger.warning("Daily signal limit reached, skipping screening")
                    result['skip_reason'] = 'daily_limit_reached'
                    return result

                # Collect stock data with focus on quality stocks
                stocks_data = await self.data_collector.fetch_filtered_data(
                    min_price=self.config.screening_criteria.min_price,
                    max_price=self.config.screening_criteria.max_price,
                    min_volume=self.config.screening_criteria.min_volume * 1.5,  # Higher volume for overnight
                    exclude_sectors=self.config.screening_criteria.exclude_sectors
                )

                if not stocks_data:
                    logger.warning("No stock data available for overnight screening")
                    result['success'] = True
                    result['skip_reason'] = 'no_data'
                    return result

                logger.info(f"Collected data for {len(stocks_data)} stocks")

                # Run overnight screening
                signals = self.screener.screen_overnight_setups(
                    stocks_data,
                    self.config.default_capital_idr
                )

                # Update tracking
                self.last_overnight_run = datetime.now()
                self.daily_signal_count += len(signals)
                self.workflow_stats['signals_generated'] += len(signals)

                # Send notifications for all overnight signals (they're rarer)
                notification_results = []
                for signal in signals[:3]:  # Max 3 overnight signals
                    try:
                        notify_result = await notify_signal(
                            signal,
                            self.config,
                            priority="normal"
                        )
                        notification_results.append(notify_result)
                    except Exception as e:
                        logger.error(f"Failed to send signal notification: {e}")

                # Save results
                await self._save_workflow_results(workflow_id, 'overnight', signals)

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
                        logger.error(f"Failed to send summary notification: {e}")

                # Prepare result
                result.update({
                    'success': True,
                    'signals': [s.to_dict() for s in signals],
                    'stats': {
                        'total_screened': len(stocks_data),
                        'signals_found': len(signals),
                        'execution_time': (datetime.now() - start_time).total_seconds()
                    },
                    'notifications_sent': len(notification_results)
                })

                logger.info(f"Overnight screening completed: {len(signals)} signals found")
                return result

            except Exception as e:
                logger.error(f"Overnight screening workflow failed: {e}")
                result['errors'].append(str(e))
                result['traceback'] = traceback.format_exc()
                return result

    async def run_risk_review(self) -> Dict[str, Any]:
        """
        Execute daily risk review workflow.

        Returns:
            Dictionary with review results
        """
        with LogContext("risk_review"):
            try:
                logger.info("Starting daily risk review")

                # Load recent signals (last 24 hours)
                recent_signals = await self._load_recent_signals(hours=24)

                if not recent_signals:
                    logger.info("No recent signals found for risk review")
                    return {'success': True, 'signals_reviewed': 0}

                # Calculate risk metrics
                risk_analysis = calculate_portfolio_risk(
                    recent_signals,
                    self.config.default_capital_idr,
                    self.config
                )

                # Generate risk report
                risk_report = self._generate_risk_report(risk_analysis, recent_signals)

                # Send risk summary if there are concerns
                if risk_analysis.get('total_risk_percentage', 0) > 80:  # More than 80% of allowed risk
                    await notify_market_update(
                        "Risk Review Alert",
                        f"Portfolio risk utilization: {risk_analysis['total_risk_percentage']:.1f}%",
                        self.config,
                        priority="high"
                    )

                logger.info(f"Risk review completed: {len(recent_signals)} signals reviewed")
                return {
                    'success': True,
                    'signals_reviewed': len(recent_signals),
                    'risk_analysis': risk_analysis
                }

            except Exception as e:
                logger.error(f"Risk review workflow failed: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }

    async def run_market_status_check(self) -> Dict[str, Any]:
        """
        Execute market status check workflow.

        Returns:
            Dictionary with market status
        """
        with LogContext("market_status_check"):
            try:
                logger.info("Running market status check")

                # Get market status
                market_status = self.data_collector.get_market_status()

                # Run data validation
                sample_symbols = self.config.default_tickers[:10]
                stocks_data = await self.data_collector.fetch_realtime_data(sample_symbols)

                validation_results = self.data_validator.validate_multiple_stocks(stocks_data)
                validation_summary = self.data_validator.get_validation_summary(validation_results)

                # Send morning update
                status_text = f"""📊 Daily Market Check

🕒 Market Status: {'Open' if market_status['is_market_open'] else 'Closed'}
📈 Data Quality: {validation_summary['validation_rate']:.1f}% pass rate
🎯 Today's Signals: {self.daily_signal_count}
📅 Next: {'Screening at 09:00' if not market_status['is_market_open'] else 'Market active'}"""

                await notify_market_update(
                    "Market Status Check",
                    status_text,
                    self.config,
                    priority="low"
                )

                return {
                    'success': True,
                    'market_status': market_status,
                    'data_quality': validation_summary['validation_rate'],
                    'daily_signal_count': self.daily_signal_count
                }

            except Exception as e:
                logger.error(f"Market status check failed: {e}")
                return {'success': False, 'error': str(e)}

    async def reset_daily_stats(self):
        """Reset daily statistics at midnight."""
        logger.info("Resetting daily statistics")

        # Save daily summary before reset
        daily_summary = {
            'date': self.workflow_stats['last_reset_date'].isoformat(),
            'signals_generated': self.daily_signal_count,
            'total_runs': self.workflow_stats['total_runs'],
            'success_rate': (self.workflow_stats['successful_runs'] /
                           max(self.workflow_stats['total_runs'], 1)) * 100
        }

        # Save summary to file
        summary_file = self.logs_dir / f"daily_summary_{self.workflow_stats['last_reset_date'].strftime('%Y%m%d')}.json"
        with open(summary_file, 'w') as f:
            json.dump(daily_summary, f, indent=2, default=str)

        # Reset counters
        self.daily_signal_count = 0
        self.workflow_stats['last_reset_date'] = datetime.now().date()

    def _check_daily_limit(self) -> bool:
        """Check if daily signal limit has been reached."""
        current_date = datetime.now().date()

        # Reset if new day
        if current_date != self.workflow_stats['last_reset_date']:
            self.daily_signal_count = 0
            self.workflow_stats['last_reset_date'] = current_date

        return self.daily_signal_count < self.config.notifications.max_signals_per_day

    async def _save_workflow_results(self, workflow_id: str, workflow_type: str, signals: List):
        """Save workflow results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        results = {
            'workflow_id': workflow_id,
            'workflow_type': workflow_type,
            'timestamp': timestamp,
            'signals_count': len(signals),
            'signals': [signal.to_dict() for signal in signals]
        }

        json_file = self.logs_dir / f"{workflow_type}_screening_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save CSV for Excel compatibility
        if signals:
            from ..notifications.formatter import NotificationFormatter
            formatter = NotificationFormatter(self.config)
            df = formatter.format_csv_signals(signals)

            csv_file = self.logs_dir / f"{workflow_type}_signals_{timestamp}.csv"
            df.to_csv(csv_file, index=False)

    async def _load_recent_signals(self, hours: int = 24) -> List:
        """Load recent signals from log files."""
        signals = []
        cutoff_time = datetime.now() - timedelta(hours=hours)

        try:
            # Load from JSON files
            for json_file in self.logs_dir.glob("*_screening_*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    file_time = datetime.fromisoformat(data['timestamp'].replace('_', ' '))
                    if file_time >= cutoff_time:
                        # Convert back to signal objects (simplified)
                        for signal_data in data.get('signals', []):
                            signals.append(signal_data)  # Keep as dict for simplicity

                except Exception as e:
                    logger.debug(f"Could not load {json_file}: {e}")

        except Exception as e:
            logger.error(f"Error loading recent signals: {e}")

        return signals

    def _generate_risk_report(self, risk_analysis: Dict, signals: List) -> str:
        """Generate human-readable risk report."""
        if not risk_analysis or 'error' in risk_analysis:
            return "Risk analysis unavailable"

        total_risk = risk_analysis.get('total_risk_percentage', 0)
        total_signals = risk_analysis.get('total_signals', 0)

        if total_risk > 80:
            risk_level = "🔴 HIGH"
        elif total_risk > 60:
            risk_level = "🟡 MEDIUM"
        else:
            risk_level = "🟢 LOW"

        report = f"""Risk Level: {risk_level}
Portfolio Risk: {total_risk:.1f}%
Active Signals: {total_signals}
Recommendations: {'Reduce position sizes' if total_risk > 80 else 'Risk within limits'}"""

        return report

    def _job_executed(self, event):
        """Handle successful job execution."""
        self.workflow_stats['total_runs'] += 1
        self.workflow_stats['successful_runs'] += 1
        logger.debug(f"Job executed successfully: {event.job_id}")

    def _job_error(self, event):
        """Handle job execution errors."""
        self.workflow_stats['total_runs'] += 1
        self.workflow_stats['failed_runs'] += 1
        logger.error(f"Job failed: {event.job_id} - {event.exception}")

    async def shutdown(self):
        """Shutdown the workflow orchestrator."""
        if self.is_running:
            logger.info("Shutting down workflow orchestrator...")
            self.scheduler.shutdown(wait=True)
            self.is_running = False
            logger.info("Workflow orchestrator shut down complete")

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        return {
            'is_running': self.is_running,
            'last_intraday_run': self.last_intraday_run.isoformat() if self.last_intraday_run else None,
            'last_overnight_run': self.last_overnight_run.isoformat() if self.last_overnight_run else None,
            'daily_signal_count': self.daily_signal_count,
            'workflow_stats': self.workflow_stats.copy(),
            'scheduled_jobs': [
                {
                    'id': job.id,
                    'name': job.name,
                    'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None
                }
                for job in self.scheduler.get_jobs()
            ]
        }


# Convenience functions
async def create_orchestrator(config: TradingConfig) -> WorkflowOrchestrator:
    """Create and initialize a workflow orchestrator."""
    orchestrator = WorkflowOrchestrator(config)
    if await orchestrator.initialize():
        return orchestrator
    else:
        raise RuntimeError("Failed to initialize workflow orchestrator")


async def run_single_workflow(
    workflow_type: str,
    config: TradingConfig
) -> Dict[str, Any]:
    """Run a single workflow without scheduling."""
    orchestrator = WorkflowOrchestrator(config)

    try:
        if workflow_type == 'intraday':
            return await orchestrator.run_intraday_screening()
        elif workflow_type == 'overnight':
            return await orchestrator.run_overnight_screening()
        elif workflow_type == 'risk_review':
            return await orchestrator.run_risk_review()
        elif workflow_type == 'market_status':
            return await orchestrator.run_market_status_check()
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

    finally:
        await orchestrator.shutdown()

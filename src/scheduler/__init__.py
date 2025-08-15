# -*- coding: utf-8 -*-
"""
Scheduler Package for Indonesian Stock Screener
==============================================

This package provides workflow scheduling and orchestration functionality
for automating Indonesian stock screening operations. It handles job scheduling,
error recovery, and coordination between different system components.

Modules:
    workflows: Main workflow orchestrator with scheduling capabilities
"""

from .workflows import (
    WorkflowOrchestrator,
    create_orchestrator,
    run_single_workflow,
)

__all__ = [
    "WorkflowOrchestrator",
    "create_orchestrator",
    "run_single_workflow",
]

__version__ = "1.0.0"
__author__ = "Indonesian Stock Screener Team"
__description__ = "Workflow scheduling and orchestration for Indonesian stock screening"

# Convenience function for quick workflow execution
async def run_screening_workflow(mode='both', config=None):
    """
    Run screening workflow in specified mode.

    Args:
        mode: Screening mode ('intraday', 'overnight', 'both')
        config: TradingConfig object

    Returns:
        Dictionary with workflow results
    """
    if config is None:
        from ..config import load_config
        config = load_config()

    results = {}

    try:
        if mode in ['intraday', 'both']:
            intraday_result = await run_single_workflow('intraday', config)
            results['intraday'] = intraday_result

        if mode in ['overnight', 'both']:
            overnight_result = await run_single_workflow('overnight', config)
            results['overnight'] = overnight_result

        return results

    except Exception as e:
        return {'error': f'Workflow execution failed: {str(e)}'}


async def start_automated_screening(config=None):
    """
    Start automated screening with full scheduling.

    Args:
        config: TradingConfig object

    Returns:
        WorkflowOrchestrator instance
    """
    if config is None:
        from ..config import load_config
        config = load_config()

    try:
        orchestrator = await create_orchestrator(config)
        return orchestrator
    except Exception as e:
        from loguru import logger
        logger.error(f"Failed to start automated screening: {e}")
        raise


def get_market_schedule():
    """
    Get IDX market schedule information.

    Returns:
        Dictionary with market hours and schedule
    """
    return {
        'market_open': '09:00',
        'market_close': '15:00',
        'timezone': 'Asia/Jakarta',
        'trading_days': 'Monday-Friday',
        'intraday_screening': 'Every 30 minutes during market hours',
        'overnight_screening': 'Daily at 17:00 WIB',
        'risk_review': 'Daily at 08:00 WIB',
        'market_status_check': 'Daily at 08:00 WIB'
    }


def validate_schedule_config(config):
    """
    Validate scheduling configuration.

    Args:
        config: TradingConfig object

    Returns:
        Tuple of (is_valid, issues)
    """
    issues = []

    # Check cron expressions
    try:
        from croniter import croniter

        cron_jobs = [
            ('intraday_screening_cron', config.scheduling.intraday_screening_cron),
            ('overnight_screening_cron', config.scheduling.overnight_screening_cron),
            ('risk_review_cron', config.scheduling.risk_review_cron)
        ]

        for name, cron_expr in cron_jobs:
            if not croniter.is_valid(cron_expr):
                issues.append(f"Invalid cron expression for {name}: {cron_expr}")

    except ImportError:
        issues.append("croniter package not available for validation")
    except Exception as e:
        issues.append(f"Error validating cron expressions: {str(e)}")

    # Check market hours
    if not (0 <= config.scheduling.market_open_hour <= 23):
        issues.append("Invalid market open hour")

    if not (0 <= config.scheduling.market_close_hour <= 23):
        issues.append("Invalid market close hour")

    if config.scheduling.market_open_hour >= config.scheduling.market_close_hour:
        issues.append("Market open hour must be before close hour")

    # Check notification limits
    if config.notifications.max_signals_per_day <= 0:
        issues.append("Max signals per day must be positive")

    if config.notifications.signal_cooldown_minutes < 0:
        issues.append("Signal cooldown cannot be negative")

    return len(issues) == 0, issues

# -*- coding: utf-8 -*-
"""
Notifications Package for Indonesian Stock Screener
==================================================

This package provides notification and messaging functionality for delivering
trading signals and market updates through various channels including Telegram,
email, and console output.

Modules:
    telegram_bot: Telegram bot implementation for signal delivery
    formatter: Message formatting utilities for different channels
"""

from .telegram_bot import (
    TelegramSignalBot,
    create_bot,
    send_signal_to_telegram,
    send_market_update_to_telegram,
)

from .formatter import (
    NotificationFormatter,
    format_signal_for_channel,
)

__all__ = [
    # Telegram Bot
    "TelegramSignalBot",
    "create_bot",
    "send_signal_to_telegram",
    "send_market_update_to_telegram",

    # Formatters
    "NotificationFormatter",
    "format_signal_for_channel",
]

__version__ = "1.0.0"
__author__ = "Indonesian Stock Screener Team"
__description__ = "Notification and messaging system for Indonesian stock screening"

# Convenience function for quick signal delivery
async def notify_signal(signal, config, channels=None, priority="normal"):
    """
    Send signal notification through specified channels.

    Args:
        signal: TradingSignal to send
        config: TradingConfig object
        channels: List of channels ('telegram', 'console') or None for all enabled
        priority: Signal priority level

    Returns:
        Dictionary with delivery results for each channel
    """
    if channels is None:
        channels = []
        if config.enable_telegram_notifications and config.notifications.telegram_bot_token:
            channels.append('telegram')
        channels.append('console')  # Always include console

    results = {}

    for channel in channels:
        try:
            if channel == 'telegram':
                success = await send_signal_to_telegram(signal, config, priority)
                results[channel] = {'success': success, 'error': None}

            elif channel == 'console':
                formatter = NotificationFormatter(config)
                message = formatter.format_signal_console(signal)
                print("\n" + message + "\n")
                results[channel] = {'success': True, 'error': None}

            else:
                results[channel] = {'success': False, 'error': f'Unknown channel: {channel}'}

        except Exception as e:
            results[channel] = {'success': False, 'error': str(e)}

    return results


async def notify_market_update(title, content, config, priority="normal", channels=None):
    """
    Send market update through specified channels.

    Args:
        title: Update title
        content: Update content
        config: TradingConfig object
        priority: Update priority level
        channels: List of channels or None for all enabled

    Returns:
        Dictionary with delivery results for each channel
    """
    if channels is None:
        channels = []
        if config.enable_telegram_notifications and config.notifications.telegram_bot_token:
            channels.append('telegram')
        channels.append('console')

    results = {}

    for channel in channels:
        try:
            if channel == 'telegram':
                success = await send_market_update_to_telegram(content, config, priority)
                results[channel] = {'success': success, 'error': None}

            elif channel == 'console':
                print(f"\n📊 {title}\n{content}\n")
                results[channel] = {'success': True, 'error': None}

            else:
                results[channel] = {'success': False, 'error': f'Unknown channel: {channel}'}

        except Exception as e:
            results[channel] = {'success': False, 'error': str(e)}

    return results

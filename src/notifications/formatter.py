# -*- coding: utf-8 -*-
"""
Notification Formatter for Indonesian Stock Screener
===================================================

Utility functions for formatting notifications, alerts, and messages
for various delivery channels including Telegram, email, and console output.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Union
import pandas as pd

from ..data.models import TradingSignal, SignalType, RiskLevel, StockData
from ..config.settings import TradingConfig


class NotificationFormatter:
    """
    Formatter for creating consistent notification messages across different channels.

    Supports multiple output formats:
    - Telegram (HTML formatting)
    - Console (plain text with colors)
    - Email (HTML)
    - JSON (structured data)
    """

    def __init__(self, config: TradingConfig):
        """
        Initialize notification formatter.

        Args:
            config: Trading configuration for formatting preferences
        """
        self.config = config
        self.timezone = "WIB"

        # Emoji mappings for different contexts
        self.signal_emojis = {
            SignalType.INTRADAY_REBOUND: "🚀",
            SignalType.OVERNIGHT_SETUP: "🌙",
            SignalType.BREAKOUT: "⚡",
            SignalType.PULLBACK: "📉",
            SignalType.REVERSAL: "🔄"
        }

        self.risk_emojis = {
            RiskLevel.LOW: "🟢",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.HIGH: "🔴"
        }

        self.priority_emojis = {
            "low": "ℹ️",
            "normal": "📊",
            "high": "🚨",
            "urgent": "🔥"
        }

    def format_signal_telegram(
        self,
        signal: TradingSignal,
        priority: str = "normal",
        include_analysis: bool = True
    ) -> str:
        """
        Format trading signal for Telegram delivery.

        Args:
            signal: Trading signal to format
            priority: Signal priority level
            include_analysis: Include detailed technical analysis

        Returns:
            Formatted HTML message for Telegram
        """
        signal_emoji = self.signal_emojis.get(signal.signal_type, "📊")
        priority_emoji = self.priority_emojis.get(priority, "📊")
        risk_emoji = self.risk_emojis.get(signal.risk_level, "🟡")

        # Header
        lines = [
            f"{priority_emoji} <b>TRADING SIGNAL</b> {signal_emoji}",
            ""
        ]

        # Basic signal info
        lines.extend([
            f"📊 <b>{signal.symbol}</b>",
            f"🎯 {signal.signal_type.value.replace('_', ' ').title()}",
            f"{risk_emoji} Risk: {signal.risk_level.value.title()}",
            ""
        ])

        # Price levels
        lines.extend([
            f"💰 Entry: <b>IDR {signal.entry_price:,.0f}</b>",
            f"🛑 Stop Loss: IDR {signal.risk_params.stop_loss:,.0f}",
        ])

        # Take profit levels
        for i, tp in enumerate(signal.risk_params.take_profit_levels, 1):
            if len(signal.risk_params.take_profit_levels) > 1:
                lines.append(f"🎯 TP{i}: IDR {tp.price:,.0f} ({tp.percentage:.0f}%)")
            else:
                lines.append(f"🎯 Take Profit: IDR {tp.price:,.0f}")

        lines.append("")

        # Risk metrics
        lines.extend([
            f"📈 Risk/Reward: <b>{signal.risk_params.risk_reward_ratio:.1f}:1</b>",
            f"🎲 Confidence: <b>{signal.confidence_score:.0%}</b>"
        ])

        # Position sizing if available
        if signal.position_sizing and signal.position_sizing.shares > 0:
            lines.extend([
                "",
                "💼 <b>Position Suggestion:</b>",
                f"   • Shares: {signal.position_sizing.shares:,} ({signal.position_sizing.lots} lots)",
                f"   • Value: IDR {signal.position_sizing.position_value:,.0f}",
                f"   • Risk: IDR {signal.position_sizing.risk_amount:,.0f} ({signal.position_sizing.risk_percentage:.1f}%)"
            ])

        # Technical context
        if include_analysis and signal.context:
            tech_lines = []
            if signal.context.rsi:
                tech_lines.append(f"RSI: {signal.context.rsi:.0f}")
            if signal.context.market_condition:
                tech_lines.append(f"Market: {signal.context.market_condition.title()}")
            if signal.context.volume_analysis:
                tech_lines.append(f"Volume: {signal.context.volume_analysis.replace('_', ' ').title()}")

            if tech_lines:
                lines.extend(["", f"📊 {' | '.join(tech_lines)}"])

        # Reasoning
        lines.extend([
            "",
            f"💡 <b>Analysis:</b> {signal.entry_reasoning}"
        ])

        # Tags
        if signal.tags:
            tag_text = " ".join(f"#{tag}" for tag in signal.tags[:3])
            lines.append(f"🏷️ {tag_text}")

        # Footer
        lines.extend([
            "",
            f"⏰ {signal.timestamp.strftime('%H:%M:%S')} {self.timezone}",
            f"🆔 <code>{signal.signal_id[-8:]}</code>",
            "",
            "⚠️ <i>This is not financial advice. Trade at your own risk.</i>"
        ])

        return "\n".join(lines)

    def format_signal_console(
        self,
        signal: TradingSignal,
        width: int = 80,
        use_colors: bool = True
    ) -> str:
        """
        Format trading signal for console output.

        Args:
            signal: Trading signal to format
            width: Console width for formatting
            use_colors: Use ANSI color codes

        Returns:
            Formatted console message
        """
        # Color codes (if enabled)
        if use_colors:
            BOLD = '\033[1m'
            GREEN = '\033[92m'
            YELLOW = '\033[93m'
            RED = '\033[91m'
            BLUE = '\033[94m'
            RESET = '\033[0m'
        else:
            BOLD = GREEN = YELLOW = RED = BLUE = RESET = ''

        # Risk color
        risk_color = {
            RiskLevel.LOW: GREEN,
            RiskLevel.MEDIUM: YELLOW,
            RiskLevel.HIGH: RED
        }.get(signal.risk_level, YELLOW)

        lines = []

        # Header
        header = f" {signal.signal_type.value.replace('_', ' ').upper()} SIGNAL "
        header_line = "=" * width
        centered_header = header.center(width, "=")

        lines.extend([
            header_line,
            f"{BOLD}{BLUE}{centered_header}{RESET}",
            header_line,
            ""
        ])

        # Basic info
        lines.extend([
            f"{BOLD}Symbol:{RESET} {signal.symbol}",
            f"{BOLD}Entry Price:{RESET} IDR {signal.entry_price:,.0f}",
            f"{BOLD}Stop Loss:{RESET} IDR {signal.risk_params.stop_loss:,.0f}",
            f"{BOLD}Take Profit:{RESET} IDR {signal.risk_params.primary_take_profit:,.0f}",
            "",
            f"{BOLD}Risk/Reward:{RESET} {signal.risk_params.risk_reward_ratio:.1f}:1",
            f"{BOLD}Confidence:{RESET} {signal.confidence_score:.1%}",
            f"{BOLD}Risk Level:{RESET} {risk_color}{signal.risk_level.value.upper()}{RESET}",
            ""
        ])

        # Position sizing
        if signal.position_sizing and signal.position_sizing.shares > 0:
            lines.extend([
                f"{BOLD}Position Size:{RESET} {signal.position_sizing.shares:,} shares ({signal.position_sizing.lots} lots)",
                f"{BOLD}Position Value:{RESET} IDR {signal.position_sizing.position_value:,.0f}",
                f"{BOLD}Risk Amount:{RESET} IDR {signal.position_sizing.risk_amount:,.0f} ({signal.position_sizing.risk_percentage:.1f}%)",
                ""
            ])

        # Analysis
        lines.extend([
            f"{BOLD}Analysis:{RESET} {signal.entry_reasoning}",
            "",
            f"{BOLD}Time:{RESET} {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')} {self.timezone}",
            f"{BOLD}Signal ID:{RESET} {signal.signal_id[-12:]}",
        ])

        # Footer
        lines.extend([
            "",
            "=" * width,
            f"{'DISCLAIMER: Not financial advice - Trade at your own risk'.center(width)}",
            "=" * width
        ])

        return "\n".join(lines)

    def format_signal_summary_table(self, signals: List[TradingSignal]) -> str:
        """
        Format multiple signals as a summary table.

        Args:
            signals: List of trading signals

        Returns:
            Formatted table string
        """
        if not signals:
            return "No signals to display."

        # Prepare data for table
        data = []
        for i, signal in enumerate(signals, 1):
            data.append({
                '#': i,
                'Symbol': signal.symbol,
                'Type': signal.signal_type.value.replace('_', ' ').title()[:8],
                'Entry': f"{signal.entry_price:,.0f}",
                'SL': f"{signal.risk_params.stop_loss:,.0f}",
                'TP': f"{signal.risk_params.primary_take_profit:,.0f}",
                'R/R': f"{signal.risk_params.risk_reward_ratio:.1f}",
                'Conf': f"{signal.confidence_score:.0%}",
                'Risk': signal.risk_level.value.title()[:3],
                'Time': signal.timestamp.strftime('%H:%M')
            })

        # Create DataFrame for better formatting
        df = pd.DataFrame(data)

        # Format as string table
        table_str = df.to_string(index=False, max_colwidth=10)

        return table_str

    def format_market_update_telegram(
        self,
        title: str,
        content: str,
        priority: str = "normal",
        include_timestamp: bool = True
    ) -> str:
        """
        Format market update for Telegram.

        Args:
            title: Update title
            content: Update content
            priority: Priority level
            include_timestamp: Include timestamp

        Returns:
            Formatted HTML message
        """
        priority_emoji = self.priority_emojis.get(priority, "📊")

        lines = [
            f"{priority_emoji} <b>{title}</b>",
            "",
            content
        ]

        if include_timestamp:
            lines.extend([
                "",
                f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {self.timezone}</i>"
            ])

        return "\n".join(lines)

    def format_screening_summary_telegram(
        self,
        intraday_count: int,
        overnight_count: int,
        total_screened: int,
        execution_time: float,
        top_signals: Optional[List[TradingSignal]] = None
    ) -> str:
        """
        Format screening session summary for Telegram.

        Args:
            intraday_count: Number of intraday signals found
            overnight_count: Number of overnight signals found
            total_screened: Total stocks screened
            execution_time: Screening execution time in seconds
            top_signals: Top signals to highlight

        Returns:
            Formatted summary message
        """
        lines = [
            "📊 <b>Screening Session Complete</b>",
            "",
            f"🔍 Stocks Screened: {total_screened}",
            f"🚀 Intraday Signals: {intraday_count}",
            f"🌙 Overnight Signals: {overnight_count}",
            f"⏱️ Execution Time: {execution_time:.1f}s",
            ""
        ]

        total_signals = intraday_count + overnight_count

        if total_signals == 0:
            lines.append("💭 No trading opportunities found at this time.")
        else:
            lines.append(f"✅ Found {total_signals} trading opportunities")

            if top_signals:
                lines.extend([
                    "",
                    "<b>🏆 Top Signals:</b>"
                ])

                for i, signal in enumerate(top_signals[:3], 1):
                    signal_type = signal.signal_type.value.replace('_', ' ').title()
                    lines.append(
                        f"{i}. <b>{signal.symbol}</b> - {signal_type} "
                        f"(Conf: {signal.confidence_score:.0%})"
                    )

        lines.extend([
            "",
            f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {self.timezone}"
        ])

        return "\n".join(lines)

    def format_error_notification(
        self,
        error_title: str,
        error_details: str,
        priority: str = "high"
    ) -> str:
        """
        Format error notification for Telegram.

        Args:
            error_title: Error title
            error_details: Error details
            priority: Error priority

        Returns:
            Formatted error message
        """
        priority_emoji = self.priority_emojis.get(priority, "🚨")

        lines = [
            f"{priority_emoji} <b>SYSTEM ERROR</b>",
            "",
            f"<b>Error:</b> {error_title}",
            "",
            f"<b>Details:</b>",
            f"<code>{error_details}</code>",
            "",
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {self.timezone}",
            "",
            "Please check the system logs for more information."
        ]

        return "\n".join(lines)

    def format_performance_report(
        self,
        period_days: int,
        total_signals: int,
        avg_confidence: float,
        signal_types: Dict[str, int],
        top_symbols: List[Tuple[str, int]]
    ) -> str:
        """
        Format performance report for Telegram.

        Args:
            period_days: Reporting period in days
            total_signals: Total signals generated
            avg_confidence: Average confidence score
            signal_types: Signal type distribution
            top_symbols: Most active symbols

        Returns:
            Formatted performance report
        """
        lines = [
            "📈 <b>PERFORMANCE REPORT</b>",
            "",
            f"📅 Period: Last {period_days} days",
            f"🎯 Total Signals: {total_signals}",
            f"🎲 Avg Confidence: {avg_confidence:.1%}",
            f"📊 Daily Average: {total_signals / period_days:.1f}",
            ""
        ]

        if signal_types:
            lines.append("<b>📋 By Strategy:</b>")
            for signal_type, count in signal_types.items():
                percentage = (count / total_signals * 100) if total_signals > 0 else 0
                lines.append(f"• {signal_type.replace('_', ' ').title()}: {count} ({percentage:.0f}%)")
            lines.append("")

        if top_symbols:
            lines.append("<b>🏆 Top Symbols:</b>")
            for symbol, count in top_symbols[:5]:
                lines.append(f"• {symbol}: {count} signals")
            lines.append("")

        lines.extend([
            f"📅 Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} {self.timezone}",
            "",
            "<i>Historical performance doesn't guarantee future results.</i>"
        ])

        return "\n".join(lines)

    def format_json_signal(self, signal: TradingSignal) -> Dict:
        """
        Format signal as structured JSON data.

        Args:
            signal: Trading signal to format

        Returns:
            Dictionary representation for JSON serialization
        """
        return {
            'signal_id': signal.signal_id,
            'symbol': signal.symbol,
            'signal_type': signal.signal_type.value,
            'timestamp': signal.timestamp.isoformat(),
            'entry_price': signal.entry_price,
            'stop_loss': signal.risk_params.stop_loss,
            'take_profit_levels': [
                {
                    'price': tp.price,
                    'percentage': tp.percentage,
                    'reasoning': tp.reasoning
                }
                for tp in signal.risk_params.take_profit_levels
            ],
            'risk_reward_ratio': signal.risk_params.risk_reward_ratio,
            'confidence_score': signal.confidence_score,
            'risk_level': signal.risk_level.value,
            'reasoning': signal.entry_reasoning,
            'position_sizing': {
                'shares': signal.position_sizing.shares,
                'lots': signal.position_sizing.lots,
                'position_value': signal.position_sizing.position_value,
                'risk_amount': signal.position_sizing.risk_amount,
                'risk_percentage': signal.position_sizing.risk_percentage
            } if signal.position_sizing else None,
            'context': {
                'market_condition': signal.context.market_condition,
                'volume_analysis': signal.context.volume_analysis,
                'rsi': signal.context.rsi,
                'technical_setup': signal.context.technical_setup
            } if signal.context else None,
            'tags': signal.tags,
            'expires_at': signal.expires_at.isoformat() if signal.expires_at else None
        }

    def format_csv_signals(self, signals: List[TradingSignal]) -> pd.DataFrame:
        """
        Format signals as CSV-ready DataFrame.

        Args:
            signals: List of trading signals

        Returns:
            DataFrame ready for CSV export
        """
        data = []

        for signal in signals:
            row = {
                'Signal_ID': signal.signal_id,
                'Symbol': signal.symbol,
                'Type': signal.signal_type.value,
                'Timestamp': signal.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Entry_Price': signal.entry_price,
                'Stop_Loss': signal.risk_params.stop_loss,
                'Take_Profit_1': signal.risk_params.take_profit_levels[0].price if signal.risk_params.take_profit_levels else None,
                'Take_Profit_2': signal.risk_params.take_profit_levels[1].price if len(signal.risk_params.take_profit_levels) > 1 else None,
                'Risk_Reward_Ratio': signal.risk_params.risk_reward_ratio,
                'Confidence': signal.confidence_score,
                'Risk_Level': signal.risk_level.value,
                'Shares': signal.position_sizing.shares if signal.position_sizing else 0,
                'Position_Value': signal.position_sizing.position_value if signal.position_sizing else 0,
                'Risk_Amount': signal.position_sizing.risk_amount if signal.position_sizing else 0,
                'Risk_Percentage': signal.position_sizing.risk_percentage if signal.position_sizing else 0,
                'Reasoning': signal.entry_reasoning,
                'Tags': ','.join(signal.tags) if signal.tags else '',
                'RSI': signal.context.rsi if signal.context else None,
                'Market_Condition': signal.context.market_condition if signal.context else None,
                'Volume_Analysis': signal.context.volume_analysis if signal.context else None
            }
            data.append(row)

        return pd.DataFrame(data)


def format_signal_for_channel(
    signal: TradingSignal,
    channel: str,
    config: TradingConfig,
    **kwargs
) -> Union[str, Dict]:
    """
    Convenience function to format signal for specific channel.

    Args:
        signal: Trading signal to format
        channel: Output channel ('telegram', 'console', 'json')
        config: Trading configuration
        **kwargs: Additional formatting options

    Returns:
        Formatted message (string) or structured data (dict)
    """
    formatter = NotificationFormatter(config)

    if channel == 'telegram':
        return formatter.format_signal_telegram(signal, **kwargs)
    elif channel == 'console':
        return formatter.format_signal_console(signal, **kwargs)
    elif channel == 'json':
        return formatter.format_json_signal(signal)
    else:
        raise ValueError(f"Unknown channel: {channel}")

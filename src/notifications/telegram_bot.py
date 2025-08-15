# -*- coding: utf-8 -*-
"""
Telegram Bot for Indonesian Stock Screener
==========================================

Telegram bot implementation for delivering trading signals and notifications.
Provides formatted messages with charts, interactive commands, and user management.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from pathlib import Path

import telegram
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes
)
from telegram.error import TelegramError, BadRequest, Forbidden
from loguru import logger

from ..config.settings import TradingConfig
from ..data.models import TradingSignal, SignalStatus
from ..utils.logger import LogContext


class TelegramSignalBot:
    """
    Telegram bot for delivering Indonesian stock screening signals.

    Features:
    - Automated signal delivery with rich formatting
    - Interactive commands for signal management
    - User subscription management
    - Rate limiting and spam protection
    - Signal history and statistics
    """

    def __init__(self, config: TradingConfig):
        """
        Initialize Telegram bot.

        Args:
            config: Trading configuration with Telegram settings
        """
        self.config = config
        self.bot_token = config.notifications.telegram_bot_token
        self.chat_ids = set(config.notifications.telegram_chat_ids)
        self.max_signals_per_day = config.notifications.max_signals_per_day
        self.signal_cooldown_minutes = config.notifications.signal_cooldown_minutes

        # Bot state
        self.bot: Optional[Bot] = None
        self.application: Optional[Application] = None
        self.is_running = False

        # Signal tracking
        self.daily_signal_count = 0
        self.last_reset_date = datetime.now().date()
        self.signal_history: List[Dict] = []
        self.last_signal_time: Dict[str, datetime] = {}  # symbol -> last signal time

        # User management
        self.authorized_users: Set[int] = set()
        self.blocked_users: Set[int] = set()

        # Rate limiting
        self.user_command_count: Dict[int, List[datetime]] = {}
        self.max_commands_per_minute = 10

        logger.info(f"TelegramSignalBot initialized for {len(self.chat_ids)} chat(s)")

    async def initialize(self) -> bool:
        """
        Initialize the Telegram bot.

        Returns:
            True if initialization successful, False otherwise
        """
        if not self.bot_token:
            logger.error("Telegram bot token not provided")
            return False

        try:
            self.bot = Bot(token=self.bot_token)

            # Test bot connection
            bot_info = await self.bot.get_me()
            logger.info(f"Bot connected: @{bot_info.username} ({bot_info.first_name})")

            # Initialize application for webhook/polling
            self.application = Application.builder().token(self.bot_token).build()

            # Add command handlers
            self._setup_handlers()

            # Load user data if exists
            await self._load_user_data()

            return True

        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            return False

    def _setup_handlers(self):
        """Setup command and callback handlers."""
        if not self.application:
            return

        # Command handlers
        self.application.add_handler(CommandHandler("start", self._cmd_start))
        self.application.add_handler(CommandHandler("help", self._cmd_help))
        self.application.add_handler(CommandHandler("status", self._cmd_status))
        self.application.add_handler(CommandHandler("signals", self._cmd_signals))
        self.application.add_handler(CommandHandler("subscribe", self._cmd_subscribe))
        self.application.add_handler(CommandHandler("unsubscribe", self._cmd_unsubscribe))
        self.application.add_handler(CommandHandler("stats", self._cmd_stats))
        self.application.add_handler(CommandHandler("config", self._cmd_config))

        # Callback query handler for inline keyboards
        self.application.add_handler(CallbackQueryHandler(self._handle_callback))

        # Message handler for unknown commands
        self.application.add_handler(MessageHandler(filters.TEXT, self._handle_unknown))

    async def send_trading_signal(
        self,
        signal: TradingSignal,
        priority: str = "normal"
    ) -> bool:
        """
        Send trading signal to authorized chats.

        Args:
            signal: Trading signal to send
            priority: Signal priority ("low", "normal", "high")

        Returns:
            True if sent successfully to at least one chat
        """
        if not self.bot:
            logger.warning("Bot not initialized, cannot send signal")
            return False

        # Check daily limit
        if not self._check_daily_limit():
            logger.warning(f"Daily signal limit ({self.max_signals_per_day}) reached")
            return False

        # Check cooldown for this symbol
        if not self._check_signal_cooldown(signal.symbol):
            logger.info(f"Signal cooldown active for {signal.symbol}")
            return False

        with LogContext("telegram_signal_send", symbol=signal.symbol):
            try:
                # Format signal message
                message = self._format_signal_message(signal, priority)
                keyboard = self._create_signal_keyboard(signal)

                success_count = 0

                for chat_id in self.chat_ids:
                    try:
                        await self.bot.send_message(
                            chat_id=chat_id,
                            text=message,
                            parse_mode='HTML',
                            reply_markup=keyboard,
                            disable_web_page_preview=True
                        )
                        success_count += 1
                        logger.debug(f"Signal sent to chat {chat_id}")

                    except Forbidden:
                        logger.warning(f"Bot blocked by chat {chat_id}")
                        self.chat_ids.discard(chat_id)

                    except BadRequest as e:
                        logger.warning(f"Bad request for chat {chat_id}: {e}")

                    except Exception as e:
                        logger.error(f"Failed to send to chat {chat_id}: {e}")

                if success_count > 0:
                    self._update_signal_tracking(signal)
                    logger.info(f"Signal sent to {success_count} chats")
                    return True
                else:
                    logger.error("Failed to send signal to any chat")
                    return False

            except Exception as e:
                logger.error(f"Error sending trading signal: {e}")
                return False

    async def send_market_update(self, update_text: str, priority: str = "low") -> bool:
        """
        Send market update message.

        Args:
            update_text: Update message text
            priority: Update priority

        Returns:
            True if sent successfully
        """
        if not self.bot or not self.chat_ids:
            return False

        try:
            priority_emoji = {"low": "ℹ️", "normal": "📊", "high": "🚨"}
            emoji = priority_emoji.get(priority, "📊")

            message = f"{emoji} <b>Market Update</b>\n\n{update_text}\n\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S WIB')}</i>"

            success_count = 0
            for chat_id in self.chat_ids:
                try:
                    await self.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode='HTML'
                    )
                    success_count += 1
                except Exception as e:
                    logger.debug(f"Failed to send update to {chat_id}: {e}")

            return success_count > 0

        except Exception as e:
            logger.error(f"Error sending market update: {e}")
            return False

    def _format_signal_message(self, signal: TradingSignal, priority: str) -> str:
        """Format trading signal for Telegram."""

        # Priority emoji
        priority_emoji = {"low": "📝", "normal": "🚀", "high": "⚡"}
        signal_emoji = priority_emoji.get(priority, "🚀")

        # Risk level emoji
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}
        risk_color = risk_emoji.get(signal.risk_level.value, "🟡")

        # Signal type formatting
        signal_type_text = signal.signal_type.value.replace('_', ' ').title()

        message_lines = [
            f"{signal_emoji} <b>TRADING SIGNAL</b> {signal_emoji}",
            "",
            f"📊 <b>{signal.symbol}</b>",
            f"🎯 Strategy: {signal_type_text}",
            f"{risk_color} Risk: {signal.risk_level.value.title()}",
            "",
            f"💰 Entry: <b>IDR {signal.entry_price:,.0f}</b>",
            f"🛑 Stop Loss: IDR {signal.risk_params.stop_loss:,.0f}",
        ]

        # Add take profit levels
        for i, tp in enumerate(signal.risk_params.take_profit_levels, 1):
            percentage_text = f" ({tp.percentage:.0f}%)" if len(signal.risk_params.take_profit_levels) > 1 else ""
            message_lines.append(f"🎯 TP {i}: IDR {tp.price:,.0f}{percentage_text}")

        message_lines.extend([
            "",
            f"📈 R/R Ratio: <b>{signal.risk_params.risk_reward_ratio:.1f}:1</b>",
            f"🎲 Confidence: <b>{signal.confidence_score:.0%}</b>",
        ])

        # Add position sizing if available
        if signal.position_sizing:
            message_lines.extend([
                "",
                f"💼 Suggested Position:",
                f"   • Shares: {signal.position_sizing.shares:,} ({signal.position_sizing.lots} lots)",
                f"   • Value: IDR {signal.position_sizing.position_value:,.0f}",
                f"   • Risk: IDR {signal.position_sizing.risk_amount:,.0f} ({signal.position_sizing.risk_percentage:.1f}%)",
            ])

        # Add technical context
        if signal.context.rsi:
            message_lines.append(f"📊 RSI: {signal.context.rsi:.0f}")

        if signal.context.technical_setup:
            message_lines.append(f"🔧 Setup: {signal.context.technical_setup}")

        message_lines.extend([
            "",
            f"💡 <b>Reasoning:</b> {signal.entry_reasoning}",
        ])

        # Add tags if any
        if signal.tags:
            tag_text = " ".join(f"#{tag}" for tag in signal.tags[:3])  # Limit to 3 tags
            message_lines.append(f"🏷️ {tag_text}")

        message_lines.extend([
            "",
            f"⏰ {signal.timestamp.strftime('%H:%M:%S')} WIB",
            f"🆔 <code>{signal.signal_id[-8:]}</code>",  # Short ID for display
        ])

        # Add disclaimer
        message_lines.extend([
            "",
            "⚠️ <i>Trading involves risk. This is not financial advice.</i>"
        ])

        return "\n".join(message_lines)

    def _create_signal_keyboard(self, signal: TradingSignal) -> InlineKeyboardMarkup:
        """Create inline keyboard for signal interaction."""

        keyboard = [
            [
                InlineKeyboardButton("📊 Analysis", callback_data=f"analysis_{signal.signal_id}"),
                InlineKeyboardButton("📈 Chart", callback_data=f"chart_{signal.symbol}"),
            ],
            [
                InlineKeyboardButton("✅ Executed", callback_data=f"executed_{signal.signal_id}"),
                InlineKeyboardButton("❌ Skip", callback_data=f"skip_{signal.signal_id}"),
            ],
            [
                InlineKeyboardButton("🔔 Set Alert", callback_data=f"alert_{signal.symbol}"),
            ]
        ]

        return InlineKeyboardMarkup(keyboard)

    def _check_daily_limit(self) -> bool:
        """Check if daily signal limit is reached."""
        current_date = datetime.now().date()

        # Reset counter if new day
        if current_date != self.last_reset_date:
            self.daily_signal_count = 0
            self.last_reset_date = current_date

        return self.daily_signal_count < self.max_signals_per_day

    def _check_signal_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period."""
        if symbol not in self.last_signal_time:
            return True

        time_since_last = datetime.now() - self.last_signal_time[symbol]
        return time_since_last.total_seconds() >= (self.signal_cooldown_minutes * 60)

    def _update_signal_tracking(self, signal: TradingSignal):
        """Update signal tracking data."""
        self.daily_signal_count += 1
        self.last_signal_time[signal.symbol] = datetime.now()

        # Add to history (keep last 100 signals)
        self.signal_history.append({
            'signal_id': signal.signal_id,
            'symbol': signal.symbol,
            'type': signal.signal_type.value,
            'confidence': signal.confidence_score,
            'timestamp': signal.timestamp.isoformat(),
            'entry_price': signal.entry_price,
        })

        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]

    def _check_rate_limit(self, user_id: int) -> bool:
        """Check if user is rate limited."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        if user_id not in self.user_command_count:
            self.user_command_count[user_id] = []

        # Remove old timestamps
        self.user_command_count[user_id] = [
            ts for ts in self.user_command_count[user_id] if ts > cutoff
        ]

        # Check limit
        if len(self.user_command_count[user_id]) >= self.max_commands_per_minute:
            return False

        # Add current timestamp
        self.user_command_count[user_id].append(now)
        return True

    # Command Handlers
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user_id = update.effective_user.id

        if not self._check_rate_limit(user_id):
            await update.message.reply_text("⏱️ Too many commands. Please wait a moment.")
            return

        welcome_message = """
🇮🇩 <b>Indonesian Stock Screener Bot</b>

Welcome! This bot delivers automated Indonesian stock trading signals based on technical analysis.

<b>Available Commands:</b>
/help - Show all commands
/status - Market and bot status
/signals - Recent signals
/subscribe - Subscribe to signals
/unsubscribe - Unsubscribe from signals
/stats - Signal statistics
/config - Configuration info

<b>Features:</b>
• Real-time IDX stock screening
• Technical analysis with VWAP, ATR, RSI, EMA
• Risk management with position sizing
• Intraday and overnight strategies

⚠️ <i>Disclaimer: This bot provides educational information only. Trading involves risk and you should do your own research.</i>
        """

        await update.message.reply_text(welcome_message, parse_mode='HTML')

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_text = """
<b>📚 Bot Commands</b>

<b>General:</b>
/start - Welcome message
/help - This help message
/status - Market and bot status

<b>Signals:</b>
/signals - Show recent signals (last 10)
/subscribe - Subscribe to signal notifications
/unsubscribe - Unsubscribe from notifications
/stats - Signal performance statistics

<b>Information:</b>
/config - Bot configuration
/market - Current market status

<b>Signal Types:</b>
🚀 <b>Intraday Rebound</b> - Same day trades on oversold bounces
🌙 <b>Overnight Setup</b> - End-of-day positions for next day

<b>Risk Levels:</b>
🟢 Low Risk - High confidence, conservative setup
🟡 Medium Risk - Good setup with normal risk
🔴 High Risk - Speculative or lower confidence

<b>Interactive Buttons:</b>
📊 Analysis - Detailed technical analysis
📈 Chart - Price chart (if available)
✅ Executed - Mark signal as traded
❌ Skip - Mark signal as skipped
🔔 Alert - Set price alert

Need help? Contact the administrator.
        """

        await update.message.reply_text(help_text, parse_mode='HTML')

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        try:
            status_message = f"""
<b>📊 Bot Status</b>

<b>Market Status:</b> 🟢 Active
<b>Bot Status:</b> 🟢 Online
<b>Last Update:</b> {datetime.now().strftime('%H:%M:%S WIB')}

<b>Today's Activity:</b>
• Signals Sent: {self.daily_signal_count}/{self.max_signals_per_day}
• Active Chats: {len(self.chat_ids)}
• Signal Types: Intraday & Overnight

<b>Configuration:</b>
• Signal Cooldown: {self.signal_cooldown_minutes} minutes
• Max Daily Signals: {self.max_signals_per_day}
• VWAP Filter: {'✅' if self.config.enable_vwap_filter else '❌'}
• ATR TP/SL: {'✅' if self.config.enable_atr_tp_sl else '❌'}

<b>Recent Performance:</b>
• Signals Generated: {len(self.signal_history)}
• Average Confidence: {sum(s['confidence'] for s in self.signal_history[-10:]) / min(len(self.signal_history), 10):.1%}
            """

            await update.message.reply_text(status_message, parse_mode='HTML')

        except Exception as e:
            await update.message.reply_text(f"❌ Error getting status: {str(e)}")

    async def _cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command."""
        if not self.signal_history:
            await update.message.reply_text("📭 No signals found in history.")
            return

        recent_signals = self.signal_history[-10:]  # Last 10 signals

        message_lines = ["<b>📊 Recent Signals</b>\n"]

        for i, signal in enumerate(reversed(recent_signals), 1):
            timestamp = datetime.fromisoformat(signal['timestamp']).strftime('%m/%d %H:%M')
            signal_type_short = signal['type'].replace('_', ' ').title()[:8]

            message_lines.append(
                f"{i}. <b>{signal['symbol']}</b> - {signal_type_short}\n"
                f"   Entry: {signal['entry_price']:,.0f} | Conf: {signal['confidence']:.0%} | {timestamp}"
            )

        message_lines.append(f"\n📈 Total signals today: {self.daily_signal_count}")

        await update.message.reply_text("\n".join(message_lines), parse_mode='HTML')

    async def _cmd_subscribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /subscribe command."""
        chat_id = str(update.effective_chat.id)

        if chat_id in self.chat_ids:
            await update.message.reply_text("✅ You're already subscribed to signals!")
        else:
            self.chat_ids.add(chat_id)
            await self._save_user_data()
            await update.message.reply_text(
                "🔔 <b>Subscribed successfully!</b>\n\n"
                "You'll now receive trading signals from the Indonesian Stock Screener.\n"
                "Use /unsubscribe to stop receiving signals.",
                parse_mode='HTML'
            )

    async def _cmd_unsubscribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /unsubscribe command."""
        chat_id = str(update.effective_chat.id)

        if chat_id not in self.chat_ids:
            await update.message.reply_text("ℹ️ You're not currently subscribed.")
        else:
            self.chat_ids.discard(chat_id)
            await self._save_user_data()
            await update.message.reply_text(
                "🔕 <b>Unsubscribed successfully!</b>\n\n"
                "You'll no longer receive trading signals.\n"
                "Use /subscribe to resubscribe anytime.",
                parse_mode='HTML'
            )

    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command."""
        if not self.signal_history:
            await update.message.reply_text("📊 No statistics available yet.")
            return

        # Calculate statistics
        total_signals = len(self.signal_history)
        intraday_count = sum(1 for s in self.signal_history if 'intraday' in s['type'])
        overnight_count = sum(1 for s in self.signal_history if 'overnight' in s['type'])

        avg_confidence = sum(s['confidence'] for s in self.signal_history) / total_signals

        # Top symbols
        symbol_counts = {}
        for signal in self.signal_history:
            symbol_counts[signal['symbol']] = symbol_counts.get(signal['symbol'], 0) + 1

        top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        stats_message = f"""
<b>📊 Signal Statistics</b>

<b>Overall:</b>
• Total Signals: {total_signals}
• Average Confidence: {avg_confidence:.1%}
• Daily Average: {total_signals / 7:.1f} (last 7 days)

<b>By Strategy:</b>
• Intraday Rebound: {intraday_count} ({intraday_count/total_signals:.1%})
• Overnight Setup: {overnight_count} ({overnight_count/total_signals:.1%})

<b>Top Symbols:</b>
        """

        for symbol, count in top_symbols:
            stats_message += f"• {symbol}: {count} signals\n"

        stats_message += f"\n<i>Statistics based on last {total_signals} signals</i>"

        await update.message.reply_text(stats_message, parse_mode='HTML')

    async def _cmd_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /config command."""
        config_message = f"""
<b>⚙️ Bot Configuration</b>

<b>Signal Limits:</b>
• Max per day: {self.max_signals_per_day}
• Cooldown: {self.signal_cooldown_minutes} minutes
• Today's count: {self.daily_signal_count}

<b>Features:</b>
• VWAP Filter: {'✅ Enabled' if self.config.enable_vwap_filter else '❌ Disabled'}
• ATR TP/SL: {'✅ Enabled' if self.config.enable_atr_tp_sl else '❌ Disabled'}
• Charts: {'✅ Enabled' if self.config.notifications.include_charts else '❌ Disabled'}

<b>Risk Management:</b>
• Max Risk/Trade: {self.config.risk_management.max_risk_per_trade:.1%}
• Max Portfolio Risk: {self.config.risk_management.max_portfolio_risk:.1%}
• Min R/R Ratio: {self.config.risk_management.min_rr_ratio}

<b>Technical Settings:</b>
• RSI Period: {self.config.indicators.rsi_period}
• ATR Period: {self.config.indicators.atr_period}
• EMA Periods: {', '.join(map(str, self.config.indicators.ema_periods))}

<b>Active Subscribers:</b> {len(self.chat_ids)}
        """

        await update.message.reply_text(config_message, parse_mode='HTML')

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks."""
        query = update.callback_query
        await query.answer()

        data = query.data
        user_id = update.effective_user.id

        if not self._check_rate_limit(user_id):
            await query.message.reply_text("⏱️ Too many commands. Please wait.")
            return

        try:
            if data.startswith('analysis_'):
                signal_id = data.replace('analysis_', '')
                await self._show_signal_analysis(query, signal_id)

            elif data.startswith('chart_'):
                symbol = data.replace('chart_', '')
                await self._show_chart_info(query, symbol)

            elif data.startswith('executed_'):
                signal_id = data.replace('executed_', '')
                await query.message.reply_text(f"✅ Signal {signal_id[-8:]} marked as executed.")

            elif data.startswith('skip_'):
                signal_id = data.replace('skip_', '')
                await query.message.reply_text(f"❌ Signal {signal_id[-8:]} marked as skipped.")

            elif data.startswith('alert_'):
                symbol = data.replace('alert_', '')
                await query.message.reply_text(f"🔔 Alert set for {symbol} (feature coming soon)")

        except Exception as e:
            logger.error(f"Callback error: {e}")
            await query.message.reply_text("❌ Error processing request.")

    async def _show_signal_analysis(self, query, signal_id: str):
        """Show detailed signal analysis."""
        # Find signal in history
        signal_data = None
        for signal in self.signal_history:
            if signal['signal_id'].endswith(signal_id):
                signal_data = signal
                break

        if not signal_data:
            await query.message.reply_text("❌ Signal not found in history.")
            return

        analysis_text = f"""
<b>📊 Signal Analysis</b>

<b>Signal:</b> {signal_data['symbol']} - {signal_data['type'].title()}
<b>Entry:</b> IDR {signal_data['entry_price']:,.0f}
<b>Confidence:</b> {signal_data['confidence']:.0%}
<b>Time:</b> {signal_data['timestamp']}

<b>Technical Analysis:</b>
• Strategy: {signal_data['type'].replace('_', ' ').title()}
• Risk Level: Based on confidence and market conditions
• Position Sizing: Calculated using risk management rules

<b>Market Context:</b>
• IDX market hours: 09:00 - 15:00 WIB
• Data source: Yahoo Finance (10-15 min delay)
• Analysis: VWAP, ATR, RSI, EMA indicators

<i>For detailed analysis, use the main application.</i>
        """

        await query.message.reply_text(analysis_text, parse_mode='HTML')

    async def _show_chart_info(self, query, symbol: str):
        """Show chart information."""
        chart_info = f"""
<b>📈 Chart Information</b>

<b>Symbol:</b> {symbol}
<b>Exchange:</b> Indonesia Stock Exchange (IDX)

<b>Recommended Charts:</b>
• TradingView: tradingview.com/chart?symbol=IDX:{symbol[:-3]}
• Stockbit: stockbit.com/symbol/{symbol[:-3]}
• RTI Business: rtibusiness.co.id

<b>Key Levels to Watch:</b>
• Support/Resistance levels from technical analysis
• EMA lines (5, 13, 21 periods)
• VWAP line for intraday trading

<i>Charts feature will be integrated in future updates.</i>
        """

        await query.message.reply_text(chart_info, parse_mode='HTML')

    async def _handle_unknown(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle unknown commands."""
        await update.message.reply_text(
            "❓ Unknown command. Use /help to see available commands.",
            parse_mode='HTML'
        )

    async def _load_user_data(self):
        """Load user data from file."""
        try:
            data_file = Path(__file__).parent.parent.parent / "logs" / "telegram_users.json"
            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    self.chat_ids.update(data.get('chat_ids', []))
                    logger.info(f"Loaded {len(self.chat_ids)} chat IDs")
        except Exception as e:
            logger.warning(f"Could not load user data: {e}")

    async def _save_user_data(self):
        """Save user data to file."""
        try:
            data_file = Path(__file__).parent.parent.parent / "logs" / "telegram_users.json"
            data_file.parent.mkdir(exist_ok=True)

            data = {
                'chat_ids': list(self.chat_ids),
                'last_updated': datetime.now().isoformat()
            }

            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Could not save user data: {e}")

    async def start_polling(self):
        """Start bot in polling mode."""
        if not self.application:
            logger.error("Application not initialized")
            return

        try:
            logger.info("Starting Telegram bot in polling mode...")
            self.is_running = True

            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()

            logger.info("Telegram bot started successfully")

        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            self.is_running = False

    async def stop(self):
        """Stop the bot."""
        if self.application and self.is_running:
            logger.info("Stopping Telegram bot...")
            try:
                await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
                self.is_running = False
                logger.info("Telegram bot stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping bot: {e}")

    async def cleanup(self):
        """Cleanup bot resources."""
        await self._save_user_data()
        if self.is_running:
            await self.stop()

    def get_stats(self) -> Dict:
        """Get bot statistics."""
        return {
            'is_running': self.is_running,
            'chat_count': len(self.chat_ids),
            'daily_signal_count': self.daily_signal_count,
            'signal_history_count': len(self.signal_history),
            'max_signals_per_day': self.max_signals_per_day,
            'signal_cooldown_minutes': self.signal_cooldown_minutes,
            'last_reset_date': self.last_reset_date.isoformat() if self.last_reset_date else None
        }


# Utility functions for bot management
async def create_bot(config: TradingConfig) -> Optional[TelegramSignalBot]:
    """
    Create and initialize a Telegram bot instance.

    Args:
        config: Trading configuration with Telegram settings

    Returns:
        TelegramSignalBot instance if successful, None otherwise
    """
    if not config.notifications.telegram_bot_token:
        logger.warning("No Telegram bot token provided")
        return None

    bot = TelegramSignalBot(config)
    if await bot.initialize():
        return bot
    else:
        logger.error("Failed to initialize Telegram bot")
        return None


async def send_signal_to_telegram(signal: TradingSignal, config: TradingConfig, priority: str = "normal") -> bool:
    """
    Convenience function to send a signal via Telegram.

    Args:
        signal: Trading signal to send
        config: Trading configuration
        priority: Signal priority level

    Returns:
        True if sent successfully, False otherwise
    """
    bot = await create_bot(config)
    if not bot:
        return False

    try:
        result = await bot.send_trading_signal(signal, priority)
        await bot.cleanup()
        return result
    except Exception as e:
        logger.error(f"Error sending signal to Telegram: {e}")
        return False


async def send_market_update_to_telegram(
    update_text: str,
    config: TradingConfig,
    priority: str = "low"
) -> bool:
    """
    Convenience function to send market update via Telegram.

    Args:
        update_text: Update message text
        config: Trading configuration
        priority: Update priority level

    Returns:
        True if sent successfully, False otherwise
    """
    bot = await create_bot(config)
    if not bot:
        return False

    try:
        result = await bot.send_market_update(update_text, priority)
        await bot.cleanup()
        return result
    except Exception as e:
        logger.error(f"Error sending market update to Telegram: {e}")
        return False

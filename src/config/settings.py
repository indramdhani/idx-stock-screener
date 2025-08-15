# -*- coding: utf-8 -*-
"""
Core Configuration Settings for Indonesian Stock Screener
=========================================================

Pydantic-based configuration management with validation and environment variable support.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import yaml


class RiskManagementConfig(BaseModel):
    """Risk management configuration parameters"""

    max_risk_per_trade: float = Field(0.02, description="Maximum risk per trade (2%)")
    max_portfolio_risk: float = Field(0.06, description="Maximum total portfolio risk")
    default_stop_loss_atr: float = Field(2.0, description="Stop loss in ATR multiples")
    default_take_profit_ratio: float = Field(2.0, description="Risk-reward ratio")
    min_rr_ratio: float = Field(1.5, description="Minimum risk-reward ratio")

    @validator('max_risk_per_trade')
    def validate_max_risk_per_trade(cls, v):
        if not 0.005 <= v <= 0.1:  # 0.5% to 10%
            raise ValueError('max_risk_per_trade must be between 0.5% and 10%')
        return v

    @validator('max_portfolio_risk')
    def validate_max_portfolio_risk(cls, v):
        if not 0.02 <= v <= 0.3:  # 2% to 30%
            raise ValueError('max_portfolio_risk must be between 2% and 30%')
        return v


class ScreeningCriteria(BaseModel):
    """Stock screening criteria configuration"""

    min_volume: int = Field(1_000_000, description="Minimum daily volume")
    min_price: int = Field(1000, description="Minimum stock price (IDR)")
    max_price: int = Field(50000, description="Maximum stock price (IDR)")
    min_market_cap: Optional[int] = Field(None, description="Minimum market cap")
    exclude_sectors: List[str] = Field(default_factory=list)
    exclude_tickers: List[str] = Field(default_factory=list, description="Tickers to exclude")

    @validator('min_volume')
    def validate_min_volume(cls, v):
        if v < 100_000:
            raise ValueError('min_volume must be at least 100,000')
        return v


class TechnicalIndicators(BaseModel):
    """Technical indicator parameters"""

    rsi_oversold: int = Field(30, description="RSI oversold threshold")
    rsi_overbought: int = Field(70, description="RSI overbought threshold")
    rsi_period: int = Field(14, description="RSI calculation period")

    ema_periods: List[int] = Field([5, 13, 21], description="EMA periods")

    vwap_deviation_threshold: float = Field(0.02, description="VWAP deviation %")

    volume_spike_threshold: float = Field(1.5, description="Volume spike multiplier")
    volume_average_period: int = Field(10, description="Volume average period")

    atr_period: int = Field(14, description="ATR calculation period")

    @validator('rsi_oversold')
    def validate_rsi_oversold(cls, v):
        if not 10 <= v <= 40:
            raise ValueError('rsi_oversold must be between 10 and 40')
        return v

    @validator('rsi_overbought')
    def validate_rsi_overbought(cls, v):
        if not 60 <= v <= 90:
            raise ValueError('rsi_overbought must be between 60 and 90')
        return v


class NotificationConfig(BaseModel):
    """Notification system configuration"""

    telegram_bot_token: Optional[str] = Field(None, description="Telegram bot token")
    telegram_chat_ids: List[str] = Field(default_factory=list, description="Telegram chat IDs")

    max_signals_per_day: int = Field(10, description="Maximum signals per day")
    signal_cooldown_minutes: int = Field(30, description="Cooldown between signals for same ticker")
    include_charts: bool = Field(False, description="Include charts in notifications")

    @validator('telegram_chat_ids')
    def validate_chat_ids(cls, v):
        if v:
            for chat_id in v:
                if not chat_id.strip():
                    raise ValueError('Empty chat ID found')
        return v


class DataConfig(BaseModel):
    """Data collection configuration"""

    data_source: str = Field("yahoo", description="Primary data source")
    data_refresh_interval: int = Field(300, description="Data refresh interval in seconds")
    intraday_interval: str = Field("5m", description="Intraday data interval")
    historical_period: str = Field("30d", description="Historical data period")

    # Data quality parameters
    min_data_points: int = Field(20, description="Minimum data points required")
    max_data_age_hours: int = Field(24, description="Maximum age of data in hours")

    @validator('data_source')
    def validate_data_source(cls, v):
        allowed_sources = ['yahoo', 'idx', 'custom']
        if v not in allowed_sources:
            raise ValueError(f'data_source must be one of {allowed_sources}')
        return v


class SchedulingConfig(BaseModel):
    """Scheduling configuration"""

    intraday_screening_cron: str = Field("*/15 9-15 * * 1-5", description="Intraday screening schedule")
    overnight_screening_cron: str = Field("0 17 * * 1-5", description="Overnight screening schedule")
    risk_review_cron: str = Field("0 8 * * 1-5", description="Risk review schedule")

    market_open_hour: int = Field(9, description="Market opening hour (24h format)")
    market_close_hour: int = Field(15, description="Market closing hour (24h format)")

    timezone: str = Field("Asia/Jakarta", description="Timezone for scheduling")


class TradingConfig(BaseModel):
    """Main trading configuration"""

    # Core configurations
    risk_management: RiskManagementConfig = RiskManagementConfig()
    screening_criteria: ScreeningCriteria = ScreeningCriteria()
    indicators: TechnicalIndicators = TechnicalIndicators()
    notifications: NotificationConfig = NotificationConfig()
    data: DataConfig = DataConfig()
    scheduling: SchedulingConfig = SchedulingConfig()

    # Default tickers to screen
    default_tickers: List[str] = Field(
        default_factory=lambda: [
            "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK",
            "AMRT.JK", "INDF.JK", "UNVR.JK", "PGAS.JK", "ADRO.JK",
            "ASII.JK", "GGRM.JK", "ICBP.JK", "KLBF.JK", "INTP.JK",
            "SMGR.JK", "JSMR.JK", "UNTR.JK", "SIDO.JK", "MNCN.JK"
        ],
        description="Default list of tickers to screen"
    )

    # Capital configuration
    default_capital_idr: int = Field(100_000_000, description="Default capital in IDR")

    # Feature flags
    enable_vwap_filter: bool = Field(False, description="Enable VWAP filtering")
    enable_atr_tp_sl: bool = Field(False, description="Enable ATR-based TP/SL")
    enable_telegram_notifications: bool = Field(True, description="Enable Telegram notifications")

    @validator('default_capital_idr')
    def validate_default_capital(cls, v):
        if v < 1_000_000:  # Minimum 1M IDR
            raise ValueError('default_capital_idr must be at least 1,000,000 IDR')
        return v

    def load_from_yaml(self, config_path: Path) -> 'TradingConfig':
        """Load configuration from YAML file"""
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            return self.__class__(**config_data)
        return self

    def load_from_env(self) -> 'TradingConfig':
        """Load sensitive configuration from environment variables"""
        env_updates = {}

        # Load Telegram configuration from environment
        if telegram_token := os.getenv('TELEGRAM_BOT_TOKEN'):
            env_updates.setdefault('notifications', {})['telegram_bot_token'] = telegram_token

        if chat_ids := os.getenv('TELEGRAM_CHAT_IDS'):
            env_updates.setdefault('notifications', {})['telegram_chat_ids'] = chat_ids.split(',')

        # Load capital from environment
        if capital := os.getenv('DEFAULT_CAPITAL_IDR'):
            try:
                env_updates['default_capital_idr'] = int(capital)
            except ValueError:
                pass  # Keep default value

        # Update configuration if we have environment overrides
        if env_updates:
            current_dict = self.dict()
            for key, value in env_updates.items():
                if key in current_dict:
                    if isinstance(current_dict[key], dict) and isinstance(value, dict):
                        current_dict[key].update(value)
                    else:
                        current_dict[key] = value

            return self.__class__(**current_dict)

        return self

    def save_to_yaml(self, config_path: Path) -> None:
        """Save configuration to YAML file (excluding sensitive data)"""
        config_dict = self.dict()

        # Remove sensitive information
        if 'notifications' in config_dict:
            config_dict['notifications'].pop('telegram_bot_token', None)
            config_dict['notifications'].pop('telegram_chat_ids', None)

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


def load_config(config_path: Optional[Path] = None) -> TradingConfig:
    """
    Load configuration from YAML file and environment variables.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        TradingConfig: Loaded configuration
    """
    # Start with default configuration
    config = TradingConfig()

    # Load from YAML if provided
    if config_path and config_path.exists():
        config = config.load_from_yaml(config_path)

    # Override with environment variables
    config = config.load_from_env()

    return config

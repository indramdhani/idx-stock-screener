# Indonesian Stock Screener Implementation Plan

## 1. PROJECT OVERVIEW

**Brief Description:**
An automated Indonesian stock screening system that identifies intraday and overnight rebound opportunities using technical indicators, calculates risk parameters, and delivers trading signals via Telegram.

**Key Objectives:**
- Automate daily stock screening for IDX (Indonesian Stock Exchange)
- Implement risk management with configurable parameters
- Provide actionable trading signals with entry/exit points
- Deliver real-time notifications through Telegram
- Support both intraday and overnight trading strategies

## 2. COMPONENTS & PAGES

### Core Components:
- **Data Collector**: Fetches real-time/historical stock data from IDX
- **Technical Analyzer**: Calculates VWAP, ATR, EMA, RSI, Support/Resistance
- **Stock Screener**: Filters stocks based on technical criteria
- **Risk Calculator**: Computes position sizing, stop loss, take profit
- **Signal Generator**: Creates formatted trading signals
- **Telegram Bot**: Delivers notifications to users
- **Scheduler**: Manages automated execution timing

### System Architecture:
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Scheduler     │───▶│  Data Collector  │───▶│ Technical       │
│  (Cron/Action)  │    │   (IDX API)      │    │ Analyzer        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Telegram Bot   │◀───│ Signal Generator │◀───│ Stock Screener  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Risk Calculator  │◀───│ Position Sizer  │
                       └──────────────────┘    └─────────────────┘
```

## 3. PROJECT STRUCTURE

```
indonesian-stock-screener/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   └── trading_config.yaml
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collectors/
│   │   │   ├── __init__.py
│   │   │   ├── idx_collector.py
│   │   │   └── data_validator.py
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── stock.py
│   │       └── signal.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── indicators/
│   │   │   ├── __init__.py
│   │   │   ├── vwap.py
│   │   │   ├── atr.py
│   │   │   ├── ema.py
│   │   │   ├── rsi.py
│   │   │   └── support_resistance.py
│   │   ├── screener.py
│   │   └── risk_calculator.py
│   ├── notifications/
│   │   ├── __init__.py
│   │   ├── telegram_bot.py
│   │   └── signal_formatter.py
│   ├── scheduler/
│   │   ├── __init__.py
│   │   ├── job_scheduler.py
│   │   └── workflows.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── helpers.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── scripts/
│   ├── setup.py
│   ├── run_screener.py
│   └── deploy.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       ├── scheduled-screening.yml
│       └── deploy.yml
├── requirements.txt
├── README.md
└── main.py
```

## 4. TOOLS & TECHNOLOGIES

### Backend Framework & Core Libraries:
```python
# Core Dependencies
pandas>=2.0.0           # Data manipulation
numpy>=1.24.0           # Numerical computing
yfinance>=0.2.18        # Stock data (fallback)
requests>=2.31.0        # HTTP requests
python-telegram-bot>=20.0  # Telegram integration

# Technical Analysis
talib>=0.4.25           # Technical indicators
ta>=0.10.2              # Alternative TA library

# Scheduling & Workflow
schedule>=1.2.0         # Simple scheduling
croniter>=1.4.0         # Cron expression parsing

# Configuration & Logging
pydantic>=2.0.0         # Config validation
PyYAML>=6.0             # YAML config files
loguru>=0.7.0           # Enhanced logging

# Data Storage (Optional)
sqlite3                 # Lightweight database
redis>=4.5.0            # Caching (if needed)
```

### IDX Data Sources:
- **Primary**: Yahoo Finance Indonesia
- **Alternative**: IDX official API or RTI (Real Time Information)
- **Backup**: Web scraping with BeautifulSoup4

### Hosting Options Analysis:

**AWS Lambda + CloudWatch (Recommended for MVP):**
- ✅ Serverless, pay-per-execution
- ✅ Reliable scheduling with CloudWatch Events
- ✅ Scalable and managed infrastructure
- ❌ 15-minute execution limit
- ❌ Setup complexity

**GitHub Actions :**
- ✅ Free tier: 2000 minutes/month
- ✅ Built-in scheduling with cron
- ✅ Easy deployment and version control
- ❌ Limited to 6-hour maximum job duration
- ❌ No persistent storage

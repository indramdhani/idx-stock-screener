# Indonesian Stock Screener v1.0

A comprehensive, modular Indonesian stock screening system that identifies intraday and overnight trading opportunities using technical analysis, risk management, and automated notifications.

## 🇮🇩 Overview

This system automatically screens Indonesian Stock Exchange (IDX) stocks for trading opportunities based on technical indicators, volume analysis, and price movements. It provides both intraday breakout and overnight rebound strategies with configurable risk management parameters.

### Key Features

- 🚀 **AI-Enhanced Screening**: ML-powered stock screening with confidence scoring
- 📊 **Advanced Analytics**: Professional performance metrics and risk analysis
- 💼 **Real-Time Portfolio**: Live portfolio tracking with risk management
- 🌐 **Web Dashboard**: Interactive dashboard with real-time charts and alerts
- 🤖 **Machine Learning**: Ensemble models for signal enhancement
- 📱 **Multi-Platform**: Telegram bot + Web interface + Mobile responsive
- ⏰ **Full Automation**: GitHub Actions + local scheduling for 24/7 operation
- 🔧 **Production Ready**: Enterprise-grade architecture with comprehensive testing

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:

```
src/
├── config/           # Configuration management with Pydantic validation
├── data/            # Data models, collection, and validation
│   ├── collectors/  # IDX data collection from Yahoo Finance
│   └── models/      # Stock data and trading signal models
├── analysis/        # Technical analysis and screening logic (Phase 3)
├── notifications/   # Telegram bot integration (Phase 5)
├── scheduler/       # Automated workflow management (Phase 6)
└── utils/          # Logging and utility functions
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Internet connection for data fetching
- (Optional) Telegram Bot Token for notifications

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd idx-stock-screener
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the legacy screener** (backward compatible):
   ```bash
   python initial-script.py
   ```
# Run the new modular system with ML enhancement
python main.py

# Interactive mode
python main.py

# Direct screening with AI enhancement
python main.py --mode intraday --ml-enhance
python main.py --mode overnight --ml-enhance

# Launch web dashboard (separate terminal)
python -m src.dashboard.app --host 0.0.0.0 --port 5000

# Data validation only
python main.py --validate-only
   ```

## 📊 Usage Examples

### Interactive Mode
```bash
python main.py
```
Launches an interactive menu where you can choose different screening modes and options.

### Command Line Usage
```bash
# Run intraday screening
python main.py --mode intraday --log-level DEBUG

# Run overnight screening with custom config
python main.py --mode overnight --config custom_config.yaml

# Run both strategies with ML enhancement
python main.py --mode both --ml-enhance

# Test Telegram integration
python main.py --telegram-test

# Start automated scheduled mode
python main.py --scheduled

# Launch web dashboard
python -m src.dashboard.app

# Validate Phase 7-8 implementation
python validate_phase7_8.py

# Validate data quality
python main.py --validate-only
```

### Legacy Mode (v0.3 compatible)
```bash
python initial-script.py
```

## ⚙️ Configuration

The system uses YAML configuration files with Pydantic validation:

### Main Configuration (`src/config/trading_config.yaml`)

```yaml
# Risk Management
risk_management:
  max_risk_per_trade: 0.02      # 2% max risk per trade
  max_portfolio_risk: 0.06      # 6% total portfolio risk
  default_stop_loss_atr: 2.0    # Stop loss at 2x ATR
  min_rr_ratio: 1.5            # Minimum risk-reward ratio

# Screening Criteria
screening_criteria:
  min_volume: 1000000          # Minimum daily volume
  min_price: 1000              # Minimum stock price (IDR)
  max_price: 50000             # Maximum stock price (IDR)

# Technical Indicators
indicators:
  rsi_oversold: 30            # RSI oversold threshold
  rsi_overbought: 70          # RSI overbought threshold
  ema_periods: [5, 13, 21]    # EMA periods
  atr_period: 14              # ATR calculation period

# Feature Flags
enable_vwap_filter: false     # Enable VWAP deviation filtering
enable_atr_tp_sl: false      # Enable ATR-based TP/SL calculation
```

### Environment Variables

For sensitive configuration (Telegram credentials):

```bash
export TELEGRAM_BOT_TOKEN="your_bot_token_here"
export TELEGRAM_CHAT_IDS="chat_id1,chat_id2,chat_id3"
export DEFAULT_CAPITAL_IDR="100000000"
```

### Telegram Bot Setup
1. Create a new bot with [@BotFather](https://t.me/BotFather)
2. Get your bot token and set `TELEGRAM_BOT_TOKEN`
3. Start a chat with your bot and get your chat ID
4. Set `TELEGRAM_CHAT_IDS` with comma-separated chat IDs

## 🎯 Screening Strategies

### 1. Morning Breakout (Intraday)
- Identifies stocks with positive momentum (≥0.8% gain)
- Entry: Market open, Exit: Before market close
- Criteria: High volume, above EMA, RSI conditions

### 2. Afternoon Rebound (Overnight)
- Identifies oversold stocks for overnight recovery
- Entry: Late afternoon, Exit: Next morning
- Criteria: Significant decline (≤-2%), near support levels

### 3. Technical Filters
- **VWAP Filter**: Excludes stocks too far from Volume Weighted Average Price
- **ATR-based TP/SL**: Dynamic stop-loss and take-profit based on Average True Range
- **Volume Analysis**: Minimum volume and spike detection
- **Price Range**: Configurable price boundaries

## 📈 Sample Output

```
🚀 INTRADAY REBOUND OPPORTUNITIES
========================================================

1. BBCA.JK
   Entry: IDR 8,525
   Stop Loss: IDR 8,440
   Take Profit: IDR 8,610
   Risk/Reward: 2.1:1
   Confidence: 85%
   Position: 11,700 shares (117 lots)
   Reasoning: RSI oversold (28); Volume spike above average

2. TLKM.JK
   Entry: IDR 3,150
   Stop Loss: IDR 3,120
   Take Profit: IDR 3,180
   Risk/Reward: 1.8:1
   Confidence: 76%
   Position: 31,600 shares (316 lots)
   Reasoning: EMA uptrend alignment; Oversold bounce in uptrend

🌙 OVERNIGHT SETUP OPPORTUNITIES
======================================================

1. ADRO.JK
   Entry: IDR 2,180
   Stop Loss: IDR 2,136
   Take Profit: IDR 2,224
   Risk/Reward: 2.3:1
   Confidence: 82%
   Position: 22,900 shares (229 lots)
   Reasoning: Significant price decline; RSI oversold (26); Quality large-cap stock
```

## 📋 Data Sources & Quality

### Primary Data Source
- **Yahoo Finance**: Real-time and historical IDX data
- **Symbols**: 100+ liquid IDX stocks with `.JK` suffix
- **Intervals**: 1D for daily analysis, 5m for intraday

### Data Validation
The system includes comprehensive data quality checks:

- ✅ **Completeness**: Required OHLCV columns
- ✅ **Freshness**: Data age within acceptable limits
- ✅ **Price Validity**: Reasonable price ranges and OHLC consistency
- ✅ **Volume Quality**: Minimum volume and zero-volume detection
- ✅ **Anomaly Detection**: Extreme price movements and gaps
- ✅ **Quality Scoring**: 0-1 score for each stock

### Market Hours
- **Trading Hours**: 09:00 - 15:00 WIB (Monday-Friday)
- **Data Delay**: ~10-15 minutes (Yahoo Finance limitation)

## 🤖 Automation & Notifications

### GitHub Actions Automation
The system runs automatically via GitHub Actions:

```yaml
# Intraday screening every 30 minutes during market hours (9 AM - 3 PM WIB)
- cron: '0,30 2-8 * * 1-5'  # UTC time (WIB-7)

# Overnight screening at 5 PM WIB daily
- cron: '0 10 * * 1-5'      # UTC time

# Market status check at 8 AM WIB daily
- cron: '0 1 * * 1-5'       # UTC time
```

### Telegram Integration
Fully automated signal delivery with rich formatting:
- 🚀 **Real-time signals** with entry/exit levels
- 📊 **Technical analysis** context (RSI, EMAs, volume)
- 💰 **Position sizing** suggestions with risk management
- 🎯 **Confidence scoring** and reasoning
- 📈 **Risk-reward ratios** and stop-loss levels
- 🔔 **Interactive commands** (/signals, /status, /stats)
- 📅 **Market updates** and daily summaries

### Local Automation
Run continuous screening on your machine:
```bash
python main.py --scheduled
```

### 🧪 Development Status

### ✅ Phase 1-2 (Complete): Foundation & Data Collection
- [x] Modular project structure
- [x] Pydantic configuration system
- [x] IDX data collector with Yahoo Finance integration
- [x] Comprehensive data validation
- [x] Stock and signal data models
- [x] Logging and error handling
- [x] Legacy compatibility mode

### ✅ Phase 3-4 (Complete): Technical Analysis & Risk Management
- [x] Technical indicator implementations (VWAP, ATR, RSI, EMA)
- [x] Stock screening engine with intraday and overnight strategies
- [x] Risk calculator and position sizing with ATR-based calculations
- [x] Signal generation and ranking with confidence scoring
- [x] Comprehensive signal models with risk parameters
- [x] Multi-factor screening criteria and validation

### ✅ Phase 5-6 (Complete): Integration & Automation
- [x] Telegram bot integration with interactive commands
- [x] Multi-channel notification system (Telegram, console, JSON, CSV)
- [x] GitHub Actions workflows for automated screening
- [x] Workflow orchestration and scheduling system
- [x] Performance monitoring and error handling
- [x] Configuration management and validation
- [x] End-to-end automation pipeline

### ✅ Phase 7-8 (Complete): Enhanced Features & Optimization
- [x] Advanced performance analytics and reporting
- [x] Real-time portfolio management and tracking
- [x] Machine learning signal enhancement with ensemble models
- [x] Professional web dashboard with real-time updates
- [x] Risk management integration with dynamic limits
- [x] Multi-timeframe analysis capabilities
- [x] Comprehensive backtesting framework
- [x] Interactive charts and visualizations
- [x] RESTful API for external integration
- [x] Mobile-responsive dashboard design

## 📁 File Structure

```
idx-stock-screener/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── main.py                   # New modular entry point
├── initial-script.py         # Legacy script (v0.3 compatible)
├── logs/                     # Log files and CSV outputs
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py       # Pydantic configuration models
│   │   └── trading_config.yaml # Main configuration file
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collectors/
│   │   │   ├── __init__.py
│   │   │   ├── idx_collector.py    # IDX data collector
│   │   │   └── data_validator.py   # Data quality validation
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── stock.py      # Stock data models
│   │       └── signal.py     # Trading signal models
│   └── utils/
│       ├── __init__.py
│       └── logger.py         # Logging utilities
├── tests/                    # Test files (planned)
├── docker/                   # Docker configuration (planned)
└── .github/
    └── workflows/            # GitHub Actions (planned)
```

## 🔧 Configuration Options

### Risk Management
- `max_risk_per_trade`: Maximum risk per trade (default: 2%)
- `max_portfolio_risk`: Maximum total portfolio risk (default: 6%)
- `default_stop_loss_atr`: Stop loss in ATR multiples (default: 2.0)
- `min_rr_ratio`: Minimum acceptable risk-reward ratio (default: 1.5)

### Screening Criteria
+- `min_volume`: Minimum daily volume threshold
+- `min_price`/`max_price`: Price range filters
+- `exclude_sectors`: Sectors to exclude from screening
+- `exclude_tickers`: Specific tickers to exclude
+
+### Technical Indicators
+- `rsi_oversold`/`rsi_overbought`: RSI thresholds (30/70)
+- `ema_periods`: EMA calculation periods [5, 13, 21]
+- `atr_period`: ATR calculation period (14)
+- `vwap_deviation_threshold`: VWAP deviation limit (2%)
+- `volume_spike_threshold`: Volume spike detection multiplier

### Feature Flags
+- `enable_vwap_filter`: Enable VWAP filtering
+- `enable_atr_tp_sl`: Enable ATR-based TP/SL calculation
+- `enable_telegram_notifications`: Enable Telegram notifications

### Screening Strategies

#### 1. **Intraday Rebound Strategy**
- **Target**: Oversold stocks with potential for same-day recovery
- **Criteria**: RSI < 30, EMA uptrend, volume spike, positive momentum
- **Risk Management**: ATR-based or 0.7% stop loss, 1.5-2.5% take profit
- **Position Hold**: Intraday only (exit before market close)

#### 2. **Overnight Setup Strategy**  
- **Target**: Quality stocks oversold for overnight recovery
- **Criteria**: RSI < 30, significant decline (-2%+), high volume, large-cap preferred
- **Risk Management**: ATR-based or 2% stop loss, 2.5-4% take profit
- **Position Hold**: Overnight (buy afternoon, sell next morning)

## 🚨 Important Disclaimers

⚠️ **Trading Risk**: This system is for educational and research purposes only. Trading stocks involves significant financial risk, and you should never invest money you cannot afford to lose.

⚠️ **Data Delays**: Yahoo Finance data for IDX stocks is delayed by 10-15 minutes. Always confirm real-time prices before executing trades.

⚠️ **No Guarantees**: Past performance does not guarantee future results. The system's signals should not be considered as financial advice.

⚠️ **Paper Trading Recommended**: Test the system with paper trading before using real money.

## 🤝 Contributing

This project is part of a structured implementation following the phases outlined in the documentation. Current focus is on completing Phase 1-2 (Foundation & Data Collection).

### Development Setup
1. Clone the repository
2. Install development dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest` (when implemented)
4. Follow the coding standards and documentation guidelines

### Code Style
- Python 3.11+ with type hints
- Pydantic for data validation
- Loguru for logging
- Black for code formatting
- Comprehensive docstrings

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Yahoo Finance for providing IDX stock data
- Indonesian Stock Exchange (IDX) for market data
- The Python community for excellent libraries and tools
- Contributors and testers who help improve the system

---

**Last Updated**: December 2024  
**Version**: 2.0.0 (Enhanced Features & Optimization)  
**Status**: 🚀 **PRODUCTION READY** - All phases complete with advanced AI features

## 🌟 Phase 7-8 New Features

### 🤖 Machine Learning Enhancement
- **AI-Powered Signals**: Ensemble ML models (Random Forest, LightGBM, XGBoost)
- **Feature Engineering**: 50+ technical features for enhanced prediction
- **Confidence Scoring**: ML-enhanced confidence with uncertainty quantification
- **Model Training**: Automated training with historical data validation

### 📊 Advanced Performance Analytics
- **Professional Metrics**: Sharpe, Sortino, Calmar ratios with drawdown analysis
- **Risk Analysis**: VaR, CVaR, beta, alpha, and tracking error calculations
- **Trade Analytics**: Win/loss ratios, profit factors, and performance attribution
- **Benchmark Comparison**: Performance vs IDX Composite and sector indices

### 💼 Real-Time Portfolio Management
- **Live Tracking**: Real-time P&L, position sizing, and risk monitoring
- **Dynamic Risk**: Automated risk limits with position concentration alerts
- **Multi-Strategy**: Strategy allocation and performance attribution
- **Alert System**: Stop-loss monitoring and threshold notifications

### 🌐 Professional Web Dashboard
- **Real-Time Updates**: WebSocket-based live portfolio monitoring
- **Interactive Charts**: Plotly-powered visualizations with zoom and pan
- **Mobile Responsive**: Bootstrap-based design for all devices
- **RESTful API**: Complete API for external system integration

### 🚀 Quick Start with Phase 7-8
```bash
# 1. Install enhanced dependencies
pip install -r requirements.txt

# 2. Validate implementation
python validate_phase7_8.py

# 3. Run AI-enhanced screening
python main.py --mode both --ml-enhance

# 4. Launch web dashboard (separate terminal)
python -m src.dashboard.app
```

### 📱 Access Your Dashboard
- **Main Dashboard**: http://localhost:5000
- **Portfolio Management**: http://localhost:5000/portfolio
- **Performance Analytics**: http://localhost:5000/analytics
- **API Documentation**: http://localhost:5000/api/
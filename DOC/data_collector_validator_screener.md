# IDX Data Collector, Validator, and Screener Workflow

## Overview

This document explains how the Indonesian Stock Screener system collects, validates, and analyzes stock data from the IDX (Indonesian Stock Exchange). It covers the three core modules:

- **IDX Data Collector**: Fetches and prepares raw stock data.
- **Data Validator**: Ensures data quality and integrity for analysis.
- **Stock Screener**: Applies technical and strategic criteria to generate actionable trading signals.

---

## 1. IDX Data Collector

### Purpose

The IDX Data Collector is responsible for gathering real-time and historical stock data from external sources (primarily Yahoo Finance, with support for other APIs). It prepares the data for downstream validation and analysis.

### Key Functions

- **Symbol Management**: Loads and manages the list of IDX stock symbols (e.g., `BBCA.JK`, `TLKM.JK`).
- **Data Fetching**:
  - **Daily Data**: Retrieves OHLCV (Open, High, Low, Close, Volume) data for the last N days (typically 30).
  - **Intraday Data**: Fetches minute-by-minute or hourly data for the current trading day.
  - **Stock Info**: Collects metadata such as company name, sector, market cap, and shares outstanding.
- **Session Management**: Uses persistent HTTP sessions for efficient data requests.
- **Error Handling**: Tracks failed symbols and implements retry logic.

### Example Workflow

1. Load symbols from configuration.
2. For each symbol:
   - Fetch daily and intraday data.
   - Collect stock info.
   - Package into a `StockData` object.
3. Return a dictionary of `{symbol: StockData}` for validation.

---

## 2. Data Validator

### Purpose

The Data Validator ensures that the collected stock data is complete, fresh, and suitable for technical analysis. It applies a series of rules to detect anomalies, missing values, and other quality issues.

### Validation Rules

- **Data Completeness**: Checks for required columns (`Open`, `High`, `Low`, `Close`, `Volume`) and sufficient history.
- **Data Freshness**: Ensures the latest data point is recent (e.g., within the last 24-48 hours).
- **Price Validity**: Detects unrealistic prices (e.g., negative or zero values, out-of-range prices).
- **Volume Validity**: Flags stocks with low or zero trading volume.
- **Price Anomalies**: Identifies extreme price changes or consecutive limit moves.
- **Gap Detection**: Finds significant price gaps between trading sessions.
- **Missing Data**: Calculates the percentage of missing values.
- **Quality Score**: Aggregates rule results into a normalized score (0-1).

### Workflow

1. For each `StockData` object:
   - Apply all validation rules.
   - Record errors, warnings, and info messages.
   - Cache results for efficiency.
2. Filter out stocks that do not meet minimum quality thresholds.
3. Provide summary statistics (e.g., valid/invalid counts, average quality score).

---

## 3. Stock Screener

### Purpose

The Stock Screener analyzes validated stock data to identify trading opportunities based on technical indicators and strategic criteria. It generates actionable signals for intraday and overnight trading.

### Screening Logic

- **Technical Indicators**: Calculates RSI, EMA, ATR, VWAP, and other metrics.
- **Strategic Filters**:
  - **Intraday Rebound**: Looks for oversold stocks with bullish momentum and volume spikes.
  - **Overnight Setup**: Identifies quality stocks with oversold conditions and potential for gap-up recovery.
- **Signal Generation**:
  - Evaluates each stock against screening criteria.
  - Calculates entry price, stop loss, take profit, and risk/reward ratio.
  - Assigns a confidence score based on how many conditions are met.
- **Ranking & Selection**: Sorts signals by confidence and selects top candidates for notification.

### Example Workflow

1. Receive validated stocks from the Data Validator.
2. For each stock:
   - Calculate technical indicators.
   - Evaluate against strategy-specific conditions.
   - If criteria are met, generate a `TradingSignal`.
3. Rank signals and deliver top picks via Telegram or dashboard.

---

## Integration & Flow

The modules work together in the following sequence:

```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Data Collector│───▶│ Data Validator│───▶│ Stock Screener│
└───────────────┘    └───────────────┘    └───────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
  Raw Stock Data      Validated Stock Data   Trading Signals
```

- **Data Collector**: Gathers and structures raw data.
- **Validator**: Filters out poor-quality stocks.
- **Screener**: Applies trading logic and generates signals.

---

## Troubleshooting & Best Practices

- **No Signals Generated**: If the screener produces no signals, review logs to see which validation or screening conditions are failing. Consider relaxing thresholds or testing with historical data.
- **Data Issues**: Ensure data sources are reliable and up-to-date. Missing or stale data will prevent signal generation.
- **Configuration**: Regularly review and update screening criteria and validation rules to match current market conditions.

---

## References

- [PHASE7_8_FEATURES.md](./PHASE7_8_FEATURES.md) — Advanced features and configuration options.
- [implementation-strategy.md](./implementation-strategy.md) — Project structure and workflow details.
- [rule.md](./rule.md) — System architecture and component overview.

---

*For further details, consult the source code in `src/data/collectors/`, `src/data/models/`, and `src/analysis/`.*

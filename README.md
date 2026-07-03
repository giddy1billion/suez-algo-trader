# Alpaca Algo Trader 🤖📈

A fully robust, scalable algorithmic trading bot with AI/ML integration.  
Uses **Alpaca Markets API** for commission-free trading (US stocks + crypto).

## Features

- ✅ **Paper + Live trading** — switch with one env var
- ✅ **3 built-in strategies**: Momentum, Mean Reversion, ML (XGBoost)
- ✅ **Risk Management** — position sizing, daily loss limits, exposure caps
- ✅ **Backtesting engine** — test strategies on historical data before going live
- ✅ **ML/AI integration** — XGBoost with 30+ engineered features, auto-retraining
- ✅ **Real-time WebSocket streaming** — sub-second data for stocks + crypto
- ✅ **Notifications** — Telegram + Discord alerts on every trade
- ✅ **SQLite persistence** — full trade history, signals, portfolio snapshots
- ✅ **Emergency controls** — panic liquidation, circuit breakers
- ✅ **Docker ready** — deploy anywhere

## Quick Start

### 1. Setup

```bash
cd algo-trader
source venv/Scripts/activate   # Windows Git Bash
# OR
.\venv\Scripts\activate        # Windows PowerShell

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your Alpaca API keys
```

Get free API keys: https://app.alpaca.markets/signup

### 3. Run

```bash
# Paper trading (safe — no real money)
python main.py

# Check account status
python main.py --status

# Backtest before going live
python main.py --backtest --strategy momentum

# Dry run (signals only, no orders)
python main.py --dry-run

# Train ML model
python main.py --train --strategy ml

# Live trading (⚠️ real money!)
python main.py --live
```

## Strategies

### Momentum
Multi-indicator trend following:
- EMA crossover (12/26) for direction
- RSI (14) for overbought/oversold
- MACD histogram for momentum
- Volume spike confirmation
- ATR-based dynamic stop-loss/take-profit

### Mean Reversion
Statistical mean reversion:
- Bollinger Bands for range boundaries
- Z-score for deviation measurement
- RSI for confirmation
- Targets SMA as exit point

### ML (XGBoost)
Machine learning prediction:
- 30+ engineered features
- XGBoost multi-class classifier (UP/DOWN/FLAT)
- Time-series cross-validation
- Auto-retrain on schedule
- Confidence threshold filtering

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   main.py                        │
├──────────┬──────────┬───────────┬───────────────┤
│ Strategy │Execution │  Risk     │ Notifications │
│ Engine   │ Engine   │ Manager   │               │
├──────────┼──────────┼───────────┼───────────────┤
│ Momentum │ Orders   │ Position  │ Telegram      │
│ MeanRev  │ Brackets │ Sizing    │ Discord       │
│ ML/XGB   │ Trailing │ Drawdown  │ Console       │
└────┬─────┴────┬─────┴─────┬────┴───────┬───────┘
     │          │           │            │
     ▼          ▼           ▼            ▼
┌──────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│ Alpaca   │ │ SQLite │ │ Models │ │  Logs    │
│ REST+WS  │ │  DB    │ │.joblib │ │          │
└──────────┘ └────────┘ └────────┘ └──────────┘
```

## Risk Management

All trades pass through the risk manager before execution:

| Rule | Default | Description |
|------|---------|-------------|
| Max risk per trade | 2% | Maximum portfolio % at risk per trade |
| Daily loss limit | 5% | Stop trading if down this much today |
| Max portfolio exposure | 80% | Never invest more than this % |
| Max single stock | 15% | No single position larger than this |
| Default stop-loss | 3% | Automatic stop-loss on every trade |
| Default take-profit | 6% | 2:1 reward-to-risk ratio |

## CLI Options

```
python main.py --help

Options:
  --live              ⚠️ Enable LIVE trading
  --strategy, -s      Strategy: momentum, mean_reversion, ml
  --symbols           Comma-separated (AAPL,TSLA,BTC/USD)
  --timeframe, -tf    1Min, 5Min, 15Min, 30Min, 1Hour, 4Hour, 1Day
  --lookback          Historical bars to analyze (default: 200)
  --interval, -i      Seconds between cycles (default: 60)
  --dry-run           Signals only, no order execution
  --backtest          Run backtest on historical data
  --train             Train/retrain ML model
  --status            Show account status
```

## Project Structure

```
algo-trader/
├── main.py                  # Entry point & CLI
├── config/
│   └── settings.py          # Pydantic configuration
├── src/
│   ├── broker/
│   │   └── alpaca_client.py # REST + WebSocket client
│   ├── strategy/
│   │   ├── base.py          # Abstract strategy interface
│   │   ├── momentum.py      # Momentum strategy
│   │   ├── mean_reversion.py# Mean reversion strategy
│   │   └── ml_strategy.py   # ML/XGBoost strategy
│   ├── ml/
│   │   └── features.py      # Feature engineering pipeline
│   ├── risk/
│   │   └── manager.py       # Risk management engine
│   ├── execution/
│   │   └── engine.py        # Trade execution orchestrator
│   ├── data/
│   │   └── store.py         # SQLAlchemy models + DB manager
│   ├── notifications/
│   │   └── alerts.py        # Telegram + Discord notifications
│   └── utils/
│       └── logger.py        # Structured logging
├── backtesting/
│   └── backtest.py          # Backtesting engine
├── models/                  # Saved ML models
├── data_cache/              # SQLite DB + cached data
├── logs/                    # Application logs
├── tests/                   # Test suite
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
└── README.md
```

## Alpaca Account Setup

1. **Sign up** at https://app.alpaca.markets/signup (free)
2. **Paper account** is auto-created with $100,000 virtual cash
3. Go to **API Keys** → Generate new key
4. Copy `API Key` and `Secret Key` to your `.env` file
5. For live: fund account + generate live API keys separately

## Safety Notes

- **Always start with paper trading** — validate your strategy first
- **Backtest thoroughly** before any live trading
- **Never risk more than you can afford to lose**
- The bot has built-in daily loss limits and circuit breakers
- Live mode requires typing "YES" as confirmation
- All trades are logged for audit trail

# Suez Algo Trader 🤖📈

[![Build & Deploy](https://github.com/giddy1billion/suez-algo-trader/actions/workflows/deploy.yml/badge.svg)](https://github.com/giddy1billion/suez-algo-trader/actions/workflows/deploy.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Private-red.svg)]()

An institutional-grade, event-driven algorithmic trading platform with adaptive AI/ML intelligence.  
Uses **Alpaca Markets API** for commission-free trading (US equities + crypto).  
Fully manageable via **Telegram** — monitor, trade, configure, backtest, and train ML models from your phone.  
Deployed on **Azure Container Instances** via GitHub Actions CI/CD pipeline.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Telegram Bot — Full Operations Guide](#telegram-bot--full-operations-guide)
  - [Setup](#telegram-setup)
  - [Authentication & Security](#authentication--security)
  - [Information Commands](#information-commands)
  - [Trading Commands](#trading-commands)
  - [Bot Control Commands](#bot-control-commands)
  - [Configuration Commands](#configuration-commands)
  - [Backtesting & ML Commands](#backtesting--ml-commands)
  - [Advanced Research Commands](#advanced-research-commands)
  - [System & Monitoring Commands](#system--monitoring-commands)
  - [Sector Management Commands](#sector-management-commands)
  - [Model Governance Commands](#model-governance-commands)
  - [Automated Notifications](#automated-notifications)
- [Strategies](#strategies)
- [Architecture](#architecture)
- [Adaptive Intelligence Layer](#adaptive-intelligence-layer)
- [Risk Management](#risk-management)
- [Configuration System](#configuration-system)
- [CLI Options](#cli-options)
- [Project Structure](#project-structure)
- [Environment Configuration](#environment-configuration)
- [Alpaca Account Setup](#alpaca-account-setup)
- [Deployment](#deployment)
- [Testing](#testing)
- [Safety Notes](#safety-notes)

---

## Features

### Core Trading
- ✅ **Paper + Live trading** — switch with one env var, live mode requires explicit CLI confirmation
- ✅ **Multi-strategy orchestrator** — run multiple strategies concurrently with independent schedules and capital weights
- ✅ **5 built-in strategies**: Momentum, Mean Reversion, ML (XGBoost), Composable, Composable MR
- ✅ **Asset-class-aware execution** — separate handling for equities (market hours) and crypto (24/7)
- ✅ **Real-time WebSocket streaming** — sub-second bar data + live trade update stream from Alpaca

### Adaptive Intelligence
- ✅ **Adaptive Intelligence Layer** — regime classification, trade quality scoring, drift detection, strategy routing
- ✅ **Market State Engine** — full market fingerprint with multi-dimensional regime analysis
- ✅ **Meta-Strategy Engine** — dynamic strategy ranking and selection per market regime
- ✅ **Concept Drift Monitor** — detect model degradation in real-time, trigger retraining
- ✅ **Capital Allocator** — dynamic position sizing via Kelly criterion with sector correlation filters
- ✅ **Decision Explainer** — human-readable reasons for every trade accept/reject
- ✅ **Decision Journal** — full audit trail for all intelligence decisions
- ✅ **Counterfactual Engine** — what-if analysis on alternative trade decisions

### ML/AI
- ✅ **XGBoost ML strategy** — 30+ engineered features, time-series cross-validation
- ✅ **Model governance** — version registry, lineage tracking (git hash, config hash, dataset hash), rollback
- ✅ **A/B testing framework** — shadow mode and split-capital comparison with statistical significance
- ✅ **Prediction registry** — full lifecycle tracking: prediction → outcome → quality grading
- ✅ **Feature store** — cached feature computation with schema versioning
- ✅ **Auto-retraining** — scheduled + drift-triggered + self-healing retry with exponential backoff
- ✅ **Model promotion gates** — min Sharpe, max drawdown, min precision, WF validation, Monte Carlo probability

### Backtesting
- ✅ **Backtrader engine** — event-driven backtesting with full analyzers
- ✅ **VectorBT engine** — vectorized high-performance backtests with parameter sweeps
- ✅ **Walk-forward optimization** — out-of-sample parameter validation with stability scoring
- ✅ **Monte Carlo simulation** — probability of profit/ruin analysis with configurable trials
- ✅ **Portfolio-level backtesting** — multi-symbol combined strategy evaluation
- ✅ **Asset-class-aware parameters** — per-symbol backtest config via layered configuration
- ✅ **Execution simulator** — realistic slippage, partial fills, latency, and spread modeling

### Risk Management
- ✅ **4-layer risk engine** — Portfolio, Account, Exposure, and Execution risk layers
- ✅ **Position sizing** — ATR-based dynamic stops, Kelly criterion, correlation-aware
- ✅ **Circuit breakers** — daily loss limits, max drawdown, consecutive loss halt
- ✅ **PDT protection** — Pattern Day Trader rule awareness ($25K threshold)
- ✅ **Sector concentration limits** — max sector exposure with dynamic lookup
- ✅ **Emergency controls** — panic liquidation, risk halts, cooldown after large losses

### Operations & Observability
- ✅ **Full Telegram bot interface** — 50+ commands, inline buttons, real-time control
- ✅ **Telegram audit forwarder** — ALL events + WARNING+ logs forwarded as rich HTML notifications
- ✅ **Event-driven architecture** — event bus, persistent event store, replay, crash recovery
- ✅ **CQRS read models** — incremental projections for fast dashboard queries
- ✅ **State snapshotting** — periodic persistence for sub-second recovery
- ✅ **Portfolio reconciliation** — periodic broker ↔ internal state sync with auto-fix
- ✅ **Health monitoring** — CPU, memory, latency, component status, structured logging
- ✅ **Notifications** — Telegram + Discord alerts on trades, signals, errors, daily summaries
- ✅ **SQLite persistence** — trade history, signals, portfolio snapshots, trade journal, event store

### Configuration & Automation
- ✅ **Database-backed configuration** — persisted settings survive restarts, seeded from .env
- ✅ **Layered configuration** — System Default → Environment → Strategy → Exchange → User Override
- ✅ **Runtime hot-swap** — change any parameter live via Telegram without restart
- ✅ **Asset-class scheduler** — DAG-based activity orchestration with dependency tracking
- ✅ **Automation scheduler** — periodic backtests, ML retraining, parameter sweeps, daily summaries

### Deployment
- ✅ **Docker containerized** — Python 3.12-slim, non-root user, health checks
- ✅ **Azure CI/CD** — GitHub Actions → Azure Container Registry → Azure Container Instances
- ✅ **Log Analytics integration** — Azure Monitor workspace for production observability

---

## Quick Start

### Prerequisites

- Python 3.12+
- Alpaca Markets account (free paper trading account included)
- Telegram bot token (optional, for remote management)

### 1. Setup

```bash
cd algo-trader

# Create virtual environment
python -m venv venv

# Activate
source venv/Scripts/activate   # Windows Git Bash
# OR
.\venv\Scripts\activate        # Windows PowerShell
# OR
source venv/bin/activate       # Linux/macOS

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your Alpaca API keys + Telegram bot token
```

Get free API keys: https://app.alpaca.markets/signup

> **Note:** On first run, the configuration service seeds the database from your `.env` file. Subsequent changes made via Telegram or the config service persist across restarts.

### 3. Run

```bash
# Paper trading (safe — no real money)
python main.py

# Check account status
python main.py --status

# Multi-strategy mode (concurrent strategies)
python main.py --strategy multi

# Custom multi-strategy config
python main.py --strategy multi --strategies "momentum:AAPL,MSFT:1Hour:60:1.0;ml:NVDA,TSLA:15Min:120:1.5"

# Backtest before going live
python main.py --backtest --strategy momentum
python main.py --backtest-bt --strategy momentum     # Backtrader engine
python main.py --backtest-vbt                         # VectorBT vectorized

# Dry run (signals only, no orders)
python main.py --dry-run

# Train ML model
python main.py --train

# Disable WebSocket streaming
python main.py --no-stream

# Live trading (⚠️ real money! requires typing "YES" to confirm)
python main.py --live
```

---

## Telegram Bot — Full Operations Guide

The Telegram bot provides a complete management interface. You can monitor your portfolio, execute trades, configure strategy parameters, run backtests, train ML models, and manage risk — all from your phone.

### Telegram Setup

#### 1. Create a Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the prompts
3. Copy the **bot token** (looks like `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

#### 2. Get Your Chat ID

1. Search for **@userinfobot** on Telegram and send `/start`
2. It will reply with your numeric **chat ID** (e.g., `123456789`)

#### 3. Configure .env

```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

#### 4. Start the Bot

```bash
python main.py
```

The bot starts automatically alongside the trading engine. Send `/start` to your bot on Telegram to verify it's connected.

---

### Authentication & Security

- The **first user** to send `/start` is automatically registered as an authorized user
- If `TELEGRAM_CHAT_ID` is set in `.env`, only that chat ID can interact with the bot
- All trading commands (buy/sell/close) require **inline button confirmation** — no accidental trades
- Emergency actions (`/closeall`) require explicit confirmation via button press
- Unauthorized users receive no response

---

### Information Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/start` | Welcome message + connection status | `/start` |
| `/status` | Account equity, cash, buying power, day P&L, position count | `/status` |
| `/positions` | All open positions with entry price, qty, and unrealized P&L | `/positions` |
| `/orders` | All pending/open orders with type and price | `/orders` |
| `/pnl` | Today's P&L summary: trades, win rate, daily return, halt status | `/pnl` |
| `/trades` | Last 10 executed trades with symbol, side, qty, price, P&L | `/trades` |
| `/signals` | Current strategy signals for all watched symbols | `/signals` |
| `/help` | Full command list organized by category | `/help` |

#### Example: `/status` Response

```
Account Status [ACTIVE]
==============================
Equity:       $  102,345.67
Cash:         $   45,123.89
Buying Power: $   90,247.78
──────────────────────────────
Day P&L:      $     +234.56 (+0.23%)
Positions:              5
Day Trades:             2
```

#### Example: `/positions` Response

```
Open Positions:

AAPL     | 50.0000 @ $178.23 | PnL: +$234.50 (2.6%)
TSLA     | 10.0000 @ $245.10 | PnL: -$45.20 (-1.8%)
BTC/USD  | 0.5000 @ $43250.00 | PnL: +$625.00 (2.9%)

Total Unrealized: +$814.30
```

---

### Trading Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/buy SYMBOL QTY` | Place a market buy order | `/buy AAPL 10` |
| `/sell SYMBOL QTY` | Place a market sell order | `/sell TSLA 5` |
| `/close SYMBOL` | Close entire position in a symbol | `/close AAPL` |
| `/closeall` | **EMERGENCY**: Close ALL positions + cancel ALL orders | `/closeall` |
| `/cancelall` | Cancel all pending orders | `/cancelall` |

#### How Trading Works via Telegram

1. You send a command like `/buy AAPL 10`
2. The bot shows a **confirmation card** with inline buttons:
   - ✅ **Confirm BUY** — executes the order
   - ❌ **Cancel** — cancels the action
3. Tap **Confirm** to execute, or **Cancel** to abort
4. Bot responds with order ID on success, or error message on failure

#### Emergency Liquidation (`/closeall`)

This is the nuclear option:
1. Sends a confirmation prompt: "EMERGENCY: Close ALL positions and cancel ALL orders?"
2. You must tap **"YES - CLOSE ALL"** to confirm
3. On confirmation: cancels every open order, then market-sells every position
4. Use this when things go wrong fast

---

### Bot Control Commands

| Command | Description |
|---------|-------------|
| `/pause` | **Pause auto-trading** — the bot stops generating/executing signals but stays online. Manual commands still work. |
| `/resume` | **Resume auto-trading** — re-enables the strategy loop. Also clears any risk halt. |
| `/strategy` | Show the currently active strategy name, symbols, timeframe, lookback |
| `/risk` | Show all risk parameters (max risk/trade, daily loss limit, exposure, leverage, etc.) |

#### Pause vs. Risk Halt

- **Pause** (`/pause`): Manual stop. You control when to resume.
- **Risk Halt**: Automatic stop when daily loss limit is breached. Bot sends a `🚨 RISK HALT` notification. Use `/resume` to clear it.

---

### Configuration Commands

All configuration can be changed **at runtime** without restarting the bot. Changes take effect on the next trading cycle.

#### Universal Setter (`/set`)

The most powerful command — change ANY configurable parameter:

```
/set PARAM VALUE
```

| Parameter | Type | Range | Example |
|-----------|------|-------|---------|
| `timeframe` | choice | 1Min, 5Min, 15Min, 30Min, 1Hour, 4Hour, 1Day | `/set timeframe 15Min` |
| `lookback` | int | 50–5000 | `/set lookback 500` |
| `interval` | int | 10–3600 (seconds) | `/set interval 120` |
| `momentum_fast_ema` | int | 2–100 | `/set momentum_fast_ema 8` |
| `momentum_slow_ema` | int | 5–500 | `/set momentum_slow_ema 21` |
| `momentum_rsi_period` | int | 2–100 | `/set momentum_rsi_period 10` |
| `momentum_rsi_oversold` | int | 5–50 | `/set momentum_rsi_oversold 25` |
| `momentum_rsi_overbought` | int | 50–95 | `/set momentum_rsi_overbought 75` |
| `momentum_atr_period` | int | 2–100 | `/set momentum_atr_period 10` |
| `momentum_atr_sl_mult` | float | 0.5–10.0 | `/set momentum_atr_sl_mult 1.5` |
| `momentum_atr_tp_mult` | float | 0.5–20.0 | `/set momentum_atr_tp_mult 4.0` |
| `mean_rev_bb_period` | int | 5–100 | `/set mean_rev_bb_period 30` |
| `mean_rev_bb_std` | float | 0.5–5.0 | `/set mean_rev_bb_std 2.5` |
| `mean_rev_zscore_entry` | float | 0.5–5.0 | `/set mean_rev_zscore_entry 1.5` |
| `mean_rev_zscore_exit` | float | 0.0–3.0 | `/set mean_rev_zscore_exit 0.3` |
| `mean_rev_rsi_period` | int | 2–100 | `/set mean_rev_rsi_period 10` |
| `ml_min_confidence` | float | 0.3–0.99 | `/set ml_min_confidence 0.70` |
| `ml_retrain_interval_hours` | int | 1–168 | `/set ml_retrain_interval_hours 12` |
| `max_position_size_pct` | float | 0.001–1.0 | `/set max_position_size_pct 0.03` |
| `max_daily_loss_pct` | float | 0.005–1.0 | `/set max_daily_loss_pct 0.03` |
| `max_portfolio_exposure` | float | 0.1–2.0 | `/set max_portfolio_exposure 0.90` |
| `max_single_stock_pct` | float | 0.01–1.0 | `/set max_single_stock_pct 0.20` |
| `max_leverage` | float | 0.1–10.0 | `/set max_leverage 2.0` |
| `max_open_positions` | int | 1–200 | `/set max_open_positions 30` |
| `max_orders_per_day` | int | 1–1000 | `/set max_orders_per_day 200` |
| `max_correlated_positions` | int | 1–20 | `/set max_correlated_positions 5` |
| `default_stop_loss_pct` | float | 0.005–0.5 | `/set default_stop_loss_pct 0.04` |
| `default_take_profit_pct` | float | 0.005–1.0 | `/set default_take_profit_pct 0.08` |
| `auto_backtest_interval_hours` | int | 0–168 | `/set auto_backtest_interval_hours 3` |
| `auto_train_interval_hours` | int | 0–168 | `/set auto_train_interval_hours 12` |
| `auto_sweep_interval_hours` | int | 0–168 | `/set auto_sweep_interval_hours 24` |
| `auto_train_bars` | int | 100–10000 | `/set auto_train_bars 2000` |
| `notify_on_trade` | bool | true/false | `/set notify_on_trade false` |
| `notify_on_error` | bool | true/false | `/set notify_on_error true` |
| `notify_on_signal` | bool | true/false | `/set notify_on_signal true` |
| `backtest_initial_cash` | float | 100–10,000,000 | `/set backtest_initial_cash 50000` |
| `backtest_commission_pct` | float | 0.0–0.1 | `/set backtest_commission_pct 0.002` |
| `backtest_slippage_pct` | float | 0.0–0.1 | `/set backtest_slippage_pct 0.001` |

#### Specialized Setters

| Command | Description | Example |
|---------|-------------|---------|
| `/setrisk PARAM VALUE` | Set risk parameters with validation | `/setrisk max_daily_loss_pct 0.03` |
| `/setstrategy NAME` | Switch active strategy (momentum, mean_reversion, ml) | `/setstrategy ml` |
| `/setsymbols SYM1,SYM2` | Change watched symbols | `/setsymbols AAPL,TSLA,BTC/USD,ETH/USD` |
| `/setinterval SECS` | Change trading cycle interval (10–3600s) | `/setinterval 120` |
| `/settf TIMEFRAME` | Change candle timeframe | `/settf 15Min` |
| `/setlookback BARS` | Change historical lookback (50–5000) | `/setlookback 500` |
| `/setauto TYPE HOURS` | Configure automation schedule (0 = disable) | `/setauto train 12` |
| `/setnotify TYPE on\|off` | Toggle notification types | `/setnotify signal on` |

#### Viewing Configuration (`/config`)

```
/config           — Show ALL settings
/config risk      — Risk parameters only
/config strategy  — Strategy settings only
/config momentum  — Momentum parameters
/config ml        — ML settings
/config auto      — Automation schedule
/config notify    — Notification preferences
```

---

### Backtesting & ML Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/backtest [SYMBOL] [DAYS]` | Run Backtrader backtest on historical data | `/backtest AAPL 30` |
| `/backtestvbt [SYMBOL] [DAYS]` | Run VectorBT vectorized backtest (faster) | `/backtestvbt TSLA 60` |
| `/sweep [SYMBOL] [DAYS]` | Full EMA parameter sweep — find optimal fast/slow combos | `/sweep BTC/USD 90` |
| `/train [SYMBOLS] [BARS]` | Train/retrain XGBoost ML model | `/train AAPL,TSLA,NVDA 2000` |
| `/modelinfo` | Show ML model metadata: type, date, features, estimators | `/modelinfo` |
| `/predict [SYMBOL]` | Get ML prediction with confidence + suggested SL/TP | `/predict BTC/USD` |

#### `/backtest` — Backtrader Backtest

Runs a full event-driven backtest using Backtrader. Returns: return %, trade count, win rate, Sharpe ratio, max drawdown, final portfolio value.

```
/backtest AAPL 60
```

Response:
```
Backtest Results: AAPL
==============================
Strategy:     Momentum
Return:       12.34%
Trades:       23
Win Rate:     65.2%
Sharpe:       1.45
Max DD:       -8.23%
Final Value:  $11,234.00
```

#### `/backtestvbt` — VectorBT Backtest

Faster vectorized backtest. Also includes a mini parameter sweep showing top 3 parameter combinations.

```
/backtestvbt TSLA 30
```

#### `/sweep` — Parameter Sweep

Tests all combinations of fast EMA (5–30) and slow EMA (20–100). Returns top 10 combinations sorted by return.

```
/sweep NVDA 60
```

Response:
```
Parameter Sweep: NVDA
===================================
Fast  Slow  Return    Trades  WinRate
8     21    18.3%     45      62%
10    26    16.7%     38      65%
12    30    15.2%     32      68%
...
```

#### `/train` — ML Model Training

Trains an XGBoost multi-class classifier (BUY/HOLD/SELL) on specified symbols:

```
/train AAPL,TSLA,NVDA,BTC/USD 2000
```

- Without arguments: trains on all configured symbols with 1000 bars
- Takes 1–3 minutes depending on data volume
- Model is saved and strategy reloads automatically on next cycle

#### `/predict` — ML Prediction

Get a real-time ML prediction for any symbol:

```
/predict BTC/USD
```

Response:
```
🔮 ML Prediction: BTC/USD
==============================
Direction:  BUY 📈
Confidence: 78% (STRONG)
Price:      $43,567.89

Probabilities:
  📉 Sell:  12%
  ➡️ Hold:  10%
  📈 Buy:   78%

Key Indicators:
  RSI(14): 42.35
  Return(1bar): 0.0123
  Volatility(5): 0.0234
  BB %B: 0.35

Suggested Levels:
  Stop Loss:    $42,890.00
  Take Profit:  $44,590.00
```

Strength interpretation:
- **STRONG** (≥80%): High-conviction signal
- **MODERATE** (≥65%): Decent signal, consider position sizing
- **WEAK** (<65%): Low conviction, bot won't auto-trade below `ml_min_confidence`

---

### Advanced Research Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/walkforward [SYMBOL] [BARS]` | Walk-forward optimization with out-of-sample validation | `/walkforward BTC/USD 2000` |
| `/montecarlo [SYMBOL] [BARS] [SIMS]` | Monte Carlo simulation for risk analysis | `/montecarlo AAPL 1000 500` |
| `/portfolio [BARS]` | Portfolio-level backtest across all configured symbols | `/portfolio 500` |
| `/models` | List all model versions in registry | `/models` |
| `/rollback vXXX` | Rollback to a previous model version | `/rollback v003` |
| `/journal [SYMBOL] [N]` | View trade journal entries | `/journal AAPL 10` |
| `/journalstats` | Journal performance analytics by confidence & model version | `/journalstats` |

#### `/walkforward` — Walk-Forward Optimization

Tests strategy robustness by training on rolling windows and testing out-of-sample:

```
/walkforward TSLA 2000
```

Response includes: OOS return, OOS Sharpe, OOS max drawdown, OOS win rate, parameter stability score, and best parameters per window.

#### `/montecarlo` — Monte Carlo Simulation

Runs N simulated equity curves by resampling historical trade results:

```
/montecarlo AAPL 1000 500
```

Response:
```
Monte Carlo: AAPL
==============================
Simulations: 500
Trades Used: 47

Return Distribution:
  P5 (worst):  -12.34%
  P25:         +2.45%
  Median:      +8.67%
  P75:         +15.23%
  P95 (best):  +28.90%

Risk Metrics:
  Expected Return: +9.12%
  Std Dev: 11.45%
  Median Max DD: -15.67%
  P(Profit): 72.4%
  P(Ruin <50%): 2.1%
```

#### `/portfolio` — Portfolio Backtest

Runs the active strategy across all configured symbols simultaneously:

```
/portfolio 1000
```

Returns aggregate metrics + per-symbol breakdown of P&L and trade counts.

#### `/models` — Model Version Registry

```
/models
```

Shows version history: version ID, training date, accuracy, symbols used, feature count, samples, and which version is currently active.

#### `/rollback` — Model Rollback

Revert to a previous model version if the current one underperforms:

```
/rollback v003
```

The model reloads on the next trading cycle. Use `/models` to see available versions.

#### `/journal` & `/journalstats` — Trade Journal

Every trade is logged with entry/exit price, P&L, confidence, strategy, model version, and exit reason.

```
/journal TSLA 5        — Last 5 TSLA trades
/journal 20            — Last 20 trades (all symbols)
/journalstats          — 30-day analytics: win rate by confidence bucket, by model version
```

---

### System & Monitoring Commands

| Command | Description |
|---------|-------------|
| `/health` | Full component health check (broker, DB, ML, Telegram, scheduler) |
| `/healthops` | Ops-formatted health: CPU, memory, disk, uptime, component latencies |
| `/metrics [DAYS]` | Live trading performance metrics (default 30 days) |
| `/performance` | Sharpe, Sortino, win rate, profit factor, expectancy, max drawdown |
| `/riskreport` | Current risk exposure: gross/net, VaR, drawdown, P&L, open positions |
| `/latency` | Broker API and component latency measurements |
| `/system` | System info: Python version, platform, PID, threads, memory, disk |
| `/uptime` | Bot uptime and start time |
| `/events [N]` | Recent event bus activity (last N events) |
| `/activetrades` | Active trade lifecycles from the trade manager |
| `/reconcile` | Trigger portfolio reconciliation: match broker positions vs. internal state |

#### `/health` — Component Health Check

```
System Health
═══════════════════════
✅ System: CPU 12.3%, RAM 245MB
✅ Broker: Connected (23ms)
✅ Database: OK (1.2MB, 2ms)
✅ ML Model: Loaded (3h old)
✅ Telegram: Polling
✅ Scheduler: 3 jobs active
```

#### `/performance` — Live Metrics

```
📈 Live Performance (30d)
════════════════════════════
Sharpe Ratio:        1.45
Sortino Ratio:       2.12
Win Rate:           64.3%
Profit Factor:       1.89
Expectancy:       +$23.45
────────────────────────────
Total Trades:          87
Avg Win:          +$45.67
Avg Loss:          $24.12
Max Drawdown:      -6.78%
Cur Drawdown:      -1.23%
Consec Wins:            7
Consec Losses:          3
```

#### `/riskreport` — Risk Metrics

```
🛡️ Risk Report
════════════════════════════
Gross Exposure:  $   82,345.00
Net Exposure:    $   67,890.00
Portfolio VaR:   $    1,234.00
Drawdown:              -2.34%
Daily P&L:       $     +456.78
Open Positions:             5
Cash Ratio:            45.2%
Portfolio Heat:          0.67
```

---

### Sector Management Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/setsector SYMBOL SECTOR` | Manually classify a symbol's sector (persists to DB) | `/setsector PLTR technology` |
| `/sectors` | List all manually cached sector classifications | `/sectors` |

**Valid sectors:** `communication_services`, `consumer_discretionary`, `consumer_staples`, `crypto`, `energy`, `financials`, `healthcare`, `industrials`, `materials`, `real_estate`, `technology`, `utilities`

Sector classifications are used by the Portfolio Risk Layer for concentration checks. Symbols that can't be auto-resolved get classified as `unmapped_equity` until manually set via `/setsector`.

---

### Model Governance Commands

| Command | Description |
|---------|-------------|
| `/governance` | Model governance audit: versions deployed, retired, completeness score |
| `/modelaudit` | Validate current model for deployment: git commit, CV accuracy, WF Sharpe, MC prob |
| `/replay [SESSION_ID]` | Event replay: list or inspect past trading sessions |
| `/recover` | Trigger crash recovery: reconcile positions, replay missed events |

---

### Automated Notifications

The bot automatically sends notifications without any command input:

| Notification | When | Example |
|---|---|---|
| **Trade Executed** | Every buy/sell order fills | `🟢 BUY AAPL — Qty: 10 @ $178.23, Conf: 72%, SL: $172.88, TP: $189.55` |
| **Position Closed** | Every position exit | `💰 CLOSED TSLA — PnL: +$234.56 (+3.2%)` |
| **Signal Generated** | When `notify_on_signal` is ON | `📊 SIGNAL: BUY BTC/USD — Confidence: 78%` |
| **Risk Halt** | Daily loss limit breached | `🚨 RISK HALT — Trading paused. Reason: Daily loss limit -5.2% exceeded` |
| **Error** | System/broker errors | `⚠️ ERROR — Broker connection timeout. Context: order_placement` |
| **Bot Started** | On startup | `🤖 Bot Started — Mode: PAPER, Strategy: momentum, Symbols: AAPL, TSLA...` |
| **Daily Summary** | End of trading day | `📊 Daily Summary — Trades: 12, Win Rate: 67%, PnL: +$567.89` |
| **ML Trained** | After model retraining | `🧠 ML Model Retrained — Accuracy: 68.5%, Features: 32, Samples: 5000` |

#### Controlling Notifications

```
/setnotify trade on      — Enable trade alerts
/setnotify trade off     — Disable trade alerts
/setnotify signal on     — Enable signal alerts (noisy but informative)
/setnotify error off     — Disable error alerts (not recommended)
```

---

## Strategies

### Momentum
Multi-indicator trend following:
- EMA crossover (12/26) for direction
- RSI (14) for overbought/oversold
- MACD histogram for momentum
- Volume spike confirmation
- ATR-based dynamic stop-loss/take-profit

**Telegram-configurable params:** `momentum_fast_ema`, `momentum_slow_ema`, `momentum_rsi_period`, `momentum_rsi_oversold`, `momentum_rsi_overbought`, `momentum_atr_period`, `momentum_atr_sl_mult`, `momentum_atr_tp_mult`

### Mean Reversion
Statistical mean reversion:
- Bollinger Bands for range boundaries
- Z-score for deviation measurement
- RSI for confirmation
- Targets SMA as exit point
- Configurable minimum confidence threshold

**Telegram-configurable params:** `mean_rev_bb_period`, `mean_rev_bb_std`, `mean_rev_zscore_entry`, `mean_rev_zscore_exit`, `mean_rev_rsi_period`, `mean_rev_min_confidence`

### ML (XGBoost)
Machine learning prediction:
- 30+ engineered features (momentum, volatility, volume, pattern recognition)
- XGBoost multi-class classifier (BUY/HOLD/SELL)
- Time-series cross-validation (prevents look-ahead bias)
- Auto-retrain on schedule + drift-triggered
- Confidence threshold filtering
- Full model governance with versioning and lineage

**Telegram-configurable params:** `ml_min_confidence`, `ml_retrain_interval_hours`

### Composable
A rule-composition framework allowing custom indicator combinations via presets. Two built-in presets:
- **Composable Momentum** — combines multiple momentum indicators with customizable weights
- **Composable Mean Reversion** — statistical mean reversion with pluggable exit rules

### Multi-Strategy Orchestrator (`--strategy multi`)
Run multiple strategies concurrently, each with independent:
- Symbol universe
- Timeframe and cycle interval
- Capital allocation weight
- Performance tracking (win rate, P&L, trade count)
- Enable/disable at runtime via Telegram

```bash
# Auto-configures: momentum (equities), mean_reversion (equities, 15Min), 
# ML (all symbols), crypto_momentum (crypto, 5Min)
python main.py --strategy multi

# Explicit config: name:symbols:timeframe:interval:weight
python main.py --strategy multi --strategies "momentum:AAPL,MSFT:1Hour:60:1.0;ml:NVDA:15Min:120:1.5"
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                                    main.py                                          │
│                           (Entry point + Trading Loop)                              │
├──────────┬──────────────┬──────────────┬──────────┬────────────┬───────────────────┤
│ Strategy │  Adaptive    │  Execution   │   Risk   │  Telegram  │   Monitoring &    │
│Orchestr. │Intelligence  │   Engine     │  Engine  │    Bot     │   Observability   │
├──────────┼──────────────┼──────────────┼──────────┼────────────┼───────────────────┤
│ Momentum │ Market State │ Orders       │Portfolio │ 50+ Cmds   │ Health Monitor    │
│ MeanRev  │ Regime Class.│ Signal Gate  │ Account  │ Audit Fwd  │ Live Metrics      │
│ ML/XGB   │ Trade Scorer │ Exec Simul.  │ Exposure │ Sector Mgmt│ Ops Handler       │
│Composable│ Drift Monitor│ Signal Bridge│Execution │ Config Cmds│ Telemetry         │
│ Multi    │ Meta Strategy│ Trade Stream │ PDT/VaR  │ Runtime Sw │ Structured Logs   │
│          │ Capital Alloc│              │ Sector   │            │                   │
│          │ Explainability              │ Circuit  │            │                   │
│          │ Counterfactual              │ Breakers │            │                   │
└────┬─────┴──────┬───────┴──────┬───────┴────┬─────┴──────┬─────┴────────┬──────────┘
     │            │              │            │            │              │
     ▼            ▼              ▼            ▼            ▼              ▼
┌──────────┐ ┌─────────┐  ┌──────────┐ ┌─────────┐ ┌──────────┐  ┌──────────────┐
│ Alpaca   │ │ SQLite  │  │  Models  │ │Telegram │ │  Event   │  │     Logs     │
│ REST+WS  │ │  DB     │  │ .joblib  │ │   API   │ │  Store   │  │  (structlog) │
│ Streams  │ │ Config  │  │ Registry │ │   Bot   │ │ Snapshots│  │  Azure Log   │
└──────────┘ └─────────┘  └──────────┘ └─────────┘ └──────────┘  └──────────────┘
```

### Event-Driven Core

The system is built on a lightweight, thread-safe, in-process event bus:

- **Domain Events**: `SignalGenerated`, `OrderSubmitted`, `OrderFilled`, `OrderRejected`, `PositionOpened`, `PositionClosed`, `RiskHalt`, `DriftDetected`, `BacktestTriggered`, `ModelTrainingCompleted`, `SchedulerEvent`
- **Event Store**: Persistent SQLite-backed event log with session tracking
- **Event Replay**: Reconstruct state from event history (for debugging or recovery)
- **Crash Recovery**: On startup, reconciles broker positions with internal state and replays missed events
- **CQRS Read Models**: Incremental projections for fast dashboard queries without re-scanning events
- **State Snapshots**: Periodic persistence (every 500 events) for sub-second cold-start recovery
- **Portfolio Reconciliation**: Periodic broker ↔ internal state sync with auto-fix for safe discrepancies

### Signal Pipeline

Every trade signal passes through a validation pipeline before execution:

```
Strategy Signal → Signal Package → Validation Gate → Intelligence Layer → Risk Engine → Execution
                   (completeness)   (confidence,      (regime, score,      (4-layer     (broker
                                    risk/reward,       drift, routing,      checks)      + simulator)
                                    provenance)        allocation)
```

- **Signal Package**: Comprehensive execution package with entry zone, stop loss, take profit levels, holding period, confidence decay schedule, and model provenance
- **Validation Gate**: Blocks incomplete signals from execution (configurable strictness)
- **Intelligence Layer**: Scores trade quality, checks regime compatibility, allocates capital
- **Risk Engine**: 4 independent risk layers evaluate every order independently

---

## Adaptive Intelligence Layer

The intelligence subsystem orchestrates multiple components for adaptive decision-making:

| Component | Responsibility |
|-----------|---------------|
| **Market State Engine** | Computes market fingerprint (volatility, trend, breadth, correlation) |
| **Regime Classifier** | Classifies current market regime (trending, mean-reverting, volatile, quiet) |
| **Meta-Strategy Engine** | Ranks strategies by expected performance in current regime |
| **Strategy Router** | Enables/disables strategies based on regime compatibility |
| **Drift Monitor** | Detects concept drift via rolling accuracy windows |
| **Trade Quality Scorer** | Composite score (0–100) evaluating trade worthiness |
| **Capital Allocator** | Dynamic sizing via Kelly criterion + correlation filters + sector limits |
| **Decision Explainer** | Human-readable explanations for accept/reject decisions |
| **Decision Journal** | Full audit trail of every intelligence decision |
| **Counterfactual Engine** | What-if analysis comparing actual vs alternative outcomes |

Configuration:
```env
INTELLIGENCE_ENABLED=true
INTELLIGENCE_MIN_TRADE_SCORE=70       # Minimum score to allow trade (0-100)
INTELLIGENCE_DRIFT_WINDOW=200         # Rolling window for drift detection
INTELLIGENCE_DRIFT_MIN_SAMPLES=50     # Minimum samples before drift alerts
INTELLIGENCE_DRIFT_ALERT_DROP=0.12    # Accuracy drop threshold for alert
```

---

## Risk Management

The platform implements a **4-layer risk engine** where every order must pass all enabled layers independently:

### Layer 1: Portfolio Risk

| Rule | Default | Description |
|------|---------|-------------|
| Max positions | 10 | Concurrent open positions cap |
| Max single stock | 20% | No single position larger than this |
| Max sector exposure | 40% | Maximum capital in one sector |
| Max correlation | 0.80 | Block trades in highly correlated assets |
| Max gross exposure | 200% | Total absolute exposure |
| Max net exposure | 100% | Net long-short exposure |
| Max VaR | 5% | Portfolio Value-at-Risk limit |
| Max portfolio heat | 10% | Total portfolio risk (sum of position risks) |

### Layer 2: Account Risk

| Rule | Default | Description |
|------|---------|-------------|
| Max daily loss | 3% | Auto-halt when breached |
| Max weekly loss | 7% | Weekly drawdown circuit breaker |
| Max drawdown | 15% | Peak-to-trough limit |
| Min cash reserve | 20% | Always keep this % in cash |
| PDT threshold | $25,000 | Pattern Day Trader rule awareness |
| Consecutive loss limit | 5 | Halt after N consecutive losses |
| Daily trade limit | 20 | Max trades per day |

### Layer 3: Exposure Risk

| Rule | Default | Description |
|------|---------|-------------|
| Require stop loss | Yes | Block orders without a stop loss |
| Max ADV % | 1% | Maximum percentage of average daily volume |
| Max trade concentration | 5% | Single trade vs portfolio |
| Max overnight exposure | 60% | Reduce exposure before close |
| Earnings blackout | 1 day | No trading around earnings dates |
| High volatility threshold | 3% | Daily move threshold |
| High vol size reduction | 50% | Reduce position size in volatile markets |

### Layer 4: Execution Risk

| Rule | Default | Description |
|------|---------|-------------|
| Max spread | 0.5% | Block illiquid instruments |
| Min volume | 10,000 | Minimum daily volume |
| Max slippage | 0.3% | Expected slippage limit |
| Max orders/minute | 10 | Rate limiting |
| Cooldown after loss | 5 min | Pause after large loss |
| Large loss threshold | 1% | What constitutes a "large" loss |

### Legacy Risk Manager (Backward-Compatible)

Additionally, the primary risk manager provides top-level guardrails:

| Rule | Default | Range | Description |
|------|---------|-------|-------------|
| Max risk per trade | 2% | 0.1%–100% | Maximum portfolio % at risk per trade |
| Daily loss limit | 5% | 0.5%–100% | Auto-halt trading if down this much today |
| Max portfolio exposure | 80% | 10%–200% | Never invest more than this % of equity |
| Max single stock | 15% | 1%–100% | No single position larger than this |
| Max leverage | 1.0x | 0.1x–10x | Maximum leverage allowed |
| Max open positions | 20 | 1–200 | Cap on concurrent positions |
| Max orders/day | 100 | 1–1000 | Rate limit on daily order count |
| Max correlated | 3 | 1–20 | Max positions in correlated assets |
| Default stop-loss | 3% | 0.5%–50% | Automatic stop-loss on every trade |
| Default take-profit | 6% | 0.5%–100% | Default take-profit target |

All parameters configurable via `/setrisk` or `/set` on Telegram. Risk layer enabled/disabled individually:
```env
RISK_PORTFOLIO_LAYER_ENABLED=true
RISK_ACCOUNT_LAYER_ENABLED=true
RISK_EXPOSURE_LAYER_ENABLED=true
RISK_EXECUTION_LAYER_ENABLED=true
```

---

## Configuration System

The platform uses a sophisticated multi-level configuration system:

### Configuration Precedence

```
User Override (Telegram /set)  ←  highest priority
    ↓
Exchange-specific overrides
    ↓
Strategy-specific overrides
    ↓
Environment variables (.env)
    ↓
System defaults               ←  lowest priority
```

### Database-Backed Persistence

- On first run: seeds database from `.env` + built-in defaults
- Subsequent runs: loads from database (user changes persist across restarts)
- Telegram `/set` commands persist to database immediately
- Configuration snapshots for audit trail

### Asset-Class Awareness

Different parameters for equities vs crypto:
- Equities: market-hours-only trading, `1Hour` default timeframe, standard fees
- Crypto: 24/7 trading, `5Min` default timeframe, adjusted risk parameters
- Backtesting parameters adapt per-symbol via layered config

---

## CLI Options

```
python main.py --help

Options:
  --live              ⚠️ Enable LIVE trading (requires explicit "YES" confirmation)
  --strategy, -s      Strategy: momentum, mean_reversion, ml, multi
  --strategies        Multi-strategy config: "name:symbols:tf:interval:weight;..."
  --symbols           Comma-separated (AAPL,TSLA,BTC/USD)
  --timeframe, -tf    1Min, 5Min, 15Min, 30Min, 1Hour, 4Hour, 1Day
  --lookback          Historical bars to analyze (default: 200)
  --interval, -i      Seconds between cycles (default: 60)
  --dry-run           Signals only, no order execution
  --backtest          Run custom backtest engine
  --backtest-bt       Run Backtrader event-driven backtest
  --backtest-vbt      Run VectorBT vectorized backtest + parameter sweep
  --train             Train/retrain ML model on configured symbols
  --status            Show account status and exit
  --no-telegram       Disable Telegram bot
  --no-stream         Disable WebSocket streaming for real-time data
```

### Examples

```bash
# Paper trade with ML strategy on specific symbols
python main.py --strategy ml --symbols NVDA,TSLA,AAPL --interval 120

# Multi-strategy: momentum on equities, ML on crypto
python main.py --strategy multi --strategies "momentum:AAPL,MSFT:1Hour:60:1.0;ml:BTC/USD,ETH/USD:5Min:30:1.5"

# VectorBT backtest with 4-hour candles
python main.py --backtest-vbt --timeframe 4Hour --lookback 500

# Dry run (no orders) without Telegram
python main.py --dry-run --no-telegram --no-stream
```

---

## Project Structure

```
algo-trader/
├── main.py                            # Entry point, CLI, trading loop, orchestrator setup
├── config/
│   ├── settings.py                    # Pydantic settings (all env vars + validation)
│   └── __init__.py
├── src/
│   ├── broker/
│   │   ├── alpaca_client.py           # Alpaca REST + WebSocket client (bars, orders, streaming)
│   │   ├── base.py                    # Abstract broker interface
│   │   ├── paper.py                   # Paper trading broker implementation
│   │   └── replay_broker.py          # Replay broker for backtesting
│   ├── strategy/
│   │   ├── base.py                    # Abstract strategy interface + TradeSignal
│   │   ├── momentum.py                # Momentum (EMA/RSI/MACD/ATR)
│   │   ├── mean_reversion.py          # Mean reversion (BB/Z-score/RSI)
│   │   ├── ml_strategy.py             # ML/XGBoost strategy wrapper
│   │   ├── composable.py              # Composable rule-based strategy framework
│   │   ├── orchestrator.py            # Multi-strategy concurrent orchestrator
│   │   ├── signal_package.py          # Trade signal package with validation gate
│   │   ├── signal_bridge.py           # Bridge raw signals → full signal packages
│   │   └── strategy_store.py          # User-defined strategy persistence
│   ├── intelligence/
│   │   ├── orchestrator.py            # Top-level adaptive intelligence coordinator
│   │   ├── models.py                  # Intelligence decision data models
│   │   ├── regime/
│   │   │   └── classifier.py          # Multi-dimensional regime classifier
│   │   ├── scoring/
│   │   │   └── trade_quality.py       # Composite trade quality scoring (0–100)
│   │   ├── drift/
│   │   │   └── monitor.py             # Concept drift detection via rolling windows
│   │   ├── allocator/
│   │   │   ├── capital_allocator.py   # Kelly criterion + dynamic sizing
│   │   │   ├── correlation_filter.py  # Correlation-based position blocking
│   │   │   └── portfolio_allocator.py # Portfolio-level allocation optimization
│   │   ├── routing/
│   │   │   └── strategy_router.py     # Enable/disable strategies per regime
│   │   ├── explainability/
│   │   │   └── explainer.py           # Human-readable decision explanations
│   │   ├── journal/
│   │   │   └── decision_journal.py    # Full audit trail of intelligence decisions
│   │   ├── counterfactual/
│   │   │   └── engine.py              # What-if analysis engine
│   │   ├── meta_strategy/
│   │   │   └── engine.py              # Strategy ranking per market state
│   │   └── market_state/
│   │       └── engine.py              # Market fingerprint computation
│   ├── ml/
│   │   ├── features.py                # Feature engineering (30+ features)
│   │   ├── feature_store.py           # Feature caching & schema versioning
│   │   ├── predictor.py               # Model inference wrapper
│   │   ├── training_pipeline.py       # Full training pipeline with CV
│   │   ├── dataset_builder.py         # Training dataset construction
│   │   ├── model_registry.py          # Model version management
│   │   ├── governance.py              # Model governance, lineage, promotion gates
│   │   ├── ab_testing.py              # A/B testing framework (shadow + split modes)
│   │   └── retraining_trigger.py      # Drift-triggered + scheduled retraining
│   ├── risk/
│   │   ├── manager.py                 # Primary risk manager (legacy guardrails)
│   │   ├── engine.py                  # 4-layer risk engine orchestrator
│   │   ├── models.py                  # Risk data models
│   │   ├── portfolio_risk.py          # Portfolio risk layer (positions, sector, VaR)
│   │   ├── account_risk.py            # Account risk layer (P&L limits, drawdown)
│   │   ├── exposure_risk.py           # Exposure risk layer (concentration, overnight)
│   │   └── execution_risk.py          # Execution risk layer (spread, volume, rate)
│   ├── execution/
│   │   ├── engine.py                  # Trade execution orchestrator
│   │   ├── simulator.py               # Execution realism simulator (slippage, fills)
│   │   └── sector_lookup.py           # Dynamic sector classification with DB cache
│   ├── data/
│   │   ├── store.py                   # SQLAlchemy models + DB manager
│   │   └── journal.py                 # Trade journal persistence
│   ├── notifications/
│   │   ├── alerts.py                  # Simple webhook notifications (Telegram + Discord)
│   │   ├── telegram_bot.py            # Full interactive Telegram bot (50+ commands)
│   │   ├── telegram_audit_forwarder.py # Rich HTML event + log forwarding to Telegram
│   │   ├── telegram_config_commands.py # Runtime config & strategy management commands
│   │   ├── telegram_runtime_commands.py# Hot-swap, A/B test, environment switching
│   │   └── telegram_sector_commands.py # Sector classification commands
│   ├── monitoring/
│   │   ├── health.py                  # Component health monitoring (heartbeats)
│   │   ├── metrics.py                 # Live trading metrics (Sharpe, Sortino, etc.)
│   │   ├── ops_commands.py            # Ops command handlers (health, latency, system)
│   │   └── telemetry.py              # Telemetry & observability
│   ├── core/
│   │   ├── events.py                  # Domain events (20+ event types)
│   │   ├── bus.py                     # Thread-safe event bus (pub/sub)
│   │   ├── event_store.py             # Persistent SQLite event storage + replay
│   │   ├── subscribers.py             # Default event subscribers (audit, journal, notifications)
│   │   ├── state_machine.py           # Trade lifecycle state machine
│   │   ├── recovery.py                # Crash recovery (broker reconciliation + event replay)
│   │   ├── replay.py                  # Event replay engine for debugging
│   │   ├── reconciliation.py          # Broker ↔ internal state sync with auto-fix
│   │   ├── projections.py             # CQRS read model projections
│   │   ├── snapshots.py               # State snapshotting for fast recovery
│   │   ├── runtime.py                 # Runtime manager (hot-swap, env switching)
│   │   ├── runtime_state.py           # Runtime state (pause/resume, halt tracking)
│   │   ├── environment.py             # Broker environment manager
│   │   ├── circuit_breaker.py         # Circuit breaker implementation
│   │   └── audit_log.py               # Audit trail logging
│   ├── config/
│   │   ├── initializer.py             # Configuration service bootstrap
│   │   ├── service.py                 # Configuration service (CRUD + caching)
│   │   ├── repository.py              # Database-backed config repository
│   │   ├── layered.py                 # Multi-level config with precedence
│   │   ├── backtest_params.py         # Asset-class-aware backtest parameters
│   │   ├── typed_config.py            # Typed configuration accessors
│   │   ├── validation.py              # Configuration value validation
│   │   ├── seed.py                    # First-run configuration seeding
│   │   ├── snapshots.py               # Configuration snapshots
│   │   └── models.py                  # Configuration data models
│   ├── predictions/
│   │   ├── registry.py                # Prediction lifecycle tracking
│   │   ├── outcome_recorder.py        # Outcome recording (win/loss/timeout)
│   │   ├── calibration.py             # Prediction calibration analysis
│   │   └── metrics.py                 # Prediction accuracy metrics
│   ├── scheduler/
│   │   ├── asset_class_scheduler.py   # DAG-based activity orchestrator
│   │   ├── activity_graph.py          # Activity dependency graph
│   │   ├── triggers.py                # Activity trigger conditions
│   │   └── market_status.py           # Market hours/status service
│   ├── market/
│   │   ├── calendars.py               # Exchange calendar definitions
│   │   ├── sessions.py                # Trading session management
│   │   ├── holidays.py                # Market holiday database
│   │   ├── instruments.py             # Instrument metadata
│   │   ├── timezones.py               # Timezone handling
│   │   ├── annualization.py           # Annualization factors per asset class
│   │   ├── gap_detection.py           # Price gap detection
│   │   └── ...                        # (exchanges, constraints, corporate actions)
│   ├── portfolio/
│   │   └── optimizer.py               # Portfolio optimization
│   └── utils/
│       └── logger.py                  # Structured logging (structlog)
├── backtesting/
│   ├── backtest.py                    # Simple custom backtester
│   ├── bt_adapter.py                  # Backtrader adapter + strategies
│   ├── vbt_adapter.py                 # VectorBT adapter + parameter sweep
│   ├── walk_forward.py                # Walk-forward optimization
│   ├── monte_carlo.py                 # Monte Carlo simulation engine
│   ├── portfolio_backtest.py          # Multi-symbol portfolio backtest
│   ├── runner.py                      # Backtest runner orchestration
│   └── param_validator.py             # Parameter validation for backtests
├── models/                            # Saved ML models (versioned .joblib)
├── data_cache/                        # SQLite DB + cached data + events + snapshots
├── logs/                              # Application logs (structlog JSON)
├── tests/                             # Comprehensive test suite (50+ test files)
├── .github/
│   └── workflows/
│       └── deploy.yml                 # CI/CD: Build → ACR → Azure Container Instances
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variable template
├── Dockerfile                         # Python 3.12-slim, non-root, health check
├── pytest.ini                         # Test configuration
├── verify_integration.py              # Integration verification script
└── README.md
```

---

## Environment Configuration

All configuration is done via `.env` file (see `.env.example` for template).  
Settings are loaded by Pydantic with validation, type coercion, and range checking.

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADING_MODE` | `paper` | `paper` or `live` |
| `ALPACA_PAPER_API_KEY` | — | Your Alpaca paper API key |
| `ALPACA_PAPER_SECRET_KEY` | — | Your Alpaca paper secret |
| `ALPACA_LIVE_API_KEY` | — | Your Alpaca live API key (for live mode) |
| `ALPACA_LIVE_SECRET_KEY` | — | Your Alpaca live secret (for live mode) |
| `ALPACA_DATA_FEED` | `iex` | `iex` (free) or `sip` (paid, all exchanges) |

### Telegram

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | — | Bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | — | Your numeric chat ID (for auth; auto-detects if empty) |
| `DISCORD_WEBHOOK_URL` | — | Discord webhook for notifications (optional) |

### Trading

| Variable | Default | Description |
|----------|---------|-------------|
| `ACTIVE_STRATEGY` | `momentum` | Default strategy (`momentum`, `mean_reversion`, `ml`, `multi`) |
| `TRADING_SYMBOLS` | `AAPL,MSFT,GOOGL,AMZN,NVDA,BTC/USD,ETH/USD,SOL/USD,AAVE/USD,ADA/USD` | Comma-separated symbols |
| `TIMEFRAME` | `1Hour` | Candle timeframe |
| `LOOKBACK_BARS` | `200` | Historical bars to analyze |
| `TRADING_INTERVAL` | `60` | Seconds between trading cycles |
| `MULTI_STRATEGY_CONFIG` | — | Multi-strategy format: `name:symbols:tf:interval:weight;...` |

### Intelligence Layer

| Variable | Default | Description |
|----------|---------|-------------|
| `INTELLIGENCE_ENABLED` | `true` | Enable adaptive intelligence layer |
| `INTELLIGENCE_MIN_TRADE_SCORE` | `70` | Minimum trade quality score (0–100) |
| `INTELLIGENCE_DRIFT_WINDOW` | `200` | Rolling window for drift detection |
| `INTELLIGENCE_DRIFT_MIN_SAMPLES` | `50` | Min samples before drift alerts fire |
| `INTELLIGENCE_DRIFT_ALERT_DROP` | `0.12` | Accuracy drop threshold |

### Risk Engine (4-Layer)

| Variable | Default | Description |
|----------|---------|-------------|
| `RISK_PORTFOLIO_LAYER_ENABLED` | `true` | Enable portfolio risk layer |
| `RISK_ACCOUNT_LAYER_ENABLED` | `true` | Enable account risk layer |
| `RISK_EXPOSURE_LAYER_ENABLED` | `true` | Enable exposure risk layer |
| `RISK_EXECUTION_LAYER_ENABLED` | `true` | Enable execution risk layer |
| `RISK_MAX_DAILY_LOSS_PCT` | `0.03` | Daily loss circuit breaker |
| `RISK_MAX_DRAWDOWN_PCT` | `0.15` | Maximum peak-to-trough drawdown |
| `RISK_MIN_CASH_RESERVE_PCT` | `0.20` | Minimum cash buffer |

### Automation

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_BACKTEST_INTERVAL_HOURS` | `6` | Auto-backtest every N hours (0=off) |
| `AUTO_TRAIN_INTERVAL_HOURS` | `24` | Auto-retrain ML every N hours (0=off) |
| `AUTO_SWEEP_INTERVAL_HOURS` | `12` | Auto parameter sweep every N hours (0=off) |
| `AUTO_TRAIN_BARS` | `1000` | Bars to use for auto-training |

### Model Governance

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_MIN_SHARPE_RATIO` | `0.5` | Minimum Sharpe for model promotion |
| `MODEL_MAX_DRAWDOWN_PCT` | `0.20` | Maximum drawdown for promotion |
| `MODEL_MIN_PRECISION` | `0.50` | Minimum prediction precision |
| `MODEL_MIN_CV_ACCURACY` | `0.52` | Minimum cross-validation accuracy |
| `MODEL_MIN_WALK_FORWARD_SHARPE` | `0.0` | Walk-forward Sharpe gate |
| `MODEL_MIN_MONTE_CARLO_PROB_PROFIT` | `0.50` | Monte Carlo profit probability gate |
| `MODEL_MIN_BACKTEST_TRADES` | `30` | Minimum trades for statistical validity |

### Execution Simulator

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_EXECUTION_SIMULATOR` | `true` | Enable execution realism modeling |
| `EXECUTION_SIMULATOR_PRESET` | `realistic` | `realistic`, `conservative`, or `ideal` |

### Self-Healing

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_MAX_RETRIES` | `3` | Max training retries on failure |
| `MODEL_RETRY_BACKOFF_SECONDS` | `60` | Base backoff between retries |
| `MODEL_STALE_THRESHOLD_HOURS` | `168` | Trigger retraining if model older than this |

---

## Alpaca Account Setup

1. **Sign up** at https://app.alpaca.markets/signup (free)
2. **Paper account** is auto-created with $100,000 virtual cash
3. Go to **API Keys** → Generate new key
4. Copy `API Key` and `Secret Key` to your `.env` file
5. For live: fund account + generate live API keys separately

---

## Deployment

### Docker (Local)

```bash
# Build
docker build -t algo-trader .

# Run (paper mode)
docker run -d --name trader --env-file .env algo-trader

# Run with live data feed and custom interval
docker run -d --name trader --env-file .env \
  -e ALPACA_DATA_FEED=sip \
  algo-trader python main.py --interval 120

# View logs
docker logs -f trader

# Stop
docker stop trader && docker rm trader
```

The Docker image uses:
- Python 3.12-slim base
- Non-root `trader` user (UID 1001)
- Health check (30s interval)
- Numba JIT cache redirected to writable directory
- Persistent volumes recommended for `data_cache/`, `models/`, and `logs/`

### Azure CI/CD (Production)

The repository includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) that:

1. **Builds** the Docker image on every push to `main`
2. **Pushes** to Azure Container Registry (`suezacr.azurecr.io`)
3. **Deploys** to Azure Container Instances with:
   - 1 CPU, 2 GB RAM
   - Always-restart policy
   - Secrets injected via secure environment variables
   - Azure Log Analytics integration for observability
4. **Verifies** the container reaches `Running` state

Required GitHub Secrets:
```
ACR_USERNAME, ACR_PASSWORD          # Azure Container Registry credentials
AZ_CREDENTIALS                      # Azure service principal JSON
TRADING_MODE                        # paper or live
ALPACA_PAPER_API_KEY, ALPACA_PAPER_SECRET_KEY
ALPACA_LIVE_API_KEY, ALPACA_LIVE_SECRET_KEY
TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
LOG_ANALYTICS_WORKSPACE_ID, LOG_ANALYTICS_WORKSPACE_KEY
```

---

## Testing

The test suite covers 50+ test files across all subsystems:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_risk_engine.py

# Run with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "test_intelligence"
```

Key test areas:
- Event bus and subscribers
- Risk engine (all 4 layers)
- Strategy orchestrator
- Intelligence layer (drift, scoring, routing)
- ML lifecycle and governance
- Configuration service and persistence
- Telegram command handling
- Portfolio reconciliation
- Monte Carlo simulation determinism
- Signal pipeline integration

---

## Safety Notes

- ⚠️ **Always start with paper trading** — validate your strategy first
- ⚠️ **Backtest thoroughly** before any live trading (use `/backtest`, `/walkforward`, `/montecarlo`)
- ⚠️ **Never risk more than you can afford to lose**
- The bot has built-in daily loss limits, weekly drawdown limits, and multi-layer circuit breakers
- Live mode requires typing "YES" as confirmation on CLI startup
- All trades are logged to SQLite for full audit trail and forensic analysis
- Use `/pause` on Telegram to immediately halt auto-trading
- Use `/closeall` for emergency liquidation (requires button confirmation)
- Risk halts are automatic — the bot notifies you and stops trading
- Model rollback (`/rollback`) lets you revert to known-good ML models instantly
- The event store enables full crash recovery (`/recover`)
- Portfolio reconciliation runs every 5 minutes to detect and fix state drift
- The intelligence layer blocks low-quality trades even when the strategy fires a signal
- Model promotion gates prevent deploying underperforming models to production
- All Telegram commands are rate-limited (10 commands/minute per user)

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.12+ |
| Trading API | Alpaca Markets (REST + WebSocket) |
| ML Framework | XGBoost, scikit-learn |
| Backtesting | Backtrader, VectorBT, custom engine |
| Database | SQLite (SQLAlchemy + aiosqlite) |
| Telegram Bot | aiogram 3.x (async) |
| Configuration | Pydantic v2 + pydantic-settings |
| Scheduling | APScheduler + custom DAG scheduler |
| Logging | structlog (structured JSON) |
| HTTP | httpx, aiohttp |
| Visualization | matplotlib, plotly |
| Deployment | Docker, Azure Container Instances |
| CI/CD | GitHub Actions |
| Monitoring | Azure Log Analytics |

---

## License

Private repository. All rights reserved.

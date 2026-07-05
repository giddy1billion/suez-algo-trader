# Alpaca Algo Trader 🤖📈

A fully robust, scalable algorithmic trading bot with AI/ML integration.  
Uses **Alpaca Markets API** for commission-free trading (US stocks + crypto).  
Fully manageable via **Telegram** — monitor, trade, configure, backtest, and train ML models from your phone.

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
- [Risk Management](#risk-management)
- [CLI Options](#cli-options)
- [Project Structure](#project-structure)
- [Environment Configuration](#environment-configuration)
- [Alpaca Account Setup](#alpaca-account-setup)
- [Docker Deployment](#docker-deployment)
- [Safety Notes](#safety-notes)

---

## Features

- ✅ **Paper + Live trading** — switch with one env var
- ✅ **Full Telegram bot interface** — 50+ commands, inline buttons, real-time control
- ✅ **5 built-in strategies**: Momentum, Mean Reversion, ML (XGBoost), Composable, Composable MR
- ✅ **Risk Management** — position sizing, daily loss limits, exposure caps, circuit breakers
- ✅ **Backtesting engines** — Backtrader, VectorBT vectorized backtests, parameter sweeps
- ✅ **ML/AI integration** — XGBoost with 30+ engineered features, auto-retraining, model versioning
- ✅ **Walk-forward optimization** — out-of-sample parameter validation
- ✅ **Monte Carlo simulation** — probability of profit/ruin analysis
- ✅ **Portfolio-level backtesting** — multi-symbol combined strategy evaluation
- ✅ **Model governance** — version registry, lineage tracking, rollback, audit
- ✅ **Real-time WebSocket streaming** — sub-second data for stocks + crypto
- ✅ **Notifications** — Telegram + Discord alerts on trades, signals, errors, daily summaries
- ✅ **Event-driven architecture** — event bus, event store, replay, recovery
- ✅ **Health monitoring** — CPU, memory, latency, component status
- ✅ **SQLite persistence** — full trade history, signals, portfolio snapshots, trade journal
- ✅ **Emergency controls** — panic liquidation, circuit breakers, risk halts
- ✅ **Runtime configuration** — change any parameter live via Telegram without restart
- ✅ **Automation scheduler** — periodic backtests, retraining, parameter sweeps
- ✅ **Docker ready** — deploy anywhere

---

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
# Edit .env with your Alpaca API keys + Telegram bot token
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

**Telegram-configurable params:** `mean_rev_bb_period`, `mean_rev_bb_std`, `mean_rev_zscore_entry`, `mean_rev_zscore_exit`, `mean_rev_rsi_period`

### ML (XGBoost)
Machine learning prediction:
- 30+ engineered features
- XGBoost multi-class classifier (UP/DOWN/FLAT)
- Time-series cross-validation
- Auto-retrain on schedule
- Confidence threshold filtering

**Telegram-configurable params:** `ml_min_confidence`, `ml_retrain_interval_hours`

### Composable
A rule-composition framework allowing custom indicator combinations via presets.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        main.py                            │
├──────────┬──────────────┬──────────┬───────────┬───────────┬────────────┤
│ Strategy │ Intelligence │Execution │   Risk    │Telegram   │ Monitoring │
│ Engine   │    Layer     │ Engine   │  Manager  │   Bot     │  & Health  │
├──────────┼──────────────┼──────────┼───────────┼───────────┼────────────┤
│ Momentum │ Regime       │ Orders   │ Position  │ Commands  │ HealthMon  │
│ MeanRev  │ Scoring      │ Brackets │ Sizing    │ Alerts    │ Metrics    │
│ ML/XGB   │ Drift        │ Trailing │ Drawdown  │ Controls  │ OpsHandler │
│Composable│ Routing      │ Recovery │ VaR       │ Config    │ Telemetry  │
└────┬─────┴──────┬───────┴────┬─────┴─────┬─────┴─────┬─────┴──────┬─────┘
     │            │            │           │           │            │
     ▼            ▼            ▼           ▼           ▼            ▼
┌──────────┐  ┌────────┐   ┌────────┐ ┌────────┐ ┌────────┐  ┌────────┐
│ Alpaca   │  │ SQLite │   │ Models │ │Telegram│ │ Event  │  │  Logs  │
│ REST+WS  │  │  DB    │   │.joblib │ │  API   │ │ Store  │  │        │
└──────────┘  └────────┘   └────────┘ └────────┘ └────────┘  └────────┘
```

### Event-Driven Core

The system uses an internal event bus for decoupled communication:
- **Events**: `TradeExecuted`, `PositionOpened`, `PositionClosed`, `RiskHalt`, `SignalGenerated`
- **Event Store**: Persistent event log for replay and recovery
- **Subscribers**: Risk manager, notifications, trade journal all react to events
- **Recovery**: On crash, replays events to reconstruct state

---

## Risk Management

All trades pass through the risk manager before execution:

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

All parameters configurable via `/setrisk` or `/set` on Telegram.

---

## CLI Options

```
python main.py --help

Options:
  --live              ⚠️ Enable LIVE trading
  --strategy, -s      Strategy: momentum, mean_reversion, ml, composable
  --symbols           Comma-separated (AAPL,TSLA,BTC/USD)
  --timeframe, -tf    1Min, 5Min, 15Min, 30Min, 1Hour, 4Hour, 1Day
  --lookback          Historical bars to analyze (default: 200)
  --interval, -i      Seconds between cycles (default: 60)
  --dry-run           Signals only, no order execution
  --backtest          Run backtest on historical data
  --train             Train/retrain ML model
  --status            Show account status
```

---

## Project Structure

```
algo-trader/
├── main.py                        # Entry point, CLI, trading loop
├── config/
│   └── settings.py                # Pydantic configuration (all env vars)
├── src/
│   ├── broker/
│   │   └── alpaca_client.py       # Alpaca REST + WebSocket client
│   ├── strategy/
│   │   ├── base.py                # Abstract strategy interface
│   │   ├── momentum.py            # Momentum (EMA/RSI/MACD/ATR)
│   │   ├── mean_reversion.py      # Mean reversion (BB/Z-score)
│   │   ├── ml_strategy.py         # ML/XGBoost strategy
│   │   ├── composable.py          # Composable rule-based strategy
│   │   └── orchestrator.py        # Multi-strategy orchestrator
│   ├── intelligence/
│   │   ├── regime/                # Multi-dimensional regime classifier
│   │   ├── scoring/               # Trade quality scoring engine
│   │   ├── drift/                 # Concept drift monitor
│   │   ├── allocator/             # Dynamic capital allocation
│   │   ├── routing/               # Strategy enable/disable router
│   │   ├── explainability/        # Decision explanations
│   │   └── orchestrator.py        # Intelligence subsystem coordinator
│   ├── ml/
│   │   ├── features.py            # Feature engineering (30+ features)
│   │   ├── feature_store.py       # Feature caching & storage
│   │   ├── model_registry.py      # Model version management
│   │   ├── governance.py          # Model governance & lineage
│   │   ├── finrl_adapter.py       # FinRL integration adapter
│   │   └── liu_adapter.py         # Liu ML adapter
│   ├── risk/
│   │   └── manager.py             # Risk management engine
│   ├── execution/
│   │   └── engine.py              # Trade execution orchestrator
│   ├── data/
│   │   └── store.py               # SQLAlchemy models + DB manager
│   ├── notifications/
│   │   ├── alerts.py              # Simple webhook notifications
│   │   └── telegram_bot.py        # Full interactive Telegram bot (50+ commands)
│   ├── monitoring/
│   │   ├── health.py              # Component health monitoring
│   │   ├── metrics.py             # Live trading metrics (Sharpe, etc.)
│   │   ├── ops_commands.py        # Ops command formatters
│   │   └── telemetry.py           # Telemetry & observability
│   ├── core/
│   │   ├── bus.py                 # Event bus (pub/sub)
│   │   ├── events.py              # Domain events
│   │   ├── event_store.py         # Persistent event storage
│   │   ├── subscribers.py         # Event subscribers
│   │   ├── state_machine.py       # Trade state machine
│   │   ├── recovery.py            # Crash recovery
│   │   ├── replay.py              # Event replay engine
│   │   ├── reconciliation.py      # Broker ↔ internal state sync
│   │   ├── snapshots.py           # State snapshots
│   │   ├── projections.py         # Event projections
│   │   └── audit_log.py           # Audit trail
│   ├── portfolio/
│   │   └── optimizer.py           # Portfolio optimization
│   └── utils/
│       └── logger.py              # Structured logging (structlog)
├── backtesting/
│   ├── backtest.py                # Simple custom backtester
│   ├── bt_adapter.py              # Backtrader adapter
│   ├── vbt_adapter.py             # VectorBT adapter
│   ├── walk_forward.py            # Walk-forward optimization
│   ├── monte_carlo.py             # Monte Carlo simulation
│   └── portfolio_backtest.py      # Multi-symbol portfolio backtest
├── models/                        # Saved ML models (versioned .joblib)
├── data_cache/                    # SQLite DB + cached data + events
├── logs/                          # Application logs
├── tests/                         # Test suite
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
└── README.md
```

---

## Environment Configuration

All configuration is done via `.env` file (see `.env.example` for template):

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
| `TELEGRAM_CHAT_ID` | — | Your numeric chat ID (for auth) |

### Trading

| Variable | Default | Description |
|----------|---------|-------------|
| `ACTIVE_STRATEGY` | `momentum` | Default strategy |
| `TRADING_SYMBOLS` | `AAPL,MSFT,...` | Comma-separated symbols |
| `TIMEFRAME` | `1Hour` | Candle timeframe |
| `LOOKBACK_BARS` | `200` | Historical bars to analyze |
| `TRADING_INTERVAL` | `60` | Seconds between trading cycles |

### Automation

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_BACKTEST_INTERVAL_HOURS` | `6` | Auto-backtest every N hours (0=off) |
| `AUTO_TRAIN_INTERVAL_HOURS` | `24` | Auto-retrain ML every N hours (0=off) |
| `AUTO_SWEEP_INTERVAL_HOURS` | `12` | Auto parameter sweep every N hours (0=off) |
| `AUTO_TRAIN_BARS` | `1000` | Bars to use for auto-training |

---

## Alpaca Account Setup

1. **Sign up** at https://app.alpaca.markets/signup (free)
2. **Paper account** is auto-created with $100,000 virtual cash
3. Go to **API Keys** → Generate new key
4. Copy `API Key` and `Secret Key` to your `.env` file
5. For live: fund account + generate live API keys separately

---

## Docker Deployment

```bash
# Build
docker build -t algo-trader .

# Run (paper mode)
docker run -d --name trader --env-file .env algo-trader

# Run with live data feed
docker run -d --name trader --env-file .env -e ALPACA_DATA_FEED=sip algo-trader

# View logs
docker logs -f trader
```

---

## Safety Notes

- ⚠️ **Always start with paper trading** — validate your strategy first
- ⚠️ **Backtest thoroughly** before any live trading (use `/backtest`, `/walkforward`, `/montecarlo`)
- ⚠️ **Never risk more than you can afford to lose**
- The bot has built-in daily loss limits and circuit breakers
- Live mode requires typing "YES" as confirmation on CLI
- All trades are logged to SQLite for full audit trail
- Use `/pause` on Telegram to immediately halt auto-trading
- Use `/closeall` for emergency liquidation
- Risk halts are automatic — the bot notifies you and stops trading
- Model rollback (`/rollback`) lets you revert to known-good ML models
- The event store enables full crash recovery (`/recover`)

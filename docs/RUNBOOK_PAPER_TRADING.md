# Operational Runbook — Suez Algo Trader Paper Trading

## Overview

This runbook documents the operational readiness of the Suez Algo Trader for
automated paper trading. All items below have been verified and tested.

---

## 1. Training Singleton Enforcement

### Problem
Multiple training triggers (APScheduler, main loop auto-retrain, Telegram /train,
calibration monitor) could fire concurrently, producing duplicate
`ModelTrainingStarted` events and wasting resources.

### Solution
- **Distributed TrainingLock** (`src/ml/training_lock.py`) uses the shared cache
  backend (Redis in production, LocalCache for single-instance) to ensure exactly
  one training job runs at any time across all instances.
- **Instance identity** (hostname:PID) is logged when the lock is acquired.
- **All training entry points** now check `runtime_manager.is_training()` before
  launching a new pipeline.
- **Lock TTL** (1 hour) with heartbeat renewal prevents deadlocks if holder crashes.

### Verification
```bash
python -m pytest tests/test_training_lock.py tests/test_e2e_paper_trading.py::TestSchedulerSingleton -v
```

### Evidence
- `TestSchedulerSingleton::test_five_concurrent_training_attempts_only_one_wins` — 5 threads race, exactly 1 wins
- `TestOperationalEvidence::test_no_duplicate_training_events` — only 1 ModelTrainingStarted emitted

---

## 2. Git Commit Hash Lineage

### Problem
The `GIT_COMMIT_HASH` build arg was never passed during Docker build in CI,
and the `GIT_COMMIT` environment variable was not injected into the running
container. This caused governance metadata to have empty `git_commit` fields.

### Solution
- **CI Workflow** (`.github/workflows/deploy.yml`):
  - Added `build-args: GIT_COMMIT_HASH=${{ github.sha }}` to Docker build step
  - Added `GIT_COMMIT="${{ github.sha }}"` to container environment variables
- **Resolution order** in governance:
  1. `GIT_COMMIT` / `SOURCE_VERSION` / `GITHUB_SHA` env vars (Priority 1)
  2. `git rev-parse HEAD` CLI (Priority 2)
  3. `.git_commit` file embedded at build time (Priority 3)

### Verification
```bash
python -m pytest tests/test_e2e_paper_trading.py::TestGitCommitLineage -v
```

---

## 3. Scheduler Deduplication

### Problem
The main loop (line ~1546) could trigger `_train_ml_model()` without checking
`is_training()`, racing with the APScheduler `auto_train` job and calibration
drift handler.

### Fix Applied
- Main loop auto-retrain now checks `runtime_manager.is_training()` first
- Telegram trigger_train path uses `runtime_manager.train_model()` (which has lock)
- All paths converge to the governed `TrainingPipeline` with distributed lock

---

## 4. /buy & /sell Command Schema

### Problems Fixed
- **No quantity validation**: NaN, Inf, negative, zero accepted → now rejected
- **No symbol validation**: arbitrary strings passed to broker → now validated
- **Broker response not checked**: `order['id']` could KeyError on error dict → now handled
- **AlpacaBroker.market_order()** missing `client_order_id` parameter → added

### Verification
```bash
python -m pytest tests/test_e2e_paper_trading.py::TestTelegramCommandValidation -v
```

---

## 5. Signal → Verdict → Execution Path

The complete pipeline for paper trading:

```
Strategy.generate_signals()
    → adapt_signal() [LegacyTradeSignal → frozen TradeSignal]
    → Signal strength gate (≥ 0.55)
    → Existing position check
    → Intelligence orchestrator (optional)
    → Signal package gate (optional)
    → Position sizing
    → DecisionOrchestrator → DecisionContract
    → Risk Engine evaluation
    → Broker.market_order() or Broker.bracket_order()
    → OrderSubmitted event
```

### Verification
```bash
python -m pytest tests/test_e2e_paper_trading.py::TestSignalToExecution -v
```

---

## 6. Paper Order Execution

### Verification
```bash
python -m pytest tests/test_e2e_paper_trading.py::TestPaperOrderPlacement -v
```

---

## 7. Recovery & Reconciliation

- **Training lock TTL**: Auto-expires (1 hour) if holder crashes
- **Lock released on failure**: Even if `_execute_pipeline` raises, the finally block releases
- **Portfolio reconciliation**: Runs every 5 cycles (~5 minutes) with auto-fix

### Verification
```bash
python -m pytest tests/test_e2e_paper_trading.py::TestRecovery -v
python -m pytest tests/test_e2e_paper_trading.py::TestReconciliation -v
```

---

## 8. Full Validation Suite

Run the complete end-to-end paper trading validation:

```bash
python -m pytest tests/test_e2e_paper_trading.py tests/test_training_lock.py tests/test_protective_exits.py -v
```

Expected: **77 tests pass** covering:
- Telegram command validation (7 tests)
- Signal-to-execution pipeline (2 tests)
- Paper order placement (3 tests)
- Scheduler singleton (4 tests)
- Git commit lineage (4 tests)
- Reconciliation (2 tests)
- Recovery (3 tests)
- Operational evidence (3 tests)
- Training lock unit tests (15 tests)
- Protective exits (34 tests)

---

## 9. Protective Exits — SL/TP for Every Position

### Behavior

Every position opened by the system (signal-driven OR manual) now receives
protective stop-loss and take-profit levels submitted as bracket orders.

### Order Entry Paths

| Path | Entry Method | SL/TP Source |
|------|-------------|--------------|
| Signal-driven (strategy) | `bracket_order()` | Strategy ATR → DecisionContract → defaults |
| Manual `/buy` or `/sell` | `bracket_order()` | ProtectiveExits module (percentage defaults) |
| CLI `--train` mode | N/A (training only) | N/A |

### Resolution Priority (SL/TP Source)

1. **Strategy-provided** — ATR-based levels from momentum/ML strategy signals
2. **DecisionContract** — Contract may override with its own recommended levels
3. **ATR-based** — If ATR value available but no strategy hint
4. **Percentage defaults** — Always available fallback (3% SL, 6% TP)

### Configuration (`ProtectiveExitConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_stop_loss_pct` | 0.03 (3%) | Default SL distance from entry |
| `default_take_profit_pct` | 0.06 (6%) | Default TP distance from entry |
| `max_stop_loss_pct` | 0.10 (10%) | Maximum allowed SL distance |
| `min_stop_loss_pct` | 0.005 (0.5%) | Minimum SL distance (prevents too-tight stops) |
| `max_take_profit_pct` | 0.30 (30%) | Maximum TP distance |
| `min_take_profit_pct` | 0.01 (1%) | Minimum TP distance |
| `enforce_bracket` | True | Always submit as bracket order |
| `min_risk_reward` | 1.5 | Minimum TP/SL ratio |

### Validation

```bash
python -m pytest tests/test_protective_exits.py -v
```

---

## 10. Pre-Live Checklist

Before enabling live trading:

- [ ] All 77 validation tests pass
- [ ] Paper trading has run ≥48 hours without manual intervention
- [ ] Telegram /buy and /sell submit bracket orders with SL/TP
- [ ] No duplicate ModelTrainingStarted events in logs
- [ ] Git commit hash present in all governance lineage records
- [ ] Reconciliation shows zero drift
- [ ] Risk engine approves and rejects correctly
- [ ] No stale locks older than 1 hour in cache
- [ ] Daily summary Telegram notifications arrive at 16:05 ET

---

## 11. Monitoring Commands

```
/status     — Current bot status, positions, PnL
/health     — Component health check
/training   — Current training pipeline status
/governance — Model lineage and git commit info
/reconcile  — Force portfolio reconciliation
```

---

## Files Modified

| File | Change |
|------|--------|
| `src/ml/training_lock.py` | NEW — Distributed training singleton lock |
| `src/ml/training_pipeline.py` | Integrated TrainingLock, added `training_lock` param |
| `.github/workflows/deploy.yml` | Added GIT_COMMIT_HASH build arg + GIT_COMMIT env var |
| `main.py` | Fixed duplicate scheduler race conditions |
| `src/notifications/telegram_bot.py` | Fixed /buy & /sell: validation + bracket orders with SL/TP |
| `src/broker/alpaca_client.py` | Added `client_order_id` to match BrokerProtocol |
| `src/risk/protective_exits.py` | NEW — Shared protective exits module (SL/TP for all entries) |
| `tests/test_training_lock.py` | NEW — Training lock unit tests |
| `tests/test_e2e_paper_trading.py` | NEW — End-to-end validation suite |
| `tests/test_protective_exits.py` | NEW — Protective exits regression tests |
| `docs/RUNBOOK_PAPER_TRADING.md` | NEW — This runbook |

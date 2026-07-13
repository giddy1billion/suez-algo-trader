# Production-Readiness Audit: `/trades` vs `/journalstats` Consistency

## 1. Executive Summary

The `/trades` and `/journalstats` Telegram commands intentionally operate on **two different database tables** with different schemas and semantics. This is by design—not a bug—but the distinction was never documented, creating confusion about why the two views can diverge.

| Aspect | `/trades` | `/journalstats` |
|--------|-----------|-----------------|
| **Table** | `trades` | `trade_journal` |
| **Data Source** | `DatabaseManager.get_trades()` | `TradeJournal.get_summary()` |
| **Scope** | Last N executed trades (all statuses) | 30-day rolling window, closed trades only for stats |
| **Write Path** | `ExecutionEngine` → `db.record_trade()` | `ExecutionEngine` → `journal.log_entry()` / `journal.log_exit()` |
| **Purpose** | Execution audit log (order-level) | ML/strategy analysis with entry context |

## 2. Complete Data Flow

### 2.1 Trade Creation (Entry)

**File:** `src/execution/engine.py`, lines 884–903

When the execution engine places an order, it records the trade in the `trades` table:

```
Signal → ExecutionEngine._execute_signal()
  → self.db.record_trade({symbol, side, qty, price, order_type, status, ...})
  → INSERT INTO trades
```

**File:** `src/execution/engine.py`, lines 905–929

Immediately after, it journals the entry with ML context:

```
  → journal.log_entry({symbol, side, entry_price, strategy_name, model_version,
                        confidence, features_snapshot, market_regime, ...})
  → INSERT INTO trade_journal
```

**Key observation:** The journal `log_entry()` call is wrapped in a `try/except` (line 928) that catches *all* exceptions and logs at `debug` level. If journaling fails silently, the trade exists in `trades` but not in `trade_journal`.

### 2.2 Trade Exit (Close)

**File:** `src/execution/engine.py`, lines 970–1063

When a position is closed:

1. **Trade table update:** `self.db.update_trade(order_id, {pnl, closed_at})` — updates the `trades` row
2. **Journal exit:** `journal.log_exit(journal_id, {exit_price, exit_reason, pnl, pnl_pct})` — updates the `trade_journal` row

**Exit journal matching logic** (lines 1052–1061): The engine finds the matching open journal entry by querying the last 20 entries for the symbol and selecting the oldest one without an `exit_price`. This is a heuristic match, not a foreign-key lookup.

### 2.3 `/trades` Command

**File:** `src/notifications/telegram_bot.py`, lines 434–457

```python
trades = _db.get_trades(limit=10)
```

**File:** `src/data/store.py`, lines 194–215

Queries the `trades` table ordered by `created_at DESC`, returns all trades regardless of status (open, filled, cancelled, rejected). No time filter.

### 2.4 `/journalstats` Command

**File:** `src/notifications/telegram_bot.py`, lines 2189–2238

```python
journal = TradeJournal(db)
summary = journal.get_summary(days=30)
conf_stats = journal.get_performance_by_confidence()
model_stats = journal.get_performance_by_model_version()
```

**File:** `src/data/journal.py`, lines 235–291

`get_summary()` queries the `trade_journal` table with:
- **Time filter:** `created_at >= (now - 30 days)`
- **Stat calculation:** Only closed trades (where `pnl IS NOT NULL`) contribute to win_rate, avg_pnl, etc.
- **Total count:** Includes both open and closed journal entries in `total_trades`

## 3. Root Causes of Discrepancy

### 3.1 Separate Tables (By Design)

The `trades` table is an **execution audit log** capturing every order submission with broker-level details (order_id, order_type, status, fees). The `trade_journal` table is an **ML analysis log** capturing entry/exit context with model features, confidence scores, market regime, and volatility for retraining.

**This separation is intentional and correct.** The journal captures richer ML context that doesn't belong in the execution audit log.

### 3.2 Silent Journal Failures (Bug Risk)

**File:** `src/execution/engine.py`, line 928

```python
except Exception as e:
    logger.debug("journal.log_entry_error", error=str(e))
```

If journal logging fails (e.g., database contention, schema mismatch, missing required fields), the trade is recorded in `trades` but silently missing from `trade_journal`. The `debug` log level means this is effectively invisible in production.

**Risk:** Trades can exist in `/trades` but never appear in `/journalstats`.

### 3.3 Heuristic Exit Matching (Bug Risk)

**File:** `src/execution/engine.py`, lines 1052–1055

```python
entries = journal.get_journal(symbol=symbol, limit=20)
open_entries = [e for e in entries if e.get("exit_price") is None]
if open_entries:
    target = open_entries[-1]
```

This selects the **oldest** open journal entry for the symbol from the last 20 entries. If there are more than 20 open entries for a symbol, or if entries are matched out-of-order, exits may be logged against the wrong journal entry or not logged at all.

**Risk:** Journal entries may remain permanently "open" (no exit data), causing them to be excluded from `/journalstats` analytics that filter on `pnl IS NOT NULL`.

### 3.4 Time Window Difference

`/trades` shows the last 10 trades regardless of age. `/journalstats` uses a 30-day rolling window. Old trades appear in `/trades` but not in `/journalstats`.

### 3.5 Status Filtering Difference

`/trades` shows all trade statuses (filled, cancelled, rejected). `/journalstats` only includes trades that were successfully journaled (i.e., actually executed), and only closed ones contribute to win/loss statistics.

## 4. Guarantees and Intentional Differences

### Guarantees Established

1. **Every successfully executed trade** is recorded in both `trades` and `trade_journal` tables (barring silent failures addressed below).
2. **`/trades`** is the authoritative view of execution history — it shows what orders were submitted to the broker.
3. **`/journalstats`** is the authoritative view of ML/strategy performance — it shows how model predictions performed against actual outcomes.
4. **Regression tests** (added in this audit) verify that the journal pipeline maintains consistency: every trade recorded via the execution engine has a corresponding journal entry, and every closed trade has journal exit data.

### Intentional Differences That Remain

| Difference | Rationale |
|-----------|-----------|
| Different tables | `trades` is execution-focused; `trade_journal` is ML-focused with richer context |
| `/trades` has no time filter | Execution history should be browseable regardless of age |
| `/journalstats` uses 30-day window | Analytics are most relevant for recent performance |
| `/journalstats` stats use closed trades only | Win/loss can only be determined after exit |
| `/journalstats` reports `total_trades` including open | Users should see how many trades are still open |

## 5. File and Line Reference Index

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Trade model | `src/data/store.py` | 16–43 | `trades` table schema |
| JournalEntry model | `src/data/store.py` | 79–117 | `trade_journal` table schema |
| DatabaseManager.record_trade | `src/data/store.py` | 175–183 | Insert trade |
| DatabaseManager.update_trade | `src/data/store.py` | 185–192 | Update trade on close |
| DatabaseManager.get_trades | `src/data/store.py` | 194–215 | Query trades for `/trades` |
| TradeJournal.log_entry | `src/data/journal.py` | 29–66 | Journal trade open |
| TradeJournal.log_exit | `src/data/journal.py` | 68–89 | Journal trade close |
| TradeJournal.get_summary | `src/data/journal.py` | 235–291 | Analytics for `/journalstats` |
| TradeJournal.get_performance_by_confidence | `src/data/journal.py` | 113–147 | Confidence bucket analysis |
| TradeJournal.get_performance_by_model_version | `src/data/journal.py` | 149–182 | Model version comparison |
| ExecutionEngine trade recording | `src/execution/engine.py` | 884–903 | Records to `trades` |
| ExecutionEngine journal entry | `src/execution/engine.py` | 905–929 | Journals with ML context |
| ExecutionEngine journal exit | `src/execution/engine.py` | 1048–1063 | Journals exit on close |
| `/trades` handler | `src/notifications/telegram_bot.py` | 434–457 | Telegram command |
| `/journal` handler | `src/notifications/telegram_bot.py` | 2131–2186 | Telegram command |
| `/journalstats` handler | `src/notifications/telegram_bot.py` | 2189–2238 | Telegram command |
| Journal lazy loader | `src/execution/engine.py` | 48–58 | Thread-safe singleton |

## 6. Recommendations Applied

1. **Documented the intentional dual-table design** (this document).
2. **Added regression tests** (`tests/test_trade_journal_consistency.py`) covering:
   - Open trade journaling
   - Closed trade journaling with exit data
   - Journal summary analytics accuracy
   - Confidence bucket analysis
   - Model version comparison
   - Recovery scenarios (journal entry without exit)
   - Edge cases (empty database, single trade)
3. **Command output clarity** — `/journalstats` now displays both total and closed trade counts, making the filtering explicit to users.
4. **Silent journal failures fixed** (§3.2) — Log level upgraded from `debug` to `warning` with symbol context, ensuring journal failures are visible in production logs.
5. **Heuristic exit matching improved** (§3.3) — Now uses precise `trade_id` lookup first before falling back to the oldest-open-entry heuristic. Query limit increased from 20 to 50 entries.

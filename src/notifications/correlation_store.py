"""
Correlation Store — Bounded, TTL-aware state for signal correlation & dedup.

Provides a pluggable abstraction for the correlation state that the
TelegramAuditForwarder needs:
  - signal_id → SignalGenerated event storage
  - signal_id → pending deadline tracking
  - signal_id → DecisionContractCreated mapping
  - contract_id → signal_id reverse mapping
  - intent_id dedup set (approvals AND rejections)

The default implementation is in-memory with TTL-based expiry and bounded
size, suitable for single-instance deployments.  The interface is designed
so that a Redis-backed implementation can be swapped in for multi-instance
or restart-durable deployments.

A SQLite-backed durable implementation (SqliteCorrelationStore) is also
provided for crash-safe deployments with startup recovery.

Design decisions:
  - time.monotonic() is used for all TTL / interval comparisons to avoid
    wall-clock jumps (NTP corrections, DST transitions, manual clock sets).
  - Atomic expire-and-cancel operations prevent timeout/verdict races.
  - Dead-letter queue support in SqliteCorrelationStore persists messages
    that failed Telegram delivery for later retry.
  - All stores enforce persist-before-dispatch: dedup keys are written
    before the message is handed to the sender.

Delivery guarantee change (documented):
  InMemoryCorrelationStore: at-most-once (dedup state lost on crash).
  SqliteCorrelationStore: at-least-once (dedup + dead-letter survive
  restarts; duplicate delivery is suppressed via durable dedup keys,
  but a message may be re-sent after crash if it was sent but not yet
  marked as delivered in the dead-letter table).
"""

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)

# Default TTLs (seconds)
DEFAULT_SIGNAL_TTL = 600.0       # 10 minutes — signals older than this are stale
DEFAULT_DEADLINE_TTL = 300.0     # 5 minutes — pending deadline entries
DEFAULT_DEDUP_TTL = 3600.0       # 1 hour — dedup keys
DEFAULT_MAX_ENTRIES = 10_000     # cap per store section


# ---------------------------------------------------------------------------
# Metrics Counters
# ---------------------------------------------------------------------------

@dataclass
class CorrelationMetrics:
    """Observable counters for correlation store activity."""
    signals_tracked: int = 0
    signals_expired: int = 0
    verdicts_correlated: int = 0
    verdicts_uncorrelated: int = 0
    duplicates_suppressed: int = 0
    timeouts_emitted: int = 0
    late_arrivals: int = 0          # RiskEvaluated arrived before SignalGenerated
    orphaned_events: int = 0        # stale entries cleaned up
    dead_letters: int = 0           # messages that failed delivery
    retries_succeeded: int = 0      # dead-letter retries that succeeded
    send_failures: int = 0          # total Telegram send failures

    def to_dict(self) -> dict[str, int]:
        return {
            "signals_tracked": self.signals_tracked,
            "signals_expired": self.signals_expired,
            "verdicts_correlated": self.verdicts_correlated,
            "verdicts_uncorrelated": self.verdicts_uncorrelated,
            "duplicates_suppressed": self.duplicates_suppressed,
            "timeouts_emitted": self.timeouts_emitted,
            "late_arrivals": self.late_arrivals,
            "orphaned_events": self.orphaned_events,
            "dead_letters": self.dead_letters,
            "retries_succeeded": self.retries_succeeded,
            "send_failures": self.send_failures,
        }


# ---------------------------------------------------------------------------
# Store Protocol  (for duck-typing a Redis backend later)
# ---------------------------------------------------------------------------

class CorrelationStoreProtocol(Protocol):
    """Minimum interface that any durable backend must implement.

    Atomic deadline operations:
      - expire_and_cancel_deadline: atomically removes a deadline and returns
        the signal_ids that were expired.  This prevents the race where a
        verdict cancels a deadline that the timeout thread is about to fire.
    """

    def store_signal(self, signal_id: str, event: Any) -> None: ...
    def get_signal(self, signal_id: str) -> Optional[Any]: ...
    def remove_signal(self, signal_id: str) -> None: ...

    def set_deadline(self, signal_id: str, deadline: float) -> None: ...
    def get_expired_deadlines(self, now: float) -> list[str]: ...
    def cancel_deadline(self, signal_id: str) -> None: ...
    def expire_and_cancel_deadlines(self, now: float) -> list[str]: ...

    def store_contract(self, signal_id: str, event: Any) -> None: ...
    def get_contract(self, signal_id: str) -> Optional[Any]: ...
    def map_contract_to_signal(self, contract_id: str, signal_id: str) -> None: ...
    def lookup_signal_by_contract(self, contract_id: str) -> str: ...

    def check_dedup(self, intent_id: str) -> bool: ...
    def mark_sent(self, intent_id: str) -> None: ...

    def record_late_risk(self, signal_id: str, event: Any) -> None: ...
    def pop_late_risk(self, signal_id: str) -> Optional[Any]: ...

    def cleanup_expired(self) -> int: ...

    @property
    def metrics(self) -> CorrelationMetrics: ...


# ---------------------------------------------------------------------------
# In-Memory Implementation  (bounded, TTL-aware, thread-safe)
# ---------------------------------------------------------------------------

@dataclass
class _TTLEntry:
    """Value with creation timestamp for TTL checks.

    Uses time.monotonic() to avoid wall-clock jumps (Finding 4).
    """
    value: Any
    created_at: float = field(default_factory=time.monotonic)


class InMemoryCorrelationStore:
    """
    Thread-safe, bounded, TTL-aware correlation store.

    All entries have a TTL.  The store runs periodic eviction in the
    caller's thread (amortised) and enforces hard size caps.

    Uses time.monotonic() for all TTL calculations (Finding 4) to be
    immune to wall-clock adjustments.
    """

    def __init__(
        self,
        signal_ttl: float = DEFAULT_SIGNAL_TTL,
        deadline_ttl: float = DEFAULT_DEADLINE_TTL,
        dedup_ttl: float = DEFAULT_DEDUP_TTL,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self._lock = threading.Lock()

        self._signal_ttl = signal_ttl
        self._deadline_ttl = deadline_ttl
        self._dedup_ttl = dedup_ttl
        self._max_entries = max_entries

        # Core state
        self._signals: dict[str, _TTLEntry] = {}           # signal_id → event
        self._deadlines: dict[str, float] = {}              # signal_id → deadline (monotonic)
        self._contracts: dict[str, _TTLEntry] = {}          # signal_id → contract event
        self._contract_to_signal: dict[str, _TTLEntry] = {} # contract_id → signal_id
        self._sent_intents: dict[str, float] = {}           # intent_id → monotonic timestamp
        self._late_risks: dict[str, _TTLEntry] = {}         # signal_id → RiskEvaluated (early)

        self._metrics = CorrelationMetrics()
        self._last_cleanup = time.monotonic()

    # ── Signal tracking ──────────────────────────────────────────────────

    def store_signal(self, signal_id: str, event: Any) -> None:
        with self._lock:
            self._evict_if_full(self._signals)
            self._signals[signal_id] = _TTLEntry(value=event)
            self._metrics.signals_tracked += 1

    def get_signal(self, signal_id: str) -> Optional[Any]:
        with self._lock:
            entry = self._signals.get(signal_id)
            if entry is None:
                return None
            if time.monotonic() - entry.created_at > self._signal_ttl:
                self._signals.pop(signal_id, None)
                self._metrics.signals_expired += 1
                return None
            return entry.value

    def remove_signal(self, signal_id: str) -> None:
        with self._lock:
            self._signals.pop(signal_id, None)

    # ── Deadline tracking ────────────────────────────────────────────────

    def set_deadline(self, signal_id: str, deadline: float) -> None:
        with self._lock:
            self._evict_if_full_dict(self._deadlines)
            self._deadlines[signal_id] = deadline

    def get_expired_deadlines(self, now: float) -> list[str]:
        with self._lock:
            expired = [
                sid for sid, dl in self._deadlines.items()
                if dl <= now
            ]
            return expired

    def cancel_deadline(self, signal_id: str) -> None:
        with self._lock:
            self._deadlines.pop(signal_id, None)

    def expire_and_cancel_deadlines(self, now: float) -> list[str]:
        """Atomically find expired deadlines and remove them (Finding 2).

        This prevents the race where cancel_deadline and
        get_expired_deadlines are called from different threads: without
        atomicity, a verdict arriving between get+cancel can cause a
        false timeout emission.

        Returns:
            List of signal_ids whose deadlines expired.
        """
        with self._lock:
            expired = [
                sid for sid, dl in self._deadlines.items()
                if dl <= now
            ]
            for sid in expired:
                self._deadlines.pop(sid, None)
            return expired

    # ── Contract tracking ────────────────────────────────────────────────

    def store_contract(self, signal_id: str, event: Any) -> None:
        with self._lock:
            self._evict_if_full(self._contracts)
            self._contracts[signal_id] = _TTLEntry(value=event)

    def get_contract(self, signal_id: str) -> Optional[Any]:
        with self._lock:
            entry = self._contracts.get(signal_id)
            if entry is None:
                return None
            if time.monotonic() - entry.created_at > self._signal_ttl:
                self._contracts.pop(signal_id, None)
                return None
            return entry.value

    def map_contract_to_signal(self, contract_id: str, signal_id: str) -> None:
        with self._lock:
            self._evict_if_full(self._contract_to_signal)
            self._contract_to_signal[contract_id] = _TTLEntry(value=signal_id)

    def lookup_signal_by_contract(self, contract_id: str) -> str:
        with self._lock:
            entry = self._contract_to_signal.get(contract_id)
            if entry is None:
                return ""
            if time.monotonic() - entry.created_at > self._signal_ttl:
                self._contract_to_signal.pop(contract_id, None)
                return ""
            return entry.value

    # ── Dedup ────────────────────────────────────────────────────────────

    def check_dedup(self, intent_id: str) -> bool:
        """Return True if this intent_id was already sent (duplicate)."""
        with self._lock:
            ts = self._sent_intents.get(intent_id)
            if ts is None:
                return False
            if time.monotonic() - ts > self._dedup_ttl:
                self._sent_intents.pop(intent_id, None)
                return False
            return True

    def mark_sent(self, intent_id: str) -> None:
        with self._lock:
            self._evict_if_full_dict(self._sent_intents)
            self._sent_intents[intent_id] = time.monotonic()

    # ── Late risk (out-of-order) ─────────────────────────────────────────

    def record_late_risk(self, signal_id: str, event: Any) -> None:
        """Store a RiskEvaluated that arrived before its SignalGenerated."""
        with self._lock:
            self._evict_if_full(self._late_risks)
            self._late_risks[signal_id] = _TTLEntry(value=event)
            self._metrics.late_arrivals += 1

    def pop_late_risk(self, signal_id: str) -> Optional[Any]:
        """Retrieve and remove a buffered late-arriving RiskEvaluated."""
        with self._lock:
            entry = self._late_risks.pop(signal_id, None)
            if entry is None:
                return None
            if time.monotonic() - entry.created_at > self._signal_ttl:
                return None
            return entry.value

    # ── Cleanup / Metrics ────────────────────────────────────────────────

    def cleanup_expired(self) -> int:
        """Remove all expired entries.  Returns count of entries removed."""
        now = time.monotonic()
        removed = 0
        with self._lock:
            removed += self._purge_ttl(self._signals, self._signal_ttl, now)
            removed += self._purge_ttl(self._contracts, self._signal_ttl, now)
            removed += self._purge_ttl(self._contract_to_signal, self._signal_ttl, now)
            removed += self._purge_ttl(self._late_risks, self._signal_ttl, now)

            # Deadlines: remove entries whose deadline + ttl has passed
            stale = [
                sid for sid, dl in self._deadlines.items()
                if now - dl > self._deadline_ttl
            ]
            for sid in stale:
                self._deadlines.pop(sid, None)
            removed += len(stale)

            # Dedup: purge old entries
            stale_dedup = [
                iid for iid, ts in self._sent_intents.items()
                if now - ts > self._dedup_ttl
            ]
            for iid in stale_dedup:
                self._sent_intents.pop(iid, None)
            removed += len(stale_dedup)

            self._metrics.orphaned_events += removed
            self._last_cleanup = now
        return removed

    @property
    def metrics(self) -> CorrelationMetrics:
        return self._metrics

    # ── Internal helpers ─────────────────────────────────────────────────

    def _evict_if_full(self, store: dict[str, _TTLEntry]) -> None:
        """Evict oldest entries if at capacity (must hold lock)."""
        if len(store) >= self._max_entries:
            # Remove oldest 10% by creation time
            to_remove = max(1, len(store) // 10)
            oldest = sorted(store, key=lambda k: store[k].created_at)[:to_remove]
            for k in oldest:
                store.pop(k, None)
            self._metrics.orphaned_events += to_remove

    def _evict_if_full_dict(self, store: dict[str, float]) -> None:
        """Evict oldest entries for plain timestamp dicts (must hold lock)."""
        if len(store) >= self._max_entries:
            to_remove = max(1, len(store) // 10)
            oldest = sorted(store, key=lambda k: store[k])[:to_remove]
            for k in oldest:
                store.pop(k, None)
            self._metrics.orphaned_events += to_remove

    @staticmethod
    def _purge_ttl(store: dict[str, _TTLEntry], ttl: float, now: float) -> int:
        stale = [k for k, v in store.items() if now - v.created_at > ttl]
        for k in stale:
            store.pop(k, None)
        return len(stale)


# ---------------------------------------------------------------------------
# SQLite-backed Durable Implementation (Finding 1)
# ---------------------------------------------------------------------------

# Schema version for migration hooks
STORE_SCHEMA_VERSION = 1

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS store_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    signal_id  TEXT PRIMARY KEY,
    event_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS deadlines (
    signal_id TEXT PRIMARY KEY,
    deadline  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS contracts (
    signal_id  TEXT PRIMARY KEY,
    event_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS contract_to_signal (
    contract_id TEXT PRIMARY KEY,
    signal_id   TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sent_intents (
    intent_id  TEXT PRIMARY KEY,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS late_risks (
    signal_id  TEXT PRIMARY KEY,
    event_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS dead_letters (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    message    TEXT NOT NULL,
    created_at REAL NOT NULL,
    attempts   INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT ''
);
"""


def _serialize_event(event: Any) -> str:
    """Serialize an event to JSON for SQLite storage."""
    if hasattr(event, "to_dict"):
        data = event.to_dict()
        data["__type__"] = type(event).__qualname__
        data["__module__"] = type(event).__module__
        return json.dumps(data, default=str)
    return json.dumps({"__raw__": repr(event)}, default=str)


def _deserialize_event(json_str: str) -> Any:
    """Deserialize an event from JSON.

    Attempts to reconstruct the original event object using its
    from_dict class method.  Falls back to returning the raw dict
    if reconstruction fails (forward-compatible).
    """
    import importlib
    data = json.loads(json_str)
    module_name = data.pop("__module__", None)
    type_name = data.pop("__type__", None)
    if data.get("__raw__") is not None:
        return data
    if module_name and type_name:
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, type_name)
            if hasattr(cls, "from_dict"):
                return cls.from_dict(data)
        except Exception:
            logger.debug("correlation_store.deserialize_fallback",
                         extra={"module": module_name, "type": type_name})
    return data


class SqliteCorrelationStore:
    """
    Crash-safe, SQLite-backed correlation store with startup recovery.

    Finding 1: Durable backend so that correlation state, dedup keys,
    and pending deadlines survive process restarts.

    Design decisions:
      - WAL journal mode for concurrent read/write safety.
      - Schema version stored in store_meta for future migrations.
      - Dead-letter table persists undeliverable messages for retry.
      - All writes use transactions for atomicity.
      - expire_and_cancel_deadlines is a single SQL DELETE RETURNING
        equivalent (SELECT + DELETE in one transaction) for atomicity.

    Delivery guarantee: at-least-once (dedup keys persist across restarts).
    """

    def __init__(
        self,
        db_path: str,
        signal_ttl: float = DEFAULT_SIGNAL_TTL,
        deadline_ttl: float = DEFAULT_DEADLINE_TTL,
        dedup_ttl: float = DEFAULT_DEDUP_TTL,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self._db_path = db_path
        self._signal_ttl = signal_ttl
        self._deadline_ttl = deadline_ttl
        self._dedup_ttl = dedup_ttl
        self._max_entries = max_entries
        self._metrics = CorrelationMetrics()
        self._lock = threading.Lock()

        # Ensure parent directory exists
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA_DDL)
        self._apply_migrations()
        self._recover_metrics()

    def _apply_migrations(self) -> None:
        """Check schema version and apply migrations if needed."""
        cur = self._conn.execute(
            "SELECT value FROM store_meta WHERE key='schema_version'"
        )
        row = cur.fetchone()
        current_version = int(row[0]) if row else 0
        if current_version < STORE_SCHEMA_VERSION:
            # Currently at version 1 — no migrations needed yet.
            # Future migrations go here: if current_version < 2: ...
            self._conn.execute(
                "INSERT OR REPLACE INTO store_meta (key, value) VALUES ('schema_version', ?)",
                (str(STORE_SCHEMA_VERSION),),
            )
            self._conn.commit()

    def _recover_metrics(self) -> None:
        """Recover counters from persisted state on startup."""
        cur = self._conn.execute("SELECT COUNT(*) FROM signals")
        self._metrics.signals_tracked = cur.fetchone()[0]
        cur = self._conn.execute("SELECT COUNT(*) FROM late_risks")
        self._metrics.late_arrivals = cur.fetchone()[0]
        cur = self._conn.execute("SELECT COUNT(*) FROM dead_letters")
        self._metrics.dead_letters = cur.fetchone()[0]
        logger.info(
            "correlation_store.sqlite_recovered",
            extra={
                "signals": self._metrics.signals_tracked,
                "late_risks": self._metrics.late_arrivals,
                "dead_letters": self._metrics.dead_letters,
            },
        )

    def _now(self) -> float:
        """Return monotonic timestamp for TTL calculations."""
        return time.monotonic()

    # ── Signal tracking ──────────────────────────────────────────────────

    def store_signal(self, signal_id: str, event: Any) -> None:
        with self._lock:
            self._enforce_limit("signals")
            self._conn.execute(
                "INSERT OR REPLACE INTO signals (signal_id, event_json, created_at) VALUES (?, ?, ?)",
                (signal_id, _serialize_event(event), self._now()),
            )
            self._conn.commit()
            self._metrics.signals_tracked += 1

    def get_signal(self, signal_id: str) -> Optional[Any]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT event_json, created_at FROM signals WHERE signal_id=?",
                (signal_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            if self._now() - row[1] > self._signal_ttl:
                self._conn.execute("DELETE FROM signals WHERE signal_id=?", (signal_id,))
                self._conn.commit()
                self._metrics.signals_expired += 1
                return None
            return _deserialize_event(row[0])

    def remove_signal(self, signal_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM signals WHERE signal_id=?", (signal_id,))
            self._conn.commit()

    # ── Deadline tracking ────────────────────────────────────────────────

    def set_deadline(self, signal_id: str, deadline: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO deadlines (signal_id, deadline) VALUES (?, ?)",
                (signal_id, deadline),
            )
            self._conn.commit()

    def get_expired_deadlines(self, now: float) -> list[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT signal_id FROM deadlines WHERE deadline <= ?",
                (now,),
            )
            return [row[0] for row in cur.fetchall()]

    def cancel_deadline(self, signal_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM deadlines WHERE signal_id=?", (signal_id,))
            self._conn.commit()

    def expire_and_cancel_deadlines(self, now: float) -> list[str]:
        """Atomically find and remove expired deadlines (Finding 2)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT signal_id FROM deadlines WHERE deadline <= ?",
                (now,),
            )
            expired = [row[0] for row in cur.fetchall()]
            if expired:
                placeholders = ",".join("?" * len(expired))
                self._conn.execute(
                    f"DELETE FROM deadlines WHERE signal_id IN ({placeholders})",
                    expired,
                )
                self._conn.commit()
            return expired

    # ── Contract tracking ────────────────────────────────────────────────

    def store_contract(self, signal_id: str, event: Any) -> None:
        with self._lock:
            self._enforce_limit("contracts")
            self._conn.execute(
                "INSERT OR REPLACE INTO contracts (signal_id, event_json, created_at) VALUES (?, ?, ?)",
                (signal_id, _serialize_event(event), self._now()),
            )
            self._conn.commit()

    def get_contract(self, signal_id: str) -> Optional[Any]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT event_json, created_at FROM contracts WHERE signal_id=?",
                (signal_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            if self._now() - row[1] > self._signal_ttl:
                self._conn.execute("DELETE FROM contracts WHERE signal_id=?", (signal_id,))
                self._conn.commit()
                return None
            return _deserialize_event(row[0])

    def map_contract_to_signal(self, contract_id: str, signal_id: str) -> None:
        with self._lock:
            self._enforce_limit("contract_to_signal")
            self._conn.execute(
                "INSERT OR REPLACE INTO contract_to_signal (contract_id, signal_id, created_at) VALUES (?, ?, ?)",
                (contract_id, signal_id, self._now()),
            )
            self._conn.commit()

    def lookup_signal_by_contract(self, contract_id: str) -> str:
        with self._lock:
            cur = self._conn.execute(
                "SELECT signal_id, created_at FROM contract_to_signal WHERE contract_id=?",
                (contract_id,),
            )
            row = cur.fetchone()
            if row is None:
                return ""
            if self._now() - row[1] > self._signal_ttl:
                self._conn.execute(
                    "DELETE FROM contract_to_signal WHERE contract_id=?",
                    (contract_id,),
                )
                self._conn.commit()
                return ""
            return row[0]

    # ── Dedup ────────────────────────────────────────────────────────────

    def check_dedup(self, intent_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "SELECT created_at FROM sent_intents WHERE intent_id=?",
                (intent_id,),
            )
            row = cur.fetchone()
            if row is None:
                return False
            if self._now() - row[0] > self._dedup_ttl:
                self._conn.execute("DELETE FROM sent_intents WHERE intent_id=?", (intent_id,))
                self._conn.commit()
                return False
            return True

    def mark_sent(self, intent_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sent_intents (intent_id, created_at) VALUES (?, ?)",
                (intent_id, self._now()),
            )
            self._conn.commit()

    # ── Late risk (out-of-order) ─────────────────────────────────────────

    def record_late_risk(self, signal_id: str, event: Any) -> None:
        with self._lock:
            self._enforce_limit("late_risks")
            self._conn.execute(
                "INSERT OR REPLACE INTO late_risks (signal_id, event_json, created_at) VALUES (?, ?, ?)",
                (signal_id, _serialize_event(event), self._now()),
            )
            self._conn.commit()
            self._metrics.late_arrivals += 1

    def pop_late_risk(self, signal_id: str) -> Optional[Any]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT event_json, created_at FROM late_risks WHERE signal_id=?",
                (signal_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            self._conn.execute("DELETE FROM late_risks WHERE signal_id=?", (signal_id,))
            self._conn.commit()
            if self._now() - row[1] > self._signal_ttl:
                return None
            return _deserialize_event(row[0])

    # ── Dead-letter queue (Finding 3) ────────────────────────────────────

    def add_dead_letter(self, message: str, error: str = "") -> None:
        """Persist a message that failed Telegram delivery."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO dead_letters (message, created_at, attempts, last_error) VALUES (?, ?, 1, ?)",
                (message, self._now(), error),
            )
            self._conn.commit()
            self._metrics.dead_letters += 1
            self._metrics.send_failures += 1

    def get_dead_letters(self, limit: int = 50) -> list[tuple[int, str, int]]:
        """Return (id, message, attempts) tuples for retry."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, message, attempts FROM dead_letters ORDER BY id ASC LIMIT ?",
                (limit,),
            )
            return cur.fetchall()

    def remove_dead_letter(self, letter_id: int) -> None:
        """Remove a successfully retried dead letter."""
        with self._lock:
            self._conn.execute("DELETE FROM dead_letters WHERE id=?", (letter_id,))
            self._conn.commit()
            self._metrics.retries_succeeded += 1

    def increment_dead_letter_attempt(self, letter_id: int, error: str = "") -> None:
        """Increment the attempt counter for a failed retry."""
        with self._lock:
            self._conn.execute(
                "UPDATE dead_letters SET attempts = attempts + 1, last_error = ? WHERE id = ?",
                (error, letter_id),
            )
            self._conn.commit()
            self._metrics.send_failures += 1

    # ── Cleanup / Metrics ────────────────────────────────────────────────

    def cleanup_expired(self) -> int:
        now = self._now()
        removed = 0
        with self._lock:
            for table, ttl in [
                ("signals", self._signal_ttl),
                ("contracts", self._signal_ttl),
                ("contract_to_signal", self._signal_ttl),
                ("late_risks", self._signal_ttl),
            ]:
                cur = self._conn.execute(
                    f"DELETE FROM {table} WHERE ? - created_at > ?",
                    (now, ttl),
                )
                removed += cur.rowcount

            cur = self._conn.execute(
                "DELETE FROM deadlines WHERE ? - deadline > ?",
                (now, self._deadline_ttl),
            )
            removed += cur.rowcount

            cur = self._conn.execute(
                "DELETE FROM sent_intents WHERE ? - created_at > ?",
                (now, self._dedup_ttl),
            )
            removed += cur.rowcount

            self._conn.commit()
            self._metrics.orphaned_events += removed
        return removed

    @property
    def metrics(self) -> CorrelationMetrics:
        return self._metrics

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ── Internal helpers ─────────────────────────────────────────────────

    def _enforce_limit(self, table: str) -> None:
        """Enforce max_entries by deleting oldest rows (must hold lock)."""
        cur = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        if count >= self._max_entries:
            to_remove = max(1, count // 10)
            self._conn.execute(
                f"DELETE FROM {table} WHERE rowid IN "
                f"(SELECT rowid FROM {table} ORDER BY created_at ASC LIMIT ?)",
                (to_remove,),
            )
            self._conn.commit()
            self._metrics.orphaned_events += to_remove

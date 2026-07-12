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
"""

import logging
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
        }


# ---------------------------------------------------------------------------
# Store Protocol  (for duck-typing a Redis backend later)
# ---------------------------------------------------------------------------

class CorrelationStoreProtocol(Protocol):
    """Minimum interface that any durable backend must implement."""

    def store_signal(self, signal_id: str, event: Any) -> None: ...
    def get_signal(self, signal_id: str) -> Optional[Any]: ...
    def remove_signal(self, signal_id: str) -> None: ...

    def set_deadline(self, signal_id: str, deadline: float) -> None: ...
    def get_expired_deadlines(self, now: float) -> list[str]: ...
    def cancel_deadline(self, signal_id: str) -> None: ...

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
    """Value with creation timestamp for TTL checks."""
    value: Any
    created_at: float = field(default_factory=time.time)


class InMemoryCorrelationStore:
    """
    Thread-safe, bounded, TTL-aware correlation store.

    All entries have a TTL.  The store runs periodic eviction in the
    caller's thread (amortised) and enforces hard size caps.
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
        self._deadlines: dict[str, float] = {}              # signal_id → deadline timestamp
        self._contracts: dict[str, _TTLEntry] = {}          # signal_id → contract event
        self._contract_to_signal: dict[str, _TTLEntry] = {} # contract_id → signal_id
        self._sent_intents: dict[str, float] = {}           # intent_id → timestamp
        self._late_risks: dict[str, _TTLEntry] = {}         # signal_id → RiskEvaluated (early)

        self._metrics = CorrelationMetrics()
        self._last_cleanup = time.time()

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
            if time.time() - entry.created_at > self._signal_ttl:
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
            if time.time() - entry.created_at > self._signal_ttl:
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
            if time.time() - entry.created_at > self._signal_ttl:
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
            if time.time() - ts > self._dedup_ttl:
                self._sent_intents.pop(intent_id, None)
                return False
            return True

    def mark_sent(self, intent_id: str) -> None:
        with self._lock:
            self._evict_if_full_dict(self._sent_intents)
            self._sent_intents[intent_id] = time.time()

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
            if time.time() - entry.created_at > self._signal_ttl:
                return None
            return entry.value

    # ── Cleanup / Metrics ────────────────────────────────────────────────

    def cleanup_expired(self) -> int:
        """Remove all expired entries.  Returns count of entries removed."""
        now = time.time()
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

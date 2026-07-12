"""
Production Hardening Tests — Validates all audit findings.

Finding 1: Durable SQLite CorrelationStore with crash-safe recovery
Finding 2: Atomic deadline-expiration (no timeout/verdict race)
Finding 3: Telegram delivery retries, dead-letter persistence, failure metrics
Finding 4: time.monotonic() for TTL/interval calculations
Finding 5: Schema-version validation and migration hooks
Finding 6: Persist-before-dispatch and durable deduplication
"""

import json
import os
import sqlite3
import threading
import time
from unittest.mock import MagicMock, patch, call

import pytest

from src.core.events import (
    Event,
    EventBus,
    SCHEMA_VERSION,
    SignalGenerated,
    DecisionContractCreated,
    RiskEvaluated,
    RiskHalt,
    register_event_migration,
    _MIGRATIONS,
)
from src.notifications.correlation_store import (
    InMemoryCorrelationStore,
    SqliteCorrelationStore,
    CorrelationMetrics,
    STORE_SCHEMA_VERSION,
)
from src.notifications.telegram_audit_forwarder import (
    TelegramAuditForwarder,
    _format_signal,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_signal(signal_id="sig-1", symbol="AAPL", side="BUY",
                 strategy="ml_momentum", strategy_version="v1.0",
                 signal_strength=0.83, source="engine") -> SignalGenerated:
    return SignalGenerated(
        signal_id=signal_id, symbol=symbol, signal=side, side=side,
        strategy=strategy, strategy_version=strategy_version,
        signal_strength=signal_strength, source=source,
    )


def _make_risk(signal_id="sig-1", symbol="AAPL", contract_id="",
               approved=True, adjusted_qty=10.0,
               reasons=None) -> RiskEvaluated:
    return RiskEvaluated(
        symbol=symbol, signal_id=signal_id, contract_id=contract_id,
        approved=approved, adjusted_qty=adjusted_qty,
        reasons=reasons or [],
    )


def _make_forwarder(send_fn=None, store=None, timeout_seconds=60.0,
                    timeout_interval=1.0, max_retries=3):
    send = send_fn or MagicMock()
    f = TelegramAuditForwarder(
        send,
        risk_verdict_timeout_seconds=timeout_seconds,
        timeout_check_interval=timeout_interval,
        correlation_store=store,
        max_send_retries=max_retries,
    )
    f._get_active_model_version = lambda: "v1.0"
    return f


def _drain(forwarder, seconds=0.5):
    time.sleep(seconds)


def _sent_msgs(send_fn):
    return [c[0][0] for c in send_fn.call_args_list]


def _combined(send_fn):
    return "\n".join(_sent_msgs(send_fn))


# ─────────────────────────────────────────────────────────────────────────────
# Finding 1: Durable SQLite CorrelationStore
# ─────────────────────────────────────────────────────────────────────────────


class TestSqliteCorrelationStore:
    """Test the durable SQLite-backed correlation store."""

    def test_basic_signal_roundtrip(self, tmp_path):
        db = str(tmp_path / "corr.db")
        store = SqliteCorrelationStore(db)
        sig = _make_signal()
        store.store_signal("sig-1", sig)
        result = store.get_signal("sig-1")
        assert result is not None
        store.close()

    def test_signal_ttl_expiry(self, tmp_path):
        db = str(tmp_path / "corr.db")
        store = SqliteCorrelationStore(db, signal_ttl=0.3)
        store.store_signal("sig-ttl", {"test": True})
        assert store.get_signal("sig-ttl") is not None
        time.sleep(0.5)
        assert store.get_signal("sig-ttl") is None
        assert store.metrics.signals_expired >= 1
        store.close()

    def test_dedup_roundtrip(self, tmp_path):
        db = str(tmp_path / "corr.db")
        store = SqliteCorrelationStore(db)
        assert store.check_dedup("intent:x") is False
        store.mark_sent("intent:x")
        assert store.check_dedup("intent:x") is True
        store.close()

    def test_dedup_ttl_expiry(self, tmp_path):
        db = str(tmp_path / "corr.db")
        store = SqliteCorrelationStore(db, dedup_ttl=0.3)
        store.mark_sent("intent:y")
        assert store.check_dedup("intent:y") is True
        time.sleep(0.5)
        assert store.check_dedup("intent:y") is False
        store.close()

    def test_deadline_set_and_expire(self, tmp_path):
        db = str(tmp_path / "corr.db")
        store = SqliteCorrelationStore(db)
        store.set_deadline("sig-d", time.monotonic() - 1)
        expired = store.get_expired_deadlines(time.monotonic())
        assert "sig-d" in expired
        store.close()

    def test_deadline_cancel(self, tmp_path):
        db = str(tmp_path / "corr.db")
        store = SqliteCorrelationStore(db)
        store.set_deadline("sig-c", time.monotonic() + 100)
        store.cancel_deadline("sig-c")
        expired = store.get_expired_deadlines(time.monotonic() + 200)
        assert "sig-c" not in expired
        store.close()

    def test_contract_roundtrip(self, tmp_path):
        db = str(tmp_path / "corr.db")
        store = SqliteCorrelationStore(db)
        store.store_contract("sig-1", {"contract": True})
        assert store.get_contract("sig-1") is not None
        store.map_contract_to_signal("c-1", "sig-1")
        assert store.lookup_signal_by_contract("c-1") == "sig-1"
        store.close()

    def test_late_risk_roundtrip(self, tmp_path):
        db = str(tmp_path / "corr.db")
        store = SqliteCorrelationStore(db)
        store.record_late_risk("sig-lr", {"late": True})
        result = store.pop_late_risk("sig-lr")
        assert result is not None
        assert store.pop_late_risk("sig-lr") is None  # popped
        assert store.metrics.late_arrivals >= 1
        store.close()

    def test_cleanup_expired(self, tmp_path):
        db = str(tmp_path / "corr.db")
        store = SqliteCorrelationStore(db, signal_ttl=0.2, dedup_ttl=0.2)
        for i in range(20):
            store.store_signal(f"sig-{i}", {"i": i})
            store.mark_sent(f"intent-{i}")
        time.sleep(0.4)
        removed = store.cleanup_expired()
        assert removed >= 20
        store.close()

    def test_crash_recovery_state_survives(self, tmp_path):
        """Simulate crash: close store, reopen, verify state persists."""
        db = str(tmp_path / "corr.db")
        store1 = SqliteCorrelationStore(db)
        store1.store_signal("sig-crash", _make_signal())
        store1.mark_sent("intent:crash")
        store1.set_deadline("sig-crash", time.monotonic() + 100)
        store1.close()

        # "Restart" — reopen the same DB
        store2 = SqliteCorrelationStore(db)
        assert store2.get_signal("sig-crash") is not None
        assert store2.check_dedup("intent:crash") is True
        expired = store2.get_expired_deadlines(time.monotonic() + 200)
        assert "sig-crash" in expired
        store2.close()

    def test_schema_version_persisted(self, tmp_path):
        db = str(tmp_path / "corr.db")
        store = SqliteCorrelationStore(db)
        conn = sqlite3.connect(db)
        cur = conn.execute("SELECT value FROM store_meta WHERE key='schema_version'")
        row = cur.fetchone()
        assert row is not None
        assert int(row[0]) == STORE_SCHEMA_VERSION
        conn.close()
        store.close()

    def test_metrics_recovery_on_startup(self, tmp_path):
        db = str(tmp_path / "corr.db")
        store1 = SqliteCorrelationStore(db)
        store1.store_signal("s1", {"x": 1})
        store1.store_signal("s2", {"x": 2})
        store1.close()

        store2 = SqliteCorrelationStore(db)
        assert store2.metrics.signals_tracked >= 2
        store2.close()


# ─────────────────────────────────────────────────────────────────────────────
# Finding 1 + Pipeline: SQLite store works with forwarder
# ─────────────────────────────────────────────────────────────────────────────


class TestSqliteWithForwarder:
    """End-to-end: forwarder using SqliteCorrelationStore."""

    def test_full_pipeline_with_sqlite_store(self, tmp_path):
        db = str(tmp_path / "corr.db")
        store = SqliteCorrelationStore(db)
        send_fn = MagicMock()
        f = _make_forwarder(send_fn, store=store)

        f.handle(_make_signal())
        f.handle(_make_risk(contract_id=""))
        _drain(f, 0.8)
        f.stop()

        assert send_fn.call_count >= 1
        combined = _combined(send_fn)
        assert "AAPL" in combined
        store.close()

    def test_crash_recovery_with_sqlite_dedup(self, tmp_path):
        """After crash, new forwarder with same DB deduplicates."""
        db = str(tmp_path / "corr.db")
        store1 = SqliteCorrelationStore(db)
        send_fn = MagicMock()
        f1 = _make_forwarder(send_fn, store=store1)

        f1.handle(_make_signal())
        f1.handle(_make_risk(contract_id=""))
        _drain(f1, 0.5)
        f1.stop()
        first_count = send_fn.call_count
        assert first_count >= 1
        store1.close()

        # "Restart"
        send_fn.reset_mock()
        store2 = SqliteCorrelationStore(db)
        f2 = _make_forwarder(send_fn, store=store2)
        f2.handle(_make_risk(contract_id=""))
        _drain(f2, 0.5)
        f2.stop()
        assert send_fn.call_count == 0  # deduped
        store2.close()


# ─────────────────────────────────────────────────────────────────────────────
# Finding 2: Atomic deadline expiration
# ─────────────────────────────────────────────────────────────────────────────


class TestAtomicDeadlineExpiration:
    """Verify expire_and_cancel_deadlines prevents timeout/verdict races."""

    def test_expire_and_cancel_in_memory(self):
        store = InMemoryCorrelationStore()
        store.set_deadline("sig-a", time.monotonic() - 1)
        store.set_deadline("sig-b", time.monotonic() + 100)

        expired = store.expire_and_cancel_deadlines(time.monotonic())
        assert "sig-a" in expired
        assert "sig-b" not in expired

        # sig-a deadline should be removed
        assert "sig-a" not in store.get_expired_deadlines(time.monotonic() + 200)

    def test_expire_and_cancel_sqlite(self, tmp_path):
        db = str(tmp_path / "corr.db")
        store = SqliteCorrelationStore(db)
        store.set_deadline("sig-a", time.monotonic() - 1)
        store.set_deadline("sig-b", time.monotonic() + 100)

        expired = store.expire_and_cancel_deadlines(time.monotonic())
        assert "sig-a" in expired
        assert "sig-b" not in expired

        remaining = store.get_expired_deadlines(time.monotonic() + 200)
        assert "sig-a" not in remaining
        store.close()

    def test_concurrent_verdict_and_timeout_no_false_timeout(self):
        """Simulate concurrent verdict arrival and timeout check."""
        send_fn = MagicMock()
        store = InMemoryCorrelationStore()
        f = _make_forwarder(send_fn, store=store, timeout_seconds=0.5,
                            timeout_interval=0.2)

        f.handle(_make_signal(signal_id="sig-race"))
        time.sleep(0.1)  # Deadline set but not expired yet
        # Verdict arrives before timeout
        f.handle(_make_risk(signal_id="sig-race", contract_id=""))
        time.sleep(1.5)  # Wait past timeout period
        f.stop()

        combined = _combined(send_fn)
        assert "NO VERDICT RECEIVED" not in combined


# ─────────────────────────────────────────────────────────────────────────────
# Finding 3: Telegram delivery retries, dead-letter, failure metrics
# ─────────────────────────────────────────────────────────────────────────────


class TestTelegramRetryAndDeadLetter:
    """Verify retry, dead-letter, and failure tracking."""

    def test_retry_on_transient_failure(self):
        """First send fails, second succeeds — message is delivered."""
        call_count = {"n": 0}
        def flaky_send(msg):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ConnectionError("transient")

        f = _make_forwarder(flaky_send, max_retries=3)
        f.handle(RiskHalt(reason="test", level="WARNING"))
        _drain(f, 2.0)
        f.stop()
        assert f._events_sent == 1
        assert f._store.metrics.send_failures >= 1

    def test_all_retries_exhausted_dead_letters_message(self, tmp_path):
        """All retries fail — message is dead-lettered in SQLite store."""
        db = str(tmp_path / "corr.db")
        store = SqliteCorrelationStore(db)
        failing_send = MagicMock(side_effect=Exception("Network down"))
        f = _make_forwarder(failing_send, store=store, max_retries=2)

        f.handle(RiskHalt(reason="test", level="WARNING"))
        _drain(f, 3.0)
        f.stop()

        assert f._events_sent == 0
        dead = store.get_dead_letters()
        assert len(dead) >= 1
        assert "Network down" in dead[0][1] or store.metrics.dead_letters >= 1
        store.close()

    def test_dead_letter_retry_and_removal(self, tmp_path):
        """Dead letters can be retried and removed on success."""
        db = str(tmp_path / "corr.db")
        store = SqliteCorrelationStore(db)
        store.add_dead_letter("test message", "error")

        letters = store.get_dead_letters()
        assert len(letters) == 1
        letter_id, msg, attempts = letters[0]
        assert msg == "test message"
        assert attempts == 1

        store.increment_dead_letter_attempt(letter_id, "still failing")
        letters = store.get_dead_letters()
        assert letters[0][2] == 2

        store.remove_dead_letter(letter_id)
        assert len(store.get_dead_letters()) == 0
        assert store.metrics.retries_succeeded == 1
        store.close()

    def test_failure_metrics_tracked(self):
        """Send failures increment metrics."""
        failing_send = MagicMock(side_effect=Exception("fail"))
        f = _make_forwarder(failing_send, max_retries=2)

        f.handle(RiskHalt(reason="test", level="WARNING"))
        _drain(f, 3.0)
        f.stop()

        assert f._store.metrics.send_failures >= 2
        stats = f.stats
        assert stats["send_failures"] >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Finding 4: time.monotonic() for TTL/interval
# ─────────────────────────────────────────────────────────────────────────────


class TestMonotonicClock:
    """Verify TTL calculations use monotonic clock, not wall clock."""

    def test_ttl_entry_uses_monotonic(self):
        """_TTLEntry.created_at should be close to time.monotonic()."""
        from src.notifications.correlation_store import _TTLEntry
        before = time.monotonic()
        entry = _TTLEntry(value="test")
        after = time.monotonic()
        assert before <= entry.created_at <= after

    def test_in_memory_store_deadline_uses_monotonic(self):
        """Deadlines set with monotonic timestamps."""
        store = InMemoryCorrelationStore()
        mono_deadline = time.monotonic() + 10
        store.set_deadline("sig-mono", mono_deadline)
        # Should NOT be expired since deadline is 10s in the future
        expired = store.get_expired_deadlines(time.monotonic())
        assert "sig-mono" not in expired
        # Should be expired with future now
        expired = store.get_expired_deadlines(mono_deadline + 1)
        assert "sig-mono" in expired

    def test_forwarder_sets_monotonic_deadline(self):
        """TelegramAuditForwarder uses time.monotonic() for deadlines."""
        send_fn = MagicMock()
        store = InMemoryCorrelationStore()
        f = _make_forwarder(send_fn, store=store, timeout_seconds=100)

        before = time.monotonic()
        f.handle(_make_signal(signal_id="sig-mono-fwd"))
        after = time.monotonic()

        # The deadline should be set in the monotonic range
        with store._lock:
            dl = store._deadlines.get("sig-mono-fwd")
        assert dl is not None
        assert before + 100 <= dl <= after + 100
        f.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Finding 5: Schema-version validation and migration hooks
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaVersionValidation:
    """Verify event deserialization validates schema version."""

    def test_event_serializes_schema_version(self):
        event = Event(source="test")
        d = event.to_dict()
        assert d["_schema_version"] == SCHEMA_VERSION

    def test_event_deserializes_current_version(self):
        event = Event(source="test")
        d = event.to_dict()
        restored = Event.from_dict(d)
        assert restored.source == "test"

    def test_event_deserializes_without_version(self):
        """Old events without _schema_version still deserialize."""
        d = {"source": "old_event"}
        restored = Event.from_dict(d)
        assert restored.source == "old_event"

    def test_migration_hook_applied(self):
        """Register a migration and verify it's applied."""
        # Save existing migrations
        saved = dict(_MIGRATIONS)
        try:
            def migrate_099(data):
                data["source"] = data.get("source", "") + "_migrated"
                return data

            register_event_migration("0.9.0", migrate_099)

            d = {"source": "old", "_schema_version": "0.9.0", "_type": "Event"}
            restored = Event.from_dict(d)
            assert "migrated" in restored.source
        finally:
            _MIGRATIONS.clear()
            _MIGRATIONS.update(saved)

    def test_signal_event_roundtrip_with_version(self):
        """SignalGenerated serializes and deserializes with version."""
        sig = _make_signal()
        d = sig.to_dict()
        assert d["_schema_version"] == SCHEMA_VERSION
        restored = SignalGenerated.from_dict(d)
        assert restored.symbol == "AAPL"
        assert restored.signal_id == "sig-1"


# ─────────────────────────────────────────────────────────────────────────────
# Finding 6: Persist-before-dispatch and durable dedup
# ─────────────────────────────────────────────────────────────────────────────


class TestPersistBeforeDispatch:
    """Verify dedup key is persisted before message dispatch."""

    def test_dedup_key_set_before_enqueue(self):
        """The intent is marked as sent BEFORE the message is enqueued."""
        send_fn = MagicMock()
        store = InMemoryCorrelationStore()
        f = _make_forwarder(send_fn, store=store)

        f.handle(_make_signal())

        # Intercept: check dedup state is set before queue drain
        check_results = []
        original_enqueue = f._enqueue_message

        def patched_enqueue(msg):
            # At this point, mark_sent should already have been called
            check_results.append(store.check_dedup("intent:sig-1"))
            original_enqueue(msg)

        f._enqueue_message = patched_enqueue
        f.handle(_make_risk(contract_id=""))
        _drain(f, 0.5)
        f.stop()

        assert len(check_results) == 1
        assert check_results[0] is True  # dedup was set BEFORE enqueue

    def test_durable_dedup_with_sqlite_survives_restart(self, tmp_path):
        """Dedup keys in SQLite survive process restart."""
        db = str(tmp_path / "corr.db")
        store1 = SqliteCorrelationStore(db)
        store1.mark_sent("intent:durable-1")
        store1.close()

        store2 = SqliteCorrelationStore(db)
        assert store2.check_dedup("intent:durable-1") is True
        store2.close()


# ─────────────────────────────────────────────────────────────────────────────
# Concurrency scenarios
# ─────────────────────────────────────────────────────────────────────────────


class TestConcurrency:
    """Thread-safety and concurrency tests."""

    def test_concurrent_signal_store_and_get(self):
        store = InMemoryCorrelationStore()
        errors = []

        def writer():
            for i in range(200):
                store.store_signal(f"sig-w-{i}", {"i": i})

        def reader():
            for i in range(200):
                store.get_signal(f"sig-w-{i}")

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # No crash = success
        assert store.metrics.signals_tracked >= 200

    def test_concurrent_dedup_no_double_send(self):
        """Two threads checking dedup should not both succeed."""
        store = InMemoryCorrelationStore()
        store.mark_sent("intent:concurrent")
        results = []

        def checker():
            results.append(store.check_dedup("intent:concurrent"))

        threads = [threading.Thread(target=checker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert all(r is True for r in results)

    def test_concurrent_sqlite_operations(self, tmp_path):
        """SQLite store handles concurrent access safely."""
        db = str(tmp_path / "corr.db")
        store = SqliteCorrelationStore(db)
        errors = []

        def writer():
            try:
                for i in range(50):
                    store.store_signal(f"sig-cw-{i}", {"i": i})
                    store.mark_sent(f"intent-cw-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert len(errors) == 0
        store.close()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics completeness
# ─────────────────────────────────────────────────────────────────────────────


class TestMetricsCompleteness:
    """Verify new metrics fields are present and serializable."""

    def test_metrics_includes_new_fields(self):
        m = CorrelationMetrics()
        d = m.to_dict()
        assert "dead_letters" in d
        assert "retries_succeeded" in d
        assert "send_failures" in d

    def test_stats_includes_failure_metrics(self):
        send_fn = MagicMock()
        f = _make_forwarder(send_fn)
        stats = f.stats
        assert "send_failures" in stats
        assert "dead_letters" in stats
        f.stop()

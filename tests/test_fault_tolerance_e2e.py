"""
End-to-end tests for fault tolerance refactoring.

Tests cover:
1. Non-blocking EventBus with slow handler isolation
2. Risk decision persistence and timestamps
3. Idempotent Telegram notifications (duplicate suppression)
4. Operational circuit breaker (broker/risk-engine fault detection)
5. Integration: trading remains responsive under fault conditions
"""

import json
import time
import threading
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.core.events import EventBus, Event, SignalGenerated, OrderFilled
from src.core.operational_circuit_breaker import (
    OperationalCircuitBreaker,
    OperationalState,
    FailureDomain,
)
from src.notifications.idempotent_sender import IdempotentNotifier
from src.risk.decision_store import RiskDecisionStore


# ===========================================================================
# 1. Non-blocking EventBus — slow handler isolation
# ===========================================================================


class TestNonBlockingEventBus:
    """Verify non-blocking delivery isolates slow subscribers."""

    def test_slow_handler_does_not_block_fast_handlers(self):
        """A slow handler must not delay delivery to other handlers."""
        bus = EventBus(non_blocking=True, handler_timeout=0.5, max_workers=4)
        fast_received = []
        slow_started = threading.Event()

        def slow_handler(event):
            slow_started.set()
            time.sleep(5.0)  # Intentionally very slow

        def fast_handler(event):
            fast_received.append(event)

        bus.subscribe(SignalGenerated, slow_handler)
        bus.subscribe(SignalGenerated, fast_handler)

        start = time.monotonic()
        event = SignalGenerated(symbol="AAPL", signal="BUY", confidence=0.9, strategy="test")
        bus.publish(event)
        elapsed = time.monotonic() - start

        # Fast handler received the event
        assert len(fast_received) == 1
        assert fast_received[0].symbol == "AAPL"

        # Publish completed within timeout (not blocked by 5s sleep)
        assert elapsed < 2.0

        # Bus recorded the timeout
        assert bus.timeout_count >= 1
        bus.shutdown()

    def test_non_blocking_bus_records_timeout_metrics(self):
        """Timeout metrics are tracked when handlers exceed deadline."""
        bus = EventBus(non_blocking=True, handler_timeout=0.2)

        def sleepy(event):
            time.sleep(1.0)

        bus.subscribe(SignalGenerated, sleepy)
        bus.publish(SignalGenerated(symbol="X", signal="BUY", confidence=0.5, strategy="t"))

        assert bus.timeout_count == 1
        assert bus.slow_handler_count == 1
        bus.shutdown()

    def test_blocking_bus_preserves_original_behavior(self):
        """Default blocking mode still works as before."""
        bus = EventBus()  # blocking by default
        received = []
        bus.subscribe(SignalGenerated, lambda e: received.append(e))
        bus.publish(SignalGenerated(symbol="MSFT", signal="SELL", confidence=0.8, strategy="t"))
        assert len(received) == 1
        assert not bus.non_blocking

    def test_non_blocking_multiple_slow_handlers(self):
        """Multiple slow handlers are all isolated independently."""
        bus = EventBus(non_blocking=True, handler_timeout=0.3, max_workers=4)
        completed = []

        def slow_a(event):
            time.sleep(2.0)
            completed.append("a")

        def slow_b(event):
            time.sleep(2.0)
            completed.append("b")

        def fast_c(event):
            completed.append("c")

        bus.subscribe(SignalGenerated, slow_a)
        bus.subscribe(SignalGenerated, slow_b)
        bus.subscribe(SignalGenerated, fast_c)

        start = time.monotonic()
        bus.publish(SignalGenerated(symbol="X", signal="BUY", confidence=0.5, strategy="t"))
        elapsed = time.monotonic() - start

        # Fast handler completed, but slow handlers timed out
        assert "c" in completed
        assert elapsed < 2.0
        assert bus.timeout_count >= 2
        bus.shutdown()

    def test_non_blocking_handler_exception_isolated(self):
        """Exceptions in non-blocking handlers don't crash the bus."""
        bus = EventBus(non_blocking=True, handler_timeout=1.0)
        received = []

        def failing(event):
            raise RuntimeError("boom")

        def good(event):
            received.append(event)

        bus.subscribe(SignalGenerated, failing)
        bus.subscribe(SignalGenerated, good)
        bus.publish(SignalGenerated(symbol="X", signal="BUY", confidence=0.5, strategy="t"))

        assert len(received) == 1
        bus.shutdown()

    def test_non_blocking_high_throughput(self):
        """Non-blocking bus handles burst of events without degradation."""
        bus = EventBus(non_blocking=True, handler_timeout=1.0, max_workers=8)
        received = []
        bus.subscribe(SignalGenerated, lambda e: received.append(e))

        start = time.monotonic()
        for i in range(100):
            bus.publish(SignalGenerated(symbol=f"SYM{i}", signal="BUY", confidence=0.5, strategy="t"))
        elapsed = time.monotonic() - start

        assert len(received) == 100
        assert elapsed < 5.0  # Should be fast even with thread pool overhead
        bus.shutdown()


# ===========================================================================
# 2. Risk Decision Persistence
# ===========================================================================


class TestRiskDecisionStore:
    """Verify risk decisions are persisted with timestamps."""

    def setup_method(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.store = RiskDecisionStore(db_path=self._tmp.name)

    def teardown_method(self):
        self.store.close()

    def test_persist_and_query_decision(self):
        """Decisions are persisted and retrievable."""
        row_id = self.store.persist(
            symbol="AAPL",
            side="BUY",
            requested_qty=100.0,
            adjusted_qty=80.0,
            approved=True,
            risk_score=25.0,
            reasons=["Portfolio heat reduced qty"],
            layer_details={"account_risk": {"action": "APPROVE", "reason": "OK"}},
            confidence=0.85,
            strategy="momentum",
            signal_id="sig_001",
        )
        assert row_id is not None and row_id > 0

        results = self.store.query(symbol="AAPL")
        assert len(results) == 1
        entry = results[0]
        assert entry["symbol"] == "AAPL"
        assert entry["approved"] is True
        assert entry["adjusted_qty"] == 80.0
        assert entry["risk_score"] == 25.0
        assert "Portfolio heat reduced qty" in entry["reasons"]
        assert entry["layer_details"]["account_risk"]["action"] == "APPROVE"

    def test_timestamps_are_utc_and_present(self):
        """Every persisted decision has a UTC timestamp."""
        self.store.persist(
            symbol="MSFT", side="SELL", requested_qty=50.0,
            adjusted_qty=0.0, approved=False, risk_score=80.0,
            reasons=["Daily loss limit"], layer_details={},
        )
        results = self.store.query()
        assert len(results) == 1
        ts = results[0]["timestamp_utc"]
        # Must be parseable ISO format
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None or "+" in ts or "Z" in ts

    def test_query_filters(self):
        """Filters by symbol, approval status work correctly."""
        self.store.persist(symbol="AAPL", side="BUY", requested_qty=10, adjusted_qty=10,
                          approved=True, risk_score=10, reasons=[], layer_details={})
        self.store.persist(symbol="TSLA", side="SELL", requested_qty=5, adjusted_qty=0,
                          approved=False, risk_score=90, reasons=["rejected"], layer_details={})
        self.store.persist(symbol="AAPL", side="SELL", requested_qty=10, adjusted_qty=10,
                          approved=True, risk_score=15, reasons=[], layer_details={})

        assert self.store.count() == 3
        assert self.store.count(approved=True) == 2
        assert self.store.count(approved=False) == 1

        aapl_decisions = self.store.query(symbol="AAPL")
        assert len(aapl_decisions) == 2

        rejected = self.store.query(approved=False)
        assert len(rejected) == 1
        assert rejected[0]["symbol"] == "TSLA"

    def test_persistence_survives_reopen(self):
        """Data survives close and reopen (durability)."""
        self.store.persist(symbol="GME", side="BUY", requested_qty=100, adjusted_qty=100,
                          approved=True, risk_score=5, reasons=["OK"], layer_details={})
        self.store.close()

        # Reopen
        store2 = RiskDecisionStore(db_path=self._tmp.name)
        results = store2.query(symbol="GME")
        assert len(results) == 1
        assert results[0]["approved"] is True
        store2.close()

    def test_concurrent_writes(self):
        """Store handles concurrent writes safely."""
        errors = []

        def writer(sym):
            try:
                for i in range(20):
                    self.store.persist(
                        symbol=sym, side="BUY", requested_qty=float(i),
                        adjusted_qty=float(i), approved=True, risk_score=10.0,
                        reasons=[], layer_details={},
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"SYM{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert self.store.count() == 100


# ===========================================================================
# 3. Idempotent Telegram Notifications
# ===========================================================================


class TestIdempotentNotifier:
    """Verify notifications are deduplicated and retries are safe."""

    def test_duplicate_notification_suppressed(self):
        """Same message sent twice is delivered only once."""
        send_mock = MagicMock()
        notifier = IdempotentNotifier(send_fn=send_mock)

        result1 = notifier.send("chat123", "Trade executed: BUY AAPL")
        result2 = notifier.send("chat123", "Trade executed: BUY AAPL")

        assert result1 is True
        assert result2 is True
        # Only one actual send
        assert send_mock.call_count == 1
        assert notifier.metrics["total_sent"] == 1
        assert notifier.metrics["total_suppressed"] == 1

    def test_different_messages_both_sent(self):
        """Different messages are not deduplicated."""
        send_mock = MagicMock()
        notifier = IdempotentNotifier(send_fn=send_mock)

        notifier.send("chat123", "Trade: BUY AAPL")
        notifier.send("chat123", "Trade: SELL MSFT")

        assert send_mock.call_count == 2

    def test_explicit_dedup_key(self):
        """Explicit dedup_key controls deduplication."""
        send_mock = MagicMock()
        notifier = IdempotentNotifier(send_fn=send_mock)

        notifier.send("chat1", "msg A", dedup_key="signal_001")
        notifier.send("chat2", "msg B", dedup_key="signal_001")  # Same key = suppressed

        assert send_mock.call_count == 1
        assert notifier.metrics["total_suppressed"] == 1

    def test_retry_on_transient_failure(self):
        """Retries on failure with eventual success."""
        call_count = [0]

        def flaky_send(chat_id, message):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("network timeout")

        notifier = IdempotentNotifier(send_fn=flaky_send, max_retries=3)
        result = notifier.send("chat123", "Alert: risk breach")

        assert result is True
        assert call_count[0] == 3  # Failed twice, succeeded third time
        assert notifier.metrics["total_sent"] == 1
        assert notifier.metrics["total_retries"] == 2

    def test_permanent_failure_after_retries(self):
        """Returns False when all retries are exhausted."""
        def always_fail(chat_id, message):
            raise ConnectionError("service down")

        notifier = IdempotentNotifier(send_fn=always_fail, max_retries=3)
        result = notifier.send("chat123", "Alert")

        assert result is False
        assert notifier.metrics["total_failed"] == 1

    def test_ttl_expiry_allows_resend(self):
        """After TTL expires, the same message can be sent again."""
        send_mock = MagicMock()
        notifier = IdempotentNotifier(send_fn=send_mock, dedup_ttl=0.1)

        notifier.send("chat123", "alert")
        assert send_mock.call_count == 1

        time.sleep(0.2)  # Wait for TTL to expire

        notifier.send("chat123", "alert")
        assert send_mock.call_count == 2

    def test_is_duplicate_check(self):
        """is_duplicate returns correct status."""
        send_mock = MagicMock()
        notifier = IdempotentNotifier(send_fn=send_mock)

        assert notifier.is_duplicate("chat1", "msg") is False
        notifier.send("chat1", "msg")
        assert notifier.is_duplicate("chat1", "msg") is True

    def test_concurrent_duplicate_sends(self):
        """Concurrent sends of the same message only deliver once."""
        send_count = [0]
        send_lock = threading.Lock()

        def counting_send(chat_id, message):
            time.sleep(0.05)  # Simulate network latency
            with send_lock:
                send_count[0] += 1

        notifier = IdempotentNotifier(send_fn=counting_send)

        threads = [
            threading.Thread(target=notifier.send, args=("chat1", "same message"))
            for _ in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # With atomic check-and-reserve, only one send should get through
        assert send_count[0] == 1
        assert notifier.metrics["total_suppressed"] == 9


# ===========================================================================
# 4. Operational Circuit Breaker
# ===========================================================================


class TestOperationalCircuitBreaker:
    """Verify circuit breaker trips on repeated failures and recovers."""

    def test_closed_initially(self):
        """Breaker starts in CLOSED state (trading allowed)."""
        cb = OperationalCircuitBreaker()
        assert cb.state == OperationalState.CLOSED
        assert cb.is_trading_allowed is True

    def test_trips_after_threshold_failures(self):
        """Breaker opens after exceeding failure threshold."""
        cb = OperationalCircuitBreaker(failure_threshold=3, window_seconds=10.0)

        cb.record_failure(FailureDomain.BROKER, "Connection refused")
        assert cb.state == OperationalState.CLOSED

        cb.record_failure(FailureDomain.BROKER, "Timeout")
        assert cb.state == OperationalState.CLOSED

        state = cb.record_failure(FailureDomain.RISK_ENGINE, "Evaluation error")
        assert state == OperationalState.OPEN
        assert cb.is_trading_allowed is False

    def test_mixed_domain_failures_accumulate(self):
        """Failures from different domains all count toward threshold."""
        cb = OperationalCircuitBreaker(failure_threshold=3, window_seconds=10.0)

        cb.record_failure(FailureDomain.BROKER, "err1")
        cb.record_failure(FailureDomain.RISK_ENGINE, "err2")
        cb.record_failure(FailureDomain.NOTIFICATION, "err3")

        assert cb.state == OperationalState.OPEN

    def test_recovery_timeout_transitions_to_half_open(self):
        """After recovery_timeout, breaker moves to HALF_OPEN for probing."""
        cb = OperationalCircuitBreaker(
            failure_threshold=2, window_seconds=10.0,
            recovery_timeout=0.1,  # Very short for testing
        )

        cb.record_failure(FailureDomain.BROKER, "err1")
        cb.record_failure(FailureDomain.BROKER, "err2")
        assert cb.state == OperationalState.OPEN

        time.sleep(0.15)
        assert cb.state == OperationalState.HALF_OPEN

    def test_probe_success_closes_breaker(self):
        """Successful probes in HALF_OPEN state close the breaker."""
        cb = OperationalCircuitBreaker(
            failure_threshold=2, window_seconds=10.0,
            recovery_timeout=0.1, probe_success_threshold=2,
        )

        cb.record_failure(FailureDomain.BROKER, "err1")
        cb.record_failure(FailureDomain.BROKER, "err2")
        time.sleep(0.15)

        assert cb.state == OperationalState.HALF_OPEN

        cb.record_success(FailureDomain.BROKER)
        assert cb.state == OperationalState.HALF_OPEN  # Need 2

        cb.record_success(FailureDomain.BROKER)
        assert cb.state == OperationalState.CLOSED
        assert cb.is_trading_allowed is True

    def test_failures_expire_outside_window(self):
        """Old failures outside the time window don't count."""
        cb = OperationalCircuitBreaker(
            failure_threshold=3, window_seconds=0.1,
        )

        cb.record_failure(FailureDomain.BROKER, "err1")
        cb.record_failure(FailureDomain.BROKER, "err2")
        time.sleep(0.15)  # Let them expire

        # This should NOT trip because earlier failures expired
        cb.record_failure(FailureDomain.BROKER, "err3")
        assert cb.state == OperationalState.CLOSED

    def test_force_open_and_close(self):
        """Manual force open/close works."""
        cb = OperationalCircuitBreaker()

        cb.force_open("maintenance")
        assert cb.state == OperationalState.OPEN
        assert cb.is_trading_allowed is False

        cb.force_close()
        assert cb.state == OperationalState.CLOSED
        assert cb.is_trading_allowed is True

    def test_status_report(self):
        """Status report includes all relevant metrics."""
        cb = OperationalCircuitBreaker(failure_threshold=5)
        cb.record_failure(FailureDomain.BROKER, "err1")
        cb.record_failure(FailureDomain.RISK_ENGINE, "err2")

        status = cb.get_status()
        assert status["state"] == "CLOSED"
        assert status["trading_allowed"] is True
        assert status["failure_counts"]["BROKER"] == 1
        assert status["failure_counts"]["RISK_ENGINE"] == 1
        assert status["total_recent_failures"] == 2
        assert status["failure_threshold"] == 5

    def test_failure_history_audit(self):
        """Failure history is available for audit."""
        cb = OperationalCircuitBreaker()
        cb.record_failure(FailureDomain.BROKER, "timeout", {"endpoint": "/orders"})
        cb.record_failure(FailureDomain.RISK_ENGINE, "OOM", {"layer": "portfolio"})

        history = cb.get_failure_history()
        assert len(history) == 2
        assert history[0]["domain"] == "BROKER"
        assert history[0]["error"] == "timeout"
        assert history[0]["context"]["endpoint"] == "/orders"
        assert history[1]["domain"] == "RISK_ENGINE"

    def test_event_bus_notified_on_trip(self):
        """EventBus receives notification when breaker trips."""
        bus = EventBus()
        events_received = []
        bus.subscribe(None, lambda e: events_received.append(e))

        cb = OperationalCircuitBreaker(
            failure_threshold=2, event_bus=bus,
        )
        cb.record_failure(FailureDomain.BROKER, "err1")
        cb.record_failure(FailureDomain.BROKER, "err2")

        # Should have received a CircuitBreakerTripped event
        tripped = [e for e in events_received if type(e).__name__ == "CircuitBreakerTripped"]
        assert len(tripped) == 1
        assert tripped[0].source == "operational_circuit_breaker"

    def test_on_state_change_callback(self):
        """State change callback is invoked on transitions."""
        transitions = []
        cb = OperationalCircuitBreaker(
            failure_threshold=2, window_seconds=10.0,
            recovery_timeout=0.1,
            on_state_change=lambda old, new: transitions.append((old, new)),
        )

        cb.record_failure(FailureDomain.BROKER, "err1")
        cb.record_failure(FailureDomain.BROKER, "err2")
        assert transitions[-1] == (OperationalState.CLOSED, OperationalState.OPEN)

        time.sleep(0.15)
        _ = cb.state  # Trigger check
        assert transitions[-1] == (OperationalState.OPEN, OperationalState.HALF_OPEN)


# ===========================================================================
# 5. End-to-End Integration: System remains responsive under faults
# ===========================================================================


class TestFaultToleranceIntegration:
    """Integration tests combining all fault tolerance mechanisms."""

    def test_trading_halts_on_repeated_broker_failures(self):
        """
        Scenario: Broker repeatedly fails → circuit breaker trips →
        trading is safely halted.
        """
        bus = EventBus(non_blocking=True, handler_timeout=1.0)
        cb = OperationalCircuitBreaker(
            failure_threshold=3, window_seconds=60.0, event_bus=bus,
        )
        halt_events = []
        bus.subscribe(None, lambda e: halt_events.append(e))

        # Simulate repeated broker failures
        for i in range(3):
            cb.record_failure(FailureDomain.BROKER, f"Connection refused #{i}")

        assert cb.is_trading_allowed is False
        # System published halt event
        tripped = [e for e in halt_events if type(e).__name__ == "CircuitBreakerTripped"]
        assert len(tripped) >= 1
        bus.shutdown()

    def test_trading_halts_on_repeated_risk_engine_failures(self):
        """
        Scenario: Risk engine throws repeatedly → trading halted.
        """
        cb = OperationalCircuitBreaker(failure_threshold=3, window_seconds=60.0)

        cb.record_failure(FailureDomain.RISK_ENGINE, "Evaluation timeout")
        cb.record_failure(FailureDomain.RISK_ENGINE, "Database lock")
        cb.record_failure(FailureDomain.RISK_ENGINE, "OOM in portfolio layer")

        assert cb.is_trading_allowed is False
        status = cb.get_status()
        assert status["failure_counts"]["RISK_ENGINE"] == 3

    def test_slow_handler_doesnt_delay_risk_decisions(self):
        """
        Scenario: A slow audit handler doesn't delay critical path
        (risk evaluation + order submission).
        """
        bus = EventBus(non_blocking=True, handler_timeout=0.5)
        critical_path_events = []

        def slow_audit_handler(event):
            time.sleep(3.0)  # Simulates slow disk write

        def fast_risk_handler(event):
            critical_path_events.append(("risk_evaluated", time.monotonic()))

        bus.subscribe(SignalGenerated, slow_audit_handler)
        bus.subscribe(SignalGenerated, fast_risk_handler)

        start = time.monotonic()
        bus.publish(SignalGenerated(symbol="AAPL", signal="BUY", confidence=0.9, strategy="test"))
        elapsed = time.monotonic() - start

        # Critical path handler was not delayed
        assert len(critical_path_events) == 1
        assert elapsed < 2.0  # Well under the 3s slow handler time
        bus.shutdown()

    def test_duplicate_notification_retries_are_safe(self):
        """
        Scenario: Network glitch causes retry → duplicate suppressed.
        """
        actual_sends = []

        def telegram_send(chat_id, message):
            actual_sends.append((chat_id, message))

        notifier = IdempotentNotifier(send_fn=telegram_send)

        # First send succeeds
        notifier.send("chat1", "🚨 Risk breach: AAPL", dedup_key="risk_001")
        # Retry (same dedup key)
        notifier.send("chat1", "🚨 Risk breach: AAPL", dedup_key="risk_001")
        # Another retry
        notifier.send("chat1", "🚨 Risk breach: AAPL", dedup_key="risk_001")

        # Only one actual delivery
        assert len(actual_sends) == 1
        assert notifier.metrics["total_suppressed"] == 2

    def test_risk_decisions_auditable_after_crash(self):
        """
        Scenario: Risk decisions are persisted so they survive crashes.
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        store = RiskDecisionStore(db_path=db_path)

        # Persist multiple decisions
        store.persist(
            symbol="AAPL", side="BUY", requested_qty=100, adjusted_qty=100,
            approved=True, risk_score=15, reasons=["All checks passed"],
            layer_details={"account": {"action": "APPROVE"}},
        )
        store.persist(
            symbol="TSLA", side="BUY", requested_qty=50, adjusted_qty=0,
            approved=False, risk_score=85, reasons=["Daily loss limit exceeded"],
            layer_details={"account": {"action": "REJECT"}},
        )
        store.close()

        # Simulate crash + restart
        store2 = RiskDecisionStore(db_path=db_path)
        results = store2.query()
        assert len(results) == 2
        # Most recent first
        assert results[0]["symbol"] == "TSLA"
        assert results[0]["approved"] is False
        assert results[1]["symbol"] == "AAPL"
        assert results[1]["approved"] is True
        store2.close()

    def test_circuit_breaker_recovery_after_broker_heals(self):
        """
        Scenario: Broker fails → trips → recovers → probes succeed → resumes.
        """
        cb = OperationalCircuitBreaker(
            failure_threshold=2, window_seconds=60.0,
            recovery_timeout=0.1, probe_success_threshold=2,
        )

        # Trip it
        cb.record_failure(FailureDomain.BROKER, "err1")
        cb.record_failure(FailureDomain.BROKER, "err2")
        assert cb.is_trading_allowed is False

        # Wait for half-open
        time.sleep(0.15)
        assert cb.state == OperationalState.HALF_OPEN

        # Simulate successful broker probes
        cb.record_success(FailureDomain.BROKER)
        cb.record_success(FailureDomain.BROKER)

        # Trading resumes
        assert cb.is_trading_allowed is True
        assert cb.state == OperationalState.CLOSED

    def test_full_pipeline_under_combined_faults(self):
        """
        Scenario: Multiple fault conditions simultaneously:
        - Slow event handlers
        - Duplicate notification attempts
        - Broker instability

        System should remain responsive and auditable.
        """
        bus = EventBus(non_blocking=True, handler_timeout=0.5, max_workers=4)
        cb = OperationalCircuitBreaker(failure_threshold=5, event_bus=bus)

        sends = []
        notifier = IdempotentNotifier(send_fn=lambda c, m: sends.append(m))

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = RiskDecisionStore(db_path=db_path)

        # Register a slow audit handler
        def slow_audit(event):
            time.sleep(2.0)

        bus.subscribe(SignalGenerated, slow_audit)

        # Fast critical path
        decisions_made = []

        def risk_evaluator(event):
            store.persist(
                symbol=event.symbol, side=event.signal,
                requested_qty=100, adjusted_qty=100,
                approved=True, risk_score=20,
                reasons=["OK"], layer_details={},
            )
            decisions_made.append(event.symbol)
            # Send notification
            notifier.send("chat1", f"Signal: {event.symbol}", dedup_key=event.event_id)

        bus.subscribe(SignalGenerated, risk_evaluator)

        # Simulate broker failures (not enough to trip)
        for i in range(3):
            cb.record_failure(FailureDomain.BROKER, f"timeout #{i}")

        # Trading still allowed (threshold is 5)
        assert cb.is_trading_allowed is True

        # Publish events - should complete quickly despite slow audit handler
        start = time.monotonic()
        for sym in ["AAPL", "MSFT", "GOOG"]:
            bus.publish(SignalGenerated(symbol=sym, signal="BUY", confidence=0.9, strategy="test"))
        elapsed = time.monotonic() - start

        # Verify responsiveness
        assert elapsed < 3.0
        assert len(decisions_made) == 3

        # Verify persistence
        assert store.count() == 3

        # Verify dedup (try sending duplicates)
        notifier.send("chat1", "Signal: AAPL", dedup_key=bus.get_history(SignalGenerated)[0].event_id)
        assert notifier.metrics["total_suppressed"] >= 1

        # Verify system is still auditable
        history = store.query()
        assert all(h["timestamp_utc"] is not None for h in history)

        store.close()
        bus.shutdown()

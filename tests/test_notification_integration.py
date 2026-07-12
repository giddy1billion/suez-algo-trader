"""
End-to-End Integration Tests for the Notification Pipeline.

These tests exercise the REAL EventBus → TelegramAuditForwarder pipeline
rather than constructing canonical objects directly.  They cover:

1. Normal correlated flow (SignalGenerated → Contract → RiskEvaluated)
2. Restart recovery (shared CorrelationStore survives forwarder replacement)
3. Duplicate delivery (both approval and rejection dedup)
4. Multi-instance operation (two forwarders sharing a store)
5. Out-of-order events (RiskEvaluated before SignalGenerated)
6. Proactive timeout generation (background timer fires)
7. Correlation TTL cleanup (stale entries expire)
8. Full signal_id collision resistance (UUIDv4 uniqueness)
9. Late signal reconciliation (buffered risk verdicts)
10. Metrics counters accuracy
"""

import time
import threading
from unittest.mock import MagicMock, patch

import pytest

from src.core.events import (
    EventBus,
    SignalGenerated,
    DecisionContractCreated,
    RiskEvaluated,
    RiskHalt,
)
from src.notifications.telegram_audit_forwarder import (
    TelegramAuditForwarder,
    _format_signal,
)
from src.notifications.correlation_store import (
    InMemoryCorrelationStore,
    CorrelationMetrics,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_signal(signal_id: str = "sig-1", symbol: str = "AAPL", side: str = "BUY",
                 strategy: str = "ml_momentum", strategy_version: str = "v1.0",
                 signal_strength: float = 0.83, source: str = "engine",
                 features: dict = None) -> SignalGenerated:
    return SignalGenerated(
        signal_id=signal_id,
        symbol=symbol,
        signal=side,
        side=side,
        strategy=strategy,
        strategy_version=strategy_version,
        signal_strength=signal_strength,
        source=source,
        features=features or {},
    )


def _make_contract(contract_id: str = "contract-1", signal_id: str = "sig-1",
                   symbol: str = "AAPL", side: str = "BUY",
                   decision: str = "execute") -> DecisionContractCreated:
    return DecisionContractCreated(
        contract_id=contract_id,
        signal_id=signal_id,
        decision=decision,
        symbol=symbol,
        side=side,
        recommended_stop_loss=145.0,
        recommended_take_profit=165.0,
    )


def _make_risk(signal_id: str = "sig-1", symbol: str = "AAPL",
               contract_id: str = "contract-1", approved: bool = True,
               adjusted_qty: float = 10.0, reasons: list = None) -> RiskEvaluated:
    return RiskEvaluated(
        symbol=symbol,
        signal_id=signal_id,
        contract_id=contract_id,
        approved=approved,
        adjusted_qty=adjusted_qty,
        reasons=reasons or [],
    )


def _create_forwarder(send_fn=None, store=None, timeout_seconds=60.0,
                      timeout_interval=1.0) -> TelegramAuditForwarder:
    """Create a forwarder with fast timeout checking for tests."""
    send = send_fn or MagicMock()
    f = TelegramAuditForwarder(
        send,
        risk_verdict_timeout_seconds=timeout_seconds,
        timeout_check_interval=timeout_interval,
        correlation_store=store,
    )
    f._get_active_model_version = lambda: "v1.0"
    return f


def _drain(forwarder: TelegramAuditForwarder, seconds: float = 0.5) -> None:
    """Wait for sender thread to drain queued messages."""
    time.sleep(seconds)


def _sent_messages(send_fn: MagicMock) -> list[str]:
    """Extract all sent messages from mock."""
    return [call[0][0] for call in send_fn.call_args_list]


def _combined_output(send_fn: MagicMock) -> str:
    """All sent messages concatenated."""
    return "\n".join(_sent_messages(send_fn))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Normal Correlated Flow via Real EventBus
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalCorrelatedFlow:
    """End-to-end: SignalGenerated → Contract → RiskEvaluated through EventBus."""

    def test_full_pipeline_via_event_bus(self):
        """Events published to EventBus are received and correlated by forwarder."""
        send_fn = MagicMock()
        bus = EventBus()
        forwarder = _create_forwarder(send_fn)
        forwarder.register(bus)

        # Publish events through the REAL event bus
        bus.publish(_make_signal())
        bus.publish(_make_contract())
        bus.publish(_make_risk())

        _drain(forwarder, 1.0)
        forwarder.stop()

        combined = _combined_output(send_fn)
        # Contract event + approved intent = 2 messages
        assert send_fn.call_count == 2
        # The final message should contain the actionable command
        final = _sent_messages(send_fn)[-1]
        assert "AAPL" in final
        assert _format_signal(_make_signal()) in final

    def test_rejected_signal_shows_risk_warning(self):
        """Rejected risk verdict produces warning, no actionable command."""
        send_fn = MagicMock()
        bus = EventBus()
        forwarder = _create_forwarder(send_fn)
        forwarder.register(bus)

        bus.publish(_make_signal(signal_id="sig-rej"))
        bus.publish(_make_risk(signal_id="sig-rej", approved=False,
                               reasons=["max exposure"], contract_id=""))

        _drain(forwarder, 1.0)
        forwarder.stop()

        combined = _combined_output(send_fn)
        assert "RISK REJECTED" in combined
        assert "/buy" not in combined
        assert "/sell" not in combined


# ─────────────────────────────────────────────────────────────────────────────
# 2. Restart Recovery (Shared Store)
# ─────────────────────────────────────────────────────────────────────────────


class TestRestartRecovery:
    """Simulate process restart: old forwarder stops, new one uses same store."""

    def test_shared_store_survives_forwarder_replacement(self):
        """A new forwarder with the same store can dedup previously-sent intents."""
        send_fn = MagicMock()
        store = InMemoryCorrelationStore()
        
        # First forwarder processes the signal
        f1 = _create_forwarder(send_fn, store=store)
        f1.handle(_make_signal())
        f1.handle(_make_contract())
        f1.handle(_make_risk())
        _drain(f1, 0.5)
        f1.stop()
        first_count = send_fn.call_count

        # "Restart" — new forwarder, SAME store
        send_fn.reset_mock()
        f2 = _create_forwarder(send_fn, store=store)
        # Replay the same risk event (duplicate after restart)
        f2.handle(_make_risk())
        _drain(f2, 0.5)
        f2.stop()

        # The second forwarder should deduplicate
        assert send_fn.call_count == 0
        assert store.metrics.duplicates_suppressed >= 1

    def test_pending_deadline_survives_forwarder_replacement(self):
        """Pending deadline set by old forwarder is visible to new forwarder."""
        send_fn = MagicMock()
        store = InMemoryCorrelationStore()

        # First forwarder records signal with deadline
        f1 = _create_forwarder(send_fn, store=store, timeout_seconds=1.0)
        f1.handle(_make_signal(signal_id="sig-restart"))
        f1.stop()  # "crash"

        # New forwarder picks up the store — should emit timeout
        send_fn.reset_mock()
        f2 = _create_forwarder(send_fn, store=store, timeout_seconds=1.0,
                               timeout_interval=0.5)
        time.sleep(2.0)
        f2.stop()

        combined = _combined_output(send_fn)
        assert "NO VERDICT RECEIVED" in combined


# ─────────────────────────────────────────────────────────────────────────────
# 3. Duplicate Delivery (Approval + Rejection Dedup)
# ─────────────────────────────────────────────────────────────────────────────


class TestDuplicateDelivery:
    """Verify dedup for both approved and rejected signals."""

    def test_duplicate_approved_risk_suppressed(self):
        """Second identical approved RiskEvaluated is not sent."""
        send_fn = MagicMock()
        forwarder = _create_forwarder(send_fn)

        forwarder.handle(_make_signal())
        forwarder.handle(_make_contract())
        forwarder.handle(_make_risk())
        forwarder.handle(_make_risk())  # duplicate
        forwarder.handle(_make_risk())  # triple

        _drain(forwarder, 0.5)
        forwarder.stop()

        # Only 2 messages: contract + one approved intent
        assert send_fn.call_count == 2
        assert forwarder.correlation_metrics.duplicates_suppressed == 2

    def test_duplicate_rejected_risk_suppressed(self):
        """Second identical rejected RiskEvaluated is not sent."""
        send_fn = MagicMock()
        forwarder = _create_forwarder(send_fn)

        forwarder.handle(_make_signal(signal_id="sig-rej-dup"))
        risk = _make_risk(signal_id="sig-rej-dup", approved=False,
                          reasons=["limit"], contract_id="")
        forwarder.handle(risk)
        forwarder.handle(risk)  # duplicate

        _drain(forwarder, 0.5)
        forwarder.stop()

        # One rejection message only
        msgs = _sent_messages(send_fn)
        risk_msgs = [m for m in msgs if "RISK REJECTED" in m]
        assert len(risk_msgs) == 1
        assert forwarder.correlation_metrics.duplicates_suppressed >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multi-Instance Operation (Shared Store)
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiInstance:
    """Two forwarders sharing a CorrelationStore."""

    def test_two_forwarders_dedup_across_instances(self):
        """Signal processed by instance A is deduped by instance B."""
        send_a = MagicMock()
        send_b = MagicMock()
        store = InMemoryCorrelationStore()

        fa = _create_forwarder(send_a, store=store)
        fb = _create_forwarder(send_b, store=store)

        # Instance A processes the full flow
        fa.handle(_make_signal())
        fa.handle(_make_contract())
        fa.handle(_make_risk())
        _drain(fa, 0.5)

        # Instance B receives the same risk event (e.g., from event replay)
        fb.handle(_make_risk())
        _drain(fb, 0.5)

        fa.stop()
        fb.stop()

        # Instance A sent the message; instance B should have deduped it
        assert send_a.call_count >= 1
        assert send_b.call_count == 0
        assert store.metrics.duplicates_suppressed >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. Out-of-Order Events
# ─────────────────────────────────────────────────────────────────────────────


class TestOutOfOrder:
    """RiskEvaluated arriving before SignalGenerated."""

    def test_risk_before_signal_is_buffered_and_reconciled(self):
        """RiskEvaluated arriving first is buffered; when SignalGenerated arrives,
        the verdict is processed with full correlation."""
        send_fn = MagicMock()
        forwarder = _create_forwarder(send_fn)

        # Risk arrives FIRST (out of order)
        forwarder.handle(_make_risk(signal_id="sig-ooo", contract_id=""))
        _drain(forwarder, 0.3)
        assert send_fn.call_count == 0  # buffered, not sent yet

        # Signal arrives LATER
        forwarder.handle(_make_signal(signal_id="sig-ooo"))
        _drain(forwarder, 0.5)
        forwarder.stop()

        # Now the message should have been sent WITH signal correlation
        assert send_fn.call_count >= 1
        combined = _combined_output(send_fn)
        assert "AAPL" in combined
        assert forwarder.correlation_metrics.late_arrivals >= 1

    def test_risk_before_signal_does_not_produce_false_timeout(self):
        """When risk arrives before signal and is reconciled,
        no false 'NO VERDICT RECEIVED' should appear."""
        send_fn = MagicMock()
        forwarder = _create_forwarder(send_fn, timeout_seconds=1.0,
                                      timeout_interval=0.5)

        # Out-of-order: risk then signal
        forwarder.handle(_make_risk(signal_id="sig-no-false-timeout", contract_id=""))
        forwarder.handle(_make_signal(signal_id="sig-no-false-timeout"))
        time.sleep(2.0)  # Wait past timeout period
        forwarder.stop()

        combined = _combined_output(send_fn)
        # Should NOT contain timeout warning since verdict was delivered
        assert "NO VERDICT RECEIVED" not in combined


# ─────────────────────────────────────────────────────────────────────────────
# 6. Proactive Timeout Generation
# ─────────────────────────────────────────────────────────────────────────────


class TestProactiveTimeout:
    """Background timer fires timeouts without needing another event."""

    def test_timeout_fires_without_any_subsequent_event(self):
        """The background timer should emit 'NO VERDICT' even if no other
        events arrive after the signal."""
        send_fn = MagicMock()
        forwarder = _create_forwarder(send_fn, timeout_seconds=1.0,
                                      timeout_interval=0.5)

        forwarder.handle(_make_signal(signal_id="sig-proactive-timeout"))
        # Wait for background timer to detect and emit timeout
        time.sleep(2.5)
        forwarder.stop()

        combined = _combined_output(send_fn)
        assert "NO VERDICT RECEIVED" in combined
        assert "no action taken" in combined
        assert forwarder.correlation_metrics.timeouts_emitted >= 1

    def test_verdict_cancels_timeout_atomically(self):
        """Processing a verdict cancels the pending timeout so no false
        timeout is emitted after the verdict."""
        send_fn = MagicMock()
        forwarder = _create_forwarder(send_fn, timeout_seconds=2.0,
                                      timeout_interval=0.5)

        forwarder.handle(_make_signal(signal_id="sig-cancel-timeout"))
        time.sleep(0.3)  # Signal tracked, deadline set
        forwarder.handle(_make_risk(signal_id="sig-cancel-timeout", contract_id=""))
        time.sleep(3.0)  # Wait past what would have been the timeout
        forwarder.stop()

        combined = _combined_output(send_fn)
        assert "NO VERDICT RECEIVED" not in combined


# ─────────────────────────────────────────────────────────────────────────────
# 7. Correlation TTL Cleanup
# ─────────────────────────────────────────────────────────────────────────────


class TestCorrelationCleanup:
    """TTL expiry and bounded store behavior."""

    def test_expired_signal_not_returned(self):
        """After TTL, stored signals are not returned."""
        store = InMemoryCorrelationStore(signal_ttl=0.5)
        store.store_signal("sig-ttl", {"test": True})
        assert store.get_signal("sig-ttl") is not None

        time.sleep(0.7)
        assert store.get_signal("sig-ttl") is None
        assert store.metrics.signals_expired >= 1

    def test_expired_dedup_key_allows_resend(self):
        """After dedup TTL, the same intent can be sent again."""
        store = InMemoryCorrelationStore(dedup_ttl=0.5)
        store.mark_sent("intent:test")
        assert store.check_dedup("intent:test") is True

        time.sleep(0.7)
        assert store.check_dedup("intent:test") is False

    def test_cleanup_removes_stale_entries(self):
        """Explicit cleanup purges all expired state."""
        store = InMemoryCorrelationStore(signal_ttl=0.3, dedup_ttl=0.3)
        for i in range(100):
            store.store_signal(f"sig-{i}", {"i": i})
            store.mark_sent(f"intent-{i}")

        time.sleep(0.5)
        removed = store.cleanup_expired()
        assert removed >= 100  # At least the signals and dedup keys

    def test_bounded_store_evicts_oldest_on_overflow(self):
        """When max_entries is reached, oldest entries are evicted."""
        store = InMemoryCorrelationStore(max_entries=50)
        for i in range(60):
            store.store_signal(f"sig-{i}", {"i": i})

        # Oldest entries should have been evicted
        assert store.get_signal("sig-0") is None
        assert store.metrics.orphaned_events >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 8. Signal ID Collision Resistance
# ─────────────────────────────────────────────────────────────────────────────


class TestSignalIdCollisionResistance:
    """Verify UUIDv4 signal_id uniqueness."""

    def test_signal_ids_are_full_uuid_length(self):
        """TradeSignal generates 128-bit (32 hex char) signal IDs."""
        from src.strategy.base import TradeSignal
        ids = set()
        for _ in range(10_000):
            sig = TradeSignal(symbol="AAPL")
            ids.add(sig.signal_id)
            # Format: SIG-<32 hex chars>
            assert sig.signal_id.startswith("SIG-")
            hex_part = sig.signal_id[4:]
            assert len(hex_part) == 32
            int(hex_part, 16)  # Must be valid hex

        # All 10,000 IDs must be unique
        assert len(ids) == 10_000

    def test_backward_compat_old_format_accepted(self):
        """Old 8-char signal_ids still work in the correlation pipeline."""
        send_fn = MagicMock()
        forwarder = _create_forwarder(send_fn)

        # Simulate old-format signal_id (backward compat)
        forwarder.handle(_make_signal(signal_id="SIG-abcd1234"))
        forwarder.handle(_make_risk(signal_id="SIG-abcd1234", contract_id=""))

        _drain(forwarder, 0.5)
        forwarder.stop()

        assert send_fn.call_count >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 9. Metrics Accuracy
# ─────────────────────────────────────────────────────────────────────────────


class TestMetricsAccuracy:
    """Verify correlation metrics counters are accurate."""

    def test_correlated_vs_uncorrelated_counts(self):
        """Verify verdicts_correlated and verdicts_uncorrelated."""
        send_fn = MagicMock()
        forwarder = _create_forwarder(send_fn)

        # Correlated: signal then risk
        forwarder.handle(_make_signal(signal_id="sig-corr"))
        forwarder.handle(_make_risk(signal_id="sig-corr", contract_id=""))

        # Uncorrelated: risk with unknown signal_id (no buffering, empty)
        forwarder.handle(_make_risk(signal_id="", contract_id=""))

        _drain(forwarder, 0.5)
        forwarder.stop()

        m = forwarder.correlation_metrics
        assert m.verdicts_correlated >= 1
        assert m.verdicts_uncorrelated >= 1

    def test_duplicate_suppression_counter(self):
        """Verify duplicates_suppressed counter."""
        send_fn = MagicMock()
        forwarder = _create_forwarder(send_fn)

        forwarder.handle(_make_signal())
        forwarder.handle(_make_risk(contract_id=""))
        forwarder.handle(_make_risk(contract_id=""))  # duplicate
        forwarder.handle(_make_risk(contract_id=""))  # triple

        _drain(forwarder, 0.5)
        forwarder.stop()

        assert forwarder.correlation_metrics.duplicates_suppressed == 2

    def test_metrics_serializable(self):
        """Metrics can be serialized to dict for monitoring."""
        store = InMemoryCorrelationStore()
        store.store_signal("s1", {})
        d = store.metrics.to_dict()
        assert isinstance(d, dict)
        assert "signals_tracked" in d
        assert d["signals_tracked"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 10. EventBus Full Pipeline Integration
# ─────────────────────────────────────────────────────────────────────────────


class TestEventBusFullPipeline:
    """Drive events through the real EventBus, not forwarder.handle() directly."""

    def test_event_bus_delivers_correlated_flow(self):
        """Full flow through EventBus: signal → contract → risk → Telegram."""
        send_fn = MagicMock()
        bus = EventBus()
        store = InMemoryCorrelationStore()
        forwarder = _create_forwarder(send_fn, store=store)
        forwarder.register(bus)

        bus.publish(_make_signal(signal_id="sig-bus-1"))
        bus.publish(_make_contract(signal_id="sig-bus-1", contract_id="c-bus-1"))
        bus.publish(_make_risk(signal_id="sig-bus-1", contract_id="c-bus-1"))

        _drain(forwarder, 1.0)
        forwarder.stop()

        # Contract message + actionable intent message
        assert send_fn.call_count == 2
        combined = _combined_output(send_fn)
        assert "AAPL" in combined

    def test_event_bus_out_of_order_reconciliation(self):
        """EventBus: risk arrives before signal and is reconciled."""
        send_fn = MagicMock()
        bus = EventBus()
        store = InMemoryCorrelationStore()
        forwarder = _create_forwarder(send_fn, store=store)
        forwarder.register(bus)

        # Out of order through the bus
        bus.publish(_make_risk(signal_id="sig-ooo-bus", contract_id=""))
        _drain(forwarder, 0.3)
        assert send_fn.call_count == 0  # buffered

        bus.publish(_make_signal(signal_id="sig-ooo-bus"))
        _drain(forwarder, 0.5)
        forwarder.stop()

        assert send_fn.call_count >= 1
        assert store.metrics.late_arrivals >= 1

    def test_event_bus_timeout_without_verdict(self):
        """EventBus: signal without verdict times out proactively."""
        send_fn = MagicMock()
        bus = EventBus()
        forwarder = _create_forwarder(send_fn, timeout_seconds=1.0,
                                      timeout_interval=0.5)
        forwarder.register(bus)

        bus.publish(_make_signal(signal_id="sig-bus-timeout"))
        time.sleep(2.5)
        forwarder.stop()

        combined = _combined_output(send_fn)
        assert "NO VERDICT RECEIVED" in combined

    def test_event_bus_multiple_signals_independent(self):
        """Multiple independent signals processed correctly through bus."""
        send_fn = MagicMock()
        bus = EventBus()
        forwarder = _create_forwarder(send_fn)
        forwarder.register(bus)

        # Signal A
        bus.publish(_make_signal(signal_id="sig-A", symbol="AAPL"))
        bus.publish(_make_risk(signal_id="sig-A", contract_id="", symbol="AAPL"))

        # Signal B
        bus.publish(_make_signal(signal_id="sig-B", symbol="TSLA",
                                 side="SELL"))
        bus.publish(_make_risk(signal_id="sig-B", contract_id="", symbol="TSLA"))

        _drain(forwarder, 1.0)
        forwarder.stop()

        combined = _combined_output(send_fn)
        assert "AAPL" in combined
        assert "TSLA" in combined
        # Two separate intent messages
        assert send_fn.call_count == 2

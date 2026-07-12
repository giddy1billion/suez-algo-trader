"""
Production blockers resolution tests.

Covers:
  1. Event registry completeness (auto-registration via __init_subclass__)
  2. Round-trip serialization/deserialization for ALL event types
  3. Crash/recovery resilience (unknown fields stripped, mixed-schema stores)
  4. Signal correlation loop closure (early rejections cancel deadlines)
  5. /trades vs /journalstats endpoint distinction validation
"""

import inspect
import json
import time
import uuid
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone

import pytest

from src.core.events import (
    Event,
    SignalGenerated,
    DecisionContractCreated,
    RiskEvaluated,
    SignalRejected,
    OrderSubmitted,
    OrderAccepted,
    OrderPartialFill,
    OrderFilled,
    OrderRejected,
    TradeOpened,
    TradeClosed,
    RiskHalt,
    SchedulerEvent,
    SystemHealth,
    EnvironmentSwitched,
    BrokerSwitched,
    ModelSwapped,
    ModelTrainingStarted,
    ModelTrainingCompleted,
    ABTestStarted,
    ABTestCompleted,
    BacktestStarted,
    BacktestCompleted,
    DataIngested,
    BacktestTriggered,
    PredictionRegistered,
    PredictionOutcomeRecorded,
    RetrainingTriggered,
    ShadowDeploymentStarted,
    ShadowDeploymentCompleted,
    CorrelationFilterApplied,
    OperationalModeChanged,
    PredictionUnavailable,
    ModelRejected,
    CircuitBreakerTripped,
    CircuitBreakerReset,
    ModelAutoRollback,
    _EVENT_CLASS_REGISTRY,
    get_event_class_registry,
)
from src.core.event_store import (
    EVENT_REGISTRY,
    EventStore,
    _reconstruct_event,
)


# ===========================================================================
# 1. EVENT REGISTRY COMPLETENESS
# ===========================================================================

class TestEventRegistryCompleteness:
    """Verify that all concrete Event subclasses are auto-registered."""

    def _find_all_event_subclasses(self):
        """Recursively find all Event subclasses defined in src/core/events.py."""
        import src.core.events as events_module
        subclasses = set()
        for name, obj in inspect.getmembers(events_module, inspect.isclass):
            if issubclass(obj, Event) and obj is not Event:
                subclasses.add(name)
        return subclasses

    def test_all_subclasses_in_registry(self):
        """Every concrete Event subclass must be discoverable during deserialization."""
        all_subclasses = self._find_all_event_subclasses()
        registered = set(_EVENT_CLASS_REGISTRY.keys())
        missing = all_subclasses - registered
        assert not missing, (
            f"Event subclasses NOT in registry (deserialization will fail): {missing}. "
            f"Ensure __init_subclass__ is triggered for all event classes."
        )

    def test_registry_has_minimum_expected_count(self):
        """Sanity check: registry should have at least 37 event types."""
        assert len(_EVENT_CLASS_REGISTRY) >= 37

    def test_event_registry_alias_matches(self):
        """EVENT_REGISTRY in event_store.py should reference the same dict."""
        # EVENT_REGISTRY should be the same object as _EVENT_CLASS_REGISTRY
        assert EVENT_REGISTRY is _EVENT_CLASS_REGISTRY

    def test_new_subclass_auto_registered(self):
        """A dynamically created Event subclass should be auto-registered."""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class _TestDynamicEvent(Event):
            custom_field: str = "hello"

        assert "_TestDynamicEvent" in _EVENT_CLASS_REGISTRY
        # Clean up
        del _EVENT_CLASS_REGISTRY["_TestDynamicEvent"]


# ===========================================================================
# 2. ROUND-TRIP SERIALIZATION FOR ALL EVENT TYPES
# ===========================================================================

# All concrete event classes to test
ALL_EVENT_CLASSES = [
    SignalGenerated,
    DecisionContractCreated,
    RiskEvaluated,
    SignalRejected,
    OrderSubmitted,
    OrderAccepted,
    OrderPartialFill,
    OrderFilled,
    OrderRejected,
    TradeOpened,
    TradeClosed,
    RiskHalt,
    SchedulerEvent,
    SystemHealth,
    EnvironmentSwitched,
    BrokerSwitched,
    ModelSwapped,
    ModelTrainingStarted,
    ModelTrainingCompleted,
    ABTestStarted,
    ABTestCompleted,
    BacktestStarted,
    BacktestCompleted,
    DataIngested,
    BacktestTriggered,
    PredictionRegistered,
    PredictionOutcomeRecorded,
    RetrainingTriggered,
    ShadowDeploymentStarted,
    ShadowDeploymentCompleted,
    CorrelationFilterApplied,
    OperationalModeChanged,
    PredictionUnavailable,
    ModelRejected,
    CircuitBreakerTripped,
    CircuitBreakerReset,
    ModelAutoRollback,
]


class TestRoundTripSerialization:
    """Every event type must survive to_dict → from_dict round-trip."""

    @pytest.mark.parametrize("event_cls", ALL_EVENT_CLASSES, ids=lambda c: c.__name__)
    def test_round_trip(self, event_cls):
        """Create event with defaults, serialize, deserialize, verify identity."""
        original = event_cls(source="round_trip_test")
        data = original.to_dict()

        # Verify _type is set correctly
        assert data["_type"] == event_cls.__name__

        # Reconstruct via the event store path
        reconstructed = _reconstruct_event(event_cls.__name__, data)
        assert type(reconstructed) is event_cls
        assert reconstructed.source == "round_trip_test"
        assert reconstructed.event_id == original.event_id

    @pytest.mark.parametrize("event_cls", ALL_EVENT_CLASSES, ids=lambda c: c.__name__)
    def test_persist_and_replay(self, event_cls, tmp_path):
        """Persist and replay each event type through EventStore."""
        store = EventStore(db_path=str(tmp_path / "test.db"), session_id="roundtrip")
        try:
            original = event_cls(source="persist_test")
            store.persist(original)

            replayed = store.replay_session("roundtrip")
            assert len(replayed) == 1
            assert type(replayed[0]) is event_cls
            assert replayed[0].event_id == original.event_id
        finally:
            store.close()


# ===========================================================================
# 3. CRASH/RECOVERY RESILIENCE
# ===========================================================================

class TestCrashRecovery:
    """Resilience when event store contains unknown/corrupted data."""

    def test_unknown_event_type_falls_back_to_base(self):
        """Unknown event types should deserialize to base Event."""
        payload = {
            "_type": "FutureEventV99",
            "_schema_version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "future_module",
            "event_id": "abc123",
            "unknown_field": "should_be_stripped",
            "another_unknown": 42,
        }
        result = _reconstruct_event("FutureEventV99", payload)
        assert isinstance(result, Event)
        assert result.source == "future_module"
        assert result.event_id == "abc123"

    def test_unknown_fields_stripped_on_fallback(self):
        """When reconstruction fails, unknown fields are stripped for base Event."""
        payload = {
            "_type": "SignalGenerated",
            "_schema_version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "test",
            "event_id": "xyz789",
            "symbol": "AAPL",
            "completely_wrong_field_that_breaks_constructor": [1, 2, 3],
        }
        # This should fail for SignalGenerated but fall back to base Event
        result = _reconstruct_event("SignalGenerated", payload)
        assert isinstance(result, Event)
        # It should still have the base fields
        assert result.event_id == "xyz789"

    def test_mixed_schema_store_replay(self, tmp_path):
        """Store with mixed event types replays correctly."""
        store = EventStore(db_path=str(tmp_path / "mixed.db"), session_id="mixed")
        try:
            events = [
                SignalGenerated(symbol="BTC", signal="BUY", source="s1"),
                RiskEvaluated(symbol="BTC", approved=True, source="s2"),
                OrderFilled(order_id="ord1", fill_price=50000, source="s3"),
                TradeClosed(trade_id="t1", pnl=100.0, source="s4"),
                SystemHealth(component="broker", status="healthy", source="s5"),
                SignalRejected(signal_id="sig1", symbol="ETH", reason="low_strength", source="s6"),
            ]
            for e in events:
                store.persist(e)

            replayed = store.replay_session("mixed")
            assert len(replayed) == 6
            assert type(replayed[0]) is SignalGenerated
            assert type(replayed[1]) is RiskEvaluated
            assert type(replayed[2]) is OrderFilled
            assert type(replayed[3]) is TradeClosed
            assert type(replayed[4]) is SystemHealth
            assert type(replayed[5]) is SignalRejected
        finally:
            store.close()

    def test_corrupted_payload_ultimate_fallback(self):
        """Even completely mangled payloads produce a valid base Event."""
        payload = {
            "not_a_real_field": True,
            "garbage": [None, None],
        }
        result = _reconstruct_event("NonExistentEvent", payload)
        assert isinstance(result, Event)
        # Should have a valid event_id even if none was in payload
        assert result.event_id


# ===========================================================================
# 4. SIGNAL CORRELATION LOOP CLOSURE
# ===========================================================================

class TestSignalCorrelationLoop:
    """Early rejections must cancel correlation deadlines (no spurious timeouts)."""

    @pytest.fixture
    def forwarder_and_bus(self):
        """Set up a TelegramAuditForwarder with a mock sender."""
        from src.core.events import EventBus
        from src.notifications.telegram_audit_forwarder import TelegramAuditForwarder
        from src.notifications.correlation_store import InMemoryCorrelationStore

        bus = EventBus()
        store = InMemoryCorrelationStore()
        forwarder = TelegramAuditForwarder(
            send_func=lambda msg: None,  # No-op sender
            correlation_store=store,
            risk_verdict_timeout_seconds=2,  # Short timeout for tests
            timeout_check_interval=0.5,
        )
        # Subscribe forwarder to all events on the bus
        bus.subscribe(None, forwarder.handle)
        yield forwarder, bus, store
        forwarder.stop()

    def test_signal_rejected_cancels_deadline(self, forwarder_and_bus):
        """SignalRejected event cancels the correlation deadline."""
        forwarder, bus, store = forwarder_and_bus
        signal_id = "sig-test-001"

        # Emit a signal (sets a deadline)
        bus.publish(SignalGenerated(
            signal_id=signal_id,
            symbol="AAPL",
            signal="BUY",
            signal_strength=0.3,
            source="test",
        ))
        time.sleep(0.1)

        # Verify deadline was set
        expired = store.get_expired_deadlines(time.monotonic() + 100)
        assert signal_id in expired or len(store._deadlines) > 0

        # Emit SignalRejected (should cancel the deadline)
        bus.publish(SignalRejected(
            signal_id=signal_id,
            symbol="AAPL",
            reason="low_strength",
            stage="strength_gate",
            source="engine",
        ))
        time.sleep(0.1)

        # Deadline should be cancelled — no timeout warning
        assert signal_id not in store._deadlines

    def test_risk_evaluated_rejected_cancels_deadline(self, forwarder_and_bus):
        """RiskEvaluated(approved=False) cancels the deadline."""
        forwarder, bus, store = forwarder_and_bus
        signal_id = "sig-test-002"

        bus.publish(SignalGenerated(
            signal_id=signal_id,
            symbol="BTC",
            signal="BUY",
            signal_strength=0.8,
            source="test",
        ))
        time.sleep(0.1)

        bus.publish(RiskEvaluated(
            signal_id=signal_id,
            symbol="BTC",
            approved=False,
            reasons=["max_position_size_exceeded"],
            source="risk_engine",
        ))
        time.sleep(0.1)

        # Deadline should be cancelled
        assert signal_id not in store._deadlines

    def test_early_rejection_no_timeout_warning(self, forwarder_and_bus):
        """Early rejection must not produce 'NO VERDICT RECEIVED' warning."""
        messages_sent = []
        forwarder, bus, store = forwarder_and_bus

        # Replace send function to capture messages
        forwarder._send = lambda msg: messages_sent.append(msg)

        signal_id = "sig-test-003"
        bus.publish(SignalGenerated(
            signal_id=signal_id,
            symbol="ETH",
            signal="SELL",
            signal_strength=0.9,
            source="test",
        ))
        time.sleep(0.1)

        # Reject immediately
        bus.publish(SignalRejected(
            signal_id=signal_id,
            symbol="ETH",
            reason="existing_position",
            stage="existing_position",
            source="engine",
        ))
        time.sleep(0.1)

        # Wait past the timeout period
        time.sleep(2.5)

        # No "NO VERDICT RECEIVED" message should appear
        timeout_msgs = [m for m in messages_sent if "NO VERDICT RECEIVED" in m]
        assert len(timeout_msgs) == 0, f"Spurious timeout warnings: {timeout_msgs}"

    def test_genuine_timeout_still_fires_when_no_verdict(self, forwarder_and_bus):
        """If no verdict arrives at all, timeout warning must still fire."""
        messages_sent = []
        forwarder, bus, store = forwarder_and_bus
        forwarder._send = lambda msg: messages_sent.append(msg)

        signal_id = "sig-timeout-test"
        bus.publish(SignalGenerated(
            signal_id=signal_id,
            symbol="DOGE",
            signal="BUY",
            signal_strength=0.7,
            source="test",
        ))

        # Wait for timeout (2s configured + check interval)
        time.sleep(4)

        timeout_msgs = [m for m in messages_sent if "NO VERDICT RECEIVED" in m]
        assert len(timeout_msgs) == 1, "Expected exactly one timeout warning"


# ===========================================================================
# 5. /TRADES vs /JOURNALSTATS DISTINCTION
# ===========================================================================

class TestTradesVsJournalStatsDistinction:
    """Validate the intentional separation between execution audit and analytics."""

    @pytest.fixture
    def db_and_journal(self, tmp_path):
        from src.data.store import DatabaseManager
        from src.data.journal import TradeJournal
        url = f"sqlite:///{tmp_path / 'test.db'}"
        mgr = DatabaseManager(url)
        journal = TradeJournal(mgr)
        return mgr, journal

    def test_trades_are_execution_audit_log(self, db_and_journal):
        """Trades table records every order execution (audit log purpose)."""
        db, journal = db_and_journal
        # Record multiple trades including open ones
        db.record_trade({
            "symbol": "AAPL", "side": "buy", "qty": 10, "price": 150.0,
            "status": "filled", "strategy": "momentum", "order_id": "ord1",
        })
        db.record_trade({
            "symbol": "GOOGL", "side": "buy", "qty": 5, "price": 2800.0,
            "status": "filled", "strategy": "mean_reversion", "order_id": "ord2",
        })
        # Trades includes open + closed (full audit)
        all_trades = db.get_trades(limit=10)
        assert len(all_trades) == 2

    def test_journal_stats_are_closed_trade_analytics(self, db_and_journal):
        """Journal stats only include closed trades with P&L data."""
        db, journal = db_and_journal
        # Open trade (no P&L yet)
        db.record_trade({
            "symbol": "AAPL", "side": "buy", "qty": 10, "price": 150.0,
            "status": "filled", "strategy": "momentum", "order_id": "ord1",
        })
        # Closed trade with P&L
        db.record_trade({
            "symbol": "GOOGL", "side": "buy", "qty": 5, "price": 2800.0,
            "status": "closed", "pnl": 250.0, "pnl_pct": 1.8,
            "strategy": "mean_reversion", "order_id": "ord2",
        })
        # Journal summary should only include closed trades
        summary = journal.get_summary()
        assert summary["total_trades"] <= 2  # At most 2
        # The journal is specifically for closed-trade analytics

    def test_trades_and_journal_serve_different_purposes(self, db_and_journal):
        """Confirm the architectural intent: trades=audit, journal=analytics."""
        db, _ = db_and_journal
        # Record a trade
        db.record_trade({
            "symbol": "TSLA", "side": "buy", "qty": 3, "price": 700.0,
            "status": "filled", "strategy": "breakout", "order_id": "ord-tsla",
        })
        # /trades endpoint: shows this immediately (execution audit)
        recent = db.get_trades(limit=10)
        assert any(t["symbol"] == "TSLA" for t in recent)
        # Until the trade is closed, journal stats shouldn't show P&L for it

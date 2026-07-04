"""Tests for the persistent EventStore and EventPersistenceSubscriber."""

import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.core.event_store import (
    EVENT_REGISTRY,
    EventPersistenceSubscriber,
    EventStore,
    _reconstruct_event,
    register_event_class,
)
from src.core.events import (
    Event,
    EventBus,
    OrderFilled,
    RiskHalt,
    SignalGenerated,
    SystemHealth,
    TradeClosed,
    TradeOpened,
)


@pytest.fixture
def db_path(tmp_path):
    """Provide a temporary DB path for tests."""
    return str(tmp_path / "test_events.db")


@pytest.fixture
def store(db_path):
    """Create an EventStore instance for testing."""
    es = EventStore(db_path=db_path, session_id="test-session-001")
    yield es
    es.close()


@pytest.fixture
def event_bus():
    """Create a fresh EventBus."""
    return EventBus()


# ---------------------------------------------------------------------------
# EventStore Tests
# ---------------------------------------------------------------------------

class TestEventStorePersist:
    """Tests for persisting events."""

    def test_persist_basic_event(self, store):
        event = Event(source="test", event_id="evt001")
        store.persist(event)
        assert store.count_events() == 1

    def test_persist_signal_generated(self, store):
        event = SignalGenerated(
            symbol="BTCUSDT",
            signal="BUY",
            confidence=0.85,
            strategy="macd_cross",
            price=50000.0,
            source="strategy_engine",
        )
        store.persist(event)
        events = store.get_session_events("test-session-001")
        assert len(events) == 1
        assert events[0]["event_type"] == "SignalGenerated"

    def test_persist_multiple_events(self, store):
        for i in range(10):
            store.persist(Event(source=f"src_{i}"))
        assert store.count_events() == 10

    def test_persist_stores_correct_payload(self, store):
        event = OrderFilled(
            order_id="ORD123",
            fill_price=100.5,
            fill_qty=2.0,
            fees=0.1,
            source="broker",
        )
        store.persist(event)
        events = store.get_latest_events(1)
        assert events[0]["event_type"] == "OrderFilled"
        assert events[0]["source"] == "broker"


class TestEventStoreRetrieve:
    """Tests for retrieving events."""

    def test_get_session_events(self, db_path):
        store1 = EventStore(db_path=db_path, session_id="session-A")
        store2 = EventStore(db_path=db_path, session_id="session-B")

        store1.persist(Event(source="A"))
        store2.persist(Event(source="B"))
        store2.persist(Event(source="B2"))

        events_a = store1.get_session_events("session-A")
        events_b = store1.get_session_events("session-B")

        assert len(events_a) == 1
        assert len(events_b) == 2

        store1.close()
        store2.close()

    def test_get_events_by_type(self, store):
        store.persist(SignalGenerated(symbol="BTC", signal="BUY"))
        store.persist(RiskHalt(reason="drawdown"))
        store.persist(SignalGenerated(symbol="ETH", signal="SELL"))

        signals = store.get_events_by_type("SignalGenerated")
        assert len(signals) == 2

        halts = store.get_events_by_type("RiskHalt")
        assert len(halts) == 1

    def test_get_events_by_type_with_limit(self, store):
        for i in range(20):
            store.persist(Event(source=f"src_{i}"))

        events = store.get_events_by_type("Event", limit=5)
        assert len(events) == 5

    def test_get_events_since(self, store):
        past = datetime(2020, 1, 1, tzinfo=timezone.utc)
        recent = datetime.now(timezone.utc) - timedelta(minutes=5)

        store.persist(Event(timestamp=past, source="old"))
        store.persist(Event(source="new"))

        since_recent = store.get_events_since(recent)
        assert len(since_recent) == 1
        assert since_recent[0]["source"] == "new"

    def test_get_latest_events(self, store):
        for i in range(10):
            store.persist(Event(source=f"evt_{i}"))

        latest = store.get_latest_events(3)
        assert len(latest) == 3
        # Most recent first
        assert latest[0]["source"] == "evt_9"

    def test_count_events_all(self, store):
        store.persist(Event())
        store.persist(Event())
        assert store.count_events() == 2

    def test_count_events_by_session(self, db_path):
        store = EventStore(db_path=db_path, session_id="s1")
        store.persist(Event())
        store.persist(Event())

        store2 = EventStore(db_path=db_path, session_id="s2")
        store2.persist(Event())

        assert store.count_events(session_id="s1") == 2
        assert store.count_events(session_id="s2") == 1

        store.close()
        store2.close()


class TestEventStoreReplay:
    """Tests for event replay/reconstruction."""

    def test_replay_session_basic(self, store):
        store.persist(SignalGenerated(symbol="BTC", signal="BUY", confidence=0.9))
        store.persist(OrderFilled(order_id="O1", fill_price=50000.0, fill_qty=1.0))

        events = store.replay_session("test-session-001")
        assert len(events) == 2
        assert isinstance(events[0], SignalGenerated)
        assert events[0].symbol == "BTC"
        assert events[0].confidence == 0.9
        assert isinstance(events[1], OrderFilled)
        assert events[1].order_id == "O1"

    def test_replay_preserves_types(self, store):
        store.persist(TradeOpened(trade_id="T1", symbol="ETH", side="BUY", entry_price=3000.0))
        store.persist(TradeClosed(trade_id="T1", symbol="ETH", exit_price=3500.0, pnl=500.0))

        events = store.replay_session("test-session-001")
        assert isinstance(events[0], TradeOpened)
        assert isinstance(events[1], TradeClosed)
        assert events[1].pnl == 500.0

    def test_replay_unknown_type_falls_back(self, store):
        # Manually insert an event with unknown type
        import json

        payload = json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "source": "x", "event_id": "e1"})
        store._conn.execute(
            "INSERT INTO events (event_type, event_id, timestamp, source, payload, session_id) VALUES (?, ?, ?, ?, ?, ?)",
            ("UnknownEvent", "e1", datetime.now(timezone.utc).isoformat(), "x", payload, "test-session-001"),
        )
        store._conn.commit()

        events = store.replay_session("test-session-001")
        assert len(events) == 1
        assert isinstance(events[0], Event)


class TestEventStoreCleanup:
    """Tests for event cleanup."""

    def test_cleanup_old_events(self, store):
        old_ts = datetime.now(timezone.utc) - timedelta(days=60)
        recent_ts = datetime.now(timezone.utc)

        store.persist(Event(timestamp=old_ts, source="old"))
        store.persist(Event(timestamp=recent_ts, source="recent"))

        assert store.count_events() == 2
        store.cleanup_old_events(days=30)
        assert store.count_events() == 1

        remaining = store.get_latest_events(10)
        assert remaining[0]["source"] == "recent"


# ---------------------------------------------------------------------------
# EventPersistenceSubscriber Tests
# ---------------------------------------------------------------------------

class TestEventPersistenceSubscriber:
    """Tests for the subscriber that bridges EventBus → EventStore."""

    def test_subscriber_persists_events(self, store, event_bus):
        subscriber = EventPersistenceSubscriber(store)
        subscriber.attach(event_bus)

        event_bus.publish(SignalGenerated(symbol="BTC", signal="BUY"))
        event_bus.publish(RiskHalt(reason="test"))

        assert store.count_events() == 2

    def test_subscriber_handles_serialization_error(self, store, event_bus):
        """Subscriber should not crash on serialization errors."""
        subscriber = EventPersistenceSubscriber(store)
        subscriber.attach(event_bus)

        # Close the store to force an error
        store.close()

        # Should not raise
        event_bus.publish(Event(source="will_fail"))

    def test_subscriber_captures_all_event_types(self, store, event_bus):
        subscriber = EventPersistenceSubscriber(store)
        subscriber.attach(event_bus)

        event_bus.publish(SignalGenerated(symbol="X", signal="BUY"))
        event_bus.publish(SystemHealth(component="db", status="healthy"))
        event_bus.publish(OrderFilled(order_id="O1", fill_price=100.0))

        assert store.count_events() == 3
        types = {e["event_type"] for e in store.get_session_events("test-session-001")}
        assert types == {"SignalGenerated", "SystemHealth", "OrderFilled"}


# ---------------------------------------------------------------------------
# Registry Tests
# ---------------------------------------------------------------------------

class TestEventRegistry:
    """Tests for event class registry."""

    def test_all_event_classes_registered(self):
        expected = [
            "Event", "SignalGenerated", "RiskEvaluated", "OrderSubmitted",
            "OrderAccepted", "OrderPartialFill", "OrderFilled", "OrderRejected",
            "TradeOpened", "TradeClosed", "RiskHalt", "SchedulerEvent", "SystemHealth",
        ]
        for name in expected:
            assert name in EVENT_REGISTRY

    def test_register_custom_event(self):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class CustomEvent(Event):
            custom_field: str = ""

        register_event_class(CustomEvent)
        assert "CustomEvent" in EVENT_REGISTRY

        # Clean up
        del EVENT_REGISTRY["CustomEvent"]

    def test_reconstruct_event(self):
        payload = {
            "_type": "SignalGenerated",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "test",
            "event_id": "abc123",
            "symbol": "ETH",
            "signal": "SELL",
            "confidence": 0.75,
            "strategy": "rsi",
            "price": 2000.0,
            "indicators": {},
        }
        event = _reconstruct_event("SignalGenerated", payload)
        assert isinstance(event, SignalGenerated)
        assert event.symbol == "ETH"
        assert event.confidence == 0.75

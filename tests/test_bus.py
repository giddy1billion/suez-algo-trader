"""Tests for Message Bus abstraction and Replay modes."""

import json
import pytest
from unittest.mock import MagicMock, patch, call

from src.core.bus import MessageBus, InMemoryBus, PersistentBus, create_bus
from src.core.events import (
    Event,
    SignalGenerated,
    RiskEvaluated,
    OrderSubmitted,
    OrderFilled,
    OrderRejected,
    TradeOpened,
    TradeClosed,
)
from src.core.replay import ReplayEngine, ReplayReport


# ---------------------------------------------------------------------------
# InMemoryBus Tests
# ---------------------------------------------------------------------------


class TestInMemoryBus:
    def test_publish_and_subscribe(self):
        bus = InMemoryBus()
        received = []
        bus.subscribe(SignalGenerated, lambda e: received.append(e))

        event = SignalGenerated(symbol="BTCUSDT", signal="BUY", confidence=0.9)
        bus.publish(event)

        assert len(received) == 1
        assert received[0] is event

    def test_wildcard_subscription(self):
        bus = InMemoryBus()
        received = []
        bus.subscribe(None, lambda e: received.append(e))

        bus.publish(SignalGenerated(symbol="ETH", signal="BUY"))
        bus.publish(OrderFilled(order_id="123", fill_price=100.0))

        assert len(received) == 2

    def test_unsubscribe(self):
        bus = InMemoryBus()
        received = []
        handler = lambda e: received.append(e)

        bus.subscribe(SignalGenerated, handler)
        bus.publish(SignalGenerated(symbol="X", signal="BUY"))
        assert len(received) == 1

        bus.unsubscribe(SignalGenerated, handler)
        bus.publish(SignalGenerated(symbol="Y", signal="SELL"))
        assert len(received) == 1  # no new event received

    def test_get_history(self):
        bus = InMemoryBus(max_history=100)
        for i in range(10):
            bus.publish(SignalGenerated(symbol=f"SYM{i}", signal="BUY"))

        history = bus.get_history(limit=5)
        assert len(history) == 5

    def test_is_message_bus_instance(self):
        bus = InMemoryBus()
        assert isinstance(bus, MessageBus)


# ---------------------------------------------------------------------------
# PersistentBus Tests
# ---------------------------------------------------------------------------


class TestPersistentBus:
    def test_events_persisted_before_delivery(self):
        """Events should be persisted BEFORE subscribers receive them."""
        store = MagicMock()
        bus = PersistentBus(event_store=store)

        delivery_order = []

        def handler(e):
            # At this point, persist should already have been called
            delivery_order.append(("delivered", store.persist.call_count))

        store.persist.side_effect = lambda e: delivery_order.append(("persisted", 0))

        bus.subscribe(SignalGenerated, handler)
        event = SignalGenerated(symbol="BTC", signal="BUY")
        bus.publish(event)

        # persist was called
        store.persist.assert_called_once_with(event)
        # persisted happened before delivery
        assert delivery_order[0][0] == "persisted"
        assert delivery_order[1][0] == "delivered"

    def test_persist_failure_still_dispatches(self):
        """If persistence fails, event should still be dispatched."""
        store = MagicMock()
        store.persist.side_effect = RuntimeError("DB error")
        bus = PersistentBus(event_store=store)

        received = []
        bus.subscribe(None, lambda e: received.append(e))
        bus.publish(SignalGenerated(symbol="X", signal="BUY"))

        assert len(received) == 1

    def test_is_message_bus_instance(self):
        store = MagicMock()
        bus = PersistentBus(event_store=store)
        assert isinstance(bus, MessageBus)


# ---------------------------------------------------------------------------
# create_bus Factory Tests
# ---------------------------------------------------------------------------


class TestCreateBus:
    def test_memory_backend(self):
        bus = create_bus("memory")
        assert isinstance(bus, InMemoryBus)

    def test_persistent_backend(self):
        store = MagicMock()
        bus = create_bus("persistent", event_store=store)
        assert isinstance(bus, PersistentBus)

    def test_persistent_backend_requires_store(self):
        with pytest.raises(ValueError, match="event_store"):
            create_bus("persistent")

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="Unknown bus backend"):
            create_bus("kafka")

    def test_max_history_kwarg(self):
        bus = create_bus("memory", max_history=50)
        assert isinstance(bus, InMemoryBus)


# ---------------------------------------------------------------------------
# Replay Mode Tests
# ---------------------------------------------------------------------------


class TestReplayModes:
    """Test replay mode filtering."""

    def _make_store_with_events(self, events):
        """Create a mock event store that returns serialized events."""
        raw_events = []
        for e in events:
            raw_events.append({
                "event_type": type(e).__name__,
                "payload": json.dumps(e.to_dict()),
                "timestamp": e.timestamp.isoformat(),
                "event_id": e.event_id,
            })
        store = MagicMock()
        store.get_session_events.return_value = raw_events
        return store

    def test_strategy_mode_filters_signals_only(self):
        events = [
            SignalGenerated(symbol="BTC", signal="BUY", confidence=0.8, strategy="test"),
            RiskEvaluated(symbol="BTC", approved=True, risk_score=0.3),
            OrderSubmitted(symbol="BTC", side="BUY", qty=1.0, order_id="o1"),
            OrderFilled(order_id="o1", fill_price=50000.0, fill_qty=1.0),
            TradeOpened(trade_id="t1", symbol="BTC", side="BUY", entry_price=50000.0, qty=1.0),
        ]
        store = self._make_store_with_events(events)
        engine = ReplayEngine(store)
        report = engine.replay("test_session", mode="strategy")

        # Only SignalGenerated and RiskEvaluated should be replayed
        assert report.events_replayed == 2
        assert report.signals_count == 1

    def test_execution_mode_filters_orders_and_trades(self):
        events = [
            SignalGenerated(symbol="BTC", signal="BUY", confidence=0.8, strategy="test"),
            RiskEvaluated(symbol="BTC", approved=True, risk_score=0.3),
            OrderSubmitted(symbol="BTC", side="BUY", qty=1.0, order_id="o1"),
            OrderFilled(order_id="o1", fill_price=50000.0, fill_qty=1.0),
            TradeOpened(trade_id="t1", symbol="BTC", side="BUY", entry_price=50000.0, qty=1.0),
            TradeClosed(trade_id="t1", symbol="BTC", exit_price=51000.0, pnl=1000.0),
        ]
        store = self._make_store_with_events(events)
        engine = ReplayEngine(store)
        report = engine.replay("test_session", mode="execution")

        # OrderSubmitted, OrderFilled, TradeOpened, TradeClosed = 4
        assert report.events_replayed == 4
        assert report.trades_opened == 1
        assert report.trades_closed == 1

    def test_full_mode_replays_everything(self):
        events = [
            SignalGenerated(symbol="BTC", signal="BUY", confidence=0.8, strategy="test"),
            RiskEvaluated(symbol="BTC", approved=True, risk_score=0.3),
            OrderSubmitted(symbol="BTC", side="BUY", qty=1.0, order_id="o1"),
            OrderFilled(order_id="o1", fill_price=50000.0, fill_qty=1.0),
            TradeOpened(trade_id="t1", symbol="BTC", side="BUY", entry_price=50000.0, qty=1.0),
        ]
        store = self._make_store_with_events(events)
        engine = ReplayEngine(store)
        report = engine.replay("test_session", mode="full")

        assert report.events_replayed == 5

    def test_mode_combined_with_event_filter(self):
        events = [
            SignalGenerated(symbol="BTC", signal="BUY", confidence=0.8, strategy="test"),
            SignalGenerated(symbol="ETH", signal="SELL", confidence=0.6, strategy="test"),
            RiskEvaluated(symbol="BTC", approved=True, risk_score=0.3),
        ]
        store = self._make_store_with_events(events)
        engine = ReplayEngine(store)

        # strategy mode + filter to BTC only
        report = engine.replay(
            "test_session",
            mode="strategy",
            event_filter=lambda e: getattr(e, "symbol", "") == "BTC",
        )

        # Only BTC SignalGenerated and BTC RiskEvaluated pass both filters
        assert report.events_replayed == 2

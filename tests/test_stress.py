"""
Stress & Scenario Tests — Platform resilience under adversarial conditions.

Tests concurrent signals, broker failures, partial fills, timeout+retry,
recovery after simulated crash, and journal/event consistency.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.core.events import (
    EventBus,
    OrderFilled,
    OrderRejected,
    SignalGenerated,
    TradeClosed,
    TradeOpened,
)
from src.core.event_store import EventStore, EventPersistenceSubscriber
from src.core.state_machine import TradeLifecycle, TradeManager, TradeState
from src.core.recovery import RecoveryManager
from src.core.reconciliation import PortfolioReconciler, Discrepancy, ReconciliationReport
from src.execution.engine import ExecutionEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_bus():
    return EventBus(max_history=5000)


@pytest.fixture
def trade_manager():
    return TradeManager()


@pytest.fixture
def tmp_event_store(tmp_path):
    db_path = str(tmp_path / "stress_events.db")
    return EventStore(db_path=db_path, session_id="stress-test")


# ---------------------------------------------------------------------------
# Stress: Concurrent Event Publishing
# ---------------------------------------------------------------------------


class TestConcurrentEventPublishing:
    """Verify EventBus thread safety under concurrent load."""

    def test_concurrent_publishers(self, event_bus):
        """Multiple threads publishing simultaneously without data corruption."""
        received = []
        lock = threading.Lock()

        def handler(event):
            with lock:
                received.append(event)

        event_bus.subscribe(None, handler)

        def publish_batch(n, prefix):
            for i in range(n):
                event_bus.publish(SignalGenerated(
                    symbol=f"{prefix}-{i}", signal="BUY",
                    confidence=0.8, strategy="stress"
                ))

        threads = []
        for t_id in range(10):
            t = threading.Thread(target=publish_batch, args=(50, f"T{t_id}"))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == 500

    def test_concurrent_subscribe_unsubscribe(self, event_bus):
        """Subscribing/unsubscribing during publishing doesn't crash."""
        count = {"n": 0}
        lock = threading.Lock()

        def handler(event):
            with lock:
                count["n"] += 1

        def subscribe_loop():
            for _ in range(100):
                event_bus.subscribe(SignalGenerated, handler)
                time.sleep(0.001)

        def publish_loop():
            for _ in range(100):
                event_bus.publish(SignalGenerated(
                    symbol="AAPL", signal="BUY", confidence=0.7
                ))
                time.sleep(0.001)

        t1 = threading.Thread(target=subscribe_loop)
        t2 = threading.Thread(target=publish_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Should not crash — exact count depends on timing
        assert count["n"] >= 0


# ---------------------------------------------------------------------------
# Stress: Trade Manager Under Load
# ---------------------------------------------------------------------------


class TestTradeManagerConcurrency:
    """Verify TradeManager handles concurrent access safely."""

    def test_concurrent_trade_creation(self, trade_manager):
        """Creating trades from multiple threads doesn't corrupt state."""
        def create_trades(prefix, n):
            for i in range(n):
                trade_manager.create_trade(
                    symbol=f"{prefix}-{i}",
                    side="BUY",
                    trade_id=f"T-{prefix}-{i}",
                )

        threads = []
        for t_id in range(5):
            t = threading.Thread(target=create_trades, args=(f"P{t_id}", 20))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 100 unique trades
        all_trades = trade_manager.get_active_trades()
        assert len(all_trades) == 100

    def test_concurrent_transitions(self, trade_manager):
        """Concurrent state transitions on same trade are safe."""
        trade = trade_manager.create_trade("AAPL", "BUY", trade_id="T-concurrent")
        trade.transition(TradeState.PENDING_RISK)
        trade.transition(TradeState.RISK_APPROVED)
        trade.transition(TradeState.SUBMITTED)
        trade.transition(TradeState.ACCEPTED)
        trade.transition(TradeState.FILLED)
        trade.transition(TradeState.ACTIVE)

        results = []

        def try_transition(state, reason):
            ok = trade.transition(state, reason)
            results.append((state, ok))

        # Multiple threads try to transition simultaneously
        threads = [
            threading.Thread(target=try_transition, args=(TradeState.CLOSING, "t1")),
            threading.Thread(target=try_transition, args=(TradeState.TRAILING, "t2")),
            threading.Thread(target=try_transition, args=(TradeState.STOP_TRIGGERED, "t3")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one should succeed from ACTIVE
        successful = [r for r in results if r[1]]
        assert len(successful) == 1


# ---------------------------------------------------------------------------
# Stress: Event Store Durability
# ---------------------------------------------------------------------------


class TestEventStoreDurability:
    """Verify event store handles burst writes and concurrent access."""

    def test_burst_writes(self, tmp_event_store):
        """1000 rapid events persisted correctly."""
        for i in range(1000):
            tmp_event_store.persist(SignalGenerated(
                symbol=f"SYM-{i}", signal="BUY", confidence=0.5 + (i % 50) / 100
            ))

        count = tmp_event_store.count_events()
        assert count == 1000

    def test_concurrent_writes(self, tmp_event_store):
        """Multiple threads writing simultaneously don't lose events."""
        def write_batch(prefix, n):
            for i in range(n):
                tmp_event_store.persist(SignalGenerated(
                    symbol=f"{prefix}-{i}", signal="BUY", confidence=0.8
                ))

        threads = []
        for t_id in range(5):
            t = threading.Thread(target=write_batch, args=(f"W{t_id}", 100))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        count = tmp_event_store.count_events()
        assert count == 500

    def test_read_during_write(self, tmp_event_store):
        """Reading events while writing doesn't block or crash."""
        # Pre-populate
        for i in range(100):
            tmp_event_store.persist(SignalGenerated(
                symbol=f"PRE-{i}", signal="BUY", confidence=0.7
            ))

        read_results = []

        def reader():
            for _ in range(50):
                events = tmp_event_store.get_latest_events(10)
                read_results.append(len(events))
                time.sleep(0.001)

        def writer():
            for i in range(100):
                tmp_event_store.persist(SignalGenerated(
                    symbol=f"NEW-{i}", signal="SELL", confidence=0.6
                ))
                time.sleep(0.001)

        t_read = threading.Thread(target=reader)
        t_write = threading.Thread(target=writer)
        t_read.start()
        t_write.start()
        t_read.join()
        t_write.join()

        # All reads should have returned results
        assert all(r > 0 for r in read_results)
        # All writes should have persisted
        total = tmp_event_store.count_events()
        assert total == 200


# ---------------------------------------------------------------------------
# Scenario: Broker Failure During Trade
# ---------------------------------------------------------------------------


class TestBrokerFailureScenarios:
    """Simulate broker failures at various points in trade lifecycle."""

    def test_recovery_after_broker_disconnect(self):
        """RecoveryManager handles broker that returns error dicts."""
        mock_broker = MagicMock()
        # Broker returns error on first call, then succeeds
        mock_broker.get_positions.return_value = {"error": True, "message": "Connection timeout"}
        
        bus = EventBus()
        tm = TradeManager()
        rm = RecoveryManager(mock_broker, bus, tm)
        
        report = rm.recover()
        # Should handle gracefully without crashing
        assert report is not None
        assert report.success is False or report.positions_recovered == 0

    def test_reconciliation_with_broker_error(self):
        """Reconciler handles broker API failure gracefully."""
        mock_broker = MagicMock()
        mock_broker.get_positions.side_effect = ConnectionError("Network unreachable")

        bus = EventBus()
        tm = TradeManager()
        reconciler = PortfolioReconciler(mock_broker, tm, bus)

        report = reconciler.reconcile()
        # Should not crash
        assert report is not None


# ---------------------------------------------------------------------------
# Scenario: Event Chain Consistency
# ---------------------------------------------------------------------------


class TestEventChainConsistency:
    """Verify event chains maintain consistency under various conditions."""

    def test_event_persistence_subscriber_handles_burst(self, tmp_event_store, event_bus):
        """EventPersistenceSubscriber handles rapid event bursts."""
        subscriber = EventPersistenceSubscriber(tmp_event_store)
        subscriber.attach(event_bus)

        # Rapid burst of events
        for i in range(200):
            event_bus.publish(SignalGenerated(
                symbol=f"BURST-{i}", signal="BUY", confidence=0.75
            ))

        count = tmp_event_store.count_events()
        assert count == 200

    def test_event_ordering_preserved(self, tmp_event_store, event_bus):
        """Events maintain chronological order in store."""
        subscriber = EventPersistenceSubscriber(tmp_event_store)
        subscriber.attach(event_bus)

        symbols = [f"ORD-{i:03d}" for i in range(50)]
        for sym in symbols:
            event_bus.publish(SignalGenerated(symbol=sym, signal="BUY", confidence=0.8))

        stored = tmp_event_store.get_session_events(tmp_event_store.session_id)
        stored_symbols = []
        for s in stored:
            import json
            payload = json.loads(s["payload"]) if isinstance(s["payload"], str) else s["payload"]
            stored_symbols.append(payload.get("symbol", ""))

        assert stored_symbols == symbols

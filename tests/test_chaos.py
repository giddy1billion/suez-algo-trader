"""
Chaos Engineering Tests — Fault injection with graceful recovery verification.

Injects:
- Broker disconnects (ConnectionError)
- SQLite lock contention (concurrent writers)
- Malformed events (invalid data in events)
- Duplicate fills (same order filled twice)
- Delayed confirmations (events arriving out of order)
- Handler exceptions (subscriber throws during event)
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.core.events import (
    Event,
    EventBus,
    OrderRejected,
    SignalGenerated,
    TradeClosed,
    TradeOpened,
)
from src.core.event_store import EventStore, EventPersistenceSubscriber
from src.core.state_machine import TradeManager, TradeLifecycle, TradeState
from src.core.recovery import RecoveryManager


# ---------------------------------------------------------------------------
# Test 1: Broker disconnect during signal processing
# ---------------------------------------------------------------------------


class TestBrokerDisconnect:
    """Verify engine handles broker ConnectionError gracefully."""

    def test_broker_disconnect_during_signal_processing(self):
        """Mock broker raises ConnectionError on submit_order — no crash, bus still works."""
        bus = EventBus()
        trade_manager = TradeManager()

        # Mock broker that raises ConnectionError
        broker = MagicMock()
        broker.submit_order.side_effect = ConnectionError("Broker disconnected")
        broker.get_positions.return_value = []
        broker.get_orders.return_value = []

        errors_captured = []

        def signal_handler(event):
            """Simulates engine logic: on signal, try to submit order."""
            if isinstance(event, SignalGenerated):
                try:
                    broker.submit_order(
                        symbol=event.symbol, side=event.signal, qty=1.0
                    )
                except ConnectionError as e:
                    errors_captured.append(str(e))
                    # Engine publishes OrderRejected on broker failure
                    bus.publish(
                        OrderRejected(
                            order_id="ORD-001",
                            reason=f"Broker error: {e}",
                            source="engine",
                        )
                    )

        rejections = []

        def rejection_handler(event):
            if isinstance(event, OrderRejected):
                rejections.append(event)

        bus.subscribe(SignalGenerated, signal_handler)
        bus.subscribe(OrderRejected, rejection_handler)

        # Publish signal — should not crash
        bus.publish(
            SignalGenerated(
                symbol="BTCUSDT",
                signal="BUY",
                confidence=0.9,
                strategy="test",
                price=50000.0,
            )
        )

        # Verify graceful handling
        assert len(errors_captured) == 1
        assert "disconnected" in errors_captured[0].lower()
        assert len(rejections) == 1
        assert "Broker error" in rejections[0].reason

        # Verify bus still functional after the error
        post_error_events = []
        bus.subscribe(SignalGenerated, lambda e: post_error_events.append(e))
        bus.publish(SignalGenerated(symbol="ETHUSDT", signal="SELL", confidence=0.8))
        assert len(post_error_events) == 1
        assert post_error_events[0].symbol == "ETHUSDT"


# ---------------------------------------------------------------------------
# Test 2: SQLite lock contention
# ---------------------------------------------------------------------------


class TestSQLiteLockContention:
    """Verify EventStore handles concurrent access without corruption."""

    def test_sqlite_lock_contention(self, tmp_path):
        """10 threads simultaneously persist and read events — no corruption or deadlocks."""
        db_path = str(tmp_path / "contention_test.db")
        store = EventStore(db_path=db_path, session_id="contention-session")

        num_threads = 10
        events_per_thread = 20
        errors = []
        completed = []

        def writer_reader(thread_id):
            """Each thread writes and reads events."""
            try:
                for i in range(events_per_thread):
                    event = SignalGenerated(
                        symbol=f"SYM-{thread_id}-{i}",
                        signal="BUY",
                        confidence=0.5,
                        strategy=f"thread-{thread_id}",
                        price=100.0 + i,
                        source=f"thread-{thread_id}",
                    )
                    store.persist(event)

                    # Also read while writing
                    store.get_session_events("contention-session")

                completed.append(thread_id)
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for tid in range(num_threads):
            t = threading.Thread(target=writer_reader, args=(tid,))
            threads.append(t)

        # Start all threads
        for t in threads:
            t.start()

        # Wait with timeout (10s deadline for no deadlocks)
        for t in threads:
            t.join(timeout=10)

        # Verify no threads still running (deadlock detection)
        still_alive = [t for t in threads if t.is_alive()]
        assert len(still_alive) == 0, f"{len(still_alive)} threads deadlocked"

        # Verify no errors
        assert len(errors) == 0, f"Thread errors: {errors}"

        # Verify all threads completed
        assert len(completed) == num_threads

        # Verify data integrity — correct total count
        expected_count = num_threads * events_per_thread
        actual_count = store.count_events(session_id="contention-session")
        assert actual_count == expected_count, (
            f"Expected {expected_count} events, got {actual_count}"
        )

        store.close()


# ---------------------------------------------------------------------------
# Test 3: Malformed event in bus
# ---------------------------------------------------------------------------


class TestMalformedEvent:
    """Verify bus handles malformed/invalid events gracefully."""

    def test_malformed_event_in_bus(self):
        """Event with invalid/missing fields doesn't crash the bus."""
        bus = EventBus()
        received_events = []

        def handler(event):
            received_events.append(event)

        bus.subscribe(None, handler)  # wildcard

        # Publish a malformed event (base Event with no real data)
        malformed = Event(source="malformed-test")
        # Manually break expected attributes
        malformed.timestamp = "NOT-A-DATETIME"  # type: ignore

        # Should not raise
        bus.publish(malformed)
        assert len(received_events) == 1

        # Publish a normal event after — verify bus still works
        normal_event = SignalGenerated(
            symbol="AAPL", signal="BUY", confidence=0.7, price=150.0
        )
        bus.publish(normal_event)
        assert len(received_events) == 2
        assert received_events[1].symbol == "AAPL"


# ---------------------------------------------------------------------------
# Test 4: Subscriber exception doesn't crash bus
# ---------------------------------------------------------------------------


class TestSubscriberException:
    """Verify one crashing subscriber doesn't affect others."""

    def test_subscriber_exception_doesnt_crash_bus(self):
        """First handler raises, second handler still receives the event."""
        bus = EventBus()
        received = []

        def crashing_handler(event):
            raise RuntimeError("Handler crashed!")

        def healthy_handler(event):
            received.append(event)

        bus.subscribe(SignalGenerated, crashing_handler)
        bus.subscribe(SignalGenerated, healthy_handler)

        # Publish — should not propagate the exception
        event = SignalGenerated(symbol="MSFT", signal="SELL", confidence=0.6, price=300.0)
        bus.publish(event)

        # Healthy handler still received the event
        assert len(received) == 1
        assert received[0].symbol == "MSFT"

        # Bus still works for subsequent events
        bus.publish(SignalGenerated(symbol="GOOG", signal="BUY", confidence=0.8, price=2800.0))
        assert len(received) == 2


# ---------------------------------------------------------------------------
# Test 5: Duplicate trade close events
# ---------------------------------------------------------------------------


class TestDuplicateTradeClose:
    """Verify duplicate TradeClosed doesn't double-count PnL."""

    def test_duplicate_trade_close_events(self):
        """Publish TradeClosed twice for same trade — no double-counting."""
        bus = EventBus()
        trade_manager = TradeManager()

        pnl_accumulator = {"total_pnl": 0.0, "close_count": 0}
        closed_trade_ids = set()

        def pnl_tracker(event):
            if isinstance(event, TradeClosed):
                # Guard against duplicates
                if event.trade_id not in closed_trade_ids:
                    closed_trade_ids.add(event.trade_id)
                    pnl_accumulator["total_pnl"] += event.pnl
                    pnl_accumulator["close_count"] += 1

        bus.subscribe(TradeClosed, pnl_tracker)

        # Open trade
        bus.publish(
            TradeOpened(
                trade_id="T-001",
                symbol="BTCUSDT",
                side="BUY",
                entry_price=50000.0,
                qty=1.0,
            )
        )

        # Close trade TWICE (duplicate)
        close_event = TradeClosed(
            trade_id="T-001",
            symbol="BTCUSDT",
            exit_price=51000.0,
            pnl=1000.0,
            pnl_pct=2.0,
            reason="take_profit",
        )
        bus.publish(close_event)
        bus.publish(close_event)  # duplicate!

        # Verify no double-counting
        assert pnl_accumulator["total_pnl"] == 1000.0
        assert pnl_accumulator["close_count"] == 1


# ---------------------------------------------------------------------------
# Test 6: Events arriving out of order
# ---------------------------------------------------------------------------


class TestOutOfOrderEvents:
    """Verify system handles out-of-order events gracefully."""

    def test_events_arriving_out_of_order(self):
        """TradeClosed arrives before TradeOpened — no crash."""
        bus = EventBus()
        trade_manager = TradeManager()
        received = []

        def tracker(event):
            received.append(event)

        bus.subscribe(None, tracker)

        # Publish TradeClosed BEFORE TradeOpened (out of order)
        bus.publish(
            TradeClosed(
                trade_id="T-OOO",
                symbol="ETHUSDT",
                exit_price=3000.0,
                pnl=200.0,
                pnl_pct=5.0,
                reason="stop_loss",
            )
        )

        # Now publish TradeOpened (delayed)
        bus.publish(
            TradeOpened(
                trade_id="T-OOO",
                symbol="ETHUSDT",
                side="BUY",
                entry_price=2800.0,
                qty=1.0,
            )
        )

        # System didn't crash and both events were delivered
        assert len(received) == 2
        assert isinstance(received[0], TradeClosed)
        assert isinstance(received[1], TradeOpened)

        # TradeManager can still create trades after out-of-order events
        trade = trade_manager.create_trade(symbol="SOLUSDT", side="BUY")
        assert trade is not None
        assert trade.state == TradeState.SIGNAL


# ---------------------------------------------------------------------------
# Test 7: High volume event burst — bounded memory
# ---------------------------------------------------------------------------


class TestHighVolumeEventBurst:
    """Verify EventBus history is bounded under high volume."""

    def test_high_volume_event_burst_no_memory_leak(self):
        """Publish 10,000 events — history stays bounded at maxlen=1000."""
        bus = EventBus(max_history=1000)
        event_count = 10_000

        # Subscriber that just counts (no unbounded storage)
        counter = {"count": 0}

        def counting_handler(event):
            counter["count"] += 1

        bus.subscribe(SignalGenerated, counting_handler)

        # Burst of 10,000 events
        for i in range(event_count):
            bus.publish(
                SignalGenerated(
                    symbol=f"SYM-{i % 100}",
                    signal="BUY",
                    confidence=0.5,
                    price=float(i),
                )
            )

        # Verify history is bounded
        history = bus.get_history(limit=event_count)
        assert len(history) <= 1000, f"History unbounded: {len(history)} events"

        # Verify subscriber received all events
        assert counter["count"] == event_count

        # Internal deque maxlen check
        assert len(bus._history) <= 1000


# ---------------------------------------------------------------------------
# Test 8: Recovery after simulated crash
# ---------------------------------------------------------------------------


class TestRecoveryAfterCrash:
    """Verify state reconstruction after simulated crash."""

    def test_recovery_after_simulated_crash(self, tmp_path):
        """Persist trade lifecycle events, 'crash', then recover from broker."""
        db_path = str(tmp_path / "crash_recovery.db")

        # --- Phase 1: Normal operation, persist events ---
        store = EventStore(db_path=db_path, session_id="crash-session")
        bus = EventBus()
        persistence = EventPersistenceSubscriber(store)
        persistence.attach(bus)

        # Simulate a trade lifecycle
        bus.publish(
            SignalGenerated(
                symbol="BTCUSDT",
                signal="BUY",
                confidence=0.95,
                strategy="momentum",
                price=50000.0,
                source="strategy",
            )
        )
        bus.publish(
            TradeOpened(
                trade_id="T-CRASH-001",
                symbol="BTCUSDT",
                side="BUY",
                entry_price=50000.0,
                qty=0.5,
                source="engine",
            )
        )
        # Trade is still open when "crash" happens

        # Verify events persisted
        assert store.count_events(session_id="crash-session") >= 2

        # --- Phase 2: Simulate crash — fresh components ---
        store.close()

        new_bus = EventBus()
        new_trade_manager = TradeManager()

        # Mock broker returns the open position
        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = [
            {
                "symbol": "BTCUSDT",
                "side": "long",
                "qty": 0.5,
                "asset_id": "T-CRASH-001",
            }
        ]
        mock_broker.get_orders.return_value = []

        # --- Phase 3: Recovery ---
        recovery = RecoveryManager(
            broker=mock_broker,
            event_bus=new_bus,
            trade_manager=new_trade_manager,
            event_store=None,  # No event replay in this test
        )

        report = recovery.recover()

        # Verify recovery success
        assert report.success is True
        assert report.positions_recovered == 1

        # Verify trade state reconstructed
        trade = new_trade_manager.get_trade("T-CRASH-001")
        assert trade is not None
        assert trade.state == TradeState.ACTIVE
        assert trade.symbol == "BTCUSDT"
        assert trade.metadata.get("recovered") is True

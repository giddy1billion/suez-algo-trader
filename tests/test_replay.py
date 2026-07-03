"""
Tests for the Deterministic Replay Engine.
"""

import pytest
from datetime import datetime, timezone

from src.core.events import (
    EventBus,
    SignalGenerated,
    TradeOpened,
    TradeClosed,
    OrderRejected,
)
from src.core.event_store import EventStore, EventPersistenceSubscriber
from src.core.replay import ReplayEngine, ReplayReport
from src.core.state_machine import TradeManager


@pytest.fixture
def tmp_event_store(tmp_path):
    """Create a temporary event store."""
    db_path = str(tmp_path / "test_events.db")
    return EventStore(db_path=db_path, session_id="test-session-001")


@pytest.fixture
def populated_store(tmp_event_store):
    """Event store pre-populated with a full trade lifecycle."""
    store = tmp_event_store

    # Simulate a full trade session
    events = [
        SignalGenerated(
            symbol="AAPL", signal="BUY", confidence=0.85,
            strategy="momentum", price=150.0, source="engine"
        ),
        TradeOpened(
            trade_id="T-abc12345", symbol="AAPL", side="BUY",
            entry_price=150.0, qty=10, stop_loss=145.0, take_profit=160.0,
            source="engine"
        ),
        SignalGenerated(
            symbol="MSFT", signal="BUY", confidence=0.72,
            strategy="ml", price=380.0, source="engine"
        ),
        OrderRejected(
            order_id="ORD-xyz", reason="Insufficient buying power",
            source="broker"
        ),
        TradeClosed(
            trade_id="T-abc12345", symbol="AAPL", exit_price=155.0,
            pnl=50.0, pnl_pct=3.33, reason="take_profit",
            source="engine"
        ),
    ]

    for event in events:
        store.persist(event)

    return store


class TestReplayEngine:
    """Tests for ReplayEngine."""

    def test_replay_empty_session(self, tmp_event_store):
        """Replaying a non-existent session returns error report."""
        engine = ReplayEngine(tmp_event_store)
        report = engine.replay("nonexistent-session")

        assert report.events_replayed == 0
        assert len(report.errors) > 0

    def test_replay_full_session(self, populated_store):
        """Replay correctly processes all events."""
        engine = ReplayEngine(populated_store)
        report = engine.replay("test-session-001")

        assert report.events_replayed == 5
        assert report.signals_count == 2
        assert report.trades_opened == 1
        assert report.trades_closed == 1
        assert report.orders_rejected == 1
        assert report.final_pnl == 50.0

    def test_replay_reconstructs_trade_manager(self, populated_store):
        """Replay rebuilds TradeManager state."""
        engine = ReplayEngine(populated_store)
        tm = TradeManager()
        report = engine.replay("test-session-001", trade_manager=tm)

        trade = tm.get_trade("T-abc12345")
        assert trade is not None
        assert trade.is_terminal  # Should be CLOSED

    def test_replay_with_custom_event_bus(self, populated_store):
        """Replay publishes into provided EventBus."""
        engine = ReplayEngine(populated_store)
        bus = EventBus(max_history=100)
        received = []
        bus.subscribe(None, lambda e: received.append(e))

        engine.replay("test-session-001", event_bus=bus)
        assert len(received) == 5

    def test_replay_stop_after(self, populated_store):
        """stop_after limits event replay count."""
        engine = ReplayEngine(populated_store)
        report = engine.replay("test-session-001", stop_after=2)

        assert report.events_replayed == 2

    def test_replay_with_filter(self, populated_store):
        """event_filter selectively replays events."""
        engine = ReplayEngine(populated_store)
        # Only replay signal events
        report = engine.replay(
            "test-session-001",
            event_filter=lambda e: isinstance(e, SignalGenerated),
        )

        assert report.signals_count == 2
        assert report.trades_opened == 0  # Filtered out

    def test_replay_with_observer(self, populated_store):
        """Custom observers receive replayed events."""
        engine = ReplayEngine(populated_store)
        observed = []
        engine.add_observer(lambda e: observed.append(type(e).__name__))

        engine.replay("test-session-001")
        assert "SignalGenerated" in observed
        assert "TradeOpened" in observed
        assert "TradeClosed" in observed

    def test_replay_report_has_timeline(self, populated_store):
        """Report includes event timeline for debugging."""
        engine = ReplayEngine(populated_store)
        report = engine.replay("test-session-001")

        assert len(report.event_timeline) == 5
        assert report.event_timeline[0]["type"] == "SignalGenerated"
        assert report.event_timeline[-1]["type"] == "TradeClosed"

    def test_list_sessions(self, populated_store):
        """list_sessions returns available sessions."""
        engine = ReplayEngine(populated_store)
        sessions = engine.list_sessions()

        assert len(sessions) >= 1
        assert sessions[0]["session_id"] == "test-session-001"
        assert sessions[0]["event_count"] == 5

    def test_get_session_summary(self, populated_store):
        """get_session_summary returns type counts without full replay."""
        engine = ReplayEngine(populated_store)
        summary = engine.get_session_summary("test-session-001")

        assert summary["found"] is True
        assert summary["total_events"] == 5
        assert "SignalGenerated" in summary["event_types"]
        assert summary["event_types"]["SignalGenerated"] == 2

    def test_compare_sessions_same(self, populated_store):
        """Comparing a session with itself shows all matches."""
        engine = ReplayEngine(populated_store)
        comparison = engine.compare_sessions("test-session-001", "test-session-001")

        assert comparison["events_match"] is True
        assert comparison["pnl_match"] is True

    def test_replay_duration_tracked(self, populated_store):
        """Replay measures its own execution time."""
        engine = ReplayEngine(populated_store)
        report = engine.replay("test-session-001")

        assert report.duration_seconds >= 0.0

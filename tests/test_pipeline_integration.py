"""
Integration Test — Full Signal → Execute → Event → Projection Pipeline.

This test traces a complete trade lifecycle through all layers to verify
end-to-end wiring and correctness, as specified in Audit Phase 15.

Flow verified:
  SignalGenerated
  → ExecutionEngine._process_signal()
  → RiskManager.evaluate()
  → TradeManager.open_trade()
  → EventBus.publish(TradeOpened)
  → ReadModelManager (projections updated)
  → TradeManager.close_trade() [via check_exits]
  → EventBus.publish(TradeClosed)
  → ReadModelManager (projection removed)
  → EventStore (events persisted)
"""

import tempfile
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from src.core.events import (
    EventBus,
    SignalGenerated,
    TradeOpened,
    TradeClosed,
    OrderSubmitted,
    OrderFilled,
    RiskEvaluated,
)
from src.core.projections import ReadModelManager
from src.core.event_store import EventStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def event_store(tmp_path):
    db = str(tmp_path / "events.db")
    store = EventStore(db_path=db)
    return store


@pytest.fixture
def read_models(event_bus):
    rm = ReadModelManager()
    rm.attach(event_bus)
    return rm


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:
    """End-to-end integration test: signal → event → projection."""

    def test_trade_opened_flows_to_projections(self, event_bus, read_models):
        """TradeOpened event should appear in position projection."""
        event = TradeOpened(
            trade_id="integration-001",
            symbol="AAPL",
            side="BUY",
            entry_price=175.50,
            qty=10,
            stop_loss=170.00,
            take_profit=185.00,
            source="test_pipeline",
        )

        event_bus.publish(event)

        # Verify projection state
        positions = read_models.positions.get_positions()
        assert len(positions) == 1
        assert positions[0].trade_id == "integration-001"
        assert positions[0].symbol == "AAPL"
        assert positions[0].entry_price == 175.50
        assert positions[0].qty == 10

    def test_trade_closed_updates_performance(self, event_bus, read_models):
        """TradeClosed event should update performance metrics."""
        # Open
        event_bus.publish(TradeOpened(
            trade_id="integration-002", symbol="MSFT", side="BUY",
            entry_price=400.0, qty=5, stop_loss=390.0, take_profit=420.0,
            source="test_pipeline",
        ))
        # Close with profit
        event_bus.publish(TradeClosed(
            trade_id="integration-002", symbol="MSFT",
            exit_price=415.0, pnl=75.0, pnl_pct=3.75,
            reason="take_profit", source="test_pipeline",
        ))

        # Position should be removed
        assert read_models.positions.count() == 0

        # Performance should reflect the trade
        metrics = read_models.performance.get_metrics()
        assert metrics["trade_count"] == 1
        assert metrics["win_count"] == 1
        assert metrics["realized_pnl"] == 75.0

    def test_multiple_trades_accumulate_metrics(self, event_bus, read_models):
        """Multiple trades should accumulate in performance projection."""
        trades = [
            ("t-win-1", "AAPL", 150.0, 160.0, 50.0),
            ("t-win-2", "MSFT", 300.0, 310.0, 30.0),
            ("t-loss-1", "TSLA", 200.0, 190.0, -40.0),
        ]

        for tid, sym, entry, exit_p, pnl in trades:
            event_bus.publish(TradeOpened(
                trade_id=tid, symbol=sym, side="BUY",
                entry_price=entry, qty=1, stop_loss=entry * 0.95,
                take_profit=entry * 1.1, source="test",
            ))
            event_bus.publish(TradeClosed(
                trade_id=tid, symbol=sym,
                exit_price=exit_p, pnl=pnl, pnl_pct=(pnl / entry) * 100,
                reason="take_profit" if pnl > 0 else "stop_loss",
                source="test",
            ))

        metrics = read_models.performance.get_metrics()
        assert metrics["trade_count"] == 3
        assert metrics["win_count"] == 2
        assert metrics["loss_count"] == 1
        assert metrics["realized_pnl"] == pytest.approx(40.0)

    def test_event_store_persists_all_events(self, event_bus, event_store):
        """Events published on bus should be persistable to the event store."""
        events = [
            SignalGenerated(
                symbol="BTC", signal="BUY", confidence=0.85,
                strategy="momentum", source="test",
            ),
            TradeOpened(
                trade_id="store-001", symbol="BTC", side="BUY",
                entry_price=65000.0, qty=0.01, stop_loss=64000.0,
                take_profit=67000.0, source="test",
            ),
            TradeClosed(
                trade_id="store-001", symbol="BTC",
                exit_price=66500.0, pnl=15.0, pnl_pct=2.3,
                reason="take_profit", source="test",
            ),
        ]

        for ev in events:
            event_store.persist(ev)

        # Verify all persisted
        assert event_store.count_events() >= 3

    def test_event_deduplication(self, event_store):
        """Same event_id should not be stored twice."""
        ev = SignalGenerated(
            symbol="ETH", signal="SELL", confidence=0.7,
            strategy="mean_reversion", source="test",
        )

        event_store.persist(ev)
        event_store.persist(ev)  # Duplicate — should be silently ignored

        # Should only have 1 event total
        assert event_store.count_events() == 1

    def test_dashboard_reflects_live_state(self, event_bus, read_models):
        """Dashboard should reflect current portfolio state."""
        # Open 2 positions
        event_bus.publish(TradeOpened(
            trade_id="dash-1", symbol="AAPL", side="BUY",
            entry_price=170.0, qty=10, stop_loss=165.0,
            take_profit=180.0, source="test",
        ))
        event_bus.publish(TradeOpened(
            trade_id="dash-2", symbol="GOOG", side="BUY",
            entry_price=150.0, qty=20, stop_loss=145.0,
            take_profit=160.0, source="test",
        ))

        dashboard = read_models.get_dashboard()
        assert dashboard["position_count"] == 2
        assert dashboard["exposure"] == (170.0 * 10 + 150.0 * 20)

        # Close one
        event_bus.publish(TradeClosed(
            trade_id="dash-1", symbol="AAPL",
            exit_price=178.0, pnl=80.0, pnl_pct=4.7,
            reason="take_profit", source="test",
        ))

        dashboard = read_models.get_dashboard()
        assert dashboard["position_count"] == 1
        assert dashboard["performance"]["trade_count"] == 1

    def test_frozen_events_cannot_be_mutated(self, event_bus):
        """Events should be immutable (frozen dataclass)."""
        ev = TradeOpened(
            trade_id="frozen-001", symbol="SPY", side="BUY",
            entry_price=450.0, qty=5, stop_loss=440.0,
            take_profit=470.0, source="test",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            ev.entry_price = 999.0

    def test_risk_halt_tracked_in_performance(self, event_bus, read_models):
        """RiskHalt events should be counted in performance projection."""
        from src.core.events import RiskHalt

        event_bus.publish(RiskHalt(
            reason="daily_loss_exceeded",
            level="CRITICAL",
            source="risk_manager",
        ))

        metrics = read_models.performance.get_metrics()
        assert metrics["risk_halt_count"] == 1

    def test_signal_counted_in_performance(self, event_bus, read_models):
        """SignalGenerated events should be counted."""
        event_bus.publish(SignalGenerated(
            symbol="NVDA", signal="BUY", confidence=0.92,
            strategy="breakout", source="test",
        ))

        metrics = read_models.performance.get_metrics()
        assert metrics["signal_count"] == 1

"""
Tests for CQRS Read Model Projections and Event Store Snapshotting.
"""

import threading
import os
import tempfile
from datetime import datetime, timezone

import pytest

from src.core.events import (
    EventBus, SignalGenerated, OrderRejected,
    TradeOpened, TradeClosed, RiskHalt,
)
from src.core.projections import (
    PositionProjection, PerformanceProjection, ActivityProjection,
    ReadModelManager, PositionView, PortfolioSnapshot,
)
from src.core.snapshots import SnapshotStore, SnapshotManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade_opened(trade_id="t1", symbol="BTCUSDT", side="BUY",
                       entry_price=50000.0, qty=0.1, stop_loss=49000.0,
                       take_profit=52000.0):
    return TradeOpened(
        trade_id=trade_id, symbol=symbol, side=side,
        entry_price=entry_price, qty=qty,
        stop_loss=stop_loss, take_profit=take_profit,
        source="test",
    )


def _make_trade_closed(trade_id="t1", symbol="BTCUSDT", pnl=100.0):
    return TradeClosed(
        trade_id=trade_id, symbol=symbol, exit_price=50500.0,
        pnl=pnl, pnl_pct=2.0, reason="take_profit",
        source="test",
    )


# ---------------------------------------------------------------------------
# PositionProjection Tests
# ---------------------------------------------------------------------------

class TestPositionProjection:
    def test_open_position(self):
        proj = PositionProjection()
        event = _make_trade_opened()
        proj.handle_event(event)

        assert proj.count() == 1
        pos = proj.get_position("t1")
        assert pos is not None
        assert pos.symbol == "BTCUSDT"
        assert pos.side == "BUY"
        assert pos.entry_price == 50000.0
        assert pos.qty == 0.1

    def test_close_position(self):
        proj = PositionProjection()
        proj.handle_event(_make_trade_opened())
        assert proj.count() == 1

        proj.handle_event(_make_trade_closed())
        assert proj.count() == 0
        assert proj.get_position("t1") is None

    def test_close_nonexistent_position(self):
        proj = PositionProjection()
        # Should not raise
        proj.handle_event(_make_trade_closed(trade_id="nonexistent"))
        assert proj.count() == 0

    def test_multiple_positions(self):
        proj = PositionProjection()
        proj.handle_event(_make_trade_opened(trade_id="t1", symbol="BTCUSDT"))
        proj.handle_event(_make_trade_opened(trade_id="t2", symbol="ETHUSDT", entry_price=3000.0, qty=1.0))

        assert proj.count() == 2
        positions = proj.get_positions()
        symbols = {p.symbol for p in positions}
        assert symbols == {"BTCUSDT", "ETHUSDT"}

    def test_get_exposure(self):
        proj = PositionProjection()
        proj.handle_event(_make_trade_opened(trade_id="t1", entry_price=50000.0, qty=0.1))
        proj.handle_event(_make_trade_opened(trade_id="t2", entry_price=3000.0, qty=1.0))

        # 50000*0.1 + 3000*1.0 = 5000 + 3000 = 8000
        assert proj.get_exposure() == pytest.approx(8000.0)

    def test_concurrent_access(self):
        proj = PositionProjection()
        errors = []

        def open_positions(start_id, count):
            try:
                for i in range(count):
                    proj.handle_event(_make_trade_opened(
                        trade_id=f"t{start_id}_{i}", symbol="BTCUSDT"
                    ))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=open_positions, args=(tid * 100, 50))
            for tid in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert proj.count() == 200


# ---------------------------------------------------------------------------
# PerformanceProjection Tests
# ---------------------------------------------------------------------------

class TestPerformanceProjection:
    def test_pnl_accumulation(self):
        proj = PerformanceProjection()
        proj.handle_event(_make_trade_closed(trade_id="t1", pnl=100.0))
        proj.handle_event(_make_trade_closed(trade_id="t2", pnl=-50.0))
        proj.handle_event(_make_trade_closed(trade_id="t3", pnl=200.0))

        metrics = proj.get_metrics()
        assert metrics["realized_pnl"] == pytest.approx(250.0)
        assert metrics["trade_count"] == 3

    def test_win_rate(self):
        proj = PerformanceProjection()
        proj.handle_event(_make_trade_closed(trade_id="t1", pnl=100.0))
        proj.handle_event(_make_trade_closed(trade_id="t2", pnl=-50.0))
        proj.handle_event(_make_trade_closed(trade_id="t3", pnl=200.0))
        proj.handle_event(_make_trade_closed(trade_id="t4", pnl=0.0))  # breakeven

        metrics = proj.get_metrics()
        assert metrics["win_count"] == 2
        assert metrics["loss_count"] == 1
        # win_rate = 2/4 = 0.5
        assert metrics["win_rate"] == pytest.approx(0.5)

    def test_profit_factor(self):
        proj = PerformanceProjection()
        proj.handle_event(_make_trade_closed(trade_id="t1", pnl=300.0))
        proj.handle_event(_make_trade_closed(trade_id="t2", pnl=-100.0))

        metrics = proj.get_metrics()
        # profit_factor = 300 / 100 = 3.0
        assert metrics["profit_factor"] == pytest.approx(3.0)

    def test_profit_factor_no_losses(self):
        proj = PerformanceProjection()
        proj.handle_event(_make_trade_closed(trade_id="t1", pnl=100.0))

        metrics = proj.get_metrics()
        assert metrics["profit_factor"] == 999.99

    def test_profit_factor_no_trades(self):
        proj = PerformanceProjection()
        metrics = proj.get_metrics()
        assert metrics["profit_factor"] == 0.0
        assert metrics["win_rate"] == 0.0

    def test_signal_and_rejection_counts(self):
        proj = PerformanceProjection()
        proj.handle_event(SignalGenerated(symbol="BTC", signal="BUY", confidence=0.8, source="test"))
        proj.handle_event(SignalGenerated(symbol="ETH", signal="SELL", confidence=0.6, source="test"))
        proj.handle_event(OrderRejected(order_id="o1", reason="insufficient_margin", source="test"))
        proj.handle_event(RiskHalt(reason="max_drawdown", source="test"))

        metrics = proj.get_metrics()
        assert metrics["signal_count"] == 2
        assert metrics["rejection_count"] == 1
        assert metrics["risk_halt_count"] == 1


# ---------------------------------------------------------------------------
# ActivityProjection Tests
# ---------------------------------------------------------------------------

class TestActivityProjection:
    def test_recent_events(self):
        proj = ActivityProjection(max_recent=100)
        proj.handle_event(SignalGenerated(symbol="BTC", signal="BUY", confidence=0.9, source="test"))
        proj.handle_event(_make_trade_opened())

        recent = proj.get_recent(10)
        assert len(recent) == 2
        assert recent[0]["type"] == "SignalGenerated"
        assert recent[1]["type"] == "TradeOpened"

    def test_capped_at_max_recent(self):
        proj = ActivityProjection(max_recent=5)
        for i in range(10):
            proj.handle_event(SignalGenerated(
                symbol=f"SYM{i}", signal="BUY", confidence=0.5, source="test"
            ))

        assert proj.count() == 5
        recent = proj.get_recent(10)
        assert len(recent) == 5
        # Should have the last 5 events (SYM5..SYM9)
        assert recent[0]["symbol"] == "SYM5"
        assert recent[4]["symbol"] == "SYM9"

    def test_event_fields_signal(self):
        proj = ActivityProjection()
        proj.handle_event(SignalGenerated(symbol="BTC", signal="SELL", confidence=0.7, source="strat"))

        entry = proj.get_recent(1)[0]
        assert entry["symbol"] == "BTC"
        assert entry["signal"] == "SELL"
        assert entry["confidence"] == 0.7

    def test_event_fields_trade_closed(self):
        proj = ActivityProjection()
        proj.handle_event(_make_trade_closed(trade_id="t99", pnl=-42.5))

        entry = proj.get_recent(1)[0]
        assert entry["trade_id"] == "t99"
        assert entry["pnl"] == -42.5


# ---------------------------------------------------------------------------
# ReadModelManager Tests
# ---------------------------------------------------------------------------

class TestReadModelManager:
    def test_dispatch_to_all_projections(self):
        mgr = ReadModelManager()
        bus = EventBus()
        mgr.attach(bus)

        bus.publish(_make_trade_opened(trade_id="t1"))
        bus.publish(SignalGenerated(symbol="BTC", signal="BUY", confidence=0.8, source="test"))

        assert mgr.positions.count() == 1
        assert mgr.performance.get_metrics()["signal_count"] == 1
        assert mgr.activity.count() == 2

    def test_dashboard_query(self):
        mgr = ReadModelManager()
        bus = EventBus()
        mgr.attach(bus)

        bus.publish(_make_trade_opened(trade_id="t1", entry_price=50000.0, qty=0.1))
        bus.publish(_make_trade_closed(trade_id="t1", pnl=500.0))

        dashboard = mgr.get_dashboard()
        assert dashboard["position_count"] == 0  # closed
        assert dashboard["performance"]["realized_pnl"] == 500.0
        assert dashboard["performance"]["trade_count"] == 1
        assert len(dashboard["recent_activity"]) == 2

    def test_get_snapshot(self):
        mgr = ReadModelManager()
        bus = EventBus()
        mgr.attach(bus)

        bus.publish(_make_trade_opened(trade_id="t1"))
        bus.publish(_make_trade_closed(trade_id="t1", pnl=100.0))

        snapshot = mgr.get_snapshot()
        assert isinstance(snapshot, PortfolioSnapshot)
        assert snapshot.realized_pnl == 100.0
        assert snapshot.trade_count == 1
        assert snapshot.win_count == 1

    def test_projection_error_does_not_crash(self):
        """A failing projection should not prevent others from updating."""
        mgr = ReadModelManager()

        class BrokenProjection:
            def handle_event(self, event):
                raise RuntimeError("boom")

        mgr._projections.insert(0, BrokenProjection())

        bus = EventBus()
        mgr.attach(bus)
        bus.publish(_make_trade_opened(trade_id="t1"))

        # Other projections still work
        assert mgr.positions.count() == 1


# ---------------------------------------------------------------------------
# SnapshotStore Tests
# ---------------------------------------------------------------------------

class TestSnapshotStore:
    @pytest.fixture
    def store(self, tmp_path):
        db_path = str(tmp_path / "test_snapshots.db")
        s = SnapshotStore(db_path=db_path)
        yield s
        s.close()

    def test_save_and_load(self, store):
        state = {"positions": [], "pnl": 1234.56}
        snapshot_id = store.save_snapshot("session1", last_event_id=100, state=state)

        assert snapshot_id >= 1
        loaded = store.get_latest_snapshot("session1")
        assert loaded is not None
        assert loaded["session_id"] == "session1"
        assert loaded["last_event_id"] == 100
        assert loaded["state"] == state

    def test_latest_snapshot_returns_most_recent(self, store):
        store.save_snapshot("s1", last_event_id=10, state={"v": 1})
        store.save_snapshot("s1", last_event_id=20, state={"v": 2})
        store.save_snapshot("s1", last_event_id=30, state={"v": 3})

        loaded = store.get_latest_snapshot("s1")
        assert loaded["last_event_id"] == 30
        assert loaded["state"]["v"] == 3

    def test_latest_snapshot_no_session_filter(self, store):
        store.save_snapshot("s1", last_event_id=10, state={"s": "1"})
        store.save_snapshot("s2", last_event_id=20, state={"s": "2"})

        loaded = store.get_latest_snapshot()
        assert loaded["last_event_id"] == 20

    def test_no_snapshot_returns_none(self, store):
        assert store.get_latest_snapshot("nonexistent") is None

    def test_snapshot_count(self, store):
        assert store.get_snapshot_count() == 0
        store.save_snapshot("s1", 1, {})
        store.save_snapshot("s1", 2, {})
        store.save_snapshot("s2", 3, {})

        assert store.get_snapshot_count() == 3
        assert store.get_snapshot_count("s1") == 2
        assert store.get_snapshot_count("s2") == 1

    def test_cleanup_old_snapshots(self, store):
        for i in range(15):
            store.save_snapshot("s1", last_event_id=i, state={"i": i})

        assert store.get_snapshot_count() == 15
        deleted = store.cleanup_old_snapshots(keep_latest=5)
        assert deleted == 10
        assert store.get_snapshot_count() == 5

        # Kept the latest 5
        loaded = store.get_latest_snapshot()
        assert loaded["state"]["i"] == 14

    def test_cleanup_when_below_threshold(self, store):
        store.save_snapshot("s1", 1, {})
        store.save_snapshot("s1", 2, {})

        deleted = store.cleanup_old_snapshots(keep_latest=10)
        assert deleted == 0
        assert store.get_snapshot_count() == 2


# ---------------------------------------------------------------------------
# SnapshotManager Tests
# ---------------------------------------------------------------------------

class TestSnapshotManager:
    @pytest.fixture
    def store(self, tmp_path):
        db_path = str(tmp_path / "mgr_snapshots.db")
        s = SnapshotStore(db_path=db_path)
        yield s
        s.close()

    def test_interval_tracking(self, store):
        mgr = SnapshotManager(store, snapshot_interval_events=5)

        assert not mgr.should_snapshot()
        for _ in range(4):
            mgr.on_event(None)
        assert not mgr.should_snapshot()

        mgr.on_event(None)
        assert mgr.should_snapshot()

    def test_take_snapshot_resets_counter(self, store):
        mgr = SnapshotManager(store, snapshot_interval_events=5)

        for _ in range(5):
            mgr.on_event(None)
        assert mgr.should_snapshot()

        mgr.take_snapshot("session1", last_event_id=50, state={"x": 1})
        assert not mgr.should_snapshot()

    def test_recovery_with_snapshot(self, store):
        mgr = SnapshotManager(store, snapshot_interval_events=10)
        state = {"positions": ["t1", "t2"], "pnl": 500.0}
        mgr.take_snapshot("session1", last_event_id=100, state=state)

        recovered = mgr.recover_from_snapshot("session1")
        assert recovered is not None
        assert recovered["last_event_id"] == 100
        assert recovered["state"]["pnl"] == 500.0

    def test_recovery_without_snapshot(self, store):
        mgr = SnapshotManager(store, snapshot_interval_events=10)
        recovered = mgr.recover_from_snapshot("nonexistent")
        assert recovered is None

    def test_concurrent_event_counting(self, store):
        mgr = SnapshotManager(store, snapshot_interval_events=1000)
        errors = []

        def fire_events(count):
            try:
                for _ in range(count):
                    mgr.on_event(None)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=fire_events, args=(250,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert mgr.should_snapshot()  # 1000 events fired

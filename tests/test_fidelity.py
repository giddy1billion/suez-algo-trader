"""
Fidelity Tests — Prove exact correctness invariants.

1. Replay Fidelity: Original execution → persist → replay → state == original state (EXACT)
2. E2E Recovery: Trade open → persist → crash → recover → reconcile → verify state
"""

from unittest.mock import MagicMock

import pytest

from src.core.events import (
    Event,
    EventBus,
    SignalGenerated,
    TradeClosed,
    TradeOpened,
)
from src.core.event_store import EventStore, EventPersistenceSubscriber
from src.core.state_machine import TradeManager, TradeLifecycle, TradeState
from src.core.recovery import RecoveryManager
from src.core.reconciliation import (
    PortfolioReconciler,
    MISSING_INTERNAL,
)
from src.core.replay import ReplayEngine


# ---------------------------------------------------------------------------
# Test 1: Replay produces exact same state
# ---------------------------------------------------------------------------


class TestReplayFidelity:
    """Verify replay produces EXACTLY the same state as original execution."""

    def test_replay_produces_exact_same_state(self, tmp_path):
        """Original execution → persist → replay → state matches exactly."""
        db_path = str(tmp_path / "fidelity_replay.db")

        # --- Phase 1: Original Execution ---
        store = EventStore(db_path=db_path, session_id="fidelity-001")
        bus = EventBus()
        trade_manager = TradeManager()
        persistence = EventPersistenceSubscriber(store)
        persistence.attach(bus)

        # Track state during original execution
        original_pnl = {"total": 0.0}
        original_trades = {"opened": 0, "closed": 0}

        def original_tracker(event):
            if isinstance(event, TradeOpened):
                original_trades["opened"] += 1
                trade_manager.create_trade(
                    symbol=event.symbol, side=event.side, trade_id=event.trade_id
                )
                lc = trade_manager.get_trade(event.trade_id)
                if lc:
                    lc.transition(TradeState.PENDING_RISK, "live")
                    lc.transition(TradeState.RISK_APPROVED, "live")
                    lc.transition(TradeState.SUBMITTED, "live")
                    lc.transition(TradeState.ACCEPTED, "live")
                    lc.transition(TradeState.FILLED, "live")
                    lc.transition(TradeState.ACTIVE, "live")
            elif isinstance(event, TradeClosed):
                original_trades["closed"] += 1
                original_pnl["total"] += event.pnl
                lc = trade_manager.get_trade(event.trade_id)
                if lc and not lc.is_terminal:
                    lc.transition(TradeState.CLOSING, "live")
                    lc.transition(TradeState.CLOSED, "live")

        bus.subscribe(TradeOpened, original_tracker)
        bus.subscribe(TradeClosed, original_tracker)

        # Publish realistic trade sequence
        bus.publish(SignalGenerated(
            symbol="BTCUSDT", signal="BUY", confidence=0.9,
            strategy="momentum", price=50000.0, source="strategy",
        ))
        bus.publish(TradeOpened(
            trade_id="T-F001", symbol="BTCUSDT", side="BUY",
            entry_price=50000.0, qty=1.0, source="engine",
        ))
        bus.publish(TradeClosed(
            trade_id="T-F001", symbol="BTCUSDT",
            exit_price=51000.0, pnl=1000.0, pnl_pct=2.0,
            reason="take_profit", source="engine",
        ))

        # Record final state
        original_event_count = store.count_events(session_id="fidelity-001")

        # --- Phase 2: Replay ---
        replay_engine = ReplayEngine(event_store=store)
        report = replay_engine.replay("fidelity-001")

        # --- Phase 3: Assert EXACT equality ---
        assert report.events_replayed == original_event_count
        assert report.trades_opened == original_trades["opened"]
        assert report.trades_closed == original_trades["closed"]
        assert report.final_pnl == original_pnl["total"]
        assert report.signals_count == 1

        store.close()

    def test_replay_with_multiple_trades_exact_pnl(self, tmp_path):
        """5 trades with different PnLs — replay produces exact PnL sum."""
        db_path = str(tmp_path / "multi_trade_replay.db")
        store = EventStore(db_path=db_path, session_id="multi-trade-001")
        bus = EventBus()
        persistence = EventPersistenceSubscriber(store)
        persistence.attach(bus)

        # Define 5 trades with specific PnLs
        trades = [
            {"id": "T-M001", "symbol": "BTCUSDT", "pnl": 1500.0, "pnl_pct": 3.0},
            {"id": "T-M002", "symbol": "ETHUSDT", "pnl": -200.0, "pnl_pct": -1.0},
            {"id": "T-M003", "symbol": "SOLUSDT", "pnl": 750.0, "pnl_pct": 5.0},
            {"id": "T-M004", "symbol": "ADAUSDT", "pnl": -100.0, "pnl_pct": -0.5},
            {"id": "T-M005", "symbol": "DOTUSDT", "pnl": 3000.0, "pnl_pct": 10.0},
        ]

        expected_total_pnl = sum(t["pnl"] for t in trades)  # 4950.0

        # Open all trades
        for t in trades:
            bus.publish(SignalGenerated(
                symbol=t["symbol"], signal="BUY", confidence=0.8,
                strategy="multi", price=100.0,
            ))
            bus.publish(TradeOpened(
                trade_id=t["id"], symbol=t["symbol"], side="BUY",
                entry_price=100.0, qty=1.0,
            ))

        # Close all trades
        for t in trades:
            bus.publish(TradeClosed(
                trade_id=t["id"], symbol=t["symbol"],
                exit_price=100.0 + t["pnl"], pnl=t["pnl"],
                pnl_pct=t["pnl_pct"], reason="strategy_exit",
            ))

        # Replay
        replay_engine = ReplayEngine(event_store=store)
        report = replay_engine.replay("multi-trade-001")

        # Assert EXACT PnL match (not approximate)
        assert report.final_pnl == expected_total_pnl
        assert report.trades_opened == 5
        assert report.trades_closed == 5
        assert report.signals_count == 5

        store.close()


# ---------------------------------------------------------------------------
# Test 2: Recovery reconstructs open positions
# ---------------------------------------------------------------------------


class TestRecoveryReconstruction:
    """Verify recovery correctly reconstructs open positions from broker."""

    def test_recovery_reconstructs_open_positions(self, tmp_path):
        """3 open positions recovered from broker — all ACTIVE in TradeManager."""
        db_path = str(tmp_path / "recovery_open.db")
        store = EventStore(db_path=db_path, session_id="recovery-open-001")
        bus = EventBus()
        persistence = EventPersistenceSubscriber(store)
        persistence.attach(bus)

        # Persist events for 3 open positions (no TradeClosed!)
        positions_data = [
            {"id": "T-R001", "symbol": "BTCUSDT", "side": "BUY", "qty": 1.0},
            {"id": "T-R002", "symbol": "ETHUSDT", "side": "BUY", "qty": 5.0},
            {"id": "T-R003", "symbol": "SOLUSDT", "side": "SELL", "qty": 100.0},
        ]

        for p in positions_data:
            bus.publish(SignalGenerated(
                symbol=p["symbol"], signal=p["side"], confidence=0.9, price=100.0,
            ))
            bus.publish(TradeOpened(
                trade_id=p["id"], symbol=p["symbol"], side=p["side"],
                entry_price=100.0, qty=p["qty"],
            ))

        store.close()

        # --- Simulate crash: fresh components ---
        new_bus = EventBus()
        new_trade_manager = TradeManager()

        # Mock broker returns 3 open positions
        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = [
            {"symbol": p["symbol"], "side": "long" if p["side"] == "BUY" else "short",
             "qty": p["qty"], "asset_id": p["id"]}
            for p in positions_data
        ]
        mock_broker.get_orders.return_value = []

        # Run recovery
        recovery = RecoveryManager(
            broker=mock_broker,
            event_bus=new_bus,
            trade_manager=new_trade_manager,
            event_store=None,
        )
        report = recovery.recover()

        # Assert: 3 positions recovered, all ACTIVE
        assert report.success is True
        assert report.positions_recovered == 3

        for p in positions_data:
            trade = new_trade_manager.get_trade(p["id"])
            assert trade is not None, f"Trade {p['id']} not found"
            assert trade.state == TradeState.ACTIVE
            assert trade.symbol == p["symbol"]
            assert trade.metadata.get("recovered") is True

    def test_recovery_then_close_completes_correctly(self, tmp_path):
        """Recover 3 positions, close one — 2 remain open, PnL correct."""
        new_bus = EventBus()
        new_trade_manager = TradeManager()

        # Mock broker with 3 positions
        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = [
            {"symbol": "BTCUSDT", "side": "long", "qty": 1.0, "asset_id": "T-RC01"},
            {"symbol": "ETHUSDT", "side": "long", "qty": 5.0, "asset_id": "T-RC02"},
            {"symbol": "SOLUSDT", "side": "short", "qty": 50.0, "asset_id": "T-RC03"},
        ]
        mock_broker.get_orders.return_value = []

        # Recover
        recovery = RecoveryManager(
            broker=mock_broker,
            event_bus=new_bus,
            trade_manager=new_trade_manager,
        )
        report = recovery.recover()
        assert report.positions_recovered == 3

        # Close one trade
        trade = new_trade_manager.get_trade("T-RC01")
        assert trade is not None
        trade.transition(TradeState.CLOSING, "manual close")
        trade.transition(TradeState.CLOSED, "manual close")

        # Verify: 2 active, 1 closed
        active_trades = new_trade_manager.get_active_trades()
        assert len(active_trades) == 2

        closed_trades = new_trade_manager.get_trades_by_state(TradeState.CLOSED)
        assert len(closed_trades) == 1
        assert closed_trades[0].trade_id == "T-RC01"


# ---------------------------------------------------------------------------
# Test 3: Reconciliation detects missing position
# ---------------------------------------------------------------------------


class TestReconciliation:
    """Verify reconciler detects discrepancies and auto-fixes."""

    def test_reconciliation_detects_missing_position(self):
        """Broker has AAPL + MSFT, internal only has AAPL — detect MISSING_INTERNAL."""
        bus = EventBus()
        trade_manager = TradeManager()

        # Create internal trade for AAPL only
        aapl_trade = trade_manager.create_trade(
            symbol="AAPL", side="BUY", trade_id="T-AAPL"
        )
        aapl_trade.transition(TradeState.PENDING_RISK, "test")
        aapl_trade.transition(TradeState.RISK_APPROVED, "test")
        aapl_trade.transition(TradeState.SUBMITTED, "test")
        aapl_trade.transition(TradeState.ACCEPTED, "test")
        aapl_trade.transition(TradeState.FILLED, "test")
        aapl_trade.transition(TradeState.ACTIVE, "test")

        # Broker has both AAPL and MSFT
        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = [
            {"symbol": "AAPL", "side": "long", "qty": 10, "asset_id": "T-AAPL"},
            {"symbol": "MSFT", "side": "long", "qty": 20, "asset_id": "T-MSFT"},
        ]

        reconciler = PortfolioReconciler(
            broker=mock_broker,
            trade_manager=trade_manager,
            event_bus=bus,
        )

        report = reconciler.reconcile()

        # Verify MSFT detected as MISSING_INTERNAL
        assert report.is_reconciled is False
        assert len(report.discrepancies) >= 1

        msft_disc = [d for d in report.discrepancies if d.symbol == "MSFT"]
        assert len(msft_disc) == 1
        assert msft_disc[0].type == MISSING_INTERNAL

        # Auto-fix should create MSFT lifecycle
        fixes = reconciler.auto_fix(report)
        assert len(fixes) >= 1
        assert any("MSFT" in f for f in fixes)

        # Verify MSFT now tracked
        msft_trade = trade_manager.get_trade("T-MSFT")
        assert msft_trade is not None
        assert msft_trade.state == TradeState.ACTIVE
        assert msft_trade.symbol == "MSFT"


# ---------------------------------------------------------------------------
# Test 4: Full lifecycle — persist, recover, replay
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    """End-to-end: execute → persist → crash → recover → replay → compare."""

    def test_full_lifecycle_persist_recover_replay(self, tmp_path):
        """Full E2E fidelity test — recovered state matches original."""
        db_path = str(tmp_path / "full_lifecycle.db")

        # === Phase 1: Execute trades and persist ===
        store = EventStore(db_path=db_path, session_id="lifecycle-001")
        bus = EventBus()
        trade_manager = TradeManager()
        persistence = EventPersistenceSubscriber(store)
        persistence.attach(bus)

        # Track original state
        original_state = {"pnl": 0.0, "opened": 0, "closed": 0, "signals": 0}

        def live_tracker(event):
            if isinstance(event, SignalGenerated):
                original_state["signals"] += 1
            elif isinstance(event, TradeOpened):
                original_state["opened"] += 1
            elif isinstance(event, TradeClosed):
                original_state["closed"] += 1
                original_state["pnl"] += event.pnl

        bus.subscribe(None, live_tracker)

        # Execute a full trade lifecycle
        bus.publish(SignalGenerated(
            symbol="BTCUSDT", signal="BUY", confidence=0.95,
            strategy="ml_model", price=45000.0,
        ))
        bus.publish(TradeOpened(
            trade_id="T-LC01", symbol="BTCUSDT", side="BUY",
            entry_price=45000.0, qty=2.0,
        ))
        bus.publish(SignalGenerated(
            symbol="ETHUSDT", signal="BUY", confidence=0.88,
            strategy="ml_model", price=3000.0,
        ))
        bus.publish(TradeOpened(
            trade_id="T-LC02", symbol="ETHUSDT", side="BUY",
            entry_price=3000.0, qty=10.0,
        ))
        # Close first trade
        bus.publish(TradeClosed(
            trade_id="T-LC01", symbol="BTCUSDT",
            exit_price=47000.0, pnl=4000.0, pnl_pct=4.44,
            reason="take_profit",
        ))

        original_event_count = store.count_events(session_id="lifecycle-001")

        # === Phase 2: "Crash" — fresh components ===
        new_bus = EventBus()
        new_trade_manager = TradeManager()

        # Mock broker: only ETHUSDT is still open (BTC was closed)
        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = [
            {"symbol": "ETHUSDT", "side": "long", "qty": 10.0, "asset_id": "T-LC02"},
        ]
        mock_broker.get_orders.return_value = []

        # === Phase 3: Recovery ===
        recovery = RecoveryManager(
            broker=mock_broker,
            event_bus=new_bus,
            trade_manager=new_trade_manager,
        )
        recovery_report = recovery.recover()
        assert recovery_report.success is True
        assert recovery_report.positions_recovered == 1

        # Verify ETHUSDT recovered
        eth_trade = new_trade_manager.get_trade("T-LC02")
        assert eth_trade is not None
        assert eth_trade.state == TradeState.ACTIVE
        assert eth_trade.symbol == "ETHUSDT"

        # === Phase 4: Replay and compare ===
        replay_engine = ReplayEngine(event_store=store)
        replay_report = replay_engine.replay("lifecycle-001")

        # Compare replay vs original
        assert replay_report.events_replayed == original_event_count
        assert replay_report.signals_count == original_state["signals"]
        assert replay_report.trades_opened == original_state["opened"]
        assert replay_report.trades_closed == original_state["closed"]
        assert replay_report.final_pnl == original_state["pnl"]

        store.close()

"""
Tests for crash recovery and portfolio reconciliation modules.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from src.core.recovery import RecoveryManager, RecoveryReport
from src.core.reconciliation import (
    PortfolioReconciler,
    ReconciliationReport,
    Discrepancy,
    MISSING_INTERNAL,
    MISSING_BROKER,
    QTY_MISMATCH,
    SIDE_MISMATCH,
)
from src.core.state_machine import TradeManager, TradeState, TradeLifecycle
from src.core.events import EventBus, SystemHealth


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_broker():
    broker = MagicMock()
    broker.get_positions.return_value = []
    broker.get_orders.return_value = []
    broker.get_account.return_value = {"equity": 100000}
    return broker


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def trade_manager():
    return TradeManager()


@pytest.fixture
def recovery_manager(mock_broker, event_bus, trade_manager):
    return RecoveryManager(
        broker=mock_broker,
        event_bus=event_bus,
        trade_manager=trade_manager,
    )


@pytest.fixture
def reconciler(mock_broker, trade_manager, event_bus):
    return PortfolioReconciler(
        broker=mock_broker,
        trade_manager=trade_manager,
        event_bus=event_bus,
        interval_seconds=60,
    )


# ---------------------------------------------------------------------------
# Recovery Tests
# ---------------------------------------------------------------------------


class TestRecoveryManager:
    def test_recover_empty_broker(self, recovery_manager, mock_broker):
        """Recovery with no open positions succeeds cleanly."""
        mock_broker.get_positions.return_value = []
        report = recovery_manager.recover()

        assert report.success is True
        assert report.positions_recovered == 0
        assert report.orphans_detected == 0

    def test_recover_with_positions(self, recovery_manager, mock_broker, trade_manager):
        """Recovery reconstructs lifecycles for broker positions."""
        mock_broker.get_positions.return_value = [
            {"symbol": "AAPL", "side": "long", "qty": "10", "asset_id": "pos-aapl"},
            {"symbol": "TSLA", "side": "short", "qty": "5", "asset_id": "pos-tsla"},
        ]

        report = recovery_manager.recover()

        assert report.success is True
        assert report.positions_recovered == 2

        # Verify lifecycles are in ACTIVE state
        aapl_trade = trade_manager.get_trade("pos-aapl")
        assert aapl_trade is not None
        assert aapl_trade.state == TradeState.ACTIVE
        assert aapl_trade.symbol == "AAPL"
        assert aapl_trade.metadata["recovered"] is True

        tsla_trade = trade_manager.get_trade("pos-tsla")
        assert tsla_trade is not None
        assert tsla_trade.state == TradeState.ACTIVE
        assert tsla_trade.symbol == "TSLA"

    def test_recover_publishes_health_event(self, recovery_manager, event_bus, mock_broker):
        """Recovery publishes SystemHealth event on completion."""
        mock_broker.get_positions.return_value = []
        events_received = []
        event_bus.subscribe(SystemHealth, lambda e: events_received.append(e))

        recovery_manager.recover()

        assert len(events_received) == 1
        assert events_received[0].component == "recovery_manager"
        assert events_received[0].status == "healthy"

    def test_recover_broker_failure(self, recovery_manager, mock_broker, event_bus):
        """Recovery handles broker API failure gracefully."""
        mock_broker.get_positions.side_effect = ConnectionError("Broker unavailable")
        events_received = []
        event_bus.subscribe(SystemHealth, lambda e: events_received.append(e))

        report = recovery_manager.recover()

        assert report.success is False
        assert "Recovery failed" in report.warnings[0]
        assert events_received[0].status == "degraded"

    def test_orphan_detection(self, recovery_manager, mock_broker, trade_manager):
        """Orphan detection finds broker positions not tracked internally."""
        # Broker has AAPL and GOOGL
        mock_broker.get_positions.return_value = [
            {"symbol": "AAPL", "side": "long", "qty": "10", "asset_id": "pos-aapl"},
            {"symbol": "GOOGL", "side": "long", "qty": "3", "asset_id": "pos-googl"},
        ]

        report = recovery_manager.recover()

        # Both get recovered, so no orphans (they are now tracked)
        assert report.success is True
        assert report.positions_recovered == 2
        assert report.orphans_detected == 0

    def test_orphan_detection_with_existing_trades(
        self, recovery_manager, mock_broker, trade_manager
    ):
        """Orphan detection when some positions already tracked."""
        # Pre-register AAPL in trade manager
        trade = trade_manager.create_trade(
            symbol="AAPL", side="BUY", trade_id="pos-aapl"
        )
        trade.transition(TradeState.PENDING_RISK, "test")
        trade.transition(TradeState.RISK_APPROVED, "test")
        trade.transition(TradeState.SUBMITTED, "test")
        trade.transition(TradeState.ACCEPTED, "test")
        trade.transition(TradeState.FILLED, "test")
        trade.transition(TradeState.ACTIVE, "test")

        # Broker has AAPL (tracked) and GOOGL (will be recovered)
        mock_broker.get_positions.return_value = [
            {"symbol": "AAPL", "side": "long", "qty": "10", "asset_id": "pos-aapl"},
            {"symbol": "GOOGL", "side": "long", "qty": "3", "asset_id": "pos-googl"},
        ]

        report = recovery_manager.recover()

        assert report.success is True
        # Only GOOGL gets recovered (AAPL already tracked)
        assert report.positions_recovered == 1

    def test_recover_with_event_store(self, mock_broker, event_bus, trade_manager):
        """Recovery replays events from event store."""
        mock_event_store = MagicMock()
        mock_event_store.get_recent_events.return_value = [
            SystemHealth(component="test", status="healthy"),
            SystemHealth(component="test2", status="healthy"),
        ]
        mock_broker.get_positions.return_value = []

        manager = RecoveryManager(
            broker=mock_broker,
            event_bus=event_bus,
            trade_manager=trade_manager,
            event_store=mock_event_store,
        )

        report = manager.recover()

        assert report.success is True
        assert report.events_replayed == 2


# ---------------------------------------------------------------------------
# Reconciliation Tests
# ---------------------------------------------------------------------------


class TestPortfolioReconciler:
    def test_reconcile_matching_state(self, reconciler, mock_broker, trade_manager):
        """Reconciliation passes when broker and internal match."""
        # Setup internal trade
        trade = trade_manager.create_trade(
            symbol="AAPL", side="BUY", trade_id="t-aapl"
        )
        trade.transition(TradeState.PENDING_RISK, "test")
        trade.transition(TradeState.RISK_APPROVED, "test")
        trade.transition(TradeState.SUBMITTED, "test")
        trade.transition(TradeState.ACCEPTED, "test")
        trade.transition(TradeState.FILLED, "test")
        trade.transition(TradeState.ACTIVE, "test")
        trade.metadata["broker_qty"] = 10

        # Broker matches
        mock_broker.get_positions.return_value = [
            {"symbol": "AAPL", "side": "long", "qty": "10"},
        ]

        report = reconciler.reconcile()

        assert report.is_reconciled is True
        assert report.broker_positions == 1
        assert report.internal_positions == 1
        assert len(report.discrepancies) == 0

    def test_reconcile_missing_internal(self, reconciler, mock_broker, trade_manager):
        """Detect position in broker but not tracked internally."""
        mock_broker.get_positions.return_value = [
            {"symbol": "AAPL", "side": "long", "qty": "10"},
        ]

        report = reconciler.reconcile()

        assert report.is_reconciled is False
        assert len(report.discrepancies) == 1
        assert report.discrepancies[0].type == MISSING_INTERNAL
        assert report.discrepancies[0].symbol == "AAPL"
        assert report.discrepancies[0].severity == "HIGH"

    def test_reconcile_missing_broker(self, reconciler, mock_broker, trade_manager):
        """Detect position tracked internally but not in broker."""
        trade = trade_manager.create_trade(
            symbol="TSLA", side="BUY", trade_id="t-tsla"
        )
        trade.transition(TradeState.PENDING_RISK, "test")
        trade.transition(TradeState.RISK_APPROVED, "test")
        trade.transition(TradeState.SUBMITTED, "test")
        trade.transition(TradeState.ACCEPTED, "test")
        trade.transition(TradeState.FILLED, "test")
        trade.transition(TradeState.ACTIVE, "test")

        mock_broker.get_positions.return_value = []

        report = reconciler.reconcile()

        assert report.is_reconciled is False
        assert len(report.discrepancies) == 1
        assert report.discrepancies[0].type == MISSING_BROKER
        assert report.discrepancies[0].symbol == "TSLA"

    def test_reconcile_qty_mismatch(self, reconciler, mock_broker, trade_manager):
        """Detect quantity mismatch between broker and internal."""
        trade = trade_manager.create_trade(
            symbol="AAPL", side="BUY", trade_id="t-aapl"
        )
        trade.transition(TradeState.PENDING_RISK, "test")
        trade.transition(TradeState.RISK_APPROVED, "test")
        trade.transition(TradeState.SUBMITTED, "test")
        trade.transition(TradeState.ACCEPTED, "test")
        trade.transition(TradeState.FILLED, "test")
        trade.transition(TradeState.ACTIVE, "test")
        trade.metadata["broker_qty"] = 10

        # Broker has different qty
        mock_broker.get_positions.return_value = [
            {"symbol": "AAPL", "side": "long", "qty": "15"},
        ]

        report = reconciler.reconcile()

        assert report.is_reconciled is False
        assert len(report.discrepancies) == 1
        assert report.discrepancies[0].type == QTY_MISMATCH
        assert report.discrepancies[0].severity == "MEDIUM"

    def test_reconcile_side_mismatch(self, reconciler, mock_broker, trade_manager):
        """Detect side mismatch between broker and internal."""
        trade = trade_manager.create_trade(
            symbol="AAPL", side="BUY", trade_id="t-aapl"
        )
        trade.transition(TradeState.PENDING_RISK, "test")
        trade.transition(TradeState.RISK_APPROVED, "test")
        trade.transition(TradeState.SUBMITTED, "test")
        trade.transition(TradeState.ACCEPTED, "test")
        trade.transition(TradeState.FILLED, "test")
        trade.transition(TradeState.ACTIVE, "test")
        trade.metadata["broker_qty"] = 10

        # Broker says short but internal says BUY
        mock_broker.get_positions.return_value = [
            {"symbol": "AAPL", "side": "short", "qty": "10"},
        ]

        report = reconciler.reconcile()

        assert report.is_reconciled is False
        assert len(report.discrepancies) == 1
        assert report.discrepancies[0].type == SIDE_MISMATCH
        assert report.discrepancies[0].severity == "HIGH"

    def test_reconcile_broker_failure(self, reconciler, mock_broker):
        """Reconciliation handles broker API failure."""
        mock_broker.get_positions.side_effect = ConnectionError("timeout")

        report = reconciler.reconcile()

        assert report.is_reconciled is False
        assert len(report.discrepancies) == 1
        assert report.discrepancies[0].severity == "HIGH"

    def test_reconcile_publishes_warnings(
        self, reconciler, mock_broker, trade_manager, event_bus
    ):
        """Reconciliation publishes SystemHealth events for discrepancies."""
        mock_broker.get_positions.return_value = [
            {"symbol": "AAPL", "side": "long", "qty": "10"},
        ]
        events_received = []
        event_bus.subscribe(SystemHealth, lambda e: events_received.append(e))

        reconciler.reconcile()

        assert len(events_received) == 1
        assert events_received[0].component == "portfolio_reconciler"
        assert events_received[0].status == "degraded"

    def test_auto_fix_missing_internal(self, reconciler, mock_broker, trade_manager):
        """Auto-fix creates lifecycle for MISSING_INTERNAL discrepancies."""
        mock_broker.get_positions.return_value = [
            {"symbol": "AAPL", "side": "long", "qty": "10", "asset_id": "fix-aapl"},
        ]

        report = reconciler.reconcile()
        assert report.is_reconciled is False

        fixes = reconciler.auto_fix(report)

        assert len(fixes) == 1
        assert "AAPL" in fixes[0]

        # Verify lifecycle was created
        trade = trade_manager.get_trade("fix-aapl")
        assert trade is not None
        assert trade.state == TradeState.ACTIVE
        assert trade.symbol == "AAPL"
        assert trade.metadata["auto_fixed"] is True

    def test_auto_fix_does_not_fix_missing_broker(
        self, reconciler, mock_broker, trade_manager
    ):
        """Auto-fix does NOT attempt to fix MISSING_BROKER discrepancies."""
        trade = trade_manager.create_trade(
            symbol="TSLA", side="BUY", trade_id="t-tsla"
        )
        trade.transition(TradeState.PENDING_RISK, "test")
        trade.transition(TradeState.RISK_APPROVED, "test")
        trade.transition(TradeState.SUBMITTED, "test")
        trade.transition(TradeState.ACCEPTED, "test")
        trade.transition(TradeState.FILLED, "test")
        trade.transition(TradeState.ACTIVE, "test")

        mock_broker.get_positions.return_value = []

        report = reconciler.reconcile()
        fixes = reconciler.auto_fix(report)

        # No fixes applied for MISSING_BROKER
        assert len(fixes) == 0

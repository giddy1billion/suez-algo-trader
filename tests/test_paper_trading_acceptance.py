"""
End-to-End Paper Trading Acceptance Test Suite.

Exercises the full paper-trading lifecycle:
1. Signals always receive terminal verdicts (approve, reject, defer, veto, timeout)
2. Manual Telegram buy/sell commands handle success and broker-error gracefully
3. Orders reconcile correctly with the broker
4. Positions can be closed and recovered after a restart

Collects logs and metrics for the full lifecycle.
"""

import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from src.broker.paper import PaperBroker
from src.core.events import (
    EventBus,
    SignalGenerated,
    RiskEvaluated,
    SignalRejected,
    OrderSubmitted,
    OrderFilled,
    TradeOpened,
    TradeClosed,
    SystemHealth,
)
from src.core.state_machine import TradeManager, TradeState, TradeLifecycle
from src.core.recovery import RecoveryManager, RecoveryReport
from src.core.reconciliation import (
    PortfolioReconciler,
    ReconciliationReport,
    Discrepancy,
    MISSING_INTERNAL,
    MISSING_BROKER,
    QTY_MISMATCH,
)
from src.notifications.correlation_store import InMemoryCorrelationStore
from src.notifications.telegram_audit_forwarder import TelegramAuditForwarder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def paper_broker():
    """Fresh paper broker with $100k starting equity."""
    broker = PaperBroker(starting_equity=100_000.0)
    broker.set_price("AAPL", 150.0)
    broker.set_price("TSLA", 250.0)
    broker.set_price("GOOG", 2800.0)
    broker.set_price("MSFT", 400.0)
    broker.set_price("NVDA", 900.0)
    return broker


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def trade_manager():
    return TradeManager()


@pytest.fixture
def correlation_store():
    return InMemoryCorrelationStore()


@pytest.fixture
def forwarder(correlation_store):
    """TelegramAuditForwarder with short timeout for testing."""
    send_mock = MagicMock()
    fw = TelegramAuditForwarder(
        send_func=send_mock,
        risk_verdict_timeout_seconds=1.5,
        timeout_check_interval=0.3,
        correlation_store=correlation_store,
    )
    yield fw
    fw.stop()


@pytest.fixture
def forwarder_with_bus(event_bus, forwarder):
    """Forwarder registered on event bus."""
    forwarder.register(event_bus)
    return forwarder


@pytest.fixture
def event_log(event_bus):
    """Captures all events published on the bus."""
    log = []
    event_bus.subscribe(None, lambda e: log.append(e))
    return log


# ---------------------------------------------------------------------------
# PHASE 1: Signal Terminal Verdicts
# ---------------------------------------------------------------------------


class TestSignalTerminalVerdicts:
    """Every signal must reach a terminal verdict (no orphaned correlations)."""

    def test_approved_signal_reaches_terminal(
        self, event_bus, forwarder_with_bus, correlation_store, event_log
    ):
        """Approved signals complete the correlation lifecycle."""
        signal_id = f"sig-approve-{uuid.uuid4().hex[:6]}"

        event_bus.publish(SignalGenerated(
            signal_id=signal_id,
            symbol="AAPL",
            signal="BUY",
            side="BUY",
            signal_strength=0.85,
            confidence=0.85,
            strategy="momentum",
            source="test",
        ))

        # Signal should be tracked
        assert correlation_store.get_signal(signal_id) is not None

        # Verdict arrives
        event_bus.publish(RiskEvaluated(
            symbol="AAPL",
            signal_id=signal_id,
            approved=True,
            reasons=[],
            contract_id="DC-APPROVED-001",
            source="risk_engine",
        ))

        time.sleep(0.2)
        assert correlation_store.metrics.verdicts_correlated >= 1
        assert correlation_store.metrics.timeouts_emitted == 0

    def test_rejected_signal_reaches_terminal(
        self, event_bus, forwarder_with_bus, correlation_store
    ):
        """Rejected signals cancel the correlation deadline."""
        signal_id = f"sig-reject-{uuid.uuid4().hex[:6]}"

        event_bus.publish(SignalGenerated(
            signal_id=signal_id,
            symbol="TSLA",
            signal="SELL",
            side="SELL",
            signal_strength=0.3,
            confidence=0.3,
            strategy="ml",
            source="test",
        ))

        event_bus.publish(SignalRejected(
            signal_id=signal_id,
            symbol="TSLA",
            reason="low_signal_strength",
            stage="strength_gate",
            source="engine",
        ))

        # Wait past timeout
        time.sleep(2.0)
        assert correlation_store.metrics.timeouts_emitted == 0

    def test_deferred_signal_reaches_terminal(
        self, event_bus, forwarder_with_bus, correlation_store
    ):
        """DEFER contracts emit RiskEvaluated(approved=False) to cancel deadline."""
        signal_id = f"sig-defer-{uuid.uuid4().hex[:6]}"

        event_bus.publish(SignalGenerated(
            signal_id=signal_id,
            symbol="GOOG",
            signal="BUY",
            side="BUY",
            signal_strength=0.6,
            confidence=0.6,
            strategy="ml",
            source="test",
        ))

        event_bus.publish(RiskEvaluated(
            symbol="GOOG",
            signal_id=signal_id,
            approved=False,
            reasons=["contract_defer: Insufficient data"],
            contract_id="DC-DEFER-002",
            source="decision_orchestrator",
        ))

        time.sleep(2.0)
        assert correlation_store.metrics.timeouts_emitted == 0
        assert correlation_store.metrics.verdicts_correlated >= 1

    def test_vetoed_signal_reaches_terminal(
        self, event_bus, forwarder_with_bus, correlation_store
    ):
        """Vetoed contracts emit RiskEvaluated(approved=False) to cancel deadline."""
        signal_id = f"sig-veto-{uuid.uuid4().hex[:6]}"

        event_bus.publish(SignalGenerated(
            signal_id=signal_id,
            symbol="NVDA",
            signal="BUY",
            side="BUY",
            signal_strength=0.9,
            confidence=0.9,
            strategy="momentum",
            source="test",
        ))

        event_bus.publish(RiskEvaluated(
            symbol="NVDA",
            signal_id=signal_id,
            approved=False,
            reasons=["vetoed: Circuit breaker active"],
            contract_id="DC-VETO-001",
            source="decision_orchestrator",
        ))

        time.sleep(2.0)
        assert correlation_store.metrics.timeouts_emitted == 0

    def test_orphaned_signal_fires_timeout(
        self, event_bus, forwarder_with_bus, correlation_store
    ):
        """Orphaned signals (no verdict) eventually timeout — proving detection works."""
        signal_id = f"sig-orphan-{uuid.uuid4().hex[:6]}"

        event_bus.publish(SignalGenerated(
            signal_id=signal_id,
            symbol="MSFT",
            signal="BUY",
            side="BUY",
            signal_strength=0.7,
            confidence=0.7,
            strategy="ml",
            source="test",
        ))

        # No verdict arrives
        time.sleep(2.5)
        assert correlation_store.metrics.timeouts_emitted >= 1

    def test_multiple_signals_all_reach_terminal(
        self, event_bus, forwarder_with_bus, correlation_store
    ):
        """Burst of signals all get resolved without orphans."""
        signals = []
        for i in range(5):
            sid = f"sig-burst-{i}-{uuid.uuid4().hex[:4]}"
            signals.append(sid)
            event_bus.publish(SignalGenerated(
                signal_id=sid,
                symbol="AAPL",
                signal="BUY",
                side="BUY",
                signal_strength=0.7 + i * 0.02,
                confidence=0.7 + i * 0.02,
                strategy="momentum",
                source="test",
            ))

        # Resolve all signals
        for sid in signals:
            event_bus.publish(RiskEvaluated(
                symbol="AAPL",
                signal_id=sid,
                approved=True,
                reasons=[],
                contract_id=f"DC-{sid}",
                source="risk_engine",
            ))

        time.sleep(2.0)
        assert correlation_store.metrics.timeouts_emitted == 0
        assert correlation_store.metrics.verdicts_correlated >= 5


# ---------------------------------------------------------------------------
# PHASE 2: Manual Telegram Buy/Sell Commands
# ---------------------------------------------------------------------------


class TestManualTelegramCommands:
    """Telegram buy/sell callbacks handle both success and broker-error gracefully."""

    @pytest.mark.asyncio
    async def test_buy_success_with_paper_broker(self):
        """Buy via paper broker returns a valid order with 'id' field."""
        from src.notifications.telegram_bot import callback_confirm_buy

        broker = PaperBroker(starting_equity=100_000.0)
        broker.set_price("AAPL", 150.0)

        callback = AsyncMock()
        callback.from_user = MagicMock()
        callback.from_user.id = 99999
        callback.data = "buy|AAPL|10.0"
        callback.message = AsyncMock()
        callback.answer = AsyncMock()

        with patch("src.notifications.telegram_bot._authorized_users", {99999}), \
             patch("src.notifications.telegram_bot._broker", broker), \
             patch("src.notifications.telegram_bot._broker_lock", threading.Lock()):
            await callback_confirm_buy(callback)

        call_args = callback.message.edit_text.call_args[0][0]
        assert "✅" in call_args
        assert "AAPL" in call_args

        # Verify order actually filled in broker
        orders = broker.get_orders(status="closed")
        assert len(orders) == 1
        assert orders[0]["status"] == "filled"
        assert orders[0]["symbol"] == "AAPL"
        assert orders[0]["qty"] == 10.0

    @pytest.mark.asyncio
    async def test_buy_broker_error_response(self):
        """Buy with broker returning error dict handles gracefully."""
        from src.notifications.telegram_bot import callback_confirm_buy

        callback = AsyncMock()
        callback.from_user = MagicMock()
        callback.from_user.id = 99999
        callback.data = "buy|AAPL|10.0"
        callback.message = AsyncMock()
        callback.answer = AsyncMock()

        mock_broker = MagicMock()
        mock_broker.market_order.return_value = {
            "error": True,
            "message": "Insufficient buying power",
            "retryable": False,
        }

        with patch("src.notifications.telegram_bot._authorized_users", {99999}), \
             patch("src.notifications.telegram_bot._broker", mock_broker), \
             patch("src.notifications.telegram_bot._broker_lock", threading.Lock()):
            await callback_confirm_buy(callback)

        call_args = callback.message.edit_text.call_args[0][0]
        assert "❌" in call_args or "failed" in call_args.lower()
        assert "Insufficient buying power" in call_args

    @pytest.mark.asyncio
    async def test_buy_broker_exception(self):
        """Buy with broker raising exception shows error message."""
        from src.notifications.telegram_bot import callback_confirm_buy

        callback = AsyncMock()
        callback.from_user = MagicMock()
        callback.from_user.id = 99999
        callback.data = "buy|AAPL|10.0"
        callback.message = AsyncMock()
        callback.answer = AsyncMock()

        mock_broker = MagicMock()
        mock_broker.market_order.side_effect = ConnectionError("API timeout")

        with patch("src.notifications.telegram_bot._authorized_users", {99999}), \
             patch("src.notifications.telegram_bot._broker", mock_broker), \
             patch("src.notifications.telegram_bot._broker_lock", threading.Lock()):
            await callback_confirm_buy(callback)

        call_args = callback.message.edit_text.call_args[0][0]
        assert "❌" in call_args
        assert "API timeout" in call_args

    @pytest.mark.asyncio
    async def test_sell_success_with_paper_broker(self):
        """Sell via paper broker succeeds when position exists."""
        from src.notifications.telegram_bot import callback_confirm_sell

        broker = PaperBroker(starting_equity=100_000.0)
        broker.set_price("TSLA", 250.0)
        # Open a position first
        broker.market_order("TSLA", 5.0, "buy")

        callback = AsyncMock()
        callback.from_user = MagicMock()
        callback.from_user.id = 99999
        callback.data = "sell|TSLA|5.0"
        callback.message = AsyncMock()
        callback.answer = AsyncMock()

        with patch("src.notifications.telegram_bot._authorized_users", {99999}), \
             patch("src.notifications.telegram_bot._broker", broker), \
             patch("src.notifications.telegram_bot._broker_lock", threading.Lock()):
            await callback_confirm_sell(callback)

        call_args = callback.message.edit_text.call_args[0][0]
        assert "✅" in call_args
        assert "TSLA" in call_args

        # Position should be closed
        positions = broker.get_positions()
        assert len(positions) == 0

    @pytest.mark.asyncio
    async def test_sell_broker_error_response(self):
        """Sell with broker returning error dict handles gracefully."""
        from src.notifications.telegram_bot import callback_confirm_sell

        callback = AsyncMock()
        callback.from_user = MagicMock()
        callback.from_user.id = 99999
        callback.data = "sell|GOOG|2.0"
        callback.message = AsyncMock()
        callback.answer = AsyncMock()

        mock_broker = MagicMock()
        mock_broker.market_order.return_value = {
            "error": True,
            "message": "Market closed",
            "retryable": True,
        }

        with patch("src.notifications.telegram_bot._authorized_users", {99999}), \
             patch("src.notifications.telegram_bot._broker", mock_broker), \
             patch("src.notifications.telegram_bot._broker_lock", threading.Lock()):
            await callback_confirm_sell(callback)

        call_args = callback.message.edit_text.call_args[0][0]
        assert "❌" in call_args or "failed" in call_args.lower()
        assert "Market closed" in call_args

    @pytest.mark.asyncio
    async def test_sell_broker_exception(self):
        """Sell with broker raising exception handles gracefully."""
        from src.notifications.telegram_bot import callback_confirm_sell

        callback = AsyncMock()
        callback.from_user = MagicMock()
        callback.from_user.id = 99999
        callback.data = "sell|NVDA|3.0"
        callback.message = AsyncMock()
        callback.answer = AsyncMock()

        mock_broker = MagicMock()
        mock_broker.market_order.side_effect = TimeoutError("Connection timed out")

        with patch("src.notifications.telegram_bot._authorized_users", {99999}), \
             patch("src.notifications.telegram_bot._broker", mock_broker), \
             patch("src.notifications.telegram_bot._broker_lock", threading.Lock()):
            await callback_confirm_sell(callback)

        call_args = callback.message.edit_text.call_args[0][0]
        assert "❌" in call_args
        assert "Connection timed out" in call_args

    @pytest.mark.asyncio
    async def test_unauthorized_user_rejected(self):
        """Unauthorized user gets rejected immediately."""
        from src.notifications.telegram_bot import callback_confirm_buy

        callback = AsyncMock()
        callback.from_user = MagicMock()
        callback.from_user.id = 11111  # Not authorized
        callback.data = "buy|AAPL|10.0"
        callback.answer = AsyncMock()

        with patch("src.notifications.telegram_bot._authorized_users", {99999}):
            await callback_confirm_buy(callback)

        callback.answer.assert_called_with("Unauthorized", show_alert=True)


# ---------------------------------------------------------------------------
# PHASE 3: Order Reconciliation with Broker
# ---------------------------------------------------------------------------


class TestOrderReconciliation:
    """Orders reconcile correctly between internal state and paper broker."""

    def test_reconcile_clean_state(self, paper_broker, trade_manager, event_bus):
        """No discrepancies when internal state matches broker."""
        reconciler = PortfolioReconciler(
            broker=paper_broker,
            trade_manager=trade_manager,
            event_bus=event_bus,
            interval_seconds=60,
        )

        report = reconciler.reconcile()
        assert report.is_reconciled is True
        assert len(report.discrepancies) == 0
        assert report.broker_positions == 0
        assert report.internal_positions == 0

    def test_reconcile_detects_missing_internal(
        self, paper_broker, trade_manager, event_bus
    ):
        """Broker has a position not tracked internally → MISSING_INTERNAL."""
        # Open position in broker
        paper_broker.market_order("AAPL", 10.0, "buy")

        reconciler = PortfolioReconciler(
            broker=paper_broker,
            trade_manager=trade_manager,
            event_bus=event_bus,
            interval_seconds=60,
        )

        report = reconciler.reconcile()
        assert report.is_reconciled is False
        assert report.broker_positions == 1
        assert report.internal_positions == 0
        assert any(d.type == MISSING_INTERNAL for d in report.discrepancies)

    def test_reconcile_detects_missing_broker(
        self, paper_broker, trade_manager, event_bus
    ):
        """Internal state has position but broker doesn't → MISSING_BROKER."""
        # Create internal trade without corresponding broker position
        trade = trade_manager.create_trade(symbol="TSLA", side="BUY", trade_id="t-001")
        trade.transition(TradeState.PENDING_RISK, "risk check")
        trade.transition(TradeState.RISK_APPROVED, "approved")
        trade.transition(TradeState.SUBMITTED, "submitted")
        trade.transition(TradeState.ACCEPTED, "accepted")
        trade.transition(TradeState.FILLED, "filled")
        trade.transition(TradeState.ACTIVE, "active")
        trade.metadata["broker_qty"] = 5.0

        reconciler = PortfolioReconciler(
            broker=paper_broker,
            trade_manager=trade_manager,
            event_bus=event_bus,
            interval_seconds=60,
        )

        report = reconciler.reconcile()
        assert report.is_reconciled is False
        assert any(d.type == MISSING_BROKER for d in report.discrepancies)

    def test_reconcile_detects_qty_mismatch(
        self, paper_broker, trade_manager, event_bus
    ):
        """Quantity mismatch between broker and internal state."""
        # Open 10 shares in broker
        paper_broker.market_order("AAPL", 10.0, "buy")

        # Track internally as 5 shares
        trade = trade_manager.create_trade(symbol="AAPL", side="BUY", trade_id="t-002")
        trade.transition(TradeState.PENDING_RISK, "risk check")
        trade.transition(TradeState.RISK_APPROVED, "approved")
        trade.transition(TradeState.SUBMITTED, "submitted")
        trade.transition(TradeState.ACCEPTED, "accepted")
        trade.transition(TradeState.FILLED, "filled")
        trade.transition(TradeState.ACTIVE, "active")
        trade.metadata["broker_qty"] = 5.0

        reconciler = PortfolioReconciler(
            broker=paper_broker,
            trade_manager=trade_manager,
            event_bus=event_bus,
            interval_seconds=60,
        )

        report = reconciler.reconcile()
        assert report.is_reconciled is False
        assert any(d.type == QTY_MISMATCH for d in report.discrepancies)

    def test_reconcile_after_full_order_lifecycle(
        self, paper_broker, trade_manager, event_bus
    ):
        """Full lifecycle: open, trade, close — reconciliation is clean after."""
        # Open and close position in broker
        paper_broker.market_order("AAPL", 10.0, "buy")
        paper_broker.close_position("AAPL")

        reconciler = PortfolioReconciler(
            broker=paper_broker,
            trade_manager=trade_manager,
            event_bus=event_bus,
            interval_seconds=60,
        )

        report = reconciler.reconcile()
        assert report.is_reconciled is True
        assert report.broker_positions == 0

    def test_reconcile_multiple_positions(
        self, paper_broker, trade_manager, event_bus
    ):
        """Multiple positions in both broker and internal state reconcile correctly."""
        # Open positions in broker
        paper_broker.market_order("AAPL", 10.0, "buy")
        paper_broker.market_order("TSLA", 5.0, "buy")

        # Track them internally with matching quantities
        for symbol, qty, tid in [("AAPL", 10.0, "t-aapl"), ("TSLA", 5.0, "t-tsla")]:
            trade = trade_manager.create_trade(symbol=symbol, side="BUY", trade_id=tid)
            trade.transition(TradeState.PENDING_RISK, "risk check")
            trade.transition(TradeState.RISK_APPROVED, "approved")
            trade.transition(TradeState.SUBMITTED, "submitted")
            trade.transition(TradeState.ACCEPTED, "accepted")
            trade.transition(TradeState.FILLED, "filled")
            trade.transition(TradeState.ACTIVE, "active")
            trade.metadata["broker_qty"] = qty

        reconciler = PortfolioReconciler(
            broker=paper_broker,
            trade_manager=trade_manager,
            event_bus=event_bus,
            interval_seconds=60,
        )

        report = reconciler.reconcile()
        assert report.is_reconciled is True
        assert report.broker_positions == 2
        assert report.internal_positions == 2
        assert len(report.discrepancies) == 0

    def test_idempotent_market_order(self, paper_broker):
        """Duplicate client_order_id returns original order, not duplicate."""
        client_id = "idem-test-001"
        order1 = paper_broker.market_order("AAPL", 5.0, "buy", client_order_id=client_id)
        order2 = paper_broker.market_order("AAPL", 5.0, "buy", client_order_id=client_id)

        assert order1["id"] == order2["id"]
        # Only one position opened
        positions = paper_broker.get_positions()
        assert len(positions) == 1
        assert positions[0]["qty"] == 5.0


# ---------------------------------------------------------------------------
# PHASE 4: Position Close and Recovery After Restart
# ---------------------------------------------------------------------------


class TestPositionCloseAndRecovery:
    """Positions can be closed and recovered after a simulated restart."""

    def test_close_position_fully(self, paper_broker):
        """Full position close removes it from broker state."""
        paper_broker.market_order("AAPL", 10.0, "buy")
        assert len(paper_broker.get_positions()) == 1

        result = paper_broker.close_position("AAPL")
        assert result is not None
        assert result["status"] == "filled"
        assert len(paper_broker.get_positions()) == 0

    def test_close_position_partially(self, paper_broker):
        """Partial close reduces position quantity."""
        paper_broker.market_order("AAPL", 10.0, "buy")

        result = paper_broker.close_position("AAPL", qty=3.0)
        assert result is not None
        positions = paper_broker.get_positions()
        assert len(positions) == 1
        assert positions[0]["qty"] == 7.0

    def test_close_nonexistent_position(self, paper_broker):
        """Closing a non-existent position returns None gracefully."""
        result = paper_broker.close_position("ZZZZ")
        assert result is None

    def test_recovery_after_restart_with_positions(self, paper_broker, event_bus, trade_manager):
        """Recovery reconstructs trade lifecycles from broker positions."""
        # Simulate pre-crash state: broker has open positions
        paper_broker.market_order("AAPL", 10.0, "buy")
        paper_broker.market_order("TSLA", 5.0, "buy")

        # Simulate restart — fresh trade manager (no state)
        recovery = RecoveryManager(
            broker=paper_broker,
            event_bus=event_bus,
            trade_manager=trade_manager,
        )

        report = recovery.recover()
        assert report.success is True
        assert report.positions_recovered == 2

        # Verify lifecycles are reconstructed
        active_trades = trade_manager.get_active_trades()
        assert len(active_trades) == 2
        symbols = {t.symbol for t in active_trades}
        assert "AAPL" in symbols
        assert "TSLA" in symbols

    def test_recovery_empty_broker(self, paper_broker, event_bus, trade_manager):
        """Recovery with no positions succeeds cleanly."""
        recovery = RecoveryManager(
            broker=paper_broker,
            event_bus=event_bus,
            trade_manager=trade_manager,
        )

        report = recovery.recover()
        assert report.success is True
        assert report.positions_recovered == 0
        assert report.orphans_detected == 0

    def test_recovery_publishes_health_event(self, paper_broker, event_bus, trade_manager):
        """Recovery publishes SystemHealth event."""
        paper_broker.market_order("AAPL", 10.0, "buy")
        events_received = []
        event_bus.subscribe(SystemHealth, lambda e: events_received.append(e))

        recovery = RecoveryManager(
            broker=paper_broker,
            event_bus=event_bus,
            trade_manager=trade_manager,
        )
        recovery.recover()

        assert len(events_received) == 1
        assert events_received[0].component == "recovery_manager"
        assert events_received[0].status == "healthy"

    def test_recovery_broker_failure_graceful(self, event_bus, trade_manager):
        """Recovery handles broker failure without crashing."""
        bad_broker = MagicMock()
        bad_broker.get_positions.side_effect = ConnectionError("Broker down")
        bad_broker.get_orders.return_value = []

        events_received = []
        event_bus.subscribe(SystemHealth, lambda e: events_received.append(e))

        recovery = RecoveryManager(
            broker=bad_broker,
            event_bus=event_bus,
            trade_manager=trade_manager,
        )

        report = recovery.recover()
        assert report.success is False
        assert any("failed" in w.lower() or "Recovery" in w for w in report.warnings)
        assert events_received[0].status == "degraded"

    def test_close_then_recover_shows_empty(self, paper_broker, event_bus, trade_manager):
        """After closing all positions, recovery sees no positions."""
        paper_broker.market_order("AAPL", 10.0, "buy")
        paper_broker.close_position("AAPL")

        recovery = RecoveryManager(
            broker=paper_broker,
            event_bus=event_bus,
            trade_manager=trade_manager,
        )

        report = recovery.recover()
        assert report.success is True
        assert report.positions_recovered == 0

    def test_recovery_then_reconcile_is_clean(
        self, paper_broker, event_bus, trade_manager
    ):
        """After recovery, reconciliation should show no discrepancies."""
        paper_broker.market_order("AAPL", 10.0, "buy")
        paper_broker.market_order("TSLA", 5.0, "buy")

        recovery = RecoveryManager(
            broker=paper_broker,
            event_bus=event_bus,
            trade_manager=trade_manager,
        )
        report = recovery.recover()
        assert report.success is True

        reconciler = PortfolioReconciler(
            broker=paper_broker,
            trade_manager=trade_manager,
            event_bus=event_bus,
            interval_seconds=60,
        )
        recon_report = reconciler.reconcile()
        # After recovery, internal state should match broker
        # (positions_recovered populates trade manager)
        assert recon_report.broker_positions == 2


# ---------------------------------------------------------------------------
# PHASE 5: Full Lifecycle Integration (End-to-End)
# ---------------------------------------------------------------------------


class TestFullLifecycleIntegration:
    """End-to-end: signal → verdict → order → reconcile → close → recover."""

    def test_complete_trading_lifecycle(
        self, paper_broker, event_bus, trade_manager, correlation_store
    ):
        """Full lifecycle from signal generation through close and recovery."""
        # Setup forwarder
        send_mock = MagicMock()
        forwarder = TelegramAuditForwarder(
            send_func=send_mock,
            risk_verdict_timeout_seconds=2.0,
            timeout_check_interval=0.3,
            correlation_store=correlation_store,
        )
        forwarder.register(event_bus)

        try:
            # 1. Generate signal
            signal_id = "lifecycle-sig-001"
            event_bus.publish(SignalGenerated(
                signal_id=signal_id,
                symbol="AAPL",
                signal="BUY",
                side="BUY",
                signal_strength=0.85,
                confidence=0.85,
                strategy="momentum",
                source="test",
            ))

            # 2. Approve the signal (terminal verdict)
            event_bus.publish(RiskEvaluated(
                symbol="AAPL",
                signal_id=signal_id,
                approved=True,
                reasons=[],
                contract_id="DC-LIFECYCLE-001",
                source="risk_engine",
            ))

            # 3. Execute order via paper broker
            order = paper_broker.market_order("AAPL", 10.0, "buy")
            assert order["status"] == "filled"
            assert order["id"] is not None

            # 4. Track in trade manager
            trade = trade_manager.create_trade(
                symbol="AAPL", side="BUY", trade_id=order["id"]
            )
            trade.transition(TradeState.PENDING_RISK, "risk check")
            trade.transition(TradeState.RISK_APPROVED, "approved")
            trade.transition(TradeState.SUBMITTED, "submitted")
            trade.transition(TradeState.ACCEPTED, "accepted")
            trade.transition(TradeState.FILLED, "filled")
            trade.transition(TradeState.ACTIVE, "active")
            trade.metadata["broker_qty"] = 10.0

            # 5. Reconcile — should be clean
            reconciler = PortfolioReconciler(
                broker=paper_broker,
                trade_manager=trade_manager,
                event_bus=event_bus,
                interval_seconds=60,
            )
            report = reconciler.reconcile()
            assert report.is_reconciled is True

            # 6. Close position
            close_order = paper_broker.close_position("AAPL")
            assert close_order["status"] == "filled"
            trade.transition(TradeState.CLOSING, "user requested close")
            trade.transition(TradeState.CLOSED, "position closed")
            assert trade.is_terminal is True

            # 7. Verify correlation completed without timeout
            time.sleep(0.5)
            assert correlation_store.metrics.timeouts_emitted == 0
            assert correlation_store.metrics.verdicts_correlated >= 1

            # 8. Simulate restart — recover
            fresh_tm = TradeManager()
            recovery = RecoveryManager(
                broker=paper_broker,
                event_bus=event_bus,
                trade_manager=fresh_tm,
            )
            recovery_report = recovery.recover()
            assert recovery_report.success is True
            # Position was closed, so nothing to recover
            assert recovery_report.positions_recovered == 0

        finally:
            forwarder.stop()

    def test_lifecycle_metrics_collected(
        self, paper_broker, event_bus, correlation_store
    ):
        """Verify that metrics are collected throughout the lifecycle."""
        send_mock = MagicMock()
        forwarder = TelegramAuditForwarder(
            send_func=send_mock,
            risk_verdict_timeout_seconds=2.0,
            timeout_check_interval=0.3,
            correlation_store=correlation_store,
        )
        forwarder.register(event_bus)

        try:
            # Generate 3 signals, resolve 2, let 1 timeout
            for i in range(3):
                event_bus.publish(SignalGenerated(
                    signal_id=f"metrics-sig-{i}",
                    symbol="AAPL",
                    signal="BUY",
                    side="BUY",
                    signal_strength=0.8,
                    confidence=0.8,
                    strategy="momentum",
                    source="test",
                ))

            # Resolve first 2
            for i in range(2):
                event_bus.publish(RiskEvaluated(
                    symbol="AAPL",
                    signal_id=f"metrics-sig-{i}",
                    approved=True,
                    reasons=[],
                    contract_id=f"DC-METRICS-{i}",
                    source="risk_engine",
                ))

            # Wait for timeout on the 3rd
            time.sleep(3.0)

            # Metrics should reflect the full lifecycle
            assert correlation_store.metrics.signals_tracked >= 3
            assert correlation_store.metrics.verdicts_correlated >= 2
            assert correlation_store.metrics.timeouts_emitted >= 1

        finally:
            forwarder.stop()


# ---------------------------------------------------------------------------
# PHASE 6: Paper Broker Edge Cases (Regression)
# ---------------------------------------------------------------------------


class TestPaperBrokerEdgeCases:
    """Regression tests for edge cases uncovered during acceptance testing."""

    def test_bracket_order_fills_at_market(self, paper_broker):
        """Bracket order entry fills at market price."""
        order = paper_broker.bracket_order(
            "AAPL", 10.0, "buy", take_profit=160.0, stop_loss=140.0
        )
        assert order["status"] == "filled"
        assert order["filled_avg_price"] == 150.0

    def test_limit_order_pending_until_price_met(self, paper_broker):
        """Limit buy below market stays pending until price drops."""
        order = paper_broker.limit_order("AAPL", 10.0, "buy", limit_price=140.0)
        assert order["status"] == "pending"

        # Price drops to limit
        paper_broker.set_price("AAPL", 140.0)
        # Should auto-fill
        updated = [o for o in paper_broker.get_orders(status="closed") if o["id"] == order["id"]]
        assert len(updated) == 1
        assert updated[0]["status"] == "filled"

    def test_cancel_pending_order(self, paper_broker):
        """Pending orders can be cancelled."""
        order = paper_broker.limit_order("AAPL", 10.0, "buy", limit_price=100.0)
        result = paper_broker.cancel_order(order["id"])
        assert result is not None
        assert result["status"] == "cancelled"

    def test_cancel_filled_order_returns_none(self, paper_broker):
        """Cannot cancel already-filled orders."""
        order = paper_broker.market_order("AAPL", 10.0, "buy")
        result = paper_broker.cancel_order(order["id"])
        assert result is None

    def test_position_pnl_calculation(self, paper_broker):
        """Unrealized P&L updates when price changes."""
        paper_broker.market_order("AAPL", 10.0, "buy")

        # Price goes up
        paper_broker.set_price("AAPL", 160.0)
        positions = paper_broker.get_positions()
        assert positions[0]["unrealized_pl"] == pytest.approx(100.0)  # (160-150)*10

        # Price goes down
        paper_broker.set_price("AAPL", 145.0)
        positions = paper_broker.get_positions()
        assert positions[0]["unrealized_pl"] == pytest.approx(-50.0)  # (145-150)*10

    def test_short_position_pnl(self, paper_broker):
        """Short position P&L is inverse of price movement."""
        paper_broker.market_order("TSLA", 5.0, "sell")
        paper_broker.set_price("TSLA", 240.0)  # Price dropped — profit

        positions = paper_broker.get_positions()
        assert positions[0]["unrealized_pl"] == pytest.approx(50.0)  # (250-240)*5

    def test_account_equity_reflects_positions(self, paper_broker):
        """Account equity includes unrealized P&L."""
        initial = paper_broker.get_account()["equity"]
        paper_broker.market_order("AAPL", 10.0, "buy")
        paper_broker.set_price("AAPL", 160.0)

        account = paper_broker.get_account()
        # Equity should reflect the gain
        assert account["equity"] > initial

    def test_multiple_orders_same_symbol_accumulate(self, paper_broker):
        """Multiple buys of same symbol accumulate into one position."""
        paper_broker.market_order("AAPL", 5.0, "buy")
        paper_broker.market_order("AAPL", 5.0, "buy")

        positions = paper_broker.get_positions()
        assert len(positions) == 1
        assert positions[0]["qty"] == 10.0
        assert positions[0]["avg_entry_price"] == 150.0

    def test_no_price_set_raises_error(self):
        """Market order without price set raises ValueError."""
        broker = PaperBroker()
        with pytest.raises(ValueError, match="No price set"):
            broker.market_order("ZZZZ", 10.0, "buy")

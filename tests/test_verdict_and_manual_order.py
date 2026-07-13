"""
Regression tests for:
1. Verdict timeout fix — non-executable contracts (DEFER, vetoed, expired) now
   emit terminal RiskEvaluated events so correlation deadlines are cancelled.
2. Manual /buy command fix — error responses from broker are handled gracefully
   without KeyError on missing 'id' field.

These tests cover both the automatic signal pipeline and manual order flows.
"""

import time
import threading
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from src.core.events import (
    EventBus,
    SignalGenerated,
    RiskEvaluated,
    SignalRejected,
    DecisionContractCreated,
)
from src.notifications.telegram_audit_forwarder import TelegramAuditForwarder
from src.notifications.correlation_store import InMemoryCorrelationStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def correlation_store():
    return InMemoryCorrelationStore()


@pytest.fixture
def forwarder(correlation_store):
    """Create a TelegramAuditForwarder with in-memory store and short timeout."""
    send_mock = MagicMock()
    fw = TelegramAuditForwarder(
        send_func=send_mock,
        risk_verdict_timeout_seconds=2.0,  # short for testing
        timeout_check_interval=0.5,
        correlation_store=correlation_store,
    )
    yield fw
    fw.stop()


@pytest.fixture
def forwarder_with_bus(event_bus, forwarder):
    """Forwarder registered on event bus."""
    forwarder.register(event_bus)
    return forwarder


# ---------------------------------------------------------------------------
# Test: Verdict timeout fix (automatic signal pipeline)
# ---------------------------------------------------------------------------


class TestVerdictTimeoutFix:
    """Verify that non-executable contracts emit terminal events."""

    def test_risk_evaluated_cancels_deadline(self, event_bus, forwarder_with_bus, correlation_store):
        """Normal flow: RiskEvaluated arrives and cancels the deadline."""
        signal_id = "sig-test-001"

        # 1. Publish signal (sets deadline)
        event_bus.publish(SignalGenerated(
            signal_id=signal_id,
            symbol="AAPL",
            signal="BUY",
            side="BUY",
            signal_strength=0.8,
            confidence=0.8,
            strategy="momentum",
            source="test",
        ))

        # Verify deadline was set
        assert correlation_store.get_signal(signal_id) is not None

        # 2. Publish RiskEvaluated (should cancel deadline)
        event_bus.publish(RiskEvaluated(
            symbol="AAPL",
            signal_id=signal_id,
            approved=True,
            reasons=[],
            contract_id="DC-001",
            source="risk_engine",
        ))

        # Verify deadline was cancelled (no timeout should fire)
        time.sleep(0.1)
        assert correlation_store.metrics.verdicts_correlated >= 1
        assert correlation_store.metrics.timeouts_emitted == 0

    def test_signal_rejected_cancels_deadline(self, event_bus, forwarder_with_bus, correlation_store):
        """SignalRejected cancels the deadline (e.g., low strength, DEFER contracts)."""
        signal_id = "sig-test-002"

        # 1. Publish signal
        event_bus.publish(SignalGenerated(
            signal_id=signal_id,
            symbol="TSLA",
            signal="BUY",
            side="BUY",
            signal_strength=0.3,
            confidence=0.3,
            strategy="ml",
            source="test",
        ))

        # 2. Publish SignalRejected (simulates low_strength_rejected)
        event_bus.publish(SignalRejected(
            signal_id=signal_id,
            symbol="TSLA",
            reason="signal_strength 0.300 < 0.550",
            stage="strength_gate",
            source="engine",
        ))

        # Deadline should be cancelled — no timeout warning
        time.sleep(2.5)  # Wait past the 2s timeout
        assert correlation_store.metrics.timeouts_emitted == 0

    def test_non_executable_contract_emits_risk_evaluated(self, event_bus, forwarder_with_bus, correlation_store):
        """
        Regression test: A contract that is not executable (DEFER/vetoed/expired)
        must emit RiskEvaluated(approved=False) to cancel the correlation deadline.
        
        Before the fix, only REJECT contracts emitted terminal events. DEFER and
        vetoed contracts would silently fall through, causing 60s timeout warnings.
        """
        signal_id = "sig-test-003"

        # 1. Publish signal
        event_bus.publish(SignalGenerated(
            signal_id=signal_id,
            symbol="GOOG",
            signal="BUY",
            side="BUY",
            signal_strength=0.7,
            confidence=0.7,
            strategy="momentum",
            source="test",
        ))

        # 2. Simulate the engine emitting RiskEvaluated for a DEFER contract
        # (This is what the fix now does — emit terminal verdict for ALL non-executable)
        event_bus.publish(RiskEvaluated(
            symbol="GOOG",
            signal_id=signal_id,
            approved=False,
            reasons=["contract_defer: Insufficient data for decision"],
            contract_id="DC-DEFER-001",
            source="decision_orchestrator",
        ))

        # Verify deadline cancelled - no timeout fires
        time.sleep(2.5)
        assert correlation_store.metrics.timeouts_emitted == 0
        assert correlation_store.metrics.verdicts_correlated >= 1

    def test_timeout_fires_when_no_verdict(self, event_bus, forwarder_with_bus, correlation_store):
        """
        Sanity check: If neither RiskEvaluated nor SignalRejected arrives,
        the timeout warning should still fire after the deadline.
        """
        signal_id = "sig-test-orphan"

        # 1. Publish signal (sets 2s deadline)
        event_bus.publish(SignalGenerated(
            signal_id=signal_id,
            symbol="NVDA",
            signal="BUY",
            side="BUY",
            signal_strength=0.9,
            confidence=0.9,
            strategy="ml",
            source="test",
        ))

        # 2. Wait for timeout (2s deadline + check interval)
        time.sleep(3.0)

        # Timeout should have fired
        assert correlation_store.metrics.timeouts_emitted >= 1


class TestExecutionEngineVerdictEmission:
    """
    Test that ExecutionEngine._process_signal emits terminal events for
    all non-executable contract decisions (not just REJECT).
    """

    def test_defer_contract_emits_risk_evaluated(self):
        """DEFER contract should emit RiskEvaluated(approved=False)."""
        from src.execution.engine import ExecutionEngine
        from src.strategy.base import TradeSignal, Side
        from src.intelligence.confidence.decision_contract import (
            DecisionContract, Decision, StageAssessment,
        )
        from unittest.mock import MagicMock, patch

        # Setup minimal engine
        broker = MagicMock()
        broker.get_account.return_value = {"portfolio_value": 100000, "cash": 50000}
        broker.get_positions.return_value = []

        risk_mgr = MagicMock()
        risk_mgr.can_trade.return_value = (True, "")
        risk_mgr.limits = MagicMock()
        risk_mgr.limits.max_single_stock_pct = 0.15

        db = MagicMock()
        event_bus = EventBus()
        published_events = []
        event_bus.subscribe(None, lambda e: published_events.append(e))

        # Mock decision orchestrator that returns a DEFER contract
        mock_orchestrator = MagicMock()
        defer_contract = MagicMock()
        defer_contract.is_executable = False
        defer_contract.decision = Decision.DEFER
        defer_contract.contract_id = "DC-DEFER-TEST"
        defer_contract.recommendation = "Insufficient market data"
        defer_contract.vetoed = False
        defer_contract.final_confidence = 0.4
        defer_contract.recommended_position_pct = 0.0
        defer_contract.recommended_stop_loss = 0.0
        defer_contract.recommended_take_profit = 0.0
        defer_contract.risk_grade = "C"
        defer_contract.stage_scores = {}
        defer_contract.veto_reason = ""
        defer_contract.valid_until = datetime.now(timezone.utc) + timedelta(minutes=5)
        mock_orchestrator.evaluate.return_value = defer_contract

        engine = ExecutionEngine(
            broker=broker,
            risk_manager=risk_mgr,
            db=db,
            event_bus=event_bus,
            decision_orchestrator=mock_orchestrator,
            min_signal_confidence=0.5,
        )

        # Create a signal
        signal = TradeSignal(
            symbol="AAPL",
            side=Side.BUY,
            signal_strength=0.7,
            signal_id="test-defer-signal",
            strategy_id="momentum",
            strategy_version="v1",
            features={"observed_price": 150.0},
            timestamp=datetime.now(timezone.utc),
        )

        # Process
        result = engine._process_signal(signal, 100000, [], {"AAPL": MagicMock(__len__=lambda s: 200)})

        # Should return None (rejected)
        assert result is None

        # Should have emitted RiskEvaluated(approved=False)
        risk_events = [e for e in published_events if isinstance(e, RiskEvaluated)]
        assert len(risk_events) >= 1
        risk_ev = risk_events[-1]
        assert risk_ev.approved is False
        assert risk_ev.signal_id == "test-defer-signal"
        assert "defer" in risk_ev.reasons[0].lower()

    def test_vetoed_contract_emits_risk_evaluated(self):
        """Vetoed contract should emit RiskEvaluated(approved=False)."""
        from src.execution.engine import ExecutionEngine
        from src.strategy.base import TradeSignal, Side
        from src.intelligence.confidence.decision_contract import Decision

        broker = MagicMock()
        broker.get_account.return_value = {"portfolio_value": 100000, "cash": 50000}
        broker.get_positions.return_value = []

        risk_mgr = MagicMock()
        risk_mgr.can_trade.return_value = (True, "")
        risk_mgr.limits = MagicMock()
        risk_mgr.limits.max_single_stock_pct = 0.15

        db = MagicMock()
        event_bus = EventBus()
        published_events = []
        event_bus.subscribe(None, lambda e: published_events.append(e))

        mock_orchestrator = MagicMock()
        vetoed_contract = MagicMock()
        vetoed_contract.is_executable = False
        vetoed_contract.decision = Decision.EXECUTE  # would be execute, but vetoed
        vetoed_contract.contract_id = "DC-VETOED-TEST"
        vetoed_contract.recommendation = "Circuit breaker active"
        vetoed_contract.vetoed = True
        vetoed_contract.final_confidence = 0.8
        vetoed_contract.recommended_position_pct = 5.0
        vetoed_contract.recommended_stop_loss = 145.0
        vetoed_contract.recommended_take_profit = 160.0
        vetoed_contract.risk_grade = "B"
        vetoed_contract.stage_scores = {}
        vetoed_contract.veto_reason = "Circuit breaker triggered"
        vetoed_contract.valid_until = datetime.now(timezone.utc) + timedelta(minutes=5)
        mock_orchestrator.evaluate.return_value = vetoed_contract

        engine = ExecutionEngine(
            broker=broker,
            risk_manager=risk_mgr,
            db=db,
            event_bus=event_bus,
            decision_orchestrator=mock_orchestrator,
            min_signal_confidence=0.5,
        )

        signal = TradeSignal(
            symbol="MSFT",
            side=Side.BUY,
            signal_strength=0.8,
            signal_id="test-vetoed-signal",
            strategy_id="ml",
            strategy_version="v2",
            features={"observed_price": 400.0},
            timestamp=datetime.now(timezone.utc),
        )

        result = engine._process_signal(signal, 100000, [], {"MSFT": MagicMock(__len__=lambda s: 200)})
        assert result is None

        risk_events = [e for e in published_events if isinstance(e, RiskEvaluated)]
        assert len(risk_events) >= 1
        risk_ev = risk_events[-1]
        assert risk_ev.approved is False
        assert risk_ev.signal_id == "test-vetoed-signal"
        assert "vetoed" in risk_ev.reasons[0].lower()


# ---------------------------------------------------------------------------
# Test: Manual /buy command fix
# ---------------------------------------------------------------------------


class TestManualBuyCommandFix:
    """
    Regression test: /buy command handles error responses from broker
    without KeyError on missing 'id' field.
    """

    @pytest.mark.asyncio
    async def test_buy_callback_handles_error_response(self):
        """When broker returns error dict (no 'id'), command should show error message."""
        from src.notifications.telegram_bot import callback_confirm_buy, _authorized_users

        # Mock the callback query
        callback = AsyncMock()
        callback.from_user = MagicMock()
        callback.from_user.id = 12345
        callback.data = "buy|AAPL|10.0"
        callback.message = AsyncMock()
        callback.answer = AsyncMock()

        # Patch _authorized_users and _broker
        error_response = {"error": True, "message": "Insufficient buying power", "retryable": False}

        with patch("src.notifications.telegram_bot._authorized_users", {12345}), \
             patch("src.notifications.telegram_bot._broker") as mock_broker, \
             patch("src.notifications.telegram_bot._broker_lock", threading.Lock()):
            mock_broker.market_order.return_value = error_response

            await callback_confirm_buy(callback)

            # Should show error message, not crash with KeyError
            callback.message.edit_text.assert_called_once()
            call_args = callback.message.edit_text.call_args[0][0]
            assert "failed" in call_args.lower() or "error" in call_args.lower()
            assert "Insufficient buying power" in call_args

    @pytest.mark.asyncio
    async def test_buy_callback_handles_success_response(self):
        """When broker returns success dict with 'id', command should show order ID."""
        from src.notifications.telegram_bot import callback_confirm_buy, _authorized_users

        callback = AsyncMock()
        callback.from_user = MagicMock()
        callback.from_user.id = 12345
        callback.data = "buy|AAPL|10.0"
        callback.message = AsyncMock()
        callback.answer = AsyncMock()

        success_response = {
            "id": "abc12345-def6-7890-ghij-klmnopqrstuv",
            "symbol": "AAPL",
            "side": "buy",
            "qty": 10.0,
            "status": "filled",
        }

        with patch("src.notifications.telegram_bot._authorized_users", {12345}), \
             patch("src.notifications.telegram_bot._broker") as mock_broker, \
             patch("src.notifications.telegram_bot._broker_lock", threading.Lock()):
            mock_broker.market_order.return_value = success_response

            await callback_confirm_buy(callback)

            callback.message.edit_text.assert_called_once()
            call_args = callback.message.edit_text.call_args[0][0]
            assert "abc12345" in call_args
            assert "AAPL" in call_args

    @pytest.mark.asyncio
    async def test_buy_callback_handles_exception(self):
        """When broker raises an exception, command should show error message."""
        from src.notifications.telegram_bot import callback_confirm_buy

        callback = AsyncMock()
        callback.from_user = MagicMock()
        callback.from_user.id = 12345
        callback.data = "buy|AAPL|10.0"
        callback.message = AsyncMock()
        callback.answer = AsyncMock()

        with patch("src.notifications.telegram_bot._authorized_users", {12345}), \
             patch("src.notifications.telegram_bot._broker") as mock_broker, \
             patch("src.notifications.telegram_bot._broker_lock", threading.Lock()):
            mock_broker.market_order.side_effect = ConnectionError("API timeout")

            await callback_confirm_buy(callback)

            callback.message.edit_text.assert_called_once()
            call_args = callback.message.edit_text.call_args[0][0]
            assert "failed" in call_args.lower()
            assert "API timeout" in call_args

    @pytest.mark.asyncio
    async def test_sell_callback_handles_error_response(self):
        """Sell callback also handles error responses gracefully."""
        from src.notifications.telegram_bot import callback_confirm_sell

        callback = AsyncMock()
        callback.from_user = MagicMock()
        callback.from_user.id = 12345
        callback.data = "sell|TSLA|5.0"
        callback.message = AsyncMock()
        callback.answer = AsyncMock()

        error_response = {"error": True, "message": "Market closed", "retryable": True}

        with patch("src.notifications.telegram_bot._authorized_users", {12345}), \
             patch("src.notifications.telegram_bot._broker") as mock_broker, \
             patch("src.notifications.telegram_bot._broker_lock", threading.Lock()):
            mock_broker.market_order.return_value = error_response

            await callback_confirm_sell(callback)

            callback.message.edit_text.assert_called_once()
            call_args = callback.message.edit_text.call_args[0][0]
            assert "failed" in call_args.lower() or "error" in call_args.lower()
            assert "Market closed" in call_args

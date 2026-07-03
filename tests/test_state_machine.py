"""Tests for trade lifecycle state machine."""
import pytest
from src.core.state_machine import TradeLifecycle, TradeState, TradeManager


class TestTradeLifecycle:
    def test_initial_state_is_signal(self):
        t = TradeLifecycle("T-001", "AAPL", "buy")
        assert t.state == TradeState.SIGNAL

    def test_happy_path_transitions(self):
        t = TradeLifecycle("T-002", "AAPL", "buy")
        assert t.transition(TradeState.PENDING_RISK) is True
        assert t.transition(TradeState.RISK_APPROVED) is True
        assert t.transition(TradeState.SUBMITTED) is True
        assert t.transition(TradeState.ACCEPTED) is True
        assert t.transition(TradeState.FILLED) is True
        assert t.transition(TradeState.ACTIVE) is True
        assert t.transition(TradeState.CLOSED) is True
        assert t.is_terminal is True

    def test_invalid_transition_rejected(self):
        t = TradeLifecycle("T-003", "AAPL", "buy")
        # Cannot go directly from SIGNAL to CLOSED
        assert t.transition(TradeState.CLOSED) is False
        assert t.state == TradeState.SIGNAL  # Unchanged

    def test_risk_rejected_is_terminal(self):
        t = TradeLifecycle("T-004", "AAPL", "buy")
        t.transition(TradeState.PENDING_RISK)
        t.transition(TradeState.RISK_REJECTED)
        assert t.is_terminal is True
        assert t.valid_transitions == []

    def test_error_reachable_from_non_terminal(self):
        t = TradeLifecycle("T-005", "AAPL", "buy")
        t.transition(TradeState.PENDING_RISK)
        assert t.transition(TradeState.ERROR) is True

    def test_error_not_reachable_from_terminal(self):
        t = TradeLifecycle("T-006", "AAPL", "buy")
        t.transition(TradeState.PENDING_RISK)
        t.transition(TradeState.RISK_REJECTED)
        assert t.transition(TradeState.ERROR) is False

    def test_history_tracked(self):
        t = TradeLifecycle("T-007", "MSFT", "sell")
        t.transition(TradeState.PENDING_RISK, "risk check")
        t.transition(TradeState.RISK_APPROVED, "all layers passed")
        assert len(t.history) == 3  # SIGNAL + 2 transitions
        assert t.history[1][0] == TradeState.PENDING_RISK
        assert t.history[1][2] == "risk check"


class TestTradeManager:
    def test_create_trade(self):
        mgr = TradeManager()
        trade = mgr.create_trade("AAPL", "buy")
        assert trade.trade_id.startswith("T-")
        assert mgr.get_trade(trade.trade_id) is not None

    def test_get_active_trades(self):
        mgr = TradeManager()
        trade = mgr.create_trade("AAPL", "buy")
        trade.transition(TradeState.PENDING_RISK)
        trade.transition(TradeState.RISK_APPROVED)
        trade.transition(TradeState.SUBMITTED)
        trade.transition(TradeState.ACCEPTED)
        trade.transition(TradeState.FILLED)
        trade.transition(TradeState.ACTIVE)
        active = mgr.get_active_trades()
        assert len(active) >= 1

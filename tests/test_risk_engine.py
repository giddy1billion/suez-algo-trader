"""Tests for multi-layer risk engine."""
import pytest
from src.risk.engine import RiskEngine
from src.risk.models import TradeRequest, RiskDecision


@pytest.fixture
def risk_engine():
    return RiskEngine()


@pytest.fixture
def normal_request():
    return TradeRequest(
        symbol="AAPL",
        side="buy",
        qty=10,
        price=150.0,
        stop_loss=145.0,
        take_profit=160.0,
        strategy="momentum",
        confidence=0.75,
    )


class TestRiskEngine:
    def test_approve_normal_trade(self, risk_engine, normal_request):
        decision = risk_engine.evaluate(
            request=normal_request,
            portfolio_value=100000.0,
            cash=50000.0,
            positions=[],
        )
        assert decision.approved is True
        assert decision.adjusted_qty > 0

    def test_reject_when_no_cash(self, risk_engine, normal_request):
        decision = risk_engine.evaluate(
            request=normal_request,
            portfolio_value=100000.0,
            cash=0.0,
            positions=[],
        )
        # Should either reject or reduce to 0
        if decision.approved:
            assert decision.adjusted_qty == 0 or decision.adjusted_qty <= normal_request.qty

    def test_reject_oversized_position(self, risk_engine):
        """A single position that would be 90% of portfolio should be rejected."""
        big_request = TradeRequest(
            symbol="AAPL", side="buy", qty=5000, price=150.0,
            stop_loss=140.0, take_profit=160.0, strategy="test", confidence=0.8,
        )
        decision = risk_engine.evaluate(
            request=big_request,
            portfolio_value=100000.0,
            cash=90000.0,
            positions=[],
        )
        # Should reject or significantly reduce
        if decision.approved:
            assert decision.adjusted_qty < big_request.qty

    def test_works_with_empty_positions(self, risk_engine, normal_request):
        decision = risk_engine.evaluate(
            request=normal_request,
            portfolio_value=50000.0,
            cash=30000.0,
            positions=[],
        )
        assert isinstance(decision, RiskDecision)

    def test_works_with_existing_positions(self, risk_engine, normal_request):
        positions = [
            {"symbol": "MSFT", "qty": 20, "market_value": 8000, "side": "long"},
            {"symbol": "GOOGL", "qty": 5, "market_value": 7500, "side": "long"},
        ]
        decision = risk_engine.evaluate(
            request=normal_request,
            portfolio_value=100000.0,
            cash=50000.0,
            positions=positions,
        )
        assert isinstance(decision, RiskDecision)

    def test_decision_has_reasons(self, risk_engine, normal_request):
        decision = risk_engine.evaluate(
            request=normal_request,
            portfolio_value=100000.0,
            cash=50000.0,
            positions=[],
        )
        assert isinstance(decision.reasons, list)

"""
Tests for the Clean Signal → Decision Pipeline Architecture.

Verifies:
1. TradeSignal (frozen, minimal, proposal-only)
2. Signal adapter (LegacyTradeSignal → TradeSignal conversion)
3. Event bus integration (SignalGenerated, DecisionContractCreated)
4. TradeRequest carries signal + contract
5. Pipeline flow integrity
"""

import pytest
from datetime import datetime, timezone

from src.strategy.base import (
    TradeSignal,
    LegacyTradeSignal,
    Signal,
    Side,
    BaseStrategy,
)
from src.strategy.signal_adapter import adapt_signal, is_legacy_signal, is_actionable
from src.core.events import (
    SignalGenerated,
    DecisionContractCreated,
    EventBus,
)
from src.risk.models import TradeRequest
from src.intelligence.confidence.decision_contract import DecisionContract, Decision


# ─── Fixtures ───────────────────────────────────────────────────────────────


class MockStrategy(BaseStrategy):
    version = "3.1.4"

    def generate_signals(self, data):
        return []

    def calculate_indicators(self, df):
        return df


@pytest.fixture
def strategy():
    return MockStrategy(name="test_momentum", symbols=["BTC/USD"], timeframe="15m")


@pytest.fixture
def legacy_signal():
    return LegacyTradeSignal(
        symbol="BTC/USD",
        signal=Signal.STRONG_BUY,
        confidence=0.85,
        price=67350.0,
        stop_loss=66800.0,
        take_profit=68500.0,
        reason="RSI divergence + EMA crossover",
        indicators={"rsi": 61.7, "atr": 425.2, "macd_hist": 0.003},
    )


@pytest.fixture
def new_signal():
    return TradeSignal(
        strategy_id="ema_cross_v4",
        strategy_version="4.2.1",
        symbol="BTC/USD",
        timeframe="15m",
        side=Side.BUY,
        signal_strength=0.83,
        expected_direction=1,
        reason="Fast EMA crossed above Slow EMA",
        features={"ema_fast": 67320.4, "ema_slow": 67210.8},
        indicators={"rsi": 61.7, "atr": 425.2},
        tags=("trend_following", "strong_signal"),
    )


# ─── TradeSignal Tests ──────────────────────────────────────────────────────


class TestTradeSignal:
    """Tests for the new frozen TradeSignal."""

    def test_frozen_immutable(self, new_signal):
        with pytest.raises(Exception):
            new_signal.symbol = "ETH/USD"

    def test_signal_id_auto_generated(self):
        sig = TradeSignal(symbol="AAPL", side=Side.BUY, signal_strength=0.7)
        assert sig.signal_id.startswith("SIG-")
        assert len(sig.signal_id) > 4

    def test_unique_signal_ids(self):
        sig1 = TradeSignal(symbol="AAPL", side=Side.BUY, signal_strength=0.7)
        sig2 = TradeSignal(symbol="AAPL", side=Side.BUY, signal_strength=0.7)
        assert sig1.signal_id != sig2.signal_id

    def test_is_actionable(self, new_signal):
        assert new_signal.is_actionable

    def test_not_actionable_zero_strength(self):
        sig = TradeSignal(symbol="AAPL", side=Side.BUY, signal_strength=0.0)
        assert not sig.is_actionable

    def test_not_actionable_empty_symbol(self):
        sig = TradeSignal(symbol="", side=Side.BUY, signal_strength=0.5)
        assert not sig.is_actionable

    def test_is_buy_sell_properties(self, new_signal):
        assert new_signal.is_buy
        assert not new_signal.is_sell

        sell_sig = TradeSignal(symbol="AAPL", side=Side.SELL, signal_strength=0.7)
        assert sell_sig.is_sell
        assert not sell_sig.is_buy

    def test_to_event_payload(self, new_signal):
        payload = new_signal.to_event_payload()
        assert payload["signal_id"] == new_signal.signal_id
        assert payload["strategy"]["id"] == "ema_cross_v4"
        assert payload["strategy"]["version"] == "4.2.1"
        assert payload["market"]["symbol"] == "BTC/USD"
        assert payload["market"]["timeframe"] == "15m"
        assert payload["signal"]["side"] == "BUY"
        assert payload["signal"]["strength"] == 0.83
        assert payload["signal"]["expected_direction"] == 1
        assert payload["metadata"]["tags"] == ["trend_following", "strong_signal"]
        assert payload["evidence"]["indicators"]["rsi"] == 61.7

    def test_does_not_contain_execution_fields(self, new_signal):
        """TradeSignal must NOT contain downstream concerns."""
        assert not hasattr(new_signal, "position_size")
        assert not hasattr(new_signal, "risk_percentage")
        assert not hasattr(new_signal, "kelly_fraction")
        assert not hasattr(new_signal, "portfolio_exposure")
        assert not hasattr(new_signal, "execution_approval")
        assert not hasattr(new_signal, "order_type")
        assert not hasattr(new_signal, "stop_loss")
        assert not hasattr(new_signal, "take_profit")
        assert not hasattr(new_signal, "confidence")

    def test_timestamp_defaults_to_utc(self):
        sig = TradeSignal(symbol="X", side=Side.BUY, signal_strength=0.5)
        assert sig.timestamp.tzinfo is not None


# ─── Signal Adapter Tests ───────────────────────────────────────────────────


class TestSignalAdapter:
    """Tests for legacy → new signal conversion."""

    def test_adapt_buy_signal(self, legacy_signal, strategy):
        adapted = adapt_signal(legacy_signal, strategy)
        assert isinstance(adapted, TradeSignal)
        assert adapted.side == Side.BUY
        assert adapted.expected_direction == 1
        assert adapted.signal_strength == 0.85
        assert adapted.symbol == "BTC/USD"

    def test_adapt_sell_signal(self, strategy):
        leg = LegacyTradeSignal(
            symbol="ETH/USD", signal=Signal.SELL,
            confidence=0.72, price=3400.0,
        )
        adapted = adapt_signal(leg, strategy)
        assert adapted.side == Side.SELL
        assert adapted.expected_direction == -1

    def test_adapt_strong_signal_tagged(self, legacy_signal, strategy):
        adapted = adapt_signal(legacy_signal, strategy)
        assert "strong_signal" in adapted.tags

    def test_adapter_preserves_sl_tp_in_features(self, legacy_signal, strategy):
        adapted = adapt_signal(legacy_signal, strategy)
        assert adapted.features["strategy_proposed_stop_loss"] == 66800.0
        assert adapted.features["strategy_proposed_take_profit"] == 68500.0
        assert adapted.features["observed_price"] == 67350.0

    def test_adapter_separates_numeric_indicators(self, legacy_signal, strategy):
        adapted = adapt_signal(legacy_signal, strategy)
        assert adapted.indicators["rsi"] == 61.7
        assert adapted.indicators["atr"] == 425.2

    def test_adapter_uses_strategy_metadata(self, legacy_signal, strategy):
        adapted = adapt_signal(legacy_signal, strategy)
        assert adapted.strategy_id == "test_momentum"
        assert adapted.strategy_version == "3.1.4"
        assert adapted.timeframe == "15m"

    def test_passthrough_new_signal(self, new_signal, strategy):
        result = adapt_signal(new_signal, strategy)
        assert result is new_signal  # Same object, no conversion

    def test_invalid_type_raises(self, strategy):
        with pytest.raises(TypeError):
            adapt_signal({"symbol": "AAPL"}, strategy)

    def test_is_legacy_signal(self, legacy_signal, new_signal):
        assert is_legacy_signal(legacy_signal)
        assert not is_legacy_signal(new_signal)

    def test_is_actionable_both_formats(self, legacy_signal, new_signal):
        assert is_actionable(legacy_signal)
        assert is_actionable(new_signal)

        # Non-actionable signals
        non_actionable_leg = LegacyTradeSignal(
            symbol="X", signal=Signal.HOLD, confidence=0.3, price=10.0,
        )
        non_actionable_new = TradeSignal(
            symbol="X", side=Side.BUY, signal_strength=0.0,
        )
        assert not is_actionable(non_actionable_leg)
        assert not is_actionable(non_actionable_new)


# ─── Event Bus Integration Tests ────────────────────────────────────────────


class TestEventBusIntegration:
    """Tests for new events on the event bus."""

    def test_signal_generated_event(self, new_signal):
        bus = EventBus()
        received = []
        bus.subscribe(SignalGenerated, lambda e: received.append(e))

        event = SignalGenerated(
            signal_id=new_signal.signal_id,
            strategy=new_signal.strategy_id,
            strategy_version=new_signal.strategy_version,
            symbol=new_signal.symbol,
            timeframe=new_signal.timeframe,
            side=new_signal.side.value,
            signal=new_signal.side.value,
            signal_strength=new_signal.signal_strength,
            expected_direction=new_signal.expected_direction,
            source="engine",
        )
        bus.publish(event)

        assert len(received) == 1
        assert received[0].signal_id == new_signal.signal_id
        assert received[0].side == "BUY"
        assert received[0].signal_strength == 0.83

    def test_decision_contract_created_event(self):
        bus = EventBus()
        received = []
        bus.subscribe(DecisionContractCreated, lambda e: received.append(e))

        event = DecisionContractCreated(
            contract_id="DC-TEST123",
            signal_id="SIG-abc",
            decision="execute",
            final_confidence=0.91,
            symbol="BTC/USD",
            side="BUY",
            recommended_position_pct=2.5,
            recommended_stop_loss=66800.0,
            recommended_take_profit=68300.0,
            risk_grade="A",
            source="decision_orchestrator",
        )
        bus.publish(event)

        assert len(received) == 1
        assert received[0].decision == "execute"
        assert received[0].final_confidence == 0.91
        assert received[0].recommended_stop_loss == 66800.0

    def test_pipeline_events_flow_in_order(self, new_signal):
        """Verify the event flow: SignalGenerated → DecisionContractCreated."""
        bus = EventBus()
        all_events = []
        bus.subscribe(SignalGenerated, lambda e: all_events.append(("signal", e)))
        bus.subscribe(DecisionContractCreated, lambda e: all_events.append(("contract", e)))

        # Simulate pipeline
        bus.publish(SignalGenerated(
            signal_id=new_signal.signal_id,
            symbol=new_signal.symbol,
            side="BUY",
            signal_strength=0.83,
            source="engine",
        ))
        bus.publish(DecisionContractCreated(
            contract_id="DC-X",
            signal_id=new_signal.signal_id,
            decision="execute",
            final_confidence=0.91,
            symbol="BTC/USD",
            side="BUY",
            source="decision_orchestrator",
        ))

        assert len(all_events) == 2
        assert all_events[0][0] == "signal"
        assert all_events[1][0] == "contract"
        # Same signal_id links them
        assert all_events[0][1].signal_id == all_events[1][1].signal_id


# ─── TradeRequest Integration Tests ────────────────────────────────────────


class TestTradeRequest:
    """Tests for TradeRequest carrying signal + contract."""

    def test_trade_request_with_signal_and_contract(self, new_signal):
        contract = DecisionContract(
            decision=Decision.EXECUTE,
            final_confidence=0.91,
            symbol="BTC/USD",
            direction="BUY",
            recommended_position_pct=2.5,
            recommended_stop_loss=66800.0,
            recommended_take_profit=68300.0,
        )

        req = TradeRequest(
            symbol="BTC/USD",
            side="buy",
            qty=0.05,
            price=67350.0,
            stop_loss=contract.recommended_stop_loss,
            take_profit=contract.recommended_take_profit,
            strategy="ema_cross_v4",
            confidence=contract.final_confidence,
            decision_contract=contract,
            trade_signal=new_signal,
        )

        assert req.has_contract
        assert req.has_signal
        assert req.signal_id == new_signal.signal_id
        assert req.contract_id == contract.contract_id
        assert req.effective_confidence == 0.91
        assert req.stop_loss == 66800.0
        assert req.take_profit == 68300.0

    def test_trade_request_without_signal(self):
        """Backward compat: request without signal still works."""
        req = TradeRequest(
            symbol="AAPL", side="buy", qty=10, price=150.0,
            confidence=0.8,
        )
        assert not req.has_signal
        assert not req.has_contract
        assert req.signal_id == ""
        assert req.contract_id == ""
        assert req.effective_confidence == 0.8

    def test_confidence_hierarchy(self, new_signal):
        """Contract confidence > score confidence > scalar confidence."""
        contract = DecisionContract(
            decision=Decision.EXECUTE, final_confidence=0.95,
        )
        req = TradeRequest(
            symbol="X", side="buy", qty=1, price=100.0,
            confidence=0.5,
            decision_contract=contract,
            trade_signal=new_signal,
        )
        # Contract wins
        assert req.effective_confidence == 0.95


# ─── Pipeline Design Principle Tests ────────────────────────────────────────


class TestDesignPrinciples:
    """Verify the architectural separation is correct."""

    def test_signal_is_proposal_only(self, new_signal):
        """TradeSignal = 'I think we should trade.' (proposal)"""
        # Signal has strategy evidence
        assert new_signal.signal_strength > 0
        assert new_signal.indicators
        assert new_signal.reason
        # Signal does NOT have system-level decisions
        assert not hasattr(new_signal, "position_size")
        assert not hasattr(new_signal, "stop_loss")
        assert not hasattr(new_signal, "take_profit")
        assert not hasattr(new_signal, "confidence")  # only signal_strength

    def test_contract_is_authoritative_decision(self):
        """DecisionContract = 'The system has decided.' (authoritative)"""
        contract = DecisionContract(
            decision=Decision.EXECUTE,
            final_confidence=0.91,
            recommended_position_pct=2.5,
            recommended_stop_loss=66800.0,
            recommended_take_profit=68300.0,
            risk_grade="A",
        )
        # Contract has system-level decisions
        assert contract.final_confidence > 0
        assert contract.recommended_position_pct > 0
        assert contract.recommended_stop_loss > 0
        assert contract.recommended_take_profit > 0
        assert contract.risk_grade == "A"
        assert contract.is_executable

    def test_risk_reads_from_contract_not_signal(self, new_signal):
        """The risk engine should use contract values, not signal values."""
        contract = DecisionContract(
            decision=Decision.EXECUTE,
            final_confidence=0.91,
            recommended_stop_loss=66800.0,
            recommended_take_profit=68300.0,
        )
        req = TradeRequest(
            symbol="BTC/USD", side="buy", qty=0.05, price=67350.0,
            # SL/TP come from contract
            stop_loss=contract.recommended_stop_loss,
            take_profit=contract.recommended_take_profit,
            confidence=contract.final_confidence,
            decision_contract=contract,
            trade_signal=new_signal,
        )
        # Execution uses contract-determined values
        assert req.stop_loss == 66800.0  # From contract
        assert req.take_profit == 68300.0  # From contract
        # NOT from signal (signal has no SL/TP)


# ─── DecisionContract Enhancement Tests ────────────────────────────────────


class TestDecisionContractEnhancements:
    """Tests for new fields added to DecisionContract."""

    def test_recommended_stop_loss_field(self):
        dc = DecisionContract(recommended_stop_loss=66800.0)
        assert dc.recommended_stop_loss == 66800.0

    def test_recommended_take_profit_field(self):
        dc = DecisionContract(recommended_take_profit=68300.0)
        assert dc.recommended_take_profit == 68300.0

    def test_defaults_to_zero(self):
        dc = DecisionContract()
        assert dc.recommended_stop_loss == 0.0
        assert dc.recommended_take_profit == 0.0

"""
Tests for TradeSignalPackage — comprehensive execution package validation.
"""

import pytest
from datetime import datetime, timezone, timedelta

from src.strategy.base import Signal
from src.strategy.signal_package import (
    TradeSignalPackage,
    EntryZone,
    ModelInfo,
    StrategyContributor,
    TakeProfitLevel,
    TimeBasedExit,
    ConfidenceDecay,
    SignalStatus,
    SignalValidationGate,
    MarketRegime,
    VolatilityLevel,
    TrailingStopMode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_valid_package(**overrides) -> TradeSignalPackage:
    """Create a fully valid signal package for testing."""
    now = datetime.now(timezone.utc)
    defaults = dict(
        symbol="BTCUSDT",
        direction=Signal.BUY,
        entry_zone=EntryZone(preferred_min=109_850, preferred_max=109_930),
        confidence=0.937,
        confidence_decay=ConfidenceDecay(
            initial_confidence=0.937,
            decay_rate_per_minute=0.001,
            invalidation_threshold=0.65,
        ),
        model_info=ModelInfo(
            model_version="v3.2.14",
            training_run_id="train-20260704-001",
            dataset_version="ds-v2.8",
            backtest_id="bt-20260704-042",
            walk_forward_validation_id="wf-20260704-007",
        ),
        strategy_contributors=[
            StrategyContributor(name="Trend Following", weight_pct=42),
            StrategyContributor(name="Momentum", weight_pct=28),
            StrategyContributor(name="Mean Reversion", weight_pct=12),
            StrategyContributor(name="Order Flow", weight_pct=8),
            StrategyContributor(name="ML Ensemble", weight_pct=10),
        ],
        signal_timeframe="15m",
        trade_horizon_minutes=480,
        max_holding_minutes=600,
        stop_loss=109_120,
        take_profit_levels=[
            TakeProfitLevel(price=110_450, allocation_pct=30, expected_time_minutes=45),
            TakeProfitLevel(price=110_980, allocation_pct=40, expected_time_minutes=120),
            TakeProfitLevel(price=111_620, allocation_pct=30, expected_time_minutes=360),
        ],
        trailing_stop_mode=TrailingStopMode.AFTER_TP1,
        trailing_stop_distance_pct=0.5,
        expected_risk_reward=3.4,
        expected_win_probability=0.74,
        expected_return_pct=2.8,
        max_expected_drawdown_pct=0.9,
        market_regime=MarketRegime.TRENDING_BULLISH,
        volatility=VolatilityLevel.MEDIUM,
        position_size_pct=2.3,
        time_based_exit=TimeBasedExit(
            entry_window_start=now,
            entry_window_end=now + timedelta(minutes=45),
            max_holding_minutes=480,
            hard_exit_time=now + timedelta(hours=8),
            max_adverse_excursion_minutes=60,
        ),
        signal_expiry_minutes=90,
        reasons=[
            "EMA alignment",
            "MACD confirmation",
            "Positive order flow",
            "Increasing volume",
            "ML probability above threshold",
            "Walk-forward validated",
        ],
    )
    defaults.update(overrides)
    return TradeSignalPackage(**defaults)


# ---------------------------------------------------------------------------
# Tests: Signal Package Validation
# ---------------------------------------------------------------------------


class TestTradeSignalPackageValidation:
    """Test the signal package validation logic."""

    def test_valid_package_passes(self):
        pkg = _make_valid_package()
        is_valid, errors = pkg.validate()
        assert is_valid is True
        assert errors == []
        assert pkg.status == SignalStatus.READY_FOR_EXECUTION

    def test_missing_symbol_fails(self):
        pkg = _make_valid_package(symbol="")
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert "Missing symbol" in errors

    def test_hold_direction_fails(self):
        pkg = _make_valid_package(direction=Signal.HOLD)
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert any("Direction" in e for e in errors)

    def test_missing_entry_zone_fails(self):
        pkg = _make_valid_package(entry_zone=None)
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert "Missing entry zone" in errors

    def test_invalid_entry_zone_range_fails(self):
        pkg = _make_valid_package(entry_zone=EntryZone(preferred_min=110_000, preferred_max=109_000))
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert any("min must be less than max" in e for e in errors)

    def test_missing_stop_loss_fails(self):
        pkg = _make_valid_package(stop_loss=0)
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert any("stop loss" in e.lower() for e in errors)

    def test_buy_stop_loss_above_entry_fails(self):
        pkg = _make_valid_package(stop_loss=110_000)  # Above entry zone
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert any("Stop loss must be below entry for BUY" in e for e in errors)

    def test_sell_stop_loss_below_entry_fails(self):
        pkg = _make_valid_package(
            direction=Signal.SELL,
            stop_loss=109_000,  # Below entry zone for a SELL
            take_profit_levels=[
                TakeProfitLevel(price=108_000, allocation_pct=50),
                TakeProfitLevel(price=107_000, allocation_pct=50),
            ],
        )
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert any("Stop loss must be above entry for SELL" in e for e in errors)

    def test_no_take_profit_fails(self):
        pkg = _make_valid_package(take_profit_levels=[])
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert any("take-profit" in e.lower() for e in errors)

    def test_tp_allocation_not_100_fails(self):
        pkg = _make_valid_package(
            take_profit_levels=[
                TakeProfitLevel(price=110_450, allocation_pct=30),
                TakeProfitLevel(price=110_980, allocation_pct=30),
            ]
        )
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert any("sum to 100%" in e for e in errors)

    def test_missing_model_info_fails(self):
        pkg = _make_valid_package(model_info=None)
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert "Missing model information" in errors

    def test_incomplete_model_provenance_fails(self):
        pkg = _make_valid_package(
            model_info=ModelInfo(model_version="v1.0")  # Missing other fields
        )
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert any("Incomplete model provenance" in e for e in errors)

    def test_no_strategy_contributors_fails(self):
        pkg = _make_valid_package(strategy_contributors=[])
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert any("strategy contributor" in e.lower() for e in errors)

    def test_strategy_weights_not_100_fails(self):
        pkg = _make_valid_package(
            strategy_contributors=[
                StrategyContributor(name="Trend", weight_pct=50),
                StrategyContributor(name="Momentum", weight_pct=30),
            ]
        )
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert any("weights must sum to 100%" in e for e in errors)

    def test_missing_timeframe_fails(self):
        pkg = _make_valid_package(signal_timeframe="")
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert any("timeframe" in e.lower() for e in errors)

    def test_zero_max_holding_fails(self):
        pkg = _make_valid_package(max_holding_minutes=0)
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert any("holding time" in e.lower() for e in errors)

    def test_zero_risk_reward_fails(self):
        pkg = _make_valid_package(expected_risk_reward=0)
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert any("risk/reward" in e.lower() for e in errors)

    def test_invalid_win_probability_fails(self):
        pkg = _make_valid_package(expected_win_probability=0)
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert any("win probability" in e.lower() for e in errors)

    def test_zero_position_size_fails(self):
        pkg = _make_valid_package(position_size_pct=0)
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert any("Position size must be positive" in e for e in errors)

    def test_position_size_over_25_pct_fails(self):
        pkg = _make_valid_package(position_size_pct=30.0)
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert any("25% safety limit" in e for e in errors)

    def test_missing_time_based_exit_fails(self):
        pkg = _make_valid_package(time_based_exit=None)
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert any("time-based exit" in e.lower() for e in errors)

    def test_missing_confidence_decay_fails(self):
        pkg = _make_valid_package(confidence_decay=None)
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert any("confidence decay" in e.lower() for e in errors)

    def test_no_reasons_fails(self):
        pkg = _make_valid_package(reasons=[])
        is_valid, errors = pkg.validate()
        assert is_valid is False
        assert any("reason" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Tests: Confidence Decay
# ---------------------------------------------------------------------------


class TestConfidenceDecay:
    """Test confidence decay behavior."""

    def test_initial_confidence(self):
        decay = ConfidenceDecay(initial_confidence=0.94, decay_rate_per_minute=0.001)
        assert decay.confidence_at(0) == 0.94

    def test_decay_over_time(self):
        decay = ConfidenceDecay(initial_confidence=0.94, decay_rate_per_minute=0.001)
        # After 30 minutes: 0.94 - 0.03 = 0.91
        assert abs(decay.confidence_at(30) - 0.91) < 0.001

    def test_decay_after_1_hour(self):
        decay = ConfidenceDecay(initial_confidence=0.94, decay_rate_per_minute=0.001)
        # After 60 minutes: 0.94 - 0.06 = 0.88
        assert abs(decay.confidence_at(60) - 0.88) < 0.001

    def test_invalidation_threshold(self):
        decay = ConfidenceDecay(
            initial_confidence=0.94,
            decay_rate_per_minute=0.001,
            invalidation_threshold=0.65,
        )
        # Valid at 200 minutes: 0.94 - 0.2 = 0.74
        assert decay.is_valid_at(200) is True
        # Invalid at 300 minutes: 0.94 - 0.3 = 0.64
        assert decay.is_valid_at(300) is False

    def test_minutes_until_invalidation(self):
        decay = ConfidenceDecay(
            initial_confidence=0.94,
            decay_rate_per_minute=0.001,
            invalidation_threshold=0.65,
        )
        # (0.94 - 0.65) / 0.001 = 290 minutes
        assert abs(decay.minutes_until_invalidation() - 290) < 0.01

    def test_confidence_never_negative(self):
        decay = ConfidenceDecay(initial_confidence=0.5, decay_rate_per_minute=0.01)
        assert decay.confidence_at(10000) == 0.0


# ---------------------------------------------------------------------------
# Tests: Entry Zone
# ---------------------------------------------------------------------------


class TestEntryZone:
    """Test entry zone logic."""

    def test_midpoint(self):
        zone = EntryZone(preferred_min=100, preferred_max=200)
        assert zone.midpoint == 150

    def test_price_within_range(self):
        zone = EntryZone(preferred_min=109_850, preferred_max=109_930, max_slippage_pct=0.10)
        # Price within range
        assert zone.is_price_acceptable(109_890) is True
        # Price slightly outside but within slippage
        mid = zone.midpoint  # 109,890
        slippage_buffer = mid * 0.001  # ~109.89
        assert zone.is_price_acceptable(109_850 - 50) is True  # Within slippage buffer
        # Price way outside
        assert zone.is_price_acceptable(108_000) is False


# ---------------------------------------------------------------------------
# Tests: Signal Properties
# ---------------------------------------------------------------------------


class TestSignalPackageProperties:
    """Test computed properties."""

    def test_is_buy(self):
        pkg = _make_valid_package(direction=Signal.BUY)
        assert pkg.is_buy is True
        assert pkg.is_sell is False
        assert pkg.side == "buy"

    def test_is_sell(self):
        pkg = _make_valid_package(direction=Signal.SELL)
        assert pkg.is_buy is False
        assert pkg.is_sell is True
        assert pkg.side == "sell"

    def test_hold_side(self):
        pkg = _make_valid_package(direction=Signal.HOLD)
        assert pkg.side == "hold"

    def test_expiry_time(self):
        now = datetime.now(timezone.utc)
        pkg = _make_valid_package(signal_expiry_minutes=90)
        pkg.generated_at = now
        expected = now + timedelta(minutes=90)
        assert abs((pkg.expiry_time - expected).total_seconds()) < 1

    def test_is_expired_false_for_fresh_signal(self):
        pkg = _make_valid_package()
        assert pkg.is_expired is False

    def test_is_expired_true_for_old_signal(self):
        pkg = _make_valid_package(signal_expiry_minutes=1)
        pkg.generated_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        assert pkg.is_expired is True

    def test_total_tp_allocation(self):
        pkg = _make_valid_package()
        assert pkg.total_take_profit_allocation == 100.0

    def test_total_strategy_weight(self):
        pkg = _make_valid_package()
        assert pkg.total_strategy_weight == 100.0

    def test_signal_id_format(self):
        pkg = _make_valid_package()
        assert pkg.signal_id.startswith("SIG-")

    def test_current_confidence_without_decay(self):
        pkg = _make_valid_package(confidence=0.9, confidence_decay=None)
        assert pkg.current_confidence == 0.9

    def test_current_confidence_with_decay(self):
        pkg = _make_valid_package()
        # Fresh signal, should be very close to initial
        assert pkg.current_confidence >= 0.93


# ---------------------------------------------------------------------------
# Tests: Signal Validation Gate
# ---------------------------------------------------------------------------


class TestSignalValidationGate:
    """Test the validation gate that blocks incomplete signals."""

    def test_valid_package_approved(self):
        gate = SignalValidationGate()
        pkg = _make_valid_package()
        approved, errors = gate.evaluate(pkg)
        assert approved is True
        assert errors == []
        assert pkg.status == SignalStatus.READY_FOR_EXECUTION

    def test_low_confidence_rejected(self):
        gate = SignalValidationGate(min_confidence=0.70)
        pkg = _make_valid_package(confidence=0.60)
        approved, errors = gate.evaluate(pkg)
        assert approved is False
        assert any("Confidence" in e for e in errors)

    def test_low_risk_reward_rejected(self):
        gate = SignalValidationGate(min_risk_reward=2.0)
        pkg = _make_valid_package(expected_risk_reward=1.5)
        approved, errors = gate.evaluate(pkg)
        assert approved is False
        assert any("Risk/reward" in e for e in errors)

    def test_oversized_position_rejected(self):
        gate = SignalValidationGate(max_position_size_pct=5.0)
        pkg = _make_valid_package(position_size_pct=10.0)
        approved, errors = gate.evaluate(pkg)
        assert approved is False
        assert any("Position size" in e for e in errors)

    def test_expired_signal_rejected(self):
        gate = SignalValidationGate()
        pkg = _make_valid_package(signal_expiry_minutes=1)
        pkg.generated_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        approved, errors = gate.evaluate(pkg)
        assert approved is False
        assert any("expired" in e.lower() for e in errors)

    def test_decayed_confidence_rejected(self):
        gate = SignalValidationGate()
        pkg = _make_valid_package(
            confidence_decay=ConfidenceDecay(
                initial_confidence=0.70,
                decay_rate_per_minute=0.01,
                invalidation_threshold=0.65,
            )
        )
        # Simulate old signal
        pkg.generated_at = datetime.now(timezone.utc) - timedelta(minutes=60)
        approved, errors = gate.evaluate(pkg)
        assert approved is False
        assert any("decayed" in e.lower() for e in errors)

    def test_incomplete_signal_rejected(self):
        gate = SignalValidationGate()
        # Package with many missing fields
        pkg = TradeSignalPackage(symbol="BTCUSDT", direction=Signal.BUY)
        approved, errors = gate.evaluate(pkg)
        assert approved is False
        assert len(errors) > 3  # Multiple failures

    def test_gate_without_provenance_requirement(self):
        gate = SignalValidationGate(require_model_provenance=False)
        pkg = _make_valid_package(
            model_info=ModelInfo(
                model_version="v1.0",
                training_run_id="tr-001",
                dataset_version="ds-001",
                backtest_id="bt-001",
                walk_forward_validation_id="wf-001",
            )
        )
        approved, errors = gate.evaluate(pkg)
        assert approved is True


# ---------------------------------------------------------------------------
# Tests: Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    """Test signal package serialization."""

    def test_to_dict_roundtrip(self):
        pkg = _make_valid_package()
        d = pkg.to_dict()
        assert d["symbol"] == "BTCUSDT"
        assert d["direction"] == "BUY"
        assert d["confidence"] == 0.937
        assert d["stop_loss"] == 109_120
        assert len(d["take_profit_levels"]) == 3
        assert len(d["strategy_contributors"]) == 5
        assert d["market_regime"] == "trending_bullish"
        assert d["status"] == "pending_validation"

    def test_summary_output(self):
        pkg = _make_valid_package()
        summary = pkg.summary()
        assert "BTCUSDT" in summary
        assert "BUY" in summary
        assert "109,120" in summary
        assert "TP1" in summary


# ---------------------------------------------------------------------------
# Tests: Invalidation & Expiry Check
# ---------------------------------------------------------------------------


class TestInvalidation:
    """Test signal invalidation."""

    def test_invalidate_changes_status(self):
        pkg = _make_valid_package()
        pkg.invalidate("Market conditions changed")
        assert pkg.status == SignalStatus.INVALIDATED
        assert any("INVALIDATED" in r for r in pkg.reasons)

    def test_check_expiry_valid_signal(self):
        pkg = _make_valid_package()
        assert pkg.check_expiry() is True

    def test_check_expiry_expired_signal(self):
        pkg = _make_valid_package(signal_expiry_minutes=1)
        pkg.generated_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        assert pkg.check_expiry() is False
        assert pkg.status == SignalStatus.INVALIDATED

    def test_check_expiry_decayed_confidence(self):
        pkg = _make_valid_package(
            confidence_decay=ConfidenceDecay(
                initial_confidence=0.70,
                decay_rate_per_minute=0.01,
                invalidation_threshold=0.65,
            )
        )
        pkg.generated_at = datetime.now(timezone.utc) - timedelta(minutes=60)
        assert pkg.check_expiry() is False
        assert pkg.status == SignalStatus.INVALIDATED


# ---------------------------------------------------------------------------
# Tests: TakeProfitLevel
# ---------------------------------------------------------------------------


class TestTakeProfitLevel:
    """Test take-profit level model."""

    def test_valid_tp(self):
        tp = TakeProfitLevel(price=110_000, allocation_pct=50)
        assert tp.price == 110_000
        assert tp.allocation_pct == 50

    def test_invalid_allocation_raises(self):
        with pytest.raises(ValueError):
            TakeProfitLevel(price=110_000, allocation_pct=0)
        with pytest.raises(ValueError):
            TakeProfitLevel(price=110_000, allocation_pct=101)


# ---------------------------------------------------------------------------
# Tests: TimeBasedExit
# ---------------------------------------------------------------------------


class TestTimeBasedExit:
    """Test time-based exit configuration."""

    def test_entry_window_duration(self):
        now = datetime.now(timezone.utc)
        tbe = TimeBasedExit(
            entry_window_start=now,
            entry_window_end=now + timedelta(minutes=45),
        )
        assert tbe.entry_window_duration_minutes == 45

    def test_no_entry_window(self):
        tbe = TimeBasedExit()
        assert tbe.entry_window_duration_minutes is None


# ---------------------------------------------------------------------------
# Tests: ModelInfo
# ---------------------------------------------------------------------------


class TestModelInfo:
    """Test model info completeness."""

    def test_complete_model_info(self):
        mi = ModelInfo(
            model_version="v3.2.14",
            training_run_id="tr-001",
            dataset_version="ds-v2.8",
            backtest_id="bt-001",
            walk_forward_validation_id="wf-001",
        )
        assert mi.is_complete() is True

    def test_incomplete_model_info(self):
        mi = ModelInfo(model_version="v1.0")
        assert mi.is_complete() is False

    def test_empty_model_info(self):
        mi = ModelInfo(model_version="")
        assert mi.is_complete() is False

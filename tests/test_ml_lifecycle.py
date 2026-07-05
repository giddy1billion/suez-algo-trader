"""Tests for ML Lifecycle Hardening — Phases 1-6.

Covers:
- Phase 1: Prediction provenance & NO_SIGNAL sentinel
- Phase 2: Model promotion gate with configurable thresholds
- Phase 3: Circuit breaker trip/reset
- Phase 4: Audit trail linking
- Phase 5: Self-healing (retry, stale detection, rollback)
- Phase 6: New event types
"""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.events import (
    CircuitBreakerReset,
    CircuitBreakerTripped,
    EventBus,
    ModelAutoRollback,
    ModelRejected,
    PredictionUnavailable,
)
from src.core.circuit_breaker import (
    CircuitBreakerReason,
    CircuitBreakerState,
    TradingCircuitBreaker,
)
from src.core.audit_log import TradeAuditTrail
from src.ml.governance import ModelGovernance, ModelLineage, ModelStatus, ValidationResult
from src.predictions.registry import PredictionRecord, PredictionRegistry
from src.strategy.base import Signal, TradeSignal


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Prediction Provenance & NO_SIGNAL Sentinel
# ──────────────────────────────────────────────────────────────────────────────


class TestPredictionProvenance:
    """Test enriched PredictionRecord with provenance fields."""

    def test_record_has_provenance_fields(self):
        record = PredictionRecord(
            asset="AAPL",
            direction="long",
            confidence=0.85,
            training_timestamp="2026-01-01T00:00:00+00:00",
            dataset_version="ds001",
            feature_set_version="fs001",
            validation_metrics={"cv_accuracy": 0.65},
            probability_distribution=[0.1, 0.2, 0.7],
            feature_importance={"rsi_14": 0.15, "ema_slope": 0.10},
        )
        assert record.training_timestamp == "2026-01-01T00:00:00+00:00"
        assert record.dataset_version == "ds001"
        assert record.feature_set_version == "fs001"
        assert record.validation_metrics == {"cv_accuracy": 0.65}
        assert record.probability_distribution == [0.1, 0.2, 0.7]
        assert record.feature_importance == {"rsi_14": 0.15, "ema_slope": 0.10}

    def test_has_required_provenance_complete(self):
        record = PredictionRecord(
            asset="AAPL",
            direction="long",
            confidence=0.85,
            model_version="v003",
            training_timestamp="2026-01-01T00:00:00+00:00",
            dataset_version="ds001",
            feature_set_version="fs001",
            validation_metrics={"cv_accuracy": 0.65},
        )
        assert record.has_required_provenance() is True

    def test_has_required_provenance_missing_model_version(self):
        record = PredictionRecord(
            asset="AAPL",
            direction="long",
            confidence=0.85,
            training_timestamp="2026-01-01T00:00:00+00:00",
            dataset_version="ds001",
            feature_set_version="fs001",
            validation_metrics={"cv_accuracy": 0.65},
        )
        assert record.has_required_provenance() is False

    def test_has_required_provenance_missing_training_timestamp(self):
        record = PredictionRecord(
            asset="AAPL",
            direction="long",
            model_version="v003",
            dataset_version="ds001",
            feature_set_version="fs001",
            validation_metrics={"cv_accuracy": 0.65},
        )
        assert record.has_required_provenance() is False

    def test_has_required_provenance_missing_validation_metrics(self):
        record = PredictionRecord(
            asset="AAPL",
            direction="long",
            model_version="v003",
            training_timestamp="2026-01-01T00:00:00+00:00",
            dataset_version="ds001",
            feature_set_version="fs001",
        )
        assert record.has_required_provenance() is False

    def test_register_rejects_missing_provenance(self, tmp_path):
        registry = PredictionRegistry(storage_path=str(tmp_path / "pred"))
        result = registry.register(
            asset="AAPL",
            direction="long",
            confidence=0.80,
        )
        assert result is None
        assert registry.active_count == 0

    def test_register_accepts_complete_provenance(self, tmp_path):
        registry = PredictionRegistry(storage_path=str(tmp_path / "pred"))
        result = registry.register(
            asset="AAPL",
            direction="long",
            confidence=0.80,
            model_version="v003",
            training_timestamp="2026-01-01T00:00:00+00:00",
            dataset_version="ds001",
            feature_set_version="fs001",
            validation_metrics={"cv_accuracy": 0.65},
        )
        assert result is not None
        assert result.prediction_id
        assert registry.active_count == 1

    def test_provenance_round_trips_through_dict(self):
        record = PredictionRecord(
            asset="AAPL",
            direction="long",
            confidence=0.85,
            model_version="v003",
            training_timestamp="2026-01-01T00:00:00+00:00",
            dataset_version="ds001",
            feature_set_version="fs001",
            validation_metrics={"cv_accuracy": 0.65},
            probability_distribution=[0.1, 0.2, 0.7],
        )
        d = record.to_dict()
        restored = PredictionRecord.from_dict(d)
        assert restored.training_timestamp == record.training_timestamp
        assert restored.dataset_version == record.dataset_version
        assert restored.validation_metrics == record.validation_metrics


class TestNoSignalSentinel:
    """Test NO_SIGNAL enum value and behavior."""

    def test_no_signal_exists(self):
        assert Signal.NO_SIGNAL.value == -99

    def test_no_signal_not_actionable(self):
        sig = TradeSignal(
            symbol="AAPL",
            signal=Signal.NO_SIGNAL,
            confidence=0.0,
            price=100.0,
            reason="PREDICTION_UNAVAILABLE",
        )
        assert sig.is_actionable is False

    def test_hold_not_actionable(self):
        sig = TradeSignal(
            symbol="AAPL",
            signal=Signal.HOLD,
            confidence=0.8,
            price=100.0,
        )
        assert sig.is_actionable is False

    def test_buy_is_actionable(self):
        sig = TradeSignal(
            symbol="AAPL",
            signal=Signal.BUY,
            confidence=0.8,
            price=100.0,
        )
        assert sig.is_actionable is True

    def test_ml_strategy_no_model_returns_no_signal(self):
        """When no model is loaded, MLStrategy returns NO_SIGNAL per symbol."""
        from src.strategy.ml_strategy import MLStrategy

        strategy = MLStrategy(
            symbols=["AAPL", "MSFT"],
            model_path="/nonexistent/model.joblib",
        )
        # model should be None since path doesn't exist
        assert strategy.model is None

        data = {
            "AAPL": pd.DataFrame({"close": [100.0]}),
            "MSFT": pd.DataFrame({"close": [200.0]}),
        }
        signals = strategy.generate_signals(data)
        assert len(signals) == 2
        for sig in signals:
            assert sig.signal == Signal.NO_SIGNAL
            assert sig.confidence == 0.0
            assert "PREDICTION_UNAVAILABLE" in sig.reason
            assert sig.is_actionable is False


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Model Promotion Gate
# ──────────────────────────────────────────────────────────────────────────────


class TestModelPromotionGate:
    """Test hardened model validation with configurable thresholds."""

    @pytest.fixture
    def governance(self, tmp_path):
        return ModelGovernance(governance_dir=str(tmp_path / "governance"))

    def _record_model(self, governance, version="v001", **overrides):
        """Helper to record a model with configurable metrics."""
        defaults = {
            "version": version,
            "features": ["rsi_14", "ema_slope"],
            "metrics": {
                "cv_accuracy": 0.65,
                "sharpe": 0.8,
                "sharpe_ratio": 0.8,
                "max_drawdown": 0.10,
                "n_trades": 50,
            },
            "hyperparameters": {"n_estimators": 200, "max_depth": 6},
            "seed": 42,
        }
        defaults.update(overrides)

        # Create a mock dataset
        dataset = pd.DataFrame({
            "close": range(100),
            "volume": range(100),
        })

        return governance.record_training(
            version=defaults["version"],
            features=defaults["features"],
            dataset=dataset,
            metrics=defaults["metrics"],
            hyperparameters=defaults["hyperparameters"],
            seed=defaults["seed"],
            walk_forward_results={"sharpe": 0.5, "total_return": 0.1},
            monte_carlo_results={"probability_of_profit": 0.65, "median_return": 0.05, "p5_return": -0.02},
        )

    def test_model_status_enum(self):
        assert ModelStatus.CANDIDATE.value == "candidate"
        assert ModelStatus.REJECTED.value == "rejected"
        assert ModelStatus.DEPLOYED.value == "deployed"

    def test_validation_result_dataclass(self):
        result = ValidationResult(
            is_valid=True,
            checks=[{"check": "cv_accuracy", "passed": True, "detail": "OK"}],
            issues=[],
            model_status="approved",
        )
        d = result.to_dict()
        assert d["is_valid"] is True
        assert len(d["checks"]) == 1

    def test_valid_model_passes(self, governance):
        self._record_model(governance, version="v001")
        is_valid, issues = governance.validate_for_deployment("v001")
        assert is_valid is True
        assert len(issues) == 0

    def test_low_cv_accuracy_fails(self, governance):
        self._record_model(governance, version="v002", metrics={
            "cv_accuracy": 0.40,
            "sharpe": 0.8,
            "sharpe_ratio": 0.8,
            "max_drawdown": 0.10,
            "n_trades": 50,
        })
        is_valid, issues = governance.validate_for_deployment("v002")
        assert is_valid is False
        assert any("CV accuracy" in i for i in issues)

    def test_low_sharpe_fails(self, governance):
        self._record_model(governance, version="v003", metrics={
            "cv_accuracy": 0.65,
            "sharpe": 0.1,
            "sharpe_ratio": 0.1,
            "max_drawdown": 0.10,
            "n_trades": 50,
        })
        is_valid, issues = governance.validate_for_deployment("v003")
        assert is_valid is False
        assert any("Sharpe ratio" in i for i in issues)

    def test_high_drawdown_fails(self, governance):
        self._record_model(governance, version="v004", metrics={
            "cv_accuracy": 0.65,
            "sharpe": 0.8,
            "sharpe_ratio": 0.8,
            "max_drawdown": 0.30,
            "n_trades": 50,
        })
        is_valid, issues = governance.validate_for_deployment("v004")
        assert is_valid is False
        assert any("drawdown" in i.lower() for i in issues)

    def test_insufficient_trades_fails(self, governance):
        self._record_model(governance, version="v005", metrics={
            "cv_accuracy": 0.65,
            "sharpe": 0.8,
            "sharpe_ratio": 0.8,
            "max_drawdown": 0.10,
            "n_trades": 10,
        })
        is_valid, issues = governance.validate_for_deployment("v005")
        assert is_valid is False
        assert any("trades" in i.lower() for i in issues)

    def test_nonexistent_version_fails(self, governance):
        is_valid, issues = governance.validate_for_deployment("nonexistent")
        assert is_valid is False
        assert "not found" in issues[0]

    def test_validation_stores_result(self, governance):
        self._record_model(governance, version="v006")
        governance.validate_for_deployment("v006")
        result = governance._last_validation_result
        assert isinstance(result, ValidationResult)
        assert len(result.checks) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Circuit Breakers
# ──────────────────────────────────────────────────────────────────────────────


class TestCircuitBreaker:
    """Test TradingCircuitBreaker trip/reset/check behavior."""

    def test_initial_state_is_normal(self):
        cb = TradingCircuitBreaker()
        assert cb.state == CircuitBreakerState.NORMAL
        assert cb.is_trading_allowed() is True

    def test_trip_enters_safe_mode(self):
        cb = TradingCircuitBreaker()
        cb.trip(CircuitBreakerReason.NO_ACTIVE_MODEL.value)
        assert cb.state == CircuitBreakerState.SAFE_MODE
        assert cb.is_trading_allowed() is False
        assert CircuitBreakerReason.NO_ACTIVE_MODEL.value in cb.active_reasons

    def test_reset_clears_state(self):
        cb = TradingCircuitBreaker()
        cb.trip(CircuitBreakerReason.NO_ACTIVE_MODEL.value)
        cb.reset(CircuitBreakerReason.NO_ACTIVE_MODEL.value)
        assert cb.state == CircuitBreakerState.NORMAL
        assert cb.is_trading_allowed() is True

    def test_reset_requires_all_reasons_cleared(self):
        cb = TradingCircuitBreaker()
        cb.trip(CircuitBreakerReason.NO_ACTIVE_MODEL.value)
        cb.trip(CircuitBreakerReason.DATA_QUALITY.value)

        # Clear one reason — still in safe mode
        cb.reset(CircuitBreakerReason.NO_ACTIVE_MODEL.value)
        assert cb.state == CircuitBreakerState.SAFE_MODE

        # Clear second reason — back to normal
        cb.reset(CircuitBreakerReason.DATA_QUALITY.value)
        assert cb.state == CircuitBreakerState.NORMAL

    def test_check_all_with_no_model(self):
        predictor = MagicMock()
        predictor.is_loaded = False

        cb = TradingCircuitBreaker()
        state, reasons = cb.check_all(predictor=predictor)
        assert state == CircuitBreakerState.SAFE_MODE
        assert CircuitBreakerReason.NO_ACTIVE_MODEL.value in reasons

    def test_check_all_with_healthy_system(self):
        predictor = MagicMock()
        predictor.is_loaded = True

        cb = TradingCircuitBreaker()
        state, reasons = cb.check_all(predictor=predictor)
        assert state == CircuitBreakerState.NORMAL
        assert len(reasons) == 0

    def test_publishes_trip_event(self):
        bus = EventBus()
        events = []
        bus.subscribe(CircuitBreakerTripped, lambda e: events.append(e))

        cb = TradingCircuitBreaker(event_bus=bus)
        cb.trip(CircuitBreakerReason.NO_ACTIVE_MODEL.value)

        assert len(events) == 1
        assert events[0].state == CircuitBreakerState.SAFE_MODE.value

    def test_publishes_reset_event(self):
        bus = EventBus()
        events = []
        bus.subscribe(CircuitBreakerReset, lambda e: events.append(e))

        cb = TradingCircuitBreaker(event_bus=bus)
        cb.trip(CircuitBreakerReason.NO_ACTIVE_MODEL.value)
        cb.reset(CircuitBreakerReason.NO_ACTIVE_MODEL.value)

        assert len(events) == 1
        assert events[0].previous_state == CircuitBreakerState.SAFE_MODE.value

    def test_get_status(self):
        cb = TradingCircuitBreaker()
        status = cb.get_status()
        assert status["state"] == "NORMAL"
        assert status["trading_allowed"] is True

    def test_duplicate_trip_reason(self):
        cb = TradingCircuitBreaker()
        cb.trip(CircuitBreakerReason.NO_ACTIVE_MODEL.value)
        cb.trip(CircuitBreakerReason.NO_ACTIVE_MODEL.value)
        assert cb.active_reasons.count(CircuitBreakerReason.NO_ACTIVE_MODEL.value) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: Audit Trail
# ──────────────────────────────────────────────────────────────────────────────


class TestTradeAuditTrail:
    """Test audit trail dataclass linking trades to predictions and models."""

    def test_create_complete_trail(self):
        trail = TradeAuditTrail(
            trade_id="trade_001",
            signal_id="sig_001",
            prediction_id="pred_001",
            model_version="v003",
            training_run_id="run_001",
            backtest_run_id="bt_001",
            dataset_snapshot_hash="abc123",
            feature_snapshot_hash="def456",
        )
        assert trail.is_complete() is True

    def test_incomplete_trail(self):
        trail = TradeAuditTrail(trade_id="trade_001")
        assert trail.is_complete() is False

    def test_to_dict_from_dict(self):
        trail = TradeAuditTrail(
            trade_id="trade_001",
            prediction_id="pred_001",
            model_version="v003",
        )
        d = trail.to_dict()
        restored = TradeAuditTrail.from_dict(d)
        assert restored.trade_id == "trade_001"
        assert restored.prediction_id == "pred_001"
        assert restored.model_version == "v003"

    def test_timestamp_auto_populated(self):
        trail = TradeAuditTrail(trade_id="trade_001")
        assert trail.timestamp  # Should be auto-populated


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5: Self-Healing & Recovery
# ──────────────────────────────────────────────────────────────────────────────


class TestSelfHealing:
    """Test training pipeline retry, stale model detection, and rollback."""

    @pytest.fixture
    def governance(self, tmp_path):
        return ModelGovernance(governance_dir=str(tmp_path / "governance"))

    def test_check_stale_model_no_deployed(self, governance, tmp_path):
        """No deployed model should be considered stale."""
        from src.ml.model_registry import ModelRegistry
        from src.ml.training_pipeline import TrainingPipeline

        registry = ModelRegistry(models_dir=str(tmp_path / "models"))
        pipeline = TrainingPipeline(registry=registry, governance=governance)
        assert pipeline.check_stale_model() is True

    def test_check_stale_model_recent(self, governance, tmp_path):
        """Recently deployed model should not be stale."""
        from src.ml.model_registry import ModelRegistry
        from src.ml.training_pipeline import TrainingPipeline

        registry = ModelRegistry(models_dir=str(tmp_path / "models"))
        pipeline = TrainingPipeline(registry=registry, governance=governance)

        # Record a recent model
        dataset = pd.DataFrame({"close": range(100), "volume": range(100)})
        governance.record_training(
            version="v001",
            features=["rsi_14"],
            dataset=dataset,
            metrics={"cv_accuracy": 0.65},
            hyperparameters={"n_estimators": 200},
        )
        governance.deploy("v001", reason="test")

        assert pipeline.check_stale_model() is False


# ──────────────────────────────────────────────────────────────────────────────
# Phase 6: New Event Types
# ──────────────────────────────────────────────────────────────────────────────


class TestNewEvents:
    """Test all new event types added for ML lifecycle hardening."""

    def test_prediction_unavailable_event(self):
        event = PredictionUnavailable(
            symbol="AAPL",
            reason="no active model",
            source="ml_strategy",
        )
        d = event.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["reason"] == "no active model"
        assert d["_type"] == "PredictionUnavailable"

    def test_model_rejected_event(self):
        event = ModelRejected(
            version="v003",
            reasons=["Low Sharpe", "High drawdown"],
            source="training_pipeline",
        )
        d = event.to_dict()
        assert d["version"] == "v003"
        assert len(d["reasons"]) == 2

    def test_circuit_breaker_tripped_event(self):
        event = CircuitBreakerTripped(
            state="SAFE_MODE",
            reasons=["NO_ACTIVE_MODEL"],
            source="circuit_breaker",
        )
        d = event.to_dict()
        assert d["state"] == "SAFE_MODE"

    def test_circuit_breaker_reset_event(self):
        event = CircuitBreakerReset(
            previous_state="SAFE_MODE",
            source="circuit_breaker",
        )
        d = event.to_dict()
        assert d["previous_state"] == "SAFE_MODE"

    def test_model_auto_rollback_event(self):
        event = ModelAutoRollback(
            from_version="v003",
            to_version="v002",
            reason="underperformance",
            source="training_pipeline",
        )
        d = event.to_dict()
        assert d["from_version"] == "v003"
        assert d["to_version"] == "v002"

    def test_event_bus_publishes_new_events(self):
        bus = EventBus()
        received = []
        bus.subscribe(ModelRejected, lambda e: received.append(e))
        bus.publish(ModelRejected(version="v001", reasons=["bad"], source="test"))
        assert len(received) == 1
        assert received[0].version == "v001"

    def test_event_bus_wildcard_catches_new_events(self):
        bus = EventBus()
        received = []
        bus.subscribe(None, lambda e: received.append(e))

        bus.publish(PredictionUnavailable(symbol="AAPL", reason="test"))
        bus.publish(CircuitBreakerTripped(state="SAFE_MODE", reasons=["test"]))
        bus.publish(ModelAutoRollback(from_version="v2", to_version="v1", reason="test"))

        assert len(received) == 3

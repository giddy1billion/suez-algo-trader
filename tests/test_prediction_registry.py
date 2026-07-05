"""Tests for Prediction Registry (Phase 3)."""

import json
import os
import tempfile
from datetime import datetime, timezone

import pytest

from src.predictions.registry import PredictionRecord, PredictionRegistry
from src.predictions.metrics import PredictionMetrics, MetricsSnapshot
from src.predictions.calibration import CalibrationAnalyzer, CalibrationReport


class TestPredictionRecord:
    """Test prediction record data class."""

    def test_create_record(self):
        record = PredictionRecord(
            asset="AAPL",
            direction="long",
            confidence=0.85,
            expected_horizon=24,
        )
        assert record.asset == "AAPL"
        assert record.direction == "long"
        assert record.confidence == 0.85
        assert record.resolved is False
        assert record.prediction_id  # Auto-generated

    def test_to_dict_from_dict(self):
        record = PredictionRecord(
            asset="BTC/USD",
            direction="short",
            confidence=0.72,
            model_version="v1.2",
        )
        d = record.to_dict()
        restored = PredictionRecord.from_dict(d)
        assert restored.asset == "BTC/USD"
        assert restored.direction == "short"
        assert restored.confidence == 0.72


class TestPredictionRegistry:
    """Test prediction registry lifecycle."""

    @pytest.fixture
    def registry(self, tmp_path):
        return PredictionRegistry(storage_path=str(tmp_path / "predictions"))

    def test_register_prediction(self, registry):
        record = registry.register(
            asset="AAPL",
            direction="long",
            confidence=0.80,
        )
        assert record.prediction_id
        assert registry.active_count == 1

    def test_record_outcome(self, registry):
        record = registry.register(
            asset="AAPL",
            direction="long",
            confidence=0.80,
            expected_return=0.02,
        )
        result = registry.record_outcome(record.prediction_id, actual_return=0.015)
        assert result is not None
        assert result.resolved is True
        assert result.direction_correct is True
        assert result.quality_grade in ("excellent", "good", "fair")
        assert registry.active_count == 0
        assert registry.resolved_count == 1

    def test_record_outcome_wrong_direction(self, registry):
        record = registry.register(
            asset="AAPL",
            direction="long",
            confidence=0.80,
        )
        result = registry.record_outcome(record.prediction_id, actual_return=-0.05)
        assert result.direction_correct is False
        assert result.quality_grade == "poor"

    def test_record_outcome_not_found(self, registry):
        result = registry.record_outcome("nonexistent", actual_return=0.01)
        assert result is None

    def test_get_active_predictions(self, registry):
        registry.register(asset="AAPL", direction="long", confidence=0.80)
        registry.register(asset="MSFT", direction="short", confidence=0.70)
        registry.register(asset="AAPL", direction="long", confidence=0.60)

        all_active = registry.get_active_predictions()
        assert len(all_active) == 3

        aapl_active = registry.get_active_predictions(asset="AAPL")
        assert len(aapl_active) == 2

    def test_get_resolved_predictions(self, registry):
        r1 = registry.register(asset="AAPL", direction="long", confidence=0.80)
        r2 = registry.register(asset="MSFT", direction="short", confidence=0.70)
        registry.record_outcome(r1.prediction_id, actual_return=0.02)

        resolved = registry.get_resolved_predictions()
        assert len(resolved) == 1
        assert resolved[0].asset == "AAPL"

    def test_summary(self, registry):
        r = registry.register(asset="AAPL", direction="long", confidence=0.80)
        registry.record_outcome(r.prediction_id, actual_return=0.01)

        summary = registry.get_summary()
        assert summary["resolved"] == 1
        assert summary["accuracy"] == 1.0

    def test_persist_active(self, registry):
        registry.register(asset="AAPL", direction="long", confidence=0.80)
        registry.persist_active()
        # Verify file was created
        active_file = registry._storage_path / "active.json"
        assert active_file.exists()


class TestPredictionMetrics:
    """Test prediction metrics computation."""

    def _make_predictions(self, count=50, accuracy=0.7):
        """Create test predictions with specified accuracy."""
        predictions = []
        for i in range(count):
            correct = i < int(count * accuracy)
            p = PredictionRecord(
                asset="AAPL",
                direction="long",
                confidence=0.75 if correct else 0.65,
                resolved=True,
                direction_correct=correct,
                actual_return=0.02 if correct else -0.01,
                absolute_error=0.005 if correct else 0.02,
            )
            predictions.append(p)
        return predictions

    def test_compute_basic_metrics(self):
        metrics = PredictionMetrics(window=100)
        predictions = self._make_predictions(50, accuracy=0.7)
        result = metrics.compute(predictions)
        assert isinstance(result, MetricsSnapshot)
        assert result.total_predictions == 50
        assert abs(result.directional_accuracy - 0.7) < 0.01

    def test_empty_predictions(self):
        metrics = PredictionMetrics()
        result = metrics.compute([])
        assert result.total_predictions == 0
        assert result.directional_accuracy == 0.0

    def test_brier_score(self):
        metrics = PredictionMetrics()
        # Perfect predictions: confidence matches outcome
        predictions = []
        for i in range(20):
            predictions.append(PredictionRecord(
                resolved=True,
                direction_correct=True,
                confidence=1.0,
                actual_return=0.01,
                absolute_error=0.0,
                asset="AAPL",
                direction="long",
            ))
        result = metrics.compute(predictions)
        assert result.brier_score == 0.0  # Perfect calibration

    def test_hit_rate_by_bucket(self):
        metrics = PredictionMetrics()
        predictions = self._make_predictions(50)
        result = metrics.compute(predictions)
        assert isinstance(result.hit_rate_by_bucket, dict)
        assert len(result.hit_rate_by_bucket) > 0

    def test_per_asset_accuracy(self):
        metrics = PredictionMetrics()
        predictions = []
        for asset in ["AAPL", "MSFT"]:
            for i in range(10):
                predictions.append(PredictionRecord(
                    asset=asset,
                    direction="long",
                    confidence=0.75,
                    resolved=True,
                    direction_correct=True,
                    actual_return=0.01,
                    absolute_error=0.005,
                ))
        result = metrics.compute(predictions)
        assert "AAPL" in result.per_asset_accuracy
        assert result.per_asset_accuracy["AAPL"] == 1.0


class TestCalibrationAnalyzer:
    """Test calibration analysis."""

    def _make_calibrated_predictions(self, count=100):
        """Create well-calibrated predictions."""
        import random
        random.seed(42)
        predictions = []
        for i in range(count):
            conf = random.uniform(0.5, 1.0)
            correct = random.random() < conf
            predictions.append(PredictionRecord(
                asset="AAPL",
                direction="long",
                confidence=conf,
                resolved=True,
                direction_correct=correct,
                actual_return=0.01 if correct else -0.01,
                absolute_error=0.005,
            ))
        return predictions

    def test_analyze_insufficient_data(self):
        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze([])
        assert "Insufficient" in result.recommendation

    def test_analyze_calibrated_predictions(self):
        analyzer = CalibrationAnalyzer(n_bins=5)
        predictions = self._make_calibrated_predictions(200)
        result = analyzer.analyze(predictions)
        assert isinstance(result, CalibrationReport)
        assert len(result.calibration_curve) > 0
        assert 0 <= result.expected_calibration_error <= 1
        assert 0 <= result.brier_score <= 1

    def test_reliability_diagram_data(self):
        analyzer = CalibrationAnalyzer()
        predictions = self._make_calibrated_predictions(100)
        data = analyzer.compute_reliability_diagram_data(predictions)
        assert "predicted" in data
        assert "actual" in data
        assert "counts" in data

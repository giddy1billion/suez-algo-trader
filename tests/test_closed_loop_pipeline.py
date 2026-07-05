"""Tests for Closed-Loop Learning Pipeline (Phase 4)."""

import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.ml.dataset_builder import DatasetBuilder, DatasetVersion
from src.ml.retraining_trigger import RetrainingEvidence, RetrainingTrigger
from src.predictions.registry import PredictionRecord


class TestDatasetBuilder:
    """Test dataset building from prediction outcomes."""

    @pytest.fixture
    def builder(self, tmp_path):
        return DatasetBuilder(
            storage_dir=str(tmp_path / "datasets"),
            min_records=10,  # Low threshold for testing
        )

    def _make_predictions(self, count=20):
        """Create resolved predictions for testing."""
        predictions = []
        for i in range(count):
            predictions.append(PredictionRecord(
                prediction_id=f"pred_{i:04d}",
                timestamp=datetime(2024, 1, 1 + i % 28, tzinfo=timezone.utc).isoformat(),
                asset=["AAPL", "MSFT", "GOOGL"][i % 3],
                direction="long" if i % 2 == 0 else "short",
                confidence=0.6 + (i % 4) * 0.1,
                expected_return=0.02,
                model_version="v1.0",
                strategy="momentum",
                resolved=True,
                actual_return=0.015 if i % 3 != 0 else -0.01,
                direction_correct=i % 3 != 0,
                absolute_error=0.005,
                quality_grade="good" if i % 3 != 0 else "poor",
            ))
        return predictions

    def test_build_dataset_success(self, builder):
        predictions = self._make_predictions(20)
        result = builder.build_dataset(predictions)
        assert result is not None
        assert isinstance(result, DatasetVersion)
        assert result.record_count == 20
        assert "AAPL" in result.symbols
        assert result.file_path

    def test_build_dataset_insufficient_data(self, builder):
        predictions = self._make_predictions(5)
        result = builder.build_dataset(predictions)
        assert result is None

    def test_build_dataset_ignores_unresolved(self, builder):
        predictions = self._make_predictions(20)
        # Add unresolved predictions
        for i in range(10):
            predictions.append(PredictionRecord(
                asset="NVDA",
                direction="long",
                confidence=0.8,
                resolved=False,
            ))
        result = builder.build_dataset(predictions)
        # Should only include the 20 resolved
        assert result.record_count == 20

    def test_get_latest_dataset(self, builder):
        predictions = self._make_predictions(20)
        builder.build_dataset(predictions)
        latest = builder.get_latest_dataset()
        assert latest is not None
        assert latest.record_count == 20

    def test_list_versions(self, builder):
        predictions = self._make_predictions(20)
        builder.build_dataset(predictions)
        versions = builder.list_versions()
        assert len(versions) == 1
        assert "version" in versions[0]

    def test_get_dataset_loads_parquet(self, builder):
        predictions = self._make_predictions(20)
        version_info = builder.build_dataset(predictions)
        df = builder.get_dataset(version_info.version)
        assert df is not None
        assert len(df) == 20

    def test_manifest_persistence(self, tmp_path):
        builder1 = DatasetBuilder(storage_dir=str(tmp_path / "ds"), min_records=10)
        predictions = self._make_predictions(20)
        builder1.build_dataset(predictions)

        # Create new builder pointing to same dir
        builder2 = DatasetBuilder(storage_dir=str(tmp_path / "ds"), min_records=10)
        assert builder2.total_versions == 1


class TestRetrainingTrigger:
    """Test evidence-driven retraining decisions."""

    @pytest.fixture
    def trigger(self):
        return RetrainingTrigger(
            min_outcomes=100,
            drift_threshold=0.15,
            max_frequency_hours=1.0,
            scheduled_interval_hours=24.0,
            brier_threshold=0.30,
        )

    def test_no_retrain_when_insufficient_outcomes(self, trigger):
        # Record a recent training so scheduled fallback doesn't fire
        trigger.record_training_completed()
        result = trigger.should_retrain(new_outcome_count=50)
        assert result is None

    def test_retrain_on_sufficient_outcomes(self, trigger):
        result = trigger.should_retrain(new_outcome_count=150)
        assert result is not None
        assert result.reason == "sufficient_outcomes"
        assert result.outcome_count == 150

    def test_retrain_on_drift(self, trigger):
        result = trigger.should_retrain(
            new_outcome_count=50,
            drift_score=0.20,
        )
        assert result is not None
        assert result.reason == "drift_detected"

    def test_retrain_on_calibration_degradation(self, trigger):
        result = trigger.should_retrain(
            new_outcome_count=50,
            brier_score=0.35,
        )
        assert result is not None
        assert result.reason == "calibration_degradation"

    def test_retrain_on_accuracy_drop(self, trigger):
        result = trigger.should_retrain(
            new_outcome_count=50,
            current_accuracy=0.55,
            baseline_accuracy=0.70,
        )
        assert result is not None
        assert result.reason == "accuracy_degradation"

    def test_frequency_limiting(self, trigger):
        # Record a recent training
        trigger.record_training_completed()

        # Should not allow retraining immediately even with evidence
        result = trigger.should_retrain(new_outcome_count=500)
        assert result is None

    def test_manual_trigger(self, trigger):
        evidence = trigger.trigger_manual()
        assert evidence.reason == "manual"

    def test_get_status(self, trigger):
        status = trigger.get_status()
        assert "min_outcomes" in status
        assert "drift_threshold" in status
        assert "can_retrain" in status
        assert status["can_retrain"] is True

    def test_evidence_summary(self):
        evidence = RetrainingEvidence(
            reason="drift_detected",
            drift_score=0.18,
            outcome_count=200,
        )
        summary = evidence.summary()
        assert "drift_detected" in summary
        assert "0.180" in summary

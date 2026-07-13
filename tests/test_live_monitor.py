"""
End-to-end tests for live model monitoring and automated rollback.

Tests simulate drift and deteriorating performance and verify:
1. Rollback occurs exactly once when criteria are met
2. System remains in a consistent state after rollback
3. Auditable events are emitted for all state transitions
4. Circuit breaker limits daily rollbacks
"""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.ml.live_monitor import (
    AuditEvent,
    LiveModelMonitor,
    LiveMonitorConfig,
    MonitorEventType,
    PerformanceSnapshot,
    RollbackReason,
    RollbackRecord,
)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def config():
    """Test config with low thresholds for fast triggering."""
    return LiveMonitorConfig(
        sharpe_window=20,
        sharpe_threshold=-0.5,
        sharpe_sustained_periods=3,
        accuracy_window=50,
        accuracy_degradation_pvalue=0.05,
        accuracy_baseline=0.52,
        drift_psi_threshold=0.25,
        min_observations=10,
        evaluation_interval_seconds=0.0,
        max_rollbacks_per_day=3,
        n_distribution_bins=3,
    )


@pytest.fixture
def monitor(config):
    """Fresh monitor instance with test config."""
    return LiveModelMonitor(config=config)


@pytest.fixture
def activated_monitor(monitor):
    """Monitor with an active model and baseline."""
    monitor.activate_model(
        "v2.0",
        previous_version="v1.0",
        baseline_distribution=[0.33, 0.34, 0.33],
        baseline_accuracy=0.55,
    )
    return monitor


# ──────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────


def feed_good_outcomes(monitor: LiveModelMonitor, n: int, model: str = "v2.0"):
    """Feed positive trade outcomes."""
    for _ in range(n):
        monitor.record_prediction(model, predicted_class=np.random.randint(0, 3), confidence=0.7)
        monitor.record_outcome(model, realized_return=0.005, was_correct=True)


def feed_bad_outcomes(monitor: LiveModelMonitor, n: int, model: str = "v2.0"):
    """Feed negative trade outcomes (losses + wrong predictions)."""
    for _ in range(n):
        monitor.record_prediction(model, predicted_class=np.random.randint(0, 3), confidence=0.4)
        monitor.record_outcome(model, realized_return=-0.02, was_correct=False)


def feed_mixed_bad_outcomes(monitor: LiveModelMonitor, n: int, model: str = "v2.0"):
    """Feed mostly bad but some ok outcomes — still degrading."""
    rng = np.random.default_rng(42)
    for i in range(n):
        is_bad = rng.random() < 0.75  # 75% bad trades
        if is_bad:
            monitor.record_prediction(model, predicted_class=0, confidence=0.4)
            monitor.record_outcome(model, realized_return=-0.015, was_correct=False)
        else:
            monitor.record_prediction(model, predicted_class=2, confidence=0.6)
            monitor.record_outcome(model, realized_return=0.003, was_correct=True)


# ──────────────────────────────────────────────────────────────────────────
# Tests: Basic Lifecycle
# ──────────────────────────────────────────────────────────────────────────


class TestModelActivation:
    def test_activate_model_sets_state(self, monitor):
        monitor.activate_model("v2.0", previous_version="v1.0")
        assert monitor.active_model == "v2.0"
        assert not monitor.has_rolled_back

    def test_activate_emits_promotion_event(self, monitor):
        monitor.activate_model("v2.0", previous_version="v1.0")
        events = monitor.audit_log
        assert len(events) == 1
        assert events[0].event_type == MonitorEventType.MODEL_PROMOTED
        assert events[0].model_version == "v2.0"

    def test_no_evaluation_without_active_model(self, monitor):
        result = monitor.evaluate()
        assert result is None

    def test_no_evaluation_with_insufficient_data(self, activated_monitor):
        # Only 5 observations, need 10
        feed_good_outcomes(activated_monitor, 5)
        result = activated_monitor.evaluate()
        assert result is None


# ──────────────────────────────────────────────────────────────────────────
# Tests: Sharpe-Based Rollback
# ──────────────────────────────────────────────────────────────────────────


class TestSharpeRollback:
    @pytest.fixture
    def sharpe_monitor(self, config):
        """Monitor configured to only test Sharpe criterion."""
        config.accuracy_baseline = 0.01  # Disable accuracy criterion
        config.accuracy_degradation_pvalue = 0.0001
        config.drift_psi_threshold = 100.0  # Disable drift criterion
        monitor = LiveModelMonitor(config=config)
        monitor.activate_model("v2.0", previous_version="v1.0")
        return monitor

    def test_sustained_negative_sharpe_triggers_rollback(self, sharpe_monitor):
        """Rollback after sustained_periods consecutive evaluations with Sharpe < threshold."""
        config = sharpe_monitor.config
        rng = np.random.default_rng(42)

        # Feed varying losses to produce negative Sharpe with nonzero std
        for _ in range(30):
            ret = -0.015 + rng.normal(0, 0.005)  # Mean ~ -1.5% with noise
            sharpe_monitor.record_prediction("v2.0", predicted_class=1, confidence=0.5)
            sharpe_monitor.record_outcome("v2.0", realized_return=ret, was_correct=True)

        # Need 3 consecutive evaluations below threshold
        for i in range(config.sharpe_sustained_periods):
            result = sharpe_monitor.evaluate()
            assert result is not None

        # Verify rollback occurred
        assert sharpe_monitor.has_rolled_back
        assert sharpe_monitor.active_model == "v1.0"
        assert sharpe_monitor.rollback_record is not None
        assert sharpe_monitor.rollback_record.reason == RollbackReason.SUSTAINED_NEGATIVE_SHARPE
        assert sharpe_monitor.rollback_record.from_version == "v2.0"
        assert sharpe_monitor.rollback_record.to_version == "v1.0"

    def test_sharpe_recovers_resets_counter(self, sharpe_monitor):
        """If Sharpe recovers between evaluations, counter resets."""
        rng = np.random.default_rng(43)

        # Feed bad returns and evaluate twice
        for _ in range(15):
            ret = -0.015 + rng.normal(0, 0.005)
            sharpe_monitor.record_prediction("v2.0", predicted_class=1, confidence=0.5)
            sharpe_monitor.record_outcome("v2.0", realized_return=ret, was_correct=True)

        sharpe_monitor.evaluate()
        sharpe_monitor.evaluate()

        # Now feed good returns to recover Sharpe
        for _ in range(30):
            ret = 0.01 + rng.normal(0, 0.003)
            sharpe_monitor.record_prediction("v2.0", predicted_class=2, confidence=0.7)
            sharpe_monitor.record_outcome("v2.0", realized_return=ret, was_correct=True)

        sharpe_monitor.evaluate()

        # Should NOT have rolled back
        assert not sharpe_monitor.has_rolled_back
        assert sharpe_monitor.active_model == "v2.0"

    def test_warning_emitted_before_rollback(self, sharpe_monitor):
        """Performance warnings are emitted before the actual rollback."""
        config = sharpe_monitor.config
        config.sharpe_sustained_periods = 3  # Need 3 to rollback
        rng = np.random.default_rng(44)

        # Feed bad returns with variance
        for _ in range(30):
            ret = -0.015 + rng.normal(0, 0.005)
            sharpe_monitor.record_prediction("v2.0", predicted_class=1, confidence=0.5)
            sharpe_monitor.record_outcome("v2.0", realized_return=ret, was_correct=True)

        # First evaluation — warning only (1 of 3)
        sharpe_monitor.evaluate()
        assert not sharpe_monitor.has_rolled_back

        warnings = [
            e for e in sharpe_monitor.audit_log
            if e.event_type == MonitorEventType.PERFORMANCE_WARNING
        ]
        assert len(warnings) >= 1


# ──────────────────────────────────────────────────────────────────────────
# Tests: Accuracy-Based Rollback
# ──────────────────────────────────────────────────────────────────────────


class TestAccuracyRollback:
    def test_accuracy_degradation_triggers_rollback(self, config):
        """Statistically significant accuracy drop triggers rollback."""
        # Use higher baseline so degradation is easier to detect
        config.accuracy_baseline = 0.60
        config.sharpe_threshold = -10.0  # Disable Sharpe criterion
        config.sharpe_sustained_periods = 100
        config.drift_psi_threshold = 100.0  # Disable drift criterion
        monitor = LiveModelMonitor(config=config)
        monitor.activate_model(
            "v2.0",
            previous_version="v1.0",
            baseline_accuracy=0.60,
        )

        # Feed outcomes that are significantly below 60% accuracy
        # ~ 30% accuracy should be statistically significant with n=50
        rng = np.random.default_rng(123)
        for _ in range(60):
            is_correct = rng.random() < 0.25  # 25% accuracy
            monitor.record_prediction("v2.0", predicted_class=1, confidence=0.5)
            ret = 0.003 if is_correct else -0.001
            monitor.record_outcome("v2.0", realized_return=ret, was_correct=is_correct)

        result = monitor.evaluate()
        assert result is not None
        assert monitor.has_rolled_back
        assert monitor.rollback_record.reason == RollbackReason.ACCURACY_DEGRADATION

    def test_no_rollback_when_accuracy_acceptable(self, config):
        """No rollback when accuracy is at or above baseline."""
        config.sharpe_threshold = -10.0
        config.sharpe_sustained_periods = 100
        config.drift_psi_threshold = 100.0
        monitor = LiveModelMonitor(config=config)
        monitor.activate_model("v2.0", previous_version="v1.0", baseline_accuracy=0.52)

        # Feed outcomes at 60% accuracy (above 52% baseline)
        rng = np.random.default_rng(456)
        for _ in range(60):
            is_correct = rng.random() < 0.60
            monitor.record_prediction("v2.0", predicted_class=1, confidence=0.6)
            ret = 0.003 if is_correct else -0.002
            monitor.record_outcome("v2.0", realized_return=ret, was_correct=is_correct)

        monitor.evaluate()
        assert not monitor.has_rolled_back


# ──────────────────────────────────────────────────────────────────────────
# Tests: Drift-Based Rollback
# ──────────────────────────────────────────────────────────────────────────


class TestDriftRollback:
    def test_prediction_drift_triggers_rollback(self, config):
        """Large PSI triggers rollback."""
        config.sharpe_threshold = -10.0
        config.sharpe_sustained_periods = 100
        config.accuracy_baseline = 0.01  # Disable accuracy criterion
        config.drift_psi_threshold = 0.10  # Low threshold for test
        monitor = LiveModelMonitor(config=config)
        monitor.activate_model(
            "v2.0",
            previous_version="v1.0",
            baseline_distribution=[0.33, 0.34, 0.33],  # uniform
        )

        # Feed predictions heavily biased to class 0 (drift)
        for _ in range(50):
            monitor.record_prediction("v2.0", predicted_class=0, confidence=0.8)
            monitor.record_outcome("v2.0", realized_return=-0.001, was_correct=True)

        result = monitor.evaluate()
        assert monitor.has_rolled_back
        assert monitor.rollback_record.reason == RollbackReason.PREDICTION_DRIFT

    def test_no_drift_with_matching_distribution(self, config):
        """No drift when distribution matches baseline."""
        config.sharpe_threshold = -10.0
        config.sharpe_sustained_periods = 100
        config.accuracy_baseline = 0.01
        monitor = LiveModelMonitor(config=config)
        monitor.activate_model(
            "v2.0",
            previous_version="v1.0",
            baseline_distribution=[0.33, 0.34, 0.33],
        )

        # Feed balanced predictions
        rng = np.random.default_rng(789)
        for _ in range(50):
            cls = rng.integers(0, 3)
            monitor.record_prediction("v2.0", predicted_class=int(cls), confidence=0.6)
            monitor.record_outcome("v2.0", realized_return=0.001, was_correct=True)

        monitor.evaluate()
        assert not monitor.has_rolled_back


# ──────────────────────────────────────────────────────────────────────────
# Tests: Rollback Occurs Exactly Once
# ──────────────────────────────────────────────────────────────────────────


class TestRollbackExactlyOnce:
    def test_rollback_occurs_only_once(self, activated_monitor):
        """Even with continued bad data, rollback only happens once."""
        feed_bad_outcomes(activated_monitor, 50)

        # Evaluate many times
        rollback_count = 0
        for _ in range(10):
            result = activated_monitor.evaluate()
            if result is None and activated_monitor.has_rolled_back:
                # After rollback, evaluate returns None
                pass

        # Count rollback events in audit log
        rollback_events = [
            e for e in activated_monitor.audit_log
            if e.event_type == MonitorEventType.ROLLBACK_TRIGGERED
        ]
        assert len(rollback_events) == 1

    def test_second_evaluate_after_rollback_returns_none(self, activated_monitor):
        """After rollback, further evaluations return None."""
        feed_bad_outcomes(activated_monitor, 50)

        # Trigger rollback
        for _ in range(5):
            activated_monitor.evaluate()

        assert activated_monitor.has_rolled_back

        # Further evaluations return None
        result = activated_monitor.evaluate()
        assert result is None

    def test_callback_invoked_exactly_once(self, config):
        """Rollback callback is called exactly once."""
        callback = MagicMock()
        monitor = LiveModelMonitor(config=config, on_rollback=callback)
        monitor.activate_model("v2.0", previous_version="v1.0")

        feed_bad_outcomes(monitor, 50, model="v2.0")
        for _ in range(10):
            monitor.evaluate()

        assert callback.call_count == 1
        record = callback.call_args[0][0]
        assert isinstance(record, RollbackRecord)
        assert record.from_version == "v2.0"
        assert record.to_version == "v1.0"


# ──────────────────────────────────────────────────────────────────────────
# Tests: Consistent State After Rollback
# ──────────────────────────────────────────────────────────────────────────


class TestConsistentStateAfterRollback:
    def test_active_model_is_previous_version(self, activated_monitor):
        """After rollback, active model is the previous (stable) version."""
        feed_bad_outcomes(activated_monitor, 50)
        for _ in range(5):
            activated_monitor.evaluate()

        assert activated_monitor.active_model == "v1.0"

    def test_rollback_record_has_complete_metrics(self, activated_monitor):
        """Rollback record contains all metrics at time of rollback."""
        feed_bad_outcomes(activated_monitor, 50)
        for _ in range(5):
            activated_monitor.evaluate()

        record = activated_monitor.rollback_record
        assert record is not None
        assert record.from_version == "v2.0"
        assert record.to_version == "v1.0"
        assert "live_sharpe" in record.metrics_at_rollback
        assert "rolling_accuracy" in record.metrics_at_rollback
        assert "psi_score" in record.metrics_at_rollback
        assert record.evaluation_window_size > 0

    def test_performance_summary_reflects_rollback(self, activated_monitor):
        """Performance summary shows rollback state."""
        feed_bad_outcomes(activated_monitor, 50)
        for _ in range(5):
            activated_monitor.evaluate()

        summary = activated_monitor.get_performance_summary()
        assert summary["rollback_occurred"] is True
        assert summary["model_version"] == "v1.0"

    def test_no_new_predictions_recorded_after_rollback(self, activated_monitor):
        """Predictions for the old model are not recorded after rollback."""
        feed_bad_outcomes(activated_monitor, 50)
        for _ in range(5):
            activated_monitor.evaluate()

        # Try to record prediction for the rolled-back model
        initial_count = len(activated_monitor._predictions)
        activated_monitor.record_prediction("v2.0", predicted_class=1, confidence=0.5)
        # Should not be recorded since v2.0 is no longer active
        assert len(activated_monitor._predictions) == initial_count


# ──────────────────────────────────────────────────────────────────────────
# Tests: Audit Events
# ──────────────────────────────────────────────────────────────────────────


class TestAuditEvents:
    def test_promotion_event_on_activation(self, monitor):
        """Promotion event is logged when model is activated."""
        monitor.activate_model("v2.0", previous_version="v1.0")
        events = [e for e in monitor.audit_log if e.event_type == MonitorEventType.MODEL_PROMOTED]
        assert len(events) == 1
        assert events[0].details["previous_version"] == "v1.0"

    def test_evaluation_event_on_evaluate(self, activated_monitor):
        """Evaluation event is logged on each successful evaluation."""
        feed_good_outcomes(activated_monitor, 15)
        activated_monitor.evaluate()

        events = [e for e in activated_monitor.audit_log if e.event_type == MonitorEventType.EVALUATION_COMPLETED]
        assert len(events) == 1

    def test_rollback_events_complete_chain(self, activated_monitor):
        """Full chain: promotion → warnings → rollback_triggered → rollback_completed."""
        feed_bad_outcomes(activated_monitor, 50)
        for _ in range(5):
            activated_monitor.evaluate()

        event_types = [e.event_type for e in activated_monitor.audit_log]

        assert MonitorEventType.MODEL_PROMOTED in event_types
        assert MonitorEventType.ROLLBACK_TRIGGERED in event_types
        assert MonitorEventType.ROLLBACK_COMPLETED in event_types

    def test_alert_callback_receives_all_events(self, config):
        """Alert callback is invoked for every event."""
        alerts = []
        monitor = LiveModelMonitor(config=config, on_alert=lambda e: alerts.append(e))
        monitor.activate_model("v2.0", previous_version="v1.0")

        feed_bad_outcomes(monitor, 50, model="v2.0")
        for _ in range(5):
            monitor.evaluate()

        assert len(alerts) > 0
        # All alerts should be AuditEvent instances
        assert all(isinstance(a, AuditEvent) for a in alerts)

    def test_event_serialization(self, activated_monitor):
        """All events can be serialized to dict."""
        feed_bad_outcomes(activated_monitor, 50)
        for _ in range(5):
            activated_monitor.evaluate()

        for event in activated_monitor.audit_log:
            d = event.to_dict()
            assert "timestamp" in d
            assert "event_type" in d
            assert "model_version" in d
            assert "severity" in d


# ──────────────────────────────────────────────────────────────────────────
# Tests: Circuit Breaker
# ──────────────────────────────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_max_rollbacks_per_day_enforced(self, config):
        """Circuit breaker stops rollbacks after daily limit."""
        config.max_rollbacks_per_day = 1
        config.sharpe_sustained_periods = 1  # Immediate rollback
        monitor = LiveModelMonitor(config=config)

        # First activation and rollback
        monitor.activate_model("v2.0", previous_version="v1.0")
        feed_bad_outcomes(monitor, 30, model="v2.0")
        monitor.evaluate()
        assert monitor.has_rolled_back

        # Re-activate and try again
        monitor.activate_model("v3.0", previous_version="v2.0")
        feed_bad_outcomes(monitor, 30, model="v3.0")
        monitor.evaluate()
        monitor.evaluate()
        monitor.evaluate()

        # Should NOT have rolled back due to circuit breaker
        assert not monitor.has_rolled_back  # reset on re-activation
        # But _rollbacks_today should still be at limit
        assert monitor._rollbacks_today >= 1


# ──────────────────────────────────────────────────────────────────────────
# Tests: End-to-End Simulation — Drift Scenario
# ──────────────────────────────────────────────────────────────────────────


class TestEndToEndDriftScenario:
    """
    Simulate a realistic scenario where a model starts well then drifts.
    Verify the monitoring system detects and rolls back appropriately.
    """

    def test_gradual_drift_triggers_rollback(self, config):
        """
        Scenario: Model starts performing well, then gradually drifts.
        Expected: System detects degradation and rolls back exactly once.
        """
        config.sharpe_sustained_periods = 3
        config.min_observations = 10
        monitor = LiveModelMonitor(config=config)
        monitor.activate_model(
            "v2.0",
            previous_version="v1.0",
            baseline_distribution=[0.33, 0.34, 0.33],
            baseline_accuracy=0.55,
        )

        rng = np.random.default_rng(42)

        # Phase 1: Good performance (20 trades)
        for _ in range(20):
            cls = rng.integers(0, 3)
            monitor.record_prediction("v2.0", predicted_class=int(cls), confidence=0.7)
            monitor.record_outcome("v2.0", realized_return=0.005, was_correct=True)

        # No rollback yet
        monitor.evaluate()
        assert not monitor.has_rolled_back

        # Phase 2: Gradual degradation (feed bad outcomes)
        for _ in range(40):
            monitor.record_prediction("v2.0", predicted_class=0, confidence=0.4)
            monitor.record_outcome("v2.0", realized_return=-0.015, was_correct=False)

        # Evaluate repeatedly to trigger sustained negative Sharpe
        for _ in range(5):
            monitor.evaluate()

        # Should have rolled back
        assert monitor.has_rolled_back
        assert monitor.active_model == "v1.0"

        # Verify exactly one rollback
        rollback_events = [
            e for e in monitor.audit_log
            if e.event_type == MonitorEventType.ROLLBACK_TRIGGERED
        ]
        assert len(rollback_events) == 1

    def test_sudden_accuracy_collapse(self, config):
        """
        Scenario: Model accuracy suddenly collapses.
        Expected: Accuracy degradation detected, rollback triggered once.
        """
        config.sharpe_threshold = -10.0  # Disable Sharpe
        config.sharpe_sustained_periods = 100
        config.drift_psi_threshold = 100.0  # Disable drift
        config.accuracy_baseline = 0.55
        config.min_observations = 10

        monitor = LiveModelMonitor(config=config)
        monitor.activate_model(
            "v2.0",
            previous_version="v1.0",
            baseline_accuracy=0.55,
        )

        # Phase 1: Normal accuracy (~60%)
        rng = np.random.default_rng(100)
        for _ in range(20):
            is_correct = rng.random() < 0.60
            monitor.record_prediction("v2.0", predicted_class=1, confidence=0.6)
            monitor.record_outcome("v2.0", realized_return=0.002 if is_correct else -0.001, was_correct=is_correct)

        monitor.evaluate()
        assert not monitor.has_rolled_back

        # Phase 2: Accuracy collapse (~20%)
        for _ in range(40):
            is_correct = rng.random() < 0.20
            monitor.record_prediction("v2.0", predicted_class=1, confidence=0.3)
            monitor.record_outcome("v2.0", realized_return=0.002 if is_correct else -0.003, was_correct=is_correct)

        monitor.evaluate()

        assert monitor.has_rolled_back
        assert monitor.rollback_record.reason == RollbackReason.ACCURACY_DEGRADATION
        assert monitor.active_model == "v1.0"

        # Exactly one rollback
        rollback_events = [
            e for e in monitor.audit_log
            if e.event_type == MonitorEventType.ROLLBACK_TRIGGERED
        ]
        assert len(rollback_events) == 1

    def test_healthy_model_not_rolled_back(self, config):
        """
        Scenario: Model performing well throughout.
        Expected: No rollback, healthy evaluations.
        """
        monitor = LiveModelMonitor(config=config)
        monitor.activate_model(
            "v2.0",
            previous_version="v1.0",
            baseline_distribution=[0.33, 0.34, 0.33],
            baseline_accuracy=0.52,
        )

        rng = np.random.default_rng(999)
        for _ in range(100):
            cls = rng.integers(0, 3)
            monitor.record_prediction("v2.0", predicted_class=int(cls), confidence=0.7)
            monitor.record_outcome("v2.0", realized_return=0.004, was_correct=True)

        for _ in range(5):
            monitor.evaluate()

        assert not monitor.has_rolled_back
        assert monitor.active_model == "v2.0"

    def test_no_rollback_without_previous_version(self, config):
        """If no previous version set, rollback cannot occur."""
        monitor = LiveModelMonitor(config=config)
        monitor.activate_model("v1.0", previous_version=None)

        feed_bad_outcomes(monitor, 50, model="v1.0")
        for _ in range(5):
            monitor.evaluate()

        assert not monitor.has_rolled_back


# ──────────────────────────────────────────────────────────────────────────
# Tests: Force Rollback
# ──────────────────────────────────────────────────────────────────────────


class TestForceRollback:
    def test_manual_rollback(self, activated_monitor):
        """Manual rollback works and records correctly."""
        feed_good_outcomes(activated_monitor, 15)
        record = activated_monitor.force_rollback(reason="manual override")

        assert record is not None
        assert record.reason == RollbackReason.MANUAL
        assert activated_monitor.active_model == "v1.0"
        assert activated_monitor.has_rolled_back

    def test_manual_rollback_only_once(self, activated_monitor):
        """Cannot force rollback twice."""
        feed_good_outcomes(activated_monitor, 15)
        record1 = activated_monitor.force_rollback()
        record2 = activated_monitor.force_rollback()

        assert record1 is not None
        assert record2 is None

    def test_no_manual_rollback_without_previous(self, config):
        """Cannot force rollback if no previous version."""
        monitor = LiveModelMonitor(config=config)
        monitor.activate_model("v1.0", previous_version=None)
        record = monitor.force_rollback()
        assert record is None


# ──────────────────────────────────────────────────────────────────────────
# Tests: Performance Metrics
# ──────────────────────────────────────────────────────────────────────────


class TestPerformanceMetrics:
    def test_sharpe_calculation(self, activated_monitor):
        """Sharpe is computed correctly from returns."""
        # Feed consistent positive returns
        for _ in range(30):
            activated_monitor.record_outcome("v2.0", realized_return=0.01, was_correct=True)
            activated_monitor.record_prediction("v2.0", predicted_class=2, confidence=0.7)

        result = activated_monitor.evaluate()
        assert result is not None
        assert result.live_sharpe > 0  # Positive returns → positive Sharpe

    def test_accuracy_calculation(self, activated_monitor):
        """Accuracy is correctly calculated."""
        for _ in range(20):
            activated_monitor.record_prediction("v2.0", predicted_class=2, confidence=0.7)
            activated_monitor.record_outcome("v2.0", realized_return=0.01, was_correct=True)
        for _ in range(10):
            activated_monitor.record_prediction("v2.0", predicted_class=0, confidence=0.5)
            activated_monitor.record_outcome("v2.0", realized_return=-0.01, was_correct=False)

        result = activated_monitor.evaluate()
        assert result is not None
        assert abs(result.rolling_accuracy - 20.0 / 30.0) < 0.01

    def test_psi_zero_when_no_baseline(self, config):
        """PSI is 0 when no baseline distribution is set."""
        monitor = LiveModelMonitor(config=config)
        monitor.activate_model("v2.0", previous_version="v1.0")

        for _ in range(15):
            monitor.record_prediction("v2.0", predicted_class=0, confidence=0.7)
            monitor.record_outcome("v2.0", realized_return=0.01, was_correct=True)

        result = monitor.evaluate()
        assert result is not None
        assert result.psi_score == 0.0

    def test_summary_with_insufficient_data(self, monitor):
        """Summary reports insufficient data when no observations."""
        monitor.activate_model("v2.0", previous_version="v1.0")
        summary = monitor.get_performance_summary()
        assert summary["status"] == "insufficient_data"

"""Tests for Backtest Triggers (Phase 2)."""

import time
from datetime import datetime, timezone

import pytest

from src.scheduler.triggers import (
    DataArrivalTrigger,
    DriftTrigger,
    ManualTrigger,
    ModelTrainedTrigger,
    ParameterChangeTrigger,
    ScheduleTrigger,
    TriggerContext,
)


class TestBacktestTriggers:
    """Test event-driven backtest trigger conditions."""

    def test_data_accumulation_trigger(self):
        """Backtest triggered when new data exceeds threshold."""
        trigger = DataArrivalTrigger(threshold=100)
        context = TriggerContext(accumulated_bars={"AAPL": 50})
        assert trigger.evaluate(context) is False

        context = TriggerContext(accumulated_bars={"AAPL": 101})
        assert trigger.evaluate(context) is True

    def test_data_trigger_resets_after_acknowledge(self):
        """Trigger resets after acknowledging data."""
        trigger = DataArrivalTrigger(threshold=100)
        context = TriggerContext(accumulated_bars={"AAPL": 150})
        assert trigger.evaluate(context) is True

        trigger.acknowledge(context)
        # Same count should not re-trigger
        assert trigger.evaluate(context) is False

        # New data should trigger again
        context2 = TriggerContext(accumulated_bars={"AAPL": 260})
        assert trigger.evaluate(context2) is True

    def test_parameter_change_detection(self):
        """Backtest triggered when strategy parameters change."""
        trigger = ParameterChangeTrigger(component="momentum_strategy")
        ctx1 = TriggerContext(parameter_hashes={"momentum_strategy": "abc123"})
        trigger.evaluate(ctx1)  # Initialize

        # Same params
        assert trigger.evaluate(ctx1) is False

        # Changed params
        ctx2 = TriggerContext(parameter_hashes={"momentum_strategy": "def456"})
        assert trigger.evaluate(ctx2) is True

    def test_feature_version_change(self):
        """Backtest triggered when feature store version bumps."""
        trigger = ParameterChangeTrigger(component="feature_store")
        ctx1 = TriggerContext(parameter_hashes={"feature_store": "v1.0"})
        trigger.evaluate(ctx1)

        ctx2 = TriggerContext(parameter_hashes={"feature_store": "v1.1"})
        assert trigger.evaluate(ctx2) is True

    def test_model_retrained_trigger(self):
        """Backtest triggered after model retraining completes."""
        trigger = ModelTrainedTrigger()
        t1 = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
        ctx1 = TriggerContext(last_model_trained=t1)
        trigger.evaluate(ctx1)  # Initialize

        # Same training time
        assert trigger.evaluate(ctx1) is False

        # New training completed
        t2 = datetime(2024, 6, 2, 12, 0, tzinfo=timezone.utc)
        ctx2 = TriggerContext(last_model_trained=t2)
        assert trigger.evaluate(ctx2) is True

    def test_drift_degradation_trigger(self):
        """Backtest triggered when drift monitor detects degradation."""
        trigger = DriftTrigger(threshold=0.12)

        ctx_normal = TriggerContext(drift_scores={"AAPL": 0.05, "MSFT": 0.08})
        assert trigger.evaluate(ctx_normal) is False

        ctx_drifted = TriggerContext(drift_scores={"AAPL": 0.05, "MSFT": 0.15})
        assert trigger.evaluate(ctx_drifted) is True

    def test_scheduled_fallback_trigger(self):
        """Backtest triggered on schedule as fallback."""
        trigger = ScheduleTrigger(interval_seconds=0.05)
        ctx = TriggerContext()
        trigger.evaluate(ctx)  # Initialize
        time.sleep(0.06)
        assert trigger.evaluate(ctx) is True

    def test_scheduled_trigger_resets(self):
        """Schedule trigger resets after firing."""
        trigger = ScheduleTrigger(interval_seconds=0.05)
        ctx = TriggerContext()
        trigger.evaluate(ctx)
        time.sleep(0.06)
        assert trigger.evaluate(ctx) is True
        trigger.reset()
        # Should not fire immediately after reset
        assert trigger.evaluate(ctx) is False

    def test_multiple_triggers_or_logic(self):
        """Activity fires when ANY trigger is met (OR logic)."""
        from src.scheduler.activity_graph import ActivityNode

        data_trigger = DataArrivalTrigger(threshold=100)
        drift_trigger = DriftTrigger(threshold=0.12)
        schedule_trigger = ScheduleTrigger(interval_seconds=3600)

        node = ActivityNode(
            name="backtest",
            callable=lambda: None,
            triggers=[data_trigger, drift_trigger, schedule_trigger],
        )

        # Only drift is met
        ctx = TriggerContext(
            accumulated_bars={"AAPL": 50},  # Below threshold
            drift_scores={"AAPL": 0.20},  # Above threshold
        )
        assert node.should_trigger(ctx) is True

    def test_disabled_activity_never_triggers(self):
        """Disabled activities don't trigger regardless of conditions."""
        from src.scheduler.activity_graph import ActivityNode

        trigger = ManualTrigger()
        trigger.activate()

        node = ActivityNode(
            name="disabled_test",
            callable=lambda: None,
            triggers=[trigger],
            enabled=False,
        )
        ctx = TriggerContext()
        assert node.should_trigger(ctx) is False

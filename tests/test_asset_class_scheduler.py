"""Tests for Asset-Class Scheduler (Phase 1)."""

import time
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.scheduler.activity_graph import ActivityGraph, ActivityNode, ActivityStatus, AssetClass
from src.scheduler.asset_class_scheduler import AssetClassScheduler
from src.scheduler.market_status import MarketStatusService, MarketPhase
from src.scheduler.triggers import (
    DataArrivalTrigger,
    DriftTrigger,
    ManualTrigger,
    ModelTrainedTrigger,
    ParameterChangeTrigger,
    ScheduleTrigger,
    TriggerContext,
    compute_parameter_hash,
)


class TestTriggers:
    """Test trigger evaluation logic."""

    def test_data_arrival_trigger_not_met(self):
        trigger = DataArrivalTrigger(threshold=100)
        context = TriggerContext(accumulated_bars={"AAPL": 50})
        assert trigger.evaluate(context) is False

    def test_data_arrival_trigger_met(self):
        trigger = DataArrivalTrigger(threshold=100)
        context = TriggerContext(accumulated_bars={"AAPL": 150})
        assert trigger.evaluate(context) is True

    def test_data_arrival_trigger_specific_symbol(self):
        trigger = DataArrivalTrigger(threshold=100, symbol="AAPL")
        context = TriggerContext(accumulated_bars={"AAPL": 50, "MSFT": 200})
        assert trigger.evaluate(context) is False

    def test_drift_trigger_below_threshold(self):
        trigger = DriftTrigger(threshold=0.12)
        context = TriggerContext(drift_scores={"AAPL": 0.05})
        assert trigger.evaluate(context) is False

    def test_drift_trigger_above_threshold(self):
        trigger = DriftTrigger(threshold=0.12)
        context = TriggerContext(drift_scores={"AAPL": 0.15})
        assert trigger.evaluate(context) is True

    def test_schedule_trigger_initial(self):
        trigger = ScheduleTrigger(interval_seconds=1)
        context = TriggerContext()
        # First evaluation initializes
        assert trigger.evaluate(context) is False

    def test_schedule_trigger_elapsed(self):
        trigger = ScheduleTrigger(interval_seconds=0.01)
        context = TriggerContext()
        trigger.evaluate(context)  # Initialize
        time.sleep(0.02)
        assert trigger.evaluate(context) is True

    def test_parameter_change_trigger(self):
        trigger = ParameterChangeTrigger(component="strategy")
        context1 = TriggerContext(parameter_hashes={"strategy": "hash_a"})
        trigger.evaluate(context1)  # Initialize

        context2 = TriggerContext(parameter_hashes={"strategy": "hash_b"})
        assert trigger.evaluate(context2) is True

    def test_parameter_change_no_change(self):
        trigger = ParameterChangeTrigger(component="strategy")
        context = TriggerContext(parameter_hashes={"strategy": "hash_a"})
        trigger.evaluate(context)
        assert trigger.evaluate(context) is False

    def test_manual_trigger(self):
        trigger = ManualTrigger()
        context = TriggerContext()
        assert trigger.evaluate(context) is False
        trigger.activate()
        assert trigger.evaluate(context) is True
        trigger.reset()
        assert trigger.evaluate(context) is False

    def test_model_trained_trigger(self):
        trigger = ModelTrainedTrigger()
        t1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 2, tzinfo=timezone.utc)

        context1 = TriggerContext(last_model_trained=t1)
        trigger.evaluate(context1)  # Initialize

        context2 = TriggerContext(last_model_trained=t2)
        assert trigger.evaluate(context2) is True

    def test_compute_parameter_hash(self):
        params = {"lr": 0.01, "epochs": 100}
        h1 = compute_parameter_hash(params)
        h2 = compute_parameter_hash(params)
        assert h1 == h2

        params2 = {"lr": 0.02, "epochs": 100}
        h3 = compute_parameter_hash(params2)
        assert h1 != h3


class TestActivityGraph:
    """Test activity graph DAG execution."""

    def test_add_activity(self):
        graph = ActivityGraph()
        node = ActivityNode(name="test", callable=lambda: "done")
        graph.add_activity(node)
        assert len(graph.activities) == 1

    def test_duplicate_activity_raises(self):
        graph = ActivityGraph()
        node = ActivityNode(name="test", callable=lambda: None)
        graph.add_activity(node)
        with pytest.raises(ValueError):
            graph.add_activity(node)

    def test_get_ready_activities_with_triggers(self):
        graph = ActivityGraph()
        trigger = DataArrivalTrigger(threshold=50)
        node = ActivityNode(name="bt", callable=lambda: None, triggers=[trigger])
        graph.add_activity(node)

        context = TriggerContext(accumulated_bars={"AAPL": 100})
        ready = graph.get_ready_activities(context)
        assert len(ready) == 1
        assert ready[0].name == "bt"

    def test_dependency_ordering(self):
        graph = ActivityGraph()
        trigger = ManualTrigger()
        trigger.activate()

        node_a = ActivityNode(name="A", callable=lambda: None, triggers=[trigger])
        node_b = ActivityNode(
            name="B", callable=lambda: None, triggers=[trigger], dependencies=["A"]
        )
        graph.add_activity(node_a)
        graph.add_activity(node_b)

        context = TriggerContext()
        ready = graph.get_ready_activities(context)
        # B should not be ready because A hasn't completed
        assert all(r.name != "B" for r in ready)

    def test_execute_activity_success(self):
        graph = ActivityGraph()
        trigger = ManualTrigger()
        trigger.activate()
        node = ActivityNode(name="test", callable=lambda: "result", triggers=[trigger])
        graph.add_activity(node)

        context = TriggerContext()
        result = graph.execute_activity(node, context)
        assert result.status == ActivityStatus.COMPLETED
        assert result.result == "result"

    def test_execute_activity_failure(self):
        graph = ActivityGraph()
        trigger = ManualTrigger()
        trigger.activate()

        def failing():
            raise ValueError("oops")

        node = ActivityNode(name="fail", callable=failing, triggers=[trigger])
        graph.add_activity(node)

        context = TriggerContext()
        result = graph.execute_activity(node, context)
        assert result.status == ActivityStatus.FAILED
        assert "oops" in result.error

    def test_execute_activity_skips_when_already_running(self):
        graph = ActivityGraph()
        trigger = ManualTrigger()
        trigger.activate()
        started = threading.Event()
        release = threading.Event()
        call_count = {"n": 0}

        def slow_callable():
            call_count["n"] += 1
            started.set()
            release.wait(timeout=1.0)
            return "ok"

        node = ActivityNode(name="slow", callable=slow_callable, triggers=[trigger])
        graph.add_activity(node)
        context = TriggerContext()
        first_result = {}

        def _run_first():
            first_result["value"] = graph.execute_activity(node, context)

        thread = threading.Thread(target=_run_first)
        thread.start()
        assert started.wait(timeout=1.0)

        second_result = graph.execute_activity(node, context)
        release.set()
        thread.join(timeout=1.0)

        assert first_result["value"].status == ActivityStatus.COMPLETED
        assert second_result.status == ActivityStatus.SKIPPED
        assert second_result.error == "activity_already_running"
        assert call_count["n"] == 1

    def test_asset_class_gating(self):
        graph = ActivityGraph()
        trigger = ManualTrigger()
        trigger.activate()

        equity_node = ActivityNode(
            name="eq", callable=lambda: None, triggers=[trigger], asset_class=AssetClass.EQUITY
        )
        crypto_node = ActivityNode(
            name="cr", callable=lambda: None, triggers=[trigger], asset_class=AssetClass.CRYPTO
        )
        graph.add_activity(equity_node)
        graph.add_activity(crypto_node)

        context = TriggerContext()
        ready = graph.get_ready_activities(context, active_asset_class=AssetClass.EQUITY)
        names = [r.name for r in ready]
        assert "eq" in names
        assert "cr" not in names


class TestMarketStatusService:
    """Test market status service."""

    def test_crypto_always_continuous(self):
        service = MarketStatusService()
        status = service.get_crypto_status()
        assert status.phase == MarketPhase.CONTINUOUS
        assert status.is_trading is True

    def test_equity_symbols(self):
        service = MarketStatusService(equity_symbols=["AAPL", "MSFT"])
        assert service.equity_symbols == ["AAPL", "MSFT"]

    def test_all_symbols(self):
        service = MarketStatusService(
            equity_symbols=["AAPL"], crypto_symbols=["BTC/USD"]
        )
        assert "AAPL" in service.all_symbols
        assert "BTC/USD" in service.all_symbols

    def test_is_any_market_open(self):
        service = MarketStatusService()
        # Crypto is always open
        assert service.is_any_market_open() is True


class TestAssetClassScheduler:
    """Test the top-level scheduler."""

    def test_initialization(self):
        scheduler = AssetClassScheduler()
        assert scheduler.is_running is False

    def test_register_activity(self):
        scheduler = AssetClassScheduler()
        node = ActivityNode(name="test", callable=lambda: None, triggers=[ManualTrigger()])
        scheduler.register_activity(node)
        status = scheduler.get_status()
        assert status["graph"]["total_activities"] == 1

    def test_manual_trigger(self):
        executed = []
        scheduler = AssetClassScheduler()
        trigger = ManualTrigger()
        trigger.activate()
        node = ActivityNode(
            name="manual_test",
            callable=lambda: executed.append(True),
            triggers=[trigger],
        )
        scheduler.register_activity(node)
        result = scheduler.trigger_manual("manual_test")
        assert result is not None
        assert result.status == ActivityStatus.COMPLETED

    def test_tick_executes_ready(self):
        executed = []
        scheduler = AssetClassScheduler()
        trigger = DataArrivalTrigger(threshold=10)
        node = ActivityNode(
            name="tick_test",
            callable=lambda: executed.append(True),
            triggers=[trigger],
            asset_class=AssetClass.CRYPTO,  # Always active
        )
        scheduler.register_activity(node)

        # Update context
        scheduler._context.accumulated_bars["BTC/USD"] = 50

        results = scheduler.tick()
        assert len(results) == 1
        assert len(executed) == 1

    def test_get_status(self):
        scheduler = AssetClassScheduler()
        status = scheduler.get_status()
        assert "running" in status
        assert "operational_mode" in status
        assert "market_status" in status

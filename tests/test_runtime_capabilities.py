"""Integration test for runtime capabilities."""
import sys
sys.path.insert(0, '.')

import numpy as np
from src.core.events import EventBus, EnvironmentSwitched, BrokerSwitched, ModelSwapped
from src.core.environment import BrokerManager, EnvironmentManager, SwitchState
from src.broker.paper import PaperBroker
from config.settings import TradingMode


# Top-level mock models (must be picklable for joblib)
class MockModelV1:
    """Mock model that always predicts 1."""
    def predict(self, X):
        return np.ones(len(X))
    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])


class MockModelV2:
    """Mock model that always predicts 0."""
    def predict(self, X):
        return np.zeros(len(X))


def test_broker_manager():
    """Test 1: BrokerManager hot-swap"""
    print("=== Test 1: BrokerManager Hot-Swap ===")
    bus = EventBus()
    events_received = []
    bus.subscribe(BrokerSwitched, lambda e: events_received.append(e))

    broker1 = PaperBroker(starting_equity=100000)
    manager = BrokerManager(broker1, event_bus=bus)

    assert manager.broker_name == "paper"
    assert manager.is_paper is True
    print(f"  Initial broker: {manager.broker_name}, paper={manager.is_paper}")

    # Create second paper broker (simulating a swap)
    broker2 = PaperBroker(starting_equity=50000)
    result = manager.switch_broker(broker2, drain_positions=True)
    assert result["success"] is True
    assert len(events_received) == 1
    print(f"  Swap successful, events: {len(events_received)}")
    print("  ✅ BrokerManager hot-swap works!")
    return manager, bus


def test_environment_manager(manager, bus):
    """Test 2: EnvironmentManager"""
    print("\n=== Test 2: EnvironmentManager ===")
    env_events = []
    bus.subscribe(EnvironmentSwitched, lambda e: env_events.append(e))

    def mock_broker_factory(mode):
        return PaperBroker(starting_equity=100000 if mode == TradingMode.PAPER else 200000)

    env_mgr = EnvironmentManager(
        broker_manager=manager,
        broker_factory=mock_broker_factory,
        event_bus=bus,
    )

    print(f"  Current mode: {env_mgr.current_mode.value}")
    assert env_mgr.state == SwitchState.IDLE

    # Switch to paper (force since already paper)
    result = env_mgr.switch_environment(TradingMode.PAPER, force=True, reason="test")
    assert result["success"] is True
    assert env_mgr.current_mode == TradingMode.PAPER
    print(f"  Mode after switch: {env_mgr.current_mode.value}")
    print(f"  Events received: {len(env_events)}")
    print("  ✅ EnvironmentManager works!")

    # Status
    status = env_mgr.get_status()
    assert status["current_mode"] == "paper"
    assert status["state"] == "idle"
    print(f"  Status: mode={status['current_mode']}, state={status['state']}")
    print("  ✅ Status reporting works!")


def test_backtest_runner():
    """Test 3: Multi-strategy backtest runner"""
    print("\n=== Test 3: BacktestRunner ===")
    import numpy as np
    import pandas as pd
    from backtesting.runner import BacktestRunner, BacktestStatus
    from src.strategy.momentum import MomentumStrategy

    bus = EventBus()
    runner = BacktestRunner(max_workers=2, event_bus=bus)

    # Create fake OHLCV data
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    data = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    }, index=dates)

    # Create strategies
    strategy1 = MomentumStrategy(symbols=["TEST"], timeframe="1Hour", lookback=200)

    # Run synchronous multi-strategy backtest
    result = runner.run_multiple(
        strategies=[strategy1],
        data={"TEST": data},
    )

    assert result.status == BacktestStatus.COMPLETED
    print(f"  Run ID: {result.run_id}")
    print(f"  Status: {result.status.value}")
    print(f"  Jobs completed: {len(result.completed_jobs)}")
    print(f"  Duration: {result.duration_seconds:.2f}s")

    if result.completed_jobs:
        job = result.completed_jobs[0]
        print(f"  Strategy: {job.strategy_name}, Return: {job.result.total_return_pct:.2%}")
    print("  ✅ BacktestRunner works!")


def test_model_predictor():
    """Test 4: ModelPredictor hot-swap"""
    print("\n=== Test 4: ModelPredictor ===")
    from src.ml.model_registry import ModelRegistry
    from src.ml.predictor import ModelPredictor
    import tempfile, shutil

    # Create temp registry
    tmpdir = tempfile.mkdtemp()
    registry = ModelRegistry(models_dir=tmpdir)

    # Register first model
    v1 = registry.save_version(
        model=MockModelV1(),
        features=["f1", "f2", "f3"],
        metrics={"accuracy": 0.7},
        symbols=["AAPL"],
    )
    print(f"  Registered {v1}")

    # Create predictor
    predictor = ModelPredictor(registry=registry, auto_reload=False)
    assert predictor.is_loaded
    assert predictor.current_version == v1
    print(f"  Predictor loaded: {predictor.current_version}")

    # Predict
    X = np.random.randn(5, 3)
    preds = predictor.predict(X)
    assert len(preds) == 5
    assert all(p == 1.0 for p in preds)
    print(f"  Predictions (v1): {preds}")

    # Register second model and swap
    v2 = registry.save_version(
        model=MockModelV2(),
        features=["f1", "f2", "f3"],
        metrics={"accuracy": 0.8},
        symbols=["AAPL"],
    )
    print(f"  Registered {v2}")

    result = predictor.swap_model(v2)
    assert predictor.current_version == v2
    print(f"  Swapped to: {predictor.current_version}")

    preds2 = predictor.predict(X)
    assert all(p == 0.0 for p in preds2)
    print(f"  Predictions (v2): {preds2}")

    # Metrics
    metrics = predictor.get_metrics()
    assert metrics["prediction_count"] == 2
    assert metrics["swap_count"] == 1
    print(f"  Metrics: predictions={metrics['prediction_count']}, swaps={metrics['swap_count']}")

    predictor.stop()
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("  ✅ ModelPredictor hot-swap works!")


def test_ab_testing():
    """Test 5: A/B Testing Framework"""
    print("\n=== Test 5: A/B Testing ===")
    import tempfile, shutil
    from src.ml.model_registry import ModelRegistry
    from src.ml.ab_testing import ABTestManager, ABTestMode, ABTestStatus

    tmpdir = tempfile.mkdtemp()
    registry = ModelRegistry(models_dir=tmpdir)

    # Register two models
    v1 = registry.save_version(MockModelV1(), ["f1"], {"acc": 0.6}, ["AAPL"])
    v2 = registry.save_version(MockModelV2(), ["f1"], {"acc": 0.7}, ["AAPL"])
    # v2 is now active; rollback to v1 to make it champion
    registry.rollback(v1)

    ab = ABTestManager(registry=registry)

    # Start test
    test_id = ab.start_test(
        challenger_version=v2,
        mode=ABTestMode.SHADOW,
        min_trades=5,
        auto_promote=False,
    )
    print(f"  Test started: {test_id}")

    status = ab.get_test_status()
    assert status["status"] == "active"
    print(f"  Status: {status['status']}")

    # Record trades
    for i in range(6):
        ab.record_trade(v1, {"pnl": 10.0 + i, "pnl_pct": 0.01 + i*0.001, "symbol": "AAPL"})
        ab.record_trade(v2, {"pnl": 15.0 + i, "pnl_pct": 0.015 + i*0.001, "symbol": "AAPL"})

    status = ab.get_test_status()
    print(f"  After trades - Champion: {status['champion']['total_trades']} trades, "
          f"Challenger: {status['challenger']['total_trades']} trades")

    shutil.rmtree(tmpdir, ignore_errors=True)
    print("  ✅ A/B Testing works!")


def test_runtime_manager():
    """Test 6: RuntimeManager unified interface"""
    print("\n=== Test 6: RuntimeManager ===")
    from src.core.runtime import RuntimeManager
    from src.core.environment import BrokerManager

    bus = EventBus()
    broker = PaperBroker(starting_equity=100000)
    broker_mgr = BrokerManager(broker, event_bus=bus)

    runtime = RuntimeManager(
        broker_manager=broker_mgr,
        event_bus=bus,
    )

    # Check status
    status = runtime.get_status()
    assert status["environment"]["mode"] == "paper"
    assert status["capabilities"]["env_switching"] is True
    assert status["capabilities"]["model_hot_swap"] is True
    print(f"  Mode: {status['environment']['mode']}")
    print(f"  Capabilities: {list(status['capabilities'].keys())}")

    # Check properties
    assert runtime.is_paper is True
    assert runtime.is_live is False
    assert runtime.current_mode == "paper"
    print(f"  is_paper={runtime.is_paper}, current_mode={runtime.current_mode}")

    runtime.shutdown()
    print("  ✅ RuntimeManager works!")


if __name__ == "__main__":
    test_broker_manager_result = test_broker_manager()
    test_environment_manager(*test_broker_manager_result)
    test_backtest_runner()
    test_model_predictor()
    test_ab_testing()
    test_runtime_manager()
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✅")
    print("="*60)

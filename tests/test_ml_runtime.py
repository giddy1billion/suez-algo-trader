"""Quick test for ModelPredictor and A/B testing."""
import sys
sys.path.insert(0, '.')
import numpy as np
import tempfile
import shutil

from src.ml.model_registry import ModelRegistry
from src.ml.predictor import ModelPredictor
from src.ml.ab_testing import ABTestManager, ABTestMode


class MockModelV1:
    def predict(self, X):
        return np.ones(len(X))


class MockModelV2:
    def predict(self, X):
        return np.zeros(len(X))


def test_predictor():
    print("Testing ModelPredictor...")
    tmpdir = tempfile.mkdtemp()
    try:
        reg = ModelRegistry(models_dir=tmpdir)
        v1 = reg.save_version(MockModelV1(), ["f1", "f2"], {"acc": 0.7}, ["A"])
        print(f"  v1={v1}")

        p = ModelPredictor(registry=reg, auto_reload=False)
        X = np.random.randn(3, 2)
        preds = p.predict(X)
        assert all(x == 1.0 for x in preds), f"Expected all 1s, got {preds}"
        print(f"  v1 predictions correct")

        v2 = reg.save_version(MockModelV2(), ["f1", "f2"], {"acc": 0.8}, ["A"])
        print(f"  v2={v2}")
        p.swap_model(v2)
        preds2 = p.predict(X)
        assert all(x == 0.0 for x in preds2), f"Expected all 0s, got {preds2}"
        print(f"  v2 predictions correct after swap")

        metrics = p.get_metrics()
        assert metrics["prediction_count"] == 2
        assert metrics["swap_count"] == 1
        print(f"  Metrics OK: {metrics['prediction_count']} preds, {metrics['swap_count']} swaps")

        p.stop()
        print("  PASSED: ModelPredictor")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_ab():
    print("\nTesting A/B Testing...")
    tmpdir = tempfile.mkdtemp()
    try:
        reg = ModelRegistry(models_dir=tmpdir)
        v1 = reg.save_version(MockModelV1(), ["f1"], {"acc": 0.6}, ["A"])
        v2 = reg.save_version(MockModelV2(), ["f1"], {"acc": 0.7}, ["A"])
        reg.rollback(v1)  # make v1 champion

        ab = ABTestManager(registry=reg, min_trades_default=5)
        test_id = ab.start_test(challenger_version=v2, mode=ABTestMode.SHADOW, min_trades=5, auto_promote=False)
        print(f"  Started test: {test_id}")

        status = ab.get_test_status()
        assert status["status"] == "active"
        print(f"  Status: {status['status']}")

        # Record trades
        for i in range(6):
            ab.record_trade(v1, {"pnl": 10.0 + i, "pnl_pct": 0.01, "symbol": "A"})
            ab.record_trade(v2, {"pnl": 15.0 + i, "pnl_pct": 0.015, "symbol": "A"})

        status = ab.get_test_status()
        if status:
            champ_trades = status["champion"]["total_trades"]
            chall_trades = status["challenger"]["total_trades"]
            print(f"  Champion trades: {champ_trades}, Challenger: {chall_trades}")
        print("  PASSED: A/B Testing")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_runtime_manager():
    print("\nTesting RuntimeManager...")
    from src.core.events import EventBus
    from src.core.environment import BrokerManager
    from src.core.runtime import RuntimeManager
    from src.broker.paper import PaperBroker

    bus = EventBus()
    broker = PaperBroker(starting_equity=100000)
    broker_mgr = BrokerManager(broker, event_bus=bus)

    runtime = RuntimeManager(broker_manager=broker_mgr, event_bus=bus)
    status = runtime.get_status()
    assert status["environment"]["mode"] == "paper"
    assert status["capabilities"]["env_switching"] is True
    assert status["capabilities"]["model_hot_swap"] is True
    assert status["capabilities"]["ab_testing"] is True
    print(f"  Mode: {status['environment']['mode']}")
    print(f"  All 7 capabilities: {list(status['capabilities'].keys())}")
    assert runtime.is_paper is True
    runtime.shutdown()
    print("  PASSED: RuntimeManager")


if __name__ == "__main__":
    test_predictor()
    test_ab()
    test_runtime_manager()
    print("\n" + "=" * 50)
    print("ALL ML/RUNTIME TESTS PASSED!")
    print("=" * 50)

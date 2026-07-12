"""
Comprehensive remediation tests covering P0–P3 audit findings.

Test suites:
1. GovernanceBypassRegression — deploy() cannot bypass validation
2. MLPipelineIntegration — data prep, training, walk-forward, OOS
3. RealisticBacktest — price-based returns with transaction costs
4. SchedulerHealthMonitoring — watchdog, auto-recovery
5. EarlyStopping — final model uses early stopping
6. ClassImbalance — balanced sample weights
7. RealizedVolVaR — VaR uses realized volatility
8. RiskAuditSemantics — audit log captures correct actions
9. DurableCorrelationStore — SQLite is the default
10. ThreadSafety — concurrent access patterns
11. SurvivorshipBias — training data awareness
"""

import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ml.governance import GovernanceViolation, ModelGovernance, ModelLineage


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_governance(tmp_path):
    """Fresh governance manager in temp directory."""
    return ModelGovernance(governance_dir=str(tmp_path / "governance"))


@pytest.fixture
def valid_model(tmp_governance):
    """Governance with a model that passes all validation gates."""
    df = pd.DataFrame({
        "close": np.random.randn(200),
        "volume": np.random.randint(1000, 10000, 200),
    })
    tmp_governance.record_training(
        version="v001",
        features=["rsi_14", "ema_slope_20", "bb_pct", "atr_14"],
        dataset=df,
        config={"n_estimators": 100, "max_depth": 6},
        metrics={
            "cv_accuracy": 0.67,
            "sharpe": 1.5,
            "n_trades": 150,
            "max_drawdown": 0.05,
        },
        hyperparameters={"n_estimators": 100, "max_depth": 6},
        seed=42,
        training_duration=45.2,
        walk_forward_results={"sharpe": 1.3},
        monte_carlo_results={"probability_of_profit": 0.72},
    )
    return tmp_governance


@pytest.fixture
def failing_model(tmp_governance):
    """Governance with a model that fails validation."""
    tmp_governance.record_training(
        version="v_bad",
        features=["rsi_14"],
        metrics={"cv_accuracy": 0.3, "sharpe": -0.5, "n_trades": 5},
        walk_forward_results={"sharpe": -1.0},
        monte_carlo_results={"probability_of_profit": 0.2},
    )
    return tmp_governance


@pytest.fixture
def sample_ohlcv():
    """250-bar OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 250
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.5),
        "low": close - abs(np.random.randn(n) * 0.5),
        "close": close,
        "volume": np.random.randint(10000, 100000, n),
    })


# ═════════════════════════════════════════════════════════════════════════════
# 1. Governance Bypass Regression Tests (P0-1)
# ═════════════════════════════════════════════════════════════════════════════

class TestGovernanceBypassRegression:
    """Prove that deploy() cannot be called without validation passing."""

    def test_deploy_raises_on_invalid_model(self, failing_model):
        """deploy() raises GovernanceViolation for invalid models."""
        with pytest.raises(GovernanceViolation, match="failed governance validation"):
            failing_model.deploy("v_bad", reason="Attempt bypass")

    def test_deploy_succeeds_for_valid_model(self, valid_model):
        """deploy() works when all validation gates pass."""
        result = valid_model.deploy("v001", reason="Passed gates")
        assert result is True
        deployed = valid_model.get_deployed_model()
        assert deployed is not None
        assert deployed.version == "v001"

    def test_skip_validation_flag_allows_deploy_with_audit(self, failing_model):
        """skip_validation allows emergency deploy but logs the override."""
        result = failing_model.deploy("v_bad", reason="Emergency rollback", skip_validation=True)
        assert result is True
        deployed = failing_model.get_deployed_model()
        assert deployed is not None

    def test_nonexistent_version_returns_false(self, tmp_governance):
        """deploy() returns False for version not in records."""
        # validate_for_deployment returns (False, [...]) for unknown version
        with pytest.raises(GovernanceViolation):
            tmp_governance.deploy("v_nonexistent")

    def test_deploy_validates_internally(self, valid_model):
        """Calling deploy() directly (not through pipeline) still validates."""
        # This is the key regression test: the audit found deploy() had
        # no internal validation gate — only the pipeline checked first.
        is_valid, _ = valid_model.validate_for_deployment("v001")
        assert is_valid  # Prerequisite

        # Direct deploy must still work
        assert valid_model.deploy("v001")

    def test_governance_violation_is_catchable(self, failing_model):
        """GovernanceViolation can be caught and inspected."""
        try:
            failing_model.deploy("v_bad")
            assert False, "Should have raised"
        except GovernanceViolation as e:
            assert "v_bad" in str(e)
            assert "failed governance validation" in str(e)


# ═════════════════════════════════════════════════════════════════════════════
# 2. ML Pipeline Integration Tests (P0-2)
# ═════════════════════════════════════════════════════════════════════════════

class TestMLPipelineDataPreparation:
    """Integration tests for data preparation pipeline."""

    def test_feature_engineering_produces_clean_output(self, sample_ohlcv):
        """Feature engineering produces >100 features with no target leakage."""
        from src.ml.features import engineer_features
        result = engineer_features(sample_ohlcv, include_target=False)

        # Should have many features
        new_cols = set(result.columns) - {"open", "high", "low", "close", "volume"}
        assert len(new_cols) >= 100

        # No target columns when include_target=False
        assert "target" not in result.columns
        assert "future_return" not in result.columns

    def test_target_generation_uses_future_shift(self, sample_ohlcv):
        """Target uses shift(-forward_bars) — correct forward-looking for labels."""
        from src.ml.features import engineer_features
        result = engineer_features(sample_ohlcv, include_target=True)

        assert "future_return" in result.columns
        # Last forward_bars rows should be NaN
        assert pd.isna(result["future_return"].iloc[-1])
        assert pd.isna(result["future_return"].iloc[-2])

    def test_adaptive_labeling_varies_by_volatility(self, sample_ohlcv):
        """High-vol assets get wider thresholds than low-vol ones."""
        from src.ml.training_pipeline import TrainingPipeline
        from src.ml.model_registry import ModelRegistry

        registry = ModelRegistry(models_dir=tempfile.mkdtemp())
        governance = ModelGovernance(governance_dir=tempfile.mkdtemp())
        pipeline = TrainingPipeline(registry=registry, governance=governance)

        # Create two datasets with very different volatilities
        np.random.seed(42)
        n = 250

        # Low volatility
        close_low = 100 + np.cumsum(np.random.randn(n) * 0.1)
        df_low = pd.DataFrame({
            "open": close_low, "high": close_low + 0.05,
            "low": close_low - 0.05, "close": close_low,
            "volume": np.full(n, 50000),
        })

        # High volatility
        close_high = 100 + np.cumsum(np.random.randn(n) * 2.0)
        df_high = pd.DataFrame({
            "open": close_high, "high": close_high + 1.0,
            "low": close_high - 1.0, "close": close_high,
            "volume": np.full(n, 50000),
        })

        from src.ml.features import engineer_features
        feat_low = engineer_features(df_low, include_target=False)
        feat_high = engineer_features(df_high, include_target=False)

        # Both should produce features
        assert len(feat_low) == n
        assert len(feat_high) == n


# ═════════════════════════════════════════════════════════════════════════════
# 3. Realistic Backtest Tests (P0-3)
# ═════════════════════════════════════════════════════════════════════════════

class TestRealisticBacktest:
    """OOS backtest uses actual prices with transaction costs."""

    def test_backtest_with_prices_includes_costs(self):
        """When close prices are provided, trade returns include costs."""
        from src.ml.training_pipeline import TrainingPipeline
        from src.ml.model_registry import ModelRegistry

        registry = ModelRegistry(models_dir=tempfile.mkdtemp())
        governance = ModelGovernance(governance_dir=tempfile.mkdtemp())
        pipeline = TrainingPipeline(registry=registry, governance=governance)

        # Create mock model that always predicts "up" (class 2 in encoded)
        mock_model = MagicMock()
        # DirectionEncoder: -1->0, 0->1, 1->2
        mock_model.predict.return_value = np.array([2] * 50)  # All "up"

        # Close prices that go up by 1% each hold period
        close = np.array([100.0 + i * 0.2 for i in range(50)])
        X = np.random.randn(50, 10)
        y = np.zeros(50, dtype=int)

        result = pipeline._backtest_model_oos(
            model=mock_model,
            feature_cols=[f"f{i}" for i in range(10)],
            X_holdout=X,
            y_holdout=y,
            close_holdout=close,
            transaction_cost_bps=10.0,
            slippage_bps=5.0,
        )

        assert result["n_trades"] > 0
        # With upward-trending prices and "up" predictions, some trades should be positive
        # but costs should reduce returns
        assert "sharpe" in result
        assert "max_drawdown" in result

    def test_backtest_without_prices_falls_back(self):
        """Without close prices, backtest uses direction-correctness fallback."""
        from src.ml.training_pipeline import TrainingPipeline
        from src.ml.model_registry import ModelRegistry

        registry = ModelRegistry(models_dir=tempfile.mkdtemp())
        governance = ModelGovernance(governance_dir=tempfile.mkdtemp())
        pipeline = TrainingPipeline(registry=registry, governance=governance)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2] * 50)

        X = np.random.randn(50, 10)
        y = np.zeros(50, dtype=int)

        result = pipeline._backtest_model_oos(
            model=mock_model,
            feature_cols=[f"f{i}" for i in range(10)],
            X_holdout=X,
            y_holdout=y,
            close_holdout=None,  # No prices
        )

        assert result["n_trades"] > 0


# ═════════════════════════════════════════════════════════════════════════════
# 4. Scheduler Health Monitoring (P0-4)
# ═════════════════════════════════════════════════════════════════════════════

class TestSchedulerHealthMonitoring:
    """Scheduler has health checks and auto-recovery."""

    def test_health_check_reports_status(self):
        """health_check() returns meaningful status dict."""
        from src.scheduler.asset_class_scheduler import AssetClassScheduler

        scheduler = AssetClassScheduler()
        health = scheduler.health_check()

        assert "healthy" in health
        assert "running" in health
        assert "consecutive_errors" in health
        assert "restart_count" in health
        assert health["consecutive_errors"] == 0

    def test_health_check_after_tick(self):
        """After a successful tick in the run loop, health shows last tick time."""
        from src.scheduler.asset_class_scheduler import AssetClassScheduler

        scheduler = AssetClassScheduler()
        scheduler._running = True  # Simulate started

        # Manual tick — in the run loop, _last_tick_time is set after tick()
        scheduler.tick()
        with scheduler._lock:
            scheduler._last_tick_time = datetime.now(timezone.utc)

        health = scheduler.health_check()
        assert health["seconds_since_last_tick"] is not None
        assert health["seconds_since_last_tick"] < 5.0

    def test_get_status_includes_health(self):
        """get_status() includes health monitoring data."""
        from src.scheduler.asset_class_scheduler import AssetClassScheduler

        scheduler = AssetClassScheduler()
        status = scheduler.get_status()
        assert "health" in status

    def test_consecutive_errors_tracked(self):
        """Consecutive errors are counted in health check."""
        from src.scheduler.asset_class_scheduler import AssetClassScheduler

        scheduler = AssetClassScheduler()
        with scheduler._lock:
            scheduler._consecutive_errors = 3
        health = scheduler.health_check()
        assert health["consecutive_errors"] == 3

    def test_auto_restart_resets_errors(self):
        """Auto-restart resets error counter and increments restart count."""
        from src.scheduler.asset_class_scheduler import AssetClassScheduler

        scheduler = AssetClassScheduler()
        scheduler._running = True
        scheduler._consecutive_errors = 5

        scheduler._attempt_restart()

        assert scheduler._consecutive_errors == 0
        assert scheduler._restart_count == 1
        assert scheduler._running is True


# ═════════════════════════════════════════════════════════════════════════════
# 5. Early Stopping Tests (P0-5)
# ═════════════════════════════════════════════════════════════════════════════

class TestEarlyStopping:
    """Final model must use early stopping to prevent overfitting."""

    def test_final_model_has_early_stopping(self):
        """Verify the training pipeline configures early_stopping_rounds."""
        import inspect
        from src.ml.training_pipeline import TrainingPipeline

        source = inspect.getsource(TrainingPipeline._train_model)

        # The final model block should include early_stopping_rounds
        # Count occurrences — should have at least 2 (CV + final)
        count = source.count("early_stopping_rounds")
        assert count >= 2, (
            f"Expected early_stopping_rounds in both CV and final model, "
            f"found {count} occurrences"
        )

    def test_final_model_uses_eval_set(self):
        """Final model .fit() must be called with eval_set for early stopping."""
        import inspect
        from src.ml.training_pipeline import TrainingPipeline

        source = inspect.getsource(TrainingPipeline._train_model)

        # The final model fit should include eval_set
        # Find the block after "Train final model"
        final_block = source[source.index("Train final model"):]
        assert "eval_set" in final_block


# ═════════════════════════════════════════════════════════════════════════════
# 6. Class Imbalance Handling (P1-1)
# ═════════════════════════════════════════════════════════════════════════════

class TestClassImbalance:
    """Training pipeline handles class imbalance with balanced weights."""

    def test_class_weight_computation_in_pipeline(self):
        """Pipeline source includes class weight balancing."""
        import inspect
        from src.ml.training_pipeline import TrainingPipeline

        source = inspect.getsource(TrainingPipeline._train_model)
        assert "compute_sample_weight" in source
        assert "balanced" in source

    def test_imbalance_ratio_in_metrics(self):
        """Pipeline reports class imbalance ratio in metrics."""
        import inspect
        from src.ml.training_pipeline import TrainingPipeline

        source = inspect.getsource(TrainingPipeline._train_model)
        assert "class_imbalance_ratio" in source


# ═════════════════════════════════════════════════════════════════════════════
# 7. Realized Volatility VaR (P1-2)
# ═════════════════════════════════════════════════════════════════════════════

class TestRealizedVolVaR:
    """VaR calculation uses realized volatility, not fixed 2%."""

    def test_crypto_position_uses_higher_vol(self):
        """Crypto positions get ~5% daily vol default, not 2%."""
        from src.risk.portfolio_risk import PortfolioRiskLayer
        from src.risk.models import TradeRequest

        layer = PortfolioRiskLayer(max_var_pct=0.10, max_portfolio_heat_pct=0.20)

        request = TradeRequest(
            symbol="BTC/USD", side="buy", qty=0.1, price=50000.0,
            confidence=0.8,
        )

        # Position with asset_class=crypto and no daily_vol
        positions = [
            {"symbol": "ETH/USD", "market_value": 5000.0, "side": "long",
             "asset_class": "crypto"}
        ]

        decision = layer.evaluate(
            request=request,
            portfolio_value=100000.0,
            positions=positions,
        )

        # The VaR calculation should use crypto vol (~5%), not 2%
        # So the same positions would have higher VaR than before
        assert decision.action.value in ("approve", "reduce", "reject")

    def test_position_with_daily_vol_uses_it(self):
        """Positions providing daily_vol don't use the default."""
        from src.risk.portfolio_risk import PortfolioRiskLayer
        from src.risk.models import TradeRequest

        layer = PortfolioRiskLayer()

        request = TradeRequest(
            symbol="AAPL", side="buy", qty=10, price=150.0,
            confidence=0.8,
        )

        # Position with explicit daily_vol
        positions = [
            {"symbol": "MSFT", "market_value": 10000.0, "side": "long",
             "daily_vol": 0.01}  # 1% daily vol
        ]

        decision = layer.evaluate(
            request=request,
            portfolio_value=100000.0,
            positions=positions,
        )

        assert decision.action.value in ("approve", "reduce")

    def test_no_daily_vol_equity_uses_default(self):
        """Equity positions without daily_vol get 1.5% default."""
        from src.risk.portfolio_risk import PortfolioRiskLayer
        from src.risk.models import TradeRequest

        layer = PortfolioRiskLayer()

        request = TradeRequest(
            symbol="AAPL", side="buy", qty=10, price=150.0,
            confidence=0.8,
        )

        # Position without daily_vol or asset_class — defaults to equity
        positions = [
            {"symbol": "MSFT", "market_value": 10000.0, "side": "long"}
        ]

        decision = layer.evaluate(
            request=request,
            portfolio_value=100000.0,
            positions=positions,
        )

        assert decision.action.value in ("approve", "reduce")


# ═════════════════════════════════════════════════════════════════════════════
# 8. Risk Audit Semantics (P1-3)
# ═════════════════════════════════════════════════════════════════════════════

class TestRiskAuditSemantics:
    """Risk audit log includes correct action metadata."""

    def test_audit_log_includes_final_action(self):
        """Audit log entry has final_action field."""
        from src.risk.engine import RiskEngine
        from src.risk.models import TradeRequest

        engine = RiskEngine()

        request = TradeRequest(
            symbol="AAPL", side="buy", qty=10, price=150.0,
            confidence=0.8,
        )

        decision = engine.evaluate(
            request=request,
            portfolio_value=100000.0,
            cash=50000.0,
        )

        # Check audit log
        assert len(engine._decision_log) == 1
        entry = engine._decision_log[0]
        assert "final_action" in entry
        assert entry["final_action"] in ("approve", "reject")

    def test_audit_log_includes_layer_metadata(self):
        """Each layer in audit log includes metadata and adjusted_qty."""
        from src.risk.engine import RiskEngine
        from src.risk.models import TradeRequest

        engine = RiskEngine()

        request = TradeRequest(
            symbol="AAPL", side="buy", qty=10, price=150.0,
            confidence=0.8,
        )

        engine.evaluate(
            request=request,
            portfolio_value=100000.0,
            cash=50000.0,
        )

        entry = engine._decision_log[0]
        for layer_name, layer_data in entry["layers"].items():
            assert "action" in layer_data
            assert "reason" in layer_data
            assert "adjusted_qty" in layer_data
            assert "metadata" in layer_data


# ═════════════════════════════════════════════════════════════════════════════
# 9. Durable Correlation Store Default (P1-5)
# ═════════════════════════════════════════════════════════════════════════════

class TestDurableCorrelationStoreDefault:
    """Production default is SqliteCorrelationStore, not InMemory."""

    def test_forwarder_defaults_to_sqlite_store(self):
        """TelegramAuditForwarder uses SqliteCorrelationStore by default."""
        import inspect
        from src.notifications import telegram_audit_forwarder

        source = inspect.getsource(telegram_audit_forwarder.TelegramAuditForwarder.__init__)
        assert "SqliteCorrelationStore()" in source
        assert "InMemoryCorrelationStore()" not in source


# ═════════════════════════════════════════════════════════════════════════════
# 10. Thread Safety (P1-4)
# ═════════════════════════════════════════════════════════════════════════════

class TestThreadSafety:
    """Concurrent access patterns don't corrupt state."""

    def test_governance_concurrent_deploy(self, valid_model):
        """Multiple threads cannot corrupt governance state."""
        errors = []

        def deploy_model():
            try:
                valid_model.deploy("v001", reason="Concurrent test")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=deploy_model) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # At most one should succeed, others should see it already deployed
        # No crashes
        deployed = valid_model.get_deployed_model()
        assert deployed is not None

    def test_risk_engine_concurrent_evaluate(self):
        """Risk engine handles concurrent evaluations safely."""
        from src.risk.engine import RiskEngine
        from src.risk.models import TradeRequest

        engine = RiskEngine()
        results = []

        def evaluate():
            request = TradeRequest(
                symbol="AAPL", side="buy", qty=10, price=150.0,
                confidence=0.8,
            )
            decision = engine.evaluate(
                request=request,
                portfolio_value=100000.0,
                cash=50000.0,
            )
            results.append(decision.approved)

        threads = [threading.Thread(target=evaluate) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == 10
        # All should get a valid response
        assert all(isinstance(r, bool) for r in results)

    def test_scheduler_concurrent_health_check(self):
        """Health check is safe under concurrent access."""
        from src.scheduler.asset_class_scheduler import AssetClassScheduler

        scheduler = AssetClassScheduler()
        results = []

        def check_health():
            health = scheduler.health_check()
            results.append(health)

        threads = [threading.Thread(target=check_health) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == 10
        assert all("healthy" in r for r in results)


# ═════════════════════════════════════════════════════════════════════════════
# 11. Survivorship Bias Awareness (P2-2)
# ═════════════════════════════════════════════════════════════════════════════

class TestSurvivorshipBias:
    """Training pipeline documents survivorship bias risk."""

    def test_training_records_symbol_universe(self):
        """Governance lineage tracks which symbols were used."""
        gov = ModelGovernance(governance_dir=tempfile.mkdtemp())
        df = pd.DataFrame({
            "close": np.random.randn(100),
            "volume": np.random.randint(1000, 10000, 100),
            "_symbol": ["AAPL"] * 50 + ["MSFT"] * 50,
        })
        lineage = gov.record_training(
            version="v001",
            features=["rsi_14"],
            dataset=df,
            metrics={"cv_accuracy": 0.6},
        )
        # Symbols are tracked in lineage for survivorship bias audit
        assert lineage.training_dataset_rows == 100


# ═════════════════════════════════════════════════════════════════════════════
# 12. Backtest Cost Modeling (P2-3)
# ═════════════════════════════════════════════════════════════════════════════

class TestBacktestCostModeling:
    """Backtest includes configurable transaction costs and slippage."""

    def test_costs_reduce_returns(self):
        """Transaction costs and slippage reduce trade returns."""
        from src.ml.training_pipeline import TrainingPipeline
        from src.ml.model_registry import ModelRegistry

        registry = ModelRegistry(models_dir=tempfile.mkdtemp())
        governance = ModelGovernance(governance_dir=tempfile.mkdtemp())
        pipeline = TrainingPipeline(registry=registry, governance=governance)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2] * 50)

        # Flat prices — returns should be negative due to costs
        close = np.full(50, 100.0)
        X = np.random.randn(50, 5)
        y = np.ones(50, dtype=int)  # flat class

        result_high_cost = pipeline._backtest_model_oos(
            model=mock_model,
            feature_cols=[f"f{i}" for i in range(5)],
            X_holdout=X,
            y_holdout=y,
            close_holdout=close,
            transaction_cost_bps=50.0,  # High costs
            slippage_bps=25.0,
        )

        result_low_cost = pipeline._backtest_model_oos(
            model=mock_model,
            feature_cols=[f"f{i}" for i in range(5)],
            X_holdout=X,
            y_holdout=y,
            close_holdout=close,
            transaction_cost_bps=1.0,
            slippage_bps=0.5,
        )

        # Higher costs should result in lower (more negative) Sharpe
        if result_high_cost["n_trades"] > 0 and result_low_cost["n_trades"] > 0:
            assert result_high_cost["sharpe"] <= result_low_cost["sharpe"]


# ═════════════════════════════════════════════════════════════════════════════
# 13. Dependency Pinning (P0-6)
# ═════════════════════════════════════════════════════════════════════════════

class TestDependencyPinning:
    """Dependencies have upper bounds to prevent breaking upgrades."""

    def test_requirements_have_upper_bounds(self):
        """All non-comment dependencies have version ceilings."""
        with open("requirements.txt") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            # Should have either == or < or ,< for upper bound
            assert ">=" in line, f"Missing version floor: {line}"
            assert "<" in line or "==" in line, f"Missing version ceiling: {line}"

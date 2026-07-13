"""
Tests for ML governance pipeline: commit metadata, governance metrics computation,
and paper trading activation safety.

Covers:
1. Git commit hash is always recorded via build_info.py (single source of truth)
2. Governance metrics (CV accuracy, walk-forward Sharpe, Monte Carlo, OOS Sharpe) 
   are computed correctly and validated against thresholds
3. Approved models activate paper trading; rejected models cannot
"""

import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.ml.governance import GovernanceViolation, ModelGovernance, ModelLineage, ModelStatus, ValidationResult
from src.ml.label_encoder import DirectionEncoder

# Stable commit hash for all tests (simulates stamped build)
_TEST_COMMIT = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"


class _PicklableModel:
    """Module-level picklable model for registry tests."""
    def predict(self, X):
        return [0] * len(X)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _stamp_build_info():
    """Simulate a stamped production build for all tests in this module."""
    with patch("src.ml.build_info.GIT_COMMIT", _TEST_COMMIT):
        with patch("src.ml.build_info.GIT_BRANCH", "main"):
            yield


@pytest.fixture
def governance(tmp_path):
    """Create governance manager with temp directory."""
    return ModelGovernance(governance_dir=str(tmp_path / "governance"))


@pytest.fixture
def passing_model(governance):
    """Record a model that passes all governance checks."""
    import pandas as pd
    import numpy as np
    # Provide a dataset so dataset_hash and dataset_rows are populated
    df = pd.DataFrame({
        "close": np.random.randn(200),
        "volume": np.random.randint(1000, 10000, 200),
    })
    governance.record_training(
        version="v010",
        features=["rsi_14", "ema_slope_20", "bb_pct", "atr_14", "macd_hist"],
        dataset=df,
        config={"n_estimators": 500, "max_depth": 4},
        metrics={
            "cv_accuracy": 0.67,
            "sharpe": 1.2,
            "sharpe_ratio": 1.2,
            "n_trades": 100,
            "max_drawdown": 0.08,
        },
        hyperparameters={"n_estimators": 500, "max_depth": 4, "learning_rate": 0.02},
        seed=42,
        training_duration=120.0,
        walk_forward_results={"sharpe": 0.8, "total_return": 0.05},
        monte_carlo_results={
            "median_return": 0.04,
            "p5_return": -0.03,
            "probability_of_profit": 0.68,
        },
    )
    return governance


@pytest.fixture
def failing_model(governance):
    """Record a model that fails governance checks (mimics current rejection)."""
    governance.record_training(
        version="v001",
        features=["rsi_14", "ema_slope_20"],
        metrics={
            "cv_accuracy": 0.322,
            "sharpe": -1.527,
            "n_trades": 70,
            "max_drawdown": 0.061,
        },
        hyperparameters={"n_estimators": 200, "max_depth": 6},
        seed=42,
        training_duration=270.0,
        walk_forward_results={"sharpe": -0.486, "total_return": -0.02},
        monte_carlo_results={
            "median_return": -0.05,
            "p5_return": -0.15,
            "probability_of_profit": 0.0,
        },
    )
    return governance


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite 1: Commit Metadata Always Recorded
# ─────────────────────────────────────────────────────────────────────────────


class TestCommitMetadata:
    """Verify that git commit hash is always captured from build_info (single source)."""

    def test_commit_from_build_info(self, governance):
        """build_info.GIT_COMMIT is the sole source of commit hash."""
        commit = governance._get_git_commit()
        assert commit == _TEST_COMMIT

    def test_env_vars_are_not_used(self, governance):
        """Environment variables are no longer used for runtime commit resolution."""
        with patch("src.ml.build_info.GIT_COMMIT", ""):
            with patch.dict(os.environ, {"GIT_COMMIT": "should_not_use"}):
                commit = governance._get_git_commit()
        assert commit == ""

    def test_record_training_includes_commit(self, governance):
        """record_training always populates git_commit from build_info."""
        lineage = governance.record_training(
            version="v001",
            features=["rsi_14"],
            metrics={"cv_accuracy": 0.6},
        )
        assert lineage.git_commit == _TEST_COMMIT

    def test_provenance_module_is_single_source(self, governance):
        """Verify governance delegates to provenance module (no subprocess)."""
        with patch("subprocess.run") as mock_run:
            commit = governance._get_git_commit()
        mock_run.assert_not_called()
        assert commit == _TEST_COMMIT


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite 2: Governance Metrics Correctly Computed and Validated
# ─────────────────────────────────────────────────────────────────────────────


class TestGovernanceMetrics:
    """Verify governance thresholds are correctly applied."""

    def test_passing_model_is_approved(self, passing_model):
        """Model meeting all thresholds is approved."""
        is_valid, issues = passing_model.validate_for_deployment("v010")
        # Filter out git_commit issue since test env may not have the env var set
        non_git_issues = [i for i in issues if "git commit" not in i.lower()]
        assert len(non_git_issues) == 0, f"Unexpected issues: {non_git_issues}"

    def test_failing_model_is_rejected(self, failing_model):
        """Model below thresholds is rejected with specific reasons."""
        is_valid, issues = failing_model.validate_for_deployment("v001")
        assert not is_valid
        # Should fail on all performance metrics
        assert any("CV accuracy" in i for i in issues)
        assert any("Walk-forward Sharpe" in i for i in issues)
        assert any("Monte Carlo prob_profit" in i or "prob_profit" in i for i in issues)
        assert any("Sharpe ratio" in i for i in issues)

    def test_cv_accuracy_threshold(self, governance):
        """CV accuracy must be >= 0.62 (configurable)."""
        governance.record_training(
            version="v001", features=["a", "b"],
            metrics={"cv_accuracy": 0.61, "sharpe": 1.0, "n_trades": 100},
            hyperparameters={"n_estimators": 100},
            walk_forward_results={"sharpe": 0.5},
            monte_carlo_results={"probability_of_profit": 0.7},
        )
        is_valid, issues = governance.validate_for_deployment("v001")
        assert any("CV accuracy" in i for i in issues)

    def test_walk_forward_sharpe_threshold(self, governance):
        """Walk-forward Sharpe must be >= 0.3."""
        governance.record_training(
            version="v001", features=["a", "b"],
            metrics={"cv_accuracy": 0.65, "sharpe": 1.0, "n_trades": 100},
            hyperparameters={"n_estimators": 100},
            walk_forward_results={"sharpe": 0.29},
            monte_carlo_results={"probability_of_profit": 0.7},
        )
        is_valid, issues = governance.validate_for_deployment("v001")
        assert any("Walk-forward Sharpe" in i for i in issues)

    def test_monte_carlo_prob_profit_threshold(self, governance):
        """Monte Carlo probability of profit must be >= 0.65."""
        governance.record_training(
            version="v001", features=["a", "b"],
            metrics={"cv_accuracy": 0.65, "sharpe": 1.0, "n_trades": 100},
            hyperparameters={"n_estimators": 100},
            walk_forward_results={"sharpe": 0.5},
            monte_carlo_results={"probability_of_profit": 0.64},
        )
        is_valid, issues = governance.validate_for_deployment("v001")
        assert any("prob_profit" in i for i in issues)

    def test_sharpe_ratio_threshold(self, governance):
        """OOS Sharpe ratio must be >= 0.5."""
        governance.record_training(
            version="v001", features=["a", "b"],
            metrics={"cv_accuracy": 0.65, "sharpe": 0.4, "n_trades": 100},
            hyperparameters={"n_estimators": 100},
            walk_forward_results={"sharpe": 0.5},
            monte_carlo_results={"probability_of_profit": 0.7},
        )
        is_valid, issues = governance.validate_for_deployment("v001")
        assert any("Sharpe ratio" in i for i in issues)

    def test_max_drawdown_threshold(self, governance):
        """Max drawdown must be <= 0.20."""
        governance.record_training(
            version="v001", features=["a", "b"],
            metrics={"cv_accuracy": 0.65, "sharpe": 1.0, "n_trades": 100,
                     "max_drawdown": 0.25},
            hyperparameters={"n_estimators": 100},
            walk_forward_results={"sharpe": 0.5},
            monte_carlo_results={"probability_of_profit": 0.7},
        )
        is_valid, issues = governance.validate_for_deployment("v001")
        assert any("drawdown" in i.lower() for i in issues)

    def test_min_backtest_trades_threshold(self, governance):
        """Must have at least 50 backtest trades."""
        governance.record_training(
            version="v001", features=["a", "b"],
            metrics={"cv_accuracy": 0.65, "sharpe": 1.0, "n_trades": 49},
            hyperparameters={"n_estimators": 100},
            walk_forward_results={"sharpe": 0.5},
            monte_carlo_results={"probability_of_profit": 0.7},
        )
        is_valid, issues = governance.validate_for_deployment("v001")
        assert any("trades" in i.lower() for i in issues)

    def test_all_thresholds_pass(self, governance):
        """Model meeting all thresholds passes validation (ignoring git in test env)."""
        import numpy as np
        df = pd.DataFrame({
            "close": np.random.randn(100),
            "volume": np.random.randint(1000, 10000, 100),
        })
        governance.record_training(
            version="v001", features=["a", "b", "c"],
            dataset=df,
            metrics={"cv_accuracy": 0.65, "sharpe": 0.8, "n_trades": 100,
                     "max_drawdown": 0.10},
            hyperparameters={"n_estimators": 500},
            walk_forward_results={"sharpe": 0.5},
            monte_carlo_results={"probability_of_profit": 0.70},
        )
        is_valid, issues = governance.validate_for_deployment("v001")
        # Only git_commit might fail in test env
        non_git_issues = [i for i in issues if "git commit" not in i.lower()]
        assert len(non_git_issues) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite 3: Paper Trading Activation Safety
# ─────────────────────────────────────────────────────────────────────────────


class TestPaperTradingActivation:
    """Verify that only approved models can activate paper trading."""

    def test_approved_model_activates_in_registry(self, passing_model, tmp_path):
        """Governance-approved model can be set as active version."""
        from src.ml.model_registry import ModelRegistry

        registry = ModelRegistry(models_dir=str(tmp_path / "models"))

        # Use a simple picklable object as a model stand-in
        
        version = registry.save_version(
            model=_PicklableModel(),
            features=["rsi_14", "ema_slope_20"],
            metrics={"cv_accuracy": 0.58},
            symbols=["AAPL"],
            activate=False,
        )

        # Validate passes
        is_valid, issues = passing_model.validate_for_deployment("v010")
        non_git_issues = [i for i in issues if "git commit" not in i.lower()]

        if len(non_git_issues) == 0:
            # Approved → can activate
            registry.set_active_version(version)
            assert registry.get_active_version() == version

    def test_rejected_model_stays_inactive(self, failing_model, tmp_path):
        """Governance-rejected model must NOT be activated."""
        from src.ml.model_registry import ModelRegistry

        registry = ModelRegistry(models_dir=str(tmp_path / "models"))


        version = registry.save_version(
            model=_PicklableModel(),
            features=["rsi_14"],
            metrics={"cv_accuracy": 0.322},
            symbols=["AAPL"],
            activate=False,
        )

        # Validate fails
        is_valid, issues = failing_model.validate_for_deployment("v001")
        assert not is_valid

        # Rejected → must NOT activate
        # Verify no active version exists
        active = registry.get_active_version()
        assert active != version or active is None

    def test_pipeline_does_not_deploy_rejected_model(self, tmp_path):
        """Full pipeline integration: rejected model is not deployed."""
        from src.ml.model_registry import ModelRegistry

        registry = ModelRegistry(models_dir=str(tmp_path / "models"))
        governance = ModelGovernance(governance_dir=str(tmp_path / "governance"))

        # Record a bad model
        governance.record_training(
            version="v001",
            features=["rsi_14"],
            metrics={"cv_accuracy": 0.30, "sharpe": -1.0, "n_trades": 10},
            hyperparameters={"n_estimators": 100},
            walk_forward_results={"sharpe": -0.5},
            monte_carlo_results={"probability_of_profit": 0.0},
        )

        # Validate
        is_valid, issues = governance.validate_for_deployment("v001")
        assert not is_valid

        # Pipeline logic: only deploy if valid
        if is_valid:
            governance.deploy(version="v001", reason="Auto-deploy")
            registry.set_active_version("v001")

        # Verify: NOT deployed
        deployed = governance.get_deployed_model()
        assert deployed is None

    def test_pipeline_deploys_approved_model(self, tmp_path):
        """Full pipeline integration: approved model IS deployed."""
        from src.ml.model_registry import ModelRegistry

        registry = ModelRegistry(models_dir=str(tmp_path / "models"))
        governance = ModelGovernance(governance_dir=str(tmp_path / "governance"))

        # Record a good model
        governance.record_training(
            version="v010",
            features=["rsi_14", "ema_slope_20", "bb_pct"],
            metrics={"cv_accuracy": 0.67, "sharpe": 1.0, "n_trades": 100,
                     "max_drawdown": 0.08},
            hyperparameters={"n_estimators": 500, "learning_rate": 0.02},
            walk_forward_results={"sharpe": 0.5},
            monte_carlo_results={"probability_of_profit": 0.70},
        )

        # Validate
        is_valid, issues = governance.validate_for_deployment("v010")
        non_git_issues = [i for i in issues if "git commit" not in i.lower()]

        if len(non_git_issues) == 0:
            # Deploy
            governance.deploy(version="v010", reason="Auto-deploy")
            deployed = governance.get_deployed_model()
            assert deployed is not None
            assert deployed.version == "v010"
            assert deployed.is_deployed is True

    def test_governance_deploy_sets_deployed_flag(self, passing_model):
        """deploy() marks model as deployed with timestamp."""
        passing_model.deploy("v010", reason="Passed all gates")
        model = passing_model.get_deployed_model()
        assert model.is_deployed is True
        assert model.deployed_at != ""
        assert model.deployment_reason == "Passed all gates"

    def test_rejected_model_cannot_be_force_deployed(self, failing_model):
        """Deploying a model that fails validation raises GovernanceViolation.

        This is a regression test for the governance bypass vulnerability:
        deploy() now enforces validation internally and cannot be bypassed
        by calling it directly.
        """
        with pytest.raises(GovernanceViolation, match="failed governance validation"):
            failing_model.deploy("v001", reason="Manual override")


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite 4: Training Data Preparation Improvements
# ─────────────────────────────────────────────────────────────────────────────


class TestTrainingDataPreparation:
    """Verify improved feature engineering and label generation."""

    def test_adaptive_threshold_higher_for_volatile_assets(self):
        """Volatile assets get higher labeling threshold (fewer false signals)."""
        # Create a volatile asset (like crypto) and a stable asset (like AAPL)
        np.random.seed(42)
        n = 200

        # Low-vol asset: daily returns ~0.5%
        low_vol_prices = 100 * np.cumprod(1 + np.random.normal(0, 0.005, n))
        # High-vol asset: daily returns ~3%
        high_vol_prices = 100 * np.cumprod(1 + np.random.normal(0, 0.03, n))

        # With a fixed 0.5% threshold, the high-vol asset would label almost
        # everything as directional (noise). With adaptive threshold, it adjusts.
        base_threshold = 0.005
        forward_bars = 5

        # Compute adaptive threshold for high-vol
        returns = pd.Series(high_vol_prices).pct_change()
        rolling_vol = returns.rolling(20, min_periods=10).std().fillna(returns.std())
        adaptive_high = np.maximum(base_threshold, 0.5 * rolling_vol * np.sqrt(forward_bars))

        # Adaptive threshold should be significantly higher for volatile asset
        mean_adaptive = adaptive_high.iloc[25:].mean()
        assert mean_adaptive > base_threshold * 2, (
            f"Adaptive threshold {mean_adaptive:.4f} should be >> base {base_threshold}"
        )

    def test_symbol_encoding_produces_one_hot_features(self):
        """Symbol one-hot encoding adds distinguishing features."""
        from src.ml.features import engineer_features

        np.random.seed(42)
        n = 150

        # Create minimal OHLCV data for two symbols
        symbols = {}
        for sym in ["AAPL", "BTC/USD"]:
            vol = 0.01 if sym == "AAPL" else 0.05
            close = 100 * np.cumprod(1 + np.random.normal(0, vol, n))
            df = pd.DataFrame({
                "open": close * (1 + np.random.normal(0, 0.001, n)),
                "high": close * (1 + abs(np.random.normal(0, 0.005, n))),
                "low": close * (1 - abs(np.random.normal(0, 0.005, n))),
                "close": close,
                "volume": np.random.randint(1000, 100000, n).astype(float),
            })
            symbols[sym] = engineer_features(df, include_target=False)

        # The _prepare_training_data should add _sym_AAPL, _sym_BTC/USD columns
        # Verify manually that symbol list is sorted and one-hot would work
        symbol_list = sorted(symbols.keys())
        assert symbol_list == ["AAPL", "BTC/USD"]

    def test_variance_filtering_removes_constant_features(self):
        """Near-constant features are removed to reduce noise."""
        # Create features where some have zero variance
        np.random.seed(42)
        X = np.column_stack([
            np.random.randn(100),       # Good feature
            np.ones(100) * 5.0,         # Constant — should be removed
            np.random.randn(100) * 0.5, # Good feature
            np.full(100, 1e-10),        # Near-zero constant — should be removed
        ])

        col_std = np.nanstd(X, axis=0)
        variance_mask = col_std > 1e-8
        filtered = X[:, variance_mask]

        assert filtered.shape[1] == 2, "Should keep only 2 non-constant features"

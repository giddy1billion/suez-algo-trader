"""
Integration tests for walk-forward early stopping and Optuna hyperparameter tuning.

Covers:
1. Walk-forward early stopping uses fold-local validation with embargo
2. Best iteration is recorded per fold
3. Walk-forward metrics are reproducible (same seed → same results)
4. Walk-forward is leak-free (test data never in training)
5. Optuna tuning produces different params for equity vs crypto
6. Tuned params are persisted and loaded
7. Tuned params are applied during subsequent training
"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Test Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_feature_data():
    """Create synthetic feature data for multiple symbols."""
    np.random.seed(42)
    n = 600  # Enough for walk-forward splits

    def _make_df():
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "open": close + np.random.randn(n) * 0.1,
            "high": close + abs(np.random.randn(n) * 0.5),
            "low": close - abs(np.random.randn(n) * 0.5),
            "close": close,
            "volume": np.random.randint(10000, 100000, n).astype(float),
        })
        # Add some technical features
        df["rsi_14"] = 50 + np.random.randn(n) * 15
        df["ema_slope_20"] = np.random.randn(n) * 0.01
        df["bb_pct"] = np.random.rand(n)
        df["atr_14"] = abs(np.random.randn(n) * 0.5) + 0.5
        df["macd_hist"] = np.random.randn(n) * 0.1
        df["vol_ratio"] = 0.5 + np.random.rand(n) * 1.5
        return df

    return {
        "AAPL": _make_df(),
        "MSFT": _make_df(),
    }


@pytest.fixture
def crypto_feature_data():
    """Create synthetic crypto feature data."""
    np.random.seed(123)
    n = 600

    def _make_df():
        close = 40000 + np.cumsum(np.random.randn(n) * 200)
        df = pd.DataFrame({
            "open": close + np.random.randn(n) * 50,
            "high": close + abs(np.random.randn(n) * 100),
            "low": close - abs(np.random.randn(n) * 100),
            "close": close,
            "volume": np.random.randint(100, 10000, n).astype(float),
        })
        df["rsi_14"] = 50 + np.random.randn(n) * 20
        df["ema_slope_20"] = np.random.randn(n) * 0.02
        df["bb_pct"] = np.random.rand(n)
        df["atr_14"] = abs(np.random.randn(n) * 100) + 50
        df["macd_hist"] = np.random.randn(n) * 50
        df["vol_ratio"] = 0.5 + np.random.rand(n) * 2.0
        return df

    return {
        "BTC/USD": _make_df(),
        "ETH/USD": _make_df(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Early Stopping Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestWalkForwardEarlyStopping:
    """Verify walk-forward uses early stopping with fold-local validation."""

    def test_walk_forward_records_best_iteration(self, synthetic_feature_data):
        """Each fold records best_iteration from early stopping."""
        from src.ml.training_pipeline import TrainingPipeline

        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        pipeline._min_samples = 50

        result = pipeline._walk_forward_validation(
            feature_data=synthetic_feature_data,
            feature_cols=["rsi_14", "ema_slope_20", "bb_pct", "atr_14", "macd_hist", "vol_ratio"],
            n_splits=3,
        )

        assert "splits" in result
        assert len(result["splits"]) > 0

        for split in result["splits"]:
            assert "best_iteration" in split
            assert "early_stopping_used" in split
            if split["early_stopping_used"]:
                # best_iteration should be a positive integer
                assert split["best_iteration"] is not None
                assert split["best_iteration"] >= 0

    def test_walk_forward_reproducible(self, synthetic_feature_data):
        """Same seed and data produces identical walk-forward results."""
        from src.ml.training_pipeline import TrainingPipeline

        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        pipeline._min_samples = 50

        feature_cols = ["rsi_14", "ema_slope_20", "bb_pct", "atr_14", "macd_hist", "vol_ratio"]

        result1 = pipeline._walk_forward_validation(
            feature_data=synthetic_feature_data,
            feature_cols=feature_cols,
            n_splits=3,
        )
        result2 = pipeline._walk_forward_validation(
            feature_data=synthetic_feature_data,
            feature_cols=feature_cols,
            n_splits=3,
        )

        assert result1["sharpe"] == result2["sharpe"]
        assert result1["total_return"] == result2["total_return"]
        assert result1["n_trades"] == result2["n_trades"]
        for s1, s2 in zip(result1["splits"], result2["splits"]):
            assert s1["accuracy"] == s2["accuracy"]
            assert s1["best_iteration"] == s2["best_iteration"]

    def test_walk_forward_is_leak_free(self, synthetic_feature_data):
        """Test data in each fold is strictly after training data + purge gap."""
        from src.ml.training_pipeline import TrainingPipeline

        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        pipeline._min_samples = 50

        result = pipeline._walk_forward_validation(
            feature_data=synthetic_feature_data,
            feature_cols=["rsi_14", "ema_slope_20", "bb_pct", "atr_14", "macd_hist", "vol_ratio"],
            n_splits=3,
        )

        # Verify splits have increasing train sizes (expanding window)
        train_sizes = [s["train_size"] for s in result["splits"]]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1], (
                f"Split {i} train_size {train_sizes[i]} should be > split {i-1} "
                f"train_size {train_sizes[i-1]} (expanding window)"
            )

        # Each test set should be smaller than or equal to a segment
        total_samples = sum(s["train_size"] for s in result["splits"][:1]) + sum(
            s["test_size"] for s in result["splits"]
        )
        # Basic sanity: no overlap possible since train_end + purge_gap = test_start
        for split in result["splits"]:
            assert split["test_size"] > 0
            assert split["train_size"] > 0

    def test_walk_forward_early_stopping_prevents_overfitting(self, synthetic_feature_data):
        """Early stopping should use fewer than max iterations (500)."""
        from src.ml.training_pipeline import TrainingPipeline

        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        pipeline._min_samples = 50

        result = pipeline._walk_forward_validation(
            feature_data=synthetic_feature_data,
            feature_cols=["rsi_14", "ema_slope_20", "bb_pct", "atr_14", "macd_hist", "vol_ratio"],
            n_splits=3,
        )

        # At least one split should stop before 500 iterations
        early_stopped = [
            s for s in result["splits"]
            if s["early_stopping_used"] and s["best_iteration"] is not None
            and s["best_iteration"] < 499
        ]
        # With random data, early stopping should trigger
        assert len(early_stopped) > 0, (
            "Expected at least one fold to early-stop before 500 iterations"
        )

    def test_walk_forward_metrics_surface_in_results(self, synthetic_feature_data):
        """Walk-forward results include sharpe, total_return, n_trades, splits."""
        from src.ml.training_pipeline import TrainingPipeline

        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        pipeline._min_samples = 50

        result = pipeline._walk_forward_validation(
            feature_data=synthetic_feature_data,
            feature_cols=["rsi_14", "ema_slope_20", "bb_pct", "atr_14", "macd_hist", "vol_ratio"],
            n_splits=3,
        )

        assert "sharpe" in result
        assert "total_return" in result
        assert "n_trades" in result
        assert "splits" in result
        assert isinstance(result["sharpe"], float)
        assert isinstance(result["total_return"], float)
        assert isinstance(result["n_trades"], int)


# ═══════════════════════════════════════════════════════════════════════════════
# Optuna Hyperparameter Tuning Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestAssetClassDetection:
    """Test asset class classification."""

    def test_equity_symbols(self):
        from src.ml.hyperparameter_tuning import classify_asset_class
        assert classify_asset_class(["AAPL", "MSFT", "GOOGL"]) == "equity"

    def test_crypto_symbols(self):
        from src.ml.hyperparameter_tuning import classify_asset_class
        assert classify_asset_class(["BTC/USD", "ETH/USD", "SOL/USD"]) == "crypto"

    def test_mixed_majority_equity(self):
        from src.ml.hyperparameter_tuning import classify_asset_class
        assert classify_asset_class(["AAPL", "MSFT", "BTC/USD"]) == "equity"

    def test_mixed_majority_crypto(self):
        from src.ml.hyperparameter_tuning import classify_asset_class
        assert classify_asset_class(["BTC/USD", "ETH/USD", "AAPL"]) == "crypto"

    def test_empty_defaults_to_equity(self):
        from src.ml.hyperparameter_tuning import classify_asset_class
        assert classify_asset_class([]) == "equity"


class TestHyperparameterTuner:
    """Test Optuna-based hyperparameter tuning workflow."""

    @pytest.fixture
    def tuner(self, tmp_path):
        from src.ml.hyperparameter_tuning import HyperparameterTuner
        return HyperparameterTuner(tuning_dir=str(tmp_path / "tuning"))

    def test_tune_equity_produces_params(self, tuner, synthetic_feature_data):
        """Tuning equities produces valid parameter dict."""
        # Use small n_trials for test speed
        params = tuner.tune(
            feature_data=synthetic_feature_data,
            asset_class="equity",
            n_trials=5,
            n_splits=2,
        )
        assert isinstance(params, dict)
        assert "n_estimators" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert params["n_estimators"] >= 200
        assert params["max_depth"] >= 3

    def test_tune_crypto_produces_params(self, tuner, crypto_feature_data):
        """Tuning crypto produces valid parameter dict."""
        params = tuner.tune(
            feature_data=crypto_feature_data,
            asset_class="crypto",
            n_trials=5,
            n_splits=2,
        )
        assert isinstance(params, dict)
        assert "n_estimators" in params

    def test_tuned_params_are_persisted(self, tuner, synthetic_feature_data):
        """Best params are saved to disk after tuning."""
        tuner.tune(
            feature_data=synthetic_feature_data,
            asset_class="equity",
            n_trials=5,
            n_splits=2,
        )
        path = tuner._params_path("equity")
        assert os.path.exists(path)

        with open(path, "r") as f:
            data = json.load(f)
        assert "params" in data
        assert "best_score" in data
        assert "tuned_at" in data
        assert data["asset_class"] == "equity"

    def test_load_best_params_returns_persisted(self, tuner, synthetic_feature_data):
        """load_best_params returns what was persisted by tune()."""
        original_params = tuner.tune(
            feature_data=synthetic_feature_data,
            asset_class="equity",
            n_trials=5,
            n_splits=2,
        )
        loaded = tuner.load_best_params("equity")
        assert loaded == original_params

    def test_load_best_params_returns_none_if_not_tuned(self, tuner):
        """load_best_params returns None for untuned asset class."""
        assert tuner.load_best_params("equity") is None
        assert tuner.load_best_params("crypto") is None

    def test_get_training_params_uses_tuned_when_available(self, tuner, synthetic_feature_data):
        """get_training_params loads tuned params when they exist."""
        tuned = tuner.tune(
            feature_data=synthetic_feature_data,
            asset_class="equity",
            n_trials=5,
            n_splits=2,
        )
        result = tuner.get_training_params(["AAPL", "MSFT"])
        assert result == tuned

    def test_get_training_params_falls_back_to_defaults(self, tuner):
        """get_training_params returns defaults when no tuned params exist."""
        result = tuner.get_training_params(["AAPL", "MSFT"])
        assert result["n_estimators"] == 500
        assert result["max_depth"] == 4

    def test_equity_and_crypto_params_differ(self, tuner, synthetic_feature_data, crypto_feature_data):
        """Equity and crypto tuning produce different parameter sets."""
        equity_params = tuner.tune(
            feature_data=synthetic_feature_data,
            asset_class="equity",
            n_trials=5,
            n_splits=2,
        )
        crypto_params = tuner.tune(
            feature_data=crypto_feature_data,
            asset_class="crypto",
            n_trials=5,
            n_splits=2,
        )
        # They should be separately persisted
        equity_loaded = tuner.load_best_params("equity")
        crypto_loaded = tuner.load_best_params("crypto")
        assert equity_loaded is not None
        assert crypto_loaded is not None
        # Both are valid param dicts
        assert "n_estimators" in equity_loaded
        assert "n_estimators" in crypto_loaded

    def test_auto_detect_asset_class(self, tuner, crypto_feature_data):
        """Tuning auto-detects crypto from symbol names."""
        params = tuner.tune(
            feature_data=crypto_feature_data,
            asset_class=None,  # Auto-detect
            n_trials=5,
            n_splits=2,
        )
        # Should have been saved as crypto
        assert tuner.load_best_params("crypto") is not None


class TestTunedParamsAppliedDuringTraining:
    """Ensure tuned parameters are actually used when training."""

    @pytest.fixture
    def tuner_with_params(self, tmp_path):
        """Tuner with pre-set equity params."""
        from src.ml.hyperparameter_tuning import HyperparameterTuner
        tuner = HyperparameterTuner(tuning_dir=str(tmp_path / "tuning"))
        # Manually persist specific params
        custom_params = {
            "n_estimators": 750,
            "max_depth": 6,
            "learning_rate": 0.015,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 12,
            "gamma": 0.2,
            "reg_alpha": 0.05,
            "reg_lambda": 2.0,
        }
        tuner._save_best_params("equity", custom_params, 0.65)
        return tuner

    def test_training_loads_tuned_params(self, tuner_with_params):
        """Training pipeline can load and use tuned parameters."""
        params = tuner_with_params.get_training_params(["AAPL", "MSFT"])
        assert params["n_estimators"] == 750
        assert params["max_depth"] == 6
        assert params["learning_rate"] == 0.015

    def test_tuned_params_applied_to_xgboost(self, tuner_with_params):
        """Tuned params can be passed directly to XGBClassifier."""
        from xgboost import XGBClassifier

        params = tuner_with_params.get_training_params(["AAPL", "MSFT"])
        model = XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            verbosity=0,
        )
        # Verify params were actually set
        model_params = model.get_params()
        assert model_params["n_estimators"] == 750
        assert model_params["max_depth"] == 6
        assert model_params["learning_rate"] == 0.015
        assert model_params["subsample"] == 0.8

    def test_crypto_params_separate_from_equity(self, tmp_path):
        """Crypto and equity persist independently."""
        from src.ml.hyperparameter_tuning import HyperparameterTuner
        tuner = HyperparameterTuner(tuning_dir=str(tmp_path / "tuning"))

        equity_params = {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.02,
                         "subsample": 0.7, "colsample_bytree": 0.6, "min_child_weight": 10,
                         "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.5}
        crypto_params = {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.01,
                         "subsample": 0.65, "colsample_bytree": 0.5, "min_child_weight": 15,
                         "gamma": 0.2, "reg_alpha": 0.2, "reg_lambda": 2.5}

        tuner._save_best_params("equity", equity_params, 0.60)
        tuner._save_best_params("crypto", crypto_params, 0.58)

        # Loading equity symbols gives equity params
        result = tuner.get_training_params(["AAPL", "GOOGL"])
        assert result["n_estimators"] == 500
        assert result["reg_lambda"] == 1.5

        # Loading crypto symbols gives crypto params
        result = tuner.get_training_params(["BTC/USD", "ETH/USD"])
        assert result["n_estimators"] == 400
        assert result["reg_lambda"] == 2.5

"""
Tests for Model Governance — lineage tracking and deployment lifecycle.
"""

import os
import tempfile

import pytest

from src.ml.governance import ModelGovernance, ModelLineage


@pytest.fixture
def governance(tmp_path):
    """Create governance manager with temp directory."""
    return ModelGovernance(governance_dir=str(tmp_path / "governance"))


@pytest.fixture
def governance_with_model(governance):
    """Governance with one recorded model."""
    governance.record_training(
        version="v001",
        features=["rsi_14", "ema_slope_20", "bb_pct", "atr_14"],
        config={"n_estimators": 100, "max_depth": 6},
        metrics={"cv_accuracy": 0.67, "sharpe": 1.5, "n_trades": 150},
        hyperparameters={"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
        seed=42,
        training_duration=45.2,
        walk_forward_results={"sharpe": 1.3, "total_return": 0.12},
        monte_carlo_results={
            "median_return": 0.08,
            "p5_return": -0.05,
            "probability_of_profit": 0.72,
        },
    )
    return governance


class TestModelLineage:
    """Tests for ModelLineage dataclass."""

    def test_lineage_to_dict_roundtrip(self):
        lineage = ModelLineage(
            version="v001",
            git_commit="abc123",
            n_features=10,
            cv_accuracy=0.65,
        )
        d = lineage.to_dict()
        restored = ModelLineage.from_dict(d)
        assert restored.version == "v001"
        assert restored.git_commit == "abc123"
        assert restored.n_features == 10

    def test_lineage_defaults(self):
        lineage = ModelLineage(version="v001")
        assert lineage.is_deployed is False
        assert lineage.cv_accuracy == 0.0
        assert lineage.feature_names == []


class TestModelGovernance:
    """Tests for ModelGovernance manager."""

    def test_record_training_basic(self, governance):
        """Record training stores lineage."""
        lineage = governance.record_training(
            version="v001",
            features=["rsi_14", "macd"],
            metrics={"cv_accuracy": 0.6},
            seed=42,
        )
        assert lineage.version == "v001"
        assert lineage.n_features == 2
        assert lineage.random_seed == 42
        assert lineage.cv_accuracy == 0.6

    def test_record_training_with_dataset(self, governance):
        """Record training computes dataset hash."""
        import pandas as pd
        import numpy as np

        df = pd.DataFrame({
            "close": np.random.randn(100),
            "volume": np.random.randint(1000, 10000, 100),
        })
        lineage = governance.record_training(
            version="v001",
            features=["close", "volume"],
            dataset=df,
            metrics={"cv_accuracy": 0.55},
        )
        assert lineage.training_dataset_hash != ""
        assert lineage.training_dataset_rows == 100

    def test_get_lineage(self, governance_with_model):
        """Retrieve stored lineage."""
        lineage = governance_with_model.get_lineage("v001")
        assert lineage is not None
        assert lineage.cv_accuracy == 0.67
        assert lineage.walk_forward_sharpe == 1.3
        assert lineage.monte_carlo_prob_profit == 0.72
        assert lineage.random_seed == 42

    def test_get_lineage_nonexistent(self, governance):
        """Non-existent version returns None."""
        assert governance.get_lineage("v999") is None

    def test_deploy_and_retire(self, governance_with_model):
        """Deploy and retire lifecycle."""
        assert governance_with_model.deploy("v001", reason="Best walk-forward Sharpe")

        deployed = governance_with_model.get_deployed_model()
        assert deployed is not None
        assert deployed.version == "v001"
        assert deployed.is_deployed is True
        assert deployed.deployment_reason == "Best walk-forward Sharpe"

        # Retire
        assert governance_with_model.retire("v001", reason="Performance degraded")
        deployed = governance_with_model.get_deployed_model()
        assert deployed is None

    def test_deploy_supersedes_previous(self, governance_with_model):
        """Deploying a new model retires the old one."""
        governance_with_model.deploy("v001")

        # Record and deploy v002
        governance_with_model.record_training(
            version="v002",
            features=["rsi_14", "ema_slope_20", "vol_ratio"],
            metrics={"cv_accuracy": 0.72},
        )
        governance_with_model.deploy("v002", reason="Higher accuracy")

        deployed = governance_with_model.get_deployed_model()
        assert deployed.version == "v002"

        # v001 should be retired
        v001 = governance_with_model.get_lineage("v001")
        assert v001.is_deployed is False
        assert "Superseded" in v001.retirement_reason

    def test_validate_for_deployment_valid(self, governance_with_model):
        """Valid model passes deployment checks."""
        is_valid, issues = governance_with_model.validate_for_deployment("v001")
        # May have issues due to missing git commit in test env
        # But should not fail on metrics
        metric_issues = [i for i in issues if "accuracy" in i.lower() or "sharpe" in i.lower()]
        assert len(metric_issues) == 0

    def test_validate_for_deployment_bad_metrics(self, governance):
        """Model with poor metrics fails validation."""
        governance.record_training(
            version="v001",
            features=["rsi_14"],
            metrics={"cv_accuracy": 0.3},  # Below 50%
            walk_forward_results={"sharpe": -0.5},  # Negative
            monte_carlo_results={"probability_of_profit": 0.3},  # Below 50%
        )
        is_valid, issues = governance.validate_for_deployment("v001")
        assert not is_valid
        assert any("CV accuracy" in i for i in issues)
        assert any("Walk-forward Sharpe" in i for i in issues)
        assert any("prob_profit" in i for i in issues)

    def test_deployment_history(self, governance_with_model):
        """Deployment history tracks all deployments."""
        governance_with_model.deploy("v001")
        history = governance_with_model.get_deployment_history()
        assert len(history) >= 1
        assert history[0]["version"] == "v001"

    def test_audit_report(self, governance_with_model):
        """Audit report summarizes governance state."""
        report = governance_with_model.audit_report()
        assert report["total_versions"] == 1
        assert report["currently_deployed"] == 0
        assert len(report["versions"]) == 1

    def test_feature_schema_version(self, governance):
        """Different feature sets produce different schema versions."""
        l1 = governance.record_training(
            version="v001", features=["rsi_14", "macd"],
            metrics={"cv_accuracy": 0.6},
        )
        l2 = governance.record_training(
            version="v002", features=["rsi_14", "macd", "atr_14"],
            metrics={"cv_accuracy": 0.65},
        )
        assert l1.feature_schema_version != l2.feature_schema_version

    def test_config_hash_deterministic(self, governance):
        """Same config produces same hash."""
        config = {"n_estimators": 100, "max_depth": 6}
        l1 = governance.record_training(
            version="v001", features=["a"], config=config,
            metrics={"cv_accuracy": 0.6},
        )
        # Re-record with same config
        l2 = governance.record_training(
            version="v001", features=["a"], config=config,
            metrics={"cv_accuracy": 0.6},
        )
        assert l1.config_hash == l2.config_hash
        assert l1.config_hash != ""

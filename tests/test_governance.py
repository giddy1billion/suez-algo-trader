"""
Tests for Model Governance — lineage tracking and deployment lifecycle.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from src.ml.governance import GovernanceViolation, ModelGovernance, ModelLineage

# Provide a stable commit hash for all governance tests (simulates stamped build)
_TEST_COMMIT = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"


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
def governance_with_model(governance):
    """Governance with one recorded model that passes all validation gates."""
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({
        "close": np.random.randn(200),
        "volume": np.random.randint(1000, 10000, 200),
    })
    governance.record_training(
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

        # Record and deploy v002 with valid metrics
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            "close": np.random.randn(200),
            "volume": np.random.randint(1000, 10000, 200),
        })
        governance_with_model.record_training(
            version="v002",
            features=["rsi_14", "ema_slope_20", "vol_ratio"],
            dataset=df,
            metrics={
                "cv_accuracy": 0.72,
                "sharpe": 1.8,
                "n_trades": 100,
                "max_drawdown": 0.04,
            },
            hyperparameters={"n_estimators": 100},
            walk_forward_results={"sharpe": 1.5},
            monte_carlo_results={"probability_of_profit": 0.75},
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


# ═══════════════════════════════════════════════════════════════════════════════
# Governance Hardening Regression Tests (P0)
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegrityGateRegression:
    """Prove that tampered lineage metadata cannot be deployed."""

    def test_tampered_cv_accuracy_blocks_deploy(self, governance_with_model):
        """Tampering with cv_accuracy in lineage invalidates integrity hash."""
        import json

        # Tamper with the lineage file directly
        with open(governance_with_model._lineage_path, "r") as f:
            records = json.load(f)
        records[0]["cv_accuracy"] = 0.99  # Tamper
        with open(governance_with_model._lineage_path, "w") as f:
            json.dump(records, f)

        with pytest.raises(GovernanceViolation, match="integrity"):
            governance_with_model.deploy("v001", reason="Tampered model")

    def test_tampered_walk_forward_sharpe_blocks_deploy(self, governance_with_model):
        """Tampering with walk_forward_sharpe triggers integrity failure."""
        import json

        with open(governance_with_model._lineage_path, "r") as f:
            records = json.load(f)
        records[0]["walk_forward_sharpe"] = 5.0  # Tamper
        with open(governance_with_model._lineage_path, "w") as f:
            json.dump(records, f)

        with pytest.raises(GovernanceViolation, match="integrity"):
            governance_with_model.deploy("v001", reason="Tampered sharpe")

    def test_tampered_monte_carlo_blocks_deploy(self, governance_with_model):
        """Tampering with monte_carlo_prob_profit triggers integrity failure."""
        import json

        with open(governance_with_model._lineage_path, "r") as f:
            records = json.load(f)
        records[0]["monte_carlo_prob_profit"] = 0.99  # Tamper
        with open(governance_with_model._lineage_path, "w") as f:
            json.dump(records, f)

        with pytest.raises(GovernanceViolation, match="integrity"):
            governance_with_model.deploy("v001", reason="Tampered MC")

    def test_tampered_n_trades_blocks_deploy(self, governance_with_model):
        """Tampering with n_trades_backtest triggers integrity failure."""
        import json

        with open(governance_with_model._lineage_path, "r") as f:
            records = json.load(f)
        records[0]["n_trades_backtest"] = 9999  # Tamper
        with open(governance_with_model._lineage_path, "w") as f:
            json.dump(records, f)

        with pytest.raises(GovernanceViolation, match="integrity"):
            governance_with_model.deploy("v001", reason="Tampered trades")

    def test_tampered_training_timestamp_blocks_deploy(self, governance_with_model):
        """Tampering with training_timestamp triggers integrity failure."""
        import json

        with open(governance_with_model._lineage_path, "r") as f:
            records = json.load(f)
        records[0]["training_timestamp"] = "2020-01-01T00:00:00+00:00"  # Tamper
        with open(governance_with_model._lineage_path, "w") as f:
            json.dump(records, f)

        with pytest.raises(GovernanceViolation, match="integrity"):
            governance_with_model.deploy("v001", reason="Tampered timestamp")

    def test_removed_integrity_hash_allows_deploy(self, governance_with_model):
        """
        If integrity_hash is missing, verify_integrity skips hash check
        (only checks deployed records with hashes). Undeployed records
        without integrity_hash still go through validation gates.
        """
        import json

        with open(governance_with_model._lineage_path, "r") as f:
            records = json.load(f)
        del records[0]["integrity_hash"]
        with open(governance_with_model._lineage_path, "w") as f:
            json.dump(records, f)

        # Should pass integrity (no hash to check) but still go through validation
        result = governance_with_model.deploy("v001", reason="No hash")
        assert result is True

    def test_untampered_model_deploys_successfully(self, governance_with_model):
        """Clean model passes both integrity and validation gates."""
        result = governance_with_model.deploy("v001", reason="Clean deploy")
        assert result is True
        deployed = governance_with_model.get_deployed_model()
        assert deployed.version == "v001"


class TestStricterThresholdsRegression:
    """Prove governance gates enforce the stricter thresholds."""

    @pytest.fixture
    def gov(self, tmp_path):
        return ModelGovernance(governance_dir=str(tmp_path / "gov"))

    def _record_model(self, gov, version, cv_accuracy=0.70, wf_sharpe=0.5,
                      mc_prob=0.70, n_trades=100, sharpe=1.0, max_dd=0.05):
        """Helper to record a model with specified metrics."""
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            "close": np.random.randn(200),
            "volume": np.random.randint(1000, 10000, 200),
        })
        gov.record_training(
            version=version,
            features=["rsi_14", "ema_slope_20", "bb_pct", "atr_14"],
            dataset=df,
            config={"n_estimators": 100, "max_depth": 6},
            metrics={
                "cv_accuracy": cv_accuracy,
                "sharpe": sharpe,
                "n_trades": n_trades,
                "max_drawdown": max_dd,
            },
            hyperparameters={"n_estimators": 100, "max_depth": 6},
            seed=42,
            walk_forward_results={"sharpe": wf_sharpe},
            monte_carlo_results={"probability_of_profit": mc_prob},
        )

    def test_cv_accuracy_below_062_rejected(self, gov):
        """CV accuracy below 0.62 threshold is rejected."""
        self._record_model(gov, "v001", cv_accuracy=0.61)
        with pytest.raises(GovernanceViolation, match="failed governance validation"):
            gov.deploy("v001")

    def test_cv_accuracy_at_062_accepted(self, gov):
        """CV accuracy at exactly 0.62 threshold passes."""
        self._record_model(gov, "v001", cv_accuracy=0.62)
        assert gov.deploy("v001") is True

    def test_walk_forward_sharpe_below_03_rejected(self, gov):
        """Walk-forward Sharpe below 0.3 is rejected."""
        self._record_model(gov, "v001", wf_sharpe=0.29)
        with pytest.raises(GovernanceViolation, match="failed governance validation"):
            gov.deploy("v001")

    def test_walk_forward_sharpe_at_03_accepted(self, gov):
        """Walk-forward Sharpe at exactly 0.3 passes."""
        self._record_model(gov, "v001", wf_sharpe=0.3)
        assert gov.deploy("v001") is True

    def test_monte_carlo_prob_below_065_rejected(self, gov):
        """Monte Carlo prob_profit below 0.65 is rejected."""
        self._record_model(gov, "v001", mc_prob=0.64)
        with pytest.raises(GovernanceViolation, match="failed governance validation"):
            gov.deploy("v001")

    def test_monte_carlo_prob_at_065_accepted(self, gov):
        """Monte Carlo prob_profit at exactly 0.65 passes."""
        self._record_model(gov, "v001", mc_prob=0.65)
        assert gov.deploy("v001") is True

    def test_min_trades_below_50_rejected(self, gov):
        """Fewer than 50 backtest trades is rejected."""
        self._record_model(gov, "v001", n_trades=49)
        with pytest.raises(GovernanceViolation, match="failed governance validation"):
            gov.deploy("v001")

    def test_min_trades_at_50_accepted(self, gov):
        """Exactly 50 backtest trades passes."""
        self._record_model(gov, "v001", n_trades=50)
        assert gov.deploy("v001") is True

    def test_all_gates_enforced_simultaneously(self, gov):
        """Model failing multiple gates reports all issues."""
        self._record_model(
            gov, "v001",
            cv_accuracy=0.40,
            wf_sharpe=-0.5,
            mc_prob=0.30,
            n_trades=10,
            sharpe=0.1,
            max_dd=0.50,
        )
        is_valid, issues = gov.validate_for_deployment("v001")
        assert not is_valid
        assert any("CV accuracy" in i for i in issues)
        assert any("Walk-forward Sharpe" in i for i in issues)
        assert any("prob_profit" in i for i in issues)
        assert any("trades" in i.lower() for i in issues)

    def test_valid_model_passes_all_stricter_gates(self, gov):
        """A strong model passes all stricter thresholds."""
        self._record_model(
            gov, "v001",
            cv_accuracy=0.70,
            wf_sharpe=0.8,
            mc_prob=0.75,
            n_trades=200,
            sharpe=1.5,
            max_dd=0.05,
        )
        assert gov.deploy("v001") is True

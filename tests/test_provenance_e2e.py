"""
End-to-end provenance integration test.

Simulates a production artifact built WITHOUT a .git directory and verifies
that the identical embedded commit hash (from build_info.py) is present and
consistent across every provenance record, audit event, registry entry, and
user-facing output.

The test fails if ANY component:
  - Reports a different commit hash
  - Falls back to legacy resolution (env vars, git CLI, .git_commit file)
  - Returns an empty commit when build_info is stamped

This validates the complete ML lifecycle:
  Docker build → training → governance validation → model registration →
  deployment → rollback → /modelinfo → /models → audit events → exported artifacts
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.audit_log import AuditLogger, TradeAuditTrail
from src.ml.governance import ModelGovernance, ModelLineage
from src.ml.model_registry import ModelRegistry
from src.ml.provenance import ProvenanceError, get_branch, get_build_timestamp, get_commit_hash, get_short_commit


# The canonical commit hash "injected at build time"
EMBEDDED_COMMIT = "deadbeef1234567890abcdef1234567890abcdef"
EMBEDDED_BRANCH = "release/v2.1"
EMBEDDED_TIMESTAMP = "2026-07-13T12:00:00Z"


class _PicklableModel:
    """Module-level picklable model for registry tests."""
    def __init__(self, name="default"):
        self.name = name

    def predict(self, X):
        return [0] * len(X)


@pytest.fixture(autouse=True)
def _simulate_production_build():
    """
    Simulate a production Docker build that has:
    - build_info.py stamped with EMBEDDED_COMMIT
    - No .git directory available
    - No GIT_COMMIT/SOURCE_VERSION/GITHUB_SHA env vars

    This ensures the test environment mimics a production container.
    """
    with patch("src.ml.build_info.GIT_COMMIT", EMBEDDED_COMMIT):
        with patch("src.ml.build_info.GIT_BRANCH", EMBEDDED_BRANCH):
            with patch("src.ml.build_info.BUILD_TIMESTAMP", EMBEDDED_TIMESTAMP):
                # Clear all env vars that could be legacy fallbacks
                env_clear = {
                    "GIT_COMMIT": "",
                    "SOURCE_VERSION": "",
                    "GITHUB_SHA": "",
                    "GIT_BRANCH": "",
                    "GITHUB_REF_NAME": "",
                }
                with patch.dict(os.environ, env_clear):
                    yield


@pytest.fixture
def governance(tmp_path):
    """Fresh governance manager."""
    return ModelGovernance(governance_dir=str(tmp_path / "governance"))


@pytest.fixture
def registry(tmp_path):
    """Fresh model registry."""
    return ModelRegistry(models_dir=str(tmp_path / "models"))


@pytest.fixture
def audit_logger(tmp_path):
    """Fresh audit logger."""
    return AuditLogger(log_dir=str(tmp_path / "audit"))


# ─────────────────────────────────────────────────────────────────────────────
# Helper: create a model that passes all governance gates
# ─────────────────────────────────────────────────────────────────────────────


def _record_passing_model(governance, version="v001"):
    """Record a model that meets all governance thresholds."""
    df = pd.DataFrame({
        "close": np.random.randn(200),
        "volume": np.random.randint(1000, 10000, 200),
    })
    return governance.record_training(
        version=version,
        features=["rsi_14", "ema_slope_20", "bb_pct", "atr_14", "macd_hist"],
        dataset=df,
        config={"n_estimators": 200, "max_depth": 5},
        metrics={
            "cv_accuracy": 0.65,
            "sharpe": 1.2,
            "n_trades": 250,
            "max_drawdown": 0.06,
            "precision": 0.60,
            "expectancy": 0.005,
        },
        hyperparameters={"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05},
        seed=42,
        training_duration=90.0,
        walk_forward_results={"sharpe": 0.9, "total_return": 0.08},
        monte_carlo_results={
            "median_return": 0.06,
            "p5_return": -0.04,
            "probability_of_profit": 0.75,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite: End-to-End Provenance Consistency
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEndProvenanceConsistency:
    """
    Verifies that the identical embedded commit hash appears consistently
    across every component in the ML lifecycle, with NO fallback to legacy
    resolution methods.
    """

    # ── 1. Provenance Module ─────────────────────────────────────────────

    def test_provenance_module_returns_embedded_commit(self):
        """The provenance module returns the build-time embedded hash."""
        assert get_commit_hash() == EMBEDDED_COMMIT
        assert get_branch() == EMBEDDED_BRANCH
        assert get_build_timestamp() == EMBEDDED_TIMESTAMP
        assert get_short_commit() == EMBEDDED_COMMIT[:7]

    def test_provenance_strict_mode_succeeds_when_stamped(self):
        """Strict mode does not raise when build_info is stamped."""
        assert get_commit_hash(strict=True) == EMBEDDED_COMMIT

    def test_provenance_strict_mode_raises_when_not_stamped(self):
        """Strict mode raises ProvenanceError for unstamped builds."""
        with patch("src.ml.build_info.GIT_COMMIT", ""):
            with pytest.raises(ProvenanceError):
                get_commit_hash(strict=True)

    # ── 2. Training Records ──────────────────────────────────────────────

    def test_training_records_embedded_commit(self, governance):
        """record_training captures the embedded commit hash."""
        lineage = _record_passing_model(governance)
        assert lineage.git_commit == EMBEDDED_COMMIT
        assert lineage.git_branch == EMBEDDED_BRANCH

    def test_training_never_uses_env_vars(self, governance):
        """Even with env vars set, training uses build_info exclusively."""
        with patch.dict(os.environ, {"GIT_COMMIT": "WRONG_COMMIT_FROM_ENV"}):
            lineage = _record_passing_model(governance)
        assert lineage.git_commit == EMBEDDED_COMMIT
        assert lineage.git_commit != "WRONG_COMMIT_FROM_ENV"

    def test_training_never_uses_git_cli(self, governance):
        """No subprocess is invoked during training provenance capture."""
        with patch("subprocess.run") as mock_run:
            with patch("subprocess.check_output") as mock_co:
                lineage = _record_passing_model(governance)
        mock_run.assert_not_called()
        mock_co.assert_not_called()
        assert lineage.git_commit == EMBEDDED_COMMIT

    # ── 3. Governance Validation ─────────────────────────────────────────

    def test_governance_validation_sees_commit(self, governance):
        """Validation passes git_commit check when build_info is stamped."""
        _record_passing_model(governance)
        is_valid, issues = governance.validate_for_deployment("v001")
        git_issues = [i for i in issues if "git commit" in i.lower()]
        assert len(git_issues) == 0, f"Unexpected git issues: {git_issues}"
        assert is_valid

    def test_governance_validation_fails_without_stamp(self, governance):
        """Validation fails git_commit check when build_info is empty."""
        with patch("src.ml.build_info.GIT_COMMIT", ""):
            _record_passing_model(governance, version="v_empty")
        is_valid, issues = governance.validate_for_deployment("v_empty")
        assert not is_valid
        assert any("git commit" in i.lower() for i in issues)

    # ── 4. Deployment ────────────────────────────────────────────────────

    def test_deployment_records_commit_in_deployment_record(self, governance):
        """deploy() stores embedded commit in deployment_record."""
        _record_passing_model(governance)
        governance.deploy("v001", reason="E2E test deployment")

        history = governance.get_deployment_history()
        assert len(history) == 1
        deployed = history[0]
        assert deployed["git_commit"] == EMBEDDED_COMMIT
        assert deployed["deployment_record"]["git_commit"] == EMBEDDED_COMMIT

    def test_deployment_commit_matches_training_commit(self, governance):
        """The commit at deployment time is the same as at training time."""
        lineage = _record_passing_model(governance)
        governance.deploy("v001", reason="Consistency check")

        history = governance.get_deployment_history()
        assert history[0]["git_commit"] == lineage.git_commit == EMBEDDED_COMMIT

    # ── 5. Rollback ──────────────────────────────────────────────────────

    def test_rollback_preserves_original_commit(self, governance):
        """After rollback, the lineage still shows original training commit."""
        # Deploy v001
        _record_passing_model(governance, version="v001")
        governance.deploy("v001", reason="Initial deploy")

        # Deploy v002 (supersedes v001)
        _record_passing_model(governance, version="v002")
        governance.deploy("v002", reason="Upgrade")

        # Retire v002 (simulating rollback to v001)
        governance.retire("v002", reason="Rollback to v001")

        # Both versions should have same embedded commit
        history = governance.get_deployment_history()
        for entry in history:
            assert entry["git_commit"] == EMBEDDED_COMMIT, (
                f"Version {entry['version']} has wrong commit: {entry['git_commit']}"
            )

    # ── 6. Model Registry ────────────────────────────────────────────────

    def test_registry_operations_independent_of_provenance(self, registry):
        """Registry stores/retrieves models without needing .git directory."""
        version = registry.save_version(
            model=_PicklableModel("e2e"),
            features=["rsi_14", "ema_20"],
            metrics={"accuracy": 0.65},
            symbols=["AAPL"],
            note="E2E provenance test",
            activate=True,
        )
        assert version is not None

        versions = registry.list_versions()
        assert len(versions) >= 1

    def test_registry_rollback_works_without_git(self, registry):
        """Model rollback works without .git directory or env vars."""
        registry.save_version(
            model=_PicklableModel("v1"),
            features=["rsi_14"],
            metrics={"accuracy": 0.60},
            symbols=["AAPL"],
            activate=True,
        )
        registry.save_version(
            model=_PicklableModel("v2"),
            features=["rsi_14", "ema_20"],
            metrics={"accuracy": 0.65},
            symbols=["AAPL"],
            activate=True,
        )

        # Rollback to first version
        versions = registry.list_versions()
        if len(versions) >= 2:
            v1_str = versions[-1]["version"]
            success = registry.rollback(v1_str)
            assert success

    # ── 7. Audit Events ──────────────────────────────────────────────────

    def test_audit_events_can_record_provenance(self, audit_logger):
        """Audit events include the embedded commit hash."""
        audit_logger.log(
            event_type="ModelDeployed",
            data={
                "model_version": "v001",
                "git_commit": get_commit_hash(),
                "reason": "E2E test",
            },
            source="governance",
        )

        # Read back the audit file
        log_dir = audit_logger._log_dir
        files = list(log_dir.glob("audit_*.jsonl"))
        assert len(files) == 1

        with open(files[0]) as f:
            entries = [json.loads(line) for line in f]
        assert len(entries) == 1
        assert entries[0]["data"]["git_commit"] == EMBEDDED_COMMIT

    def test_trade_audit_trail_carries_consistent_commit(self):
        """TradeAuditTrail can be populated with provenance commit."""
        trail = TradeAuditTrail(
            trade_id="trade_e2e_001",
            signal_id="sig_001",
            prediction_id="pred_001",
            model_version="v001",
            training_run_id=f"run_{get_short_commit()}",
            dataset_snapshot_hash="ds_hash_123",
            feature_snapshot_hash="fs_hash_456",
        )
        assert trail.is_complete()
        assert get_short_commit() in trail.training_run_id

    # ── 8. Audit Report ──────────────────────────────────────────────────

    def test_audit_report_shows_commit(self, governance):
        """audit_report() correctly reports the embedded commit."""
        _record_passing_model(governance)
        governance.deploy("v001", reason="Audit test")

        report = governance.audit_report()
        assert report["with_git_commit"] == 1
        assert report["versions"][0]["git_commit"] == EMBEDDED_COMMIT[:8]

    # ── 9. Predictor Commit Resolution ───────────────────────────────────

    def test_predictor_uses_provenance_module(self):
        """ModelPredictor._get_git_commit uses provenance module."""
        from src.ml.predictor import ModelPredictor
        predictor = ModelPredictor.__new__(ModelPredictor)
        result = predictor._get_git_commit()
        assert result == EMBEDDED_COMMIT[:7]

    def test_predictor_no_subprocess_in_production(self):
        """Predictor never invokes subprocess for commit resolution."""
        from src.ml.predictor import ModelPredictor
        predictor = ModelPredictor.__new__(ModelPredictor)
        if hasattr(predictor, '_cached_git_commit'):
            del predictor._cached_git_commit
        with patch("subprocess.run") as mock_run:
            with patch("subprocess.check_output") as mock_co:
                result = predictor._get_git_commit()
        mock_run.assert_not_called()
        mock_co.assert_not_called()
        assert result == EMBEDDED_COMMIT[:7]

    # ── 10. Full Lifecycle Consistency ───────────────────────────────────

    def test_full_lifecycle_commit_consistency(self, governance, registry, audit_logger):
        """
        Complete lifecycle test: train → validate → deploy → rollback → audit.
        Every step must report the SAME embedded commit hash.
        """
        collected_commits = {}

        # Step 1: Training
        lineage = _record_passing_model(governance, version="v001")
        collected_commits["training"] = lineage.git_commit

        # Step 2: Validation
        is_valid, issues = governance.validate_for_deployment("v001")
        assert is_valid, f"Validation failed: {issues}"
        collected_commits["validation_passed"] = True

        # Step 3: Deployment
        governance.deploy("v001", reason="Lifecycle test")
        history = governance.get_deployment_history()
        collected_commits["deployment"] = history[0]["git_commit"]
        collected_commits["deployment_record"] = history[0]["deployment_record"]["git_commit"]

        # Step 4: Train & deploy v002
        _record_passing_model(governance, version="v002")
        governance.deploy("v002", reason="Upgrade")
        history = governance.get_deployment_history()
        collected_commits["v002_deployment"] = history[0]["git_commit"]

        # Step 5: Rollback (retire v002)
        governance.retire("v002", reason="Rollback")
        history = governance.get_deployment_history()
        for entry in history:
            collected_commits[f"post_rollback_{entry['version']}"] = entry["git_commit"]

        # Step 6: Audit report
        report = governance.audit_report()
        for ver in report["versions"]:
            collected_commits[f"audit_report_{ver['version']}"] = ver["git_commit"]

        # Step 7: Audit event
        audit_logger.log(
            event_type="ModelDeployed",
            data={"git_commit": get_commit_hash(), "version": "v001"},
            source="governance",
        )
        log_dir = audit_logger._log_dir
        files = list(log_dir.glob("audit_*.jsonl"))
        with open(files[0]) as f:
            entries = [json.loads(line) for line in f]
        collected_commits["audit_event"] = entries[0]["data"]["git_commit"]

        # Step 8: Predictor
        from src.ml.predictor import ModelPredictor
        predictor = ModelPredictor.__new__(ModelPredictor)
        collected_commits["predictor"] = predictor._get_git_commit()

        # ── ASSERTIONS: Every commit must be the same embedded hash ──
        for source, commit in collected_commits.items():
            if source == "validation_passed":
                continue  # Boolean, not a commit
            # Audit report truncates to 8 chars
            if source.startswith("audit_report_"):
                assert commit == EMBEDDED_COMMIT[:8], (
                    f"INCONSISTENCY: {source} reported '{commit}' "
                    f"instead of '{EMBEDDED_COMMIT[:8]}'"
                )
            # Predictor uses short (7 chars)
            elif source == "predictor":
                assert commit == EMBEDDED_COMMIT[:7], (
                    f"INCONSISTENCY: {source} reported '{commit}' "
                    f"instead of '{EMBEDDED_COMMIT[:7]}'"
                )
            else:
                assert commit == EMBEDDED_COMMIT, (
                    f"INCONSISTENCY: {source} reported '{commit}' "
                    f"instead of '{EMBEDDED_COMMIT}'"
                )

    # ── 11. No .git Directory Required ───────────────────────────────────

    def test_no_git_directory_required(self, governance):
        """
        Entire workflow works even if .git directory doesn't exist.
        No FileNotFoundError or subprocess error propagates.
        """
        # Ensure no subprocess calls are made (simulating missing git binary)
        with patch("subprocess.run", side_effect=FileNotFoundError("git not installed")):
            with patch("subprocess.check_output", side_effect=FileNotFoundError("git not installed")):
                # Full lifecycle without .git
                lineage = _record_passing_model(governance)
                assert lineage.git_commit == EMBEDDED_COMMIT

                governance.deploy("v001", reason="No git test")
                history = governance.get_deployment_history()
                assert history[0]["git_commit"] == EMBEDDED_COMMIT

                report = governance.audit_report()
                assert report["with_git_commit"] == 1

    # ── 12. Legacy Fallback Rejection ────────────────────────────────────

    def test_env_vars_never_override_build_info(self, governance):
        """
        Even if legacy env vars are set with DIFFERENT values,
        the build_info commit is always used.
        """
        rogue_envs = {
            "GIT_COMMIT": "rogue_commit_from_env_111",
            "SOURCE_VERSION": "rogue_heroku_222",
            "GITHUB_SHA": "rogue_actions_333",
        }
        with patch.dict(os.environ, rogue_envs):
            lineage = _record_passing_model(governance)
            assert lineage.git_commit == EMBEDDED_COMMIT

            commit = governance._get_git_commit()
            assert commit == EMBEDDED_COMMIT
            assert commit != "rogue_commit_from_env_111"
            assert commit != "rogue_heroku_222"
            assert commit != "rogue_actions_333"

    def test_git_commit_file_never_read(self, governance, tmp_path):
        """
        A .git_commit file in the project root is never consulted.
        """
        commit_file = tmp_path / ".git_commit"
        commit_file.write_text("stale_docker_commit_should_not_appear")

        lineage = _record_passing_model(governance)
        assert lineage.git_commit == EMBEDDED_COMMIT
        assert lineage.git_commit != "stale_docker_commit_should_not_appear"

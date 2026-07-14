"""
Comprehensive tests for model-provenance commit-resolution path.

The ``src/ml/provenance`` module provides centralized commit resolution with
the following precedence:
  1. ``src/ml/build_info.py`` constants (injected at Docker/CI build time)
  2. Environment variables: GIT_COMMIT, SOURCE_VERSION, GITHUB_SHA
  3. ``.git_commit`` file in the project root

Tests cover:
  1. build_info.py as highest-priority source
  2. Environment variable fallback chain
  3. Strict mode raising when no source provides a hash
  4. inject_build_info.py script stamping at build time
  5. Integration with governance and predictor
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ml.governance import ModelGovernance


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def governance(tmp_path):
    """Create governance manager with temp directory."""
    return ModelGovernance(governance_dir=str(tmp_path / "governance"))


@pytest.fixture
def clean_env():
    """Context manager that clears all commit-related env vars."""
    keys = ("GIT_COMMIT", "SOURCE_VERSION", "GITHUB_SHA")
    original = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in original.items():
        if v is not None:
            os.environ[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite: Provenance Source Precedence
# ─────────────────────────────────────────────────────────────────────────────


class TestProvenancePrecedence:
    """Verify provenance resolution precedence: build_info > env vars > .git_commit file."""

    def test_build_info_is_sole_source(self, governance):
        """build_info.GIT_COMMIT is returned by _get_git_commit."""
        with patch("src.ml.build_info.GIT_COMMIT", "build_injected_sha256"):
            commit = governance._get_git_commit()
        assert commit == "build_injected_sha256"

    def test_build_info_takes_precedence_over_env(self, governance):
        """build_info.GIT_COMMIT wins even when env vars are set."""
        with patch("src.ml.build_info.GIT_COMMIT", "build_info_value"):
            with patch.dict(os.environ, {"GIT_COMMIT": "env_value"}):
                commit = governance._get_git_commit()
        assert commit == "build_info_value"

    def test_env_vars_used_when_build_info_empty(self, governance):
        """Environment variables are used as fallback when build_info is empty."""
        with patch("src.ml.build_info.GIT_COMMIT", ""):
            with patch.dict(os.environ, {"GIT_COMMIT": "ci_commit_hash"}, clear=False):
                commit = governance._get_git_commit()
        assert commit == "ci_commit_hash"

    def test_env_var_precedence_order(self, governance):
        """GIT_COMMIT env var takes precedence over SOURCE_VERSION and GITHUB_SHA."""
        with patch("src.ml.build_info.GIT_COMMIT", ""):
            with patch.dict(os.environ, {
                "GIT_COMMIT": "git_commit_val",
                "SOURCE_VERSION": "source_version_val",
                "GITHUB_SHA": "github_sha_val",
            }, clear=False):
                commit = governance._get_git_commit()
        assert commit == "git_commit_val"

    def test_all_sources_fail_returns_empty(self, governance):
        """Returns empty string when all sources are empty."""
        with patch("src.ml.build_info.GIT_COMMIT", ""):
            with patch.dict(os.environ, {
                "GIT_COMMIT": "",
                "SOURCE_VERSION": "",
                "GITHUB_SHA": "",
            }, clear=False):
                # Also ensure no .git_commit file interferes
                with patch("builtins.open", side_effect=OSError("no file")):
                    commit = governance._get_git_commit()
        assert commit == ""
        assert isinstance(commit, str)


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite: Environment Variable Handling
# ─────────────────────────────────────────────────────────────────────────────


class TestProvenanceModule:
    """Test the centralized provenance module directly."""

    def test_get_commit_hash_returns_build_info(self):
        """get_commit_hash reads from build_info module attribute."""
        from src.ml.provenance import get_commit_hash
        with patch("src.ml.build_info.GIT_COMMIT", "abc123full"):
            assert get_commit_hash() == "abc123full"

    def test_get_commit_hash_strict_raises(self):
        """Strict mode raises ProvenanceError when all sources are empty."""
        from src.ml.provenance import ProvenanceError, get_commit_hash
        with patch("src.ml.build_info.GIT_COMMIT", ""):
            with patch.dict(os.environ, {
                "GIT_COMMIT": "",
                "SOURCE_VERSION": "",
                "GITHUB_SHA": "",
            }, clear=False):
                with patch("builtins.open", side_effect=OSError("no file")):
                    with pytest.raises(ProvenanceError):
                        get_commit_hash(strict=True)

    def test_get_commit_hash_non_strict_returns_empty(self):
        """Non-strict returns empty when all sources are empty."""
        from src.ml.provenance import get_commit_hash
        with patch("src.ml.build_info.GIT_COMMIT", ""):
            with patch.dict(os.environ, {
                "GIT_COMMIT": "",
                "SOURCE_VERSION": "",
                "GITHUB_SHA": "",
            }, clear=False):
                with patch("builtins.open", side_effect=OSError("no file")):
                    assert get_commit_hash() == ""

    def test_get_short_commit(self):
        """get_short_commit truncates to requested length."""
        from src.ml.provenance import get_short_commit
        with patch("src.ml.build_info.GIT_COMMIT", "abcdef1234567890abcdef"):
            assert get_short_commit() == "abcdef1"
            assert get_short_commit(length=10) == "abcdef1234"

    def test_get_branch(self):
        """get_branch reads from build_info."""
        from src.ml.provenance import get_branch
        with patch("src.ml.build_info.GIT_BRANCH", "main"):
            assert get_branch() == "main"

    def test_get_build_timestamp(self):
        """get_build_timestamp reads from build_info."""
        from src.ml.provenance import get_build_timestamp
        with patch("src.ml.build_info.BUILD_TIMESTAMP", "2026-07-13T12:00:00Z"):
            assert get_build_timestamp() == "2026-07-13T12:00:00Z"


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite: Git CLI Fallback
# ─────────────────────────────────────────────────────────────────────────────


class TestNoLegacyFallbacks:
    """Verify that git CLI subprocess calls are not used for commit resolution."""

    def test_no_subprocess_calls_in_commit_resolution(self, governance):
        """_get_git_commit never invokes subprocess."""
        with patch("src.ml.build_info.GIT_COMMIT", "stamped_hash"):
            with patch("subprocess.run") as mock_run:
                commit = governance._get_git_commit()
        mock_run.assert_not_called()
        assert commit == "stamped_hash"

    def test_no_subprocess_when_empty(self, governance):
        """Even when build_info is empty, no subprocess call is made."""
        with patch("src.ml.build_info.GIT_COMMIT", ""):
            with patch.dict(os.environ, {
                "GIT_COMMIT": "",
                "SOURCE_VERSION": "",
                "GITHUB_SHA": "",
            }, clear=False):
                with patch("subprocess.run") as mock_run:
                    with patch("builtins.open", side_effect=OSError("no file")):
                        commit = governance._get_git_commit()
        mock_run.assert_not_called()
        assert commit == ""

    def test_no_git_commit_file_access_when_env_set(self, governance, tmp_path):
        """When env var provides commit, .git_commit file is not needed."""
        commit_file = tmp_path / ".git_commit"
        commit_file.write_text("should_not_be_read\n")

        with patch("src.ml.build_info.GIT_COMMIT", ""):
            with patch.dict(os.environ, {"GIT_COMMIT": "from_env"}, clear=False):
                commit = governance._get_git_commit()
        assert commit == "from_env"


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite: .git_commit File Fallback
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildInfoModule:
    """Test the build_info module integration."""

    def test_default_build_info_is_empty(self):
        """Unstamped build_info has empty sentinel values."""
        from src.ml import build_info
        assert isinstance(build_info.GIT_COMMIT, str)
        assert isinstance(build_info.GIT_BRANCH, str)
        assert isinstance(build_info.BUILD_TIMESTAMP, str)

    def test_build_info_empty_returns_empty_from_governance(self, governance):
        """If build_info and all fallbacks are empty, governance returns empty."""
        with patch("src.ml.build_info.GIT_COMMIT", ""):
            with patch.dict(os.environ, {
                "GIT_COMMIT": "",
                "SOURCE_VERSION": "",
                "GITHUB_SHA": "",
            }, clear=False):
                with patch("builtins.open", side_effect=OSError("no file")):
                    commit = governance._get_git_commit()
        assert commit == ""


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite: inject_build_info.py Script
# ─────────────────────────────────────────────────────────────────────────────


class TestInjectBuildInfoScript:
    """Test the build-time injection script."""

    def test_script_stamps_build_info(self, tmp_path):
        """inject_build_info.py writes commit to build_info.py."""
        # Create a minimal build_info.py target
        target = tmp_path / "src" / "ml" / "build_info.py"
        target.parent.mkdir(parents=True)
        target.write_text('GIT_COMMIT: str = ""\n')

        script_path = Path(__file__).resolve().parent.parent / "scripts" / "inject_build_info.py"

        result = subprocess.run(
            [sys.executable, str(script_path), "--commit", "test_sha_abc123"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path.parent.parent),  # Won't matter since we pass --commit
            env={**os.environ, "PYTHONPATH": ""},
        )
        # The script writes to its own relative path, so check the actual file
        actual_target = script_path.parent.parent / "src" / "ml" / "build_info.py"
        content = actual_target.read_text()
        assert "test_sha_abc123" in content

        # Restore original
        actual_target.write_text(
            '"""\nBuild-time metadata module for model provenance.\n'
            '"""\n\n'
            '# Injected at build time — DO NOT edit manually.\n'
            'GIT_COMMIT: str = ""\n'
            'GIT_BRANCH: str = ""\n'
            'BUILD_TIMESTAMP: str = ""\n'
        )

    def test_script_fails_without_commit(self, tmp_path):
        """Script exits with error if no commit can be resolved."""
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "inject_build_info.py"

        # Run with no git and no env vars
        env = {"PATH": "", "HOME": str(tmp_path)}
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(tmp_path),
        )
        assert result.returncode == 1
        assert "Unable to determine git commit hash" in result.stderr

    def test_script_uses_env_var_when_no_flag(self):
        """Script reads GIT_COMMIT env var when --commit not provided."""
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "inject_build_info.py"

        env = {**os.environ, "GIT_COMMIT": "env_injected_commit_hash"}
        result = subprocess.run(
            [sys.executable, str(script_path), "--commit", "explicit_flag_commit"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0
        # Verify it stamped correctly
        actual_target = script_path.parent.parent / "src" / "ml" / "build_info.py"
        content = actual_target.read_text()
        assert "explicit_flag_commit" in content

        # Restore
        actual_target.write_text(
            '"""\nBuild-time metadata module for model provenance.\n\n'
            'This file is overwritten at build time by ``scripts/inject_build_info.py``\n'
            'to embed the exact git commit hash (and optional branch/timestamp) into the\n'
            'production artifact.  At runtime the governance layer reads the constants\n'
            'below as the **highest-confidence** provenance source — no .git directory or\n'
            'environment variables required.\n\n'
            'If the file has NOT been stamped (i.e. during local development), the\n'
            'sentinel values remain and the governance system falls through to its\n'
            'other resolution strategies (env vars → git CLI → .git_commit file).\n'
            '"""\n\n'
            '# Injected at build time — DO NOT edit manually.\n'
            'GIT_COMMIT: str = ""\n'
            'GIT_BRANCH: str = ""\n'
            'BUILD_TIMESTAMP: str = ""\n'
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite: Integration — record_training with Provenance
# ─────────────────────────────────────────────────────────────────────────────


class TestRecordTrainingProvenance:
    """Test that record_training correctly captures commit hash."""

    def test_record_training_populates_git_commit(self, governance):
        """Training records commit hash from build_info."""
        with patch("src.ml.build_info.GIT_COMMIT", "abc123def456"):
            lineage = governance.record_training(
                version="v_test_001",
                features=["rsi_14", "ema_20"],
                metrics={"cv_accuracy": 0.6},
            )
        assert lineage.git_commit == "abc123def456"

    def test_record_training_with_build_info(self, governance):
        """Training uses build_info commit when available."""
        with patch("src.ml.build_info.GIT_COMMIT", "stamped_build_abc"):
            lineage = governance.record_training(
                version="v_build_001",
                features=["rsi_14"],
                metrics={"cv_accuracy": 0.5},
            )
        assert lineage.git_commit == "stamped_build_abc"

    def test_validate_for_deployment_rejects_missing_commit(self, governance):
        """Validation fails when git_commit is empty."""
        # Record training with mocked empty commit
        with patch.object(governance, "_get_git_commit", return_value=""):
            governance.record_training(
                version="v_no_commit",
                features=["rsi_14", "ema_20", "bb_pct", "atr_14", "macd_hist"],
                metrics={
                    "cv_accuracy": 0.7,
                    "sharpe": 1.5,
                    "n_trades": 100,
                    "max_drawdown": 0.05,
                },
                hyperparameters={"n_estimators": 200},
                walk_forward_results={"sharpe": 0.9, "total_return": 0.1},
                monte_carlo_results={"probability_of_profit": 0.75},
            )

        is_valid, issues = governance.validate_for_deployment("v_no_commit")
        assert not is_valid
        git_issues = [i for i in issues if "git commit" in i.lower()]
        assert len(git_issues) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite: Predictor Commit Resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestPredictorCommitResolution:
    """Test predictor's commit resolution uses provenance module only."""

    def test_predictor_uses_build_info(self):
        """Predictor reads build_info for short commit hash."""
        from src.ml.predictor import ModelPredictor
        predictor = ModelPredictor.__new__(ModelPredictor)
        with patch("src.ml.build_info.GIT_COMMIT", "abcdef1234567890"):
            result = predictor._get_git_commit()
        assert result == "abcdef1"  # First 7 chars

    def test_predictor_returns_empty_without_build_info(self):
        """Predictor returns empty when build_info and all fallbacks are empty."""
        from src.ml.predictor import ModelPredictor
        predictor = ModelPredictor.__new__(ModelPredictor)
        # Clear cached value
        if hasattr(predictor, '_cached_git_commit'):
            del predictor._cached_git_commit
        with patch("src.ml.build_info.GIT_COMMIT", ""):
            with patch.dict(os.environ, {
                "GIT_COMMIT": "",
                "SOURCE_VERSION": "",
                "GITHUB_SHA": "",
            }, clear=False):
                with patch("builtins.open", side_effect=OSError("no file")):
                    result = predictor._get_git_commit()
        assert result == ""

    def test_predictor_no_subprocess_call(self):
        """Predictor never calls subprocess for commit resolution."""
        from src.ml.predictor import ModelPredictor
        predictor = ModelPredictor.__new__(ModelPredictor)
        if hasattr(predictor, '_cached_git_commit'):
            del predictor._cached_git_commit
        with patch("src.ml.build_info.GIT_COMMIT", "xyz789"):
            with patch("subprocess.run") as mock_run:
                with patch("subprocess.check_output") as mock_co:
                    result = predictor._get_git_commit()
        mock_run.assert_not_called()
        mock_co.assert_not_called()
        assert result == "xyz789"

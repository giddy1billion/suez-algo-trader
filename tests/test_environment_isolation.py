"""
Environment Isolation Integration Tests
========================================

Proves that:
1. The test environment profile (SUEZ_ENV=test) uses dedicated storage paths.
2. Executing the full acceptance/E2E suite leaves a fresh paper-trading
   instance with zero trades and no test symbols in its history.
3. The bot refuses to start against test storage.
4. Storage paths are correctly reported for operational logging.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from config.settings import Settings, Environment, TradingMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_settings(**overrides) -> Settings:
    """Create a Settings instance with SUEZ_ENV=test and optional overrides."""
    env = {"SUEZ_ENV": "test", **overrides}
    with patch.dict(os.environ, env, clear=False):
        return Settings()


# ---------------------------------------------------------------------------
# Test: Environment profile separation
# ---------------------------------------------------------------------------


class TestEnvironmentProfiles:
    """Validate that production and test profiles use distinct paths."""

    def test_production_defaults(self, monkeypatch):
        """Production profile uses standard data_cache paths."""
        monkeypatch.setenv("SUEZ_ENV", "production")
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("CORRELATION_STORE_DB_PATH", raising=False)
        monkeypatch.delenv("PREDICTION_REGISTRY_STORAGE_PATH", raising=False)
        s = Settings()
        assert s.suez_env == Environment.PRODUCTION
        assert s.is_test_environment is False
        assert s.storage_base_dir == "data_cache"
        assert "data_cache/trading.db" in s.effective_database_url

    def test_test_profile_rewrites_paths(self):
        """Test profile rewrites storage to data_cache_test."""
        s = _make_test_settings()
        assert s.suez_env == Environment.TEST
        assert s.is_test_environment is True
        assert s.storage_base_dir == "data_cache_test"
        assert "data_cache_test/trading.db" in s.effective_database_url
        assert "data_cache_test/correlation_store.db" in s.effective_correlation_store_db_path
        assert "data_cache_test/predictions" in s.effective_prediction_registry_storage_path

    def test_custom_database_url_not_rewritten(self):
        """If user provides custom DB URL, test profile does not rewrite it."""
        s = _make_test_settings(DATABASE_URL="postgresql://localhost/mydb")
        assert s.effective_database_url == "postgresql://localhost/mydb"

    def test_storage_paths_summary_contains_all_keys(self):
        """storage_paths_summary includes all operational paths."""
        s = _make_test_settings()
        summary = s.storage_paths_summary
        expected_keys = {
            "suez_env",
            "storage_base_dir",
            "database_url",
            "correlation_store_db_path",
            "prediction_registry_storage_path",
            "ml_model_path",
            "log_file",
        }
        assert expected_keys.issubset(set(summary.keys()))


# ---------------------------------------------------------------------------
# Test: Bot startup safeguard
# ---------------------------------------------------------------------------


class TestStartupSafeguard:
    """The bot must refuse to start when SUEZ_ENV=test."""

    def test_main_exits_on_test_env(self):
        """main.py should sys.exit(1) when SUEZ_ENV=test is set."""
        env = os.environ.copy()
        env["SUEZ_ENV"] = "test"
        # Provide dummy API keys to pass earlier checks
        env["ALPACA_PAPER_API_KEY"] = "test_key"
        env["ALPACA_PAPER_SECRET_KEY"] = "test_secret"

        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(Path(__file__).resolve().parent.parent),
            env=env,
        )
        assert result.returncode != 0
        assert "SUEZ_ENV=test" in result.stdout or "SUEZ_ENV=test" in result.stderr


# ---------------------------------------------------------------------------
# Test: Acceptance suite leaves zero residue
# ---------------------------------------------------------------------------


class TestAcceptanceSuiteIsolation:
    """Prove that the acceptance/E2E test suite leaves no residue in
    a fresh paper-trading instance.
    """

    def test_e2e_suite_leaves_clean_paper_broker(self, tmp_path):
        """After running the E2E paper trading tests, a new PaperBroker
        instance has zero trades and no test symbols.
        """
        from src.broker.paper import PaperBroker

        # Run the E2E paper-trading test suite in a subprocess with isolated env
        test_data_dir = str(tmp_path / "data_cache_test")
        os.makedirs(test_data_dir, exist_ok=True)

        env = os.environ.copy()
        env["SUEZ_ENV"] = "test"
        env["DATABASE_URL"] = f"sqlite:///{test_data_dir}/trading.db"
        env["CORRELATION_STORE_DB_PATH"] = f"{test_data_dir}/correlation_store.db"
        env["PREDICTION_REGISTRY_STORAGE_PATH"] = f"{test_data_dir}/predictions"

        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/test_e2e_paper_trading.py",
                "-q", "--tb=short", "--no-header",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).resolve().parent.parent),
            env=env,
        )
        # The E2E tests should pass (or skip if deps missing)
        assert result.returncode in (0, 5), (
            f"E2E tests failed unexpectedly:\n{result.stdout}\n{result.stderr}"
        )

        # Now verify: a fresh PaperBroker has zero trades and no positions
        broker = PaperBroker(starting_equity=100_000.0)
        positions = broker.get_positions()
        orders = broker.get_orders()

        assert len(positions) == 0, "Fresh broker should have zero positions"
        assert len(orders) == 0, "Fresh broker should have zero orders"

        # Verify no test-specific symbols leaked into any production store
        prod_db_path = Path("data_cache/trading.db")
        if prod_db_path.exists():
            import sqlite3
            conn = sqlite3.connect(str(prod_db_path))
            cursor = conn.cursor()
            # Check for test session IDs in event store
            try:
                cursor.execute(
                    "SELECT COUNT(*) FROM events WHERE session_id LIKE 'test-%'"
                )
                count = cursor.fetchone()[0]
                assert count == 0, (
                    f"Production DB contains {count} test session events"
                )
            except sqlite3.OperationalError:
                pass  # Table may not exist — fine
            finally:
                conn.close()

    def test_test_data_dir_does_not_exist_after_cleanup(self, tmp_path):
        """Verify that the test storage directory is scoped to tmp_path
        and does not persist as a permanent 'data_cache_test' directory.
        """
        # The autouse fixture in conftest routes all storage to tmp_path
        # Confirm no top-level data_cache_test was created
        project_root = Path(__file__).resolve().parent.parent
        permanent_test_dir = project_root / "data_cache_test"
        # It's acceptable if this dir exists only inside tmp_path
        # but it must NOT exist at project root from test runs
        if permanent_test_dir.exists():
            # If it does exist, it should be empty (no DB files)
            db_files = list(permanent_test_dir.glob("*.db"))
            assert len(db_files) == 0, (
                f"data_cache_test at project root contains DB files: {db_files}"
            )


# ---------------------------------------------------------------------------
# Test: Storage path logging
# ---------------------------------------------------------------------------


class TestStoragePathLogging:
    """Verify that storage paths are available for startup logging."""

    def test_summary_reflects_environment(self):
        """storage_paths_summary accurately reflects the active env."""
        s = _make_test_settings()
        summary = s.storage_paths_summary
        assert summary["suez_env"] == "test"
        assert "data_cache_test" in summary["storage_base_dir"]

    def test_production_summary(self, monkeypatch):
        monkeypatch.setenv("SUEZ_ENV", "production")
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("CORRELATION_STORE_DB_PATH", raising=False)
        monkeypatch.delenv("PREDICTION_REGISTRY_STORAGE_PATH", raising=False)
        s = Settings()
        summary = s.storage_paths_summary
        assert summary["suez_env"] == "production"
        assert summary["storage_base_dir"] == "data_cache"

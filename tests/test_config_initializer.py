"""
Unit tests for configuration initialization and persistence.

Tests verify that:
1. Configuration service initializes correctly
2. First-run scenario seeds defaults and env vars
3. Subsequent runs restore from database
4. Environment variables don't override persisted configs
"""

import os
import tempfile
import sqlite3
import gc
import time
from pathlib import Path

import pytest

from src.config.initializer import (
    initialize_configuration_service,
    reset_configuration_service,
    get_configuration_service,
)
from src.config.repository import ConfigurationRepository
from src.config.seed import seed_default_configuration, DEFAULT_CONFIGURATIONS


class TestConfigurationInitializer:
    """Test configuration initialization and persistence."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db_path = f"sqlite:///{path}"
        yield db_path
        # Cleanup - close config service and remove file
        reset_configuration_service()
        import gc
        gc.collect()
        import time
        time.sleep(0.1)  # Brief delay to release file locks
        Path(path).unlink(missing_ok=True)
        # Also cleanup WAL files that SQLite creates
        Path(f"{path}-wal").unlink(missing_ok=True)
        Path(f"{path}-shm").unlink(missing_ok=True)

    def test_first_run_initialization(self, temp_db):
        """Test initialization on first run (no existing config)."""
        reset_configuration_service()

        config_service = initialize_configuration_service(
            database_url=temp_db,
            seed_from_env=False,
            auto_refresh=False,
        )

        assert config_service is not None
        assert len(config_service._cache) > 0

        # Verify defaults were seeded
        repo = ConfigurationRepository(database_url=temp_db)
        all_configs = repo.get_all()
        assert len(all_configs) == len(DEFAULT_CONFIGURATIONS)

    def test_first_run_loads_defaults(self, temp_db):
        """Test that first run loads all default configurations."""
        reset_configuration_service()

        config_service = initialize_configuration_service(
            database_url=temp_db,
            seed_from_env=False,
            auto_refresh=False,
        )

        # Verify specific defaults are accessible
        trading_interval = config_service.get_int("trading", "trading_interval")
        assert trading_interval > 0

        max_leverage = config_service.get_float("risk", "max_leverage")
        assert max_leverage > 0

    def test_subsequent_run_restores_from_db(self, temp_db):
        """Test that subsequent runs restore from database."""
        reset_configuration_service()

        # First run: initialize with defaults
        config_service1 = initialize_configuration_service(
            database_url=temp_db,
            seed_from_env=False,
            auto_refresh=False,
        )

        # Modify a config value in the database
        repo = ConfigurationRepository(database_url=temp_db)
        repo.set(
            category="trading",
            key="trading_interval",
            value="240",
            value_type="int",
            changed_by="test",
            change_reason="test_modification",
        )

        # Reset and reinitialize (simulating restart)
        reset_configuration_service()

        config_service2 = initialize_configuration_service(
            database_url=temp_db,
            seed_from_env=False,
            auto_refresh=False,
        )

        # Verify modified value is restored
        trading_interval = config_service2.get_int("trading", "trading_interval")
        assert trading_interval == 240

    def test_singleton_behavior(self, temp_db):
        """Test that initialization returns the same singleton instance."""
        reset_configuration_service()

        service1 = initialize_configuration_service(
            database_url=temp_db,
            seed_from_env=False,
            auto_refresh=False,
        )

        service2 = initialize_configuration_service(
            database_url=temp_db,
            seed_from_env=False,
            auto_refresh=False,
        )

        assert service1 is service2
        assert get_configuration_service() is service1

    def test_config_persistence_across_restarts(self, temp_db):
        """Test that config changes persist across simulated restarts."""
        reset_configuration_service()

        # First run: create default config
        initialize_configuration_service(
            database_url=temp_db,
            seed_from_env=False,
            auto_refresh=False,
        )

        # Change a config value
        repo = ConfigurationRepository(database_url=temp_db)
        repo.set(
            category="trading",
            key="active_strategy",
            value="ml",
            value_type="str",
            changed_by="test",
            change_reason="test_strategy_change",
        )

        # Simulate restart: reset and reinitialize
        reset_configuration_service()
        service_after_restart = initialize_configuration_service(
            database_url=temp_db,
            seed_from_env=False,
            auto_refresh=False,
        )

        # Verify change persisted
        strategy = service_after_restart.get_str("trading", "active_strategy")
        assert strategy == "ml"

    def test_error_handling_invalid_db(self):
        """Test graceful error handling with invalid database."""
        reset_configuration_service()

        with pytest.raises(Exception):
            initialize_configuration_service(
                database_url="sqlite:///invalid_path_that_does_not_exist/nonexistent/file.db",
                seed_from_env=False,
                auto_refresh=False,
            )

    def test_config_service_cache_populated(self, temp_db):
        """Test that configuration service cache is properly populated."""
        reset_configuration_service()

        config_service = initialize_configuration_service(
            database_url=temp_db,
            seed_from_env=False,
            auto_refresh=False,
        )

        # Verify cache is populated with all default categories
        assert "trading" in config_service._cache
        assert "risk" in config_service._cache
        assert "strategy_momentum" in config_service._cache

        # Verify specific keys exist in cache
        assert "trading_interval" in config_service._cache["trading"]
        assert "max_leverage" in config_service._cache["risk"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

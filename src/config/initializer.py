"""
Configuration Initializer — Bootstrap the database-backed configuration system.

Handles the startup sequence for loading persisted configs from the database,
with fallback to environment variables on first run.

Usage:
    from src.config.initializer import initialize_configuration_service
    config_service = initialize_configuration_service(database_url)
"""

import os
from typing import Optional

from src.config.repository import ConfigurationRepository
from src.config.service import ConfigurationService
from src.config.seed import seed_default_configuration, seed_from_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level singleton
_config_service: Optional[ConfigurationService] = None


def initialize_configuration_service(
    database_url: str = "sqlite:///data_cache/trading.db",
    seed_from_env: bool = True,
    auto_refresh: bool = True,
) -> ConfigurationService:
    """
    Initialize the configuration service, restoring persisted configs from DB.

    Configuration startup precedence:
    1. Check if config snapshot exists in database
    2. If yes: load from database (user changes persist)
    3. If no (first run):
       a. Seed with defaults from DEFAULT_CONFIGURATIONS
       b. If seed_from_env=True: overlay with current environment variables
       c. Save to database
    4. Create and return ConfigurationService instance

    Args:
        database_url: Database connection string (default: SQLite trading.db)
        seed_from_env: On first run, overlay env vars over defaults (default: True)
        auto_refresh: Enable background refresh of config from DB (default: True)

    Returns:
        ConfigurationService singleton instance

    Raises:
        Exception: If database initialization fails
    """
    global _config_service

    if _config_service is not None:
        logger.debug("config_initializer.already_initialized")
        return _config_service

    try:
        logger.info("config_initializer.starting", database_url=database_url)

        # Initialize repository (creates database and schema if needed)
        repo = ConfigurationRepository(database_url=database_url)

        # Check if configuration already exists in database
        existing_configs = repo.get_all()
        is_first_run = len(existing_configs) == 0

        if is_first_run:
            logger.info(
                "config_initializer.first_run_detected",
                action="seeding_database_with_defaults",
            )

            # Seed with defaults
            seeded_count = seed_default_configuration(
                database_url=database_url,
                overwrite=False,
                changed_by="system:initializer",
            )
            logger.info(
                "config_initializer.defaults_seeded",
                count=seeded_count,
            )

            # Overlay environment variables if enabled
            if seed_from_env:
                logger.info(
                    "config_initializer.overlaying_env_vars",
                    action="seeding_env_overrides",
                )
                env_count = seed_from_settings(
                    database_url=database_url,
                    changed_by="system:initializer:env",
                )
                logger.info(
                    "config_initializer.env_overlay_complete",
                    count=env_count,
                )
        else:
            logger.info(
                "config_initializer.existing_config_found",
                entry_count=len(existing_configs),
                action="loading_from_database",
            )

        # Create ConfigurationService with persisted/seeded configs
        _config_service = ConfigurationService(
            repository=repo,
            refresh_interval_seconds=60,
            auto_refresh=auto_refresh,
        )

        logger.info(
            "config_initializer.service_created",
            cache_size=len(_config_service._cache),
            auto_refresh=auto_refresh,
        )

        return _config_service

    except Exception as e:
        logger.error(
            "config_initializer.failed",
            error=str(e),
            database_url=database_url,
        )
        raise


def get_configuration_service() -> Optional[ConfigurationService]:
    """
    Get the initialized ConfigurationService singleton.

    Returns None if initialization has not been called yet.
    """
    return _config_service


def reset_configuration_service() -> None:
    """Reset the configuration service singleton (for testing)."""
    global _config_service
    if _config_service is not None:
        _config_service.stop()
    _config_service = None

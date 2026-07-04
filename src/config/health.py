"""
Configuration Health Endpoint — Reports configuration system status.

Provides diagnostic information about:
- Configuration version
- Last refresh time
- Cache age
- Invalid keys
- Pending updates
- Database connectivity
- Configuration hash
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigHealthCheck:
    """
    Health check for the configuration system.

    Reports on the overall health and status of configuration,
    making it easy to diagnose synchronization issues.
    """

    def __init__(
        self,
        config_service: Optional[Any] = None,
        repository: Optional[Any] = None,
    ):
        self._service = config_service
        self._repo = repository

    def check(self) -> dict[str, Any]:
        """
        Perform a full health check of the configuration system.

        Returns a status dict with all diagnostic information.
        """
        status: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "healthy": True,
            "checks": {},
        }

        # Configuration version (based on highest version in cache)
        status["checks"]["config_version"] = self._check_version()

        # Last refresh
        status["checks"]["last_refresh"] = self._check_last_refresh()

        # Cache age
        status["checks"]["cache_age"] = self._check_cache_age()

        # Cache size
        status["checks"]["cache_size"] = self._check_cache_size()

        # Database connectivity
        status["checks"]["database"] = self._check_database()

        # Configuration hash
        status["checks"]["config_hash"] = self._check_config_hash()

        # Determine overall health
        for check_name, check_result in status["checks"].items():
            if check_result.get("status") == "error":
                status["healthy"] = False
                break

        return status

    def _check_version(self) -> dict[str, Any]:
        """Check configuration version."""
        if self._service is None:
            return {"status": "unknown", "message": "Service not initialized"}

        try:
            # Get max version from all cached entries
            max_version = 0
            with self._service._lock:
                for cat_entries in self._service._raw_cache.values():
                    for entry in cat_entries.values():
                        if hasattr(entry, "version") and entry.version > max_version:
                            max_version = entry.version

            return {
                "status": "ok",
                "version": max_version,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _check_last_refresh(self) -> dict[str, Any]:
        """Check when cache was last refreshed."""
        if self._service is None:
            return {"status": "unknown", "message": "Service not initialized"}

        last_refresh = self._service.last_refresh
        if last_refresh is None:
            return {"status": "warning", "message": "Never refreshed"}

        return {
            "status": "ok",
            "last_refresh": last_refresh.isoformat(),
        }

    def _check_cache_age(self) -> dict[str, Any]:
        """Check how stale the cache is."""
        if self._service is None:
            return {"status": "unknown", "message": "Service not initialized"}

        last_refresh = self._service.last_refresh
        if last_refresh is None:
            return {"status": "warning", "age_seconds": None}

        age = (datetime.now(timezone.utc) - last_refresh).total_seconds()

        # Consider cache stale if older than 5 minutes
        if age > 300:
            return {
                "status": "warning",
                "age_seconds": age,
                "message": "Cache may be stale",
            }
        return {
            "status": "ok",
            "age_seconds": age,
        }

    def _check_cache_size(self) -> dict[str, Any]:
        """Check cache size."""
        if self._service is None:
            return {"status": "unknown", "message": "Service not initialized"}

        size = self._service.cache_size
        if size == 0:
            return {
                "status": "warning",
                "entries": 0,
                "message": "Cache is empty",
            }
        return {
            "status": "ok",
            "entries": size,
        }

    def _check_database(self) -> dict[str, Any]:
        """Check database connectivity."""
        if self._repo is None:
            return {"status": "unknown", "message": "Repository not available"}

        try:
            # Attempt a simple query to verify connectivity
            entries = self._repo.get_all()
            return {
                "status": "ok",
                "entries_in_db": len(entries),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Database unreachable: {e}",
            }

    def _check_config_hash(self) -> dict[str, Any]:
        """Compute a hash of the current configuration for comparison."""
        if self._service is None:
            return {"status": "unknown", "message": "Service not initialized"}

        try:
            with self._service._lock:
                # Create a deterministic hash of all cached values
                cache_data = {}
                for cat, entries in sorted(self._service._cache.items()):
                    cache_data[cat] = {k: str(v) for k, v in sorted(entries.items())}

            content = json.dumps(cache_data, sort_keys=True)
            config_hash = hashlib.sha256(content.encode()).hexdigest()

            return {
                "status": "ok",
                "hash": config_hash[:16],  # Shortened for readability
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

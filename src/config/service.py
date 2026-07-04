"""
Configuration Service — Centralized runtime configuration with caching.

This is the single source of truth for all runtime settings after startup.
No application component should read directly from environment variables
for business configuration after initialization.
"""

import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from src.config.models import SystemConfiguration
from src.config.repository import ConfigurationRepository
from src.config.events import ConfigEventBus, ConfigurationChangedEvent
from src.config.lock import ConfigurationLock
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level singleton
_config_service: Optional["ConfigurationService"] = None
_config_lock = threading.Lock()


class ConfigurationService:
    """
    Centralized configuration service backed by the database.

    Features:
    - In-memory cache for fast reads
    - Typed accessors (get_int, get_float, get_bool, get_str, get_json)
    - Cache invalidation on update
    - Thread-safe operations
    - Audit logging for all changes
    - Periodic background refresh
    - Validation before persisting values
    """

    def __init__(
        self,
        repository: ConfigurationRepository,
        refresh_interval_seconds: int = 60,
        auto_refresh: bool = True,
        event_bus: Optional[ConfigEventBus] = None,
        config_lock: Optional[ConfigurationLock] = None,
    ):
        self._repo = repository
        self._cache: dict[str, dict[str, Any]] = {}  # {category: {key: parsed_value}}
        self._raw_cache: dict[str, dict[str, SystemConfiguration]] = {}  # full objects
        self._lock = threading.RLock()
        self._refresh_interval = refresh_interval_seconds
        self._last_refresh: Optional[datetime] = None
        self._refresh_thread: Optional[threading.Thread] = None
        self._running = False
        self._callbacks: list = []
        self._event_bus = event_bus or ConfigEventBus()
        self._config_lock = config_lock or ConfigurationLock()

        # Initial load
        self.refresh()

        # Start background refresh if enabled
        if auto_refresh and refresh_interval_seconds > 0:
            self._start_refresh_loop()

    def refresh(self) -> int:
        """
        Reload all configuration from database into cache.

        Returns count of entries loaded.
        """
        with self._lock:
            entries = self._repo.get_all()
            new_cache: dict[str, dict[str, Any]] = {}
            new_raw: dict[str, dict[str, SystemConfiguration]] = {}

            for entry in entries:
                cat = entry.category
                if cat not in new_cache:
                    new_cache[cat] = {}
                    new_raw[cat] = {}

                new_cache[cat][entry.key] = self._parse_value(
                    entry.value, entry.value_type
                )
                new_raw[cat][entry.key] = entry

            self._cache = new_cache
            self._raw_cache = new_raw
            self._last_refresh = datetime.now(timezone.utc)

            logger.debug(
                "config_service.refreshed",
                entries=len(entries),
                categories=len(new_cache),
            )
            return len(entries)

    # ─── Typed Accessors ──────────────────────────────────────────────────

    def get(self, category: str, key: str, default: Any = None) -> Any:
        """Get a configuration value with optional default."""
        with self._lock:
            cat_cache = self._cache.get(category, {})
            return cat_cache.get(key, default)

    def get_str(self, category: str, key: str, default: str = "") -> str:
        """Get a string configuration value."""
        value = self.get(category, key, default)
        return str(value) if value is not None else default

    def get_int(self, category: str, key: str, default: int = 0) -> int:
        """Get an integer configuration value."""
        value = self.get(category, key)
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def get_float(self, category: str, key: str, default: float = 0.0) -> float:
        """Get a float configuration value."""
        value = self.get(category, key)
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def get_bool(self, category: str, key: str, default: bool = False) -> bool:
        """Get a boolean configuration value."""
        value = self.get(category, key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    def get_json(self, category: str, key: str, default: Any = None) -> Any:
        """Get a JSON configuration value (parsed)."""
        value = self.get(category, key)
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(str(value))
        except (json.JSONDecodeError, TypeError):
            return default

    def get_category(self, category: str) -> dict[str, Any]:
        """Get all values for a category as a dict."""
        with self._lock:
            return dict(self._cache.get(category, {}))

    def get_all_categories(self) -> list[str]:
        """Get list of all configuration categories."""
        with self._lock:
            return list(self._cache.keys())

    # ─── Write Operations ─────────────────────────────────────────────────

    def set(
        self,
        category: str,
        key: str,
        value: Any,
        changed_by: str = "system",
        change_reason: str = "",
        value_type: Optional[str] = None,
        description: str = "",
        validation_rule: str = "",
    ) -> bool:
        """
        Update a configuration value.

        Validates, persists to DB, updates cache, emits events, and notifies listeners.
        Returns True on success.
        """
        # Check configuration lock
        if not self._config_lock.can_modify(changed_by):
            logger.warning(
                "config_service.locked",
                category=category,
                key=key,
                user=changed_by,
            )
            return False

        # Auto-detect value type if not provided
        if value_type is None:
            value_type = self._detect_type(value)

        # Convert value to string for storage
        str_value = self._serialize_value(value, value_type)

        # Validate if rule exists
        existing = self._get_raw(category, key)
        if existing and existing.validation_rule:
            if not self._validate(str_value, existing.validation_rule, value_type):
                logger.warning(
                    "config_service.validation_failed",
                    category=category,
                    key=key,
                    value=str_value,
                    rule=existing.validation_rule,
                )
                return False

        # Check editability
        if existing and not existing.is_editable:
            logger.warning(
                "config_service.not_editable",
                category=category,
                key=key,
            )
            return False

        # Capture old value for event
        old_value = existing.value if existing else None

        # Persist to database
        self._repo.set(
            category=category,
            key=key,
            value=str_value,
            value_type=value_type,
            changed_by=changed_by,
            change_reason=change_reason,
            description=description,
            validation_rule=validation_rule,
        )

        # Update cache
        with self._lock:
            if category not in self._cache:
                self._cache[category] = {}
            self._cache[category][key] = self._parse_value(str_value, value_type)

        # Emit structured change event (for distributed invalidation)
        new_version = (existing.version + 1) if existing else 1
        self._event_bus.emit(
            category=category,
            key=key,
            old_value=old_value,
            new_value=str_value,
            updated_by=changed_by,
            version=new_version,
            change_reason=change_reason,
        )

        # Notify legacy listeners
        self._notify_change(category, key, value)

        logger.info(
            "config_service.updated",
            category=category,
            key=key,
            changed_by=changed_by,
        )
        return True

    def bulk_set(
        self,
        entries: list[dict],
        changed_by: str = "system",
        change_reason: str = "",
    ) -> int:
        """
        Bulk update configuration entries and refresh cache.

        Each entry: {category, key, value, value_type?, description?, ...}
        Returns count of entries processed.
        """
        # Serialize values
        for entry in entries:
            vtype = entry.get("value_type") or self._detect_type(entry["value"])
            entry["value_type"] = vtype
            entry["value"] = self._serialize_value(entry["value"], vtype)

        count = self._repo.bulk_set(entries, changed_by, change_reason)
        self.refresh()
        return count

    # ─── Metadata ─────────────────────────────────────────────────────────

    def get_metadata(self, category: str, key: str) -> Optional[dict]:
        """Get full metadata for a configuration entry."""
        raw = self._get_raw(category, key)
        if not raw:
            return None
        return {
            "id": raw.id,
            "category": raw.category,
            "key": raw.key,
            "value": raw.value,
            "value_type": raw.value_type,
            "description": raw.description,
            "is_secret": raw.is_secret,
            "is_editable": raw.is_editable,
            "validation_rule": raw.validation_rule,
            "version": raw.version,
            "updated_by": raw.updated_by,
            "created_at": raw.created_at.isoformat() if raw.created_at else None,
            "updated_at": raw.updated_at.isoformat() if raw.updated_at else None,
        }

    def get_audit_log(
        self,
        category: Optional[str] = None,
        key: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get audit log entries as dicts."""
        entries = self._repo.get_audit_log(category, key, limit)
        return [
            {
                "category": e.category,
                "key": e.key,
                "old_value": e.old_value,
                "new_value": e.new_value,
                "old_version": e.old_version,
                "new_version": e.new_version,
                "changed_by": e.changed_by,
                "change_reason": e.change_reason,
                "changed_at": e.changed_at.isoformat() if e.changed_at else None,
            }
            for e in entries
        ]

    # ─── Listeners ────────────────────────────────────────────────────────

    def on_change(self, callback):
        """
        Register a callback for configuration changes.

        Callback signature: callback(category: str, key: str, new_value: Any)
        """
        self._callbacks.append(callback)

    # ─── Lifecycle ────────────────────────────────────────────────────────

    def stop(self):
        """Stop the background refresh loop."""
        self._running = False
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=5)

    @property
    def event_bus(self) -> ConfigEventBus:
        """Access the configuration event bus."""
        return self._event_bus

    @property
    def config_lock(self) -> ConfigurationLock:
        """Access the configuration lock."""
        return self._config_lock

    @property
    def last_refresh(self) -> Optional[datetime]:
        return self._last_refresh

    @property
    def cache_size(self) -> int:
        """Total number of cached configuration entries."""
        with self._lock:
            return sum(len(v) for v in self._cache.values())

    # ─── Private ──────────────────────────────────────────────────────────

    def _get_raw(self, category: str, key: str) -> Optional[SystemConfiguration]:
        """Get raw SystemConfiguration object from cache."""
        with self._lock:
            return self._raw_cache.get(category, {}).get(key)

    def _start_refresh_loop(self):
        """Start background periodic refresh."""
        self._running = True
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop,
            daemon=True,
            name="config-refresh",
        )
        self._refresh_thread.start()

    def _refresh_loop(self):
        """Background thread that periodically refreshes the cache."""
        while self._running:
            time.sleep(self._refresh_interval)
            if not self._running:
                break
            try:
                self.refresh()
            except Exception as e:
                logger.error("config_service.refresh_error", error=str(e))

    def _notify_change(self, category: str, key: str, value: Any):
        """Notify all registered callbacks of a change."""
        for cb in self._callbacks:
            try:
                cb(category, key, value)
            except Exception as e:
                logger.error(
                    "config_service.callback_error",
                    callback=str(cb),
                    error=str(e),
                )

    @staticmethod
    def _parse_value(value: str, value_type: str) -> Any:
        """Parse a stored string value into its typed representation."""
        if value_type == "int":
            return int(value)
        elif value_type == "float":
            return float(value)
        elif value_type == "bool":
            return value.lower() in ("true", "1", "yes", "on")
        elif value_type == "json":
            return json.loads(value)
        return value  # str

    @staticmethod
    def _serialize_value(value: Any, value_type: str) -> str:
        """Serialize a typed value to string for storage."""
        if value_type == "json":
            return json.dumps(value) if not isinstance(value, str) else value
        if value_type == "bool":
            if isinstance(value, bool):
                return "true" if value else "false"
            return str(value).lower()
        return str(value)

    @staticmethod
    def _detect_type(value: Any) -> str:
        """Auto-detect the value type."""
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, (dict, list)):
            return "json"
        return "str"

    @staticmethod
    def _validate(value: str, rule: str, value_type: str) -> bool:
        """
        Validate a value against a rule string.

        Supported rules:
        - "range:min:max" — numeric range
        - "options:a,b,c" — allowed values
        - "min_length:N" — minimum string length
        - "max_length:N" — maximum string length
        """
        if not rule:
            return True

        try:
            if rule.startswith("range:"):
                parts = rule.split(":")
                min_val = float(parts[1])
                max_val = float(parts[2])
                num_val = float(value)
                return min_val <= num_val <= max_val

            elif rule.startswith("options:"):
                options = rule[8:].split(",")
                return value in options

            elif rule.startswith("min_length:"):
                min_len = int(rule.split(":")[1])
                return len(value) >= min_len

            elif rule.startswith("max_length:"):
                max_len = int(rule.split(":")[1])
                return len(value) <= max_len

        except (ValueError, IndexError):
            logger.warning("config_service.invalid_rule", rule=rule)

        return True


def get_config_service() -> Optional["ConfigurationService"]:
    """Get the global ConfigurationService singleton (None if not initialized)."""
    return _config_service


def init_config_service(
    database_url: str = "sqlite:///data_cache/trading.db",
    refresh_interval_seconds: int = 60,
    auto_refresh: bool = True,
) -> "ConfigurationService":
    """
    Initialize the global ConfigurationService singleton.

    Should be called once during application startup after database connection
    is established.
    """
    global _config_service
    with _config_lock:
        if _config_service is not None:
            return _config_service

        repo = ConfigurationRepository(database_url)
        _config_service = ConfigurationService(
            repository=repo,
            refresh_interval_seconds=refresh_interval_seconds,
            auto_refresh=auto_refresh,
        )
        logger.info(
            "config_service.initialized",
            entries=_config_service.cache_size,
            refresh_interval=refresh_interval_seconds,
        )
        return _config_service


def reset_config_service():
    """Reset the singleton (for testing)."""
    global _config_service
    with _config_lock:
        if _config_service:
            _config_service.stop()
        _config_service = None

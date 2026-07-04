"""
Configuration Bridge — Connects ConfigurationService to legacy Settings.

Provides a compatibility layer so existing code that reads from `settings`
can transparently get values from the database-backed configuration service
when available, with fallback to the Pydantic Settings singleton.
"""

from typing import Any, Optional

from src.config.service import ConfigurationService, get_config_service
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigBridge:
    """
    Bridge between the new ConfigurationService and legacy settings access.

    Usage:
        bridge = ConfigBridge(config_service)
        max_leverage = bridge.get_float("risk", "max_leverage", fallback=1.0)

    Or use the module-level accessor:
        from src.config.bridge import runtime_config
        max_leverage = runtime_config("risk", "max_leverage", default=1.0)
    """

    def __init__(self, service: ConfigurationService):
        self._service = service

    def get(self, category: str, key: str, default: Any = None) -> Any:
        return self._service.get(category, key, default)

    def get_str(self, category: str, key: str, default: str = "") -> str:
        return self._service.get_str(category, key, default)

    def get_int(self, category: str, key: str, default: int = 0) -> int:
        return self._service.get_int(category, key, default)

    def get_float(self, category: str, key: str, default: float = 0.0) -> float:
        return self._service.get_float(category, key, default)

    def get_bool(self, category: str, key: str, default: bool = False) -> bool:
        return self._service.get_bool(category, key, default)


def runtime_config(category: str, key: str, default: Any = None) -> Any:
    """
    Module-level accessor for runtime configuration.

    Falls back to `default` if the ConfigurationService is not yet initialized
    or the key doesn't exist.
    """
    svc = get_config_service()
    if svc is None:
        return default
    return svc.get(category, key, default)

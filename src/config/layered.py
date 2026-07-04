"""
Multi-Level Configuration — Support configuration precedence.

Provides a layered configuration system where settings can be overridden
at different levels: System Default → Environment → Strategy → Exchange → User Override.
"""

from enum import IntEnum
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigLevel(IntEnum):
    """
    Configuration precedence levels (lower number = lower priority).

    Higher-level values override lower-level values.
    """

    SYSTEM_DEFAULT = 0
    ENVIRONMENT = 1
    STRATEGY = 2
    EXCHANGE = 3
    USER_OVERRIDE = 4


class LayeredConfig:
    """
    Multi-level configuration with precedence-based resolution.

    Supports per-strategy and per-exchange configuration overrides
    instead of one global value.

    Example:
        layered = LayeredConfig()
        layered.set("max_position", 10, level=ConfigLevel.SYSTEM_DEFAULT)
        layered.set("max_position", 5, level=ConfigLevel.STRATEGY, context="BTC")
        layered.set("max_position", 2, level=ConfigLevel.STRATEGY, context="ETH")

        layered.get("max_position")  # 10 (system default)
        layered.get("max_position", strategy="BTC")  # 5
        layered.get("max_position", strategy="ETH")  # 2
    """

    def __init__(self):
        # {key: {level: {context: value}}}
        self._layers: dict[str, dict[ConfigLevel, dict[str, Any]]] = {}

    def set(
        self,
        key: str,
        value: Any,
        level: ConfigLevel = ConfigLevel.SYSTEM_DEFAULT,
        context: str = "__global__",
    ) -> None:
        """
        Set a configuration value at a specific level and context.

        Args:
            key: Configuration key name.
            value: The value to set.
            level: The precedence level.
            context: Optional context (strategy name, exchange name, user ID).
        """
        if key not in self._layers:
            self._layers[key] = {}
        if level not in self._layers[key]:
            self._layers[key][level] = {}
        self._layers[key][level][context] = value

    def get(
        self,
        key: str,
        default: Any = None,
        strategy: Optional[str] = None,
        exchange: Optional[str] = None,
        user: Optional[str] = None,
    ) -> Any:
        """
        Get a configuration value with precedence resolution.

        Resolution order (highest to lowest priority):
        1. User Override (matching user)
        2. Exchange (matching exchange)
        3. Strategy (matching strategy)
        4. Environment
        5. System Default

        Args:
            key: Configuration key name.
            default: Default value if nothing found at any level.
            strategy: Optional strategy name for context-specific lookup.
            exchange: Optional exchange name for context-specific lookup.
            user: Optional user ID for user-specific overrides.
        """
        if key not in self._layers:
            return default

        layers = self._layers[key]

        # Check from highest to lowest priority
        checks = [
            (ConfigLevel.USER_OVERRIDE, user),
            (ConfigLevel.EXCHANGE, exchange),
            (ConfigLevel.STRATEGY, strategy),
            (ConfigLevel.ENVIRONMENT, "__global__"),
            (ConfigLevel.SYSTEM_DEFAULT, "__global__"),
        ]

        for level, context in checks:
            if level not in layers:
                continue
            level_data = layers[level]

            # Check specific context first, then global
            if context and context in level_data:
                return level_data[context]
            if "__global__" in level_data and context != "__global__":
                # Only use global for ENVIRONMENT and SYSTEM_DEFAULT
                if level in (ConfigLevel.ENVIRONMENT, ConfigLevel.SYSTEM_DEFAULT):
                    return level_data["__global__"]

        return default

    def get_effective_value(
        self,
        key: str,
        strategy: Optional[str] = None,
        exchange: Optional[str] = None,
        user: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Get the effective value with metadata about which level provided it.

        Returns dict with 'value', 'level', and 'context' or None if not found.
        """
        if key not in self._layers:
            return None

        layers = self._layers[key]

        checks = [
            (ConfigLevel.USER_OVERRIDE, user),
            (ConfigLevel.EXCHANGE, exchange),
            (ConfigLevel.STRATEGY, strategy),
            (ConfigLevel.ENVIRONMENT, "__global__"),
            (ConfigLevel.SYSTEM_DEFAULT, "__global__"),
        ]

        for level, context in checks:
            if level not in layers:
                continue
            level_data = layers[level]
            if context and context in level_data:
                return {
                    "value": level_data[context],
                    "level": level.name,
                    "context": context,
                }
            if "__global__" in level_data and level in (
                ConfigLevel.ENVIRONMENT,
                ConfigLevel.SYSTEM_DEFAULT,
            ):
                return {
                    "value": level_data["__global__"],
                    "level": level.name,
                    "context": "__global__",
                }

        return None

    def get_all_overrides(self, key: str) -> list[dict[str, Any]]:
        """
        Get all configured values for a key across all levels and contexts.

        Returns list of dicts with 'value', 'level', and 'context'.
        """
        if key not in self._layers:
            return []

        overrides = []
        for level in sorted(self._layers[key].keys()):
            for context, value in self._layers[key][level].items():
                overrides.append({
                    "value": value,
                    "level": level.name,
                    "context": context,
                })
        return overrides

    def remove(
        self,
        key: str,
        level: ConfigLevel,
        context: str = "__global__",
    ) -> bool:
        """Remove a specific value at a given level and context."""
        if key not in self._layers:
            return False
        if level not in self._layers[key]:
            return False
        if context not in self._layers[key][level]:
            return False
        del self._layers[key][level][context]
        return True

    def list_keys(self) -> list[str]:
        """List all configured keys."""
        return list(self._layers.keys())

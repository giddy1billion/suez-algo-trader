"""
Runtime Configuration Management — Database-backed configuration service.

Provides centralized, auditable, hot-reloadable configuration management.
Environment variables are used only for secrets and bootstrap; all runtime
business configuration lives in the database.

Features:
- Database-backed configuration with caching
- Distributed cache invalidation via event publishing
- Configuration snapshots (export/import)
- Startup validation (fail-fast)
- Configuration dependency validation
- Multi-level configuration with precedence
- Secrets management (env-var references only)
- Structured configuration change events
- Emergency configuration lock (maintenance mode)
- Configuration health endpoint
- Strongly typed configuration objects
"""

from src.config.models import SystemConfiguration, ConfigurationAuditLog
from src.config.repository import ConfigurationRepository
from src.config.service import ConfigurationService, get_config_service
from src.config.events import (
    ConfigurationChangedEvent,
    ConfigEventBus,
    EventPublisher,
    InProcessEventPublisher,
    RedisEventPublisher,
)
from src.config.snapshots import ConfigSnapshot, SnapshotManager
from src.config.validation import StartupValidator, ValidationReport, ValidationResult
from src.config.dependencies import DependencyValidator, DependencyRule
from src.config.layered import LayeredConfig, ConfigLevel
from src.config.secrets import SecretsManager, SecretReference, PROTECTED_SECRETS
from src.config.lock import ConfigurationLock, LockState
from src.config.health import ConfigHealthCheck
from src.config.typed_config import (
    RiskConfig,
    TradingConfig,
    MLConfig,
    TelegramConfig,
    ExchangeConfig,
    RiskEngineConfig,
    BacktestConfig,
    build_risk_config,
    build_trading_config,
    build_ml_config,
    build_telegram_config,
    build_exchange_config,
)

__all__ = [
    # Core
    "SystemConfiguration",
    "ConfigurationAuditLog",
    "ConfigurationRepository",
    "ConfigurationService",
    "get_config_service",
    # Events & Cache Invalidation
    "ConfigurationChangedEvent",
    "ConfigEventBus",
    "EventPublisher",
    "InProcessEventPublisher",
    "RedisEventPublisher",
    # Snapshots
    "ConfigSnapshot",
    "SnapshotManager",
    # Validation
    "StartupValidator",
    "ValidationReport",
    "ValidationResult",
    # Dependencies
    "DependencyValidator",
    "DependencyRule",
    # Multi-level
    "LayeredConfig",
    "ConfigLevel",
    # Secrets
    "SecretsManager",
    "SecretReference",
    "PROTECTED_SECRETS",
    # Lock
    "ConfigurationLock",
    "LockState",
    # Health
    "ConfigHealthCheck",
    # Typed Config
    "RiskConfig",
    "TradingConfig",
    "MLConfig",
    "TelegramConfig",
    "ExchangeConfig",
    "RiskEngineConfig",
    "BacktestConfig",
    "build_risk_config",
    "build_trading_config",
    "build_ml_config",
    "build_telegram_config",
    "build_exchange_config",
]

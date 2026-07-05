"""
Tests for the configuration improvements:
- Events & distributed cache invalidation
- Snapshots
- Startup validation
- Dependency validation
- Multi-level configuration
- Secrets management
- Emergency lock
- Health check
- Strongly typed configuration objects
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from src.config.events import (
    ConfigurationChangedEvent,
    ConfigEventBus,
    InProcessEventPublisher,
)
from src.config.snapshots import ConfigSnapshot, SnapshotManager
from src.config.validation import StartupValidator, ValidationReport
from src.config.dependencies import DependencyValidator, DependencyRule
from src.config.layered import LayeredConfig, ConfigLevel
from src.config.secrets import SecretsManager, SecretReference, PROTECTED_SECRETS
from src.config.lock import ConfigurationLock
from src.config.health import ConfigHealthCheck
from src.config.typed_config import (
    RiskConfig,
    TradingConfig,
    MLConfig,
    TelegramConfig,
    ExchangeConfig,
    build_risk_config,
    build_trading_config,
    build_ml_config,
    build_telegram_config,
    build_exchange_config,
)


# ─── Events Tests ─────────────────────────────────────────────────────────────


class TestConfigurationChangedEvent:
    def test_create_event(self):
        event = ConfigurationChangedEvent(
            category="risk",
            key="max_leverage",
            old_value="1.0",
            new_value="2.0",
            updated_by="admin",
            version=2,
        )
        assert event.category == "risk"
        assert event.key == "max_leverage"
        assert event.old_value == "1.0"
        assert event.new_value == "2.0"
        assert event.updated_by == "admin"
        assert event.version == 2

    def test_serialize_deserialize(self):
        event = ConfigurationChangedEvent(
            category="trading",
            key="active_strategy",
            old_value="momentum",
            new_value="ml",
            updated_by="system",
        )
        json_str = event.to_json()
        restored = ConfigurationChangedEvent.from_dict(json.loads(json_str))
        assert restored.category == event.category
        assert restored.key == event.key
        assert restored.new_value == event.new_value


class TestConfigEventBus:
    def test_emit_notifies_callbacks(self):
        bus = ConfigEventBus()
        received = []
        bus.on_change(lambda e: received.append(e))

        bus.emit(
            category="risk",
            key="max_leverage",
            old_value="1.0",
            new_value="2.0",
            updated_by="admin",
        )

        assert len(received) == 1
        assert received[0].key == "max_leverage"

    def test_emit_publishes_to_publishers(self):
        bus = ConfigEventBus()
        publisher = InProcessEventPublisher()
        bus.add_publisher(publisher)

        received = []
        publisher.subscribe(lambda e: received.append(e))

        bus.emit(
            category="trading",
            key="mode",
            old_value="paper",
            new_value="live",
            updated_by="admin",
        )

        assert len(received) == 1
        assert received[0].new_value == "live"

    def test_event_history(self):
        bus = ConfigEventBus()
        bus.emit("a", "b", None, "c", "system")
        bus.emit("d", "e", "c", "f", "system")

        history = bus.get_recent_events()
        assert len(history) == 2


class TestInProcessEventPublisher:
    def test_publish_subscribe(self):
        publisher = InProcessEventPublisher()
        received = []
        publisher.subscribe(lambda e: received.append(e))

        event = ConfigurationChangedEvent(
            category="test",
            key="key",
            old_value=None,
            new_value="val",
            updated_by="test",
        )
        result = publisher.publish(event)

        assert result is True
        assert len(received) == 1


# ─── Snapshots Tests ──────────────────────────────────────────────────────────


class TestConfigSnapshot:
    def test_create_snapshot(self):
        snapshot = ConfigSnapshot(
            version="20260704_120000",
            description="Test snapshot",
            entries=[
                {"category": "risk", "key": "max_leverage", "value": "2.0"},
            ],
        )
        assert snapshot.version == "20260704_120000"
        assert len(snapshot.entries) == 1

    def test_checksum_integrity(self):
        snapshot = ConfigSnapshot(
            version="v1",
            entries=[{"category": "a", "key": "b", "value": "c"}],
        )
        snapshot.checksum = snapshot.compute_checksum()
        assert snapshot.verify_integrity()

        # Tamper with entries
        snapshot.entries[0]["value"] = "d"
        assert not snapshot.verify_integrity()

    def test_json_roundtrip(self):
        snapshot = ConfigSnapshot(
            version="v1",
            entries=[
                {"category": "risk", "key": "leverage", "value": "1.5"},
            ],
        )
        snapshot.checksum = snapshot.compute_checksum()
        json_str = snapshot.to_json()
        restored = ConfigSnapshot.from_json(json_str)
        assert restored.version == "v1"
        assert len(restored.entries) == 1
        assert restored.verify_integrity()


class TestSnapshotManager:
    def test_export_snapshot(self):
        repo = MagicMock()
        entry = MagicMock()
        entry.category = "risk"
        entry.key = "max_leverage"
        entry.value = "2.0"
        entry.value_type = "float"
        entry.description = "Max leverage"
        entry.is_secret = False
        entry.is_editable = True
        entry.validation_rule = "range:0.1:10.0"
        entry.version = 3
        repo.get_all.return_value = [entry]

        manager = SnapshotManager(repo)
        snapshot = manager.export_snapshot(version="test_v1")

        assert snapshot.version == "test_v1"
        assert len(snapshot.entries) == 1
        assert snapshot.entries[0]["value"] == "2.0"

    def test_restore_snapshot_dry_run(self):
        repo = MagicMock()
        repo.get.return_value = None  # No existing entries

        manager = SnapshotManager(repo)
        snapshot = ConfigSnapshot(
            version="v1",
            entries=[
                {
                    "category": "risk",
                    "key": "max_leverage",
                    "value": "2.0",
                    "value_type": "float",
                    "description": "",
                    "is_secret": False,
                    "is_editable": True,
                    "validation_rule": "",
                },
            ],
        )
        snapshot.checksum = snapshot.compute_checksum()

        result = manager.restore_snapshot(snapshot, dry_run=True)
        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["created"] == 1
        repo.set.assert_not_called()


# ─── Startup Validation Tests ─────────────────────────────────────────────────


class TestStartupValidator:
    def _make_config(self, **overrides):
        defaults = {
            "max_leverage": 1.0,
            "max_position_size_pct": 0.02,
            "max_daily_loss_pct": 0.05,
            "max_portfolio_exposure": 0.80,
            "max_single_stock_pct": 0.15,
            "default_stop_loss_pct": 0.03,
            "default_take_profit_pct": 0.06,
            "trading_interval": 60,
            "trading_mode": "paper",
            "alpaca_paper_api_key": "test_key",
            "alpaca_paper_secret_key": "test_secret",
            "alpaca_live_api_key": "",
            "alpaca_live_secret_key": "",
            "symbols_list": ["AAPL", "MSFT"],
            "active_strategy": "momentum",
            "ml_model_path": "",
            "ml_min_confidence": 0.65,
            "risk_max_daily_loss_pct": 0.03,
            "risk_max_drawdown_pct": 0.15,
        }
        defaults.update(overrides)
        config = MagicMock()
        for k, v in defaults.items():
            setattr(config, k, v)
        return config

    def test_valid_config_passes(self):
        config = self._make_config()
        validator = StartupValidator()
        report = validator.validate_all(config)
        assert report.passed

    def test_zero_leverage_fails(self):
        config = self._make_config(max_leverage=0)
        validator = StartupValidator()
        report = validator.validate_all(config)
        assert not report.passed
        assert any("leverage" in r.check for r in report.errors)

    def test_stop_loss_greater_than_take_profit_warns(self):
        config = self._make_config(
            default_stop_loss_pct=0.10,
            default_take_profit_pct=0.05,
        )
        validator = StartupValidator()
        report = validator.validate_all(config)
        assert any("stop_loss" in r.check for r in report.warnings)


# ─── Dependency Validation Tests ──────────────────────────────────────────────


class TestDependencyValidator:
    def test_ml_strategy_requires_model_path(self):
        config = MagicMock()
        config.active_strategy = "ml"
        config.ml_model_path = ""
        config.ml_min_confidence = 0.65
        config.ml_retrain_interval_hours = 24

        validator = DependencyValidator()
        report = validator.validate(config)
        assert not report.passed

    def test_non_ml_strategy_no_dependency_check(self):
        config = MagicMock()
        config.active_strategy = "momentum"
        config.trading_mode = "paper"
        config.telegram_bot_token = ""
        config.intelligence_enabled = False
        config.multi_strategy_config = ""

        validator = DependencyValidator()
        report = validator.validate(config)
        assert report.passed

    def test_custom_rule(self):
        rule = DependencyRule(
            name="custom",
            description="Test rule",
            condition=lambda cfg: getattr(cfg, "feature_enabled", False),
            required_keys=["feature_api_key"],
        )
        validator = DependencyValidator(rules=[rule])

        config = MagicMock()
        config.feature_enabled = True
        config.feature_api_key = ""

        report = validator.validate(config)
        assert not report.passed


# ─── Multi-Level Configuration Tests ─────────────────────────────────────────


class TestLayeredConfig:
    def test_system_default(self):
        lc = LayeredConfig()
        lc.set("max_position", 10, level=ConfigLevel.SYSTEM_DEFAULT)
        assert lc.get("max_position") == 10

    def test_strategy_override(self):
        lc = LayeredConfig()
        lc.set("max_position", 10, level=ConfigLevel.SYSTEM_DEFAULT)
        lc.set("max_position", 5, level=ConfigLevel.STRATEGY, context="BTC")
        lc.set("max_position", 2, level=ConfigLevel.STRATEGY, context="ETH")

        assert lc.get("max_position") == 10  # Default
        assert lc.get("max_position", strategy="BTC") == 5
        assert lc.get("max_position", strategy="ETH") == 2

    def test_precedence_order(self):
        lc = LayeredConfig()
        lc.set("timeout", 30, level=ConfigLevel.SYSTEM_DEFAULT)
        lc.set("timeout", 15, level=ConfigLevel.STRATEGY, context="fast")
        lc.set("timeout", 5, level=ConfigLevel.USER_OVERRIDE, context="admin")

        assert lc.get("timeout", user="admin") == 5
        assert lc.get("timeout", strategy="fast") == 15
        assert lc.get("timeout") == 30

    def test_effective_value_metadata(self):
        lc = LayeredConfig()
        lc.set("fee", 0.01, level=ConfigLevel.EXCHANGE, context="binance")

        result = lc.get_effective_value("fee", exchange="binance")
        assert result is not None
        assert result["value"] == 0.01
        assert result["level"] == "EXCHANGE"
        assert result["context"] == "binance"

    def test_default_when_not_found(self):
        lc = LayeredConfig()
        assert lc.get("nonexistent", default=42) == 42


# ─── Secrets Management Tests ─────────────────────────────────────────────────


class TestSecretsManager:
    def test_protected_secrets(self):
        manager = SecretsManager()
        assert manager.is_protected("alpaca_live_api_key")
        assert manager.is_protected("telegram_bot_token")
        assert not manager.is_protected("trading_interval")

    def test_resolve_from_env(self):
        manager = SecretsManager()
        with patch.dict(os.environ, {"ALPACA_LIVE_API_KEY": "test_key_123"}):
            value = manager.resolve("alpaca_live_api_key")
            assert value == "test_key_123"

    def test_redact_value(self):
        manager = SecretsManager()
        redacted = manager.redact_value("alpaca_live_api_key", "sk-very-secret-key-12345")
        assert "****" in redacted
        assert "sk-very-secret-key-12345" != redacted

    def test_validate_no_secrets_in_config(self):
        manager = SecretsManager()
        entries = [
            {"key": "alpaca_live_api_key", "value": "sk-12345", "is_secret": False},
            {"key": "trading_interval", "value": "60", "is_secret": False},
        ]
        violations = manager.validate_no_secrets_in_config(entries)
        assert len(violations) >= 1

    def test_get_all_references(self):
        manager = SecretsManager()
        refs = manager.get_all_references()
        assert "alpaca_live_api_key" in refs
        assert refs["alpaca_live_api_key"]["source"] == "env"


# ─── Emergency Lock Tests ─────────────────────────────────────────────────────


class TestConfigurationLock:
    def test_initially_unlocked(self):
        lock = ConfigurationLock(super_admins=["admin"])
        assert not lock.is_locked
        assert lock.can_modify("anyone")

    def test_engage_by_super_admin(self):
        lock = ConfigurationLock(super_admins=["admin"])
        result = lock.engage(locked_by="admin", reason="Market crash")
        assert result is True
        assert lock.is_locked

    def test_engage_by_non_admin_fails(self):
        lock = ConfigurationLock(super_admins=["admin"])
        result = lock.engage(locked_by="trader", reason="reasons")
        assert result is False
        assert not lock.is_locked

    def test_locked_blocks_non_admin(self):
        lock = ConfigurationLock(super_admins=["admin"])
        lock.engage(locked_by="admin", reason="test")
        assert not lock.can_modify("trader")
        assert lock.can_modify("admin")

    def test_release(self):
        lock = ConfigurationLock(super_admins=["admin"])
        lock.engage(locked_by="admin", reason="test")
        lock.release(released_by="admin")
        assert not lock.is_locked
        assert lock.can_modify("trader")


# ─── Health Check Tests ───────────────────────────────────────────────────────


class TestConfigHealthCheck:
    def test_health_with_no_service(self):
        health = ConfigHealthCheck()
        result = health.check()
        assert "timestamp" in result
        assert "checks" in result

    def test_health_with_mock_service(self):
        service = MagicMock()
        service._lock = MagicMock()
        service._lock.__enter__ = MagicMock(return_value=None)
        service._lock.__exit__ = MagicMock(return_value=False)
        service._raw_cache = {}
        service._cache = {"risk": {"leverage": "1.0"}}
        service.last_refresh = None
        service.cache_size = 1

        repo = MagicMock()
        repo.get_all.return_value = []

        health = ConfigHealthCheck(config_service=service, repository=repo)
        result = health.check()
        assert result["checks"]["database"]["status"] == "ok"


# ─── Typed Configuration Tests ────────────────────────────────────────────────


class TestTypedConfig:
    def test_risk_config_immutable(self):
        rc = RiskConfig(max_leverage=2.0)
        assert rc.max_leverage == 2.0
        with pytest.raises(Exception):
            rc.max_leverage = 3.0  # type: ignore

    def test_trading_config_symbols_list(self):
        tc = TradingConfig(trading_symbols="AAPL,MSFT,GOOGL")
        assert tc.symbols_list == ["AAPL", "MSFT", "GOOGL"]

    def test_telegram_config_is_configured(self):
        tc = TelegramConfig(bot_token="token", chat_id="123")
        assert tc.is_configured

        tc2 = TelegramConfig()
        assert not tc2.is_configured

    def test_exchange_config_live_check(self):
        ec = ExchangeConfig(live_api_key="key", live_secret_key="secret")
        assert ec.is_live_configured

        ec2 = ExchangeConfig()
        assert not ec2.is_live_configured

    def test_build_risk_config_from_settings(self):
        settings = MagicMock()
        settings.max_position_size_pct = 0.03
        settings.max_daily_loss_pct = 0.04
        settings.max_portfolio_exposure = 0.75
        settings.max_single_stock_pct = 0.12
        settings.max_leverage = 1.5
        settings.max_open_positions = 15
        settings.max_orders_per_day = 50
        settings.max_correlated_positions = 2
        settings.default_stop_loss_pct = 0.02
        settings.default_take_profit_pct = 0.05

        rc = build_risk_config(settings=settings)
        assert rc.max_leverage == 1.5
        assert rc.max_position_size_pct == 0.03

    def test_build_trading_config_from_service(self):
        service = MagicMock()
        service.get_str.side_effect = lambda cat, key, default="": {
            ("trading", "trading_mode"): "paper",
            ("trading", "active_strategy"): "momentum",
            ("trading", "trading_symbols"): "AAPL,NVDA",
            ("trading", "timeframe"): "1Hour",
        }.get((cat, key), default)
        service.get_int.side_effect = lambda cat, key, default=0: {
            ("trading", "trading_interval"): 30,
            ("trading", "max_consecutive_errors"): 3,
            ("trading", "error_cooldown_seconds"): 120,
            ("trading", "lookback_bars"): 100,
        }.get((cat, key), default)
        service.get_bool.side_effect = lambda cat, key, default=False: {
            ("trading", "enable_auto_trading"): True,
        }.get((cat, key), default)

        tc = build_trading_config(config_service=service)
        assert tc.trading_interval == 30
        assert tc.active_strategy == "momentum"

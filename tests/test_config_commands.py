"""
Tests for Telegram Config Commands module.
"""

import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.notifications.telegram_config_commands import (
    _mask_secret,
    _apply_setting_update,
    _generate_env_content,
    set_config_components,
)


class TestMaskSecret:
    def test_masks_long_string(self):
        result = _mask_secret("abcdefghijklmnop")
        assert result.endswith("mnop")
        assert "•" in result
        assert len(result) == 16

    def test_masks_short_string(self):
        result = _mask_secret("short")
        assert result == "••••••••"

    def test_empty_string(self):
        result = _mask_secret("")
        assert result == "(not set)"

    def test_none_value(self):
        result = _mask_secret(None)
        assert result == "(not set)"


class TestGenerateEnvContent:
    def test_generates_valid_env(self):
        """Test that env content generation works with settings."""
        from config.settings import Settings

        # Create mock settings
        settings = Settings(
            trading_mode="paper",
            alpaca_paper_api_key="test_key_12345678",
            alpaca_paper_secret_key="test_secret_12345678",
            telegram_bot_token="123456:ABC-DEF",
            telegram_chat_id="12345",
        )

        # Inject settings
        set_config_components(
            settings=settings,
            strategy_store=None,
            authorized_users=set(),
            runtime_lock=threading.Lock(),
            runtime_changes={},
        )

        content = _generate_env_content()
        assert "TRADING_MODE=paper" in content
        # Secrets must be redacted — never written in plaintext
        assert "ALPACA_PAPER_API_KEY=REDACTED_USE_ENV_VAR_OR_SECRETS_MANAGER" in content
        assert "TELEGRAM_BOT_TOKEN=REDACTED_USE_ENV_VAR_OR_SECRETS_MANAGER" in content
        assert "TELEGRAM_CHAT_ID=12345" in content
        # Ensure actual secret values are NOT present
        assert "test_key_12345678" not in content
        assert "123456:ABC-DEF" not in content

    def test_env_contains_all_sections(self):
        from config.settings import Settings
        settings = Settings()

        set_config_components(
            settings=settings,
            strategy_store=None,
            authorized_users=set(),
            runtime_lock=threading.Lock(),
            runtime_changes={},
        )

        content = _generate_env_content()
        # Check all major sections exist
        assert "Alpaca Paper" in content
        assert "Alpaca Live" in content
        assert "Risk Management" in content
        assert "Strategy" in content
        assert "ML Model" in content
        assert "Notifications" in content
        assert "Automation" in content
        assert "Execution Simulator" in content


class TestSetConfigComponents:
    def test_sets_module_state(self):
        """Verify set_config_components properly injects dependencies."""
        from config.settings import Settings
        from src.strategy.strategy_store import StrategyStore

        settings = Settings()
        store = StrategyStore(path=Path("nonexistent_test.json"))
        lock = threading.Lock()
        changes = {"strategy_name": None}

        set_config_components(
            settings=settings,
            strategy_store=store,
            authorized_users={123, 456},
            runtime_lock=lock,
            runtime_changes=changes,
        )

        # Import module state
        from src.notifications import telegram_config_commands as mod
        assert mod._settings is settings
        assert mod._strategy_store is store
        assert mod._authorized_users == {123, 456}
        assert mod._runtime_lock is lock
        assert mod._runtime_changes is changes


class TestConfigAuditPersistence:
    def test_apply_setting_update_persists_to_config_service(self):
        from config.settings import Settings, TradingMode

        settings = Settings()
        lock = threading.Lock()
        config_service = MagicMock()
        config_service.set.return_value = True

        set_config_components(
            settings=settings,
            strategy_store=None,
            authorized_users=set(),
            runtime_lock=lock,
            runtime_changes={},
            config_service=config_service,
        )

        _apply_setting_update(
            "trading_mode",
            TradingMode.LIVE,
            category="trading",
            key="trading_mode",
            changed_by="telegram:123",
            change_reason="test",
            value_type="str",
            stored_value="live",
        )

        assert settings.trading_mode == TradingMode.LIVE
        config_service.set.assert_called_once_with(
            category="trading",
            key="trading_mode",
            value="live",
            changed_by="telegram:123",
            change_reason="test",
            value_type="str",
        )

    def test_apply_setting_update_rejects_failed_persistence(self):
        from config.settings import Settings

        settings = Settings()
        lock = threading.Lock()
        config_service = MagicMock()
        config_service.set.return_value = False

        set_config_components(
            settings=settings,
            strategy_store=None,
            authorized_users=set(),
            runtime_lock=lock,
            runtime_changes={},
            config_service=config_service,
        )

        with pytest.raises(RuntimeError, match="persist_failed:trading.trading_mode"):
            _apply_setting_update(
                "trading_mode",
                "live",
                category="trading",
                key="trading_mode",
                changed_by="telegram:123",
                change_reason="test",
                value_type="str",
            )

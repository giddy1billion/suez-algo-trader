"""
Tests for the Runtime Configuration Management system.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.config.models import ConfigBase, SystemConfiguration, ConfigurationAuditLog
from src.config.repository import ConfigurationRepository
from src.config.service import ConfigurationService, reset_config_service, init_config_service
from src.config.seed import seed_default_configuration, DEFAULT_CONFIGURATIONS


@pytest.fixture
def repo(tmp_path):
    """Create a fresh configuration repository with in-memory database."""
    db_url = f"sqlite:///{tmp_path}/test_config.db"
    return ConfigurationRepository(db_url)


@pytest.fixture
def service(repo):
    """Create a configuration service with no auto-refresh."""
    svc = ConfigurationService(repo, auto_refresh=False)
    yield svc
    svc.stop()


@pytest.fixture(autouse=True)
def cleanup_singleton():
    """Reset the global singleton after each test."""
    yield
    reset_config_service()


class TestConfigurationRepository:
    """Tests for ConfigurationRepository CRUD operations."""

    def test_set_and_get(self, repo):
        """Test basic set and get operations."""
        repo.set("trading", "max_leverage", "2.5", value_type="float", changed_by="test")
        result = repo.get("trading", "max_leverage")

        assert result is not None
        assert result.value == "2.5"
        assert result.value_type == "float"
        assert result.version == 1

    def test_update_increments_version(self, repo):
        """Test that updating a value increments the version."""
        repo.set("trading", "max_leverage", "1.0", value_type="float")
        repo.set("trading", "max_leverage", "2.0", value_type="float")
        result = repo.get("trading", "max_leverage")

        assert result.value == "2.0"
        assert result.version == 2

    def test_get_nonexistent(self, repo):
        """Test getting a non-existent key returns None."""
        result = repo.get("nonexistent", "key")
        assert result is None

    def test_get_by_category(self, repo):
        """Test loading all entries for a category."""
        repo.set("risk", "max_leverage", "1.0", value_type="float")
        repo.set("risk", "max_positions", "20", value_type="int")
        repo.set("trading", "interval", "60", value_type="int")

        results = repo.get_by_category("risk")
        assert len(results) == 2

    def test_get_all(self, repo):
        """Test loading all entries."""
        repo.set("risk", "max_leverage", "1.0", value_type="float")
        repo.set("trading", "interval", "60", value_type="int")

        results = repo.get_all()
        assert len(results) == 2

    def test_delete(self, repo):
        """Test deleting a configuration entry."""
        repo.set("trading", "temp_key", "temp_value")
        assert repo.get("trading", "temp_key") is not None

        deleted = repo.delete("trading", "temp_key", changed_by="test")
        assert deleted is True
        assert repo.get("trading", "temp_key") is None

    def test_delete_nonexistent(self, repo):
        """Test deleting a non-existent entry returns False."""
        assert repo.delete("nonexistent", "key") is False

    def test_audit_log_on_create(self, repo):
        """Test that creating a value generates an audit log entry."""
        repo.set("trading", "leverage", "1.0", changed_by="admin")
        logs = repo.get_audit_log("trading", "leverage")

        assert len(logs) == 1
        assert logs[0].old_value is None
        assert logs[0].new_value == "1.0"
        assert logs[0].changed_by == "admin"
        assert logs[0].new_version == 1

    def test_audit_log_on_update(self, repo):
        """Test that updating a value generates an audit log entry."""
        repo.set("trading", "leverage", "1.0", changed_by="admin")
        repo.set("trading", "leverage", "2.0", changed_by="user1", change_reason="increased")
        logs = repo.get_audit_log("trading", "leverage")

        assert len(logs) == 2
        # Most recent first
        assert logs[0].old_value == "1.0"
        assert logs[0].new_value == "2.0"
        assert logs[0].changed_by == "user1"
        assert logs[0].change_reason == "increased"

    def test_audit_log_on_delete(self, repo):
        """Test that deleting generates an audit log entry."""
        repo.set("trading", "temp", "val")
        repo.delete("trading", "temp", changed_by="admin")
        logs = repo.get_audit_log("trading", "temp")

        assert len(logs) == 2
        assert logs[0].new_value == "<DELETED>"

    def test_bulk_set(self, repo):
        """Test bulk upsert of entries."""
        entries = [
            {"category": "risk", "key": "leverage", "value": "1.5", "value_type": "float"},
            {"category": "risk", "key": "positions", "value": "10", "value_type": "int"},
            {"category": "trading", "key": "interval", "value": "30", "value_type": "int"},
        ]
        count = repo.bulk_set(entries, changed_by="bulk_test")
        assert count == 3
        assert repo.get("risk", "leverage").value == "1.5"
        assert repo.get("trading", "interval").value == "30"


class TestConfigurationService:
    """Tests for ConfigurationService with caching and typed access."""

    def test_get_after_set(self, service, repo):
        """Test that get returns value after set."""
        repo.set("trading", "interval", "120", value_type="int")
        service.refresh()

        assert service.get_int("trading", "interval") == 120

    def test_typed_accessors(self, service, repo):
        """Test all typed accessors."""
        repo.set("test", "str_val", "hello", value_type="str")
        repo.set("test", "int_val", "42", value_type="int")
        repo.set("test", "float_val", "3.14", value_type="float")
        repo.set("test", "bool_true", "true", value_type="bool")
        repo.set("test", "bool_false", "false", value_type="bool")
        repo.set("test", "json_val", '{"a": 1}', value_type="json")
        service.refresh()

        assert service.get_str("test", "str_val") == "hello"
        assert service.get_int("test", "int_val") == 42
        assert service.get_float("test", "float_val") == pytest.approx(3.14)
        assert service.get_bool("test", "bool_true") is True
        assert service.get_bool("test", "bool_false") is False
        assert service.get_json("test", "json_val") == {"a": 1}

    def test_defaults(self, service):
        """Test that defaults are returned for missing keys."""
        assert service.get_str("missing", "key", "fallback") == "fallback"
        assert service.get_int("missing", "key", 99) == 99
        assert service.get_float("missing", "key", 1.5) == 1.5
        assert service.get_bool("missing", "key", True) is True

    def test_set_updates_cache(self, service):
        """Test that set() updates the cache immediately."""
        service.set("live", "test_key", 42, changed_by="test")
        assert service.get_int("live", "test_key") == 42

    def test_set_validates(self, service, repo):
        """Test that set() validates against rules."""
        repo.set("risk", "leverage", "1.0", value_type="float", validation_rule="range:0.1:10.0")
        service.refresh()

        # Valid value
        assert service.set("risk", "leverage", 5.0, changed_by="test") is True
        assert service.get_float("risk", "leverage") == 5.0

        # Invalid value (out of range)
        assert service.set("risk", "leverage", 15.0, changed_by="test") is False
        # Value should remain unchanged
        assert service.get_float("risk", "leverage") == 5.0

    def test_non_editable_cannot_be_changed(self, service, repo):
        """Test that non-editable values cannot be changed via service."""
        repo.set("system", "version", "1.0", value_type="str", is_editable=False)
        service.refresh()

        assert service.set("system", "version", "2.0", changed_by="test") is False
        assert service.get_str("system", "version") == "1.0"

    def test_get_category(self, service, repo):
        """Test getting all values for a category."""
        repo.set("risk", "leverage", "2.0", value_type="float")
        repo.set("risk", "positions", "10", value_type="int")
        service.refresh()

        cat = service.get_category("risk")
        assert cat == {"leverage": 2.0, "positions": 10}

    def test_get_all_categories(self, service, repo):
        """Test getting list of all categories."""
        repo.set("risk", "x", "1", value_type="int")
        repo.set("trading", "y", "2", value_type="int")
        service.refresh()

        categories = service.get_all_categories()
        assert "risk" in categories
        assert "trading" in categories

    def test_cache_size(self, service, repo):
        """Test cache size reporting."""
        repo.set("a", "x", "1", value_type="int")
        repo.set("a", "y", "2", value_type="int")
        repo.set("b", "z", "3", value_type="int")
        service.refresh()

        assert service.cache_size == 3

    def test_change_callback(self, service):
        """Test that change callbacks are invoked."""
        changes = []
        service.on_change(lambda cat, key, val: changes.append((cat, key, val)))

        service.set("test", "notified", "hello", changed_by="test")
        assert len(changes) == 1
        assert changes[0] == ("test", "notified", "hello")

    def test_get_audit_log(self, service):
        """Test audit log retrieval via service."""
        service.set("test", "key1", "v1", changed_by="user1")
        service.set("test", "key1", "v2", changed_by="user2", change_reason="updated")

        logs = service.get_audit_log("test", "key1")
        assert len(logs) >= 2
        assert logs[0]["changed_by"] == "user2"
        assert logs[0]["change_reason"] == "updated"

    def test_get_metadata(self, service, repo):
        """Test metadata retrieval."""
        repo.set("meta", "test_key", "val", value_type="str", description="A test key")
        service.refresh()

        meta = service.get_metadata("meta", "test_key")
        assert meta is not None
        assert meta["category"] == "meta"
        assert meta["key"] == "test_key"
        assert meta["description"] == "A test key"
        assert meta["version"] == 1

    def test_bulk_set(self, service):
        """Test bulk update via service."""
        entries = [
            {"category": "bulk", "key": "a", "value": 1},
            {"category": "bulk", "key": "b", "value": 2.5},
            {"category": "bulk", "key": "c", "value": True},
        ]
        count = service.bulk_set(entries, changed_by="bulk_test")
        assert count == 3
        assert service.get_int("bulk", "a") == 1
        assert service.get_float("bulk", "b") == 2.5
        assert service.get_bool("bulk", "c") is True


class TestConfigurationSeed:
    """Tests for the configuration seeding module."""

    def test_seed_default_configuration(self, tmp_path):
        """Test seeding default configuration values."""
        db_url = f"sqlite:///{tmp_path}/seed_test.db"
        count = seed_default_configuration(db_url)

        assert count == len(DEFAULT_CONFIGURATIONS)

        # Verify some values exist
        repo = ConfigurationRepository(db_url)
        leverage = repo.get("risk", "max_leverage")
        assert leverage is not None
        assert leverage.value == "1.0"
        assert leverage.value_type == "float"

    def test_seed_does_not_overwrite(self, tmp_path):
        """Test that seed doesn't overwrite existing values."""
        db_url = f"sqlite:///{tmp_path}/seed_test.db"
        repo = ConfigurationRepository(db_url)

        # Pre-set a value
        repo.set("risk", "max_leverage", "5.0", value_type="float")

        # Seed (should not overwrite)
        seed_default_configuration(db_url, overwrite=False)

        result = repo.get("risk", "max_leverage")
        assert result.value == "5.0"  # Unchanged

    def test_seed_with_overwrite(self, tmp_path):
        """Test that seed can overwrite when requested."""
        db_url = f"sqlite:///{tmp_path}/seed_test.db"
        repo = ConfigurationRepository(db_url)

        # Pre-set a value
        repo.set("risk", "max_leverage", "5.0", value_type="float")

        # Seed with overwrite
        seed_default_configuration(db_url, overwrite=True)

        result = repo.get("risk", "max_leverage")
        assert result.value == "1.0"  # Overwritten to default


class TestConfigurationServiceInit:
    """Tests for singleton initialization."""

    def test_init_config_service(self, tmp_path):
        """Test initializing the global config service."""
        db_url = f"sqlite:///{tmp_path}/init_test.db"
        svc = init_config_service(db_url, auto_refresh=False)

        assert svc is not None
        assert svc.cache_size == 0  # Empty DB

    def test_init_is_singleton(self, tmp_path):
        """Test that init returns the same instance."""
        db_url = f"sqlite:///{tmp_path}/init_test.db"
        svc1 = init_config_service(db_url, auto_refresh=False)
        svc2 = init_config_service(db_url, auto_refresh=False)

        assert svc1 is svc2


class TestValidation:
    """Tests for configuration validation rules."""

    def test_range_validation(self, service, repo):
        """Test range validation rule."""
        repo.set("test", "ranged", "5.0", value_type="float", validation_rule="range:0:10")
        service.refresh()

        assert service.set("test", "ranged", 7.5, changed_by="test") is True
        assert service.set("test", "ranged", 15.0, changed_by="test") is False
        assert service.set("test", "ranged", -1.0, changed_by="test") is False

    def test_options_validation(self, service, repo):
        """Test options validation rule."""
        repo.set("test", "mode", "paper", value_type="str", validation_rule="options:paper,live")
        service.refresh()

        assert service.set("test", "mode", "live", changed_by="test") is True
        assert service.set("test", "mode", "invalid", changed_by="test") is False

    def test_min_length_validation(self, service, repo):
        """Test min_length validation rule."""
        repo.set("test", "name", "hello", value_type="str", validation_rule="min_length:3")
        service.refresh()

        assert service.set("test", "name", "valid", changed_by="test") is True
        assert service.set("test", "name", "ab", changed_by="test") is False


class TestConfigBridge:
    """Tests for the configuration bridge module."""

    def test_runtime_config_without_service(self):
        """Test that runtime_config returns default when service not initialized."""
        from src.config.bridge import runtime_config
        reset_config_service()

        assert runtime_config("any", "key", "default_val") == "default_val"

    def test_runtime_config_with_service(self, tmp_path):
        """Test that runtime_config returns DB value when service is initialized."""
        from src.config.bridge import runtime_config

        db_url = f"sqlite:///{tmp_path}/bridge_test.db"
        svc = init_config_service(db_url, auto_refresh=False)
        svc.set("test", "bridge_key", "db_value", changed_by="test")

        assert runtime_config("test", "bridge_key", "fallback") == "db_value"

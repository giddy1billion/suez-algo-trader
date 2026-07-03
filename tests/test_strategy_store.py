"""
Tests for Strategy Store — CRUD operations for user-defined strategies.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.strategy.strategy_store import (
    StrategyStore,
    StrategyDefinition,
    STRATEGY_TEMPLATES,
    VALID_TIMEFRAMES,
)


@pytest.fixture
def temp_store(tmp_path):
    """Create a StrategyStore with a temp file."""
    return StrategyStore(path=tmp_path / "strategies.json")


class TestStrategyDefinition:
    def test_to_dict_roundtrip(self):
        strat = StrategyDefinition(
            name="test",
            template="momentum",
            symbols=["AAPL", "MSFT"],
            timeframe="1Hour",
            lookback=200,
            params={"fast_ema": 12},
        )
        data = strat.to_dict()
        restored = StrategyDefinition.from_dict(data)
        assert restored.name == "test"
        assert restored.symbols == ["AAPL", "MSFT"]
        assert restored.params == {"fast_ema": 12}

    def test_multi_config_entry(self):
        strat = StrategyDefinition(
            name="my_scalper",
            template="scalping",
            symbols=["AAPL", "TSLA"],
            timeframe="5Min",
            interval=30,
            weight=1.5,
        )
        entry = strat.to_multi_config_entry()
        assert entry == "my_scalper:AAPL,TSLA:5Min:30:1.5"


class TestStrategyStore:
    def test_create_strategy(self, temp_store):
        success, msg = temp_store.create(
            name="my_momentum",
            template="momentum",
            symbols=["AAPL", "MSFT"],
            timeframe="1Hour",
        )
        assert success is True
        assert "created" in msg.lower()
        assert temp_store.count == 1

    def test_create_duplicate_fails(self, temp_store):
        temp_store.create("test", template="momentum", symbols=["AAPL"])
        success, msg = temp_store.create("test", template="momentum", symbols=["MSFT"])
        assert success is False
        assert "already exists" in msg

    def test_create_invalid_template(self, temp_store):
        success, msg = temp_store.create("test", template="nonexistent", symbols=["AAPL"])
        assert success is False
        assert "Unknown template" in msg

    def test_create_invalid_timeframe(self, temp_store):
        success, msg = temp_store.create("test", template="momentum", symbols=["AAPL"], timeframe="99Min")
        assert success is False
        assert "Invalid timeframe" in msg

    def test_create_no_symbols(self, temp_store):
        success, msg = temp_store.create("test", template="momentum", symbols=[])
        assert success is False
        assert "symbol" in msg.lower()

    def test_create_merges_template_params(self, temp_store):
        temp_store.create(
            "test",
            template="momentum",
            symbols=["AAPL"],
            params={"fast_ema": 8},
        )
        strat = temp_store.get("test")
        # Custom param is applied
        assert strat.params["fast_ema"] == 8
        # Template defaults are preserved
        assert strat.params["slow_ema"] == 26
        assert strat.params["rsi_period"] == 14

    def test_list_strategies(self, temp_store):
        temp_store.create("s1", template="momentum", symbols=["AAPL"])
        temp_store.create("s2", template="swing", symbols=["MSFT"])
        strategies = temp_store.list_strategies()
        assert len(strategies) == 2
        names = {s.name for s in strategies}
        assert names == {"s1", "s2"}

    def test_get_nonexistent(self, temp_store):
        assert temp_store.get("missing") is None

    def test_update_symbols(self, temp_store):
        temp_store.create("test", template="momentum", symbols=["AAPL"])
        success, _ = temp_store.update("test", symbols=["MSFT", "GOOGL"])
        assert success is True
        strat = temp_store.get("test")
        assert strat.symbols == ["MSFT", "GOOGL"]

    def test_update_timeframe(self, temp_store):
        temp_store.create("test", template="momentum", symbols=["AAPL"])
        success, _ = temp_store.update("test", timeframe="5Min")
        assert success is True
        assert temp_store.get("test").timeframe == "5Min"

    def test_update_invalid_timeframe(self, temp_store):
        temp_store.create("test", template="momentum", symbols=["AAPL"])
        success, msg = temp_store.update("test", timeframe="99Min")
        assert success is False

    def test_update_params(self, temp_store):
        temp_store.create("test", template="momentum", symbols=["AAPL"])
        success, _ = temp_store.update("test", params={"fast_ema": 5, "slow_ema": 13})
        strat = temp_store.get("test")
        assert strat.params["fast_ema"] == 5
        assert strat.params["slow_ema"] == 13

    def test_update_nonexistent(self, temp_store):
        success, msg = temp_store.update("missing", timeframe="5Min")
        assert success is False
        assert "not found" in msg

    def test_delete(self, temp_store):
        temp_store.create("test", template="momentum", symbols=["AAPL"])
        success, _ = temp_store.delete("test")
        assert success is True
        assert temp_store.count == 0

    def test_delete_nonexistent(self, temp_store):
        success, msg = temp_store.delete("missing")
        assert success is False

    def test_activate_deactivate(self, temp_store):
        temp_store.create("test", template="momentum", symbols=["AAPL"])
        temp_store.deactivate("test")
        assert temp_store.get("test").active is False
        temp_store.activate("test")
        assert temp_store.get("test").active is True

    def test_get_active_strategies(self, temp_store):
        temp_store.create("s1", template="momentum", symbols=["AAPL"])
        temp_store.create("s2", template="swing", symbols=["MSFT"])
        temp_store.deactivate("s2")
        active = temp_store.get_active_strategies()
        assert len(active) == 1
        assert active[0].name == "s1"

    def test_get_multi_config_string(self, temp_store):
        temp_store.create("s1", template="momentum", symbols=["AAPL", "MSFT"], timeframe="1Hour", interval=60)
        temp_store.create("s2", template="scalping", symbols=["TSLA"], timeframe="5Min", interval=30)
        config_str = temp_store.get_multi_config_string()
        assert "s1:AAPL,MSFT:1Hour:60:1.0" in config_str
        assert "s2:TSLA:5Min:30:1.0" in config_str
        assert ";" in config_str

    def test_duplicate(self, temp_store):
        temp_store.create("original", template="momentum", symbols=["AAPL"], params={"fast_ema": 8})
        success, _ = temp_store.duplicate("original", "copy")
        assert success is True
        assert temp_store.count == 2
        copy = temp_store.get("copy")
        assert copy.params["fast_ema"] == 8
        assert copy.symbols == ["AAPL"]
        assert copy.template == "momentum"

    def test_persistence(self, tmp_path):
        path = tmp_path / "strats.json"
        store1 = StrategyStore(path=path)
        store1.create("persistent", template="swing", symbols=["BTC/USD"], timeframe="4Hour")

        # Load from same file
        store2 = StrategyStore(path=path)
        assert store2.count == 1
        strat = store2.get("persistent")
        assert strat.template == "swing"
        assert strat.symbols == ["BTC/USD"]

    def test_get_templates(self, temp_store):
        templates = temp_store.get_templates()
        assert "momentum" in templates
        assert "mean_reversion" in templates
        assert "scalping" in templates
        assert "swing" in templates
        assert "description" in templates["momentum"]
        assert "params" in templates["momentum"]


class TestStrategyTemplates:
    def test_all_templates_have_params(self):
        for name, tmpl in STRATEGY_TEMPLATES.items():
            assert "description" in tmpl
            assert "params" in tmpl
            assert len(tmpl["params"]) > 0

    def test_valid_timeframes(self):
        assert "1Min" in VALID_TIMEFRAMES
        assert "1Hour" in VALID_TIMEFRAMES
        assert "1Day" in VALID_TIMEFRAMES

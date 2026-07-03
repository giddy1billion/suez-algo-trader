"""
Strategy Store — Persistent CRUD for user-defined trading strategies.

Stores strategy definitions as JSON in config/strategies.json.
Supports creating strategies from templates (momentum, mean_reversion, composable)
with custom parameters, symbols, and timeframes.

Usage:
    store = StrategyStore()
    store.create("my_scalper", template="momentum", symbols=["AAPL", "TSLA"],
                 timeframe="5Min", params={"fast_ema": 5, "slow_ema": 13})
    store.list_strategies()
    store.activate("my_scalper")
    store.delete("my_scalper")
"""

import json
import os
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default storage path
STRATEGIES_FILE = Path("config/strategies.json")

# Strategy templates with default parameters
STRATEGY_TEMPLATES = {
    "momentum": {
        "description": "Trend-following with EMA crossover, RSI, and volume confirmation",
        "params": {
            "fast_ema": 12,
            "slow_ema": 26,
            "signal_ema": 9,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "atr_period": 14,
            "atr_sl_multiplier": 2.0,
            "atr_tp_multiplier": 3.0,
            "volume_ma_period": 20,
            "volume_spike_threshold": 1.5,
        },
    },
    "mean_reversion": {
        "description": "Mean reversion with Bollinger Bands, RSI, and z-score",
        "params": {
            "bb_period": 20,
            "bb_std": 2.0,
            "zscore_entry": 2.0,
            "zscore_exit": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 25,
            "rsi_overbought": 75,
            "atr_period": 14,
            "atr_sl_multiplier": 1.5,
            "atr_tp_multiplier": 2.5,
        },
    },
    "scalping": {
        "description": "Fast scalping with tight stops and quick exits",
        "params": {
            "fast_ema": 5,
            "slow_ema": 13,
            "rsi_period": 7,
            "rsi_oversold": 25,
            "rsi_overbought": 75,
            "atr_period": 10,
            "atr_sl_multiplier": 1.0,
            "atr_tp_multiplier": 1.5,
            "volume_spike_threshold": 2.0,
        },
    },
    "swing": {
        "description": "Swing trading with wider stops and longer holds",
        "params": {
            "fast_ema": 20,
            "slow_ema": 50,
            "rsi_period": 14,
            "rsi_oversold": 35,
            "rsi_overbought": 65,
            "atr_period": 20,
            "atr_sl_multiplier": 3.0,
            "atr_tp_multiplier": 5.0,
            "volume_spike_threshold": 1.3,
        },
    },
}

# Valid timeframes for Alpaca
VALID_TIMEFRAMES = ["1Min", "5Min", "15Min", "30Min", "1Hour", "4Hour", "1Day"]


class StrategyDefinition:
    """A persisted strategy definition (not the live instance, but the blueprint)."""

    def __init__(
        self,
        name: str,
        template: str,
        symbols: list[str],
        timeframe: str = "1Hour",
        lookback: int = 200,
        params: dict = None,
        active: bool = True,
        interval: int = 60,
        weight: float = 1.0,
        created_at: str = None,
        updated_at: str = None,
        description: str = "",
    ):
        self.name = name
        self.template = template
        self.symbols = symbols
        self.timeframe = timeframe
        self.lookback = lookback
        self.params = params or {}
        self.active = active
        self.interval = interval
        self.weight = weight
        self.description = description
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.updated_at = updated_at or self.created_at

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "template": self.template,
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "lookback": self.lookback,
            "params": self.params,
            "active": self.active,
            "interval": self.interval,
            "weight": self.weight,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyDefinition":
        return cls(**data)

    def to_multi_config_entry(self) -> str:
        """Convert to multi_strategy_config format: name:symbols:timeframe:interval:weight"""
        symbols_str = ",".join(self.symbols)
        return f"{self.name}:{symbols_str}:{self.timeframe}:{self.interval}:{self.weight}"


class StrategyStore:
    """
    Persistent store for user-defined strategy blueprints.
    Thread-safe CRUD operations with JSON file backing.
    """

    def __init__(self, path: Optional[Path] = None):
        self._path = path or STRATEGIES_FILE
        self._lock = threading.Lock()
        self._strategies: dict[str, StrategyDefinition] = {}
        self._load()

    def _load(self) -> None:
        """Load strategies from JSON file."""
        if not self._path.exists():
            self._strategies = {}
            return

        try:
            with open(self._path, "r") as f:
                data = json.load(f)
            self._strategies = {
                name: StrategyDefinition.from_dict(entry)
                for name, entry in data.get("strategies", {}).items()
            }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"strategy_store.load_error: {e}")
            self._strategies = {}

    def _save(self) -> None:
        """Persist strategies to JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "strategies": {
                name: strat.to_dict()
                for name, strat in self._strategies.items()
            },
        }
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    def create(
        self,
        name: str,
        template: str = "momentum",
        symbols: list[str] = None,
        timeframe: str = "1Hour",
        lookback: int = 200,
        params: dict = None,
        interval: int = 60,
        weight: float = 1.0,
        description: str = "",
    ) -> tuple[bool, str]:
        """
        Create a new strategy definition.

        Returns:
            (success: bool, message: str)
        """
        with self._lock:
            # Validate name
            name = name.lower().replace(" ", "_")
            if not name.isidentifier():
                return False, f"Invalid name: '{name}'. Use letters, digits, underscores."

            if name in self._strategies:
                return False, f"Strategy '{name}' already exists. Use edit or delete first."

            # Validate template
            if template not in STRATEGY_TEMPLATES:
                valid = ", ".join(STRATEGY_TEMPLATES.keys())
                return False, f"Unknown template: '{template}'. Valid: {valid}"

            # Validate timeframe
            if timeframe not in VALID_TIMEFRAMES:
                return False, f"Invalid timeframe: '{timeframe}'. Valid: {', '.join(VALID_TIMEFRAMES)}"

            # Validate symbols
            if not symbols or len(symbols) == 0:
                return False, "At least one symbol is required."

            # Merge template defaults with user params
            merged_params = deepcopy(STRATEGY_TEMPLATES[template]["params"])
            if params:
                for key, val in params.items():
                    if key in merged_params:
                        merged_params[key] = type(merged_params[key])(val)
                    else:
                        merged_params[key] = val

            strat = StrategyDefinition(
                name=name,
                template=template,
                symbols=[s.upper() for s in symbols],
                timeframe=timeframe,
                lookback=lookback,
                params=merged_params,
                active=True,
                interval=interval,
                weight=weight,
                description=description or STRATEGY_TEMPLATES[template]["description"],
            )

            self._strategies[name] = strat
            self._save()
            return True, f"Strategy '{name}' created successfully."

    def list_strategies(self) -> list[StrategyDefinition]:
        """Return all strategy definitions."""
        with self._lock:
            return list(self._strategies.values())

    def get(self, name: str) -> Optional[StrategyDefinition]:
        """Get a strategy by name."""
        with self._lock:
            return self._strategies.get(name.lower())

    def update(self, name: str, **kwargs) -> tuple[bool, str]:
        """
        Update strategy fields.

        Supported kwargs: symbols, timeframe, lookback, params, interval, weight,
                         active, description
        """
        with self._lock:
            name = name.lower()
            if name not in self._strategies:
                return False, f"Strategy '{name}' not found."

            strat = self._strategies[name]

            if "symbols" in kwargs:
                symbols = kwargs["symbols"]
                if isinstance(symbols, str):
                    symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
                strat.symbols = symbols

            if "timeframe" in kwargs:
                tf = kwargs["timeframe"]
                if tf not in VALID_TIMEFRAMES:
                    return False, f"Invalid timeframe: '{tf}'. Valid: {', '.join(VALID_TIMEFRAMES)}"
                strat.timeframe = tf

            if "lookback" in kwargs:
                strat.lookback = int(kwargs["lookback"])

            if "interval" in kwargs:
                strat.interval = int(kwargs["interval"])

            if "weight" in kwargs:
                strat.weight = float(kwargs["weight"])

            if "active" in kwargs:
                strat.active = bool(kwargs["active"])

            if "description" in kwargs:
                strat.description = kwargs["description"]

            if "params" in kwargs:
                for key, val in kwargs["params"].items():
                    strat.params[key] = val

            strat.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()
            return True, f"Strategy '{name}' updated."

    def delete(self, name: str) -> tuple[bool, str]:
        """Delete a strategy."""
        with self._lock:
            name = name.lower()
            if name not in self._strategies:
                return False, f"Strategy '{name}' not found."

            del self._strategies[name]
            self._save()
            return True, f"Strategy '{name}' deleted."

    def activate(self, name: str) -> tuple[bool, str]:
        """Activate a strategy."""
        return self.update(name, active=True)

    def deactivate(self, name: str) -> tuple[bool, str]:
        """Deactivate a strategy."""
        return self.update(name, active=False)

    def get_active_strategies(self) -> list[StrategyDefinition]:
        """Return only active strategies."""
        with self._lock:
            return [s for s in self._strategies.values() if s.active]

    def get_multi_config_string(self) -> str:
        """Generate multi_strategy_config string from active strategies."""
        active = self.get_active_strategies()
        if not active:
            return ""
        return ";".join(s.to_multi_config_entry() for s in active)

    def get_templates(self) -> dict:
        """Return available templates with descriptions."""
        return {
            name: {
                "description": tmpl["description"],
                "params": list(tmpl["params"].keys()),
            }
            for name, tmpl in STRATEGY_TEMPLATES.items()
        }

    def duplicate(self, source_name: str, new_name: str) -> tuple[bool, str]:
        """Duplicate an existing strategy with a new name."""
        with self._lock:
            source = self._strategies.get(source_name.lower())
            if not source:
                return False, f"Strategy '{source_name}' not found."

            new_name = new_name.lower().replace(" ", "_")
            if new_name in self._strategies:
                return False, f"Strategy '{new_name}' already exists."

            new_strat = StrategyDefinition(
                name=new_name,
                template=source.template,
                symbols=list(source.symbols),
                timeframe=source.timeframe,
                lookback=source.lookback,
                params=deepcopy(source.params),
                active=True,
                interval=source.interval,
                weight=source.weight,
                description=f"Copy of {source.name}",
            )
            self._strategies[new_name] = new_strat
            self._save()
            return True, f"Strategy '{new_name}' created (copy of '{source_name}')."

    @property
    def count(self) -> int:
        return len(self._strategies)

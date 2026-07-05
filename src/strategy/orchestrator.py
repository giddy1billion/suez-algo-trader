"""
Strategy Orchestrator — Concurrent multi-strategy execution.

Runs multiple strategies in parallel, each with its own symbol set, timeframe,
and schedule. Aggregates signals through a shared risk manager and tracks
per-strategy performance independently.

Architecture:
    Orchestrator
        ├── StrategySlot("momentum", symbols=["AAPL","MSFT"], timeframe="1Hour")
        ├── StrategySlot("ml", symbols=["NVDA","TSLA"], timeframe="15Min")
        └── StrategySlot("mean_reversion", symbols=["BTC/USD"], timeframe="5Min")

Each slot runs on its own cadence (interval) and can be enabled/disabled at runtime.
"""

import copy
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.strategy.base import BaseStrategy, LegacyTradeSignal
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StrategySlot:
    """Configuration and state for one strategy in the orchestrator."""

    name: str
    strategy: BaseStrategy
    symbols: list[str]
    timeframe: str
    interval: int = 60  # seconds between cycles for this strategy
    enabled: bool = True
    weight: float = 1.0  # capital allocation weight (relative)

    # Runtime state (not user-configured)
    last_cycle: Optional[datetime] = None
    cycle_count: int = 0
    total_signals: int = 0
    total_trades: int = 0
    realized_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.0

    @property
    def is_due(self) -> bool:
        """Check if this strategy is due for a new cycle."""
        if not self.enabled:
            return False
        if self.last_cycle is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_cycle).total_seconds()
        return elapsed >= self.interval

    def record_cycle(self, signals: int, trades: list[dict]):
        """Record cycle results for this strategy."""
        with self._lock:
            self.last_cycle = datetime.now(timezone.utc)
            self.cycle_count += 1
            self.total_signals += signals
            self.total_trades += len(trades)
            for t in trades:
                pnl = t.get("pnl", 0)
                if pnl > 0:
                    self.win_count += 1
                elif pnl < 0:
                    self.loss_count += 1
                self.realized_pnl += pnl

    def get_stats(self) -> dict:
        """Get per-strategy performance stats."""
        with self._lock:
            return {
                "name": self.name,
                "enabled": self.enabled,
                "symbols": self.symbols,
                "timeframe": self.timeframe,
                "interval": self.interval,
                "weight": self.weight,
                "cycle_count": self.cycle_count,
                "total_signals": self.total_signals,
                "total_trades": self.total_trades,
                "realized_pnl": round(self.realized_pnl, 2),
                "win_rate": round(self.win_rate * 100, 1),
                "last_cycle": self.last_cycle.isoformat() if self.last_cycle else None,
            }


class StrategyOrchestrator:
    """
    Manages concurrent execution of multiple trading strategies.

    Each strategy runs on its own schedule/interval and symbol set.
    The orchestrator:
    - Decides which strategies are due for execution each tick
    - Runs them sequentially (thread-safe with broker) or in parallel
    - Tracks per-strategy P&L and metrics
    - Allows runtime enable/disable via Telegram
    """

    def __init__(self):
        self._slots: dict[str, StrategySlot] = {}
        self._lock = threading.Lock()
        self._total_cycles = 0

    def add_strategy(
        self,
        name: str,
        strategy: BaseStrategy,
        symbols: list[str],
        timeframe: str = "1Hour",
        interval: int = 60,
        weight: float = 1.0,
        enabled: bool = True,
    ) -> None:
        """Register a strategy slot."""
        with self._lock:
            self._slots[name] = StrategySlot(
                name=name,
                strategy=strategy,
                symbols=symbols,
                timeframe=timeframe,
                interval=interval,
                weight=weight,
                enabled=enabled,
            )
        logger.info(
            "orchestrator.strategy_added",
            name=name,
            symbols=len(symbols),
            interval=interval,
        )

    def remove_strategy(self, name: str) -> bool:
        """Remove a strategy slot."""
        with self._lock:
            if name in self._slots:
                del self._slots[name]
                logger.info("orchestrator.strategy_removed", name=name)
                return True
            return False

    def enable_strategy(self, name: str) -> bool:
        """Enable a strategy at runtime."""
        with self._lock:
            if name in self._slots:
                self._slots[name].enabled = True
                logger.info("orchestrator.strategy_enabled", name=name)
                return True
            return False

    def disable_strategy(self, name: str) -> bool:
        """Disable a strategy at runtime (stops generating signals)."""
        with self._lock:
            if name in self._slots:
                self._slots[name].enabled = False
                logger.info("orchestrator.strategy_disabled", name=name)
                return True
            return False

    def get_due_strategies(self) -> list[StrategySlot]:
        """Get strategies that are due for execution this tick."""
        with self._lock:
            return [slot for slot in self._slots.values() if slot.is_due]

    def run_due_strategies(self, engine) -> list[dict]:
        """
        Execute all strategies that are due.

        Args:
            engine: ExecutionEngine instance (shared, thread-safe with broker lock)

        Returns:
            Combined list of trade results from all strategies that ran.
        """
        due = self.get_due_strategies()
        if not due:
            return []

        # Compute normalized weights for capital allocation
        weights = self.get_weights()

        all_results = []
        for slot in due:
            try:
                # Create a shallow copy to avoid mutating shared strategy state
                # (race condition if multiple slots share the same strategy instance)
                strategy_instance = copy.copy(slot.strategy)
                strategy_instance.symbols = slot.symbols
                strategy_instance.timeframe = slot.timeframe

                capital_weight = weights.get(slot.name, 1.0)
                results = engine.run_cycle(strategy_instance, capital_weight=capital_weight)
                signal_count = len(slot.symbols)  # approximate
                slot.record_cycle(signal_count, results)

                # Tag results with strategy name for tracking
                for r in results:
                    r["_strategy"] = slot.name

                all_results.extend(results)
                self._total_cycles += 1

                logger.info(
                    "orchestrator.cycle_complete",
                    strategy=slot.name,
                    trades=len(results),
                )
            except Exception as e:
                logger.error(
                    "orchestrator.strategy_error",
                    strategy=slot.name,
                    error=str(e),
                )

        return all_results

    def get_all_stats(self) -> dict:
        """Get combined orchestrator statistics."""
        with self._lock:
            strategies = {name: slot.get_stats() for name, slot in self._slots.items()}
            total_pnl = sum(s["realized_pnl"] for s in strategies.values())
            active = sum(1 for s in strategies.values() if s["enabled"])

            return {
                "total_strategies": len(self._slots),
                "active_strategies": active,
                "total_cycles": self._total_cycles,
                "total_pnl": round(total_pnl, 2),
                "strategies": strategies,
            }

    def get_weights(self) -> dict[str, float]:
        """Get normalized capital allocation weights."""
        with self._lock:
            enabled = {n: s.weight for n, s in self._slots.items() if s.enabled}
            total_weight = sum(enabled.values())
            if total_weight == 0:
                return {}
            return {n: w / total_weight for n, w in enabled.items()}

    def set_weight(self, name: str, weight: float) -> bool:
        """Set capital allocation weight for a strategy."""
        with self._lock:
            if name in self._slots and weight > 0:
                self._slots[name].weight = weight
                logger.info("orchestrator.weight_set", name=name, weight=weight)
                return True
            return False

    @property
    def strategy_names(self) -> list[str]:
        with self._lock:
            return list(self._slots.keys())

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(1 for s in self._slots.values() if s.enabled)

    def __len__(self) -> int:
        return len(self._slots)

    def __repr__(self) -> str:
        return f"StrategyOrchestrator(strategies={len(self._slots)}, active={self.active_count})"

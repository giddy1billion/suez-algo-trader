"""
Trigger Types — Event-driven conditions for scheduler activities.

Each trigger evaluates whether its condition is met and can be combined
with other triggers using AND/OR logic in the activity graph.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class TriggerContext:
    """Context passed to trigger evaluation."""

    current_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accumulated_bars: dict[str, int] = field(default_factory=dict)  # symbol -> bar count
    drift_scores: dict[str, float] = field(default_factory=dict)  # symbol -> drift score
    parameter_hashes: dict[str, str] = field(default_factory=dict)  # component -> hash
    last_model_trained: Optional[datetime] = None
    last_backtest_run: Optional[datetime] = None
    last_retraining: Optional[datetime] = None
    events_received: list[str] = field(default_factory=list)  # event type names


class Trigger(ABC):
    """Base class for all trigger types."""

    @abstractmethod
    def evaluate(self, context: TriggerContext) -> bool:
        """Evaluate whether the trigger condition is met."""
        ...

    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the trigger."""
        ...

    def reset(self) -> None:
        """Reset trigger state after activation."""
        pass


class DataArrivalTrigger(Trigger):
    """Triggers when sufficient new data has accumulated."""

    def __init__(self, threshold: int = 100, symbol: Optional[str] = None):
        self.threshold = threshold
        self.symbol = symbol
        self._last_counts: dict[str, int] = {}

    def evaluate(self, context: TriggerContext) -> bool:
        if self.symbol:
            count = context.accumulated_bars.get(self.symbol, 0)
            last = self._last_counts.get(self.symbol, 0)
            return (count - last) >= self.threshold

        # Check if any symbol has enough new data
        for sym, count in context.accumulated_bars.items():
            last = self._last_counts.get(sym, 0)
            if (count - last) >= self.threshold:
                return True
        return False

    def reset(self) -> None:
        self._last_counts = dict(self._last_counts)

    def acknowledge(self, context: TriggerContext) -> None:
        """Record current counts after trigger fires."""
        self._last_counts = dict(context.accumulated_bars)

    def description(self) -> str:
        sym = self.symbol or "any"
        return f"DataArrival(threshold={self.threshold}, symbol={sym})"


class DriftTrigger(Trigger):
    """Triggers when drift monitor detects degradation above threshold."""

    def __init__(self, threshold: float = 0.12, symbol: Optional[str] = None):
        self.threshold = threshold
        self.symbol = symbol

    def evaluate(self, context: TriggerContext) -> bool:
        if self.symbol:
            score = context.drift_scores.get(self.symbol, 0.0)
            return score >= self.threshold

        # Check if any symbol has drift above threshold
        for score in context.drift_scores.values():
            if score >= self.threshold:
                return True
        return False

    def description(self) -> str:
        sym = self.symbol or "any"
        return f"Drift(threshold={self.threshold}, symbol={sym})"


class ScheduleTrigger(Trigger):
    """Triggers on a periodic schedule (fallback)."""

    def __init__(self, interval_seconds: float = 3600):
        self.interval_seconds = interval_seconds
        self._last_fired: Optional[float] = None

    def evaluate(self, context: TriggerContext) -> bool:
        now = time.time()
        if self._last_fired is None:
            self._last_fired = now
            return False
        return (now - self._last_fired) >= self.interval_seconds

    def reset(self) -> None:
        self._last_fired = time.time()

    def description(self) -> str:
        hours = self.interval_seconds / 3600
        return f"Schedule(every={hours:.1f}h)"


class ParameterChangeTrigger(Trigger):
    """Triggers when strategy/feature parameters change (hash comparison)."""

    def __init__(self, component: str = "strategy"):
        self.component = component
        self._last_hash: Optional[str] = None

    def evaluate(self, context: TriggerContext) -> bool:
        current_hash = context.parameter_hashes.get(self.component)
        if current_hash is None:
            return False
        if self._last_hash is None:
            self._last_hash = current_hash
            return False
        return current_hash != self._last_hash

    def reset(self) -> None:
        pass

    def acknowledge(self, context: TriggerContext) -> None:
        """Update stored hash after trigger fires."""
        self._last_hash = context.parameter_hashes.get(self.component)

    def description(self) -> str:
        return f"ParameterChange(component={self.component})"


class ModelTrainedTrigger(Trigger):
    """Triggers when a model training completes."""

    def __init__(self):
        self._last_seen: Optional[datetime] = None

    def evaluate(self, context: TriggerContext) -> bool:
        if context.last_model_trained is None:
            return False
        if self._last_seen is None:
            self._last_seen = context.last_model_trained
            return False
        return context.last_model_trained > self._last_seen

    def reset(self) -> None:
        pass

    def acknowledge(self, context: TriggerContext) -> None:
        """Record last seen training time."""
        self._last_seen = context.last_model_trained

    def description(self) -> str:
        return "ModelTrained()"


class ManualTrigger(Trigger):
    """Triggers on manual activation (external call)."""

    def __init__(self):
        self._activated = False

    def activate(self) -> None:
        """Externally activate this trigger."""
        self._activated = True

    def evaluate(self, context: TriggerContext) -> bool:
        return self._activated

    def reset(self) -> None:
        self._activated = False

    def description(self) -> str:
        return "Manual()"


def compute_parameter_hash(params: dict[str, Any]) -> str:
    """Compute a deterministic hash of parameters for change detection."""
    serialized = str(sorted(params.items()))
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]

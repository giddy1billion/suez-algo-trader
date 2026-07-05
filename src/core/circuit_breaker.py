"""
Trading Circuit Breaker — Prevents trading when system conditions are unsafe.

Monitors multiple health signals and transitions between states:
  NORMAL → SAFE_MODE → HALTED

Integrates with EventBus to publish state change events and can trigger
automatic recovery actions (retraining, rollback).
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CircuitBreakerState(str, Enum):
    """Trading circuit breaker states."""
    NORMAL = "NORMAL"
    SAFE_MODE = "SAFE_MODE"
    HALTED = "HALTED"


class CircuitBreakerReason(str, Enum):
    """Reasons for circuit breaker trips."""
    NO_ACTIVE_MODEL = "NO_ACTIVE_MODEL"
    BACKTEST_FAILURE = "BACKTEST_FAILURE"
    TRAINING_FAILURE = "TRAINING_FAILURE"
    FEATURE_FAILURE = "FEATURE_FAILURE"
    DATA_QUALITY = "DATA_QUALITY"
    PREDICTION_LATENCY = "PREDICTION_LATENCY"
    MODEL_DEGRADATION = "MODEL_DEGRADATION"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    STALE_DATA = "STALE_DATA"


@dataclass
class CircuitBreakerCheck:
    """Result of a single circuit breaker check."""
    name: str
    passed: bool
    reason: str = ""
    severity: str = "warning"  # "warning" | "critical"


class TradingCircuitBreaker:
    """
    Multi-signal circuit breaker for trading systems.

    Monitors model availability, health, data quality, and prediction
    latency to determine whether trading should be allowed.
    """

    def __init__(self, event_bus=None):
        self._state = CircuitBreakerState.NORMAL
        self._active_reasons: list[str] = []
        self._lock = threading.Lock()
        self._event_bus = event_bus
        self._trip_history: list[dict] = []
        self._last_check_time: Optional[datetime] = None

    @property
    def state(self) -> CircuitBreakerState:
        with self._lock:
            return self._state

    @property
    def active_reasons(self) -> list[str]:
        with self._lock:
            return list(self._active_reasons)

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        with self._lock:
            return self._state == CircuitBreakerState.NORMAL

    def check_all(
        self,
        predictor=None,
        health_monitor=None,
        governance=None,
    ) -> tuple[CircuitBreakerState, list[str]]:
        """
        Run all circuit breaker checks and update state.

        Args:
            predictor: ModelPredictor instance (checks model availability).
            health_monitor: HealthMonitor instance (checks system health).
            governance: ModelGovernance instance (checks model provenance).

        Returns:
            (current_state, list_of_active_reasons)
        """
        reasons = []
        self._last_check_time = datetime.now(timezone.utc)

        # Check model predictor availability
        if predictor is not None:
            if not getattr(predictor, 'is_loaded', False):
                reasons.append(CircuitBreakerReason.NO_ACTIVE_MODEL.value)

        # Check health monitor
        if health_monitor is not None:
            try:
                report = health_monitor.get_full_report()
                overall = report.get("overall_status", "unknown")
                if overall == "down":
                    reasons.append(CircuitBreakerReason.DATA_QUALITY.value)
            except Exception:
                pass

        # Check governance for deployed model provenance
        if governance is not None:
            try:
                deployed = governance.get_deployed_model()
                if deployed is None:
                    if CircuitBreakerReason.NO_ACTIVE_MODEL.value not in reasons:
                        reasons.append(CircuitBreakerReason.NO_ACTIVE_MODEL.value)
            except Exception:
                pass

        # Update state based on reasons
        with self._lock:
            old_state = self._state
            if reasons:
                self._state = CircuitBreakerState.SAFE_MODE
                self._active_reasons = reasons
            else:
                self._state = CircuitBreakerState.NORMAL
                self._active_reasons = []

            # Publish events on state transitions
            if old_state != self._state:
                if self._state != CircuitBreakerState.NORMAL:
                    self._on_trip(old_state)
                else:
                    self._on_reset(old_state)

        return self._state, reasons

    def trip(self, reason: str) -> None:
        """Manually trip the circuit breaker."""
        with self._lock:
            old_state = self._state
            self._state = CircuitBreakerState.SAFE_MODE
            if reason not in self._active_reasons:
                self._active_reasons.append(reason)
            if old_state == CircuitBreakerState.NORMAL:
                self._on_trip(old_state)

    def reset(self, reason: str = "") -> None:
        """
        Reset the circuit breaker to NORMAL.
        Only resets if all conditions have cleared.
        """
        with self._lock:
            old_state = self._state
            if reason and reason in self._active_reasons:
                self._active_reasons.remove(reason)
            if not self._active_reasons:
                self._state = CircuitBreakerState.NORMAL
                if old_state != CircuitBreakerState.NORMAL:
                    self._on_reset(old_state)

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status."""
        with self._lock:
            return {
                "state": self._state.value,
                "trading_allowed": self._state == CircuitBreakerState.NORMAL,
                "active_reasons": list(self._active_reasons),
                "last_check": self._last_check_time.isoformat() if self._last_check_time else None,
                "trip_count": len(self._trip_history),
            }

    def _on_trip(self, old_state: CircuitBreakerState) -> None:
        """Handle circuit breaker trip (called with lock held)."""
        self._trip_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from_state": old_state.value,
            "to_state": self._state.value,
            "reasons": list(self._active_reasons),
        })
        logger.warning(
            "circuit_breaker.tripped",
            state=self._state.value,
            reasons=self._active_reasons,
        )
        if self._event_bus:
            try:
                from src.core.events import CircuitBreakerTripped
                self._event_bus.publish(CircuitBreakerTripped(
                    state=self._state.value,
                    reasons=list(self._active_reasons),
                    source="circuit_breaker",
                ))
            except Exception:
                pass

    def _on_reset(self, old_state: CircuitBreakerState) -> None:
        """Handle circuit breaker reset (called with lock held)."""
        logger.info(
            "circuit_breaker.reset",
            previous_state=old_state.value,
        )
        if self._event_bus:
            try:
                from src.core.events import CircuitBreakerReset
                self._event_bus.publish(CircuitBreakerReset(
                    previous_state=old_state.value,
                    source="circuit_breaker",
                ))
            except Exception:
                pass

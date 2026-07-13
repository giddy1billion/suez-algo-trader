"""
Operational Circuit Breaker — Detects repeated broker and risk-engine failures.

Unlike the TradingCircuitBreaker (which monitors model/data health), this
circuit breaker monitors operational infrastructure: broker connectivity,
risk-engine evaluation errors, and notification delivery failures.

When failure counts exceed thresholds within a rolling window, trading is
safely halted to prevent cascading failures.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class OperationalState(str, Enum):
    """Operational circuit breaker states."""
    CLOSED = "CLOSED"       # Normal operation — all systems healthy
    OPEN = "OPEN"           # Failures detected — trading halted
    HALF_OPEN = "HALF_OPEN"  # Probing — allowing limited traffic to test recovery
    HALTED = "HALTED"       # Manual halt — all trading prevented until explicitly resumed


class FailureDomain(str, Enum):
    """Categories of operational failures tracked."""
    BROKER = "BROKER"
    RISK_ENGINE = "RISK_ENGINE"
    NOTIFICATION = "NOTIFICATION"


@dataclass
class FailureRecord:
    """A single failure event."""
    domain: FailureDomain
    timestamp: float  # monotonic
    wall_clock: str   # ISO UTC
    error: str
    context: dict = field(default_factory=dict)


class OperationalCircuitBreaker:
    """
    Monitors operational health and halts trading on repeated failures.

    Configuration:
    - failure_threshold: Number of failures within the window to trip.
    - window_seconds: Rolling time window for counting failures.
    - recovery_timeout: Seconds to wait before transitioning to HALF_OPEN.
    - probe_success_threshold: Successful probes needed to close.

    Thread-safe. Publishes state transitions via optional event_bus.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        window_seconds: float = 60.0,
        recovery_timeout: float = 30.0,
        probe_success_threshold: int = 3,
        event_bus: Optional[Any] = None,
        on_state_change: Optional[Callable[[OperationalState, OperationalState], None]] = None,
    ):
        self._failure_threshold = failure_threshold
        self._window_seconds = window_seconds
        self._recovery_timeout = recovery_timeout
        self._probe_success_threshold = probe_success_threshold
        self._event_bus = event_bus
        self._on_state_change = on_state_change

        self._state = OperationalState.CLOSED
        self._lock = threading.Lock()

        # Failure tracking per domain
        self._failures: dict[FailureDomain, deque] = {
            domain: deque() for domain in FailureDomain
        }
        # Global failure history for audit
        self._failure_history: list[FailureRecord] = []

        # Recovery tracking
        self._opened_at: Optional[float] = None
        self._probe_successes: int = 0

        # Metrics
        self._total_trips: int = 0
        self._total_recoveries: int = 0

    @property
    def state(self) -> OperationalState:
        """Current circuit breaker state."""
        with self._lock:
            self._check_recovery_timeout()
            return self._state

    @property
    def is_trading_allowed(self) -> bool:
        """Whether trading operations should proceed."""
        with self._lock:
            self._check_recovery_timeout()
            return self._state not in (OperationalState.OPEN, OperationalState.HALTED)

    def record_failure(
        self,
        domain: FailureDomain,
        error: str,
        context: Optional[dict] = None,
    ) -> OperationalState:
        """
        Record a failure event and potentially trip the breaker.

        Args:
            domain: Which operational domain failed.
            error: Human-readable error description.
            context: Optional dict with additional context.

        Returns:
            Current state after recording the failure.
        """
        now = time.monotonic()
        record = FailureRecord(
            domain=domain,
            timestamp=now,
            wall_clock=datetime.now(timezone.utc).isoformat(),
            error=error,
            context=context or {},
        )

        with self._lock:
            self._failures[domain].append(record)
            self._failure_history.append(record)
            # Trim history
            if len(self._failure_history) > 1000:
                self._failure_history = self._failure_history[-500:]

            # Expire old failures
            self._expire_failures(now)

            # Check if threshold breached
            total_recent = sum(len(q) for q in self._failures.values())
            if total_recent >= self._failure_threshold and self._state == OperationalState.CLOSED:
                self._trip(now)

            return self._state

    def record_success(self, domain: FailureDomain) -> OperationalState:
        """
        Record a successful operation. Used in HALF_OPEN state to close the breaker.

        Args:
            domain: Which operational domain succeeded.

        Returns:
            Current state after recording the success.
        """
        with self._lock:
            self._check_recovery_timeout()
            if self._state == OperationalState.HALF_OPEN:
                self._probe_successes += 1
                if self._probe_successes >= self._probe_success_threshold:
                    self._close()
            return self._state

    def force_open(self, reason: str = "manual") -> None:
        """Manually trip the circuit breaker."""
        with self._lock:
            if self._state != OperationalState.OPEN:
                self._trip(time.monotonic(), reason=reason)

    def force_close(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._close()

    def halt(self, reason: str = "manual_halt") -> None:
        """Enter HALTED state — prevents all trading until explicitly resumed.

        Unlike OPEN (which auto-recovers via HALF_OPEN), HALTED requires
        an explicit call to resume() to re-enable trading.
        """
        with self._lock:
            old_state = self._state
            if old_state == OperationalState.HALTED:
                return
            self._state = OperationalState.HALTED
            logger.warning("operational_circuit_breaker.halted", reason=reason)
            if self._on_state_change:
                try:
                    self._on_state_change(old_state, self._state)
                except Exception:
                    pass

    def resume(self) -> None:
        """Exit HALTED state and return to CLOSED (normal operation)."""
        with self._lock:
            if self._state != OperationalState.HALTED:
                return
            old_state = self._state
            self._state = OperationalState.CLOSED
            logger.info("operational_circuit_breaker.resumed_from_halt")
            if self._on_state_change:
                try:
                    self._on_state_change(old_state, self._state)
                except Exception:
                    pass

    def get_status(self) -> dict[str, Any]:
        """Get current status and metrics."""
        with self._lock:
            self._check_recovery_timeout()
            now = time.monotonic()
            self._expire_failures(now)
            return {
                "state": self._state.value,
                "trading_allowed": self._state not in (OperationalState.OPEN, OperationalState.HALTED),
                "failure_counts": {
                    domain.value: len(q) for domain, q in self._failures.items()
                },
                "total_recent_failures": sum(len(q) for q in self._failures.values()),
                "failure_threshold": self._failure_threshold,
                "window_seconds": self._window_seconds,
                "total_trips": self._total_trips,
                "total_recoveries": self._total_recoveries,
                "probe_successes": self._probe_successes if self._state == OperationalState.HALF_OPEN else 0,
            }

    def get_failure_history(self, limit: int = 50) -> list[dict]:
        """Get recent failure records for audit."""
        with self._lock:
            records = self._failure_history[-limit:]
        return [
            {
                "domain": r.domain.value,
                "timestamp": r.wall_clock,
                "error": r.error,
                "context": r.context,
            }
            for r in records
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trip(self, now: float, reason: str = "threshold_exceeded") -> None:
        """Transition to OPEN state (called with lock held)."""
        old_state = self._state
        self._state = OperationalState.OPEN
        self._opened_at = now
        self._probe_successes = 0
        self._total_trips += 1

        logger.warning(
            "operational_circuit_breaker.tripped",
            reason=reason,
            failure_counts={d.value: len(q) for d, q in self._failures.items()},
        )

        if self._on_state_change:
            try:
                self._on_state_change(old_state, self._state)
            except Exception:
                pass

        if self._event_bus:
            try:
                from src.core.events import CircuitBreakerTripped
                self._event_bus.publish(CircuitBreakerTripped(
                    state=self._state.value,
                    reasons=[reason],
                    source="operational_circuit_breaker",
                ))
            except Exception:
                pass

    def _close(self) -> None:
        """Transition to CLOSED state (called with lock held)."""
        old_state = self._state
        self._state = OperationalState.CLOSED
        self._opened_at = None
        self._probe_successes = 0
        self._total_recoveries += 1

        # Clear failure queues on recovery
        for q in self._failures.values():
            q.clear()

        logger.info("operational_circuit_breaker.recovered")

        if self._on_state_change:
            try:
                self._on_state_change(old_state, self._state)
            except Exception:
                pass

        if self._event_bus:
            try:
                from src.core.events import CircuitBreakerReset
                self._event_bus.publish(CircuitBreakerReset(
                    previous_state=old_state.value,
                    source="operational_circuit_breaker",
                ))
            except Exception:
                pass

    def _check_recovery_timeout(self) -> None:
        """Check if recovery timeout has elapsed to transition to HALF_OPEN."""
        if self._state == OperationalState.OPEN and self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._recovery_timeout:
                old_state = self._state
                self._state = OperationalState.HALF_OPEN
                self._probe_successes = 0
                logger.info(
                    "operational_circuit_breaker.half_open",
                    elapsed_seconds=elapsed,
                )
                if self._on_state_change:
                    try:
                        self._on_state_change(old_state, self._state)
                    except Exception:
                        pass

    def _expire_failures(self, now: float) -> None:
        """Remove failures older than the window (called with lock held)."""
        cutoff = now - self._window_seconds
        for q in self._failures.values():
            while q and q[0].timestamp < cutoff:
                q.popleft()

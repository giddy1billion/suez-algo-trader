"""
Runtime State — Centralized state accessible to all components.

Provides a thread-safe container for runtime state (pause, trading mode,
operating mode) that components can inject and check without circular imports.

Operating Modes (auto-managed by health score):
    NORMAL     → Full trading (health ≥ 0.8)
    WARNING    → Full trading + alerts (health 0.6–0.8)
    DEGRADED   → Reduced position sizes (health 0.4–0.6)
    SAFE_MODE  → No new entries, exits only (health 0.2–0.4)
    READ_ONLY  → No orders at all, monitoring only (health 0.1–0.2)
    HALT       → System stopped, manual intervention required (health < 0.1)
"""

import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class OperatingMode(str, Enum):
    """System operating mode — controls what actions are permitted."""
    NORMAL = "normal"
    WARNING = "warning"
    DEGRADED = "degraded"
    SAFE_MODE = "safe_mode"
    READ_ONLY = "read_only"
    HALT = "halt"


# Ordered from most permissive to most restrictive
_MODE_SEVERITY = [
    OperatingMode.NORMAL,
    OperatingMode.WARNING,
    OperatingMode.DEGRADED,
    OperatingMode.SAFE_MODE,
    OperatingMode.READ_ONLY,
    OperatingMode.HALT,
]

# Health score thresholds for auto-transition (lower bound for each mode)
_MODE_THRESHOLDS = {
    OperatingMode.NORMAL: 0.80,
    OperatingMode.WARNING: 0.60,
    OperatingMode.DEGRADED: 0.40,
    OperatingMode.SAFE_MODE: 0.20,
    OperatingMode.READ_ONLY: 0.10,
    OperatingMode.HALT: 0.0,
}

# Position size multiplier per mode
MODE_SIZE_MULTIPLIER = {
    OperatingMode.NORMAL: 1.0,
    OperatingMode.WARNING: 0.8,
    OperatingMode.DEGRADED: 0.5,
    OperatingMode.SAFE_MODE: 0.0,   # No new entries
    OperatingMode.READ_ONLY: 0.0,
    OperatingMode.HALT: 0.0,
}


class RuntimeState:
    """
    Thread-safe container for runtime state that all components can access.

    Centralizes state management so that:
    - ExecutionEngine checks operating mode before trading
    - HealthMonitor updates system health → mode auto-transitions
    - TelegramAuditForwarder checks if events should be suppressed
    - Telegram bot can set pause state without exposing globals
    """

    def __init__(self):
        """Initialize runtime state with defaults."""
        self._paused = False
        self._operating_mode = OperatingMode.NORMAL
        self._health_score: float = 1.0
        self._mode_reason: str = ""
        self._mode_changed_at: Optional[datetime] = None
        self._mode_history: list[dict] = []
        self._lock = threading.RLock()

    # ──────────────────────────────────────────────────────────────────────
    # Pause/Resume (backward-compatible)
    # ──────────────────────────────────────────────────────────────────────

    def is_paused(self) -> bool:
        """Check if trading is paused. Thread-safe."""
        with self._lock:
            return self._paused or self._operating_mode == OperatingMode.HALT

    def set_paused(self, paused: bool) -> None:
        """Set pause state. Thread-safe."""
        with self._lock:
            self._paused = paused

    def pause(self) -> None:
        """Convenience method to pause trading."""
        self.set_paused(True)

    def resume(self) -> None:
        """Convenience method to resume trading."""
        self.set_paused(False)

    @property
    def paused(self) -> bool:
        """Property accessor for pause state."""
        return self.is_paused()

    @paused.setter
    def paused(self, value: bool) -> None:
        """Property setter for pause state."""
        self.set_paused(value)

    # ──────────────────────────────────────────────────────────────────────
    # Operating Mode (system health-driven)
    # ──────────────────────────────────────────────────────────────────────

    @property
    def operating_mode(self) -> OperatingMode:
        """Current operating mode."""
        with self._lock:
            return self._operating_mode

    @property
    def health_score(self) -> float:
        """Current system health score (0.0–1.0)."""
        with self._lock:
            return self._health_score

    @property
    def can_open_positions(self) -> bool:
        """Whether the current mode allows opening new positions."""
        with self._lock:
            if self._paused:
                return False
            return self._operating_mode in (
                OperatingMode.NORMAL,
                OperatingMode.WARNING,
                OperatingMode.DEGRADED,
            )

    @property
    def can_close_positions(self) -> bool:
        """Whether the current mode allows closing positions."""
        with self._lock:
            if self._paused:
                return False
            return self._operating_mode in (
                OperatingMode.NORMAL,
                OperatingMode.WARNING,
                OperatingMode.DEGRADED,
                OperatingMode.SAFE_MODE,
            )

    @property
    def position_size_multiplier(self) -> float:
        """Position size multiplier for current mode."""
        with self._lock:
            return MODE_SIZE_MULTIPLIER.get(self._operating_mode, 0.0)

    def update_health(self, health_score: float, reason: str = "") -> Optional[OperatingMode]:
        """
        Update system health score and auto-transition mode if needed.

        Args:
            health_score: System health 0.0–1.0 (from HealthMonitor).
            reason: Why the health changed.

        Returns:
            New OperatingMode if a transition occurred, None otherwise.
        """
        with self._lock:
            self._health_score = max(0.0, min(1.0, health_score))
            new_mode = self._compute_mode(self._health_score)

            if new_mode != self._operating_mode:
                old_mode = self._operating_mode
                self._operating_mode = new_mode
                self._mode_reason = reason
                self._mode_changed_at = datetime.now(timezone.utc)
                self._mode_history.append({
                    "from": old_mode.value,
                    "to": new_mode.value,
                    "health_score": self._health_score,
                    "reason": reason,
                    "timestamp": self._mode_changed_at.isoformat(),
                })
                if len(self._mode_history) > 100:
                    self._mode_history = self._mode_history[-100:]

                logger.warning(
                    "operating_mode.transition",
                    from_mode=old_mode.value,
                    to_mode=new_mode.value,
                    health_score=round(self._health_score, 3),
                    reason=reason,
                )
                return new_mode
            return None

    def force_mode(self, mode: OperatingMode, reason: str = "manual") -> None:
        """Force a specific operating mode (manual override)."""
        with self._lock:
            old_mode = self._operating_mode
            self._operating_mode = mode
            self._mode_reason = reason
            self._mode_changed_at = datetime.now(timezone.utc)
            self._mode_history.append({
                "from": old_mode.value,
                "to": mode.value,
                "health_score": self._health_score,
                "reason": f"FORCED: {reason}",
                "timestamp": self._mode_changed_at.isoformat(),
            })
            logger.info(
                "operating_mode.forced",
                mode=mode.value,
                reason=reason,
            )

    def get_mode_status(self) -> dict:
        """Get full operating mode status for monitoring/telegram."""
        with self._lock:
            return {
                "mode": self._operating_mode.value,
                "health_score": round(self._health_score, 3),
                "reason": self._mode_reason,
                "changed_at": self._mode_changed_at.isoformat() if self._mode_changed_at else None,
                "can_open": self.can_open_positions,
                "can_close": self.can_close_positions,
                "size_multiplier": self.position_size_multiplier,
                "paused": self._paused,
                "transitions": len(self._mode_history),
            }

    def _compute_mode(self, score: float) -> OperatingMode:
        """Determine operating mode from health score."""
        for mode in _MODE_SEVERITY:
            threshold = _MODE_THRESHOLDS[mode]
            if score >= threshold:
                return mode
        return OperatingMode.HALT

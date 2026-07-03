"""
Trade Lifecycle State Machine.

Manages the state of trades through their lifecycle with strict
transition validation and full history tracking.
"""

import logging
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class TradeState(Enum):
    """All possible states in the trade lifecycle."""

    SIGNAL = "signal"
    PENDING_RISK = "pending_risk"
    RISK_APPROVED = "risk_approved"
    RISK_REJECTED = "risk_rejected"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    ACTIVE = "active"
    STOP_TRIGGERED = "stop_triggered"
    TRAILING = "trailing"
    CLOSING = "closing"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Transition Table
# ---------------------------------------------------------------------------

# Maps each state to the set of states it can transition to (excluding ERROR).
_TRANSITIONS: dict[TradeState, set[TradeState]] = {
    TradeState.SIGNAL: {TradeState.PENDING_RISK},
    TradeState.PENDING_RISK: {TradeState.RISK_APPROVED, TradeState.RISK_REJECTED},
    TradeState.RISK_APPROVED: {TradeState.SUBMITTED},
    TradeState.RISK_REJECTED: set(),
    TradeState.SUBMITTED: {TradeState.ACCEPTED, TradeState.CANCELLED, TradeState.EXPIRED},
    TradeState.ACCEPTED: {
        TradeState.PARTIALLY_FILLED,
        TradeState.FILLED,
        TradeState.CANCELLED,
        TradeState.EXPIRED,
    },
    TradeState.PARTIALLY_FILLED: {
        TradeState.FILLED,
        TradeState.CANCELLED,
        TradeState.EXPIRED,
    },
    TradeState.FILLED: {TradeState.ACTIVE},
    TradeState.ACTIVE: {
        TradeState.STOP_TRIGGERED,
        TradeState.TRAILING,
        TradeState.CLOSING,
        TradeState.CLOSED,
    },
    TradeState.STOP_TRIGGERED: {TradeState.CLOSING},
    TradeState.TRAILING: {TradeState.CLOSING, TradeState.STOP_TRIGGERED},
    TradeState.CLOSING: {TradeState.CLOSED},
    TradeState.CLOSED: set(),
    TradeState.CANCELLED: set(),
    TradeState.EXPIRED: set(),
    TradeState.ERROR: set(),
}


# ---------------------------------------------------------------------------
# Trade Lifecycle
# ---------------------------------------------------------------------------


class TradeLifecycle:
    """
    Manages the state of a single trade through its lifecycle.

    State transitions are validated against the transition table.
    ERROR is always reachable from any non-terminal state.
    """

    def __init__(self, trade_id: str, symbol: str, side: str) -> None:
        self.trade_id = trade_id
        self.symbol = symbol
        self.side = side
        self.state = TradeState.SIGNAL
        self.history: list[tuple[TradeState, datetime, str]] = [
            (TradeState.SIGNAL, datetime.now(timezone.utc), "created")
        ]
        self.metadata: dict = {}
        self._lock = threading.Lock()

    def transition(self, new_state: TradeState, reason: str = "") -> bool:
        """
        Attempt a state transition.

        Returns True if the transition was valid and applied,
        False if the transition is not allowed.
        """
        with self._lock:
            if not self._is_valid_transition(new_state):
                logger.warning(
                    "Invalid transition for trade %s: %s → %s (reason: %s)",
                    self.trade_id,
                    self.state.value,
                    new_state.value,
                    reason,
                )
                return False

            old_state = self.state
            self.state = new_state
            self.history.append((new_state, datetime.now(timezone.utc), reason))
            logger.info(
                "Trade %s: %s → %s (%s)",
                self.trade_id,
                old_state.value,
                new_state.value,
                reason or "no reason",
            )
            return True

    @property
    def valid_transitions(self) -> list[TradeState]:
        """Return valid next states from current state."""
        with self._lock:
            valid = list(_TRANSITIONS.get(self.state, set()))
            # ERROR is always reachable from non-terminal states
            terminal = {TradeState.CLOSED, TradeState.CANCELLED, TradeState.EXPIRED, TradeState.ERROR, TradeState.RISK_REJECTED}
            if self.state not in terminal:
                valid.append(TradeState.ERROR)
            return valid

    @property
    def is_terminal(self) -> bool:
        """Check if trade is in a terminal state."""
        return self.state in {
            TradeState.CLOSED,
            TradeState.CANCELLED,
            TradeState.EXPIRED,
            TradeState.ERROR,
            TradeState.RISK_REJECTED,
        }

    @property
    def duration(self) -> Optional[float]:
        """Duration from creation to current time in seconds."""
        if not self.history:
            return None
        start = self.history[0][1]
        return (datetime.now(timezone.utc) - start).total_seconds()

    def _is_valid_transition(self, new_state: TradeState) -> bool:
        """Check if a transition is valid from the current state."""
        # ERROR is always reachable from non-terminal states
        terminal = {TradeState.CLOSED, TradeState.CANCELLED, TradeState.EXPIRED, TradeState.ERROR, TradeState.RISK_REJECTED}
        if new_state == TradeState.ERROR and self.state not in terminal:
            return True

        allowed = _TRANSITIONS.get(self.state, set())
        return new_state in allowed

    def to_dict(self) -> dict:
        """Serialize trade lifecycle to a dictionary."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "state": self.state.value,
            "history": [
                {"state": s.value, "timestamp": t.isoformat(), "reason": r}
                for s, t, r in self.history
            ],
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"<TradeLifecycle {self.trade_id} [{self.state.value}] {self.symbol} {self.side}>"


# ---------------------------------------------------------------------------
# Trade Manager
# ---------------------------------------------------------------------------


class TradeManager:
    """Manages all active trade lifecycles."""

    def __init__(self) -> None:
        self._trades: dict[str, TradeLifecycle] = {}
        self._lock = threading.Lock()

    def create_trade(
        self,
        symbol: str,
        side: str,
        trade_id: Optional[str] = None,
        **metadata: dict,
    ) -> TradeLifecycle:
        """
        Create a new trade lifecycle in SIGNAL state.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT").
            side: Trade side ("BUY" or "SELL").
            trade_id: Optional custom ID. Auto-generated if not provided.
            **metadata: Additional metadata to attach.

        Returns:
            The created TradeLifecycle instance.
        """
        if trade_id is None:
            trade_id = f"T-{uuid.uuid4().hex[:8]}"

        trade = TradeLifecycle(trade_id=trade_id, symbol=symbol, side=side)
        trade.metadata.update(metadata)

        with self._lock:
            self._trades[trade_id] = trade

        logger.info("Created trade %s for %s %s", trade_id, side, symbol)
        return trade

    def get_trade(self, trade_id: str) -> Optional[TradeLifecycle]:
        """Get a trade by ID."""
        with self._lock:
            return self._trades.get(trade_id)

    def get_active_trades(self) -> list[TradeLifecycle]:
        """Get all non-terminal trades."""
        with self._lock:
            return [t for t in self._trades.values() if not t.is_terminal]

    def get_trades_by_state(self, state: TradeState) -> list[TradeLifecycle]:
        """Get all trades currently in a specific state."""
        with self._lock:
            return [t for t in self._trades.values() if t.state == state]

    def get_trades_by_symbol(self, symbol: str) -> list[TradeLifecycle]:
        """Get all trades for a specific symbol."""
        with self._lock:
            return [t for t in self._trades.values() if t.symbol == symbol]

    def get_trade_history(self, trade_id: str) -> list[tuple[TradeState, datetime, str]]:
        """Get the full state transition log for a trade."""
        trade = self.get_trade(trade_id)
        if trade is None:
            return []
        with trade._lock:
            return list(trade.history)

    def remove_terminal(self) -> int:
        """Remove all terminal trades from memory. Returns count removed."""
        with self._lock:
            terminal_ids = [
                tid for tid, t in self._trades.items() if t.is_terminal
            ]
            for tid in terminal_ids:
                del self._trades[tid]
            return len(terminal_ids)

    @property
    def count(self) -> int:
        """Total number of tracked trades."""
        with self._lock:
            return len(self._trades)

    @property
    def active_count(self) -> int:
        """Number of non-terminal trades."""
        with self._lock:
            return sum(1 for t in self._trades.values() if not t.is_terminal)

    def summary(self) -> dict[str, int]:
        """Get count of trades per state."""
        with self._lock:
            counts: dict[str, int] = {}
            for t in self._trades.values():
                key = t.state.value
                counts[key] = counts.get(key, 0) + 1
            return counts

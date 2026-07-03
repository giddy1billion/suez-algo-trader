"""
Event Bus System — Lightweight, thread-safe, in-process pub/sub.

Provides event classes for the algo-trader lifecycle and a central
EventBus for decoupled communication between components.
"""

import asyncio
import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Event schema version — increment when event structure changes
SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Base Event
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """Base event with timestamp and metadata."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to a dictionary."""
        data = asdict(self)
        data["_type"] = type(self).__name__
        data["_schema_version"] = SCHEMA_VERSION
        # Convert datetime to ISO string
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Deserialize event from a dictionary (base implementation)."""
        data = data.copy()
        data.pop("_type", None)
        data.pop("_schema_version", None)  # Remove version before constructing
        ts = data.get("timestamp")
        if isinstance(ts, str):
            data["timestamp"] = datetime.fromisoformat(ts)
        return cls(**data)


# ---------------------------------------------------------------------------
# Signal & Risk Events
# ---------------------------------------------------------------------------

@dataclass
class SignalGenerated(Event):
    """Emitted when a strategy generates a trade signal."""

    symbol: str = ""
    signal: str = ""  # "BUY" | "SELL"
    confidence: float = 0.0
    strategy: str = ""
    price: float = 0.0
    indicators: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskEvaluated(Event):
    """Emitted after risk management evaluates a signal."""

    symbol: str = ""
    approved: bool = False
    reasons: list[str] = field(default_factory=list)
    adjusted_qty: float = 0.0
    risk_score: float = 0.0


# ---------------------------------------------------------------------------
# Order Events
# ---------------------------------------------------------------------------

@dataclass
class OrderSubmitted(Event):
    """Emitted when an order is sent to the broker."""

    symbol: str = ""
    side: str = ""  # "BUY" | "SELL"
    qty: float = 0.0
    price: float = 0.0
    order_type: str = "MARKET"
    order_id: str = ""


@dataclass
class OrderAccepted(Event):
    """Emitted when the broker acknowledges an order."""

    order_id: str = ""
    broker_timestamp: Optional[datetime] = None


@dataclass
class OrderPartialFill(Event):
    """Emitted on partial order fills."""

    order_id: str = ""
    filled_qty: float = 0.0
    remaining_qty: float = 0.0
    fill_price: float = 0.0


@dataclass
class OrderFilled(Event):
    """Emitted when an order is fully filled."""

    order_id: str = ""
    fill_price: float = 0.0
    fill_qty: float = 0.0
    fees: float = 0.0


@dataclass
class OrderRejected(Event):
    """Emitted when an order is rejected by the broker."""

    order_id: str = ""
    reason: str = ""


# ---------------------------------------------------------------------------
# Trade Events
# ---------------------------------------------------------------------------

@dataclass
class TradeOpened(Event):
    """Emitted when a trade position is opened."""

    trade_id: str = ""
    symbol: str = ""
    side: str = ""
    entry_price: float = 0.0
    qty: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0


@dataclass
class TradeClosed(Event):
    """Emitted when a trade position is closed."""

    trade_id: str = ""
    symbol: str = ""
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------------
# System Events
# ---------------------------------------------------------------------------

@dataclass
class RiskHalt(Event):
    """Emitted when risk limits trigger a halt."""

    reason: str = ""
    level: str = "WARNING"  # "WARNING" | "CRITICAL"


@dataclass
class SchedulerEvent(Event):
    """Emitted for scheduled job lifecycle."""

    job_name: str = ""
    status: str = ""  # "started" | "completed" | "failed"


@dataclass
class SystemHealth(Event):
    """Emitted for component health checks."""

    component: str = ""
    status: str = ""  # "healthy" | "degraded" | "down"
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------

# Sentinel for wildcard subscriptions (subscribe to ALL events)
_WILDCARD = object()


class EventBus:
    """
    Pub/Sub event bus. Thread-safe, supports sync and async handlers.

    Usage:
        bus = EventBus()
        bus.subscribe(SignalGenerated, my_handler)
        bus.subscribe(None, audit_all)  # wildcard — receives all events
        bus.publish(SignalGenerated(symbol="BTCUSDT", signal="BUY", ...))
    """

    def __init__(self, max_history: int = 1000) -> None:
        self._subscribers: dict[Any, list[Callable]] = {}
        self._lock = threading.Lock()
        self._history: deque[Event] = deque(maxlen=max_history)
        self._max_history = max_history

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(self, event_type: Optional[type], handler: Callable) -> None:
        """
        Subscribe a handler to an event type.

        Args:
            event_type: The Event subclass to listen for, or None for wildcard.
            handler: A callable (sync function or async coroutine function).
        """
        key = event_type if event_type is not None else _WILDCARD
        with self._lock:
            if key not in self._subscribers:
                self._subscribers[key] = []
            if handler not in self._subscribers[key]:
                self._subscribers[key].append(handler)

    def unsubscribe(self, event_type: Optional[type], handler: Callable) -> None:
        """Remove a handler subscription."""
        key = event_type if event_type is not None else _WILDCARD
        with self._lock:
            handlers = self._subscribers.get(key, [])
            if handler in handlers:
                handlers.remove(handler)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, event: Event) -> None:
        """
        Publish an event synchronously. All handlers are called in order.
        Exceptions in handlers are logged but do not propagate.
        """
        self._record(event)
        handlers = self._get_handlers(type(event))
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    # Run async handler in event loop if available
                    self._run_async_handler(handler, event)
                else:
                    handler(event)
            except Exception:
                logger.exception(
                    "Handler %s raised an exception for event %s",
                    getattr(handler, "__name__", repr(handler)),
                    type(event).__name__,
                )

    def publish_async(self, event: Event) -> None:
        """
        Publish an event, running async handlers via asyncio.
        Falls back to publish() for sync handlers.
        """
        self._record(event)
        handlers = self._get_handlers(type(event))
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    self._run_async_handler(handler, event)
                else:
                    handler(event)
            except Exception:
                logger.exception(
                    "Async handler %s raised an exception for event %s",
                    getattr(handler, "__name__", repr(handler)),
                    type(event).__name__,
                )

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_history(
        self, event_type: Optional[type] = None, limit: int = 100
    ) -> list[Event]:
        """
        Return recent events, optionally filtered by type.

        Args:
            event_type: Filter to this event class, or None for all.
            limit: Max number of events to return.
        """
        with self._lock:
            if event_type is None:
                items = list(self._history)
            else:
                items = [e for e in self._history if isinstance(e, event_type)]
        return items[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._history.clear()

    @property
    def subscriber_count(self) -> int:
        """Total number of handler registrations."""
        with self._lock:
            return sum(len(h) for h in self._subscribers.values())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record(self, event: Event) -> None:
        """Add event to history."""
        with self._lock:
            self._history.append(event)

    def _get_handlers(self, event_type: type) -> list[Callable]:
        """Get all handlers for an event type (specific + wildcard)."""
        with self._lock:
            specific = list(self._subscribers.get(event_type, []))
            wildcard = list(self._subscribers.get(_WILDCARD, []))
        return specific + wildcard

    def _run_async_handler(self, handler: Callable, event: Event) -> None:
        """Execute an async handler, creating a loop if necessary."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(handler(event))
        except RuntimeError:
            # No running loop — run in a new one on a thread
            def _run() -> None:
                try:
                    asyncio.run(handler(event))
                except Exception:
                    logger.exception(
                        "Async handler %s failed",
                        getattr(handler, "__name__", repr(handler)),
                    )

            t = threading.Thread(target=_run, daemon=True)
            t.start()

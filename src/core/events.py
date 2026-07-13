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

# Migration hooks registry: version → callable(data_dict) → data_dict
# Each hook transforms event data from its version to the next.
# Example: _MIGRATIONS["0.9.0"] = lambda d: {**d, "new_field": "default"}
_MIGRATIONS: dict[str, Callable] = {}


def register_event_migration(from_version: str, migration_fn: Callable) -> None:
    """Register a migration hook for event deserialization (Finding 5).

    When deserializing events serialized at an older schema version,
    registered migrations are applied in version order to bring the
    data dict up to the current SCHEMA_VERSION.

    Args:
        from_version: The schema version this migration upgrades FROM.
        migration_fn: Callable that takes a data dict and returns
                      the upgraded data dict.
    """
    _MIGRATIONS[from_version] = migration_fn


def _apply_migrations(data: dict[str, Any], stored_version: str) -> dict[str, Any]:
    """Apply migration hooks to bring data from stored_version to current."""
    if stored_version == SCHEMA_VERSION:
        return data
    # Sort migration keys and apply sequentially
    sorted_versions = sorted(_MIGRATIONS.keys())
    for ver in sorted_versions:
        if ver >= stored_version and ver < SCHEMA_VERSION:
            data = _MIGRATIONS[ver](data)
    return data


# ---------------------------------------------------------------------------
# Base Event
# ---------------------------------------------------------------------------

# Auto-populated registry: every Event subclass is registered via __init_subclass__.
# Used by the event store for deserialization without manual maintenance.
_EVENT_CLASS_REGISTRY: dict[str, type] = {}


@dataclass(frozen=True)
class Event:
    """Base event with timestamp and metadata. Immutable after creation."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __init_subclass__(cls, **kwargs):
        """Automatically register every Event subclass for deserialization."""
        super().__init_subclass__(**kwargs)
        _EVENT_CLASS_REGISTRY[cls.__name__] = cls

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
        """Deserialize event from a dictionary (base implementation).

        Finding 5: Validates _schema_version and applies any registered
        migration hooks to bring old-format data up to the current
        schema before constructing the event object.
        """
        data = data.copy()
        data.pop("_type", None)
        stored_version = data.pop("_schema_version", SCHEMA_VERSION)

        # Validate and migrate (Finding 5)
        if stored_version != SCHEMA_VERSION:
            logger.info(
                "event.schema_migration",
                extra={"from_version": stored_version, "to_version": SCHEMA_VERSION},
            )
            data = _apply_migrations(data, stored_version)

        ts = data.get("timestamp")
        if isinstance(ts, str):
            data["timestamp"] = datetime.fromisoformat(ts)
        return cls(**data)


# ---------------------------------------------------------------------------
# Signal & Risk Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalGenerated(Event):
    """
    Emitted when a strategy generates a trade signal (proposal).

    This represents the strategy's proposal BEFORE any system-level evaluation.
    The payload follows the clean architecture: only what the strategy knows.
    """

    # Signal identity
    signal_id: str = ""

    # Strategy provenance
    strategy: str = ""
    strategy_version: str = ""

    # Market context
    symbol: str = ""
    timeframe: str = ""

    # Signal proposal
    signal: str = ""  # "BUY" | "SELL" (kept for backward compat)
    side: str = ""    # "BUY" | "SELL" (canonical field)
    signal_strength: float = 0.0
    expected_direction: int = 0  # +1 or -1

    # Legacy compat (deprecated — use signal_strength instead)
    confidence: float = 0.0
    price: float = 0.0

    # Evidence
    reason: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    indicators: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionContractCreated(Event):
    """
    Emitted when the DecisionOrchestrator produces an authoritative decision.

    This is the first point where the system has actually DECIDED something.
    The signal proposed; the contract decides.
    """

    contract_id: str = ""
    signal_id: str = ""

    # Decision outcome
    decision: str = ""  # "execute" | "reject" | "reduce" | "defer"
    final_confidence: float = 0.0

    # Market context
    symbol: str = ""
    side: str = ""

    # Execution parameters (determined by intelligence layer)
    recommended_position_pct: float = 0.0
    recommended_stop_loss: float = 0.0
    recommended_take_profit: float = 0.0
    risk_grade: str = ""

    # Assessment breakdown
    stage_scores: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    # Governance
    vetoed: bool = False
    veto_reason: str = ""
    expires_at: str = ""


@dataclass(frozen=True)
class RiskEvaluated(Event):
    """Emitted after risk management evaluates a signal."""

    symbol: str = ""
    signal_id: str = ""
    approved: bool = False
    reasons: list[str] = field(default_factory=list)
    adjusted_qty: float = 0.0
    risk_score: float = 0.0
    contract_id: str = ""  # Links to DecisionContract evaluated


@dataclass(frozen=True)
class SignalRejected(Event):
    """Emitted when a signal is rejected before reaching risk evaluation.

    This is a terminal verdict event that allows the notification correlation
    layer to cancel the pending deadline without waiting for a timeout.
    Covers: low strength, existing position, intelligence rejection,
    signal gate rejection, zero quantity, and state machine errors.
    """

    signal_id: str = ""
    symbol: str = ""
    reason: str = ""
    stage: str = ""  # "strength_gate" | "existing_position" | "intelligence" | "signal_gate" | "zero_qty" | "state_machine"


# ---------------------------------------------------------------------------
# Order Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderSubmitted(Event):
    """Emitted when an order is sent to the broker."""

    symbol: str = ""
    side: str = ""  # "BUY" | "SELL"
    qty: float = 0.0
    price: float = 0.0
    order_type: str = "MARKET"
    order_id: str = ""


@dataclass(frozen=True)
class OrderAccepted(Event):
    """Emitted when the broker acknowledges an order."""

    order_id: str = ""
    broker_timestamp: Optional[datetime] = None


@dataclass(frozen=True)
class OrderPartialFill(Event):
    """Emitted on partial order fills."""

    order_id: str = ""
    filled_qty: float = 0.0
    remaining_qty: float = 0.0
    fill_price: float = 0.0


@dataclass(frozen=True)
class OrderFilled(Event):
    """Emitted when an order is fully filled."""

    order_id: str = ""
    fill_price: float = 0.0
    fill_qty: float = 0.0
    fees: float = 0.0


@dataclass(frozen=True)
class OrderRejected(Event):
    """Emitted when an order is rejected by the broker."""

    order_id: str = ""
    reason: str = ""


# ---------------------------------------------------------------------------
# Trade Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TradeOpened(Event):
    """Emitted when a trade position is opened."""

    trade_id: str = ""
    symbol: str = ""
    side: str = ""
    entry_price: float = 0.0
    qty: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    contract_id: str = ""  # Links to DecisionContract that approved this trade


@dataclass(frozen=True)
class TradeClosed(Event):
    """Emitted when a trade position is closed."""

    trade_id: str = ""
    symbol: str = ""
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""
    # Enriched fields for closed-loop feedback
    entry_price: float = 0.0
    side: str = ""  # "buy" or "sell"
    entry_time: str = ""  # ISO timestamp
    exit_time: str = ""  # ISO timestamp
    model_version: str = ""
    strategy_name: str = ""
    signal_package_id: str = ""
    contract_id: str = ""  # Links to DecisionContract that approved this trade


# ---------------------------------------------------------------------------
# System Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskHalt(Event):
    """Emitted when risk limits trigger a halt."""

    reason: str = ""
    level: str = "WARNING"  # "WARNING" | "CRITICAL"


@dataclass(frozen=True)
class SchedulerEvent(Event):
    """Emitted for scheduled job lifecycle."""

    job_name: str = ""
    status: str = ""  # "started" | "completed" | "failed"


@dataclass(frozen=True)
class SystemHealth(Event):
    """Emitted for component health checks."""

    component: str = ""
    status: str = ""  # "healthy" | "degraded" | "down"
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Runtime Switching Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvironmentSwitched(Event):
    """Emitted when trading environment switches between paper and live."""

    old_mode: str = ""
    new_mode: str = ""
    broker_name: str = ""
    positions_closed: int = 0
    reason: str = ""


@dataclass(frozen=True)
class BrokerSwitched(Event):
    """Emitted when the broker instance is hot-swapped."""

    old_broker: str = ""
    new_broker: str = ""
    open_positions_migrated: int = 0


# ---------------------------------------------------------------------------
# ML Runtime Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSwapped(Event):
    """Emitted when an ML model is hot-swapped in production."""

    old_version: str = ""
    new_version: str = ""
    strategy: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ModelTrainingStarted(Event):
    """Emitted when model training pipeline begins."""

    pipeline_id: str = ""
    symbols: list[str] = field(default_factory=list)
    trigger: str = ""  # "manual" | "scheduled" | "performance_decay"


@dataclass(frozen=True)
class ModelTrainingCompleted(Event):
    """Emitted when model training pipeline finishes."""

    pipeline_id: str = ""
    version: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    auto_deployed: bool = False


@dataclass(frozen=True)
class ABTestStarted(Event):
    """Emitted when an A/B test begins."""

    test_id: str = ""
    champion_version: str = ""
    challenger_version: str = ""
    allocation_pct: float = 0.0


@dataclass(frozen=True)
class ABTestCompleted(Event):
    """Emitted when an A/B test concludes."""

    test_id: str = ""
    winner: str = ""
    champion_sharpe: float = 0.0
    challenger_sharpe: float = 0.0
    trades_evaluated: int = 0


# ---------------------------------------------------------------------------
# Backtest Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestStarted(Event):
    """Emitted when a backtest run begins."""

    run_id: str = ""
    strategies: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    engine: str = "native"


@dataclass(frozen=True)
class BacktestCompleted(Event):
    """Emitted when a backtest run finishes."""

    run_id: str = ""
    strategy: str = ""
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Data & Scheduler Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataIngested(Event):
    """Emitted when new market data is ingested."""

    asset_class: str = ""  # "equity" | "crypto"
    symbol: str = ""
    bar_count: int = 0


@dataclass(frozen=True)
class BacktestTriggered(Event):
    """Emitted when a backtest is triggered by the scheduler."""

    reason: str = ""
    symbols: list[str] = field(default_factory=list)
    trigger_source: str = ""  # "data_accumulation" | "drift" | "parameter_change" | "model_retrained" | "scheduled"


@dataclass(frozen=True)
class PredictionRegistered(Event):
    """Emitted when a new prediction is registered."""

    prediction_id: str = ""
    asset: str = ""
    direction: str = ""  # "long" | "short"
    confidence: float = 0.0
    horizon: str = ""


@dataclass(frozen=True)
class PredictionOutcomeRecorded(Event):
    """Emitted when a prediction outcome is recorded."""

    prediction_id: str = ""
    actual_return: float = 0.0
    quality_grade: str = ""  # "excellent" | "good" | "fair" | "poor"


@dataclass(frozen=True)
class RetrainingTriggered(Event):
    """Emitted when model retraining is triggered by evidence."""

    reason: str = ""
    evidence_summary: str = ""


@dataclass(frozen=True)
class ShadowDeploymentStarted(Event):
    """Emitted when a model enters shadow deployment."""

    model_version: str = ""
    baseline_version: str = ""


@dataclass(frozen=True)
class ShadowDeploymentCompleted(Event):
    """Emitted when shadow deployment evaluation completes."""

    model_version: str = ""
    result: str = ""  # "promoted" | "rejected"
    promoted: bool = False


@dataclass(frozen=True)
class CorrelationFilterApplied(Event):
    """Emitted when the correlation filter processes signals."""

    signals_received: int = 0
    signals_passed: int = 0
    signals_reduced: int = 0
    signals_skipped: int = 0


@dataclass(frozen=True)
class OperationalModeChanged(Event):
    """Emitted when the system operational mode changes."""

    old_mode: str = ""  # "research" | "paper" | "live"
    new_mode: str = ""
    reason: str = ""


# ---------------------------------------------------------------------------
# ML Lifecycle Hardening Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictionUnavailable(Event):
    """Emitted when a prediction cannot be generated."""

    symbol: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ModelRejected(Event):
    """Emitted when a model fails promotion validation."""

    version: str = ""
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CircuitBreakerTripped(Event):
    """Emitted when the trading circuit breaker trips."""

    state: str = ""  # "SAFE_MODE" | "HALTED"
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CircuitBreakerReset(Event):
    """Emitted when the trading circuit breaker resets."""

    previous_state: str = ""


@dataclass(frozen=True)
class ModelAutoRollback(Event):
    """Emitted when a model is automatically rolled back."""

    from_version: str = ""
    to_version: str = ""
    reason: str = ""


# ---------------------------------------------------------------------------
# Protective Exit Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProtectiveExitConfigured(Event):
    """Emitted when protective exit levels are computed for a new position."""

    symbol: str = ""
    side: str = ""  # "buy" or "sell"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    risk_reward_ratio: float = 0.0
    source: str = ""  # "strategy" | "atr" | "default" | "decision_contract"
    trade_source: str = ""  # "signal" | "manual" | "telegram"


@dataclass(frozen=True)
class ProtectiveExitAdjusted(Event):
    """Emitted when protective exit levels are adjusted (clamped/enforced)."""

    symbol: str = ""
    field_adjusted: str = ""  # "stop_loss" | "take_profit"
    original_value: float = 0.0
    adjusted_value: float = 0.0
    reason: str = ""  # "clamped_max" | "clamped_min" | "risk_reward_enforced"


@dataclass(frozen=True)
class ProtectiveExitExecuted(Event):
    """Emitted when a protective exit order (SL or TP) is filled by the broker."""

    symbol: str = ""
    exit_type: str = ""  # "stop_loss" | "take_profit"
    trigger_price: float = 0.0
    fill_price: float = 0.0
    qty: float = 0.0
    pnl: float = 0.0
    order_id: str = ""
    trade_id: str = ""
    broker_acknowledged: bool = False


def get_event_class_registry() -> dict[str, type]:
    """Return the auto-populated event class registry (read-only copy).

    All concrete Event subclasses are automatically registered via
    __init_subclass__ when their class body is evaluated.
    """
    return dict(_EVENT_CLASS_REGISTRY)


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

"""
Telegram Audit & Alert Forwarder.

Subscribes to ALL events on the event bus and forwards them to Telegram
in real-time. Also provides a Python logging handler that sends WARNING+
level logs to Telegram.

Design: Forward all events to Telegram, except when trading is paused.
Trading-related events (signals, orders, trades) are suppressed when paused,
but system events (health, errors) are still sent.

Correlation & Dedup:
    All correlation state (signals, deadlines, dedup keys) lives in a
    pluggable CorrelationStore with TTL-based cleanup.  A proactive
    background timer checks for timeouts independently of event arrival.
    Out-of-order delivery (RiskEvaluated arriving before SignalGenerated)
    is buffered and reconciled.  Deduplication covers both approvals
    and rejections.
"""

import asyncio
import hashlib
import logging
import math
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from src.core.events import (
    Event,
    EventBus,
    DecisionContractCreated,
    OrderAccepted,
    OrderFilled,
    OrderPartialFill,
    OrderRejected,
    OrderSubmitted,
    RiskEvaluated,
    RiskHalt,
    SchedulerEvent,
    SignalGenerated,
    SystemHealth,
    TradeClosed,
    TradeOpened,
)
from src.core.runtime_state import RuntimeState
from src.market.constraints import get_constraints
from src.market.registry import classify_symbol
from src.ml.model_registry import ModelRegistry
from src.notifications.signal_formatter import SignalPackage, format_signal_message
from src.notifications.correlation_store import (
    InMemoryCorrelationStore,
    CorrelationMetrics,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Event → Telegram Message Formatters
# ─────────────────────────────────────────────────────────────────────────────

def _format_signal(event: SignalGenerated) -> str:
    side = event.side or event.signal
    emoji = "📶" if side == "BUY" else "📉" if side == "SELL" else "⏸️"
    strength = event.signal_strength if event.signal_strength > 0 else event.confidence
    strategy_info = f"\nStrategy: <code>{event.strategy}</code>" if event.strategy else ""
    return (
        f"{emoji} <b>Signal: {side} {event.symbol}</b>\n"
        f"Strength: <code>{strength:.2f}</code>"
        f"{strategy_info}\n"
        f"Source: <code>{event.source}</code>"
    )


def _format_risk_evaluated(event: RiskEvaluated) -> str:
    approved = getattr(event, "approved", None)
    if approved is False:
        emoji = "🛡️"
        status = "REJECTED"
    elif approved is True:
        emoji = "✅"
        status = "APPROVED"
    else:
        emoji = "🔍"
        status = "EVALUATED"
    reasons = getattr(event, "reasons", [])
    reason_str = "; ".join(reasons) if reasons else "N/A"
    symbol = getattr(event, "symbol", "")
    return (
        f"{emoji} <b>Risk {status}</b>"
        f"{f' — {symbol}' if symbol else ''}\n"
        f"Reason: <code>{reason_str}</code>\n"
        f"Score: <code>{getattr(event, 'risk_score', 0):.2f}</code>"
    )


def _format_decision_contract(event: DecisionContractCreated) -> str:
    decision = event.decision.upper()
    if decision == "EXECUTE":
        emoji = "✅"
    elif decision == "REJECT":
        emoji = "❌"
    elif decision == "REDUCE":
        emoji = "⚠️"
    else:
        emoji = "⏳"
    return (
        f"{emoji} <b>Decision: {decision}</b> — {event.symbol}\n"
        f"Confidence: <code>{event.final_confidence:.2f}</code>\n"
        f"Position: <code>{event.recommended_position_pct:.1f}%</code> | "
        f"Grade: <code>{event.risk_grade or 'N/A'}</code>\n"
        f"Contract: <code>{event.contract_id[:16]}</code>"
    )


def _format_order_submitted(event: OrderSubmitted) -> str:
    return (
        f"📤 <b>Order Submitted</b>\n"
        f"ID: <code>{event.order_id}</code>\n"
        f"Symbol: <code>{getattr(event, 'symbol', '?')}</code> | "
        f"Side: <code>{getattr(event, 'side', '?')}</code> | "
        f"Qty: <code>{getattr(event, 'qty', '?')}</code>"
    )


def _format_order_accepted(event: OrderAccepted) -> str:
    return (
        f"✅ <b>Order Accepted</b>\n"
        f"ID: <code>{event.order_id}</code>"
    )


def _format_order_partial_fill(event: OrderPartialFill) -> str:
    return (
        f"⏳ <b>Partial Fill</b>\n"
        f"ID: <code>{event.order_id}</code>\n"
        f"Filled: <code>{event.filled_qty}</code> @ <code>${event.fill_price:.2f}</code>"
    )


def _format_order_filled(event: OrderFilled) -> str:
    return (
        f"💰 <b>Order Filled</b>\n"
        f"ID: <code>{event.order_id}</code>\n"
        f"Price: <code>${event.fill_price:.2f}</code>"
    )


def _format_order_rejected(event: OrderRejected) -> str:
    return (
        f"⛔ <b>Order REJECTED</b>\n"
        f"ID: <code>{event.order_id}</code>\n"
        f"Reason: <code>{event.reason}</code>"
    )


def _format_trade_opened(event: TradeOpened) -> str:
    return (
        f"📈 <b>Trade Opened: {event.side} {event.symbol}</b>\n"
        f"Entry: <code>${event.entry_price:.2f}</code> | Qty: <code>{event.qty}</code>\n"
        f"SL: <code>{event.stop_loss}</code> | TP: <code>{event.take_profit}</code>"
    )


def _format_trade_closed(event: TradeClosed) -> str:
    emoji = "💰" if event.pnl >= 0 else "💸"
    contract_line = ""
    if event.contract_id:
        contract_line = f"\nContract: <code>{event.contract_id[:20]}</code>"
    return (
        f"{emoji} <b>Trade Closed: {event.symbol}</b>\n"
        f"PnL: <code>${event.pnl:.2f} ({event.pnl_pct:.2f}%)</code>\n"
        f"Reason: <code>{event.reason}</code>"
        f"{contract_line}"
    )


def _format_risk_halt(event: RiskHalt) -> str:
    return (
        f"🚨 <b>RISK HALT [{event.level}]</b>\n"
        f"Reason: <code>{event.reason}</code>"
    )


def _format_scheduler_event(event: SchedulerEvent) -> str:
    status = getattr(event, "status", "unknown")
    icon = {"started": "⏱️", "completed": "✅", "failed": "❌"}.get(status, "📅")
    return (
        f"{icon} <b>Scheduler: {getattr(event, 'job_name', '?')}</b>\n"
        f"Status: <code>{status}</code>\n"
        f"Detail: <code>{getattr(event, 'detail', '')}</code>"
    )


def _format_system_health(event: SystemHealth) -> str:
    status = event.status
    icon = {"healthy": "🟢", "degraded": "🟡", "down": "🔴"}.get(status, "⚪")
    metrics = getattr(event, "metrics", {})
    latency = metrics.get("latency_ms", "N/A")
    return (
        f"{icon} <b>Health: {event.component}</b>\n"
        f"Status: <code>{status}</code>\n"
        f"Latency: <code>{latency}ms</code>"
    )


def _format_generic(event: Event) -> str:
    data = event.to_dict()
    # Compact display
    lines = [f"📋 <b>Event: {type(event).__name__}</b>"]
    for k, v in list(data.items())[:8]:
        if k in ("timestamp", "event_id"):
            continue
        lines.append(f"  {k}: <code>{v}</code>")
    return "\n".join(lines)


# Formatter dispatch table
_FORMATTERS: dict[type, Callable] = {
    DecisionContractCreated: _format_decision_contract,
    OrderSubmitted: _format_order_submitted,
    OrderAccepted: _format_order_accepted,
    OrderPartialFill: _format_order_partial_fill,
    OrderFilled: _format_order_filled,
    OrderRejected: _format_order_rejected,
    TradeOpened: _format_trade_opened,
    TradeClosed: _format_trade_closed,
    RiskHalt: _format_risk_halt,
    SchedulerEvent: _format_scheduler_event,
    SystemHealth: _format_system_health,
}


# ─────────────────────────────────────────────────────────────────────────────
# TelegramAuditForwarder — Wildcard subscriber, sends ALL events
# ─────────────────────────────────────────────────────────────────────────────


class TelegramAuditForwarder:
    """
    Subscribes to ALL event bus events and forwards them to Telegram.

    Trading-related events (signals, orders, trades) are suppressed when paused.
    System events (health, errors, scheduler) are always forwarded.

    Uses a background sender thread with a queue to avoid blocking the
    main trading loop on Telegram API latency.

    Correlation / Dedup / Timeout Design:
        - All state is held in a CorrelationStore (TTL-bounded, replaceable).
        - A background timer thread proactively checks for verdict timeouts.
        - Deduplication covers both approved AND rejected intents.
        - Out-of-order RiskEvaluated (before SignalGenerated) is buffered
          and reconciled when the signal arrives.
    """

    def __init__(
        self,
        send_func: Callable[[str], None],
        runtime_state: Optional[RuntimeState] = None,
        risk_verdict_timeout_seconds: float = 60.0,
        bracket_orders_supported_provider: Optional[Callable[[], bool]] = None,
        correlation_store: Optional[InMemoryCorrelationStore] = None,
        timeout_check_interval: float = 5.0,
    ) -> None:
        """
        Args:
            send_func: Synchronous callable that sends an HTML-formatted
                       message to Telegram (e.g., NotificationManager._send_telegram
                       or telegram_bot.send_sync).
            runtime_state: RuntimeState for checking pause state (optional).
                          If not provided, never suppresses events.
            correlation_store: Pluggable store for correlation state.
                             Defaults to a bounded in-memory store.
            timeout_check_interval: How often (seconds) the background timer
                                   checks for verdict timeouts.
        """
        self._send = send_func
        self._runtime_state = runtime_state or RuntimeState()
        self._queue: queue.Queue = queue.Queue(maxsize=5000)
        self._running = True
        self._sender_thread = threading.Thread(
            target=self._sender_loop,
            daemon=True,
            name="TelegramAuditSender",
        )
        self._sender_thread.start()
        self._events_sent = 0
        self._events_dropped = 0
        self._events_suppressed = 0
        self._risk_verdict_timeout_seconds = max(1.0, risk_verdict_timeout_seconds)
        self._bracket_orders_supported_provider = bracket_orders_supported_provider
        self._timeout_check_interval = max(0.5, timeout_check_interval)

        # Pluggable correlation store (default: in-memory with TTL)
        self._store = correlation_store or InMemoryCorrelationStore()

        # Track last health status per component to suppress repeated identical notifications
        self._last_health_status: dict[str, str] = {}

        # Proactive background timer for timeout emission
        self._timeout_thread = threading.Thread(
            target=self._timeout_loop,
            daemon=True,
            name="TelegramTimeoutChecker",
        )
        self._timeout_thread.start()

    def handle(self, event: Event) -> None:
        """Handle any event — format and enqueue for Telegram delivery.
        
        Suppresses trading-related events (signals, orders, trades) when paused,
        but always forwards system events.
        """
        try:
            # Suppress trading events when paused
            if self._runtime_state.is_paused():
                if isinstance(event, (SignalGenerated, RiskEvaluated, OrderSubmitted, OrderAccepted,
                                    OrderFilled, OrderPartialFill, OrderRejected,
                                    TradeOpened, TradeClosed)):
                    self._events_suppressed += 1
                    return  # Skip this event

            if isinstance(event, SignalGenerated):
                self._track_signal_generated(event)
                return

            if isinstance(event, DecisionContractCreated):
                self._track_decision_contract(event)

            if isinstance(event, RiskEvaluated):
                self._handle_risk_evaluated(event)
                return

            # Suppress health events: only forward status CHANGES to non-healthy
            if isinstance(event, SystemHealth):
                prev = self._last_health_status.get(event.component)
                self._last_health_status[event.component] = event.status
                if prev == event.status:
                    return  # Same status as last time — suppress
                # Suppress "healthy" events entirely unless recovering from degraded/down
                if event.status == "healthy" and prev not in ("degraded", "down"):
                    return  # No need to notify "healthy" on first appearance
            
            formatter = _FORMATTERS.get(type(event), _format_generic)
            message = formatter(event)

            # Add timestamp footer
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            message += f"\n<i>{ts}</i>"

            # Non-blocking enqueue
            try:
                self._queue.put_nowait(message)
            except queue.Full:
                self._events_dropped += 1
        except Exception:
            pass  # Never allow forwarder to interfere with event processing

    def _track_signal_generated(self, event: SignalGenerated) -> None:
        signal_id = (getattr(event, "signal_id", "") or "").strip()
        if not signal_id:
            signal_id = f"sig:{event.event_id}"
        self._store.store_signal(signal_id, event)
        self._store.set_deadline(signal_id, time.time() + self._risk_verdict_timeout_seconds)

        # Check if a RiskEvaluated arrived out-of-order before this signal
        late_risk = self._store.pop_late_risk(signal_id)
        if late_risk is not None:
            logger.info(
                "telegram_audit.late_signal_reconciled",
                extra={"signal_id": signal_id},
            )
            self._handle_risk_evaluated(late_risk)

    def _track_decision_contract(self, event: DecisionContractCreated) -> None:
        signal_id = (event.signal_id or "").strip()
        if signal_id:
            self._store.store_contract(signal_id, event)
        if event.contract_id and signal_id:
            self._store.map_contract_to_signal(event.contract_id, signal_id)

    def _emit_no_verdict_timeouts(self) -> None:
        """Check for expired deadlines and emit timeout warnings."""
        now = time.time()
        expired = self._store.get_expired_deadlines(now)
        for signal_id in expired:
            signal_event = self._store.get_signal(signal_id)
            if signal_event:
                block = _format_signal(signal_event)
                warning = (
                    f"{block}\n"
                    f"⚠️ NO VERDICT RECEIVED — no action taken "
                    f"(timeout={int(self._risk_verdict_timeout_seconds)}s)"
                )
                self._enqueue_message(warning)
                self._store.metrics.timeouts_emitted += 1
            # Atomically cancel the deadline
            self._store.cancel_deadline(signal_id)

    def _timeout_loop(self) -> None:
        """Proactive background timer that checks for verdict timeouts."""
        while self._running:
            try:
                self._emit_no_verdict_timeouts()
            except Exception:
                logger.debug("telegram_audit.timeout_check_error", exc_info=True)
            time.sleep(self._timeout_check_interval)

    def _handle_risk_evaluated(self, event: RiskEvaluated) -> None:
        signal_id = (getattr(event, "signal_id", "") or "").strip()
        if not signal_id and event.contract_id:
            signal_id = self._store.lookup_signal_by_contract(event.contract_id)
        signal_event = self._store.get_signal(signal_id) if signal_id else None
        contract_event = self._store.get_contract(signal_id) if signal_id else None

        # Out-of-order: RiskEvaluated arrived before SignalGenerated
        if signal_event is None and signal_id:
            self._store.record_late_risk(signal_id, event)
            logger.info(
                "telegram_audit.risk_before_signal_buffered",
                extra={"signal_id": signal_id},
            )
            return

        if signal_event is not None:
            self._store.metrics.verdicts_correlated += 1
        else:
            self._store.metrics.verdicts_uncorrelated += 1

        intent = self._build_trade_intent(event, signal_event, contract_event, signal_id)

        # Dedup: suppress BOTH approved AND rejected duplicate intents
        if self._store.check_dedup(intent.trade_intent_id):
            self._store.metrics.duplicates_suppressed += 1
            return

        rendered = format_signal_message(intent)
        self._enqueue_message(rendered)
        self._store.mark_sent(intent.trade_intent_id)

        # Atomically cancel the pending deadline for this signal
        if signal_id:
            self._store.cancel_deadline(signal_id)

    def _build_trade_intent(
        self,
        risk_event: RiskEvaluated,
        signal_event: Optional[SignalGenerated],
        contract_event: Optional[DecisionContractCreated],
        signal_id: str,
    ) -> SignalPackage:
        symbol = (
            (signal_event.symbol if signal_event else "")
            or getattr(risk_event, "symbol", "")
        )
        direction = ""
        if signal_event:
            direction = (signal_event.side or signal_event.signal or "").upper()
        confidence = 0.0
        if signal_event:
            confidence = (
                signal_event.signal_strength
                if signal_event.signal_strength > 0
                else signal_event.confidence
            )
        strategy = signal_event.strategy if signal_event else ""
        source = signal_event.source if signal_event else getattr(risk_event, "source", "")
        tags = tuple(getattr(signal_event, "tags", tuple()) or tuple())
        is_fallback_source = any(str(tag).lower() == "fallback" for tag in tags)
        provenance = "fallback" if is_fallback_source else (source or "unknown")

        active_model_version = self._get_active_model_version()
        strategy_version = signal_event.strategy_version if signal_event else ""
        model_active = bool(active_model_version) and (
            not strategy_version or active_model_version == strategy_version
        )

        instrument = classify_symbol(symbol) if symbol else None
        constraints = get_constraints(instrument) if instrument else None
        qty_step = constraints.lot_size if constraints else None

        explicit_qty = self._read_finite_positive(
            getattr(signal_event, "features", {}).get("position_size")
            if signal_event
            else None
        )
        computed_qty = self._read_finite_positive(getattr(risk_event, "adjusted_qty", None))
        if computed_qty is not None and constraints is not None:
            computed_qty = constraints.round_quantity(computed_qty)
            if computed_qty <= 0:
                computed_qty = None

        stop_loss = None
        take_profit = None
        if contract_event:
            stop_loss = self._read_finite_positive(contract_event.recommended_stop_loss)
            take_profit = self._read_finite_positive(contract_event.recommended_take_profit)
        if stop_loss is None and signal_event:
            stop_loss = self._read_finite_positive(
                getattr(signal_event, "features", {}).get("strategy_proposed_stop_loss")
            )
        if take_profit is None and signal_event:
            take_profit = self._read_finite_positive(
                getattr(signal_event, "features", {}).get("strategy_proposed_take_profit")
            )
        if constraints is not None:
            if stop_loss is not None:
                stop_loss = constraints.round_price(stop_loss)
            if take_profit is not None:
                take_profit = constraints.round_price(take_profit)

        risk_rejection_reason = "; ".join(getattr(risk_event, "reasons", []) or [])
        intent_id = self._stable_intent_id(risk_event, signal_id, symbol, direction, strategy)

        return SignalPackage(
            trade_intent_id=intent_id,
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            strength=confidence if isinstance(confidence, (int, float)) else 0.0,
            strategy=strategy,
            source=source,
            provenance=provenance,
            signal_block=_format_signal(signal_event) if signal_event else _format_generic(risk_event),
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=explicit_qty,
            auto_sized_qty=computed_qty,
            quantity_step=qty_step,
            model_active=model_active,
            risk_approved=bool(getattr(risk_event, "approved", False)),
            risk_rejection_reason=risk_rejection_reason,
            is_fallback_source=is_fallback_source,
            bracket_orders_supported=self._get_bracket_orders_supported(),
        )

    def _enqueue_message(self, message: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        with_timestamp = f"{message}\n<i>{ts}</i>"
        try:
            self._queue.put_nowait(with_timestamp)
        except queue.Full:
            self._events_dropped += 1

    def _get_active_model_version(self) -> str:
        try:
            return ModelRegistry().get_active_version() or ""
        except Exception:
            return ""

    def _get_bracket_orders_supported(self) -> bool:
        if self._bracket_orders_supported_provider is None:
            return False
        try:
            return bool(self._bracket_orders_supported_provider())
        except Exception:
            return False

    @staticmethod
    def _read_finite_positive(value: Any) -> Optional[float]:
        if value is None or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(float(value)) or float(value) <= 0:
            return None
        return float(value)

    @staticmethod
    def _stable_intent_id(
        risk_event: RiskEvaluated,
        signal_id: str,
        symbol: str,
        direction: str,
        strategy: str,
    ) -> str:
        if risk_event.contract_id:
            return f"intent:{risk_event.contract_id}"
        if signal_id:
            return f"intent:{signal_id}"
        raw = f"{symbol}|{direction}|{strategy}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"intent:{digest}"

    def _sender_loop(self) -> None:
        """Background thread that drains the queue and sends to Telegram."""
        while self._running:
            try:
                message = self._queue.get(timeout=1.0)
                try:
                    self._send(message)
                    self._events_sent += 1
                except Exception:
                    pass  # Silently handle send failures to avoid recursion
                # Small delay to avoid Telegram rate limits (30 msg/sec max)
                time.sleep(0.05)
            except queue.Empty:
                continue

    def register(self, bus: EventBus) -> None:
        """Register as wildcard subscriber — receives ALL events."""
        bus.subscribe(None, self.handle)

    def stop(self) -> None:
        """Gracefully stop the sender and timeout threads."""
        self._running = False
        if self._sender_thread.is_alive():
            self._sender_thread.join(timeout=5.0)
        if self._timeout_thread.is_alive():
            self._timeout_thread.join(timeout=5.0)
        # Final cleanup of expired state
        self._store.cleanup_expired()

    @property
    def stats(self) -> dict:
        return {
            "events_sent": self._events_sent,
            "events_dropped": self._events_dropped,
            "events_suppressed": self._events_suppressed,
            "queue_size": self._queue.qsize(),
            "running": self._running,
        }

    @property
    def correlation_metrics(self) -> CorrelationMetrics:
        """Observable correlation/dedup metrics."""
        return self._store.metrics


# ─────────────────────────────────────────────────────────────────────────────
# TelegramLogHandler — Python logging handler that sends WARNING+ to Telegram
# ─────────────────────────────────────────────────────────────────────────────


class TelegramLogHandler(logging.Handler):
    """
    Python logging handler that forwards WARNING, ERROR, and CRITICAL
    level log messages to Telegram.

    This ensures that ALL system warnings and errors are surfaced to the
    user in real-time — not just event-bus events but any logger.warning(),
    logger.error(), or logger.critical() call anywhere in the codebase.
    """

    def __init__(self, send_func: Callable[[str], None], level=logging.WARNING):
        super().__init__(level=level)
        self._send = send_func
        self._queue: queue.Queue = queue.Queue(maxsize=2000)
        self._running = True
        self._sender_thread = threading.Thread(
            target=self._drain_loop,
            daemon=True,
            name="TelegramLogSender",
        )
        self._sender_thread.start()
        self._messages_sent = 0

    def emit(self, record: logging.LogRecord) -> None:
        """Format and enqueue log record for Telegram delivery."""
        try:
            level = record.levelname
            icon = {
                "WARNING": "⚠️",
                "ERROR": "🔴",
                "CRITICAL": "🆘",
            }.get(level, "📝")

            # Truncate long messages
            msg = self.format(record)
            if len(msg) > 500:
                msg = msg[:497] + "..."

            text = (
                f"{icon} <b>[{level}]</b> <code>{record.name}</code>\n"
                f"<pre>{_escape_html(msg)}</pre>\n"
                f"<i>{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}</i>"
            )

            try:
                self._queue.put_nowait(text)
            except queue.Full:
                pass  # Drop silently to avoid recursion
        except Exception:
            pass  # Never allow logging handler to raise

    def _drain_loop(self) -> None:
        """Background sender for log messages."""
        while self._running:
            try:
                message = self._queue.get(timeout=1.0)
                try:
                    self._send(message)
                    self._messages_sent += 1
                except Exception:
                    pass  # Avoid recursion if Telegram send itself logs
                time.sleep(0.1)  # Rate limit log messages
            except queue.Empty:
                continue

    def stop(self) -> None:
        self._running = False
        if self._sender_thread.is_alive():
            self._sender_thread.join(timeout=5.0)


def _escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram HTML parse mode."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Setup Helper
# ─────────────────────────────────────────────────────────────────────────────


def setup_telegram_full_audit(
    event_bus: EventBus,
    send_func: Callable[[str], None],
    runtime_state: Optional[RuntimeState] = None,
    attach_log_handler: bool = True,
) -> dict[str, Any]:
    """
    Wire up full Telegram audit forwarding.

    This function:
    1. Creates a TelegramAuditForwarder (wildcard event subscriber)
    2. Optionally creates a TelegramLogHandler (WARNING+ to Telegram)
    3. Registers both so system events are forwarded (trading events suppressed when paused)

    Args:
        event_bus: The application event bus.
        send_func: Callable that sends HTML-formatted text to Telegram.
        runtime_state: RuntimeState for pause checking (optional).
        attach_log_handler: Whether to also send WARNING+ logs to Telegram.

    Returns:
        Dict with the forwarder and handler instances for lifecycle management.
    """
    # 1. Event bus → Telegram (all events, respecting pause state)
    forwarder = TelegramAuditForwarder(send_func, runtime_state=runtime_state)
    forwarder.register(event_bus)
    logger.info("telegram_audit.forwarder_registered")

    result: dict[str, Any] = {"forwarder": forwarder}

    # 2. Python logging → Telegram (WARNING+)
    if attach_log_handler:
        log_handler = TelegramLogHandler(send_func, level=logging.WARNING)
        log_handler.setFormatter(logging.Formatter("%(message)s"))
        # Attach to root logger to capture ALL warnings/errors from any module
        logging.getLogger().addHandler(log_handler)
        result["log_handler"] = log_handler
        logger.info("telegram_audit.log_handler_attached")

    return result

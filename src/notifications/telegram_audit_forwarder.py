"""
Telegram Audit & Alert Forwarder.

Subscribes to ALL events on the event bus and forwards them to Telegram
in real-time. Also provides a Python logging handler that sends WARNING+
level logs to Telegram.

Design: Forward all events to Telegram, except when trading is paused.
Trading-related events (signals, orders, trades) are suppressed when paused,
but system events (health, errors) are still sent.
"""

import asyncio
import logging
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
    SignalGenerated: _format_signal,
    DecisionContractCreated: _format_decision_contract,
    RiskEvaluated: _format_risk_evaluated,
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
    """

    def __init__(
        self,
        send_func: Callable[[str], None],
        runtime_state: Optional[RuntimeState] = None,
    ) -> None:
        """
        Args:
            send_func: Synchronous callable that sends an HTML-formatted
                       message to Telegram (e.g., NotificationManager._send_telegram
                       or telegram_bot.send_sync).
            runtime_state: RuntimeState for checking pause state (optional).
                          If not provided, never suppresses events.
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

    def handle(self, event: Event) -> None:
        """Handle any event — format and enqueue for Telegram delivery.
        
        Suppresses trading-related events (signals, orders, trades) when paused,
        but always forwards system events.
        """
        try:
            # Suppress trading events when paused
            if self._runtime_state.is_paused():
                if isinstance(event, (SignalGenerated, OrderSubmitted, OrderAccepted, 
                                    OrderFilled, OrderPartialFill, OrderRejected, 
                                    TradeOpened, TradeClosed)):
                    self._events_suppressed += 1
                    return  # Skip this event
            
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
        """Gracefully stop the sender thread."""
        self._running = False
        if self._sender_thread.is_alive():
            self._sender_thread.join(timeout=5.0)

    @property
    def stats(self) -> dict:
        return {
            "events_sent": self._events_sent,
            "events_dropped": self._events_dropped,
            "events_suppressed": self._events_suppressed,
            "queue_size": self._queue.qsize(),
            "running": self._running,
        }


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

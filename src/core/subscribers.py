"""
Default Event Bus Subscribers.

Sets up standard subscribers that wire the event system into
logging, auditing, metrics, and notifications.
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from .audit_log import AuditLogger
from .events import (
    Event,
    EventBus,
    OrderFilled,
    OrderRejected,
    OrderSubmitted,
    RiskHalt,
    SignalGenerated,
    TradeClosed,
    TradeOpened,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audit Subscriber — logs ALL events to immutable audit trail
# ---------------------------------------------------------------------------


class AuditSubscriber:
    """Listens for ALL events and writes them to the audit log."""

    def __init__(self, audit_logger: Optional[AuditLogger] = None) -> None:
        self._audit = audit_logger or AuditLogger()

    def handle(self, event: Event) -> None:
        """Write event to audit log."""
        try:
            data = event.to_dict()
            trade_id = data.get("trade_id", "")
            symbol = data.get("symbol", "")
            self._audit.log(
                event_type=type(event).__name__,
                data=data,
                trade_id=trade_id,
                symbol=symbol,
                source=event.source,
            )
        except Exception:
            logger.exception("AuditSubscriber failed for %s", type(event).__name__)

    def register(self, bus: EventBus) -> None:
        """Register as wildcard subscriber on the bus."""
        bus.subscribe(None, self.handle)


# ---------------------------------------------------------------------------
# Journal Subscriber — logs trade closures for the trading journal
# ---------------------------------------------------------------------------


class JournalSubscriber:
    """Listens for TradeClosed events and logs to the trading journal."""

    def __init__(self, journal_path: str = "data_cache/journal.jsonl") -> None:
        self._journal_path = journal_path

    def handle(self, event: TradeClosed) -> None:
        """Log completed trade to journal."""
        import json
        import os

        try:
            os.makedirs(os.path.dirname(self._journal_path), exist_ok=True)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trade_id": event.trade_id,
                "symbol": event.symbol,
                "exit_price": event.exit_price,
                "pnl": event.pnl,
                "pnl_pct": event.pnl_pct,
                "reason": event.reason,
                "source": event.source,
            }
            with open(self._journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")

            logger.info(
                "Journal: %s %s closed | PnL: %.2f (%.2f%%)",
                event.trade_id,
                event.symbol,
                event.pnl,
                event.pnl_pct,
            )
        except Exception:
            logger.exception("JournalSubscriber failed for trade %s", event.trade_id)

    def register(self, bus: EventBus) -> None:
        """Register for TradeClosed events."""
        bus.subscribe(TradeClosed, self.handle)


# ---------------------------------------------------------------------------
# Metrics Subscriber — maintains running trade metrics
# ---------------------------------------------------------------------------


class MetricsSubscriber:
    """Listens for TradeClosed events and updates running performance metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.total_pnl: float = 0.0
        self.max_win: float = 0.0
        self.max_loss: float = 0.0
        self.consecutive_wins: int = 0
        self.consecutive_losses: int = 0
        self._last_was_win: Optional[bool] = None

    def handle(self, event: TradeClosed) -> None:
        """Update metrics on trade close."""
        try:
            with self._lock:
                self.total_trades += 1
                self.total_pnl += event.pnl

                if event.pnl > 0:
                    self.winning_trades += 1
                    self.max_win = max(self.max_win, event.pnl)
                    if self._last_was_win is True:
                        self.consecutive_wins += 1
                    else:
                        self.consecutive_wins = 1
                        self.consecutive_losses = 0
                    self._last_was_win = True
                elif event.pnl < 0:
                    self.losing_trades += 1
                    self.max_loss = min(self.max_loss, event.pnl)
                    if self._last_was_win is False:
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 1
                        self.consecutive_wins = 0
                    self._last_was_win = False
                else:
                    # Break-even trade: track total but don't affect win/loss counts
                    self.consecutive_wins = 0
                    self.consecutive_losses = 0
                    self._last_was_win = None

                logger.debug(
                    "Metrics updated: %d trades, %.2f total PnL, %.1f%% win rate",
                    self.total_trades,
                    self.total_pnl,
                    self.win_rate * 100,
                )
        except Exception:
            logger.exception("MetricsSubscriber failed")

    @property
    def win_rate(self) -> float:
        """Current win rate as a ratio (0.0 to 1.0). Excludes break-even trades."""
        decided_trades = self.winning_trades + self.losing_trades
        if decided_trades == 0:
            return 0.0
        return self.winning_trades / decided_trades

    def to_dict(self) -> dict[str, Any]:
        """Serialize current metrics."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "max_win": self.max_win,
            "max_loss": self.max_loss,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
        }

    def register(self, bus: EventBus) -> None:
        """Register for TradeClosed events."""
        bus.subscribe(TradeClosed, self.handle)


# ---------------------------------------------------------------------------
# Notification Subscriber — sends alerts via Telegram if configured
# ---------------------------------------------------------------------------


class NotificationSubscriber:
    """Listens for key trade events and sends notifications via Telegram."""

    def __init__(self, send_func: Optional[Any] = None) -> None:
        """
        Args:
            send_func: A callable(message: str) that sends a notification.
                       If None, notifications are only logged.
        """
        self._send = send_func

    def handle_trade_opened(self, event: TradeOpened) -> None:
        """Notify on trade open."""
        msg = (
            f"📈 Trade Opened: {event.side} {event.symbol}\n"
            f"Entry: {event.entry_price} | Qty: {event.qty}\n"
            f"SL: {event.stop_loss} | TP: {event.take_profit}"
        )
        self._notify(msg)

    def handle_trade_closed(self, event: TradeClosed) -> None:
        """Notify on trade close."""
        emoji = "✅" if event.pnl >= 0 else "❌"
        msg = (
            f"{emoji} Trade Closed: {event.symbol}\n"
            f"PnL: {event.pnl:.2f} ({event.pnl_pct:.2f}%)\n"
            f"Reason: {event.reason}"
        )
        self._notify(msg)

    def handle_risk_halt(self, event: RiskHalt) -> None:
        """Notify on risk halt."""
        msg = f"🚨 RISK HALT [{event.level}]: {event.reason}"
        self._notify(msg)

    def handle_order_rejected(self, event: OrderRejected) -> None:
        """Notify on order rejection."""
        msg = f"⚠️ Order Rejected: {event.order_id}\nReason: {event.reason}"
        self._notify(msg)

    def _notify(self, message: str) -> None:
        """Send notification or log if no send function configured."""
        logger.info("Notification: %s", message.replace("\n", " | "))
        if self._send:
            try:
                self._send(message)
            except Exception:
                logger.exception("Failed to send notification")

    def register(self, bus: EventBus) -> None:
        """Register for relevant trade events."""
        bus.subscribe(TradeOpened, self.handle_trade_opened)
        bus.subscribe(TradeClosed, self.handle_trade_closed)
        bus.subscribe(RiskHalt, self.handle_risk_halt)
        bus.subscribe(OrderRejected, self.handle_order_rejected)


# ---------------------------------------------------------------------------
# Setup helper
# ---------------------------------------------------------------------------


def setup_default_subscribers(
    bus: EventBus,
    audit_logger: Optional[AuditLogger] = None,
    notification_send_func: Optional[Any] = None,
) -> dict[str, Any]:
    """
    Wire up all default subscribers to the event bus.

    Args:
        bus: The EventBus instance.
        audit_logger: Optional custom AuditLogger.
        notification_send_func: Optional callable for sending notifications.

    Returns:
        Dict of subscriber instances for reference/testing.
    """
    audit = AuditSubscriber(audit_logger)
    journal = JournalSubscriber()
    metrics = MetricsSubscriber()
    notifications = NotificationSubscriber(notification_send_func)

    audit.register(bus)
    journal.register(bus)
    metrics.register(bus)
    notifications.register(bus)

    logger.info("Default event subscribers registered (%d handlers)", bus.subscriber_count)

    return {
        "audit": audit,
        "journal": journal,
        "metrics": metrics,
        "notifications": notifications,
    }

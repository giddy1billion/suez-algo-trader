"""
Paper-Trading Dashboard — Real-time metrics for paper trading sessions.

Aggregates protective exit telemetry, order flow metrics, and position
health into a dashboard view suitable for display in Telegram or web UI.

Integrates:
- Protective exit configured/adjusted/executed counters
- Broker order acknowledgment tracking
- Position-level SL/TP status
- Daily summary generation
"""

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProtectiveExitStats:
    """Aggregated stats for protective exit telemetry."""

    configured_total: int = 0
    configured_by_source: dict = field(default_factory=dict)
    adjusted_total: int = 0
    adjusted_by_reason: dict = field(default_factory=dict)
    executed_stop_loss: int = 0
    executed_take_profit: int = 0
    executed_total: int = 0
    broker_acknowledged: int = 0
    broker_rejected: int = 0


@dataclass
class OrderFlowEntry:
    """Single order flow entry for tracking."""

    timestamp: datetime
    symbol: str
    side: str
    order_type: str  # "bracket" | "market" | "stop_limit"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: str = "submitted"  # "submitted" | "acknowledged" | "filled" | "rejected"
    broker_response_ms: float = 0.0


class PaperTradingDashboard:
    """
    Dashboard for paper-trading sessions with protective exit metrics.

    Usage:
        dashboard = PaperTradingDashboard()

        # Record events as they occur
        dashboard.record_exit_configured("AAPL", "buy", 150.0, 145.5, 159.0, "strategy")
        dashboard.record_exit_adjusted("AAPL", "stop_loss", 143.0, 145.5, "clamped_min")
        dashboard.record_exit_executed("AAPL", "stop_loss", 145.5, 145.3, 10, -47.0)
        dashboard.record_order_submitted("AAPL", "buy", "bracket", sl=145.5, tp=159.0)
        dashboard.record_order_acknowledged("AAPL", response_ms=12.5)

        # Get dashboard data
        data = dashboard.get_dashboard_data()
        text = dashboard.get_dashboard_text()
    """

    def __init__(self, max_history: int = 500):
        self._lock = threading.Lock()
        self._stats = ProtectiveExitStats()
        self._order_flow: deque[OrderFlowEntry] = deque(maxlen=max_history)
        self._exit_events: deque[dict] = deque(maxlen=max_history)
        self._start_time = datetime.now(timezone.utc)

    # --- Recording methods ---

    def record_exit_configured(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        source: str,
        trade_source: str = "signal",
    ) -> None:
        """Record that protective exits were configured for a position."""
        with self._lock:
            self._stats.configured_total += 1
            self._stats.configured_by_source[source] = (
                self._stats.configured_by_source.get(source, 0) + 1
            )
            self._exit_events.append({
                "type": "configured",
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "source": source,
                "trade_source": trade_source,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    def record_exit_adjusted(
        self,
        symbol: str,
        field_name: str,
        original: float,
        adjusted: float,
        reason: str,
    ) -> None:
        """Record that a protective exit level was adjusted."""
        with self._lock:
            self._stats.adjusted_total += 1
            self._stats.adjusted_by_reason[reason] = (
                self._stats.adjusted_by_reason.get(reason, 0) + 1
            )
            self._exit_events.append({
                "type": "adjusted",
                "symbol": symbol,
                "field": field_name,
                "original": original,
                "adjusted": adjusted,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    def record_exit_executed(
        self,
        symbol: str,
        exit_type: str,
        trigger_price: float,
        fill_price: float,
        qty: float,
        pnl: float,
        broker_acknowledged: bool = True,
    ) -> None:
        """Record that a protective exit was executed."""
        with self._lock:
            self._stats.executed_total += 1
            if exit_type == "stop_loss":
                self._stats.executed_stop_loss += 1
            elif exit_type == "take_profit":
                self._stats.executed_take_profit += 1

            if broker_acknowledged:
                self._stats.broker_acknowledged += 1
            else:
                self._stats.broker_rejected += 1

            self._exit_events.append({
                "type": "executed",
                "symbol": symbol,
                "exit_type": exit_type,
                "trigger_price": trigger_price,
                "fill_price": fill_price,
                "qty": qty,
                "pnl": pnl,
                "broker_acknowledged": broker_acknowledged,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    def record_order_submitted(
        self,
        symbol: str,
        side: str,
        order_type: str,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> None:
        """Record a new order submission."""
        with self._lock:
            self._order_flow.append(OrderFlowEntry(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                side=side,
                order_type=order_type,
                stop_loss=sl,
                take_profit=tp,
                status="submitted",
            ))

    def record_order_acknowledged(
        self, symbol: str, response_ms: float = 0.0
    ) -> None:
        """Record broker acknowledgment for an order."""
        with self._lock:
            # Update the most recent order for this symbol
            for entry in reversed(self._order_flow):
                if entry.symbol == symbol and entry.status == "submitted":
                    entry.status = "acknowledged"
                    entry.broker_response_ms = response_ms
                    break

    # --- Dashboard output ---

    def get_dashboard_data(self) -> dict:
        """Get structured dashboard data for API/web consumption."""
        with self._lock:
            uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()

            # Recent orders summary
            recent_orders = []
            for entry in list(self._order_flow)[-10:]:
                recent_orders.append({
                    "symbol": entry.symbol,
                    "side": entry.side,
                    "order_type": entry.order_type,
                    "stop_loss": entry.stop_loss,
                    "take_profit": entry.take_profit,
                    "status": entry.status,
                    "response_ms": entry.broker_response_ms,
                    "timestamp": entry.timestamp.isoformat(),
                })

            return {
                "uptime_seconds": round(uptime, 1),
                "protective_exits": {
                    "configured_total": self._stats.configured_total,
                    "configured_by_source": dict(self._stats.configured_by_source),
                    "adjusted_total": self._stats.adjusted_total,
                    "adjusted_by_reason": dict(self._stats.adjusted_by_reason),
                    "executed_total": self._stats.executed_total,
                    "executed_stop_loss": self._stats.executed_stop_loss,
                    "executed_take_profit": self._stats.executed_take_profit,
                    "broker_acknowledged": self._stats.broker_acknowledged,
                    "broker_rejected": self._stats.broker_rejected,
                },
                "recent_orders": recent_orders,
                "recent_events": list(self._exit_events)[-10:],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def get_dashboard_text(self) -> str:
        """Get formatted text for Telegram display."""
        with self._lock:
            stats = self._stats
            uptime_hours = (
                datetime.now(timezone.utc) - self._start_time
            ).total_seconds() / 3600

            lines = [
                "<b>📊 Paper Trading Dashboard</b>",
                f"{'═' * 28}",
                f"⏱ Uptime: {uptime_hours:.1f}h",
                "",
                "<b>🛡 Protective Exits</b>",
                f"  Configured: {stats.configured_total}",
            ]

            if stats.configured_by_source:
                for src, count in stats.configured_by_source.items():
                    lines.append(f"    • {src}: {count}")

            lines.append(f"  Adjusted: {stats.adjusted_total}")
            if stats.adjusted_by_reason:
                for reason, count in stats.adjusted_by_reason.items():
                    lines.append(f"    • {reason}: {count}")

            lines.extend([
                f"  Executed: {stats.executed_total}",
                f"    • Stop-Loss: {stats.executed_stop_loss}",
                f"    • Take-Profit: {stats.executed_take_profit}",
                "",
                "<b>🔗 Broker Status</b>",
                f"  Acknowledged: {stats.broker_acknowledged}",
                f"  Rejected: {stats.broker_rejected}",
            ])

            # Recent executed exits
            executed_events = [
                e for e in self._exit_events if e.get("type") == "executed"
            ]
            if executed_events:
                lines.append("")
                lines.append("<b>Recent Exits:</b>")
                for ev in executed_events[-5:]:
                    emoji = "🔴" if ev["exit_type"] == "stop_loss" else "🟢"
                    lines.append(
                        f"  {emoji} {ev['symbol']} {ev['exit_type'].upper()} "
                        f"@ ${ev['fill_price']:.2f} (P&L: ${ev['pnl']:+.2f})"
                    )

            return "\n".join(lines)

    # --- Daily Report ---

    def get_daily_report(self, date: Optional[datetime] = None) -> dict:
        """Generate daily report data for protective exits."""
        target_date = (date or datetime.now(timezone.utc)).strftime("%Y-%m-%d")

        with self._lock:
            # Filter events for the target date
            day_events = [
                e for e in self._exit_events
                if e.get("timestamp", "")[:10] == target_date
            ]

            configured = [e for e in day_events if e["type"] == "configured"]
            adjusted = [e for e in day_events if e["type"] == "adjusted"]
            executed = [e for e in day_events if e["type"] == "executed"]

            sl_executed = [e for e in executed if e.get("exit_type") == "stop_loss"]
            tp_executed = [e for e in executed if e.get("exit_type") == "take_profit"]

            total_pnl = sum(e.get("pnl", 0.0) for e in executed)
            sl_pnl = sum(e.get("pnl", 0.0) for e in sl_executed)
            tp_pnl = sum(e.get("pnl", 0.0) for e in tp_executed)

            return {
                "date": target_date,
                "protective_exits": {
                    "configured": len(configured),
                    "adjusted": len(adjusted),
                    "executed_total": len(executed),
                    "executed_stop_loss": len(sl_executed),
                    "executed_take_profit": len(tp_executed),
                    "total_pnl": round(total_pnl, 2),
                    "sl_pnl": round(sl_pnl, 2),
                    "tp_pnl": round(tp_pnl, 2),
                },
                "adjustments_detail": [
                    {
                        "symbol": e.get("symbol"),
                        "field": e.get("field"),
                        "reason": e.get("reason"),
                    }
                    for e in adjusted
                ],
                "executions_detail": [
                    {
                        "symbol": e.get("symbol"),
                        "exit_type": e.get("exit_type"),
                        "fill_price": e.get("fill_price"),
                        "pnl": e.get("pnl"),
                        "broker_acknowledged": e.get("broker_acknowledged"),
                    }
                    for e in executed
                ],
            }

    def get_daily_report_text(self, date: Optional[datetime] = None) -> str:
        """Generate formatted daily report text."""
        report = self.get_daily_report(date)
        pe = report["protective_exits"]

        lines = [
            f"<b>📋 Daily Report — {report['date']}</b>",
            f"{'═' * 32}",
            "",
            "<b>🛡 Protective Exits Summary</b>",
            f"  Configured: {pe['configured']}",
            f"  Adjusted: {pe['adjusted']}",
            f"  Executed: {pe['executed_total']}",
            f"    • Stop-Loss: {pe['executed_stop_loss']} (P&L: ${pe['sl_pnl']:+.2f})",
            f"    • Take-Profit: {pe['executed_take_profit']} (P&L: ${pe['tp_pnl']:+.2f})",
            f"  Total Exit P&L: ${pe['total_pnl']:+.2f}",
        ]

        if report["adjustments_detail"]:
            lines.append("")
            lines.append("<b>⚙️ Adjustments:</b>")
            for adj in report["adjustments_detail"][:10]:
                lines.append(
                    f"  • {adj['symbol']} {adj['field']}: {adj['reason']}"
                )

        if report["executions_detail"]:
            lines.append("")
            lines.append("<b>📈 Executions:</b>")
            for ex in report["executions_detail"][:10]:
                emoji = "🔴" if ex["exit_type"] == "stop_loss" else "🟢"
                ack = "✓" if ex["broker_acknowledged"] else "✗"
                lines.append(
                    f"  {emoji} {ex['symbol']} @ ${ex['fill_price']:.2f} "
                    f"(${ex['pnl']:+.2f}) [{ack}]"
                )

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all dashboard state."""
        with self._lock:
            self._stats = ProtectiveExitStats()
            self._order_flow.clear()
            self._exit_events.clear()
            self._start_time = datetime.now(timezone.utc)

"""
Operational command implementations for Telegram bot integration.

Each format_* method returns a Markdown-formatted string (HTML parse mode)
ready to send via Telegram. Messages are kept under 4096 chars.
"""

import platform
import sys
import threading
from datetime import datetime, timezone

from src.utils.logger import get_logger

logger = get_logger(__name__)


class OpsCommandHandler:
    """Generates formatted responses for operational Telegram commands."""

    def __init__(
        self,
        health_monitor=None,
        event_bus=None,
        trade_manager=None,
        broker=None,
        reconciler=None,
        risk_manager=None,
        metrics=None,
    ):
        self._health = health_monitor
        self._event_bus = event_bus
        self._trade_manager = trade_manager
        self._broker = broker
        self._reconciler = reconciler
        self._risk_manager = risk_manager
        self._metrics = metrics

    # ──────────────────────────────────────────────────────────────────────
    # /health
    # ──────────────────────────────────────────────────────────────────────

    def format_health(self) -> str:
        """Format /health response."""
        if not self._health:
            return "⚠️ Health monitor not configured."

        try:
            report = self._health.get_full_report()
            overall = report.get("overall_status", "unknown")
            icon = {"healthy": "✅", "degraded": "⚠️", "down": "🔴"}.get(overall, "❓")
            uptime = self._format_uptime(report.get("uptime_seconds", 0))
            sys_metrics = report.get("system", {})

            lines = [
                f"<b>{icon} System Health</b>",
                f"{'═' * 28}",
                f"Overall: <code>{overall.upper()}</code>",
                f"Uptime:  <code>{uptime}</code>",
                "",
                "<b>System Resources:</b>",
                f"  Memory: <code>{sys_metrics.get('memory_mb', '?')} MB</code>",
                f"  CPU:    <code>{sys_metrics.get('cpu_percent', '?')}%</code>",
                f"  Disk:   <code>{sys_metrics.get('disk_percent', '?')}% used</code>",
                "",
                "<b>Components:</b>",
            ]

            components = report.get("components", {})
            for name, info in components.items():
                status = info.get("status", "unknown")
                s_icon = {"healthy": "🟢", "degraded": "🟡", "down": "🔴"}.get(status, "⚪")
                latency = info.get("latency_ms", 0)
                lat_str = f" ({latency:.0f}ms)" if latency > 0 else ""
                lines.append(f"  {s_icon} {name}{lat_str}")

            issues = report.get("issues", [])
            if issues:
                lines.append("")
                lines.append("<b>Issues:</b>")
                for issue in issues[:5]:
                    lines.append(f"  ⚠️ {issue}")

            return "\n".join(lines)
        except Exception as e:
            logger.warning("ops.health_format_error", error=str(e))
            return f"❌ Error generating health report: {e}"

    # ──────────────────────────────────────────────────────────────────────
    # /riskreport
    # ──────────────────────────────────────────────────────────────────────

    def format_risk_report(self) -> str:
        """Format /riskreport response."""
        if not self._risk_manager:
            return "⚠️ Risk manager not configured."

        try:
            # Try new-style RiskEngine.get_metrics()
            if hasattr(self._risk_manager, "get_metrics"):
                metrics = self._risk_manager.get_metrics()
                lines = [
                    "<b>🛡️ Risk Report</b>",
                    f"{'═' * 28}",
                    f"Gross Exposure:  <code>${getattr(metrics, 'gross_exposure', 0):>12,.2f}</code>",
                    f"Net Exposure:    <code>${getattr(metrics, 'net_exposure', 0):>12,.2f}</code>",
                    f"Portfolio VaR:   <code>${getattr(metrics, 'portfolio_var', 0):>12,.2f}</code>",
                    f"Drawdown:        <code>{getattr(metrics, 'drawdown', 0):>+12.2f}%</code>",
                    f"Daily P&L:       <code>${getattr(metrics, 'daily_pnl', 0):>+12,.2f}</code>",
                    f"Open Positions:  <code>{getattr(metrics, 'open_positions', 0):>12d}</code>",
                    f"Cash Ratio:      <code>{getattr(metrics, 'cash_ratio', 0):>12.1%}</code>",
                    f"Portfolio Heat:  <code>{getattr(metrics, 'portfolio_heat', 0):>12.2f}</code>",
                ]
            elif hasattr(self._risk_manager, "daily_stats"):
                # Legacy RiskManager
                ds = self._risk_manager.daily_stats
                lines = [
                    "<b>🛡️ Risk Report</b>",
                    f"{'═' * 28}",
                    f"Daily Return:    <code>{ds.daily_return_pct:>+12.2%}</code>",
                    f"Realized P&L:    <code>${ds.realized_pnl:>+12,.2f}</code>",
                    f"Unrealized P&L:  <code>${ds.unrealized_pnl:>+12,.2f}</code>",
                    f"Trades Today:    <code>{ds.trades_today:>12d}</code>",
                    f"Win Rate:        <code>{ds.win_rate:>12.1%}</code>",
                    f"Halted:          <code>{'YES' if ds.is_halted else 'NO':>12}</code>",
                ]
            else:
                return "⚠️ Risk manager has no recognized interface."

            if hasattr(self._risk_manager, "is_halted") and self._risk_manager.is_halted:
                lines.append(f"\n🚨 <b>HALTED:</b> {self._risk_manager.halt_reason}")

            return "\n".join(lines)
        except Exception as e:
            logger.warning("ops.risk_report_error", error=str(e))
            return f"❌ Error generating risk report: {e}"

    # ──────────────────────────────────────────────────────────────────────
    # /reconcile
    # ──────────────────────────────────────────────────────────────────────

    def format_reconciliation(self) -> str:
        """Format /reconcile response (triggers reconciliation)."""
        if not self._reconciler:
            return "⚠️ Reconciler not configured."

        try:
            if hasattr(self._reconciler, "reconcile"):
                result = self._reconciler.reconcile()
            elif hasattr(self._reconciler, "run"):
                result = self._reconciler.run()
            else:
                return "⚠️ Reconciler has no reconcile() or run() method."

            if isinstance(result, dict):
                matched = result.get("matched", 0)
                mismatched = result.get("mismatched", 0)
                missing = result.get("missing", 0)
                extra = result.get("extra", 0)
                lines = [
                    "<b>🔄 Reconciliation Results</b>",
                    f"{'═' * 28}",
                    f"Matched:    <code>{matched}</code>",
                    f"Mismatched: <code>{mismatched}</code>",
                    f"Missing:    <code>{missing}</code>",
                    f"Extra:      <code>{extra}</code>",
                ]
                if mismatched == 0 and missing == 0 and extra == 0:
                    lines.append("\n✅ All positions reconciled.")
                else:
                    lines.append("\n⚠️ Discrepancies found — review needed.")
                return "\n".join(lines)
            else:
                return f"🔄 Reconciliation complete:\n<code>{str(result)[:3000]}</code>"
        except Exception as e:
            logger.warning("ops.reconcile_error", error=str(e))
            return f"❌ Reconciliation error: {e}"

    # ──────────────────────────────────────────────────────────────────────
    # /events
    # ──────────────────────────────────────────────────────────────────────

    def format_events(self, n: int = 10) -> str:
        """Format /events response."""
        if not self._event_bus:
            return "⚠️ Event bus not configured."

        try:
            history = self._event_bus.get_history(limit=n)
            if not history:
                return "📡 No events recorded yet."

            lines = ["<b>📡 Recent Events</b>", f"{'═' * 28}"]
            for ev in reversed(history[-n:]):
                ts = ev.timestamp.strftime("%H:%M:%S") if hasattr(ev, "timestamp") else "?"
                etype = type(ev).__name__
                symbol = getattr(ev, "symbol", "")
                extra = f" {symbol}" if symbol else ""
                lines.append(f"<code>{ts}</code> {etype}{extra}")

            total = len(self._event_bus.get_history(limit=9999))
            lines.append(f"\nTotal events: {total}")
            if hasattr(self._event_bus, "subscriber_count"):
                lines.append(f"Subscribers: {self._event_bus.subscriber_count}")

            return "\n".join(lines)
        except Exception as e:
            logger.warning("ops.events_error", error=str(e))
            return f"❌ Events error: {e}"

    # ──────────────────────────────────────────────────────────────────────
    # /latency
    # ──────────────────────────────────────────────────────────────────────

    def format_latency(self) -> str:
        """Format /latency response."""
        lines = ["<b>⏱️ Latency Metrics</b>", f"{'═' * 28}"]

        # Broker latency
        if self._broker:
            try:
                import time as _time
                start = _time.perf_counter()
                if hasattr(self._broker, "get_account"):
                    self._broker.get_account()
                elif hasattr(self._broker, "account"):
                    _ = self._broker.account
                broker_ms = (_time.perf_counter() - start) * 1000
                lines.append(f"Broker API:  <code>{broker_ms:>8.1f} ms</code>")
            except Exception as e:
                lines.append(f"Broker API:  <code>ERROR ({e})</code>")
        else:
            lines.append("Broker API:  <code>N/A</code>")

        # Component latencies from health monitor
        if self._health:
            lines.append("")
            lines.append("<b>Component Latencies:</b>")
            report = self._health.get_full_report()
            components = report.get("components", {})
            for name, info in components.items():
                lat = info.get("latency_ms", 0)
                if lat > 0:
                    lines.append(f"  {name}: <code>{lat:.1f} ms</code>")

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    # /performance
    # ──────────────────────────────────────────────────────────────────────

    def format_performance(self) -> str:
        """Format /performance response."""
        if not self._metrics:
            return "⚠️ Live metrics not configured."

        try:
            m = self._metrics.get_metrics(period_days=30)
            lines = [
                "<b>📈 Live Performance (30d)</b>",
                f"{'═' * 28}",
                f"Sharpe Ratio:   <code>{m.get('sharpe_ratio', 0):>10.2f}</code>",
                f"Sortino Ratio:  <code>{m.get('sortino_ratio', 0):>10.2f}</code>",
                f"Win Rate:       <code>{m.get('win_rate', 0):>10.1%}</code>",
                f"Profit Factor:  <code>{m.get('profit_factor', 0):>10.2f}</code>",
                f"Expectancy:     <code>${m.get('expectancy', 0):>+9.2f}</code>",
                f"{'─' * 28}",
                f"Total Trades:   <code>{m.get('total_trades', 0):>10d}</code>",
                f"Avg Win:        <code>${m.get('average_win', 0):>+9.2f}</code>",
                f"Avg Loss:       <code>${m.get('average_loss', 0):>9.2f}</code>",
                f"Max Drawdown:   <code>{m.get('max_drawdown', 0):>+10.2f}%</code>",
                f"Cur Drawdown:   <code>{m.get('current_drawdown', 0):>+10.2f}%</code>",
                f"Consec Wins:    <code>{m.get('max_consecutive_wins', 0):>10d}</code>",
                f"Consec Losses:  <code>{m.get('max_consecutive_losses', 0):>10d}</code>",
            ]
            return "\n".join(lines)
        except Exception as e:
            logger.warning("ops.performance_error", error=str(e))
            return f"❌ Performance error: {e}"

    # ──────────────────────────────────────────────────────────────────────
    # /system
    # ──────────────────────────────────────────────────────────────────────

    def format_system(self) -> str:
        """Format /system response."""
        try:
            import psutil

            proc = psutil.Process()
            mem = proc.memory_info()
            uptime = self._health.uptime_seconds if self._health else 0

            lines = [
                "<b>🖥️ System Info</b>",
                f"{'═' * 28}",
                f"Python:    <code>{sys.version.split()[0]}</code>",
                f"Platform:  <code>{platform.system()} {platform.release()}</code>",
                f"Uptime:    <code>{self._format_uptime(uptime)}</code>",
                f"PID:       <code>{proc.pid}</code>",
                f"Threads:   <code>{threading.active_count()}</code>",
                f"{'─' * 28}",
                f"RSS Mem:   <code>{mem.rss / (1024*1024):.1f} MB</code>",
                f"CPU:       <code>{psutil.cpu_percent(interval=None):.1f}%</code>",
            ]

            disk = psutil.disk_usage("C:\\" if platform.system() == "Windows" else "/")
            lines.append(f"Disk Free: <code>{disk.free / (1024**3):.1f} GB</code>")
            lines.append(f"Disk Used: <code>{disk.percent:.1f}%</code>")

            return "\n".join(lines)
        except Exception as e:
            logger.warning("ops.system_error", error=str(e))
            return f"❌ System info error: {e}"

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """Format seconds into human-readable uptime string."""
        seconds = int(seconds)
        days, rem = divmod(seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, secs = divmod(rem, 60)
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    # ──────────────────────────────────────────────────────────────────────
    # /strategies
    # ──────────────────────────────────────────────────────────────────────

    def format_strategies(self, orchestrator) -> str:
        """Format /strategies response showing all strategy slots and stats."""
        if orchestrator is None:
            return "⚠️ Strategy orchestrator not configured."

        if len(orchestrator) == 0:
            return "⚠️ No strategies registered."

        all_stats = orchestrator.get_all_stats()
        strategies = all_stats.get("strategies", {})
        lines = [
            "<b>📊 Strategy Orchestrator</b>",
            f"{'═' * 28}",
            f"Active: <code>{all_stats['active_strategies']}/{all_stats['total_strategies']}</code>",
            f"Total Cycles: <code>{all_stats['total_cycles']}</code>",
            f"Total P&amp;L: <code>${all_stats['total_pnl']:.2f}</code>",
            "",
        ]

        for name, s in strategies.items():
            active = s.get("enabled", False)
            icon = "🟢" if active else "🔴"
            cycles = s.get("cycle_count", 0)
            signals = s.get("total_signals", 0)
            trades = s.get("total_trades", 0)
            pnl = s.get("realized_pnl", 0)
            win_rate = s.get("win_rate", 0)
            weight = s.get("weight", 1.0)
            symbols = s.get("symbols", [])
            timeframe = s.get("timeframe", "?")
            interval = s.get("interval", 0)

            lines.append(f"{icon} <b>{name}</b> (w={weight:.1f})")
            lines.append(f"   Symbols: <code>{', '.join(symbols[:5])}</code>")
            lines.append(f"   TF: <code>{timeframe}</code> | Interval: <code>{interval}s</code>")
            lines.append(f"   Cycles: <code>{cycles}</code> | Signals: <code>{signals}</code> | Trades: <code>{trades}</code>")
            lines.append(f"   P&amp;L: <code>${pnl:.2f}</code> | Win: <code>{win_rate}%</code>")
            lines.append("")

        return "\n".join(lines)

"""
Live Trading Metrics — calculates and tracks real-time trading performance.

Thread-safe, designed for continuous use during live trading.
"""

import math
import threading
from collections import deque
from datetime import datetime, timezone, timedelta

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LiveMetrics:
    """Calculates and tracks real-time trading performance metrics."""

    def __init__(self, max_trades: int = 5000, max_equity_points: int = 10000, max_daily_records: int = 365):
        self._trades: list[dict] = []
        self._equity_curve: list[tuple[datetime, float]] = []
        self._daily_pnl: dict[str, float] = {}  # date_str -> pnl
        self._lock = threading.Lock()
        self._max_trades = max_trades
        self._max_equity_points = max_equity_points
        self._max_daily_records = max_daily_records

    def record_trade(self, trade: dict):
        """Record a completed trade.

        Expected trade dict keys:
            - pnl: float (profit/loss in $)
            - entry_time: datetime
            - exit_time: datetime
            - side: str ("long" or "short")
            - symbol: str
            - quantity: float (optional)
        """
        with self._lock:
            self._trades.append(trade)
            if len(self._trades) > self._max_trades:
                self._trades = self._trades[-self._max_trades:]

            # Update daily PnL
            exit_time = trade.get("exit_time", datetime.now(timezone.utc))
            if isinstance(exit_time, datetime):
                date_key = exit_time.strftime("%Y-%m-%d")
            else:
                date_key = str(exit_time)[:10]

            self._daily_pnl[date_key] = self._daily_pnl.get(date_key, 0.0) + trade.get("pnl", 0.0)

            # Trim old daily records
            while len(self._daily_pnl) > self._max_daily_records:
                oldest = next(iter(self._daily_pnl))
                del self._daily_pnl[oldest]

    def record_equity(self, equity: float):
        """Record current equity value for drawdown calculations."""
        with self._lock:
            self._equity_curve.append((datetime.now(timezone.utc), equity))
            if len(self._equity_curve) > self._max_equity_points:
                self._equity_curve = self._equity_curve[-self._max_equity_points:]

    def get_metrics(self, period_days: int = 30) -> dict:
        """Returns comprehensive metrics dict for the specified period."""
        with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)

            # Filter trades for period
            trades = [
                t for t in self._trades
                if t.get("exit_time", datetime.now(timezone.utc)) >= cutoff
            ]

            if not trades:
                return self._empty_metrics()

            pnls = [t.get("pnl", 0.0) for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]

            total_trades = len(trades)
            win_rate = len(wins) / total_trades if total_trades > 0 else 0.0

            average_win = sum(wins) / len(wins) if wins else 0.0
            average_loss = abs(sum(losses) / len(losses)) if losses else 0.0
            win_loss_ratio = average_win / average_loss if average_loss > 0 else float("inf")

            # Profit factor
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            # Expectancy
            expectancy = sum(pnls) / total_trades if total_trades > 0 else 0.0

            # Daily returns for ratio calculations
            daily_returns = self._get_daily_returns(cutoff)

            # Sharpe ratio (annualized)
            sharpe_ratio = self._calc_sharpe(daily_returns)

            # Sortino ratio
            sortino_ratio = self._calc_sortino(daily_returns)

            # Max drawdown and current drawdown
            max_dd, current_dd = self._calc_drawdowns()

            # Calmar ratio (annualized return / max drawdown)
            calmar_ratio = self._calc_calmar(daily_returns, max_dd)

            # Consecutive wins/losses
            max_consec_wins, max_consec_losses = self._calc_consecutive(pnls)

            # Kelly fraction
            kelly = self._calc_kelly(win_rate, win_loss_ratio)

            # Holding period
            avg_holding = self._calc_avg_holding(trades)

            # Exposure
            exposure_pct = self._calc_exposure(trades, period_days)

            # Turnover (annualized trades)
            turnover = total_trades * (365.0 / max(period_days, 1))

            # Recent daily PnL (last 5 days)
            recent_pnl = self._get_recent_daily_pnl(5)

            # VaR and CVaR
            var_95, cvar_95 = self._calc_var(daily_returns)

            return {
                "sharpe_ratio": round(sharpe_ratio, 2),
                "sortino_ratio": round(sortino_ratio, 2),
                "calmar_ratio": round(calmar_ratio, 2),
                "max_drawdown": round(max_dd, 2),
                "current_drawdown": round(current_dd, 2),
                "win_rate": round(win_rate, 4),
                "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.99,
                "expectancy": round(expectancy, 2),
                "average_win": round(average_win, 2),
                "average_loss": round(average_loss, 2),
                "win_loss_ratio": round(win_loss_ratio, 2) if win_loss_ratio != float("inf") else 999.99,
                "total_trades": total_trades,
                "avg_holding_period": avg_holding,
                "max_consecutive_wins": max_consec_wins,
                "max_consecutive_losses": max_consec_losses,
                "kelly_fraction": round(kelly, 4),
                "exposure_pct": round(exposure_pct, 2),
                "turnover": round(turnover, 1),
                "daily_pnl": recent_pnl,
                "var_95": round(var_95, 2),
                "cvar_95": round(cvar_95, 2),
            }

    def get_summary_text(self, period_days: int = 30) -> str:
        """Formatted text for Telegram /metrics command."""
        m = self.get_metrics(period_days)

        if m["total_trades"] == 0:
            return (
                f"<b>Trading Metrics ({period_days}d)</b>\n"
                f"{'═' * 23}\n"
                f"No trades recorded in this period."
            )

        text = (
            f"<b>Trading Metrics ({period_days}d)</b>\n"
            f"{'═' * 23}\n"
            f"Sharpe:    {m['sharpe_ratio']:.2f}\n"
            f"Sortino:   {m['sortino_ratio']:.2f}\n"
            f"Calmar:    {m['calmar_ratio']:.2f}\n"
            f"Max DD:    {m['max_drawdown']:.1f}%\n"
            f"Win Rate:  {m['win_rate'] * 100:.1f}%\n"
            f"PF:        {m['profit_factor']:.2f}\n"
            f"Expectancy: ${m['expectancy']:.2f}\n"
            f"Kelly:     {m['kelly_fraction'] * 100:.1f}%\n"
            f"VaR 95%:   -${abs(m['var_95']):.0f}\n"
            f"Trades:    {m['total_trades']}\n"
            f"Exposure:  {m['exposure_pct']:.0f}%\n"
        )

        # Add recent daily PnL
        if m["daily_pnl"]:
            text += f"\n<b>Recent P&L:</b>\n"
            for date_str, pnl in m["daily_pnl"].items():
                emoji = "🟢" if pnl >= 0 else "🔴"
                text += f"  {emoji} {date_str}: ${pnl:+.2f}\n"

        return text

    def _empty_metrics(self) -> dict:
        """Return empty metrics dict."""
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "win_loss_ratio": 0.0,
            "total_trades": 0,
            "avg_holding_period": "0m",
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "kelly_fraction": 0.0,
            "exposure_pct": 0.0,
            "turnover": 0.0,
            "daily_pnl": {},
            "var_95": 0.0,
            "cvar_95": 0.0,
        }

    def _get_daily_returns(self, cutoff: datetime) -> list[float]:
        """Get daily returns from equity curve."""
        points = [(t, e) for t, e in self._equity_curve if t >= cutoff]
        if len(points) < 2:
            # Fall back to daily PnL-based returns
            cutoff_str = cutoff.strftime("%Y-%m-%d")
            sorted_days = sorted(k for k in self._daily_pnl.keys() if k >= cutoff_str)
            if sorted_days and self._equity_curve:
                # Approximate returns from PnL / last known equity
                base_equity = self._equity_curve[-1][1] if self._equity_curve else 10000.0
                return [self._daily_pnl[d] / base_equity for d in sorted_days]
            return []

        # Group by date and compute daily returns
        daily_eq = {}
        for t, e in points:
            day = t.strftime("%Y-%m-%d")
            daily_eq[day] = e  # Last equity of the day

        days = sorted(daily_eq.keys())
        returns = []
        for i in range(1, len(days)):
            prev = daily_eq[days[i - 1]]
            curr = daily_eq[days[i]]
            if prev > 0:
                returns.append((curr - prev) / prev)

        return returns

    def _calc_sharpe(self, daily_returns: list[float]) -> float:
        """Annualized Sharpe ratio (risk-free rate = 0)."""
        if len(daily_returns) < 2:
            return 0.0
        mean_r = sum(daily_returns) / len(daily_returns)
        std_r = _std(daily_returns)
        if std_r == 0:
            return 0.0
        return (mean_r / std_r) * math.sqrt(252)

    def _calc_sortino(self, daily_returns: list[float]) -> float:
        """Annualized Sortino ratio (downside deviation)."""
        if len(daily_returns) < 2:
            return 0.0
        mean_r = sum(daily_returns) / len(daily_returns)
        downside = [r for r in daily_returns if r < 0]
        if not downside:
            # No downside returns: Sortino is theoretically infinite; cap for display safety
            return 0.0 if mean_r <= 0 else 99.99
        downside_std = _std(downside)
        if downside_std == 0:
            return 0.0
        return (mean_r / downside_std) * math.sqrt(252)

    def _calc_calmar(self, daily_returns: list[float], max_dd: float) -> float:
        """Calmar ratio: annualized return / max drawdown."""
        if not daily_returns or max_dd == 0:
            return 0.0
        mean_daily_return = sum(daily_returns) / len(daily_returns)
        annualized_return = ((1 + mean_daily_return) ** 252) - 1
        return annualized_return / (abs(max_dd) / 100.0) if max_dd != 0 else 0.0

    def _calc_drawdowns(self) -> tuple[float, float]:
        """Calculate max drawdown (%) and current drawdown (%)."""
        if not self._equity_curve:
            return 0.0, 0.0

        equities = [e for _, e in self._equity_curve]
        peak = equities[0]
        max_dd = 0.0

        for eq in equities:
            if eq > peak:
                peak = eq
            dd = ((peak - eq) / peak) * 100 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        # Current drawdown
        current_peak = max(equities)
        current_eq = equities[-1]
        current_dd = ((current_peak - current_eq) / current_peak) * 100 if current_peak > 0 else 0.0

        return max_dd, current_dd

    def _calc_consecutive(self, pnls: list[float]) -> tuple[int, int]:
        """Max consecutive wins and losses."""
        max_wins = 0
        max_losses = 0
        curr_wins = 0
        curr_losses = 0

        for pnl in pnls:
            if pnl > 0:
                curr_wins += 1
                curr_losses = 0
                max_wins = max(max_wins, curr_wins)
            elif pnl < 0:
                curr_losses += 1
                curr_wins = 0
                max_losses = max(max_losses, curr_losses)
            else:
                curr_wins = 0
                curr_losses = 0

        return max_wins, max_losses

    def _calc_kelly(self, win_rate: float, win_loss_ratio: float) -> float:
        """Kelly criterion fraction."""
        if win_loss_ratio <= 0 or win_loss_ratio == float("inf"):
            return 0.0
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        return max(0.0, min(kelly, 1.0))

    def _calc_avg_holding(self, trades: list[dict]) -> str:
        """Average holding period as human-readable string."""
        durations = []
        for t in trades:
            entry = t.get("entry_time")
            exit_t = t.get("exit_time")
            if entry and exit_t and isinstance(entry, datetime) and isinstance(exit_t, datetime):
                durations.append((exit_t - entry).total_seconds())

        if not durations:
            return "N/A"

        avg_secs = sum(durations) / len(durations)
        if avg_secs < 60:
            return f"{avg_secs:.0f}s"
        elif avg_secs < 3600:
            return f"{avg_secs / 60:.0f}m"
        elif avg_secs < 86400:
            return f"{avg_secs / 3600:.1f}h"
        else:
            return f"{avg_secs / 86400:.1f}d"

    def _calc_exposure(self, trades: list[dict], period_days: int) -> float:
        """Percentage of time in market."""
        total_seconds = period_days * 86400
        if total_seconds == 0:
            return 0.0

        in_market_seconds = 0.0
        for t in trades:
            entry = t.get("entry_time")
            exit_t = t.get("exit_time")
            if entry and exit_t and isinstance(entry, datetime) and isinstance(exit_t, datetime):
                in_market_seconds += (exit_t - entry).total_seconds()

        return min((in_market_seconds / total_seconds) * 100, 100.0)

    def _get_recent_daily_pnl(self, days: int) -> dict[str, float]:
        """Get last N days of PnL."""
        sorted_dates = sorted(self._daily_pnl.keys(), reverse=True)[:days]
        return {d: round(self._daily_pnl[d], 2) for d in sorted(sorted_dates)}

    def _calc_var(self, daily_returns: list[float]) -> tuple[float, float]:
        """1-day parametric VaR and CVaR at 95% confidence."""
        if len(daily_returns) < 5:
            return 0.0, 0.0

        # Convert returns to dollar PnL using last equity
        last_equity = self._equity_curve[-1][1] if self._equity_curve else 10000.0

        mean_r = sum(daily_returns) / len(daily_returns)
        std_r = _std(daily_returns)

        if std_r == 0:
            return 0.0, 0.0

        # Parametric VaR (normal distribution assumption)
        # z-score for 95% = 1.645
        var_95_pct = mean_r - 1.645 * std_r
        var_95_dollar = var_95_pct * last_equity

        # CVaR (expected shortfall) - average of returns below VaR
        sorted_returns = sorted(daily_returns)
        cutoff_idx = max(1, int(len(sorted_returns) * 0.05))
        tail_returns = sorted_returns[:cutoff_idx]
        cvar_pct = sum(tail_returns) / len(tail_returns) if tail_returns else var_95_pct
        cvar_95_dollar = cvar_pct * last_equity

        return var_95_dollar, cvar_95_dollar


def _std(values: list[float]) -> float:
    """Sample standard deviation (Bessel-corrected, divides by n-1)."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)

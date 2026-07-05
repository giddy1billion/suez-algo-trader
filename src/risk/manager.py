"""
Risk Manager — Legacy compatibility shim.

This module is preserved for backward compatibility. New code should use
the multi-layer RiskEngine (src.risk.engine) directly.

All trading decisions pass through here before execution.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timezone, timedelta
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _market_date_et() -> date:
    """Get current date in US/Eastern (market timezone)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("US/Eastern")).date()
    except ImportError:
        # Fallback: approximate ET as UTC-5
        return (datetime.now(timezone.utc) - timedelta(hours=5)).date()


def _week_start_et() -> date:
    """Monday of the current week, in US/Eastern."""
    d = _market_date_et()
    return d - timedelta(days=d.weekday())


@dataclass
class RiskLimits:
    """Configuration for risk parameters."""
    max_position_size_pct: float = 0.02      # Max 2% of portfolio per trade
    max_daily_loss_pct: float = 0.05         # Stop trading if down 5% today
    max_portfolio_exposure: float = 0.80     # Max 80% invested at once
    max_single_stock_pct: float = 0.15       # No single position > 15% of portfolio
    max_leverage: float = 1.0                # No leverage by default
    max_open_positions: int = 20             # Max concurrent positions
    max_orders_per_day: int = 100            # Circuit breaker
    default_stop_loss_pct: float = 0.03      # 3% stop loss
    default_take_profit_pct: float = 0.06    # 6% take profit (2:1 ratio)
    max_correlated_positions: int = 3        # Max positions in same sector


@dataclass
class DailyStats:
    """Track daily P&L and trade counts."""
    date: date = field(default_factory=_market_date_et)
    starting_equity: float = 0.0
    current_equity: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades_today: int = 0
    wins: int = 0
    losses: int = 0
    is_halted: bool = False
    halt_reason: str = ""
    week_start: date = field(default_factory=_week_start_et)
    weekly_realized_pnl: float = 0.0

    @property
    def daily_return_pct(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        return (self.current_equity - self.starting_equity) / self.starting_equity

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0


class RiskManager:
    """
    Gatekeeper for all trades. Evaluates whether a proposed trade
    is within acceptable risk parameters before allowing execution.
    """

    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        self.daily_stats = DailyStats()
        self._trade_log: list[dict] = []

    def reset_daily(self, starting_equity: float):
        """Reset daily tracking (call at market open). Weekly P&L persists
        across days and only resets when a new ISO week begins."""
        prev_week_start = self.daily_stats.week_start
        prev_weekly_pnl = self.daily_stats.weekly_realized_pnl

        current_week_start = _week_start_et()

        # Roll over weekly P&L only if we've entered a new week
        carried_weekly_pnl = 0.0 if current_week_start != prev_week_start else prev_weekly_pnl

        self.daily_stats = DailyStats(
            date=_market_date_et(),
            starting_equity=starting_equity,
            current_equity=starting_equity,
            week_start=current_week_start,
            weekly_realized_pnl=carried_weekly_pnl,
        )
        logger.info("risk.daily_reset", equity=starting_equity, weekly_pnl=carried_weekly_pnl)

    def update_equity(self, current_equity: float):
        """Update current equity for drawdown tracking."""
        self.daily_stats.current_equity = current_equity

    @property
    def weekly_pnl(self) -> float:
        """Accumulated realized P&L for the current ISO week."""
        return self.daily_stats.weekly_realized_pnl

    # ──────────────────────────────────────────────────────────────────────
    # Pre-Trade Checks
    # ──────────────────────────────────────────────────────────────────────

    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed right now (daily loss limit, etc.)."""
        if self.daily_stats.is_halted:
            return False, f"Trading halted: {self.daily_stats.halt_reason}"

        # Daily loss limit (only halts on losses, not profits)
        if self.daily_stats.daily_return_pct <= -self.limits.max_daily_loss_pct:
            self.daily_stats.is_halted = True
            self.daily_stats.halt_reason = f"Daily loss limit hit: {self.daily_stats.daily_return_pct:.2%}"
            logger.warning("risk.halted", reason=self.daily_stats.halt_reason)
            return False, self.daily_stats.halt_reason

        # Max orders per day
        if self.daily_stats.trades_today >= self.limits.max_orders_per_day:
            return False, f"Max daily orders reached: {self.limits.max_orders_per_day}"

        return True, "OK"

    def evaluate_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        portfolio_value: float,
        current_positions: list[dict],
    ) -> tuple[bool, str, dict]:
        """
        Evaluate whether a proposed trade is within risk limits.
        Returns (approved, reason, adjusted_params).
        """
        can, reason = self.can_trade()
        if not can:
            return False, reason, {}

        trade_value = qty * price
        adjustments = {}

        # 1. Position size check
        position_pct = trade_value / portfolio_value if portfolio_value > 0 else 1.0
        if position_pct > self.limits.max_single_stock_pct:
            max_qty = (self.limits.max_single_stock_pct * portfolio_value) / price
            adjustments["qty_reduced"] = {"original": qty, "adjusted": max_qty}
            qty = max_qty
            trade_value = qty * price
            logger.info("risk.qty_reduced", symbol=symbol, original_pct=position_pct,
                       max_pct=self.limits.max_single_stock_pct)

        # 2. Max risk per trade
        risk_amount = trade_value * self.limits.default_stop_loss_pct
        max_risk = portfolio_value * self.limits.max_position_size_pct
        if risk_amount > max_risk:
            max_qty = (max_risk / self.limits.default_stop_loss_pct) / price
            adjustments["risk_reduced"] = {"original_qty": qty, "max_qty": max_qty}
            qty = min(qty, max_qty)
            trade_value = qty * price

        # 3. Portfolio exposure check
        total_exposure = sum(abs(float(p.get("market_value", 0))) for p in current_positions)
        if (total_exposure + trade_value) / portfolio_value > self.limits.max_portfolio_exposure:
            remaining = (self.limits.max_portfolio_exposure * portfolio_value) - total_exposure
            if remaining <= 0:
                return False, "Portfolio exposure limit reached", {}
            max_qty = remaining / price
            adjustments["exposure_reduced"] = {"max_qty": max_qty}
            qty = min(qty, max_qty)

        # 4. Max open positions
        if side.lower() == "buy" and len(current_positions) >= self.limits.max_open_positions:
            return False, f"Max open positions ({self.limits.max_open_positions}) reached", {}

        # 5. Leverage check
        if portfolio_value > 0:
            effective_leverage = (total_exposure + trade_value) / portfolio_value
            if effective_leverage > self.limits.max_leverage:
                return False, f"Leverage limit ({self.limits.max_leverage}x) would be exceeded", {}

        if qty <= 0:
            return False, "Adjusted quantity is zero or negative", {}

        return True, "Approved", {
            "adjusted_qty": qty,
            "trade_value": qty * price,
            "risk_pct": (qty * price * self.limits.default_stop_loss_pct) / portfolio_value,
            "stop_loss": price * (1 - self.limits.default_stop_loss_pct) if side == "buy"
                        else price * (1 + self.limits.default_stop_loss_pct),
            "take_profit": price * (1 + self.limits.default_take_profit_pct) if side == "buy"
                          else price * (1 - self.limits.default_take_profit_pct),
            **adjustments,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Position Sizing
    # ──────────────────────────────────────────────────────────────────────

    def calculate_position_size(
        self, price: float, stop_loss_price: float,
        portfolio_value: float, risk_pct: float = None
    ) -> float:
        """
        Calculate optimal position size using fixed-risk model.
        Risk = (entry - stop_loss) * qty <= risk_pct * portfolio
        """
        risk_pct = risk_pct or self.limits.max_position_size_pct
        risk_per_share = abs(price - stop_loss_price)

        if risk_per_share <= 0:
            return 0.0

        max_risk_amount = portfolio_value * risk_pct
        qty = max_risk_amount / risk_per_share

        # Cap at max single stock allocation
        max_by_allocation = (self.limits.max_single_stock_pct * portfolio_value) / price
        qty = min(qty, max_by_allocation)

        return round(qty, 6)  # Support fractional shares

    # ──────────────────────────────────────────────────────────────────────
    # Trade Recording
    # ──────────────────────────────────────────────────────────────────────

    def record_trade(self, trade: dict):
        """Record a completed trade for daily and weekly stats."""
        self.daily_stats.trades_today += 1
        pnl = trade.get("pnl", 0)
        self.daily_stats.realized_pnl += pnl
        self.daily_stats.weekly_realized_pnl += pnl
        if pnl > 0:
            self.daily_stats.wins += 1
        elif pnl < 0:
            self.daily_stats.losses += 1
        self._trade_log.append({**trade, "timestamp": datetime.now().isoformat()})

    def get_daily_summary(self) -> dict:
        """Get today's risk/performance summary."""
        return {
            "date": str(self.daily_stats.date),
            "trades": self.daily_stats.trades_today,
            "win_rate": f"{self.daily_stats.win_rate:.1%}",
            "daily_pnl": self.daily_stats.realized_pnl,
            "daily_return": f"{self.daily_stats.daily_return_pct:.2%}",
            "is_halted": self.daily_stats.is_halted,
            "positions_allowed": self.limits.max_open_positions,
        }

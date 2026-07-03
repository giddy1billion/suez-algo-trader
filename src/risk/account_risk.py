"""
Account-Level Risk Layer — Enforces account-wide safety limits.

Checks:
- Daily loss limit
- Weekly loss limit
- Max drawdown from peak (auto-halt)
- Minimum cash reserve
- Pattern Day Trader (PDT) rule
- Consecutive loss limit
- Daily trade count limit
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, date, timedelta
from typing import Optional

from src.risk.models import TradeRequest, LayerDecision, RiskAction
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AccountRiskLayer:
    """
    Evaluates whether account-level constraints permit a new trade.
    Tracks daily/weekly P&L, drawdown, and trade frequency.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.03,
        max_weekly_loss_pct: float = 0.07,
        max_drawdown_pct: float = 0.15,
        min_cash_reserve_pct: float = 0.20,
        max_day_trades_5d: int = 3,
        pdt_account_threshold: float = 25_000.0,
        consecutive_loss_limit: int = 5,
        daily_trade_limit: int = 20,
        enabled: bool = True,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.min_cash_reserve_pct = min_cash_reserve_pct
        self.max_day_trades_5d = max_day_trades_5d
        self.pdt_account_threshold = pdt_account_threshold
        self.consecutive_loss_limit = consecutive_loss_limit
        self.daily_trade_limit = daily_trade_limit
        self.enabled = enabled

        # Internal state
        self._lock = threading.Lock()
        self._peak_equity: float = 0.0
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._daily_date: date = date.today()
        self._week_start: date = date.today() - timedelta(days=date.today().weekday())
        self._trades_today: int = 0
        self._consecutive_losses: int = 0
        self._day_trades: deque = deque(maxlen=100)  # (date, was_day_trade) pairs
        self._is_halted: bool = False
        self._halt_reason: str = ""

    # ──────────────────────────────────────────────────────────────────────
    # State Management
    # ──────────────────────────────────────────────────────────────────────

    def update_state(
        self,
        current_equity: float,
        daily_pnl: float,
        weekly_pnl: float,
        cash: float,
    ) -> None:
        """Update account state with latest figures. Call before evaluate()."""
        with self._lock:
            today = date.today()
            if today != self._daily_date:
                self._daily_date = today
                self._trades_today = 0
                self._daily_pnl = 0.0

            week_start = today - timedelta(days=today.weekday())
            if week_start != self._week_start:
                self._week_start = week_start
                self._weekly_pnl = 0.0

            self._daily_pnl = daily_pnl
            self._weekly_pnl = weekly_pnl
            self._peak_equity = max(self._peak_equity, current_equity)
            self._current_equity = current_equity
            self._cash = cash

    def record_trade_result(self, pnl: float, was_day_trade: bool = False) -> None:
        """Record a completed trade result for consecutive loss tracking."""
        with self._lock:
            self._trades_today += 1
            if pnl < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

            if was_day_trade:
                self._day_trades.append(date.today())

    def reset_halt(self) -> None:
        """Manually reset a halt condition (e.g., next trading day)."""
        with self._lock:
            self._is_halted = False
            self._halt_reason = ""
            logger.info("account_risk.halt_reset")

    @property
    def is_halted(self) -> bool:
        return self._is_halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    # ──────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        request: TradeRequest,
        portfolio_value: float,
        cash: float,
        account_value: Optional[float] = None,
    ) -> LayerDecision:
        """
        Evaluate trade against account-level risk constraints.

        Args:
            request: The proposed trade.
            portfolio_value: Total portfolio equity.
            cash: Available cash in the account.
            account_value: Total account value (for PDT check).

        Returns:
            LayerDecision with APPROVE or REJECT.
        """
        if not self.enabled:
            return LayerDecision(
                layer_name="account_risk",
                action=RiskAction.APPROVE,
                reason="Layer disabled",
            )

        with self._lock:
            return self._evaluate_locked(request, portfolio_value, cash, account_value)

    def _evaluate_locked(
        self,
        request: TradeRequest,
        portfolio_value: float,
        cash: float,
        account_value: Optional[float],
    ) -> LayerDecision:
        """Core evaluation logic (under lock)."""
        account_value = account_value or portfolio_value

        # Check if already halted
        if self._is_halted:
            return LayerDecision(
                layer_name="account_risk",
                action=RiskAction.REJECT,
                reason=f"Account halted: {self._halt_reason}",
            )

        # 1. Daily loss limit
        if portfolio_value > 0:
            daily_loss_pct = -self._daily_pnl / portfolio_value if self._daily_pnl < 0 else 0.0
            if daily_loss_pct >= self.max_daily_loss_pct:
                self._is_halted = True
                self._halt_reason = f"Daily loss limit hit: {daily_loss_pct:.1%}"
                logger.warning("account_risk.daily_loss_halt", loss_pct=f"{daily_loss_pct:.1%}")
                return LayerDecision(
                    layer_name="account_risk",
                    action=RiskAction.REJECT,
                    reason=self._halt_reason,
                )

        # 2. Weekly loss limit
        if portfolio_value > 0:
            weekly_loss_pct = -self._weekly_pnl / portfolio_value if self._weekly_pnl < 0 else 0.0
            if weekly_loss_pct >= self.max_weekly_loss_pct:
                self._is_halted = True
                self._halt_reason = f"Weekly loss limit hit: {weekly_loss_pct:.1%}"
                logger.warning("account_risk.weekly_loss_halt", loss_pct=f"{weekly_loss_pct:.1%}")
                return LayerDecision(
                    layer_name="account_risk",
                    action=RiskAction.REJECT,
                    reason=self._halt_reason,
                )

        # 3. Max drawdown from peak
        if self._peak_equity > 0:
            current_equity = getattr(self, "_current_equity", portfolio_value)
            drawdown = (self._peak_equity - current_equity) / self._peak_equity
            if drawdown >= self.max_drawdown_pct:
                self._is_halted = True
                self._halt_reason = f"Max drawdown hit: {drawdown:.1%} from peak"
                logger.warning("account_risk.drawdown_halt", drawdown=f"{drawdown:.1%}")
                return LayerDecision(
                    layer_name="account_risk",
                    action=RiskAction.REJECT,
                    reason=self._halt_reason,
                )

        # 4. Minimum cash reserve
        if portfolio_value > 0:
            trade_cost = request.notional_value
            remaining_cash = cash - trade_cost
            min_cash = self.min_cash_reserve_pct * portfolio_value
            if request.side == "buy" and remaining_cash < min_cash:
                return LayerDecision(
                    layer_name="account_risk",
                    action=RiskAction.REJECT,
                    reason=f"Insufficient cash reserve: need {min_cash:.0f}, would have {remaining_cash:.0f}",
                    metadata={"min_cash": min_cash, "remaining_cash": remaining_cash},
                )

        # 5. PDT rule (< 4 day trades in 5 business days if under 25K)
        if account_value < self.pdt_account_threshold:
            cutoff = date.today() - timedelta(days=5)
            recent_day_trades = sum(1 for dt in self._day_trades if dt >= cutoff)
            if recent_day_trades >= self.max_day_trades_5d:
                return LayerDecision(
                    layer_name="account_risk",
                    action=RiskAction.REJECT,
                    reason=f"PDT limit: {recent_day_trades} day trades in last 5 days (account < ${self.pdt_account_threshold:,.0f})",
                )

        # 6. Consecutive loss limit
        if self._consecutive_losses >= self.consecutive_loss_limit:
            return LayerDecision(
                layer_name="account_risk",
                action=RiskAction.REJECT,
                reason=f"Consecutive loss limit ({self.consecutive_loss_limit}) reached — pausing",
                metadata={"consecutive_losses": self._consecutive_losses},
            )

        # 7. Daily trade count limit
        if self._trades_today >= self.daily_trade_limit:
            return LayerDecision(
                layer_name="account_risk",
                action=RiskAction.REJECT,
                reason=f"Daily trade limit ({self.daily_trade_limit}) reached",
            )

        return LayerDecision(
            layer_name="account_risk",
            action=RiskAction.APPROVE,
            reason="All account checks passed",
        )

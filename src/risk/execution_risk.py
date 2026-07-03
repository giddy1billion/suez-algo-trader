"""
Execution Quality Risk Layer — Last-mile checks before order submission.

Checks:
- Bid-ask spread threshold
- Liquidity (volume) check
- Slippage estimation
- Rate limiting (max orders per minute)
- Cooldown after large loss
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional

from src.risk.models import TradeRequest, LayerDecision, RiskAction
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionRiskLayer:
    """
    Evaluates execution quality risk — ensures orders are placed only when
    market microstructure is favorable and rate limits are respected.
    """

    def __init__(
        self,
        max_spread_pct: float = 0.005,
        min_volume: int = 10_000,
        max_slippage_pct: float = 0.003,
        max_orders_per_minute: int = 10,
        cooldown_after_large_loss_minutes: int = 5,
        large_loss_threshold_pct: float = 0.01,
        enabled: bool = True,
    ):
        self.max_spread_pct = max_spread_pct
        self.min_volume = min_volume
        self.max_slippage_pct = max_slippage_pct
        self.max_orders_per_minute = max_orders_per_minute
        self.cooldown_after_large_loss_minutes = cooldown_after_large_loss_minutes
        self.large_loss_threshold_pct = large_loss_threshold_pct
        self.enabled = enabled

        # Internal state
        self._lock = threading.Lock()
        self._order_timestamps: deque = deque(maxlen=200)
        self._last_large_loss_time: Optional[float] = None
        self._last_large_loss_amount: float = 0.0

    # ──────────────────────────────────────────────────────────────────────
    # State Management
    # ──────────────────────────────────────────────────────────────────────

    def record_order(self) -> None:
        """Record that an order was placed (for rate limiting)."""
        with self._lock:
            self._order_timestamps.append(time.time())

    def record_large_loss(self, loss_pct: float) -> None:
        """Record a large loss event to trigger cooldown."""
        with self._lock:
            if abs(loss_pct) >= self.large_loss_threshold_pct:
                self._last_large_loss_time = time.time()
                self._last_large_loss_amount = loss_pct
                logger.warning(
                    "execution_risk.large_loss_cooldown",
                    loss_pct=f"{loss_pct:.2%}",
                    cooldown_minutes=self.cooldown_after_large_loss_minutes,
                )

    # ──────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        request: TradeRequest,
        portfolio_value: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        current_volume: Optional[int] = None,
        average_daily_volume: Optional[int] = None,
    ) -> LayerDecision:
        """
        Evaluate execution quality risk.

        Args:
            request: The proposed trade.
            portfolio_value: Total portfolio equity.
            bid: Current best bid price.
            ask: Current best ask price.
            current_volume: Current session volume.
            average_daily_volume: Average daily volume.

        Returns:
            LayerDecision with APPROVE or REJECT.
        """
        if not self.enabled:
            return LayerDecision(
                layer_name="execution_risk",
                action=RiskAction.APPROVE,
                reason="Layer disabled",
            )

        with self._lock:
            return self._evaluate_locked(
                request, portfolio_value, bid, ask, current_volume, average_daily_volume
            )

    def _evaluate_locked(
        self,
        request: TradeRequest,
        portfolio_value: float,
        bid: Optional[float],
        ask: Optional[float],
        current_volume: Optional[int],
        average_daily_volume: Optional[int],
    ) -> LayerDecision:
        """Core evaluation logic (under lock)."""

        # 1. Spread check
        if bid is not None and ask is not None and bid > 0:
            spread = ask - bid
            spread_pct = spread / bid
            if spread_pct > self.max_spread_pct:
                return LayerDecision(
                    layer_name="execution_risk",
                    action=RiskAction.REJECT,
                    reason=f"Spread too wide: {spread_pct:.3%} > {self.max_spread_pct:.3%}",
                    metadata={"spread": spread, "spread_pct": spread_pct},
                )

        # 2. Liquidity check
        if current_volume is not None and current_volume < self.min_volume:
            return LayerDecision(
                layer_name="execution_risk",
                action=RiskAction.REJECT,
                reason=f"Insufficient volume: {current_volume:,} < {self.min_volume:,}",
                metadata={"current_volume": current_volume},
            )

        # 3. Slippage estimate
        if average_daily_volume is not None and average_daily_volume > 0:
            # Simple slippage model: slippage proportional to order size / ADV
            participation_rate = request.qty / average_daily_volume
            estimated_slippage = participation_rate * 0.1  # 10% market impact coefficient
            if estimated_slippage > self.max_slippage_pct:
                return LayerDecision(
                    layer_name="execution_risk",
                    action=RiskAction.REJECT,
                    reason=f"Expected slippage too high: {estimated_slippage:.3%} > {self.max_slippage_pct:.3%}",
                    metadata={
                        "estimated_slippage": estimated_slippage,
                        "participation_rate": participation_rate,
                    },
                )

        # 4. Rate limiting
        now = time.time()
        one_minute_ago = now - 60
        recent_orders = sum(1 for t in self._order_timestamps if t > one_minute_ago)
        if recent_orders >= self.max_orders_per_minute:
            return LayerDecision(
                layer_name="execution_risk",
                action=RiskAction.REJECT,
                reason=f"Rate limit: {recent_orders} orders in last minute (max {self.max_orders_per_minute})",
            )

        # 5. Cooldown after large loss
        if self._last_large_loss_time is not None:
            elapsed_minutes = (now - self._last_large_loss_time) / 60.0
            if elapsed_minutes < self.cooldown_after_large_loss_minutes:
                remaining = self.cooldown_after_large_loss_minutes - elapsed_minutes
                return LayerDecision(
                    layer_name="execution_risk",
                    action=RiskAction.REJECT,
                    reason=f"Cooldown active: {remaining:.1f} min remaining after large loss ({self._last_large_loss_amount:.2%})",
                    metadata={"cooldown_remaining_min": remaining},
                )
            else:
                # Cooldown expired, clear it
                self._last_large_loss_time = None
                self._last_large_loss_amount = 0.0

        return LayerDecision(
            layer_name="execution_risk",
            action=RiskAction.APPROVE,
            reason="All execution checks passed",
        )

"""
Position/Exposure Risk Layer — Evaluates individual trade quality and exposure.

Checks:
- Stop loss required
- Position size vs average daily volume (ADV)
- Single trade concentration
- Overnight exposure limits
- Earnings/event blackout
- Volatility regime adjustment
"""

from __future__ import annotations

import threading
from datetime import datetime, date
from typing import Optional

from src.risk.models import TradeRequest, LayerDecision, RiskAction
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExposureRiskLayer:
    """
    Evaluates individual position exposure risk. Ensures trades have proper
    risk controls and don't take outsized positions relative to liquidity.
    """

    def __init__(
        self,
        require_stop_loss: bool = True,
        max_adv_pct: float = 0.01,
        max_trade_concentration_pct: float = 0.05,
        max_overnight_exposure_pct: float = 0.60,
        earnings_blackout_days: int = 1,
        high_vol_threshold: float = 0.03,
        high_vol_size_reduction: float = 0.50,
        enabled: bool = True,
    ):
        self.require_stop_loss = require_stop_loss
        self.max_adv_pct = max_adv_pct
        self.max_trade_concentration_pct = max_trade_concentration_pct
        self.max_overnight_exposure_pct = max_overnight_exposure_pct
        self.earnings_blackout_days = earnings_blackout_days
        self.high_vol_threshold = high_vol_threshold
        self.high_vol_size_reduction = high_vol_size_reduction
        self.enabled = enabled
        self._lock = threading.Lock()

    def evaluate(
        self,
        request: TradeRequest,
        portfolio_value: float,
        average_daily_volume: Optional[float] = None,
        daily_volatility: Optional[float] = None,
        earnings_date: Optional[date] = None,
        is_market_hours: bool = True,
        current_positions: Optional[list[dict]] = None,
    ) -> LayerDecision:
        """
        Evaluate trade against position/exposure constraints.

        Args:
            request: The proposed trade.
            portfolio_value: Total portfolio equity.
            average_daily_volume: Average daily share volume for the symbol.
            daily_volatility: Daily return standard deviation for the symbol.
            earnings_date: Next earnings date for the symbol (if known).
            is_market_hours: Whether market is currently open.
            current_positions: List of current positions.

        Returns:
            LayerDecision with APPROVE, REJECT, or REDUCE.
        """
        if not self.enabled:
            return LayerDecision(
                layer_name="exposure_risk",
                action=RiskAction.APPROVE,
                reason="Layer disabled",
            )

        with self._lock:
            return self._evaluate_locked(
                request, portfolio_value, average_daily_volume,
                daily_volatility, earnings_date, is_market_hours, current_positions,
            )

    def _evaluate_locked(
        self,
        request: TradeRequest,
        portfolio_value: float,
        average_daily_volume: Optional[float],
        daily_volatility: Optional[float],
        earnings_date: Optional[date],
        is_market_hours: bool,
        current_positions: Optional[list[dict]],
    ) -> LayerDecision:
        """Core evaluation logic (under lock)."""
        adjusted_qty = request.qty

        # 1. Stop loss required
        if self.require_stop_loss and request.stop_loss is None:
            return LayerDecision(
                layer_name="exposure_risk",
                action=RiskAction.REJECT,
                reason="Stop loss is required for all trades",
            )

        # 2. Position size vs ADV
        if average_daily_volume is not None and average_daily_volume > 0:
            adv_pct = request.qty / average_daily_volume
            if adv_pct > self.max_adv_pct:
                max_qty = self.max_adv_pct * average_daily_volume
                adjusted_qty = min(adjusted_qty, max_qty)
                logger.info(
                    "exposure_risk.adv_reduced",
                    symbol=request.symbol,
                    adv_pct=f"{adv_pct:.2%}",
                    max_pct=f"{self.max_adv_pct:.2%}",
                )

        # 3. Single trade concentration
        if portfolio_value > 0:
            trade_pct = (adjusted_qty * request.price) / portfolio_value
            if trade_pct > self.max_trade_concentration_pct:
                max_value = self.max_trade_concentration_pct * portfolio_value
                adjusted_qty = min(adjusted_qty, max_value / request.price)

        # 4. Overnight exposure limits
        if not is_market_hours:
            # Near market close — check overnight exposure
            positions = current_positions or []
            total_exposure = sum(abs(float(p.get("market_value", 0))) for p in positions)
            new_exposure = total_exposure + adjusted_qty * request.price
            if portfolio_value > 0 and new_exposure / portfolio_value > self.max_overnight_exposure_pct:
                remaining = (self.max_overnight_exposure_pct * portfolio_value) - total_exposure
                if remaining <= 0:
                    return LayerDecision(
                        layer_name="exposure_risk",
                        action=RiskAction.REJECT,
                        reason=f"Overnight exposure limit ({self.max_overnight_exposure_pct:.0%}) would be exceeded",
                    )
                adjusted_qty = min(adjusted_qty, remaining / request.price)

        # 5. Earnings blackout
        if earnings_date is not None:
            days_to_earnings = (earnings_date - date.today()).days
            if 0 <= days_to_earnings <= self.earnings_blackout_days:
                return LayerDecision(
                    layer_name="exposure_risk",
                    action=RiskAction.REJECT,
                    reason=f"Earnings blackout: {request.symbol} reports in {days_to_earnings} day(s)",
                    metadata={"earnings_date": str(earnings_date)},
                )

        # 6. Volatility adjustment
        if daily_volatility is not None and daily_volatility > self.high_vol_threshold:
            vol_adjusted_qty = adjusted_qty * self.high_vol_size_reduction
            if vol_adjusted_qty < adjusted_qty:
                logger.info(
                    "exposure_risk.vol_adjusted",
                    symbol=request.symbol,
                    daily_vol=f"{daily_volatility:.2%}",
                    reduction=f"{self.high_vol_size_reduction:.0%}",
                )
                adjusted_qty = vol_adjusted_qty

        # Final decision
        if adjusted_qty <= 0:
            return LayerDecision(
                layer_name="exposure_risk",
                action=RiskAction.REJECT,
                reason="Adjusted quantity is zero after exposure checks",
            )

        if adjusted_qty < request.qty:
            return LayerDecision(
                layer_name="exposure_risk",
                action=RiskAction.REDUCE,
                reason="Position size reduced by exposure constraints",
                adjusted_qty=adjusted_qty,
                metadata={
                    "original_qty": request.qty,
                    "reduction_pct": 1 - (adjusted_qty / request.qty),
                },
            )

        return LayerDecision(
            layer_name="exposure_risk",
            action=RiskAction.APPROVE,
            reason="All exposure checks passed",
        )

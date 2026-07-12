"""
Portfolio-Level Risk Layer — Evaluates portfolio-wide constraints.

Checks:
- Max number of positions
- Single-stock allocation limits
- Sector concentration
- Correlation risk
- Gross/net exposure limits
- Portfolio VaR
- Portfolio heat (aggregate risk)
"""

from __future__ import annotations

import threading
from typing import Optional

from src.risk.models import TradeRequest, LayerDecision, RiskAction
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioRiskLayer:
    """
    Evaluates whether a trade is acceptable from a portfolio construction
    standpoint. Enforces diversification, exposure limits, and VaR constraints.
    """

    def __init__(
        self,
        max_positions: int = 10,
        max_single_stock_pct: float = 0.20,
        max_sector_exposure_pct: float = 0.40,
        max_correlation: float = 0.80,
        max_gross_exposure_pct: float = 2.00,
        max_net_exposure_pct: float = 1.00,
        max_var_pct: float = 0.05,
        max_portfolio_heat_pct: float = 0.10,
        enabled: bool = True,
    ):
        self.max_positions = max_positions
        self.max_single_stock_pct = max_single_stock_pct
        self.max_sector_exposure_pct = max_sector_exposure_pct
        self.max_correlation = max_correlation
        self.max_gross_exposure_pct = max_gross_exposure_pct
        self.max_net_exposure_pct = max_net_exposure_pct
        self.max_var_pct = max_var_pct
        self.max_portfolio_heat_pct = max_portfolio_heat_pct
        self.enabled = enabled
        self._lock = threading.Lock()

    def evaluate(
        self,
        request: TradeRequest,
        portfolio_value: float,
        positions: list[dict],
        sector_map: Optional[dict[str, str]] = None,
        correlation_matrix: Optional[dict] = None,
    ) -> LayerDecision:
        """
        Evaluate trade against portfolio-level risk constraints.

        Args:
            request: The proposed trade.
            portfolio_value: Total portfolio equity.
            positions: List of current position dicts with keys:
                symbol, market_value, side, qty, sector (optional).
            sector_map: Optional mapping of symbol -> sector.
            correlation_matrix: Optional dict of (sym1,sym2) -> correlation.

        Returns:
            LayerDecision with APPROVE, REJECT, or REDUCE.
        """
        if not self.enabled:
            return LayerDecision(
                layer_name="portfolio_risk",
                action=RiskAction.APPROVE,
                reason="Layer disabled",
            )

        with self._lock:
            return self._evaluate_locked(
                request, portfolio_value, positions, sector_map, correlation_matrix
            )

    def _evaluate_locked(
        self,
        request: TradeRequest,
        portfolio_value: float,
        positions: list[dict],
        sector_map: Optional[dict[str, str]],
        correlation_matrix: Optional[dict],
    ) -> LayerDecision:
        """Core evaluation logic (called under lock)."""
        if portfolio_value <= 0:
            return LayerDecision(
                layer_name="portfolio_risk",
                action=RiskAction.REJECT,
                reason="Portfolio value is zero or negative",
            )

        # 1. Max positions check
        if request.side == "buy":
            open_count = len([p for p in positions if float(p.get("market_value", 0)) != 0])
            if open_count >= self.max_positions:
                return LayerDecision(
                    layer_name="portfolio_risk",
                    action=RiskAction.REJECT,
                    reason=f"Max positions ({self.max_positions}) reached",
                )

        # 2. Single-stock allocation
        trade_value = request.notional_value
        allocation_pct = trade_value / portfolio_value
        adjusted_qty = request.qty

        if allocation_pct > self.max_single_stock_pct:
            max_value = self.max_single_stock_pct * portfolio_value
            adjusted_qty = max_value / request.price
            logger.info(
                "portfolio_risk.allocation_reduced",
                symbol=request.symbol,
                original_pct=f"{allocation_pct:.1%}",
                max_pct=f"{self.max_single_stock_pct:.1%}",
            )

        # 3. Sector exposure
        if sector_map:
            trade_sector = sector_map.get(request.symbol, "unknown")
            sector_exposure = sum(
                abs(float(p.get("market_value", 0)))
                for p in positions
                if sector_map.get(p.get("symbol", ""), "unknown") == trade_sector
            )
            new_sector_exposure = (sector_exposure + adjusted_qty * request.price) / portfolio_value
            if new_sector_exposure > self.max_sector_exposure_pct:
                remaining = (self.max_sector_exposure_pct * portfolio_value) - sector_exposure
                if remaining <= 0:
                    return LayerDecision(
                        layer_name="portfolio_risk",
                        action=RiskAction.REJECT,
                        reason=f"Sector '{trade_sector}' exposure limit ({self.max_sector_exposure_pct:.0%}) reached",
                    )
                adjusted_qty = min(adjusted_qty, remaining / request.price)

        # 4. Correlation risk
        if correlation_matrix and positions:
            for pos in positions:
                pos_symbol = pos.get("symbol", "")
                pair = tuple(sorted([request.symbol, pos_symbol]))
                corr = correlation_matrix.get(pair, 0.0)
                if abs(corr) > self.max_correlation:
                    return LayerDecision(
                        layer_name="portfolio_risk",
                        action=RiskAction.REJECT,
                        reason=f"High correlation ({corr:.2f}) with existing position {pos_symbol}",
                        metadata={"correlated_with": pos_symbol, "correlation": corr},
                    )

        # 5. Gross exposure
        gross_exposure = sum(abs(float(p.get("market_value", 0))) for p in positions)
        new_gross = gross_exposure + adjusted_qty * request.price
        if new_gross / portfolio_value > self.max_gross_exposure_pct:
            remaining = (self.max_gross_exposure_pct * portfolio_value) - gross_exposure
            if remaining <= 0:
                return LayerDecision(
                    layer_name="portfolio_risk",
                    action=RiskAction.REJECT,
                    reason=f"Gross exposure limit ({self.max_gross_exposure_pct:.0%}) reached",
                )
            adjusted_qty = min(adjusted_qty, remaining / request.price)

        # 6. Net exposure
        long_exposure = sum(
            float(p.get("market_value", 0))
            for p in positions
            if p.get("side") == "long"
        )
        short_exposure = sum(
            abs(float(p.get("market_value", 0)))
            for p in positions
            if p.get("side") == "short"
        )
        if request.side == "buy":
            new_net = (long_exposure + adjusted_qty * request.price - short_exposure) / portfolio_value
        else:
            new_net = (long_exposure - (short_exposure + adjusted_qty * request.price)) / portfolio_value

        if abs(new_net) > self.max_net_exposure_pct:
            return LayerDecision(
                layer_name="portfolio_risk",
                action=RiskAction.REJECT,
                reason=f"Net exposure ({new_net:.0%}) would exceed limit ({self.max_net_exposure_pct:.0%})",
            )

        # 7. Portfolio VaR (simplified parametric estimate)
        # Uses position-level realized volatility when available,
        # falling back to asset-class defaults if daily_vol is absent.
        _asset_class_vol_defaults = {
            "crypto": 0.05,   # ~5% daily vol typical for BTC/ETH
            "equity": 0.015,  # ~1.5% daily vol typical for large-cap equity
        }
        default_vol = 0.02

        portfolio_heat = 0.0
        for p in positions:
            pos_vol = float(p.get("daily_vol", 0.0))
            if pos_vol <= 0:
                asset_class = str(p.get("asset_class", "equity")).lower()
                pos_vol = _asset_class_vol_defaults.get(asset_class, default_vol)
            portfolio_heat += abs(float(p.get("market_value", 0))) * pos_vol

        # Estimate trade volatility from position data or asset-class default
        trade_asset_class = "equity"  # default; callers can set via request metadata
        trade_vol = _asset_class_vol_defaults.get(trade_asset_class, default_vol)
        trade_heat = adjusted_qty * request.price * trade_vol
        total_heat = portfolio_heat + trade_heat
        var_estimate = total_heat * 1.65  # 95% confidence, 1-day

        if var_estimate / portfolio_value > self.max_var_pct:
            return LayerDecision(
                layer_name="portfolio_risk",
                action=RiskAction.REJECT,
                reason=f"Portfolio VaR ({var_estimate/portfolio_value:.1%}) exceeds limit ({self.max_var_pct:.1%})",
                metadata={"var_estimate": var_estimate, "var_pct": var_estimate / portfolio_value},
            )

        # 8. Portfolio heat check
        if total_heat / portfolio_value > self.max_portfolio_heat_pct:
            # Reduce rather than reject
            max_heat_remaining = (self.max_portfolio_heat_pct * portfolio_value) - portfolio_heat
            if max_heat_remaining <= 0:
                return LayerDecision(
                    layer_name="portfolio_risk",
                    action=RiskAction.REJECT,
                    reason="Portfolio heat limit reached",
                )
            # Scale down qty so trade contributes only the remaining heat budget
            max_trade_value = max_heat_remaining / trade_vol
            adjusted_qty = min(adjusted_qty, max_trade_value / request.price)

        # Determine final action
        if adjusted_qty < request.qty:
            return LayerDecision(
                layer_name="portfolio_risk",
                action=RiskAction.REDUCE,
                reason="Position size reduced by portfolio constraints",
                adjusted_qty=adjusted_qty,
                metadata={
                    "original_qty": request.qty,
                    "reduction_pct": 1 - (adjusted_qty / request.qty),
                },
            )

        return LayerDecision(
            layer_name="portfolio_risk",
            action=RiskAction.APPROVE,
            reason="All portfolio checks passed",
        )

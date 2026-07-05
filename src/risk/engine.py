"""
Risk Engine Orchestrator — Chains all risk layers to produce a final decision.

The RiskEngine sits between strategy signal generation and order execution.
Each layer independently evaluates the trade and can APPROVE, REJECT, or REDUCE.
If any layer rejects, the trade is blocked. If a layer reduces, position size
is adjusted downward before passing to the next layer.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Optional

from src.risk.models import (
    TradeRequest,
    RiskDecision,
    RiskMetrics,
    LayerDecision,
    RiskAction,
)
from src.risk.portfolio_risk import PortfolioRiskLayer
from src.risk.account_risk import AccountRiskLayer
from src.risk.exposure_risk import ExposureRiskLayer
from src.risk.execution_risk import ExecutionRiskLayer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RiskEngine:
    """
    Multi-layer risk engine that evaluates trade requests through independent
    risk layers. Thread-safe — can be called concurrently from the main loop,
    Telegram commands, and scheduler.

    Layer evaluation order:
    1. Account Risk (daily/weekly limits, drawdown, PDT)
    2. Portfolio Risk (diversification, exposure, VaR)
    3. Exposure Risk (position quality, liquidity, events)
    4. Execution Risk (spread, slippage, rate limiting)
    """

    def __init__(
        self,
        portfolio_layer: Optional[PortfolioRiskLayer] = None,
        account_layer: Optional[AccountRiskLayer] = None,
        exposure_layer: Optional[ExposureRiskLayer] = None,
        execution_layer: Optional[ExecutionRiskLayer] = None,
    ):
        self.portfolio_layer = portfolio_layer or PortfolioRiskLayer()
        self.account_layer = account_layer or AccountRiskLayer()
        self.exposure_layer = exposure_layer or ExposureRiskLayer()
        self.execution_layer = execution_layer or ExecutionRiskLayer()

        self._lock = threading.Lock()
        self._decision_log: list[dict] = []
        self._metrics = RiskMetrics()

    # ──────────────────────────────────────────────────────────────────────
    # Core Evaluation
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        request: TradeRequest,
        portfolio_value: float = 0.0,
        cash: float = 0.0,
        positions: Optional[list[dict]] = None,
        market_data: Optional[dict] = None,
    ) -> RiskDecision:
        """
        Evaluate a trade request through all risk layers.

        Args:
            request: The trade to evaluate.
            portfolio_value: Total account equity.
            cash: Available cash.
            positions: List of current positions.
            market_data: Optional dict with market context:
                - bid, ask: Current quote
                - volume: Current session volume
                - adv: Average daily volume
                - daily_vol: Daily volatility
                - earnings_date: Next earnings date
                - sector_map: symbol -> sector mapping
                - correlation_matrix: correlation data
                - is_market_hours: bool

        Returns:
            RiskDecision with final approval status and adjusted quantity.
        """
        positions = positions or []
        market_data = market_data or {}
        layer_decisions: dict[str, LayerDecision] = {}
        current_qty = request.qty
        reasons: list[str] = []

        # Pre-check: reject signals with insufficient confidence.
        # Signals at or below 0.5 confidence are likely placeholder/fallback
        # values from strategies that failed to compute real predictions
        # (e.g., no active ML model, missing features, exception branches).
        if request.confidence <= 0.5:
            return self._build_decision(
                approved=False,
                adjusted_qty=0.0,
                reasons=[f"Confidence {request.confidence:.2f} below minimum threshold (>0.5 required)"],
                layer_decisions=layer_decisions,
                request=request,
            )

        # Layer 1: Account Risk
        account_decision = self.account_layer.evaluate(
            request=request,
            portfolio_value=portfolio_value,
            cash=cash,
            account_value=portfolio_value,
        )
        layer_decisions["account_risk"] = account_decision

        if account_decision.action == RiskAction.REJECT:
            return self._build_decision(
                approved=False,
                adjusted_qty=0.0,
                reasons=[account_decision.reason],
                layer_decisions=layer_decisions,
                request=request,
            )

        # Layer 2: Portfolio Risk
        portfolio_decision = self.portfolio_layer.evaluate(
            request=request,
            portfolio_value=portfolio_value,
            positions=positions,
            sector_map=market_data.get("sector_map"),
            correlation_matrix=market_data.get("correlation_matrix"),
        )
        layer_decisions["portfolio_risk"] = portfolio_decision

        if portfolio_decision.action == RiskAction.REJECT:
            return self._build_decision(
                approved=False,
                adjusted_qty=0.0,
                reasons=[portfolio_decision.reason],
                layer_decisions=layer_decisions,
                request=request,
            )
        if portfolio_decision.action == RiskAction.REDUCE:
            current_qty = min(current_qty, portfolio_decision.adjusted_qty or current_qty)
            reasons.append(portfolio_decision.reason)

        # Layer 3: Exposure Risk — use reduced qty
        adjusted_request = TradeRequest(
            symbol=request.symbol,
            side=request.side,
            qty=current_qty,
            price=request.price,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            strategy=request.strategy,
            confidence=request.confidence,
            urgency=request.urgency,
        )

        exposure_decision = self.exposure_layer.evaluate(
            request=adjusted_request,
            portfolio_value=portfolio_value,
            average_daily_volume=market_data.get("adv"),
            daily_volatility=market_data.get("daily_vol"),
            earnings_date=market_data.get("earnings_date"),
            is_market_hours=market_data.get("is_market_hours", True),
            current_positions=positions,
        )
        layer_decisions["exposure_risk"] = exposure_decision

        if exposure_decision.action == RiskAction.REJECT:
            return self._build_decision(
                approved=False,
                adjusted_qty=0.0,
                reasons=[exposure_decision.reason],
                layer_decisions=layer_decisions,
                request=request,
            )
        if exposure_decision.action == RiskAction.REDUCE:
            current_qty = min(current_qty, exposure_decision.adjusted_qty or current_qty)
            reasons.append(exposure_decision.reason)

        # Layer 4: Execution Risk
        execution_decision = self.execution_layer.evaluate(
            request=adjusted_request,
            portfolio_value=portfolio_value,
            bid=market_data.get("bid"),
            ask=market_data.get("ask"),
            current_volume=market_data.get("volume"),
            average_daily_volume=market_data.get("adv"),
        )
        layer_decisions["execution_risk"] = execution_decision

        if execution_decision.action == RiskAction.REJECT:
            return self._build_decision(
                approved=False,
                adjusted_qty=0.0,
                reasons=[execution_decision.reason],
                layer_decisions=layer_decisions,
                request=request,
            )

        # All layers passed — record the order for rate limiting
        self.execution_layer.record_order()

        return self._build_decision(
            approved=True,
            adjusted_qty=current_qty,
            reasons=reasons if reasons else ["All risk checks passed"],
            layer_decisions=layer_decisions,
            request=request,
        )

    # ──────────────────────────────────────────────────────────────────────
    # State Management
    # ──────────────────────────────────────────────────────────────────────

    def update_account_state(
        self,
        current_equity: float,
        daily_pnl: float,
        weekly_pnl: float,
        cash: float,
    ) -> None:
        """Update account-level state for risk tracking."""
        self.account_layer.update_state(current_equity, daily_pnl, weekly_pnl, cash)

    def record_trade_result(self, pnl: float, portfolio_value: float, was_day_trade: bool = False) -> None:
        """Record a completed trade for consecutive loss / cooldown tracking."""
        self.account_layer.record_trade_result(pnl, was_day_trade)
        if portfolio_value > 0:
            loss_pct = pnl / portfolio_value
            if pnl < 0:
                self.execution_layer.record_large_loss(loss_pct)

    def get_metrics(self) -> RiskMetrics:
        """Return current risk metrics snapshot."""
        return self._metrics

    def reset_daily(self) -> None:
        """Reset daily counters (call at start of trading day)."""
        self.account_layer.reset_halt()

    @property
    def is_halted(self) -> bool:
        """Check if trading is halted by any layer."""
        return self.account_layer.is_halted

    @property
    def halt_reason(self) -> str:
        """Get halt reason if trading is halted."""
        return self.account_layer.halt_reason

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _build_decision(
        self,
        approved: bool,
        adjusted_qty: float,
        reasons: list[str],
        layer_decisions: dict[str, LayerDecision],
        request: TradeRequest,
    ) -> RiskDecision:
        """Build final RiskDecision and log it."""
        # Calculate risk score (0-100)
        risk_score = self._calculate_risk_score(layer_decisions, request)

        decision = RiskDecision(
            approved=approved,
            adjusted_qty=adjusted_qty,
            reasons=reasons,
            layer_decisions=layer_decisions,
            risk_score=risk_score,
        )

        # Audit log
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": request.symbol,
            "side": request.side,
            "requested_qty": request.qty,
            "adjusted_qty": adjusted_qty,
            "approved": approved,
            "risk_score": risk_score,
            "reasons": reasons,
            "layers": {
                name: {"action": d.action.value, "reason": d.reason}
                for name, d in layer_decisions.items()
            },
        }

        with self._lock:
            self._decision_log.append(log_entry)
            # Keep last 1000 decisions in memory
            if len(self._decision_log) > 1000:
                self._decision_log = self._decision_log[-500:]

        if approved:
            logger.info(
                "risk_engine.approved",
                symbol=request.symbol,
                side=request.side,
                qty=adjusted_qty,
                risk_score=risk_score,
            )
        else:
            logger.info(
                "risk_engine.rejected",
                symbol=request.symbol,
                side=request.side,
                reasons=reasons,
                risk_score=risk_score,
            )

        return decision

    def _calculate_risk_score(
        self, layer_decisions: dict[str, LayerDecision], request: TradeRequest
    ) -> float:
        """
        Calculate aggregate risk score 0-100 based on layer outcomes.
        Higher = riskier.
        """
        score = 0.0

        # Base score from layer outcomes
        for decision in layer_decisions.values():
            if decision.action == RiskAction.REJECT:
                score += 30
            elif decision.action == RiskAction.REDUCE:
                score += 15

        # Confidence adjustment (low confidence = higher risk)
        if request.confidence < 0.3:
            score += 20
        elif request.confidence < 0.5:
            score += 10

        # Stop loss presence
        if request.stop_loss is None:
            score += 15

        return min(score, 100.0)

    def get_decision_log(self, limit: int = 50) -> list[dict]:
        """Get recent risk decisions for audit/analysis."""
        with self._lock:
            return self._decision_log[-limit:]

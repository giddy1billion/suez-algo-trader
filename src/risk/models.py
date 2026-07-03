"""
Risk Engine Data Models — Shared types for the multi-layer risk system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class RiskAction(str, Enum):
    """Possible outcomes from a risk layer evaluation."""
    APPROVE = "approve"
    REJECT = "reject"
    REDUCE = "reduce"


@dataclass
class TradeRequest:
    """Incoming trade request to be evaluated by the risk engine."""
    symbol: str
    side: str  # "buy" or "sell"
    qty: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: str = "unknown"
    confidence: float = 0.0
    urgency: float = 0.5  # 0.0 = low urgency, 1.0 = immediate

    @property
    def notional_value(self) -> float:
        return self.qty * self.price

    @property
    def risk_per_share(self) -> float:
        if self.stop_loss is None:
            return 0.0
        return abs(self.price - self.stop_loss)

    @property
    def total_risk(self) -> float:
        return self.risk_per_share * self.qty


@dataclass
class LayerDecision:
    """Decision from a single risk layer."""
    layer_name: str
    action: RiskAction
    reason: str
    adjusted_qty: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskDecision:
    """Final aggregated decision from the risk engine."""
    approved: bool
    adjusted_qty: float
    reasons: list[str] = field(default_factory=list)
    layer_decisions: dict[str, LayerDecision] = field(default_factory=dict)
    risk_score: float = 0.0  # 0 = no risk, 100 = maximum risk
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def was_reduced(self) -> bool:
        return any(
            d.action == RiskAction.REDUCE
            for d in self.layer_decisions.values()
        )

    @property
    def rejection_reasons(self) -> list[str]:
        return [
            d.reason for d in self.layer_decisions.values()
            if d.action == RiskAction.REJECT
        ]


@dataclass
class RiskMetrics:
    """Current portfolio risk metrics snapshot."""
    portfolio_var: float = 0.0        # Value at Risk (1-day, 95%)
    gross_exposure: float = 0.0       # Sum of absolute position values
    net_exposure: float = 0.0         # Long - Short exposure
    portfolio_heat: float = 0.0       # Sum of all position risk amounts
    daily_pnl: float = 0.0           # Today's realized + unrealized P&L
    drawdown: float = 0.0            # Current drawdown from peak (%)
    open_positions: int = 0
    cash_ratio: float = 1.0          # Cash / Total equity
    timestamp: datetime = field(default_factory=datetime.now)

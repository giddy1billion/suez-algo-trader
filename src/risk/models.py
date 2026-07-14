"""
Risk Engine Data Models — Shared types for the multi-layer risk system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from src.intelligence.confidence.models import ConfidenceScore
    from src.intelligence.confidence.decision_contract import DecisionContract
    from src.strategy.base import TradeSignal


class RiskAction(str, Enum):
    """Possible outcomes from a risk layer evaluation."""
    APPROVE = "approve"
    REJECT = "reject"
    REDUCE = "reduce"


@dataclass
class TradeRequest:
    """
    Incoming trade request to be evaluated by the risk engine.

    Clean Architecture:
        The TradeRequest carries both the strategy's PROPOSAL (TradeSignal)
        and the system's DECISION (DecisionContract). The risk engine reads
        from the contract for execution parameters (SL, TP, position size),
        NOT from the signal.

    Flow:
        TradeSignal → DecisionOrchestrator → DecisionContract → TradeRequest → RiskEngine
    """
    symbol: str
    side: str  # "buy" or "sell"
    qty: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: str = "unknown"
    confidence: float = 0.0
    urgency: float = 0.5  # 0.0 = low urgency, 1.0 = immediate

    # Rich confidence object (optional — backward-compatible)
    confidence_score: Optional[ConfidenceScore] = None

    # Decision Contract — the authoritative decision object.
    # When present, this is the single source of truth for the trade.
    decision_contract: Optional[DecisionContract] = None

    # Source signal — the strategy's original proposal (audit trail)
    trade_signal: Optional[TradeSignal] = None

    @property
    def effective_confidence(self) -> float:
        """Return the best available confidence value (contract > score > scalar)."""
        if self.decision_contract is not None:
            return self.decision_contract.final_confidence
        if self.confidence_score is not None:
            return self.confidence_score.value
        return self.confidence

    @property
    def has_contract(self) -> bool:
        """Whether this request carries an authoritative DecisionContract."""
        return self.decision_contract is not None

    @property
    def has_signal(self) -> bool:
        """Whether this request carries the source TradeSignal."""
        return self.trade_signal is not None

    @property
    def signal_id(self) -> str:
        """Get signal_id from the attached trade signal, or empty string."""
        if self.trade_signal is not None:
            return self.trade_signal.signal_id
        return ""

    @property
    def contract_id(self) -> str:
        """Get contract_id from the attached decision contract, or empty string."""
        if self.decision_contract is not None:
            return self.decision_contract.contract_id
        return ""

    @property
    def notional_value(self) -> float:
        return self.qty * self.price

    def __post_init__(self):
        """Validate trade request inputs to prevent garbage data propagating to risk layers."""
        import math
        if self.qty < 0:
            raise ValueError(f"TradeRequest.qty must be non-negative, got {self.qty}")
        if self.price <= 0 or (isinstance(self.price, float) and (math.isnan(self.price) or math.isinf(self.price))):
            raise ValueError(f"TradeRequest.price must be positive and finite, got {self.price}")
        if isinstance(self.confidence, float) and (math.isnan(self.confidence) or math.isinf(self.confidence)):
            raise ValueError(f"TradeRequest.confidence must be finite, got {self.confidence}")
        if self.side not in ("buy", "sell"):
            raise ValueError(f"TradeRequest.side must be 'buy' or 'sell', got '{self.side}'")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("TradeRequest.symbol must be non-empty")

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

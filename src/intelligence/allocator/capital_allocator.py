"""Capital allocator with regime- and drift-aware sizing multipliers."""

from __future__ import annotations

from typing import Optional

from src.intelligence.models import AllocationDecision, DriftState, MarketState


class CapitalAllocator:
    def allocate(
        self,
        market_state: MarketState,
        drift_state: DriftState,
        portfolio_context: Optional[dict] = None,
    ) -> AllocationDecision:
        portfolio_context = portfolio_context or {}
        qty_multiplier = 1.0
        max_exposure_multiplier = 1.0
        reasons: list[str] = []

        if market_state.stress == "Panic":
            qty_multiplier *= 0.4
            max_exposure_multiplier *= 0.5
            reasons.append("Stress panic: size reduced")
        elif market_state.stress == "Elevated":
            qty_multiplier *= 0.75
            reasons.append("Elevated stress: moderate size reduction")

        if market_state.volatility == "Extreme":
            qty_multiplier *= 0.6
            reasons.append("Extreme volatility: additional reduction")
        elif market_state.volatility == "Compression":
            qty_multiplier *= 1.05
            reasons.append("Compression regime: slight size increase")

        if drift_state.degrading:
            qty_multiplier *= 0.7
            max_exposure_multiplier *= 0.8
            reasons.append("Concept drift degrading: defensive allocation")

        corr_risk = float(portfolio_context.get("correlation_risk", 0.0))
        if corr_risk > 0.7:
            qty_multiplier *= 0.7
            reasons.append("High correlation risk: reduced allocation")

        qty_multiplier = max(0.1, min(qty_multiplier, 1.5))
        max_exposure_multiplier = max(0.3, min(max_exposure_multiplier, 1.5))
        return AllocationDecision(
            qty_multiplier=qty_multiplier,
            max_exposure_multiplier=max_exposure_multiplier,
            reasons=reasons,
        )


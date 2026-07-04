"""Human-readable explanations for intelligence decisions."""

from __future__ import annotations

from src.intelligence.models import AllocationDecision, MarketState, RoutingDecision, TradeQualityResult


class DecisionExplainer:
    def build(
        self,
        accepted: bool,
        market_state: MarketState,
        routing: RoutingDecision,
        quality: TradeQualityResult,
        allocation: AllocationDecision,
    ) -> str:
        lines = []
        lines.append("Trade Accepted" if accepted else "Trade Rejected")
        lines.append(f"Regime: {market_state.overall_regime} ({market_state.confidence:.0%})")
        lines.append(f"Routing: {'PASS' if routing.enabled else 'BLOCK'} - {routing.reason}")

        for item in quality.positives:
            lines.append(f"✔ {item}")
        for item in quality.negatives:
            lines.append(f"✘ {item}")

        for reason in allocation.reasons:
            lines.append(f"• Allocation: {reason}")
        lines.append(f"Final Score: {quality.score:.1f}")
        lines.append(f"Qty Multiplier: {allocation.qty_multiplier:.2f}x")
        return "\n".join(lines)


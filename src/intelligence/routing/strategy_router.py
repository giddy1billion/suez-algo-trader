"""Strategy routing decisions based on regime fit and drift state."""

from __future__ import annotations

from src.intelligence.models import DriftState, MarketState, RoutingDecision


class StrategyRouter:
    def evaluate(self, strategy_name: str, market_state: MarketState, drift_state: DriftState) -> RoutingDecision:
        name = strategy_name.lower()

        if "ml" in name and drift_state.degrading:
            return RoutingDecision(enabled=False, reason="ML strategy paused due to concept drift")

        if "momentum" in name:
            if "Uptrend" in market_state.trend or "Downtrend" in market_state.trend:
                if market_state.stress != "Panic":
                    return RoutingDecision(enabled=True, reason="Momentum aligned with trend regime")
            return RoutingDecision(enabled=False, reason="Momentum blocked by regime mismatch")

        if "mean" in name:
            if market_state.trend == "Sideways" and market_state.volatility != "Extreme":
                return RoutingDecision(enabled=True, reason="Mean reversion aligned with ranging regime")
            return RoutingDecision(enabled=False, reason="Mean reversion blocked outside ranging regime")

        if "breakout" in name:
            if market_state.volatility in ("Expansion", "Extreme"):
                return RoutingDecision(enabled=True, reason="Breakout aligned with volatility expansion")
            return RoutingDecision(enabled=False, reason="Breakout blocked in low-volatility regime")

        return RoutingDecision(enabled=True, reason="No explicit routing rule; strategy allowed")


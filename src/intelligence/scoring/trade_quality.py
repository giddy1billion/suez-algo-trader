"""Trade quality scoring with regime/portfolio/execution adjustments."""

from __future__ import annotations

from typing import Optional

from src.intelligence.models import MarketState, TradeQualityResult


class TradeQualityEngine:
    def __init__(self, threshold: float = 70.0):
        self.threshold = threshold

    def score(
        self,
        strategy_name: str,
        signal_confidence: float,
        market_state: MarketState,
        indicators: Optional[dict] = None,
        portfolio_context: Optional[dict] = None,
        execution_context: Optional[dict] = None,
    ) -> TradeQualityResult:
        indicators = indicators or {}
        portfolio_context = portfolio_context or {}
        execution_context = execution_context or {}
        contributions: dict[str, float] = {}
        positives: list[str] = []
        negatives: list[str] = []

        trend_strength = min(abs(market_state.diagnostics.get("ret_20", 0.0)) / 0.10, 1.0)
        contributions["trend_strength"] = trend_strength * 18.0

        momentum_raw = indicators.get("momentum", indicators.get("rsi", 50.0))
        if isinstance(momentum_raw, (int, float)):
            momentum_score = max(min((float(momentum_raw) - 40.0) / 30.0, 1.0), 0.0)
        else:
            momentum_score = 0.5
        contributions["momentum"] = momentum_score * 15.0

        liq_ratio = market_state.diagnostics.get("liquidity_ratio", 1.0)
        contributions["volume_confirmation"] = max(min(liq_ratio - 0.5, 1.0), 0.0) * 12.0
        contributions["ml_confidence"] = max(min(signal_confidence, 1.0), 0.0) * 15.0

        rr = indicators.get("rr", 0.0)
        rr_bonus = 10.0 if isinstance(rr, (int, float)) and rr >= 2.0 else 0.0
        contributions["risk_reward_bonus"] = rr_bonus

        if "momentum" in strategy_name.lower() and "Uptrend" in market_state.trend:
            contributions["regime_strategy_fit"] = 8.0
            positives.append("Regime supports momentum")
        elif "mean" in strategy_name.lower() and market_state.trend == "Sideways":
            contributions["regime_strategy_fit"] = 8.0
            positives.append("Regime supports mean reversion")
        else:
            contributions["regime_strategy_fit"] = -6.0
            negatives.append("Regime mismatch")

        corr_risk = float(portfolio_context.get("correlation_risk", 0.0))
        corr_penalty = min(max(corr_risk, 0.0), 1.0) * 10.0
        contributions["correlation_penalty"] = -corr_penalty
        if corr_penalty > 5:
            negatives.append("Correlation too high")

        if market_state.volatility in ("Expansion", "Extreme"):
            contributions["high_vol_penalty"] = -5.0
            negatives.append("High volatility penalty")
        else:
            contributions["high_vol_penalty"] = 2.0

        spread_pct = float(execution_context.get("spread_pct", 0.0))
        slippage_penalty = min(max(spread_pct / 0.005, 0.0), 1.0) * 8.0
        contributions["execution_penalty"] = -slippage_penalty

        total = sum(contributions.values())
        score = max(0.0, min(100.0, total))
        accepted = score >= self.threshold

        if score >= 85:
            positives.append("High-conviction setup")
        elif score < self.threshold:
            negatives.append(f"Score below threshold ({score:.1f} < {self.threshold:.1f})")

        if signal_confidence >= 0.8:
            positives.append(f"ML confidence {signal_confidence:.0%}")
        elif signal_confidence < 0.6:
            negatives.append(f"Low confidence {signal_confidence:.0%}")

        return TradeQualityResult(
            score=score,
            threshold=self.threshold,
            accepted=accepted,
            contributions=contributions,
            positives=positives,
            negatives=negatives,
        )

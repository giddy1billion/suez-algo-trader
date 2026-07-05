"""
Market Regime Adjuster — discounts confidence in incompatible regimes.

A trend-following model may report 0.94 confidence in a sideways market.
That confidence should be discounted because the model was trained for
(and performs best in) trending conditions.

Uses the existing 8-dimensional market fingerprint to compute regime
compatibility between the model's optimal conditions and current state.

Evaluated AFTER decay, as the final adjustment before scoring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from src.intelligence.confidence.models import GateResult, GateVerdict


@dataclass
class RegimeProfile:
    """
    A model/strategy's optimal operating regime.

    Each dimension is a preferred range (0.0–1.0) where the strategy
    performs best. Dimensions outside the range get penalized.
    """

    # Preferred regime labels (from MarketFingerprint)
    preferred_trend: list[str] = field(default_factory=lambda: ["Strong Uptrend", "Weak Uptrend", "Strong Downtrend", "Weak Downtrend"])
    preferred_volatility: list[str] = field(default_factory=lambda: ["Normal", "Compression"])
    preferred_stress: list[str] = field(default_factory=lambda: ["Calm", "Elevated"])

    # How sensitive is this strategy to regime mismatch? (0 = ignore, 1 = critical)
    trend_sensitivity: float = 0.5
    volatility_sensitivity: float = 0.3
    stress_sensitivity: float = 0.4
    liquidity_sensitivity: float = 0.2
    momentum_sensitivity: float = 0.3

    @classmethod
    def momentum_strategy(cls) -> RegimeProfile:
        """Profile for momentum/trend strategies."""
        return cls(
            preferred_trend=["Strong Uptrend", "Weak Uptrend", "Strong Downtrend", "Weak Downtrend"],
            preferred_volatility=["Normal", "Expansion"],
            preferred_stress=["Calm", "Elevated"],
            trend_sensitivity=0.8,  # Very sensitive to trend regime
            volatility_sensitivity=0.3,
            stress_sensitivity=0.4,
            momentum_sensitivity=0.7,
        )

    @classmethod
    def mean_reversion_strategy(cls) -> RegimeProfile:
        """Profile for mean-reversion strategies."""
        return cls(
            preferred_trend=["Sideways", "Weak Uptrend", "Weak Downtrend"],
            preferred_volatility=["Normal", "Compression"],
            preferred_stress=["Calm"],
            trend_sensitivity=0.7,
            volatility_sensitivity=0.5,
            stress_sensitivity=0.6,
            momentum_sensitivity=0.2,
        )

    @classmethod
    def ml_strategy(cls) -> RegimeProfile:
        """Profile for ML strategies (generally more adaptive)."""
        return cls(
            preferred_trend=["Strong Uptrend", "Weak Uptrend", "Sideways", "Weak Downtrend", "Strong Downtrend"],
            preferred_volatility=["Normal", "Compression", "Expansion"],
            preferred_stress=["Calm", "Elevated"],
            trend_sensitivity=0.3,  # ML adapts better
            volatility_sensitivity=0.4,
            stress_sensitivity=0.5,
            momentum_sensitivity=0.2,
        )

    @classmethod
    def for_strategy(cls, strategy_name: str) -> RegimeProfile:
        """Factory method for strategy-based regime profiles."""
        profiles = {
            "momentum": cls.momentum_strategy,
            "mean_reversion": cls.mean_reversion_strategy,
            "ml": cls.ml_strategy,
        }
        factory = profiles.get(strategy_name.lower(), cls.ml_strategy)
        return factory()


@dataclass
class RegimeAdjustConfig:
    """Configuration for regime adjustment behavior."""

    # Maximum confidence discount from regime mismatch
    max_discount: float = 0.25

    # Minimum regime compatibility to avoid any discount
    compatibility_threshold: float = 0.7

    # Below this compatibility → reject outright
    reject_threshold: float = 0.3

    # Weight given to overall market fingerprint confidence
    fingerprint_confidence_weight: float = 0.3


class MarketRegimeAdjuster:
    """
    Adjusts confidence based on strategy-regime compatibility.

    Takes the current market state (from MarketFingerprint) and the
    strategy's regime profile, computes compatibility, and applies
    an appropriate discount.
    """

    def __init__(self, config: Optional[RegimeAdjustConfig] = None):
        self.config = config or RegimeAdjustConfig()

    def compute_compatibility(
        self,
        regime_profile: RegimeProfile,
        current_trend: str = "",
        current_volatility: str = "",
        current_stress: str = "",
        current_liquidity: str = "",
        current_momentum: str = "",
        fingerprint_confidence: float = 1.0,
    ) -> float:
        """
        Compute regime compatibility score (0.0–1.0).

        1.0 = perfect match, 0.0 = complete mismatch.
        """
        scores = []

        # Trend compatibility
        if current_trend and regime_profile.trend_sensitivity > 0:
            match = 1.0 if current_trend in regime_profile.preferred_trend else 0.0
            weighted = match * regime_profile.trend_sensitivity + (1 - regime_profile.trend_sensitivity)
            scores.append(weighted)

        # Volatility compatibility
        if current_volatility and regime_profile.volatility_sensitivity > 0:
            match = 1.0 if current_volatility in regime_profile.preferred_volatility else 0.0
            weighted = match * regime_profile.volatility_sensitivity + (1 - regime_profile.volatility_sensitivity)
            scores.append(weighted)

        # Stress compatibility
        if current_stress and regime_profile.stress_sensitivity > 0:
            match = 1.0 if current_stress in regime_profile.preferred_stress else 0.0
            weighted = match * regime_profile.stress_sensitivity + (1 - regime_profile.stress_sensitivity)
            scores.append(weighted)

        if not scores:
            return 1.0  # No regime data available — no adjustment

        raw_compatibility = sum(scores) / len(scores)

        # Factor in fingerprint confidence (if low, we're less certain about regime)
        adjusted = (
            raw_compatibility * (1 - self.config.fingerprint_confidence_weight)
            + fingerprint_confidence * self.config.fingerprint_confidence_weight
        )

        return max(0.0, min(1.0, adjusted))

    def evaluate(
        self,
        strategy_name: str,
        current_trend: str = "",
        current_volatility: str = "",
        current_stress: str = "",
        current_liquidity: str = "",
        current_momentum: str = "",
        fingerprint_confidence: float = 1.0,
        regime_profile: Optional[RegimeProfile] = None,
    ) -> GateResult:
        """
        Evaluate regime compatibility and return adjustment.

        Args:
            strategy_name: Name of the strategy (for profile lookup)
            current_*: Current market state labels from MarketFingerprint
            fingerprint_confidence: Overall confidence of the market state assessment
            regime_profile: Explicit profile override (default: auto from strategy name)
        """
        start = time.perf_counter()

        profile = regime_profile or RegimeProfile.for_strategy(strategy_name)

        compatibility = self.compute_compatibility(
            regime_profile=profile,
            current_trend=current_trend,
            current_volatility=current_volatility,
            current_stress=current_stress,
            current_liquidity=current_liquidity,
            current_momentum=current_momentum,
            fingerprint_confidence=fingerprint_confidence,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Perfect or near-perfect match
        if compatibility >= self.config.compatibility_threshold:
            return GateResult(
                gate_name="regime_adjustment",
                verdict=GateVerdict.PASS,
                score=compatibility,
                reason=f"Regime compatible ({compatibility:.2f}) for {strategy_name}",
                details={
                    "compatibility": compatibility,
                    "trend": current_trend,
                    "volatility": current_volatility,
                    "stress": current_stress,
                },
                elapsed_ms=elapsed_ms,
            )

        # Critical mismatch
        if compatibility < self.config.reject_threshold:
            return GateResult(
                gate_name="regime_adjustment",
                verdict=GateVerdict.REJECT,
                score=compatibility,
                reason=(
                    f"Regime critically incompatible ({compatibility:.2f}) for {strategy_name}. "
                    f"Current: trend={current_trend}, vol={current_volatility}, stress={current_stress}"
                ),
                details={
                    "compatibility": compatibility,
                    "reject_threshold": self.config.reject_threshold,
                },
                elapsed_ms=elapsed_ms,
            )

        # Moderate mismatch → discount
        # Scale discount: at threshold → 0%, at reject_threshold → max_discount
        range_span = self.config.compatibility_threshold - self.config.reject_threshold
        distance = self.config.compatibility_threshold - compatibility
        discount = (distance / range_span) * self.config.max_discount

        return GateResult(
            gate_name="regime_adjustment",
            verdict=GateVerdict.ADJUST,
            score=compatibility,
            adjustment=-discount,
            reason=(
                f"Regime mismatch for {strategy_name}: compatibility={compatibility:.2f} "
                f"(discount={discount:.1%}). "
                f"Current: trend={current_trend}, vol={current_volatility}, stress={current_stress}"
            ),
            details={
                "compatibility": compatibility,
                "discount": discount,
                "strategy": strategy_name,
            },
            elapsed_ms=elapsed_ms,
        )

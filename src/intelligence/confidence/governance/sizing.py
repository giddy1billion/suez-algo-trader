"""Position sizing engine — separates 'Can I trade?' from 'How much?'

Key insight: A signal with confidence=0.61 shouldn't be REJECTED.
It should be EXECUTED at reduced size. Only hard data-quality floors
cause rejection; everything else scales the position.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SizingDecision:
    """The sizing half of the decision — separate from approval."""

    execution_allowed: bool  # Can this trade execute at all?
    execution_reason: str  # Why allowed/blocked
    sizing_confidence: float  # 0.0-1.0: how much to allocate
    risk_multiplier: float  # 0.0-1.0: scale position by this
    expected_edge: float  # Expected return (bps)
    expected_drawdown: float  # Worst-case drawdown estimate
    uncertainty_penalty: float  # Size reduction from uncertainty
    regime_discount: float  # Size reduction from regime mismatch
    base_position_pct: float  # Starting position % before adjustments
    final_position_pct: float  # After all adjustments

    def to_dict(self) -> dict:
        """Serialize for audit logging."""
        return {
            "execution_allowed": self.execution_allowed,
            "execution_reason": self.execution_reason,
            "sizing_confidence": round(self.sizing_confidence, 4),
            "risk_multiplier": round(self.risk_multiplier, 4),
            "expected_edge": round(self.expected_edge, 2),
            "expected_drawdown": round(self.expected_drawdown, 4),
            "uncertainty_penalty": round(self.uncertainty_penalty, 4),
            "regime_discount": round(self.regime_discount, 4),
            "base_position_pct": round(self.base_position_pct, 4),
            "final_position_pct": round(self.final_position_pct, 4),
        }


@dataclass
class SizingEngineConfig:
    """Configuration for the sizing engine."""

    max_position_pct: float = 0.05  # Maximum 5% of portfolio per position
    min_position_pct: float = 0.005  # Minimum 0.5% to bother trading
    base_position_pct: float = 0.02  # Default 2% position
    hard_confidence_floor: float = 0.35  # Below this, block execution entirely
    min_data_quality: float = 0.5  # Minimum data quality score to trade
    uncertainty_weight: float = 0.4  # How much uncertainty reduces size
    regime_weight: float = 0.3  # How much regime mismatch reduces size
    kelly_fraction: float = 0.25  # Fractional Kelly (conservative)


class SizingEngine:
    """Computes position sizing using Kelly-like logic and continuous scaling.

    Philosophy: Never reject a signal just for moderate confidence.
    Reduce size instead. Only hard floors (data quality, stale data)
    cause outright rejection.
    """

    def __init__(self, config: Optional[SizingEngineConfig] = None) -> None:
        self.config = config or SizingEngineConfig()

    def compute(
        self,
        raw_confidence: float,
        uncertainty: float = 0.0,
        regime_compatibility: float = 1.0,
        data_quality: float = 1.0,
        model_health: float = 1.0,
        expected_edge_bps: float = 0.0,
    ) -> SizingDecision:
        """Compute sizing decision from raw inputs.

        Args:
            raw_confidence: Model confidence 0.0-1.0
            uncertainty: Uncertainty score 0.0-1.0 (0=certain, 1=uncertain)
            regime_compatibility: How well current regime matches model training 0.0-1.0
            data_quality: Data freshness/completeness score 0.0-1.0
            model_health: Model performance score 0.0-1.0
            expected_edge_bps: Expected edge in basis points

        Returns:
            SizingDecision with execution approval and position sizing
        """
        # --- HARD REJECTION CHECKS (only data quality issues) ---
        if data_quality < self.config.min_data_quality:
            return self._reject(
                reason=f"Data quality too low: {data_quality:.2f} < {self.config.min_data_quality:.2f}",
                raw_confidence=raw_confidence,
            )

        if raw_confidence < self.config.hard_confidence_floor:
            return self._reject(
                reason=f"Confidence below hard floor: {raw_confidence:.3f} < {self.config.hard_confidence_floor:.2f}",
                raw_confidence=raw_confidence,
            )

        # --- CONTINUOUS SIZING (no binary reject for moderate confidence) ---
        # Uncertainty penalty: high uncertainty → reduce size
        uncertainty_penalty = uncertainty * self.config.uncertainty_weight
        uncertainty_penalty = min(uncertainty_penalty, 0.8)  # Cap at 80% reduction

        # Regime discount: poor regime match → reduce size
        regime_discount = (1.0 - regime_compatibility) * self.config.regime_weight
        regime_discount = min(regime_discount, 0.6)  # Cap at 60% reduction

        # Model health multiplier
        health_multiplier = max(0.3, model_health)

        # Kelly-like sizing: size proportional to edge/variance
        # Simplified: confidence * base_size * adjustments
        sizing_confidence = raw_confidence * health_multiplier

        # Risk multiplier combines all discount factors
        risk_multiplier = max(0.0, 1.0 - uncertainty_penalty - regime_discount)
        risk_multiplier = risk_multiplier * health_multiplier

        # Compute expected drawdown from uncertainty and confidence
        expected_drawdown = (1.0 - raw_confidence) * (1.0 + uncertainty)
        expected_drawdown = min(expected_drawdown, 1.0)

        # Final position size using fractional Kelly
        edge = max(expected_edge_bps, (raw_confidence - 0.5) * 200)  # Convert to bps if not provided
        variance_proxy = max(0.01, uncertainty + (1.0 - regime_compatibility) * 0.5)
        kelly_size = (edge / 10000.0) / variance_proxy if variance_proxy > 0 else 0.0
        kelly_size = kelly_size * self.config.kelly_fraction

        # Base position scaled by confidence and Kelly
        base_pct = self.config.base_position_pct
        final_pct = base_pct * sizing_confidence * risk_multiplier

        # Apply Kelly adjustment (blend Kelly sizing with confidence-based)
        if kelly_size > 0:
            final_pct = final_pct * 0.7 + kelly_size * 0.3

        # Clamp to configured bounds
        final_pct = max(self.config.min_position_pct, min(self.config.max_position_pct, final_pct))

        # If final position is below minimum, still allow but at minimum size
        if final_pct < self.config.min_position_pct:
            final_pct = self.config.min_position_pct

        return SizingDecision(
            execution_allowed=True,
            execution_reason="Signal approved for sized execution",
            sizing_confidence=round(sizing_confidence, 4),
            risk_multiplier=round(risk_multiplier, 4),
            expected_edge=round(edge, 2),
            expected_drawdown=round(expected_drawdown, 4),
            uncertainty_penalty=round(uncertainty_penalty, 4),
            regime_discount=round(regime_discount, 4),
            base_position_pct=round(base_pct, 4),
            final_position_pct=round(final_pct, 6),
        )

    def _reject(self, reason: str, raw_confidence: float) -> SizingDecision:
        """Create a rejection decision."""
        return SizingDecision(
            execution_allowed=False,
            execution_reason=reason,
            sizing_confidence=0.0,
            risk_multiplier=0.0,
            expected_edge=0.0,
            expected_drawdown=1.0,
            uncertainty_penalty=0.0,
            regime_discount=0.0,
            base_position_pct=self.config.base_position_pct,
            final_position_pct=0.0,
        )

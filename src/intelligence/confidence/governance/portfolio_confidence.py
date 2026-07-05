"""Portfolio-level confidence — correlation-aware position sizing.

Key insight: AAPL 0.83 + MSFT 0.81 + QQQ 0.85 are effectively the same bet.
This module discounts confidence when a new signal is highly correlated
with existing positions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfidence:
    """Portfolio-level confidence accounting for position correlation."""

    individual_confidence: float  # This signal's raw confidence
    portfolio_adjusted_confidence: float  # After correlation discount
    correlation_penalty: float  # How much was deducted
    similar_positions: list[str] = field(default_factory=list)  # Highly correlated symbols
    effective_bet_count: float = 1.0  # Diversification ratio
    concentration_risk: float = 0.0  # 0=diversified, 1=concentrated

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "individual_confidence": round(self.individual_confidence, 4),
            "portfolio_adjusted_confidence": round(self.portfolio_adjusted_confidence, 4),
            "correlation_penalty": round(self.correlation_penalty, 4),
            "similar_positions": self.similar_positions,
            "effective_bet_count": round(self.effective_bet_count, 2),
            "concentration_risk": round(self.concentration_risk, 4),
        }


class PortfolioConfidenceEngine:
    """Computes portfolio-adjusted confidence for new signals.

    For each new signal, computes its correlation with every existing position
    and applies a discount proportional to the max correlation. This prevents
    inadvertent concentration in correlated bets.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.6,
        max_penalty: float = 0.3,
        penalty_scaling: float = 0.5,
    ) -> None:
        """Initialize the engine.

        Args:
            correlation_threshold: Correlation above which penalty applies
            max_penalty: Maximum penalty to apply (caps discount)
            penalty_scaling: How aggressively to penalize (0-1)
        """
        self.correlation_threshold = correlation_threshold
        self.max_penalty = max_penalty
        self.penalty_scaling = penalty_scaling

    def compute(
        self,
        symbol: str,
        confidence: float,
        current_positions: dict[str, float],
        correlation_matrix: Optional[dict[tuple[str, str], float]] = None,
        position_weights: Optional[dict[str, float]] = None,
    ) -> PortfolioConfidence:
        """Compute portfolio-adjusted confidence for a new signal.

        Args:
            symbol: The symbol being evaluated
            confidence: Raw confidence for this signal
            current_positions: Dict of symbol → position size (can be notional or %)
            correlation_matrix: Dict of (sym1, sym2) → correlation coefficient
            position_weights: Dict of symbol → weight (fraction of portfolio).
                             If None, computed from current_positions.

        Returns:
            PortfolioConfidence with adjusted values
        """
        if not current_positions:
            return PortfolioConfidence(
                individual_confidence=confidence,
                portfolio_adjusted_confidence=confidence,
                correlation_penalty=0.0,
                similar_positions=[],
                effective_bet_count=1.0,
                concentration_risk=0.0,
            )

        if correlation_matrix is None:
            correlation_matrix = {}

        # Compute position weights if not provided
        if position_weights is None:
            total_exposure = sum(abs(v) for v in current_positions.values())
            if total_exposure > 0:
                position_weights = {
                    s: abs(v) / total_exposure for s, v in current_positions.items()
                }
            else:
                position_weights = {s: 1.0 / len(current_positions) for s in current_positions}

        # Find correlations with existing positions
        correlations: list[tuple[str, float]] = []
        for existing_symbol in current_positions:
            if existing_symbol == symbol:
                continue
            # Look up correlation (check both orderings)
            corr = correlation_matrix.get(
                (symbol, existing_symbol),
                correlation_matrix.get((existing_symbol, symbol), 0.0),
            )
            correlations.append((existing_symbol, abs(corr)))

        # Identify similar positions (above threshold)
        similar_positions = [
            sym for sym, corr in correlations if corr >= self.correlation_threshold
        ]

        # Compute penalty based on max correlation with existing positions
        # Weighted by position size: large correlated positions penalize more
        correlation_penalty = 0.0
        if correlations:
            weighted_correlations = []
            for sym, corr in correlations:
                weight = position_weights.get(sym, 0.0)
                if corr >= self.correlation_threshold:
                    excess = corr - self.correlation_threshold
                    weighted_correlations.append(excess * (1.0 + weight) * self.penalty_scaling)

            if weighted_correlations:
                # Use max correlation-based penalty, not sum (avoid over-penalizing)
                correlation_penalty = min(max(weighted_correlations), self.max_penalty)

        # Apply penalty
        adjusted_confidence = confidence * (1.0 - correlation_penalty)
        adjusted_confidence = max(0.0, adjusted_confidence)

        # Effective bet count: 1/sum(w_i^2) — higher = more diversified
        weights = list(position_weights.values())
        # Include the new position weight estimate
        new_weight = 1.0 / (len(current_positions) + 1)
        all_weights = weights + [new_weight]
        # Normalize
        total_w = sum(all_weights)
        if total_w > 0:
            all_weights = [w / total_w for w in all_weights]
        sum_sq = sum(w * w for w in all_weights)
        effective_bet_count = 1.0 / sum_sq if sum_sq > 0 else len(all_weights)

        # Concentration risk: max single position / total
        max_weight = max(all_weights) if all_weights else 0.0
        concentration_risk = max_weight  # 1/N → 0 concentration, 1.0 → full concentration

        return PortfolioConfidence(
            individual_confidence=confidence,
            portfolio_adjusted_confidence=round(adjusted_confidence, 4),
            correlation_penalty=round(correlation_penalty, 4),
            similar_positions=similar_positions,
            effective_bet_count=round(effective_bet_count, 2),
            concentration_risk=round(concentration_risk, 4),
        )

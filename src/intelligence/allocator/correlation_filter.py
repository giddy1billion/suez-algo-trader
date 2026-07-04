"""
Correlation Filter — Correlated signal deduplication.

When multiple correlated assets signal simultaneously:
1. Rank by trade quality score
2. Allow the top-ranked signal through unchanged
3. Apply position reduction for secondary correlated signals
4. Skip highly correlated positions beyond configurable threshold
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeCandidate:
    """A trade signal candidate for correlation filtering."""

    symbol: str
    direction: str  # "long" | "short"
    quality_score: float  # From trade quality scorer
    confidence: float
    strategy: str = ""
    sector: Optional[str] = None


@dataclass
class FilterResult:
    """Result of correlation filtering on a batch of candidates."""

    passed: list[TradeCandidate] = field(default_factory=list)
    reduced: list[tuple[TradeCandidate, float]] = field(default_factory=list)  # (candidate, reduction_factor)
    skipped: list[TradeCandidate] = field(default_factory=list)
    signals_received: int = 0
    signals_passed: int = 0
    signals_reduced: int = 0
    signals_skipped: int = 0


class CorrelationFilter:
    """
    Filters correlated signals to prevent portfolio concentration.

    When multiple correlated assets signal simultaneously:
    - Top-ranked signal passes through unchanged
    - Secondary correlated signals get position reduction
    - Tertiary and beyond are skipped entirely
    """

    def __init__(
        self,
        correlation_threshold: float = 0.0,
        max_correlated_positions: int = 2,
        reduction_factor: float = 0.5,
    ):
        self._threshold = correlation_threshold or settings.portfolio_correlation_threshold
        self._max_correlated = max_correlated_positions
        self._reduction_factor = reduction_factor
        # Pre-defined correlation groups (can be updated dynamically)
        self._correlation_matrix: dict[tuple[str, str], float] = {}
        self._correlation_groups: list[set[str]] = []

    def filter_signals(
        self,
        candidates: list[TradeCandidate],
        correlation_matrix: Optional[dict[tuple[str, str], float]] = None,
    ) -> FilterResult:
        """
        Filter a batch of trade candidates for correlation.

        Args:
            candidates: List of trade candidates to filter
            correlation_matrix: Optional pairwise correlations {(sym1, sym2): corr}

        Returns:
            FilterResult with passed, reduced, and skipped candidates
        """
        if correlation_matrix:
            self._correlation_matrix = correlation_matrix

        result = FilterResult(signals_received=len(candidates))

        if not candidates:
            return result

        # Sort by quality score (highest first)
        sorted_candidates = sorted(candidates, key=lambda c: c.quality_score, reverse=True)

        # Track which symbols have been accepted
        accepted_symbols: list[str] = []

        for candidate in sorted_candidates:
            correlated_count = self._count_correlated(candidate.symbol, accepted_symbols)

            if correlated_count == 0:
                # No correlation with accepted positions — pass through
                result.passed.append(candidate)
                result.signals_passed += 1
                accepted_symbols.append(candidate.symbol)
            elif correlated_count < self._max_correlated:
                # Some correlation — reduce position
                result.reduced.append((candidate, self._reduction_factor))
                result.signals_reduced += 1
                accepted_symbols.append(candidate.symbol)
            else:
                # Too many correlated positions — skip
                result.skipped.append(candidate)
                result.signals_skipped += 1

        logger.debug(
            "correlation_filter.applied",
            received=result.signals_received,
            passed=result.signals_passed,
            reduced=result.signals_reduced,
            skipped=result.signals_skipped,
        )

        return result

    def _count_correlated(self, symbol: str, accepted: list[str]) -> int:
        """Count how many accepted positions are correlated with this symbol."""
        count = 0
        for accepted_sym in accepted:
            corr = self._get_correlation(symbol, accepted_sym)
            if abs(corr) >= self._threshold:
                count += 1
        return count

    def _get_correlation(self, sym1: str, sym2: str) -> float:
        """Get pairwise correlation between two symbols."""
        if sym1 == sym2:
            return 1.0
        # Check both orderings
        corr = self._correlation_matrix.get((sym1, sym2))
        if corr is not None:
            return corr
        corr = self._correlation_matrix.get((sym2, sym1))
        if corr is not None:
            return corr
        # Default: check if in same correlation group
        for group in self._correlation_groups:
            if sym1 in group and sym2 in group:
                return self._threshold  # Assume threshold-level correlation
        return 0.0

    def update_correlation_matrix(
        self, matrix: dict[tuple[str, str], float]
    ) -> None:
        """Update the correlation matrix with new data."""
        self._correlation_matrix = matrix

    def set_correlation_groups(self, groups: list[set[str]]) -> None:
        """Set pre-defined correlation groups (e.g., tech stocks)."""
        self._correlation_groups = groups

    @staticmethod
    def compute_correlation_matrix(
        returns_df: "pd.DataFrame",
    ) -> dict[tuple[str, str], float]:
        """
        Compute pairwise correlations from a returns DataFrame.

        Args:
            returns_df: DataFrame with symbols as columns, returns as rows

        Returns:
            Dict mapping (sym1, sym2) -> correlation
        """
        corr_matrix = returns_df.corr()
        result = {}
        symbols = corr_matrix.columns.tolist()
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i < j:
                    result[(sym1, sym2)] = float(corr_matrix.iloc[i, j])
        return result

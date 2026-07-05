"""
Portfolio Allocator — Full portfolio-level capital allocation with correlation awareness.

Determines:
- Position size (Kelly criterion, capped)
- Correlation-adjusted exposure
- Sector/asset-class concentration limits
- Daily risk budget remaining
- Maximum leverage constraint
- Cash reserve minimum
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from config.settings import settings
from src.intelligence.allocator.correlation_filter import (
    CorrelationFilter,
    FilterResult,
    TradeCandidate,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AllocationResult:
    """Result of portfolio-level allocation decision."""

    symbol: str
    approved: bool
    position_size_pct: float = 0.0  # Fraction of portfolio
    position_value: float = 0.0
    qty: float = 0.0
    kelly_fraction: float = 0.0
    correlation_adjustment: float = 1.0
    sector_limit_hit: bool = False
    daily_budget_remaining_pct: float = 1.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class PortfolioState:
    """Current portfolio state for allocation decisions."""

    total_value: float = 0.0
    cash: float = 0.0
    positions: list[dict[str, Any]] = field(default_factory=list)
    daily_pnl: float = 0.0
    daily_risk_budget: float = 0.0
    sector_exposures: dict[str, float] = field(default_factory=dict)  # sector -> pct


class PortfolioAllocator:
    """
    Full portfolio-level capital allocation with correlation awareness.

    Pipeline:
    Signal → Trade Quality Score → Regime Check → Portfolio Correlation Check
        → Exposure Check → Liquidity Check → Capital Allocation → Execution
    """

    def __init__(
        self,
        correlation_threshold: float = 0.0,
        max_sector_concentration: float = 0.0,
        cash_reserve_min: float = 0.0,
        max_kelly_fraction: float = 0.0,
        max_leverage: float = 1.0,
    ):
        self._correlation_threshold = (
            correlation_threshold or settings.portfolio_correlation_threshold
        )
        self._max_sector_concentration = (
            max_sector_concentration or settings.portfolio_max_sector_concentration
        )
        self._cash_reserve_min = cash_reserve_min or settings.portfolio_cash_reserve_min
        self._max_kelly = max_kelly_fraction or settings.portfolio_max_kelly_fraction
        self._max_leverage = max_leverage or settings.max_leverage

        self._correlation_filter = CorrelationFilter(
            correlation_threshold=self._correlation_threshold,
        )

    def allocate(
        self,
        candidate: TradeCandidate,
        portfolio_state: PortfolioState,
        win_rate: float = 0.5,
        avg_win_loss_ratio: float = 1.5,
        current_price: float = 0.0,
        correlation_matrix: Optional[dict] = None,
    ) -> AllocationResult:
        """
        Determine capital allocation for a trade candidate.

        Args:
            candidate: The trade candidate
            portfolio_state: Current portfolio state
            win_rate: Historical win rate for Kelly calculation
            avg_win_loss_ratio: Average win / average loss
            current_price: Current price of the asset
            correlation_matrix: Optional pairwise correlations

        Returns:
            AllocationResult with position sizing decision
        """
        reasons = []

        # Check cash reserve
        cash_pct = portfolio_state.cash / portfolio_state.total_value if portfolio_state.total_value > 0 else 0
        if cash_pct <= self._cash_reserve_min:
            return AllocationResult(
                symbol=candidate.symbol,
                approved=False,
                reasons=["Cash reserve minimum breached"],
            )

        # Check sector concentration
        if candidate.sector and candidate.sector in portfolio_state.sector_exposures:
            sector_exp = portfolio_state.sector_exposures[candidate.sector]
            if sector_exp >= self._max_sector_concentration:
                return AllocationResult(
                    symbol=candidate.symbol,
                    approved=False,
                    sector_limit_hit=True,
                    reasons=[f"Sector '{candidate.sector}' at concentration limit ({sector_exp:.1%})"],
                )

        # Kelly criterion (capped)
        kelly = self._compute_kelly(win_rate, avg_win_loss_ratio)
        capped_kelly = min(kelly, self._max_kelly)
        reasons.append(f"Kelly={kelly:.3f}, capped={capped_kelly:.3f}")

        # Correlation adjustment
        corr_adjustment = self._compute_correlation_adjustment(
            candidate.symbol, portfolio_state.positions, correlation_matrix
        )
        reasons.append(f"Correlation adjustment={corr_adjustment:.2f}")

        # Position size
        available_capital = portfolio_state.cash - (
            portfolio_state.total_value * self._cash_reserve_min
        )
        position_size_pct = capped_kelly * corr_adjustment
        position_value = portfolio_state.total_value * position_size_pct

        # Cap by available capital
        position_value = min(position_value, available_capital)

        # Compute quantity
        qty = position_value / current_price if current_price > 0 else 0

        # Daily budget check
        budget_remaining = 1.0
        if portfolio_state.daily_risk_budget > 0:
            budget_used = abs(portfolio_state.daily_pnl) / portfolio_state.daily_risk_budget
            budget_remaining = max(0, 1.0 - budget_used)
            if budget_remaining < 0.1:
                return AllocationResult(
                    symbol=candidate.symbol,
                    approved=False,
                    daily_budget_remaining_pct=budget_remaining,
                    reasons=["Daily risk budget exhausted"],
                )
            # Scale position by remaining budget
            position_value *= budget_remaining
            qty = position_value / current_price if current_price > 0 else 0
            reasons.append(f"Budget remaining={budget_remaining:.1%}")

        # Leverage check
        total_exposure = sum(
            abs(p.get("market_value", 0)) for p in portfolio_state.positions
        )
        new_exposure = total_exposure + position_value
        leverage = new_exposure / portfolio_state.total_value if portfolio_state.total_value > 0 else 0
        if leverage > self._max_leverage:
            return AllocationResult(
                symbol=candidate.symbol,
                approved=False,
                reasons=[f"Leverage limit ({leverage:.2f}x > {self._max_leverage}x)"],
            )

        return AllocationResult(
            symbol=candidate.symbol,
            approved=True,
            position_size_pct=position_size_pct,
            position_value=position_value,
            qty=qty,
            kelly_fraction=capped_kelly,
            correlation_adjustment=corr_adjustment,
            daily_budget_remaining_pct=budget_remaining,
            reasons=reasons,
        )

    def filter_batch(
        self,
        candidates: list[TradeCandidate],
        correlation_matrix: Optional[dict] = None,
    ) -> FilterResult:
        """Run correlation filter on a batch of candidates."""
        return self._correlation_filter.filter_signals(candidates, correlation_matrix)

    def _compute_kelly(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Compute Kelly criterion fraction.

        Kelly% = W - (1-W)/R
        where W = win probability, R = win/loss ratio
        """
        if win_loss_ratio <= 0:
            return 0.0
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        return max(0.0, kelly)

    def _compute_correlation_adjustment(
        self,
        symbol: str,
        positions: list[dict],
        correlation_matrix: Optional[dict] = None,
    ) -> float:
        """Compute position size reduction based on portfolio correlation."""
        if not positions or not correlation_matrix:
            return 1.0

        max_corr = 0.0
        for pos in positions:
            pos_symbol = pos.get("symbol", "")
            corr = abs(
                correlation_matrix.get((symbol, pos_symbol), 0.0)
                or correlation_matrix.get((pos_symbol, symbol), 0.0)
            )
            max_corr = max(max_corr, corr)

        # Reduce position as correlation increases
        if max_corr >= self._correlation_threshold:
            # Linear reduction: at threshold -> 0.7x, at 1.0 -> 0.3x
            reduction = 1.0 - (max_corr - self._correlation_threshold) * 2.0
            return max(0.3, min(1.0, reduction))

        return 1.0

    def update_correlation_matrix(self, matrix: dict) -> None:
        """Update correlation filter with new correlation data."""
        self._correlation_filter.update_correlation_matrix(matrix)

    @property
    def correlation_filter(self) -> CorrelationFilter:
        return self._correlation_filter

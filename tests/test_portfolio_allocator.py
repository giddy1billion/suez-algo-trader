"""Tests for Portfolio Allocator and Correlation Filter (Phase 5)."""

import pytest

from src.intelligence.allocator.correlation_filter import (
    CorrelationFilter,
    FilterResult,
    TradeCandidate,
)
from src.intelligence.allocator.portfolio_allocator import (
    AllocationResult,
    PortfolioAllocator,
    PortfolioState,
)


class TestCorrelationFilter:
    """Test correlation-based signal filtering."""

    @pytest.fixture
    def filter(self):
        return CorrelationFilter(
            correlation_threshold=0.70,
            max_correlated_positions=2,
            reduction_factor=0.5,
        )

    def test_no_candidates(self, filter):
        result = filter.filter_signals([])
        assert result.signals_received == 0
        assert result.signals_passed == 0

    def test_uncorrelated_candidates_all_pass(self, filter):
        candidates = [
            TradeCandidate(symbol="AAPL", direction="long", quality_score=80, confidence=0.8),
            TradeCandidate(symbol="XOM", direction="long", quality_score=75, confidence=0.7),
            TradeCandidate(symbol="JNJ", direction="long", quality_score=70, confidence=0.6),
        ]
        result = filter.filter_signals(candidates)
        assert result.signals_passed == 3
        assert result.signals_reduced == 0
        assert result.signals_skipped == 0

    def test_correlated_candidates_top_ranked_passes(self, filter):
        candidates = [
            TradeCandidate(symbol="AAPL", direction="long", quality_score=80, confidence=0.8),
            TradeCandidate(symbol="MSFT", direction="long", quality_score=75, confidence=0.7),
            TradeCandidate(symbol="GOOGL", direction="long", quality_score=70, confidence=0.6),
        ]
        # All tech stocks correlated
        corr_matrix = {
            ("AAPL", "MSFT"): 0.85,
            ("AAPL", "GOOGL"): 0.80,
            ("MSFT", "GOOGL"): 0.82,
        }
        result = filter.filter_signals(candidates, correlation_matrix=corr_matrix)
        # AAPL passes (best), MSFT reduced, GOOGL skipped
        assert result.signals_passed == 1
        assert result.signals_reduced == 1
        assert result.signals_skipped == 1

    def test_ranking_by_quality_score(self, filter):
        candidates = [
            TradeCandidate(symbol="MSFT", direction="long", quality_score=60, confidence=0.6),
            TradeCandidate(symbol="AAPL", direction="long", quality_score=90, confidence=0.9),
        ]
        corr_matrix = {("AAPL", "MSFT"): 0.85}
        result = filter.filter_signals(candidates, correlation_matrix=corr_matrix)
        # AAPL should pass (higher score), MSFT should be reduced
        assert result.passed[0].symbol == "AAPL"
        assert result.reduced[0][0].symbol == "MSFT"

    def test_correlation_groups(self, filter):
        filter.set_correlation_groups([{"AAPL", "MSFT", "GOOGL"}])
        candidates = [
            TradeCandidate(symbol="AAPL", direction="long", quality_score=80, confidence=0.8),
            TradeCandidate(symbol="MSFT", direction="long", quality_score=75, confidence=0.7),
            TradeCandidate(symbol="XOM", direction="long", quality_score=70, confidence=0.6),
        ]
        result = filter.filter_signals(candidates)
        # XOM is not in the group, so uncorrelated
        assert result.signals_passed >= 2  # AAPL + XOM at minimum


class TestPortfolioAllocator:
    """Test portfolio-level capital allocation."""

    @pytest.fixture
    def allocator(self):
        return PortfolioAllocator(
            correlation_threshold=0.70,
            max_sector_concentration=0.40,
            cash_reserve_min=0.10,
            max_kelly_fraction=0.25,
            max_leverage=2.0,
        )

    @pytest.fixture
    def portfolio_state(self):
        return PortfolioState(
            total_value=100000.0,
            cash=50000.0,
            positions=[],
            daily_pnl=-500.0,
            daily_risk_budget=3000.0,
            sector_exposures={"tech": 0.20, "energy": 0.10},
        )

    def test_basic_allocation(self, allocator, portfolio_state):
        candidate = TradeCandidate(
            symbol="AAPL",
            direction="long",
            quality_score=80,
            confidence=0.8,
            sector="tech",
        )
        result = allocator.allocate(
            candidate=candidate,
            portfolio_state=portfolio_state,
            win_rate=0.6,
            avg_win_loss_ratio=1.5,
            current_price=150.0,
        )
        assert result.approved is True
        assert result.qty > 0
        assert result.kelly_fraction > 0

    def test_cash_reserve_rejection(self, allocator):
        state = PortfolioState(
            total_value=100000.0,
            cash=5000.0,  # Only 5% cash
            positions=[],
        )
        candidate = TradeCandidate(
            symbol="AAPL", direction="long", quality_score=80, confidence=0.8
        )
        result = allocator.allocate(
            candidate=candidate,
            portfolio_state=state,
            win_rate=0.6,
            avg_win_loss_ratio=1.5,
            current_price=150.0,
        )
        assert result.approved is False
        assert "Cash reserve" in result.reasons[0]

    def test_sector_concentration_rejection(self, allocator):
        state = PortfolioState(
            total_value=100000.0,
            cash=50000.0,
            positions=[],
            sector_exposures={"tech": 0.45},  # Above 0.40 limit
        )
        candidate = TradeCandidate(
            symbol="AAPL",
            direction="long",
            quality_score=80,
            confidence=0.8,
            sector="tech",
        )
        result = allocator.allocate(
            candidate=candidate,
            portfolio_state=state,
            win_rate=0.6,
            avg_win_loss_ratio=1.5,
            current_price=150.0,
        )
        assert result.approved is False
        assert result.sector_limit_hit is True

    def test_leverage_limit(self, allocator):
        state = PortfolioState(
            total_value=100000.0,
            cash=50000.0,
            positions=[
                {"symbol": "MSFT", "market_value": 150000.0},
                {"symbol": "GOOGL", "market_value": 50000.0},
            ],  # Already 2x leveraged
        )
        candidate = TradeCandidate(
            symbol="AAPL", direction="long", quality_score=80, confidence=0.8
        )
        result = allocator.allocate(
            candidate=candidate,
            portfolio_state=state,
            win_rate=0.6,
            avg_win_loss_ratio=1.5,
            current_price=150.0,
        )
        assert result.approved is False
        assert "Leverage" in result.reasons[0]

    def test_kelly_criterion(self, allocator):
        state = PortfolioState(total_value=100000.0, cash=80000.0)
        candidate = TradeCandidate(
            symbol="AAPL", direction="long", quality_score=80, confidence=0.8
        )
        # Win rate 60%, win/loss ratio 1.5
        # Kelly = 0.6 - 0.4/1.5 = 0.6 - 0.267 = 0.333, capped at 0.25
        result = allocator.allocate(
            candidate=candidate,
            portfolio_state=state,
            win_rate=0.6,
            avg_win_loss_ratio=1.5,
            current_price=150.0,
        )
        assert result.kelly_fraction == 0.25  # Capped

    def test_daily_budget_exhausted(self, allocator):
        state = PortfolioState(
            total_value=100000.0,
            cash=50000.0,
            daily_pnl=-2900.0,  # Almost exhausted (budget is 3000)
            daily_risk_budget=3000.0,
        )
        candidate = TradeCandidate(
            symbol="AAPL", direction="long", quality_score=80, confidence=0.8
        )
        result = allocator.allocate(
            candidate=candidate,
            portfolio_state=state,
            win_rate=0.6,
            avg_win_loss_ratio=1.5,
            current_price=150.0,
        )
        assert result.approved is False
        assert "budget" in result.reasons[0].lower()

    def test_filter_batch(self, allocator):
        candidates = [
            TradeCandidate(symbol="AAPL", direction="long", quality_score=80, confidence=0.8),
            TradeCandidate(symbol="MSFT", direction="long", quality_score=75, confidence=0.7),
        ]
        result = allocator.filter_batch(candidates)
        assert isinstance(result, FilterResult)
        assert result.signals_received == 2

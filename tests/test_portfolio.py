"""Tests for the Portfolio Optimizer module."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio.optimizer import PortfolioOptimizer, OptimizationMethod, PortfolioAllocation


def _make_returns(n_assets=5, n_days=252, seed=42):
    rng = np.random.default_rng(seed)
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.02, (n_days, n_assets)),
        columns=[f"ASSET_{i}" for i in range(n_assets)]
    )
    return returns


@pytest.fixture
def optimizer():
    return PortfolioOptimizer()


@pytest.fixture
def returns_df():
    return _make_returns()


class TestEqualWeight:
    def test_all_weights_equal(self, optimizer, returns_df):
        alloc = optimizer.optimize(returns_df, method=OptimizationMethod.EQUAL_WEIGHT)
        weights = list(alloc.weights.values())
        assert all(abs(w - weights[0]) < 1e-10 for w in weights)

    def test_weights_sum_to_one(self, optimizer, returns_df):
        alloc = optimizer.optimize(returns_df, method=OptimizationMethod.EQUAL_WEIGHT)
        assert abs(alloc.total_weight - 1.0) < 1e-10


class TestRiskParity:
    def test_lower_vol_gets_higher_weight(self):
        # Use higher max_weight so constraints don't flatten the result
        opt = PortfolioOptimizer(max_weight=0.80)
        rng = np.random.default_rng(99)
        n_days = 252
        returns = pd.DataFrame({
            "LOW_VOL": rng.normal(0.0005, 0.01, n_days),
            "HIGH_VOL": rng.normal(0.0005, 0.04, n_days),
        })
        alloc = opt.optimize(returns, method=OptimizationMethod.RISK_PARITY)
        assert alloc.weights["LOW_VOL"] > alloc.weights["HIGH_VOL"]

    def test_weights_sum_to_one(self, optimizer, returns_df):
        alloc = optimizer.optimize(returns_df, method=OptimizationMethod.RISK_PARITY)
        assert abs(alloc.total_weight - 1.0) < 1e-10


class TestMinVariance:
    def test_valid_weights(self, optimizer, returns_df):
        alloc = optimizer.optimize(returns_df, method=OptimizationMethod.MIN_VARIANCE)
        assert abs(alloc.total_weight - 1.0) < 1e-10
        assert all(w >= 0 for w in alloc.weights.values())

    def test_produces_weights_summing_to_one(self, optimizer, returns_df):
        alloc = optimizer.optimize(returns_df, method=OptimizationMethod.MIN_VARIANCE)
        assert abs(sum(alloc.weights.values()) - 1.0) < 1e-10


class TestKelly:
    def test_positive_expectancy_gets_weight(self):
        # Use higher max_weight so Kelly differences aren't flattened
        opt = PortfolioOptimizer(max_weight=0.80)
        rng = np.random.default_rng(7)
        n_days = 252
        returns = pd.DataFrame({
            "WINNER": rng.normal(0.002, 0.01, n_days),
            "LOSER": rng.normal(-0.002, 0.01, n_days),
        })
        alloc = opt.optimize(returns, method=OptimizationMethod.KELLY)
        # Winner should get higher weight
        assert alloc.weights["WINNER"] > alloc.weights["LOSER"]


class TestHRP:
    def test_valid_weights(self, optimizer, returns_df):
        alloc = optimizer.optimize(returns_df, method=OptimizationMethod.HRP)
        assert abs(alloc.total_weight - 1.0) < 1e-10
        assert all(w >= 0 for w in alloc.weights.values())

    def test_handles_correlated_assets(self, optimizer):
        rng = np.random.default_rng(11)
        n_days = 252
        base = rng.normal(0.0005, 0.02, n_days)
        returns = pd.DataFrame({
            "A": base + rng.normal(0, 0.001, n_days),
            "B": base + rng.normal(0, 0.001, n_days),
            "C": rng.normal(0.0005, 0.02, n_days),
        })
        alloc = optimizer.optimize(returns, method=OptimizationMethod.HRP)
        assert abs(alloc.total_weight - 1.0) < 1e-10


class TestVolTarget:
    def test_portfolio_vol_near_target(self, optimizer, returns_df):
        target = 0.15
        alloc = optimizer.optimize(returns_df, method=OptimizationMethod.VOL_TARGET, target_vol=target)
        # Allow 50% tolerance due to constraints and estimation noise
        assert abs(alloc.expected_volatility - target) / target < 0.5


class TestConstraints:
    def test_no_weight_exceeds_max(self):
        optimizer = PortfolioOptimizer(max_weight=0.30)
        # Use risk parity with varied vols — some assets would naturally get > 30%
        rng = np.random.default_rng(55)
        n_days = 252
        returns = pd.DataFrame({
            "CALM": rng.normal(0.0005, 0.005, n_days),   # very low vol
            "MED1": rng.normal(0.0005, 0.02, n_days),
            "MED2": rng.normal(0.0005, 0.02, n_days),
            "WILD": rng.normal(0.0005, 0.05, n_days),
        })
        alloc = optimizer.optimize(returns, method=OptimizationMethod.RISK_PARITY)
        for w in alloc.weights.values():
            assert w <= optimizer.max_weight + 1e-10


class TestPositionSizes:
    def test_converts_weights_to_dollars(self, optimizer, returns_df):
        alloc = optimizer.optimize(returns_df, method=OptimizationMethod.EQUAL_WEIGHT)
        capital = 100_000.0
        sizes = alloc.get_position_sizes(capital)
        assert abs(sum(sizes.values()) - capital) < 1e-6
        for sym in alloc.weights:
            assert abs(sizes[sym] - alloc.weights[sym] * capital) < 1e-6


class TestEdgeCases:
    def test_single_asset(self, optimizer):
        returns = _make_returns(n_assets=1)
        for method in OptimizationMethod:
            alloc = optimizer.optimize(returns, method=method)
            assert abs(alloc.total_weight - 1.0) < 1e-10

    def test_empty_dataframe(self, optimizer):
        empty = pd.DataFrame()
        alloc = optimizer.optimize(empty, method=OptimizationMethod.RISK_PARITY)
        assert alloc.weights == {}

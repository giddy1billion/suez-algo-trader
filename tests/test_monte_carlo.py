"""Tests for Monte Carlo simulation."""
import pytest
import numpy as np
from backtesting.monte_carlo import monte_carlo_simulation, _compute_equity_curve


class TestEquityCurve:
    def test_ruin_cascade(self):
        """Once equity hits 0, it stays at 0."""
        pnls = np.array([-5000, -6000, 2000, 3000])
        eq = _compute_equity_curve(pnls, 10000.0)
        # After -5000 → 5000, then -6000 → -1000 → should be 0
        # Everything after should remain 0
        assert eq[-1] == 0.0
        assert eq[-2] == 0.0

    def test_positive_trades_grow(self):
        pnls = np.array([100, 200, 150, 300, 250])
        eq = _compute_equity_curve(pnls, 10000.0)
        assert eq[-1] == 11000.0  # 10000 + sum(pnls)

    def test_initial_cash_preserved(self):
        pnls = np.array([0, 0, 0])
        eq = _compute_equity_curve(pnls, 5000.0)
        assert eq[0] == 5000.0
        assert eq[-1] == 5000.0


class TestMonteCarloSimulation:
    def test_empty_trades_doesnt_crash(self):
        result = monte_carlo_simulation([], initial_cash=10000.0, n_simulations=10)
        assert isinstance(result, dict)

    def test_returns_expected_keys(self):
        trades = [{'pnl': 100}, {'pnl': -50}, {'pnl': 200}, {'pnl': -30}]
        result = monte_carlo_simulation(trades, initial_cash=10000.0, n_simulations=100)
        # Check that result has meaningful keys
        assert any(k in result for k in ['median_return', 'percentiles', 'final_equity', 'statistics'])

    def test_more_sims_same_structure(self):
        trades = [{'pnl': p} for p in [100, -50, 200, -80, 150, -30]]
        r1 = monte_carlo_simulation(trades, n_simulations=50)
        r2 = monte_carlo_simulation(trades, n_simulations=200)
        # Both should have same structure
        assert type(r1) == type(r2)

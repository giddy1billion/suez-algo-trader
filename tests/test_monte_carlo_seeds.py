"""Tests for Monte Carlo simulation seed behavior (determinism and randomness)."""

import numpy as np
import pytest

from backtesting.monte_carlo import monte_carlo_simulation


@pytest.fixture
def sample_trades():
    """Simple trade list for MC testing."""
    return [
        {"pnl": 150.0, "return": 0.015},
        {"pnl": -80.0, "return": -0.008},
        {"pnl": 200.0, "return": 0.020},
        {"pnl": -50.0, "return": -0.005},
        {"pnl": 300.0, "return": 0.030},
        {"pnl": -120.0, "return": -0.012},
        {"pnl": 100.0, "return": 0.010},
        {"pnl": -30.0, "return": -0.003},
        {"pnl": 250.0, "return": 0.025},
        {"pnl": -90.0, "return": -0.009},
    ]


class TestMonteCarloSeeds:
    """Verify deterministic and random seed behavior."""

    def test_monte_carlo_deterministic_with_seed(self, sample_trades):
        """Same seed must produce identical results across runs."""
        result1 = monte_carlo_simulation(sample_trades, seed=42, n_simulations=500)
        result2 = monte_carlo_simulation(sample_trades, seed=42, n_simulations=500)

        assert result1["expected_return"] == result2["expected_return"]
        assert result1["median_return"] == result2["median_return"]
        assert result1["probability_of_profit"] == result2["probability_of_profit"]
        assert result1["median_max_drawdown"] == result2["median_max_drawdown"]

        # Equity curve percentiles must match exactly
        for key in result1["equity_curves_summary"]:
            np.testing.assert_array_equal(
                result1["equity_curves_summary"][key],
                result2["equity_curves_summary"][key],
                err_msg=f"Equity curve {key} differs with same seed",
            )

    def test_monte_carlo_random_without_seed(self, sample_trades):
        """seed=None must produce different results across runs."""
        result1 = monte_carlo_simulation(sample_trades, seed=None, n_simulations=500)
        result2 = monte_carlo_simulation(sample_trades, seed=None, n_simulations=500)

        # With 500 simulations and no seed, it's astronomically unlikely
        # that all percentile values match. Check equity curves differ.
        curves1 = result1["equity_curves_summary"]["p50"]
        curves2 = result2["equity_curves_summary"]["p50"]

        # At least one value should differ
        assert curves1 != curves2, (
            "Two unseeded MC runs produced identical p50 equity curves — "
            "seed=None may not be providing randomness"
        )

"""
Monte Carlo Simulation — Equity curve robustness testing.
Shuffles trade sequences to build confidence intervals on outcomes.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _compute_equity_curve(pnls: np.ndarray, initial_cash: float) -> np.ndarray:
    """Compute equity curve from PnL sequence. Once ruined (<=0), stays at 0."""
    equity = np.empty(len(pnls) + 1)
    equity[0] = initial_cash
    equity[1:] = initial_cash + np.cumsum(pnls)
    # Once equity hits 0 or below, it stays at 0 (ruin is permanent)
    ruin_idx = np.where(equity <= 0)[0]
    if len(ruin_idx) > 0:
        first_ruin = ruin_idx[0]
        equity[first_ruin:] = 0.0
    return equity


def _compute_max_drawdown_from_equity(equity: np.ndarray) -> float:
    """Max drawdown as a fraction from an equity curve."""
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(peak > 0, (equity - peak) / peak, 0.0)
    if len(drawdowns) == 0:
        return 0.0
    return float(np.min(drawdowns))


def monte_carlo_simulation(
    trades: List[Dict[str, Any]],
    initial_cash: float = 10000.0,
    n_simulations: int = 1000,
    confidence_levels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run Monte Carlo simulation by shuffling trade order.

    Builds confidence intervals on final equity and drawdown by randomly
    reordering the historical trade sequence many times.

    Args:
        trades: List of trade dicts with at minimum a 'pnl' key.
            Can also have a 'return' key.
        initial_cash: Starting capital for equity curve simulation.
        n_simulations: Number of random shuffles to perform.
        confidence_levels: Percentile levels to compute (default: [5, 25, 50, 75, 95]).

    Returns:
        Dict with simulation statistics and percentile equity curves.
    """
    if confidence_levels is None:
        confidence_levels = [5, 25, 50, 75, 95]

    n_trades = len(trades)

    if n_trades == 0:
        logger.warning("No trades provided for Monte Carlo simulation")
        return _empty_mc_result(n_simulations, confidence_levels)

    # Extract PnL and returns arrays
    pnls = np.array([t.get("pnl", 0.0) for t in trades], dtype=float)
    returns = np.array([t.get("return", 0.0) for t in trades], dtype=float)

    # Storage for simulation results
    final_returns = np.empty(n_simulations)
    max_drawdowns = np.empty(n_simulations)
    equity_curves = np.empty((n_simulations, n_trades + 1))

    rng = np.random.default_rng()

    for i in range(n_simulations):
        # Shuffle trade order
        indices = rng.permutation(n_trades)
        shuffled_pnls = pnls[indices]

        # Compute equity curve
        equity = _compute_equity_curve(shuffled_pnls, initial_cash)
        equity_curves[i] = equity

        # Final return
        final_value = equity[-1]
        final_returns[i] = (final_value - initial_cash) / initial_cash

        # Max drawdown
        max_drawdowns[i] = _compute_max_drawdown_from_equity(equity)

    # Compute statistics
    median_return = float(np.percentile(final_returns, 50))
    p5_return = float(np.percentile(final_returns, 5))
    p25_return = float(np.percentile(final_returns, 25))
    p75_return = float(np.percentile(final_returns, 75))
    p95_return = float(np.percentile(final_returns, 95))

    median_max_dd = float(np.percentile(max_drawdowns, 50))
    # Worst-case drawdown is the 95th percentile of drawdown magnitude (most negative)
    p5_max_dd = float(np.percentile(max_drawdowns, 5))

    probability_of_profit = float(np.sum(final_returns > 0) / n_simulations)
    # Ruin: equity drops below 50% of initial at any point
    min_equity_per_sim = np.min(equity_curves, axis=1)
    probability_of_ruin = float(np.sum(min_equity_per_sim < initial_cash * 0.5) / n_simulations)

    expected_return = float(np.mean(final_returns))
    return_std = float(np.std(final_returns, ddof=1)) if n_simulations > 1 else 0.0

    # Percentile equity curves for plotting
    equity_curves_summary = {}
    for p in confidence_levels:
        equity_curves_summary[f"p{p}"] = np.percentile(equity_curves, p, axis=0).tolist()

    logger.info(
        "Monte Carlo complete: %d simulations, %d trades, "
        "median return=%.4f, P(profit)=%.1f%%, P(ruin)=%.1f%%",
        n_simulations,
        n_trades,
        median_return,
        probability_of_profit * 100,
        probability_of_ruin * 100,
    )

    return {
        "n_simulations": n_simulations,
        "n_trades": n_trades,
        "median_return": median_return,
        "p5_return": p5_return,
        "p25_return": p25_return,
        "p75_return": p75_return,
        "p95_return": p95_return,
        "median_max_drawdown": median_max_dd,
        "p5_max_drawdown": p5_max_dd,
        "probability_of_profit": probability_of_profit,
        "probability_of_ruin": probability_of_ruin,
        "expected_return": expected_return,
        "return_std": return_std,
        "equity_curves_summary": equity_curves_summary,
    }


def _empty_mc_result(
    n_simulations: int, confidence_levels: List[int]
) -> Dict[str, Any]:
    """Return empty result when no trades are available."""
    equity_curves_summary = {f"p{p}": [] for p in confidence_levels}
    return {
        "n_simulations": n_simulations,
        "n_trades": 0,
        "median_return": 0.0,
        "p5_return": 0.0,
        "p25_return": 0.0,
        "p75_return": 0.0,
        "p95_return": 0.0,
        "median_max_drawdown": 0.0,
        "p5_max_drawdown": 0.0,
        "probability_of_profit": 0.0,
        "probability_of_ruin": 0.0,
        "expected_return": 0.0,
        "return_std": 0.0,
        "equity_curves_summary": equity_curves_summary,
    }


def monte_carlo_from_backtest(
    df: pd.DataFrame,
    fast_ema: int = 12,
    slow_ema: int = 26,
    initial_cash: float = 10000.0,
    fees: float = 0.001,
    n_simulations: int = 1000,
) -> Dict[str, Any]:
    """Run EMA crossover backtest then Monte Carlo on the resulting trades.

    Uses the numpy EMA crossover backtest logic (same as
    vbt_adapter._numpy_ema_crossover_backtest) to generate trades,
    then runs Monte Carlo simulation on those trades.

    Args:
        df: DataFrame with 'close' column.
        fast_ema: Fast EMA period.
        slow_ema: Slow EMA period.
        initial_cash: Starting capital.
        fees: Trading fee fraction.
        n_simulations: Number of Monte Carlo shuffles.

    Returns:
        Dict with Monte Carlo results plus backtest metadata.
    """
    if "close" not in df.columns:
        logger.error("DataFrame must have a 'close' column")
        return _empty_mc_result(n_simulations, [5, 25, 50, 75, 95])

    close = df["close"].values.astype(float)
    n = len(close)

    if n < slow_ema + 2:
        logger.warning(
            "Insufficient data for EMA backtest: need at least %d bars, got %d",
            slow_ema + 2,
            n,
        )
        return _empty_mc_result(n_simulations, [5, 25, 50, 75, 95])

    # Calculate EMAs
    fast = pd.Series(close).ewm(span=fast_ema, adjust=False).mean().values
    slow_arr = pd.Series(close).ewm(span=slow_ema, adjust=False).mean().values

    # Crossover signals
    fast_above = fast > slow_arr
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    entries[1:] = fast_above[1:] & ~fast_above[:-1]
    exits[1:] = ~fast_above[1:] & fast_above[:-1]

    # Simulate trades
    trades: List[Dict[str, Any]] = []
    cash = initial_cash
    position = 0.0
    entry_price = 0.0

    for i in range(n):
        if entries[i] and position == 0:
            invest_amount = cash
            qty = (invest_amount * (1 - fees)) / close[i]
            position = qty
            entry_price = close[i]
            cash -= invest_amount
        elif exits[i] and position > 0:
            proceeds = position * close[i] * (1 - fees)
            pnl = proceeds - (position * entry_price)
            trades.append({
                "entry_price": entry_price,
                "exit_price": close[i],
                "pnl": pnl,
                "return": (close[i] - entry_price) / entry_price,
            })
            cash += proceeds
            position = 0.0

    if not trades:
        logger.warning("EMA backtest produced no trades")
        return _empty_mc_result(n_simulations, [5, 25, 50, 75, 95])

    logger.info(
        "EMA backtest produced %d trades, running Monte Carlo with %d simulations",
        len(trades),
        n_simulations,
    )

    # Run Monte Carlo
    result = monte_carlo_simulation(
        trades=trades,
        initial_cash=initial_cash,
        n_simulations=n_simulations,
    )

    # Add backtest context
    result["backtest_params"] = {
        "fast_ema": fast_ema,
        "slow_ema": slow_ema,
        "fees": fees,
    }
    result["backtest_n_trades"] = len(trades)
    result["backtest_total_pnl"] = float(sum(t["pnl"] for t in trades))

    return result

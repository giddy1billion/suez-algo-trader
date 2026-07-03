"""
Walk-Forward Optimization — Rolling window train/validate/step approach.
Produces realistic out-of-sample performance estimates by never peeking ahead.
"""

import itertools
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _compute_sharpe(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    """Annualized Sharpe ratio from an array of per-trade returns."""
    if len(returns) < 2:
        return 0.0
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0
    return float(mean / std * np.sqrt(periods_per_year))


def _compute_max_drawdown(returns: np.ndarray) -> float:
    """Max drawdown from a sequence of per-trade returns (as fractions)."""
    if len(returns) == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    return float(np.min(drawdowns))


def _compute_cumulative_return(returns: np.ndarray) -> float:
    """Cumulative return from per-trade returns."""
    if len(returns) == 0:
        return 0.0
    return float(np.prod(1.0 + returns) - 1.0)


class WalkForwardOptimizer:
    """Rolling window walk-forward optimizer.

    Splits data into sequential train/test windows, optimizes parameters
    on the train set, then evaluates out-of-sample on the test set.
    """

    def __init__(
        self,
        train_window: int = 500,
        test_window: int = 100,
        step: int = 100,
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step = step

    def run(
        self,
        df: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame, Dict[str, Any]], List[Dict]],
        param_grid: Dict[str, List[Any]],
        metric: str = "return",
    ) -> Dict[str, Any]:
        """Run walk-forward optimization.

        Args:
            df: DataFrame with at minimum a 'close' column.
            strategy_fn: Function taking (df_slice, params_dict) and returning
                a list of trade dicts with keys: entry_price, exit_price, pnl, return.
            param_grid: Dict mapping param names to lists of values to sweep.
            metric: Which trade metric to optimize on ('return', 'sharpe', 'pnl').

        Returns:
            Dict with OOS performance metrics and diagnostics.
        """
        n = len(df)
        min_required = self.train_window + self.test_window
        if n < min_required:
            logger.warning(
                "Insufficient data for walk-forward: need %d bars, got %d",
                min_required,
                n,
            )
            return self._empty_result()

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combos = [
            dict(zip(param_names, combo))
            for combo in itertools.product(*param_values)
        ]

        if not param_combos:
            logger.warning("Empty param_grid provided")
            return self._empty_result()

        oos_trades: List[Dict] = []
        best_params_per_window: List[Dict] = []
        is_returns_all: List[float] = []
        n_windows = 0

        start = 0
        while start + min_required <= n:
            train_end = start + self.train_window
            test_end = min(train_end + self.test_window, n)

            train_df = df.iloc[start:train_end].reset_index(drop=True)
            test_df = df.iloc[train_end:test_end].reset_index(drop=True)

            # --- In-sample optimization ---
            best_score = -np.inf
            best_params = param_combos[0]
            best_is_return = 0.0

            for params in param_combos:
                try:
                    trades = strategy_fn(train_df, params)
                except Exception as e:
                    logger.debug("Strategy error with params %s: %s", params, e)
                    continue

                if not trades:
                    continue

                score = self._score_trades(trades, metric)
                if score > best_score:
                    best_score = score
                    best_params = params
                    returns_arr = np.array([t.get("return", 0.0) for t in trades])
                    best_is_return = float(np.mean(returns_arr)) if len(returns_arr) > 0 else 0.0

            best_params_per_window.append(best_params)
            is_returns_all.append(best_is_return)

            # --- Out-of-sample evaluation ---
            try:
                oos_window_trades = strategy_fn(test_df, best_params)
            except Exception as e:
                logger.debug("OOS strategy error: %s", e)
                oos_window_trades = []

            oos_trades.extend(oos_window_trades)
            n_windows += 1

            # Step forward
            start += self.step

        if n_windows == 0:
            logger.warning("No walk-forward windows could be formed")
            return self._empty_result()

        # --- Compute OOS metrics ---
        oos_returns = np.array([t.get("return", 0.0) for t in oos_trades])
        oos_pnls = np.array([t.get("pnl", 0.0) for t in oos_trades])

        oos_cumulative_return = _compute_cumulative_return(oos_returns)
        oos_sharpe = _compute_sharpe(oos_returns)
        oos_max_dd = _compute_max_drawdown(oos_returns)
        oos_win_rate = float(np.sum(oos_pnls > 0) / len(oos_pnls)) if len(oos_pnls) > 0 else 0.0

        # Param stability: fraction of windows that chose the most common params
        params_tuples = [tuple(sorted(p.items())) for p in best_params_per_window]
        most_common_count = Counter(params_tuples).most_common(1)[0][1]
        param_stability = most_common_count / n_windows

        is_return = float(np.mean(is_returns_all)) if is_returns_all else 0.0

        logger.info(
            "Walk-forward complete: %d windows, %d OOS trades, OOS return=%.4f, OOS Sharpe=%.2f",
            n_windows,
            len(oos_trades),
            oos_cumulative_return,
            oos_sharpe,
        )

        return {
            "oos_trades": oos_trades,
            "oos_return": oos_cumulative_return,
            "oos_sharpe": oos_sharpe,
            "oos_max_drawdown": oos_max_dd,
            "oos_win_rate": oos_win_rate,
            "best_params_per_window": best_params_per_window,
            "param_stability": param_stability,
            "n_windows": n_windows,
            "is_return": is_return,
        }

    def _score_trades(self, trades: List[Dict], metric: str) -> float:
        """Score a set of trades by the given metric."""
        if not trades:
            return -np.inf

        returns = np.array([t.get("return", 0.0) for t in trades])

        if metric == "sharpe":
            return _compute_sharpe(returns)
        elif metric == "pnl":
            return float(np.sum([t.get("pnl", 0.0) for t in trades]))
        else:  # default: cumulative return
            return _compute_cumulative_return(returns)

    def _empty_result(self) -> Dict[str, Any]:
        """Return an empty result dict when optimization cannot proceed."""
        return {
            "oos_trades": [],
            "oos_return": 0.0,
            "oos_sharpe": 0.0,
            "oos_max_drawdown": 0.0,
            "oos_win_rate": 0.0,
            "best_params_per_window": [],
            "param_stability": 0.0,
            "n_windows": 0,
            "is_return": 0.0,
        }


def _ema_crossover_strategy(df: pd.DataFrame, params: Dict[str, Any]) -> List[Dict]:
    """EMA crossover strategy compatible with WalkForwardOptimizer.

    Args:
        df: DataFrame with 'close' column.
        params: Dict with 'fast_ema', 'slow_ema', and optionally 'fees'.

    Returns:
        List of trade dicts.
    """
    fast_ema = params.get("fast_ema", 12)
    slow_ema = params.get("slow_ema", 26)
    fees = params.get("fees", 0.001)

    if fast_ema >= slow_ema:
        return []

    close = df["close"].values.astype(float)
    n = len(close)

    if n < slow_ema + 2:
        return []

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
    trades: List[Dict] = []
    position = False
    entry_price = 0.0

    for i in range(n):
        if entries[i] and not position:
            entry_price = close[i]
            position = True
        elif exits[i] and position:
            exit_price = close[i]
            gross_return = (exit_price - entry_price) / entry_price
            net_return = gross_return - 2 * fees  # entry + exit fee
            pnl = entry_price * net_return  # pnl per unit
            trades.append({
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "return": net_return,
            })
            position = False

    return trades


def walk_forward_ema_backtest(
    df: pd.DataFrame,
    train_window: int = 500,
    test_window: int = 100,
    step: int = 100,
    initial_cash: float = 10000.0,
    fees: float = 0.001,
) -> dict:
    """Walk-forward optimization using EMA crossover strategy.

    Uses fast EMA range [8, 10, ..., 28] and slow EMA range [20, 22, ..., 54]
    matching the parameter space from vbt_adapter._numpy_ema_crossover_backtest.

    Args:
        df: DataFrame with 'close' column (OHLCV data).
        train_window: Number of bars for the training window.
        test_window: Number of bars for the test window.
        step: Step size to advance between windows.
        initial_cash: Starting capital (used for context/logging).
        fees: Trading fee fraction (applied on entry and exit).

    Returns:
        Dict with walk-forward results including OOS metrics.
    """
    param_grid = {
        "fast_ema": list(range(8, 30, 2)),
        "slow_ema": list(range(20, 56, 2)),
        "fees": [fees],
    }

    optimizer = WalkForwardOptimizer(
        train_window=train_window,
        test_window=test_window,
        step=step,
    )

    result = optimizer.run(
        df=df,
        strategy_fn=_ema_crossover_strategy,
        param_grid=param_grid,
        metric="return",
    )

    # Add context info
    result["initial_cash"] = initial_cash
    result["fees"] = fees
    result["strategy"] = "ema_crossover"
    result["param_ranges"] = {
        "fast_ema": "8-28 step 2",
        "slow_ema": "20-54 step 2",
    }

    return result

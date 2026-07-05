"""
Walk-Forward Parameter Validation — Prevents overfitting of backtest parameters.

Before any parameter change (EMA periods, fees, stops) can be promoted to
production config, it must survive out-of-sample walk-forward testing.

This module integrates:
- WalkForwardOptimizer (existing) for rolling train/test validation
- InstrumentRegistry for proper symbol classification
- LayeredConfig for reading/writing validated parameters

Usage:
    from backtesting.param_validator import validate_params, ValidationResult

    result = validate_params(
        df=crypto_data,
        symbol="BTC/USD",
        candidate_params={"fast_ema": 21, "slow_ema": 55},
    )
    if result.passed:
        # Safe to promote to LayeredConfig
        result.promote()
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtesting.walk_forward import WalkForwardOptimizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Minimum OOS Sharpe to consider params viable (not great, just not destructive)
MIN_OOS_SHARPE = -0.5
# Minimum OOS return to pass
MIN_OOS_RETURN = -0.05
# Minimum parameter stability (fraction of windows choosing similar params)
MIN_PARAM_STABILITY = 0.3
# Minimum OOS trades to be statistically meaningful
MIN_OOS_TRADES = 10


@dataclass
class ValidationResult:
    """Result of walk-forward parameter validation."""

    symbol: str
    asset_class: str
    candidate_params: Dict[str, Any]
    passed: bool
    oos_sharpe: float = 0.0
    oos_return: float = 0.0
    oos_max_drawdown: float = 0.0
    oos_win_rate: float = 0.0
    oos_trades: int = 0
    param_stability: float = 0.0
    n_windows: int = 0
    best_params_per_window: List[Dict] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)

    def promote(self) -> bool:
        """Promote validated params to LayeredConfig at EXCHANGE level."""
        if not self.passed:
            logger.warning(
                "param_validator.cannot_promote",
                symbol=self.symbol,
                reasons=self.rejection_reasons,
            )
            return False

        from src.config.backtest_params import set_asset_class_override
        for key, value in self.candidate_params.items():
            set_asset_class_override(self.asset_class, key, value)

        logger.info(
            "param_validator.promoted",
            asset_class=self.asset_class,
            params=self.candidate_params,
            oos_sharpe=round(self.oos_sharpe, 3),
            oos_return=round(self.oos_return, 4),
        )
        return True

    def summary(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        lines = [
            f"\n{'='*60}",
            f"WALK-FORWARD VALIDATION: {self.symbol} ({self.asset_class})",
            f"{'='*60}",
            f"Status: {status}",
            f"Candidate Params: {self.candidate_params}",
            f"{'─'*60}",
            f"OOS Sharpe:      {self.oos_sharpe:.3f}",
            f"OOS Return:      {self.oos_return:.2%}",
            f"OOS Max DD:      {self.oos_max_drawdown:.2%}",
            f"OOS Win Rate:    {self.oos_win_rate:.1%}",
            f"OOS Trades:      {self.oos_trades}",
            f"Param Stability: {self.param_stability:.1%}",
            f"Windows:         {self.n_windows}",
        ]
        if self.rejection_reasons:
            lines.append(f"{'─'*60}")
            lines.append("Rejection Reasons:")
            for reason in self.rejection_reasons:
                lines.append(f"  • {reason}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


def _ema_strategy_with_stops(df: pd.DataFrame, params: Dict[str, Any]) -> List[Dict]:
    """
    EMA crossover strategy with ATR stops and cooldown for walk-forward.

    Compatible with WalkForwardOptimizer's strategy_fn interface.
    """
    fast_ema = params.get("fast_ema", 12)
    slow_ema = params.get("slow_ema", 26)
    fees = params.get("fees", 0.001)
    cooldown = params.get("cooldown_bars", 0)
    atr_mult = params.get("atr_stop_multiplier", 0.0)

    if fast_ema >= slow_ema:
        return []

    close = df["close"].values.astype(float)
    n = len(close)

    if n < slow_ema + 2:
        return []

    # Calculate EMAs
    fast = pd.Series(close).ewm(span=fast_ema, adjust=False).mean().values
    slow_arr = pd.Series(close).ewm(span=slow_ema, adjust=False).mean().values

    # Compute ATR if stops enabled
    atr = None
    if atr_mult > 0 and "high" in df.columns and "low" in df.columns:
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        atr = np.full(n, np.nan)
        if n >= 14:
            atr[13] = np.mean(tr[:14])
            for i in range(14, n):
                atr[i] = (atr[i - 1] * 13 + tr[i]) / 14

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
    stop_price = 0.0
    last_exit_bar = -cooldown - 1

    for i in range(n):
        # Check stop-loss
        if position and atr_mult > 0 and stop_price > 0:
            check_price = df["low"].iloc[i] if "low" in df.columns else close[i]
            if check_price <= stop_price:
                gross_return = (stop_price - entry_price) / entry_price
                net_return = gross_return - 2 * fees
                trades.append({
                    "entry_price": entry_price,
                    "exit_price": stop_price,
                    "pnl": entry_price * net_return,
                    "return": net_return,
                })
                position = False
                last_exit_bar = i
                continue

        if entries[i] and not position:
            if (i - last_exit_bar) < cooldown:
                continue
            entry_price = close[i]
            position = True
            if atr is not None and not np.isnan(atr[i]):
                stop_price = entry_price - (atr[i] * atr_mult)
            else:
                stop_price = 0.0
        elif exits[i] and position:
            exit_price = close[i]
            gross_return = (exit_price - entry_price) / entry_price
            net_return = gross_return - 2 * fees
            trades.append({
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": entry_price * net_return,
                "return": net_return,
            })
            position = False
            last_exit_bar = i

    return trades


def validate_params(
    df: pd.DataFrame,
    symbol: str,
    candidate_params: Optional[Dict[str, Any]] = None,
    train_window: int = 400,
    test_window: int = 100,
    step: int = 50,
    min_oos_sharpe: float = MIN_OOS_SHARPE,
    min_oos_return: float = MIN_OOS_RETURN,
    min_param_stability: float = MIN_PARAM_STABILITY,
    min_oos_trades: int = MIN_OOS_TRADES,
) -> ValidationResult:
    """
    Validate candidate backtest parameters via walk-forward optimization.

    If candidate_params is provided, tests only those params (single-combo validation).
    Otherwise, runs a full parameter sweep to find the best OOS params.

    Args:
        df: OHLCV DataFrame with sufficient history (>= train + test window).
        symbol: Symbol for InstrumentRegistry classification.
        candidate_params: Specific params to validate (e.g., {"fast_ema": 21, "slow_ema": 55}).
        train_window: Training window size in bars.
        test_window: Test window size in bars.
        step: Step size between windows.
        min_oos_sharpe: Minimum OOS Sharpe to pass.
        min_oos_return: Minimum OOS cumulative return to pass.
        min_param_stability: Minimum fraction of windows selecting similar params.
        min_oos_trades: Minimum OOS trade count for statistical significance.

    Returns:
        ValidationResult with pass/fail decision and OOS metrics.
    """
    from src.market.registry import classify_symbol as _classify
    instrument = _classify(symbol)
    asset_class = instrument.asset_class.value

    # Build parameter grid
    if candidate_params:
        # Single-point validation: test only the candidate
        param_grid = {k: [v] for k, v in candidate_params.items()}
        # Ensure fees are included
        if "fees" not in param_grid:
            from src.config.backtest_params import get_backtest_config
            defaults = get_backtest_config(symbol)
            param_grid["fees"] = [defaults["fees"]]
    else:
        # Full sweep: use appropriate ranges per asset class
        from src.config.backtest_params import get_backtest_config
        defaults = get_backtest_config(symbol)
        if asset_class == "crypto":
            param_grid = {
                "fast_ema": [13, 17, 21, 26, 34],
                "slow_ema": [34, 42, 55, 70, 89],
                "fees": [defaults["fees"]],
                "cooldown_bars": [0, 3, 5],
                "atr_stop_multiplier": [0.0, 2.0, 2.5, 3.0],
            }
        else:
            param_grid = {
                "fast_ema": [8, 10, 12, 15, 18],
                "slow_ema": [20, 26, 30, 35, 40],
                "fees": [defaults["fees"]],
                "cooldown_bars": [0, 2, 3],
                "atr_stop_multiplier": [0.0, 1.5, 2.0, 2.5],
            }

    optimizer = WalkForwardOptimizer(
        train_window=train_window,
        test_window=test_window,
        step=step,
    )

    result = optimizer.run(
        df=df,
        strategy_fn=_ema_strategy_with_stops,
        param_grid=param_grid,
        metric="return",
    )

    # Evaluate pass/fail
    rejection_reasons = []
    oos_trades = len(result.get("oos_trades", []))

    if oos_trades < min_oos_trades:
        rejection_reasons.append(
            f"Insufficient OOS trades: {oos_trades} < {min_oos_trades}"
        )
    if result["oos_sharpe"] < min_oos_sharpe:
        rejection_reasons.append(
            f"OOS Sharpe too low: {result['oos_sharpe']:.3f} < {min_oos_sharpe}"
        )
    if result["oos_return"] < min_oos_return:
        rejection_reasons.append(
            f"OOS return too low: {result['oos_return']:.4f} < {min_oos_return}"
        )
    if result["param_stability"] < min_param_stability:
        rejection_reasons.append(
            f"Param instability: {result['param_stability']:.2f} < {min_param_stability}"
        )

    passed = len(rejection_reasons) == 0

    # Determine the effective candidate params from validation
    effective_params = candidate_params or {}
    if not candidate_params and result["best_params_per_window"]:
        # Use the most common best params across windows
        from collections import Counter
        params_tuples = [tuple(sorted(p.items())) for p in result["best_params_per_window"]]
        most_common = Counter(params_tuples).most_common(1)[0][0]
        effective_params = dict(most_common)

    validation = ValidationResult(
        symbol=symbol,
        asset_class=asset_class,
        candidate_params=effective_params,
        passed=passed,
        oos_sharpe=result["oos_sharpe"],
        oos_return=result["oos_return"],
        oos_max_drawdown=result["oos_max_drawdown"],
        oos_win_rate=result["oos_win_rate"],
        oos_trades=oos_trades,
        param_stability=result["param_stability"],
        n_windows=result["n_windows"],
        best_params_per_window=result["best_params_per_window"],
        rejection_reasons=rejection_reasons,
    )

    logger.info(
        "param_validator.complete",
        symbol=symbol,
        asset_class=asset_class,
        passed=passed,
        oos_sharpe=round(result["oos_sharpe"], 3),
        oos_return=round(result["oos_return"], 4),
        oos_trades=oos_trades,
    )

    return validation


def validate_and_promote(
    df: pd.DataFrame,
    symbol: str,
    candidate_params: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> ValidationResult:
    """
    Validate parameters and auto-promote to LayeredConfig if they pass.

    Convenience wrapper around validate_params() + result.promote().
    """
    result = validate_params(df, symbol, candidate_params, **kwargs)
    if result.passed:
        result.promote()
    return result

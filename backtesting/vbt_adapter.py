"""
VectorBT Integration — Ultra-fast vectorized backtesting.
Leverages NumPy for batch signal evaluation across symbols and timeframes.

Falls back to a pure numpy/pandas vectorized engine if vectorbt/numba
can't be imported (common on Python 3.13 where numba DLLs may fail).
"""

import os
import tempfile

import numpy as np
import pandas as pd
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Numba cache fix — must be set BEFORE numba is imported (vectorbt triggers it).
# In containerized environments the non-root user cannot write .nbi/.nbc cache
# files into site-packages. Redirect to a writable directory.
# ──────────────────────────────────────────────────────────────────────────
if "NUMBA_CACHE_DIR" not in os.environ:
    _numba_cache = os.path.join(tempfile.gettempdir(), ".numba_cache")
    os.makedirs(_numba_cache, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = _numba_cache

# Check if vectorbt is usable
_VBT_AVAILABLE = False
try:
    import vectorbt as vbt
    _VBT_AVAILABLE = True
except (ImportError, OSError):
    logger.debug("vbt.unavailable", msg="Using pure numpy fallback")


# ──────────────────────────────────────────────────────────────────────────
# Pure numpy/pandas fallback (no vectorbt/numba required)
# ──────────────────────────────────────────────────────────────────────────


def _safe_profit_factor(wins: list, losses: list) -> float:
    """Compute profit factor safely avoiding division by zero."""
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0.0
    gross_profit = sum(t['pnl'] for t in wins) if wins else 0.0
    if gross_loss < 1e-9:
        # Cap at 999.99 for downstream safety instead of infinity
        return 999.99 if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute Average True Range over a numpy array of OHLC data."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _numpy_ema_crossover_backtest(
    df: pd.DataFrame,
    fast_ema: int = 12,
    slow_ema: int = 26,
    initial_cash: float = 10000.0,
    fees: float = 0.001,
    risk_per_trade: float = 1.0,
    atr_stop_multiplier: float = 0.0,
    cooldown_bars: int = 0,
    annualization_periods: float = 252.0,
) -> dict:
    """Vectorized EMA crossover backtest using pure numpy/pandas.

    Args:
        risk_per_trade: Fraction of equity to risk per trade (1.0 = all-in, 0.02 = 2%).
        atr_stop_multiplier: If > 0, use ATR-based stop-loss (N × ATR below entry).
            Set to 0 to disable stop-loss (original behavior).
        cooldown_bars: Minimum bars to wait between exit and next entry.
            Set to 0 to disable cooldown (original behavior).
        annualization_periods: Periods per year for Sharpe calculation.
            Equity default ~252 (trades/year proxy). Adjust for crypto/intraday.
    """
    close = df['close'].values.astype(float)
    n = len(close)

    if n < slow_ema + 2:
        return {
            "total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0,
            "win_rate": 0.0, "total_trades": 0, "profit_factor": 0.0,
            "final_value": initial_cash, "trades": [], "equity_curve": [initial_cash],
        }

    # Validate data - drop NaN from close
    if np.any(np.isnan(close)):
        close = pd.Series(close).ffill().bfill().values

    # Calculate EMAs
    fast = pd.Series(close).ewm(span=fast_ema, adjust=False).mean().values
    slow = pd.Series(close).ewm(span=slow_ema, adjust=False).mean().values

    # Compute ATR if stop-loss is enabled
    has_stops = atr_stop_multiplier > 0
    atr = None
    if has_stops:
        high = df['high'].values.astype(float) if 'high' in df.columns else close
        low = df['low'].values.astype(float) if 'low' in df.columns else close
        if np.any(np.isnan(high)):
            high = pd.Series(high).ffill().bfill().values
        if np.any(np.isnan(low)):
            low = pd.Series(low).ffill().bfill().values
        atr = _compute_atr(high, low, close, period=14)

    # Signals: +1 when fast crosses above slow, -1 when below
    fast_above = fast > slow
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    entries[1:] = fast_above[1:] & ~fast_above[:-1]  # cross above
    exits[1:] = ~fast_above[1:] & fast_above[:-1]    # cross below

    # Simulate trades with stop-loss and cooldown
    cash = initial_cash
    position = 0.0
    trades = []
    entry_price = 0.0
    stop_price = 0.0
    last_exit_bar = -cooldown_bars - 1  # Allow immediate first entry

    for i in range(n):
        # Check stop-loss on open position
        if position > 0 and has_stops and stop_price > 0:
            check_price = df['low'].iloc[i] if 'low' in df.columns else close[i]
            if check_price <= stop_price:
                exit_at = stop_price
                proceeds = position * exit_at * (1 - fees)
                pnl = proceeds - (position * entry_price)
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': exit_at,
                    'pnl': pnl,
                    'return': (exit_at - entry_price) / entry_price,
                    'exit_reason': 'stop_loss',
                })
                cash += proceeds
                position = 0.0
                last_exit_bar = i
                continue

        if entries[i] and position == 0:
            # Enforce cooldown
            if (i - last_exit_bar) < cooldown_bars:
                continue
            # Buy — use risk_per_trade fraction of available cash
            invest_amount = cash * min(risk_per_trade, 1.0)
            qty = (invest_amount * (1 - fees)) / close[i]
            position = qty
            entry_price = close[i]
            cash -= invest_amount
            # Set stop-loss
            if has_stops and atr is not None and not np.isnan(atr[i]):
                stop_price = entry_price - (atr[i] * atr_stop_multiplier)
            else:
                stop_price = 0.0
        elif exits[i] and position > 0:
            # Sell on EMA cross
            proceeds = position * close[i] * (1 - fees)
            pnl = proceeds - (position * entry_price)
            trades.append({
                'entry_price': entry_price,
                'exit_price': close[i],
                'pnl': pnl,
                'return': (close[i] - entry_price) / entry_price,
                'exit_reason': 'signal',
            })
            cash += proceeds
            position = 0.0
            last_exit_bar = i

    # Final value
    final_value = cash + position * close[-1] * (1 - fees) if position > 0 else cash
    total_return = (final_value - initial_cash) / initial_cash

    # Metrics
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    win_rate = len(wins) / len(trades) if trades else 0.0

    # Trade Sharpe scaled by annualization_periods
    if trades:
        returns = np.array([t['return'] for t in trades])
        sharpe = (returns.mean() / returns.std() * np.sqrt(annualization_periods)) if returns.std() > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown from equity curve
    equity = np.full(n, initial_cash, dtype=float)
    pos = 0.0
    c = initial_cash
    ep = 0.0
    sp = 0.0
    last_exit_eq = -cooldown_bars - 1
    for i in range(n):
        # Stop-loss exit in equity tracking
        if pos > 0 and has_stops and sp > 0:
            check_p = df['low'].iloc[i] if 'low' in df.columns else close[i]
            if check_p <= sp:
                c += pos * sp * (1 - fees)
                pos = 0.0
                last_exit_eq = i
                equity[i] = c
                continue

        if entries[i] and pos == 0 and (i - last_exit_eq) >= cooldown_bars:
            invest_amount = c * min(risk_per_trade, 1.0)
            qty = (invest_amount * (1 - fees)) / close[i]
            pos = qty
            ep = close[i]
            c -= invest_amount
            if has_stops and atr is not None and not np.isnan(atr[i]):
                sp = ep - (atr[i] * atr_stop_multiplier)
            else:
                sp = 0.0
        elif exits[i] and pos > 0:
            c += pos * close[i] * (1 - fees)
            pos = 0.0
            last_exit_eq = i
        equity[i] = c + pos * close[i]

    running_max = np.maximum.accumulate(equity)
    # Guard against division by zero when running_max is 0
    running_max = np.where(running_max > 0, running_max, 1.0)
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': abs(max_drawdown),
        'win_rate': win_rate,
        'total_trades': len(trades),
        'profit_factor': _safe_profit_factor(wins, losses),
        'final_value': final_value,
        'trades': trades,
    }


def _numpy_parameter_sweep(
    df: pd.DataFrame,
    fast_range: range = range(8, 30, 2),
    slow_range: range = range(20, 56, 2),
    initial_cash: float = 10000.0,
    fees: float = 0.001,
    risk_per_trade: float = 1.0,
    atr_stop_multiplier: float = 0.0,
    cooldown_bars: int = 0,
    annualization_periods: float = 252.0,
) -> pd.DataFrame:
    """Brute-force parameter sweep using numpy fallback.
    Default ranges include (12, 26) for direct comparison with single backtest.
    """
    results = []
    for fast_w in fast_range:
        for slow_w in slow_range:
            if fast_w >= slow_w:
                continue
            try:
                metrics = _numpy_ema_crossover_backtest(
                    df, fast_w, slow_w, initial_cash, fees,
                    risk_per_trade=risk_per_trade,
                    atr_stop_multiplier=atr_stop_multiplier,
                    cooldown_bars=cooldown_bars,
                    annualization_periods=annualization_periods,
                )
                results.append({
                    'fast_window': fast_w,
                    'slow_window': slow_w,
                    'total_return': metrics['total_return'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
                    'win_rate': metrics['win_rate'],
                    'total_trades': metrics['total_trades'],
                })
            except Exception:
                continue

    return pd.DataFrame(results) if results else pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────
# Public API (delegates to vectorbt if available, numpy otherwise)
# ──────────────────────────────────────────────────────────────────────────

def vectorbt_momentum_backtest(
    df: pd.DataFrame,
    fast_ema: int = 12,
    slow_ema: int = 26,
    initial_cash: float = 10000.0,
    fees: float = 0.001,
    risk_per_trade: float = 1.0,
    atr_stop_multiplier: float = 0.0,
    cooldown_bars: int = 0,
    annualization_periods: float = 252.0,
) -> dict:
    """
    Vectorized EMA crossover backtest.
    Uses vectorbt if available, falls back to pure numpy.

    Args:
        df: OHLCV DataFrame with DatetimeIndex and 'close' column
        fast_ema: Fast EMA period
        slow_ema: Slow EMA period
        initial_cash: Starting capital
        fees: Trading fees (0.001 = 0.1%)
        risk_per_trade: Fraction of equity to invest per trade (1.0 = all-in)
        atr_stop_multiplier: ATR-based stop-loss multiplier (0 = disabled)
        cooldown_bars: Minimum bars between exit and next entry (0 = disabled)
        annualization_periods: Periods/year for Sharpe ratio annualization

    Returns:
        Dict with performance metrics
    """
    if not _VBT_AVAILABLE:
        logger.debug("vbt.using_numpy_fallback")
        return _numpy_ema_crossover_backtest(
            df, fast_ema, slow_ema, initial_cash, fees, risk_per_trade,
            atr_stop_multiplier=atr_stop_multiplier,
            cooldown_bars=cooldown_bars,
            annualization_periods=annualization_periods,
        )

    close = df['close']

    # EMA crossover signals
    fast = vbt.MA.run(close, fast_ema, short_name='fast', ewm=True)
    slow = vbt.MA.run(close, slow_ema, short_name='slow', ewm=True)

    entries = fast.ma_crossed_above(slow)
    exits = fast.ma_crossed_below(slow)

    # Run portfolio simulation
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=initial_cash,
        fees=fees,
        freq='1h',
    )

    stats = portfolio.stats()

    return {
        'total_return': portfolio.total_return(),
        'sharpe_ratio': portfolio.sharpe_ratio(),
        'max_drawdown': portfolio.max_drawdown(),
        'win_rate': portfolio.trades.win_rate() if len(portfolio.trades.records_readable) > 0 else 0,
        'total_trades': portfolio.trades.count(),
        'profit_factor': portfolio.trades.profit_factor() if len(portfolio.trades.records_readable) > 0 else 0,
        'final_value': portfolio.final_value(),
        'stats': stats,
        'portfolio': portfolio,
    }


def vectorbt_parameter_sweep(
    df: pd.DataFrame,
    fast_range: range = range(8, 30, 2),
    slow_range: range = range(20, 56, 2),
    initial_cash: float = 10000.0,
    fees: float = 0.001,
    risk_per_trade: float = 1.0,
    atr_stop_multiplier: float = 0.0,
    cooldown_bars: int = 0,
    annualization_periods: float = 252.0,
) -> pd.DataFrame:
    """
    Parameter optimization — tests ALL EMA combinations.
    Uses vectorbt's batch engine if available, otherwise numpy loop.

    Default ranges include (12, 26) for direct comparison with single backtest.

    Returns:
        DataFrame with columns: fast_window, slow_window, total_return,
        sharpe_ratio, max_drawdown, win_rate, total_trades
    """
    if not _VBT_AVAILABLE:
        logger.debug("vbt.sweep_numpy_fallback")
        return _numpy_parameter_sweep(
            df, fast_range, slow_range, initial_cash, fees,
            risk_per_trade=risk_per_trade,
            atr_stop_multiplier=atr_stop_multiplier,
            cooldown_bars=cooldown_bars,
            annualization_periods=annualization_periods,
        )

    # Run one parameter pair at a time to avoid vectorbt index-broadcast
    # alignment failures such as "Index at position 0 could not be aligned".
    results = []
    for fast_w in fast_range:
        for slow_w in slow_range:
            if fast_w >= slow_w:
                continue
            try:
                metrics = vectorbt_momentum_backtest(
                    df,
                    fast_ema=fast_w,
                    slow_ema=slow_w,
                    initial_cash=initial_cash,
                    fees=fees,
                    risk_per_trade=risk_per_trade,
                    atr_stop_multiplier=atr_stop_multiplier,
                    cooldown_bars=cooldown_bars,
                    annualization_periods=annualization_periods,
                )
            except Exception as e:
                logger.warning(
                    "vbt.sweep_combo_failed",
                    fast_window=fast_w,
                    slow_window=slow_w,
                    error=str(e),
                )
                continue

            results.append({
                "fast_window": fast_w,
                "slow_window": slow_w,
                "total_return": float(metrics.get("total_return", 0.0)),
                "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "win_rate": float(metrics.get("win_rate", 0.0)),
                "total_trades": int(metrics.get("total_trades", 0)),
            })

    if results:
        return pd.DataFrame(results)

    logger.warning("vbt.sweep_all_vectorbt_combos_failed", action="fallback_numpy")
    return _numpy_parameter_sweep(
        df, fast_range, slow_range, initial_cash, fees,
        risk_per_trade=risk_per_trade,
        atr_stop_multiplier=atr_stop_multiplier,
        cooldown_bars=cooldown_bars,
        annualization_periods=annualization_periods,
    )


def vectorbt_multi_symbol_backtest(
    data: dict[str, pd.DataFrame],
    fast_ema: int = 12,
    slow_ema: int = 26,
    initial_cash: float = 10000.0,
    use_asset_class_params: bool = True,
) -> dict:
    """
    Run the same strategy across multiple symbols and aggregate results.

    Args:
        data: Dict of symbol -> OHLCV DataFrame
        fast_ema: Fast EMA period (used when use_asset_class_params=False)
        slow_ema: Slow EMA period (used when use_asset_class_params=False)
        initial_cash: Total starting capital (split equally across symbols)
        use_asset_class_params: If True (default), resolve per-symbol params via LayeredConfig

    Returns:
        Combined performance metrics
    """
    results = {}
    per_symbol_cash = initial_cash / len(data) if data else initial_cash

    for symbol, df in data.items():
        try:
            if use_asset_class_params:
                from src.config.backtest_params import get_backtest_config
                params = get_backtest_config(symbol)
                result = vectorbt_momentum_backtest(
                    df,
                    fast_ema=params["fast_ema"],
                    slow_ema=params["slow_ema"],
                    initial_cash=per_symbol_cash,
                    fees=params["fees"],
                    risk_per_trade=params["risk_per_trade"],
                    atr_stop_multiplier=params["atr_stop_multiplier"],
                    cooldown_bars=params["cooldown_bars"],
                    annualization_periods=params["annualization_periods"],
                )
            else:
                result = vectorbt_momentum_backtest(
                    df, fast_ema=fast_ema, slow_ema=slow_ema,
                    initial_cash=per_symbol_cash
                )
            results[symbol] = {
                'return': result['total_return'],
                'sharpe': result['sharpe_ratio'],
                'max_dd': result['max_drawdown'],
                'trades': result['total_trades'],
            }
        except Exception as e:
            logger.error("vbt.symbol_error", symbol=symbol, error=str(e))

    if not results:
        return {}

    returns = [r['return'] for r in results.values()]
    return {
        'per_symbol': results,
        'avg_return': np.mean(returns),
        'total_return': sum(returns),
        'best_symbol': max(results.items(), key=lambda x: x[1]['return'])[0],
        'worst_symbol': min(results.items(), key=lambda x: x[1]['return'])[0],
    }

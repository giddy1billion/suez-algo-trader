"""
VectorBT Integration — Ultra-fast vectorized backtesting.
Leverages NumPy for batch signal evaluation across symbols and timeframes.

Falls back to a pure numpy/pandas vectorized engine if vectorbt/numba
can't be imported (common on Python 3.13 where numba DLLs may fail).
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Check if vectorbt is usable
_VBT_AVAILABLE = False
try:
    import vectorbt as vbt
    # Disable numba caching to prevent "no locator available for file" errors
    # in Docker/overlay filesystems where numba can't reliably stat module files.
    try:
        vbt.settings.caching['disable_machinery'] = True
    except (AttributeError, KeyError, TypeError):
        pass
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


def _numpy_ema_crossover_backtest(
    df: pd.DataFrame,
    fast_ema: int = 12,
    slow_ema: int = 26,
    initial_cash: float = 10000.0,
    fees: float = 0.001,
    risk_per_trade: float = 1.0,
) -> dict:
    """Vectorized EMA crossover backtest using pure numpy/pandas.
    
    Args:
        risk_per_trade: Fraction of equity to risk per trade (1.0 = all-in, 0.02 = 2%)
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

    # Signals: +1 when fast crosses above slow, -1 when below
    fast_above = fast > slow
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    entries[1:] = fast_above[1:] & ~fast_above[:-1]  # cross above
    exits[1:] = ~fast_above[1:] & fast_above[:-1]    # cross below

    # Simulate trades
    cash = initial_cash
    position = 0.0
    trades = []
    entry_price = 0.0

    for i in range(n):
        if entries[i] and position == 0:
            # Buy — use risk_per_trade fraction of available cash
            invest_amount = cash * min(risk_per_trade, 1.0)
            qty = (invest_amount * (1 - fees)) / close[i]
            position = qty
            entry_price = close[i]
            cash -= invest_amount
        elif exits[i] and position > 0:
            # Sell
            proceeds = position * close[i] * (1 - fees)
            pnl = proceeds - (position * entry_price)
            trades.append({
                'entry_price': entry_price,
                'exit_price': close[i],
                'pnl': pnl,
                'return': (close[i] - entry_price) / entry_price,
            })
            cash += proceeds
            position = 0.0

    # Final value
    final_value = cash + position * close[-1] * (1 - fees) if position > 0 else cash
    total_return = (final_value - initial_cash) / initial_cash

    # Metrics
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    win_rate = len(wins) / len(trades) if trades else 0.0

    # Trade Sharpe (not annualized daily Sharpe) — computed from per-trade returns,
    # scaled by sqrt(252) as a rough annualization proxy.
    if trades:
        returns = np.array([t['return'] for t in trades])
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown from equity curve
    equity = np.full(n, initial_cash, dtype=float)
    pos = 0.0
    c = initial_cash
    for i in range(n):
        if entries[i] and pos == 0:
            invest_amount = c * min(risk_per_trade, 1.0)
            qty = (invest_amount * (1 - fees)) / close[i]
            pos = qty
            c -= invest_amount
        elif exits[i] and pos > 0:
            c += pos * close[i] * (1 - fees)
            pos = 0.0
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
    }


def _numpy_parameter_sweep(
    df: pd.DataFrame,
    fast_range: range = range(8, 30, 2),
    slow_range: range = range(20, 56, 2),
    initial_cash: float = 10000.0,
    fees: float = 0.001,
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
                metrics = _numpy_ema_crossover_backtest(df, fast_w, slow_w, initial_cash, fees)
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

    Returns:
        Dict with performance metrics
    """
    if not _VBT_AVAILABLE:
        logger.debug("vbt.using_numpy_fallback")
        return _numpy_ema_crossover_backtest(df, fast_ema, slow_ema, initial_cash, fees, risk_per_trade)

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
        return _numpy_parameter_sweep(df, fast_range, slow_range, initial_cash, fees)

    close = df['close']

    # Generate all EMA combinations at once
    fast_emas = vbt.MA.run(close, list(fast_range), short_name='fast', ewm=True)
    slow_emas = vbt.MA.run(close, list(slow_range), short_name='slow', ewm=True)

    # All crossover combinations
    entries = fast_emas.ma_crossed_above(slow_emas)
    exits = fast_emas.ma_crossed_below(slow_emas)

    # Run all portfolios at once
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=initial_cash,
        fees=fees,
        freq='1h',
    )

    # Collect metrics into DataFrame
    total_returns = portfolio.total_return()
    sharpe_ratios = portfolio.sharpe_ratio()
    max_drawdowns = portfolio.max_drawdown()

    results = []
    if hasattr(total_returns, 'index') and hasattr(total_returns.index, 'to_frame'):
        idx_df = total_returns.index.to_frame(index=False)
        for i in range(len(idx_df)):
            results.append({
                'fast_window': idx_df.iloc[i, 0],
                'slow_window': idx_df.iloc[i, 1],
                'total_return': total_returns.iloc[i],
                'sharpe_ratio': sharpe_ratios.iloc[i],
                'max_drawdown': max_drawdowns.iloc[i],
                'win_rate': 0,
                'total_trades': 0,
            })
    else:
        # Scalar result
        results.append({
            'fast_window': list(fast_range)[0],
            'slow_window': list(slow_range)[0],
            'total_return': float(total_returns) if np.isscalar(total_returns) else 0,
            'sharpe_ratio': float(sharpe_ratios) if np.isscalar(sharpe_ratios) else 0,
            'max_drawdown': float(max_drawdowns) if np.isscalar(max_drawdowns) else 0,
            'win_rate': 0,
            'total_trades': 0,
        })

    return pd.DataFrame(results)


def vectorbt_multi_symbol_backtest(
    data: dict[str, pd.DataFrame],
    fast_ema: int = 12,
    slow_ema: int = 26,
    initial_cash: float = 10000.0,
) -> dict:
    """
    Run the same strategy across multiple symbols and aggregate results.

    Args:
        data: Dict of symbol -> OHLCV DataFrame
        fast_ema: Fast EMA period
        slow_ema: Slow EMA period

    Returns:
        Combined performance metrics
    """
    results = {}
    per_symbol_cash = initial_cash / len(data) if data else initial_cash

    for symbol, df in data.items():
        try:
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

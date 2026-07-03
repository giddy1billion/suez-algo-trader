"""
VectorBT Integration — Ultra-fast vectorized backtesting.
Leverages NumPy for batch signal evaluation across symbols and timeframes.
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


def vectorbt_momentum_backtest(
    df: pd.DataFrame,
    fast_ema: int = 12,
    slow_ema: int = 26,
    initial_cash: float = 10000.0,
    fees: float = 0.001,
) -> dict:
    """
    Vectorized EMA crossover backtest using vectorbt.

    Args:
        df: OHLCV DataFrame with DatetimeIndex and 'close' column
        fast_ema: Fast EMA period
        slow_ema: Slow EMA period
        initial_cash: Starting capital
        fees: Trading fees (0.001 = 0.1%)

    Returns:
        Dict with performance metrics
    """
    try:
        import vectorbt as vbt
    except ImportError:
        raise ImportError(
            "vectorbt is not installed. Install with: pip install vectorbt"
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
    fast_range: range = range(5, 30, 5),
    slow_range: range = range(20, 100, 10),
    initial_cash: float = 10000.0,
    fees: float = 0.001,
) -> pd.DataFrame:
    """
    Vectorized parameter optimization — tests ALL combinations simultaneously.
    This is where vectorbt shines: testing 100+ parameter combos in seconds.

    Args:
        df: OHLCV DataFrame
        fast_range: Range of fast EMA periods to test
        slow_range: Range of slow EMA periods to test

    Returns:
        DataFrame with metrics for each parameter combination
    """
    try:
        import vectorbt as vbt
    except ImportError:
        raise ImportError(
            "vectorbt is not installed. Install with: pip install vectorbt"
        )

    close = df['close']

    # Generate all EMA combinations at once
    fast_emas = vbt.MA.run(close, list(fast_range), short_name='fast', ewm=True)
    slow_emas = vbt.MA.run(close, list(slow_range), short_name='slow', ewm=True)

    # All crossover combinations
    entries, exits = fast_emas.ma_crossed_above(slow_emas), fast_emas.ma_crossed_below(slow_emas)

    # Run all portfolios at once
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=initial_cash,
        fees=fees,
        freq='1h',
    )

    # Get metrics for each combo
    total_returns = portfolio.total_return()
    sharpe_ratios = portfolio.sharpe_ratio()
    max_drawdowns = portfolio.max_drawdown()

    # Find best parameters
    best_idx = total_returns.idxmax() if hasattr(total_returns, 'idxmax') else None

    return {
        'total_returns': total_returns,
        'sharpe_ratios': sharpe_ratios,
        'max_drawdowns': max_drawdowns,
        'best_params': best_idx,
        'best_return': total_returns.max() if hasattr(total_returns, 'max') else None,
        'portfolio': portfolio,
    }


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
    for symbol, df in data.items():
        try:
            result = vectorbt_momentum_backtest(
                df, fast_ema=fast_ema, slow_ema=slow_ema,
                initial_cash=initial_cash / len(data)
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

    # Aggregate
    returns = [r['return'] for r in results.values()]
    return {
        'per_symbol': results,
        'avg_return': np.mean(returns),
        'total_return': sum(returns),
        'best_symbol': max(results.items(), key=lambda x: x[1]['return'])[0],
        'worst_symbol': min(results.items(), key=lambda x: x[1]['return'])[0],
    }

"""
Functional verification of caller argument completeness audit fixes.
Tests that all backtest callers now properly pass asset-class-aware parameters.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('ALPACA_API_KEY', 'test')
os.environ.setdefault('ALPACA_SECRET_KEY', 'test')
os.environ.setdefault('TELEGRAM_BOT_TOKEN', 'test')
os.environ.setdefault('TELEGRAM_CHAT_ID', '0')

import numpy as np
import pandas as pd
import inspect


def _make_ohlcv(n=300, seed=42):
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        'open': close + np.random.randn(n) * 0.1,
        'high': close + abs(np.random.randn(n) * 0.3),
        'low': close - abs(np.random.randn(n) * 0.3),
        'close': close,
        'volume': np.random.randint(1000, 10000, n).astype(float),
    }, index=pd.date_range('2024-01-01', periods=n, freq='h'))


def test_vbt_adapter_returns_trades():
    """_numpy_ema_crossover_backtest now includes 'trades' in return dict."""
    from backtesting.vbt_adapter import _numpy_ema_crossover_backtest
    df = _make_ohlcv()
    result = _numpy_ema_crossover_backtest(df, fast_ema=12, slow_ema=26)
    assert 'trades' in result, "trades key missing from result"
    assert isinstance(result['trades'], list)
    assert len(result['trades']) > 0
    # Each trade has required fields
    t = result['trades'][0]
    assert 'entry_price' in t
    assert 'exit_price' in t
    assert 'pnl' in t
    assert 'return' in t


def test_monte_carlo_uses_vbt_adapter():
    """monte_carlo_from_backtest now delegates to _numpy_ema_crossover_backtest."""
    from backtesting.monte_carlo import monte_carlo_from_backtest
    df = _make_ohlcv()
    mc = monte_carlo_from_backtest(df, fast_ema=12, slow_ema=26, fees=0.001, n_simulations=50)
    assert mc['n_trades'] > 0
    # New params should be recorded
    assert 'risk_per_trade' in mc['backtest_params']
    assert 'atr_stop_multiplier' in mc['backtest_params']
    assert 'cooldown_bars' in mc['backtest_params']


def test_monte_carlo_supports_new_params():
    """monte_carlo_from_backtest accepts risk_per_trade, atr_stop_multiplier, cooldown."""
    from backtesting.monte_carlo import monte_carlo_from_backtest
    df = _make_ohlcv()
    sig = inspect.signature(monte_carlo_from_backtest)
    assert 'risk_per_trade' in sig.parameters
    assert 'atr_stop_multiplier' in sig.parameters
    assert 'cooldown_bars' in sig.parameters

    # Call with all params
    mc = monte_carlo_from_backtest(
        df, fast_ema=21, slow_ema=55, fees=0.0015,
        risk_per_trade=0.3, atr_stop_multiplier=2.5, cooldown_bars=4,
        n_simulations=50,
    )
    assert mc['backtest_params']['risk_per_trade'] == 0.3
    assert mc['backtest_params']['atr_stop_multiplier'] == 2.5
    assert mc['backtest_params']['cooldown_bars'] == 4


def test_walk_forward_accepts_new_params():
    """walk_forward_ema_backtest accepts cooldown_bars and atr_stop_multiplier."""
    from backtesting.walk_forward import walk_forward_ema_backtest
    sig = inspect.signature(walk_forward_ema_backtest)
    assert 'cooldown_bars' in sig.parameters
    assert 'atr_stop_multiplier' in sig.parameters

    df = _make_ohlcv()
    wf = walk_forward_ema_backtest(
        df, train_window=150, test_window=50, step=50,
        fees=0.0015, cooldown_bars=3, atr_stop_multiplier=2.0,
    )
    assert 'n_windows' in wf
    assert wf['n_windows'] > 0


def test_walk_forward_strategy_supports_stops():
    """_ema_crossover_strategy in walk_forward.py handles cooldown and ATR stops."""
    from backtesting.walk_forward import _ema_crossover_strategy
    df = _make_ohlcv()
    # With stops and cooldown
    trades = _ema_crossover_strategy(df, {
        'fast_ema': 12, 'slow_ema': 26, 'fees': 0.001,
        'cooldown_bars': 2, 'atr_stop_multiplier': 2.0,
    })
    assert isinstance(trades, list)
    # Without stops (backward compat)
    trades_no_stops = _ema_crossover_strategy(df, {
        'fast_ema': 12, 'slow_ema': 26, 'fees': 0.001,
    })
    assert isinstance(trades_no_stops, list)
    assert len(trades_no_stops) > 0


def test_param_validator_delegates_to_walk_forward():
    """param_validator._ema_strategy_with_stops delegates to walk_forward."""
    from backtesting.param_validator import _ema_strategy_with_stops
    df = _make_ohlcv()
    trades = _ema_strategy_with_stops(df, {
        'fast_ema': 12, 'slow_ema': 26, 'fees': 0.001,
        'cooldown_bars': 2, 'atr_stop_multiplier': 2.0,
    })
    assert len(trades) > 0


def test_multi_symbol_defaults_to_asset_class_aware():
    """vectorbt_multi_symbol_backtest defaults use_asset_class_params=True."""
    from backtesting.vbt_adapter import vectorbt_multi_symbol_backtest
    sig = inspect.signature(vectorbt_multi_symbol_backtest)
    assert sig.parameters['use_asset_class_params'].default is True


def test_backtest_config_differentiation():
    """get_backtest_config produces different params for equity vs crypto."""
    from src.config.backtest_params import get_backtest_config
    eq = get_backtest_config('AAPL')
    cr = get_backtest_config('BTC/USD')
    assert eq['fast_ema'] != cr['fast_ema']
    assert eq['fees'] != cr['fees']
    assert eq['annualization_periods'] != cr['annualization_periods']
    assert eq['risk_per_trade'] != cr['risk_per_trade']


def test_backtester_for_symbol():
    """Backtester.for_symbol resolves correct costs."""
    from backtesting.backtest import Backtester
    from unittest.mock import MagicMock
    strategy = MagicMock()
    strategy.name = "test"

    bt_equity = Backtester.for_symbol(strategy, "AAPL", initial_capital=10000.0)
    bt_crypto = Backtester.for_symbol(strategy, "BTC/USD", initial_capital=10000.0)
    assert bt_equity.commission_pct < bt_crypto.commission_pct


if __name__ == '__main__':
    tests = [
        test_vbt_adapter_returns_trades,
        test_monte_carlo_uses_vbt_adapter,
        test_monte_carlo_supports_new_params,
        test_walk_forward_accepts_new_params,
        test_walk_forward_strategy_supports_stops,
        test_param_validator_delegates_to_walk_forward,
        test_multi_symbol_defaults_to_asset_class_aware,
        test_backtest_config_differentiation,
        test_backtester_for_symbol,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASSED: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {t.__name__} — {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)

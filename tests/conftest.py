"""Shared test fixtures for algo-trader test suite."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta


@pytest.fixture
def sample_ohlcv():
    """Generate 200 bars of realistic OHLCV data with DatetimeIndex."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    # Random walk for close
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(100000, 5000000, n).astype(float)
    
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume,
    }, index=dates)


@pytest.fixture
def small_ohlcv():
    """50 bars — below minimum for feature engineering."""
    np.random.seed(99)
    n = 50
    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    close = 50 + np.cumsum(np.random.randn(n) * 0.3)
    return pd.DataFrame({
        'open': close + 0.1, 'high': close + 0.5,
        'low': close - 0.5, 'close': close,
        'volume': np.random.randint(50000, 1000000, n).astype(float),
    }, index=dates)


@pytest.fixture
def sample_trades():
    """12 sample trades for metrics/MC testing."""
    now = datetime.now(timezone.utc)
    trades = []
    pnls = [150, -80, 200, -50, 300, -120, 100, -30, 250, -90, 180, -60]
    for i, pnl in enumerate(pnls):
        trades.append({
            'pnl': pnl,
            'symbol': 'AAPL',
            'side': 'long',
            'entry_time': now - timedelta(hours=24-i),
            'exit_time': now - timedelta(hours=23-i),
            'quantity': 10,
        })
    return trades


@pytest.fixture
def trending_ohlcv():
    """200 bars with clear uptrend (for testing entry/filter models)."""
    np.random.seed(7)
    n = 200
    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    # Steady uptrend
    trend = np.linspace(100, 130, n)
    noise = np.random.randn(n) * 0.3
    close = trend + noise
    high = close + np.abs(np.random.randn(n) * 0.4)
    low = close - np.abs(np.random.randn(n) * 0.4)
    open_ = close - np.random.rand(n) * 0.2
    volume = np.random.randint(500000, 5000000, n).astype(float)
    # Add a volume spike near end
    volume[-10:] = volume[-10:] * 3
    
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume,
    }, index=dates)

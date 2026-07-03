"""
ML Feature Engineering Pipeline.
Prepares raw market data for ML model consumption.
"""

import numpy as np
import pandas as pd
from typing import Optional


def engineer_features(df: pd.DataFrame, include_target: bool = False, forward_bars: int = 5, threshold: float = 0.005) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Input: OHLCV DataFrame with columns [open, high, low, close, volume]
    Output: DataFrame with all engineered features (and optionally target)
    """
    df = df.copy()

    # --- Price Returns ---
    for period in [1, 2, 3, 5, 10, 20, 50]:
        df[f'ret_{period}'] = df['close'].pct_change(period)

    # --- Volatility ---
    df['vol_5'] = df['ret_1'].rolling(5).std()
    df['vol_10'] = df['ret_1'].rolling(10).std()
    df['vol_20'] = df['ret_1'].rolling(20).std()
    df['vol_ratio'] = df['vol_5'] / df['vol_20'].replace(0, np.nan)

    # --- Trend Indicators ---
    for period in [5, 10, 20, 50, 100, 200]:
        sma = df['close'].rolling(period).mean()
        df[f'dist_sma_{period}'] = (df['close'] - sma) / sma

    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # --- Oscillators ---
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Stochastic
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14).replace(0, np.nan)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # --- Bollinger Bands ---
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_pct'] = (df['close'] - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)
    df['bb_width'] = (4 * bb_std) / bb_mid

    # --- Volume Features ---
    df['vol_ma_20'] = df['volume'].rolling(20).mean()
    df['vol_spike'] = df['volume'] / df['vol_ma_20'].replace(0, np.nan)
    df['vol_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean().replace(0, np.nan)

    # --- Candlestick Features ---
    body = df['close'] - df['open']
    range_ = df['high'] - df['low']
    df['body_pct'] = body / range_.replace(0, np.nan)
    df['upper_wick'] = (df['high'] - df[['close', 'open']].max(axis=1)) / range_.replace(0, np.nan)
    df['lower_wick'] = (df[['close', 'open']].min(axis=1) - df['low']) / range_.replace(0, np.nan)

    # --- ATR ---
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / df['close']

    # --- Day/Time Features (if datetime index) ---
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # --- Target ---
    if include_target:
        df['future_return'] = df['close'].shift(-forward_bars) / df['close'] - 1
        df['target'] = np.where(
            df['future_return'] > threshold, 1,
            np.where(df['future_return'] < -threshold, -1, 0)
        )

    return df


def get_feature_names(include_time_features: bool = True) -> list[str]:
    """Return the list of feature column names (excluding target).
    
    Args:
        include_time_features: Whether to include hour/dow cyclical features.
            Only set True when the data has a DatetimeIndex.
    """
    features = [
        'ret_1', 'ret_2', 'ret_3', 'ret_5', 'ret_10', 'ret_20', 'ret_50',
        'vol_5', 'vol_10', 'vol_20', 'vol_ratio',
        'dist_sma_5', 'dist_sma_10', 'dist_sma_20', 'dist_sma_50', 'dist_sma_100', 'dist_sma_200',
        'macd', 'macd_signal', 'macd_hist',
        'rsi_14', 'stoch_k', 'stoch_d',
        'bb_pct', 'bb_width',
        'vol_spike', 'vol_trend',
        'body_pct', 'upper_wick', 'lower_wick',
        'atr_pct',
    ]
    if include_time_features:
        features.extend(['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'])
    return features

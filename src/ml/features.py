"""
ML Feature Engineering Pipeline — Institutional Quality (120+ features).

Produces features across 8 categories: Trend, Volatility, Momentum, Volume/Order Flow,
Time, Regime, Statistical, and Cross-Asset/Relative.

Anti-Leakage Guarantees:
- No .shift(-N) in feature computation (only in target generation)
- All rolling windows use min_periods to avoid early NaN issues
- EMAs computed with adjust=False (no future weighting)
- Minimum warmup: ~100 bars for all features to stabilize

Requires ONLY numpy and pandas.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Suppress pandas PerformanceWarning about DataFrame fragmentation.
# This module intentionally adds columns one-by-one for readability, and
# defragments with df.copy() before returning.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Minimum bars needed before features are valid (due to rolling windows)
MINIMUM_BARS_REQUIRED = 100


# ---------------------------------------------------------------------------
# Helper functions (vectorized, no look-ahead)
# ---------------------------------------------------------------------------

def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Division replacing zero/inf with NaN."""
    return numerator / denominator.replace(0, np.nan)


def _rolling_rank(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of current value within rolling window (0-1)."""
    def _rank_pct(x):
        if len(x) < 2:
            return np.nan
        return (x.values < x.values[-1]).sum() / (len(x) - 1)
    return series.rolling(window, min_periods=window).apply(_rank_pct, raw=False)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range using previous close."""
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)


def _count_trailing(x, value) -> int:
    """Count trailing consecutive occurrences of value at end of array."""
    count = 0
    for i in range(len(x) - 1, -1, -1):
        if x[i] == value:
            count += 1
        else:
            break
    return count


def _ema(series: pd.Series, span: int) -> pd.Series:
    """EMA with adjust=False (no future weighting)."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    """RSI using rolling mean of gains/losses."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period, min_periods=period).mean()
    rs = _safe_div(gain, loss)
    return 100.0 - (100.0 / (1.0 + rs))


# ---------------------------------------------------------------------------
# Main feature engineering pipeline
# ---------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    include_target: bool = False,
    forward_bars: int = 5,
    threshold: float = 0.005,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline — 120+ features, institutional quality.

    Input: OHLCV DataFrame with columns [open, high, low, close, volume].
           DatetimeIndex enables time features.
    Output: DataFrame with all engineered feature columns added.

    Args:
        df: OHLCV data. Must have columns: open, high, low, close, volume.
        include_target: If True, appends future_return and target columns.
        forward_bars: Lookahead bars for target computation.
        threshold: Return threshold for classification target.
    """
    # Input validation
    if df is None or len(df) == 0:
        raise ValueError("Cannot engineer features from empty DataFrame")

    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    missing = required_cols - set(c.lower() for c in df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) < MINIMUM_BARS_REQUIRED:
        logger.warning(
            "ml.features.insufficient_bars",
            bars=len(df),
            minimum=MINIMUM_BARS_REQUIRED,
        )

    df = df.copy()

    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']
    volume = df['volume'].replace(0, np.nan)  # guard against zero volume

    # Pre-compute returns (used extensively)
    ret_1 = close.pct_change(1)
    log_ret = np.log(close / close.shift(1))

    # ======================================================================
    # 1. PRICE RETURNS (baseline, backward compatible)
    # ======================================================================
    for period in [1, 2, 3, 5, 10, 20, 50]:
        df[f'ret_{period}'] = close.pct_change(period)

    # ======================================================================
    # 2. TREND FEATURES (18 features)
    # ======================================================================

    # SMA distance (backward compatible)
    for period in [5, 10, 20, 50, 100, 200]:
        sma = close.rolling(period, min_periods=period).mean()
        df[f'dist_sma_{period}'] = _safe_div(close - sma, sma)

    # EMA slopes (normalized by price)
    for span in [5, 10, 20, 50]:
        ema = _ema(close, span)
        df[f'ema_slope_{span}'] = _safe_div(ema - ema.shift(1), close)

    # MACD (backward compatible names)
    df['ema_12'] = _ema(close, 12)
    df['ema_26'] = _ema(close, 26)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = _ema(df['macd'], 9)
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Linear regression slope (20 bars), normalized
    def _linreg_slope(x):
        if len(x) < 20:
            return np.nan
        y = x.values
        t = np.arange(len(y))
        slope = np.polyfit(t, y, 1)[0]
        return slope
    df['linreg_slope_20'] = close.rolling(20, min_periods=20).apply(_linreg_slope, raw=False)
    df['linreg_slope_20'] = _safe_div(df['linreg_slope_20'], close)

    # ADX and DI+/DI-
    tr = _true_range(high, low, close)
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    # Zero out when other DM is larger
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

    atr_14_raw = tr.rolling(14, min_periods=14).mean()
    plus_di = 100 * _safe_div(plus_dm.rolling(14, min_periods=14).mean(), atr_14_raw)
    minus_di = 100 * _safe_div(minus_dm.rolling(14, min_periods=14).mean(), atr_14_raw)
    dx = 100 * _safe_div((plus_di - minus_di).abs(), plus_di + minus_di)
    df['adx'] = dx.rolling(14, min_periods=14).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    df['di_diff'] = plus_di - minus_di

    # Trend persistence: bars since EMA 10/50 crossover
    ema_10 = _ema(close, 10)
    ema_50 = _ema(close, 50)
    cross_signal = (ema_10 > ema_50).astype(int)
    cross_change = cross_signal.diff().abs()
    # Count bars since last crossover
    cross_groups = cross_change.cumsum()
    df['trend_persistence'] = cross_groups.groupby(cross_groups).cumcount()

    # Market structure: higher-highs and lower-lows count (20 bar window)
    hh = (high > high.shift(1)).astype(int)
    ll = (low < low.shift(1)).astype(int)
    df['higher_highs_20'] = hh.rolling(20, min_periods=5).sum()
    df['lower_lows_20'] = ll.rolling(20, min_periods=5).sum()
    df['market_structure'] = df['higher_highs_20'] - df['lower_lows_20']

    # Hurst exponent estimate (simplified R/S method, 50-bar window)
    def _hurst_rs(x):
        if len(x) < 20:
            return np.nan
        y = np.diff(x)
        m = np.mean(y)
        z = np.cumsum(y - m)
        r = np.max(z) - np.min(z)
        s = np.std(y, ddof=1)
        if s == 0:
            return np.nan
        return np.log(r / s) / np.log(len(y))
    df['hurst_50'] = close.rolling(50, min_periods=50).apply(_hurst_rs, raw=True)

    # ======================================================================
    # 3. VOLATILITY FEATURES (18 features)
    # ======================================================================

    # Simple realized volatility (backward compatible)
    df['vol_5'] = ret_1.rolling(5, min_periods=5).std()
    df['vol_10'] = ret_1.rolling(10, min_periods=10).std()
    df['vol_20'] = ret_1.rolling(20, min_periods=20).std()
    df['vol_ratio'] = _safe_div(df['vol_5'], df['vol_20'])

    # Parkinson volatility (high-low based)
    hl_log = np.log(_safe_div(high, low))
    for w in [10, 20]:
        df[f'parkinson_vol_{w}'] = np.sqrt(
            (hl_log ** 2).rolling(w, min_periods=w).mean() / (4 * np.log(2))
        )

    # Garman-Klass volatility (20-day)
    gk_term = (0.5 * hl_log ** 2
               - (2 * np.log(2) - 1) * np.log(_safe_div(close, open_)) ** 2)
    df['garman_klass_20'] = np.sqrt(gk_term.rolling(20, min_periods=20).mean())

    # Yang-Zhang volatility (20-day)
    oc_log = np.log(_safe_div(open_, close.shift(1)))
    co_log = np.log(_safe_div(close, open_))
    yz_open_var = oc_log.rolling(20, min_periods=20).var()
    yz_close_var = co_log.rolling(20, min_periods=20).var()
    yz_rs_var = (hl_log ** 2).rolling(20, min_periods=20).mean() / (4 * np.log(2))
    k = 0.34 / (1.34 + 21.0 / 22.0)
    df['yang_zhang_20'] = np.sqrt(yz_open_var + k * yz_close_var + (1 - k) * yz_rs_var)

    # ATR multiple periods (backward compatible atr_14)
    df['atr_14'] = atr_14_raw
    df['atr_pct'] = _safe_div(df['atr_14'], close)
    for p in [7, 21]:
        atr_p = tr.rolling(p, min_periods=p).mean()
        df[f'atr_{p}'] = atr_p
        df[f'atr_pct_{p}'] = _safe_div(atr_p, close)

    # Intraday range / close
    df['intraday_range'] = _safe_div(high - low, close)

    # Volatility of volatility
    df['vol_of_vol'] = df['vol_20'].rolling(20, min_periods=10).std()

    # Rolling entropy (simplified: std of absolute returns distribution)
    abs_ret = ret_1.abs()
    def _rolling_entropy(x):
        if len(x) < 10:
            return np.nan
        # Approximate entropy via histogram-based method
        counts, _ = np.histogram(x, bins=10)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))
    df['return_entropy_20'] = abs_ret.rolling(20, min_periods=20).apply(_rolling_entropy, raw=True)

    # Volatility regime ratio (short vs long)
    df['vol_ratio_7_21'] = _safe_div(
        ret_1.rolling(7, min_periods=7).std(),
        ret_1.rolling(21, min_periods=21).std()
    )

    # ======================================================================
    # 4. MOMENTUM FEATURES (20 features)
    # ======================================================================

    # RSI (backward compatible)
    df['rsi_14'] = _rsi(close, 14)

    # Multi-timeframe RSI
    df['rsi_7'] = _rsi(close, 7)
    df['rsi_21'] = _rsi(close, 21)
    df['rsi_fast_slow_diff'] = df['rsi_7'] - df['rsi_21']

    # Rate of Change
    for p in [5, 10, 20]:
        df[f'roc_{p}'] = _safe_div(close - close.shift(p), close.shift(p)) * 100

    # Momentum acceleration (ROC of ROC)
    roc_10 = df['roc_10']
    df['momentum_accel'] = roc_10 - roc_10.shift(5)

    # Stochastic oscillator (backward compatible)
    low_14 = low.rolling(14, min_periods=14).min()
    high_14 = high.rolling(14, min_periods=14).max()
    df['stoch_k'] = 100 * _safe_div(close - low_14, high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(3, min_periods=3).mean()

    # Williams %R
    df['williams_r'] = -100 * _safe_div(high_14 - close, high_14 - low_14)

    # CCI (Commodity Channel Index)
    typical_price = (high + low + close) / 3.0
    tp_sma = typical_price.rolling(20, min_periods=20).mean()
    tp_mad = typical_price.rolling(20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    df['cci_20'] = _safe_div(typical_price - tp_sma, 0.015 * tp_mad)

    # Ultimate Oscillator
    bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    tr_uo = _true_range(high, low, close)
    avg7 = _safe_div(bp.rolling(7, min_periods=7).sum(), tr_uo.rolling(7, min_periods=7).sum())
    avg14 = _safe_div(bp.rolling(14, min_periods=14).sum(), tr_uo.rolling(14, min_periods=14).sum())
    avg28 = _safe_div(bp.rolling(28, min_periods=28).sum(), tr_uo.rolling(28, min_periods=28).sum())
    df['ultimate_osc'] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7.0

    # TSI (True Strength Index)
    pc = close.diff()
    double_smooth_pc = _ema(_ema(pc, 25), 13)
    double_smooth_abs_pc = _ema(_ema(pc.abs(), 25), 13)
    df['tsi'] = 100 * _safe_div(double_smooth_pc, double_smooth_abs_pc)

    # Bollinger Bands (backward compatible)
    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    df['bb_pct'] = _safe_div(close - (bb_mid - 2 * bb_std), 4 * bb_std)
    df['bb_width'] = _safe_div(4 * bb_std, bb_mid)

    # ======================================================================
    # 5. VOLUME / ORDER FLOW FEATURES (16 features)
    # ======================================================================

    vol_ma_20 = volume.rolling(20, min_periods=10).mean()
    df['vol_ma_20'] = vol_ma_20

    # Volume spike / trend (backward compatible)
    df['vol_spike'] = _safe_div(volume, vol_ma_20)
    df['vol_trend'] = _safe_div(
        volume.rolling(5, min_periods=5).mean(),
        volume.rolling(20, min_periods=10).mean()
    )

    # Volume ratio
    df['volume_ratio_10'] = _safe_div(volume, volume.rolling(10, min_periods=10).mean())

    # Dollar volume
    df['dollar_volume'] = close * volume
    df['dollar_volume_ratio'] = _safe_div(
        df['dollar_volume'],
        df['dollar_volume'].rolling(20, min_periods=10).mean()
    )

    # Volume momentum (ROC of volume)
    df['volume_roc_5'] = _safe_div(volume - volume.shift(5), volume.shift(5))
    df['volume_roc_10'] = _safe_div(volume - volume.shift(10), volume.shift(10))

    # OBV and OBV slope
    obv = (np.sign(ret_1).fillna(0) * volume).cumsum()
    df['obv_slope_10'] = _safe_div(obv - obv.shift(10), obv.rolling(10, min_periods=10).mean().abs())

    # Accumulation/Distribution Line
    clv = _safe_div((close - low) - (high - close), high - low)
    ad = (clv * volume).cumsum()
    df['ad_slope_10'] = _safe_div(ad - ad.shift(10), ad.rolling(10, min_periods=10).mean().abs())

    # Money Flow Index (14 period)
    mf_raw = typical_price * volume
    pos_mf = mf_raw.where(typical_price > typical_price.shift(1), 0)
    neg_mf = mf_raw.where(typical_price < typical_price.shift(1), 0)
    mf_ratio = _safe_div(
        pos_mf.rolling(14, min_periods=14).sum(),
        neg_mf.rolling(14, min_periods=14).sum()
    )
    df['mfi_14'] = 100 - _safe_div(pd.Series(100, index=df.index), 1 + mf_ratio)

    # Force Index
    df['force_index_13'] = _ema(ret_1 * volume, 13)
    df['force_index_norm'] = _safe_div(df['force_index_13'], vol_ma_20)

    # Volume-price trend
    vpt = (ret_1 * volume).cumsum()
    df['vpt_slope_10'] = _safe_div(vpt - vpt.shift(10), vpt.rolling(10, min_periods=10).mean().abs())

    # Up volume vs down volume ratio (10 bar)
    up_vol = volume.where(ret_1 > 0, 0)
    down_vol = volume.where(ret_1 < 0, 0)
    df['up_down_vol_ratio'] = _safe_div(
        up_vol.rolling(10, min_periods=5).sum(),
        down_vol.rolling(10, min_periods=5).sum()
    )

    # VWAP deviation (rolling 20-bar approximation)
    cum_tp_vol = (typical_price * volume).rolling(20, min_periods=10).sum()
    cum_vol = volume.rolling(20, min_periods=10).sum()
    vwap_20 = _safe_div(cum_tp_vol, cum_vol)
    df['vwap_dev_20'] = _safe_div(close - vwap_20, vwap_20)

    # ======================================================================
    # 6. CANDLESTICK / PRICE ACTION FEATURES (backward compatible + new)
    # ======================================================================

    body = close - open_
    range_ = high - low
    df['body_pct'] = _safe_div(body, range_)
    df['upper_wick'] = _safe_div(high - pd.concat([close, open_], axis=1).max(axis=1), range_)
    df['lower_wick'] = _safe_div(pd.concat([close, open_], axis=1).min(axis=1) - low, range_)

    # Gap features
    df['gap_pct'] = _safe_div(open_ - close.shift(1), close.shift(1))

    # ======================================================================
    # 7. TIME FEATURES (12 features, if datetime index)
    # ======================================================================

    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        dow = df.index.dayofweek
        month = df.index.month
        day = df.index.day

        # Cyclical encodings
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)

        # Binary flags
        df['is_monday'] = (dow == 0).astype(int)
        df['is_friday'] = (dow == 4).astype(int)

        # Days to month end
        df['days_to_month_end'] = pd.Series(
            [(pd.Timestamp(d.year, d.month, 1) + pd.offsets.MonthEnd(0)).day - d.day
             for d in df.index],
            index=df.index
        )
        df['quarter'] = ((month - 1) // 3).astype(int)

        # First/last hour flags (for intraday data)
        df['is_first_hour'] = (hour == df.index[0].hour).astype(int) if len(df) > 0 else 0
        df['is_last_hour'] = (hour == 15).astype(int)  # typical market close hour

    # ======================================================================
    # 8. REGIME FEATURES (12 features)
    # ======================================================================

    # Trending score (ADX-based)
    df['is_trending'] = (df['adx'] > 25).astype(int)
    df['trending_score'] = df['adx'] / 50.0  # normalized 0-~1

    # Volatility regime
    vol_median_60 = df['vol_20'].rolling(60, min_periods=30).median()
    df['vol_regime'] = _safe_div(df['vol_20'], vol_median_60)
    df['high_vol_regime'] = (df['vol_20'] > vol_median_60).astype(int)

    # Bull/Bear regime
    sma_200 = close.rolling(200, min_periods=100).mean()
    df['bull_bear'] = (close > sma_200).astype(int)

    # Mean reversion potential (z-score vs 50 SMA)
    sma_50 = close.rolling(50, min_periods=50).mean()
    std_50 = close.rolling(50, min_periods=50).std()
    df['zscore_50'] = _safe_div(close - sma_50, std_50)

    # Regime persistence (bars in current bull/bear regime)
    regime_signal = df['bull_bear'].diff().abs().fillna(0)
    regime_groups = regime_signal.cumsum()
    df['regime_persistence'] = regime_groups.groupby(regime_groups).cumcount()

    # Risk-on/off proxy (vol regime + trend direction)
    df['risk_on_proxy'] = df['bull_bear'] * (1 - df['high_vol_regime'])

    # Drawdown from recent peak (20 bar)
    rolling_max_20 = close.rolling(20, min_periods=5).max()
    df['drawdown_20'] = _safe_div(close - rolling_max_20, rolling_max_20)

    # Recovery ratio (distance recovered from 20-bar low)
    rolling_min_20 = close.rolling(20, min_periods=5).min()
    df['recovery_ratio'] = _safe_div(close - rolling_min_20, rolling_max_20 - rolling_min_20)

    # ======================================================================
    # 9. STATISTICAL FEATURES (16 features)
    # ======================================================================

    # Z-score of price vs SMA
    std_20 = close.rolling(20, min_periods=20).std()
    sma_20 = close.rolling(20, min_periods=20).mean()
    df['zscore_20'] = _safe_div(close - sma_20, std_20)

    # Rolling skewness and kurtosis
    df['skew_20'] = ret_1.rolling(20, min_periods=20).skew()
    df['kurt_20'] = ret_1.rolling(20, min_periods=20).kurt()

    # Autocorrelation of returns
    df['autocorr_1'] = ret_1.rolling(20, min_periods=20).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )
    df['autocorr_5'] = ret_1.rolling(30, min_periods=30).apply(
        lambda x: pd.Series(x).autocorr(lag=5), raw=False
    )

    # Percentile rank (current price vs last 100 bars)
    df['price_pctrank_100'] = _rolling_rank(close, 100)

    # Rolling Sharpe (20 bar, annualized approx)
    mean_ret_20 = ret_1.rolling(20, min_periods=20).mean()
    df['sharpe_20'] = _safe_div(mean_ret_20, df['vol_20']) * np.sqrt(252)

    # Rolling Sortino (20 bar)
    downside_ret = ret_1.where(ret_1 < 0, 0)
    downside_std = downside_ret.rolling(20, min_periods=20).std()
    df['sortino_20'] = _safe_div(mean_ret_20, downside_std) * np.sqrt(252)

    # Max return in window
    df['max_ret_10'] = ret_1.rolling(10, min_periods=5).max()
    df['min_ret_10'] = ret_1.rolling(10, min_periods=5).min()

    # Return dispersion
    df['ret_range_20'] = df['max_ret_10'] - df['min_ret_10']

    # Consecutive up/down bars
    up_bar = (ret_1 > 0).astype(int)
    down_bar = (ret_1 < 0).astype(int)
    df['consecutive_up'] = up_bar.rolling(10, min_periods=1).apply(
        lambda x: _count_trailing(x, 1), raw=True
    )
    df['consecutive_down'] = down_bar.rolling(10, min_periods=1).apply(
        lambda x: _count_trailing(x, 1), raw=True
    )

    # Tail ratio (95th percentile / 5th percentile of returns)
    def _tail_ratio(x):
        if len(x) < 20:
            return np.nan
        p95 = np.abs(np.percentile(x, 95))
        p5 = np.abs(np.percentile(x, 5))
        return p95 / p5 if p5 != 0 else np.nan
    df['tail_ratio_50'] = ret_1.rolling(50, min_periods=30).apply(_tail_ratio, raw=True)

    # ======================================================================
    # 10. CROSS-ASSET / RELATIVE FEATURES (5 features)
    # Computed as self-relative metrics when single-asset
    # ======================================================================

    # Return rank within own history (20-bar percentile)
    df['ret_pctrank_20'] = _rolling_rank(ret_1, 20)

    # Volume rank within own history
    df['vol_pctrank_50'] = _rolling_rank(volume, 50)

    # Relative strength (momentum rank)
    df['momentum_rank_50'] = _rolling_rank(close.pct_change(10), 50)

    # Price distance from 52-bar high/low (normalized)
    high_52 = close.rolling(52, min_periods=20).max()
    low_52 = close.rolling(52, min_periods=20).min()
    df['dist_52_high'] = _safe_div(close - high_52, high_52)
    df['dist_52_low'] = _safe_div(close - low_52, low_52)

    # Relative volume strength (volume percentile in own history)
    df['vol_strength_100'] = _rolling_rank(volume, 100)

    # ======================================================================
    # TARGET (uses look-ahead — only when explicitly requested)
    # ======================================================================
    if include_target:
        df['future_return'] = close.shift(-forward_bars) / close - 1
        df['target'] = np.where(
            df['future_return'] > threshold, 1,
            np.where(df['future_return'] < -threshold, -1, 0)
        )

    # Drop intermediate columns not needed as features
    drop_cols = ['ema_12', 'ema_26', 'vol_ma_20', 'dollar_volume']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

    # Defragment DataFrame (avoids pandas PerformanceWarning from many inserts)
    df = df.copy()

    return df


def _count_trailing(arr, val):
    """Count trailing consecutive occurrences of val in array."""
    count = 0
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] == val:
            count += 1
        else:
            break
    return count


def get_feature_names(include_time_features: bool = True) -> list[str]:
    """Return the list of all feature column names (excluding target).

    Args:
        include_time_features: Whether to include time-based features.
            Only set True when the data has a DatetimeIndex.
    """
    features = [
        # --- Price Returns ---
        'ret_1', 'ret_2', 'ret_3', 'ret_5', 'ret_10', 'ret_20', 'ret_50',

        # --- Trend ---
        'dist_sma_5', 'dist_sma_10', 'dist_sma_20', 'dist_sma_50', 'dist_sma_100', 'dist_sma_200',
        'ema_slope_5', 'ema_slope_10', 'ema_slope_20', 'ema_slope_50',
        'macd', 'macd_signal', 'macd_hist',
        'linreg_slope_20',
        'adx', 'plus_di', 'minus_di', 'di_diff',
        'trend_persistence',
        'higher_highs_20', 'lower_lows_20', 'market_structure',
        'hurst_50',

        # --- Volatility ---
        'vol_5', 'vol_10', 'vol_20', 'vol_ratio',
        'parkinson_vol_10', 'parkinson_vol_20',
        'garman_klass_20', 'yang_zhang_20',
        'atr_pct', 'atr_pct_7', 'atr_pct_21',
        'intraday_range',
        'vol_of_vol', 'return_entropy_20', 'vol_ratio_7_21',

        # --- Momentum ---
        'rsi_14', 'rsi_7', 'rsi_21', 'rsi_fast_slow_diff',
        'roc_5', 'roc_10', 'roc_20',
        'momentum_accel',
        'stoch_k', 'stoch_d', 'williams_r',
        'cci_20', 'ultimate_osc', 'tsi',
        'bb_pct', 'bb_width',

        # --- Volume / Order Flow ---
        'vol_spike', 'vol_trend', 'volume_ratio_10',
        'dollar_volume_ratio',
        'volume_roc_5', 'volume_roc_10',
        'obv_slope_10', 'ad_slope_10',
        'mfi_14', 'force_index_norm', 'vpt_slope_10',
        'up_down_vol_ratio', 'vwap_dev_20',

        # --- Candlestick / Price Action ---
        'body_pct', 'upper_wick', 'lower_wick', 'gap_pct',

        # --- Regime ---
        'is_trending', 'trending_score',
        'vol_regime', 'high_vol_regime',
        'bull_bear', 'zscore_50',
        'regime_persistence', 'risk_on_proxy',
        'drawdown_20', 'recovery_ratio',

        # --- Statistical ---
        'zscore_20', 'skew_20', 'kurt_20',
        'autocorr_1', 'autocorr_5',
        'price_pctrank_100', 'sharpe_20', 'sortino_20',
        'max_ret_10', 'min_ret_10', 'ret_range_20',
        'consecutive_up', 'consecutive_down', 'tail_ratio_50',

        # --- Cross-Asset / Relative ---
        'ret_pctrank_20', 'vol_pctrank_50', 'momentum_rank_50',
        'dist_52_high', 'dist_52_low', 'vol_strength_100',
    ]

    if include_time_features:
        features.extend([
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'month_sin', 'month_cos',
            'is_monday', 'is_friday',
            'days_to_month_end', 'quarter',
            'is_first_hour', 'is_last_hour',
        ])

    return features

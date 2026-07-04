"""Multi-dimensional market regime classifier."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.intelligence.models import MarketState


class RegimeClassifier:
    """Classify trend, volatility, liquidity, correlation, and stress dimensions."""

    def classify(self, df: Optional[pd.DataFrame], context: Optional[dict] = None) -> MarketState:
        context = context or {}
        if df is None or len(df) < 30:
            return MarketState(
                trend="Sideways",
                volatility="Normal",
                liquidity="Medium",
                correlation_env="Neutral",
                stress="Elevated",
                overall_regime="UNKNOWN",
                confidence=0.35,
                diagnostics={"reason": "insufficient_data"},
            )

        close = df["close"].astype(float)
        volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series([0.0] * len(df), index=df.index)

        ret_20 = close.pct_change(20).iloc[-1]
        ret_5 = close.pct_change(5).iloc[-1]
        short_vol = close.pct_change().rolling(10).std().iloc[-1]
        long_vol = close.pct_change().rolling(50).std().iloc[-1]
        vol_ratio = float(short_vol / long_vol) if long_vol and not pd.isna(long_vol) else 1.0
        vol_ratio = 1.0 if pd.isna(vol_ratio) or np.isinf(vol_ratio) else vol_ratio

        if ret_20 >= 0.06:
            trend = "Strong Uptrend"
            trend_score = min(ret_20 / 0.10, 1.0)
        elif ret_20 >= 0.02:
            trend = "Weak Uptrend"
            trend_score = min(ret_20 / 0.06, 1.0)
        elif ret_20 <= -0.06:
            trend = "Strong Downtrend"
            trend_score = min(abs(ret_20) / 0.10, 1.0)
        elif ret_20 <= -0.02:
            trend = "Weak Downtrend"
            trend_score = min(abs(ret_20) / 0.06, 1.0)
        else:
            trend = "Sideways"
            trend_score = 0.6

        if vol_ratio < 0.7:
            volatility = "Compression"
            vol_score = min((1.0 - vol_ratio) / 0.5, 1.0)
        elif vol_ratio < 1.3:
            volatility = "Normal"
            vol_score = 0.7
        elif vol_ratio < 2.0:
            volatility = "Expansion"
            vol_score = min((vol_ratio - 1.0) / 1.0, 1.0)
        else:
            volatility = "Extreme"
            vol_score = 1.0

        vol_ma = volume.rolling(20).mean().iloc[-1]
        vol_ratio_liq = float(volume.iloc[-1] / vol_ma) if vol_ma and not pd.isna(vol_ma) else 1.0
        if vol_ratio_liq >= 1.25:
            liquidity = "High"
            liq_score = min(vol_ratio_liq / 2.0, 1.0)
        elif vol_ratio_liq >= 0.75:
            liquidity = "Medium"
            liq_score = 0.7
        else:
            liquidity = "Low"
            liq_score = min((1.0 - vol_ratio_liq) / 0.5, 1.0)

        benchmark_ret = context.get("benchmark_return", ret_5)
        corr = context.get("cross_asset_corr", 0.0)
        if benchmark_ret > 0.0 and corr > 0.2:
            correlation_env = "Risk-On"
            corr_score = min(abs(benchmark_ret) / 0.02, 1.0)
        elif benchmark_ret < 0.0 and corr > 0.2:
            correlation_env = "Risk-Off"
            corr_score = min(abs(benchmark_ret) / 0.02, 1.0)
        else:
            correlation_env = "Neutral"
            corr_score = 0.6

        drawdown_proxy = float(context.get("drawdown_proxy", abs(min(ret_5, 0.0))))
        if volatility == "Extreme" or drawdown_proxy >= 0.04:
            stress = "Panic"
            stress_score = 1.0
        elif volatility == "Expansion" or drawdown_proxy >= 0.02:
            stress = "Elevated"
            stress_score = 0.7
        else:
            stress = "Calm"
            stress_score = 0.8

        if "Uptrend" in trend or "Downtrend" in trend:
            regime_core = "TRENDING"
        else:
            regime_core = "RANGING"
        regime_vol = "HIGH_VOL" if volatility in ("Expansion", "Extreme") else "LOW_VOL"
        overall = f"{regime_core}_{regime_vol}"

        confidence = float(np.clip(np.mean([trend_score, vol_score, liq_score, corr_score, stress_score]), 0.0, 1.0))
        return MarketState(
            trend=trend,
            volatility=volatility,
            liquidity=liquidity,
            correlation_env=correlation_env,
            stress=stress,
            overall_regime=overall,
            confidence=confidence,
            diagnostics={
                "ret_20": float(ret_20) if not pd.isna(ret_20) else 0.0,
                "ret_5": float(ret_5) if not pd.isna(ret_5) else 0.0,
                "vol_ratio": vol_ratio,
                "liquidity_ratio": vol_ratio_liq,
            },
        )


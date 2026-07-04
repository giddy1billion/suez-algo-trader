"""
Market State Engine — Full multi-dimensional market fingerprint.

Dimensions:
  - Trend (direction + strength)
  - Volatility (compression → extreme)
  - Liquidity (volume vs norm)
  - Momentum (rate of change)
  - Correlation environment (risk-on/off)
  - Market stress (drawdown + vol spike)
  - Seasonality (day-of-week, hour-of-day effects)
  - Time-of-day session context

Outputs a MarketFingerprint consumed by all downstream intelligence modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class DimensionScore:
    """Score for a single market dimension."""
    label: str
    confidence: float  # 0.0–1.0
    value: float  # raw numeric for downstream math


@dataclass
class MarketFingerprint:
    """Complete market profile used by all intelligence subsystems."""
    timestamp: datetime
    trend: DimensionScore
    volatility: DimensionScore
    liquidity: DimensionScore
    momentum: DimensionScore
    correlation: DimensionScore
    stress: DimensionScore
    seasonality: DimensionScore
    session: DimensionScore
    overall_regime: str
    overall_confidence: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_favorable_for_trend(self) -> bool:
        return "Uptrend" in self.trend.label or "Downtrend" in self.trend.label

    @property
    def is_high_risk(self) -> bool:
        return self.stress.label in ("Panic", "Elevated") or self.volatility.label == "Extreme"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "trend": {"label": self.trend.label, "confidence": self.trend.confidence},
            "volatility": {"label": self.volatility.label, "confidence": self.volatility.confidence},
            "liquidity": {"label": self.liquidity.label, "confidence": self.liquidity.confidence},
            "momentum": {"label": self.momentum.label, "confidence": self.momentum.confidence},
            "correlation": {"label": self.correlation.label, "confidence": self.correlation.confidence},
            "stress": {"label": self.stress.label, "confidence": self.stress.confidence},
            "seasonality": {"label": self.seasonality.label, "confidence": self.seasonality.confidence},
            "session": {"label": self.session.label, "confidence": self.session.confidence},
            "overall_regime": self.overall_regime,
            "overall_confidence": self.overall_confidence,
        }


class MarketStateEngine:
    """Compute a full MarketFingerprint from OHLCV + optional context."""

    def compute(
        self,
        df: Optional[pd.DataFrame],
        context: Optional[dict] = None,
        now: Optional[datetime] = None,
    ) -> MarketFingerprint:
        context = context or {}
        now = now or datetime.now(timezone.utc)

        if df is None or len(df) < 30:
            return self._unknown_fingerprint(now)

        close = df["close"].astype(float)
        volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(np.zeros(len(df)), index=df.index)

        trend = self._classify_trend(close)
        volatility = self._classify_volatility(close)
        liquidity = self._classify_liquidity(volume)
        momentum = self._classify_momentum(close)
        correlation = self._classify_correlation(context)
        stress = self._classify_stress(close, volatility)
        seasonality = self._classify_seasonality(now)
        session = self._classify_session(now)

        scores = [trend.confidence, volatility.confidence, liquidity.confidence,
                  momentum.confidence, stress.confidence]
        overall_confidence = float(np.mean(scores))

        # Derive overall regime label
        if "Uptrend" in trend.label or "Downtrend" in trend.label:
            regime_trend = "TRENDING"
        else:
            regime_trend = "RANGING"

        if volatility.label in ("Expansion", "Extreme"):
            regime_vol = "HIGH_VOL"
        else:
            regime_vol = "LOW_VOL"

        if stress.label == "Panic":
            regime_stress = "_PANIC"
        elif stress.label == "Elevated":
            regime_stress = "_STRESSED"
        else:
            regime_stress = ""

        overall_regime = f"{regime_trend}_{regime_vol}{regime_stress}"

        return MarketFingerprint(
            timestamp=now,
            trend=trend,
            volatility=volatility,
            liquidity=liquidity,
            momentum=momentum,
            correlation=correlation,
            stress=stress,
            seasonality=seasonality,
            session=session,
            overall_regime=overall_regime,
            overall_confidence=overall_confidence,
            diagnostics={
                "bars_analyzed": len(df),
                "last_close": float(close.iloc[-1]),
            },
        )

    # ─────────────────────────────────────────────────────────────────────
    # Dimension Classifiers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_trend(close: pd.Series) -> DimensionScore:
        ret_20 = float(close.pct_change(20).iloc[-1]) if len(close) > 20 else 0.0
        ret_50 = float(close.pct_change(50).iloc[-1]) if len(close) > 50 else ret_20

        if pd.isna(ret_20):
            ret_20 = 0.0
        if pd.isna(ret_50):
            ret_50 = ret_20

        avg_ret = (ret_20 + ret_50) / 2.0

        if avg_ret >= 0.06:
            return DimensionScore("Strong Uptrend", min(avg_ret / 0.10, 1.0), avg_ret)
        elif avg_ret >= 0.02:
            return DimensionScore("Weak Uptrend", min(avg_ret / 0.06, 1.0), avg_ret)
        elif avg_ret <= -0.06:
            return DimensionScore("Strong Downtrend", min(abs(avg_ret) / 0.10, 1.0), avg_ret)
        elif avg_ret <= -0.02:
            return DimensionScore("Weak Downtrend", min(abs(avg_ret) / 0.06, 1.0), avg_ret)
        else:
            return DimensionScore("Sideways", 0.65, avg_ret)

    @staticmethod
    def _classify_volatility(close: pd.Series) -> DimensionScore:
        short_vol = close.pct_change().rolling(10).std().iloc[-1]
        long_vol = close.pct_change().rolling(50).std().iloc[-1] if len(close) > 50 else short_vol

        if pd.isna(short_vol) or pd.isna(long_vol) or long_vol == 0:
            return DimensionScore("Normal", 0.5, 1.0)

        ratio = float(short_vol / long_vol)
        if ratio < 0.7:
            return DimensionScore("Compression", min((1.0 - ratio) / 0.5, 1.0), ratio)
        elif ratio < 1.3:
            return DimensionScore("Normal", 0.7, ratio)
        elif ratio < 2.0:
            return DimensionScore("Expansion", min((ratio - 1.0) / 1.0, 1.0), ratio)
        else:
            return DimensionScore("Extreme", 1.0, ratio)

    @staticmethod
    def _classify_liquidity(volume: pd.Series) -> DimensionScore:
        if volume.sum() == 0:
            return DimensionScore("Medium", 0.5, 1.0)

        vol_ma = volume.rolling(20).mean().iloc[-1]
        if pd.isna(vol_ma) or vol_ma == 0:
            return DimensionScore("Medium", 0.5, 1.0)

        ratio = float(volume.iloc[-1] / vol_ma)
        if ratio >= 1.5:
            return DimensionScore("High", min(ratio / 2.5, 1.0), ratio)
        elif ratio >= 0.75:
            return DimensionScore("Medium", 0.7, ratio)
        else:
            return DimensionScore("Low", min((1.0 - ratio) / 0.5, 1.0), ratio)

    @staticmethod
    def _classify_momentum(close: pd.Series) -> DimensionScore:
        roc_5 = float(close.pct_change(5).iloc[-1]) if len(close) > 5 else 0.0
        roc_10 = float(close.pct_change(10).iloc[-1]) if len(close) > 10 else 0.0

        if pd.isna(roc_5):
            roc_5 = 0.0
        if pd.isna(roc_10):
            roc_10 = roc_5

        avg_mom = (roc_5 + roc_10) / 2.0

        if avg_mom >= 0.03:
            return DimensionScore("Strong Bullish", min(avg_mom / 0.06, 1.0), avg_mom)
        elif avg_mom >= 0.01:
            return DimensionScore("Mild Bullish", min(avg_mom / 0.03, 1.0), avg_mom)
        elif avg_mom <= -0.03:
            return DimensionScore("Strong Bearish", min(abs(avg_mom) / 0.06, 1.0), avg_mom)
        elif avg_mom <= -0.01:
            return DimensionScore("Mild Bearish", min(abs(avg_mom) / 0.03, 1.0), avg_mom)
        else:
            return DimensionScore("Neutral", 0.6, avg_mom)

    @staticmethod
    def _classify_correlation(context: dict) -> DimensionScore:
        benchmark_ret = context.get("benchmark_return", 0.0)
        cross_corr = context.get("cross_asset_corr", 0.0)

        if benchmark_ret > 0.005 and cross_corr > 0.3:
            return DimensionScore("Risk-On", min(abs(benchmark_ret) / 0.02, 1.0), cross_corr)
        elif benchmark_ret < -0.005 and cross_corr > 0.3:
            return DimensionScore("Risk-Off", min(abs(benchmark_ret) / 0.02, 1.0), cross_corr)
        else:
            return DimensionScore("Neutral", 0.55, cross_corr)

    @staticmethod
    def _classify_stress(close: pd.Series, volatility: DimensionScore) -> DimensionScore:
        if len(close) < 20:
            return DimensionScore("Calm", 0.5, 0.0)

        peak = close.rolling(20).max().iloc[-1]
        current = close.iloc[-1]
        drawdown = float((current - peak) / peak) if peak > 0 else 0.0

        if volatility.label == "Extreme" or drawdown <= -0.06:
            return DimensionScore("Panic", 1.0, drawdown)
        elif volatility.label == "Expansion" or drawdown <= -0.03:
            return DimensionScore("Elevated", 0.75, drawdown)
        else:
            return DimensionScore("Calm", 0.8, drawdown)

    @staticmethod
    def _classify_seasonality(now: datetime) -> DimensionScore:
        dow = now.weekday()  # 0=Mon, 4=Fri
        # Monday/Friday historically higher vol; midweek more stable
        if dow == 0:
            return DimensionScore("Monday Effect", 0.6, float(dow))
        elif dow == 4:
            return DimensionScore("Friday Effect", 0.6, float(dow))
        else:
            return DimensionScore("Midweek", 0.7, float(dow))

    @staticmethod
    def _classify_session(now: datetime) -> DimensionScore:
        hour = now.hour
        if 9 <= hour < 10:
            return DimensionScore("Market Open", 0.8, float(hour))
        elif 10 <= hour < 15:
            return DimensionScore("Core Hours", 0.85, float(hour))
        elif 15 <= hour < 16:
            return DimensionScore("Power Hour", 0.75, float(hour))
        elif 16 <= hour < 20:
            return DimensionScore("After Hours", 0.5, float(hour))
        else:
            return DimensionScore("Off Hours", 0.4, float(hour))

    @staticmethod
    def _unknown_fingerprint(now: datetime) -> MarketFingerprint:
        unknown = DimensionScore("Unknown", 0.3, 0.0)
        return MarketFingerprint(
            timestamp=now,
            trend=unknown,
            volatility=unknown,
            liquidity=unknown,
            momentum=unknown,
            correlation=unknown,
            stress=unknown,
            seasonality=unknown,
            session=unknown,
            overall_regime="UNKNOWN",
            overall_confidence=0.3,
            diagnostics={"reason": "insufficient_data"},
        )

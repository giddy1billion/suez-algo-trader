"""
Data Quality Gate — rejects signals built on unreliable inputs.

A model can be highly confident on bad data. This gate ensures
the underlying inputs (candle freshness, volume, feature completeness,
spread availability) meet minimum standards before confidence is
even considered.

Evaluated BEFORE model health and calibration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.intelligence.confidence.models import (
    DataQuality,
    GateResult,
    GateVerdict,
)


@dataclass
class DataQualityConfig:
    """Configuration for data quality thresholds."""

    max_candle_age_seconds: float = 120.0  # 2 minutes stale = degraded
    critical_candle_age_seconds: float = 600.0  # 10 minutes = reject
    min_feature_completeness: float = 0.70  # 70% features must be non-NaN
    critical_feature_completeness: float = 0.50  # Below this = hard reject
    min_bars: int = 100  # Minimum bars for feature computation
    min_volume_freshness: float = 0.3  # Volume data quality floor
    max_orderbook_delay_ms: float = 5000.0  # 5 seconds max orderbook delay
    spread_required: bool = True  # Reject if spread unavailable


class DataQualityGate:
    """
    Evaluates input data quality and produces a composite score.

    Scoring:
    - Feature completeness: 40% weight (most critical)
    - Candle freshness: 25% weight
    - Volume freshness: 15% weight
    - Bar sufficiency: 10% weight
    - Spread availability: 10% weight
    """

    def __init__(self, config: Optional[DataQualityConfig] = None):
        self.config = config or DataQualityConfig()

    def evaluate(
        self,
        features: Optional[np.ndarray] = None,
        last_candle_timestamp: Optional[float] = None,
        volume_data_age_seconds: float = 0.0,
        bars_available: int = 0,
        spread_available: bool = True,
        orderbook_delay_ms: float = 0.0,
    ) -> tuple[GateResult, DataQuality]:
        """
        Evaluate data quality and return gate result + assessment.

        Args:
            features: Feature array (checks for NaN/Inf)
            last_candle_timestamp: Unix timestamp of last candle
            volume_data_age_seconds: How old the volume data is
            bars_available: Number of bars in the lookback window
            spread_available: Whether bid/ask spread is available
            orderbook_delay_ms: Order book data delay
        """
        start = time.perf_counter()

        # ── Feature completeness ──
        feature_completeness = 1.0
        if features is not None and features.size > 0:
            valid = np.isfinite(features).sum()
            feature_completeness = valid / features.size
        elif features is not None and features.size == 0:
            feature_completeness = 0.0

        # ── Candle freshness ──
        candle_age_seconds = 0.0
        candle_freshness = 1.0
        if last_candle_timestamp is not None:
            candle_age_seconds = time.time() - last_candle_timestamp
            if candle_age_seconds >= self.config.critical_candle_age_seconds:
                candle_freshness = 0.0
            elif candle_age_seconds >= self.config.max_candle_age_seconds:
                # Linear decay from 1.0 → 0.0 between max and critical
                range_span = (
                    self.config.critical_candle_age_seconds
                    - self.config.max_candle_age_seconds
                )
                elapsed_past_max = candle_age_seconds - self.config.max_candle_age_seconds
                candle_freshness = max(0.0, 1.0 - (elapsed_past_max / range_span))

        # ── Volume freshness ──
        volume_freshness = 1.0
        if volume_data_age_seconds > 0:
            # Decay: fresh at 0s, 0.5 at 60s, 0.0 at 300s
            volume_freshness = max(0.0, 1.0 - (volume_data_age_seconds / 300.0))

        # ── Bar sufficiency ──
        bar_score = min(1.0, bars_available / max(1, self.config.min_bars))

        # ── Spread ──
        spread_score = 1.0 if spread_available else 0.0

        # ── Composite score (weighted) ──
        overall = (
            feature_completeness * 0.40
            + candle_freshness * 0.25
            + volume_freshness * 0.15
            + bar_score * 0.10
            + spread_score * 0.10
        )

        quality = DataQuality(
            last_candle_age_seconds=candle_age_seconds,
            volume_freshness=volume_freshness,
            spread_available=spread_available,
            feature_completeness=feature_completeness,
            bars_available=bars_available,
            bars_required=self.config.min_bars,
            orderbook_delay_ms=orderbook_delay_ms,
            overall_score=overall,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # ── Gate Decision ──
        if feature_completeness < self.config.critical_feature_completeness:
            return (
                GateResult(
                    gate_name="data_quality",
                    verdict=GateVerdict.REJECT,
                    score=overall,
                    reason=f"Feature completeness {feature_completeness:.0%} below critical threshold ({self.config.critical_feature_completeness:.0%})",
                    details={"feature_completeness": feature_completeness},
                    elapsed_ms=elapsed_ms,
                ),
                quality,
            )

        if candle_freshness == 0.0:
            return (
                GateResult(
                    gate_name="data_quality",
                    verdict=GateVerdict.REJECT,
                    score=overall,
                    reason=f"Candle data critically stale ({candle_age_seconds:.0f}s old, max {self.config.critical_candle_age_seconds:.0f}s)",
                    details={"candle_age_seconds": candle_age_seconds},
                    elapsed_ms=elapsed_ms,
                ),
                quality,
            )

        if (
            self.config.spread_required
            and not spread_available
            and bars_available >= self.config.min_bars
        ):
            return (
                GateResult(
                    gate_name="data_quality",
                    verdict=GateVerdict.REJECT,
                    score=overall,
                    reason="Spread data unavailable — cannot assess execution cost",
                    details={"spread_available": False},
                    elapsed_ms=elapsed_ms,
                ),
                quality,
            )

        if overall < 0.5:
            return (
                GateResult(
                    gate_name="data_quality",
                    verdict=GateVerdict.REJECT,
                    score=overall,
                    reason=f"Composite data quality {overall:.2f} below minimum (0.50)",
                    details=quality.__dict__,
                    elapsed_ms=elapsed_ms,
                ),
                quality,
            )

        # Passed — but may adjust confidence downward
        adjustment = 0.0
        verdict = GateVerdict.PASS
        reason = "Data quality sufficient"

        if overall < 0.8:
            # Partial penalty
            adjustment = -(1.0 - overall) * 0.2  # Up to -10% confidence penalty
            verdict = GateVerdict.ADJUST
            reason = f"Data quality {overall:.2f} — applying {adjustment:.1%} confidence adjustment"

        return (
            GateResult(
                gate_name="data_quality",
                verdict=verdict,
                score=overall,
                adjustment=adjustment,
                reason=reason,
                elapsed_ms=elapsed_ms,
            ),
            quality,
        )

"""
Confidence Decay — temporal degradation of signal confidence.

A prediction becomes stale. Market conditions change, order flow shifts,
and the information that generated a high-confidence signal erodes.

This module computes how much confidence should be discounted based on
elapsed time since signal generation.

Supports multiple decay curves:
- Linear: steady decay per minute
- Exponential: rapid early decay, slow tail
- Step: discrete confidence drops at time boundaries

Evaluated AFTER calibration, BEFORE regime adjustment.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.intelligence.confidence.models import GateResult, GateVerdict


class DecayCurve(str, Enum):
    """Type of confidence decay over time."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"


@dataclass
class DecayConfig:
    """Configuration for confidence decay behavior."""

    curve: DecayCurve = DecayCurve.EXPONENTIAL
    half_life_minutes: float = 60.0  # Time for confidence to halve (exponential)
    linear_rate_per_minute: float = 0.001  # ~6% per hour (linear)
    invalidation_threshold: float = 0.55  # Below this → signal is dead
    max_age_minutes: float = 480.0  # Hard cap — 8 hours max

    # Step decay boundaries (list of (minutes, multiplier))
    step_boundaries: list[tuple[float, float]] = None

    def __post_init__(self):
        if self.step_boundaries is None:
            self.step_boundaries = [
                (15.0, 0.95),   # First 15 min: 95% confidence retained
                (30.0, 0.88),   # 30 min: 88%
                (60.0, 0.78),   # 1 hour: 78%
                (120.0, 0.65),  # 2 hours: 65%
                (240.0, 0.50),  # 4 hours: 50%
                (480.0, 0.30),  # 8 hours: 30%
            ]


class ConfidenceDecayEngine:
    """
    Computes temporal decay of confidence.

    Usage:
        decay = ConfidenceDecayEngine(config)
        factor = decay.compute_factor(elapsed_minutes=45.0)
        adjusted_confidence = raw_confidence * factor
    """

    def __init__(self, config: Optional[DecayConfig] = None):
        self.config = config or DecayConfig()

    def compute_factor(self, elapsed_minutes: float) -> float:
        """
        Compute the decay factor (0.0–1.0) for a given elapsed time.

        Returns:
            Multiplier to apply to confidence. 1.0 = no decay, 0.0 = fully decayed.
        """
        if elapsed_minutes <= 0:
            return 1.0

        if elapsed_minutes >= self.config.max_age_minutes:
            return 0.0

        if self.config.curve == DecayCurve.LINEAR:
            return self._linear_decay(elapsed_minutes)
        elif self.config.curve == DecayCurve.EXPONENTIAL:
            return self._exponential_decay(elapsed_minutes)
        elif self.config.curve == DecayCurve.STEP:
            return self._step_decay(elapsed_minutes)
        else:
            return self._exponential_decay(elapsed_minutes)

    def _linear_decay(self, minutes: float) -> float:
        factor = 1.0 - (self.config.linear_rate_per_minute * minutes)
        return max(0.0, factor)

    def _exponential_decay(self, minutes: float) -> float:
        # f(t) = 0.5^(t / half_life)
        half_life = self.config.half_life_minutes
        if half_life <= 0:
            return 0.0
        factor = math.pow(0.5, minutes / half_life)
        return max(0.0, factor)

    def _step_decay(self, minutes: float) -> float:
        # Find the applicable step boundary
        factor = 1.0
        for boundary_minutes, boundary_factor in self.config.step_boundaries:
            if minutes >= boundary_minutes:
                factor = boundary_factor
            else:
                break
        return factor

    def is_expired(self, elapsed_minutes: float, current_confidence: float) -> bool:
        """Check if a signal has decayed below the invalidation threshold."""
        decayed = current_confidence * self.compute_factor(elapsed_minutes)
        return decayed < self.config.invalidation_threshold

    def evaluate(
        self,
        elapsed_minutes: float,
        current_confidence: float,
    ) -> GateResult:
        """
        Evaluate temporal decay and return gate result.

        Args:
            elapsed_minutes: Time since signal was generated
            current_confidence: Confidence before decay is applied
        """
        start = time.perf_counter()

        factor = self.compute_factor(elapsed_minutes)
        decayed_confidence = current_confidence * factor
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Hard expiry
        if elapsed_minutes >= self.config.max_age_minutes:
            return GateResult(
                gate_name="confidence_decay",
                verdict=GateVerdict.REJECT,
                score=0.0,
                reason=f"Signal expired: {elapsed_minutes:.0f}min elapsed (max {self.config.max_age_minutes:.0f}min)",
                details={"elapsed_minutes": elapsed_minutes, "factor": factor},
                elapsed_ms=elapsed_ms,
            )

        # Below invalidation threshold
        if decayed_confidence < self.config.invalidation_threshold:
            return GateResult(
                gate_name="confidence_decay",
                verdict=GateVerdict.REJECT,
                score=factor,
                reason=(
                    f"Confidence decayed below threshold: "
                    f"{current_confidence:.2f} × {factor:.3f} = {decayed_confidence:.2f} "
                    f"(min: {self.config.invalidation_threshold:.2f})"
                ),
                details={
                    "elapsed_minutes": elapsed_minutes,
                    "factor": factor,
                    "decayed_confidence": decayed_confidence,
                },
                elapsed_ms=elapsed_ms,
            )

        # No meaningful decay yet
        if factor >= 0.99:
            return GateResult(
                gate_name="confidence_decay",
                verdict=GateVerdict.PASS,
                score=factor,
                reason=f"Signal fresh ({elapsed_minutes:.1f}min old)",
                elapsed_ms=elapsed_ms,
            )

        # Apply decay as adjustment
        adjustment = -(1.0 - factor) * current_confidence
        return GateResult(
            gate_name="confidence_decay",
            verdict=GateVerdict.ADJUST,
            score=factor,
            adjustment=factor - 1.0,  # Multiplicative factor as negative adjustment
            reason=(
                f"Temporal decay: {elapsed_minutes:.0f}min elapsed, "
                f"factor={factor:.3f} ({self.config.curve.value} curve, "
                f"half-life={self.config.half_life_minutes:.0f}min)"
            ),
            details={
                "elapsed_minutes": elapsed_minutes,
                "factor": factor,
                "curve": self.config.curve.value,
                "decayed_confidence": decayed_confidence,
            },
            elapsed_ms=elapsed_ms,
        )

"""Adaptive thresholds — dynamic regime-aware confidence thresholds.

Key insight: Instead of a static 0.65 threshold, learn per-regime:
Trending=0.59, MeanReverting=0.72, HighVol=0.76, Calm=0.62.
Thresholds slowly adapt based on observed outcomes.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Safety bounds: never let thresholds escape these
MIN_THRESHOLD = 0.45
MAX_THRESHOLD = 0.85

# Default starting thresholds per regime
DEFAULT_THRESHOLDS: dict[str, float] = {
    "trending": 0.59,
    "mean_reverting": 0.72,
    "high_vol": 0.76,
    "calm": 0.62,
}


@dataclass
class AdaptiveThreshold:
    """A learned threshold for a specific regime."""

    regime: str
    threshold: float
    confidence_interval: tuple[float, float] = (0.0, 1.0)
    sample_size: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    win_rate_at_threshold: float = 0.0

    def to_dict(self) -> dict:
        """Serialize."""
        return {
            "regime": self.regime,
            "threshold": round(self.threshold, 4),
            "confidence_interval": (
                round(self.confidence_interval[0], 4),
                round(self.confidence_interval[1], 4),
            ),
            "sample_size": self.sample_size,
            "last_updated": self.last_updated.isoformat(),
            "win_rate_at_threshold": round(self.win_rate_at_threshold, 4),
        }


@dataclass
class _RegimeState:
    """Internal mutable state for a single regime."""

    threshold: float
    outcomes: list[tuple[float, bool]] = field(default_factory=list)  # (confidence, was_profitable)
    ema_win_rate: float = 0.55  # Exponential moving average of win rate
    sample_count: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AdaptiveThresholdEngine:
    """Maintains and adapts per-regime confidence thresholds.

    Uses exponential moving average of outcomes to slowly adjust thresholds.
    If signals at confidence=X in regime R win > 55% of the time,
    the threshold for R should be ≤ X.

    Thread-safe.
    """

    def __init__(
        self,
        initial_thresholds: Optional[dict[str, float]] = None,
        learning_rate: float = 0.01,
        min_samples_for_update: int = 20,
        target_win_rate: float = 0.55,
    ) -> None:
        """Initialize the engine.

        Args:
            initial_thresholds: Starting thresholds per regime. Defaults to DEFAULT_THRESHOLDS.
            learning_rate: EMA learning rate for threshold adaptation (slow = stable)
            min_samples_for_update: Minimum samples before allowing threshold updates
            target_win_rate: Win rate target that drives threshold adjustment
        """
        self._lock = threading.Lock()
        self.learning_rate = learning_rate
        self.min_samples_for_update = min_samples_for_update
        self.target_win_rate = target_win_rate

        thresholds = initial_thresholds or DEFAULT_THRESHOLDS.copy()
        self._regimes: dict[str, _RegimeState] = {}

        for regime, threshold in thresholds.items():
            clamped = max(MIN_THRESHOLD, min(MAX_THRESHOLD, threshold))
            self._regimes[regime] = _RegimeState(threshold=clamped)

    def record_outcome(self, regime: str, confidence: float, was_profitable: bool) -> None:
        """Record a trade outcome for threshold learning.

        Args:
            regime: Market regime when the trade was taken
            confidence: Model confidence at trade time
            was_profitable: Whether the trade was profitable
        """
        with self._lock:
            state = self._regimes.get(regime)
            if state is None:
                # New regime encountered — initialize with default
                default_thresh = DEFAULT_THRESHOLDS.get(regime, 0.65)
                state = _RegimeState(threshold=default_thresh)
                self._regimes[regime] = state

            state.outcomes.append((confidence, was_profitable))
            state.sample_count += 1

            # Update EMA of win rate
            win_val = 1.0 if was_profitable else 0.0
            state.ema_win_rate = (
                state.ema_win_rate * (1.0 - self.learning_rate)
                + win_val * self.learning_rate
            )

            # Adapt threshold if enough samples
            if state.sample_count >= self.min_samples_for_update:
                self._adapt_threshold(state, regime)

    def get_threshold(self, regime: str) -> float:
        """Get current optimal threshold for a regime.

        Args:
            regime: Market regime

        Returns:
            Current threshold (clamped to safety bounds)
        """
        with self._lock:
            state = self._regimes.get(regime)
            if state is None:
                return DEFAULT_THRESHOLDS.get(regime, 0.65)
            return state.threshold

    def get_adaptive_threshold(self, regime: str) -> AdaptiveThreshold:
        """Get full threshold object with metadata.

        Args:
            regime: Market regime

        Returns:
            AdaptiveThreshold with confidence interval and sample info
        """
        with self._lock:
            state = self._regimes.get(regime)
            if state is None:
                default = DEFAULT_THRESHOLDS.get(regime, 0.65)
                return AdaptiveThreshold(
                    regime=regime,
                    threshold=default,
                    confidence_interval=(default - 0.05, default + 0.05),
                    sample_size=0,
                    win_rate_at_threshold=0.0,
                )

            # Compute confidence interval from recent outcomes
            ci = self._compute_confidence_interval(state)

            return AdaptiveThreshold(
                regime=regime,
                threshold=state.threshold,
                confidence_interval=ci,
                sample_size=state.sample_count,
                last_updated=state.last_updated,
                win_rate_at_threshold=state.ema_win_rate,
            )

    def get_report(self) -> dict:
        """Get all regimes + their current thresholds + sample sizes.

        Returns:
            Dict with regime details and overall stats
        """
        with self._lock:
            regimes = {}
            for regime, state in self._regimes.items():
                regimes[regime] = {
                    "threshold": round(state.threshold, 4),
                    "sample_size": state.sample_count,
                    "ema_win_rate": round(state.ema_win_rate, 4),
                    "last_updated": state.last_updated.isoformat(),
                }

            return {
                "regimes": regimes,
                "learning_rate": self.learning_rate,
                "target_win_rate": self.target_win_rate,
                "min_samples_for_update": self.min_samples_for_update,
                "bounds": {"min": MIN_THRESHOLD, "max": MAX_THRESHOLD},
            }

    def _adapt_threshold(self, state: _RegimeState, regime: str) -> None:
        """Adapt threshold based on observed win rates at different confidence levels.

        Logic: If trades near the current threshold are winning > target_win_rate,
        the threshold can be lowered (we're being too strict). If winning less,
        raise the threshold.
        """
        # Get recent outcomes near the current threshold (within ±0.1)
        near_threshold = [
            (conf, won)
            for conf, won in state.outcomes[-200:]  # Last 200 trades
            if abs(conf - state.threshold) < 0.1
        ]

        if len(near_threshold) < 10:
            return

        # Win rate for trades near the threshold
        near_wins = sum(1 for _, won in near_threshold if won)
        near_win_rate = near_wins / len(near_threshold)

        # Adjust threshold
        if near_win_rate > self.target_win_rate:
            # Winning enough near threshold → can lower threshold slightly
            adjustment = -self.learning_rate * (near_win_rate - self.target_win_rate)
        else:
            # Not winning enough → raise threshold
            adjustment = self.learning_rate * (self.target_win_rate - near_win_rate)

        new_threshold = state.threshold + adjustment

        # Clamp to safety bounds
        new_threshold = max(MIN_THRESHOLD, min(MAX_THRESHOLD, new_threshold))
        state.threshold = new_threshold
        state.last_updated = datetime.now(timezone.utc)

        logger.debug(
            f"Regime '{regime}' threshold adapted to {new_threshold:.4f} "
            f"(near-threshold win rate: {near_win_rate:.2%})"
        )

    @staticmethod
    def _compute_confidence_interval(state: _RegimeState) -> tuple[float, float]:
        """Compute confidence interval for the threshold estimate."""
        if state.sample_count < 10:
            return (state.threshold - 0.05, state.threshold + 0.05)

        # Use recent outcomes to estimate threshold uncertainty
        recent = state.outcomes[-100:]
        if not recent:
            return (state.threshold - 0.05, state.threshold + 0.05)

        confidences = [c for c, _ in recent]
        std = float(np.std(confidences)) if len(confidences) > 1 else 0.05
        se = std / np.sqrt(len(confidences))

        lower = max(MIN_THRESHOLD, state.threshold - 1.96 * se)
        upper = min(MAX_THRESHOLD, state.threshold + 1.96 * se)
        return (round(lower, 4), round(upper, 4))

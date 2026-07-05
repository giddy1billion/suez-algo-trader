"""
Confidence Calibrator — adjusts raw confidence to match observed outcomes.

If a model says confidence=0.90 but only wins 64% of the time at that
level, confidence is overestimated. This calibrator uses historical
prediction outcomes to apply a correction factor.

Uses the existing CalibrationAnalyzer infrastructure for ECE/MCE/Brier
metrics and applies Platt-style scaling to raw confidence.

Evaluated AFTER model health, BEFORE temporal decay.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from src.intelligence.confidence.models import GateResult, GateVerdict


@dataclass
class CalibrationConfig:
    """Configuration for confidence calibration."""

    # Minimum predictions needed before calibration is applied
    min_predictions: int = 100

    # Number of calibration bins
    n_bins: int = 10

    # Maximum single-step adjustment (prevents wild swings)
    max_adjustment: float = 0.15

    # ECE threshold below which no adjustment is needed
    well_calibrated_ece: float = 0.05

    # Enable isotonic regression (vs linear scaling)
    use_isotonic: bool = False


@dataclass
class CalibrationBin:
    """A single calibration bin tracking predicted vs actual."""

    bin_lower: float
    bin_upper: float
    predicted_mean: float = 0.0
    actual_accuracy: float = 0.0
    count: int = 0

    @property
    def gap(self) -> float:
        """How far off predictions are from reality in this bin."""
        return self.predicted_mean - self.actual_accuracy


class ConfidenceCalibrator:
    """
    Adjusts confidence based on historical calibration data.

    Maintains a binned history of (predicted_confidence → actual_outcome)
    and uses it to correct future confidence values.

    Example:
        If predictions in the 0.80–0.90 bin historically win only 65%
        of the time, a raw confidence of 0.85 gets adjusted down to ~0.65.
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self._bins: list[CalibrationBin] = self._init_bins()
        self._total_predictions: int = 0
        self._ece: float = 0.0

    def _init_bins(self) -> list[CalibrationBin]:
        """Initialize empty calibration bins."""
        bins = []
        step = 1.0 / self.config.n_bins
        for i in range(self.config.n_bins):
            lower = i * step
            upper = (i + 1) * step
            bins.append(CalibrationBin(bin_lower=lower, bin_upper=upper))
        return bins

    def record_outcome(self, predicted_confidence: float, was_correct: bool) -> None:
        """
        Record a prediction outcome for future calibration.

        Call this after every trade resolves to build calibration history.
        """
        bin_idx = min(
            int(predicted_confidence * self.config.n_bins),
            self.config.n_bins - 1,
        )
        bin_data = self._bins[bin_idx]

        # Incremental mean update
        bin_data.count += 1
        n = bin_data.count
        bin_data.predicted_mean += (predicted_confidence - bin_data.predicted_mean) / n
        outcome = 1.0 if was_correct else 0.0
        bin_data.actual_accuracy += (outcome - bin_data.actual_accuracy) / n

        self._total_predictions += 1

        # Recompute ECE periodically
        if self._total_predictions % 50 == 0:
            self._recompute_ece()

    def _recompute_ece(self) -> None:
        """Recompute Expected Calibration Error from bins."""
        total_samples = sum(b.count for b in self._bins)
        if total_samples == 0:
            self._ece = 0.0
            return

        ece = 0.0
        for b in self._bins:
            if b.count > 0:
                weight = b.count / total_samples
                ece += weight * abs(b.gap)
        self._ece = ece

    def get_calibrated_confidence(self, raw_confidence: float) -> float:
        """
        Apply calibration correction to raw confidence.

        Returns adjusted confidence based on historical accuracy in
        the corresponding bin.
        """
        if self._total_predictions < self.config.min_predictions:
            return raw_confidence  # Not enough data to calibrate

        bin_idx = min(
            int(raw_confidence * self.config.n_bins),
            self.config.n_bins - 1,
        )
        bin_data = self._bins[bin_idx]

        if bin_data.count < 10:
            return raw_confidence  # Not enough data in this bin

        # The calibrated value is the actual accuracy at this confidence level
        calibrated = bin_data.actual_accuracy

        # Blend: don't jump fully to calibrated value, smooth toward it
        # Higher ECE = more aggressive correction
        correction_strength = min(1.0, self._ece / 0.20)
        adjusted = raw_confidence + correction_strength * (calibrated - raw_confidence)

        # Clamp adjustment to prevent wild swings
        max_adj = self.config.max_adjustment
        adjustment = adjusted - raw_confidence
        adjustment = max(-max_adj, min(max_adj, adjustment))

        return max(0.0, min(1.0, raw_confidence + adjustment))

    def evaluate(self, raw_confidence: float) -> GateResult:
        """
        Evaluate and calibrate a confidence value.

        Returns GateResult with the calibration adjustment applied.
        """
        start = time.perf_counter()

        if self._total_predictions < self.config.min_predictions:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return GateResult(
                gate_name="calibration",
                verdict=GateVerdict.SKIP,
                score=1.0,
                reason=f"Insufficient calibration data ({self._total_predictions}/{self.config.min_predictions} predictions)",
                elapsed_ms=elapsed_ms,
            )

        if self._ece <= self.config.well_calibrated_ece:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return GateResult(
                gate_name="calibration",
                verdict=GateVerdict.PASS,
                score=1.0 - self._ece,
                reason=f"Well-calibrated (ECE={self._ece:.3f})",
                elapsed_ms=elapsed_ms,
            )

        calibrated = self.get_calibrated_confidence(raw_confidence)
        adjustment = calibrated - raw_confidence
        elapsed_ms = (time.perf_counter() - start) * 1000

        if abs(adjustment) < 0.01:
            return GateResult(
                gate_name="calibration",
                verdict=GateVerdict.PASS,
                score=1.0 - self._ece,
                reason=f"Minimal calibration needed (ECE={self._ece:.3f}, adj={adjustment:+.3f})",
                elapsed_ms=elapsed_ms,
            )

        direction = "down" if adjustment < 0 else "up"
        return GateResult(
            gate_name="calibration",
            verdict=GateVerdict.ADJUST,
            score=1.0 - self._ece,
            adjustment=adjustment,
            reason=(
                f"Calibration {direction}: {raw_confidence:.2f} → {calibrated:.2f} "
                f"(ECE={self._ece:.3f}, adjustment={adjustment:+.3f})"
            ),
            details={
                "raw": raw_confidence,
                "calibrated": calibrated,
                "ece": self._ece,
                "total_predictions": self._total_predictions,
            },
            elapsed_ms=elapsed_ms,
        )

    def load_from_history(
        self, predictions: list[tuple[float, bool]]
    ) -> None:
        """
        Bulk-load calibration from historical predictions.

        Args:
            predictions: List of (confidence, was_correct) tuples
        """
        for confidence, was_correct in predictions:
            self.record_outcome(confidence, was_correct)

    @property
    def ece(self) -> float:
        return self._ece

    @property
    def total_predictions(self) -> int:
        return self._total_predictions

    def get_calibration_report(self) -> dict:
        """Return full calibration state for auditing."""
        return {
            "total_predictions": self._total_predictions,
            "ece": round(self._ece, 4),
            "bins": [
                {
                    "range": f"{b.bin_lower:.1f}-{b.bin_upper:.1f}",
                    "predicted_mean": round(b.predicted_mean, 3),
                    "actual_accuracy": round(b.actual_accuracy, 3),
                    "gap": round(b.gap, 3),
                    "count": b.count,
                }
                for b in self._bins
                if b.count > 0
            ],
        }

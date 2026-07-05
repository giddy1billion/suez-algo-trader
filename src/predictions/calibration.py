"""
Calibration Analyzer — Confidence calibration analysis for predictions.

Evaluates whether predicted confidence scores are well-calibrated:
a prediction with 80% confidence should be correct ~80% of the time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.predictions.registry import PredictionRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationPoint:
    """A single point on the calibration curve."""
    bin_center: float  # Predicted confidence center
    predicted_confidence: float  # Mean predicted confidence in bin
    actual_accuracy: float  # Actual accuracy in bin
    count: int  # Number of predictions in bin


@dataclass
class CalibrationReport:
    """Full calibration analysis report."""
    calibration_curve: list[CalibrationPoint] = field(default_factory=list)
    expected_calibration_error: float = 0.0  # ECE
    maximum_calibration_error: float = 0.0  # MCE
    brier_score: float = 0.0
    reliability_score: float = 0.0  # 1 - ECE (higher is better)
    is_overconfident: bool = False
    is_underconfident: bool = False
    recommendation: str = ""


class CalibrationAnalyzer:
    """
    Analyzes prediction confidence calibration.

    A well-calibrated model produces confidence scores that match
    the actual frequency of correct predictions.
    """

    def __init__(self, n_bins: int = 10):
        """
        Args:
            n_bins: Number of bins for calibration curve.
        """
        self._n_bins = n_bins

    def analyze(self, predictions: list[PredictionRecord]) -> CalibrationReport:
        """
        Perform full calibration analysis.

        Args:
            predictions: List of resolved predictions with outcomes.

        Returns:
            CalibrationReport with curve, ECE, MCE, and recommendations.
        """
        resolved = [p for p in predictions if p.resolved and p.direction_correct is not None]
        if len(resolved) < 10:
            return CalibrationReport(recommendation="Insufficient data for calibration analysis")

        # Build calibration curve
        bin_edges = np.linspace(0.5, 1.0, self._n_bins + 1)
        curve: list[CalibrationPoint] = []
        bin_weights: list[float] = []
        bin_gaps: list[float] = []

        for i in range(self._n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            bin_center = (low + high) / 2

            in_bin = [
                p for p in resolved
                if low <= p.confidence < high
            ]

            if not in_bin:
                continue

            predicted_conf = np.mean([p.confidence for p in in_bin])
            actual_acc = sum(1 for p in in_bin if p.direction_correct) / len(in_bin)

            curve.append(CalibrationPoint(
                bin_center=float(bin_center),
                predicted_confidence=float(predicted_conf),
                actual_accuracy=float(actual_acc),
                count=len(in_bin),
            ))

            weight = len(in_bin) / len(resolved)
            gap = abs(actual_acc - predicted_conf)
            bin_weights.append(weight)
            bin_gaps.append(gap)

        # Expected Calibration Error (weighted average of bin gaps)
        ece = sum(w * g for w, g in zip(bin_weights, bin_gaps)) if bin_gaps else 0.0
        mce = max(bin_gaps) if bin_gaps else 0.0

        # Brier Score
        brier_scores = [
            (p.confidence - (1.0 if p.direction_correct else 0.0)) ** 2
            for p in resolved
        ]
        brier = float(np.mean(brier_scores)) if brier_scores else 0.0

        # Determine over/under confidence
        overconfident_bins = sum(
            1 for point in curve
            if point.predicted_confidence > point.actual_accuracy
        )
        total_bins = len(curve)
        is_overconfident = overconfident_bins > total_bins * 0.6 if total_bins > 0 else False
        is_underconfident = overconfident_bins < total_bins * 0.4 if total_bins > 0 else False

        # Recommendation
        recommendation = self._generate_recommendation(ece, is_overconfident, is_underconfident)

        return CalibrationReport(
            calibration_curve=curve,
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            brier_score=brier,
            reliability_score=1.0 - ece,
            is_overconfident=is_overconfident,
            is_underconfident=is_underconfident,
            recommendation=recommendation,
        )

    def _generate_recommendation(
        self, ece: float, overconfident: bool, underconfident: bool
    ) -> str:
        """Generate actionable recommendation based on calibration."""
        if ece < 0.05:
            return "Well-calibrated. No adjustment needed."
        elif ece < 0.10:
            if overconfident:
                return "Slightly overconfident. Consider Platt scaling or temperature adjustment."
            elif underconfident:
                return "Slightly underconfident. Model is conservative."
            return "Minor calibration gap. Monitor for drift."
        elif ece < 0.20:
            if overconfident:
                return "Significantly overconfident. Apply isotonic regression or reduce confidence thresholds."
            return "Moderate calibration error. Recommend recalibration."
        else:
            return "Severe miscalibration. Retraining with calibration loss recommended."

    def compute_reliability_diagram_data(
        self, predictions: list[PredictionRecord]
    ) -> dict[str, list[float]]:
        """
        Get data for plotting a reliability diagram.

        Returns dict with 'predicted' and 'actual' lists for plotting.
        """
        report = self.analyze(predictions)
        return {
            "predicted": [p.predicted_confidence for p in report.calibration_curve],
            "actual": [p.actual_accuracy for p in report.calibration_curve],
            "counts": [p.count for p in report.calibration_curve],
        }

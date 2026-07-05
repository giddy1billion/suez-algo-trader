"""
Prediction Metrics — Rolling prediction quality metrics computation.

Computes:
- Directional accuracy (overall and per-asset)
- Precision / Recall / F1 (classification view)
- MAE / RMSE (regression view)
- Brier Score
- Hit rate by confidence bucket
- Profit per prediction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.predictions.registry import PredictionRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricsSnapshot:
    """Point-in-time metrics for prediction quality."""

    total_predictions: int = 0
    directional_accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    brier_score: float = 0.0
    avg_profit_per_prediction: float = 0.0
    hit_rate_by_bucket: dict[str, float] = field(default_factory=dict)
    per_asset_accuracy: dict[str, float] = field(default_factory=dict)


class PredictionMetrics:
    """
    Computes rolling prediction quality metrics from resolved predictions.
    """

    def __init__(self, window: int = 200):
        """
        Args:
            window: Number of recent resolved predictions to include in metrics.
        """
        self._window = window

    def compute(self, predictions: list[PredictionRecord]) -> MetricsSnapshot:
        """Compute metrics from a list of resolved predictions."""
        resolved = [p for p in predictions if p.resolved]
        if not resolved:
            return MetricsSnapshot()

        # Use most recent window
        recent = resolved[-self._window:]

        total = len(recent)
        correct = [p for p in recent if p.direction_correct]
        incorrect = [p for p in recent if not p.direction_correct]

        # Directional accuracy
        accuracy = len(correct) / total if total > 0 else 0.0

        # Precision/Recall/F1 (treating "direction correct" as positive class)
        tp = len(correct)
        fp = 0  # No false positives in this binary framing
        fn = len(incorrect)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # MAE and RMSE
        errors = [
            p.absolute_error for p in recent
            if p.absolute_error is not None
        ]
        mae = float(np.mean(errors)) if errors else 0.0
        rmse = float(np.sqrt(np.mean([e**2 for e in errors]))) if errors else 0.0

        # Brier Score (for confidence calibration)
        brier = self._compute_brier_score(recent)

        # Profit per prediction
        returns = [p.actual_return for p in recent if p.actual_return is not None]
        avg_profit = float(np.mean(returns)) if returns else 0.0

        # Hit rate by confidence bucket
        hit_rate_buckets = self._hit_rate_by_bucket(recent)

        # Per-asset accuracy
        per_asset = self._per_asset_accuracy(recent)

        return MetricsSnapshot(
            total_predictions=total,
            directional_accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            mae=mae,
            rmse=rmse,
            brier_score=brier,
            avg_profit_per_prediction=avg_profit,
            hit_rate_by_bucket=hit_rate_buckets,
            per_asset_accuracy=per_asset,
        )

    def _compute_brier_score(self, predictions: list[PredictionRecord]) -> float:
        """
        Brier Score: mean squared error between confidence and actual outcome.
        Lower is better. Measures calibration quality.
        """
        scores = []
        for p in predictions:
            if p.direction_correct is None:
                continue
            outcome = 1.0 if p.direction_correct else 0.0
            scores.append((p.confidence - outcome) ** 2)
        return float(np.mean(scores)) if scores else 0.0

    def _hit_rate_by_bucket(self, predictions: list[PredictionRecord]) -> dict[str, float]:
        """Compute hit rate grouped by confidence bucket."""
        buckets: dict[str, list[bool]] = {
            "0.5-0.6": [],
            "0.6-0.7": [],
            "0.7-0.8": [],
            "0.8-0.9": [],
            "0.9-1.0": [],
        }

        for p in predictions:
            if p.direction_correct is None:
                continue
            conf = p.confidence
            if conf < 0.6:
                key = "0.5-0.6"
            elif conf < 0.7:
                key = "0.6-0.7"
            elif conf < 0.8:
                key = "0.7-0.8"
            elif conf < 0.9:
                key = "0.8-0.9"
            else:
                key = "0.9-1.0"
            buckets[key].append(p.direction_correct)

        return {
            k: (sum(v) / len(v) if v else 0.0)
            for k, v in buckets.items()
        }

    def _per_asset_accuracy(self, predictions: list[PredictionRecord]) -> dict[str, float]:
        """Compute directional accuracy per asset."""
        by_asset: dict[str, list[bool]] = {}
        for p in predictions:
            if p.direction_correct is None:
                continue
            if p.asset not in by_asset:
                by_asset[p.asset] = []
            by_asset[p.asset].append(p.direction_correct)

        return {
            asset: sum(results) / len(results) if results else 0.0
            for asset, results in by_asset.items()
        }

"""
Retraining Trigger — Evidence-driven retraining decision logic.

Determines when model retraining should be triggered based on:
1. Sufficient new validated outcomes accumulated
2. Drift monitor detects degradation
3. Scheduled research cycle (fallback)
4. Manual override

Production model NEVER learns directly from each prediction.
Retraining only occurs when governance gates are satisfied.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrainingEvidence:
    """Evidence summary for why retraining was triggered."""

    reason: str  # "sufficient_outcomes" | "drift_detected" | "scheduled" | "manual"
    outcome_count: int = 0
    drift_score: float = 0.0
    brier_score: float = 0.0
    accuracy_drop: float = 0.0
    hours_since_last_training: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [f"reason={self.reason}"]
        if self.outcome_count:
            parts.append(f"outcomes={self.outcome_count}")
        if self.drift_score:
            parts.append(f"drift={self.drift_score:.3f}")
        if self.accuracy_drop:
            parts.append(f"accuracy_drop={self.accuracy_drop:.3f}")
        return ", ".join(parts)


class RetrainingTrigger:
    """
    Evidence-driven retraining decision logic.

    Evaluates multiple signals to determine if retraining is warranted.
    Enforces minimum frequency constraints to prevent over-training.
    """

    def __init__(
        self,
        min_outcomes: int = 0,
        drift_threshold: float = 0.0,
        max_frequency_hours: float = 0.0,
        scheduled_interval_hours: float = 0.0,
        brier_threshold: float = 0.30,
    ):
        self._min_outcomes = min_outcomes or settings.retraining_min_outcomes
        self._drift_threshold = drift_threshold or settings.retraining_drift_threshold
        self._max_frequency_hours = max_frequency_hours or settings.retraining_max_frequency_hours
        self._scheduled_interval_hours = (
            scheduled_interval_hours or settings.retraining_scheduled_interval_hours
        )
        self._brier_threshold = brier_threshold
        self._last_training_time: Optional[float] = None
        self._last_outcome_count: int = 0

    def should_retrain(
        self,
        new_outcome_count: int,
        drift_score: float = 0.0,
        brier_score: float = 0.0,
        current_accuracy: float = 0.0,
        baseline_accuracy: float = 0.0,
    ) -> Optional[RetrainingEvidence]:
        """
        Evaluate whether retraining should be triggered.

        Returns RetrainingEvidence if retraining is warranted, None otherwise.
        """
        # Check frequency constraint
        if not self._can_retrain():
            return None

        # Check evidence signals
        evidence = self._evaluate_evidence(
            new_outcome_count=new_outcome_count,
            drift_score=drift_score,
            brier_score=brier_score,
            current_accuracy=current_accuracy,
            baseline_accuracy=baseline_accuracy,
        )

        return evidence

    def record_training_completed(self) -> None:
        """Record that training was completed (for frequency limiting)."""
        self._last_training_time = time.time()

    def trigger_manual(self) -> RetrainingEvidence:
        """Create manual trigger evidence."""
        return RetrainingEvidence(
            reason="manual",
            hours_since_last_training=self._hours_since_last(),
        )

    def _can_retrain(self) -> bool:
        """Check if enough time has passed since last training."""
        if self._last_training_time is None:
            return True
        hours_elapsed = (time.time() - self._last_training_time) / 3600
        return hours_elapsed >= self._max_frequency_hours

    def _evaluate_evidence(
        self,
        new_outcome_count: int,
        drift_score: float,
        brier_score: float,
        current_accuracy: float,
        baseline_accuracy: float,
    ) -> Optional[RetrainingEvidence]:
        """Evaluate all evidence signals."""
        hours_since = self._hours_since_last()

        # Signal 1: Sufficient new outcomes accumulated
        new_since_last = new_outcome_count - self._last_outcome_count
        if new_since_last >= self._min_outcomes:
            return RetrainingEvidence(
                reason="sufficient_outcomes",
                outcome_count=new_since_last,
                hours_since_last_training=hours_since,
            )

        # Signal 2: Drift detected above threshold
        if drift_score >= self._drift_threshold:
            return RetrainingEvidence(
                reason="drift_detected",
                drift_score=drift_score,
                hours_since_last_training=hours_since,
            )

        # Signal 3: Calibration degradation (high Brier score)
        if brier_score >= self._brier_threshold:
            return RetrainingEvidence(
                reason="calibration_degradation",
                brier_score=brier_score,
                hours_since_last_training=hours_since,
            )

        # Signal 4: Accuracy drop
        if baseline_accuracy > 0 and current_accuracy > 0:
            drop = baseline_accuracy - current_accuracy
            if drop > 0.10:  # 10% accuracy drop
                return RetrainingEvidence(
                    reason="accuracy_degradation",
                    accuracy_drop=drop,
                    hours_since_last_training=hours_since,
                )

        # Signal 5: Scheduled interval (fallback)
        if hours_since >= self._scheduled_interval_hours:
            return RetrainingEvidence(
                reason="scheduled",
                hours_since_last_training=hours_since,
                outcome_count=new_since_last,
            )

        return None

    def _hours_since_last(self) -> float:
        """Hours since last training."""
        if self._last_training_time is None:
            return float("inf")
        return (time.time() - self._last_training_time) / 3600

    def get_status(self) -> dict[str, Any]:
        """Get current trigger status."""
        return {
            "last_training_time": (
                datetime.fromtimestamp(self._last_training_time, tz=timezone.utc).isoformat()
                if self._last_training_time
                else None
            ),
            "hours_since_last": self._hours_since_last(),
            "can_retrain": self._can_retrain(),
            "min_outcomes": self._min_outcomes,
            "drift_threshold": self._drift_threshold,
            "max_frequency_hours": self._max_frequency_hours,
            "scheduled_interval_hours": self._scheduled_interval_hours,
        }

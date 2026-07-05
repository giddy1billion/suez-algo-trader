"""
Model Health Gate — discounts or rejects confidence from degraded models.

A model can report high confidence while its accuracy has collapsed.
This gate uses drift monitoring, calibration metrics, and model age
to determine whether a model's confidence output should be trusted.

Evaluated AFTER data quality, BEFORE calibration adjustment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.intelligence.confidence.models import (
    GateResult,
    GateVerdict,
    ModelHealth,
)


@dataclass
class ModelHealthConfig:
    """Configuration for model health thresholds."""

    # Accuracy degradation thresholds
    accuracy_drop_warning: float = 0.08  # 8% drop → discount
    accuracy_drop_critical: float = 0.15  # 15% drop → reject
    min_predictions_for_health: int = 50  # Need this many predictions to assess

    # Calibration thresholds
    ece_warning: float = 0.10  # 10% ECE → discount
    ece_critical: float = 0.20  # 20% ECE → reject

    # Model age thresholds (hours)
    model_age_warning: float = 168.0  # 7 days old → discount
    model_age_critical: float = 720.0  # 30 days old → heavier discount

    # Health score floor (below this = reject)
    health_floor: float = 0.40


class ModelHealthGate:
    """
    Evaluates model health and produces a composite score.

    Health score composition:
    - Accuracy stability: 40% (no drift)
    - Calibration quality: 30% (ECE)
    - Model freshness: 20% (age since last training)
    - Prediction volume: 10% (enough predictions to assess)
    """

    def __init__(self, config: Optional[ModelHealthConfig] = None):
        self.config = config or ModelHealthConfig()

    def evaluate(
        self,
        model_version: str = "",
        accuracy_baseline: float = 0.0,
        accuracy_recent: float = 0.0,
        predictions_count: int = 0,
        calibration_ece: float = 0.0,
        last_retrained: Optional[datetime] = None,
        is_drift_detected: bool = False,
    ) -> tuple[GateResult, ModelHealth]:
        """
        Evaluate model health and return gate result + assessment.

        Args:
            model_version: Current model version identifier
            accuracy_baseline: Historical mean accuracy
            accuracy_recent: Recent window accuracy
            predictions_count: Total predictions made since deployment
            calibration_ece: Expected Calibration Error (0 = perfect)
            last_retrained: When the model was last retrained
            is_drift_detected: Whether drift monitor has flagged degradation
        """
        start = time.perf_counter()

        # ── Accuracy stability ──
        accuracy_drop = max(0.0, accuracy_baseline - accuracy_recent)
        has_enough_data = predictions_count >= self.config.min_predictions_for_health

        accuracy_score = 1.0
        if has_enough_data:
            if accuracy_drop >= self.config.accuracy_drop_critical:
                accuracy_score = 0.0
            elif accuracy_drop >= self.config.accuracy_drop_warning:
                # Linear degradation between warning and critical
                range_span = (
                    self.config.accuracy_drop_critical
                    - self.config.accuracy_drop_warning
                )
                accuracy_score = max(
                    0.0, 1.0 - (accuracy_drop - self.config.accuracy_drop_warning) / range_span
                )
            # Boost from drift detection override
            if is_drift_detected:
                accuracy_score = min(accuracy_score, 0.3)

        # ── Calibration quality ──
        calibration_score = 1.0
        if calibration_ece > 0:
            if calibration_ece >= self.config.ece_critical:
                calibration_score = 0.0
            elif calibration_ece >= self.config.ece_warning:
                range_span = self.config.ece_critical - self.config.ece_warning
                calibration_score = max(
                    0.0, 1.0 - (calibration_ece - self.config.ece_warning) / range_span
                )

        # ── Model freshness ──
        model_age_hours = 0.0
        freshness_score = 1.0
        if last_retrained is not None:
            model_age_hours = (datetime.now() - last_retrained).total_seconds() / 3600
            if model_age_hours >= self.config.model_age_critical:
                freshness_score = 0.5  # Old but not useless
            elif model_age_hours >= self.config.model_age_warning:
                range_span = self.config.model_age_critical - self.config.model_age_warning
                elapsed = model_age_hours - self.config.model_age_warning
                freshness_score = max(0.5, 1.0 - (elapsed / range_span) * 0.5)

        # ── Prediction volume ──
        volume_score = min(1.0, predictions_count / max(1, self.config.min_predictions_for_health))

        # ── Composite health score ──
        health_score = (
            accuracy_score * 0.40
            + calibration_score * 0.30
            + freshness_score * 0.20
            + volume_score * 0.10
        )

        # ── Calibration reliability (for downstream use) ──
        calibration_reliability = max(0.0, 1.0 - calibration_ece)

        health = ModelHealth(
            model_version=model_version,
            accuracy_baseline=accuracy_baseline,
            accuracy_recent=accuracy_recent,
            accuracy_drop=accuracy_drop,
            is_degrading=is_drift_detected or (
                has_enough_data and accuracy_drop >= self.config.accuracy_drop_warning
            ),
            predictions_count=predictions_count,
            last_retrained=last_retrained,
            model_age_hours=model_age_hours,
            calibration_ece=calibration_ece,
            calibration_reliability=calibration_reliability,
            health_score=health_score,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # ── Gate Decision ──
        if is_drift_detected and accuracy_drop >= self.config.accuracy_drop_critical:
            return (
                GateResult(
                    gate_name="model_health",
                    verdict=GateVerdict.REJECT,
                    score=health_score,
                    reason=(
                        f"Model critically degraded: accuracy dropped {accuracy_drop:.1%} "
                        f"(baseline {accuracy_baseline:.1%} → recent {accuracy_recent:.1%}) "
                        f"with active drift detection"
                    ),
                    details={
                        "accuracy_drop": accuracy_drop,
                        "drift_detected": True,
                        "model_version": model_version,
                    },
                    elapsed_ms=elapsed_ms,
                ),
                health,
            )

        if calibration_ece >= self.config.ece_critical:
            return (
                GateResult(
                    gate_name="model_health",
                    verdict=GateVerdict.REJECT,
                    score=health_score,
                    reason=(
                        f"Model severely miscalibrated: ECE={calibration_ece:.2f} "
                        f"(critical threshold: {self.config.ece_critical:.2f}). "
                        f"Confidence values are unreliable."
                    ),
                    details={"calibration_ece": calibration_ece},
                    elapsed_ms=elapsed_ms,
                ),
                health,
            )

        if health_score < self.config.health_floor:
            return (
                GateResult(
                    gate_name="model_health",
                    verdict=GateVerdict.REJECT,
                    score=health_score,
                    reason=f"Composite model health {health_score:.2f} below floor ({self.config.health_floor:.2f})",
                    details={"health_score": health_score},
                    elapsed_ms=elapsed_ms,
                ),
                health,
            )

        # Passed — may adjust confidence based on health
        if health_score >= 0.9:
            return (
                GateResult(
                    gate_name="model_health",
                    verdict=GateVerdict.PASS,
                    score=health_score,
                    reason="Model healthy",
                    elapsed_ms=elapsed_ms,
                ),
                health,
            )

        # Health is degraded but not critical → apply discount
        discount = (1.0 - health_score) * 0.3  # Up to 18% discount
        return (
            GateResult(
                gate_name="model_health",
                verdict=GateVerdict.ADJUST,
                score=health_score,
                adjustment=-discount,
                reason=(
                    f"Model health {health_score:.2f} — applying {discount:.1%} "
                    f"confidence discount (accuracy_drop={accuracy_drop:.1%}, "
                    f"ECE={calibration_ece:.2f}, age={model_age_hours:.0f}h)"
                ),
                details={
                    "discount": discount,
                    "accuracy_score": accuracy_score,
                    "calibration_score": calibration_score,
                    "freshness_score": freshness_score,
                },
                elapsed_ms=elapsed_ms,
            ),
            health,
        )

"""
Prediction Registry — Core prediction storage and lifecycle management.

Every ML/strategy signal is tracked through its full lifecycle:
prediction → outcome → quality grading → metrics aggregation.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionRecord:
    """A single prediction with full lifecycle tracking."""

    prediction_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    asset: str = ""
    direction: str = ""  # "long" | "short"
    confidence: float = 0.0
    expected_horizon: int = 24  # bars
    expected_return: float = 0.0
    model_version: str = ""
    strategy: str = ""
    features_snapshot_hash: str = ""
    # Provenance metadata
    training_timestamp: str = ""
    dataset_version: str = ""
    feature_set_version: str = ""
    validation_metrics: dict = field(default_factory=dict)
    probability_distribution: list = field(default_factory=list)
    feature_importance: Optional[dict] = None
    # Outcome fields (filled when resolved)
    outcome_timestamp: Optional[str] = None
    actual_return: Optional[float] = None
    direction_correct: Optional[bool] = None
    absolute_error: Optional[float] = None
    quality_grade: Optional[str] = None  # "excellent" | "good" | "fair" | "poor"
    resolved: bool = False

    def has_required_provenance(self) -> bool:
        """Check whether all required provenance metadata is present."""
        return bool(
            self.model_version
            and self.training_timestamp
            and self.dataset_version
            and self.feature_set_version
            and self.validation_metrics
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PredictionRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class PredictionRegistry:
    """
    Core prediction storage with lifecycle management.

    Stores predictions and their outcomes, enabling quality tracking
    and closed-loop learning.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        event_bus=None,
        max_active: int = 10000,
    ):
        self._storage_path = Path(
            storage_path or settings.prediction_registry_storage_path
        )
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._event_bus = event_bus
        self._max_active = max_active
        self._lock = threading.Lock()

        # In-memory active predictions (not yet resolved)
        self._active: dict[str, PredictionRecord] = {}
        # Resolved predictions (recent history)
        self._resolved: list[PredictionRecord] = []

        self._load_active()
        logger.info(
            "prediction_registry.initialized",
            active=len(self._active),
            storage=str(self._storage_path),
        )

    def register(
        self,
        asset: str,
        direction: str,
        confidence: float,
        expected_horizon: int = 0,
        expected_return: float = 0.0,
        model_version: str = "",
        strategy: str = "",
        features_hash: str = "",
        training_timestamp: str = "",
        dataset_version: str = "",
        feature_set_version: str = "",
        validation_metrics: Optional[dict] = None,
        probability_distribution: Optional[list] = None,
        feature_importance: Optional[dict] = None,
    ) -> Optional[PredictionRecord]:
        """Register a new prediction. Returns None if required provenance is missing."""
        horizon = expected_horizon or settings.prediction_default_horizon_bars

        record = PredictionRecord(
            asset=asset,
            direction=direction,
            confidence=confidence,
            expected_horizon=horizon,
            expected_return=expected_return,
            model_version=model_version,
            strategy=strategy,
            features_snapshot_hash=features_hash,
            training_timestamp=training_timestamp,
            dataset_version=dataset_version,
            feature_set_version=feature_set_version,
            validation_metrics=validation_metrics or {},
            probability_distribution=probability_distribution or [],
            feature_importance=feature_importance,
        )

        if not record.has_required_provenance():
            logger.warning(
                "prediction.missing_provenance",
                asset=asset,
                model_version=model_version,
                training_timestamp=training_timestamp,
                dataset_version=dataset_version,
                feature_set_version=feature_set_version,
            )
            return None

        with self._lock:
            self._active[record.prediction_id] = record

        # Publish event
        if self._event_bus:
            from src.core.events import PredictionRegistered
            self._event_bus.publish(PredictionRegistered(
                prediction_id=record.prediction_id,
                asset=asset,
                direction=direction,
                confidence=confidence,
                horizon=str(horizon),
                source="prediction_registry",
            ))

        logger.debug(
            "prediction.registered",
            id=record.prediction_id,
            asset=asset,
            direction=direction,
            confidence=confidence,
        )
        return record

    def record_outcome(
        self,
        prediction_id: str,
        actual_return: float,
    ) -> Optional[PredictionRecord]:
        """Record the outcome of a prediction."""
        with self._lock:
            record = self._active.pop(prediction_id, None)

        if record is None:
            logger.warning("prediction.outcome_not_found", id=prediction_id)
            return None

        # Determine direction correctness
        if record.direction == "long":
            direction_correct = actual_return > 0
        else:
            direction_correct = actual_return < 0

        # Compute absolute error
        absolute_error = abs(actual_return - record.expected_return)

        # Grade quality
        quality_grade = self._grade_prediction(
            direction_correct, absolute_error, record.confidence
        )

        # Update record
        record.outcome_timestamp = datetime.now(timezone.utc).isoformat()
        record.actual_return = actual_return
        record.direction_correct = direction_correct
        record.absolute_error = absolute_error
        record.quality_grade = quality_grade
        record.resolved = True

        with self._lock:
            self._resolved.append(record)
            # Keep resolved bounded
            if len(self._resolved) > self._max_active:
                self._resolved = self._resolved[-self._max_active // 2:]

        # Persist
        self._persist_outcome(record)

        # Publish event
        if self._event_bus:
            from src.core.events import PredictionOutcomeRecorded
            self._event_bus.publish(PredictionOutcomeRecorded(
                prediction_id=prediction_id,
                actual_return=actual_return,
                quality_grade=quality_grade,
                source="prediction_registry",
            ))

        return record

    def get_active_predictions(self, asset: Optional[str] = None) -> list[PredictionRecord]:
        """Get active (unresolved) predictions, optionally filtered by asset."""
        with self._lock:
            if asset:
                return [r for r in self._active.values() if r.asset == asset]
            return list(self._active.values())

    def get_resolved_predictions(
        self, limit: int = 100, asset: Optional[str] = None
    ) -> list[PredictionRecord]:
        """Get recently resolved predictions."""
        with self._lock:
            records = self._resolved
            if asset:
                records = [r for r in records if r.asset == asset]
            return records[-limit:]

    def get_expired_predictions(self, current_bar_counts: dict[str, int]) -> list[PredictionRecord]:
        """Get predictions that have exceeded their horizon."""
        expired = []
        # Simple expiry check based on timestamp comparison
        now = datetime.now(timezone.utc)
        with self._lock:
            for record in self._active.values():
                try:
                    pred_time = datetime.fromisoformat(record.timestamp)
                    # Rough expiry: assume 1 bar = 1 hour for simplicity
                    hours_elapsed = (now - pred_time).total_seconds() / 3600
                    if hours_elapsed >= record.expected_horizon:
                        expired.append(record)
                except (ValueError, TypeError):
                    pass
        return expired

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def resolved_count(self) -> int:
        return len(self._resolved)

    def get_summary(self) -> dict[str, Any]:
        """Get registry summary statistics."""
        with self._lock:
            resolved = self._resolved
        if not resolved:
            return {
                "active": len(self._active),
                "resolved": 0,
                "accuracy": 0.0,
            }
        correct = sum(1 for r in resolved if r.direction_correct)
        return {
            "active": len(self._active),
            "resolved": len(resolved),
            "accuracy": correct / len(resolved),
            "avg_confidence": sum(r.confidence for r in resolved) / len(resolved),
            "quality_distribution": self._quality_distribution(resolved),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _grade_prediction(
        self, direction_correct: bool, absolute_error: float, confidence: float
    ) -> str:
        """Grade prediction quality."""
        if direction_correct and absolute_error < 0.01:
            return "excellent"
        elif direction_correct and absolute_error < 0.03:
            return "good"
        elif direction_correct:
            return "fair"
        else:
            return "poor"

    def _quality_distribution(self, records: list[PredictionRecord]) -> dict[str, int]:
        """Count predictions by quality grade."""
        dist: dict[str, int] = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        for r in records:
            if r.quality_grade in dist:
                dist[r.quality_grade] += 1
        return dist

    def _persist_outcome(self, record: PredictionRecord) -> None:
        """Persist resolved prediction to storage."""
        try:
            outcomes_file = self._storage_path / "outcomes.jsonl"
            with open(outcomes_file, "a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as e:
            logger.warning("prediction.persist_failed", error=str(e))

    def _load_active(self) -> None:
        """Load any persisted active predictions (for recovery)."""
        active_file = self._storage_path / "active.json"
        if active_file.exists():
            try:
                with open(active_file) as f:
                    data = json.load(f)
                for item in data:
                    record = PredictionRecord.from_dict(item)
                    self._active[record.prediction_id] = record
            except Exception as e:
                logger.warning("prediction.load_failed", error=str(e))

    def persist_active(self) -> None:
        """Persist active predictions for crash recovery."""
        try:
            active_file = self._storage_path / "active.json"
            with self._lock:
                data = [r.to_dict() for r in self._active.values()]
            with open(active_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning("prediction.persist_active_failed", error=str(e))

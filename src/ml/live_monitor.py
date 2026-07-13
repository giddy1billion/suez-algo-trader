"""
Live Model Monitoring & Automated Rollback.

Tracks prediction distributions, realized performance, and drift statistics
in production. Defines configurable rollback criteria and automatically
reverts to the previous approved model when those criteria are met.

Emits auditable events and alerts for all promotions, warnings, and rollbacks.

Rollback Criteria (configurable):
1. Sustained negative live Sharpe over N consecutive evaluation periods
2. Statistically significant accuracy degradation (binomial test)
3. Significant prediction distribution drift (PSI)

Integrates with:
- AutoRollbackManager for change-point detection
- ModelHealthMonitor for composite scoring
- ModelRegistry for version management
"""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Enums & Configuration
# ──────────────────────────────────────────────────────────────────────────


class MonitorEventType(str, Enum):
    """Types of auditable monitoring events."""
    MODEL_PROMOTED = "model_promoted"
    PREDICTION_RECORDED = "prediction_recorded"
    PERFORMANCE_WARNING = "performance_warning"
    DRIFT_WARNING = "drift_warning"
    ACCURACY_DEGRADATION = "accuracy_degradation"
    SHARPE_DEGRADATION = "sharpe_degradation"
    ROLLBACK_TRIGGERED = "rollback_triggered"
    ROLLBACK_COMPLETED = "rollback_completed"
    CIRCUIT_BREAKER_TRIPPED = "circuit_breaker_tripped"
    EVALUATION_COMPLETED = "evaluation_completed"


class RollbackReason(str, Enum):
    """Reason for an automated rollback."""
    SUSTAINED_NEGATIVE_SHARPE = "sustained_negative_sharpe"
    ACCURACY_DEGRADATION = "accuracy_degradation"
    PREDICTION_DRIFT = "prediction_drift"
    MANUAL = "manual"


@dataclass
class LiveMonitorConfig:
    """Configuration for live model monitoring and rollback."""

    # Sharpe-based rollback
    sharpe_window: int = 50
    sharpe_threshold: float = -0.5
    sharpe_sustained_periods: int = 3

    # Accuracy-based rollback
    accuracy_window: int = 100
    accuracy_degradation_pvalue: float = 0.05
    accuracy_baseline: float = 0.52

    # Drift-based rollback
    drift_psi_threshold: float = 0.25

    # General
    min_observations: int = 30
    evaluation_interval_seconds: float = 300.0
    max_rollbacks_per_day: int = 3

    # Prediction distribution bins (for PSI calculation)
    n_distribution_bins: int = 3  # matches 3-class output (down/flat/up)


@dataclass
class AuditEvent:
    """Immutable auditable event for all monitoring actions."""
    timestamp: float
    event_type: MonitorEventType
    model_version: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, critical

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "model_version": self.model_version,
            "details": self.details,
            "severity": self.severity,
        }


@dataclass
class PredictionRecord:
    """Single prediction with outcome tracking."""
    timestamp: float
    model_version: str
    predicted_class: int  # 0=down, 1=flat, 2=up
    confidence: float
    realized_return: Optional[float] = None
    is_correct: Optional[bool] = None


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance metrics."""
    timestamp: float
    model_version: str
    live_sharpe: float
    rolling_accuracy: float
    n_predictions: int
    n_resolved: int
    prediction_distribution: List[float]  # class probabilities
    psi_score: float


@dataclass
class RollbackRecord:
    """Complete record of a rollback action."""
    timestamp: float
    from_version: str
    to_version: str
    reason: RollbackReason
    metrics_at_rollback: Dict[str, Any]
    evaluation_window_size: int


# ──────────────────────────────────────────────────────────────────────────
# Live Monitor
# ──────────────────────────────────────────────────────────────────────────


class LiveModelMonitor:
    """
    Production model monitoring with automated rollback.

    Continuously tracks prediction distributions, realized performance,
    and drift statistics. Triggers automated rollback when configurable
    criteria are met, ensuring at-most-once rollback semantics.

    Usage:
        monitor = LiveModelMonitor(config=LiveMonitorConfig())
        monitor.activate_model("v15.0", previous_version="v14.3")

        # Record predictions as they happen
        monitor.record_prediction("v15.0", predicted_class=2, confidence=0.72)

        # Record realized outcomes
        monitor.record_outcome("v15.0", realized_return=0.003, was_correct=True)

        # Periodic evaluation (called by scheduler or manually)
        result = monitor.evaluate()
    """

    def __init__(
        self,
        config: Optional[LiveMonitorConfig] = None,
        on_rollback: Optional[Callable[[RollbackRecord], None]] = None,
        on_alert: Optional[Callable[[AuditEvent], None]] = None,
    ):
        self.config = config or LiveMonitorConfig()
        self._on_rollback = on_rollback
        self._on_alert = on_alert

        # Active model state
        self._active_model: Optional[str] = None
        self._previous_model: Optional[str] = None
        self._model_activated_at: float = 0.0

        # Prediction tracking
        self._predictions: deque = deque(maxlen=5000)
        self._returns: deque = deque(maxlen=5000)
        self._correct: deque = deque(maxlen=5000)

        # Baseline distribution (set from initial predictions)
        self._baseline_distribution: Optional[np.ndarray] = None
        self._baseline_accuracy: Optional[float] = None

        # Sharpe degradation tracking
        self._sharpe_below_threshold_count: int = 0
        self._last_sharpe_values: deque = deque(maxlen=20)

        # Rollback state
        self._rollback_occurred: bool = False
        self._rollback_record: Optional[RollbackRecord] = None
        self._rollbacks_today: int = 0
        self._day_start: float = 0.0

        # Audit log
        self._audit_log: List[AuditEvent] = []

        # Thread safety
        self._lock = threading.Lock()

        # Last evaluation time
        self._last_evaluation_time: float = 0.0

    # ──────────────────────────────────────────────────────────────────────
    # Public API — Model Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def activate_model(
        self,
        model_version: str,
        previous_version: Optional[str] = None,
        baseline_distribution: Optional[List[float]] = None,
        baseline_accuracy: Optional[float] = None,
    ):
        """
        Activate a model for live monitoring.

        Args:
            model_version: Version string of the newly promoted model.
            previous_version: Version to rollback to if needed.
            baseline_distribution: Expected prediction class distribution.
            baseline_accuracy: Expected baseline accuracy from validation.
        """
        with self._lock:
            self._active_model = model_version
            self._previous_model = previous_version
            self._model_activated_at = time.time()

            # Reset monitoring state
            self._predictions.clear()
            self._returns.clear()
            self._correct.clear()
            self._sharpe_below_threshold_count = 0
            self._last_sharpe_values.clear()
            self._rollback_occurred = False
            self._rollback_record = None

            if baseline_distribution is not None:
                self._baseline_distribution = np.array(baseline_distribution)
            else:
                self._baseline_distribution = None

            self._baseline_accuracy = baseline_accuracy or self.config.accuracy_baseline

            self._emit_event(AuditEvent(
                timestamp=time.time(),
                event_type=MonitorEventType.MODEL_PROMOTED,
                model_version=model_version,
                details={
                    "previous_version": previous_version,
                    "baseline_accuracy": self._baseline_accuracy,
                    "baseline_distribution": baseline_distribution,
                },
                severity="info",
            ))

            logger.info(
                "live_monitor.model_activated",
                model_version=model_version,
                previous_version=previous_version,
            )

    def record_prediction(
        self,
        model_version: str,
        predicted_class: int,
        confidence: float = 0.0,
    ):
        """Record a new prediction from the active model."""
        with self._lock:
            if model_version != self._active_model:
                return

            record = PredictionRecord(
                timestamp=time.time(),
                model_version=model_version,
                predicted_class=predicted_class,
                confidence=confidence,
            )
            self._predictions.append(record)

    def record_outcome(
        self,
        model_version: str,
        realized_return: float,
        was_correct: bool,
    ):
        """
        Record a realized trade outcome.

        Args:
            model_version: Model that generated the prediction.
            realized_return: Actual P&L of the trade.
            was_correct: Whether the directional prediction was correct.
        """
        with self._lock:
            if model_version != self._active_model:
                return

            self._returns.append(realized_return)
            self._correct.append(1 if was_correct else 0)

    # ──────────────────────────────────────────────────────────────────────
    # Public API — Evaluation & Rollback
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(self) -> Optional[PerformanceSnapshot]:
        """
        Evaluate the active model and trigger rollback if criteria are met.

        Returns:
            PerformanceSnapshot if evaluation was performed, None otherwise.
        """
        with self._lock:
            if self._active_model is None:
                return None

            if self._rollback_occurred:
                return None  # Already rolled back — no further action

            n_returns = len(self._returns)
            n_predictions = len(self._predictions)

            if n_returns < self.config.min_observations:
                return None  # Not enough data

            # Calculate metrics
            live_sharpe = self._compute_live_sharpe()
            rolling_accuracy = self._compute_rolling_accuracy()
            pred_distribution = self._compute_prediction_distribution()
            psi_score = self._compute_psi(pred_distribution)

            snapshot = PerformanceSnapshot(
                timestamp=time.time(),
                model_version=self._active_model,
                live_sharpe=live_sharpe,
                rolling_accuracy=rolling_accuracy,
                n_predictions=n_predictions,
                n_resolved=n_returns,
                prediction_distribution=pred_distribution.tolist(),
                psi_score=psi_score,
            )

            # Check rollback criteria
            rollback_reason = self._check_rollback_criteria(
                live_sharpe, rolling_accuracy, psi_score
            )

            self._emit_event(AuditEvent(
                timestamp=time.time(),
                event_type=MonitorEventType.EVALUATION_COMPLETED,
                model_version=self._active_model,
                details={
                    "live_sharpe": live_sharpe,
                    "rolling_accuracy": rolling_accuracy,
                    "psi_score": psi_score,
                    "n_resolved": n_returns,
                    "rollback_triggered": rollback_reason is not None,
                },
                severity="info",
            ))

            if rollback_reason is not None:
                self._execute_rollback(rollback_reason, snapshot)

            self._last_evaluation_time = time.time()
            return snapshot

    def force_rollback(self, reason: str = "manual") -> Optional[RollbackRecord]:
        """Manually trigger a rollback. Returns the rollback record."""
        with self._lock:
            if self._active_model is None or self._previous_model is None:
                return None
            if self._rollback_occurred:
                return None

            snapshot = PerformanceSnapshot(
                timestamp=time.time(),
                model_version=self._active_model,
                live_sharpe=self._compute_live_sharpe() if len(self._returns) > 1 else 0.0,
                rolling_accuracy=self._compute_rolling_accuracy() if self._correct else 0.0,
                n_predictions=len(self._predictions),
                n_resolved=len(self._returns),
                prediction_distribution=[],
                psi_score=0.0,
            )
            self._execute_rollback(RollbackReason.MANUAL, snapshot)
            return self._rollback_record

    @property
    def active_model(self) -> Optional[str]:
        """Currently active model version."""
        return self._active_model

    @property
    def has_rolled_back(self) -> bool:
        """Whether a rollback has occurred for the current model."""
        return self._rollback_occurred

    @property
    def rollback_record(self) -> Optional[RollbackRecord]:
        """Record of the rollback that occurred, if any."""
        return self._rollback_record

    @property
    def audit_log(self) -> List[AuditEvent]:
        """Full audit log of all events."""
        return list(self._audit_log)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance metrics summary."""
        with self._lock:
            if not self._returns:
                return {"status": "insufficient_data", "n_observations": 0}

            return {
                "model_version": self._active_model,
                "n_predictions": len(self._predictions),
                "n_resolved": len(self._returns),
                "live_sharpe": self._compute_live_sharpe(),
                "rolling_accuracy": self._compute_rolling_accuracy(),
                "sharpe_below_threshold_count": self._sharpe_below_threshold_count,
                "rollback_occurred": self._rollback_occurred,
            }

    # ──────────────────────────────────────────────────────────────────────
    # Internal — Metrics Computation
    # ──────────────────────────────────────────────────────────────────────

    def _compute_live_sharpe(self) -> float:
        """Compute rolling Sharpe ratio from realized returns."""
        window = self.config.sharpe_window
        returns = list(self._returns)[-window:]
        if len(returns) < 2:
            return 0.0
        arr = np.array(returns)
        mean_r = np.mean(arr)
        std_r = np.std(arr, ddof=1)
        if std_r < 1e-10:
            return 0.0 if mean_r <= 0 else 10.0
        # Annualize assuming ~252 trading days, but keep as raw ratio
        return float(mean_r / std_r) * np.sqrt(min(252, len(returns)))

    def _compute_rolling_accuracy(self) -> float:
        """Compute rolling accuracy from recent outcomes."""
        window = self.config.accuracy_window
        correct = list(self._correct)[-window:]
        if not correct:
            return 0.0
        return float(np.mean(correct))

    def _compute_prediction_distribution(self) -> np.ndarray:
        """Compute current prediction class distribution."""
        n_bins = self.config.n_distribution_bins
        recent = list(self._predictions)[-self.config.accuracy_window:]
        if not recent:
            return np.ones(n_bins) / n_bins

        classes = [p.predicted_class for p in recent]
        counts = np.zeros(n_bins)
        for c in classes:
            if 0 <= c < n_bins:
                counts[c] += 1

        total = counts.sum()
        if total == 0:
            return np.ones(n_bins) / n_bins
        return counts / total

    def _compute_psi(self, current_distribution: np.ndarray) -> float:
        """
        Compute Population Stability Index between baseline and current distribution.

        PSI = Σ (P_i - Q_i) * ln(P_i / Q_i)
        """
        if self._baseline_distribution is None:
            return 0.0

        # Ensure same length
        baseline = self._baseline_distribution
        if len(baseline) != len(current_distribution):
            return 0.0

        # Add small epsilon to avoid log(0)
        eps = 1e-6
        p = np.clip(current_distribution, eps, 1.0)
        q = np.clip(baseline, eps, 1.0)

        psi = float(np.sum((p - q) * np.log(p / q)))
        return max(0.0, psi)

    # ──────────────────────────────────────────────────────────────────────
    # Internal — Rollback Criteria
    # ──────────────────────────────────────────────────────────────────────

    def _check_rollback_criteria(
        self,
        live_sharpe: float,
        rolling_accuracy: float,
        psi_score: float,
    ) -> Optional[RollbackReason]:
        """
        Check all rollback criteria. Returns reason if rollback should occur.

        Criteria are checked in order of severity:
        1. Sustained negative Sharpe
        2. Statistically significant accuracy degradation
        3. Prediction distribution drift
        """
        if self._previous_model is None:
            return None  # No model to rollback to

        # Check daily rollback limit (circuit breaker)
        if not self._check_daily_limit():
            return None

        # Criterion 1: Sustained negative Sharpe
        reason = self._check_sharpe_criterion(live_sharpe)
        if reason:
            return reason

        # Criterion 2: Accuracy degradation
        reason = self._check_accuracy_criterion(rolling_accuracy)
        if reason:
            return reason

        # Criterion 3: Prediction drift
        reason = self._check_drift_criterion(psi_score)
        if reason:
            return reason

        return None

    def _check_sharpe_criterion(self, live_sharpe: float) -> Optional[RollbackReason]:
        """Check if Sharpe has been sustained below threshold."""
        self._last_sharpe_values.append(live_sharpe)

        if live_sharpe < self.config.sharpe_threshold:
            self._sharpe_below_threshold_count += 1
        else:
            self._sharpe_below_threshold_count = 0

        if self._sharpe_below_threshold_count >= self.config.sharpe_sustained_periods:
            self._emit_event(AuditEvent(
                timestamp=time.time(),
                event_type=MonitorEventType.SHARPE_DEGRADATION,
                model_version=self._active_model,
                details={
                    "live_sharpe": live_sharpe,
                    "threshold": self.config.sharpe_threshold,
                    "sustained_periods": self._sharpe_below_threshold_count,
                },
                severity="critical",
            ))
            return RollbackReason.SUSTAINED_NEGATIVE_SHARPE

        # Emit warning if approaching
        if self._sharpe_below_threshold_count >= 1:
            self._emit_event(AuditEvent(
                timestamp=time.time(),
                event_type=MonitorEventType.PERFORMANCE_WARNING,
                model_version=self._active_model,
                details={
                    "live_sharpe": live_sharpe,
                    "threshold": self.config.sharpe_threshold,
                    "consecutive_below": self._sharpe_below_threshold_count,
                    "periods_until_rollback": (
                        self.config.sharpe_sustained_periods
                        - self._sharpe_below_threshold_count
                    ),
                },
                severity="warning",
            ))

        return None

    def _check_accuracy_criterion(self, rolling_accuracy: float) -> Optional[RollbackReason]:
        """Check for statistically significant accuracy degradation."""
        n = len(self._correct)
        if n < self.config.min_observations:
            return None

        window = list(self._correct)[-self.config.accuracy_window:]
        n_correct = sum(window)
        n_total = len(window)

        # One-sided binomial test: is accuracy significantly below baseline?
        result = scipy_stats.binomtest(
            n_correct, n_total, self._baseline_accuracy, alternative="less"
        )

        if result.pvalue < self.config.accuracy_degradation_pvalue:
            self._emit_event(AuditEvent(
                timestamp=time.time(),
                event_type=MonitorEventType.ACCURACY_DEGRADATION,
                model_version=self._active_model,
                details={
                    "rolling_accuracy": rolling_accuracy,
                    "baseline_accuracy": self._baseline_accuracy,
                    "p_value": result.pvalue,
                    "threshold_pvalue": self.config.accuracy_degradation_pvalue,
                    "n_total": n_total,
                    "n_correct": n_correct,
                },
                severity="critical",
            ))
            return RollbackReason.ACCURACY_DEGRADATION

        return None

    def _check_drift_criterion(self, psi_score: float) -> Optional[RollbackReason]:
        """Check for significant prediction distribution drift."""
        if psi_score > self.config.drift_psi_threshold:
            self._emit_event(AuditEvent(
                timestamp=time.time(),
                event_type=MonitorEventType.DRIFT_WARNING,
                model_version=self._active_model,
                details={
                    "psi_score": psi_score,
                    "threshold": self.config.drift_psi_threshold,
                },
                severity="critical",
            ))
            return RollbackReason.PREDICTION_DRIFT

        return None

    def _check_daily_limit(self) -> bool:
        """Check if daily rollback limit has been reached."""
        now = time.time()
        day_seconds = 86400.0

        # Reset counter at start of new day
        if now - self._day_start > day_seconds:
            self._rollbacks_today = 0
            self._day_start = now

        if self._rollbacks_today >= self.config.max_rollbacks_per_day:
            self._emit_event(AuditEvent(
                timestamp=now,
                event_type=MonitorEventType.CIRCUIT_BREAKER_TRIPPED,
                model_version=self._active_model,
                details={
                    "rollbacks_today": self._rollbacks_today,
                    "max_allowed": self.config.max_rollbacks_per_day,
                },
                severity="critical",
            ))
            return False

        return True

    # ──────────────────────────────────────────────────────────────────────
    # Internal — Rollback Execution
    # ──────────────────────────────────────────────────────────────────────

    def _execute_rollback(
        self,
        reason: RollbackReason,
        snapshot: PerformanceSnapshot,
    ):
        """
        Execute the rollback: swap active model to previous version.

        Guarantees at-most-once rollback semantics per model activation.
        """
        if self._rollback_occurred:
            return  # Already rolled back — idempotent guard

        from_version = self._active_model
        to_version = self._previous_model

        # Create rollback record
        record = RollbackRecord(
            timestamp=time.time(),
            from_version=from_version,
            to_version=to_version,
            reason=reason,
            metrics_at_rollback={
                "live_sharpe": snapshot.live_sharpe,
                "rolling_accuracy": snapshot.rolling_accuracy,
                "psi_score": snapshot.psi_score,
                "n_predictions": snapshot.n_predictions,
                "n_resolved": snapshot.n_resolved,
            },
            evaluation_window_size=len(self._returns),
        )

        # Mark rollback as occurred BEFORE any callbacks (at-most-once)
        self._rollback_occurred = True
        self._rollback_record = record
        self._rollbacks_today += 1

        # Swap active model
        self._active_model = to_version

        # Emit rollback triggered event
        self._emit_event(AuditEvent(
            timestamp=time.time(),
            event_type=MonitorEventType.ROLLBACK_TRIGGERED,
            model_version=from_version,
            details={
                "reason": reason.value,
                "rolled_back_to": to_version,
                "metrics": record.metrics_at_rollback,
            },
            severity="critical",
        ))

        # Emit rollback completed event
        self._emit_event(AuditEvent(
            timestamp=time.time(),
            event_type=MonitorEventType.ROLLBACK_COMPLETED,
            model_version=to_version,
            details={
                "rolled_back_from": from_version,
                "reason": reason.value,
            },
            severity="info",
        ))

        logger.warning(
            "live_monitor.rollback_executed",
            from_version=from_version,
            to_version=to_version,
            reason=reason.value,
            live_sharpe=snapshot.live_sharpe,
            rolling_accuracy=snapshot.rolling_accuracy,
        )

        # Invoke callback
        if self._on_rollback:
            self._on_rollback(record)

    # ──────────────────────────────────────────────────────────────────────
    # Internal — Event Emission
    # ──────────────────────────────────────────────────────────────────────

    def _emit_event(self, event: AuditEvent):
        """Emit an auditable event to the log and optional alert callback."""
        self._audit_log.append(event)

        if self._on_alert:
            self._on_alert(event)

        logger.info(
            "live_monitor.event",
            event_type=event.event_type.value,
            model_version=event.model_version,
            severity=event.severity,
        )

"""
Closed-Loop Learning Pipeline — Autonomous retraining orchestration.

Connects:
    Trade Outcomes → Experience DB → Retrain Trigger → Training Pipeline
    → Validation → Promotion → Deployment → Monitoring → (repeat)

The loop never stops. Models continuously earn the right to remain active.

Trigger Policies:
1. Performance Decay — Health score drops below threshold
2. Sample Accumulation — Enough new trades to justify retraining
3. Drift Detection — Feature or prediction drift detected
4. Scheduled — Regular interval-based retraining
5. Manual — Operator-initiated

Integrates with:
- ExperienceDatabase (feedback_loop.py) — trade outcomes
- ModelHealthMonitor (model_health.py) — health scoring
- AutoRollbackManager (auto_rollback.py) — degradation detection
- TrainingPipeline (training_pipeline.py) — model training
- PromotionEngine (promotion_engine.py) — governance
- StatisticalValidation (statistical_validation.py) — overfitting guards
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Enums & Configuration
# ──────────────────────────────────────────────────────────────────────────


class RetriggerReason(Enum):
    PERFORMANCE_DECAY = "performance_decay"
    SAMPLE_ACCUMULATION = "sample_accumulation"
    DRIFT_DETECTED = "drift_detected"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    HEALTH_WARNING = "health_warning"
    ROLLBACK_RECOVERY = "rollback_recovery"


class PipelineStage(Enum):
    IDLE = "idle"
    COLLECTING = "collecting"
    TRAINING = "training"
    VALIDATING = "validating"
    SHADOW_TESTING = "shadow_testing"
    PROMOTING = "promoting"
    MONITORING = "monitoring"


@dataclass
class RetriggerPolicy:
    """Policy controlling when retraining is triggered."""

    # Performance decay trigger
    health_score_threshold: float = 71.0  # trigger when health drops below this
    accuracy_drop_threshold: float = 0.10  # trigger on 10% accuracy drop

    # Sample accumulation trigger
    min_new_trades_for_retrain: int = 100  # minimum new trades before retraining
    max_staleness_hours: float = 168.0  # force retrain after 7 days without one

    # Drift trigger
    psi_threshold: float = 0.15  # feature drift threshold
    kl_threshold: float = 0.10  # prediction drift threshold

    # Scheduled trigger
    scheduled_interval_hours: float = 24.0  # retrain every N hours regardless

    # Guardrails
    min_retrain_interval_hours: float = 4.0  # never retrain more often than this
    max_concurrent_pipelines: int = 1  # only 1 training at a time
    cooldown_after_failure_hours: float = 2.0  # wait before retry after failure

    # Validation requirements before promotion
    require_walk_forward: bool = True
    require_monte_carlo: bool = True
    require_statistical_validation: bool = True
    min_shadow_trades: int = 20  # minimum paper trades before promotion


@dataclass
class LoopState:
    """Current state of the closed-loop pipeline."""

    stage: PipelineStage = PipelineStage.IDLE
    current_model_version: str = ""
    challenger_version: Optional[str] = None
    last_retrain_time: float = 0.0
    last_retrain_reason: Optional[RetriggerReason] = None
    new_trades_since_retrain: int = 0
    total_retrains: int = 0
    consecutive_failures: int = 0
    shadow_trades_count: int = 0
    is_training: bool = False


@dataclass
class RetriggerEvent:
    """Record of a retrain trigger."""

    timestamp: float
    reason: RetriggerReason
    model_version: str
    health_score: Optional[float] = None
    accuracy: Optional[float] = None
    new_trades: int = 0
    drift_psi: Optional[float] = None
    approved: bool = True
    rejection_reason: str = ""


# ──────────────────────────────────────────────────────────────────────────
# Closed-Loop Pipeline Manager
# ──────────────────────────────────────────────────────────────────────────


class ClosedLoopPipeline:
    """
    Autonomous model lifecycle manager.

    Monitors model health, detects degradation, triggers retraining,
    validates new models, and promotes/demotes based on evidence.

    The loop:
        Monitor → Detect → Trigger → Train → Validate → Shadow → Promote → Monitor

    Usage:
        pipeline = ClosedLoopPipeline(policy=RetriggerPolicy())

        # On each trade outcome:
        pipeline.on_trade_resolved(trade_outcome)

        # Periodic check (e.g., every 5 minutes):
        pipeline.check_triggers()

        # Integrate with training:
        pipeline.set_training_callback(training_fn)
    """

    def __init__(
        self,
        policy: Optional[RetriggerPolicy] = None,
        training_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        validation_callback: Optional[Callable[[str], Dict[str, Any]]] = None,
        promotion_callback: Optional[Callable[[str, str], bool]] = None,
    ):
        self.policy = policy or RetriggerPolicy()
        self.training_callback = training_callback
        self.validation_callback = validation_callback
        self.promotion_callback = promotion_callback

        self.state = LoopState()
        self._events: List[RetriggerEvent] = []
        self._trade_outcomes: List[Dict[str, Any]] = []
        self._health_history: List[float] = []

    def on_trade_resolved(self, outcome: Dict[str, Any]):
        """
        Record a resolved trade outcome for learning.

        This is the primary input to the closed loop. Every trade that
        closes feeds back into the system.

        Args:
            outcome: Dict with keys like 'return_pct', 'direction_correct',
                     'model_version', 'confidence', 'symbol', etc.
        """
        self._trade_outcomes.append(outcome)
        self.state.new_trades_since_retrain += 1

        # If in shadow testing mode, count shadow trades
        if self.state.stage == PipelineStage.SHADOW_TESTING:
            if outcome.get("model_version") == self.state.challenger_version:
                self.state.shadow_trades_count += 1

        logger.debug(
            "closed_loop.trade_recorded",
            new_trades=self.state.new_trades_since_retrain,
            model=outcome.get("model_version", "unknown"),
        )

    def on_health_update(self, model_version: str, health_score: float):
        """
        Record model health score update.

        Called by ModelHealthMonitor after each evaluation.
        """
        self._health_history.append(health_score)

        if health_score < self.policy.health_score_threshold:
            self._try_trigger(
                RetriggerReason.HEALTH_WARNING,
                model_version=model_version,
                health_score=health_score,
            )

    def on_drift_detected(self, model_version: str, psi: float, kl: float):
        """
        Handle drift detection event.

        Called by DriftMonitor when feature or prediction drift exceeds threshold.
        """
        if psi > self.policy.psi_threshold or kl > self.policy.kl_threshold:
            self._try_trigger(
                RetriggerReason.DRIFT_DETECTED,
                model_version=model_version,
                drift_psi=psi,
            )

    def on_rollback(self, model_version: str, rolled_back_to: str):
        """
        Handle rollback event — trigger recovery retraining.

        When a model is rolled back, we immediately start training a
        replacement using the latest data.
        """
        self.state.current_model_version = rolled_back_to
        self._try_trigger(
            RetriggerReason.ROLLBACK_RECOVERY,
            model_version=model_version,
        )

    def check_triggers(self) -> Optional[RetriggerReason]:
        """
        Check all trigger conditions and start retraining if needed.

        Call this periodically (e.g., every 5 minutes via scheduler).

        Returns:
            RetriggerReason if retraining was triggered, None otherwise.
        """
        now = time.time()

        # Already training?
        if self.state.is_training:
            return None

        # Cooldown check
        hours_since_retrain = (now - self.state.last_retrain_time) / 3600
        if hours_since_retrain < self.policy.min_retrain_interval_hours:
            return None

        # Failure cooldown
        if self.state.consecutive_failures > 0:
            cooldown = self.policy.cooldown_after_failure_hours * self.state.consecutive_failures
            if hours_since_retrain < cooldown:
                return None

        # 1. Sample accumulation trigger
        if self.state.new_trades_since_retrain >= self.policy.min_new_trades_for_retrain:
            return self._try_trigger(
                RetriggerReason.SAMPLE_ACCUMULATION,
                model_version=self.state.current_model_version,
                new_trades=self.state.new_trades_since_retrain,
            )

        # 2. Staleness trigger
        if hours_since_retrain >= self.policy.max_staleness_hours:
            return self._try_trigger(
                RetriggerReason.SCHEDULED,
                model_version=self.state.current_model_version,
            )

        # 3. Scheduled trigger
        if hours_since_retrain >= self.policy.scheduled_interval_hours:
            return self._try_trigger(
                RetriggerReason.SCHEDULED,
                model_version=self.state.current_model_version,
            )

        # 4. Performance decay (accuracy drop)
        if self._detect_accuracy_drop():
            return self._try_trigger(
                RetriggerReason.PERFORMANCE_DECAY,
                model_version=self.state.current_model_version,
            )

        return None

    def check_shadow_promotion(self) -> bool:
        """
        Check if shadow-testing challenger is ready for promotion.

        Returns:
            True if challenger was promoted.
        """
        if self.state.stage != PipelineStage.SHADOW_TESTING:
            return False

        if self.state.shadow_trades_count < self.policy.min_shadow_trades:
            return False

        # Validate challenger performance
        if self.validation_callback and self.state.challenger_version:
            validation = self.validation_callback(self.state.challenger_version)
            if not validation.get("passed", False):
                logger.warning(
                    "closed_loop.shadow_validation_failed",
                    challenger=self.state.challenger_version,
                    reasons=validation.get("reasons", []),
                )
                self.state.stage = PipelineStage.MONITORING
                self.state.challenger_version = None
                return False

        # Promote
        if self.promotion_callback and self.state.challenger_version:
            promoted = self.promotion_callback(
                self.state.current_model_version,
                self.state.challenger_version,
            )
            if promoted:
                logger.info(
                    "closed_loop.model_promoted",
                    old_champion=self.state.current_model_version,
                    new_champion=self.state.challenger_version,
                )
                self.state.current_model_version = self.state.challenger_version
                self.state.challenger_version = None
                self.state.stage = PipelineStage.MONITORING
                self.state.shadow_trades_count = 0
                return True

        return False

    def trigger_manual_retrain(self, reason: str = "operator_initiated"):
        """Manually trigger retraining (operator action)."""
        self._try_trigger(
            RetriggerReason.MANUAL,
            model_version=self.state.current_model_version,
        )

    def get_state(self) -> Dict[str, Any]:
        """Get current pipeline state for monitoring/dashboard."""
        return {
            "stage": self.state.stage.value,
            "current_model": self.state.current_model_version,
            "challenger_model": self.state.challenger_version,
            "new_trades_since_retrain": self.state.new_trades_since_retrain,
            "total_retrains": self.state.total_retrains,
            "consecutive_failures": self.state.consecutive_failures,
            "shadow_trades": self.state.shadow_trades_count,
            "is_training": self.state.is_training,
            "last_retrain_reason": self.state.last_retrain_reason.value if self.state.last_retrain_reason else None,
            "hours_since_retrain": (time.time() - self.state.last_retrain_time) / 3600 if self.state.last_retrain_time > 0 else None,
            "total_events": len(self._events),
        }

    def get_events(self, limit: int = 20) -> List[RetriggerEvent]:
        """Get recent trigger events."""
        return self._events[-limit:]

    # ──────────────────────────────────────────────────────────────────────
    # Private Methods
    # ──────────────────────────────────────────────────────────────────────

    def _try_trigger(
        self,
        reason: RetriggerReason,
        model_version: str = "",
        health_score: Optional[float] = None,
        new_trades: int = 0,
        drift_psi: Optional[float] = None,
    ) -> Optional[RetriggerReason]:
        """Attempt to trigger retraining with guardrails."""
        now = time.time()

        # Guardrail: minimum interval
        hours_since = (now - self.state.last_retrain_time) / 3600
        if self.state.last_retrain_time > 0 and hours_since < self.policy.min_retrain_interval_hours:
            return None

        # Guardrail: concurrent limit
        if self.state.is_training:
            return None

        # Record event
        event = RetriggerEvent(
            timestamp=now,
            reason=reason,
            model_version=model_version,
            health_score=health_score,
            new_trades=new_trades,
            drift_psi=drift_psi,
            approved=True,
        )
        self._events.append(event)

        logger.info(
            "closed_loop.retrain_triggered",
            reason=reason.value,
            model_version=model_version,
            new_trades=new_trades,
        )

        # Execute training
        self._start_training(reason)
        return reason

    def _start_training(self, reason: RetriggerReason):
        """Initiate the training pipeline."""
        self.state.is_training = True
        self.state.stage = PipelineStage.TRAINING
        self.state.last_retrain_reason = reason
        self.state.last_retrain_time = time.time()
        self.state.total_retrains += 1

        if self.training_callback:
            try:
                result = self.training_callback({
                    "reason": reason.value,
                    "n_new_trades": self.state.new_trades_since_retrain,
                    "current_model": self.state.current_model_version,
                })

                # Training succeeded
                if result and result.get("success"):
                    new_version = result.get("model_version", f"v{self.state.total_retrains:03d}")
                    self.state.challenger_version = new_version
                    self.state.stage = PipelineStage.SHADOW_TESTING
                    self.state.shadow_trades_count = 0
                    self.state.consecutive_failures = 0
                    self.state.new_trades_since_retrain = 0

                    logger.info(
                        "closed_loop.training_succeeded",
                        new_version=new_version,
                        stage="shadow_testing",
                    )
                else:
                    self.state.consecutive_failures += 1
                    self.state.stage = PipelineStage.MONITORING
                    logger.warning(
                        "closed_loop.training_failed",
                        failures=self.state.consecutive_failures,
                    )

            except Exception as e:
                self.state.consecutive_failures += 1
                self.state.stage = PipelineStage.MONITORING
                logger.error(f"closed_loop.training_exception: {e}")
        else:
            # No callback — just record intent
            self.state.stage = PipelineStage.MONITORING
            logger.warning("closed_loop.no_training_callback_configured")

        self.state.is_training = False

    def _detect_accuracy_drop(self) -> bool:
        """Detect significant accuracy drop from recent trade outcomes."""
        if len(self._trade_outcomes) < 30:
            return False

        recent = self._trade_outcomes[-30:]
        correct = sum(1 for t in recent if t.get("direction_correct", False))
        recent_accuracy = correct / len(recent)

        # Compare to older baseline
        if len(self._trade_outcomes) >= 100:
            older = self._trade_outcomes[-100:-30]
            older_correct = sum(1 for t in older if t.get("direction_correct", False))
            older_accuracy = older_correct / len(older) if older else 0.5
            drop = older_accuracy - recent_accuracy
            return drop >= self.policy.accuracy_drop_threshold

        return False

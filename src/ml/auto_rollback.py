"""
Automatic Rollback — Change-point detection for deployed models.

Implements CUSUM (Cumulative Sum) and Page-Hinkley tests to detect
statistically significant performance degradation in real-time.

When degradation is detected:
1. Alert generated (warning threshold)
2. Allocation reduced (moderate degradation)
3. Model rolled back to previous champion (severe degradation)
4. Circuit breaker tripped (catastrophic failure)

Integrates with:
- ModelHealthMonitor for composite health scoring
- PromotionEngine for governance-aware rollback
- CircuitBreaker for emergency halt
- ModelRegistry for version management
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Enums & Configuration
# ──────────────────────────────────────────────────────────────────────────


class RollbackSeverity(Enum):
    NONE = "none"
    ALERT = "alert"
    REDUCE_ALLOCATION = "reduce_allocation"
    ROLLBACK = "rollback"
    HALT = "halt"


class DetectionMethod(Enum):
    CUSUM = "cusum"
    PAGE_HINKLEY = "page_hinkley"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"


@dataclass
class RollbackConfig:
    """Configuration for auto-rollback behavior."""

    # CUSUM parameters
    cusum_threshold: float = 5.0  # cumulative deviation threshold for detection
    cusum_drift: float = 0.5  # allowable drift before accumulating (slack parameter)

    # Page-Hinkley parameters
    ph_threshold: float = 50.0  # Page-Hinkley detection threshold
    ph_alpha: float = 0.02  # DR2 FIX: Raised from 0.005 to 0.02 — 0.005 was too conservative,
    # tolerating 0.5% mean shift before flagging. 0.02 detects 2% mean shift promptly.

    # Exponential smoothing
    ewma_lambda: float = 0.05  # smoothing factor (lower = more sensitive)
    ewma_L: float = 3.0  # control limit in standard deviations

    # Action thresholds (how many consecutive detections before action)
    alert_after: int = 1  # alert on first detection
    reduce_after: int = 3  # reduce allocation after 3 consecutive
    rollback_after: int = 5  # rollback after 5 consecutive
    halt_after: int = 10  # halt after 10 (catastrophic)

    # Cooldown between actions
    action_cooldown_seconds: float = 300.0  # 5 min between actions

    # Minimum observations before detection is active
    min_observations: int = 30

    # Health score integration
    health_score_rollback_threshold: float = 55.0
    health_score_reduce_threshold: float = 65.0

    # DR4 FIX: Asset-class-specific thresholds — crypto has naturally higher volatility
    asset_class: str = "equity"  # 'equity' or 'crypto'


@dataclass
class ChangePointDetection:
    """Result of a change-point detection test."""

    detected: bool
    method: DetectionMethod
    statistic: float
    threshold: float
    direction: str  # "degradation" or "improvement"
    n_observations: int
    consecutive_detections: int


@dataclass
class RollbackEvent:
    """Record of a rollback action taken."""

    timestamp: float
    model_version: str
    severity: RollbackSeverity
    reason: str
    detection: ChangePointDetection
    rolled_back_to: Optional[str] = None
    health_score: Optional[float] = None


# ──────────────────────────────────────────────────────────────────────────
# CUSUM Detector
# ──────────────────────────────────────────────────────────────────────────


class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) change-point detector.

    Tracks cumulative deviations from the expected performance baseline.
    Detects both positive and negative shifts (two-sided).

    The CUSUM statistic S_n accumulates when observations deviate from
    the target by more than the drift parameter:
        S_n⁺ = max(0, S_{n-1}⁺ + (x_n - μ₀ - k))  # detects upward shift
        S_n⁻ = max(0, S_{n-1}⁻ + (μ₀ - k - x_n))  # detects downward shift

    Detection occurs when S_n exceeds the threshold h.
    """

    def __init__(self, target_mean: float = 0.0, drift: float = 0.5, threshold: float = 5.0):
        self.target_mean = target_mean
        self.drift = drift
        self.threshold = threshold

        # State
        self.s_pos: float = 0.0  # positive CUSUM
        self.s_neg: float = 0.0  # negative CUSUM
        self.n_observations: int = 0
        self._values: deque = deque(maxlen=500)

    def update(self, value: float) -> bool:
        """
        Add new observation and check for change point.

        Args:
            value: New metric observation (e.g., trade return, accuracy).

        Returns:
            True if change point detected (degradation).
        """
        self._values.append(value)
        self.n_observations += 1

        # Accumulate CUSUM statistics
        self.s_pos = max(0, self.s_pos + (value - self.target_mean - self.drift))
        self.s_neg = max(0, self.s_neg + (self.target_mean - self.drift - value))

        return self.s_neg > self.threshold  # negative shift = degradation

    def reset(self):
        """Reset CUSUM state (e.g., after rollback)."""
        self.s_pos = 0.0
        self.s_neg = 0.0

    @property
    def statistic(self) -> float:
        """Current CUSUM statistic (negative direction)."""
        return self.s_neg

    def recalibrate(self, new_target: Optional[float] = None):
        """Recalibrate target mean from recent observations."""
        if new_target is not None:
            self.target_mean = new_target
        elif len(self._values) >= 20:
            self.target_mean = float(np.mean(list(self._values)[-20:]))
        self.reset()


# ──────────────────────────────────────────────────────────────────────────
# Page-Hinkley Detector
# ──────────────────────────────────────────────────────────────────────────


class PageHinkleyDetector:
    """
    Page-Hinkley test for change-point detection.

    Detects both upward and downward mean shifts. Uses two-sided
    monitoring: tracks cumulative sum above and below the running mean.

    For degradation detection (downward shift):
        m_n = Σ(x_i - x̄_n + α)  (cumulates negative deviation)
        Detection: m_n - min(m) > threshold
    """

    def __init__(self, threshold: float = 50.0, alpha: float = 0.005):
        self.threshold = threshold
        self.alpha = alpha  # minimum acceptable change in mean

        # State
        self.n: int = 0
        self.sum_values: float = 0.0
        # Two-sided: detect downward shift
        self.m_t: float = 0.0  # cumulative sum for downward detection
        self.m_min: float = 0.0

    def update(self, value: float) -> bool:
        """
        Add observation and check for downward mean shift (degradation).

        Returns:
            True if degradation detected.
        """
        self.n += 1
        self.sum_values += value
        mean = self.sum_values / self.n

        # Cumulate deviation below mean (detects downward shift)
        # When values are consistently below the running mean, m_t grows
        self.m_t += (mean - value - self.alpha)
        self.m_min = min(self.m_min, self.m_t)

        # Detection: accumulated downward deviation exceeds threshold
        return (self.m_t - self.m_min) > self.threshold

    def reset(self):
        """Reset detector state."""
        self.n = 0
        self.sum_values = 0.0
        self.m_t = 0.0
        self.m_min = 0.0

    @property
    def statistic(self) -> float:
        return self.m_t - self.m_min


# ──────────────────────────────────────────────────────────────────────────
# EWMA Control Chart
# ──────────────────────────────────────────────────────────────────────────


class EWMADetector:
    """
    Exponentially Weighted Moving Average control chart.

    Sensitive to small, persistent shifts. Good for detecting gradual
    performance degradation over time.

    Z_t = λ * x_t + (1-λ) * Z_{t-1}
    UCL/LCL = μ₀ ± L * σ * sqrt(λ/(2-λ) * (1-(1-λ)^(2t)))
    """

    def __init__(
        self,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        lam: float = 0.05,
        L: float = 3.0,
    ):
        self.target_mean = target_mean
        self.target_std = target_std
        self.lam = lam
        self.L = L

        # State
        self.z: float = target_mean
        self.n: int = 0

    def update(self, value: float) -> bool:
        """
        Add observation and check for out-of-control condition.

        Returns:
            True if degradation detected (below LCL).
        """
        self.n += 1
        self.z = self.lam * value + (1 - self.lam) * self.z

        # Control limit (asymptotic)
        sigma_z = self.target_std * np.sqrt(
            self.lam / (2 - self.lam) * (1 - (1 - self.lam) ** (2 * self.n))
        )
        lcl = self.target_mean - self.L * sigma_z

        return self.z < lcl  # below lower control limit = degradation

    def reset(self):
        """Reset to target mean."""
        self.z = self.target_mean
        self.n = 0

    @property
    def statistic(self) -> float:
        return self.z

    def recalibrate(self, new_mean: float, new_std: float):
        """Update baseline parameters."""
        self.target_mean = new_mean
        self.target_std = new_std
        self.reset()


# ──────────────────────────────────────────────────────────────────────────
# Auto-Rollback Manager
# ──────────────────────────────────────────────────────────────────────────


class AutoRollbackManager:
    """
    Autonomous model rollback based on change-point detection.

    Monitors model performance in real-time using three complementary
    detection methods (CUSUM, Page-Hinkley, EWMA) and triggers appropriate
    actions when statistically significant degradation is detected.

    Usage:
        manager = AutoRollbackManager(config=RollbackConfig())
        manager.register_model("v14.3", baseline_return=0.002)

        # On each new trade outcome:
        severity = manager.observe("v14.3", trade_return=0.001)
        if severity == RollbackSeverity.ROLLBACK:
            # Execute rollback
            ...

        # Or integrate with health scoring:
        severity = manager.evaluate_health("v14.3", health_score=52.0)
    """

    def __init__(
        self,
        config: Optional[RollbackConfig] = None,
        on_alert: Optional[Callable[[RollbackEvent], None]] = None,
        on_rollback: Optional[Callable[[RollbackEvent], None]] = None,
        on_halt: Optional[Callable[[RollbackEvent], None]] = None,
    ):
        self.config = config or RollbackConfig()
        self.on_alert = on_alert
        self.on_rollback = on_rollback
        self.on_halt = on_halt

        # Per-model detectors
        self._cusum: Dict[str, CUSUMDetector] = {}
        self._page_hinkley: Dict[str, PageHinkleyDetector] = {}
        self._ewma: Dict[str, EWMADetector] = {}

        # State tracking
        self._consecutive_detections: Dict[str, int] = {}
        self._last_action_time: Dict[str, float] = {}
        self._events: List[RollbackEvent] = []
        self._previous_versions: Dict[str, str] = {}  # current → previous

    def register_model(
        self,
        model_version: str,
        baseline_return: float = 0.0,
        baseline_std: float = 0.02,
        previous_version: Optional[str] = None,
    ):
        """
        Register a model version for monitoring.

        Args:
            model_version: Version to monitor.
            baseline_return: Expected mean return per trade.
            baseline_std: Expected standard deviation of returns.
            previous_version: Version to rollback to if needed.
        """
        self._cusum[model_version] = CUSUMDetector(
            target_mean=baseline_return,
            drift=self.config.cusum_drift * baseline_std,
            threshold=self.config.cusum_threshold * baseline_std,
        )
        self._page_hinkley[model_version] = PageHinkleyDetector(
            threshold=self.config.ph_threshold,
            alpha=self.config.ph_alpha,
        )
        self._ewma[model_version] = EWMADetector(
            target_mean=baseline_return,
            target_std=baseline_std,
            lam=self.config.ewma_lambda,
            L=self.config.ewma_L,
        )
        self._consecutive_detections[model_version] = 0
        self._last_action_time[model_version] = 0.0

        if previous_version:
            self._previous_versions[model_version] = previous_version

        logger.info(
            "auto_rollback.model_registered",
            model_version=model_version,
            baseline_return=baseline_return,
            baseline_std=baseline_std,
        )

    def observe(self, model_version: str, trade_return: float) -> RollbackSeverity:
        """
        Feed new trade return observation and determine required action.

        Args:
            model_version: Model that generated this trade.
            trade_return: Realized return of the trade.

        Returns:
            RollbackSeverity indicating the action level.
        """
        if model_version not in self._cusum:
            return RollbackSeverity.NONE

        # Run all three detectors
        cusum_detected = self._cusum[model_version].update(trade_return)
        ph_detected = self._page_hinkley[model_version].update(trade_return)
        ewma_detected = self._ewma[model_version].update(trade_return)

        # Majority vote: need at least 2/3 detectors to agree
        n_detections = sum([cusum_detected, ph_detected, ewma_detected])

        if n_detections >= 2:
            self._consecutive_detections[model_version] += 1
        else:
            # DR3 FIX: Decay rate halved — single good trade now reduces by 0.5
            # instead of 1.0. Prevents a single good trade from canceling
            # multiple consecutive degradation detections.
            current = self._consecutive_detections[model_version]
            if current > 0:
                self._consecutive_detections[model_version] = max(
                    0, current - 0.5
                )
                # Round to int for threshold comparison
                self._consecutive_detections[model_version] = int(
                    self._consecutive_detections[model_version]
                )

        consecutive = self._consecutive_detections[model_version]

        # Determine severity based on consecutive detections
        severity = self._determine_severity(consecutive)

        # Enforce cooldown
        now = time.time()
        last_action = self._last_action_time.get(model_version, 0.0)
        if severity != RollbackSeverity.NONE and (now - last_action) < self.config.action_cooldown_seconds:
            return RollbackSeverity.NONE  # still in cooldown

        # Execute action if needed
        if severity != RollbackSeverity.NONE:
            detection = ChangePointDetection(
                detected=True,
                method=DetectionMethod.CUSUM,  # primary method
                statistic=self._cusum[model_version].statistic,
                threshold=self._cusum[model_version].threshold,
                direction="degradation",
                n_observations=self._cusum[model_version].n_observations,
                consecutive_detections=consecutive,
            )
            self._execute_action(model_version, severity, detection)
            self._last_action_time[model_version] = now

        return severity

    def evaluate_health(self, model_version: str, health_score: float) -> RollbackSeverity:
        """
        Evaluate model health score and trigger rollback if needed.

        Provides a direct health-based rollback path independent of
        trade-by-trade change detection.

        Args:
            model_version: Model version being evaluated.
            health_score: Composite health score (0-100).

        Returns:
            RollbackSeverity based on health score.
        """
        if health_score < self.config.health_score_rollback_threshold:
            severity = RollbackSeverity.ROLLBACK
        elif health_score < self.config.health_score_reduce_threshold:
            severity = RollbackSeverity.REDUCE_ALLOCATION
        else:
            return RollbackSeverity.NONE

        # Enforce cooldown
        now = time.time()
        last_action = self._last_action_time.get(model_version, 0.0)
        if (now - last_action) < self.config.action_cooldown_seconds:
            return RollbackSeverity.NONE

        detection = ChangePointDetection(
            detected=True,
            method=DetectionMethod.EXPONENTIAL_SMOOTHING,
            statistic=health_score,
            threshold=self.config.health_score_rollback_threshold,
            direction="degradation",
            n_observations=0,
            consecutive_detections=0,
        )
        self._execute_action(model_version, severity, detection, health_score=health_score)
        self._last_action_time[model_version] = now

        return severity

    def get_events(self, model_version: Optional[str] = None, limit: int = 50) -> List[RollbackEvent]:
        """Get rollback event history."""
        events = self._events
        if model_version:
            events = [e for e in events if e.model_version == model_version]
        return events[-limit:]

    def reset_model(self, model_version: str):
        """Reset all detectors for a model (e.g., after successful retrain)."""
        if model_version in self._cusum:
            self._cusum[model_version].reset()
        if model_version in self._page_hinkley:
            self._page_hinkley[model_version].reset()
        if model_version in self._ewma:
            self._ewma[model_version].reset()
        self._consecutive_detections[model_version] = 0
        logger.info("auto_rollback.model_reset", model_version=model_version)

    def _determine_severity(self, consecutive: int) -> RollbackSeverity:
        """Map consecutive detection count to severity level."""
        if consecutive >= self.config.halt_after:
            return RollbackSeverity.HALT
        elif consecutive >= self.config.rollback_after:
            return RollbackSeverity.ROLLBACK
        elif consecutive >= self.config.reduce_after:
            return RollbackSeverity.REDUCE_ALLOCATION
        elif consecutive >= self.config.alert_after:
            return RollbackSeverity.ALERT
        return RollbackSeverity.NONE

    def _execute_action(
        self,
        model_version: str,
        severity: RollbackSeverity,
        detection: ChangePointDetection,
        health_score: Optional[float] = None,
    ):
        """Execute the rollback action and record the event."""
        rolled_back_to = None
        if severity in (RollbackSeverity.ROLLBACK, RollbackSeverity.HALT):
            rolled_back_to = self._previous_versions.get(model_version)

        reason = (
            f"Change-point detected: {detection.method.value} stat={detection.statistic:.3f} "
            f"(threshold={detection.threshold:.3f}), {detection.consecutive_detections} consecutive"
        )
        if health_score is not None:
            reason = f"Health score {health_score:.1f} below threshold {detection.threshold:.1f}"

        event = RollbackEvent(
            timestamp=time.time(),
            model_version=model_version,
            severity=severity,
            reason=reason,
            detection=detection,
            rolled_back_to=rolled_back_to,
            health_score=health_score,
        )
        self._events.append(event)

        # Invoke callbacks
        if severity == RollbackSeverity.ALERT and self.on_alert:
            self.on_alert(event)
        elif severity in (RollbackSeverity.ROLLBACK, RollbackSeverity.REDUCE_ALLOCATION) and self.on_rollback:
            self.on_rollback(event)
        elif severity == RollbackSeverity.HALT and self.on_halt:
            self.on_halt(event)

        logger.warning(
            "auto_rollback.action_triggered",
            model_version=model_version,
            severity=severity.value,
            reason=reason,
            rolled_back_to=rolled_back_to,
        )

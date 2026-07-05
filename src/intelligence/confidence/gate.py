"""
Confidence Gate — the unified orchestrator for multi-dimensional confidence.

Pipeline:
    Signal → Signal Integrity → Data Quality → Model Health →
    Calibration → Temporal Decay → Regime Adjustment → Final Score

Each gate can independently REJECT (short-circuit) or ADJUST (discount).
The final ConfidenceScore carries full provenance of every gate's decision.

This replaces the single `if confidence <= 0.5: reject` with a rich,
auditable, multi-gate evaluation that produces an explainable result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import structlog

from src.intelligence.confidence.calibrator import (
    CalibrationConfig,
    ConfidenceCalibrator,
)
from src.intelligence.confidence.data_quality_gate import (
    DataQualityConfig,
    DataQualityGate,
)
from src.intelligence.confidence.decay import (
    ConfidenceDecayEngine,
    DecayConfig,
)
from src.intelligence.confidence.model_health_gate import (
    ModelHealthConfig,
    ModelHealthGate,
)
from src.intelligence.confidence.models import (
    ConfidenceBreakdown,
    ConfidenceComponent,
    ConfidenceScore,
    DataQuality,
    GateResult,
    GateVerdict,
    ModelHealth,
    SignalIntegrity,
    ThresholdProfile,
)
from src.intelligence.confidence.regime_adjuster import (
    MarketRegimeAdjuster,
    RegimeAdjustConfig,
    RegimeProfile,
)

logger = structlog.get_logger(__name__)


@dataclass
class ConfidenceGateConfig:
    """Master configuration for the confidence gate pipeline."""

    # Sub-gate configs
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    model_health: ModelHealthConfig = field(default_factory=ModelHealthConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)
    regime: RegimeAdjustConfig = field(default_factory=RegimeAdjustConfig)

    # Threshold profile (default: balanced)
    threshold_profile: ThresholdProfile = field(
        default_factory=ThresholdProfile.balanced
    )

    # Master switches
    enable_data_quality: bool = True
    enable_model_health: bool = True
    enable_calibration: bool = True
    enable_decay: bool = True
    enable_regime_adjustment: bool = True

    # Legacy compatibility: hard floor (absolute minimum regardless of profile)
    absolute_floor: float = 0.30


@dataclass
class SignalContext:
    """
    All contextual information needed to evaluate a signal's confidence.

    Populated by the execution engine or orchestrator before calling
    the confidence gate.
    """

    # Signal metadata
    symbol: str = ""
    strategy: str = ""
    raw_confidence: float = 0.0
    signal_integrity: SignalIntegrity = SignalIntegrity.REAL
    signal_generated_at: Optional[datetime] = None

    # Data quality inputs
    features: Optional[np.ndarray] = None
    last_candle_timestamp: Optional[float] = None
    volume_data_age_seconds: float = 0.0
    bars_available: int = 0
    spread_available: bool = True
    orderbook_delay_ms: float = 0.0

    # Model health inputs
    model_version: str = ""
    accuracy_baseline: float = 0.0
    accuracy_recent: float = 0.0
    predictions_count: int = 0
    calibration_ece: float = 0.0
    last_retrained: Optional[datetime] = None
    is_drift_detected: bool = False

    # Market regime inputs (from MarketFingerprint)
    current_trend: str = ""
    current_volatility: str = ""
    current_stress: str = ""
    current_liquidity: str = ""
    current_momentum: str = ""
    fingerprint_confidence: float = 1.0

    # Confidence components (from multi-strategy agreement)
    components: list[ConfidenceComponent] = field(default_factory=list)

    # Regime profile override (None = auto from strategy name)
    regime_profile: Optional[RegimeProfile] = None


class ConfidenceGate:
    """
    Master confidence gate — orchestrates the full evaluation pipeline.

    Replaces the scalar confidence check with a structured, auditable,
    multi-gate evaluation that produces a rich ConfidenceScore object.

    Usage:
        gate = ConfidenceGate(config)
        score = gate.evaluate(context)

        if score.approved:
            execute_trade(confidence=score.value)
        else:
            log_rejection(score.rejection_reason, score.breakdown)
    """

    def __init__(self, config: Optional[ConfidenceGateConfig] = None):
        self.config = config or ConfidenceGateConfig()

        # Initialize sub-gates
        self._data_quality = DataQualityGate(self.config.data_quality)
        self._model_health = ModelHealthGate(self.config.model_health)
        self._calibrator = ConfidenceCalibrator(self.config.calibration)
        self._decay = ConfidenceDecayEngine(self.config.decay)
        self._regime = MarketRegimeAdjuster(self.config.regime)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(self, context: SignalContext) -> ConfidenceScore:
        """
        Evaluate a signal through the full confidence pipeline.

        Pipeline order:
        1. Signal Integrity (explicit status check)
        2. Data Quality (input reliability)
        3. Model Health (model performance check)
        4. Calibration (adjust for historical accuracy)
        5. Temporal Decay (time-based degradation)
        6. Regime Adjustment (market compatibility)
        7. Threshold Check (profile-based approval)

        Returns:
            ConfidenceScore with full provenance and approval status.
        """
        total_start = time.perf_counter()
        gate_results: list[GateResult] = []
        confidence = context.raw_confidence

        # ── Gate 1: Signal Integrity ──
        integrity_result = self._check_integrity(context)
        gate_results.append(integrity_result)
        if integrity_result.verdict == GateVerdict.REJECT:
            return self._build_rejected(
                context, gate_results, integrity_result.reason, total_start
            )

        # ── Gate 2: Data Quality ──
        data_quality = DataQuality()
        if self.config.enable_data_quality:
            dq_result, data_quality = self._data_quality.evaluate(
                features=context.features,
                last_candle_timestamp=context.last_candle_timestamp,
                volume_data_age_seconds=context.volume_data_age_seconds,
                bars_available=context.bars_available,
                spread_available=context.spread_available,
                orderbook_delay_ms=context.orderbook_delay_ms,
            )
            gate_results.append(dq_result)
            if dq_result.verdict == GateVerdict.REJECT:
                return self._build_rejected(
                    context, gate_results, dq_result.reason, total_start,
                    data_quality=data_quality,
                )
            if dq_result.verdict == GateVerdict.ADJUST:
                confidence = max(0.0, confidence + dq_result.adjustment)

        # ── Gate 3: Model Health ──
        model_health = ModelHealth()
        if self.config.enable_model_health:
            mh_result, model_health = self._model_health.evaluate(
                model_version=context.model_version,
                accuracy_baseline=context.accuracy_baseline,
                accuracy_recent=context.accuracy_recent,
                predictions_count=context.predictions_count,
                calibration_ece=context.calibration_ece,
                last_retrained=context.last_retrained,
                is_drift_detected=context.is_drift_detected,
            )
            gate_results.append(mh_result)
            if mh_result.verdict == GateVerdict.REJECT:
                return self._build_rejected(
                    context, gate_results, mh_result.reason, total_start,
                    data_quality=data_quality,
                    model_health=model_health,
                )
            if mh_result.verdict == GateVerdict.ADJUST:
                confidence = max(0.0, confidence + mh_result.adjustment)

        # ── Gate 4: Calibration ──
        calibration_adjustment = 0.0
        if self.config.enable_calibration:
            cal_result = self._calibrator.evaluate(confidence)
            gate_results.append(cal_result)
            if cal_result.verdict == GateVerdict.ADJUST:
                calibration_adjustment = cal_result.adjustment
                confidence = max(0.0, min(1.0, confidence + calibration_adjustment))

        # ── Gate 5: Temporal Decay ──
        decay_factor = 1.0
        if self.config.enable_decay and context.signal_generated_at:
            elapsed = (datetime.now() - context.signal_generated_at).total_seconds() / 60.0
            decay_result = self._decay.evaluate(elapsed, confidence)
            gate_results.append(decay_result)
            if decay_result.verdict == GateVerdict.REJECT:
                return self._build_rejected(
                    context, gate_results, decay_result.reason, total_start,
                    data_quality=data_quality,
                    model_health=model_health,
                )
            if decay_result.verdict == GateVerdict.ADJUST:
                decay_factor = self._decay.compute_factor(elapsed)
                confidence = confidence * decay_factor

        # ── Gate 6: Regime Adjustment ──
        regime_factor = 1.0
        if self.config.enable_regime_adjustment and context.current_trend:
            regime_result = self._regime.evaluate(
                strategy_name=context.strategy,
                current_trend=context.current_trend,
                current_volatility=context.current_volatility,
                current_stress=context.current_stress,
                current_liquidity=context.current_liquidity,
                current_momentum=context.current_momentum,
                fingerprint_confidence=context.fingerprint_confidence,
                regime_profile=context.regime_profile,
            )
            gate_results.append(regime_result)
            if regime_result.verdict == GateVerdict.REJECT:
                return self._build_rejected(
                    context, gate_results, regime_result.reason, total_start,
                    data_quality=data_quality,
                    model_health=model_health,
                )
            if regime_result.verdict == GateVerdict.ADJUST:
                regime_factor = 1.0 + regime_result.adjustment
                confidence = max(0.0, confidence * regime_factor)

        # ── Gate 7: Threshold Check ──
        profile = self.config.threshold_profile
        final_confidence = max(0.0, min(1.0, confidence))

        # Absolute floor (non-negotiable)
        if final_confidence < self.config.absolute_floor:
            reason = (
                f"Final confidence {final_confidence:.3f} below absolute floor "
                f"({self.config.absolute_floor:.2f})"
            )
            gate_results.append(GateResult(
                gate_name="threshold",
                verdict=GateVerdict.REJECT,
                score=final_confidence,
                reason=reason,
            ))
            return self._build_rejected(
                context, gate_results, reason, total_start,
                data_quality=data_quality,
                model_health=model_health,
            )

        # Profile-based threshold
        if final_confidence < profile.min_confidence:
            reason = (
                f"Final confidence {final_confidence:.3f} below {profile.mode.value} "
                f"threshold ({profile.min_confidence:.2f})"
            )
            gate_results.append(GateResult(
                gate_name="threshold",
                verdict=GateVerdict.REJECT,
                score=final_confidence,
                reason=reason,
            ))
            return self._build_rejected(
                context, gate_results, reason, total_start,
                data_quality=data_quality,
                model_health=model_health,
            )

        # ── APPROVED ──
        gate_results.append(GateResult(
            gate_name="threshold",
            verdict=GateVerdict.PASS,
            score=final_confidence,
            reason=f"Confidence {final_confidence:.3f} passes {profile.mode.value} threshold ({profile.min_confidence:.2f})",
        ))

        breakdown = ConfidenceBreakdown(
            raw_model_probability=context.raw_confidence,
            calibration_adjustment=calibration_adjustment,
            data_quality_factor=data_quality.overall_score,
            feature_freshness_factor=1.0 - (data_quality.last_candle_age_seconds / max(1, self.config.data_quality.critical_candle_age_seconds)),
            model_health_factor=model_health.health_score,
            regime_compatibility_factor=regime_factor,
            strategy_agreement_factor=self._compute_strategy_agreement(context.components),
            temporal_decay_factor=decay_factor,
            final_confidence=final_confidence,
            gate_results=gate_results,
            components=context.components,
        )

        total_ms = (time.perf_counter() - total_start) * 1000

        logger.info(
            "confidence_gate.approved",
            symbol=context.symbol,
            strategy=context.strategy,
            raw=context.raw_confidence,
            final=final_confidence,
            gates_passed=len([g for g in gate_results if g.verdict == GateVerdict.PASS]),
            gates_adjusted=len([g for g in gate_results if g.verdict == GateVerdict.ADJUST]),
            elapsed_ms=round(total_ms, 2),
        )

        return ConfidenceScore(
            value=final_confidence,
            integrity=context.signal_integrity,
            approved=True,
            breakdown=breakdown,
            data_quality=data_quality,
            model_health=model_health,
            computed_at=datetime.now(),
            strategy=context.strategy,
            symbol=context.symbol,
        )

    def record_outcome(self, predicted_confidence: float, was_correct: bool) -> None:
        """Record a trade outcome for calibration learning."""
        self._calibrator.record_outcome(predicted_confidence, was_correct)

    def update_threshold_profile(self, profile: ThresholdProfile) -> None:
        """Update the threshold profile (e.g., switching modes)."""
        self.config.threshold_profile = profile
        logger.info(
            "confidence_gate.threshold_updated",
            mode=profile.mode.value,
            min_confidence=profile.min_confidence,
        )

    @property
    def calibrator(self) -> ConfidenceCalibrator:
        """Access calibrator for reporting."""
        return self._calibrator

    # ──────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _check_integrity(self, context: SignalContext) -> GateResult:
        """Gate 1: Check signal integrity status."""
        if context.signal_integrity == SignalIntegrity.REAL:
            return GateResult(
                gate_name="signal_integrity",
                verdict=GateVerdict.PASS,
                score=1.0,
                reason="Signal integrity: REAL",
            )

        # All non-REAL statuses are rejected
        return GateResult(
            gate_name="signal_integrity",
            verdict=GateVerdict.REJECT,
            score=0.0,
            reason=f"Signal integrity: {context.signal_integrity.value} — non-tradeable signal",
            details={"integrity": context.signal_integrity.value},
        )

    def _build_rejected(
        self,
        context: SignalContext,
        gate_results: list[GateResult],
        reason: str,
        start_time: float,
        data_quality: Optional[DataQuality] = None,
        model_health: Optional[ModelHealth] = None,
    ) -> ConfidenceScore:
        """Build a rejected ConfidenceScore with full provenance."""
        total_ms = (time.perf_counter() - start_time) * 1000

        breakdown = ConfidenceBreakdown(
            raw_model_probability=context.raw_confidence,
            final_confidence=0.0,
            gate_results=gate_results,
            components=context.components,
        )

        logger.info(
            "confidence_gate.rejected",
            symbol=context.symbol,
            strategy=context.strategy,
            raw=context.raw_confidence,
            reason=reason,
            gate=gate_results[-1].gate_name if gate_results else "unknown",
            elapsed_ms=round(total_ms, 2),
        )

        return ConfidenceScore(
            value=0.0,
            integrity=context.signal_integrity,
            approved=False,
            rejection_reason=reason,
            breakdown=breakdown,
            data_quality=data_quality or DataQuality(),
            model_health=model_health or ModelHealth(),
            computed_at=datetime.now(),
            strategy=context.strategy,
            symbol=context.symbol,
        )

    def _compute_strategy_agreement(
        self, components: list[ConfidenceComponent]
    ) -> float:
        """Compute inter-strategy agreement from components."""
        if not components or len(components) < 2:
            return 1.0  # Single strategy — no disagreement possible

        scores = [c.score for c in components if c.score > 0]
        if len(scores) < 2:
            return 1.0

        # Agreement = 1 - coefficient_of_variation
        mean = sum(scores) / len(scores)
        if mean == 0:
            return 0.0
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        cv = (variance ** 0.5) / mean
        return max(0.0, min(1.0, 1.0 - cv))

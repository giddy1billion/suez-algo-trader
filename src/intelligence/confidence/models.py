"""
Core data models for the multi-dimensional confidence system.

Confidence is no longer a single float — it is a structured object
carrying provenance, component scores, calibration state, and
temporal validity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────


class SignalIntegrity(str, Enum):
    """Explicit signal quality status — never infer from confidence alone."""

    REAL = "real"
    PLACEHOLDER = "placeholder"
    MODEL_UNAVAILABLE = "model_unavailable"
    FEATURES_MISSING = "features_missing"
    FEATURES_STALE = "features_stale"
    BACKTEST_FAILED = "backtest_failed"
    TRAINING_PENDING = "training_pending"
    STALE_MODEL = "stale_model"
    FALLBACK = "fallback"
    SIMULATED = "simulated"


class GateVerdict(str, Enum):
    """Outcome of a single gate evaluation."""

    PASS = "pass"
    ADJUST = "adjust"
    REJECT = "reject"
    SKIP = "skip"  # Gate not applicable (e.g., no calibration data yet)


class ThresholdMode(str, Enum):
    """Pre-configured threshold profiles."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    PAPER = "paper"
    ADAPTIVE = "adaptive"


# ──────────────────────────────────────────────────────────────────────────────
# Component Scores
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConfidenceComponent:
    """A single dimension of the confidence assessment."""

    name: str
    score: float  # 0.0–1.0 (or adjustment factor)
    weight: float = 1.0  # Contribution weight to final score
    source: str = ""  # Where this score came from
    detail: str = ""  # Human-readable explanation


@dataclass(frozen=True)
class GateResult:
    """Result from a single gate in the confidence pipeline."""

    gate_name: str
    verdict: GateVerdict
    score: float  # Gate-specific score (0.0–1.0)
    adjustment: float = 0.0  # Additive or multiplicative adjustment applied
    reason: str = ""
    details: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass
class DataQuality:
    """Assessment of input data reliability."""

    last_candle_age_seconds: float = 0.0
    volume_freshness: float = 1.0  # 1.0 = fresh, 0.0 = stale
    spread_available: bool = True
    feature_completeness: float = 1.0  # fraction of features non-NaN
    bars_available: int = 0
    bars_required: int = 100
    orderbook_delay_ms: float = 0.0
    overall_score: float = 1.0  # Composite data quality (0.0–1.0)

    @property
    def is_sufficient(self) -> bool:
        return (
            self.overall_score >= 0.5
            and self.feature_completeness >= 0.7
            and self.bars_available >= self.bars_required
        )


@dataclass
class ModelHealth:
    """Assessment of model performance and reliability."""

    model_version: str = ""
    accuracy_baseline: float = 0.0  # Historical accuracy
    accuracy_recent: float = 0.0  # Recent window accuracy
    accuracy_drop: float = 0.0  # baseline - recent
    is_degrading: bool = False
    predictions_count: int = 0
    last_retrained: Optional[datetime] = None
    model_age_hours: float = 0.0
    calibration_ece: float = 0.0  # Expected Calibration Error
    calibration_reliability: float = 1.0  # 1.0 - ECE
    health_score: float = 1.0  # Composite model health (0.0–1.0)

    @property
    def is_healthy(self) -> bool:
        return self.health_score >= 0.5 and not self.is_degrading


# ──────────────────────────────────────────────────────────────────────────────
# Confidence Breakdown (Full Provenance)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ConfidenceBreakdown:
    """
    Complete provenance of how the final confidence was computed.

    This is the audit trail — every adjustment is recorded with its
    reason and magnitude, allowing full reconstructibility.
    """

    raw_model_probability: float = 0.0
    calibration_adjustment: float = 0.0
    data_quality_factor: float = 1.0
    feature_freshness_factor: float = 1.0
    model_health_factor: float = 1.0
    regime_compatibility_factor: float = 1.0
    strategy_agreement_factor: float = 1.0
    temporal_decay_factor: float = 1.0
    final_confidence: float = 0.0

    # Per-gate results (ordered by evaluation)
    gate_results: list[GateResult] = field(default_factory=list)

    # Component scores contributing to raw confidence
    components: list[ConfidenceComponent] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for logging and audit."""
        return {
            "raw_model_probability": round(self.raw_model_probability, 4),
            "calibration_adjustment": round(self.calibration_adjustment, 4),
            "data_quality_factor": round(self.data_quality_factor, 4),
            "feature_freshness_factor": round(self.feature_freshness_factor, 4),
            "model_health_factor": round(self.model_health_factor, 4),
            "regime_compatibility_factor": round(self.regime_compatibility_factor, 4),
            "strategy_agreement_factor": round(self.strategy_agreement_factor, 4),
            "temporal_decay_factor": round(self.temporal_decay_factor, 4),
            "final_confidence": round(self.final_confidence, 4),
            "gates": [
                {
                    "gate": g.gate_name,
                    "verdict": g.verdict.value,
                    "score": round(g.score, 4),
                    "adjustment": round(g.adjustment, 4),
                    "reason": g.reason,
                }
                for g in self.gate_results
            ],
            "components": [
                {
                    "name": c.name,
                    "score": round(c.score, 4),
                    "weight": round(c.weight, 4),
                    "source": c.source,
                }
                for c in self.components
            ],
        }


# ──────────────────────────────────────────────────────────────────────────────
# The Confidence Score (First-Class Object)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ConfidenceScore:
    """
    Multi-dimensional confidence assessment — the first-class object
    that replaces the single `confidence: float` scalar.

    Carries:
    - The final numeric score (backward-compatible with existing code)
    - Signal integrity status (explicit, not inferred)
    - Full breakdown with provenance
    - Data quality and model health assessments
    - Whether the signal was approved or rejected by the confidence gate
    - Human-readable explanation
    """

    # Final score (0.0–1.0) — backward-compatible with existing float
    value: float = 0.0

    # Explicit signal status
    integrity: SignalIntegrity = SignalIntegrity.REAL

    # Did the confidence gate approve this signal?
    approved: bool = False
    rejection_reason: str = ""

    # Full provenance
    breakdown: ConfidenceBreakdown = field(default_factory=ConfidenceBreakdown)

    # Sub-assessments
    data_quality: DataQuality = field(default_factory=DataQuality)
    model_health: ModelHealth = field(default_factory=ModelHealth)

    # Metadata
    computed_at: datetime = field(default_factory=datetime.now)
    strategy: str = ""
    symbol: str = ""
    expires_at: Optional[datetime] = None

    def __float__(self) -> float:
        """Allow use as a float for backward compatibility."""
        return self.value

    def __lt__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return self.value < other
        if isinstance(other, ConfidenceScore):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return self.value <= other
        if isinstance(other, ConfidenceScore):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return self.value > other
        if isinstance(other, ConfidenceScore):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return self.value >= other
        if isinstance(other, ConfidenceScore):
            return self.value >= other.value
        return NotImplemented

    def __eq__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return self.value == other
        if isinstance(other, ConfidenceScore):
            return self.value == other.value
        return NotImplemented

    @property
    def is_valid(self) -> bool:
        """Whether this confidence score represents a tradeable signal."""
        return (
            self.approved
            and self.integrity == SignalIntegrity.REAL
            and self.data_quality.is_sufficient
            and self.model_health.is_healthy
        )

    @property
    def explanation(self) -> str:
        """Human-readable summary of the confidence assessment."""
        if not self.approved:
            return f"REJECTED: {self.rejection_reason}"

        parts = [f"Confidence: {self.value:.1%}"]

        if self.breakdown.regime_compatibility_factor < 0.9:
            parts.append(
                f"regime discount: {1 - self.breakdown.regime_compatibility_factor:.0%}"
            )
        if self.breakdown.temporal_decay_factor < 1.0:
            parts.append(
                f"decay: {1 - self.breakdown.temporal_decay_factor:.0%}"
            )
        if self.breakdown.model_health_factor < 0.9:
            parts.append(
                f"model health: {self.breakdown.model_health_factor:.0%}"
            )
        if self.breakdown.data_quality_factor < 0.9:
            parts.append(
                f"data quality: {self.breakdown.data_quality_factor:.0%}"
            )

        return " | ".join(parts)

    def to_dict(self) -> dict:
        """Full serialization for audit logging."""
        return {
            "value": round(self.value, 4),
            "integrity": self.integrity.value,
            "approved": self.approved,
            "rejection_reason": self.rejection_reason,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "computed_at": self.computed_at.isoformat(),
            "breakdown": self.breakdown.to_dict(),
            "data_quality_score": round(self.data_quality.overall_score, 4),
            "model_health_score": round(self.model_health.health_score, 4),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Threshold Profiles
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ThresholdProfile:
    """
    Configurable confidence thresholds by operational mode.

    Replaces hardcoded 0.50/0.55 with mode-aware profiles.
    """

    mode: ThresholdMode
    min_confidence: float  # Below this → reject
    preferred_confidence: float  # Below this → reduce size
    high_confidence: float  # Above this → full allocation
    data_quality_min: float  # Minimum data quality score
    model_health_min: float  # Minimum model health score
    max_decay_factor: float  # Max allowed temporal decay

    @classmethod
    def conservative(cls) -> ThresholdProfile:
        return cls(
            mode=ThresholdMode.CONSERVATIVE,
            min_confidence=0.72,
            preferred_confidence=0.80,
            high_confidence=0.90,
            data_quality_min=0.85,
            model_health_min=0.80,
            max_decay_factor=0.90,
        )

    @classmethod
    def balanced(cls) -> ThresholdProfile:
        return cls(
            mode=ThresholdMode.BALANCED,
            min_confidence=0.65,
            preferred_confidence=0.75,
            high_confidence=0.85,
            data_quality_min=0.70,
            model_health_min=0.70,
            max_decay_factor=0.80,
        )

    @classmethod
    def aggressive(cls) -> ThresholdProfile:
        return cls(
            mode=ThresholdMode.AGGRESSIVE,
            min_confidence=0.58,
            preferred_confidence=0.68,
            high_confidence=0.80,
            data_quality_min=0.60,
            model_health_min=0.60,
            max_decay_factor=0.70,
        )

    @classmethod
    def paper(cls) -> ThresholdProfile:
        return cls(
            mode=ThresholdMode.PAPER,
            min_confidence=0.40,
            preferred_confidence=0.55,
            high_confidence=0.70,
            data_quality_min=0.50,
            model_health_min=0.40,
            max_decay_factor=0.50,
        )

    @classmethod
    def for_mode(cls, mode: str) -> ThresholdProfile:
        """Factory method — resolve mode string to profile."""
        profiles = {
            "conservative": cls.conservative,
            "balanced": cls.balanced,
            "aggressive": cls.aggressive,
            "paper": cls.paper,
        }
        factory = profiles.get(mode.lower(), cls.balanced)
        return factory()

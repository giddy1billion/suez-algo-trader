"""
Decision Contract — The immutable, auditable assessment that flows through
the entire trading pipeline as the system's central governance object.

This is NOT just "confidence as a number." This IS the decision itself.

Every downstream component receives the EXACT same object:
    Market Data → Feature Store → ML Prediction → Decision Contract
        → Risk Manager → Position Sizing → Execution Engine → Trade
        → Experience Database → Dataset Registry → Model Governance

Once created, the DecisionContract is IMMUTABLE. Nothing modifies it.
It is a permanent audit record of why a trade was (or wasn't) executed.

Design principles:
1. Every stage produces a StageAssessment (not just a float)
2. Any stage can VETO (hard-fail, unconditional stop)
3. The contract carries full provenance (reproducible months later)
4. Confidence decays — the contract has an expiry
5. Downstream consumers never mutate the contract

Reference: Institutional trading desk decision governance patterns.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────


class Decision(str, Enum):
    """Final decision outcome."""
    EXECUTE = "execute"
    REJECT = "reject"
    REDUCE = "reduce"  # execute but with reduced position
    DEFER = "defer"  # wait for better conditions


class StageSeverity(str, Enum):
    """Severity of a stage assessment issue."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"  # triggers veto


class VetoAuthority(str, Enum):
    """Which stage exercised veto power."""
    DATA_QUALITY = "data_quality"
    FEATURE_QUALITY = "feature_quality"
    MODEL_RELIABILITY = "model_reliability"
    MARKET_REGIME = "market_regime"
    RISK_ALIGNMENT = "risk_alignment"
    EXECUTION_FEASIBILITY = "execution_feasibility"
    CIRCUIT_BREAKER = "circuit_breaker"
    OPERATOR = "operator"


# ──────────────────────────────────────────────────────────────────────────────
# Stage Assessment — Every gate produces one of these
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StageAssessment:
    """
    A single stage's contribution to the decision.

    NOT just a float. This is a structured object carrying:
    - The score (0-1)
    - Whether this stage passed
    - Whether this stage VETOED the entire decision
    - Evidence supporting the score
    - Warnings for downstream consumers
    - Blockers that prevent execution
    - Weight in the final calculation
    - Severity classification

    Example:
        StageAssessment(
            stage="market_regime",
            score=0.84,
            passed=True,
            weight=0.18,
            severity=StageSeverity.LOW,
            evidence={"regime": "TRENDING", "adx": 32, "vol_pct": 68},
            warnings=[],
            blockers=[],
        )
    """

    stage: str
    score: float  # 0.0-1.0
    passed: bool
    weight: float = 0.15  # contribution weight to final score (sum of all = 1.0)
    severity: StageSeverity = StageSeverity.NONE
    veto: bool = False  # if True, execution STOPS regardless of other scores
    veto_reason: str = ""

    # Rich evidence (what the stage observed)
    evidence: dict[str, Any] = field(default_factory=dict)

    # Warnings (non-blocking concerns)
    warnings: list[str] = field(default_factory=list)

    # Blockers (reasons this stage would prevent execution)
    blockers: list[str] = field(default_factory=list)

    # Timing
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    evaluation_ms: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Provenance — Full reproducibility metadata
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DecisionProvenance:
    """
    Complete provenance for reproducibility.

    Months later, you can reconstruct exactly why a trade executed:
    - Which model version generated the prediction
    - Which features were used (version + snapshot hash)
    - Which dataset trained the model
    - What market state existed at decision time
    - What validation artifacts backed the model
    """

    model_version: str = ""
    feature_set_version: str = ""
    dataset_version: str = ""
    training_run_id: str = ""
    backtest_id: str = ""
    walk_forward_id: str = ""
    prediction_id: str = ""
    market_snapshot_hash: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    git_commit: str = ""

    # Validation status at decision time
    walk_forward_passed: Optional[bool] = None
    monte_carlo_passed: Optional[bool] = None
    reality_check_passed: Optional[bool] = None
    deflated_sharpe: Optional[float] = None
    model_health_score: Optional[float] = None


# ──────────────────────────────────────────────────────────────────────────────
# The Decision Contract — IMMUTABLE central object
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DecisionContract:
    """
    THE immutable decision contract.

    This is the single object that flows through the entire system.
    Once created, it is NEVER modified. Every consumer — Risk Manager,
    Position Sizer, Execution Engine, Experience Database, Dataset
    Registry — receives this exact same object.

    It replaces:
        confidence = 0.82

    With:
        contract = DecisionContract(
            decision=Decision.EXECUTE,
            final_confidence=0.82,
            recommendation="EXECUTE with 3.2% allocation",
            stages=[...7 StageAssessments...],
            provenance=DecisionProvenance(...),
            ...
        )

    Properties:
    - Immutable (frozen=True): cannot be modified after creation
    - Auditable: full provenance chain
    - Explainable: every stage shows its reasoning
    - Temporal: has creation time and expiry
    - Veto-aware: any stage can halt execution
    """

    # ── Identity ──
    contract_id: str = field(default_factory=lambda: f"DC-{uuid.uuid4().hex[:12].upper()}")

    # ── Decision ──
    decision: Decision = Decision.REJECT
    final_confidence: float = 0.0
    recommendation: str = ""

    # ── Symbol & Direction ──
    symbol: str = ""
    direction: str = ""  # BUY, SELL, HOLD

    # ── Stage Assessments (ordered by evaluation) ──
    stages: tuple[StageAssessment, ...] = field(default_factory=tuple)

    # ── Provenance ──
    provenance: DecisionProvenance = field(default_factory=DecisionProvenance)

    # ── Timing & Expiry ──
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(minutes=5))

    # ── Veto Information ──
    vetoed: bool = False
    vetoed_by: Optional[VetoAuthority] = None
    veto_reason: str = ""

    # ── Position Sizing Recommendation ──
    recommended_position_pct: float = 0.0
    kelly_fraction: float = 0.0
    risk_grade: str = ""

    # ── Integrity Hash (for tamper detection) ──
    integrity_hash: str = ""

    # ──────────────────────────────────────────────────────────────────────
    # Computed Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def is_executable(self) -> bool:
        """Whether this contract authorizes trade execution."""
        return (
            self.decision == Decision.EXECUTE
            and not self.vetoed
            and not self.is_expired
            and self.final_confidence > 0
        )

    @property
    def is_expired(self) -> bool:
        """Whether this contract has passed its validity window."""
        return datetime.now(timezone.utc) > self.valid_until

    @property
    def age_seconds(self) -> float:
        """Seconds since this contract was created."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    @property
    def stage_scores(self) -> dict[str, float]:
        """Quick access to all stage scores as a dict."""
        return {s.stage: s.score for s in self.stages}

    @property
    def warnings(self) -> list[str]:
        """Aggregate all warnings from all stages."""
        all_warnings = []
        for stage in self.stages:
            for w in stage.warnings:
                all_warnings.append(f"[{stage.stage}] {w}")
        return all_warnings

    @property
    def blockers(self) -> list[str]:
        """Aggregate all blockers from all stages."""
        all_blockers = []
        for stage in self.stages:
            for b in stage.blockers:
                all_blockers.append(f"[{stage.stage}] {b}")
        return all_blockers

    @property
    def explanation(self) -> str:
        """Human-readable decision summary."""
        lines = [
            f"{'═'*60}",
            f"DECISION CONTRACT: {self.contract_id}",
            f"{'═'*60}",
            f"Decision: {self.decision.value.upper()} | Confidence: {self.final_confidence:.1%}",
            f"Symbol: {self.symbol} | Direction: {self.direction}",
            f"Valid until: {self.valid_until.strftime('%H:%M:%S UTC')}",
        ]
        if self.vetoed:
            lines.append(f"⛔ VETOED by {self.vetoed_by.value}: {self.veto_reason}")
        lines.append(f"{'─'*60}")
        lines.append("Stages:")
        for s in self.stages:
            status = "✅" if s.passed else ("⛔" if s.veto else "❌")
            lines.append(f"  {status} {s.stage:<25} {s.score:.0%} (weight: {s.weight:.0%})")
            if s.veto:
                lines.append(f"     VETO: {s.veto_reason}")
            for w in s.warnings:
                lines.append(f"     ⚠ {w}")
        if self.warnings:
            lines.append(f"{'─'*60}")
            lines.append(f"Warnings: {len(self.warnings)}")
        lines.append(f"{'═'*60}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Full serialization for permanent audit storage and replay."""
        return self._serialize()

    def to_audit_dict(self) -> dict[str, Any]:
        """Alias for to_dict() — kept for backward compatibility."""
        return self._serialize()

    def _serialize(self) -> dict[str, Any]:
        """Internal serialization implementation."""
        return {
            "contract_id": self.contract_id,
            "decision": self.decision.value,
            "final_confidence": self.final_confidence,
            "recommendation": self.recommendation,
            "symbol": self.symbol,
            "direction": self.direction,
            "created_at": self.created_at.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "vetoed": self.vetoed,
            "vetoed_by": self.vetoed_by.value if self.vetoed_by else None,
            "veto_reason": self.veto_reason,
            "recommended_position_pct": self.recommended_position_pct,
            "risk_grade": self.risk_grade,
            "stages": [
                {
                    "stage": s.stage,
                    "score": s.score,
                    "passed": s.passed,
                    "weight": s.weight,
                    "severity": s.severity.value,
                    "veto": s.veto,
                    "evidence": s.evidence,
                    "warnings": s.warnings,
                    "blockers": s.blockers,
                }
                for s in self.stages
            ],
            "provenance": {
                "model_version": self.provenance.model_version,
                "feature_set_version": self.provenance.feature_set_version,
                "dataset_version": self.provenance.dataset_version,
                "prediction_id": self.provenance.prediction_id,
                "walk_forward_passed": self.provenance.walk_forward_passed,
                "monte_carlo_passed": self.provenance.monte_carlo_passed,
                "model_health_score": self.provenance.model_health_score,
            },
            "integrity_hash": self.integrity_hash,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Decision Contract Builder — Produces immutable contracts
# ──────────────────────────────────────────────────────────────────────────────


class DecisionContractBuilder:
    """
    Builds an immutable DecisionContract from the confidence gate pipeline.

    Usage:
        builder = DecisionContractBuilder()
        builder.set_symbol("BTC/USD", "BUY")
        builder.set_provenance(model_version="v14.3", ...)
        builder.add_stage(StageAssessment(stage="data_quality", ...))
        builder.add_stage(StageAssessment(stage="market_regime", ...))
        contract = builder.build()

    Once build() is called, the contract is immutable and the builder
    is consumed (cannot build again without reset).
    """

    def __init__(self, validity_minutes: float = 5.0):
        self._symbol: str = ""
        self._direction: str = ""
        self._stages: list[StageAssessment] = []
        self._provenance: DecisionProvenance = DecisionProvenance()
        self._validity_minutes: float = validity_minutes
        self._position_pct: float = 0.0
        self._kelly: float = 0.0
        self._risk_grade: str = ""
        self._built: bool = False

    def set_symbol(self, symbol: str, direction: str) -> "DecisionContractBuilder":
        """Set the trading symbol and direction."""
        self._symbol = symbol
        self._direction = direction
        return self

    def set_provenance(self, **kwargs) -> "DecisionContractBuilder":
        """Set provenance metadata."""
        self._provenance = DecisionProvenance(**kwargs)
        return self

    def set_sizing(
        self,
        position_pct: float = 0.0,
        kelly: float = 0.0,
        risk_grade: str = "",
    ) -> "DecisionContractBuilder":
        """Set position sizing recommendation."""
        self._position_pct = position_pct
        self._kelly = kelly
        self._risk_grade = risk_grade
        return self

    def add_stage(self, assessment: StageAssessment) -> "DecisionContractBuilder":
        """Add a stage assessment to the contract."""
        self._stages.append(assessment)
        return self

    def build(self) -> DecisionContract:
        """
        Build the immutable DecisionContract.

        Resolves:
        - Final confidence (weighted average of passing stages)
        - Decision (EXECUTE/REJECT/REDUCE based on scores and vetoes)
        - Veto detection (any stage with veto=True halts execution)
        - Integrity hash (for tamper detection)

        Returns:
            Frozen DecisionContract.
        """
        if self._built:
            raise RuntimeError("DecisionContractBuilder already consumed. Call reset() first.")
        self._built = True

        now = datetime.now(timezone.utc)
        valid_until = now + timedelta(minutes=self._validity_minutes)

        # Check for vetoes (any stage can halt everything)
        vetoed = False
        vetoed_by: Optional[VetoAuthority] = None
        veto_reason = ""

        for stage in self._stages:
            if stage.veto:
                vetoed = True
                try:
                    vetoed_by = VetoAuthority(stage.stage)
                except ValueError:
                    vetoed_by = VetoAuthority.CIRCUIT_BREAKER
                veto_reason = stage.veto_reason or f"Stage '{stage.stage}' exercised veto authority"
                break

        # Compute final confidence (weighted average of all stages)
        total_weight = sum(s.weight for s in self._stages)
        if total_weight > 0 and not vetoed:
            final_confidence = sum(s.score * s.weight for s in self._stages) / total_weight
        else:
            final_confidence = 0.0

        final_confidence = max(0.0, min(1.0, final_confidence))

        # Determine decision
        if vetoed:
            decision = Decision.REJECT
            recommendation = f"REJECTED (veto by {vetoed_by.value}): {veto_reason}"
        elif any(not s.passed for s in self._stages):
            # Any failing non-veto stage → check if confidence still viable
            failing = [s for s in self._stages if not s.passed]
            if final_confidence >= 0.65:
                decision = Decision.REDUCE
                recommendation = f"REDUCE: {len(failing)} stage(s) failed but confidence {final_confidence:.0%} above threshold"
            else:
                decision = Decision.REJECT
                reasons = "; ".join(s.blockers[0] if s.blockers else f"{s.stage} failed" for s in failing)
                recommendation = f"REJECTED: {reasons}"
        elif final_confidence >= 0.75:
            decision = Decision.EXECUTE
            recommendation = f"EXECUTE with {self._position_pct:.1f}% allocation (confidence {final_confidence:.0%})"
        elif final_confidence >= 0.60:
            decision = Decision.REDUCE
            recommendation = f"REDUCE: confidence {final_confidence:.0%} below preferred threshold"
        else:
            decision = Decision.REJECT
            recommendation = f"REJECTED: confidence {final_confidence:.0%} too low"

        # Compute integrity hash
        hash_input = f"{self._symbol}|{self._direction}|{final_confidence:.6f}|{now.isoformat()}"
        integrity_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        return DecisionContract(
            decision=decision,
            final_confidence=final_confidence,
            recommendation=recommendation,
            symbol=self._symbol,
            direction=self._direction,
            stages=tuple(self._stages),
            provenance=self._provenance,
            created_at=now,
            valid_until=valid_until,
            vetoed=vetoed,
            vetoed_by=vetoed_by,
            veto_reason=veto_reason,
            recommended_position_pct=self._position_pct,
            kelly_fraction=self._kelly,
            risk_grade=self._risk_grade,
            integrity_hash=integrity_hash,
        )

    def reset(self) -> "DecisionContractBuilder":
        """Reset the builder for reuse."""
        self._symbol = ""
        self._direction = ""
        self._stages = []
        self._provenance = DecisionProvenance()
        self._position_pct = 0.0
        self._kelly = 0.0
        self._risk_grade = ""
        self._built = False
        return self

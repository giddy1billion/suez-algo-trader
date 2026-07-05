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
class DeploymentRecord:
    """
    Who deployed this model and when?

    Captures the full deployment lineage so months later you can trace
    a trading decision back to the exact deployment event.
    """
    deployed_by: str = ""          # Operator/system that deployed (e.g., "ci-pipeline", "john.doe")
    deployment_id: str = ""        # Unique deployment identifier (e.g., "deploy-2026-07-05-a3f2")
    deployment_pipeline: str = ""  # Pipeline that produced the deployment (e.g., "closed_loop_v2")
    deployed_at: Optional[datetime] = None  # When the model was deployed to production
    promotion_reason: str = ""     # Why this model was promoted (e.g., "champion_challenger_winner")
    approved_by: str = ""          # Who approved the promotion (operator or "auto")
    environment: str = "production"  # production, shadow, staging


@dataclass(frozen=True)
class ThresholdsApplied:
    """
    Which thresholds gated this decision?

    Records the exact numeric thresholds that were applied at decision time.
    Critical for debugging: "why was this rejected?" → "confidence 0.72 < threshold 0.75"
    """
    min_confidence: float = 0.75        # Minimum to execute
    reduce_confidence: float = 0.60     # Below this → REDUCE instead of EXECUTE
    reject_confidence: float = 0.0      # Below this → hard REJECT
    max_position_pct: float = 5.0       # Maximum position as % of portfolio
    max_drawdown_pct: float = 15.0      # Maximum drawdown before halt
    max_correlation: float = 0.70       # Maximum correlation with existing positions
    min_risk_reward: float = 1.5        # Minimum expected risk/reward ratio
    validity_minutes: float = 5.0       # Contract validity window
    veto_stages: tuple[str, ...] = field(default_factory=lambda: (
        "data_quality", "feature_quality", "circuit_breaker",
    ))


@dataclass(frozen=True)
class FeatureAttribution:
    """
    Which features mattered for this prediction?

    Carries the top-N feature importances from the explainability module.
    Answers: "Why did the model predict BUY?"
    """
    top_features: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    # e.g., (("rsi_14", 0.23), ("ema_cross", 0.18), ("volume_surge", 0.12))
    method: str = ""  # "shap", "permutation", "lime"
    baseline_prediction: float = 0.0
    total_attribution_sum: float = 0.0


@dataclass(frozen=True)
class AlternativeConsidered:
    """A signal that was evaluated but NOT taken in this cycle."""
    symbol: str = ""
    direction: str = ""
    raw_confidence: float = 0.0
    rejection_reason: str = ""
    rejected_by_stage: str = ""  # Which stage rejected it


@dataclass(frozen=True)
class DecisionRationale:
    """
    The structured "why" and "why not" of a trading decision.

    This is the core auditable answer to:
    - Why was THIS trade taken?
    - Why wasn't ANOTHER trade taken instead?
    - What was the primary signal driver?
    - What risk factors were considered?

    Every contract carries one. It's generated from stages + provenance.
    """
    # Why this trade was taken (or rejected)
    primary_reason: str = ""
    # e.g., "Strong momentum signal (RSI=72, EMA cross up) with high model confidence (0.88)"

    # Supporting evidence (top 3 factors that pushed toward EXECUTE)
    supporting_factors: tuple[str, ...] = field(default_factory=tuple)
    # e.g., ("Trending market regime (ADX=32)", "Model accuracy 74% on similar regimes", "Low spread 2.1 bps")

    # Risk factors (concerns that reduced confidence or position size)
    risk_factors: tuple[str, ...] = field(default_factory=tuple)
    # e.g., ("Elevated volatility (68th percentile)", "Correlation 0.3 with existing BTC position")

    # Why alternatives were not taken
    alternatives: tuple[AlternativeConsidered, ...] = field(default_factory=tuple)

    # Feature attribution (which features drove the prediction)
    feature_attribution: FeatureAttribution = field(default_factory=FeatureAttribution)

    # Strategy that generated the signal
    strategy_name: str = ""
    signal_type: str = ""  # e.g., "momentum_crossover", "mean_reversion_oversold"


@dataclass(frozen=True)
class DecisionProvenance:
    """
    Complete provenance for reproducibility.

    Months later, you can answer:
    - Which model version generated the prediction? → model_version
    - Which features mattered? → feature_set_version + attribution
    - Who deployed that model? → deployment.deployed_by
    - When? → deployment.deployed_at, timestamp
    - Which dataset trained it? → dataset_version
    - What validation backed it? → walk_forward_passed, etc.
    """

    # Model identity
    model_version: str = ""         # e.g., "v14.3"
    model_id: str = ""              # Unique model identifier (e.g., "momentum_btc_v14")
    model_name: str = ""            # Human-readable name (e.g., "BTC Momentum Predictor")
    model_type: str = ""            # e.g., "xgboost_multi_target", "lstm_classifier"

    # Data lineage
    feature_set_version: str = ""   # Feature store version (e.g., "fs_82")
    dataset_version: str = ""       # Training dataset version (e.g., "ds_2026_07_05")
    training_run_id: str = ""       # MLflow/internal training run ID
    training_completed_at: Optional[datetime] = None

    # Deployment lineage — WHO deployed and WHEN
    deployment: DeploymentRecord = field(default_factory=DeploymentRecord)

    # Prediction tracking
    prediction_id: str = ""         # Unique prediction identifier
    backtest_id: str = ""
    walk_forward_id: str = ""

    # Market state snapshot
    market_snapshot_hash: str = ""  # Hash of the market data used for this decision

    # Timestamps
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    git_commit: str = ""

    # Validation status at decision time
    walk_forward_passed: Optional[bool] = None
    monte_carlo_passed: Optional[bool] = None
    reality_check_passed: Optional[bool] = None
    deflated_sharpe: Optional[float] = None
    model_health_score: Optional[float] = None

    # Thresholds that were active at decision time
    thresholds: ThresholdsApplied = field(default_factory=ThresholdsApplied)


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
    recommended_stop_loss: float = 0.0
    recommended_take_profit: float = 0.0
    kelly_fraction: float = 0.0
    risk_grade: str = ""

    # ── Decision Rationale — WHY was this decision made? ──
    rationale: DecisionRationale = field(default_factory=DecisionRationale)

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
            f"{'='*60}",
            f"DECISION CONTRACT: {self.contract_id}",
            f"{'='*60}",
            f"Decision: {self.decision.value.upper()} | Confidence: {self.final_confidence:.1%}",
            f"Symbol: {self.symbol} | Direction: {self.direction}",
            f"Valid until: {self.valid_until.strftime('%H:%M:%S UTC')}",
        ]
        if self.vetoed:
            lines.append(f"VETOED by {self.vetoed_by.value}: {self.veto_reason}")
        if self.rationale.primary_reason:
            lines.append(f"{'-'*60}")
            lines.append(f"WHY: {self.rationale.primary_reason}")
            if self.rationale.supporting_factors:
                for f in self.rationale.supporting_factors:
                    lines.append(f"  + {f}")
            if self.rationale.risk_factors:
                for f in self.rationale.risk_factors:
                    lines.append(f"  - {f}")
        lines.append(f"{'-'*60}")
        lines.append("Stages:")
        for s in self.stages:
            status = "[PASS]" if s.passed else ("[VETO]" if s.veto else "[FAIL]")
            lines.append(f"  {status} {s.stage:<25} {s.score:.0%} (weight: {s.weight:.0%})")
            if s.veto:
                lines.append(f"     VETO: {s.veto_reason}")
            for w in s.warnings:
                lines.append(f"     WARN: {w}")
        if self.warnings:
            lines.append(f"{'-'*60}")
            lines.append(f"Warnings: {len(self.warnings)}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)

    @property
    def audit_answers(self) -> dict[str, str]:
        """
        Direct answers to the 8 institutional audit questions.

        Returns a dict with human-readable answers:
        1. Why was this trade taken?
        2. Why wasn't another taken?
        3. Which model approved it?
        4. Which thresholds applied?
        5. Which features mattered?
        6. Which version generated the prediction?
        7. Who deployed that model?
        8. When?
        """
        # 1. Why was this trade taken?
        if self.decision == Decision.EXECUTE:
            why_taken = self.rationale.primary_reason or self.recommendation
        elif self.decision == Decision.REJECT:
            why_taken = f"NOT taken: {self.rationale.primary_reason or self.recommendation}"
        else:
            why_taken = f"Reduced position: {self.rationale.primary_reason or self.recommendation}"

        # 2. Why wasn't another taken?
        if self.rationale.alternatives:
            alts = [
                f"{a.symbol} {a.direction}: {a.rejection_reason} (rejected by {a.rejected_by_stage})"
                for a in self.rationale.alternatives
            ]
            why_not_other = "; ".join(alts)
        else:
            why_not_other = "No alternatives evaluated in this cycle"

        # 3. Which model approved it?
        model_info = self.provenance.model_name or self.provenance.model_id or self.provenance.model_version
        model_type = f" ({self.provenance.model_type})" if self.provenance.model_type else ""
        which_model = f"{model_info}{model_type}" if model_info else "No model specified"

        # 4. Which thresholds applied?
        t = self.provenance.thresholds
        thresholds = (
            f"min_confidence={t.min_confidence:.0%}, "
            f"reduce={t.reduce_confidence:.0%}, "
            f"max_position={t.max_position_pct:.1f}%, "
            f"validity={t.validity_minutes:.0f}min, "
            f"veto_stages={list(t.veto_stages)}"
        )

        # 5. Which features mattered?
        fa = self.rationale.feature_attribution
        if fa.top_features:
            features = ", ".join(f"{name}={importance:.3f}" for name, importance in fa.top_features[:5])
            features += f" (method: {fa.method})" if fa.method else ""
        else:
            features = "Feature attribution not available"

        # 6. Which version generated the prediction?
        version = self.provenance.model_version or "Unknown"
        ds = self.provenance.dataset_version
        fs = self.provenance.feature_set_version
        version_detail = version
        if ds:
            version_detail += f", dataset={ds}"
        if fs:
            version_detail += f", features={fs}"

        # 7. Who deployed that model?
        dep = self.provenance.deployment
        if dep.deployed_by:
            who = f"{dep.deployed_by} via {dep.deployment_pipeline or 'manual'}"
            if dep.promotion_reason:
                who += f" (reason: {dep.promotion_reason})"
            if dep.approved_by:
                who += f" [approved by: {dep.approved_by}]"
        else:
            who = "Deployment record not available"

        # 8. When?
        if dep.deployed_at:
            when = dep.deployed_at.strftime("%Y-%m-%d %H:%M UTC")
        else:
            when = f"Decision at {self.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"

        return {
            "why_taken": why_taken,
            "why_not_other": why_not_other,
            "which_model": which_model,
            "thresholds_applied": thresholds,
            "features_mattered": features,
            "prediction_version": version_detail,
            "who_deployed": who,
            "when": when,
        }

    def to_dict(self) -> dict[str, Any]:
        """Full serialization for permanent audit storage and replay."""
        return self._serialize()

    def to_audit_dict(self) -> dict[str, Any]:
        """Alias for to_dict() — kept for backward compatibility."""
        return self._serialize()

    def _serialize(self) -> dict[str, Any]:
        """Internal serialization implementation — complete audit record."""
        dep = self.provenance.deployment
        fa = self.rationale.feature_attribution
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
            "rationale": {
                "primary_reason": self.rationale.primary_reason,
                "supporting_factors": list(self.rationale.supporting_factors),
                "risk_factors": list(self.rationale.risk_factors),
                "alternatives": [
                    {
                        "symbol": a.symbol,
                        "direction": a.direction,
                        "raw_confidence": a.raw_confidence,
                        "rejection_reason": a.rejection_reason,
                        "rejected_by_stage": a.rejected_by_stage,
                    }
                    for a in self.rationale.alternatives
                ],
                "feature_attribution": {
                    "top_features": [
                        {"name": name, "importance": imp}
                        for name, imp in fa.top_features
                    ],
                    "method": fa.method,
                    "baseline_prediction": fa.baseline_prediction,
                },
                "strategy_name": self.rationale.strategy_name,
                "signal_type": self.rationale.signal_type,
            },
            "provenance": {
                "model_version": self.provenance.model_version,
                "model_id": self.provenance.model_id,
                "model_name": self.provenance.model_name,
                "model_type": self.provenance.model_type,
                "feature_set_version": self.provenance.feature_set_version,
                "dataset_version": self.provenance.dataset_version,
                "training_run_id": self.provenance.training_run_id,
                "prediction_id": self.provenance.prediction_id,
                "git_commit": self.provenance.git_commit,
                "walk_forward_passed": self.provenance.walk_forward_passed,
                "monte_carlo_passed": self.provenance.monte_carlo_passed,
                "model_health_score": self.provenance.model_health_score,
                "deployment": {
                    "deployed_by": dep.deployed_by,
                    "deployment_id": dep.deployment_id,
                    "deployment_pipeline": dep.deployment_pipeline,
                    "deployed_at": dep.deployed_at.isoformat() if dep.deployed_at else None,
                    "promotion_reason": dep.promotion_reason,
                    "approved_by": dep.approved_by,
                    "environment": dep.environment,
                },
                "thresholds": {
                    "min_confidence": self.provenance.thresholds.min_confidence,
                    "reduce_confidence": self.provenance.thresholds.reduce_confidence,
                    "max_position_pct": self.provenance.thresholds.max_position_pct,
                    "validity_minutes": self.provenance.thresholds.validity_minutes,
                    "veto_stages": list(self.provenance.thresholds.veto_stages),
                },
            },
            "audit_answers": self.audit_answers,
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
        self._rationale: Optional[DecisionRationale] = None
        self._alternatives: list[AlternativeConsidered] = []
        self._feature_attribution: FeatureAttribution = FeatureAttribution()
        self._strategy_name: str = ""
        self._signal_type: str = ""
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

    def set_rationale(
        self,
        primary_reason: str = "",
        supporting_factors: Optional[list[str]] = None,
        risk_factors: Optional[list[str]] = None,
        strategy_name: str = "",
        signal_type: str = "",
    ) -> "DecisionContractBuilder":
        """Set the decision rationale (why this trade was/wasn't taken)."""
        self._strategy_name = strategy_name
        self._signal_type = signal_type
        self._rationale = DecisionRationale(
            primary_reason=primary_reason,
            supporting_factors=tuple(supporting_factors or []),
            risk_factors=tuple(risk_factors or []),
            alternatives=tuple(self._alternatives),
            feature_attribution=self._feature_attribution,
            strategy_name=strategy_name,
            signal_type=signal_type,
        )
        return self

    def add_alternative(
        self,
        symbol: str,
        direction: str,
        raw_confidence: float,
        rejection_reason: str,
        rejected_by_stage: str = "",
    ) -> "DecisionContractBuilder":
        """Record an alternative signal that was NOT taken."""
        self._alternatives.append(AlternativeConsidered(
            symbol=symbol,
            direction=direction,
            raw_confidence=raw_confidence,
            rejection_reason=rejection_reason,
            rejected_by_stage=rejected_by_stage,
        ))
        return self

    def set_feature_attribution(
        self,
        top_features: Optional[list[tuple[str, float]]] = None,
        method: str = "",
        baseline_prediction: float = 0.0,
    ) -> "DecisionContractBuilder":
        """Set which features mattered for this prediction."""
        self._feature_attribution = FeatureAttribution(
            top_features=tuple(top_features or []),
            method=method,
            baseline_prediction=baseline_prediction,
            total_attribution_sum=sum(abs(v) for _, v in (top_features or [])),
        )
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

        if not self._symbol or not self._direction:
            raise ValueError(
                f"DecisionContractBuilder requires symbol and direction. "
                f"Got symbol='{self._symbol}', direction='{self._direction}'"
            )

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

        # Build rationale (auto-generate from stages if not explicitly set)
        rationale = self._rationale or self._auto_generate_rationale(
            decision, final_confidence, vetoed, veto_reason
        )

        return DecisionContract(
            decision=decision,
            final_confidence=final_confidence,
            recommendation=recommendation,
            symbol=self._symbol,
            direction=self._direction,
            stages=tuple(self._stages),
            provenance=self._provenance,
            rationale=rationale,
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

    def _auto_generate_rationale(
        self, decision: Decision, confidence: float, vetoed: bool, veto_reason: str
    ) -> DecisionRationale:
        """Auto-generate rationale from stages when not explicitly provided."""
        # Primary reason
        if vetoed:
            primary = f"Signal vetoed: {veto_reason}"
        elif decision == Decision.EXECUTE:
            top_stage = max(self._stages, key=lambda s: s.score * s.weight) if self._stages else None
            primary = (
                f"All {len(self._stages)} quality gates passed with {confidence:.0%} confidence"
                + (f", strongest signal from {top_stage.stage} ({top_stage.score:.0%})" if top_stage else "")
            )
        elif decision == Decision.REDUCE:
            failing = [s for s in self._stages if not s.passed]
            primary = f"Confidence {confidence:.0%} viable but position reduced due to: {', '.join(s.stage for s in failing) or 'moderate confidence'}"
        else:
            failing = [s for s in self._stages if not s.passed]
            primary = f"Rejected: {', '.join(s.blockers[0] if s.blockers else s.stage for s in failing) or f'confidence {confidence:.0%} too low'}"

        # Supporting factors: stages that scored above average
        avg_score = confidence
        supporting = tuple(
            f"{s.stage}: {s.score:.0%}" + (f" ({list(s.evidence.values())[:2]})" if s.evidence else "")
            for s in sorted(self._stages, key=lambda x: x.score, reverse=True)[:3]
            if s.score >= avg_score and s.passed
        )

        # Risk factors: stages with warnings or low scores
        risk = tuple(
            w for s in self._stages for w in s.warnings
        ) + tuple(
            f"{s.stage} below threshold ({s.score:.0%})"
            for s in self._stages if s.score < 0.7 and s.passed
        )

        return DecisionRationale(
            primary_reason=primary,
            supporting_factors=supporting[:5],
            risk_factors=risk[:5],
            alternatives=tuple(self._alternatives),
            feature_attribution=self._feature_attribution,
            strategy_name=self._strategy_name,
            signal_type=self._signal_type,
        )

    def reset(self) -> "DecisionContractBuilder":
        """Reset the builder for reuse."""
        self._symbol = ""
        self._direction = ""
        self._stages = []
        self._provenance = DecisionProvenance()
        self._rationale = None
        self._alternatives = []
        self._feature_attribution = FeatureAttribution()
        self._strategy_name = ""
        self._signal_type = ""
        self._position_pct = 0.0
        self._kelly = 0.0
        self._risk_grade = ""
        self._built = False
        return self

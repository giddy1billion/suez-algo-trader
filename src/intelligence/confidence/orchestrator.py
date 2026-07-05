"""
Decision Orchestrator — Produces immutable DecisionContracts from the
confidence gate pipeline.

This is the integration layer that:
1. Takes a SignalContext (from execution engine)
2. Runs it through the ConfidenceGate stages
3. Enriches with provenance, multi-target predictions, and explainability
4. Produces a frozen DecisionContract that flows through the entire system

Every trading decision goes through here. No exceptions.

Usage:
    orchestrator = DecisionOrchestrator()

    # From the execution engine:
    contract = orchestrator.evaluate(signal_context, prediction, explanation)

    if contract.is_executable:
        risk_engine.evaluate(contract)
        execution_engine.execute(contract)
        experience_db.record(contract)
    else:
        audit_log.record_rejection(contract)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.intelligence.confidence.decision_contract import (
    AlternativeConsidered,
    Decision,
    DecisionContract,
    DecisionContractBuilder,
    DecisionProvenance,
    DeploymentRecord,
    FeatureAttribution,
    StageAssessment,
    StageSeverity,
    ThresholdsApplied,
    VetoAuthority,
)
from src.intelligence.confidence.gate import ConfidenceGate, ConfidenceGateConfig, SignalContext
from src.intelligence.confidence.models import GateVerdict
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DecisionOrchestrator:
    """
    Produces immutable DecisionContracts from raw signal contexts.

    Integrates:
    - ConfidenceGate (multi-stage confidence evaluation)
    - MultiTargetPredictor (direction, return, duration, risk)
    - PredictionExplainer (SHAP-based feature attribution)
    - ModelHealthMonitor (health scoring)
    - Risk Engine (position sizing)

    The output DecisionContract is the SINGLE source of truth for:
    - Whether to execute
    - At what size
    - Why (full provenance and evidence)
    - When it expires
    """

    def __init__(
        self,
        gate_config: Optional[ConfidenceGateConfig] = None,
        validity_minutes: float = 5.0,
    ):
        self._gate = ConfidenceGate(gate_config)
        self._validity_minutes = validity_minutes

    def evaluate(
        self,
        context: SignalContext,
        multi_target: Optional[Dict[str, Any]] = None,
        explanation: Optional[Dict[str, Any]] = None,
        provenance_kwargs: Optional[Dict[str, Any]] = None,
        position_pct: float = 0.0,
        kelly_fraction: float = 0.0,
        risk_grade: str = "",
        strategy_name: str = "",
        signal_type: str = "",
        alternatives: Optional[List[Dict[str, Any]]] = None,
        deployment: Optional[Dict[str, Any]] = None,
        thresholds: Optional[Dict[str, Any]] = None,
    ) -> DecisionContract:
        """
        Evaluate a signal and produce an immutable DecisionContract.

        Args:
            context: SignalContext with all inputs for the confidence gate.
            multi_target: Optional dict from MultiTargetPredictor.to_dict().
            explanation: Optional dict from PredictionExplainer.to_evidence_dict().
            provenance_kwargs: Optional provenance metadata overrides.
            position_pct: Recommended position size (% of portfolio).
            kelly_fraction: Kelly criterion optimal fraction.
            risk_grade: Risk grade (A-F).
            strategy_name: Strategy that generated this signal.
            signal_type: Signal type (e.g., "momentum_crossover").
            alternatives: Signals evaluated but NOT taken this cycle.
            deployment: Deployment record for the active model.
            thresholds: Thresholds active at decision time.

        Returns:
            Frozen DecisionContract — the system's decision.
        """
        start = time.perf_counter()

        # Run through confidence gate (produces ConfidenceScore)
        confidence_score = self._gate.evaluate(context)

        # Convert gate results to StageAssessments
        builder = DecisionContractBuilder(validity_minutes=self._validity_minutes)
        builder.set_symbol(context.symbol, "BUY" if confidence_score.approved else "HOLD")

        # Build provenance with deployment and thresholds
        prov_kwargs: Dict[str, Any] = {
            "model_version": context.model_version,
            "timestamp": datetime.now(timezone.utc),
        }
        if deployment:
            prov_kwargs["deployment"] = DeploymentRecord(**deployment)
        if thresholds:
            prov_kwargs["thresholds"] = ThresholdsApplied(**thresholds)
        if provenance_kwargs:
            prov_kwargs.update(provenance_kwargs)
        builder.set_provenance(**prov_kwargs)

        # Set sizing
        builder.set_sizing(
            position_pct=position_pct,
            kelly=kelly_fraction,
            risk_grade=risk_grade,
        )

        # Record alternatives (signals NOT taken this cycle)
        for alt in (alternatives or []):
            builder.add_alternative(
                symbol=alt.get("symbol", ""),
                direction=alt.get("direction", ""),
                raw_confidence=alt.get("raw_confidence", 0.0),
                rejection_reason=alt.get("rejection_reason", ""),
                rejected_by_stage=alt.get("rejected_by_stage", ""),
            )

        # Set feature attribution from explainability output
        if explanation:
            top_features = explanation.get("top_features", [])
            # Normalize: list of dicts or list of tuples
            feature_tuples = self._normalize_features(top_features)
            builder.set_feature_attribution(
                top_features=feature_tuples,
                method=explanation.get("method", "shap"),
                baseline_prediction=explanation.get("baseline_prediction", 0.0),
            )

        # Convert each gate result to a StageAssessment
        supporting_factors: List[str] = []
        risk_factors: List[str] = []

        for gate_result in confidence_score.breakdown.gate_results:
            is_veto = (
                gate_result.verdict == GateVerdict.REJECT
                and gate_result.gate_name in ("signal_integrity", "data_quality")
            )

            severity = StageSeverity.NONE
            if gate_result.verdict == GateVerdict.REJECT:
                severity = StageSeverity.CRITICAL if is_veto else StageSeverity.HIGH
            elif gate_result.verdict == GateVerdict.ADJUST and gate_result.adjustment < -0.1:
                severity = StageSeverity.MEDIUM

            blockers = [gate_result.reason] if gate_result.verdict == GateVerdict.REJECT else []
            warnings = [gate_result.reason] if gate_result.verdict == GateVerdict.ADJUST and gate_result.adjustment < -0.05 else []

            evidence = dict(gate_result.details) if gate_result.details else {}
            evidence["raw_score"] = gate_result.score
            if gate_result.adjustment != 0:
                evidence["adjustment"] = gate_result.adjustment

            stage = StageAssessment(
                stage=gate_result.gate_name,
                score=gate_result.score,
                passed=gate_result.verdict != GateVerdict.REJECT,
                weight=self._get_stage_weight(gate_result.gate_name),
                severity=severity,
                veto=is_veto,
                veto_reason=gate_result.reason if is_veto else "",
                evidence=evidence,
                warnings=warnings,
                blockers=blockers,
                evaluation_ms=gate_result.elapsed_ms,
            )
            builder.add_stage(stage)

            # Collect rationale factors from stages
            if gate_result.score >= 0.8 and gate_result.verdict != GateVerdict.REJECT:
                supporting_factors.append(
                    f"{gate_result.gate_name}: {gate_result.reason} (score={gate_result.score:.2f})"
                )
            elif gate_result.verdict == GateVerdict.ADJUST and gate_result.adjustment < -0.05:
                risk_factors.append(
                    f"{gate_result.gate_name}: {gate_result.reason} (adj={gate_result.adjustment:.3f})"
                )
            elif gate_result.verdict == GateVerdict.REJECT:
                risk_factors.append(
                    f"{gate_result.gate_name}: BLOCKED - {gate_result.reason}"
                )

        # Add multi-target evidence as an additional stage (if available)
        if multi_target:
            mt_score = multi_target.get("direction_probability", 0.5)
            mt_confidence = multi_target.get("confidence", 0.5)
            builder.add_stage(StageAssessment(
                stage="multi_target_prediction",
                score=mt_confidence,
                passed=mt_score > 0.4,
                weight=0.15,
                severity=StageSeverity.NONE if mt_score > 0.4 else StageSeverity.MEDIUM,
                evidence={
                    "direction": multi_target.get("direction", "HOLD"),
                    "expected_return_pct": multi_target.get("expected_return_pct", 0),
                    "risk_reward": multi_target.get("risk_reward_ratio", 0),
                    "probability_tp": multi_target.get("probability_tp", 0),
                    "holding_hours": multi_target.get("expected_holding_hours", 0),
                },
            ))
            if mt_score > 0.6:
                supporting_factors.append(
                    f"Multi-target model agrees: p={mt_score:.2f}, E[R]={multi_target.get('expected_return_pct', 0):.2f}%"
                )

        # Add explainability stage (if available)
        if explanation:
            top_features_raw = explanation.get("top_features", [])
            builder.add_stage(StageAssessment(
                stage="explainability",
                score=0.9 if top_features_raw else 0.5,
                passed=True,
                weight=0.05,
                evidence={
                    "top_features": top_features_raw[:5],
                    "summary": explanation.get("summary", ""),
                },
            ))

        # Build rationale from collected evidence
        primary_reason = self._synthesize_primary_reason(
            context.symbol, confidence_score.approved, confidence_score.final_score,
            strategy_name, signal_type, supporting_factors
        )
        builder.set_rationale(
            primary_reason=primary_reason,
            supporting_factors=supporting_factors[:5],
            risk_factors=risk_factors[:5],
            strategy_name=strategy_name,
            signal_type=signal_type,
        )

        # Build the immutable contract
        contract = builder.build()

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "decision_orchestrator.contract_produced",
            contract_id=contract.contract_id,
            symbol=contract.symbol,
            decision=contract.decision.value,
            confidence=round(contract.final_confidence, 3),
            stages=len(contract.stages),
            vetoed=contract.vetoed,
            has_rationale=contract.rationale is not None,
            strategy=strategy_name,
            elapsed_ms=round(elapsed_ms, 2),
        )

        return contract

    def record_outcome(self, predicted_confidence: float, was_correct: bool):
        """Pass outcome to calibrator for online learning."""
        self._gate.record_outcome(predicted_confidence, was_correct)

    def _get_stage_weight(self, gate_name: str) -> float:
        """Map gate names to decision weights."""
        weights = {
            "signal_integrity": 0.05,
            "data_quality": 0.15,
            "model_health": 0.15,
            "calibration": 0.10,
            "decay": 0.10,
            "regime": 0.15,
            "threshold": 0.15,
            "risk": 0.10,
            "execution": 0.05,
        }
        return weights.get(gate_name, 0.10)

    def _normalize_features(
        self, top_features: Any
    ) -> List[Tuple[str, float]]:
        """Normalize feature list to [(name, importance)] tuples."""
        result: List[Tuple[str, float]] = []
        if not top_features:
            return result
        for item in top_features[:10]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                result.append((str(item[0]), float(item[1])))
            elif isinstance(item, dict):
                name = item.get("feature", item.get("name", "unknown"))
                importance = item.get("importance", item.get("value", 0.0))
                result.append((str(name), float(importance)))
        return result

    def _synthesize_primary_reason(
        self,
        symbol: str,
        approved: bool,
        final_score: float,
        strategy_name: str,
        signal_type: str,
        supporting_factors: List[str],
    ) -> str:
        """Generate a human-readable primary reason from collected evidence."""
        if not approved:
            return (
                f"Signal REJECTED for {symbol}: confidence {final_score:.3f} below threshold. "
                f"Strategy: {strategy_name or 'unknown'}."
            )
        strategy_desc = f" via {strategy_name}" if strategy_name else ""
        signal_desc = f" ({signal_type})" if signal_type else ""
        factor_summary = ""
        if supporting_factors:
            factor_summary = f" Supported by: {supporting_factors[0].split(':')[0]}"
            if len(supporting_factors) > 1:
                factor_summary += f" + {len(supporting_factors) - 1} more factors"
        return (
            f"EXECUTE {symbol}{strategy_desc}{signal_desc} with "
            f"confidence {final_score:.3f}.{factor_summary}"
        )

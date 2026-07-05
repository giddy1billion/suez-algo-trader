"""
Confidence as a First-Class Object.

Transforms the single-scalar confidence value into a multi-dimensional,
auditable, calibrated decision object with full provenance tracking.

Pipeline:
    Signal → Signal Integrity → Data Quality → Model Health →
    Calibration → Decay → Regime Adjustment → Final Score →
    Decision Contract (immutable)

Each gate can independently REJECT a signal or ADJUST the confidence
with a recorded reason. The final DecisionContract carries the full
breakdown of how the decision was made, and flows immutably through
the entire system (Risk → Sizing → Execution → Experience DB).
"""

from src.intelligence.confidence.models import (
    ConfidenceScore,
    ConfidenceComponent,
    ConfidenceBreakdown,
    SignalIntegrity,
    DataQuality,
    ModelHealth,
    ThresholdProfile,
    GateVerdict,
    GateResult,
)
from src.intelligence.confidence.decision_contract import (
    DecisionContract,
    DecisionContractBuilder,
    DecisionProvenance,
    StageAssessment,
    StageSeverity,
    Decision,
    VetoAuthority,
)

__all__ = [
    # Legacy (still used by gate internals)
    "ConfidenceScore",
    "ConfidenceComponent",
    "ConfidenceBreakdown",
    "SignalIntegrity",
    "DataQuality",
    "ModelHealth",
    "ThresholdProfile",
    "GateVerdict",
    "GateResult",
    # Decision Contract (the new central object)
    "DecisionContract",
    "DecisionContractBuilder",
    "DecisionProvenance",
    "StageAssessment",
    "StageSeverity",
    "Decision",
    "VetoAuthority",
]

"""
Confidence as a First-Class Object.

Transforms the single-scalar confidence value into a multi-dimensional,
auditable, calibrated decision object with full provenance tracking.

Pipeline:
    Signal → Signal Integrity → Data Quality → Model Health →
    Calibration → Decay → Regime Adjustment → Final Score

Each gate can independently REJECT a signal or ADJUST the confidence
with a recorded reason. The final ConfidenceScore carries the full
breakdown of how the score was computed.
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

__all__ = [
    "ConfidenceScore",
    "ConfidenceComponent",
    "ConfidenceBreakdown",
    "SignalIntegrity",
    "DataQuality",
    "ModelHealth",
    "ThresholdProfile",
    "GateVerdict",
    "GateResult",
]

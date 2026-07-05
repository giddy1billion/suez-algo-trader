"""
Prediction Registry — Full lifecycle tracking for ML/strategy predictions.

Provides:
- PredictionRegistry: Core prediction storage and lifecycle
- OutcomeRecorder: Monitors markets and records outcomes at horizon expiry
- Prediction quality metrics and calibration analysis
"""

from src.predictions.registry import PredictionRegistry, PredictionRecord
from src.predictions.outcome_recorder import OutcomeRecorder
from src.predictions.metrics import PredictionMetrics
from src.predictions.calibration import CalibrationAnalyzer

__all__ = [
    "PredictionRegistry",
    "PredictionRecord",
    "OutcomeRecorder",
    "PredictionMetrics",
    "CalibrationAnalyzer",
]

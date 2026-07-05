"""Feature attribution — answers 'WHY did the model predict this?'

Every prediction answers "why?" with top-N feature contributions,
enabling auditability and trust in the decision pipeline.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureContribution:
    """A single feature's contribution to the prediction."""

    feature_name: str
    contribution: float  # Positive = pushes toward BUY, negative = toward SELL
    value: float  # The actual feature value at prediction time
    percentile: float  # Where this value falls historically (0-100)

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "feature_name": self.feature_name,
            "contribution": round(self.contribution, 6),
            "value": round(self.value, 6),
            "percentile": round(self.percentile, 2),
        }


@dataclass
class FeatureAttribution:
    """Complete attribution for a prediction."""

    contributions: list[FeatureContribution] = field(default_factory=list)
    method: str = "simple_diff"  # "shap" | "permutation" | "gradient" | "simple_diff"
    prediction_direction: str = "BUY"  # "BUY" or "SELL"
    top_bullish: list[str] = field(default_factory=list)  # Top features pushing BUY
    top_bearish: list[str] = field(default_factory=list)  # Top features pushing SELL
    attribution_hash: str = ""  # For audit matching

    def __post_init__(self) -> None:
        """Sort contributions and compute derived fields if not set."""
        if self.contributions and not self.top_bullish and not self.top_bearish:
            self._compute_derived()

    def _compute_derived(self) -> None:
        """Compute top_bullish, top_bearish, and hash from contributions."""
        # Sort by absolute contribution descending
        self.contributions = sorted(
            self.contributions, key=lambda c: abs(c.contribution), reverse=True
        )

        # Top bullish (positive contributions)
        self.top_bullish = [
            c.feature_name for c in self.contributions if c.contribution > 0
        ][:5]

        # Top bearish (negative contributions)
        self.top_bearish = [
            c.feature_name for c in self.contributions if c.contribution < 0
        ][:5]

        # Attribution hash for audit
        if not self.attribution_hash:
            hash_input = "|".join(
                f"{c.feature_name}:{c.contribution:.6f}" for c in self.contributions
            )
            self.attribution_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Serialize for audit storage."""
        return {
            "method": self.method,
            "prediction_direction": self.prediction_direction,
            "top_bullish": self.top_bullish,
            "top_bearish": self.top_bearish,
            "attribution_hash": self.attribution_hash,
            "contributions": [c.to_dict() for c in self.contributions],
        }


def compute_simple_attribution(
    feature_names: list[str],
    feature_values: list[float],
    feature_importances: list[float],
    feature_means: list[float],
    feature_stds: list[float],
    prediction_direction: str = "BUY",
    top_n: int = 10,
) -> FeatureAttribution:
    """Compute attribution without SHAP — uses feature importance × deviation.

    This is a lightweight proxy: contribution ≈ importance × (value - mean) / std.
    Features that are both important AND unusual get the highest attribution.

    Args:
        feature_names: Names of features
        feature_values: Current feature values
        feature_importances: Model feature importance scores (0-1)
        feature_means: Historical mean of each feature
        feature_stds: Historical std of each feature
        prediction_direction: "BUY" or "SELL"
        top_n: Number of top contributions to include

    Returns:
        FeatureAttribution with contributions sorted by importance
    """
    n = len(feature_names)
    if not (n == len(feature_values) == len(feature_importances) == len(feature_means) == len(feature_stds)):
        raise ValueError("All input lists must have the same length")

    contributions = []
    for i in range(n):
        std = feature_stds[i] if feature_stds[i] > 1e-10 else 1.0
        deviation = (feature_values[i] - feature_means[i]) / std
        contribution = feature_importances[i] * deviation

        # Flip sign for SELL direction
        if prediction_direction == "SELL":
            contribution = -contribution

        # Compute percentile (approximate from z-score)
        percentile = _z_to_percentile(deviation)

        contributions.append(
            FeatureContribution(
                feature_name=feature_names[i],
                contribution=contribution,
                value=feature_values[i],
                percentile=percentile,
            )
        )

    # Sort by absolute contribution and take top N
    contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
    contributions = contributions[:top_n]

    attribution = FeatureAttribution(
        contributions=contributions,
        method="simple_diff",
        prediction_direction=prediction_direction,
    )
    attribution._compute_derived()
    return attribution


def compute_shap_attribution(
    model: object,
    feature_names: list[str],
    feature_values: list[float],
    background_data: Optional[np.ndarray] = None,
    prediction_direction: str = "BUY",
    top_n: int = 10,
) -> FeatureAttribution:
    """Compute SHAP-based attribution if shap is installed.

    Falls back to simple_diff method if SHAP is not available.

    Args:
        model: Trained model object (must support predict/predict_proba)
        feature_names: Feature names
        feature_values: Current feature values
        background_data: Background dataset for SHAP (samples × features)
        prediction_direction: "BUY" or "SELL"
        top_n: Number of top contributions to include

    Returns:
        FeatureAttribution with SHAP values
    """
    try:
        import shap  # type: ignore
    except ImportError:
        logger.warning(
            "SHAP not installed. Falling back to simple_diff method. "
            "Install with: pip install shap"
        )
        # Fallback: use model feature importances if available
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            importances = [1.0 / len(feature_names)] * len(feature_names)

        means = [0.0] * len(feature_names)
        stds = [1.0] * len(feature_names)
        if background_data is not None and len(background_data) > 0:
            means = list(np.mean(background_data, axis=0))
            stds = list(np.std(background_data, axis=0))

        return compute_simple_attribution(
            feature_names=feature_names,
            feature_values=feature_values,
            feature_importances=list(importances),
            feature_means=means,
            feature_stds=stds,
            prediction_direction=prediction_direction,
            top_n=top_n,
        )

    # SHAP path
    input_array = np.array(feature_values).reshape(1, -1)

    if background_data is None:
        background_data = input_array  # Self-reference (not ideal, but functional)

    explainer = shap.Explainer(model, background_data)
    shap_values = explainer(input_array)

    # Extract SHAP values for the positive class
    values = shap_values.values[0]
    if len(values.shape) > 1:
        values = values[:, 1]  # Binary classification: take positive class

    contributions = []
    for i, name in enumerate(feature_names):
        contribution = float(values[i])
        if prediction_direction == "SELL":
            contribution = -contribution

        # Percentile from background data
        percentile = 50.0
        if background_data is not None and background_data.shape[0] > 1:
            col = background_data[:, i]
            percentile = float(np.mean(col <= feature_values[i]) * 100)

        contributions.append(
            FeatureContribution(
                feature_name=name,
                contribution=contribution,
                value=feature_values[i],
                percentile=percentile,
            )
        )

    contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
    contributions = contributions[:top_n]

    attribution = FeatureAttribution(
        contributions=contributions,
        method="shap",
        prediction_direction=prediction_direction,
    )
    attribution._compute_derived()
    return attribution


def _z_to_percentile(z: float) -> float:
    """Approximate percentile from z-score using logistic approximation."""
    # Logistic approximation of normal CDF
    percentile = 100.0 / (1.0 + np.exp(-1.7 * z))
    return float(np.clip(percentile, 0.0, 100.0))

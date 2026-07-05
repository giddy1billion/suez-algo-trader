"""Uncertainty estimation — confidence in the confidence itself.

Key insight: P(up)=0.74 ± 0.18 is very different from P(up)=0.74 ± 0.03.
This module quantifies how much we should trust the point prediction.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyEstimate:
    """Quantifies how confident we are in the confidence."""

    point_estimate: float  # The prediction (e.g., 0.74)
    lower_bound: float  # Confidence interval lower
    upper_bound: float  # Confidence interval upper
    interval_width: float  # upper - lower (uncertainty magnitude)
    estimation_method: str  # "ensemble_variance" | "bootstrap" | "naive"
    ensemble_disagreement: float  # Variance across ensemble members
    sample_size: int  # How many observations support this estimate
    epistemic_uncertainty: float  # Model uncertainty (reducible with more data)
    aleatoric_uncertainty: float  # Data uncertainty (irreducible)

    @property
    def uncertainty_score(self) -> float:
        """Single scalar uncertainty: 0=certain, 1=totally uncertain.

        Combines interval width and decomposed uncertainties into a
        normalized score suitable for downstream sizing decisions.
        """
        # Weighted combination of different uncertainty sources
        width_component = min(self.interval_width / 0.5, 1.0)  # Normalize: 0.5 width → 1.0
        epistemic_component = min(self.epistemic_uncertainty, 1.0)
        aleatoric_component = min(self.aleatoric_uncertainty, 1.0)

        score = (
            0.5 * width_component
            + 0.3 * epistemic_component
            + 0.2 * aleatoric_component
        )
        return min(1.0, max(0.0, score))

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "point_estimate": round(self.point_estimate, 4),
            "lower_bound": round(self.lower_bound, 4),
            "upper_bound": round(self.upper_bound, 4),
            "interval_width": round(self.interval_width, 4),
            "estimation_method": self.estimation_method,
            "ensemble_disagreement": round(self.ensemble_disagreement, 6),
            "sample_size": self.sample_size,
            "epistemic_uncertainty": round(self.epistemic_uncertainty, 4),
            "aleatoric_uncertainty": round(self.aleatoric_uncertainty, 4),
            "uncertainty_score": round(self.uncertainty_score, 4),
        }


class UncertaintyEstimator:
    """Estimates uncertainty in model predictions using multiple methods.

    Supports:
    - ensemble_variance: Disagreement across ensemble members (default)
    - bootstrap: Resampling-based confidence intervals
    - naive: Simple heuristic based on proximity to 0.5

    High uncertainty feeds into SizingEngine to reduce position size.
    """

    def __init__(
        self,
        method: str = "ensemble_variance",
        confidence_level: float = 0.95,
        min_ensemble_size: int = 3,
    ) -> None:
        """Initialize the estimator.

        Args:
            method: Estimation method ("ensemble_variance", "bootstrap", "naive")
            confidence_level: Confidence level for intervals (default 95%)
            min_ensemble_size: Minimum ensemble members for valid estimate
        """
        valid_methods = {"ensemble_variance", "bootstrap", "naive"}
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")

        self.method = method
        self.confidence_level = confidence_level
        self.min_ensemble_size = min_ensemble_size
        self._z_score = self._compute_z_score(confidence_level)

    def estimate(
        self,
        predictions: list[float],
        sample_size: int = 100,
        feature_importances: Optional[list[float]] = None,
    ) -> UncertaintyEstimate:
        """Compute uncertainty estimate from model predictions.

        Args:
            predictions: List of predictions (from ensemble members, bootstrap, etc.)
            sample_size: Number of training observations supporting this prediction
            feature_importances: Optional feature importance scores for bootstrap method

        Returns:
            UncertaintyEstimate with full uncertainty decomposition
        """
        if not predictions:
            raise ValueError("predictions list cannot be empty")

        predictions_arr = np.array(predictions, dtype=np.float64)

        if self.method == "ensemble_variance":
            return self._estimate_ensemble(predictions_arr, sample_size)
        elif self.method == "bootstrap":
            return self._estimate_bootstrap(predictions_arr, sample_size, feature_importances)
        else:
            return self._estimate_naive(predictions_arr, sample_size)

    def _estimate_ensemble(
        self, predictions: np.ndarray, sample_size: int
    ) -> UncertaintyEstimate:
        """Estimate uncertainty from ensemble member disagreement.

        If N models predict [0.72, 0.76, 0.71, 0.78, 0.73], the mean is 0.74
        and the std captures how much they disagree.
        """
        n = len(predictions)
        if n < self.min_ensemble_size:
            logger.warning(
                f"Ensemble size {n} < minimum {self.min_ensemble_size}. "
                "Estimate may be unreliable."
            )

        mean = float(np.mean(predictions))
        std = float(np.std(predictions, ddof=1)) if n > 1 else 0.0

        # Standard error of the mean
        se = std / math.sqrt(n) if n > 0 else std

        # Confidence interval
        margin = self._z_score * se
        lower = max(0.0, mean - margin)
        upper = min(1.0, mean + margin)

        # Epistemic uncertainty: model disagreement (reducible with more models/data)
        epistemic = std  # Direct measure of model uncertainty

        # Aleatoric uncertainty: estimate from prediction proximity to 0.5
        # Predictions near 0.5 inherently have higher data uncertainty
        aleatoric = 2.0 * min(mean, 1.0 - mean) * 0.5  # Peaks at 0.5

        return UncertaintyEstimate(
            point_estimate=mean,
            lower_bound=lower,
            upper_bound=upper,
            interval_width=upper - lower,
            estimation_method="ensemble_variance",
            ensemble_disagreement=float(np.var(predictions, ddof=1)) if n > 1 else 0.0,
            sample_size=sample_size,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
        )

    def _estimate_bootstrap(
        self,
        predictions: np.ndarray,
        sample_size: int,
        feature_importances: Optional[list[float]] = None,
    ) -> UncertaintyEstimate:
        """Estimate uncertainty via bootstrap resampling.

        Resamples the predictions (or feature importances) to construct
        confidence intervals via percentile method.
        """
        n_bootstrap = 1000
        n = len(predictions)
        mean = float(np.mean(predictions))

        # Bootstrap resampling
        rng = np.random.default_rng(seed=42)
        boot_means = np.zeros(n_bootstrap)

        data = predictions
        if feature_importances is not None and len(feature_importances) > 0:
            data = np.array(feature_importances, dtype=np.float64)

        for i in range(n_bootstrap):
            resample = rng.choice(data, size=len(data), replace=True)
            boot_means[i] = np.mean(resample)

        # Percentile confidence interval
        alpha = (1.0 - self.confidence_level) / 2.0
        lower = float(np.percentile(boot_means, alpha * 100))
        upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))

        # Clamp to valid range
        lower = max(0.0, lower)
        upper = min(1.0, upper)

        # Decompose uncertainty
        boot_std = float(np.std(boot_means))
        epistemic = boot_std  # Sampling uncertainty
        aleatoric = 2.0 * min(mean, 1.0 - mean) * 0.5

        return UncertaintyEstimate(
            point_estimate=mean,
            lower_bound=lower,
            upper_bound=upper,
            interval_width=upper - lower,
            estimation_method="bootstrap",
            ensemble_disagreement=float(np.var(predictions, ddof=1)) if n > 1 else 0.0,
            sample_size=sample_size,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
        )

    def _estimate_naive(
        self, predictions: np.ndarray, sample_size: int
    ) -> UncertaintyEstimate:
        """Simple heuristic uncertainty estimate.

        Uses prediction proximity to 0.5 and sample size as proxies.
        Useful when no ensemble or bootstrap is available.
        """
        mean = float(np.mean(predictions))
        n = len(predictions)

        # Heuristic: uncertainty is higher near 0.5 and with fewer samples
        distance_from_edge = min(mean, 1.0 - mean)
        sample_factor = 1.0 / math.sqrt(max(sample_size, 1))

        # Width estimate based on heuristics
        width = distance_from_edge * 0.5 + sample_factor * 0.3
        width = min(width, 0.5)

        lower = max(0.0, mean - width / 2)
        upper = min(1.0, mean + width / 2)

        epistemic = sample_factor * 0.5
        aleatoric = distance_from_edge * 0.5

        std = float(np.std(predictions, ddof=1)) if n > 1 else 0.0

        return UncertaintyEstimate(
            point_estimate=mean,
            lower_bound=lower,
            upper_bound=upper,
            interval_width=upper - lower,
            estimation_method="naive",
            ensemble_disagreement=float(np.var(predictions, ddof=1)) if n > 1 else 0.0,
            sample_size=sample_size,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
        )

    @staticmethod
    def _compute_z_score(confidence_level: float) -> float:
        """Compute z-score for given confidence level using inverse normal approximation."""
        # Common values
        z_table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        if confidence_level in z_table:
            return z_table[confidence_level]

        # Rational approximation (Abramowitz & Stegun 26.2.23)
        p = (1.0 + confidence_level) / 2.0
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
        return z

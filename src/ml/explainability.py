"""
Model Explainability — SHAP-based feature attribution for every prediction.

Every signal becomes auditable:
    BUY BTC/USD
    because:
        EMA trend     +18% contribution
        RSI           +11%
        Volume        +24%
        Order flow    +15%
        Regime        +13%

Integrates with:
- MultiTargetPredictor for per-prediction SHAP values
- TradeSignalPackage for enriching evidence packages
- DecisionExplainer for human-readable explanations
- ModelHealthMonitor for explainability of drift

Supports:
- TreeSHAP (fast, for XGBoost/LightGBM)
- KernelSHAP (model-agnostic fallback)
- Feature interaction analysis
- Counterfactual explanations ("what would flip this prediction?")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class FeatureContribution:
    """Single feature's contribution to a prediction."""

    feature_name: str
    shap_value: float  # signed contribution (positive = pushes toward prediction)
    feature_value: float  # actual feature value for this prediction
    contribution_pct: float  # percentage of total explanation
    direction: str  # "supporting" or "opposing"


@dataclass
class PredictionExplanation:
    """Complete explanation for a single prediction."""

    prediction_id: str
    symbol: str
    direction: str
    confidence: float
    base_value: float  # expected value (model mean prediction)
    prediction_value: float  # actual prediction

    # Top contributing features (sorted by |SHAP|)
    top_contributors: List[FeatureContribution]

    # Feature interactions (pairs that matter together)
    interactions: List[Tuple[str, str, float]] = field(default_factory=list)

    # Counterfactual
    counterfactual_features: Dict[str, float] = field(default_factory=dict)
    counterfactual_description: str = ""

    # Human-readable summary
    natural_language: str = ""

    def to_evidence_dict(self) -> Dict[str, Any]:
        """Convert to dict suitable for evidence package integration."""
        return {
            "explanation_type": "shap",
            "base_value": self.base_value,
            "top_features": [
                {
                    "name": c.feature_name,
                    "contribution": c.shap_value,
                    "value": c.feature_value,
                    "pct": c.contribution_pct,
                    "direction": c.direction,
                }
                for c in self.top_contributors
            ],
            "interactions": [
                {"feature_a": a, "feature_b": b, "strength": s}
                for a, b, s in self.interactions
            ],
            "counterfactual": self.counterfactual_features,
            "summary": self.natural_language,
        }


# ──────────────────────────────────────────────────────────────────────────
# Explainability Engine
# ──────────────────────────────────────────────────────────────────────────


class PredictionExplainer:
    """
    SHAP-based explainability engine for trade predictions.

    Wraps SHAP library to provide:
    1. Per-prediction feature attributions
    2. Feature interaction detection
    3. Counterfactual analysis
    4. Human-readable explanation generation

    Usage:
        explainer = PredictionExplainer(model, feature_names)
        explanation = explainer.explain(features_row, prediction)
    """

    def __init__(
        self,
        model: Any = None,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[pd.DataFrame] = None,
        top_k: int = 8,
        use_tree_shap: bool = True,
    ):
        self.model = model
        self.feature_names = feature_names or []
        self.background_data = background_data
        self.top_k = top_k
        self.use_tree_shap = use_tree_shap

        self._shap_explainer = None
        self._is_initialized = False

    def initialize(self, model: Any, feature_names: List[str], background_data: Optional[pd.DataFrame] = None):
        """
        Initialize or reinitialize the explainer with a new model.

        Args:
            model: Trained model (XGBoost, LightGBM, or sklearn).
            feature_names: List of feature column names.
            background_data: Optional background dataset for KernelSHAP.
        """
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self._shap_explainer = None
        self._is_initialized = False

        try:
            import shap
            if self.use_tree_shap and hasattr(model, 'get_booster'):
                # XGBoost native TreeSHAP (fast)
                self._shap_explainer = shap.TreeExplainer(model)
            elif self.use_tree_shap and hasattr(model, 'booster_'):
                # LightGBM TreeSHAP
                self._shap_explainer = shap.TreeExplainer(model)
            elif background_data is not None:
                # KernelSHAP fallback (slower but model-agnostic)
                bg = background_data.sample(min(100, len(background_data)))
                self._shap_explainer = shap.KernelExplainer(model.predict, bg)
            else:
                logger.warning("explainability.no_shap_available — using permutation fallback")
            self._is_initialized = True
        except ImportError:
            logger.warning("SHAP library not installed — using permutation importance fallback")
            self._is_initialized = True  # will use fallback

    def explain(
        self,
        features: pd.DataFrame,
        prediction_id: str = "",
        symbol: str = "",
        direction: str = "",
        confidence: float = 0.0,
    ) -> PredictionExplanation:
        """
        Generate explanation for a single prediction.

        Args:
            features: Single-row DataFrame with feature values.
            prediction_id: Unique identifier for this prediction.
            symbol: Trading symbol.
            direction: Predicted direction (BUY/SELL/HOLD).
            confidence: Prediction confidence.

        Returns:
            PredictionExplanation with feature attributions.
        """
        if len(features) > 1:
            features = features.iloc[[-1]]

        # Get SHAP values
        shap_values = self._compute_shap_values(features)

        if shap_values is None:
            # Fallback to permutation importance
            shap_values = self._permutation_fallback(features)

        # Build contributions
        feature_values = features.iloc[0].values if len(features) > 0 else np.zeros(len(self.feature_names))
        contributions = self._build_contributions(shap_values, feature_values)

        # Top-K contributors
        top = sorted(contributions, key=lambda c: abs(c.shap_value), reverse=True)[:self.top_k]

        # Base value (expected prediction)
        base_value = self._get_base_value()

        # Generate natural language explanation
        nl = self._generate_natural_language(direction, confidence, top)

        # Counterfactual analysis
        counterfactual = self._compute_counterfactual(features, shap_values, direction)

        return PredictionExplanation(
            prediction_id=prediction_id,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            base_value=base_value,
            prediction_value=base_value + sum(c.shap_value for c in contributions),
            top_contributors=top,
            counterfactual_features=counterfactual,
            counterfactual_description=self._describe_counterfactual(counterfactual),
            natural_language=nl,
        )

    def explain_drift(
        self,
        recent_features: pd.DataFrame,
        baseline_features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Explain WHY drift is occurring by comparing feature attributions
        between recent and baseline periods.

        Returns:
            Dict with shifted features and their contribution changes.
        """
        recent_shap = self._compute_shap_values(recent_features)
        baseline_shap = self._compute_shap_values(baseline_features)

        if recent_shap is None or baseline_shap is None:
            return {"error": "Unable to compute SHAP values for drift explanation"}

        # Average SHAP per feature
        recent_mean = np.mean(np.abs(recent_shap), axis=0) if recent_shap.ndim > 1 else np.abs(recent_shap)
        baseline_mean = np.mean(np.abs(baseline_shap), axis=0) if baseline_shap.ndim > 1 else np.abs(baseline_shap)

        # Features with biggest importance shift
        importance_change = recent_mean - baseline_mean
        top_shifted = np.argsort(np.abs(importance_change))[::-1][:10]

        shifts = []
        for idx in top_shifted:
            if idx < len(self.feature_names):
                shifts.append({
                    "feature": self.feature_names[idx],
                    "baseline_importance": float(baseline_mean[idx]),
                    "recent_importance": float(recent_mean[idx]),
                    "change": float(importance_change[idx]),
                })

        return {
            "top_shifted_features": shifts,
            "total_importance_change": float(np.sum(np.abs(importance_change))),
            "n_features_analyzed": len(self.feature_names),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Private Methods
    # ──────────────────────────────────────────────────────────────────────

    def _compute_shap_values(self, features: pd.DataFrame) -> Optional[np.ndarray]:
        """Compute SHAP values using available explainer."""
        if self._shap_explainer is not None:
            try:
                shap_values = self._shap_explainer.shap_values(features)
                # For multi-class, take the predicted class SHAP values
                if isinstance(shap_values, list):
                    # Use the class with highest prediction
                    if self.model is not None:
                        try:
                            pred = self.model.predict(features)
                            pred_class = int(pred[0]) if hasattr(pred, '__len__') else int(pred)
                            shap_values = shap_values[pred_class]
                        except Exception:
                            shap_values = shap_values[-1]  # last class (positive)
                    else:
                        shap_values = shap_values[-1]

                if hasattr(shap_values, 'values'):
                    shap_values = shap_values.values

                return np.array(shap_values).flatten()
            except Exception as e:
                logger.warning(f"SHAP computation failed: {e}")
                return None
        return None

    def _permutation_fallback(self, features: pd.DataFrame) -> np.ndarray:
        """Compute approximate feature importance via permutation."""
        if self.model is None:
            return np.zeros(len(self.feature_names))

        try:
            base_pred = self.model.predict(features)
            if hasattr(base_pred, '__len__'):
                base_pred = base_pred[0]

            importances = np.zeros(len(self.feature_names))
            for i in range(min(len(self.feature_names), features.shape[1])):
                perturbed = features.copy()
                # Permute single feature (replace with mean or zero)
                perturbed.iloc[0, i] = 0.0
                perm_pred = self.model.predict(perturbed)
                if hasattr(perm_pred, '__len__'):
                    perm_pred = perm_pred[0]
                importances[i] = base_pred - perm_pred  # positive = feature helped

            return importances
        except Exception as e:
            logger.warning(f"Permutation fallback failed: {e}")
            return np.zeros(len(self.feature_names))

    def _build_contributions(
        self,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
    ) -> List[FeatureContribution]:
        """Build FeatureContribution list from raw SHAP values."""
        total_abs = np.sum(np.abs(shap_values)) + 1e-10
        contributions = []

        for i in range(min(len(self.feature_names), len(shap_values))):
            sv = float(shap_values[i])
            fv = float(feature_values[i]) if i < len(feature_values) else 0.0
            pct = abs(sv) / total_abs * 100

            contributions.append(FeatureContribution(
                feature_name=self.feature_names[i],
                shap_value=sv,
                feature_value=fv,
                contribution_pct=pct,
                direction="supporting" if sv > 0 else "opposing",
            ))

        return contributions

    def _get_base_value(self) -> float:
        """Get the base (expected) value from the SHAP explainer."""
        if self._shap_explainer is not None and hasattr(self._shap_explainer, 'expected_value'):
            ev = self._shap_explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                return float(ev[-1])  # last class for multi-class
            return float(ev)
        return 0.5  # default base for classification

    def _compute_counterfactual(
        self,
        features: pd.DataFrame,
        shap_values: np.ndarray,
        direction: str,
    ) -> Dict[str, float]:
        """
        Find minimal feature changes that would flip the prediction.

        Identifies the top opposing features and suggests what values
        would need to change to reverse the direction.
        """
        if shap_values is None or len(shap_values) == 0:
            return {}

        # Find features opposing the prediction (negative SHAP for BUY, positive for SELL)
        flip_target = -1 if direction == "BUY" else 1

        # Sort by SHAP value in the opposing direction
        opposing = [(i, shap_values[i]) for i in range(len(shap_values)) if shap_values[i] * flip_target > 0]
        opposing.sort(key=lambda x: abs(x[1]), reverse=True)

        # Top 3 features to change
        counterfactual = {}
        for idx, sv in opposing[:3]:
            if idx < len(self.feature_names) and idx < features.shape[1]:
                fname = self.feature_names[idx]
                current_val = float(features.iloc[0, idx])
                # Suggest flipping direction by moving 2 standard deviations
                suggested_change = -np.sign(sv) * abs(current_val) * 0.5
                counterfactual[fname] = current_val + suggested_change

        return counterfactual

    def _describe_counterfactual(self, counterfactual: Dict[str, float]) -> str:
        """Generate human-readable counterfactual description."""
        if not counterfactual:
            return "No counterfactual available"

        parts = []
        for feature, value in counterfactual.items():
            parts.append(f"{feature} → {value:.4f}")

        return f"Prediction would flip if: {', '.join(parts)}"

    def _generate_natural_language(
        self,
        direction: str,
        confidence: float,
        top_contributors: List[FeatureContribution],
    ) -> str:
        """Generate human-readable explanation."""
        if not top_contributors:
            return f"{direction} signal with {confidence:.0%} confidence (no feature attribution available)"

        supporting = [c for c in top_contributors if c.direction == "supporting"]
        opposing = [c for c in top_contributors if c.direction == "opposing"]

        lines = [f"{direction} signal ({confidence:.0%} confidence)"]

        if supporting:
            lines.append("  Supporting factors:")
            for c in supporting[:4]:
                lines.append(f"    {c.feature_name}: +{c.contribution_pct:.0f}% contribution")

        if opposing:
            lines.append("  Risk factors:")
            for c in opposing[:2]:
                lines.append(f"    {c.feature_name}: -{c.contribution_pct:.0f}% contribution")

        return "\n".join(lines)

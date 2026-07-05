"""
Multi-Target Prediction — Single inference producing direction, return, duration,
risk, and sizing recommendations.

Instead of:
    Model → BUY/SELL

We predict:
    Model → {
        direction: BUY,
        probability: 0.89,
        expected_return: +5.2%,
        holding_time_hours: 18-36,
        max_drawdown: -1.7%,
        take_profit: +5.2%,
        stop_loss: -1.7%,
        confidence: 0.89,
        risk_score: A,
        position_size_pct: 3.2%,
    }

Architecture:
    Shared Feature Encoder (XGBoost feature importance → top-K features)
        │
        ├── Direction Head (classification: up/flat/down)
        ├── Return Magnitude Head (regression: expected_return_pct)
        ├── Duration Head (regression: expected_holding_hours)
        ├── Risk Head (regression: max_adverse_excursion)
        └── Confidence Head (calibrated probability)

Each head is a separate XGBoost model trained on the same features but
different targets, enabling independent optimization per target.

Integrates with:
- TrainingPipeline for feature engineering
- FeatureStore for feature versioning
- TradeSignalPackage for evidence-rich output
- ModelRegistry for versioned artifact storage
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ml.label_encoder import DirectionEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class MultiTargetPrediction:
    """Complete prediction from multi-target inference."""

    # Direction
    direction: str  # "BUY", "SELL", "HOLD"
    direction_probability: float  # P(direction correct)

    # Return expectations
    expected_return_pct: float
    expected_return_std: float  # uncertainty
    upside_potential_pct: float  # P75 return
    downside_risk_pct: float  # P25 return (negative)

    # Time horizon
    expected_holding_hours: float
    time_to_tp_hours: float  # expected time to take-profit
    time_to_sl_hours: float  # expected time to stop-loss

    # Risk
    max_adverse_excursion_pct: float  # expected worst drawdown
    risk_reward_ratio: float
    probability_tp: float  # P(reaching take profit)
    probability_sl: float  # P(hitting stop loss)
    probability_timeout: float  # P(time-based exit)

    # Confidence & calibration
    confidence: float  # overall signal confidence (calibrated)
    prediction_uncertainty: float  # model uncertainty estimate

    # Position sizing recommendation
    recommended_position_pct: float  # of portfolio
    kelly_fraction: float

    # Risk grade
    risk_grade: str  # A, B, C, D, F

    # Suggested levels
    suggested_tp_pct: float
    suggested_sl_pct: float
    suggested_trailing_stop_pct: float

    # Metadata
    model_version: str = ""
    inference_time_ms: float = 0.0
    feature_set_version: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for signal package integration."""
        return {
            "direction": self.direction,
            "direction_probability": self.direction_probability,
            "expected_return_pct": self.expected_return_pct,
            "expected_return_std": self.expected_return_std,
            "upside_potential_pct": self.upside_potential_pct,
            "downside_risk_pct": self.downside_risk_pct,
            "expected_holding_hours": self.expected_holding_hours,
            "time_to_tp_hours": self.time_to_tp_hours,
            "time_to_sl_hours": self.time_to_sl_hours,
            "max_adverse_excursion_pct": self.max_adverse_excursion_pct,
            "risk_reward_ratio": self.risk_reward_ratio,
            "probability_tp": self.probability_tp,
            "probability_sl": self.probability_sl,
            "probability_timeout": self.probability_timeout,
            "confidence": self.confidence,
            "prediction_uncertainty": self.prediction_uncertainty,
            "recommended_position_pct": self.recommended_position_pct,
            "kelly_fraction": self.kelly_fraction,
            "risk_grade": self.risk_grade,
            "suggested_tp_pct": self.suggested_tp_pct,
            "suggested_sl_pct": self.suggested_sl_pct,
            "suggested_trailing_stop_pct": self.suggested_trailing_stop_pct,
            "model_version": self.model_version,
            "inference_time_ms": self.inference_time_ms,
        }


@dataclass
class TargetDefinition:
    """Definition of a single prediction target."""

    name: str
    target_type: str  # "classification" or "regression"
    column_name: str  # name in training dataframe
    n_classes: int = 0  # for classification
    clip_range: Optional[Tuple[float, float]] = None  # for regression
    transform: Optional[str] = None  # "log", "sqrt", etc.


# ──────────────────────────────────────────────────────────────────────────
# Target Engineering
# ──────────────────────────────────────────────────────────────────────────


MULTI_TARGETS = [
    TargetDefinition(
        name="direction",
        target_type="classification",
        column_name="target_direction",
        n_classes=3,  # -1, 0, +1
    ),
    TargetDefinition(
        name="return_magnitude",
        target_type="regression",
        column_name="target_return_pct",
        clip_range=(-0.20, 0.20),  # clip extreme returns
    ),
    TargetDefinition(
        name="holding_duration",
        target_type="regression",
        column_name="target_holding_hours",
        clip_range=(0.5, 720.0),  # 30 min to 30 days
        transform="log",
    ),
    TargetDefinition(
        name="max_adverse_excursion",
        target_type="regression",
        column_name="target_mae_pct",
        clip_range=(0.0, 0.20),
    ),
    TargetDefinition(
        name="max_favorable_excursion",
        target_type="regression",
        column_name="target_mfe_pct",
        clip_range=(0.0, 0.30),
    ),
]


def engineer_multi_targets(
    df: pd.DataFrame,
    forward_bars: int = 5,
    direction_threshold: float = 0.005,
    holding_period_bars: int = 20,
) -> pd.DataFrame:
    """
    Engineer multiple prediction targets from OHLCV data.

    Computes:
    1. Direction: +1 (up), 0 (flat), -1 (down) based on forward return
    2. Return magnitude: actual forward return %
    3. Holding duration: bars until signal invalidation or TP/SL
    4. Max adverse excursion (MAE): worst drawdown during holding period
    5. Max favorable excursion (MFE): best gain during holding period

    Args:
        df: OHLCV DataFrame with 'close', 'high', 'low' columns.
        forward_bars: Look-ahead window for direction target.
        direction_threshold: Min return to classify as up/down (default 0.5%).
        holding_period_bars: Max bars to look ahead for MAE/MFE.

    Returns:
        DataFrame with target columns appended.
    """
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    n = len(close)

    # Target arrays
    target_direction = np.zeros(n)
    target_return = np.zeros(n)
    target_holding = np.full(n, np.nan)
    target_mae = np.zeros(n)
    target_mfe = np.zeros(n)

    for i in range(n - forward_bars):
        entry = close[i]
        if entry <= 0:
            continue

        # Forward return
        future_close = close[i + forward_bars]
        fwd_return = (future_close - entry) / entry
        target_return[i] = fwd_return

        # Direction
        if fwd_return > direction_threshold:
            target_direction[i] = 1
        elif fwd_return < -direction_threshold:
            target_direction[i] = -1
        else:
            target_direction[i] = 0

        # MAE/MFE over holding period
        end_idx = min(i + holding_period_bars, n)
        future_highs = high[i + 1:end_idx]
        future_lows = low[i + 1:end_idx]

        if len(future_highs) > 0:
            max_high = np.max(future_highs)
            min_low = np.min(future_lows)
            target_mfe[i] = (max_high - entry) / entry
            target_mae[i] = (entry - min_low) / entry  # positive = adverse

        # Holding duration: bars until direction reverses or threshold hit
        tp_level = entry * (1 + direction_threshold * 3)
        sl_level = entry * (1 - direction_threshold * 2)

        holding = float(forward_bars)
        for j in range(i + 1, end_idx):
            if high[j] >= tp_level or low[j] <= sl_level:
                holding = float(j - i)
                break
        target_holding[i] = holding

    # Append to dataframe
    result = df.copy()
    result["target_direction"] = target_direction
    result["target_return_pct"] = target_return
    result["target_holding_hours"] = target_holding  # bars → hours depends on timeframe
    result["target_mae_pct"] = target_mae
    result["target_mfe_pct"] = target_mfe

    # Drop rows where targets are NaN (end of series)
    result = result.dropna(subset=["target_holding_hours"])

    return result


# ──────────────────────────────────────────────────────────────────────────
# Multi-Target Model
# ──────────────────────────────────────────────────────────────────────────


class MultiTargetPredictor:
    """
    Multi-head prediction model producing comprehensive trade evidence.

    Each prediction target has its own XGBoost model trained on shared features.
    This allows independent hyperparameter tuning per target while maintaining
    a unified inference interface.

    Usage:
        predictor = MultiTargetPredictor()
        predictor.train(features_df, targets_df)
        prediction = predictor.predict(current_features)
    """

    def __init__(
        self,
        model_version: str = "multi_v001",
        feature_set_version: str = "fs_001",
        calibrate: bool = True,
    ):
        self.model_version = model_version
        self.feature_set_version = feature_set_version
        self.calibrate = calibrate

        # Individual models per target
        self._models: Dict[str, Any] = {}
        self._feature_names: List[str] = []
        self._is_trained: bool = False

        # Calibration (Platt scaling for direction probabilities)
        self._calibration_a: float = 1.0
        self._calibration_b: float = 0.0

        # Historical statistics for uncertainty estimation
        self._train_return_std: float = 0.02
        self._train_holding_mean: float = 24.0
        self._train_mae_mean: float = 0.02

    def train(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Train all prediction heads on shared features.

        Args:
            features: Feature matrix (T × F).
            targets: Multi-target matrix with columns matching MULTI_TARGETS.
            validation_split: Fraction for validation (time-ordered).

        Returns:
            Dict of per-target training metrics.
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("XGBoost not available, using sklearn GradientBoosting fallback")
            return self._train_sklearn_fallback(features, targets, validation_split)

        self._feature_names = list(features.columns)
        n = len(features)
        split_idx = int(n * (1 - validation_split))

        X_train = features.iloc[:split_idx]
        X_val = features.iloc[split_idx:]

        metrics = {}

        for target_def in MULTI_TARGETS:
            col = target_def.column_name
            if col not in targets.columns:
                logger.warning(f"Target column '{col}' not found, skipping {target_def.name}")
                continue

            y_train = targets[col].iloc[:split_idx].values
            y_val = targets[col].iloc[split_idx:].values

            # Apply transform
            if target_def.transform == "log":
                y_train = np.log1p(np.clip(y_train, 0, None))
                y_val = np.log1p(np.clip(y_val, 0, None))

            # Clip range
            if target_def.clip_range:
                lo, hi = target_def.clip_range
                y_train = np.clip(y_train, lo, hi)
                y_val = np.clip(y_val, lo, hi)

            if target_def.target_type == "classification":
                # Multi-class classification
                params = {
                    "objective": "multi:softprob",
                    "num_class": target_def.n_classes,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "n_estimators": 200,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 5,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "eval_metric": "mlogloss",
                    "verbosity": 0,
                }
                # Encode direction labels for classifier
                y_train_mapped = DirectionEncoder.encode(y_train)
                y_val_mapped = DirectionEncoder.encode(y_val)

                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train, y_train_mapped,
                    eval_set=[(X_val, y_val_mapped)],
                    verbose=False,
                )
                val_pred = model.predict(X_val)
                accuracy = float(np.mean(val_pred == y_val_mapped))
                metrics[target_def.name] = {"accuracy": accuracy, "type": "classification"}

            else:
                # Regression
                params = {
                    "objective": "reg:squarederror",
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "n_estimators": 150,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 5,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "verbosity": 0,
                }
                model = xgb.XGBRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                val_pred = model.predict(X_val)
                mae = float(np.mean(np.abs(val_pred - y_val)))
                r2 = float(1 - np.sum((y_val - val_pred)**2) / np.sum((y_val - y_val.mean())**2))
                metrics[target_def.name] = {"mae": mae, "r2": r2, "type": "regression"}

            self._models[target_def.name] = model

        # Store training statistics
        if "target_return_pct" in targets.columns:
            self._train_return_std = float(targets["target_return_pct"].std())
        if "target_holding_hours" in targets.columns:
            self._train_holding_mean = float(targets["target_holding_hours"].mean())
        if "target_mae_pct" in targets.columns:
            self._train_mae_mean = float(targets["target_mae_pct"].mean())

        self._is_trained = True
        logger.info(
            "multi_target.trained",
            model_version=self.model_version,
            n_samples=n,
            targets=list(metrics.keys()),
        )
        return metrics

    def predict(self, features: pd.DataFrame) -> MultiTargetPrediction:
        """
        Run multi-target inference on current market features.

        Args:
            features: Single-row DataFrame (or last row used) with feature columns.

        Returns:
            MultiTargetPrediction with all targets populated.
        """
        start_time = time.time()

        if not self._is_trained:
            return self._default_prediction()

        # Ensure single-row input
        if len(features) > 1:
            features = features.iloc[[-1]]

        X = features[self._feature_names] if self._feature_names else features

        # Direction prediction
        direction_str = "HOLD"
        direction_prob = 0.5
        if "direction" in self._models:
            model = self._models["direction"]
            proba = model.predict_proba(X)[0]  # [P(down), P(flat), P(up)]
            pred_class = int(np.argmax(proba))
            direction_str = DirectionEncoder.CLASS_NAMES.get(pred_class, "HOLD")
            direction_prob = float(proba[pred_class])

        # Return magnitude prediction
        expected_return = 0.0
        if "return_magnitude" in self._models:
            expected_return = float(self._models["return_magnitude"].predict(X)[0])

        # Duration prediction
        expected_holding = self._train_holding_mean
        if "holding_duration" in self._models:
            raw_pred = float(self._models["holding_duration"].predict(X)[0])
            expected_holding = float(np.expm1(raw_pred)) if raw_pred > 0 else self._train_holding_mean

        # MAE prediction
        mae_pct = self._train_mae_mean
        if "max_adverse_excursion" in self._models:
            mae_pct = float(self._models["max_adverse_excursion"].predict(X)[0])
            mae_pct = max(0.001, mae_pct)

        # MFE prediction
        mfe_pct = abs(expected_return) * 1.5
        if "max_favorable_excursion" in self._models:
            mfe_pct = float(self._models["max_favorable_excursion"].predict(X)[0])
            mfe_pct = max(0.001, mfe_pct)

        # Derived metrics
        risk_reward = mfe_pct / mae_pct if mae_pct > 0 else 1.0
        probability_tp = min(0.95, direction_prob * 0.85)
        probability_sl = min(0.95, 1 - probability_tp - 0.05)
        probability_timeout = max(0.0, 1 - probability_tp - probability_sl)

        # Kelly fraction
        kelly = (direction_prob * risk_reward - (1 - direction_prob)) / risk_reward if risk_reward > 0 else 0
        kelly = max(0, min(0.25, kelly))  # cap at 25%, floor at 0

        # Confidence (calibrated direction probability)
        confidence = self._calibrate_probability(direction_prob)

        # Position sizing (half-Kelly for safety)
        position_pct = kelly * 0.5 * 100  # as percentage

        # Risk grade
        risk_grade = self._compute_risk_grade(mae_pct, risk_reward, confidence)

        # Suggested levels
        suggested_tp = mfe_pct * 0.8  # conservative TP at 80% of expected MFE
        suggested_sl = mae_pct * 1.2  # SL slightly beyond expected MAE
        suggested_trailing = mae_pct * 0.6  # tighter trailing stop

        # Time estimates
        time_to_tp = expected_holding * 0.7
        time_to_sl = expected_holding * 0.4

        inference_ms = (time.time() - start_time) * 1000

        return MultiTargetPrediction(
            direction=direction_str,
            direction_probability=direction_prob,
            expected_return_pct=expected_return * 100,
            expected_return_std=self._train_return_std * 100,
            upside_potential_pct=mfe_pct * 100,
            downside_risk_pct=mae_pct * 100,
            expected_holding_hours=expected_holding,
            time_to_tp_hours=time_to_tp,
            time_to_sl_hours=time_to_sl,
            max_adverse_excursion_pct=mae_pct * 100,
            risk_reward_ratio=risk_reward,
            probability_tp=probability_tp,
            probability_sl=probability_sl,
            probability_timeout=probability_timeout,
            confidence=confidence,
            prediction_uncertainty=1 - confidence,
            recommended_position_pct=position_pct,
            kelly_fraction=kelly,
            risk_grade=risk_grade,
            suggested_tp_pct=suggested_tp * 100,
            suggested_sl_pct=suggested_sl * 100,
            suggested_trailing_stop_pct=suggested_trailing * 100,
            model_version=self.model_version,
            inference_time_ms=inference_ms,
            feature_set_version=self.feature_set_version,
        )

    def get_feature_importance(self, top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get feature importance per prediction head.

        Returns:
            Dict mapping target_name → [(feature_name, importance), ...]
        """
        importances = {}
        for name, model in self._models.items():
            try:
                imp = model.feature_importances_
                indices = np.argsort(imp)[::-1][:top_k]
                importances[name] = [
                    (self._feature_names[i], float(imp[i]))
                    for i in indices
                    if i < len(self._feature_names)
                ]
            except (AttributeError, IndexError):
                importances[name] = []
        return importances

    def _calibrate_probability(self, raw_prob: float) -> float:
        """Apply Platt scaling calibration to raw probability."""
        # Sigmoid calibration: P_cal = 1 / (1 + exp(A*f + B))
        calibrated = 1.0 / (1.0 + np.exp(self._calibration_a * raw_prob + self._calibration_b))
        return float(np.clip(calibrated, 0.01, 0.99))

    def _compute_risk_grade(self, mae: float, rr: float, confidence: float) -> str:
        """Compute risk grade A-F from multiple factors."""
        # Score 0-100 from: low MAE, high R:R, high confidence
        mae_score = max(0, 100 - mae * 1000)  # 0% MAE → 100, 10% → 0
        rr_score = min(100, rr * 33)  # R:R 3 → 100
        conf_score = confidence * 100

        composite = 0.4 * mae_score + 0.35 * rr_score + 0.25 * conf_score

        if composite >= 85:
            return "A"
        elif composite >= 70:
            return "B"
        elif composite >= 55:
            return "C"
        elif composite >= 40:
            return "D"
        else:
            return "F"

    def _default_prediction(self) -> MultiTargetPrediction:
        """Return a neutral default prediction when model is not trained."""
        return MultiTargetPrediction(
            direction="HOLD",
            direction_probability=0.33,
            expected_return_pct=0.0,
            expected_return_std=2.0,
            upside_potential_pct=0.0,
            downside_risk_pct=0.0,
            expected_holding_hours=0.0,
            time_to_tp_hours=0.0,
            time_to_sl_hours=0.0,
            max_adverse_excursion_pct=0.0,
            risk_reward_ratio=0.0,
            probability_tp=0.33,
            probability_sl=0.33,
            probability_timeout=0.34,
            confidence=0.0,
            prediction_uncertainty=1.0,
            recommended_position_pct=0.0,
            kelly_fraction=0.0,
            risk_grade="F",
            suggested_tp_pct=0.0,
            suggested_sl_pct=0.0,
            suggested_trailing_stop_pct=0.0,
            model_version=self.model_version,
            inference_time_ms=0.0,
            feature_set_version=self.feature_set_version,
        )

    def _train_sklearn_fallback(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        validation_split: float,
    ) -> Dict[str, Any]:
        """Fallback training using sklearn when XGBoost unavailable."""
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        self._feature_names = list(features.columns)
        n = len(features)
        split_idx = int(n * (1 - validation_split))

        X_train = features.iloc[:split_idx].values
        X_val = features.iloc[split_idx:].values

        metrics = {}

        for target_def in MULTI_TARGETS:
            col = target_def.column_name
            if col not in targets.columns:
                continue

            y_train = targets[col].iloc[:split_idx].values
            y_val = targets[col].iloc[split_idx:].values

            if target_def.clip_range:
                y_train = np.clip(y_train, *target_def.clip_range)
                y_val = np.clip(y_val, *target_def.clip_range)

            if target_def.target_type == "classification":
                y_train_mapped = DirectionEncoder.encode(y_train)
                y_val_mapped = DirectionEncoder.encode(y_val)
                model = GradientBoostingClassifier(
                    n_estimators=100, max_depth=5, learning_rate=0.05,
                )
                model.fit(X_train, y_train_mapped)
                accuracy = float(model.score(X_val, y_val_mapped))
                metrics[target_def.name] = {"accuracy": accuracy}
            else:
                if target_def.transform == "log":
                    y_train = np.log1p(np.clip(y_train, 0, None))
                    y_val = np.log1p(np.clip(y_val, 0, None))
                model = GradientBoostingRegressor(
                    n_estimators=100, max_depth=5, learning_rate=0.05,
                )
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                mae = float(np.mean(np.abs(val_pred - y_val)))
                metrics[target_def.name] = {"mae": mae}

            self._models[target_def.name] = model

        if "target_return_pct" in targets.columns:
            self._train_return_std = float(targets["target_return_pct"].std())
        if "target_holding_hours" in targets.columns:
            self._train_holding_mean = float(targets["target_holding_hours"].mean())
        if "target_mae_pct" in targets.columns:
            self._train_mae_mean = float(targets["target_mae_pct"].mean())

        self._is_trained = True
        return metrics

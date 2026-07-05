"""
ML Strategy — Machine Learning based signal generation.
Uses XGBoost with engineered features for price direction prediction.
"""

import os
import threading
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from src.strategy.base import BaseStrategy, LegacyTradeSignal, Signal
from src.ml.label_encoder import DirectionEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLStrategy(BaseStrategy):
    """
    ML-powered trading strategy:
    - Feature engineering from OHLCV + technical indicators
    - XGBoost classifier for direction prediction (up/down/flat)
    - Confidence threshold filtering
    - Auto-retraining on schedule
    """

    def __init__(
        self,
        symbols: list[str],
        timeframe: str = "1Hour",
        lookback: int = 500,
        model_path: str = "models/latest_model.joblib",
        min_confidence: float = 0.65,
        retrain_interval_hours: int = 24,
        fallback_strategy: str = "momentum",
    ):
        super().__init__(name="ml_xgboost", symbols=symbols, timeframe=timeframe, lookback=lookback)
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.retrain_interval_hours = retrain_interval_hours
        self.model = None
        self._last_train_time: Optional[datetime] = None
        self._feature_columns: list[str] = []
        self._model_lock = threading.Lock()  # Protects model load/save/predict
        self._fallback_strategy_name = fallback_strategy
        self._fallback_strategy = None
        self._bootstrap_attempted = False

        self._load_model()

        # Warn clearly if no model available
        if self.model is None:
            logger.warning(
                "ml.NO_MODEL_AVAILABLE",
                msg="No trained model found. System will use fallback strategy "
                    "until model is trained. Run /train or wait for auto-train.",
                path=self.model_path,
            )

    def _load_model(self):
        """Load a previously trained model from disk. Thread-safe."""
        with self._model_lock:
            if os.path.exists(self.model_path):
                try:
                    data = joblib.load(self.model_path)
                    self.model = data['model']
                    self._feature_columns = data.get('features', [])
                    self._last_train_time = data.get('trained_at')
                    logger.info("ml.model_loaded", path=self.model_path,
                               features=len(self._feature_columns))
                except Exception as e:
                    logger.error("ml.model_load_failed", error=str(e))
                    self.model = None

    def save_model(self):
        """Save the trained model to disk with versioning. Thread-safe."""
        with self._model_lock:
            if self.model is None:
                return
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump({
                'model': self.model,
                'features': self._feature_columns,
                'trained_at': datetime.now(),
            }, self.model_path)

        # Version the model
        try:
            from src.ml.model_registry import ModelRegistry
            registry = ModelRegistry()
            registry.save_version(
                model=self.model,
                features=self._feature_columns,
                metrics=getattr(self, '_last_train_metrics', {}),
                symbols=self.symbols,
            )
        except Exception as e:
            logger.warning("ml.versioning_failed", error=str(e))

        logger.info("ml.model_saved", path=self.model_path)

    @property
    def model_available(self) -> bool:
        """Whether a trained model is loaded and ready for predictions."""
        return self.model is not None

    def needs_retraining(self) -> bool:
        """Check if model needs retraining."""
        if self.model is None:
            return True
        if self._last_train_time is None:
            return True
        hours_since = (datetime.now() - self._last_train_time).total_seconds() / 3600
        return hours_since >= self.retrain_interval_hours

    def _get_fallback_strategy(self):
        """Lazy-create fallback strategy for when ML model is unavailable."""
        if self._fallback_strategy is None:
            try:
                from src.strategy.momentum import MomentumStrategy
                self._fallback_strategy = MomentumStrategy(
                    symbols=self.symbols,
                    timeframe=self.timeframe,
                    lookback=self.lookback,
                )
                logger.info("ml.fallback_created", strategy=self._fallback_strategy_name)
            except Exception as e:
                logger.error("ml.fallback_creation_failed", error=str(e))
        return self._fallback_strategy

    def _fallback_signals(self, data: dict[str, pd.DataFrame]) -> list:
        """Generate signals using fallback strategy when ML model unavailable."""
        logger.warning("ml.using_fallback", reason="no trained model available")
        fallback = self._get_fallback_strategy()
        if fallback is not None:
            try:
                signals = fallback.generate_signals(data)
                # Tag signals so they're identifiable as fallback
                for sig in signals:
                    sig.reason = f"[FALLBACK:{self._fallback_strategy_name}] {sig.reason or ''}"
                return signals
            except Exception as e:
                logger.error("ml.fallback_error", error=str(e))

        # Last resort: return NO_SIGNAL
        return [
            LegacyTradeSignal(
                symbol=symbol,
                signal=Signal.NO_SIGNAL,
                confidence=0.0,
                price=0.0,
                reason="PREDICTION_UNAVAILABLE: no model and fallback failed",
            )
            for symbol in data.keys()
        ]

    # ──────────────────────────────────────────────────────────────────────
    # Feature Engineering (delegates to shared pipeline)
    # ──────────────────────────────────────────────────────────────────────

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features using the shared feature pipeline."""
        from src.ml.features import engineer_features
        has_datetime_index = isinstance(df.index, pd.DatetimeIndex)
        return engineer_features(df, include_target=False)

    def get_feature_columns(self) -> list[str]:
        """Get the list of feature columns for the model."""
        from src.ml.features import get_feature_names
        if self._feature_columns:
            return self._feature_columns
        return get_feature_names(include_time_features=True)

    # ──────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────

    def train(self, training_data: dict[str, pd.DataFrame], forward_bars: int = 5, threshold: float = 0.005):
        """
        Train the XGBoost model on historical data.

        Args:
            training_data: Dict of symbol -> OHLCV DataFrame
            forward_bars: How many bars ahead to predict
            threshold: Minimum return to classify as up/down (vs flat)
        """
        from xgboost import XGBClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import classification_report

        all_X = []
        all_y = []

        feature_cols = self.get_feature_columns()

        for symbol, df in training_data.items():
            df = self.calculate_indicators(df)

            # Target: direction over next N bars (ONLY look-ahead in pipeline)
            # Features are strictly backward-looking (rolling, EMA with adjust=False).
            # dropna removes: first ~50 rows (feature warmup NaN) + last forward_bars rows (target NaN).
            # No temporal embargo needed because features[t] depend only on data[0:t].
            df['future_return'] = df['close'].shift(-forward_bars) / df['close'] - 1
            df['target'] = np.where(
                df['future_return'] > threshold, 1,   # UP
                np.where(df['future_return'] < -threshold, -1, 0)  # DOWN / FLAT
            )

            # Drop 'future_return' from feature set to prevent leakage
            if 'future_return' in df.columns:
                df = df.drop(columns=['future_return'])

            # Use only features that actually exist in the data
            available_cols = [c for c in feature_cols if c in df.columns and c != 'target']
            valid = df.dropna(subset=available_cols + ['target'])
            if len(valid) < 100:
                continue

            all_X.append(valid[available_cols])
            all_y.append(valid['target'])

        if not all_X:
            logger.warning("ml.no_training_data")
            return

        X = pd.concat(all_X, ignore_index=True)
        y = pd.concat(all_y, ignore_index=True)

        logger.info("ml.training", samples=len(X), features=len(feature_cols))

        # Time-series cross-validation
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
        )

        # Remap labels: -1 -> 0, 0 -> 1, 1 -> 2
        y_mapped = y.map({-1: 0, 0: 1, 1: 2})

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_mapped.iloc[train_idx], y_mapped.iloc[val_idx]
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            score = model.score(X_val, y_val)
            scores.append(score)

        self._last_train_metrics = {
            "cv_accuracy": float(np.mean(scores)),
            "cv_std": float(np.std(scores)),
            "n_samples": len(X),
            "n_features": len(self._feature_columns) if self._feature_columns else X.shape[1],
        }

        # Final fit on all data
        model.fit(X, y_mapped, verbose=False)
        self.model = model
        self._feature_columns = list(X.columns)  # Save actual columns used
        self._last_train_time = datetime.now()
        self.save_model()

        logger.info("ml.trained", cv_accuracy=f"{np.mean(scores):.3f}", std=f"{np.std(scores):.3f}")

    # ──────────────────────────────────────────────────────────────────────
    # Prediction / Signal Generation
    # ──────────────────────────────────────────────────────────────────────

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> list:
        """Generate ML-based signals. Falls back to momentum if no model. Thread-safe."""
        if self.model is None:
            return self._fallback_signals(data)

        signals = []
        feature_cols = self._feature_columns or self.get_feature_columns()

        for symbol, df in data.items():
            if len(df) < 100:
                continue

            df = self.calculate_indicators(df)
            latest = df.iloc[-1:]

            # Validate feature columns exist and check for NaN
            available_cols = [c for c in feature_cols if c in latest.columns]
            if len(available_cols) < len(feature_cols):
                missing = set(feature_cols) - set(available_cols)
                logger.warning("ml.predict_missing_features", symbol=symbol, missing=list(missing)[:5])
                continue
            features = latest[feature_cols]
            if features.isna().any(axis=1).iloc[0]:
                continue

            # Predict (thread-safe)
            with self._model_lock:
                try:
                    proba = self.model.predict_proba(features)[0]
                except Exception as e:
                    logger.error("ml.predict_failed", symbol=symbol, error=str(e))
                    continue
            # proba: [P(down), P(flat), P(up)]
            pred_class = np.argmax(proba)
            confidence = proba[pred_class]

            price = float(latest['close'].iloc[0])
            atr = float(latest['atr_14'].iloc[0]) if 'atr_14' in latest.columns and not pd.isna(latest['atr_14'].iloc[0]) else price * 0.02

            if pred_class == DirectionEncoder.UP_CLASS and confidence >= self.min_confidence:
                signal = Signal.STRONG_BUY if confidence >= 0.8 else Signal.BUY
                stop_loss = price - (atr * 2)
                take_profit = price + (atr * 3)
            elif pred_class == DirectionEncoder.DOWN_CLASS and confidence >= self.min_confidence:
                signal = Signal.STRONG_SELL if confidence >= 0.8 else Signal.SELL
                stop_loss = price + (atr * 2)
                take_profit = price - (atr * 3)
            else:
                signal = Signal.HOLD
                stop_loss = None
                take_profit = None

            signals.append(LegacyTradeSignal(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"ML pred: {DirectionEncoder.class_name(pred_class)} ({confidence:.1%})",
                indicators={
                    "prob_down": round(proba[DirectionEncoder.DOWN_CLASS], 3),
                    "prob_flat": round(proba[DirectionEncoder.FLAT_CLASS], 3),
                    "prob_up": round(proba[DirectionEncoder.UP_CLASS], 3),
                    "atr": round(atr, 4),
                },
            ))

        return [s for s in signals if s.is_actionable]

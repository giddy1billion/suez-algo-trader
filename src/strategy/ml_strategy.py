"""
ML Strategy — Machine Learning based signal generation.
Uses XGBoost with engineered features for price direction prediction.
"""

import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from src.strategy.base import BaseStrategy, TradeSignal, Signal
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
    ):
        super().__init__(name="ml_xgboost", symbols=symbols, timeframe=timeframe, lookback=lookback)
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.retrain_interval_hours = retrain_interval_hours
        self.model = None
        self._last_train_time: Optional[datetime] = None
        self._feature_columns: list[str] = []

        self._load_model()

    def _load_model(self):
        """Load a previously trained model from disk."""
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
        """Save the trained model to disk."""
        if self.model is None:
            return
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'features': self._feature_columns,
            'trained_at': datetime.now(),
        }, self.model_path)
        logger.info("ml.model_saved", path=self.model_path)

    def needs_retraining(self) -> bool:
        """Check if model needs retraining."""
        if self.model is None:
            return True
        if self._last_train_time is None:
            return True
        hours_since = (datetime.now() - self._last_train_time).total_seconds() / 3600
        return hours_since >= self.retrain_interval_hours

    # ──────────────────────────────────────────────────────────────────────
    # Feature Engineering
    # ──────────────────────────────────────────────────────────────────────

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features used by the ML model."""
        df = df.copy()

        # Price-based features
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        df['returns_20'] = df['close'].pct_change(20)

        # Volatility
        df['volatility_10'] = df['returns_1'].rolling(10).std()
        df['volatility_20'] = df['returns_1'].rolling(20).std()

        # Moving averages
        for period in [5, 10, 20, 50, 100]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'close_to_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']

        # EMA
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)

        # Volume features
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20'].replace(0, np.nan)

        # Candle patterns
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open']
        df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open']

        # Momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1

        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']

        return df

    def get_feature_columns(self) -> list[str]:
        """Get the list of feature columns for the model."""
        return [
            'returns_1', 'returns_5', 'returns_10', 'returns_20',
            'volatility_10', 'volatility_20',
            'close_to_sma_5', 'close_to_sma_10', 'close_to_sma_20', 'close_to_sma_50', 'close_to_sma_100',
            'macd', 'macd_signal', 'rsi', 'bb_pct',
            'volume_ratio', 'body_size', 'upper_shadow', 'lower_shadow',
            'momentum_5', 'momentum_10', 'atr_pct',
        ]

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

            # Target: direction over next N bars
            df['future_return'] = df['close'].shift(-forward_bars) / df['close'] - 1
            df['target'] = np.where(
                df['future_return'] > threshold, 1,   # UP
                np.where(df['future_return'] < -threshold, -1, 0)  # DOWN / FLAT
            )

            # Drop NaN rows
            valid = df.dropna(subset=feature_cols + ['target'])
            if len(valid) < 100:
                continue

            all_X.append(valid[feature_cols])
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
            use_label_encoder=False,
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

        # Final fit on all data
        model.fit(X, y_mapped, verbose=False)
        self.model = model
        self._feature_columns = feature_cols
        self._last_train_time = datetime.now()
        self.save_model()

        logger.info("ml.trained", cv_accuracy=f"{np.mean(scores):.3f}", std=f"{np.std(scores):.3f}")

    # ──────────────────────────────────────────────────────────────────────
    # Prediction / Signal Generation
    # ──────────────────────────────────────────────────────────────────────

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> list[TradeSignal]:
        """Generate ML-based signals."""
        if self.model is None:
            logger.warning("ml.no_model", msg="Train model first")
            return []

        signals = []
        feature_cols = self._feature_columns or self.get_feature_columns()

        for symbol, df in data.items():
            if len(df) < 100:
                continue

            df = self.calculate_indicators(df)
            latest = df.iloc[-1:]

            # Check for NaN features
            features = latest[feature_cols]
            if features.isna().any(axis=1).iloc[0]:
                continue

            # Predict
            proba = self.model.predict_proba(features)[0]
            # proba: [P(down), P(flat), P(up)]
            pred_class = np.argmax(proba)
            confidence = proba[pred_class]

            price = float(latest['close'].iloc[0])
            atr = float(latest['atr'].iloc[0]) if not pd.isna(latest['atr'].iloc[0]) else price * 0.02

            if pred_class == 2 and confidence >= self.min_confidence:  # UP
                signal = Signal.STRONG_BUY if confidence >= 0.8 else Signal.BUY
                stop_loss = price - (atr * 2)
                take_profit = price + (atr * 3)
            elif pred_class == 0 and confidence >= self.min_confidence:  # DOWN
                signal = Signal.STRONG_SELL if confidence >= 0.8 else Signal.SELL
                stop_loss = price + (atr * 2)
                take_profit = price - (atr * 3)
            else:
                signal = Signal.HOLD
                stop_loss = None
                take_profit = None

            signals.append(TradeSignal(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"ML pred: {'UP' if pred_class == 2 else 'DOWN' if pred_class == 0 else 'FLAT'} ({confidence:.1%})",
                indicators={
                    "prob_down": round(proba[0], 3),
                    "prob_flat": round(proba[1], 3),
                    "prob_up": round(proba[2], 3),
                    "atr": round(atr, 4),
                },
            ))

        return [s for s in signals if s.is_actionable]

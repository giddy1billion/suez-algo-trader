"""
Optuna-based Hyperparameter Tuning — Asset-class-specific optimization.

Tunes XGBoost parameters separately for equities and crypto, persists
the best configurations, and provides an API for the training pipeline
to load tuned parameters.

Usage:
    tuner = HyperparameterTuner(tuning_dir="models/tuning")
    best_params = tuner.tune(feature_data, asset_class="equity", n_trials=50)
    # Later, in training:
    params = tuner.load_best_params("equity")
"""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Asset Class Detection
# ---------------------------------------------------------------------------

CRYPTO_SUFFIXES = ("/USD", "/EUR", "/GBP", "/USDT", "/BTC")


def classify_asset_class(symbols: list[str]) -> str:
    """
    Determine asset class from symbol names.

    Returns 'crypto' if majority of symbols match crypto patterns,
    otherwise 'equity'.
    """
    if not symbols:
        return "equity"
    crypto_count = sum(
        1 for s in symbols
        if any(s.upper().endswith(suffix) for suffix in CRYPTO_SUFFIXES)
    )
    return "crypto" if crypto_count > len(symbols) / 2 else "equity"


# ---------------------------------------------------------------------------
# Hyperparameter Tuner
# ---------------------------------------------------------------------------


class HyperparameterTuner:
    """
    Optuna-based hyperparameter tuner with asset-class separation.

    Persists best configurations to disk for reproducible training.
    """

    def __init__(self, tuning_dir: str = "models/tuning"):
        self.tuning_dir = tuning_dir
        self._lock = threading.Lock()
        os.makedirs(tuning_dir, exist_ok=True)

    def _params_path(self, asset_class: str) -> str:
        return os.path.join(self.tuning_dir, f"best_params_{asset_class}.json")

    def tune(
        self,
        feature_data: dict[str, pd.DataFrame],
        asset_class: Optional[str] = None,
        n_trials: int = 50,
        n_splits: int = 3,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """
        Run Optuna hyperparameter optimization.

        Args:
            feature_data: Dict of symbol -> DataFrame with features and target.
            asset_class: 'equity' or 'crypto'. Auto-detected if None.
            n_trials: Number of Optuna trials.
            n_splits: Number of TimeSeriesSplit folds for evaluation.
            random_seed: Random seed for reproducibility.

        Returns:
            Best parameters dict.
        """
        import optuna
        from xgboost import XGBClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if asset_class is None:
            asset_class = classify_asset_class(list(feature_data.keys()))

        # Prepare combined data
        X, y = self._prepare_data(feature_data)
        if X is None or len(X) < 200:
            logger.warning("hyperparameter_tuning.insufficient_data", n_samples=len(X) if X is not None else 0)
            return self._default_params(asset_class)

        def objective(trial):
            params = self._suggest_params(trial, asset_class)

            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                # Apply embargo
                embargo = 5
                val_idx = val_idx[embargo:]
                if len(val_idx) < 20:
                    continue

                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = XGBClassifier(
                    **params,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    random_state=random_seed,
                    verbosity=0,
                    early_stopping_rounds=30,
                )
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                score = accuracy_score(y_val, model.predict(X_val))
                scores.append(score)

            return float(np.mean(scores)) if scores else 0.0

        sampler = optuna.samplers.TPESampler(seed=random_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params["n_estimators"] = best_params.pop("n_estimators", 500)

        # Persist
        self._save_best_params(asset_class, best_params, study.best_value)

        logger.info(
            "hyperparameter_tuning.complete",
            asset_class=asset_class,
            best_value=round(study.best_value, 4),
            n_trials=n_trials,
            best_params=best_params,
        )
        return best_params

    def load_best_params(self, asset_class: str) -> Optional[dict[str, Any]]:
        """
        Load persisted best parameters for an asset class.

        Returns None if no tuned parameters exist.
        """
        path = self._params_path(asset_class)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("params")
        except (json.JSONDecodeError, OSError):
            return None

    def get_training_params(self, symbols: list[str]) -> dict[str, Any]:
        """
        Get best parameters for training, with fallback to defaults.

        Determines asset class from symbols, attempts to load tuned params,
        falls back to default params if none persisted.
        """
        asset_class = classify_asset_class(symbols)
        params = self.load_best_params(asset_class)
        if params is None:
            params = self._default_params(asset_class)
            logger.info(
                "hyperparameter_tuning.using_defaults",
                asset_class=asset_class,
            )
        else:
            logger.info(
                "hyperparameter_tuning.using_tuned",
                asset_class=asset_class,
            )
        return params

    def _save_best_params(self, asset_class: str, params: dict, best_score: float) -> None:
        """Persist best parameters atomically."""
        path = self._params_path(asset_class)
        record = {
            "asset_class": asset_class,
            "params": params,
            "best_score": best_score,
            "tuned_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)

    def _prepare_data(self, feature_data: dict[str, pd.DataFrame]) -> tuple:
        """Prepare combined X, y from feature_data dict."""
        frames = []
        for symbol, df in feature_data.items():
            if "target" in df.columns:
                frames.append(df)
            elif "close" in df.columns:
                # Generate target using volatility-adaptive threshold (same as training pipeline)
                df_copy = df.copy()
                forward_bars = 5
                base_threshold = 0.005
                df_copy["future_return"] = df_copy["close"].shift(-forward_bars) / df_copy["close"] - 1
                returns = df_copy["close"].pct_change()
                rolling_vol = returns.rolling(20, min_periods=10).std().fillna(returns.std())
                adaptive_threshold = np.maximum(
                    base_threshold, 0.5 * rolling_vol * np.sqrt(forward_bars)
                )
                df_copy["target"] = np.where(
                    df_copy["future_return"] > adaptive_threshold, 1,
                    np.where(df_copy["future_return"] < -adaptive_threshold, -1, 0)
                )
                df_copy = df_copy.drop(columns=["future_return"])
                df_copy = df_copy.dropna(subset=["target"])
                frames.append(df_copy)

        if not frames:
            return None, None

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.dropna()

        # Separate features from target
        exclude_cols = {"target", "open", "high", "low", "close", "volume", "symbol",
                        "date", "datetime", "timestamp"}
        feature_cols = [c for c in combined.columns if c not in exclude_cols]

        if not feature_cols:
            return None, None

        X = combined[feature_cols].values.astype(np.float32)
        y = combined["target"].values.astype(int)

        # Encode trading labels [-1,0,1] → model classes [0,1,2]
        from src.ml.label_encoder import DirectionEncoder
        y = DirectionEncoder.encode(y)

        # Handle any remaining NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y

    def _suggest_params(self, trial, asset_class: str) -> dict[str, Any]:
        """Suggest hyperparameters for a trial, varying by asset class."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        }

        if asset_class == "crypto":
            # Crypto: higher volatility → more regularization, lower learning rate
            params["reg_lambda"] = trial.suggest_float("reg_lambda_crypto", 1.0, 8.0)
            params["min_child_weight"] = trial.suggest_int("min_child_weight_crypto", 10, 80)

        return params

    @staticmethod
    def _default_params(asset_class: str) -> dict[str, Any]:
        """Default parameters per asset class."""
        base = {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.02,
            "subsample": 0.7,
            "colsample_bytree": 0.6,
            "min_child_weight": 10,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.5,
        }
        if asset_class == "crypto":
            # Crypto needs stronger regularization
            base.update({
                "min_child_weight": 15,
                "reg_lambda": 2.5,
                "subsample": 0.65,
            })
        return base

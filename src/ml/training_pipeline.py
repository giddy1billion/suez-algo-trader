"""
Training Pipeline — End-to-end ML model training orchestrator.

Provides:
- One-command trigger for full data → features → train → validate → deploy pipeline
- Non-blocking execution in background thread
- Progress reporting via events
- Auto-validation via governance before deployment
- Configurable triggers (manual, scheduled, performance decay)
- Integration with ModelRegistry, FeatureStore, and Governance
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.ml.model_registry import ModelRegistry
from src.ml.governance import ModelGovernance
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineStatus(str, Enum):
    """Training pipeline execution status."""
    IDLE = "idle"
    FETCHING_DATA = "fetching_data"
    ENGINEERING_FEATURES = "engineering_features"
    TRAINING = "training"
    VALIDATING = "validating"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineProgress:
    """Progress tracker for a training pipeline run."""
    pipeline_id: str
    status: PipelineStatus = PipelineStatus.IDLE
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_step: str = ""
    steps_completed: int = 0
    total_steps: int = 6
    error: Optional[str] = None
    # Results
    version: Optional[str] = None
    metrics: dict = field(default_factory=dict)
    validation_issues: list = field(default_factory=list)
    auto_deployed: bool = False
    trigger: str = "manual"

    @property
    def duration_seconds(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()

    @property
    def is_running(self) -> bool:
        return self.status not in (PipelineStatus.IDLE, PipelineStatus.COMPLETED, PipelineStatus.FAILED)

    def to_dict(self) -> dict:
        return {
            "pipeline_id": self.pipeline_id,
            "status": self.status.value,
            "current_step": self.current_step,
            "progress": f"{self.steps_completed}/{self.total_steps}",
            "duration_seconds": round(self.duration_seconds, 1),
            "version": self.version,
            "metrics": self.metrics,
            "auto_deployed": self.auto_deployed,
            "trigger": self.trigger,
            "error": self.error,
        }


class TrainingPipeline:
    """
    End-to-end ML training pipeline orchestrator.

    Coordinates the full lifecycle:
    1. Fetch training data from broker
    2. Engineer features
    3. Train model (XGBoost by default)
    4. Validate via governance
    5. Register in model registry
    6. Auto-deploy if validation passes

    Runs in background thread so live trading isn't interrupted.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        governance: ModelGovernance,
        broker=None,
        event_bus=None,
        auto_deploy: bool = True,
        min_training_samples: int = 500,
    ):
        self._registry = registry
        self._governance = governance
        self._broker = broker
        self._event_bus = event_bus
        self._auto_deploy = auto_deploy
        self._min_samples = min_training_samples

        # Recovery settings (loaded from config when available)
        try:
            from config.settings import settings
            self._max_retries = settings.model_max_retries
            self._retry_backoff_seconds = settings.model_retry_backoff_seconds
            self._stale_threshold_hours = settings.model_stale_threshold_hours
        except Exception:
            self._max_retries = 3
            self._retry_backoff_seconds = 60.0
            self._stale_threshold_hours = 168.0

        # State
        self._current: Optional[PipelineProgress] = None
        self._history: list[PipelineProgress] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def train(
        self,
        symbols: list[str],
        timeframe: str = "1Hour",
        lookback_bars: int = 1000,
        trigger: str = "manual",
        callback: Optional[Callable[[PipelineProgress], None]] = None,
        data_override: Optional[dict[str, pd.DataFrame]] = None,
    ) -> str:
        """
        Trigger the full training pipeline (non-blocking).

        Args:
            symbols: List of symbols to train on.
            timeframe: Bar timeframe for data.
            lookback_bars: Number of bars to fetch per symbol.
            trigger: What triggered training ("manual", "scheduled", "performance_decay").
            callback: Optional completion callback.
            data_override: Optional pre-fetched data (skips data fetch step).

        Returns:
            pipeline_id for tracking progress.

        Raises:
            RuntimeError: If a pipeline is already running.
        """
        with self._lock:
            if self._current and self._current.is_running:
                raise RuntimeError(
                    f"Pipeline already running: {self._current.pipeline_id} "
                    f"(step: {self._current.current_step})"
                )

            pipeline_id = uuid.uuid4().hex[:12]
            progress = PipelineProgress(
                pipeline_id=pipeline_id,
                trigger=trigger,
            )
            self._current = progress

        def _run():
            try:
                self._execute_pipeline(
                    progress, symbols, timeframe, lookback_bars, data_override
                )
            except Exception as e:
                progress.status = PipelineStatus.FAILED
                progress.error = str(e)
                progress.completed_at = datetime.now(timezone.utc)
                logger.error("training_pipeline.failed", pipeline_id=pipeline_id, error=str(e))
            finally:
                with self._lock:
                    self._history.append(progress)
                if callback:
                    try:
                        callback(progress)
                    except Exception:
                        pass

        self._thread = threading.Thread(
            target=_run,
            name=f"training-pipeline-{pipeline_id}",
            daemon=True,
        )
        self._thread.start()

        # Publish start event
        if self._event_bus:
            from src.core.events import ModelTrainingStarted
            self._event_bus.publish(ModelTrainingStarted(
                pipeline_id=pipeline_id,
                symbols=symbols,
                trigger=trigger,
                source="training_pipeline",
            ))

        logger.info(
            "training_pipeline.started",
            pipeline_id=pipeline_id,
            symbols=symbols,
            timeframe=timeframe,
            trigger=trigger,
        )
        return pipeline_id

    def train_sync(
        self,
        symbols: list[str],
        timeframe: str = "1Hour",
        lookback_bars: int = 1000,
        data_override: Optional[dict[str, pd.DataFrame]] = None,
    ) -> PipelineProgress:
        """
        Run training pipeline synchronously (blocking).

        Useful for CLI and testing.
        """
        pipeline_id = uuid.uuid4().hex[:12]
        progress = PipelineProgress(pipeline_id=pipeline_id, trigger="sync")

        with self._lock:
            self._current = progress

        try:
            self._execute_pipeline(progress, symbols, timeframe, lookback_bars, data_override)
        except Exception as e:
            progress.status = PipelineStatus.FAILED
            progress.error = str(e)
            progress.completed_at = datetime.now(timezone.utc)

        with self._lock:
            self._history.append(progress)

        return progress

    # ──────────────────────────────────────────────────────────────────────
    # Status
    # ──────────────────────────────────────────────────────────────────────

    def get_progress(self) -> Optional[dict]:
        """Get current pipeline progress."""
        with self._lock:
            if self._current:
                return self._current.to_dict()
            return None

    def is_running(self) -> bool:
        """Check if a pipeline is currently running."""
        with self._lock:
            return self._current is not None and self._current.is_running

    def get_history(self, limit: int = 10) -> list[dict]:
        """Get recent pipeline execution history."""
        with self._lock:
            return [p.to_dict() for p in self._history[-limit:]]

    # ──────────────────────────────────────────────────────────────────────
    # Pipeline Execution
    # ──────────────────────────────────────────────────────────────────────

    def _execute_pipeline(
        self,
        progress: PipelineProgress,
        symbols: list[str],
        timeframe: str,
        lookback_bars: int,
        data_override: Optional[dict[str, pd.DataFrame]],
    ):
        """Execute the full training pipeline."""
        progress.started_at = datetime.now(timezone.utc)
        progress.status = PipelineStatus.FETCHING_DATA
        train_start = time.perf_counter()

        # Step 1: Fetch data
        progress.current_step = "Fetching market data"
        logger.info("training_pipeline.step", step="fetch_data", symbols=symbols)

        if data_override:
            data = data_override
        else:
            data = self._fetch_data(symbols, timeframe, lookback_bars)

        if not data:
            raise RuntimeError("No data fetched for any symbol")

        progress.steps_completed = 1

        # Step 2: Engineer features
        progress.status = PipelineStatus.ENGINEERING_FEATURES
        progress.current_step = "Engineering features"
        logger.info("training_pipeline.step", step="engineer_features")

        from src.ml.features import engineer_features
        feature_data = {}
        for symbol, df in data.items():
            try:
                featured_df = engineer_features(df, include_target=False)
                if len(featured_df) >= self._min_samples:
                    feature_data[symbol] = featured_df
                else:
                    logger.warning(
                        "training_pipeline.insufficient_data",
                        symbol=symbol,
                        rows=len(featured_df),
                        min_required=self._min_samples,
                    )
            except Exception as e:
                logger.warning("training_pipeline.feature_error", symbol=symbol, error=str(e))

        if not feature_data:
            raise RuntimeError("Feature engineering produced no usable data")

        progress.steps_completed = 2

        # Step 3: Train model
        progress.status = PipelineStatus.TRAINING
        progress.current_step = "Training XGBoost model"
        logger.info("training_pipeline.step", step="train_model")

        model, metrics, feature_cols = self._train_model(feature_data)
        progress.metrics = metrics
        progress.steps_completed = 3

        # Step 4: Register in model registry
        progress.current_step = "Registering model version"
        logger.info("training_pipeline.step", step="register")

        version = self._registry.save_version(
            model=model,
            features=feature_cols,
            metrics=metrics,
            symbols=symbols,
            note=f"Pipeline {progress.pipeline_id} ({progress.trigger})",
        )
        progress.version = version
        progress.steps_completed = 4

        # Step 5: Governance validation
        progress.status = PipelineStatus.VALIDATING
        progress.current_step = "Validating via governance"
        logger.info("training_pipeline.step", step="validate", version=version)

        training_duration = time.perf_counter() - train_start
        self._governance.record_training(
            version=version,
            features=feature_cols,
            dataset=pd.concat(feature_data.values()) if feature_data else None,
            metrics=metrics,
            hyperparameters=metrics.get("hyperparameters", {}),
            training_duration=training_duration,
        )

        is_valid, issues = self._governance.validate_for_deployment(version)
        progress.validation_issues = issues
        progress.steps_completed = 5

        # Step 6: Auto-deploy if valid, reject otherwise
        progress.status = PipelineStatus.DEPLOYING
        progress.current_step = "Deploying model"

        if is_valid and self._auto_deploy:
            self._governance.deploy(
                version=version,
                reason=f"Auto-deploy from pipeline {progress.pipeline_id}",
            )
            progress.auto_deployed = True
            logger.info("training_pipeline.auto_deployed", version=version)
        elif not is_valid:
            logger.warning(
                "training_pipeline.validation_failed",
                version=version,
                issues=issues,
            )
            # Publish ModelRejected event
            if self._event_bus:
                from src.core.events import ModelRejected
                self._event_bus.publish(ModelRejected(
                    version=version,
                    reasons=issues,
                    source="training_pipeline",
                ))
        progress.steps_completed = 6

        # Complete
        progress.status = PipelineStatus.COMPLETED
        progress.completed_at = datetime.now(timezone.utc)
        progress.current_step = "Done"

        # Publish completion event
        if self._event_bus:
            from src.core.events import ModelTrainingCompleted
            self._event_bus.publish(ModelTrainingCompleted(
                pipeline_id=progress.pipeline_id,
                version=version,
                metrics=metrics,
                duration_seconds=progress.duration_seconds,
                auto_deployed=progress.auto_deployed,
                source="training_pipeline",
            ))

        logger.info(
            "training_pipeline.completed",
            pipeline_id=progress.pipeline_id,
            version=version,
            duration_s=round(progress.duration_seconds, 1),
            auto_deployed=progress.auto_deployed,
            metrics=metrics,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Internal Steps
    # ──────────────────────────────────────────────────────────────────────

    def _fetch_data(
        self, symbols: list[str], timeframe: str, lookback_bars: int
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical data from broker."""
        if self._broker is None:
            raise RuntimeError("No broker configured for data fetch")

        data = {}
        for symbol in symbols:
            try:
                df = self._broker.get_bars_df(symbol, timeframe, lookback_bars)
                if df is not None and len(df) > 0:
                    data[symbol] = df
                    logger.info(
                        "training_pipeline.data_fetched",
                        symbol=symbol,
                        bars=len(df),
                    )
            except Exception as e:
                logger.warning(
                    "training_pipeline.fetch_error",
                    symbol=symbol,
                    error=str(e),
                )

        return data

    def _train_model(
        self, feature_data: dict[str, pd.DataFrame]
    ) -> tuple[Any, dict, list[str]]:
        """Train XGBoost model on prepared feature data."""
        from xgboost import XGBClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score, classification_report

        # Combine all symbol data
        all_dfs = []
        for symbol, df in feature_data.items():
            df_copy = df.copy()
            df_copy['_symbol'] = symbol
            all_dfs.append(df_copy)

        combined = pd.concat(all_dfs, ignore_index=True)

        # Create target: forward return direction
        forward_bars = 5
        threshold = 0.005
        combined['future_return'] = combined['close'].shift(-forward_bars) / combined['close'] - 1
        combined['target'] = np.where(
            combined['future_return'] > threshold, 1,
            np.where(combined['future_return'] < -threshold, -1, 0)
        )

        # Get feature columns (exclude meta and target)
        exclude_cols = {'target', 'future_return', '_symbol', 'open', 'high', 'low', 'close', 'volume'}
        feature_cols = [c for c in combined.columns if c not in exclude_cols]

        # Drop NaN rows
        valid = combined.dropna(subset=feature_cols + ['target'])
        if len(valid) < self._min_samples:
            raise RuntimeError(
                f"Insufficient training samples: {len(valid)} < {self._min_samples}"
            )

        X = valid[feature_cols].values
        y = valid['target'].values

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=42,
                verbosity=0,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            score = accuracy_score(y_val, model.predict(X_val))
            cv_scores.append(score)

        # Train final model on all data
        final_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            verbosity=0,
        )
        final_model.fit(X, y, verbose=False)

        metrics = {
            "cv_accuracy": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "n_samples": len(valid),
            "n_features": len(feature_cols),
            "n_symbols": len(feature_data),
            "class_distribution": {
                "up": int((y == 1).sum()),
                "flat": int((y == 0).sum()),
                "down": int((y == -1).sum()),
            },
            "hyperparameters": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
            },
        }

        return final_model, metrics, feature_cols

    # ──────────────────────────────────────────────────────────────────────
    # Self-Healing & Recovery
    # ──────────────────────────────────────────────────────────────────────

    def train_with_retry(
        self,
        symbols: list[str],
        timeframe: str = "1Hour",
        lookback_bars: int = 1000,
        trigger: str = "auto_retry",
        data_override: Optional[dict[str, pd.DataFrame]] = None,
    ) -> Optional[str]:
        """
        Train with automatic retry on failure.

        Uses exponential backoff between retries.

        Returns:
            pipeline_id of successful run, or None if all retries exhausted.
        """
        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                progress = self.train_sync(
                    symbols=symbols,
                    timeframe=timeframe,
                    lookback_bars=lookback_bars,
                    data_override=data_override,
                )
                if progress.status == PipelineStatus.COMPLETED:
                    logger.info(
                        "training_pipeline.retry_succeeded",
                        attempt=attempt,
                        pipeline_id=progress.pipeline_id,
                    )
                    return progress.pipeline_id
                last_error = progress.error or "Unknown failure"
            except Exception as e:
                last_error = str(e)

            if attempt < self._max_retries:
                backoff = self._retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "training_pipeline.retry_backoff",
                    attempt=attempt,
                    next_attempt=attempt + 1,
                    backoff_seconds=backoff,
                    error=last_error,
                )
                time.sleep(backoff)

        logger.error(
            "training_pipeline.all_retries_exhausted",
            max_retries=self._max_retries,
            last_error=last_error,
        )
        return None

    def check_stale_model(self) -> bool:
        """
        Check if the deployed model is stale (older than threshold).

        Returns True if the model is stale and retraining should be triggered.
        """
        try:
            deployed = self._governance.get_deployed_model()
            if deployed is None:
                return True  # No model at all — definitely stale

            if not deployed.training_timestamp:
                return True

            training_time = datetime.fromisoformat(deployed.training_timestamp)
            age_hours = (datetime.now(timezone.utc) - training_time).total_seconds() / 3600
            if age_hours > self._stale_threshold_hours:
                logger.info(
                    "training_pipeline.stale_model_detected",
                    model_version=deployed.version,
                    age_hours=round(age_hours, 1),
                    threshold_hours=self._stale_threshold_hours,
                )
                return True
        except Exception as e:
            logger.warning("training_pipeline.stale_check_error", error=str(e))
        return False

    def auto_rollback(self, reason: str = "") -> bool:
        """
        Roll back to the previous model version if available.

        Returns True if rollback was successful.
        """
        try:
            previous = self._registry.rollback()
            if previous:
                # Publish rollback event
                if self._event_bus:
                    from src.core.events import ModelAutoRollback
                    self._event_bus.publish(ModelAutoRollback(
                        from_version=getattr(previous, 'from_version', ''),
                        to_version=getattr(previous, 'to_version', str(previous)),
                        reason=reason,
                        source="training_pipeline",
                    ))
                logger.info(
                    "training_pipeline.auto_rollback",
                    to_version=str(previous),
                    reason=reason,
                )
                return True
        except Exception as e:
            logger.error("training_pipeline.rollback_failed", error=str(e))
        return False

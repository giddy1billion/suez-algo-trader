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
from src.ml.label_encoder import DirectionEncoder
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
        experience_db=None,
        dataset_registry=None,
        training_lock=None,
    ):
        self._registry = registry
        self._governance = governance
        self._broker = broker
        self._event_bus = event_bus
        self._auto_deploy = auto_deploy
        self._experience_db = experience_db  # ExperienceDatabase for closed-loop training
        self._dataset_registry = dataset_registry  # DatasetRegistry for lineage tracking
        self._min_samples = min_training_samples
        self._training_lock = training_lock  # TrainingLock for distributed singleton

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
        self._max_history: int = 50  # Cap to prevent memory leak
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

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
        # Distributed lock check — prevents concurrent training across instances
        if self._training_lock:
            from src.ml.training_lock import TrainingLockError
            # Generate pipeline_id early so we can use it for lock acquisition
            pipeline_id = uuid.uuid4().hex[:12]
            if not self._training_lock.try_acquire(pipeline_id):
                holder = self._training_lock.lock_holder() or "unknown"
                raise RuntimeError(
                    f"Training lock held by {holder}. "
                    f"Cannot start pipeline {pipeline_id} from "
                    f"{self._training_lock.instance_identity}."
                )
        else:
            pipeline_id = uuid.uuid4().hex[:12]

        with self._lock:
            if self._current and self._current.is_running:
                # Release distributed lock if we acquired it
                if self._training_lock:
                    self._training_lock.release(pipeline_id)
                raise RuntimeError(
                    f"Pipeline already running: {self._current.pipeline_id} "
                    f"(step: {self._current.current_step})"
                )
            self._stop_event.clear()

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
                # Release distributed lock when training completes
                if self._training_lock:
                    self._training_lock.release(pipeline_id)
                with self._lock:
                    self._history.append(progress)
                    if len(self._history) > self._max_history:
                        self._history = self._history[-self._max_history:]
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
            instance=self._training_lock.instance_identity if self._training_lock else "local",
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
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

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

    def stop(self, timeout: float = 10.0) -> None:
        """
        Request pipeline shutdown and join the worker thread when present.

        The request is cooperative; the current run exits at safe checkpoints.
        """
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("training_pipeline.stop_timeout", timeout=timeout)

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
        self._raise_if_stopping()

        # Pre-flight: check optional dependencies
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            logger.warning(
                "training_pipeline.pyarrow_missing",
                msg="pyarrow not installed. Dataset snapshots will use CSV fallback. "
                    "Install with: pip install pyarrow",
            )

        # Step 1: Fetch data
        progress.current_step = "Fetching market data"
        logger.info("training_pipeline.step", step="fetch_data", symbols=symbols)

        if data_override:
            data = data_override
        else:
            data = self._fetch_data(symbols, timeframe, lookback_bars)
        self._raise_if_stopping()

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
            self._raise_if_stopping()
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
        self._raise_if_stopping()

        progress.steps_completed = 2

        # Step 2.5: Enrich with live trade outcomes (closed-loop)
        feature_data = self._enrich_with_experience(feature_data)

        # Step 3: Train model (returns holdout data for out-of-sample governance eval)
        progress.status = PipelineStatus.TRAINING
        progress.current_step = "Training XGBoost model"
        logger.info("training_pipeline.step", step="train_model")

        model, metrics, feature_cols, X_holdout, y_holdout, close_holdout = self._train_model(feature_data)
        self._raise_if_stopping()
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
            activate=False,  # Don't activate until governance approves
        )
        progress.version = version
        progress.steps_completed = 4

        # Record dataset lineage (links dataset → model version)
        if self._dataset_registry:
            try:
                combined_df = pd.concat(feature_data.values()) if feature_data else pd.DataFrame()
                self._dataset_registry.register_training_run(
                    model_version=version,
                    dataset_df=combined_df,
                    symbols=symbols,
                    feature_columns=feature_cols,
                    pipeline_id=progress.pipeline_id,
                    metrics=metrics,
                )
                logger.info("training_pipeline.dataset_registered", version=version)
            except Exception as e:
                logger.warning("training_pipeline.dataset_registry_error", error=str(e))

        # Step 5: Governance validation (with out-of-sample backtest metrics)
        progress.status = PipelineStatus.VALIDATING
        progress.current_step = "Backtesting and validating via governance"
        logger.info("training_pipeline.step", step="validate", version=version)

        # Run out-of-sample backtest on HELD-OUT data the model never trained on
        backtest_results = self._backtest_model_oos(model, feature_cols, X_holdout, y_holdout, close_holdout)
        # Run genuine walk-forward validation (expanding-window refit + predict)
        walk_forward_results = self._walk_forward_validation(feature_data, feature_cols)
        metrics.update({
            "sharpe": backtest_results.get("sharpe", 0.0),
            "sharpe_ratio": backtest_results.get("sharpe", 0.0),
            "n_trades": backtest_results.get("n_trades", 0),
            "max_drawdown": backtest_results.get("max_drawdown", 0.0),
        })

        training_duration = time.perf_counter() - train_start
        self._governance.record_training(
            version=version,
            features=feature_cols,
            dataset=pd.concat(feature_data.values()) if feature_data else None,
            metrics=metrics,
            hyperparameters=metrics.get("hyperparameters", {}),
            training_duration=training_duration,
            walk_forward_results=walk_forward_results,
            monte_carlo_results=backtest_results.get("monte_carlo"),
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
            # Activate in registry ONLY after governance approves — this is
            # what the predictor's auto_reload watcher polls.
            self._registry.set_active_version(version)
            progress.auto_deployed = True
            logger.info("training_pipeline.auto_deployed", version=version)
        elif not is_valid:
            # Model stays inactive in registry — predictor won't load it
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
            self._raise_if_stopping()
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

    def _enrich_with_experience(
        self, feature_data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """
        Enrich training data with live trade outcome LABELS (not features).

        CRITICAL: To prevent data leakage, we NEVER mix live-trade features
        into the historical feature matrix. Live trade features were computed
        at prediction time and may contain future-looking information relative
        to the historical training window.

        Instead, we use experience data in two safe ways:
        1. Sample weights: historical bars that correspond to profitable live
           trades receive higher weight (verified signal quality).
        2. Calibration signal: the overall accuracy rate from experience
           adjusts the confidence threshold used in label generation.

        The actual features ALWAYS come from the historical bar data.
        Experience provides only outcome verification and weighting.
        """
        if self._experience_db is None:
            return feature_data

        try:
            # Fast pre-check: skip entirely if no trades recorded yet
            if hasattr(self._experience_db, 'total_trades'):
                _trade_count = self._experience_db.total_trades
                # Handle both property and method access
                if callable(_trade_count):
                    _trade_count = _trade_count()
                if _trade_count == 0:
                    return feature_data

            experience_df = self._experience_db.get_training_samples(min_trades=20)
            if experience_df is None or experience_df.empty:
                logger.info("training_pipeline.no_experience_data")
                return feature_data

            # Extract verified accuracy from live trading (calibration signal)
            recent_accuracy = self._experience_db.get_recent_accuracy(n_trades=50)

            # Use time-weighted samples if available
            if hasattr(self._experience_db, 'get_training_samples_weighted'):
                try:
                    weighted_df = self._experience_db.get_training_samples_weighted(
                        min_trades=20, half_life_days=30.0
                    )
                    if weighted_df is not None and not weighted_df.empty:
                        experience_df = weighted_df
                except Exception:
                    pass  # Fall back to unweighted

            # Build symbol-level performance weights from experience
            # Symbols with verified profitable history get boosted weights
            symbol_col = '_symbol' if '_symbol' in experience_df.columns else 'symbol'
            if symbol_col not in experience_df.columns:
                logger.info("training_pipeline.experience_no_symbol_col")
                return feature_data

            symbol_accuracy = {}
            for sym in experience_df[symbol_col].unique():
                sym_data = experience_df[experience_df[symbol_col] == sym]
                profitable_col = 'actual_profitable' if 'actual_profitable' in sym_data.columns else 'profitable'
                if profitable_col in sym_data.columns:
                    symbol_accuracy[sym] = float(sym_data[profitable_col].mean())

            # Apply sample weights to feature_data based on experience accuracy
            # Symbols with verified high accuracy get boosted; poor accuracy get reduced
            enriched_count = 0
            for symbol, df in feature_data.items():
                if symbol not in symbol_accuracy:
                    continue

                accuracy = symbol_accuracy[symbol]
                # Weight: accuracy maps to [0.5, 2.0] range
                # 50% accuracy = weight 1.0 (neutral)
                # 70% accuracy = weight 1.4 (boost)
                # 30% accuracy = weight 0.6 (reduce)
                weight = 0.5 + accuracy * 1.5
                weight = max(0.5, min(2.0, weight))

                df_copy = df.copy()
                df_copy['_sample_weight'] = weight
                df_copy['_from_experience'] = 0  # Features are historical, NOT from experience
                feature_data[symbol] = df_copy
                enriched_count += 1

            if enriched_count > 0:
                logger.info(
                    "training_pipeline.experience_enrichment",
                    method="label_weighting_only",
                    symbols_weighted=enriched_count,
                    live_accuracy=f"{recent_accuracy:.1%}",
                    leakage_prevention="features_untouched",
                )
        except Exception as e:
            logger.warning("training_pipeline.experience_enrichment_error", error=str(e))

        return feature_data

    def _prepare_training_data(
        self, feature_data: dict[str, pd.DataFrame]
    ) -> tuple[np.ndarray, np.ndarray, list[str], Optional[np.ndarray], int, Optional[np.ndarray]]:
        """
        Prepare combined feature matrix from per-symbol DataFrames.

        Computes targets per-symbol (avoiding cross-symbol leakage), concatenates,
        drops forward-looking columns, and returns the clean feature matrix.

        Improvements over naive approach:
        - Volatility-adaptive labeling threshold per symbol (avoids labeling noise as signal)
        - Symbol one-hot encoding (lets model learn asset-specific patterns)
        - Feature variance filtering (removes near-constant noise features)

        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Encoded target vector
            feature_cols: List of feature column names
            sample_weights: Optional sample weight vector
            holdout_start_idx: Index where the temporal holdout begins (last 20%)
        """
        # Combine all symbol data
        all_dfs = []
        symbol_list = sorted(feature_data.keys())
        for symbol, df in feature_data.items():
            df_copy = df.copy()
            df_copy['_symbol'] = symbol
            all_dfs.append(df_copy)

        # Compute target per-symbol BEFORE concatenation to avoid cross-symbol leakage
        # Use VOLATILITY-ADAPTIVE threshold: scale by each symbol's realized volatility
        # so high-vol assets (BTC, ETH) don't get mislabeled as directional on noise moves
        forward_bars = 5
        base_threshold = 0.005
        for df_copy in all_dfs:
            df_copy['future_return'] = df_copy['close'].shift(-forward_bars) / df_copy['close'] - 1
            # Adaptive threshold = max(base, 0.5 * rolling 20-bar realized vol * sqrt(forward_bars))
            # This prevents labeling noise as signal in volatile assets
            if 'close' in df_copy.columns and len(df_copy) > 20:
                returns = df_copy['close'].pct_change()
                rolling_vol = returns.rolling(20, min_periods=10).std().fillna(returns.std())
                adaptive_threshold = np.maximum(
                    base_threshold,
                    0.5 * rolling_vol * np.sqrt(forward_bars)
                )
            else:
                adaptive_threshold = base_threshold
            df_copy['target'] = np.where(
                df_copy['future_return'] > adaptive_threshold, 1,
                np.where(df_copy['future_return'] < -adaptive_threshold, -1, 0)
            )

        combined = pd.concat(all_dfs, ignore_index=True)

        # Add symbol one-hot encoding so model can learn asset-specific patterns
        for sym in symbol_list:
            combined[f'_sym_{sym}'] = (combined['_symbol'] == sym).astype(np.float32)

        # Get feature columns (exclude meta and target)
        exclude_cols = {'target', 'future_return', '_symbol', '_sample_weight', '_from_experience',
                        'open', 'high', 'low', 'close', 'volume'}
        feature_cols = [c for c in combined.columns if c not in exclude_cols]

        # Drop forward-looking column to ensure no leakage into features
        combined = combined.drop(columns=['future_return'], errors='ignore')

        # Extract sample weights if present (from experience enrichment)
        sample_weights = None
        if '_sample_weight' in combined.columns:
            sample_weights = combined['_sample_weight'].copy()
            combined = combined.drop(columns=['_sample_weight'], errors='ignore')
        if '_from_experience' in combined.columns:
            combined = combined.drop(columns=['_from_experience'], errors='ignore')

        # Drop NaN rows
        valid_mask = combined[feature_cols + ['target']].notna().all(axis=1)
        valid = combined[valid_mask].reset_index(drop=True)
        if sample_weights is not None:
            sample_weights = sample_weights.loc[valid_mask].to_numpy()
        if len(valid) < self._min_samples:
            raise RuntimeError(
                f"Insufficient training samples: {len(valid)} < {self._min_samples}"
            )

        # Feature variance filtering: remove near-constant features (zero-variance = noise)
        X_raw = valid[feature_cols].values
        col_std = np.nanstd(X_raw, axis=0)
        variance_mask = col_std > 1e-8  # Keep features with meaningful variance
        feature_cols = [fc for fc, keep in zip(feature_cols, variance_mask) if keep]
        X = X_raw[:, variance_mask]

        y = valid['target'].values

        # Encode trading labels [-1,0,1] → model classes [0,1,2]
        y = DirectionEncoder.encode(y)

        # Temporal holdout: last 20% reserved for governance backtest (never seen during training)
        holdout_start_idx = int(len(X) * 0.80)

        # Preserve close prices for realistic OOS backtesting (not used as features)
        close_prices = valid['close'].values if 'close' in valid.columns else None

        return X, y, feature_cols, sample_weights, holdout_start_idx, close_prices

    def _train_model(
        self, feature_data: dict[str, pd.DataFrame]
    ) -> tuple[Any, dict, list[str], np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Train XGBoost model on prepared feature data.

        IMPORTANT: The final model is trained ONLY on the first 80% of data.
        The last 20% is reserved as a temporal holdout for governance backtest
        to ensure out-of-sample evaluation integrity.

        Returns:
            final_model: Trained XGBClassifier
            metrics: Training metrics dict
            feature_cols: Feature column names used
            X_holdout: Held-out feature matrix (for governance backtest)
            y_holdout: Held-out target vector (for governance backtest)
        """
        from xgboost import XGBClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score

        X, y, feature_cols, sample_weights, holdout_start_idx, close_prices = self._prepare_training_data(feature_data)

        # Split into train and holdout — holdout is NEVER used for model fitting
        X_train_all = X[:holdout_start_idx]
        y_train_all = y[:holdout_start_idx]
        w_train_all = sample_weights[:holdout_start_idx] if sample_weights is not None else None
        X_holdout = X[holdout_start_idx:]
        y_holdout = y[holdout_start_idx:]

        logger.info(
            "training_pipeline.data_split",
            train_samples=len(X_train_all),
            holdout_samples=len(X_holdout),
            holdout_pct=f"{len(X_holdout) / len(X):.1%}",
        )

        # Time-series cross-validation (on training portion only)
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        # ── Class-imbalance handling ──────────────────────────────────
        # Compute per-class sample weights to correct class imbalance.
        # This ensures the minority direction class (up/down) is not
        # overwhelmed by the majority (usually flat) in the loss function.
        from sklearn.utils.class_weight import compute_sample_weight
        class_sample_weights = compute_sample_weight("balanced", y_train_all)
        if w_train_all is not None:
            # Combine with any experience-enrichment weights
            w_train_all = w_train_all * class_sample_weights
        else:
            w_train_all = class_sample_weights

        # Report class distribution for monitoring
        unique_classes, class_counts = np.unique(y_train_all, return_counts=True)
        class_dist = dict(zip(unique_classes.tolist(), class_counts.tolist()))
        total = sum(class_counts)
        imbalance_ratio = max(class_counts) / max(min(class_counts), 1)
        logger.info(
            "training_pipeline.class_distribution",
            distribution=class_dist,
            imbalance_ratio=round(float(imbalance_ratio), 2),
        )

        for train_idx, val_idx in tscv.split(X_train_all):
            # Adaptive embargo: min(10% of val fold, 50 bars) — avoids destroying
            # early folds while still preventing autocorrelation leakage
            embargo_bars = min(max(len(val_idx) // 10, 5), 50)
            val_idx = val_idx[embargo_bars:]
            if len(val_idx) < 20:
                continue

            X_train, X_val = X_train_all[train_idx], X_train_all[val_idx]
            y_train, y_val = y_train_all[train_idx], y_train_all[val_idx]
            w_train = w_train_all[train_idx] if w_train_all is not None else None

            model = XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.02,
                subsample=0.7,
                colsample_bytree=0.6,
                min_child_weight=10,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.5,
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=42,
                verbosity=0,
                early_stopping_rounds=30,
            )
            model.fit(X_train, y_train, sample_weight=w_train,
                      eval_set=[(X_val, y_val)], verbose=False)
            score = accuracy_score(y_val, model.predict(X_val))
            cv_scores.append(score)

        # Train final model on training portion ONLY (excludes holdout).
        # Use early stopping against the holdout to prevent overfitting,
        # matching the regularisation applied during CV folds.
        # Reserve a small internal validation split (last 10% of train) for
        # early-stopping monitoring so the holdout stays truly unseen.
        es_split = int(len(X_train_all) * 0.90)
        X_train_es = X_train_all[:es_split]
        y_train_es = y_train_all[:es_split]
        X_val_es = X_train_all[es_split:]
        y_val_es = y_train_all[es_split:]
        w_train_es = w_train_all[:es_split] if w_train_all is not None else None

        final_model = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.02,
            subsample=0.7,
            colsample_bytree=0.6,
            min_child_weight=10,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.5,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            verbosity=0,
            early_stopping_rounds=30,
        )
        final_model.fit(
            X_train_es, y_train_es,
            sample_weight=w_train_es,
            eval_set=[(X_val_es, y_val_es)],
            verbose=False,
        )

        metrics = {
            "cv_accuracy": float(np.mean(cv_scores)) if cv_scores else 0.0,
            "cv_std": float(np.std(cv_scores)) if cv_scores else 0.0,
            "n_samples": len(X_train_all),
            "n_holdout_samples": len(X_holdout),
            "n_features": len(feature_cols),
            "n_symbols": len(feature_data),
            "class_distribution": {
                "up": int((y_train_all == DirectionEncoder.UP_CLASS).sum()),
                "flat": int((y_train_all == DirectionEncoder.FLAT_CLASS).sum()),
                "down": int((y_train_all == DirectionEncoder.DOWN_CLASS).sum()),
            },
            "class_imbalance_ratio": float(imbalance_ratio),
            "hyperparameters": {
                "n_estimators": 500,
                "max_depth": 4,
                "learning_rate": 0.02,
                "subsample": 0.7,
                "colsample_bytree": 0.6,
                "min_child_weight": 10,
                "gamma": 0.1,
                "reg_alpha": 0.1,
                "reg_lambda": 1.5,
                "early_stopping_rounds": 30,
            },
        }

        close_holdout = close_prices[holdout_start_idx:] if close_prices is not None else None

        return final_model, metrics, feature_cols, X_holdout, y_holdout, close_holdout

    def _backtest_model_oos(
        self,
        model,
        feature_cols: list[str],
        X_holdout: np.ndarray,
        y_holdout: np.ndarray,
        close_holdout: Optional[np.ndarray] = None,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
    ) -> dict:
        """
        Run out-of-sample backtest on held-out data the model NEVER trained on.

        Uses **actual close prices** when available to compute realistic
        trade returns including round-trip transaction costs and slippage.
        Falls back to a simplified simulation only when prices are absent.

        Args:
            model: Trained model (only trained on first 80% of data).
            feature_cols: Feature column names.
            X_holdout: Feature matrix from the temporal holdout (last 20%).
            y_holdout: True encoded targets from the holdout.
            close_holdout: Close price array aligned with X_holdout (optional).
            transaction_cost_bps: Round-trip transaction cost in basis points.
            slippage_bps: Estimated slippage per side in basis points.

        Returns:
            Dict with sharpe, max_drawdown, n_trades, monte_carlo results.
        """
        from backtesting.monte_carlo import monte_carlo_simulation

        hold_bars = 5  # Holding period per trade
        cost_per_trade = (transaction_cost_bps + 2 * slippage_bps) / 10_000.0

        # Handle NaN in holdout features
        nan_mask = np.isnan(X_holdout) | np.isinf(X_holdout)
        if nan_mask.any():
            X_holdout = np.where(nan_mask, 0.0, X_holdout)

        # Predict on holdout
        predictions = model.predict(X_holdout)
        decoded_predictions = DirectionEncoder.decode(predictions)

        use_prices = (close_holdout is not None and len(close_holdout) == len(X_holdout))

        # Generate NON-OVERLAPPING trades: after entering, skip forward by hold_bars
        all_trades = []
        i = 0
        while i < len(decoded_predictions) - hold_bars:
            signal = decoded_predictions[i]
            if signal == 0:  # Flat — no trade, advance one bar
                i += 1
                continue

            if use_prices:
                # ── Realistic price-based return ────────────────────────
                entry_price = close_holdout[i]
                exit_price = close_holdout[i + hold_bars]
                if entry_price > 0:
                    raw_return = signal * (exit_price / entry_price - 1.0)
                    trade_return = raw_return - cost_per_trade
                else:
                    trade_return = 0.0
            else:
                # ── Fallback: direction-correctness estimate ────────────
                true_direction = DirectionEncoder.decode(np.array([y_holdout[i]]))[0]
                if true_direction == signal:
                    trade_return = 0.005 * abs(signal) - cost_per_trade
                elif true_direction == 0:
                    trade_return = -cost_per_trade
                else:
                    trade_return = -0.005 * abs(signal) - cost_per_trade

            all_trades.append({
                "pnl": 1000.0 * trade_return,  # Notional $1000 per trade
                "return": trade_return,
                "signal": signal,
            })

            # Skip forward by hold_bars — ensures non-overlapping trades
            i += hold_bars

        # Compute metrics from independent trades
        n_trades = len(all_trades)
        if n_trades < 5:
            return {
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "n_trades": n_trades,
                "monte_carlo": {"probability_of_profit": 0.0, "median_return": 0.0, "p5_return": 0.0},
            }

        trade_returns = np.array([t["return"] for t in all_trades])
        mean_ret = np.mean(trade_returns)
        std_ret = np.std(trade_returns, ddof=1) if len(trade_returns) > 1 else 1.0

        # Correct annualization: each trade spans hold_bars periods.
        # With non-overlapping trades, there are (252 / hold_bars) trades per year.
        trades_per_year = 252.0 / hold_bars
        sharpe = float(mean_ret / std_ret * np.sqrt(trades_per_year)) if std_ret > 0 else 0.0

        # Max drawdown from sequential non-overlapping trade returns
        equity = np.cumprod(1.0 + trade_returns)
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / np.where(peak > 0, peak, 1.0)
        max_drawdown = float(abs(np.min(drawdowns))) if len(drawdowns) > 0 else 0.0

        # Monte Carlo simulation on independent trades
        mc_results = {"probability_of_profit": 0.0, "median_return": 0.0, "p5_return": 0.0}
        try:
            mc_results = monte_carlo_simulation(
                trades=all_trades,
                initial_cash=10000.0,
                n_simulations=500,
                seed=42,
            )
        except Exception as e:
            logger.debug("training_pipeline.monte_carlo_error", error=str(e))

        # Compute holdout accuracy for cross-check with CV accuracy
        holdout_accuracy = float(np.mean(predictions == y_holdout))

        logger.info(
            "training_pipeline.oos_backtest_complete",
            n_trades=n_trades,
            sharpe=round(sharpe, 3),
            max_drawdown=round(max_drawdown, 4),
            holdout_accuracy=round(holdout_accuracy, 3),
            mc_prob_profit=round(mc_results.get("probability_of_profit", 0.0), 3),
            annualization_factor=round(np.sqrt(trades_per_year), 2),
        )

        return {
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "n_trades": n_trades,
            "holdout_accuracy": holdout_accuracy,
            "monte_carlo": mc_results,
        }

    def _walk_forward_validation(
        self,
        feature_data: dict[str, pd.DataFrame],
        feature_cols: list[str],
        n_splits: int = 3,
    ) -> dict:
        """
        Genuine expanding-window walk-forward validation.

        For each split:
        1. Train a fresh model on data up to split boundary
        2. Predict on the next unseen segment
        3. Simulate non-overlapping trades on predicted segment

        This ensures predictions are ALWAYS out-of-sample and mimics
        how the model would perform if retrained periodically in production.

        Returns:
            Dict with walk-forward sharpe, total_return, and per-split details.
        """
        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score

        hold_bars = 5
        primary_symbol = next(iter(sorted(feature_data.keys())), "AAPL")
        try:
            from src.config.backtest_params import get_backtest_config
            commission_pct = float(get_backtest_config(primary_symbol).get("fees"))
        except Exception:
            commission_pct = 1.0 / 1000.0
        try:
            from config.settings import settings
            slippage_pct = float(settings.backtest_slippage_pct)
        except Exception:
            slippage_pct = 1.0 / 2000.0
        cost_per_trade = commission_pct + (2.0 * slippage_pct)

        X, y, _, sample_weights, _, close_prices = self._prepare_training_data(feature_data)

        # Divide into (n_splits + 1) temporal segments
        n_total = len(X)
        segment_size = n_total // (n_splits + 1)

        if segment_size < 100:
            logger.warning(
                "training_pipeline.walk_forward_insufficient_data",
                n_total=n_total,
                segment_size=segment_size,
            )
            return {"sharpe": 0.0, "total_return": 0.0, "n_trades": 0, "splits": []}

        all_wf_trades = []
        split_results = []

        for split_idx in range(n_splits):
            # Training: all data from start up to end of segment (split_idx + 1)
            train_end = segment_size * (split_idx + 1)
            # Purge gap: skip forward_bars (5) to prevent label leakage at boundary
            purge_gap = 5
            test_start = train_end + purge_gap
            test_end = min(train_end + segment_size, n_total)

            if test_end - test_start < 20:
                continue

            X_train_wf = X[:train_end]
            y_train_wf = y[:train_end]
            w_train_wf = sample_weights[:train_end] if sample_weights is not None else None
            X_test_wf = X[test_start:test_end]
            y_test_wf = y[test_start:test_end]
            close_test_wf = (
                close_prices[test_start:test_end]
                if close_prices is not None and len(close_prices) >= test_end
                else None
            )

            # Train fresh model on expanding window with early stopping
            # Use last 10% of training data as fold-local validation (respects temporal order)
            embargo_bars = 5  # Same as purge_gap to prevent leakage
            es_split = int(len(X_train_wf) * 0.90)
            X_train_fold = X_train_wf[:es_split]
            y_train_fold = y_train_wf[:es_split]
            w_train_fold = w_train_wf[:es_split] if w_train_wf is not None else None
            # Apply embargo gap between fold-local train and validation
            val_start = es_split + embargo_bars
            X_val_fold = X_train_wf[val_start:]
            y_val_fold = y_train_wf[val_start:]

            # If validation set is too small after embargo, use full training without early stopping
            use_early_stopping = len(X_val_fold) >= 20

            wf_model = XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.02,
                subsample=0.7,
                colsample_bytree=0.6,
                min_child_weight=10,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.5,
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=42,
                verbosity=0,
                early_stopping_rounds=30 if use_early_stopping else None,
            )
            if use_early_stopping:
                wf_model.fit(
                    X_train_fold, y_train_fold, sample_weight=w_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)], verbose=False,
                )
            else:
                wf_model.fit(X_train_wf, y_train_wf, sample_weight=w_train_wf, verbose=False)

            # Predict on unseen segment
            preds = wf_model.predict(X_test_wf)
            decoded = DirectionEncoder.decode(preds)
            accuracy = float(accuracy_score(y_test_wf, preds))

            # Simulate non-overlapping trades
            i = 0
            split_trades = []
            while i < len(decoded) - hold_bars:
                signal = decoded[i]
                if signal == 0:
                    i += 1
                    continue

                if close_test_wf is not None:
                    entry_price = close_test_wf[i]
                    exit_price = close_test_wf[i + hold_bars]
                    if entry_price > 0:
                        raw_return = signal * (exit_price / entry_price - 1.0)
                        trade_return = float(raw_return - cost_per_trade)
                    else:
                        trade_return = 0.0
                else:
                    true_dir = DirectionEncoder.decode(np.array([y_test_wf[i]]))[0]
                    if true_dir == signal:
                        trade_return = 0.005 * abs(signal) - cost_per_trade
                    elif true_dir == 0:
                        trade_return = -cost_per_trade
                    else:
                        trade_return = -0.005 * abs(signal) - cost_per_trade

                split_trades.append({"return": trade_return, "signal": signal})
                i += hold_bars

            all_wf_trades.extend(split_trades)
            best_iteration = getattr(wf_model, 'best_iteration', None) if use_early_stopping else None
            split_results.append({
                "split": split_idx,
                "train_size": len(X_train_wf),
                "test_size": len(X_test_wf),
                "accuracy": accuracy,
                "n_trades": len(split_trades),
                "best_iteration": best_iteration,
                "early_stopping_used": use_early_stopping,
            })

        # Compute walk-forward metrics from all out-of-sample trades
        if len(all_wf_trades) < 5:
            wf_sharpe = 0.0
            wf_total_return = 0.0
        else:
            wf_returns = np.array([t["return"] for t in all_wf_trades])
            wf_mean = np.mean(wf_returns)
            wf_std = np.std(wf_returns, ddof=1) if len(wf_returns) > 1 else 1.0
            trades_per_year = 252.0 / hold_bars
            wf_sharpe = float(wf_mean / wf_std * np.sqrt(trades_per_year)) if wf_std > 0 else 0.0
            wf_total_return = float(np.prod(1.0 + wf_returns) - 1.0)

        logger.info(
            "training_pipeline.walk_forward_complete",
            n_splits=n_splits,
            total_wf_trades=len(all_wf_trades),
            wf_sharpe=round(wf_sharpe, 3),
            wf_total_return=round(wf_total_return, 4),
            split_details=split_results,
        )

        return {
            "sharpe": wf_sharpe,
            "total_return": wf_total_return,
            "n_trades": len(all_wf_trades),
            "splits": split_results,
        }

    # Legacy compatibility wrapper — kept for any external callers
    def _backtest_model(
        self,
        model,
        feature_cols: list[str],
        feature_data: dict[str, pd.DataFrame],
    ) -> dict:
        """
        DEPRECATED: Legacy backtest method retained for backward compatibility.

        New code should use _backtest_model_oos() with proper holdout data.
        This wrapper simulates the old interface by splitting feature_data internally,
        but uses non-overlapping trades and correct Sharpe annualization.
        """
        logger.warning(
            "training_pipeline.legacy_backtest_called",
            msg="Using deprecated _backtest_model — prefer _backtest_model_oos",
        )
        from backtesting.monte_carlo import monte_carlo_simulation

        hold_bars = 5
        all_trades = []

        for symbol, df in feature_data.items():
            try:
                # Use last 20% as backtest window
                n = len(df)
                split_idx = int(n * 0.8)
                test_df = df.iloc[split_idx:].copy()

                if len(test_df) < 20:
                    continue

                # Get features for prediction
                available_cols = [c for c in feature_cols if c in test_df.columns]
                if len(available_cols) < len(feature_cols) * 0.8:
                    continue

                X_test = test_df[available_cols].values
                nan_mask = np.isnan(X_test) | np.isinf(X_test)
                if nan_mask.any():
                    X_test = np.where(nan_mask, 0.0, X_test)

                predictions = model.predict(X_test)
                decoded_predictions = DirectionEncoder.decode(predictions)

                # NON-OVERLAPPING trades (fixed from original)
                close_prices = test_df['close'].values
                i = 0
                while i < len(decoded_predictions) - hold_bars:
                    signal = decoded_predictions[i]
                    if signal == 0:
                        i += 1
                        continue

                    entry_price = close_prices[i]
                    exit_price = close_prices[i + hold_bars]

                    if entry_price <= 0:
                        i += hold_bars
                        continue

                    trade_return = (exit_price / entry_price - 1) * signal
                    pnl = entry_price * trade_return

                    all_trades.append({
                        "pnl": pnl,
                        "return": trade_return,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "symbol": symbol,
                    })
                    i += hold_bars  # Skip to after exit
            except Exception as e:
                logger.debug("training_pipeline.backtest_symbol_error", symbol=symbol, error=str(e))
                continue

        n_trades = len(all_trades)
        if n_trades < 5:
            return {
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "n_trades": n_trades,
                "walk_forward": {"sharpe": 0.0, "total_return": 0.0},
                "monte_carlo": {"probability_of_profit": 0.0, "median_return": 0.0, "p5_return": 0.0},
            }

        trade_returns = np.array([t["return"] for t in all_trades])
        mean_ret = np.mean(trade_returns)
        std_ret = np.std(trade_returns, ddof=1) if len(trade_returns) > 1 else 1.0

        # Correct annualization for non-overlapping N-bar trades
        trades_per_year = 252.0 / hold_bars
        sharpe = float(mean_ret / std_ret * np.sqrt(trades_per_year)) if std_ret > 0 else 0.0

        equity = np.cumprod(1.0 + trade_returns)
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / np.where(peak > 0, peak, 1.0)
        max_drawdown = float(abs(np.min(drawdowns))) if len(drawdowns) > 0 else 0.0

        mc_results = {"probability_of_profit": 0.0, "median_return": 0.0, "p5_return": 0.0}
        try:
            mc_results = monte_carlo_simulation(
                trades=all_trades,
                initial_cash=10000.0,
                n_simulations=500,
                seed=42,
            )
        except Exception as e:
            logger.debug("training_pipeline.monte_carlo_error", error=str(e))

        logger.info(
            "training_pipeline.backtest_complete",
            n_trades=n_trades,
            sharpe=round(sharpe, 3),
            max_drawdown=round(max_drawdown, 4),
            mc_prob_profit=round(mc_results.get("probability_of_profit", 0.0), 3),
        )

        return {
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "n_trades": n_trades,
            "walk_forward": {"sharpe": sharpe, "total_return": float(np.prod(1.0 + trade_returns) - 1.0)},
            "monte_carlo": mc_results,
        }

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
            # Find the previous version (second-to-last, or the one before active)
            versions = self._registry.list_versions()
            if len(versions) < 2:
                logger.warning("training_pipeline.rollback_no_previous", reason=reason)
                return False

            active_version = self._registry.get_active_version()
            previous_version = None
            for v in versions:
                if v["version"] != active_version:
                    previous_version = v["version"]
                    break

            if not previous_version:
                logger.warning("training_pipeline.rollback_no_candidate", reason=reason)
                return False

            success = self._registry.rollback(previous_version)
            if success:
                # Publish rollback event
                if self._event_bus:
                    from src.core.events import ModelAutoRollback
                    self._event_bus.publish(ModelAutoRollback(
                        from_version=active_version or "",
                        to_version=previous_version,
                        reason=reason,
                        source="training_pipeline",
                    ))
                logger.info(
                    "training_pipeline.auto_rollback",
                    from_version=active_version,
                    to_version=previous_version,
                    reason=reason,
                )
                return True
        except Exception as e:
            logger.error("training_pipeline.rollback_failed", error=str(e))
        return False

    def _raise_if_stopping(self) -> None:
        if self._stop_event.is_set():
            raise RuntimeError("training_pipeline.stop_requested")

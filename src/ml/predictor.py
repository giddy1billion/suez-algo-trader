"""
Model Predictor — Centralized inference service with transparent hot-swap.

Provides:
- Single point of prediction for all ML strategies
- Automatic model reload when registry active version changes
- Thread-safe model access during hot-swap
- Version tracking and rollback capability
- Prediction latency monitoring
- Shadow mode for testing new models without affecting trading
"""

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from src.ml.model_registry import ModelRegistry
from src.ml.label_encoder import DirectionEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelPredictor:
    """
    Centralized model inference service with hot-swap.

    All ML strategies route predictions through this service.
    When the active model version changes in the registry, the
    predictor transparently reloads without interrupting predictions.

    Features:
    - Zero-downtime model switching via double-buffered loading
    - Prediction latency tracking
    - Fallback to previous model on load failure
    - Shadow predictions (run new model without using results)
    """

    def __init__(
        self,
        registry: ModelRegistry,
        event_bus=None,
        auto_reload: bool = True,
        check_interval_seconds: float = 30.0,
        feature_store=None,
        dataset_registry=None,
    ):
        self._registry = registry
        self._event_bus = event_bus
        self._auto_reload = auto_reload
        self._check_interval = check_interval_seconds
        self._feature_store = feature_store
        self._dataset_registry = dataset_registry

        # Model state (double-buffered for zero-downtime swap)
        self._model = None
        self._features: list[str] = []
        self._version: Optional[str] = None
        self._loaded_at: Optional[datetime] = None
        self._lock = threading.RLock()

        # Shadow model for A/B testing
        self._shadow_model = None
        self._shadow_version: Optional[str] = None
        self._shadow_features: list[str] = []

        # Metrics
        self._prediction_count: int = 0
        self._total_latency_ms: float = 0.0
        self._swap_count: int = 0
        self._last_check_time: Optional[float] = None
        self._errors: list[dict] = []
        self._max_errors: int = 100  # Cap to prevent memory leak

        # Auto-reload watcher
        self._watcher_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Load initial model (under lock to avoid race with watcher thread)
        with self._lock:
            self._load_active_model()

        # Start watcher if auto-reload enabled
        if auto_reload:
            self._start_watcher()

    # ──────────────────────────────────────────────────────────────────────
    # Public Prediction API
    # ──────────────────────────────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Get prediction from the active model.

        Transparently reloads the model if the active version has changed.
        Thread-safe — multiple strategies can call this concurrently.

        Args:
            features: Feature array (samples x features).

        Returns:
            Prediction array from the model.

        Raises:
            RuntimeError: If no model is loaded and loading fails.
        """
        start_time = time.perf_counter()

        with self._lock:
            # Check if we need a version refresh
            self._maybe_check_version()

            if self._model is None:
                raise RuntimeError("No model loaded. Train or register a model first.")

            # Validate features before prediction
            if features.ndim == 1:
                features = features.reshape(1, -1)

            if hasattr(self._model, 'n_features_in_'):
                expected = self._model.n_features_in_
                actual = features.shape[1]
                if actual != expected:
                    raise ValueError(
                        f"Feature count mismatch: model expects {expected}, got {actual}"
                    )

            # Check for NaN/inf
            nan_count = np.isnan(features).sum()
            if nan_count > 0:
                inf_count = np.isinf(features).sum()
                if nan_count == features.size:
                    raise ValueError("All features are NaN - cannot predict")
                logger.warning("predictor.nan_features", nan_count=int(nan_count), inf_count=int(inf_count))
                features = np.where(np.isnan(features) | np.isinf(features), 0.0, features)

            try:
                predictions = self._model.predict(features)
                self._prediction_count += 1
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self._total_latency_ms += elapsed_ms
                return predictions
            except Exception as e:
                self._errors.append({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "error": str(e),
                    "version": self._version,
                })
                if len(self._errors) > self._max_errors:
                    self._errors = self._errors[-self._max_errors:]
                raise

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get probability predictions from the active model.

        Args:
            features: Feature array (samples x features).

        Returns:
            Probability array (samples x classes).

        Raises:
            RuntimeError: If model does not support predict_proba.
        """
        start_time = time.perf_counter()

        with self._lock:
            self._maybe_check_version()

            if self._model is None:
                raise RuntimeError("No model loaded.")

            if not hasattr(self._model, 'predict_proba'):
                raise RuntimeError(f"Model {self._version} doesn't support predict_proba")

            # Validate features before prediction
            if features.ndim == 1:
                features = features.reshape(1, -1)

            if hasattr(self._model, 'n_features_in_'):
                expected = self._model.n_features_in_
                actual = features.shape[1]
                if actual != expected:
                    raise ValueError(
                        f"Feature count mismatch: model expects {expected}, got {actual}"
                    )

            # Check for NaN/inf
            nan_count = np.isnan(features).sum()
            if nan_count > 0:
                inf_count = np.isinf(features).sum()
                if nan_count == features.size:
                    raise ValueError("All features are NaN - cannot predict")
                logger.warning("predictor.nan_features", nan_count=int(nan_count), inf_count=int(inf_count))
                features = np.where(np.isnan(features) | np.isinf(features), 0.0, features)

            try:
                proba = self._model.predict_proba(features)
                self._prediction_count += 1
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self._total_latency_ms += elapsed_ms

                # Snapshot features for reproducibility (non-blocking)
                if self._feature_store:
                    try:
                        self._feature_store.snapshot_prediction_features(
                            features=features,
                            model_version=self._version,
                            prediction_id=self._prediction_count,
                        )
                    except Exception:
                        pass  # Feature store failures must not block predictions

                return proba
            except Exception as e:
                self._errors.append({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "error": str(e),
                    "version": self._version,
                })
                if len(self._errors) > self._max_errors:
                    self._errors = self._errors[-self._max_errors:]
                raise

    def predict_with_shadow(self, features: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get predictions from both active and shadow models.

        Used for A/B testing -- shadow predictions are returned but not
        used for trading decisions.

        Args:
            features: Feature array.

        Returns:
            Tuple of (active_predictions, shadow_predictions or None).
        """
        active_pred = self.predict(features)

        shadow_pred = None
        if self._shadow_model is not None:
            try:
                with self._lock:
                    shadow_pred = self._shadow_model.predict(features)
            except Exception as e:
                logger.debug("predictor.shadow_failed", error=str(e))

        return active_pred, shadow_pred

    def predict_with_lineage(
        self,
        features: np.ndarray,
        symbol: str = "unknown",
        strategy_name: str = "",
        bar_timestamp: str = "",
        market_session: str = "",
    ) -> tuple:
        """
        predict_proba + full lineage recording.

        Returns:
            (proba_array, lineage_dict) where lineage_dict contains prediction_id
            and all provenance metadata for downstream consumers.
        """
        proba = self.predict_proba(features)

        lineage = {
            "prediction_id": f"pred_{self._prediction_count}",
            "model_version": self._version or "",
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
            "bar_timestamp": bar_timestamp,
            "market_session": market_session,
            "symbol": symbol,
            "strategy": strategy_name,
            "predicted_direction": DirectionEncoder.class_name(int(np.argmax(proba[0]))),
            "predicted_confidence": float(np.max(proba[0])),
            "feature_snapshot_id": "",
            "git_commit": self._get_git_commit(),
        }

        # Record in dataset registry
        if self._dataset_registry:
            try:
                self._dataset_registry.record_prediction(
                    prediction_id=lineage["prediction_id"],
                    model_version=lineage["model_version"],
                    feature_version_id=f"fv_{lineage['model_version']}",
                    feature_snapshot_id=lineage["feature_snapshot_id"],
                    symbol=symbol,
                    predicted_direction=lineage["predicted_direction"],
                    predicted_confidence=lineage["predicted_confidence"],
                )
            except Exception:
                pass  # Lineage failures must not block predictions

        return proba, lineage

    def _get_git_commit(self) -> str:
        """Get current git commit hash (cached)."""
        if not hasattr(self, '_cached_git_commit'):
            try:
                import subprocess
                self._cached_git_commit = subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
            except Exception:
                self._cached_git_commit = ""
        return self._cached_git_commit

    # ──────────────────────────────────────────────────────────────────────
    # Model Management
    # ──────────────────────────────────────────────────────────────────────

    def swap_model(self, version: str) -> dict:
        """
        Explicitly swap to a specific model version.

        Args:
            version: Version string to load (e.g., "v003").

        Returns:
            Dict with swap details.

        Raises:
            FileNotFoundError: If version doesn't exist.
        """
        with self._lock:
            old_version = self._version
            try:
                self._load_version(version)
            except Exception as e:
                logger.error(
                    "predictor.swap_failed",
                    target_version=version,
                    current_version=old_version,
                    error=str(e),
                )
                self._errors.append({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "error": f"swap_failed: {e}",
                    "version": version,
                })
                if len(self._errors) > self._max_errors:
                    self._errors = self._errors[-self._max_errors:]
                raise
            self._swap_count += 1

            result = {
                "old_version": old_version,
                "new_version": version,
                "swap_count": self._swap_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Publish event
            if self._event_bus:
                from src.core.events import ModelSwapped
                self._event_bus.publish(ModelSwapped(
                    old_version=old_version or "",
                    new_version=version,
                    strategy="global",
                    reason="explicit_swap",
                    source="model_predictor",
                ))

            logger.info("predictor.model_swapped", **result)
            return result

    def set_shadow_model(self, version: str) -> None:
        """
        Set a shadow model for A/B comparison.

        The shadow model's predictions are available via predict_with_shadow()
        but don't affect trading.

        Args:
            version: Version string to use as shadow.
        """
        with self._lock:
            try:
                data = self._registry.get_version(version)
                self._shadow_model = data["model"]
                self._shadow_version = version
                self._shadow_features = data.get("features", [])
                logger.info("predictor.shadow_set", version=version)
            except Exception as e:
                logger.error("predictor.shadow_failed", version=version, error=str(e))
                raise

    def clear_shadow_model(self) -> None:
        """Remove the shadow model."""
        with self._lock:
            self._shadow_model = None
            self._shadow_version = None
            self._shadow_features = []

    def reload(self) -> bool:
        """Force reload the active model from registry."""
        with self._lock:
            try:
                self._load_active_model()
                return True
            except Exception as e:
                logger.error("predictor.reload_failed", error=str(e))
                return False

    # ──────────────────────────────────────────────────────────────────────
    # Status & Metrics
    # ──────────────────────────────────────────────────────────────────────

    @property
    def current_version(self) -> Optional[str]:
        return self._version

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def feature_names(self) -> list[str]:
        return self._features.copy()

    @property
    def shadow_version(self) -> Optional[str]:
        return self._shadow_version

    def get_metrics(self) -> dict:
        """Get prediction service metrics."""
        avg_latency = (
            self._total_latency_ms / self._prediction_count
            if self._prediction_count > 0 else 0.0
        )
        return {
            "version": self._version,
            "loaded_at": self._loaded_at.isoformat() if self._loaded_at else None,
            "prediction_count": self._prediction_count,
            "avg_latency_ms": round(avg_latency, 2),
            "swap_count": self._swap_count,
            "shadow_version": self._shadow_version,
            "error_count": len(self._errors),
            "auto_reload": self._auto_reload,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def stop(self):
        """Stop the auto-reload watcher."""
        self._stop_event.set()
        if self._watcher_thread and self._watcher_thread.is_alive():
            self._watcher_thread.join(timeout=5.0)
            if self._watcher_thread.is_alive():
                logger.warning("predictor.stop_timeout", timeout=5.0)

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _load_active_model(self):
        """
        Load the currently active model from registry.

        Handles gracefully:
        - No registry entries (recovery attempt)
        - Registry references missing files (stale entries from deploy)
        - No model files at all (waits for bootstrap training)
        """
        active_version = self._registry.get_active_version()
        if active_version is None:
            # Attempt self-healing: trigger registry recovery if model files exist
            self._registry._recover_orphaned_models()
            active_version = self._registry.get_active_version()

        if active_version is not None and active_version != self._version:
            try:
                self._load_version(active_version)
                return
            except (FileNotFoundError, KeyError) as e:
                # Registry references a model file that doesn't exist on disk.
                # Common in containerized deployments where .joblib is gitignored.
                # Prune the stale entry and fall through to fallback.
                logger.warning(
                    "predictor.registry_stale_entry",
                    version=active_version,
                    error=str(e),
                    msg="Registry references missing model file; pruning entry",
                )
                self._registry._prune_missing_entries()
                active_version = None
            except Exception as e:
                logger.error("predictor.load_failed_unexpected", version=active_version, error=str(e))
                active_version = None
        elif active_version == self._version:
            return  # Already loaded

        # Fallback: load latest_model.joblib directly
        if active_version is None and self._model is None:
            latest_path = self._registry.latest_path
            if os.path.exists(latest_path):
                try:
                    import joblib
                    artifact = joblib.load(latest_path)
                    self._model = artifact["model"]
                    self._features = artifact.get("features", [])
                    self._version = "v000_fallback"
                    self._loaded_at = datetime.now(timezone.utc)
                    logger.info(
                        "predictor.loaded_from_fallback",
                        path=latest_path,
                        n_features=len(self._features),
                    )
                    return
                except Exception as e:
                    logger.error("predictor.fallback_load_failed", error=str(e))

            # No model available at all — not an error, just awaiting training
            logger.info(
                "predictor.awaiting_model",
                msg="No trained model available. System will use fallback strategy until auto-train completes.",
            )

    def _load_version(self, version: str):
        """Load a specific version into the predictor."""
        old_model = self._model
        old_version = self._version

        try:
            data = self._registry.get_version(version)
            self._model = data["model"]
            self._features = data.get("features", [])
            self._version = version
            self._loaded_at = datetime.now(timezone.utc)

            logger.info(
                "predictor.model_loaded",
                version=version,
                n_features=len(self._features),
            )
        except Exception as e:
            # Rollback on failure
            logger.error("predictor.load_failed", version=version, error=str(e))
            self._model = old_model
            self._version = old_version
            raise

    def _maybe_check_version(self):
        """Check if version needs refresh (rate-limited)."""
        if not self._auto_reload:
            return

        now = time.monotonic()
        if self._last_check_time and (now - self._last_check_time) < self._check_interval:
            return

        self._last_check_time = now
        active = self._registry.get_active_version()
        if active and active != self._version:
            logger.info(
                "predictor.version_drift_detected",
                current=self._version,
                registry_active=active,
            )
            old_version = self._version
            try:
                self._load_version(active)
                self._swap_count += 1
                logger.info(
                    "predictor.auto_reload_success",
                    old_version=old_version,
                    new_version=active,
                    swap_count=self._swap_count,
                )
                if self._event_bus:
                    from src.core.events import ModelSwapped
                    self._event_bus.publish(ModelSwapped(
                        old_version=old_version or "",
                        new_version=active,
                        strategy="global",
                        reason="auto_reload",
                        source="model_predictor",
                    ))
            except Exception as e:
                self._errors.append({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "error": f"auto_swap_failed: {e}",
                    "version": active,
                })
                if len(self._errors) > self._max_errors:
                    self._errors = self._errors[-self._max_errors:]
                logger.error(
                    "predictor.auto_swap_failed",
                    target_version=active,
                    current_version=self._version,
                    error=str(e),
                    error_count=len(self._errors),
                )

    def _start_watcher(self):
        """Start background thread that watches for model version changes."""
        def _watch():
            while not self._stop_event.is_set():
                try:
                    with self._lock:
                        self._maybe_check_version()
                except Exception as e:
                    logger.debug("predictor.watcher_error", error=str(e))
                self._stop_event.wait(timeout=self._check_interval)

        self._watcher_thread = threading.Thread(
            target=_watch,
            name="model-predictor-watcher",
            daemon=True,
        )
        self._watcher_thread.start()

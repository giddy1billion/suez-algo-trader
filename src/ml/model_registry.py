"""
Model Registry — versioning, comparison, and rollback for ML trading models.

Provides atomic persistence, thread-safe writes, and full version history
so that model deployments can be audited, compared, and rolled back.
"""

import json
import os
import shutil
import tempfile
import threading
from datetime import datetime
from typing import Optional

import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)

_registry_lock = threading.RLock()


class ModelRegistry:
    """
    Manages versioned ML model artifacts with metadata tracking.

    Models are saved as joblib files in the models directory with a
    registry.json manifest that tracks all versions and their metadata.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.registry_path = os.path.join(models_dir, "registry.json")
        self.latest_path = os.path.join(models_dir, "latest_model.joblib")
        os.makedirs(models_dir, exist_ok=True)

        # Self-heal: recover orphaned models if registry is missing
        if not os.path.exists(self.registry_path):
            self._recover_orphaned_models()

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def save_version(
        self,
        model,
        features: list,
        metrics: dict,
        symbols: list,
        note: str = "",
        activate: bool = True,
    ) -> str:
        """
        Save a new model version to the registry.

        Args:
            model: Trained model instance (e.g. XGBClassifier).
            features: List of feature column names used by the model.
            metrics: Training/validation metrics dict.
            symbols: List of symbols the model was trained on.
            note: Optional human-readable note for this version.
            activate: If True, immediately mark as active (predictor will load it).
                      If False, model is registered but NOT active — governance
                      must explicitly promote it via set_active_version().

        Returns:
            Version string, e.g. "v003".
        """
        with _registry_lock:
            registry = self._load_registry()
            version_num = len(registry) + 1
            version_str = f"v{version_num:03d}"
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{version_str}_{timestamp_str}.joblib"
            filepath = os.path.join(self.models_dir, filename)

            # Save model artifact
            artifact = {
                "model": model,
                "features": features,
                "trained_at": timestamp,
            }
            joblib.dump(artifact, filepath)

            # Only update latest_model.joblib when governance approves activation.
            # This prevents rejected models from being picked up by the predictor's
            # fallback path (_load_active_model → latest_model.joblib).
            if activate:
                shutil.copy2(filepath, self.latest_path)

            if activate:
                # Mark all previous versions as inactive
                for entry in registry:
                    entry["is_active"] = False

            # Build metadata entry
            entry = {
                "version": version_str,
                "filename": filename,
                "trained_at": timestamp.isoformat(),
                "metrics": metrics,
                "symbols": symbols,
                "n_features": len(features),
                "n_samples": metrics.get("n_samples", 0),
                "note": note,
                "is_active": activate,
            }
            registry.append(entry)

            self._save_registry(registry)
            logger.info(
                "ml.registry.version_saved",
                version=version_str,
                activated=activate,
                metrics=metrics,
            )
            return version_str

    def set_active_version(self, version_str: str) -> bool:
        """
        Explicitly promote a registered version to active status.

        Only governance-approved models should be activated. The predictor's
        auto-reload watcher polls get_active_version() — this is the gate.

        Args:
            version_str: Version to activate (e.g. "v003").

        Returns:
            True if version was found and activated, False otherwise.
        """
        with _registry_lock:
            registry = self._load_registry()
            found = False
            for entry in registry:
                if entry["version"] == version_str:
                    entry["is_active"] = True
                    found = True
                else:
                    entry["is_active"] = False

            if found:
                self._save_registry(registry)
                # Update latest_model.joblib to match the activated version
                entry = self._find_entry(registry, version_str)
                filepath = os.path.join(self.models_dir, entry["filename"])
                if os.path.exists(filepath):
                    shutil.copy2(filepath, self.latest_path)
                logger.info("ml.registry.version_activated", version=version_str)
            else:
                logger.warning("ml.registry.activate_not_found", version=version_str)
            return found

    def list_versions(self) -> list[dict]:
        """
        Return all registered versions, newest first.

        Each entry contains: version, trained_at, metrics, symbols,
        is_active, filename.
        """
        registry = self._load_registry()
        return list(reversed(registry))

    def get_version(self, version_str: str) -> dict:
        """
        Load a specific version's model and metadata.

        Args:
            version_str: Version identifier, e.g. "v003".

        Returns:
            Dict with keys: model, features, metadata.

        Raises:
            FileNotFoundError: If the version file is missing.
            KeyError: If the version is not in the registry.
        """
        registry = self._load_registry()
        entry = self._find_entry(registry, version_str)

        filepath = os.path.join(self.models_dir, entry["filename"])
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Model file not found: {filepath}"
            )

        artifact = joblib.load(filepath)
        return {
            "model": artifact["model"],
            "features": artifact["features"],
            "metadata": entry,
        }

    def rollback(self, version_str: str) -> bool:
        """
        Roll back to a previous model version.

        Copies the specified version to latest_model.joblib and marks it
        as the active version in the registry.

        Args:
            version_str: Version to activate, e.g. "v001".

        Returns:
            True on success, False on failure.
        """
        with _registry_lock:
            try:
                registry = self._load_registry()
                entry = self._find_entry(registry, version_str)

                filepath = os.path.join(self.models_dir, entry["filename"])
                if not os.path.exists(filepath):
                    logger.error(
                        "ml.registry.rollback_failed",
                        version=version_str,
                        reason="file_missing",
                    )
                    return False

                shutil.copy2(filepath, self.latest_path)

                for e in registry:
                    e["is_active"] = (e["version"] == version_str)

                self._save_registry(registry)
                logger.info("ml.registry.rollback", version=version_str)
                return True
            except (KeyError, FileNotFoundError) as exc:
                logger.error(
                    "ml.registry.rollback_failed",
                    version=version_str,
                    error=str(exc),
                )
                return False

    def compare_versions(self, v1: str, v2: str) -> dict:
        """
        Compare two model versions.

        Args:
            v1: First version string.
            v2: Second version string.

        Returns:
            Dict with metrics_diff, training_date_diff, and feature_changes.
        """
        registry = self._load_registry()
        entry1 = self._find_entry(registry, v1)
        entry2 = self._find_entry(registry, v2)

        # Metrics diff
        m1 = entry1.get("metrics", {})
        m2 = entry2.get("metrics", {})
        all_keys = set(m1.keys()) | set(m2.keys())
        metrics_diff = {}
        for k in sorted(all_keys):
            val1 = m1.get(k)
            val2 = m2.get(k)
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                metrics_diff[k] = {"v1": val1, "v2": val2, "diff": val2 - val1}
            else:
                metrics_diff[k] = {"v1": val1, "v2": val2}

        # Training date diff
        t1 = datetime.fromisoformat(entry1["trained_at"])
        t2 = datetime.fromisoformat(entry2["trained_at"])
        training_date_diff = {
            "v1_trained_at": entry1["trained_at"],
            "v2_trained_at": entry2["trained_at"],
            "diff_hours": round((t2 - t1).total_seconds() / 3600, 2),
        }

        # Feature changes
        features1 = set(self._get_features_for_entry(entry1))
        features2 = set(self._get_features_for_entry(entry2))
        feature_changes = {
            "added": sorted(features2 - features1),
            "removed": sorted(features1 - features2),
            "unchanged_count": len(features1 & features2),
        }

        return {
            "v1": v1,
            "v2": v2,
            "metrics_diff": metrics_diff,
            "training_date_diff": training_date_diff,
            "feature_changes": feature_changes,
        }

    def get_active_version(self) -> Optional[str]:
        """
        Return the currently active version string, or None if no
        versions are registered.
        """
        registry = self._load_registry()
        for entry in reversed(registry):
            if entry.get("is_active"):
                return entry["version"]
        return None

    def delete_version(self, version_str: str) -> bool:
        """
        Delete a model version (file + registry entry).

        Cannot delete the currently active version.

        Args:
            version_str: Version to delete.

        Returns:
            True on success, False on failure.
        """
        with _registry_lock:
            try:
                registry = self._load_registry()
                entry = self._find_entry(registry, version_str)

                if entry.get("is_active"):
                    logger.warning(
                        "ml.registry.delete_refused",
                        version=version_str,
                        reason="cannot_delete_active",
                    )
                    return False

                # Remove file
                filepath = os.path.join(self.models_dir, entry["filename"])
                if os.path.exists(filepath):
                    os.remove(filepath)

                # Remove from registry
                registry = [e for e in registry if e["version"] != version_str]
                self._save_registry(registry)

                logger.info("ml.registry.deleted", version=version_str)
                return True
            except (KeyError, OSError) as exc:
                logger.error(
                    "ml.registry.delete_failed",
                    version=version_str,
                    error=str(exc),
                )
                return False

    def auto_version_check(self, new_metrics: dict, threshold: float = 0.05) -> bool:
        """
        Determine if a new model should replace the current active model.

        Compares cv_accuracy (or first numeric metric) of the new model
        against the active version. Returns True if the new model is
        better by at least `threshold` relative improvement, or if there
        is no active model.

        Args:
            new_metrics: Metrics dict from the new candidate model.
            threshold: Minimum relative improvement required (default 5%).

        Returns:
            True if the new model should replace the active one.
        """
        active_version = self.get_active_version()
        if active_version is None:
            return True

        registry = self._load_registry()
        try:
            active_entry = self._find_entry(registry, active_version)
        except KeyError:
            return True

        active_metrics = active_entry.get("metrics", {})

        # Compare on cv_accuracy first, then fall back to first numeric key
        compare_key = self._pick_comparison_key(active_metrics, new_metrics)
        if compare_key is None:
            return True  # No comparable metric — allow upgrade

        old_val = active_metrics[compare_key]
        new_val = new_metrics.get(compare_key)

        if not isinstance(old_val, (int, float)) or not isinstance(new_val, (int, float)):
            return True

        if old_val == 0:
            return new_val > 0

        improvement = (new_val - old_val) / abs(old_val)
        return improvement >= threshold

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _load_registry(self) -> list[dict]:
        """Load registry.json, returning empty list on missing/corrupt file."""
        if not os.path.exists(self.registry_path):
            return []
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            logger.warning("ml.registry.corrupt", reason="not_a_list")
            return []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("ml.registry.corrupt", error=str(exc))
            return []

    def _save_registry(self, registry: list[dict]) -> None:
        """Atomically write registry.json (write-to-temp then rename)."""
        with _registry_lock:
            fd, tmp_path = tempfile.mkstemp(
                suffix=".json",
                prefix="registry_",
                dir=self.models_dir,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(registry, f, indent=2, default=str)
                # Atomic replace (Windows: need to remove target first)
                if os.path.exists(self.registry_path):
                    os.replace(tmp_path, self.registry_path)
                else:
                    os.rename(tmp_path, self.registry_path)
            except Exception as e:
                logger.error("ml.registry.save_failed", error=str(e))
                # Clean up temp file on failure
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

    def _find_entry(self, registry: list[dict], version_str: str) -> dict:
        """Find a registry entry by version string. Raises KeyError if missing."""
        for entry in registry:
            if entry["version"] == version_str:
                return entry
        raise KeyError(f"Version not found in registry: {version_str}")

    def _recover_orphaned_models(self) -> None:
        """
        Auto-recover registry.json from orphaned model files on disk.

        If the registry manifest is missing but versioned .joblib files exist,
        reconstruct the registry and mark the newest as active.
        Also includes latest_model.joblib if no versioned files found.
        This prevents the predictor.no_active_version warning after data loss.
        """
        with _registry_lock:
            # Double-check under lock (another thread may have recovered already)
            if os.path.exists(self.registry_path):
                return

            model_files = sorted([
                f for f in os.listdir(self.models_dir)
                if f.startswith("v") and f.endswith(".joblib")
            ])

            # If no versioned files but latest_model.joblib exists, include it
            if not model_files and os.path.exists(self.latest_path):
                model_files = ["latest_model.joblib"]

            if not model_files:
                self._cleanup_temp_files()
                return

            logger.warning(
                "ml.registry.recovering_orphaned_models",
                count=len(model_files),
            )

            registry = []
            for i, filename in enumerate(model_files, 1):
                # Parse trained_at from filename pattern: v001_20260703_110304.joblib
                parts = filename.replace(".joblib", "").split("_")
                trained_at = datetime.now().isoformat()
                if len(parts) >= 3 and parts[0].startswith("v"):
                    try:
                        date_str = parts[1]
                        time_str = parts[2]
                        trained_at = (
                            f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                            f"T{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                        )
                    except (IndexError, ValueError):
                        pass

                entry = {
                    "version": f"v{i:03d}",
                    "filename": filename,
                    "trained_at": trained_at,
                    "metrics": {"recovered": True},
                    "symbols": [],
                    "n_features": 0,
                    "n_samples": 0,
                    "note": f"Auto-recovered from orphaned file: {filename}",
                    "is_active": (i == len(model_files)),
                }
                registry.append(entry)

            self._save_registry(registry)
            self._cleanup_temp_files()
            logger.info(
                "ml.registry.recovery_complete",
                versions=len(registry),
                active=f"v{len(registry):03d}",
            )

    def _cleanup_temp_files(self) -> None:
        """Remove orphaned registry temp files from failed atomic writes."""
        try:
            for f in os.listdir(self.models_dir):
                if f.startswith("registry_") and f.endswith(".json"):
                    os.remove(os.path.join(self.models_dir, f))
        except OSError:
            pass

    def _prune_missing_entries(self) -> None:
        """
        Remove registry entries whose model files don't exist on disk.

        Common after container deployments where .joblib files are gitignored
        but registry.json is committed. Promotes the newest existing file to active.
        """
        with _registry_lock:
            registry = self._load_registry()
            if not registry:
                return

            pruned = []
            kept = []
            for entry in registry:
                filepath = os.path.join(self.models_dir, entry["filename"])
                if os.path.exists(filepath):
                    kept.append(entry)
                else:
                    pruned.append(entry["version"])

            if not pruned:
                return

            # Mark newest surviving entry as active
            for entry in kept:
                entry["is_active"] = False
            if kept:
                kept[-1]["is_active"] = True

            self._save_registry(kept)
            logger.info(
                "ml.registry.pruned_missing",
                removed=pruned,
                remaining=len(kept),
            )

    def _get_features_for_entry(self, entry: dict) -> list[str]:
        """Load features list for a registry entry from the model file."""
        filepath = os.path.join(self.models_dir, entry["filename"])
        if not os.path.exists(filepath):
            return []
        try:
            artifact = joblib.load(filepath)
            return artifact.get("features", [])
        except Exception:
            return []

    @staticmethod
    def _pick_comparison_key(old_metrics: dict, new_metrics: dict) -> Optional[str]:
        """Pick the best metric key to compare models on."""
        # Prefer cv_accuracy
        if "cv_accuracy" in old_metrics and "cv_accuracy" in new_metrics:
            return "cv_accuracy"
        # Fall back to first shared numeric key
        for k in old_metrics:
            if k in new_metrics and isinstance(old_metrics[k], (int, float)):
                return k
        return None

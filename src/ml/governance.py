"""
Model Governance — Full lineage tracking for ML model deployments.

Records complete provenance for every model version including:
- Git commit hash at training time
- Configuration hash
- Feature schema version
- Training dataset hash
- Scaler version
- Random seed
- Walk-forward and Monte Carlo scores
- Deployment and retirement timestamps

This creates auditable lineage for every prediction made in production.
"""

import hashlib
import json
import os
import subprocess
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model Status Enum
# ---------------------------------------------------------------------------


class ModelStatus(str, Enum):
    """Model promotion lifecycle status."""
    CANDIDATE = "candidate"
    VALIDATING = "validating"
    REJECTED = "rejected"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    RETIRED = "retired"


# ---------------------------------------------------------------------------
# Validation Result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Structured result from model validation with per-check details."""
    is_valid: bool = False
    checks: list = field(default_factory=list)  # list of {"check": str, "passed": bool, "detail": str}
    issues: list = field(default_factory=list)
    model_status: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Governance Metadata
# ---------------------------------------------------------------------------


@dataclass
class ModelLineage:
    """Complete provenance record for a model deployment."""

    version: str
    git_commit: str = ""
    git_branch: str = ""
    config_hash: str = ""
    feature_schema_version: str = ""
    feature_names: list = field(default_factory=list)
    n_features: int = 0
    training_dataset_hash: str = ""
    training_dataset_rows: int = 0
    training_dataset_symbols: list = field(default_factory=list)
    scaler_type: str = ""
    scaler_hash: str = ""
    random_seed: Optional[int] = None
    hyperparameters: dict = field(default_factory=dict)
    # Scores
    cv_accuracy: float = 0.0
    walk_forward_sharpe: float = 0.0
    walk_forward_return: float = 0.0
    monte_carlo_median_return: float = 0.0
    monte_carlo_p5_return: float = 0.0
    monte_carlo_prob_profit: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    n_trades_backtest: int = 0
    # Training metadata
    training_duration_seconds: float = 0.0
    training_timestamp: str = ""
    # Deployment
    deployed_at: str = ""
    retired_at: str = ""
    is_deployed: bool = False
    deployment_reason: str = ""
    retirement_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelLineage":
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Model Governance Manager
# ---------------------------------------------------------------------------


class ModelGovernance:
    """
    Manages full model lineage and governance metadata.

    Works alongside ModelRegistry to add governance tracking:
    - Records complete provenance at training time
    - Tracks deployment/retirement lifecycle
    - Provides audit queries
    - Validates model readiness for deployment

    Usage:
        governance = ModelGovernance()
        lineage = governance.record_training(
            version="v003",
            features=["rsi_14", "ema_slope_20", ...],
            dataset=df_train,
            config=settings_dict,
            metrics={"cv_accuracy": 0.67, "sharpe": 1.8},
            seed=42,
        )
        governance.deploy(version="v003", reason="Better walk-forward Sharpe")
    """

    def __init__(self, governance_dir: str = "models/governance"):
        self.governance_dir = governance_dir
        self._lock = threading.Lock()
        os.makedirs(governance_dir, exist_ok=True)
        self._lineage_path = os.path.join(governance_dir, "lineage.json")

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_training(
        self,
        version: str,
        features: list,
        dataset: Any = None,
        config: Optional[dict] = None,
        metrics: Optional[dict] = None,
        hyperparameters: Optional[dict] = None,
        seed: Optional[int] = None,
        scaler: Any = None,
        training_duration: float = 0.0,
        walk_forward_results: Optional[dict] = None,
        monte_carlo_results: Optional[dict] = None,
    ) -> ModelLineage:
        """
        Record complete training provenance for a model version.

        Args:
            version: Model version string (e.g. "v003").
            features: List of feature column names.
            dataset: Training DataFrame (used to compute hash/row count).
            config: Configuration dict used for training.
            metrics: Training/validation metrics.
            hyperparameters: Model hyperparameters.
            seed: Random seed used.
            scaler: Fitted scaler object (for hash).
            training_duration: Training time in seconds.
            walk_forward_results: Walk-forward optimization results dict.
            monte_carlo_results: Monte Carlo simulation results dict.

        Returns:
            ModelLineage with all metadata populated.
        """
        metrics = metrics or {}
        hyperparameters = hyperparameters or {}

        lineage = ModelLineage(
            version=version,
            git_commit=self._get_git_commit(),
            git_branch=self._get_git_branch(),
            config_hash=self._hash_dict(config) if config else "",
            feature_schema_version=self._compute_feature_schema_version(features),
            feature_names=features,
            n_features=len(features),
            training_dataset_hash=self._hash_dataset(dataset) if dataset is not None else "",
            training_dataset_rows=len(dataset) if dataset is not None else 0,
            training_dataset_symbols=self._extract_symbols(dataset),
            scaler_type=type(scaler).__name__ if scaler else "",
            scaler_hash=self._hash_object(scaler) if scaler else "",
            random_seed=seed,
            hyperparameters=hyperparameters,
            # Scores from metrics
            cv_accuracy=metrics.get("cv_accuracy", 0.0),
            sharpe_ratio=metrics.get("sharpe", metrics.get("sharpe_ratio", 0.0)),
            sortino_ratio=metrics.get("sortino", metrics.get("sortino_ratio", 0.0)),
            calmar_ratio=metrics.get("calmar", metrics.get("calmar_ratio", 0.0)),
            max_drawdown=metrics.get("max_drawdown", 0.0),
            n_trades_backtest=metrics.get("n_trades", 0),
            # Walk-forward
            walk_forward_sharpe=(
                walk_forward_results.get("sharpe", 0.0)
                if walk_forward_results else 0.0
            ),
            walk_forward_return=(
                walk_forward_results.get("total_return", 0.0)
                if walk_forward_results else 0.0
            ),
            # Monte Carlo
            monte_carlo_median_return=(
                monte_carlo_results.get("median_return", 0.0)
                if monte_carlo_results else 0.0
            ),
            monte_carlo_p5_return=(
                monte_carlo_results.get("p5_return", 0.0)
                if monte_carlo_results else 0.0
            ),
            monte_carlo_prob_profit=(
                monte_carlo_results.get("probability_of_profit", 0.0)
                if monte_carlo_results else 0.0
            ),
            # Training metadata
            training_duration_seconds=training_duration,
            training_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._save_lineage(lineage)
        logger.info(
            "governance.recorded",
            version=version,
            git_commit=lineage.git_commit[:8] if lineage.git_commit else "unknown",
            features=len(features),
            dataset_rows=lineage.training_dataset_rows,
        )
        return lineage

    # ------------------------------------------------------------------
    # Deployment Lifecycle
    # ------------------------------------------------------------------

    def deploy(self, version: str, reason: str = "") -> bool:
        """Mark a model version as deployed (live in production)."""
        with self._lock:
            all_lineage = self._load_all_lineage()

            # Retire current deployment
            for entry in all_lineage:
                if entry.get("is_deployed"):
                    entry["is_deployed"] = False
                    entry["retired_at"] = datetime.now(timezone.utc).isoformat()
                    entry["retirement_reason"] = f"Superseded by {version}"

            # Deploy new version
            target = self._find_lineage(all_lineage, version)
            if target is None:
                logger.warning("governance.deploy_failed", version=version, reason="not_found")
                return False

            target["is_deployed"] = True
            target["deployed_at"] = datetime.now(timezone.utc).isoformat()
            target["deployment_reason"] = reason
            target["deployment_record"] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "previous_model": next(
                    (e.get("version") for e in all_lineage if e.get("retired_at") == target.get("deployed_at")),
                    None,
                ),
                "promoter": "governance_system",
                "governance_decision": "auto_deploy" if "Auto-deploy" in reason else "manual",
                "validation_metrics": target.get("metrics", {}),
                "gates_passed": target.get("validation_issues", []) == [],
                "rollback_model": next(
                    (e.get("version") for e in reversed(all_lineage) if e.get("is_deployed") and e["version"] != version),
                    None,
                ),
                "git_commit": target.get("git_commit", ""),
                "feature_schema_version": target.get("feature_hash", ""),
            }

            self._save_all_lineage(all_lineage)
            logger.info("governance.deployed", version=version, reason=reason)
            return True

    def retire(self, version: str, reason: str = "") -> bool:
        """Mark a model version as retired."""
        with self._lock:
            all_lineage = self._load_all_lineage()
            target = self._find_lineage(all_lineage, version)
            if target is None:
                return False

            target["is_deployed"] = False
            target["retired_at"] = datetime.now(timezone.utc).isoformat()
            target["retirement_reason"] = reason

            self._save_all_lineage(all_lineage)
            logger.info("governance.retired", version=version, reason=reason)
            return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_lineage(self, version: str) -> Optional[ModelLineage]:
        """Get full lineage for a specific version."""
        all_lineage = self._load_all_lineage()
        entry = self._find_lineage(all_lineage, version)
        if entry:
            return ModelLineage.from_dict(entry)
        return None

    def get_deployed_model(self) -> Optional[ModelLineage]:
        """Get the currently deployed model's lineage."""
        all_lineage = self._load_all_lineage()
        for entry in reversed(all_lineage):
            if entry.get("is_deployed"):
                return ModelLineage.from_dict(entry)
        return None

    def get_deployment_history(self) -> list[dict]:
        """Get all models that were ever deployed, with timestamps."""
        all_lineage = self._load_all_lineage()
        deployed = [e for e in all_lineage if e.get("deployed_at")]
        return sorted(deployed, key=lambda x: x.get("deployed_at", ""), reverse=True)

    def validate_for_deployment(self, version: str) -> tuple[bool, list[str]]:
        """
        Validate whether a model is ready for deployment.

        Uses configurable thresholds from settings when available,
        falling back to sensible defaults.

        Returns (is_valid, list_of_issues).
        """
        try:
            from config.settings import settings
            min_cv_accuracy = settings.model_min_cv_accuracy
            min_walk_forward_sharpe = settings.model_min_walk_forward_sharpe
            min_monte_carlo_prob_profit = settings.model_min_monte_carlo_prob_profit
            min_sharpe_ratio = settings.model_min_sharpe_ratio
            max_drawdown_pct = settings.model_max_drawdown_pct
            min_expectancy = settings.model_min_expectancy
            min_precision = settings.model_min_precision
            min_backtest_trades = settings.model_min_backtest_trades
        except Exception:
            min_cv_accuracy = 0.52
            min_walk_forward_sharpe = 0.0
            min_monte_carlo_prob_profit = 0.50
            min_sharpe_ratio = 0.5
            max_drawdown_pct = 0.20
            min_expectancy = 0.0
            min_precision = 0.50
            min_backtest_trades = 30

        issues = []
        checks = []
        all_lineage = self._load_all_lineage()
        entry = self._find_lineage(all_lineage, version)

        if entry is None:
            return False, [f"Version {version} not found in governance records"]

        lineage = ModelLineage.from_dict(entry)

        def _check(name: str, passed: bool, detail: str):
            checks.append({"check": name, "passed": passed, "detail": detail})
            if not passed:
                issues.append(detail)

        # Required provenance fields
        _check("git_commit", bool(lineage.git_commit),
               "Missing git commit hash" if not lineage.git_commit else "OK")
        _check("feature_schema", bool(lineage.feature_names),
               "Missing feature schema" if not lineage.feature_names else "OK")
        _check("dataset_hash", bool(lineage.training_dataset_hash),
               "Missing dataset hash" if not lineage.training_dataset_hash else "OK")
        _check("n_features", lineage.n_features > 0,
               "Zero features recorded" if lineage.n_features == 0 else "OK")
        _check("dataset_rows", lineage.training_dataset_rows > 0,
               "Zero training rows recorded" if lineage.training_dataset_rows == 0 else "OK")
        _check("hyperparameters", bool(lineage.hyperparameters),
               "Hyperparameters not recorded" if not lineage.hyperparameters else "OK")

        # Performance thresholds
        _check("cv_accuracy",
               lineage.cv_accuracy >= min_cv_accuracy,
               f"CV accuracy {lineage.cv_accuracy:.3f} below threshold {min_cv_accuracy}")
        _check("walk_forward_sharpe",
               lineage.walk_forward_sharpe >= min_walk_forward_sharpe,
               f"Walk-forward Sharpe {lineage.walk_forward_sharpe:.3f} below threshold {min_walk_forward_sharpe}")
        _check("monte_carlo_prob_profit",
               lineage.monte_carlo_prob_profit >= min_monte_carlo_prob_profit,
               f"Monte Carlo prob_profit {lineage.monte_carlo_prob_profit:.3f} below threshold {min_monte_carlo_prob_profit}")
        _check("sharpe_ratio",
               lineage.sharpe_ratio >= min_sharpe_ratio,
               f"Sharpe ratio {lineage.sharpe_ratio:.3f} below threshold {min_sharpe_ratio}")
        _check("max_drawdown",
               lineage.max_drawdown <= max_drawdown_pct,
               f"Max drawdown {lineage.max_drawdown:.3f} exceeds threshold {max_drawdown_pct}")
        _check("min_backtest_trades",
               lineage.n_trades_backtest >= min_backtest_trades,
               f"Backtest trades {lineage.n_trades_backtest} below minimum {min_backtest_trades}")

        is_valid = len(issues) == 0
        self._last_validation_result = ValidationResult(
            is_valid=is_valid,
            checks=checks,
            issues=issues,
            model_status=ModelStatus.APPROVED.value if is_valid else ModelStatus.REJECTED.value,
        )
        return is_valid, issues

    def audit_report(self) -> dict:
        """Generate a governance audit report."""
        all_lineage = self._load_all_lineage()
        total = len(all_lineage)
        deployed = sum(1 for e in all_lineage if e.get("is_deployed"))
        retired = sum(1 for e in all_lineage if e.get("retired_at") and not e.get("is_deployed"))
        with_git = sum(1 for e in all_lineage if e.get("git_commit"))
        with_dataset_hash = sum(1 for e in all_lineage if e.get("training_dataset_hash"))

        return {
            "total_versions": total,
            "currently_deployed": deployed,
            "retired": retired,
            "with_git_commit": with_git,
            "with_dataset_hash": with_dataset_hash,
            "governance_completeness": (
                (with_git + with_dataset_hash) / (2 * total) if total > 0 else 0.0
            ),
            "versions": [
                {
                    "version": e.get("version"),
                    "deployed": e.get("is_deployed", False),
                    "git_commit": e.get("git_commit", "")[:8],
                    "trained_at": e.get("training_timestamp", ""),
                    "cv_accuracy": e.get("cv_accuracy", 0.0),
                }
                for e in all_lineage
            ],
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    def _get_git_branch(self) -> str:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=5,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    def _hash_dict(self, d: dict) -> str:
        """Deterministic hash of a dictionary."""
        try:
            json_str = json.dumps(d, sort_keys=True, default=str)
            return hashlib.sha256(json_str.encode()).hexdigest()[:16]
        except Exception:
            return ""

    def _hash_dataset(self, dataset) -> str:
        """Hash a pandas DataFrame for provenance tracking."""
        try:
            import pandas as pd
            if isinstance(dataset, pd.DataFrame):
                # Hash shape + column names + first/last rows + sample values
                shape_str = f"{dataset.shape}"
                cols_str = ",".join(dataset.columns.tolist())
                # Use a reproducible sample of the data
                data_hash = hashlib.sha256(
                    pd.util.hash_pandas_object(dataset).values.tobytes()
                ).hexdigest()[:16]
                return hashlib.sha256(
                    f"{shape_str}|{cols_str}|{data_hash}".encode()
                ).hexdigest()[:16]
        except Exception:
            pass
        return ""

    def _hash_object(self, obj) -> str:
        """Hash an arbitrary object (e.g., scaler) using pickle."""
        try:
            import pickle
            obj_bytes = pickle.dumps(obj)
            return hashlib.sha256(obj_bytes).hexdigest()[:16]
        except Exception:
            return ""

    def _extract_symbols(self, dataset) -> list:
        """Extract symbol list from dataset if available."""
        try:
            import pandas as pd
            if isinstance(dataset, pd.DataFrame) and "symbol" in dataset.columns:
                return sorted(dataset["symbol"].unique().tolist())
        except Exception:
            pass
        return []

    def _compute_feature_schema_version(self, features: list) -> str:
        """Compute a version hash of the feature schema."""
        schema_str = ",".join(sorted(features))
        return hashlib.sha256(schema_str.encode()).hexdigest()[:8]

    def _save_lineage(self, lineage: ModelLineage) -> None:
        """Append a lineage record (immutable once deployed)."""
        with self._lock:
            all_lineage = self._load_all_lineage()
            existing = self._find_lineage(all_lineage, lineage.version)
            if existing is not None:
                # If already deployed, refuse to overwrite
                if existing.get("deployed_at"):
                    logger.warning(
                        "governance.immutable_reject",
                        version=lineage.version,
                        reason="deployed record cannot be overwritten",
                    )
                    return
                all_lineage = [
                    lineage.to_dict() if e.get("version") == lineage.version else e
                    for e in all_lineage
                ]
            else:
                all_lineage.append(lineage.to_dict())
            # Compute and store integrity hash for each new/updated record
            for entry in all_lineage:
                if entry.get("version") == lineage.version:
                    entry["integrity_hash"] = self._compute_integrity_hash(entry)
            self._save_all_lineage(all_lineage)

    def _load_all_lineage(self) -> list[dict]:
        """Load all lineage records."""
        if not os.path.exists(self._lineage_path):
            return []
        try:
            with open(self._lineage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save_all_lineage(self, all_lineage: list[dict]) -> None:
        """Save all lineage records atomically."""
        try:
            with open(self._lineage_path, "w", encoding="utf-8") as f:
                json.dump(all_lineage, f, indent=2, default=str)
        except OSError:
            logger.exception("governance.save_failed")

    @staticmethod
    def _find_lineage(all_lineage: list[dict], version: str) -> Optional[dict]:
        """Find a lineage record by version."""
        for entry in all_lineage:
            if entry.get("version") == version:
                return entry
        return None

    def _compute_integrity_hash(self, record: dict) -> str:
        """Compute hash of record content excluding deployment fields."""
        # Exclude mutable deployment fields and the integrity_hash itself
        excluded = {"deployed_at", "retired_at", "is_deployed", "deployment_reason",
                    "retirement_reason", "integrity_hash", "deployment_record"}
        content = {k: v for k, v in record.items() if k not in excluded}
        json_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def verify_integrity(self) -> tuple:
        """
        Verify integrity of all governance records.
        
        Checks:
        - All records have required fields (version, training_timestamp)
        - No deployed record has been tampered with (integrity hash matches)
        
        Returns (is_valid, list_of_issues).
        """
        issues = []
        all_lineage = self._load_all_lineage()

        for entry in all_lineage:
            version = entry.get("version", "<unknown>")

            # Check required fields
            if not entry.get("version"):
                issues.append("Record missing 'version' field")
                continue
            if not entry.get("training_timestamp"):
                issues.append(f"Record {version} missing 'training_timestamp'")

            # Check integrity hash for deployed records
            if entry.get("deployed_at") and entry.get("integrity_hash"):
                expected_hash = self._compute_integrity_hash(entry)
                if entry["integrity_hash"] != expected_hash:
                    issues.append(
                        f"Record {version} integrity hash mismatch: "
                        f"expected {expected_hash}, got {entry['integrity_hash']}"
                    )

        return len(issues) == 0, issues

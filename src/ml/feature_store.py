"""
Feature Store for ML prediction reproducibility.

Tracks feature versions, transformations, scaling parameters, and stores
feature snapshots at prediction time to ensure full reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureVersion:
    """Immutable record of a registered feature set configuration."""

    version_id: str
    feature_names: list[str]
    feature_hash: str
    scaling_params: dict
    encoding_params: dict
    normalization_method: str
    created_at: datetime
    description: str = ""
    parent_version: Optional[str] = None


@dataclass
class FeatureSnapshot:
    """Point-in-time capture of feature values used for a prediction."""

    snapshot_id: str
    version_id: str
    symbol: str
    timestamp: datetime
    values: dict[str, float]
    raw_values: dict[str, float]


class FeatureStore:
    """
    Production feature store backed by DuckDB.

    Provides versioned feature management, snapshot storage for prediction
    reproducibility, and drift detection via Population Stability Index.
    """

    def __init__(self, storage_path: str = "data_cache/features", store_dir: str = None) -> None:
        path = store_dir or storage_path
        self._storage_path = Path(path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._db_path = str(self._storage_path / "features.duckdb")
        self._write_lock = threading.Lock()
        self._init_db()
        logger.info("feature_store.initialized", storage_path=str(self._storage_path))

    def _get_conn(self) -> duckdb.DuckDBPyConnection:
        """Create a new connection to the DuckDB database."""
        return duckdb.connect(self._db_path)

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_versions (
                    version_id TEXT PRIMARY KEY,
                    feature_names TEXT NOT NULL,
                    feature_hash TEXT NOT NULL UNIQUE,
                    scaling_params TEXT,
                    encoding_params TEXT,
                    normalization_method TEXT DEFAULT 'standard',
                    description TEXT DEFAULT '',
                    parent_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    prediction_id TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    values_json TEXT NOT NULL,
                    raw_values_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_time
                ON feature_snapshots(symbol, timestamp)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transformation_log (
                    log_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    params TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        finally:
            conn.close()

    def register_version(
        self,
        feature_names: list[str],
        scaling_params: dict,
        encoding_params: dict | None = None,
        normalization_method: str = "standard",
        description: str = "",
        parent_version: str | None = None,
    ) -> str:
        """
        Register a new feature version.

        Returns the version_id. If an identical feature_hash already exists,
        returns the existing version_id (deduplication).
        """
        if encoding_params is None:
            encoding_params = {}

        feature_hash = self._compute_hash(feature_names, scaling_params, encoding_params)

        with self._write_lock:
            conn = self._get_conn()
            try:
                # Check for existing hash (dedup)
                result = conn.execute(
                    "SELECT version_id FROM feature_versions WHERE feature_hash = ?",
                    [feature_hash],
                ).fetchone()

                if result is not None:
                    existing_id = result[0]
                    logger.info(
                        "feature_store.version_deduplicated",
                        existing_version_id=existing_id,
                        feature_hash=feature_hash,
                    )
                    return existing_id

                # Determine parent version if not provided
                if parent_version is None:
                    active = conn.execute(
                        "SELECT version_id FROM feature_versions ORDER BY created_at DESC LIMIT 1"
                    ).fetchone()
                    if active is not None:
                        parent_version = active[0]

                version_id = self._generate_version_id()
                now = datetime.now(timezone.utc)

                conn.execute(
                    """
                    INSERT INTO feature_versions
                    (version_id, feature_names, feature_hash, scaling_params,
                     encoding_params, normalization_method, description,
                     parent_version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        version_id,
                        json.dumps(feature_names),
                        feature_hash,
                        json.dumps(scaling_params),
                        json.dumps(encoding_params),
                        normalization_method,
                        description,
                        parent_version,
                        now,
                    ],
                )

                # Log the registration
                self._log_transformation(
                    conn,
                    version_id,
                    "register",
                    {
                        "normalization_method": normalization_method,
                        "feature_count": len(feature_names),
                    },
                )

                logger.info(
                    "feature_store.version_registered",
                    version_id=version_id,
                    feature_count=len(feature_names),
                    normalization_method=normalization_method,
                )
                return version_id
            finally:
                conn.close()

    def get_version(self, version_id: str) -> FeatureVersion:
        """Retrieve a specific feature version by ID."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM feature_versions WHERE version_id = ?",
                [version_id],
            ).fetchone()

            if row is None:
                raise KeyError(f"Feature version not found: {version_id}")

            return self._row_to_version(row)
        finally:
            conn.close()

    def get_active_version(self) -> Optional[FeatureVersion]:
        """Get the currently active feature version (latest registered)."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM feature_versions ORDER BY created_at DESC LIMIT 1"
            ).fetchone()

            if row is None:
                return None

            return self._row_to_version(row)
        finally:
            conn.close()

    def snapshot_features(
        self,
        version_id: str,
        symbol: str,
        values: dict[str, float],
        raw_values: dict[str, float] | None = None,
        prediction_id: str | None = None,
    ) -> str:
        """
        Store a feature snapshot at prediction time.

        Called every time a prediction is made to preserve the exact
        feature values used. Returns snapshot_id.
        """
        snapshot_id = f"snap_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        with self._write_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO feature_snapshots
                    (snapshot_id, version_id, symbol, prediction_id,
                     timestamp, values_json, raw_values_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        snapshot_id,
                        version_id,
                        symbol,
                        prediction_id,
                        now,
                        json.dumps(values),
                        json.dumps(raw_values) if raw_values else None,
                        now,
                    ],
                )

                logger.debug(
                    "feature_store.snapshot_stored",
                    snapshot_id=snapshot_id,
                    version_id=version_id,
                    symbol=symbol,
                )
                return snapshot_id
            finally:
                conn.close()

    def get_snapshot(self, snapshot_id: str) -> Optional[FeatureSnapshot]:
        """Retrieve a stored feature snapshot."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM feature_snapshots WHERE snapshot_id = ?",
                [snapshot_id],
            ).fetchone()

            if row is None:
                return None

            return FeatureSnapshot(
                snapshot_id=row[0],
                version_id=row[1],
                symbol=row[2],
                timestamp=row[4],
                values=json.loads(row[5]),
                raw_values=json.loads(row[6]) if row[6] else {},
            )
        finally:
            conn.close()

    def get_snapshots_for_symbol(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Get feature snapshots for a symbol as a DataFrame."""
        conn = self._get_conn()
        try:
            if since is not None:
                rows = conn.execute(
                    """
                    SELECT snapshot_id, version_id, symbol, timestamp, values_json, raw_values_json
                    FROM feature_snapshots
                    WHERE symbol = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    [symbol, since, limit],
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT snapshot_id, version_id, symbol, timestamp, values_json, raw_values_json
                    FROM feature_snapshots
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    [symbol, limit],
                ).fetchall()

            if not rows:
                return pd.DataFrame()

            records = []
            for row in rows:
                values = json.loads(row[4])
                record = {
                    "snapshot_id": row[0],
                    "version_id": row[1],
                    "symbol": row[2],
                    "timestamp": row[3],
                    **values,
                }
                records.append(record)

            return pd.DataFrame(records)
        finally:
            conn.close()

    def compute_feature_drift(
        self,
        baseline_days: int = 30,
        recent_days: int = 7,
    ) -> dict:
        """
        Compute Population Stability Index (PSI) for each feature.

        Compares recent feature distributions vs historical baseline.

        Returns:
            {feature_name: {"psi": float, "baseline_mean": float,
             "recent_mean": float, "drift_detected": bool}}
        """
        now = datetime.now(timezone.utc)
        baseline_start = now - timedelta(days=baseline_days)
        recent_start = now - timedelta(days=recent_days)

        conn = self._get_conn()
        try:
            # Get baseline snapshots
            baseline_rows = conn.execute(
                """
                SELECT values_json FROM feature_snapshots
                WHERE timestamp >= ? AND timestamp < ?
                """,
                [baseline_start, recent_start],
            ).fetchall()

            # Get recent snapshots
            recent_rows = conn.execute(
                """
                SELECT values_json FROM feature_snapshots
                WHERE timestamp >= ?
                """,
                [recent_start],
            ).fetchall()
        finally:
            conn.close()

        if not baseline_rows or not recent_rows:
            logger.warning(
                "feature_store.drift_insufficient_data",
                baseline_count=len(baseline_rows) if baseline_rows else 0,
                recent_count=len(recent_rows) if recent_rows else 0,
            )
            return {}

        # Parse values
        baseline_values: dict[str, list[float]] = {}
        for (values_json,) in baseline_rows:
            values = json.loads(values_json)
            for name, val in values.items():
                baseline_values.setdefault(name, []).append(float(val))

        recent_values: dict[str, list[float]] = {}
        for (values_json,) in recent_rows:
            values = json.loads(values_json)
            for name, val in values.items():
                recent_values.setdefault(name, []).append(float(val))

        # Compute PSI per feature
        drift_results = {}
        for feature_name in baseline_values:
            if feature_name not in recent_values:
                continue

            baseline_arr = np.array(baseline_values[feature_name])
            recent_arr = np.array(recent_values[feature_name])

            if len(baseline_arr) < 10 or len(recent_arr) < 10:
                continue

            psi = self._compute_psi(baseline_arr, recent_arr)
            drift_results[feature_name] = {
                "psi": psi,
                "baseline_mean": float(np.mean(baseline_arr)),
                "recent_mean": float(np.mean(recent_arr)),
                "drift_detected": psi >= 0.25,
            }

        logger.info(
            "feature_store.drift_computed",
            features_checked=len(drift_results),
            drift_detected=sum(
                1 for v in drift_results.values() if v["drift_detected"]
            ),
        )
        return drift_results

    def get_feature_importance_history(self) -> pd.DataFrame:
        """Track how feature importance changes across model versions."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT version_id, feature_names, created_at
                FROM feature_versions
                ORDER BY created_at ASC
                """
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return pd.DataFrame()

        records = []
        for row in rows:
            version_id = row[0]
            feature_names = json.loads(row[1])
            created_at = row[2]
            for idx, name in enumerate(feature_names):
                records.append(
                    {
                        "version_id": version_id,
                        "feature_name": name,
                        "position": idx,
                        "created_at": created_at,
                    }
                )

        return pd.DataFrame(records)

    @property
    def version_count(self) -> int:
        """Number of registered feature versions."""
        conn = self._get_conn()
        try:
            result = conn.execute("SELECT COUNT(*) FROM feature_versions").fetchone()
            return result[0] if result else 0
        finally:
            conn.close()

    @property
    def snapshot_count(self) -> int:
        """Total stored snapshots."""
        conn = self._get_conn()
        try:
            result = conn.execute("SELECT COUNT(*) FROM feature_snapshots").fetchone()
            return result[0] if result else 0
        finally:
            conn.close()

    # ─── Private helpers ────────────────────────────────────────────────

    def _compute_hash(
        self,
        feature_names: list[str],
        scaling_params: dict,
        encoding_params: dict,
    ) -> str:
        """Compute deterministic SHA256 hash from feature configuration."""
        payload = json.dumps(
            {
                "features": sorted(feature_names),
                "scaling": scaling_params,
                "encoding": encoding_params,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _generate_version_id(self) -> str:
        """Generate a unique version ID with date prefix."""
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        short_id = uuid.uuid4().hex[:8]
        return f"fv_{date_str}_{short_id}"

    def _row_to_version(self, row: tuple) -> FeatureVersion:
        """Convert a database row to FeatureVersion dataclass."""
        # Column order: version_id, feature_names, feature_hash, scaling_params,
        #               encoding_params, normalization_method, description,
        #               parent_version, created_at
        return FeatureVersion(
            version_id=row[0],
            feature_names=json.loads(row[1]) if isinstance(row[1], str) else row[1],
            feature_hash=row[2],
            scaling_params=json.loads(row[3]) if isinstance(row[3], str) else (row[3] or {}),
            encoding_params=json.loads(row[4]) if isinstance(row[4], str) else (row[4] or {}),
            normalization_method=row[5] or "standard",
            created_at=row[8] if len(row) > 8 else datetime.now(timezone.utc),
            description=row[6] or "",
            parent_version=row[7],
        )

    def _log_transformation(
        self,
        conn: duckdb.DuckDBPyConnection,
        version_id: str,
        operation: str,
        params: dict,
    ) -> None:
        """Log a transformation operation."""
        log_id = f"log_{uuid.uuid4().hex[:12]}"
        conn.execute(
            """
            INSERT INTO transformation_log (log_id, version_id, operation, params, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                log_id,
                version_id,
                operation,
                json.dumps(params),
                datetime.now(timezone.utc),
            ],
        )

    def _compute_psi(
        self, baseline: np.ndarray, recent: np.ndarray, bins: int = 10
    ) -> float:
        """
        Population Stability Index.

        PSI < 0.1: No significant change
        0.1 <= PSI < 0.25: Moderate change
        PSI >= 0.25: Significant change (drift detected)
        """
        breakpoints = np.quantile(baseline, np.linspace(0, 1, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
        recent_counts = np.histogram(recent, bins=breakpoints)[0]

        # Avoid division by zero with Laplace smoothing
        baseline_pct = (baseline_counts + 1) / (len(baseline) + bins)
        recent_pct = (recent_counts + 1) / (len(recent) + bins)

        psi = np.sum((recent_pct - baseline_pct) * np.log(recent_pct / baseline_pct))
        return float(psi)

    # ─── Backward-compatible API (legacy tests) ─────────────────────────
    # The original FeatureStore used a simpler file-based API with put/get.
    # These methods provide backward compatibility for existing tests.

    def put(self, df: pd.DataFrame, symbol: str, params: dict = None) -> "FeatureSetMetadata":
        """
        Store a feature DataFrame (legacy API).
        Maps to register_version + store the actual data as Parquet.
        """
        import time as _time
        start = _time.perf_counter()

        feature_names = list(df.columns)
        scaling_params = params or {}
        feature_hash = self._compute_hash(feature_names, scaling_params, {})

        # Register version (deduplicates by hash)
        version_id = self.register_version(
            feature_names=feature_names,
            scaling_params=scaling_params,
            normalization_method="none",
            description=f"Legacy put for {symbol}",
        )

        # Store the actual data as Parquet
        data_dir = self._storage_path / "data" / symbol
        data_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = data_dir / f"{version_id}.parquet"
        df.to_parquet(str(parquet_path), index=False)

        elapsed = _time.perf_counter() - start
        data_hash = hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()[:16]

        return FeatureSetMetadata(
            version=version_id,
            symbol=symbol,
            n_rows=len(df),
            n_features=len(feature_names),
            feature_names=feature_names,
            source_data_hash=data_hash,
            computation_seconds=elapsed,
            computed_at=datetime.now(timezone.utc).isoformat(),
            parameters=scaling_params,
        )

    def get(self, symbol: str, version: str) -> Optional[pd.DataFrame]:
        """Retrieve stored feature DataFrame (legacy API)."""
        parquet_path = self._storage_path / "data" / symbol / f"{version}.parquet"
        if parquet_path.exists():
            return pd.read_parquet(str(parquet_path))
        return None

    def validate_schema(self, df: pd.DataFrame, expected_version: str, symbol: str) -> tuple:
        """Validate DataFrame schema matches a registered version (legacy API)."""
        try:
            version = self.get_version(expected_version)
        except (KeyError, Exception):
            return False, [f"Version {expected_version} not found"]
        if version is None:
            return False, [f"Version {expected_version} not found"]

        expected_cols = set(version.feature_names)
        actual_cols = set(df.columns)

        issues = []
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols

        if missing:
            issues.append(f"Missing columns: {sorted(missing)}")
        if extra:
            issues.append(f"Extra columns: {sorted(extra)}")

        return len(issues) == 0, issues

    def list_versions(self, symbol: str) -> list[str]:
        """List all versions for a symbol (legacy API)."""
        data_dir = self._storage_path / "data" / symbol
        if not data_dir.exists():
            return []
        return [f.stem for f in sorted(data_dir.glob("*.parquet"))]

    def cleanup(self, symbol: str, keep_latest: int = 3) -> int:
        """Remove old versions, keeping only the latest N (legacy API)."""
        versions = self.list_versions(symbol)
        if len(versions) <= keep_latest:
            return 0

        to_remove = versions[:-keep_latest]
        data_dir = self._storage_path / "data" / symbol
        removed = 0
        for v in to_remove:
            path = data_dir / f"{v}.parquet"
            if path.exists():
                path.unlink()
                removed += 1
        return removed


@dataclass
class FeatureSetMetadata:
    """Legacy metadata class for backward compatibility."""

    version: str
    symbol: str
    n_rows: int
    n_features: int
    feature_names: list[str]
    source_data_hash: str
    computation_seconds: float
    computed_at: str
    parameters: dict = field(default_factory=dict)

    def _row_to_version(self, row: tuple) -> FeatureVersion:
        """Convert a database row to a FeatureVersion dataclass."""
        return FeatureVersion(
            version_id=row[0],
            feature_names=json.loads(row[1]),
            feature_hash=row[2],
            scaling_params=json.loads(row[3]) if row[3] else {},
            encoding_params=json.loads(row[4]) if row[4] else {},
            normalization_method=row[5] or "standard",
            description=row[6] or "",
            parent_version=row[7],
            created_at=row[8],
        )

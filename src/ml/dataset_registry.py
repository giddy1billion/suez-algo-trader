"""Dataset Versioning and Model Lineage system.

Ensures full reproducibility: any historical prediction can be traced back
to the exact dataset, model version, and feature version used.
"""
from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetVersion:
    dataset_id: str
    version: int
    symbols: list[str]
    timeframe: str
    start_date: datetime
    end_date: datetime
    row_count: int
    feature_version_id: str
    data_hash: str
    source: str
    created_at: datetime
    description: str = ""
    parent_dataset_id: Optional[str] = None


@dataclass
class ModelLineage:
    model_version: str
    dataset_id: str
    feature_version_id: str
    training_pipeline_id: str
    parent_model_version: Optional[str] = None
    hyperparameters: dict = field(default_factory=dict)
    training_metrics: dict = field(default_factory=dict)
    training_duration_seconds: float = 0.0
    training_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    promotion_timestamp: Optional[datetime] = None
    demotion_timestamp: Optional[datetime] = None
    status: str = "registered"


@dataclass
class PredictionRecord:
    prediction_id: str
    model_version: str
    feature_version_id: str
    feature_snapshot_id: str
    symbol: str
    timestamp: datetime
    predicted_direction: str
    predicted_confidence: float
    trade_id: Optional[str] = None
    outcome_profitable: Optional[bool] = None


class DatasetRegistry:
    """Registry for dataset versions, model lineage, and prediction tracking."""

    def __init__(self, storage_path: str = "data_cache/datasets") -> None:
        self._storage_path = Path(storage_path)
        self._snapshots_path = self._storage_path / "snapshots"
        self._db_path = self._storage_path / "lineage.duckdb"
        self._write_lock = threading.Lock()

        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._snapshots_path.mkdir(parents=True, exist_ok=True)

        self._conn = duckdb.connect(str(self._db_path))
        self._init_schema()
        logger.info("dataset_registry_initialized", storage_path=storage_path)

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                symbols TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                row_count INTEGER NOT NULL,
                feature_version_id TEXT,
                data_hash TEXT NOT NULL,
                source TEXT DEFAULT 'broker_historical',
                description TEXT DEFAULT '',
                parent_dataset_id TEXT,
                parquet_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS model_lineage (
                model_version TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                feature_version_id TEXT NOT NULL,
                training_pipeline_id TEXT,
                parent_model_version TEXT,
                hyperparameters TEXT,
                training_metrics TEXT,
                training_duration_seconds DOUBLE DEFAULT 0.0,
                training_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                promotion_timestamp TIMESTAMP,
                demotion_timestamp TIMESTAMP,
                status TEXT DEFAULT 'registered'
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_records (
                prediction_id TEXT PRIMARY KEY,
                model_version TEXT NOT NULL,
                feature_version_id TEXT NOT NULL,
                feature_snapshot_id TEXT,
                symbol TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                predicted_direction TEXT,
                predicted_confidence DOUBLE,
                trade_id TEXT,
                outcome_profitable BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_model ON prediction_records(model_version)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_trade ON prediction_records(trade_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_lineage_status ON model_lineage(status)"
        )

    def _compute_hash(self, data: pd.DataFrame) -> str:
        """Compute SHA256 hash of sorted DataFrame bytes."""
        sorted_df = data.sort_index(axis=1).sort_index(axis=0)
        content = sorted_df.to_csv(index=True).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def _next_version(self) -> int:
        result = self._conn.execute("SELECT COALESCE(MAX(version), 0) FROM datasets").fetchone()
        return result[0] + 1

    # --- Dataset Registration ---

    def register_dataset(
        self,
        data: pd.DataFrame,
        symbols: list[str],
        timeframe: str,
        feature_version_id: str,
        source: str = "broker_historical",
        description: str = "",
        parent_dataset_id: Optional[str] = None,
    ) -> str:
        """Register a training dataset version. Returns dataset_id."""
        with self._write_lock:
            version = self._next_version()
            short_id = uuid.uuid4().hex[:8]
            dataset_id = f"ds_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{short_id}"
            data_hash = self._compute_hash(data)

            # Determine date range from index or columns
            if hasattr(data.index, 'min') and pd.api.types.is_datetime64_any_dtype(data.index):
                start_date = data.index.min()
                end_date = data.index.max()
            else:
                start_date = datetime.now(timezone.utc)
                end_date = datetime.now(timezone.utc)

            # Save parquet snapshot
            parquet_path = str(self._snapshots_path / f"{dataset_id}.parquet")
            data.to_parquet(parquet_path)

            now = datetime.now(timezone.utc)
            self._conn.execute(
                """
                INSERT INTO datasets (
                    dataset_id, version, symbols, timeframe, start_date, end_date,
                    row_count, feature_version_id, data_hash, source, description,
                    parent_dataset_id, parquet_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    dataset_id, version, json.dumps(symbols), timeframe,
                    start_date, end_date, len(data), feature_version_id,
                    data_hash, source, description, parent_dataset_id,
                    parquet_path, now,
                ],
            )
            logger.info(
                "dataset_registered",
                dataset_id=dataset_id,
                version=version,
                row_count=len(data),
                source=source,
            )
            return dataset_id

    def get_dataset(self, dataset_id: str) -> Optional[DatasetVersion]:
        """Retrieve dataset metadata."""
        row = self._conn.execute(
            "SELECT * FROM datasets WHERE dataset_id = ?", [dataset_id]
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dataset_version(row)

    def load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load the actual dataset Parquet file."""
        row = self._conn.execute(
            "SELECT parquet_path FROM datasets WHERE dataset_id = ?", [dataset_id]
        ).fetchone()
        if row is None or row[0] is None:
            return None
        path = Path(row[0])
        if not path.exists():
            logger.warning("parquet_file_missing", dataset_id=dataset_id, path=str(path))
            return None
        return pd.read_parquet(path)

    def get_latest_dataset(self) -> Optional[DatasetVersion]:
        """Get the most recent dataset version."""
        row = self._conn.execute(
            "SELECT * FROM datasets ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dataset_version(row)

    def _row_to_dataset_version(self, row: tuple) -> DatasetVersion:
        return DatasetVersion(
            dataset_id=row[0],
            version=row[1],
            symbols=json.loads(row[2]),
            timeframe=row[3],
            start_date=row[4],
            end_date=row[5],
            row_count=row[6],
            feature_version_id=row[7],
            data_hash=row[8],
            source=row[9],
            description=row[10],
            parent_dataset_id=row[11],
            created_at=row[13],
        )

    # --- Model Lineage ---

    def register_model(
        self,
        model_version: str,
        dataset_id: str,
        feature_version_id: str,
        pipeline_id: str,
        hyperparameters: Optional[dict] = None,
        training_metrics: Optional[dict] = None,
        training_duration: float = 0.0,
        parent_model_version: Optional[str] = None,
    ) -> None:
        """Register a model with its full training lineage."""
        with self._write_lock:
            now = datetime.now(timezone.utc)
            self._conn.execute(
                """
                INSERT INTO model_lineage (
                    model_version, dataset_id, feature_version_id,
                    training_pipeline_id, parent_model_version,
                    hyperparameters, training_metrics,
                    training_duration_seconds, training_timestamp, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'registered')
                """,
                [
                    model_version, dataset_id, feature_version_id,
                    pipeline_id, parent_model_version,
                    json.dumps(hyperparameters or {}),
                    json.dumps(training_metrics or {}),
                    training_duration, now,
                ],
            )
            logger.info(
                "model_registered",
                model_version=model_version,
                dataset_id=dataset_id,
            )

    def get_model_lineage(self, model_version: str) -> Optional[ModelLineage]:
        """Get full lineage for a model version."""
        row = self._conn.execute(
            "SELECT * FROM model_lineage WHERE model_version = ?", [model_version]
        ).fetchone()
        if row is None:
            return None
        return self._row_to_model_lineage(row)

    def set_model_status(self, model_version: str, status: str) -> None:
        """Update model status (active, demoted, rolled_back)."""
        with self._write_lock:
            now = datetime.now(timezone.utc)
            if status == "active":
                # Demote current active model
                self._conn.execute(
                    """
                    UPDATE model_lineage
                    SET status = 'demoted', demotion_timestamp = ?
                    WHERE status = 'active'
                    """,
                    [now],
                )
                self._conn.execute(
                    """
                    UPDATE model_lineage
                    SET status = ?, promotion_timestamp = ?
                    WHERE model_version = ?
                    """,
                    [status, now, model_version],
                )
            elif status == "demoted":
                self._conn.execute(
                    """
                    UPDATE model_lineage
                    SET status = ?, demotion_timestamp = ?
                    WHERE model_version = ?
                    """,
                    [status, now, model_version],
                )
            else:
                self._conn.execute(
                    "UPDATE model_lineage SET status = ? WHERE model_version = ?",
                    [status, model_version],
                )
            logger.info("model_status_updated", model_version=model_version, status=status)

    def get_active_model(self) -> Optional[ModelLineage]:
        """Get the currently active model's lineage."""
        row = self._conn.execute(
            "SELECT * FROM model_lineage WHERE status = 'active' LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return self._row_to_model_lineage(row)

    def get_model_history(self, limit: int = 20) -> list[ModelLineage]:
        """Get model version history ordered by training timestamp."""
        rows = self._conn.execute(
            "SELECT * FROM model_lineage ORDER BY training_timestamp DESC LIMIT ?",
            [limit],
        ).fetchall()
        return [self._row_to_model_lineage(r) for r in rows]

    def _row_to_model_lineage(self, row: tuple) -> ModelLineage:
        return ModelLineage(
            model_version=row[0],
            dataset_id=row[1],
            feature_version_id=row[2],
            training_pipeline_id=row[3],
            parent_model_version=row[4],
            hyperparameters=json.loads(row[5]) if row[5] else {},
            training_metrics=json.loads(row[6]) if row[6] else {},
            training_duration_seconds=row[7] or 0.0,
            training_timestamp=row[8],
            promotion_timestamp=row[9],
            demotion_timestamp=row[10],
            status=row[11],
        )

    # --- Prediction Tracking ---

    def record_prediction(
        self,
        prediction_id: str,
        model_version: str,
        feature_version_id: str,
        feature_snapshot_id: str,
        symbol: str,
        predicted_direction: str,
        predicted_confidence: float,
    ) -> None:
        """Record a prediction for lineage tracking."""
        with self._write_lock:
            now = datetime.now(timezone.utc)
            self._conn.execute(
                """
                INSERT INTO prediction_records (
                    prediction_id, model_version, feature_version_id,
                    feature_snapshot_id, symbol, timestamp,
                    predicted_direction, predicted_confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    prediction_id, model_version, feature_version_id,
                    feature_snapshot_id, symbol, now,
                    predicted_direction, predicted_confidence, now,
                ],
            )

    def link_prediction_to_trade(self, prediction_id: str, trade_id: str) -> None:
        """Link a prediction to its executed trade."""
        with self._write_lock:
            self._conn.execute(
                "UPDATE prediction_records SET trade_id = ? WHERE prediction_id = ?",
                [trade_id, prediction_id],
            )

    def record_prediction_outcome(self, prediction_id: str, profitable: bool) -> None:
        """Record whether prediction was correct."""
        with self._write_lock:
            self._conn.execute(
                "UPDATE prediction_records SET outcome_profitable = ? WHERE prediction_id = ?",
                [profitable, prediction_id],
            )

    def get_prediction_lineage(self, trade_id: str) -> Optional[dict]:
        """Full lineage from trade back to dataset."""
        row = self._conn.execute(
            """
            SELECT
                p.trade_id,
                p.prediction_id,
                p.model_version,
                p.feature_version_id,
                p.feature_snapshot_id,
                m.dataset_id,
                m.training_timestamp,
                d.source,
                d.row_count
            FROM prediction_records p
            LEFT JOIN model_lineage m ON p.model_version = m.model_version
            LEFT JOIN datasets d ON m.dataset_id = d.dataset_id
            WHERE p.trade_id = ?
            LIMIT 1
            """,
            [trade_id],
        ).fetchone()
        if row is None:
            return None
        return {
            "trade_id": row[0],
            "prediction_id": row[1],
            "model_version": row[2],
            "feature_version": row[3],
            "feature_snapshot_id": row[4],
            "model_trained_on": row[5],
            "training_timestamp": str(row[6]) if row[6] else None,
            "dataset_source": row[7],
            "dataset_row_count": row[8],
        }

    def reproduce_prediction(self, prediction_id: str) -> dict:
        """Load everything needed to reproduce a historical prediction."""
        row = self._conn.execute(
            """
            SELECT
                p.prediction_id,
                p.model_version,
                p.feature_version_id,
                p.feature_snapshot_id,
                p.symbol,
                p.predicted_direction,
                p.predicted_confidence,
                p.timestamp,
                m.dataset_id,
                m.hyperparameters,
                m.training_pipeline_id,
                d.parquet_path,
                d.data_hash
            FROM prediction_records p
            LEFT JOIN model_lineage m ON p.model_version = m.model_version
            LEFT JOIN datasets d ON m.dataset_id = d.dataset_id
            WHERE p.prediction_id = ?
            LIMIT 1
            """,
            [prediction_id],
        ).fetchone()
        if row is None:
            return {}
        return {
            "prediction_id": row[0],
            "model_version": row[1],
            "feature_version_id": row[2],
            "feature_snapshot_id": row[3],
            "symbol": row[4],
            "predicted_direction": row[5],
            "predicted_confidence": row[6],
            "prediction_timestamp": str(row[7]) if row[7] else None,
            "dataset_id": row[8],
            "hyperparameters": json.loads(row[9]) if row[9] else {},
            "training_pipeline_id": row[10],
            "parquet_path": row[11],
            "dataset_hash": row[12],
        }

    @property
    def dataset_count(self) -> int:
        result = self._conn.execute("SELECT COUNT(*) FROM datasets").fetchone()
        return result[0]

    @property
    def model_count(self) -> int:
        result = self._conn.execute("SELECT COUNT(*) FROM model_lineage").fetchone()
        return result[0]

    @property
    def prediction_count(self) -> int:
        result = self._conn.execute("SELECT COUNT(*) FROM prediction_records").fetchone()
        return result[0]

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

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker

from src.ml.models import MLBase, MLDataset, MLModelLineage, MLPredictionRecord
from src.utils.database import create_db_engine
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

    def __init__(self, storage_path: str = "data_cache/datasets", database_url: str = None, artifact_store=None) -> None:
        self._storage_path = Path(storage_path)
        self._snapshots_path = self._storage_path / "snapshots"
        self._write_lock = threading.Lock()
        self._artifact_store = artifact_store
        self._blob_container = "datasets"

        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._snapshots_path.mkdir(parents=True, exist_ok=True)

        if database_url:
            self._engine = create_db_engine(database_url)
        else:
            db_path = self._storage_path / "lineage.db"
            self._engine = create_db_engine(f"sqlite:///{db_path}")

        MLBase.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        logger.info("dataset_registry_initialized", storage_path=storage_path)

    def _compute_hash(self, data: pd.DataFrame) -> str:
        """Compute SHA256 hash of sorted DataFrame bytes."""
        sorted_df = data.sort_index(axis=1).sort_index(axis=0)
        content = sorted_df.to_csv(index=True).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def _next_version(self) -> int:
        with self._Session() as session:
            result = session.query(func.max(MLDataset.version)).scalar()
            return (result or 0) + 1

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

            # Save parquet snapshot (with CSV fallback if pyarrow unavailable)
            parquet_path = str(self._snapshots_path / f"{dataset_id}.parquet")
            try:
                data.to_parquet(parquet_path)
            except ImportError:
                # pyarrow/fastparquet not available — fall back to CSV
                csv_path = str(self._snapshots_path / f"{dataset_id}.csv")
                data.to_csv(csv_path)
                parquet_path = csv_path
                logger.warning(
                    "dataset_registry.parquet_unavailable",
                    msg="pyarrow not installed, falling back to CSV snapshot",
                    dataset_id=dataset_id,
                )

            # Upload snapshot to blob storage if configured
            if self._artifact_store:
                try:
                    snapshot_file = Path(parquet_path)
                    blob_name = snapshot_file.name
                    self._artifact_store.upload(
                        self._blob_container, blob_name, snapshot_file.read_bytes()
                    )
                except Exception as exc:
                    logger.warning(
                        "dataset_registry.blob_upload_failed",
                        dataset_id=dataset_id,
                        error=str(exc),
                    )

            now = datetime.now(timezone.utc)
            record = MLDataset(
                dataset_id=dataset_id,
                version=version,
                symbols=json.dumps(symbols),
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                row_count=len(data),
                feature_version_id=feature_version_id,
                data_hash=data_hash,
                source=source,
                description=description,
                parent_dataset_id=parent_dataset_id,
                parquet_path=parquet_path,
                created_at=now,
            )
            with self._Session() as session:
                session.add(record)
                session.commit()

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
        with self._Session() as session:
            row = session.query(MLDataset).filter(
                MLDataset.dataset_id == dataset_id
            ).first()
            if row is None:
                return None
            return self._row_to_dataset_version(row)

    def load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load the actual dataset file (Parquet or CSV fallback). Checks blob if local missing."""
        with self._Session() as session:
            row = session.query(MLDataset.parquet_path).filter(
                MLDataset.dataset_id == dataset_id
            ).first()
            if row is None or row[0] is None:
                return None
            parquet_path = row[0]

        path = Path(parquet_path)

        # If local file missing, try to restore from blob
        if not path.exists() and self._artifact_store:
            try:
                blob_name = path.name
                data = self._artifact_store.download(self._blob_container, blob_name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                logger.info(
                    "dataset_registry.blob_download_restored",
                    dataset_id=dataset_id,
                    blob_name=blob_name,
                )
            except Exception as exc:
                logger.warning(
                    "dataset_registry.blob_download_failed",
                    dataset_id=dataset_id,
                    error=str(exc),
                )

        if not path.exists():
            logger.warning("dataset_file_missing", dataset_id=dataset_id, path=str(path))
            return None
        if path.suffix == ".csv":
            return pd.read_csv(path, index_col=0)
        try:
            return pd.read_parquet(path)
        except ImportError:
            logger.warning("dataset_registry.read_parquet_unavailable", dataset_id=dataset_id)
            return None

    def get_latest_dataset(self) -> Optional[DatasetVersion]:
        """Get the most recent dataset version."""
        with self._Session() as session:
            row = session.query(MLDataset).order_by(
                MLDataset.version.desc()
            ).first()
            if row is None:
                return None
            return self._row_to_dataset_version(row)

    def _row_to_dataset_version(self, row: MLDataset) -> DatasetVersion:
        return DatasetVersion(
            dataset_id=row.dataset_id,
            version=row.version,
            symbols=json.loads(row.symbols),
            timeframe=row.timeframe,
            start_date=row.start_date,
            end_date=row.end_date,
            row_count=row.row_count,
            feature_version_id=row.feature_version_id,
            data_hash=row.data_hash,
            source=row.source,
            description=row.description,
            parent_dataset_id=row.parent_dataset_id,
            created_at=row.created_at,
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
            record = MLModelLineage(
                model_version=model_version,
                dataset_id=dataset_id,
                feature_version_id=feature_version_id,
                training_pipeline_id=pipeline_id,
                parent_model_version=parent_model_version,
                hyperparameters=json.dumps(hyperparameters or {}),
                training_metrics=json.dumps(training_metrics or {}),
                training_duration_seconds=training_duration,
                training_timestamp=now,
                status="registered",
            )
            with self._Session() as session:
                session.add(record)
                session.commit()

            logger.info(
                "model_registered",
                model_version=model_version,
                dataset_id=dataset_id,
            )

    def get_model_lineage(self, model_version: str) -> Optional[ModelLineage]:
        """Get full lineage for a model version."""
        with self._Session() as session:
            row = session.query(MLModelLineage).filter(
                MLModelLineage.model_version == model_version
            ).first()
            if row is None:
                return None
            return self._row_to_model_lineage(row)

    def set_model_status(self, model_version: str, status: str) -> None:
        """Update model status (active, demoted, rolled_back)."""
        with self._write_lock:
            now = datetime.now(timezone.utc)
            with self._Session() as session:
                if status == "active":
                    # Demote current active model
                    session.query(MLModelLineage).filter(
                        MLModelLineage.status == "active"
                    ).update({
                        MLModelLineage.status: "demoted",
                        MLModelLineage.demotion_timestamp: now,
                    })
                    session.query(MLModelLineage).filter(
                        MLModelLineage.model_version == model_version
                    ).update({
                        MLModelLineage.status: status,
                        MLModelLineage.promotion_timestamp: now,
                    })
                elif status == "demoted":
                    session.query(MLModelLineage).filter(
                        MLModelLineage.model_version == model_version
                    ).update({
                        MLModelLineage.status: status,
                        MLModelLineage.demotion_timestamp: now,
                    })
                else:
                    session.query(MLModelLineage).filter(
                        MLModelLineage.model_version == model_version
                    ).update({
                        MLModelLineage.status: status,
                    })
                session.commit()

            logger.info("model_status_updated", model_version=model_version, status=status)

    def get_active_model(self) -> Optional[ModelLineage]:
        """Get the currently active model's lineage."""
        with self._Session() as session:
            row = session.query(MLModelLineage).filter(
                MLModelLineage.status == "active"
            ).first()
            if row is None:
                return None
            return self._row_to_model_lineage(row)

    def get_model_history(self, limit: int = 20) -> list[ModelLineage]:
        """Get model version history ordered by training timestamp."""
        with self._Session() as session:
            rows = session.query(MLModelLineage).order_by(
                MLModelLineage.training_timestamp.desc()
            ).limit(limit).all()
            return [self._row_to_model_lineage(r) for r in rows]

    def _row_to_model_lineage(self, row: MLModelLineage) -> ModelLineage:
        return ModelLineage(
            model_version=row.model_version,
            dataset_id=row.dataset_id,
            feature_version_id=row.feature_version_id,
            training_pipeline_id=row.training_pipeline_id,
            parent_model_version=row.parent_model_version,
            hyperparameters=json.loads(row.hyperparameters) if row.hyperparameters else {},
            training_metrics=json.loads(row.training_metrics) if row.training_metrics else {},
            training_duration_seconds=row.training_duration_seconds or 0.0,
            training_timestamp=row.training_timestamp,
            promotion_timestamp=row.promotion_timestamp,
            demotion_timestamp=row.demotion_timestamp,
            status=row.status,
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
            record = MLPredictionRecord(
                prediction_id=prediction_id,
                model_version=model_version,
                feature_version_id=feature_version_id,
                feature_snapshot_id=feature_snapshot_id,
                symbol=symbol,
                timestamp=now,
                predicted_direction=predicted_direction,
                predicted_confidence=predicted_confidence,
                created_at=now,
            )
            with self._Session() as session:
                session.add(record)
                session.commit()

    def link_prediction_to_trade(self, prediction_id: str, trade_id: str) -> None:
        """Link a prediction to its executed trade."""
        with self._write_lock:
            with self._Session() as session:
                session.query(MLPredictionRecord).filter(
                    MLPredictionRecord.prediction_id == prediction_id
                ).update({MLPredictionRecord.trade_id: trade_id})
                session.commit()

    def record_prediction_outcome(self, prediction_id: str, profitable: bool) -> None:
        """Record whether prediction was correct."""
        with self._write_lock:
            with self._Session() as session:
                session.query(MLPredictionRecord).filter(
                    MLPredictionRecord.prediction_id == prediction_id
                ).update({MLPredictionRecord.outcome_profitable: profitable})
                session.commit()

    def get_prediction_lineage(self, trade_id: str) -> Optional[dict]:
        """Full lineage from trade back to dataset."""
        with self._Session() as session:
            pred = session.query(MLPredictionRecord).filter(
                MLPredictionRecord.trade_id == trade_id
            ).first()
            if pred is None:
                return None

            model = session.query(MLModelLineage).filter(
                MLModelLineage.model_version == pred.model_version
            ).first()

            dataset = None
            if model:
                dataset = session.query(MLDataset).filter(
                    MLDataset.dataset_id == model.dataset_id
                ).first()

            return {
                "trade_id": pred.trade_id,
                "prediction_id": pred.prediction_id,
                "model_version": pred.model_version,
                "feature_version": pred.feature_version_id,
                "feature_snapshot_id": pred.feature_snapshot_id,
                "model_trained_on": model.dataset_id if model else None,
                "training_timestamp": str(model.training_timestamp) if model and model.training_timestamp else None,
                "dataset_source": dataset.source if dataset else None,
                "dataset_row_count": dataset.row_count if dataset else None,
            }

    def reproduce_prediction(self, prediction_id: str) -> dict:
        """Load everything needed to reproduce a historical prediction."""
        with self._Session() as session:
            pred = session.query(MLPredictionRecord).filter(
                MLPredictionRecord.prediction_id == prediction_id
            ).first()
            if pred is None:
                return {}

            model = session.query(MLModelLineage).filter(
                MLModelLineage.model_version == pred.model_version
            ).first()

            dataset = None
            if model:
                dataset = session.query(MLDataset).filter(
                    MLDataset.dataset_id == model.dataset_id
                ).first()

            return {
                "prediction_id": pred.prediction_id,
                "model_version": pred.model_version,
                "feature_version_id": pred.feature_version_id,
                "feature_snapshot_id": pred.feature_snapshot_id,
                "symbol": pred.symbol,
                "predicted_direction": pred.predicted_direction,
                "predicted_confidence": pred.predicted_confidence,
                "prediction_timestamp": str(pred.timestamp) if pred.timestamp else None,
                "dataset_id": model.dataset_id if model else None,
                "hyperparameters": json.loads(model.hyperparameters) if model and model.hyperparameters else {},
                "training_pipeline_id": model.training_pipeline_id if model else None,
                "parquet_path": dataset.parquet_path if dataset else None,
                "dataset_hash": dataset.data_hash if dataset else None,
            }

    @property
    def dataset_count(self) -> int:
        with self._Session() as session:
            return session.query(func.count(MLDataset.dataset_id)).scalar()

    @property
    def model_count(self) -> int:
        with self._Session() as session:
            return session.query(func.count(MLModelLineage.model_version)).scalar()

    @property
    def prediction_count(self) -> int:
        with self._Session() as session:
            return session.query(func.count(MLPredictionRecord.prediction_id)).scalar()

    # --- Convenience Integration API ---

    def register_training_run(
        self,
        model_version: str,
        dataset_df: pd.DataFrame,
        symbols: list[str],
        feature_columns: list[str],
        pipeline_id: str,
        metrics: Optional[dict] = None,
        timeframe: str = "1Hour",
    ) -> str:
        """
        Register both dataset and model lineage in one call.

        Called from TrainingPipeline after model is trained and registered.
        Returns dataset_id.
        """
        # Register dataset
        feature_hash = hashlib.sha256(
            json.dumps(sorted(feature_columns)).encode()
        ).hexdigest()[:16]
        feature_version_id = f"fv_{feature_hash}"

        dataset_id = self.register_dataset(
            data=dataset_df,
            symbols=symbols,
            timeframe=timeframe,
            feature_version_id=feature_version_id,
            source="training_pipeline",
            description=f"Pipeline {pipeline_id}, {len(symbols)} symbols",
        )

        # Register model lineage
        self.register_model(
            model_version=model_version,
            dataset_id=dataset_id,
            feature_version_id=feature_version_id,
            pipeline_id=pipeline_id,
            hyperparameters=metrics.get("hyperparameters") if metrics else None,
            training_metrics=metrics,
        )

        return dataset_id

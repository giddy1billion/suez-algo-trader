"""
Dataset Builder — Accumulates validated outcomes into versioned datasets.

Outcomes from the Prediction Registry flow into versioned Parquet datasets
that are used for retraining. This ensures:
- Production models NEVER learn directly from each prediction
- Datasets are versioned and hashed for reproducibility
- Feature validation before inclusion
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.predictions.registry import PredictionRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetVersion:
    """Metadata for a versioned training dataset."""

    version: str
    created_at: str
    record_count: int
    feature_hash: str
    outcome_count: int
    symbols: list[str] = field(default_factory=list)
    date_range: tuple[str, str] = field(default_factory=lambda: ("", ""))
    quality_distribution: dict[str, int] = field(default_factory=dict)
    file_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "record_count": self.record_count,
            "feature_hash": self.feature_hash,
            "outcome_count": self.outcome_count,
            "symbols": self.symbols,
            "date_range": list(self.date_range),
            "quality_distribution": self.quality_distribution,
            "file_path": self.file_path,
        }


class DatasetBuilder:
    """
    Builds versioned training datasets from validated prediction outcomes.

    The builder:
    1. Accumulates resolved predictions with outcomes
    2. Validates feature completeness
    3. Creates versioned Parquet datasets
    4. Maintains a manifest of all dataset versions
    """

    def __init__(
        self,
        storage_dir: str = "data_cache/datasets",
        min_records: int = 500,
    ):
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._min_records = min_records
        self._manifest: list[DatasetVersion] = []
        self._load_manifest()

    def build_dataset(
        self,
        predictions: list[PredictionRecord],
        features_df: Optional[pd.DataFrame] = None,
    ) -> Optional[DatasetVersion]:
        """
        Build a new versioned dataset from resolved predictions.

        Args:
            predictions: Resolved predictions with outcomes
            features_df: Optional feature DataFrame to merge

        Returns:
            DatasetVersion if successful, None if insufficient data
        """
        # Filter to resolved predictions only
        resolved = [p for p in predictions if p.resolved and p.actual_return is not None]

        if len(resolved) < self._min_records:
            logger.info(
                "dataset_builder.insufficient_data",
                have=len(resolved),
                need=self._min_records,
            )
            return None

        # Build DataFrame from predictions
        records = []
        for p in resolved:
            records.append({
                "prediction_id": p.prediction_id,
                "timestamp": p.timestamp,
                "asset": p.asset,
                "direction": p.direction,
                "confidence": p.confidence,
                "expected_return": p.expected_return,
                "actual_return": p.actual_return,
                "direction_correct": p.direction_correct,
                "quality_grade": p.quality_grade,
                "model_version": p.model_version,
                "strategy": p.strategy,
                "features_hash": p.features_snapshot_hash,
            })

        df = pd.DataFrame(records)

        # Merge features if provided
        if features_df is not None and not features_df.empty:
            df = df.merge(features_df, on="prediction_id", how="left")

        # Compute version hash
        content_hash = hashlib.sha256(
            df.to_json().encode()
        ).hexdigest()[:12]
        version = f"ds_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{content_hash}"

        # Save as Parquet
        file_path = self._storage_dir / f"{version}.parquet"
        df.to_parquet(file_path, index=False)

        # Create version metadata
        symbols = sorted(df["asset"].unique().tolist())
        timestamps = pd.to_datetime(df["timestamp"])
        quality_dist = df["quality_grade"].value_counts().to_dict()

        dataset_version = DatasetVersion(
            version=version,
            created_at=datetime.now(timezone.utc).isoformat(),
            record_count=len(df),
            feature_hash=content_hash,
            outcome_count=len(resolved),
            symbols=symbols,
            date_range=(
                str(timestamps.min()) if len(timestamps) > 0 else "",
                str(timestamps.max()) if len(timestamps) > 0 else "",
            ),
            quality_distribution=quality_dist,
            file_path=str(file_path),
        )

        self._manifest.append(dataset_version)
        self._save_manifest()

        logger.info(
            "dataset_builder.built",
            version=version,
            records=len(df),
            symbols=len(symbols),
        )

        return dataset_version

    def get_latest_dataset(self) -> Optional[DatasetVersion]:
        """Get the most recent dataset version."""
        return self._manifest[-1] if self._manifest else None

    def get_dataset(self, version: str) -> Optional[pd.DataFrame]:
        """Load a specific dataset version."""
        for dv in self._manifest:
            if dv.version == version:
                path = Path(dv.file_path)
                if path.exists():
                    return pd.read_parquet(path)
        return None

    def list_versions(self, limit: int = 10) -> list[dict]:
        """List recent dataset versions."""
        return [v.to_dict() for v in self._manifest[-limit:]]

    @property
    def total_versions(self) -> int:
        return len(self._manifest)

    def _load_manifest(self) -> None:
        """Load dataset manifest from storage."""
        manifest_file = self._storage_dir / "manifest.json"
        if manifest_file.exists():
            try:
                with open(manifest_file) as f:
                    data = json.load(f)
                for item in data:
                    self._manifest.append(DatasetVersion(
                        version=item["version"],
                        created_at=item["created_at"],
                        record_count=item["record_count"],
                        feature_hash=item["feature_hash"],
                        outcome_count=item["outcome_count"],
                        symbols=item.get("symbols", []),
                        date_range=tuple(item.get("date_range", ("", ""))),
                        quality_distribution=item.get("quality_distribution", {}),
                        file_path=item.get("file_path", ""),
                    ))
            except Exception as e:
                logger.warning("dataset_builder.manifest_load_failed", error=str(e))

    def _save_manifest(self) -> None:
        """Save dataset manifest to storage."""
        manifest_file = self._storage_dir / "manifest.json"
        try:
            data = [v.to_dict() for v in self._manifest]
            with open(manifest_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("dataset_builder.manifest_save_failed", error=str(e))

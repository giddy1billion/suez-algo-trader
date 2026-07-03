"""
Feature Store — Versioned, cached feature engineering for reproducibility.

Decouples feature computation from training/inference code.
Ensures identical features are used in training and production.

Key properties:
- Each feature set is versioned (hash of schema + computation params)
- Features are cached to disk (parquet) for fast retrieval
- Schema validation ensures train/inference consistency
- Metadata tracks lineage: computation time, row count, source data hash
"""

import hashlib
import json
import os
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureSetMetadata:
    """Metadata for a computed feature set."""
    version: str  # Hash-based version
    schema: List[str]  # Ordered list of feature column names
    n_rows: int
    n_features: int
    computed_at: str
    computation_seconds: float
    source_data_hash: str
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "schema": self.schema,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "computed_at": self.computed_at,
            "computation_seconds": self.computation_seconds,
            "source_data_hash": self.source_data_hash,
            "parameters": self.parameters,
        }


class FeatureStore:
    """
    Versioned feature store with caching and schema validation.
    
    Usage:
        store = FeatureStore()
        
        # Compute and store features
        metadata = store.put(features_df, symbol="AAPL", params={"timeframe": "1H"})
        
        # Retrieve cached features
        df = store.get(symbol="AAPL", version=metadata.version)
        
        # Validate schema consistency
        is_valid = store.validate_schema(features_df, expected_version="abc123")
    """
    
    def __init__(self, store_dir: str = "data_cache/features"):
        self.store_dir = store_dir
        self._lock = threading.Lock()
        Path(store_dir).mkdir(parents=True, exist_ok=True)
        self._registry_path = os.path.join(store_dir, "registry.json")
    
    def put(self, features: pd.DataFrame, symbol: str, 
            params: Optional[dict] = None) -> FeatureSetMetadata:
        """
        Store a computed feature set.
        
        Args:
            features: DataFrame of computed features.
            symbol: Asset symbol this feature set belongs to.
            params: Computation parameters (timeframe, lookback, etc.)
        
        Returns:
            FeatureSetMetadata with version hash and schema info.
        """
        params = params or {}
        start_time = time.time()
        
        # Compute version hash from schema + params
        schema = sorted(features.columns.tolist())
        version = self._compute_version(schema, params)
        
        # Compute source data hash (for provenance)
        source_hash = self._hash_dataframe(features)
        
        # Save to parquet
        file_path = self._get_path(symbol, version)
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(file_path, index=True)
        
        computation_time = time.time() - start_time
        
        metadata = FeatureSetMetadata(
            version=version,
            schema=schema,
            n_rows=len(features),
            n_features=len(schema),
            computed_at=datetime.now(timezone.utc).isoformat(),
            computation_seconds=computation_time,
            source_data_hash=source_hash,
            parameters=params,
        )
        
        # Update registry
        self._update_registry(symbol, metadata)
        
        logger.info(
            "feature_store.put",
            symbol=symbol,
            version=version[:8],
            rows=len(features),
            features=len(schema),
        )
        return metadata
    
    def get(self, symbol: str, version: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Retrieve cached features.
        
        Args:
            symbol: Asset symbol.
            version: Specific version (None = latest).
        
        Returns:
            DataFrame or None if not found.
        """
        if version is None:
            version = self._get_latest_version(symbol)
            if version is None:
                return None
        
        file_path = self._get_path(symbol, version)
        if not os.path.exists(file_path):
            return None
        
        return pd.read_parquet(file_path)
    
    def get_metadata(self, symbol: str, version: Optional[str] = None) -> Optional[FeatureSetMetadata]:
        """Get metadata for a feature set version."""
        registry = self._load_registry()
        entries = registry.get(symbol, [])
        
        if not entries:
            return None
        
        if version is None:
            entry = entries[-1]  # Latest
        else:
            entry = next((e for e in entries if e["version"] == version), None)
        
        if entry is None:
            return None
        
        return FeatureSetMetadata(**entry)
    
    def validate_schema(self, features: pd.DataFrame, expected_version: str,
                        symbol: str = "") -> tuple:
        """
        Validate that a feature DataFrame matches an expected schema version.
        
        Returns (is_valid, list_of_issues).
        """
        issues = []
        metadata = self.get_metadata(symbol, expected_version) if symbol else None
        
        if metadata is None:
            # Compute expected schema from version
            current_schema = sorted(features.columns.tolist())
            current_version = self._compute_version(current_schema, {})
            if current_version != expected_version:
                issues.append(f"Schema version mismatch: got {current_version[:8]}, expected {expected_version[:8]}")
            return len(issues) == 0, issues
        
        expected_schema = set(metadata.schema)
        actual_schema = set(features.columns.tolist())
        
        missing = expected_schema - actual_schema
        extra = actual_schema - expected_schema
        
        if missing:
            issues.append(f"Missing features: {sorted(missing)}")
        if extra:
            issues.append(f"Unexpected features: {sorted(extra)}")
        if features.shape[1] != metadata.n_features:
            issues.append(f"Feature count mismatch: got {features.shape[1]}, expected {metadata.n_features}")
        
        return len(issues) == 0, issues
    
    def list_versions(self, symbol: str) -> List[dict]:
        """List all stored versions for a symbol."""
        registry = self._load_registry()
        return registry.get(symbol, [])
    
    def list_symbols(self) -> List[str]:
        """List all symbols with stored features."""
        registry = self._load_registry()
        return list(registry.keys())
    
    def cleanup(self, symbol: str, keep_latest: int = 5) -> int:
        """Remove old versions, keeping the N most recent."""
        registry = self._load_registry()
        entries = registry.get(symbol, [])
        
        if len(entries) <= keep_latest:
            return 0
        
        to_remove = entries[:-keep_latest]
        removed = 0
        
        for entry in to_remove:
            file_path = self._get_path(symbol, entry["version"])
            if os.path.exists(file_path):
                os.remove(file_path)
                removed += 1
        
        registry[symbol] = entries[-keep_latest:]
        self._save_registry(registry)
        return removed
    
    # --- Private ---
    
    def _compute_version(self, schema: List[str], params: dict) -> str:
        """Deterministic version hash from schema + params."""
        content = json.dumps({"schema": schema, "params": params}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Hash a DataFrame for provenance."""
        try:
            data_hash = hashlib.sha256(
                pd.util.hash_pandas_object(df).values.tobytes()
            ).hexdigest()[:16]
            return data_hash
        except Exception:
            return ""
    
    def _get_path(self, symbol: str, version: str) -> str:
        """Get file path for a feature set."""
        safe_symbol = symbol.replace("/", "_").replace("\\", "_")
        return os.path.join(self.store_dir, safe_symbol, f"{version}.parquet")
    
    def _get_latest_version(self, symbol: str) -> Optional[str]:
        """Get the latest version for a symbol."""
        registry = self._load_registry()
        entries = registry.get(symbol, [])
        return entries[-1]["version"] if entries else None
    
    def _update_registry(self, symbol: str, metadata: FeatureSetMetadata):
        """Append metadata to registry."""
        with self._lock:
            registry = self._load_registry()
            if symbol not in registry:
                registry[symbol] = []
            # Replace if same version exists, else append
            existing_versions = {e["version"] for e in registry[symbol]}
            if metadata.version in existing_versions:
                registry[symbol] = [
                    metadata.to_dict() if e["version"] == metadata.version else e
                    for e in registry[symbol]
                ]
            else:
                registry[symbol].append(metadata.to_dict())
            self._save_registry(registry)
    
    def _load_registry(self) -> dict:
        if not os.path.exists(self._registry_path):
            return {}
        try:
            with open(self._registry_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    
    def _save_registry(self, registry: dict):
        with open(self._registry_path, "w") as f:
            json.dump(registry, f, indent=2, default=str)

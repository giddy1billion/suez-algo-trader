"""
Tests for Feature Store, Adaptive Snapshotting, and Immutable Governance.
"""

import os
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

from src.ml.feature_store import FeatureStore, FeatureSetMetadata
from src.core.snapshots import SnapshotStore, SnapshotManager
from src.ml.governance import ModelGovernance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Create a temp directory for test artifacts."""
    d = os.path.join(os.path.dirname(__file__), "_test_artifacts")
    os.makedirs(d, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def feature_store(tmp_dir):
    return FeatureStore(store_dir=os.path.join(tmp_dir, "features"))


@pytest.fixture
def sample_features():
    """Sample feature DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        "rsi_14": np.random.rand(100),
        "ema_slope_20": np.random.rand(100),
        "volume_ratio": np.random.rand(100),
    })


@pytest.fixture
def snapshot_store(tmp_dir):
    db_path = os.path.join(tmp_dir, "snapshots.db")
    store = SnapshotStore(db_path=db_path)
    yield store
    store.close()


@pytest.fixture
def governance(tmp_dir):
    return ModelGovernance(governance_dir=os.path.join(tmp_dir, "governance"))


# ---------------------------------------------------------------------------
# Feature Store Tests
# ---------------------------------------------------------------------------

class TestFeatureStore:
    def test_put_get_roundtrip(self, feature_store, sample_features):
        """Stored features match retrieved features exactly."""
        metadata = feature_store.put(sample_features, symbol="AAPL")
        retrieved = feature_store.get(symbol="AAPL", version=metadata.version)

        assert retrieved is not None
        pd.testing.assert_frame_equal(retrieved, sample_features)

    def test_different_schemas_different_versions(self, feature_store):
        """Different schemas produce different versions."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

        meta1 = feature_store.put(df1, symbol="AAPL")
        meta2 = feature_store.put(df2, symbol="AAPL")

        assert meta1.version != meta2.version

    def test_same_schema_same_version(self, feature_store):
        """Same schema + params produce same version (deterministic)."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})

        meta1 = feature_store.put(df1, symbol="AAPL", params={"tf": "1H"})
        meta2 = feature_store.put(df2, symbol="AAPL", params={"tf": "1H"})

        assert meta1.version == meta2.version

    def test_validate_schema_correct(self, feature_store, sample_features):
        """Correct schema passes validation."""
        metadata = feature_store.put(sample_features, symbol="AAPL")
        is_valid, issues = feature_store.validate_schema(
            sample_features, expected_version=metadata.version, symbol="AAPL"
        )
        assert is_valid
        assert issues == []

    def test_validate_schema_wrong(self, feature_store, sample_features):
        """Wrong schema fails validation."""
        metadata = feature_store.put(sample_features, symbol="AAPL")
        wrong_df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        is_valid, issues = feature_store.validate_schema(
            wrong_df, expected_version=metadata.version, symbol="AAPL"
        )
        assert not is_valid
        assert len(issues) > 0

    def test_list_versions(self, feature_store):
        """Tracks multiple versions."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        df3 = pd.DataFrame({"p": [1], "q": [2], "r": [3]})

        feature_store.put(df1, symbol="BTC")
        feature_store.put(df2, symbol="BTC")
        feature_store.put(df3, symbol="BTC")

        versions = feature_store.list_versions("BTC")
        assert len(versions) == 3

    def test_cleanup(self, feature_store):
        """Removes old versions correctly."""
        for i in range(5):
            df = pd.DataFrame({f"col_{i}": [1, 2, 3]})
            feature_store.put(df, symbol="ETH")

        versions_before = feature_store.list_versions("ETH")
        assert len(versions_before) == 5

        removed = feature_store.cleanup("ETH", keep_latest=2)
        assert removed == 3

        versions_after = feature_store.list_versions("ETH")
        assert len(versions_after) == 2

    def test_metadata(self, feature_store, sample_features):
        """Records computation time, row count, source hash."""
        metadata = feature_store.put(sample_features, symbol="AAPL", params={"tf": "1H"})

        assert metadata.n_rows == 100
        assert metadata.n_features == 3
        assert metadata.computation_seconds >= 0
        assert metadata.source_data_hash != ""
        assert metadata.computed_at != ""
        assert metadata.parameters == {"tf": "1H"}


# ---------------------------------------------------------------------------
# Adaptive Snapshot Tests
# ---------------------------------------------------------------------------

class TestAdaptiveSnapshot:
    def test_time_based_trigger(self, snapshot_store):
        """Time-based trigger works."""
        manager = SnapshotManager(
            snapshot_store,
            snapshot_interval_events=1000,
            snapshot_interval_seconds=0.1,  # 100ms for testing
        )
        # No events processed, but wait for time trigger
        assert not manager.should_snapshot()
        time.sleep(0.15)
        assert manager.should_snapshot()

    def test_event_based_trigger(self, snapshot_store):
        """Event-based trigger still works."""
        manager = SnapshotManager(
            snapshot_store,
            snapshot_interval_events=3,
            snapshot_interval_seconds=9999,  # Won't trigger by time
        )
        manager.on_event(None)
        manager.on_event(None)
        assert not manager.should_snapshot()
        manager.on_event(None)
        assert manager.should_snapshot()

    def test_force_snapshot(self, snapshot_store):
        """Force snapshot always works regardless of interval."""
        manager = SnapshotManager(
            snapshot_store,
            snapshot_interval_events=9999,
            snapshot_interval_seconds=9999,
        )
        # Neither trigger met
        assert not manager.should_snapshot()

        # Force works anyway
        snapshot_id = manager.force_snapshot(
            session_id="test_session",
            last_event_id=42,
            state={"portfolio_value": 100000},
            reason="graceful_shutdown",
        )
        assert snapshot_id is not None
        assert snapshot_id > 0

        # Verify it was saved
        latest = snapshot_store.get_latest_snapshot("test_session")
        assert latest is not None
        assert latest["last_event_id"] == 42

    def test_take_snapshot_resets_both_counters(self, snapshot_store):
        """take_snapshot resets both event count and time."""
        manager = SnapshotManager(
            snapshot_store,
            snapshot_interval_events=2,
            snapshot_interval_seconds=0.1,
        )
        manager.on_event(None)
        manager.on_event(None)
        assert manager.should_snapshot()

        manager.take_snapshot("s1", 10, {"val": 1})
        assert not manager.should_snapshot()


# ---------------------------------------------------------------------------
# Immutable Governance Tests
# ---------------------------------------------------------------------------

class TestImmutableGovernance:
    def test_deployed_record_immutable(self, governance):
        """Once deployed, record cannot be overwritten."""
        governance.record_training(
            version="v001",
            features=["f1", "f2"],
            metrics={"cv_accuracy": 0.7, "sharpe": 1.5},
            hyperparameters={"lr": 0.01},
            seed=42,
        )
        governance.deploy("v001", reason="initial")

        # Try to overwrite
        governance.record_training(
            version="v001",
            features=["f1", "f2", "f3"],  # different features
            metrics={"cv_accuracy": 0.9},
            hyperparameters={"lr": 0.02},
            seed=99,
        )

        # Original should be preserved
        lineage = governance.get_lineage("v001")
        assert lineage.n_features == 2  # Not 3
        assert lineage.random_seed == 42  # Not 99

    def test_verify_integrity_passes(self, governance):
        """Clean records pass integrity check."""
        governance.record_training(
            version="v001",
            features=["f1", "f2"],
            metrics={"cv_accuracy": 0.7},
            hyperparameters={"lr": 0.01},
            seed=42,
        )
        governance.deploy("v001", reason="test")

        is_valid, issues = governance.verify_integrity()
        assert is_valid, f"Unexpected issues: {issues}"

    def test_verify_integrity_detects_tampering(self, governance):
        """Tampered deployed records are detected."""
        governance.record_training(
            version="v001",
            features=["f1", "f2"],
            metrics={"cv_accuracy": 0.7},
            hyperparameters={"lr": 0.01},
            seed=42,
        )
        governance.deploy("v001", reason="test")

        # Tamper with the record directly
        import json
        with open(governance._lineage_path, "r") as f:
            data = json.load(f)
        data[0]["cv_accuracy"] = 0.99  # Tamper!
        with open(governance._lineage_path, "w") as f:
            json.dump(data, f)

        is_valid, issues = governance.verify_integrity()
        assert not is_valid
        assert any("integrity hash mismatch" in i for i in issues)

    def test_integrity_hash_stored(self, governance):
        """integrity_hash field is stored in lineage records."""
        governance.record_training(
            version="v001",
            features=["f1"],
            metrics={"cv_accuracy": 0.6},
            hyperparameters={"lr": 0.01},
            seed=1,
        )
        import json
        with open(governance._lineage_path, "r") as f:
            data = json.load(f)
        assert "integrity_hash" in data[0]
        assert len(data[0]["integrity_hash"]) == 16

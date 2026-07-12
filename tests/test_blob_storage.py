"""
Tests for blob_storage module — ArtifactStore abstraction.

Tests LocalArtifactStore fully, mocks AzureBlobArtifactStore,
and validates ModelRegistry/DatasetRegistry integration with artifact stores.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.blob_storage import (
    AzureBlobArtifactStore,
    LocalArtifactStore,
    create_artifact_store,
)


# ═══════════════════════════════════════════════════════════════════════════════
# LocalArtifactStore Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestLocalArtifactStore:
    """Test the filesystem-backed artifact store."""

    @pytest.fixture
    def store(self, tmp_path):
        return LocalArtifactStore(base_dir=str(tmp_path / "artifacts"))

    def test_upload_and_download(self, store):
        data = b"hello world model bytes"
        store.upload("models", "v001.joblib", data)
        result = store.download("models", "v001.joblib")
        assert result == data

    def test_exists_true(self, store):
        store.upload("models", "v001.joblib", b"data")
        assert store.exists("models", "v001.joblib") is True

    def test_exists_false(self, store):
        assert store.exists("models", "nonexistent.joblib") is False

    def test_list_blobs_empty(self, store):
        assert store.list_blobs("models") == []

    def test_list_blobs_with_files(self, store):
        store.upload("models", "v001.joblib", b"a")
        store.upload("models", "v002.joblib", b"b")
        store.upload("models", "subdir/v003.joblib", b"c")
        blobs = store.list_blobs("models")
        assert "v001.joblib" in blobs
        assert "v002.joblib" in blobs
        assert "subdir/v003.joblib" in blobs

    def test_list_blobs_with_prefix(self, store):
        store.upload("models", "v001.joblib", b"a")
        store.upload("models", "v002.joblib", b"b")
        store.upload("models", "archive/old.joblib", b"c")
        blobs = store.list_blobs("models", prefix="v00")
        assert "v001.joblib" in blobs
        assert "v002.joblib" in blobs
        assert "archive/old.joblib" not in blobs

    def test_delete(self, store):
        store.upload("models", "v001.joblib", b"data")
        assert store.exists("models", "v001.joblib") is True
        store.delete("models", "v001.joblib")
        assert store.exists("models", "v001.joblib") is False

    def test_delete_nonexistent_no_error(self, store):
        # Should not raise
        store.delete("models", "nonexistent.joblib")

    def test_download_nonexistent_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.download("models", "nonexistent.joblib")

    def test_upload_creates_subdirectories(self, store):
        store.upload("datasets", "2024/01/snapshot.parquet", b"parquet-data")
        assert store.exists("datasets", "2024/01/snapshot.parquet") is True
        assert store.download("datasets", "2024/01/snapshot.parquet") == b"parquet-data"

    def test_multiple_containers(self, store):
        store.upload("models", "model.joblib", b"model")
        store.upload("datasets", "data.parquet", b"dataset")
        store.upload("audit-logs", "log.jsonl", b"audit")
        assert store.exists("models", "model.joblib")
        assert store.exists("datasets", "data.parquet")
        assert store.exists("audit-logs", "log.jsonl")
        assert not store.exists("models", "data.parquet")


# ═══════════════════════════════════════════════════════════════════════════════
# AzureBlobArtifactStore Tests (Mocked)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAzureBlobArtifactStoreMocked:
    """Test Azure backend with mocked Azure SDK."""

    @pytest.fixture
    def mock_blob_service(self):
        with patch("src.utils.blob_storage.AzureBlobArtifactStore._create_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client
            store = AzureBlobArtifactStore(
                account_url="https://testaccount.blob.core.windows.net",
                container_prefix="suez-trader",
            )
            yield store, mock_client

    def test_container_name_formatting(self, mock_blob_service):
        store, _ = mock_blob_service
        assert store._container_name("models") == "suez-trader-models"
        assert store._container_name("audit-logs") == "suez-trader-audit-logs"
        assert store._container_name("MY_Container") == "suez-trader-my-container"

    def test_upload_calls_sdk(self, mock_blob_service):
        store, mock_client = mock_blob_service
        mock_container = MagicMock()
        mock_container.exists.return_value = True
        mock_client.get_container_client.return_value = mock_container

        mock_blob = MagicMock()
        mock_client.get_blob_client.return_value = mock_blob

        store.upload("models", "v001.joblib", b"model-data")
        mock_blob.upload_blob.assert_called_once_with(b"model-data", overwrite=True)

    def test_download_calls_sdk(self, mock_blob_service):
        store, mock_client = mock_blob_service
        mock_blob = MagicMock()
        mock_client.get_blob_client.return_value = mock_blob
        mock_blob.download_blob.return_value.readall.return_value = b"model-data"

        result = store.download("models", "v001.joblib")
        assert result == b"model-data"

    def test_exists_returns_true(self, mock_blob_service):
        store, mock_client = mock_blob_service
        mock_blob = MagicMock()
        mock_client.get_blob_client.return_value = mock_blob
        mock_blob.get_blob_properties.return_value = {}

        assert store.exists("models", "v001.joblib") is True

    def test_exists_returns_false_on_exception(self, mock_blob_service):
        store, mock_client = mock_blob_service
        mock_blob = MagicMock()
        mock_client.get_blob_client.return_value = mock_blob
        mock_blob.get_blob_properties.side_effect = Exception("Not found")

        assert store.exists("models", "v001.joblib") is False

    def test_delete_calls_sdk(self, mock_blob_service):
        store, mock_client = mock_blob_service
        mock_container = MagicMock()
        mock_container.exists.return_value = True
        mock_client.get_container_client.return_value = mock_container

        mock_blob = MagicMock()
        mock_client.get_blob_client.return_value = mock_blob

        store.delete("models", "v001.joblib")
        mock_blob.delete_blob.assert_called_once_with(delete_snapshots="include")


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCreateArtifactStore:
    """Test the factory function."""

    def test_empty_url_returns_local(self, tmp_path):
        store = create_artifact_store(
            blob_url="", local_base_dir=str(tmp_path / "local")
        )
        assert isinstance(store, LocalArtifactStore)

    def test_none_url_returns_local(self, tmp_path):
        store = create_artifact_store(
            blob_url="", local_base_dir=str(tmp_path / "local")
        )
        assert isinstance(store, LocalArtifactStore)

    @patch("src.utils.blob_storage.AzureBlobArtifactStore._create_client")
    def test_valid_url_returns_azure(self, mock_create, tmp_path):
        mock_create.return_value = MagicMock()
        store = create_artifact_store(
            blob_url="https://test.blob.core.windows.net",
            local_base_dir=str(tmp_path / "local"),
        )
        assert isinstance(store, AzureBlobArtifactStore)

    def test_invalid_url_falls_back_to_local(self, tmp_path):
        # Simulate Azure SDK import failure / connection failure
        with patch(
            "src.utils.blob_storage.AzureBlobArtifactStore._create_client",
            side_effect=Exception("Connection refused"),
        ):
            store = create_artifact_store(
                blob_url="https://bad.blob.core.windows.net",
                local_base_dir=str(tmp_path / "local"),
            )
            assert isinstance(store, LocalArtifactStore)


# ═══════════════════════════════════════════════════════════════════════════════
# ModelRegistry + ArtifactStore Integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelRegistryWithArtifactStore:
    """Test ModelRegistry uses artifact store for uploads/downloads."""

    @pytest.fixture
    def setup(self, tmp_path):
        store = LocalArtifactStore(base_dir=str(tmp_path / "blob"))
        from src.ml.model_registry import ModelRegistry

        registry = ModelRegistry(
            models_dir=str(tmp_path / "models"),
            artifact_store=store,
        )
        return registry, store, tmp_path

    def test_save_uploads_to_store(self, setup):
        registry, store, _ = setup
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier()
        model.fit([[1, 2], [3, 4]], [0, 1])

        version = registry.save_version(
            model=model,
            features=["f1", "f2"],
            metrics={"accuracy": 0.9},
            symbols=["AAPL"],
        )

        # Should exist in blob
        blobs = store.list_blobs("models")
        assert any(version in b for b in blobs)
        # latest_model.joblib should also be in blob
        assert store.exists("models", "latest_model.joblib")

    def test_get_version_downloads_from_store(self, setup):
        registry, store, tmp_path = setup
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier()
        model.fit([[1, 2], [3, 4]], [0, 1])

        version = registry.save_version(
            model=model,
            features=["f1", "f2"],
            metrics={"accuracy": 0.9},
            symbols=["AAPL"],
        )

        # Delete local file to force blob download
        models_dir = tmp_path / "models"
        for f in models_dir.glob("*.joblib"):
            if f.name != "registry.json":
                f.unlink()

        # Should still load from blob
        result = registry.get_version(version)
        assert result["features"] == ["f1", "f2"]
        assert result["metadata"]["version"] == version

    def test_no_artifact_store_works_normally(self, tmp_path):
        from src.ml.model_registry import ModelRegistry
        from sklearn.tree import DecisionTreeClassifier

        registry = ModelRegistry(models_dir=str(tmp_path / "models"))
        model = DecisionTreeClassifier()
        model.fit([[1, 2], [3, 4]], [0, 1])

        version = registry.save_version(
            model=model,
            features=["f1", "f2"],
            metrics={"accuracy": 0.9},
            symbols=["AAPL"],
        )
        result = registry.get_version(version)
        assert result["features"] == ["f1", "f2"]


# ═══════════════════════════════════════════════════════════════════════════════
# DatasetRegistry + ArtifactStore Integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestDatasetRegistryWithArtifactStore:
    """Test DatasetRegistry uses artifact store for uploads/downloads."""

    @pytest.fixture
    def setup(self, tmp_path):
        store = LocalArtifactStore(base_dir=str(tmp_path / "blob"))
        from src.ml.dataset_registry import DatasetRegistry

        registry = DatasetRegistry(
            storage_path=str(tmp_path / "datasets"),
            artifact_store=store,
        )
        return registry, store, tmp_path

    def test_register_uploads_to_store(self, setup):
        import pandas as pd

        registry, store, _ = setup
        df = pd.DataFrame({"price": [100.0, 101.0, 102.0]})

        dataset_id = registry.register_dataset(
            data=df,
            symbols=["AAPL"],
            timeframe="1d",
            feature_version_id="fv_001",
        )

        # Check blob has the file
        blobs = store.list_blobs("datasets")
        assert len(blobs) >= 1
        assert any(dataset_id in b for b in blobs)

    def test_load_dataset_downloads_from_store(self, setup):
        import pandas as pd

        registry, store, tmp_path = setup
        df = pd.DataFrame({"price": [100.0, 101.0, 102.0]})

        dataset_id = registry.register_dataset(
            data=df,
            symbols=["AAPL"],
            timeframe="1d",
            feature_version_id="fv_001",
        )

        # Delete local snapshot to force blob download
        snapshots_dir = tmp_path / "datasets" / "snapshots"
        for f in snapshots_dir.glob("*"):
            f.unlink()

        # Should still load from blob
        loaded = registry.load_dataset(dataset_id)
        assert loaded is not None
        assert len(loaded) == 3

    def test_no_artifact_store_works_normally(self, tmp_path):
        import pandas as pd
        from src.ml.dataset_registry import DatasetRegistry

        registry = DatasetRegistry(storage_path=str(tmp_path / "datasets"))
        df = pd.DataFrame({"price": [100.0, 101.0, 102.0]})

        dataset_id = registry.register_dataset(
            data=df,
            symbols=["AAPL"],
            timeframe="1d",
            feature_version_id="fv_001",
        )
        loaded = registry.load_dataset(dataset_id)
        assert loaded is not None
        assert len(loaded) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Live Azure Tests (gated by env var)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not os.environ.get("BLOB_STORAGE_URL"),
    reason="BLOB_STORAGE_URL not set — skipping live Azure tests",
)
class TestAzureBlobLive:
    """Live integration tests against real Azure Blob Storage."""

    @pytest.fixture
    def store(self):
        url = os.environ["BLOB_STORAGE_URL"]
        return AzureBlobArtifactStore(
            account_url=url, container_prefix="suez-trader-test"
        )

    def test_upload_download_roundtrip(self, store):
        data = b"live test data"
        store.upload("test", "live_test.bin", data)
        result = store.download("test", "live_test.bin")
        assert result == data
        store.delete("test", "live_test.bin")

    def test_exists_and_list(self, store):
        store.upload("test", "exists_test.bin", b"data")
        assert store.exists("test", "exists_test.bin")
        blobs = store.list_blobs("test", prefix="exists_")
        assert "exists_test.bin" in blobs
        store.delete("test", "exists_test.bin")
        assert not store.exists("test", "exists_test.bin")

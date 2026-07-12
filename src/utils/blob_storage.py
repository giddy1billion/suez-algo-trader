"""
Artifact Store — dual-backend abstraction for local filesystem and Azure Blob Storage.

Provides a unified interface for uploading, downloading, listing, and deleting
binary artifacts. Defaults to local filesystem when no Azure Blob URL is configured.
"""

import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ArtifactStore(ABC):
    """Abstract interface for artifact storage backends."""

    @abstractmethod
    def upload(self, container: str, blob_name: str, data: bytes) -> None:
        """Upload bytes to the given container/blob_name."""
        ...

    @abstractmethod
    def download(self, container: str, blob_name: str) -> bytes:
        """Download and return bytes from the given container/blob_name."""
        ...

    @abstractmethod
    def exists(self, container: str, blob_name: str) -> bool:
        """Check whether a blob exists."""
        ...

    @abstractmethod
    def list_blobs(self, container: str, prefix: str = "") -> list[str]:
        """List blob names in a container, optionally filtered by prefix."""
        ...

    @abstractmethod
    def delete(self, container: str, blob_name: str) -> None:
        """Delete a blob from the container."""
        ...


class LocalArtifactStore(ArtifactStore):
    """
    Filesystem-based artifact store.

    Stores blobs as files under <base_dir>/<container>/<blob_name>.
    """

    def __init__(self, base_dir: str = "data_cache/artifacts"):
        self._base_dir = Path(base_dir)
        self._lock = threading.Lock()
        self._base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("artifact_store.local.initialized", base_dir=str(self._base_dir))

    def _blob_path(self, container: str, blob_name: str) -> Path:
        path = self._base_dir / container / blob_name
        return path

    def upload(self, container: str, blob_name: str, data: bytes) -> None:
        with self._lock:
            path = self._blob_path(container, blob_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            logger.debug(
                "artifact_store.local.uploaded",
                container=container,
                blob_name=blob_name,
                size=len(data),
            )

    def download(self, container: str, blob_name: str) -> bytes:
        path = self._blob_path(container, blob_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Local artifact not found: {container}/{blob_name}"
            )
        return path.read_bytes()

    def exists(self, container: str, blob_name: str) -> bool:
        return self._blob_path(container, blob_name).exists()

    def list_blobs(self, container: str, prefix: str = "") -> list[str]:
        container_dir = self._base_dir / container
        if not container_dir.exists():
            return []
        results = []
        for p in container_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(container_dir).as_posix()
                if rel.startswith(prefix):
                    results.append(rel)
        return sorted(results)

    def delete(self, container: str, blob_name: str) -> None:
        with self._lock:
            path = self._blob_path(container, blob_name)
            if path.exists():
                path.unlink()
                logger.debug(
                    "artifact_store.local.deleted",
                    container=container,
                    blob_name=blob_name,
                )


class AzureBlobArtifactStore(ArtifactStore):
    """
    Azure Blob Storage backend.

    Uses DefaultAzureCredential for Managed Identity auth (works in Azure
    and locally with az cli). Falls back to connection string if provided.
    """

    def __init__(self, account_url: str, container_prefix: str = "suez-trader"):
        self._account_url = account_url
        self._container_prefix = container_prefix
        self._lock = threading.Lock()
        self._client = self._create_client(account_url)
        logger.info(
            "artifact_store.azure.initialized",
            account_url=account_url[:40] + "...",
            container_prefix=container_prefix,
        )

    def _create_client(self, account_url: str):
        """Create BlobServiceClient with DefaultAzureCredential or connection string."""
        from azure.storage.blob import BlobServiceClient

        if account_url.startswith("DefaultEndpointsProtocol=") or "AccountKey=" in account_url:
            # Connection string
            return BlobServiceClient.from_connection_string(account_url)
        else:
            from azure.identity import DefaultAzureCredential
            credential = DefaultAzureCredential()
            return BlobServiceClient(account_url=account_url, credential=credential)

    def _container_name(self, container: str) -> str:
        """Build the full container name with prefix. Azure rules: lowercase, 3-63 chars."""
        name = f"{self._container_prefix}-{container}".lower()
        # Sanitize: only letters, numbers, hyphens
        sanitized = "".join(c if c.isalnum() or c == "-" else "-" for c in name)
        return sanitized[:63]

    def _ensure_container(self, container_name: str) -> None:
        """Create container if it doesn't exist."""
        try:
            container_client = self._client.get_container_client(container_name)
            if not container_client.exists():
                self._client.create_container(container_name)
                logger.info("artifact_store.azure.container_created", container=container_name)
        except Exception as exc:
            logger.warning(
                "artifact_store.azure.ensure_container_failed",
                container=container_name,
                error=str(exc),
            )

    def upload(self, container: str, blob_name: str, data: bytes) -> None:
        with self._lock:
            cname = self._container_name(container)
            self._ensure_container(cname)
            blob_client = self._client.get_blob_client(container=cname, blob=blob_name)
            blob_client.upload_blob(data, overwrite=True)
            logger.debug(
                "artifact_store.azure.uploaded",
                container=cname,
                blob_name=blob_name,
                size=len(data),
            )

    def download(self, container: str, blob_name: str) -> bytes:
        cname = self._container_name(container)
        blob_client = self._client.get_blob_client(container=cname, blob=blob_name)
        downloader = blob_client.download_blob()
        return downloader.readall()

    def exists(self, container: str, blob_name: str) -> bool:
        try:
            cname = self._container_name(container)
            blob_client = self._client.get_blob_client(container=cname, blob=blob_name)
            blob_client.get_blob_properties()
            return True
        except Exception:
            return False

    def list_blobs(self, container: str, prefix: str = "") -> list[str]:
        try:
            cname = self._container_name(container)
            container_client = self._client.get_container_client(cname)
            blobs = container_client.list_blobs(name_starts_with=prefix or None)
            return sorted([b.name for b in blobs])
        except Exception:
            return []

    def delete(self, container: str, blob_name: str) -> None:
        with self._lock:
            cname = self._container_name(container)
            blob_client = self._client.get_blob_client(container=cname, blob=blob_name)
            blob_client.delete_blob(delete_snapshots="include")
            logger.debug(
                "artifact_store.azure.deleted",
                container=cname,
                blob_name=blob_name,
            )


def create_artifact_store(
    blob_url: str = "",
    container_prefix: str = "suez-trader",
    local_base_dir: str = "data_cache/artifacts",
) -> ArtifactStore:
    """
    Factory function — returns Azure backend if blob_url is provided and reachable,
    otherwise falls back to local filesystem.

    Args:
        blob_url: Azure Blob Storage account URL or connection string.
        container_prefix: Prefix for Azure container names.
        local_base_dir: Base directory for local artifact storage.

    Returns:
        An ArtifactStore instance (Azure or Local).
    """
    if blob_url:
        try:
            store = AzureBlobArtifactStore(
                account_url=blob_url,
                container_prefix=container_prefix,
            )
            logger.info("artifact_store.factory.azure_active")
            return store
        except Exception as exc:
            logger.warning(
                "artifact_store.factory.azure_unavailable",
                error=str(exc),
                msg="Falling back to local filesystem",
            )

    store = LocalArtifactStore(base_dir=local_base_dir)
    logger.info("artifact_store.factory.local_active")
    return store

"""
Migrate local ML artifacts to Azure Blob Storage.

Scans local directories for model, dataset, and audit artifacts and uploads
them to the configured Azure Blob Storage account. Idempotent — skips blobs
that already exist in the target container.

Usage:
    python scripts/migrate_artifacts_to_blob.py
    python scripts/migrate_artifacts_to_blob.py --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.blob_storage import create_artifact_store
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def scan_and_upload(
    artifact_store,
    local_dir: str,
    container: str,
    extensions: list[str],
    dry_run: bool = False,
) -> dict:
    """
    Scan a local directory for files matching extensions and upload to blob.

    Returns:
        Dict with keys: uploaded, skipped, failed, total.
    """
    stats = {"uploaded": 0, "skipped": 0, "failed": 0, "total": 0}
    local_path = Path(local_dir)

    if not local_path.exists():
        logger.info(
            "migrate.directory_not_found",
            directory=local_dir,
            msg="Skipping — directory does not exist",
        )
        return stats

    files = []
    for ext in extensions:
        files.extend(local_path.rglob(f"*{ext}"))

    stats["total"] = len(files)

    for filepath in sorted(files):
        blob_name = filepath.relative_to(local_path).as_posix()

        if artifact_store.exists(container, blob_name):
            stats["skipped"] += 1
            logger.debug("migrate.skipped", container=container, blob_name=blob_name)
            continue

        if dry_run:
            stats["uploaded"] += 1
            print(f"  [DRY-RUN] Would upload: {filepath} -> {container}/{blob_name}")
            continue

        try:
            data = filepath.read_bytes()
            artifact_store.upload(container, blob_name, data)
            stats["uploaded"] += 1
            logger.info(
                "migrate.uploaded",
                container=container,
                blob_name=blob_name,
                size=len(data),
            )
        except Exception as exc:
            stats["failed"] += 1
            logger.error(
                "migrate.upload_failed",
                container=container,
                blob_name=blob_name,
                error=str(exc),
            )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate local ML artifacts to Azure Blob Storage"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be uploaded without making changes",
    )
    parser.add_argument(
        "--blob-url",
        default=os.environ.get("BLOB_STORAGE_URL", ""),
        help="Azure Blob Storage account URL (default: $BLOB_STORAGE_URL)",
    )
    args = parser.parse_args()

    if not args.blob_url:
        print("ERROR: No blob URL provided. Set BLOB_STORAGE_URL env var or use --blob-url")
        sys.exit(1)

    print(f"{'[DRY-RUN] ' if args.dry_run else ''}Migrating artifacts to Azure Blob Storage...")
    print(f"  Blob URL: {args.blob_url[:40]}...")
    print()

    artifact_store = create_artifact_store(blob_url=args.blob_url)

    # Migration targets
    migrations = [
        {
            "local_dir": "models",
            "container": "models",
            "extensions": [".joblib"],
            "label": "ML Models",
        },
        {
            "local_dir": "data_cache/datasets/snapshots",
            "container": "datasets",
            "extensions": [".parquet", ".csv"],
            "label": "Dataset Snapshots",
        },
        {
            "local_dir": "data_cache/audit",
            "container": "audit-logs",
            "extensions": [".jsonl"],
            "label": "Audit Logs",
        },
    ]

    total_stats = {"uploaded": 0, "skipped": 0, "failed": 0, "total": 0}

    for migration in migrations:
        print(f"── {migration['label']} ({migration['local_dir']}) ──")
        stats = scan_and_upload(
            artifact_store=artifact_store,
            local_dir=migration["local_dir"],
            container=migration["container"],
            extensions=migration["extensions"],
            dry_run=args.dry_run,
        )
        print(
            f"  Total: {stats['total']} | "
            f"Uploaded: {stats['uploaded']} | "
            f"Skipped: {stats['skipped']} | "
            f"Failed: {stats['failed']}"
        )
        print()

        for k in total_stats:
            total_stats[k] += stats[k]

    print("═══ Summary ═══")
    print(
        f"  Total files: {total_stats['total']} | "
        f"Uploaded: {total_stats['uploaded']} | "
        f"Skipped (already exists): {total_stats['skipped']} | "
        f"Failed: {total_stats['failed']}"
    )

    if total_stats["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

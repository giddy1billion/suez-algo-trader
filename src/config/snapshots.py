"""
Configuration Snapshots — Export and restore complete configuration sets.

Provides disaster recovery, environment cloning, easy rollback,
and change review capabilities.
"""

import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from src.config.repository import ConfigurationRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConfigSnapshot:
    """
    A complete snapshot of all configuration values at a point in time.

    Includes metadata for identification, verification, and audit trail.
    """

    version: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    created_by: str = "system"
    description: str = ""
    entries: list[dict[str, Any]] = field(default_factory=list)
    checksum: str = ""

    def compute_checksum(self) -> str:
        """Compute SHA-256 checksum of configuration entries."""
        content = json.dumps(self.entries, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize snapshot to dictionary."""
        data = asdict(self)
        data["checksum"] = self.compute_checksum()
        return data

    def to_json(self, indent: int = 2) -> str:
        """Serialize snapshot to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "ConfigSnapshot":
        """Deserialize snapshot from JSON string."""
        data = json.loads(json_str)
        entries = data.pop("entries", [])
        snapshot = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        snapshot.entries = entries
        return snapshot

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfigSnapshot":
        """Deserialize snapshot from dictionary."""
        entries = data.pop("entries", [])
        snapshot = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        snapshot.entries = entries
        return snapshot

    def verify_integrity(self) -> bool:
        """Verify the snapshot checksum matches its entries."""
        if not self.checksum:
            return True
        return self.compute_checksum() == self.checksum


class SnapshotManager:
    """
    Manages configuration snapshots for backup and restore operations.

    Features:
    - Export current configuration to a portable snapshot
    - Import/restore configuration from a snapshot
    - Verify snapshot integrity before restore
    - Selective restore (specific categories only)
    """

    def __init__(self, repository: ConfigurationRepository):
        self._repo = repository

    def export_snapshot(
        self,
        version: Optional[str] = None,
        description: str = "",
        created_by: str = "system",
        categories: Optional[list[str]] = None,
    ) -> ConfigSnapshot:
        """
        Export current configuration as a snapshot.

        Args:
            version: Snapshot version identifier. Auto-generated if not provided.
            description: Human-readable description of this snapshot.
            created_by: User who created the snapshot.
            categories: Optional list of categories to include. None = all.

        Returns:
            ConfigSnapshot with all matching configuration entries.
        """
        entries = self._repo.get_all()

        if categories:
            entries = [e for e in entries if e.category in categories]

        snapshot_entries = []
        for entry in entries:
            snapshot_entries.append({
                "category": entry.category,
                "key": entry.key,
                "value": entry.value,
                "value_type": entry.value_type,
                "description": entry.description or "",
                "is_secret": entry.is_secret,
                "is_editable": entry.is_editable,
                "validation_rule": entry.validation_rule or "",
                "version": entry.version,
            })

        if version is None:
            version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        snapshot = ConfigSnapshot(
            version=version,
            description=description,
            created_by=created_by,
            entries=snapshot_entries,
        )
        snapshot.checksum = snapshot.compute_checksum()

        logger.info(
            "snapshot.exported",
            version=version,
            entries=len(snapshot_entries),
            checksum=snapshot.checksum[:12],
        )
        return snapshot

    def restore_snapshot(
        self,
        snapshot: ConfigSnapshot,
        restored_by: str = "system",
        categories: Optional[list[str]] = None,
        dry_run: bool = False,
        skip_secrets: bool = True,
    ) -> dict[str, Any]:
        """
        Restore configuration from a snapshot.

        Args:
            snapshot: The snapshot to restore.
            restored_by: User performing the restore.
            categories: Optional list of categories to restore. None = all.
            dry_run: If True, only report what would change without applying.
            skip_secrets: If True, skip entries marked as secrets.

        Returns:
            Summary dict with counts and details of changes.
        """
        # Verify integrity
        if not snapshot.verify_integrity():
            logger.error("snapshot.integrity_check_failed", version=snapshot.version)
            return {"success": False, "error": "Integrity check failed"}

        entries_to_restore = snapshot.entries
        if categories:
            entries_to_restore = [
                e for e in entries_to_restore if e["category"] in categories
            ]
        if skip_secrets:
            entries_to_restore = [
                e for e in entries_to_restore if not e.get("is_secret", False)
            ]

        changes = {"created": 0, "updated": 0, "skipped": 0, "details": []}

        for entry in entries_to_restore:
            existing = self._repo.get(entry["category"], entry["key"])

            if existing and existing.value == entry["value"]:
                changes["skipped"] += 1
                continue

            action = "update" if existing else "create"
            changes["details"].append({
                "category": entry["category"],
                "key": entry["key"],
                "action": action,
                "old_value": existing.value if existing else None,
                "new_value": entry["value"],
            })

            if not dry_run:
                self._repo.set(
                    category=entry["category"],
                    key=entry["key"],
                    value=entry["value"],
                    value_type=entry.get("value_type", "str"),
                    changed_by=restored_by,
                    change_reason=f"Restored from snapshot {snapshot.version}",
                    description=entry.get("description", ""),
                    is_secret=entry.get("is_secret", False),
                    is_editable=entry.get("is_editable", True),
                    validation_rule=entry.get("validation_rule", ""),
                )

            if action == "create":
                changes["created"] += 1
            else:
                changes["updated"] += 1

        changes["success"] = True
        changes["dry_run"] = dry_run
        changes["total"] = len(entries_to_restore)

        logger.info(
            "snapshot.restored",
            version=snapshot.version,
            created=changes["created"],
            updated=changes["updated"],
            skipped=changes["skipped"],
            dry_run=dry_run,
        )
        return changes

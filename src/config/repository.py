"""
Configuration Repository — Database CRUD operations for configuration.
"""

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session, sessionmaker

from src.config.models import ConfigBase, SystemConfiguration, ConfigurationAuditLog


class ConfigurationRepository:
    """
    Data access layer for the system configuration table.

    Handles all database interactions: reads, writes, versioning, and audit logging.
    """

    def __init__(self, database_url: str = "sqlite:///data_cache/trading.db"):
        from src.utils.database import create_db_engine

        self._engine = create_db_engine(database_url)
        ConfigBase.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)

    def _get_session(self) -> Session:
        return self._session_factory()

    def get_all(self) -> list[SystemConfiguration]:
        """Load all configuration entries."""
        with self._get_session() as session:
            return session.query(SystemConfiguration).all()

    def get_by_category(self, category: str) -> list[SystemConfiguration]:
        """Load all entries for a given category."""
        with self._get_session() as session:
            return (
                session.query(SystemConfiguration)
                .filter(SystemConfiguration.category == category)
                .all()
            )

    def get(self, category: str, key: str) -> Optional[SystemConfiguration]:
        """Get a single configuration entry."""
        with self._get_session() as session:
            return (
                session.query(SystemConfiguration)
                .filter(
                    SystemConfiguration.category == category,
                    SystemConfiguration.key == key,
                )
                .first()
            )

    def set(
        self,
        category: str,
        key: str,
        value: str,
        value_type: str = "str",
        changed_by: str = "system",
        change_reason: str = "",
        description: str = "",
        is_secret: bool = False,
        is_editable: bool = True,
        validation_rule: str = "",
    ) -> SystemConfiguration:
        """
        Create or update a configuration entry with audit logging.

        Returns the updated/created configuration entry.
        """
        with self._get_session() as session:
            existing = (
                session.query(SystemConfiguration)
                .filter(
                    SystemConfiguration.category == category,
                    SystemConfiguration.key == key,
                )
                .first()
            )

            if existing:
                old_value = existing.value
                old_version = existing.version

                existing.value = value
                existing.value_type = value_type
                existing.version = old_version + 1
                existing.updated_by = changed_by
                existing.updated_at = datetime.now(timezone.utc)

                if description:
                    existing.description = description
                if validation_rule:
                    existing.validation_rule = validation_rule

                # Audit log
                audit = ConfigurationAuditLog(
                    category=category,
                    key=key,
                    old_value=old_value,
                    new_value=value,
                    old_version=old_version,
                    new_version=existing.version,
                    changed_by=changed_by,
                    change_reason=change_reason,
                )
                session.add(audit)
                session.commit()
                session.refresh(existing)
                return existing
            else:
                new_config = SystemConfiguration(
                    category=category,
                    key=key,
                    value=value,
                    value_type=value_type,
                    description=description,
                    is_secret=is_secret,
                    is_editable=is_editable,
                    validation_rule=validation_rule,
                    updated_by=changed_by,
                    version=1,
                )
                session.add(new_config)

                # Audit log for creation
                audit = ConfigurationAuditLog(
                    category=category,
                    key=key,
                    old_value=None,
                    new_value=value,
                    old_version=None,
                    new_version=1,
                    changed_by=changed_by,
                    change_reason=change_reason or "initial_creation",
                )
                session.add(audit)
                session.commit()
                session.refresh(new_config)
                return new_config

    def delete(self, category: str, key: str, changed_by: str = "system") -> bool:
        """Delete a configuration entry. Returns True if deleted."""
        with self._get_session() as session:
            existing = (
                session.query(SystemConfiguration)
                .filter(
                    SystemConfiguration.category == category,
                    SystemConfiguration.key == key,
                )
                .first()
            )
            if not existing:
                return False

            # Audit log
            audit = ConfigurationAuditLog(
                category=category,
                key=key,
                old_value=existing.value,
                new_value="<DELETED>",
                old_version=existing.version,
                new_version=existing.version + 1,
                changed_by=changed_by,
                change_reason="deleted",
            )
            session.add(audit)
            session.delete(existing)
            session.commit()
            return True

    def get_audit_log(
        self,
        category: Optional[str] = None,
        key: Optional[str] = None,
        limit: int = 50,
    ) -> list[ConfigurationAuditLog]:
        """Get audit log entries with optional filtering."""
        with self._get_session() as session:
            query = session.query(ConfigurationAuditLog)
            if category:
                query = query.filter(ConfigurationAuditLog.category == category)
            if key:
                query = query.filter(ConfigurationAuditLog.key == key)
            return (
                query.order_by(ConfigurationAuditLog.changed_at.desc())
                .limit(limit)
                .all()
            )

    def bulk_set(
        self,
        entries: list[dict],
        changed_by: str = "system",
        change_reason: str = "",
    ) -> int:
        """
        Bulk upsert configuration entries.

        Each entry dict should contain: category, key, value, value_type,
        and optionally: description, is_secret, is_editable, validation_rule.

        Returns count of entries processed.
        """
        count = 0
        for entry in entries:
            self.set(
                category=entry["category"],
                key=entry["key"],
                value=str(entry["value"]),
                value_type=entry.get("value_type", "str"),
                changed_by=changed_by,
                change_reason=change_reason,
                description=entry.get("description", ""),
                is_secret=entry.get("is_secret", False),
                is_editable=entry.get("is_editable", True),
                validation_rule=entry.get("validation_rule", ""),
            )
            count += 1
        return count

    def get_version_history(
        self, category: str, key: str, limit: int = 20
    ) -> list[ConfigurationAuditLog]:
        """Get version history for a specific configuration key."""
        return self.get_audit_log(category=category, key=key, limit=limit)

    def close(self) -> None:
        """Dispose the engine connection pool."""
        self._engine.dispose()

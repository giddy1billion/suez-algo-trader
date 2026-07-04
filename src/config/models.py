"""
Configuration Database Models — SQLAlchemy models for runtime configuration.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Boolean,
    Text,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase


class ConfigBase(DeclarativeBase):
    """Separate base for configuration tables."""
    pass


class SystemConfiguration(ConfigBase):
    """
    Stores runtime configuration key-value pairs organized by category.

    This is the authoritative source for all runtime settings after startup.
    """
    __tablename__ = "system_configuration"
    __table_args__ = (
        UniqueConstraint("category", "key", name="uq_category_key"),
        Index("ix_system_config_category", "category"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(50), nullable=False)
    key = Column(String(100), nullable=False)
    value = Column(Text, nullable=False)
    value_type = Column(String(20), nullable=False, default="str")  # str, int, float, bool, json
    description = Column(Text, default="")
    is_secret = Column(Boolean, default=False)
    is_editable = Column(Boolean, default=True)
    validation_rule = Column(String(200), default="")
    updated_by = Column(String(100), default="system")
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class ConfigurationAuditLog(ConfigBase):
    """
    Audit trail for all configuration changes.

    Captures who changed what, when, with old and new values.
    """
    __tablename__ = "configuration_audit_log"
    __table_args__ = (
        Index("ix_config_audit_category_key", "category", "key"),
        Index("ix_config_audit_timestamp", "changed_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(50), nullable=False)
    key = Column(String(100), nullable=False)
    old_value = Column(Text)
    new_value = Column(Text, nullable=False)
    old_version = Column(Integer)
    new_version = Column(Integer, nullable=False)
    changed_by = Column(String(100), nullable=False, default="system")
    change_reason = Column(Text, default="")
    changed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

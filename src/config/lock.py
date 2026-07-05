"""
Emergency Configuration Lock — Maintenance mode for configuration.

Provides the ability to lock all configuration changes during incidents
or market volatility, allowing only super admins to make changes while
trading continues uninterrupted.
"""

import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LockState:
    """Current state of the configuration lock."""

    is_locked: bool = False
    locked_by: str = ""
    locked_at: Optional[str] = None
    reason: str = ""
    super_admins: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_locked": self.is_locked,
            "locked_by": self.locked_by,
            "locked_at": self.locked_at,
            "reason": self.reason,
            "super_admins": self.super_admins,
        }


class ConfigurationLock:
    """
    Emergency lock for configuration changes.

    When locked:
    - Only super admins can modify configuration
    - Trading continues normally
    - All other configuration changes are rejected
    - Useful during incidents or live market volatility

    Usage:
        lock = ConfigurationLock(super_admins=["admin@example.com"])
        lock.engage(locked_by="admin@example.com", reason="Market volatility")

        # Check before allowing changes
        if not lock.can_modify(user="trader@example.com"):
            raise PermissionError("Configuration is locked")
    """

    def __init__(self, super_admins: Optional[list[str]] = None):
        self._state = LockState(
            super_admins=super_admins or ["system"],
        )
        self._lock = threading.Lock()

    @property
    def is_locked(self) -> bool:
        """Check if configuration is currently locked."""
        return self._state.is_locked

    @property
    def state(self) -> LockState:
        """Get current lock state."""
        return self._state

    def engage(self, locked_by: str, reason: str = "") -> bool:
        """
        Engage the configuration lock.

        Only super admins can engage the lock.

        Args:
            locked_by: User engaging the lock (must be a super admin).
            reason: Reason for locking configuration.

        Returns:
            True if lock was successfully engaged.
        """
        with self._lock:
            if not self._is_super_admin(locked_by):
                logger.warning(
                    "config_lock.unauthorized_engage",
                    user=locked_by,
                )
                return False

            self._state.is_locked = True
            self._state.locked_by = locked_by
            self._state.locked_at = datetime.now(timezone.utc).isoformat()
            self._state.reason = reason

            logger.warning(
                "config_lock.engaged",
                locked_by=locked_by,
                reason=reason,
            )
            return True

    def release(self, released_by: str) -> bool:
        """
        Release the configuration lock.

        Only super admins can release the lock.

        Args:
            released_by: User releasing the lock (must be a super admin).

        Returns:
            True if lock was successfully released.
        """
        with self._lock:
            if not self._is_super_admin(released_by):
                logger.warning(
                    "config_lock.unauthorized_release",
                    user=released_by,
                )
                return False

            if not self._state.is_locked:
                return True  # Already unlocked

            self._state.is_locked = False
            logger.info(
                "config_lock.released",
                released_by=released_by,
                was_locked_by=self._state.locked_by,
            )
            self._state.locked_by = ""
            self._state.locked_at = None
            self._state.reason = ""
            return True

    def can_modify(self, user: str) -> bool:
        """
        Check if a user is allowed to modify configuration.

        Returns True if:
        - Configuration is not locked, OR
        - User is a super admin
        """
        if not self._state.is_locked:
            return True
        return self._is_super_admin(user)

    def add_super_admin(self, admin: str, added_by: str) -> bool:
        """Add a new super admin (must be done by existing super admin)."""
        with self._lock:
            if not self._is_super_admin(added_by):
                return False
            if admin not in self._state.super_admins:
                self._state.super_admins.append(admin)
                logger.info(
                    "config_lock.super_admin_added",
                    admin=admin,
                    added_by=added_by,
                )
            return True

    def remove_super_admin(self, admin: str, removed_by: str) -> bool:
        """Remove a super admin (cannot remove self if last admin)."""
        with self._lock:
            if not self._is_super_admin(removed_by):
                return False
            if len(self._state.super_admins) <= 1:
                logger.warning("config_lock.cannot_remove_last_admin")
                return False
            if admin in self._state.super_admins:
                self._state.super_admins.remove(admin)
                logger.info(
                    "config_lock.super_admin_removed",
                    admin=admin,
                    removed_by=removed_by,
                )
            return True

    def _is_super_admin(self, user: str) -> bool:
        """Check if a user is a super admin."""
        return user in self._state.super_admins

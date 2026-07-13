"""
Training Lock — Distributed singleton lock to prevent concurrent training jobs.

Uses the CacheBackend (Redis or LocalCache) to ensure exactly one training
job runs at a time across all bot/scheduler instances. Logs instance identity
(hostname + PID) when the lock is acquired.

Root cause of duplicate ModelTrainingStarted events:
- Multiple schedulers (APScheduler interval trigger + calibration check +
  manual Telegram trigger) can race to invoke TrainingPipeline.train().
- In multi-container deployments, multiple replicas may each fire their
  own scheduled training. The in-process threading.Lock only guards a
  single process.

This module provides a cross-process/cross-container lock via the shared
cache backend (Redis in production, LocalCache for single-instance dev).
"""

import os
import socket
import time
import threading
from contextlib import contextmanager
from typing import Optional

from src.utils.logger import get_logger
from src.utils.redis_client import CacheBackend

logger = get_logger(__name__)

# Lock key in the cache
_LOCK_KEY = "training:lock"
# Default lock TTL — prevents deadlocks if holder crashes (seconds)
_DEFAULT_LOCK_TTL = 3600  # 1 hour max training time
# Heartbeat interval for lock renewal
_HEARTBEAT_INTERVAL = 60


def _instance_identity() -> str:
    """Return a unique identity string for this process instance."""
    hostname = socket.gethostname()
    pid = os.getpid()
    return f"{hostname}:{pid}"


class TrainingLockError(RuntimeError):
    """Raised when training lock cannot be acquired."""
    pass


class TrainingLock:
    """
    Distributed singleton lock for training jobs.

    Ensures exactly one training pipeline runs at a time across all
    instances. Uses CacheBackend for cross-process coordination.

    Usage:
        lock = TrainingLock(cache_backend)
        with lock.acquire(pipeline_id="abc123"):
            # ... run training ...
    """

    def __init__(self, cache: CacheBackend, lock_ttl: int = _DEFAULT_LOCK_TTL):
        self._cache = cache
        self._lock_ttl = lock_ttl
        self._identity = _instance_identity()
        self._local_lock = threading.Lock()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop = threading.Event()

    @property
    def instance_identity(self) -> str:
        return self._identity

    def is_locked(self) -> bool:
        """Check if training lock is currently held by any instance."""
        return self._cache.exists(_LOCK_KEY)

    def lock_holder(self) -> Optional[str]:
        """Return the identity of the current lock holder, or None."""
        raw = self._cache.get(_LOCK_KEY)
        if raw:
            # Format: "identity|pipeline_id|acquired_at"
            parts = raw.split("|", 2)
            return parts[0] if parts else raw
        return None

    def try_acquire(self, pipeline_id: str) -> bool:
        """
        Attempt to acquire the training lock (non-blocking).

        Returns True if lock was acquired, False if already held.
        Logs the instance identity on success.
        """
        with self._local_lock:
            # Check if already locked
            existing = self._cache.get(_LOCK_KEY)
            if existing:
                holder = existing.split("|", 2)[0]
                logger.warning(
                    "training_lock.already_held",
                    holder=holder,
                    requester=self._identity,
                    pipeline_id=pipeline_id,
                )
                return False

            # Acquire: write identity + pipeline_id + timestamp
            lock_value = f"{self._identity}|{pipeline_id}|{time.time():.0f}"
            self._cache.set(_LOCK_KEY, lock_value, ttl=self._lock_ttl)

            logger.info(
                "training_lock.acquired",
                instance=self._identity,
                pipeline_id=pipeline_id,
                ttl_seconds=self._lock_ttl,
            )

            # Start heartbeat to renew TTL while training is active
            self._heartbeat_stop.clear()
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
                name=f"training-lock-heartbeat-{pipeline_id}",
            )
            self._heartbeat_thread.start()
            return True

    def release(self, pipeline_id: str) -> None:
        """Release the training lock. Only the holder should call this."""
        with self._local_lock:
            # Stop heartbeat
            self._heartbeat_stop.set()
            if self._heartbeat_thread:
                self._heartbeat_thread.join(timeout=5)
                self._heartbeat_thread = None

            # Verify we are the holder before releasing
            existing = self._cache.get(_LOCK_KEY)
            if existing:
                parts = existing.split("|", 2)
                if parts[0] == self._identity:
                    self._cache.delete(_LOCK_KEY)
                    logger.info(
                        "training_lock.released",
                        instance=self._identity,
                        pipeline_id=pipeline_id,
                    )
                else:
                    logger.warning(
                        "training_lock.release_denied",
                        holder=parts[0],
                        requester=self._identity,
                        msg="Cannot release lock held by another instance",
                    )
            else:
                logger.debug(
                    "training_lock.release_noop",
                    pipeline_id=pipeline_id,
                    msg="Lock already expired or released",
                )

    @contextmanager
    def acquire(self, pipeline_id: str):
        """
        Context manager that acquires/releases the training lock.

        Raises TrainingLockError if lock is already held.
        """
        if not self.try_acquire(pipeline_id):
            holder = self.lock_holder() or "unknown"
            raise TrainingLockError(
                f"Training lock held by {holder}. "
                f"Cannot start pipeline {pipeline_id} from {self._identity}."
            )
        try:
            yield self
        finally:
            self.release(pipeline_id)

    def _heartbeat_loop(self) -> None:
        """Periodically renew the lock TTL to prevent expiry during long training."""
        while not self._heartbeat_stop.wait(timeout=_HEARTBEAT_INTERVAL):
            try:
                if self._cache.exists(_LOCK_KEY):
                    self._cache.expire(_LOCK_KEY, self._lock_ttl)
            except Exception as e:
                logger.warning("training_lock.heartbeat_error", error=str(e))

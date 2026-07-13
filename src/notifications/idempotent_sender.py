"""
Idempotent Telegram Notification Sender — Ensures at-most-once delivery semantics.

Wraps Telegram message sending with deduplication keys so that retries
(e.g., from network timeouts) do not produce duplicate messages to users.

Uses the CorrelationStore's dedup mechanism to track sent notification IDs.
"""

import hashlib
import time
import threading
from typing import Any, Callable, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class IdempotentNotifier:
    """
    Wraps a notification send function with idempotency guarantees.

    Each notification is identified by a dedup_key (derived from content hash
    or explicitly provided). If the same key is seen within the TTL window,
    the notification is suppressed.

    Thread-safe. Supports configurable TTL and retry tracking.
    """

    def __init__(
        self,
        send_fn: Callable[[str, str], Any],
        dedup_ttl: float = 3600.0,
        max_retries: int = 3,
    ):
        """
        Args:
            send_fn: Callable(chat_id, message) -> Any. The actual send function.
            dedup_ttl: How long (seconds) to remember sent notification keys.
            max_retries: Maximum retry attempts for transient failures.
        """
        self._send_fn = send_fn
        self._dedup_ttl = dedup_ttl
        self._max_retries = max_retries
        self._lock = threading.Lock()
        # dedup_key -> (timestamp_monotonic, attempt_count)
        self._sent_keys: dict[str, tuple[float, int]] = {}
        # Metrics
        self._total_sent: int = 0
        self._total_suppressed: int = 0
        self._total_failed: int = 0
        self._total_retries: int = 0

    @property
    def metrics(self) -> dict[str, int]:
        """Return notification delivery metrics."""
        return {
            "total_sent": self._total_sent,
            "total_suppressed": self._total_suppressed,
            "total_failed": self._total_failed,
            "total_retries": self._total_retries,
        }

    def send(
        self,
        chat_id: str,
        message: str,
        dedup_key: Optional[str] = None,
    ) -> bool:
        """
        Send a notification idempotently.

        Args:
            chat_id: Telegram chat ID.
            message: Message text.
            dedup_key: Optional explicit dedup key. If not provided,
                       a hash of (chat_id, message) is used.

        Returns:
            True if the message was sent (or was already sent).
            False if delivery failed after retries.
        """
        if dedup_key is None:
            dedup_key = self._compute_key(chat_id, message)

        # Atomically check-and-reserve the dedup key to prevent races
        now = time.monotonic()
        with self._lock:
            self._cleanup_expired(now)
            if dedup_key in self._sent_keys:
                self._total_suppressed += 1
                logger.debug(
                    "notification.suppressed_duplicate",
                    dedup_key=dedup_key,
                )
                return True  # Already sent — idempotent success
            # Reserve the key immediately (optimistic mark)
            self._sent_keys[dedup_key] = (now, 0)

        # Attempt delivery with retries
        last_error: Optional[Exception] = None
        retries_used = 0
        for attempt in range(1, self._max_retries + 1):
            try:
                self._send_fn(chat_id, message)
                # Confirm delivery
                with self._lock:
                    self._sent_keys[dedup_key] = (now, attempt)
                    self._total_sent += 1
                    self._total_retries += retries_used
                return True
            except Exception as e:
                last_error = e
                retries_used += 1
                if attempt < self._max_retries:
                    logger.warning(
                        "notification.retry",
                        attempt=attempt,
                        max_retries=self._max_retries,
                        error=str(e),
                    )
                    time.sleep(min(0.1 * (2 ** attempt), 2.0))

        # All retries exhausted — release the reserved key so future retries can attempt
        with self._lock:
            self._sent_keys.pop(dedup_key, None)
            self._total_failed += 1
        logger.error(
            "notification.failed_after_retries",
            dedup_key=dedup_key,
            error=str(last_error),
        )
        return False

    def is_duplicate(self, chat_id: str, message: str, dedup_key: Optional[str] = None) -> bool:
        """Check if a notification would be suppressed as duplicate."""
        if dedup_key is None:
            dedup_key = self._compute_key(chat_id, message)
        now = time.monotonic()
        with self._lock:
            self._cleanup_expired(now)
            return dedup_key in self._sent_keys

    def reset(self) -> None:
        """Clear all dedup state (for testing)."""
        with self._lock:
            self._sent_keys.clear()

    def _compute_key(self, chat_id: str, message: str) -> str:
        """Compute a dedup key from chat_id and message content."""
        content = f"{chat_id}:{message}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _cleanup_expired(self, now: float) -> None:
        """Remove expired dedup entries (called with lock held)."""
        expired = [
            key
            for key, (ts, _) in self._sent_keys.items()
            if now - ts > self._dedup_ttl
        ]
        for key in expired:
            del self._sent_keys[key]

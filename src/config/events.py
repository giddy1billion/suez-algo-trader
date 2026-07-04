"""
Configuration Change Events — Structured event model for configuration changes.

Emits structured events whenever configuration values change, enabling
integrations with logging, monitoring, notifications, and analytics.
Also provides distributed cache invalidation via pluggable event publishers.
"""

import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Structured Event Model ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ConfigurationChangedEvent:
    """
    Structured event emitted when a configuration value changes.

    Provides full context for logging, monitoring, notifications, and analytics.
    """

    category: str
    key: str
    old_value: Optional[str]
    new_value: str
    updated_by: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: int = 1
    change_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize event to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfigurationChangedEvent":
        """Deserialize event from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ─── Event Publisher Interface ───────────────────────────────────────────────


class EventPublisher(ABC):
    """
    Abstract base class for configuration event publishers.

    Implementations can publish to Redis Pub/Sub, RabbitMQ, Kafka, or
    any other messaging system to enable distributed cache invalidation.
    """

    @abstractmethod
    def publish(self, event: ConfigurationChangedEvent) -> bool:
        """
        Publish a configuration change event.

        Returns True if successfully published.
        """
        ...

    @abstractmethod
    def subscribe(self, callback: Callable[[ConfigurationChangedEvent], None]) -> None:
        """Subscribe to configuration change events."""
        ...

    def close(self) -> None:
        """Clean up resources."""
        pass


class InProcessEventPublisher(EventPublisher):
    """
    In-process event publisher for single-instance deployments.

    Immediately notifies all subscribers within the same process.
    Useful for testing and single-node deployments.
    """

    def __init__(self):
        self._subscribers: list[Callable[[ConfigurationChangedEvent], None]] = []
        self._lock = threading.Lock()

    def publish(self, event: ConfigurationChangedEvent) -> bool:
        """Publish event to all in-process subscribers."""
        with self._lock:
            for callback in self._subscribers:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(
                        "event_publisher.callback_error",
                        error=str(e),
                        event_key=f"{event.category}.{event.key}",
                    )
        return True

    def subscribe(self, callback: Callable[[ConfigurationChangedEvent], None]) -> None:
        """Register a callback for configuration change events."""
        with self._lock:
            self._subscribers.append(callback)


class RedisEventPublisher(EventPublisher):
    """
    Redis Pub/Sub event publisher for distributed cache invalidation.

    Publishes configuration change events to a Redis channel so all
    running instances can immediately refresh their caches.

    Requires: redis package (pip install redis)
    """

    CHANNEL = "config:changes"

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._redis_url = redis_url
        self._redis = None
        self._pubsub = None
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False
        self._callbacks: list[Callable[[ConfigurationChangedEvent], None]] = []

    def _get_redis(self):
        """Lazy-initialize Redis connection."""
        if self._redis is None:
            try:
                import redis

                self._redis = redis.from_url(self._redis_url)
                self._redis.ping()
                logger.info("redis_event_publisher.connected", url=self._redis_url)
            except ImportError:
                logger.warning("redis_event_publisher.redis_not_installed")
                raise
            except Exception as e:
                logger.error("redis_event_publisher.connection_failed", error=str(e))
                raise
        return self._redis

    def publish(self, event: ConfigurationChangedEvent) -> bool:
        """Publish event to Redis Pub/Sub channel."""
        try:
            r = self._get_redis()
            r.publish(self.CHANNEL, event.to_json())
            logger.debug(
                "redis_event_publisher.published",
                key=f"{event.category}.{event.key}",
            )
            return True
        except Exception as e:
            logger.error("redis_event_publisher.publish_failed", error=str(e))
            return False

    def subscribe(self, callback: Callable[[ConfigurationChangedEvent], None]) -> None:
        """Subscribe to Redis configuration change events."""
        self._callbacks.append(callback)
        if not self._running:
            self._start_listener()

    def _start_listener(self):
        """Start background thread listening for Redis messages."""
        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="redis-config-listener",
        )
        self._listener_thread.start()

    def _listen_loop(self):
        """Background loop that listens for Redis Pub/Sub messages."""
        while self._running:
            try:
                r = self._get_redis()
                self._pubsub = r.pubsub()
                self._pubsub.subscribe(self.CHANNEL)

                for message in self._pubsub.listen():
                    if not self._running:
                        break
                    if message["type"] == "message":
                        try:
                            data = json.loads(message["data"])
                            event = ConfigurationChangedEvent.from_dict(data)
                            for cb in self._callbacks:
                                cb(event)
                        except Exception as e:
                            logger.error(
                                "redis_event_publisher.parse_error", error=str(e)
                            )
            except Exception as e:
                logger.error("redis_event_publisher.listener_error", error=str(e))
                if self._running:
                    time.sleep(5)  # Retry after delay

    def close(self):
        """Stop listener and close Redis connections."""
        self._running = False
        if self._pubsub:
            try:
                self._pubsub.unsubscribe()
                self._pubsub.close()
            except Exception:
                pass
        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass


# ─── Event Bus ───────────────────────────────────────────────────────────────


class ConfigEventBus:
    """
    Central event bus for configuration changes.

    Aggregates multiple publishers and dispatches events to all of them.
    Provides both local callbacks and distributed event publishing.
    """

    def __init__(self):
        self._publishers: list[EventPublisher] = []
        self._local_callbacks: list[Callable[[ConfigurationChangedEvent], None]] = []
        self._event_history: list[ConfigurationChangedEvent] = []
        self._max_history = 1000
        self._lock = threading.Lock()

    def add_publisher(self, publisher: EventPublisher) -> None:
        """Register an event publisher."""
        self._publishers.append(publisher)

    def on_change(self, callback: Callable[[ConfigurationChangedEvent], None]) -> None:
        """Register a local callback for configuration changes."""
        with self._lock:
            self._local_callbacks.append(callback)

    def emit(
        self,
        category: str,
        key: str,
        old_value: Optional[str],
        new_value: str,
        updated_by: str = "system",
        version: int = 1,
        change_reason: str = "",
    ) -> ConfigurationChangedEvent:
        """
        Emit a configuration change event.

        Notifies local callbacks and publishes to all registered publishers.
        """
        event = ConfigurationChangedEvent(
            category=category,
            key=key,
            old_value=old_value,
            new_value=new_value,
            updated_by=updated_by,
            version=version,
            change_reason=change_reason,
        )

        # Store in history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

        # Notify local callbacks
        for cb in self._local_callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error("event_bus.callback_error", error=str(e))

        # Publish to all publishers
        for publisher in self._publishers:
            try:
                publisher.publish(event)
            except Exception as e:
                logger.error(
                    "event_bus.publish_error",
                    publisher=type(publisher).__name__,
                    error=str(e),
                )

        return event

    def get_recent_events(self, limit: int = 50) -> list[ConfigurationChangedEvent]:
        """Get recent configuration change events."""
        with self._lock:
            return list(self._event_history[-limit:])

    def close(self):
        """Clean up all publishers."""
        for publisher in self._publishers:
            try:
                publisher.close()
            except Exception:
                pass

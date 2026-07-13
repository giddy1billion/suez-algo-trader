"""
Message Bus Abstraction — Transport-agnostic event delivery.

Provides an abstract interface that business logic codes against,
with pluggable backends:
- InMemoryBus (default, current behavior)
- PersistentBus (writes to SQLite before dispatching)
- RedisBus (cross-instance broadcasting via Redis Pub/Sub)

Business logic uses MessageBus interface and is unaware of transport.
"""

import json
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Type

from src.core.events import Event, EventBus
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MessageBus(ABC):
    """Abstract message bus interface."""

    @abstractmethod
    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        ...

    @abstractmethod
    def subscribe(self, event_type: Optional[Type[Event]], handler: Callable) -> None:
        """Subscribe to events of a given type (None = wildcard)."""
        ...

    @abstractmethod
    def unsubscribe(self, event_type: Optional[Type[Event]], handler: Callable) -> None:
        """Unsubscribe a handler."""
        ...

    @abstractmethod
    def get_history(self, limit: int = 50) -> list:
        """Get recent event history."""
        ...

    @property
    @abstractmethod
    def subscriber_count(self) -> int:
        """Total number of handler registrations."""
        ...


class InMemoryBus(MessageBus):
    """In-memory message bus (wraps existing EventBus)."""

    def __init__(self, max_history: int = 1000):
        self._bus = EventBus(max_history=max_history)

    def publish(self, event: Event) -> None:
        self._bus.publish(event)

    def subscribe(self, event_type: Optional[Type[Event]], handler: Callable) -> None:
        self._bus.subscribe(event_type, handler)

    def unsubscribe(self, event_type: Optional[Type[Event]], handler: Callable) -> None:
        self._bus.unsubscribe(event_type, handler)

    def get_history(self, limit: int = 50) -> list:
        return self._bus.get_history(limit=limit)

    @property
    def subscriber_count(self) -> int:
        return self._bus.subscriber_count


class PersistentBus(MessageBus):
    """
    Persistent message bus — writes events to durable store before dispatching.

    Guarantees: events are persisted BEFORE subscribers see them.
    Useful for development/paper trading where durability matters
    but a full message queue is overkill.
    """

    def __init__(self, event_store, max_history: int = 1000):
        self._bus = EventBus(max_history=max_history)
        self._event_store = event_store

    def publish(self, event: Event) -> None:
        # Persist first (guarantee durability)
        try:
            self._event_store.persist(event)
        except Exception:
            logger.warning("PersistentBus: failed to persist %s", type(event).__name__)
        # Then dispatch to subscribers
        self._bus.publish(event)

    def subscribe(self, event_type: Optional[Type[Event]], handler: Callable) -> None:
        self._bus.subscribe(event_type, handler)

    def unsubscribe(self, event_type: Optional[Type[Event]], handler: Callable) -> None:
        self._bus.unsubscribe(event_type, handler)

    def get_history(self, limit: int = 50) -> list:
        return self._bus.get_history(limit=limit)

    @property
    def subscriber_count(self) -> int:
        return self._bus.subscriber_count


class RedisBus(MessageBus):
    """
    Redis-backed message bus — combines local dispatch with cross-instance broadcasting.

    Events are dispatched locally AND published to a Redis channel so that
    multiple container instances share events in real time.
    """

    CHANNEL = "events"

    def __init__(self, redis_url: str, key_prefix: str = "suez:", max_history: int = 1000):
        self._local_bus = EventBus(max_history=max_history)
        self._key_prefix = key_prefix
        self._channel = f"{key_prefix}{self.CHANNEL}"
        self._running = True
        self._redis = None
        self._listener_thread = None

        try:
            import redis
            self._redis = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            self._redis.ping()
            self._start_listener()
            logger.info("bus.redis.connected", channel=self._channel)
        except Exception as e:
            logger.warning("bus.redis.connection_failed", error=str(e), msg="Operating in local-only mode")
            self._redis = None

    def _start_listener(self):
        """Start background thread that listens for events from other instances."""
        def _listen():
            try:
                ps = self._redis.pubsub()
                ps.subscribe(self._channel)
                while self._running:
                    msg = ps.get_message(timeout=1.0)
                    if msg and msg["type"] == "message":
                        try:
                            data = json.loads(msg["data"])
                            if data.get("_source_instance") != id(self):
                                event = Event.from_dict(data)
                                self._local_bus.publish(event)
                        except Exception as e:
                            logger.debug("bus.redis.parse_error", error=str(e))
            except Exception as e:
                if self._running:
                    logger.error("bus.redis.listener_died", error=str(e))

        self._listener_thread = threading.Thread(target=_listen, daemon=True, name="redis-bus-listener")
        self._listener_thread.start()

    def publish(self, event: Event) -> None:
        self._local_bus.publish(event)
        if self._redis:
            try:
                payload = event.to_dict()
                payload["_source_instance"] = id(self)
                self._redis.publish(self._channel, json.dumps(payload, default=str))
            except Exception as e:
                logger.debug("bus.redis.publish_failed", error=str(e))

    def subscribe(self, event_type: Optional[Type[Event]], handler: Callable) -> None:
        self._local_bus.subscribe(event_type, handler)

    def unsubscribe(self, event_type: Optional[Type[Event]], handler: Callable) -> None:
        self._local_bus.unsubscribe(event_type, handler)

    def get_history(self, limit: int = 50) -> list:
        return self._local_bus.get_history(limit=limit)

    @property
    def subscriber_count(self) -> int:
        return self._local_bus.subscriber_count

    def close(self):
        """Stop listener and close Redis connections."""
        self._running = False
        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass


def create_bus(backend: str = "memory", **kwargs) -> MessageBus:
    """
    Factory for creating message bus instances.

    Args:
        backend: "memory" | "persistent" | "redis"
        **kwargs: Backend-specific configuration.
            - max_history (int): Event history limit (all backends).
            - event_store: Required for "persistent" backend.
            - redis_url (str): Required for "redis" backend.
            - key_prefix (str): Redis key/channel prefix (default: "suez:").

    Returns:
        MessageBus implementation.
    """
    if backend == "memory":
        return InMemoryBus(max_history=kwargs.get("max_history", 1000))
    elif backend == "persistent":
        event_store = kwargs.get("event_store")
        if event_store is None:
            raise ValueError("PersistentBus requires 'event_store' kwarg")
        return PersistentBus(event_store, max_history=kwargs.get("max_history", 1000))
    elif backend == "redis":
        redis_url = kwargs.get("redis_url", "")
        if not redis_url:
            logger.warning("bus.redis_url_empty, falling back to InMemoryBus")
            return InMemoryBus(max_history=kwargs.get("max_history", 1000))
        return RedisBus(
            redis_url=redis_url,
            key_prefix=kwargs.get("key_prefix", "suez:"),
            max_history=kwargs.get("max_history", 1000),
        )
    else:
        raise ValueError(f"Unknown bus backend: {backend}. Supported: memory, persistent, redis")

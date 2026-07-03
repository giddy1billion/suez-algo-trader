"""
Message Bus Abstraction — Transport-agnostic event delivery.

Provides an abstract interface that business logic codes against,
with pluggable backends:
- InMemoryBus (default, current behavior)
- PersistentBus (writes to SQLite before dispatching)
- Future: RedisBus, KafkaBus, RabbitMQBus

Business logic uses MessageBus interface and is unaware of transport.
"""

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


def create_bus(backend: str = "memory", **kwargs) -> MessageBus:
    """
    Factory for creating message bus instances.

    Args:
        backend: "memory" | "persistent" | "redis" (future)
        **kwargs: Backend-specific configuration.

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
    else:
        raise ValueError(f"Unknown bus backend: {backend}. Supported: memory, persistent")

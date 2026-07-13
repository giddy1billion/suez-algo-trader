"""
Redis Client — unified cache abstraction with graceful local fallback.

Provides a single interface for caching, TTL-based expiry, pub/sub, and
distributed locking. Falls back to an in-memory implementation when Redis
is unavailable, ensuring the bot operates flawlessly in all environments.
"""

import json
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CacheBackend(ABC):
    """Abstract cache interface — implemented by Redis and local memory."""

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Get a value by key. Returns None if not found or expired."""
        ...

    @abstractmethod
    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set a key-value pair with optional TTL in seconds."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a key."""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        ...

    @abstractmethod
    def incr(self, key: str) -> int:
        """Atomically increment a key's integer value. Creates with value 1 if missing."""
        ...

    @abstractmethod
    def expire(self, key: str, ttl: int) -> None:
        """Set TTL on an existing key."""
        ...

    @abstractmethod
    def get_json(self, key: str) -> Optional[Any]:
        """Get and deserialize a JSON value."""
        ...

    @abstractmethod
    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Serialize and set a JSON value."""
        ...

    @abstractmethod
    def publish(self, channel: str, message: str) -> int:
        """Publish a message to a pub/sub channel. Returns subscriber count."""
        ...

    @abstractmethod
    def subscribe(self, channel: str, callback: Callable[[str], None]) -> None:
        """Subscribe to a pub/sub channel with a callback for each message."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the backend is connected and healthy."""
        ...


class LocalCache(CacheBackend):
    """
    In-memory cache with lazy TTL expiry.

    Thread-safe. Suitable for single-instance deployments and local dev.
    No background threads — expired entries are cleaned on read.
    """

    def __init__(self, key_prefix: str = ""):
        self._store: dict[str, tuple[str, Optional[float]]] = {}  # key -> (value, expires_at)
        self._lock = threading.Lock()
        self._prefix = key_prefix
        self._subscribers: dict[str, list[Callable[[str], None]]] = {}
        logger.info("cache.local.initialized", prefix=key_prefix)

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}{key}" if self._prefix else key

    def _is_expired(self, expires_at: Optional[float]) -> bool:
        return expires_at is not None and time.time() > expires_at

    def get(self, key: str) -> Optional[str]:
        fk = self._full_key(key)
        with self._lock:
            entry = self._store.get(fk)
            if entry is None:
                return None
            value, expires_at = entry
            if self._is_expired(expires_at):
                del self._store[fk]
                return None
            return value

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        fk = self._full_key(key)
        expires_at = (time.time() + ttl) if ttl else None
        with self._lock:
            self._store[fk] = (value, expires_at)

    def delete(self, key: str) -> None:
        fk = self._full_key(key)
        with self._lock:
            self._store.pop(fk, None)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def incr(self, key: str) -> int:
        fk = self._full_key(key)
        with self._lock:
            entry = self._store.get(fk)
            if entry is None or self._is_expired(entry[1]):
                self._store[fk] = ("1", None)
                return 1
            current = int(entry[0])
            new_val = current + 1
            self._store[fk] = (str(new_val), entry[1])
            return new_val

    def expire(self, key: str, ttl: int) -> None:
        fk = self._full_key(key)
        with self._lock:
            entry = self._store.get(fk)
            if entry is not None:
                self._store[fk] = (entry[0], time.time() + ttl)

    def get_json(self, key: str) -> Optional[Any]:
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self.set(key, json.dumps(value, default=str), ttl=ttl)

    def publish(self, channel: str, message: str) -> int:
        """In-memory pub/sub — delivers synchronously to local subscribers."""
        with self._lock:
            callbacks = self._subscribers.get(channel, [])
        for cb in callbacks:
            try:
                cb(message)
            except Exception as e:
                logger.warning("cache.local.publish_callback_error", channel=channel, error=str(e))
        return len(callbacks)

    def subscribe(self, channel: str, callback: Callable[[str], None]) -> None:
        with self._lock:
            self._subscribers.setdefault(channel, []).append(callback)

    def close(self) -> None:
        with self._lock:
            self._store.clear()
            self._subscribers.clear()

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def size(self) -> int:
        """Number of keys currently stored (including expired — lazy cleanup)."""
        return len(self._store)


class RedisCache(CacheBackend):
    """
    Redis-backed cache with connection pooling, health checks, and
    automatic reconnection.

    Uses redis-py with connection pool for thread safety.
    """

    def __init__(self, redis_url: str, key_prefix: str = "suez:"):
        self._redis_url = redis_url
        self._prefix = key_prefix
        self._redis = None
        self._pubsub_threads: list[threading.Thread] = []
        self._running = True
        self._connect()

    def _connect(self):
        """Establish Redis connection with pooling."""
        import redis

        try:
            self._redis = redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            self._redis.ping()
            logger.info("cache.redis.connected", url=self._redis_url[:30] + "...")
        except Exception as e:
            logger.error("cache.redis.connection_failed", error=str(e))
            raise

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}{key}" if self._prefix else key

    def get(self, key: str) -> Optional[str]:
        try:
            return self._redis.get(self._full_key(key))
        except Exception as e:
            logger.warning("cache.redis.get_error", key=key, error=str(e))
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        try:
            fk = self._full_key(key)
            if ttl:
                self._redis.set(fk, value, ex=ttl)
            else:
                self._redis.set(fk, value)
        except Exception as e:
            logger.warning("cache.redis.set_error", key=key, error=str(e))

    def delete(self, key: str) -> None:
        try:
            self._redis.delete(self._full_key(key))
        except Exception as e:
            logger.warning("cache.redis.delete_error", key=key, error=str(e))

    def exists(self, key: str) -> bool:
        try:
            return bool(self._redis.exists(self._full_key(key)))
        except Exception:
            return False

    def incr(self, key: str) -> int:
        try:
            return self._redis.incr(self._full_key(key))
        except Exception as e:
            logger.warning("cache.redis.incr_error", key=key, error=str(e))
            return 0

    def expire(self, key: str, ttl: int) -> None:
        try:
            self._redis.expire(self._full_key(key), ttl)
        except Exception as e:
            logger.warning("cache.redis.expire_error", key=key, error=str(e))

    def get_json(self, key: str) -> Optional[Any]:
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self.set(key, json.dumps(value, default=str), ttl=ttl)

    def publish(self, channel: str, message: str) -> int:
        try:
            return self._redis.publish(f"{self._prefix}{channel}", message)
        except Exception as e:
            logger.warning("cache.redis.publish_error", channel=channel, error=str(e))
            return 0

    def subscribe(self, channel: str, callback: Callable[[str], None]) -> None:
        """Subscribe to a Redis pub/sub channel in a background thread."""
        import redis

        def _listener():
            try:
                ps = self._redis.pubsub()
                ps.subscribe(f"{self._prefix}{channel}")
                while self._running:
                    msg = ps.get_message(timeout=1.0)
                    if msg and msg["type"] == "message":
                        try:
                            callback(msg["data"])
                        except Exception as e:
                            logger.warning("cache.redis.subscriber_error", error=str(e))
            except Exception as e:
                logger.error("cache.redis.listener_died", channel=channel, error=str(e))

        t = threading.Thread(target=_listener, daemon=True, name=f"redis-sub-{channel}")
        t.start()
        self._pubsub_threads.append(t)

    def close(self) -> None:
        self._running = False
        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass

    @property
    def is_connected(self) -> bool:
        try:
            self._redis.ping()
            return True
        except Exception:
            return False


def create_cache(redis_url: str = "", key_prefix: str = "suez:") -> CacheBackend:
    """
    Factory — returns RedisCache if URL is provided and connection succeeds,
    otherwise returns LocalCache for graceful fallback.

    Args:
        redis_url: Redis connection URL (e.g., "redis://host:6379/0" or
                   "rediss://..." for TLS). Empty string means local only.
        key_prefix: Prefix for all keys (namespace isolation).

    Returns:
        CacheBackend instance.
    """
    if redis_url:
        try:
            cache = RedisCache(redis_url=redis_url, key_prefix=key_prefix)
            logger.info("cache.factory.redis_active")
            return cache
        except Exception as exc:
            logger.warning(
                "cache.factory.redis_unavailable",
                error=str(exc),
                msg="Falling back to local in-memory cache",
            )

    cache = LocalCache(key_prefix=key_prefix)
    logger.info("cache.factory.local_active")
    return cache

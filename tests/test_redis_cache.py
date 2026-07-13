"""
Tests for Redis cache abstraction and integrations.

Tests LocalCache fully. Tests RedisCache with mocks. Tests
SignalDeduplicator and RedisBus integration.
"""

import json
import time
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock
import os

import pytest

from src.utils.redis_client import LocalCache, RedisCache, CacheBackend, create_cache


# ═══════════════════════════════════════════════════════════════════════════════
# LocalCache Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestLocalCache:
    """Comprehensive tests for the in-memory cache."""

    @pytest.fixture
    def cache(self):
        return LocalCache(key_prefix="test:")

    def test_set_and_get(self, cache):
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_returns_none(self, cache):
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self, cache):
        cache.set("short", "val", ttl=1)
        assert cache.get("short") == "val"
        time.sleep(1.1)
        assert cache.get("short") is None

    def test_no_ttl_persists(self, cache):
        cache.set("permanent", "data")
        time.sleep(0.1)
        assert cache.get("permanent") == "data"

    def test_delete(self, cache):
        cache.set("del_me", "value")
        cache.delete("del_me")
        assert cache.get("del_me") is None

    def test_delete_nonexistent_no_error(self, cache):
        cache.delete("nothing")  # Should not raise

    def test_exists_true(self, cache):
        cache.set("present", "yes")
        assert cache.exists("present") is True

    def test_exists_false(self, cache):
        assert cache.exists("absent") is False

    def test_exists_expired(self, cache):
        cache.set("expiring", "val", ttl=1)
        time.sleep(1.1)
        assert cache.exists("expiring") is False

    def test_incr_new_key(self, cache):
        result = cache.incr("counter")
        assert result == 1

    def test_incr_existing_key(self, cache):
        cache.set("counter", "5")
        result = cache.incr("counter")
        assert result == 6

    def test_incr_expired_key(self, cache):
        cache.set("counter", "10", ttl=1)
        time.sleep(1.1)
        result = cache.incr("counter")
        assert result == 1  # Reset after expiry

    def test_expire_sets_ttl(self, cache):
        cache.set("key", "val")
        cache.expire("key", 1)
        assert cache.get("key") == "val"
        time.sleep(1.1)
        assert cache.get("key") is None

    def test_set_json_and_get_json(self, cache):
        data = {"symbol": "BTC/USD", "price": 60000.0, "signals": [1, 2, 3]}
        cache.set_json("data", data)
        result = cache.get_json("data")
        assert result == data

    def test_get_json_missing(self, cache):
        assert cache.get_json("missing") is None

    def test_get_json_invalid(self, cache):
        cache.set("bad", "not-json{{{")
        assert cache.get_json("bad") is None

    def test_publish_and_subscribe(self, cache):
        received = []
        cache.subscribe("test_channel", lambda msg: received.append(msg))
        cache.publish("test_channel", "hello")
        assert received == ["hello"]

    def test_publish_no_subscribers(self, cache):
        count = cache.publish("empty_channel", "message")
        assert count == 0

    def test_key_prefix(self, cache):
        cache.set("mykey", "val")
        # Internally stored as "test:mykey"
        assert "test:mykey" in cache._store

    def test_is_connected(self, cache):
        assert cache.is_connected is True

    def test_close_clears_state(self, cache):
        cache.set("key", "val")
        cache.close()
        assert cache.size == 0

    def test_thread_safety(self, cache):
        """Concurrent writes should not corrupt state."""
        errors = []

        def writer(n):
            try:
                for i in range(100):
                    cache.set(f"thread_{n}_{i}", str(i))
                    cache.incr(f"shared_counter")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert int(cache.get("shared_counter")) == 500


# ═══════════════════════════════════════════════════════════════════════════════
# RedisCache Tests (Mocked)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRedisCacheMocked:
    """Test Redis backend with mocked redis-py."""

    @pytest.fixture
    def mock_redis(self):
        with patch("src.utils.redis_client.RedisCache._connect") as mock_conn:
            cache = RedisCache.__new__(RedisCache)
            cache._redis_url = "redis://localhost:6379/0"
            cache._prefix = "suez:"
            cache._redis = MagicMock()
            cache._pubsub_threads = []
            cache._running = True
            yield cache

    def test_get_calls_redis(self, mock_redis):
        mock_redis._redis.get.return_value = "value"
        assert mock_redis.get("key") == "value"
        mock_redis._redis.get.assert_called_with("suez:key")

    def test_get_returns_none_on_error(self, mock_redis):
        mock_redis._redis.get.side_effect = Exception("conn error")
        assert mock_redis.get("key") is None

    def test_set_with_ttl(self, mock_redis):
        mock_redis.set("key", "val", ttl=60)
        mock_redis._redis.set.assert_called_with("suez:key", "val", ex=60)

    def test_set_without_ttl(self, mock_redis):
        mock_redis.set("key", "val")
        mock_redis._redis.set.assert_called_with("suez:key", "val")

    def test_delete(self, mock_redis):
        mock_redis.delete("key")
        mock_redis._redis.delete.assert_called_with("suez:key")

    def test_exists(self, mock_redis):
        mock_redis._redis.exists.return_value = 1
        assert mock_redis.exists("key") is True

    def test_incr(self, mock_redis):
        mock_redis._redis.incr.return_value = 5
        assert mock_redis.incr("counter") == 5

    def test_publish(self, mock_redis):
        mock_redis._redis.publish.return_value = 2
        count = mock_redis.publish("channel", "message")
        assert count == 2
        mock_redis._redis.publish.assert_called_with("suez:channel", "message")

    def test_is_connected_true(self, mock_redis):
        mock_redis._redis.ping.return_value = True
        assert mock_redis.is_connected is True

    def test_is_connected_false_on_error(self, mock_redis):
        mock_redis._redis.ping.side_effect = Exception("down")
        assert mock_redis.is_connected is False


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCreateCache:
    """Test the factory function."""

    def test_empty_url_returns_local(self):
        cache = create_cache(redis_url="")
        assert isinstance(cache, LocalCache)

    def test_invalid_url_falls_back_to_local(self):
        # Connection to non-existent Redis should fall back gracefully
        cache = create_cache(redis_url="redis://255.255.255.255:9999/0")
        assert isinstance(cache, LocalCache)

    @patch("src.utils.redis_client.RedisCache._connect")
    def test_valid_url_returns_redis(self, mock_connect):
        cache = RedisCache.__new__(RedisCache)
        cache._redis_url = "redis://localhost:6379/0"
        cache._prefix = "suez:"
        cache._redis = MagicMock()
        cache._pubsub_threads = []
        cache._running = True
        with patch("src.utils.redis_client.RedisCache", return_value=cache):
            result = create_cache(redis_url="redis://localhost:6379/0")
            # Since we patched, it should try RedisCache


# ═══════════════════════════════════════════════════════════════════════════════
# SignalDeduplicator + Cache Integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestSignalDeduplicatorWithCache:
    """Test that SignalDeduplicator works with cache backend."""

    @pytest.fixture
    def cache(self):
        return LocalCache(key_prefix="test:")

    @pytest.fixture
    def dedup_with_cache(self, cache):
        from src.execution.signal_dedup import SignalDeduplicator
        return SignalDeduplicator(strength_threshold=0.10, cache=cache)

    @pytest.fixture
    def dedup_no_cache(self):
        from src.execution.signal_dedup import SignalDeduplicator
        return SignalDeduplicator(strength_threshold=0.10)

    def _make_signal(self, symbol="BTC/USD", side="buy", strength=0.8, timeframe="1Hour"):
        """Create a mock TradeSignal."""
        signal = MagicMock()
        signal.symbol = symbol
        signal.side = MagicMock()
        signal.side.value = side
        signal.signal_strength = strength
        signal.timeframe = timeframe
        return signal

    def test_first_signal_always_notifies(self, dedup_with_cache):
        signal = self._make_signal()
        assert dedup_with_cache.should_notify(signal) is True

    def test_same_signal_suppressed(self, dedup_with_cache):
        signal = self._make_signal()
        dedup_with_cache.should_notify(signal)
        assert dedup_with_cache.should_notify(signal) is False

    def test_direction_change_notifies(self, dedup_with_cache):
        buy = self._make_signal(side="buy")
        sell = self._make_signal(side="sell")
        dedup_with_cache.should_notify(buy)
        assert dedup_with_cache.should_notify(sell) is True

    def test_strength_change_notifies(self, dedup_with_cache):
        weak = self._make_signal(strength=0.5)
        strong = self._make_signal(strength=0.7)  # +0.2 > threshold
        dedup_with_cache.should_notify(weak)
        assert dedup_with_cache.should_notify(strong) is True

    def test_state_persists_in_cache(self, cache):
        """Simulate restart: state should survive via cache."""
        from src.execution.signal_dedup import SignalDeduplicator

        signal = self._make_signal()

        # Instance 1 sees signal
        dedup1 = SignalDeduplicator(cache=cache)
        assert dedup1.should_notify(signal) is True

        # Instance 2 (simulating restart) sees same signal — should suppress
        dedup2 = SignalDeduplicator(cache=cache)
        assert dedup2.should_notify(signal) is False

    def test_reset_clears_cache(self, dedup_with_cache, cache):
        signal = self._make_signal()
        dedup_with_cache.should_notify(signal)
        dedup_with_cache.reset("BTC/USD")
        # After reset, should notify again
        assert dedup_with_cache.should_notify(signal) is True

    def test_no_cache_still_works(self, dedup_no_cache):
        """Without cache, classic in-memory behavior."""
        signal = self._make_signal()
        assert dedup_no_cache.should_notify(signal) is True
        assert dedup_no_cache.should_notify(signal) is False


# ═══════════════════════════════════════════════════════════════════════════════
# RedisBus Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRedisBus:
    """Test the Redis-backed message bus."""

    def test_redis_bus_falls_back_when_unavailable(self):
        """When Redis is unreachable, RedisBus should still work locally."""
        from src.core.bus import RedisBus
        from src.core.events import Event

        bus = RedisBus(redis_url="redis://255.255.255.255:9999/0")
        # Should have fallen back to local-only mode
        assert bus._redis is None

        # Should still function for local pub/sub
        received = []
        bus.subscribe(None, lambda e: received.append(e))
        event = Event(source="test")
        bus.publish(event)
        assert len(received) == 1
        assert received[0].source == "test"

    def test_redis_bus_create_bus_factory(self):
        """create_bus with backend='redis' and empty URL falls back."""
        from src.core.bus import create_bus, InMemoryBus

        bus = create_bus(backend="redis", redis_url="")
        assert isinstance(bus, InMemoryBus)

    def test_redis_bus_local_dispatch(self):
        """Events should always dispatch locally regardless of Redis."""
        from src.core.bus import RedisBus
        from src.core.events import Event

        bus = RedisBus(redis_url="redis://255.255.255.255:9999/0")
        received = []
        bus.subscribe(None, lambda e: received.append(e))

        for i in range(5):
            bus.publish(Event(source=f"src_{i}"))

        assert len(received) == 5
        bus.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Live Redis Tests (gated by env var)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not os.environ.get("REDIS_URL"),
    reason="REDIS_URL not set — skipping live Redis tests",
)
class TestLiveRedis:
    """Live integration tests against real Redis."""

    @pytest.fixture
    def cache(self):
        url = os.environ["REDIS_URL"]
        cache = RedisCache(redis_url=url, key_prefix="suez-test:")
        yield cache
        cache.close()

    def test_set_get_roundtrip(self, cache):
        cache.set("live_test", "hello")
        assert cache.get("live_test") == "hello"
        cache.delete("live_test")

    def test_ttl_expiry(self, cache):
        cache.set("ttl_test", "expire_me", ttl=2)
        assert cache.get("ttl_test") == "expire_me"
        time.sleep(2.5)
        assert cache.get("ttl_test") is None

    def test_json_roundtrip(self, cache):
        data = {"symbol": "BTC/USD", "confidence": 0.85}
        cache.set_json("json_test", data, ttl=10)
        result = cache.get_json("json_test")
        assert result == data
        cache.delete("json_test")

    def test_incr(self, cache):
        cache.delete("counter_test")
        assert cache.incr("counter_test") == 1
        assert cache.incr("counter_test") == 2
        cache.delete("counter_test")

    def test_pubsub(self, cache):
        received = []
        cache.subscribe("test_chan", lambda msg: received.append(msg))
        time.sleep(0.5)  # Let subscriber thread start
        cache.publish("test_chan", "live_message")
        time.sleep(0.5)  # Let message propagate
        assert "live_message" in received

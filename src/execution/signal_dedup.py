"""
Signal Deduplication — Suppresses repeated identical signal notifications.

Only publishes notifications when:
1. A new signal appears (new symbol or direction change)
2. Signal strength changes significantly (> threshold)
3. After a configurable cooldown (tied to strategy timeframe)

Thread-safe: uses internal lock for concurrent access from multiple cycles.
Supports optional Redis backend for state that survives container restarts.
"""

import json
import threading
from datetime import datetime, timezone
from typing import Optional

from src.strategy.base import TradeSignal
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Timeframe → cooldown seconds mapping.
# Uses the candle duration: no point re-notifying within the same bar.
_TIMEFRAME_SECONDS = {
    "1Min": 60,
    "5Min": 300,
    "15Min": 900,
    "30Min": 1800,
    "1Hour": 3600,
    "4Hour": 14400,
    "1Day": 86400,
}


class SignalDeduplicator:
    """
    Tracks last-notified signal per symbol and determines whether
    a new notification should be emitted.

    Design:
    - Optional Redis backend (state survives restarts, works across replicas)
    - Falls back to in-memory dict when no cache provided
    - Thread-safe via internal lock
    - Configurable strength threshold
    - Cooldown derived from signal's timeframe (one full candle)
    """

    def __init__(self, strength_threshold: float = 0.10, cache=None):
        """
        Args:
            strength_threshold: Minimum strength delta to trigger re-notification.
            cache: Optional CacheBackend instance (from create_cache). When provided,
                   dedup state is stored in Redis with automatic TTL-based expiry.
        """
        self._threshold = strength_threshold
        self._state: dict[str, dict] = {}  # local fallback: symbol → {side, strength, timestamp}
        self._lock = threading.Lock()
        self._cache = cache

    @staticmethod
    def timeframe_to_seconds(timeframe: str) -> int:
        """Convert timeframe string to cooldown seconds."""
        return _TIMEFRAME_SECONDS.get(timeframe, 900)  # Default 15min

    def _cache_key(self, symbol: str) -> str:
        return f"dedup:{symbol}"

    def _get_prev(self, symbol: str) -> Optional[dict]:
        """Get previous signal state from cache or local memory."""
        if self._cache:
            data = self._cache.get_json(self._cache_key(symbol))
            if data and "timestamp" in data:
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            return data
        return self._state.get(symbol)

    def _set_state(self, symbol: str, side: str, strength: float, now: datetime, cooldown: int) -> None:
        """Store signal state in cache (with TTL) or local memory."""
        entry = {"side": side, "strength": strength, "timestamp": now.isoformat()}
        if self._cache:
            # TTL = 2x cooldown to allow slightly-late signals to still dedup
            self._cache.set_json(self._cache_key(symbol), entry, ttl=cooldown * 2)
        else:
            self._state[symbol] = {
                "side": side,
                "strength": strength,
                "timestamp": now,
            }

    def should_notify(self, signal: TradeSignal) -> bool:
        """
        Determine if this signal warrants a new notification.

        Returns True if:
        - First time seeing this symbol
        - Direction changed (BUY -> SELL or vice versa)
        - Strength changed by more than threshold
        - Cooldown period has elapsed (one full candle)
        """
        now = datetime.now(timezone.utc)
        cooldown = self.timeframe_to_seconds(signal.timeframe)

        with self._lock:
            prev = self._get_prev(signal.symbol)

            if prev is None:
                self._set_state(signal.symbol, signal.side.value, signal.signal_strength, now, cooldown)
                return True

            # Direction change — always notify
            if prev["side"] != signal.side.value:
                self._set_state(signal.symbol, signal.side.value, signal.signal_strength, now, cooldown)
                return True

            # Significant strength change
            if abs(signal.signal_strength - prev["strength"]) >= self._threshold:
                self._set_state(signal.symbol, signal.side.value, signal.signal_strength, now, cooldown)
                return True

            # Cooldown expired
            elapsed = (now - prev["timestamp"]).total_seconds()
            if elapsed >= cooldown:
                self._set_state(signal.symbol, signal.side.value, signal.signal_strength, now, cooldown)
                return True

            # Suppress — same direction, similar strength, within cooldown
            return False

    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset dedup state. If symbol given, only that symbol; else all."""
        with self._lock:
            if symbol:
                if self._cache:
                    self._cache.delete(self._cache_key(symbol))
                self._state.pop(symbol, None)
            else:
                # For cache-backed: we can't enumerate all keys easily,
                # so just clear local; Redis entries will expire via TTL
                self._state.clear()

    @property
    def tracked_symbols(self) -> list[str]:
        """Return list of symbols currently being tracked (local state only)."""
        with self._lock:
            return list(self._state.keys())

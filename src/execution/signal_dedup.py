"""
Signal Deduplication — Suppresses repeated identical signal notifications.

Only publishes notifications when:
1. A new signal appears (new symbol or direction change)
2. Signal strength changes significantly (> threshold)
3. After a configurable cooldown (tied to strategy timeframe)

Thread-safe: uses internal lock for concurrent access from multiple cycles.
"""

import threading
from datetime import datetime, timezone
from typing import Optional

from src.strategy.base import TradeSignal


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
    - In-memory state (resets on restart → first signal always notifies after deploy)
    - Thread-safe via internal lock
    - Configurable strength threshold
    - Cooldown derived from signal's timeframe (one full candle)
    """

    def __init__(self, strength_threshold: float = 0.10):
        self._threshold = strength_threshold
        self._state: dict[str, dict] = {}  # symbol → {side, strength, timestamp}
        self._lock = threading.Lock()

    @staticmethod
    def timeframe_to_seconds(timeframe: str) -> int:
        """Convert timeframe string to cooldown seconds."""
        return _TIMEFRAME_SECONDS.get(timeframe, 900)  # Default 15min

    def should_notify(self, signal: TradeSignal) -> bool:
        """
        Determine if this signal warrants a new notification.

        Returns True if:
        - First time seeing this symbol
        - Direction changed (BUY → SELL or vice versa)
        - Strength changed by more than threshold
        - Cooldown period has elapsed (one full candle)
        """
        now = datetime.now(timezone.utc)
        cooldown = self.timeframe_to_seconds(signal.timeframe)

        with self._lock:
            prev = self._state.get(signal.symbol)

            if prev is None:
                self._state[signal.symbol] = {
                    "side": signal.side.value,
                    "strength": signal.signal_strength,
                    "timestamp": now,
                }
                return True

            # Direction change — always notify
            if prev["side"] != signal.side.value:
                self._state[signal.symbol] = {
                    "side": signal.side.value,
                    "strength": signal.signal_strength,
                    "timestamp": now,
                }
                return True

            # Significant strength change
            if abs(signal.signal_strength - prev["strength"]) >= self._threshold:
                self._state[signal.symbol] = {
                    "side": signal.side.value,
                    "strength": signal.signal_strength,
                    "timestamp": now,
                }
                return True

            # Cooldown expired
            elapsed = (now - prev["timestamp"]).total_seconds()
            if elapsed >= cooldown:
                self._state[signal.symbol] = {
                    "side": signal.side.value,
                    "strength": signal.signal_strength,
                    "timestamp": now,
                }
                return True

            # Suppress — same direction, similar strength, within cooldown
            return False

    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset dedup state. If symbol given, only that symbol; else all."""
        with self._lock:
            if symbol:
                self._state.pop(symbol, None)
            else:
                self._state.clear()

    @property
    def tracked_symbols(self) -> list[str]:
        """Return list of symbols currently being tracked."""
        with self._lock:
            return list(self._state.keys())

"""
Tests for signal deduplication in SignalDeduplicator.

Validates that repeated identical signals are suppressed while meaningful
changes (direction, strength, cooldown expiry) still generate notifications.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

from src.strategy.base import TradeSignal, Side
from src.execution.signal_dedup import SignalDeduplicator


def make_signal(symbol="BTC/USD", side=Side.BUY, strength=0.65, timeframe="15Min"):
    """Create a minimal TradeSignal for testing."""
    return TradeSignal(
        signal_id=f"SIG_{symbol}_{side.value}_{strength}",
        strategy_id="momentum",
        strategy_version="1.0.0",
        symbol=symbol,
        timeframe=timeframe,
        timestamp=datetime.now(timezone.utc),
        side=side,
        signal_strength=strength,
        expected_direction=1 if side == Side.BUY else -1,
        tags=[],
        reason="test",
        features={},
        indicators={},
    )


class TestSignalDedup:
    """Signal deduplication logic tests."""

    def test_first_signal_always_notifies(self):
        """First signal for any symbol should always pass through."""
        dedup = SignalDeduplicator(strength_threshold=0.10)
        sig = make_signal("BTC/USD", Side.BUY, 0.65)
        assert dedup.should_notify(sig) is True

    def test_identical_signal_suppressed(self):
        """Same symbol, same direction, same strength → suppress."""
        dedup = SignalDeduplicator(strength_threshold=0.10)
        sig1 = make_signal("BTC/USD", Side.BUY, 0.65)
        sig2 = make_signal("BTC/USD", Side.BUY, 0.65)

        assert dedup.should_notify(sig1) is True
        assert dedup.should_notify(sig2) is False

    def test_direction_change_notifies(self):
        """Direction change (BUY→SELL) should always notify."""
        dedup = SignalDeduplicator(strength_threshold=0.10)
        buy = make_signal("BTC/USD", Side.BUY, 0.65)
        sell = make_signal("BTC/USD", Side.SELL, 0.65)

        assert dedup.should_notify(buy) is True
        assert dedup.should_notify(sell) is True

    def test_significant_strength_change_notifies(self):
        """Strength change > threshold should notify."""
        dedup = SignalDeduplicator(strength_threshold=0.10)
        sig1 = make_signal("BTC/USD", Side.BUY, 0.65)
        sig2 = make_signal("BTC/USD", Side.BUY, 0.80)  # +0.15 > 0.10 threshold

        assert dedup.should_notify(sig1) is True
        assert dedup.should_notify(sig2) is True

    def test_small_strength_change_suppressed(self):
        """Strength change < threshold should be suppressed."""
        dedup = SignalDeduplicator(strength_threshold=0.10)
        sig1 = make_signal("BTC/USD", Side.BUY, 0.65)
        sig2 = make_signal("BTC/USD", Side.BUY, 0.70)  # +0.05 < 0.10 threshold

        assert dedup.should_notify(sig1) is True
        assert dedup.should_notify(sig2) is False

    def test_cooldown_expiry_notifies(self):
        """After cooldown expires, same signal should notify again."""
        dedup = SignalDeduplicator(strength_threshold=0.10)
        sig = make_signal("BTC/USD", Side.BUY, 0.65, timeframe="15Min")

        assert dedup.should_notify(sig) is True

        # Manually backdate the timestamp to simulate cooldown expiry
        with dedup._lock:
            dedup._state["BTC/USD"]["timestamp"] = (
                datetime.now(timezone.utc) - timedelta(minutes=16)
            )

        assert dedup.should_notify(sig) is True

    def test_within_cooldown_suppressed(self):
        """Within cooldown window, same signal should be suppressed."""
        dedup = SignalDeduplicator(strength_threshold=0.10)
        sig = make_signal("BTC/USD", Side.BUY, 0.65, timeframe="15Min")

        assert dedup.should_notify(sig) is True
        # Immediately re-check (well within 15min cooldown)
        assert dedup.should_notify(sig) is False

    def test_different_symbols_independent(self):
        """Each symbol has independent dedup tracking."""
        dedup = SignalDeduplicator(strength_threshold=0.10)
        btc = make_signal("BTC/USD", Side.BUY, 0.65)
        eth = make_signal("ETH/USD", Side.BUY, 0.65)

        assert dedup.should_notify(btc) is True
        assert dedup.should_notify(eth) is True
        # Repeats of each should be suppressed independently
        assert dedup.should_notify(btc) is False
        assert dedup.should_notify(eth) is False

    def test_timeframe_to_seconds_known_values(self):
        """Verify timeframe → cooldown mapping."""
        assert SignalDeduplicator.timeframe_to_seconds("1Min") == 60
        assert SignalDeduplicator.timeframe_to_seconds("5Min") == 300
        assert SignalDeduplicator.timeframe_to_seconds("15Min") == 900
        assert SignalDeduplicator.timeframe_to_seconds("1Hour") == 3600
        assert SignalDeduplicator.timeframe_to_seconds("4Hour") == 14400
        assert SignalDeduplicator.timeframe_to_seconds("1Day") == 86400

    def test_timeframe_to_seconds_unknown_defaults(self):
        """Unknown timeframe should default to 900s (15min)."""
        assert SignalDeduplicator.timeframe_to_seconds("unknown") == 900
        assert SignalDeduplicator.timeframe_to_seconds("") == 900

    def test_direction_flip_resets_tracking(self):
        """After direction flip, going back should notify again."""
        dedup = SignalDeduplicator(strength_threshold=0.10)
        buy = make_signal("BTC/USD", Side.BUY, 0.65)
        sell = make_signal("BTC/USD", Side.SELL, 0.65)

        assert dedup.should_notify(buy) is True
        assert dedup.should_notify(sell) is True
        # Going back to BUY is another direction change
        assert dedup.should_notify(buy) is True

    def test_strength_threshold_edge_case(self):
        """Just above threshold should notify."""
        dedup = SignalDeduplicator(strength_threshold=0.10)
        sig1 = make_signal("BTC/USD", Side.BUY, 0.65)
        sig2 = make_signal("BTC/USD", Side.BUY, 0.76)  # +0.11 > 0.10

        assert dedup.should_notify(sig1) is True
        assert dedup.should_notify(sig2) is True  # > threshold

    def test_reset_single_symbol(self):
        """Reset for one symbol should not affect others."""
        dedup = SignalDeduplicator(strength_threshold=0.10)
        btc = make_signal("BTC/USD", Side.BUY, 0.65)
        eth = make_signal("ETH/USD", Side.BUY, 0.65)

        dedup.should_notify(btc)
        dedup.should_notify(eth)
        assert dedup.should_notify(btc) is False

        dedup.reset("BTC/USD")
        assert dedup.should_notify(btc) is True  # Reset → first signal again
        assert dedup.should_notify(eth) is False  # ETH unaffected

    def test_reset_all(self):
        """Full reset clears all symbols."""
        dedup = SignalDeduplicator(strength_threshold=0.10)
        btc = make_signal("BTC/USD", Side.BUY, 0.65)
        eth = make_signal("ETH/USD", Side.BUY, 0.65)

        dedup.should_notify(btc)
        dedup.should_notify(eth)

        dedup.reset()
        assert dedup.should_notify(btc) is True
        assert dedup.should_notify(eth) is True

    def test_tracked_symbols(self):
        """tracked_symbols returns list of symbols seen."""
        dedup = SignalDeduplicator(strength_threshold=0.10)
        assert dedup.tracked_symbols == []

        dedup.should_notify(make_signal("BTC/USD"))
        dedup.should_notify(make_signal("ETH/USD"))
        assert sorted(dedup.tracked_symbols) == ["BTC/USD", "ETH/USD"]

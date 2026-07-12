"""
Signal Adapter — Bridges legacy strategy outputs to the new TradeSignal format.

Existing strategies return LegacyTradeSignal (mutable, with confidence/price/SL/TP).
This adapter converts them to the new frozen TradeSignal (minimal proposal-only).

Pipeline:
    Strategy.generate_signals() → LegacyTradeSignal
        ↓ adapt_signal()
    TradeSignal (frozen, clean)
        ↓
    DecisionOrchestrator → DecisionContract
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Union

from src.strategy.base import (
    BaseStrategy,
    LegacyTradeSignal,
    Signal,
    Side,
    TradeSignal,
)


def adapt_signal(
    signal: Union[TradeSignal, LegacyTradeSignal],
    strategy: BaseStrategy,
) -> TradeSignal:
    """
    Convert any signal format to the new clean TradeSignal.

    If signal is already a TradeSignal (frozen), returns it unchanged.
    If signal is a LegacyTradeSignal, converts to new format.

    Args:
        signal: Either a new TradeSignal or legacy format.
        strategy: The strategy that produced this signal.

    Returns:
        A frozen TradeSignal instance.
    """
    # Already new format — pass through
    if isinstance(signal, TradeSignal):
        return signal

    # Legacy format — convert
    if not isinstance(signal, LegacyTradeSignal):
        raise TypeError(
            f"Expected TradeSignal or LegacyTradeSignal, got {type(signal).__name__}"
        )

    # Determine side from legacy Signal enum
    if signal.signal in (Signal.BUY, Signal.STRONG_BUY):
        side = Side.BUY
        expected_direction = 1
    elif signal.signal in (Signal.SELL, Signal.STRONG_SELL):
        side = Side.SELL
        expected_direction = -1
    else:
        # HOLD/NO_SIGNAL — shouldn't reach here if filtered upstream,
        # but handle gracefully
        side = Side.BUY
        expected_direction = 0

    # Map signal enum strength to tags
    tags = []
    if signal.signal == Signal.STRONG_BUY:
        tags.append("strong_signal")
    elif signal.signal == Signal.STRONG_SELL:
        tags.append("strong_signal")

    # Extract numeric indicators from the indicators dict
    # Use hasattr(v, 'item') to catch numpy scalars (numpy 2.0+ removed float subclassing)
    numeric_indicators = {}
    general_features = {}
    if signal.indicators:
        for key, value in signal.indicators.items():
            if isinstance(value, (int, float)) or hasattr(value, 'item'):
                numeric_indicators[key] = float(value)
            elif value is None:
                numeric_indicators[key] = None
            else:
                general_features[key] = value

    # Preserve legacy SL/TP in features for the DecisionOrchestrator
    # (it may use strategy-proposed levels as hints)
    if signal.stop_loss is not None:
        general_features["strategy_proposed_stop_loss"] = float(signal.stop_loss)
    if signal.take_profit is not None:
        general_features["strategy_proposed_take_profit"] = float(signal.take_profit)
    if signal.price:
        general_features["observed_price"] = float(signal.price)

    return TradeSignal(
        signal_id=f"SIG-{uuid.uuid4().hex[:8]}",
        strategy_id=strategy.name,
        strategy_version=getattr(strategy, "version", "1.0.0"),
        symbol=signal.symbol,
        timeframe=strategy.timeframe,
        timestamp=datetime.now(timezone.utc),
        side=side,
        signal_strength=float(signal.confidence),
        expected_direction=expected_direction,
        tags=tuple(tags),
        reason=signal.reason,
        features=general_features,
        indicators=numeric_indicators,
    )


def is_legacy_signal(signal) -> bool:
    """Check if a signal is the legacy format."""
    return isinstance(signal, LegacyTradeSignal)


def is_actionable(signal: Union[TradeSignal, LegacyTradeSignal]) -> bool:
    """Check if a signal is actionable regardless of format."""
    if isinstance(signal, TradeSignal):
        return signal.is_actionable
    if isinstance(signal, LegacyTradeSignal):
        return signal.is_actionable
    return False

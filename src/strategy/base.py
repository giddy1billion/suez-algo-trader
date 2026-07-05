"""
Base Strategy — Abstract class that all trading strategies must implement.
Defines the interface for signal generation, entry/exit logic.

Architecture:
    TradeSignal is intentionally MINIMAL — it is a strategy PROPOSAL only.
    It does NOT contain position sizing, risk allocation, confidence after
    calibration, or execution approval. Those belong to the DecisionContract.

    Pipeline: Strategy → TradeSignal → DecisionOrchestrator → DecisionContract
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────


class Side(str, Enum):
    """Trade direction — intentionally binary. Strength is signal_strength."""
    BUY = "BUY"
    SELL = "SELL"


class Signal(Enum):
    """Legacy trading signal types. Retained for backward compatibility."""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2
    NO_SIGNAL = -99


# ──────────────────────────────────────────────────────────────────────────────
# TradeSignal — The Clean Architecture signal (frozen, minimal, proposal-only)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TradeSignal:
    """
    Lightweight strategy proposal — contains ONLY what the strategy knows.

    This is NOT an execution instruction. It is a proposal that says:
    "I think we should trade."

    Does NOT contain:
        - Position size
        - Risk percentage
        - Confidence after calibration
        - Kelly fraction
        - Portfolio exposure
        - Execution approval
        - Broker information
        - Stop loss / take profit chosen by the risk engine
        - Order type

    Those belong to the DecisionContract (produced by DecisionOrchestrator).
    """

    # ── Identity ──
    signal_id: str = field(default_factory=lambda: f"SIG-{uuid.uuid4().hex[:8]}")
    strategy_id: str = ""
    strategy_version: str = "1.0.0"

    # ── Market ──
    symbol: str = ""
    timeframe: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Proposal ──
    side: Side = Side.BUY
    signal_strength: float = 0.0        # 0.0-1.0, raw strategy output
    expected_direction: int = 1          # +1 (bullish) or -1 (bearish)

    # ── Strategy Metadata ──
    tags: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""

    # ── Strategy Evidence Only ──
    features: dict[str, Any] = field(default_factory=dict)
    indicators: dict[str, float] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """A signal is actionable if it has meaningful strength."""
        return self.signal_strength > 0.0 and self.symbol != ""

    @property
    def is_buy(self) -> bool:
        return self.side == Side.BUY

    @property
    def is_sell(self) -> bool:
        return self.side == Side.SELL

    def to_event_payload(self) -> dict[str, Any]:
        """Serialize to structured event bus payload."""
        return {
            "signal_id": self.signal_id,
            "strategy": {
                "id": self.strategy_id,
                "version": self.strategy_version,
            },
            "market": {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "timestamp": self.timestamp.isoformat(),
            },
            "signal": {
                "side": self.side.value,
                "strength": self.signal_strength,
                "expected_direction": self.expected_direction,
            },
            "metadata": {
                "tags": list(self.tags),
                "reason": self.reason,
            },
            "evidence": {
                "features": self.features,
                "indicators": self.indicators,
            },
        }


# ──────────────────────────────────────────────────────────────────────────────
# LegacyTradeSignal — Old mutable format (for backward compatibility)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class LegacyTradeSignal:
    """
    DEPRECATED: Old trade signal format. Retained for strategies not yet migrated.
    Use TradeSignal (frozen) for new code.
    """
    symbol: str
    signal: Signal
    confidence: float  # 0.0 to 1.0
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    indicators: dict = None

    def __post_init__(self):
        if self.indicators is None:
            self.indicators = {}

    @property
    def is_actionable(self) -> bool:
        return self.signal not in (Signal.HOLD, Signal.NO_SIGNAL) and self.confidence > 0.5


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Strategies produce TradeSignal objects — lightweight proposals.
    The DecisionOrchestrator downstream decides whether to execute.

    Strategies that haven't migrated yet can return LegacyTradeSignal;
    the signal adapter in the ExecutionEngine handles conversion.
    """

    # Override in subclass for versioning
    version: str = "1.0.0"

    def __init__(self, name: str, symbols: list[str], timeframe: str = "1Hour", lookback: int = 200):
        self.name = name
        self.symbols = symbols
        self.timeframe = timeframe
        self.lookback = lookback
        self._is_active = True

    @abstractmethod
    def generate_signals(self, data: dict[str, pd.DataFrame]) -> list:
        """
        Analyze market data and generate trading signals.

        Args:
            data: Dict mapping symbol -> DataFrame with OHLCV columns

        Returns:
            List of TradeSignal or LegacyTradeSignal objects.
            Prefer returning TradeSignal (frozen) for new strategies.
        """
        pass

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators needed for this strategy.
        Returns the DataFrame with indicator columns added.
        """
        pass

    def should_exit(self, symbol: str, position: dict, current_price: float) -> Optional[TradeSignal]:
        """
        Check if an existing position should be exited.
        Override for custom exit logic beyond stop-loss/take-profit.
        Default: relies on broker-side SL/TP orders.
        """
        return None

    def on_bar(self, symbol: str, bar: dict):
        """Called on each new bar (for streaming strategies). Override if needed."""
        pass

    def activate(self):
        self._is_active = True

    def deactivate(self):
        self._is_active = False

    @property
    def is_active(self) -> bool:
        return self._is_active

    def __repr__(self):
        return f"<{self.__class__.__name__}(name={self.name}, symbols={len(self.symbols)}, tf={self.timeframe})>"

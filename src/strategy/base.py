"""
Base Strategy — Abstract class that all trading strategies must implement.
Defines the interface for signal generation, entry/exit logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Signal(Enum):
    """Trading signal types."""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class TradeSignal:
    """A concrete trade signal with metadata."""
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
        return self.signal != Signal.HOLD and self.confidence >= 0.5


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Subclass and implement generate_signals() for your strategy logic.
    """

    def __init__(self, name: str, symbols: list[str], timeframe: str = "1Hour", lookback: int = 200):
        self.name = name
        self.symbols = symbols
        self.timeframe = timeframe
        self.lookback = lookback
        self._is_active = True

    @abstractmethod
    def generate_signals(self, data: dict[str, pd.DataFrame]) -> list[TradeSignal]:
        """
        Analyze market data and generate trading signals.

        Args:
            data: Dict mapping symbol -> DataFrame with OHLCV columns

        Returns:
            List of TradeSignal objects (one per symbol with a signal)
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

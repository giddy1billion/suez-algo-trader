"""
Abstract Broker Interface — Transport-agnostic broker protocol.

Defines the contract that all broker implementations must satisfy.
Execution logic programs against this interface, unaware of which
broker (Alpaca, IBKR, Binance, Paper, Replay) is underneath.
"""

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class BrokerCapabilities:
    """Declares what features a broker implementation supports.

    Execution logic should check capabilities before attempting operations
    to provide clear error messages instead of runtime failures.
    """

    supports_fractional: bool = False
    supports_shorting: bool = False
    supports_options: bool = False
    supports_crypto: bool = False
    supports_extended_hours: bool = False
    supports_bracket_orders: bool = False
    supports_notional_orders: bool = False
    supports_stop_limit: bool = False
    max_symbols: int = 0  # 0 = unlimited
    supported_timeframes: tuple[str, ...] = field(default_factory=lambda: ("1Min", "5Min", "15Min", "1Hour", "1Day"))


@runtime_checkable
class BrokerProtocol(Protocol):
    """Protocol defining the broker interface contract."""

    @property
    def paper(self) -> bool:
        """Whether this is a paper/simulated broker."""
        ...

    @property
    def name(self) -> str:
        """Broker identifier (e.g., 'alpaca', 'paper', 'ibkr')."""
        ...

    @property
    def capabilities(self) -> BrokerCapabilities:
        """Declare broker capabilities for feature negotiation."""
        ...

    # --- Account ---
    def get_account(self) -> dict:
        """Get account info: equity, buying_power, cash, etc."""
        ...

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        ...

    # --- Orders ---
    def market_order(self, symbol: str, qty: float, side: str,
                     time_in_force: str = "day",
                     client_order_id: Optional[str] = None) -> dict:
        """Submit a market order. Returns order dict with id, status, fill price.

        Args:
            client_order_id: Optional idempotency key. If provided, duplicate
                submissions with the same ID are rejected by the broker.
        """
        ...

    def market_order_notional(self, symbol: str, notional: float, side: str,
                              time_in_force: str = "gtc") -> dict:
        """Submit a market order by dollar amount (notional value)."""
        ...

    def limit_order(self, symbol: str, qty: float, side: str,
                    limit_price: float, time_in_force: str = "day") -> dict:
        """Submit a limit order."""
        ...

    def stop_limit_order(self, symbol: str, qty: float, side: str,
                         limit_price: float, stop_price: float,
                         time_in_force: str = "gtc") -> dict:
        """Submit a stop-limit order."""
        ...

    def bracket_order(self, symbol: str, qty: float, side: str,
                      take_profit: float, stop_loss: float,
                      time_in_force: str = "day") -> dict:
        """Submit a bracket order (entry + TP + SL)."""
        ...

    def cancel_order(self, order_id: str) -> Optional[dict]:
        """Cancel an order by ID."""
        ...

    def close_position(self, symbol: str, qty: Optional[float] = None) -> Optional[dict]:
        """Close a position (entirely, or partially if qty specified)."""
        ...

    def get_orders(self, status: str = "open", limit: int = 50) -> list[dict]:
        """Get orders filtered by status."""
        ...

    # --- Market Data ---
    def get_bars(self, symbol: str, timeframe: str = "1Hour",
                 limit: int = 200, start: Optional[datetime] = None) -> list[dict]:
        """Get historical bar data."""
        ...

    def get_bars_df(self, symbol: str, timeframe: str = "1Hour",
                    limit: int = 200) -> pd.DataFrame:
        """Get historical bars as a DataFrame with columns: open, high, low, close, volume."""
        ...

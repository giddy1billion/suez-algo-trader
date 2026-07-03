"""
Abstract Broker Interface — Transport-agnostic broker protocol.

Defines the contract that all broker implementations must satisfy.
Execution logic programs against this interface, unaware of which
broker (Alpaca, IBKR, Binance, Paper, Replay) is underneath.
"""

from typing import Optional, Protocol, runtime_checkable
from datetime import datetime

import pandas as pd


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

    # --- Account ---
    def get_account(self) -> dict:
        """Get account info: equity, buying_power, cash, etc."""
        ...

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        ...

    # --- Orders ---
    def market_order(self, symbol: str, qty: float, side: str,
                     time_in_force: str = "day") -> dict:
        """Submit a market order. Returns order dict with id, status, fill price."""
        ...

    def limit_order(self, symbol: str, qty: float, side: str,
                    limit_price: float, time_in_force: str = "day") -> dict:
        """Submit a limit order."""
        ...

    def bracket_order(self, symbol: str, qty: float, side: str,
                      take_profit: float, stop_loss: float,
                      time_in_force: str = "day") -> dict:
        """Submit a bracket order (entry + TP + SL)."""
        ...

    def cancel_order(self, order_id: str) -> Optional[dict]:
        """Cancel an order by ID."""
        ...

    def close_position(self, symbol: str) -> Optional[dict]:
        """Close an entire position for a symbol."""
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

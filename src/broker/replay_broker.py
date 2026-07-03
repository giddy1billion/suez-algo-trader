"""
Replay Broker — Feeds historical market data during replay sessions.

Used by the ReplayEngine to simulate broker responses using
previously recorded market data.
"""

import threading
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReplayBroker:
    """
    Broker that replays historical data for deterministic testing.
    Orders fill at the current bar's close price.
    Implements BrokerProtocol.
    """

    def __init__(self, historical_data: dict[str, pd.DataFrame],
                 starting_equity: float = 100_000.0):
        """
        Args:
            historical_data: Dict of symbol -> DataFrame with columns:
                             open, high, low, close, volume
            starting_equity: Starting account equity.
        """
        self._lock = threading.Lock()
        self._data = historical_data
        self._starting_equity = starting_equity
        self._cash = starting_equity
        self._positions: dict[str, dict] = {}
        self._orders: list[dict] = []
        self._bar_index: dict[str, int] = {s: 0 for s in historical_data}

        logger.info("replay_broker.initialized",
                    symbols=list(historical_data.keys()),
                    starting_equity=starting_equity)

    @property
    def paper(self) -> bool:
        """Replay broker is always simulated."""
        return True

    @property
    def name(self) -> str:
        """Broker identifier."""
        return "replay"

    def advance(self, symbol: str, bars: int = 1) -> None:
        """Advance the replay index for a symbol by N bars."""
        with self._lock:
            if symbol in self._bar_index and symbol in self._data:
                max_idx = len(self._data[symbol]) - 1
                self._bar_index[symbol] = min(
                    self._bar_index[symbol] + bars, max_idx
                )

    def current_price(self, symbol: str) -> Optional[float]:
        """Get the current bar's close price for a symbol."""
        with self._lock:
            return self._get_price(symbol)

    # --- Account ---

    def get_account(self) -> dict:
        """Get account info."""
        with self._lock:
            equity = self._cash + self._position_market_value()
            return {
                "equity": equity,
                "buying_power": self._cash,
                "cash": self._cash,
                "unrealized_pl": self._calc_unrealized_pl(),
                "starting_equity": self._starting_equity,
            }

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        with self._lock:
            result = []
            for symbol, pos in self._positions.items():
                current_price = self._get_price(symbol) or pos["avg_entry_price"]
                if pos["side"] == "buy":
                    unrealized_pl = (current_price - pos["avg_entry_price"]) * pos["qty"]
                else:
                    unrealized_pl = (pos["avg_entry_price"] - current_price) * pos["qty"]
                result.append({
                    "symbol": symbol,
                    "qty": pos["qty"],
                    "side": pos["side"],
                    "avg_entry_price": pos["avg_entry_price"],
                    "current_price": current_price,
                    "unrealized_pl": unrealized_pl,
                })
            return result

    # --- Orders ---

    def market_order(self, symbol: str, qty: float, side: str,
                     time_in_force: str = "day") -> dict:
        """Submit a market order. Fills at current bar's close."""
        with self._lock:
            price = self._get_price(symbol)
            if price is None:
                raise ValueError(f"No historical data for {symbol}")

            order = self._create_order(symbol, qty, side, "market",
                                       time_in_force=time_in_force)
            self._fill_order(order, price)
            return order

    def limit_order(self, symbol: str, qty: float, side: str,
                    limit_price: float, time_in_force: str = "day") -> dict:
        """Submit a limit order. Fills if current price satisfies limit."""
        with self._lock:
            order = self._create_order(symbol, qty, side, "limit",
                                       limit_price=limit_price,
                                       time_in_force=time_in_force)

            price = self._get_price(symbol)
            if price is not None:
                if self._should_fill_limit(side, limit_price, price):
                    self._fill_order(order, limit_price)

            return order

    def bracket_order(self, symbol: str, qty: float, side: str,
                      take_profit: float, stop_loss: float,
                      time_in_force: str = "day") -> dict:
        """Submit a bracket order. Entry fills at current bar's close."""
        with self._lock:
            price = self._get_price(symbol)
            if price is None:
                raise ValueError(f"No historical data for {symbol}")

            order = self._create_order(symbol, qty, side, "bracket",
                                       take_profit=take_profit,
                                       stop_loss=stop_loss,
                                       time_in_force=time_in_force)
            self._fill_order(order, price)
            return order

    def cancel_order(self, order_id: str) -> Optional[dict]:
        """Cancel a pending order."""
        with self._lock:
            for order in self._orders:
                if order["id"] == order_id and order["status"] == "pending":
                    order["status"] = "cancelled"
                    return order
            return None

    def close_position(self, symbol: str) -> Optional[dict]:
        """Close an entire position."""
        with self._lock:
            if symbol not in self._positions:
                return None

            pos = self._positions[symbol]
            price = self._get_price(symbol) or pos["avg_entry_price"]
            close_side = "sell" if pos["side"] == "buy" else "buy"
            order = self._create_order(symbol, pos["qty"], close_side, "market")
            self._fill_order(order, price)
            return order

    def get_orders(self, status: str = "open", limit: int = 50) -> list[dict]:
        """Get orders filtered by status."""
        with self._lock:
            if status == "open":
                filtered = [o for o in self._orders if o["status"] == "pending"]
            elif status == "closed":
                filtered = [o for o in self._orders if o["status"] in ("filled", "cancelled")]
            else:
                filtered = list(self._orders)
            return filtered[:limit]

    # --- Market Data ---

    def get_bars(self, symbol: str, timeframe: str = "1Hour",
                 limit: int = 200, start: Optional[datetime] = None) -> list[dict]:
        """Get historical bar data up to current replay index."""
        with self._lock:
            if symbol not in self._data:
                return []
            df = self._data[symbol]
            idx = self._bar_index.get(symbol, 0)
            subset = df.iloc[:idx + 1].tail(limit)
            return subset.to_dict("records")

    def get_bars_df(self, symbol: str, timeframe: str = "1Hour",
                    limit: int = 200) -> pd.DataFrame:
        """Get historical bars as DataFrame up to current replay index."""
        with self._lock:
            if symbol not in self._data:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            df = self._data[symbol]
            idx = self._bar_index.get(symbol, 0)
            return df.iloc[:idx + 1].tail(limit).reset_index(drop=True)

    # --- Internal helpers ---

    def _get_price(self, symbol: str) -> Optional[float]:
        """Get current close price from historical data."""
        if symbol not in self._data or symbol not in self._bar_index:
            return None
        df = self._data[symbol]
        idx = self._bar_index[symbol]
        if idx >= len(df):
            idx = len(df) - 1
        return float(df.iloc[idx]["close"])

    def _create_order(self, symbol: str, qty: float, side: str,
                      order_type: str, **kwargs) -> dict:
        """Create an order dict."""
        order = {
            "id": str(uuid.uuid4()),
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": order_type,
            "status": "pending",
            "filled_avg_price": None,
            "created_at": datetime.now().isoformat(),
            **kwargs,
        }
        self._orders.append(order)
        return order

    def _fill_order(self, order: dict, price: float) -> None:
        """Fill an order at the given price."""
        order["status"] = "filled"
        order["filled_avg_price"] = price

        symbol = order["symbol"]
        qty = order["qty"]
        side = order["side"]

        if symbol in self._positions:
            pos = self._positions[symbol]
            if pos["side"] == side:
                total_qty = pos["qty"] + qty
                pos["avg_entry_price"] = (
                    (pos["avg_entry_price"] * pos["qty"] + price * qty) / total_qty
                )
                pos["qty"] = total_qty
            else:
                if qty >= pos["qty"]:
                    closed_qty = pos["qty"]
                    if pos["side"] == "buy":
                        realized_pl = (price - pos["avg_entry_price"]) * closed_qty
                    else:
                        realized_pl = (pos["avg_entry_price"] - price) * closed_qty
                    self._cash += realized_pl + (pos["avg_entry_price"] * closed_qty)
                    del self._positions[symbol]

                    remaining = qty - closed_qty
                    if remaining > 0:
                        self._positions[symbol] = {
                            "qty": remaining,
                            "avg_entry_price": price,
                            "side": side,
                        }
                        self._cash -= price * remaining
                else:
                    if pos["side"] == "buy":
                        realized_pl = (price - pos["avg_entry_price"]) * qty
                    else:
                        realized_pl = (pos["avg_entry_price"] - price) * qty
                    self._cash += realized_pl + (pos["avg_entry_price"] * qty)
                    pos["qty"] -= qty
        else:
            self._positions[symbol] = {
                "qty": qty,
                "avg_entry_price": price,
                "side": side,
            }
            self._cash -= price * qty

    def _should_fill_limit(self, side: str, limit_price: float,
                           current_price: float) -> bool:
        """Check if limit order should fill."""
        if side == "buy":
            return current_price <= limit_price
        else:
            return current_price >= limit_price

    def _calc_unrealized_pl(self) -> float:
        """Calculate total unrealized P&L."""
        total = 0.0
        for symbol, pos in self._positions.items():
            current_price = self._get_price(symbol) or pos["avg_entry_price"]
            if pos["side"] == "buy":
                total += (current_price - pos["avg_entry_price"]) * pos["qty"]
            else:
                total += (pos["avg_entry_price"] - current_price) * pos["qty"]
        return total

    def _position_market_value(self) -> float:
        """Calculate total market value of positions."""
        total = 0.0
        for symbol, pos in self._positions.items():
            current_price = self._get_price(symbol) or pos["avg_entry_price"]
            total += current_price * pos["qty"]
        return total

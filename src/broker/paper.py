"""
Paper Trading Broker — Fully self-contained simulated broker.

Maintains internal state for positions, orders, and account without
any external API dependency. Useful for:
- Development without API keys
- Backtesting
- Integration testing
- Replay-based simulation
"""

import threading
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PaperBroker:
    """
    Simulated broker that tracks positions, orders, and account state
    entirely in-memory. Implements BrokerProtocol.
    """

    def __init__(self, starting_equity: float = 100_000.0):
        self._lock = threading.Lock()
        self._starting_equity = starting_equity
        self._cash = starting_equity
        self._positions: dict[str, dict] = {}  # symbol -> {qty, avg_entry_price, side}
        self._orders: list[dict] = []
        self._prices: dict[str, float] = {}  # last known prices

        logger.info("paper_broker.initialized", starting_equity=starting_equity)

    @property
    def paper(self) -> bool:
        """Always True — this is a paper broker."""
        return True

    @property
    def name(self) -> str:
        """Broker identifier."""
        return "paper"

    def set_price(self, symbol: str, price: float) -> None:
        """Set the current price for a symbol (for test/simulation use)."""
        with self._lock:
            self._prices[symbol] = price
            self._try_fill_pending_orders(symbol, price)

    # --- Account ---

    def get_account(self) -> dict:
        """Get account info including unrealized P&L."""
        with self._lock:
            unrealized_pl = self._calc_unrealized_pl()
            equity = self._cash + self._position_market_value()
            return {
                "equity": equity,
                "buying_power": self._cash,
                "cash": self._cash,
                "unrealized_pl": unrealized_pl,
                "starting_equity": self._starting_equity,
            }

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        with self._lock:
            result = []
            for symbol, pos in self._positions.items():
                current_price = self._prices.get(symbol, pos["avg_entry_price"])
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
        """Submit a market order. Fills instantly at last known price."""
        with self._lock:
            price = self._prices.get(symbol)
            if price is None:
                raise ValueError(f"No price set for {symbol}. Use set_price() first.")

            order = self._create_order(symbol, qty, side, "market",
                                       time_in_force=time_in_force)
            self._fill_order(order, price)
            logger.info("paper_broker.market_order_filled",
                        symbol=symbol, qty=qty, side=side, price=price)
            return order

    def limit_order(self, symbol: str, qty: float, side: str,
                    limit_price: float, time_in_force: str = "day") -> dict:
        """Submit a limit order. Fills when price crosses limit."""
        with self._lock:
            order = self._create_order(symbol, qty, side, "limit",
                                       limit_price=limit_price,
                                       time_in_force=time_in_force)

            # Check if it can fill immediately
            current_price = self._prices.get(symbol)
            if current_price is not None:
                if self._should_fill_limit(side, limit_price, current_price):
                    self._fill_order(order, limit_price)
                    logger.info("paper_broker.limit_order_filled_immediately",
                                symbol=symbol, price=limit_price)

            return order

    def bracket_order(self, symbol: str, qty: float, side: str,
                      take_profit: float, stop_loss: float,
                      time_in_force: str = "day") -> dict:
        """Submit a bracket order (entry + TP + SL). Entry fills at market."""
        with self._lock:
            price = self._prices.get(symbol)
            if price is None:
                raise ValueError(f"No price set for {symbol}. Use set_price() first.")

            order = self._create_order(symbol, qty, side, "bracket",
                                       take_profit=take_profit,
                                       stop_loss=stop_loss,
                                       time_in_force=time_in_force)
            self._fill_order(order, price)
            logger.info("paper_broker.bracket_order_filled",
                        symbol=symbol, qty=qty, side=side, price=price)
            return order

    def market_order_notional(self, symbol: str, notional: float, side: str,
                              time_in_force: str = "gtc") -> dict:
        """Submit a market order by dollar amount. Converts to qty internally."""
        with self._lock:
            price = self._prices.get(symbol)
            if price is None:
                raise ValueError(f"No price set for {symbol}. Use set_price() first.")
            qty = notional / price
            order = self._create_order(symbol, qty, side, "market",
                                       time_in_force=time_in_force)
            self._fill_order(order, price)
            order["notional"] = notional
            logger.info("paper_broker.market_notional_filled",
                        symbol=symbol, notional=notional, qty=qty, side=side, price=price)
            return order

    def stop_limit_order(self, symbol: str, qty: float, side: str,
                         limit_price: float, stop_price: float,
                         time_in_force: str = "gtc") -> dict:
        """Submit a stop-limit order. Fills when price hits stop, at limit."""
        with self._lock:
            order = self._create_order(symbol, qty, side, "stop_limit",
                                       limit_price=limit_price,
                                       stop_price=stop_price,
                                       time_in_force=time_in_force)
            # Check immediate fill possibility
            current_price = self._prices.get(symbol)
            if current_price is not None:
                triggered = (side.lower() == "buy" and current_price >= stop_price) or \
                           (side.lower() == "sell" and current_price <= stop_price)
                if triggered and self._should_fill_limit(side, limit_price, current_price):
                    self._fill_order(order, limit_price)
                    logger.info("paper_broker.stop_limit_filled_immediately",
                                symbol=symbol, price=limit_price)
            return order

    def cancel_order(self, order_id: str) -> Optional[dict]:
        """Cancel a pending order by ID."""
        with self._lock:
            for order in self._orders:
                if order["id"] == order_id and order["status"] == "pending":
                    order["status"] = "cancelled"
                    logger.info("paper_broker.order_cancelled", order_id=order_id)
                    return order
            return None

    def close_position(self, symbol: str, qty: Optional[float] = None) -> Optional[dict]:
        """Close a position (entirely or partially by specifying qty)."""
        with self._lock:
            # Normalize crypto symbols (BTC/USD → BTCUSD for lookup)
            lookup = symbol.replace("/", "")
            # Try both forms
            actual_symbol = symbol if symbol in self._positions else lookup

            if actual_symbol not in self._positions:
                return None

            pos = self._positions[actual_symbol]
            original_qty = pos["qty"]
            price = self._prices.get(actual_symbol, self._prices.get(symbol, pos["avg_entry_price"]))

            close_qty = qty if qty is not None else original_qty
            close_side = "sell" if pos["side"] == "buy" else "buy"
            order = self._create_order(actual_symbol, close_qty, close_side, "market")
            self._fill_order(order, price)

            if qty is not None and qty < original_qty:
                logger.info("paper_broker.position_partial_close",
                           symbol=actual_symbol, qty=close_qty, price=price)
            else:
                logger.info("paper_broker.position_closed", symbol=actual_symbol, price=price)
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
        """Get historical bar data. Returns empty list (no real data)."""
        return []

    def get_bars_df(self, symbol: str, timeframe: str = "1Hour",
                    limit: int = 200) -> pd.DataFrame:
        """Get historical bars as DataFrame. Returns empty DataFrame."""
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # --- Internal helpers ---

    def _create_order(self, symbol: str, qty: float, side: str, order_type: str,
                      **kwargs) -> dict:
        """Create an order dict and add to history."""
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
        """Fill an order at the given price, updating positions and cash."""
        order["status"] = "filled"
        order["filled_avg_price"] = price

        symbol = order["symbol"]
        qty = order["qty"]
        side = order["side"]

        if symbol in self._positions:
            pos = self._positions[symbol]
            if pos["side"] == side:
                # Adding to position
                total_qty = pos["qty"] + qty
                pos["avg_entry_price"] = (
                    (pos["avg_entry_price"] * pos["qty"] + price * qty) / total_qty
                )
                pos["qty"] = total_qty
            else:
                # Reducing/closing position
                if qty >= pos["qty"]:
                    # Close position, realize P&L
                    closed_qty = pos["qty"]
                    if pos["side"] == "buy":
                        realized_pl = (price - pos["avg_entry_price"]) * closed_qty
                    else:
                        realized_pl = (pos["avg_entry_price"] - price) * closed_qty
                    self._cash += realized_pl + (pos["avg_entry_price"] * closed_qty)
                    del self._positions[symbol]

                    # If qty > pos qty, open reverse position
                    remaining = qty - closed_qty
                    if remaining > 0:
                        self._positions[symbol] = {
                            "qty": remaining,
                            "avg_entry_price": price,
                            "side": side,
                        }
                        self._cash -= price * remaining
                else:
                    # Partial close
                    if pos["side"] == "buy":
                        realized_pl = (price - pos["avg_entry_price"]) * qty
                    else:
                        realized_pl = (pos["avg_entry_price"] - price) * qty
                    self._cash += realized_pl + (pos["avg_entry_price"] * qty)
                    pos["qty"] -= qty
        else:
            # New position
            self._positions[symbol] = {
                "qty": qty,
                "avg_entry_price": price,
                "side": side,
            }
            self._cash -= price * qty

    def _should_fill_limit(self, side: str, limit_price: float,
                           current_price: float) -> bool:
        """Check if a limit order should fill at current price."""
        if side == "buy":
            return current_price <= limit_price
        else:
            return current_price >= limit_price

    def _try_fill_pending_orders(self, symbol: str, price: float) -> None:
        """Try to fill any pending limit orders for a symbol."""
        for order in self._orders:
            if (order["status"] == "pending" and
                    order["symbol"] == symbol and
                    order["type"] == "limit"):
                limit_price = order.get("limit_price")
                if limit_price and self._should_fill_limit(order["side"], limit_price, price):
                    self._fill_order(order, limit_price)
                    logger.info("paper_broker.limit_order_filled",
                                order_id=order["id"], price=limit_price)

    def _calc_unrealized_pl(self) -> float:
        """Calculate total unrealized P&L across all positions."""
        total = 0.0
        for symbol, pos in self._positions.items():
            current_price = self._prices.get(symbol, pos["avg_entry_price"])
            if pos["side"] == "buy":
                total += (current_price - pos["avg_entry_price"]) * pos["qty"]
            else:
                total += (pos["avg_entry_price"] - current_price) * pos["qty"]
        return total

    def _position_market_value(self) -> float:
        """Calculate total market value of positions."""
        total = 0.0
        for symbol, pos in self._positions.items():
            current_price = self._prices.get(symbol, pos["avg_entry_price"])
            total += current_price * pos["qty"]
        return total

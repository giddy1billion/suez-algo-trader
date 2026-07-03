"""
Alpaca broker client — handles REST + WebSocket connections.
Supports both Paper and Live trading modes.
Auto-switches based on TRADING_MODE in configuration.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderStatus, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    CryptoBarsRequest,
    CryptoLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.live import StockDataStream, CryptoDataStream

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Map string timeframes to Alpaca TimeFrame objects
TIMEFRAME_MAP = {
    "1Min": TimeFrame(1, TimeFrameUnit.Minute),
    "5Min": TimeFrame(5, TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "30Min": TimeFrame(30, TimeFrameUnit.Minute),
    "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
    "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
    "1Day": TimeFrame(1, TimeFrameUnit.Day),
    "1Week": TimeFrame(1, TimeFrameUnit.Week),
}


class AlpacaBroker:
    """
    Unified Alpaca broker interface for both stocks and crypto.
    Handles authentication, order execution, position management,
    and real-time data streaming.
    """

    def __init__(self, api_key: str, secret_key: str, base_url: str, data_feed: str = "iex", paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.paper = paper
        self.data_feed = data_feed

        # Trading client
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )

        # Data clients
        self.stock_data = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        self.crypto_data = CryptoHistoricalDataClient(api_key=api_key, secret_key=secret_key)

        # Streaming clients (initialized on demand)
        self._stock_stream: Optional[StockDataStream] = None
        self._crypto_stream: Optional[CryptoDataStream] = None

        logger.info("broker.initialized", paper=paper, data_feed=data_feed)

    # ──────────────────────────────────────────────────────────────────────
    # Account & Portfolio
    # ──────────────────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        """Get account details (balance, buying power, etc.)."""
        account = self.trading_client.get_account()
        return {
            "id": account.id,
            "status": account.status.value if account.status else None,
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "equity": float(account.equity),
            "last_equity": float(account.last_equity),
            "long_market_value": float(account.long_market_value),
            "short_market_value": float(account.short_market_value),
            "day_trade_count": account.daytrade_count,
            "pattern_day_trader": account.pattern_day_trader,
        }

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        positions = self.trading_client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side.value if p.side else "long",
                "avg_entry_price": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "current_price": float(p.current_price),
            }
            for p in positions
        ]

    def get_position(self, symbol: str) -> Optional[dict]:
        """Get position for a specific symbol (None if no position)."""
        try:
            p = self.trading_client.get_open_position(symbol)
            return {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side.value if p.side else "long",
                "avg_entry_price": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "current_price": float(p.current_price),
            }
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────────
    # Order Execution
    # ──────────────────────────────────────────────────────────────────────

    def market_order(self, symbol: str, qty: float, side: str, time_in_force: str = "day") -> dict:
        """Place a market order."""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=tif,
        )
        order = self.trading_client.submit_order(request)
        logger.info("order.submitted", symbol=symbol, side=side, qty=qty, type="market", order_id=str(order.id))
        return self._order_to_dict(order)

    def limit_order(self, symbol: str, qty: float, side: str, limit_price: float, time_in_force: str = "day") -> dict:
        """Place a limit order."""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC

        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=tif,
            limit_price=limit_price,
        )
        order = self.trading_client.submit_order(request)
        logger.info("order.submitted", symbol=symbol, side=side, qty=qty, type="limit", price=limit_price)
        return self._order_to_dict(order)

    def bracket_order(
        self, symbol: str, qty: float, side: str,
        stop_loss_price: float, take_profit_price: float,
        time_in_force: str = "day"
    ) -> dict:
        """Place a bracket order (entry + stop-loss + take-profit)."""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=tif,
            order_class="bracket",
            stop_loss=StopLossRequest(stop_price=stop_loss_price),
            take_profit=TakeProfitRequest(limit_price=take_profit_price),
        )
        order = self.trading_client.submit_order(request)
        logger.info("order.bracket", symbol=symbol, side=side, qty=qty, sl=stop_loss_price, tp=take_profit_price)
        return self._order_to_dict(order)

    def trailing_stop_order(self, symbol: str, qty: float, side: str, trail_percent: float) -> dict:
        """Place a trailing stop order."""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        request = TrailingStopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.GTC,
            trail_percent=trail_percent,
        )
        order = self.trading_client.submit_order(request)
        logger.info("order.trailing_stop", symbol=symbol, side=side, trail_pct=trail_percent)
        return self._order_to_dict(order)

    def cancel_order(self, order_id: str):
        """Cancel a pending order."""
        self.trading_client.cancel_order_by_id(order_id)
        logger.info("order.cancelled", order_id=order_id)

    def cancel_all_orders(self):
        """Cancel all open orders."""
        self.trading_client.cancel_orders()
        logger.warning("orders.all_cancelled")

    def close_position(self, symbol: str):
        """Close an entire position."""
        self.trading_client.close_position(symbol)
        logger.info("position.closed", symbol=symbol)

    def close_all_positions(self):
        """Liquidate all positions."""
        self.trading_client.close_all_positions(cancel_orders=True)
        logger.warning("positions.all_liquidated")

    def get_orders(self, status: str = "open", limit: int = 50) -> list[dict]:
        """Get orders by status."""
        query_status = QueryOrderStatus.OPEN if status == "open" else QueryOrderStatus.ALL
        request = GetOrdersRequest(status=query_status, limit=limit)
        orders = self.trading_client.get_orders(request)
        return [self._order_to_dict(o) for o in orders]

    # ──────────────────────────────────────────────────────────────────────
    # Market Data
    # ──────────────────────────────────────────────────────────────────────

    def get_bars(self, symbol: str, timeframe: str = "1Hour", limit: int = 200, start: datetime = None):
        """Fetch historical bars (OHLCV) for stocks or crypto."""
        tf = TIMEFRAME_MAP.get(timeframe, TimeFrame(1, TimeFrameUnit.Hour))

        if not start:
            start = datetime.now() - timedelta(days=max(limit * 2, 30))

        is_crypto = "/" in symbol  # BTC/USD, ETH/USD

        if is_crypto:
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                limit=limit,
            )
            bars = self.crypto_data.get_crypto_bars(request)
        else:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                limit=limit,
                feed=self.data_feed,
            )
            bars = self.stock_data.get_stock_bars(request)

        return bars[symbol] if symbol in bars else bars.df

    def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        is_crypto = "/" in symbol

        if is_crypto:
            request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.crypto_data.get_crypto_latest_quote(request)
            return float(quote[symbol].ask_price)
        else:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self.data_feed)
            quote = self.stock_data.get_stock_latest_quote(request)
            return float(quote[symbol].ask_price)

    def get_bars_df(self, symbol: str, timeframe: str = "1Hour", limit: int = 200):
        """Fetch bars and return as a pandas DataFrame."""
        import pandas as pd

        bars = self.get_bars(symbol, timeframe, limit)

        if hasattr(bars, 'df'):
            return bars.df

        records = []
        for bar in bars:
            records.append({
                "timestamp": bar.timestamp,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            })
        return pd.DataFrame(records).set_index("timestamp")

    # ──────────────────────────────────────────────────────────────────────
    # Real-time WebSocket Streaming
    # ──────────────────────────────────────────────────────────────────────

    def get_stock_stream(self) -> StockDataStream:
        """Get or create a stock data WebSocket stream."""
        if self._stock_stream is None:
            self._stock_stream = StockDataStream(
                api_key=self.api_key,
                secret_key=self.secret_key,
                feed=self.data_feed,
            )
        return self._stock_stream

    def get_crypto_stream(self) -> CryptoDataStream:
        """Get or create a crypto data WebSocket stream."""
        if self._crypto_stream is None:
            self._crypto_stream = CryptoDataStream(
                api_key=self.api_key,
                secret_key=self.secret_key,
            )
        return self._crypto_stream

    async def stream_bars(self, symbols: list[str], handler):
        """
        Stream real-time bar data for given symbols.
        Handler receives bar updates as they arrive.
        """
        stock_symbols = [s for s in symbols if "/" not in s]
        crypto_symbols = [s for s in symbols if "/" in s]

        tasks = []

        if stock_symbols:
            stream = self.get_stock_stream()
            stream.subscribe_bars(handler, *stock_symbols)
            tasks.append(stream._run_forever())

        if crypto_symbols:
            stream = self.get_crypto_stream()
            stream.subscribe_bars(handler, *crypto_symbols)
            tasks.append(stream._run_forever())

        if tasks:
            await asyncio.gather(*tasks)

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        """Check if the stock market is currently open."""
        clock = self.trading_client.get_clock()
        return clock.is_open

    def next_market_open(self) -> datetime:
        """Get when the market next opens."""
        clock = self.trading_client.get_clock()
        return clock.next_open

    def next_market_close(self) -> datetime:
        """Get when the market next closes."""
        clock = self.trading_client.get_clock()
        return clock.next_close

    def _order_to_dict(self, order) -> dict:
        """Convert an Alpaca order object to a clean dict."""
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "side": order.side.value if order.side else None,
            "type": order.type.value if order.type else None,
            "qty": float(order.qty) if order.qty else None,
            "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "stop_price": float(order.stop_price) if order.stop_price else None,
            "status": order.status.value if order.status else None,
            "created_at": str(order.created_at) if order.created_at else None,
            "filled_at": str(order.filled_at) if order.filled_at else None,
        }

"""
Alpaca broker client — handles REST + WebSocket connections.
Supports both Paper and Live trading modes.
Auto-switches based on TRADING_MODE in configuration.
"""

import asyncio
import functools
import random
import time
import threading
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


# ──────────────────────────────────────────────────────────────────────
# Resilience utilities
# ──────────────────────────────────────────────────────────────────────

class _RateLimiter:
    """Token-bucket rate limiter: max_requests per window (seconds)."""

    def __init__(self, max_requests: int = 200, window: float = 60.0):
        self._max = max_requests
        self._window = window
        self._tokens = max_requests
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            refill = elapsed * (self._max / self._window)
            self._tokens = min(self._max, self._tokens + refill)
            self._last_refill = now

            if self._tokens < 1:
                wait = (1 - self._tokens) * (self._window / self._max)
                time.sleep(wait)
                self._tokens = 0
            else:
                self._tokens -= 1


def _is_retryable(exc: Exception) -> bool:
    """Determine if an exception warrants a retry."""
    from requests.exceptions import (
        ConnectionError, Timeout, ReadTimeout, ConnectTimeout
    )

    # Network/timeout errors — always retry
    if isinstance(exc, (ConnectionError, Timeout, ReadTimeout, ConnectTimeout, OSError)):
        return True

    # HTTP status-based errors from Alpaca SDK
    status = getattr(exc, 'status_code', None) or getattr(exc, 'code', None)
    if status is None:
        # Try extracting from response attribute
        resp = getattr(exc, 'response', None)
        if resp is not None:
            status = getattr(resp, 'status_code', None)

    if status is not None:
        if status == 429:
            return True
        if status >= 500:
            return True
        # Non-retryable client errors
        if status in (401, 403, 422):
            return False

    # Unknown errors — retry to be safe
    return True


def _retry(max_retries: int = 3, base_delay: float = 1.0):
    """Retry decorator with exponential backoff + jitter."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt == max_retries or not _is_retryable(exc):
                        raise
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning(
                        "api.retry",
                        method=fn.__name__,
                        attempt=attempt + 1,
                        delay=round(delay, 2),
                        error=str(exc),
                    )
                    time.sleep(delay)
            raise last_exc  # pragma: no cover
        return wrapper
    return decorator


def _error_dict(exc: Exception) -> dict:
    """Build a caller-friendly error response dict."""
    return {
        "error": True,
        "message": str(exc),
        "retryable": _is_retryable(exc),
    }


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

    Includes built-in retry logic, rate limiting, and error handling.
    """

    def __init__(self, api_key: str, secret_key: str, base_url: str, data_feed: str = "iex",
                 paper: bool = True, timeout: float = 30.0, rate_limit: int = 200):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.paper = paper
        self.data_feed = data_feed
        self.timeout = timeout

        # Rate limiter (requests/minute)
        self._rate_limiter = _RateLimiter(max_requests=rate_limit, window=60.0)

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

        logger.info("broker.initialized", paper=paper, data_feed=data_feed, timeout=timeout)

    def _call(self, fn, *args, **kwargs):
        """Execute an API call with rate limiting, retry, and error handling."""
        self._rate_limiter.acquire()
        return fn(*args, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    # Account & Portfolio
    # ──────────────────────────────────────────────────────────────────────

    @_retry()
    def get_account(self) -> dict:
        """Get account details (balance, buying power, etc.)."""
        try:
            account = self._call(self.trading_client.get_account)
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
        except Exception as exc:
            logger.error("account.get_failed", error=str(exc))
            raise

    @_retry()
    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        try:
            positions = self._call(self.trading_client.get_all_positions)
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
        except Exception as exc:
            logger.error("positions.get_failed", error=str(exc))
            raise

    @_retry()
    def get_position(self, symbol: str) -> Optional[dict]:
        """Get position for a specific symbol (None if no position)."""
        try:
            p = self._call(self.trading_client.get_open_position, symbol)
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

    @_retry()
    def market_order(self, symbol: str, qty: float, side: str, time_in_force: str = "day") -> dict:
        """Place a market order."""
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC

            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
            )
            order = self._call(self.trading_client.submit_order, request)
            logger.info("order.submitted", symbol=symbol, side=side, qty=qty, type="market", order_id=str(order.id))
            return self._order_to_dict(order)
        except Exception as exc:
            logger.error("order.market_failed", symbol=symbol, error=str(exc))
            if not _is_retryable(exc):
                return _error_dict(exc)
            raise

    @_retry()
    def limit_order(self, symbol: str, qty: float, side: str, limit_price: float, time_in_force: str = "day") -> dict:
        """Place a limit order."""
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC

            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
                limit_price=limit_price,
            )
            order = self._call(self.trading_client.submit_order, request)
            logger.info("order.submitted", symbol=symbol, side=side, qty=qty, type="limit", price=limit_price)
            return self._order_to_dict(order)
        except Exception as exc:
            logger.error("order.limit_failed", symbol=symbol, error=str(exc))
            if not _is_retryable(exc):
                return _error_dict(exc)
            raise

    @_retry()
    def bracket_order(
        self, symbol: str, qty: float, side: str,
        stop_loss_price: float, take_profit_price: float,
        time_in_force: str = "day"
    ) -> dict:
        """Place a bracket order (entry + stop-loss + take-profit)."""
        try:
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
            order = self._call(self.trading_client.submit_order, request)
            logger.info("order.bracket", symbol=symbol, side=side, qty=qty, sl=stop_loss_price, tp=take_profit_price)
            return self._order_to_dict(order)
        except Exception as exc:
            logger.error("order.bracket_failed", symbol=symbol, error=str(exc))
            if not _is_retryable(exc):
                return _error_dict(exc)
            raise

    @_retry()
    def trailing_stop_order(self, symbol: str, qty: float, side: str, trail_percent: float) -> dict:
        """Place a trailing stop order."""
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            request = TrailingStopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC,
                trail_percent=trail_percent,
            )
            order = self._call(self.trading_client.submit_order, request)
            logger.info("order.trailing_stop", symbol=symbol, side=side, trail_pct=trail_percent)
            return self._order_to_dict(order)
        except Exception as exc:
            logger.error("order.trailing_stop_failed", symbol=symbol, error=str(exc))
            if not _is_retryable(exc):
                return _error_dict(exc)
            raise

    @_retry()
    def cancel_order(self, order_id: str) -> Optional[dict]:
        """Cancel a pending order."""
        try:
            self._call(self.trading_client.cancel_order_by_id, order_id)
            logger.info("order.cancelled", order_id=order_id)
            return None
        except Exception as exc:
            logger.error("order.cancel_failed", order_id=order_id, error=str(exc))
            if not _is_retryable(exc):
                return _error_dict(exc)
            raise

    @_retry()
    def cancel_all_orders(self) -> Optional[dict]:
        """Cancel all open orders."""
        try:
            self._call(self.trading_client.cancel_orders)
            logger.warning("orders.all_cancelled")
            return None
        except Exception as exc:
            logger.error("orders.cancel_all_failed", error=str(exc))
            if not _is_retryable(exc):
                return _error_dict(exc)
            raise

    @_retry()
    def close_position(self, symbol: str) -> Optional[dict]:
        """Close an entire position."""
        try:
            self._call(self.trading_client.close_position, symbol)
            logger.info("position.closed", symbol=symbol)
            return None
        except Exception as exc:
            logger.error("position.close_failed", symbol=symbol, error=str(exc))
            if not _is_retryable(exc):
                return _error_dict(exc)
            raise

    @_retry()
    def close_all_positions(self) -> Optional[dict]:
        """Liquidate all positions."""
        try:
            self._call(self.trading_client.close_all_positions, cancel_orders=True)
            logger.warning("positions.all_liquidated")
            return None
        except Exception as exc:
            logger.error("positions.close_all_failed", error=str(exc))
            if not _is_retryable(exc):
                return _error_dict(exc)
            raise

    @_retry()
    def get_orders(self, status: str = "open", limit: int = 50) -> list[dict]:
        """Get orders by status."""
        try:
            query_status = QueryOrderStatus.OPEN if status == "open" else QueryOrderStatus.ALL
            request = GetOrdersRequest(status=query_status, limit=limit)
            orders = self._call(self.trading_client.get_orders, request)
            return [self._order_to_dict(o) for o in orders]
        except Exception as exc:
            logger.error("orders.get_failed", error=str(exc))
            raise

    # ──────────────────────────────────────────────────────────────────────
    # Market Data
    # ──────────────────────────────────────────────────────────────────────

    @_retry()
    def get_bars(self, symbol: str, timeframe: str = "1Hour", limit: int = 200, start: datetime = None):
        """Fetch historical bars (OHLCV) for stocks or crypto."""
        try:
            tf = TIMEFRAME_MAP.get(timeframe, TimeFrame(1, TimeFrameUnit.Hour))

            if not start:
                bars_per_day = {
                    "1Min": 390, "5Min": 78, "15Min": 26,
                    "30Min": 13, "1Hour": 7, "4Hour": 2,
                    "1Day": 1, "1Week": 0.2,
                }
                bpd = bars_per_day.get(timeframe, 7)
                days_needed = int((limit / max(bpd, 0.1)) * 1.5) + 5
                start = datetime.now() - timedelta(days=max(days_needed, 7))

            is_crypto = "/" in symbol

            if is_crypto:
                request = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start,
                    limit=limit,
                )
                bars = self._call(self.crypto_data.get_crypto_bars, request)
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start,
                    limit=limit,
                    feed=self.data_feed,
                )
                bars = self._call(self.stock_data.get_stock_bars, request)

            if symbol in bars:
                return bars[symbol]
            if hasattr(bars, 'data') and symbol in bars.data:
                return bars.data[symbol]
            return []
        except Exception as exc:
            logger.error("bars.get_failed", symbol=symbol, error=str(exc))
            raise

    @_retry()
    def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        try:
            is_crypto = "/" in symbol

            if is_crypto:
                request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
                quote = self._call(self.crypto_data.get_crypto_latest_quote, request)
                return float(quote[symbol].ask_price)
            else:
                request = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self.data_feed)
                quote = self._call(self.stock_data.get_stock_latest_quote, request)
                return float(quote[symbol].ask_price)
        except Exception as exc:
            logger.error("price.get_failed", symbol=symbol, error=str(exc))
            raise

    @_retry()
    def get_bars_df(self, symbol: str, timeframe: str = "1Hour", limit: int = 200):
        """Fetch bars and return as a pandas DataFrame."""
        import pandas as pd

        try:
            bars = self.get_bars(symbol, timeframe, limit)

            if not bars:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

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
            df = pd.DataFrame(records).set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)
            return df
        except Exception as exc:
            logger.error("bars_df.get_failed", symbol=symbol, error=str(exc))
            raise

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

    @_retry()
    def is_market_open(self) -> bool:
        """Check if the stock market is currently open."""
        try:
            clock = self._call(self.trading_client.get_clock)
            return clock.is_open
        except Exception as exc:
            logger.error("clock.failed", error=str(exc))
            raise

    @_retry()
    def next_market_open(self) -> datetime:
        """Get when the market next opens."""
        try:
            clock = self._call(self.trading_client.get_clock)
            return clock.next_open
        except Exception as exc:
            logger.error("clock.failed", error=str(exc))
            raise

    @_retry()
    def next_market_close(self) -> datetime:
        """Get when the market next closes."""
        try:
            clock = self._call(self.trading_client.get_clock)
            return clock.next_close
        except Exception as exc:
            logger.error("clock.failed", error=str(exc))
            raise

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

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
    StopLimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
    GetOrdersRequest,
    GetAssetsRequest,
    ClosePositionRequest,
)
from alpaca.trading.enums import (
    OrderSide, TimeInForce, OrderType, OrderStatus, QueryOrderStatus,
    AssetClass, AssetStatus,
)
from alpaca.trading.stream import TradingStream
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    CryptoBarsRequest,
    CryptoLatestQuoteRequest,
)
from alpaca.data.enums import DataFeed
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

    # Programming errors — never retry (fail fast for debugging)
    if isinstance(exc, (TypeError, AttributeError, KeyError, ValueError, IndexError)):
        return False

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

    # Unknown errors — retry to be safe (likely network-related)
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
                    retryable = _is_retryable(exc)
                    if attempt == max_retries or not retryable:
                        # Emit explicit exhaustion/non-retryable signal
                        if attempt == max_retries and retryable:
                            logger.error(
                                "api.retries_exhausted",
                                method=fn.__name__,
                                attempts=max_retries + 1,
                                error=str(exc),
                                retryable=True,
                            )
                        elif not retryable:
                            logger.error(
                                "api.non_retryable_failure",
                                method=fn.__name__,
                                attempt=attempt + 1,
                                error=str(exc),
                                retryable=False,
                            )
                        raise
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning(
                        "api.retry",
                        method=fn.__name__,
                        attempt=attempt + 1,
                        max_retries=max_retries,
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

        # Trade update stream (initialized on demand)
        self._trade_stream: Optional[TradingStream] = None
        self._trade_stream_thread: Optional[threading.Thread] = None
        self._trade_stream_handler = None
        self._shutdown_flag: bool = False

        logger.info("broker.initialized", paper=paper, data_feed=data_feed, timeout=timeout)

    @property
    def name(self) -> str:
        return "alpaca"

    @staticmethod
    def _normalize_symbol_for_position(symbol: str) -> str:
        """Normalize crypto symbols for position API calls (BTC/USD → BTCUSD)."""
        return symbol.replace("/", "")

    @staticmethod
    def _parse_time_in_force(tif: str) -> TimeInForce:
        """Parse time-in-force string to Alpaca enum."""
        tif_lower = tif.lower()
        if tif_lower == "gtc":
            return TimeInForce.GTC
        elif tif_lower == "ioc":
            return TimeInForce.IOC
        elif tif_lower == "day":
            return TimeInForce.DAY
        return TimeInForce.DAY

    def _call(self, fn, *args, **kwargs):
        """Execute an API call with rate limiting, timeout enforcement, and error handling."""
        import concurrent.futures
        self._rate_limiter.acquire()
        # P1-05: Enforce timeout on all broker API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn, *args, **kwargs)
            try:
                return future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"Broker API call {fn.__name__} timed out after {self.timeout}s"
                )

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
            normalized = self._normalize_symbol_for_position(symbol)
            p = self._call(self.trading_client.get_open_position, normalized)
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
    def market_order(self, symbol: str, qty: float, side: str, time_in_force: str = "day",
                     client_order_id: str = None) -> dict:
        """Place a market order."""
        import uuid
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            tif = self._parse_time_in_force(time_in_force)

            # Generate idempotency key if not provided
            if client_order_id is None:
                client_order_id = f"suez-{uuid.uuid4().hex[:16]}"

            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
                client_order_id=client_order_id,
            )
            order = self._call(self.trading_client.submit_order, request)
            logger.info("order.submitted", symbol=symbol, side=side, qty=qty, type="market",
                       order_id=str(order.id), client_order_id=client_order_id)
            return self._order_to_dict(order)
        except Exception as exc:
            logger.error("order.market_failed", symbol=symbol, error=str(exc))
            if not _is_retryable(exc):
                return _error_dict(exc)
            raise

    @_retry()
    def market_order_notional(self, symbol: str, notional: float, side: str, time_in_force: str = "gtc") -> dict:
        """Place a market order by dollar amount (notional value)."""
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            tif = self._parse_time_in_force(time_in_force)

            request = MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=order_side,
                time_in_force=tif,
            )
            order = self._call(self.trading_client.submit_order, request)
            logger.info("order.submitted", symbol=symbol, side=side, notional=notional, type="market_notional", order_id=str(order.id))
            return self._order_to_dict(order)
        except Exception as exc:
            logger.error("order.market_notional_failed", symbol=symbol, error=str(exc))
            if not _is_retryable(exc):
                return _error_dict(exc)
            raise

    @_retry()
    def limit_order(self, symbol: str, qty: float, side: str, limit_price: float, time_in_force: str = "day") -> dict:
        """Place a limit order."""
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            tif = self._parse_time_in_force(time_in_force)

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
    def stop_limit_order(
        self, symbol: str, qty: float, side: str,
        limit_price: float, stop_price: float, time_in_force: str = "gtc"
    ) -> dict:
        """Place a stop-limit order (triggers at stop_price, fills at limit_price)."""
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            tif = self._parse_time_in_force(time_in_force)

            request = StopLimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
                limit_price=limit_price,
                stop_price=stop_price,
            )
            order = self._call(self.trading_client.submit_order, request)
            logger.info("order.submitted", symbol=symbol, side=side, qty=qty,
                       type="stop_limit", limit=limit_price, stop=stop_price)
            return self._order_to_dict(order)
        except Exception as exc:
            logger.error("order.stop_limit_failed", symbol=symbol, error=str(exc))
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
            tif = self._parse_time_in_force(time_in_force)

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
    def close_position(self, symbol: str, qty: Optional[float] = None) -> Optional[dict]:
        """Close a position (entirely or partially by specifying qty)."""
        try:
            normalized = self._normalize_symbol_for_position(symbol)
            if qty is not None:
                close_options = ClosePositionRequest(qty=str(qty))
                self._call(self.trading_client.close_position, normalized, close_options=close_options)
                logger.info("position.partial_close", symbol=symbol, qty=qty)
            else:
                self._call(self.trading_client.close_position, normalized)
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
        """Fetch bars and return as a pandas DataFrame with data validation."""
        import pandas as pd
        import math

        try:
            bars = self.get_bars(symbol, timeframe, limit)

            if not bars:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

            records = []
            skipped = 0
            for bar in bars:
                o, h, l, c, v = float(bar.open), float(bar.high), float(bar.low), float(bar.close), float(bar.volume)
                # Data validation: reject invalid bars
                if any(math.isnan(x) or math.isinf(x) for x in (o, h, l, c)):
                    skipped += 1
                    continue
                if c <= 0 or o <= 0 or h <= 0 or l <= 0:
                    skipped += 1
                    continue
                if v < 0:
                    skipped += 1
                    continue
                if h < l:  # High must be >= Low
                    skipped += 1
                    continue
                records.append({
                    "timestamp": bar.timestamp,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                })

            if skipped > 0:
                logger.warning("bars_df.invalid_bars_skipped", symbol=symbol, skipped=skipped, total=len(bars))

            if not records:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

            df = pd.DataFrame(records).set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)
            df.sort_index(inplace=True)

            # Candle gap detection: warn if timestamps have unexpected gaps
            if len(df) >= 2:
                self._detect_candle_gaps(df, symbol, timeframe)

            return df
        except Exception as exc:
            logger.error("bars_df.get_failed", symbol=symbol, error=str(exc))
            raise

    # ──────────────────────────────────────────────────────────────────────
    # Data Quality — Gap Detection
    # ──────────────────────────────────────────────────────────────────────

    _TF_EXPECTED_DELTA = {
        "1Min": 60, "5Min": 300, "15Min": 900,
        "1Hour": 3600, "1Day": 86400,
    }

    def _detect_candle_gaps(self, df, symbol: str, timeframe: str) -> None:
        """Detect and log missing candle gaps in bar data.

        For stocks: accounts for weekends (65h), holidays (up to 4-day closures ~97h).
        For crypto: trades 24/7, so any gap beyond 3x interval is flagged.
        """
        expected_seconds = self._TF_EXPECTED_DELTA.get(timeframe)
        if expected_seconds is None:
            return  # Unknown timeframe - skip

        is_crypto = "/" in symbol  # e.g. BTC/USD, ETH/USD

        if is_crypto:
            # Crypto trades 24/7 — any gap > 3x interval is suspicious
            threshold = expected_seconds * 3
        else:
            # Stocks have weekends + holidays. Max normal closure:
            # 4-day weekend (Fri close 4pm → Tue open 9:30am) = ~113h
            # For daily bars, allow 5 days; for intraday allow 120h
            if timeframe == "1Day":
                threshold = 86400 * 5  # 5 calendar days
            else:
                threshold = 120 * 3600  # 120 hours (covers 4-day weekends)

        deltas = df.index.to_series().diff().dt.total_seconds().dropna()
        gaps = deltas[deltas > threshold]

        if len(gaps) > 0:
            logger.warning(
                "bars_df.candle_gaps_detected",
                symbol=symbol,
                timeframe=timeframe,
                gap_count=len(gaps),
                max_gap_hours=round(gaps.max() / 3600, 1),
                first_gap=str(gaps.index[0]),
            )

    # ──────────────────────────────────────────────────────────────────────
    # Real-time WebSocket Streaming
    # ──────────────────────────────────────────────────────────────────────

    def get_stock_stream(self) -> StockDataStream:
        """Get or create a stock data WebSocket stream."""
        if self._stock_stream is None:
            feed_enum = DataFeed(self.data_feed) if isinstance(self.data_feed, str) else self.data_feed
            self._stock_stream = StockDataStream(
                api_key=self.api_key,
                secret_key=self.secret_key,
                feed=feed_enum,
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
        qty = float(order.qty) if order.qty else 0
        filled_qty = float(order.filled_qty) if order.filled_qty else 0
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "side": order.side.value if order.side else None,
            "type": order.type.value if order.type else None,
            "qty": qty or None,
            "notional": float(order.notional) if getattr(order, 'notional', None) else None,
            "filled_qty": filled_qty,
            "remaining_qty": qty - filled_qty,
            "filled_avg_price": float(order.filled_avg_price) if getattr(order, 'filled_avg_price', None) else None,
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "stop_price": float(order.stop_price) if order.stop_price else None,
            "status": order.status.value if order.status else None,
            "created_at": str(order.created_at) if order.created_at else None,
            "filled_at": str(order.filled_at) if order.filled_at else None,
        }

    @_retry()
    def get_order_status(self, order_id: str) -> dict:
        """Query current order state from broker by order ID."""
        try:
            order = self._call(self.trading_client.get_order_by_id, order_id)
            return self._order_to_dict(order)
        except Exception as exc:
            logger.error("order.get_status_failed", order_id=order_id, error=str(exc))
            raise

    # ──────────────────────────────────────────────────────────────────────
    # Trade Update Stream (real-time order fills/cancellations)
    # ──────────────────────────────────────────────────────────────────────

    def start_trade_stream(self, handler) -> threading.Thread:
        """
        Start a background thread that streams real-time trade updates
        (fills, partial fills, cancellations, rejections) via WebSocket.

        Args:
            handler: async or sync callable receiving trade update data.
                     Data includes: event (fill/partial_fill/canceled/rejected),
                     order dict, timestamp, etc.

        Returns:
            The daemon thread running the stream (for lifecycle management).
        """
        if self._trade_stream_thread and self._trade_stream_thread.is_alive():
            logger.warning("trade_stream.already_running")
            return self._trade_stream_thread

        self._trade_stream = TradingStream(
            self.api_key,
            self.secret_key,
            paper=self.paper,
        )

        async def _handler_wrapper(data):
            """Normalize trade update data and invoke user handler."""
            try:
                order_data = getattr(data, 'order', None)
                is_dict = isinstance(order_data, dict)

                filled_qty = float(
                    order_data.get('filled_qty', '0') if is_dict
                    else getattr(order_data, 'filled_qty', '0')
                ) if order_data else 0
                total_qty = float(
                    order_data.get('qty', '0') if is_dict
                    else getattr(order_data, 'qty', '0')
                ) if order_data else 0

                update = {
                    "event": data.event if hasattr(data, 'event') else str(data.get('event', '')),
                    "order": {
                        "id": str(order_data.get('id', '')) if is_dict else str(getattr(order_data, 'id', '')),
                        "symbol": order_data.get('symbol', '') if is_dict else getattr(order_data, 'symbol', ''),
                        "side": order_data.get('side', '') if is_dict else getattr(order_data, 'side', ''),
                        "qty": total_qty,
                        "filled_qty": filled_qty,
                        "remaining_qty": total_qty - filled_qty,
                        "filled_avg_price": (
                            order_data.get('filled_avg_price', None) if is_dict
                            else getattr(order_data, 'filled_avg_price', None)
                        ),
                        "status": order_data.get('status', '') if is_dict else getattr(order_data, 'status', ''),
                        "type": order_data.get('type', '') if is_dict else getattr(order_data, 'type', ''),
                    },
                    "timestamp": str(data.timestamp) if hasattr(data, 'timestamp') else None,
                }

                # Reset backoff on successful message receipt
                self._stream_backoff = 1.0

                if asyncio.iscoroutinefunction(handler):
                    await handler(update)
                else:
                    handler(update)
            except Exception as e:
                logger.error("trade_stream.handler_error", error=str(e))

        self._trade_stream_handler = _handler_wrapper
        self._trade_stream.subscribe_trade_updates(_handler_wrapper)

        self._stream_backoff = 1.0
        self._shutdown_flag = False

        self._trade_stream_thread = threading.Thread(
            target=self._trade_stream_runner,
            name="trade-stream",
            daemon=True,
        )
        self._trade_stream_thread.start()
        logger.info("trade_stream.started")
        return self._trade_stream_thread

    def _trade_stream_runner(self):
        """Run trade stream with automatic reconnection on failure."""
        backoff = 1.0
        max_backoff = 60.0
        while not self._shutdown_flag:
            try:
                logger.info("trade_stream.connecting")
                self._trade_stream.run()
            except Exception as e:
                if self._shutdown_flag:
                    break
                logger.error("trade_stream.disconnected", error=str(e), reconnect_in=backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                # Re-create stream for fresh connection
                try:
                    self._trade_stream = TradingStream(
                        self.api_key,
                        self.secret_key,
                        paper=self.paper,
                    )
                    if self._trade_stream_handler:
                        self._trade_stream.subscribe_trade_updates(self._trade_stream_handler)
                except Exception as reinit_err:
                    logger.error("trade_stream.reinit_failed", error=str(reinit_err))
            else:
                # Clean exit (no exception) — reset backoff
                backoff = 1.0
                if not self._shutdown_flag:
                    # Unexpected clean exit, try reconnecting
                    logger.warning("trade_stream.unexpected_exit", reconnect_in=backoff)
                    time.sleep(backoff)
        logger.info("trade_stream.shutdown")

    def stop_trade_stream(self):
        """Stop the trade update stream and signal the reconnection loop to exit."""
        self._shutdown_flag = True
        if self._trade_stream:
            try:
                self._trade_stream.stop()
            except Exception as e:
                logger.debug("trade_stream.stop_error", error=str(e))
            self._trade_stream = None
        if self._trade_stream_thread and self._trade_stream_thread.is_alive():
            self._trade_stream_thread.join(timeout=5.0)
            if self._trade_stream_thread.is_alive():
                logger.warning("trade_stream.stop_timeout", timeout=5.0)
        self._trade_stream_thread = None
        logger.info("trade_stream.stopped")

    def stop_market_data_streams(self):
        """Stop stock/crypto market-data WebSocket streams."""
        if self._stock_stream:
            try:
                self._stock_stream.stop()
            except Exception as e:
                logger.debug("stock_stream.stop_error", error=str(e))
            self._stock_stream = None
        if self._crypto_stream:
            try:
                self._crypto_stream.stop()
            except Exception as e:
                logger.debug("crypto_stream.stop_error", error=str(e))
            self._crypto_stream = None

    # ──────────────────────────────────────────────────────────────────────
    # Asset Discovery
    # ──────────────────────────────────────────────────────────────────────

    @_retry()
    def get_crypto_assets(self) -> list[dict]:
        """Discover all tradeable crypto pairs on Alpaca."""
        try:
            request = GetAssetsRequest(
                asset_class=AssetClass.CRYPTO,
                status=AssetStatus.ACTIVE,
            )
            assets = self._call(self.trading_client.get_all_assets, request)
            return [
                {
                    "symbol": a.symbol,
                    "name": a.name,
                    "exchange": a.exchange.value if a.exchange else None,
                    "tradable": a.tradable,
                    "fractionable": a.fractionable,
                    "min_order_size": getattr(a, 'min_order_size', None),
                    "min_trade_increment": getattr(a, 'min_trade_increment', None),
                }
                for a in assets
                if a.tradable
            ]
        except Exception as exc:
            logger.error("assets.crypto_discovery_failed", error=str(exc))
            raise

    @_retry()
    def get_stock_assets(self, exchange: str = None) -> list[dict]:
        """Discover tradeable stock assets, optionally filtered by exchange."""
        try:
            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE,
            )
            assets = self._call(self.trading_client.get_all_assets, request)
            result = []
            for a in assets:
                if not a.tradable:
                    continue
                if exchange and a.exchange and a.exchange.value != exchange:
                    continue
                result.append({
                    "symbol": a.symbol,
                    "name": a.name,
                    "exchange": a.exchange.value if a.exchange else None,
                    "tradable": a.tradable,
                    "fractionable": a.fractionable,
                    "shortable": a.shortable,
                })
            return result
        except Exception as exc:
            logger.error("assets.stock_discovery_failed", error=str(exc))
            raise

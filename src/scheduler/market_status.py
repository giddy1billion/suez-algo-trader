"""
Market Status Service — Unified market state per asset class.

Queries market calendars and determines the current trading state
for each asset class (equity, crypto). Used by the scheduler to
gate activities appropriately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from src.market.calendars import get_calendar, MarketCalendar
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MarketPhase(str, Enum):
    """Current phase of the market session."""
    PRE_MARKET = "pre_market"
    OPEN = "open"
    POST_MARKET = "post_market"
    CLOSED = "closed"
    CONTINUOUS = "continuous"  # 24/7 markets like crypto


@dataclass
class AssetClassStatus:
    """Status for a single asset class."""
    asset_class: str
    phase: MarketPhase
    is_trading: bool
    calendar_name: str
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None
    symbols: list[str] = field(default_factory=list)


class MarketStatusService:
    """
    Provides unified market status for all asset classes.

    Integrates with existing market calendars from src/market/calendars.py
    to determine whether each asset class is currently active.
    """

    def __init__(
        self,
        equity_symbols: Optional[list[str]] = None,
        crypto_symbols: Optional[list[str]] = None,
    ):
        self._equity_symbols = equity_symbols or [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"
        ]
        self._crypto_symbols = crypto_symbols or [
            "BTC/USD", "ETH/USD", "SOL/USD", "AAVE/USD", "ADA/USD"
        ]

        # Load calendars
        try:
            self._equity_calendar: MarketCalendar = get_calendar("NYSE")
        except Exception:
            self._equity_calendar = None
            logger.warning("market_status.nyse_calendar_unavailable")

        try:
            self._crypto_calendar: MarketCalendar = get_calendar("CRYPTO")
        except Exception:
            self._crypto_calendar = None
            logger.warning("market_status.crypto_calendar_unavailable")

    def get_equity_status(self, now: Optional[datetime] = None) -> AssetClassStatus:
        """Get current equity market status."""
        now = now or datetime.now(timezone.utc)

        if self._equity_calendar is None:
            return AssetClassStatus(
                asset_class="equity",
                phase=MarketPhase.CLOSED,
                is_trading=False,
                calendar_name="NYSE",
                symbols=self._equity_symbols,
            )

        is_trading_day = self._equity_calendar.is_trading_day(now)
        is_trading_time = self._equity_calendar.is_trading_time(now)

        if is_trading_day and is_trading_time:
            phase = MarketPhase.OPEN
        elif is_trading_day:
            phase = MarketPhase.PRE_MARKET  # Simplified; could detect pre/post
        else:
            phase = MarketPhase.CLOSED

        return AssetClassStatus(
            asset_class="equity",
            phase=phase,
            is_trading=is_trading_time and is_trading_day,
            calendar_name="NYSE",
            symbols=self._equity_symbols,
        )

    def get_crypto_status(self, now: Optional[datetime] = None) -> AssetClassStatus:
        """Get current crypto market status (always continuous)."""
        return AssetClassStatus(
            asset_class="crypto",
            phase=MarketPhase.CONTINUOUS,
            is_trading=True,
            calendar_name="CRYPTO",
            symbols=self._crypto_symbols,
        )

    def get_all_statuses(self, now: Optional[datetime] = None) -> dict[str, AssetClassStatus]:
        """Get status for all asset classes."""
        return {
            "equity": self.get_equity_status(now),
            "crypto": self.get_crypto_status(now),
        }

    def is_any_market_open(self, now: Optional[datetime] = None) -> bool:
        """Check if any market is currently trading."""
        statuses = self.get_all_statuses(now)
        return any(s.is_trading for s in statuses.values())

    def get_active_symbols(self, now: Optional[datetime] = None) -> list[str]:
        """Get symbols from currently active markets."""
        symbols = []
        statuses = self.get_all_statuses(now)
        for status in statuses.values():
            if status.is_trading:
                symbols.extend(status.symbols)
        return symbols

    @property
    def equity_symbols(self) -> list[str]:
        return self._equity_symbols

    @property
    def crypto_symbols(self) -> list[str]:
        return self._crypto_symbols

    @property
    def all_symbols(self) -> list[str]:
        return self._equity_symbols + self._crypto_symbols

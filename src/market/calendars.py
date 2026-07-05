"""
Market Calendars — abstract contract and implementations.

A MarketCalendar encapsulates all temporal logic for a specific market:
trading days, trading hours, session boundaries, expected bar intervals,
and annualization factors. No business logic outside these classes should
contain calendar assumptions.
"""

from abc import ABC, abstractmethod
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

from src.market.holidays import HolidayCalendar, NoHolidayCalendar, NYSEHolidayCalendar
from src.market.sessions import (
    CRYPTO_ALWAYS_OPEN,
    NYSE_AFTER_HOURS,
    NYSE_PRE_MARKET,
    NYSE_REGULAR,
    SessionType,
    TradingSession,
    get_current_session_type,
)


class MarketCalendar(ABC):
    """
    Abstract base for all market calendars.

    Every exchange or market type must implement this interface. Strategy,
    backtesting, risk, and execution engines consume calendars via this
    contract, ensuring no hardcoded calendar assumptions leak into
    business logic.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical calendar name (e.g., 'NYSE', '24/7')."""
        ...

    @property
    @abstractmethod
    def timezone(self) -> str:
        """IANA timezone for this calendar's local time."""
        ...

    @abstractmethod
    def is_trading_day(self, dt: date) -> bool:
        """Check if a given date is a valid trading day."""
        ...

    @abstractmethod
    def is_trading_time(self, dt: datetime) -> bool:
        """
        Check if a given datetime falls within tradable hours.
        The datetime should be in UTC or timezone-aware.
        """
        ...

    @abstractmethod
    def current_session(self, dt: datetime) -> SessionType:
        """Get the current session type for a datetime."""
        ...

    @abstractmethod
    def sessions(self) -> list[TradingSession]:
        """Get all defined sessions for this calendar."""
        ...

    @abstractmethod
    def trading_minutes_per_day(self) -> int:
        """Number of tradable minutes in a regular session day."""
        ...

    @abstractmethod
    def trading_days_per_year(self) -> int:
        """Approximate number of trading days per year."""
        ...

    @abstractmethod
    def next_session_open(self, after: datetime) -> datetime:
        """Get the next session open time after the given datetime."""
        ...

    @abstractmethod
    def previous_session_close(self, before: datetime) -> datetime:
        """Get the most recent session close before the given datetime."""
        ...

    def expected_bar_interval_seconds(self, timeframe: str) -> float:
        """
        Get the base interval in seconds for a timeframe on this calendar.
        """
        tf_seconds = {
            "1Min": 60, "5Min": 300, "15Min": 900,
            "30Min": 1800, "1Hour": 3600, "4Hour": 14400,
            "1Day": 86400, "1Week": 604800,
        }
        return float(tf_seconds.get(timeframe, 3600))

    def max_expected_gap_seconds(self, timeframe: str) -> float:
        """
        Maximum expected gap (seconds) between consecutive bars,
        accounting for overnight, weekends, and holidays.

        Override in subclasses for market-specific behavior.
        """
        return self.expected_bar_interval_seconds(timeframe) * 1.5

    def annualization_factor(self, timeframe: str) -> float:
        """
        Compute sqrt(periods_per_year) for Sharpe ratio annualization.
        Derived from this calendar's session definition.
        """
        import numpy as np
        return np.sqrt(self.periods_per_year(timeframe))

    def periods_per_year(self, timeframe: str) -> float:
        """
        Number of bars per year for a given timeframe on this calendar.
        Derived from trading days and minutes per day.
        """
        base_seconds = self.expected_bar_interval_seconds(timeframe)
        if timeframe in ("1Day", "1Week"):
            # For daily/weekly, periods = trading days (or weeks)
            if timeframe == "1Day":
                return float(self.trading_days_per_year())
            else:
                return float(self.trading_days_per_year()) / 5.0
        minutes_per_year = self.trading_days_per_year() * self.trading_minutes_per_day()
        return minutes_per_year / (base_seconds / 60.0)


class NYSECalendar(MarketCalendar):
    """
    NYSE trading calendar.

    Regular session: Mon-Fri, 09:30-16:00 Eastern Time.
    Excludes NYSE-observed holidays.
    Supports pre-market (04:00-09:30) and after-hours (16:00-20:00).
    """

    def __init__(self, holiday_calendar: Optional[HolidayCalendar] = None):
        self._holidays = holiday_calendar or NYSEHolidayCalendar()
        self._regular_open = time(9, 30)
        self._regular_close = time(16, 0)

    @property
    def name(self) -> str:
        return "NYSE"

    @property
    def timezone(self) -> str:
        return "America/New_York"

    def is_trading_day(self, dt: date) -> bool:
        """A trading day is a weekday that is not an NYSE holiday."""
        if isinstance(dt, datetime):
            dt = dt.date() if not hasattr(dt, 'date') else dt.date()
        if hasattr(dt, 'weekday'):
            if dt.weekday() >= 5:
                return False
        return not self._holidays.is_holiday(dt)

    def is_trading_time(self, dt: datetime) -> bool:
        """
        Check if dt falls within regular NYSE trading hours.
        Expects timezone-aware datetime (will convert from UTC).
        """
        if not self.is_trading_day(dt):
            return False
        local_time = self._to_local_time(dt)
        return self._regular_open <= local_time < self._regular_close

    def current_session(self, dt: datetime) -> SessionType:
        """Determine which session the given datetime falls into."""
        if not self.is_trading_day(dt):
            return SessionType.CLOSED
        local_time = self._to_local_time(dt)
        return get_current_session_type(local_time, self.sessions())

    def sessions(self) -> list[TradingSession]:
        """NYSE has pre-market, regular, and after-hours sessions."""
        return [NYSE_PRE_MARKET, NYSE_REGULAR, NYSE_AFTER_HOURS]

    def trading_minutes_per_day(self) -> int:
        """NYSE regular session is 6.5 hours = 390 minutes."""
        return 390

    def trading_days_per_year(self) -> int:
        """NYSE averages ~252 trading days per year."""
        return 252

    def next_session_open(self, after: datetime) -> datetime:
        """Find the next regular session open after the given datetime."""
        current = after.date() if isinstance(after, datetime) else after
        # Move to next day if we're past today's open
        local_time = self._to_local_time(after) if isinstance(after, datetime) else time(23, 59)
        if local_time >= self._regular_open:
            current = current + timedelta(days=1)
        # Find next trading day
        for _ in range(10):  # Max 10 days ahead (handles long weekends)
            if self.is_trading_day(current):
                return datetime.combine(current, self._regular_open)
            current = current + timedelta(days=1)
        # Fallback: next weekday
        return datetime.combine(current, self._regular_open)

    def previous_session_close(self, before: datetime) -> datetime:
        """Find the most recent session close before the given datetime."""
        current = before.date() if isinstance(before, datetime) else before
        local_time = self._to_local_time(before) if isinstance(before, datetime) else time(0, 0)
        if local_time < self._regular_close:
            current = current - timedelta(days=1)
        for _ in range(10):
            if self.is_trading_day(current):
                return datetime.combine(current, self._regular_close)
            current = current - timedelta(days=1)
        return datetime.combine(current, self._regular_close)

    def max_expected_gap_seconds(self, timeframe: str) -> float:
        """
        NYSE-specific gap thresholds accounting for overnight/weekend/holidays.
        """
        if timeframe in ("1Day", "1Week"):
            # Daily: allow up to 4 days (3-day weekend + holiday)
            return self.expected_bar_interval_seconds(timeframe) * 4
        else:
            # Intraday: 90 hours covers 3-day weekends + holidays
            return 90 * 3600

    def _to_local_time(self, dt: datetime) -> time:
        """Extract local time from a datetime (assumes Eastern or converts from UTC)."""
        try:
            import pytz
            eastern = pytz.timezone("America/New_York")
            if dt.tzinfo is not None:
                local_dt = dt.astimezone(eastern)
            else:
                local_dt = pytz.utc.localize(dt).astimezone(eastern)
            return local_dt.time()
        except ImportError:
            # Without pytz, use the time as-is (caller responsibility)
            return dt.time()


class Crypto247Calendar(MarketCalendar):
    """
    24/7 cryptocurrency calendar.

    No closures, no holidays, continuous trading.
    """

    @property
    def name(self) -> str:
        return "24/7"

    @property
    def timezone(self) -> str:
        return "UTC"

    def is_trading_day(self, dt: date) -> bool:
        """Crypto trades every day."""
        return True

    def is_trading_time(self, dt: datetime) -> bool:
        """Crypto trades every moment."""
        return True

    def current_session(self, dt: datetime) -> SessionType:
        """Crypto is always in an open session."""
        return SessionType.ALWAYS_OPEN

    def sessions(self) -> list[TradingSession]:
        """Single always-open session."""
        return [CRYPTO_ALWAYS_OPEN]

    def trading_minutes_per_day(self) -> int:
        """24 hours = 1440 minutes."""
        return 1440

    def trading_days_per_year(self) -> int:
        """365 days (366 in leap years, but 365 is standard)."""
        return 365

    def next_session_open(self, after: datetime) -> datetime:
        """Always open — return current time."""
        return after

    def previous_session_close(self, before: datetime) -> datetime:
        """Never closes — return current time."""
        return before

    def max_expected_gap_seconds(self, timeframe: str) -> float:
        """
        Crypto: gaps should never exceed 1.5x the interval.
        Allow small tolerance for exchange maintenance.
        """
        return self.expected_bar_interval_seconds(timeframe) * 1.5


class NASDAQCalendar(NYSECalendar):
    """
    NASDAQ calendar — identical session hours to NYSE.

    NASDAQ follows the same holiday and session schedule as NYSE.
    Provided as a separate class for future differentiation.
    """

    @property
    def name(self) -> str:
        return "NASDAQ"


class CMECalendar(MarketCalendar):
    """
    CME futures calendar — stub for future implementation.

    CME has complex session structures (globex, pit, etc.) that
    differ by product group. This stub provides a basic framework.
    """

    @property
    def name(self) -> str:
        return "CME"

    @property
    def timezone(self) -> str:
        return "America/Chicago"

    def is_trading_day(self, dt: date) -> bool:
        if isinstance(dt, datetime):
            dt = dt.date()
        # CME is closed on weekends and some holidays
        return dt.weekday() < 5

    def is_trading_time(self, dt: datetime) -> bool:
        # CME Globex trades nearly 23 hours/day Sun-Fri
        if not self.is_trading_day(dt):
            return False
        return True  # Simplified — real implementation needs product-specific sessions

    def current_session(self, dt: datetime) -> SessionType:
        if not self.is_trading_day(dt):
            return SessionType.CLOSED
        return SessionType.REGULAR

    def sessions(self) -> list[TradingSession]:
        return [TradingSession(
            session_type=SessionType.REGULAR,
            open_time=time(17, 0),  # 5pm previous day
            close_time=time(16, 0),  # 4pm
            name="CME Globex",
        )]

    def trading_minutes_per_day(self) -> int:
        return 1380  # ~23 hours

    def trading_days_per_year(self) -> int:
        return 251

    def next_session_open(self, after: datetime) -> datetime:
        current = after + timedelta(days=1)
        return datetime.combine(current.date(), time(17, 0))

    def previous_session_close(self, before: datetime) -> datetime:
        return datetime.combine(before.date(), time(16, 0))


class ForexCalendar(MarketCalendar):
    """
    Forex calendar — stub for future implementation.

    Forex trades 24 hours, Sunday evening through Friday evening (US time).
    """

    @property
    def name(self) -> str:
        return "FOREX"

    @property
    def timezone(self) -> str:
        return "UTC"

    def is_trading_day(self, dt: date) -> bool:
        if isinstance(dt, datetime):
            dt = dt.date()
        # Forex is closed Saturday and most of Sunday
        return dt.weekday() < 5  # Simplified

    def is_trading_time(self, dt: datetime) -> bool:
        return self.is_trading_day(dt)

    def current_session(self, dt: datetime) -> SessionType:
        if not self.is_trading_day(dt):
            return SessionType.CLOSED
        return SessionType.REGULAR

    def sessions(self) -> list[TradingSession]:
        return [TradingSession(
            session_type=SessionType.REGULAR,
            open_time=time(17, 0),  # Sunday 5pm ET
            close_time=time(17, 0),  # Friday 5pm ET
            name="Forex Session",
        )]

    def trading_minutes_per_day(self) -> int:
        return 1440  # 24 hours on trading days

    def trading_days_per_year(self) -> int:
        return 260  # ~5 days/week * 52 weeks

    def next_session_open(self, after: datetime) -> datetime:
        current = after + timedelta(days=1)
        while current.weekday() >= 5:
            current += timedelta(days=1)
        return datetime.combine(current.date(), time(0, 0))

    def previous_session_close(self, before: datetime) -> datetime:
        current = before - timedelta(days=1)
        while current.weekday() >= 5:
            current -= timedelta(days=1)
        return datetime.combine(current.date(), time(23, 59))


# Calendar registry for lookup by name
_CALENDAR_REGISTRY: dict[str, type[MarketCalendar]] = {
    "NYSE": NYSECalendar,
    "NASDAQ": NASDAQCalendar,
    "CME": CMECalendar,
    "FOREX": ForexCalendar,
    "24/7": Crypto247Calendar,
}


def get_calendar(name: str) -> MarketCalendar:
    """
    Get a calendar instance by name.

    Args:
        name: Calendar name (e.g., "NYSE", "24/7", "CME").

    Returns:
        Instantiated MarketCalendar.

    Raises:
        ValueError: If calendar name is not registered.
    """
    calendar_cls = _CALENDAR_REGISTRY.get(name)
    if calendar_cls is None:
        raise ValueError(
            f"Unknown calendar: {name}. "
            f"Available: {list(_CALENDAR_REGISTRY.keys())}"
        )
    return calendar_cls()


def register_calendar(name: str, calendar_cls: type[MarketCalendar]) -> None:
    """Register a custom calendar implementation."""
    _CALENDAR_REGISTRY[name] = calendar_cls

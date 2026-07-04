"""
Utility functions for the market package.
"""

from datetime import datetime, timezone
from typing import Optional

from src.market.calendars import MarketCalendar, get_calendar
from src.market.instruments import Instrument
from src.market.registry import classify_symbol
from src.market.sessions import SessionType


def is_tradable_now(
    instrument: Instrument,
    now: Optional[datetime] = None,
) -> bool:
    """
    Check if an instrument is currently tradable based on its calendar.

    Backward-compatible with the original market_calendar.is_tradable_now().

    Args:
        instrument: Instrument to check.
        now: Current datetime (defaults to utcnow). Should be timezone-aware.

    Returns:
        True if the instrument can be traded right now.
    """
    if instrument.trades_24_7:
        return True

    if now is None:
        now = datetime.now(timezone.utc)

    calendar = get_calendar(instrument.calendar)
    return calendar.is_trading_time(now)


def filter_trading_hours(df, instrument: Instrument):
    """
    Filter a DataFrame to only include bars during valid trading hours.

    Backward-compatible with market_calendar.filter_trading_hours().
    """
    import pandas as pd

    if instrument.trades_24_7:
        return df

    calendar = get_calendar(instrument.calendar)

    # For NYSE equities, filter to trading hours
    try:
        import pytz
        eastern = pytz.timezone(calendar.timezone)
        idx_et = (
            df.index.tz_convert(eastern)
            if df.index.tz
            else df.index.tz_localize("UTC").tz_convert(eastern)
        )
    except (ImportError, TypeError):
        idx_et = df.index

    from datetime import time
    mask = pd.Series(True, index=df.index)
    mask &= idx_et.weekday < 5
    times = idx_et.time
    mask &= (times >= time(9, 30)) & (times < time(16, 0))

    return df[mask.values]

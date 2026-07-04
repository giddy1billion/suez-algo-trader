"""
Timezone normalization utilities.

Enforces the fundamental rule: ALL internal timestamps must be UTC.
Local exchange time is only used for session evaluation.

These utilities help ensure timezone safety throughout the engine.
"""

from datetime import datetime, timezone, tzinfo
from typing import Optional

import pandas as pd


class NaiveDatetimeError(TypeError):
    """Raised when a naive (timezone-unaware) datetime is detected."""

    def __init__(self, dt: datetime, context: str = ""):
        msg = (
            f"Naive datetime detected: {dt}. "
            f"All internal timestamps must be timezone-aware (UTC). "
            f"Context: {context}"
        )
        super().__init__(msg)


def ensure_utc(dt: datetime, context: str = "") -> datetime:
    """
    Ensure a datetime is timezone-aware and in UTC.

    - If already UTC, returns as-is.
    - If timezone-aware but not UTC, converts to UTC.
    - If naive, raises NaiveDatetimeError.

    Args:
        dt: Datetime to validate and normalize.
        context: Description of where this datetime came from (for errors).

    Returns:
        UTC-normalized datetime.

    Raises:
        NaiveDatetimeError: If dt is naive (no tzinfo).
    """
    if dt.tzinfo is None:
        raise NaiveDatetimeError(dt, context)
    if dt.tzinfo == timezone.utc or (hasattr(dt.tzinfo, 'zone') and dt.tzinfo.zone == 'UTC'):
        return dt
    return dt.astimezone(timezone.utc)


def to_exchange_time(dt: datetime, exchange_tz: str) -> datetime:
    """
    Convert a UTC datetime to exchange local time.

    Only use this when evaluating sessions, holidays, or trading windows.
    Never store the result — always convert back to UTC for internal use.

    Args:
        dt: UTC datetime.
        exchange_tz: IANA timezone string (e.g., "America/New_York").

    Returns:
        Datetime in exchange local time.
    """
    try:
        import pytz
        tz = pytz.timezone(exchange_tz)
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
        return dt.astimezone(tz)
    except ImportError:
        # Without pytz, return as-is (limited functionality)
        return dt


def reject_naive_datetime(dt: datetime, context: str = "") -> None:
    """
    Guard function that raises if a naive datetime is encountered.

    Use at entry points to critical subsystems to enforce UTC-only policy.

    Args:
        dt: Datetime to check.
        context: Description for error message.

    Raises:
        NaiveDatetimeError: If dt has no timezone info.
    """
    if dt.tzinfo is None:
        raise NaiveDatetimeError(dt, context)


def normalize_index_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a DataFrame's DatetimeIndex to UTC.

    - If already UTC, returns unchanged.
    - If timezone-aware but not UTC, converts to UTC.
    - If naive, localizes as UTC (assumes data is already in UTC).

    Args:
        df: DataFrame with DatetimeIndex.

    Returns:
        DataFrame with UTC-normalized index.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    if df.index.tz is None:
        # Assume naive timestamps are UTC
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    elif str(df.index.tz) != "UTC":
        df = df.copy()
        df.index = df.index.tz_convert("UTC")

    return df


def is_utc(dt: datetime) -> bool:
    """Check if a datetime is in UTC."""
    if dt.tzinfo is None:
        return False
    return dt.tzinfo == timezone.utc or (
        hasattr(dt.tzinfo, 'zone') and dt.tzinfo.zone == 'UTC'
    )


def utc_now() -> datetime:
    """Get the current time in UTC (timezone-aware)."""
    return datetime.now(timezone.utc)

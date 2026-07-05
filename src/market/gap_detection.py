"""
Gap detection — calendar-aware data quality validation.

Detects genuine missing-data gaps in bar data by consulting the instrument's
calendar. A gap is only flagged if the time between consecutive bars exceeds
what the instrument's calendar would explain (overnight, weekends, holidays).

No fixed thresholds — all thresholds derive from the calendar definition.
"""

import pandas as pd

from src.market.calendars import MarketCalendar, get_calendar
from src.market.instruments import Instrument
from src.utils.logger import get_logger

logger = get_logger(__name__)


def detect_gaps(
    df: pd.DataFrame,
    instrument: Instrument,
    timeframe: str,
) -> pd.Series:
    """
    Detect genuine missing-data gaps in bar data, respecting the
    instrument's trading calendar.

    A gap is only flagged if the time between consecutive bars exceeds
    what the instrument's calendar would explain.

    Args:
        df: DataFrame with DatetimeIndex of bar timestamps.
        instrument: Instrument metadata with calendar info.
        timeframe: Bar timeframe (e.g., "1Hour").

    Returns:
        Series of gap sizes (seconds) for bars that represent genuine gaps.
        Empty Series if no gaps found.
    """
    if len(df) < 2:
        return pd.Series(dtype=float)

    calendar = get_calendar(instrument.calendar)
    threshold = calendar.max_expected_gap_seconds(timeframe)

    deltas = df.index.to_series().diff().dt.total_seconds().dropna()
    gaps = deltas[deltas > threshold]
    return gaps


def log_gap_report(
    gaps: pd.Series,
    instrument: Instrument,
    timeframe: str,
) -> None:
    """Log a structured gap detection report."""
    if len(gaps) == 0:
        return

    logger.warning(
        "market.gaps_detected",
        symbol=instrument.symbol,
        asset_class=instrument.asset_class.value,
        calendar=instrument.calendar,
        timeframe=timeframe,
        gap_count=len(gaps),
        max_gap_hours=round(gaps.max() / 3600, 1),
        first_gap=str(gaps.index[0]),
    )


def expected_gap_seconds(
    instrument: Instrument,
    timeframe: str,
    current_bar=None,
) -> float:
    """
    Calculate the maximum expected gap (in seconds) between consecutive bars.

    Backward-compatible wrapper around calendar.max_expected_gap_seconds().

    Args:
        instrument: The Instrument with calendar metadata.
        timeframe: Bar timeframe string.
        current_bar: Unused (kept for backward compatibility).

    Returns:
        Maximum expected seconds until the next valid bar.
    """
    calendar = get_calendar(instrument.calendar)
    return calendar.max_expected_gap_seconds(timeframe)

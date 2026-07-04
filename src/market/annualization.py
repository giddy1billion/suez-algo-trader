"""
Annualization engine.

Derives annualization factors from calendar session definitions rather than
hardcoded constants. Every performance metric (Sharpe, Sortino, Calmar, CAGR)
should consume this API.
"""

import numpy as np

from src.market.calendars import MarketCalendar, get_calendar
from src.market.instruments import AssetClass, Instrument


def bars_per_year(calendar: MarketCalendar, timeframe: str) -> float:
    """
    Calculate the number of bars per year for a given calendar and timeframe.

    Derived from the calendar's session definition:
        bars/year = (trading_days * trading_minutes_per_day) / minutes_per_bar

    Args:
        calendar: The market calendar to use.
        timeframe: Bar timeframe (e.g., "1Hour", "1Day").

    Returns:
        Number of bars per year as a float.
    """
    bar_seconds = calendar.expected_bar_interval_seconds(timeframe)
    bar_minutes = bar_seconds / 60.0

    if timeframe in ("1Day", "1Week"):
        # For daily/weekly, count in trading days
        if timeframe == "1Day":
            return float(calendar.trading_days_per_year())
        else:
            return float(calendar.trading_days_per_year()) / 5.0
    else:
        # For intraday: trading_days * bars_per_day
        bars_per_day = calendar.trading_minutes_per_day() / bar_minutes
        return float(calendar.trading_days_per_year()) * bars_per_day


def get_annualization_factor(
    instrument: Instrument,
    timeframe: str,
) -> float:
    """
    Get the correct Sharpe ratio annualization factor for an instrument.

    The factor is sqrt(periods_per_year), derived from the instrument's
    calendar and the timeframe.

    Args:
        instrument: Instrument with calendar metadata.
        timeframe: Bar timeframe string (e.g., "1Hour", "1Day").

    Returns:
        sqrt(N) annualization factor where N = periods per year.
    """
    calendar = get_calendar(instrument.calendar)
    n_periods = bars_per_year(calendar, timeframe)
    return float(np.sqrt(n_periods))


def get_periods_per_year(
    asset_class: AssetClass,
    timeframe: str,
) -> float:
    """
    Get the number of trading periods per year.

    Convenience function that maps asset class to the appropriate calendar
    and computes bars per year.

    Args:
        asset_class: EQUITY, CRYPTO, etc.
        timeframe: Bar timeframe string.

    Returns:
        Number of periods per year.
    """
    calendar_name = _asset_class_to_calendar(asset_class)
    calendar = get_calendar(calendar_name)
    return bars_per_year(calendar, timeframe)


def sharpe_annualization(
    asset_class: AssetClass,
    timeframe: str,
) -> float:
    """Annualization factor specifically for Sharpe ratio."""
    return float(np.sqrt(get_periods_per_year(asset_class, timeframe)))


def sortino_annualization(
    asset_class: AssetClass,
    timeframe: str,
) -> float:
    """Annualization factor for Sortino ratio (same as Sharpe)."""
    return sharpe_annualization(asset_class, timeframe)


def cagr_periods(
    asset_class: AssetClass,
    timeframe: str,
) -> float:
    """Number of periods per year for CAGR calculation."""
    return get_periods_per_year(asset_class, timeframe)


def _asset_class_to_calendar(asset_class: AssetClass) -> str:
    """Map asset class to default calendar name."""
    mapping = {
        AssetClass.EQUITY: "NYSE",
        AssetClass.CRYPTO: "24/7",
        AssetClass.FUTURES: "CME",
        AssetClass.FOREX: "FOREX",
        AssetClass.OPTIONS: "NYSE",
    }
    return mapping.get(asset_class, "NYSE")

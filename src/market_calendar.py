"""
Market Calendar Management — First-class subsystem for multi-asset trading.

Provides:
- Instrument metadata (asset class, exchange, timezone, calendar)
- Exchange-specific trading calendars (NYSE, 24/7 crypto)
- Calendar-aware gap detection
- Session-aware utilities for backtesting and feature engineering
- Proper annualization factors per asset class

This module ensures that mixing 24/7 crypto markets with session-based equity
markets does not silently distort research results, gap detection, or risk metrics.
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Enums & Constants
# ══════════════════════════════════════════════════════════════════════════════


class AssetClass(str, Enum):
    """Supported asset classes."""
    EQUITY = "equity"
    CRYPTO = "crypto"


class ExchangeCalendar(str, Enum):
    """Supported exchange calendars."""
    NYSE = "NYSE"           # Mon-Fri, 09:30-16:00 ET (with holidays)
    TWENTY_FOUR_SEVEN = "24/7"  # Continuous trading, no closures


# NYSE holidays (Federal holidays observed by NYSE) — simplified set.
# A production system would use a maintained holiday calendar library.
_NYSE_HOLIDAYS_2024_2027 = {
    # 2024
    datetime(2024, 1, 1), datetime(2024, 1, 15), datetime(2024, 2, 19),
    datetime(2024, 3, 29), datetime(2024, 5, 27), datetime(2024, 6, 19),
    datetime(2024, 7, 4), datetime(2024, 9, 2), datetime(2024, 11, 28),
    datetime(2024, 12, 25),
    # 2025
    datetime(2025, 1, 1), datetime(2025, 1, 20), datetime(2025, 2, 17),
    datetime(2025, 4, 18), datetime(2025, 5, 26), datetime(2025, 6, 19),
    datetime(2025, 7, 4), datetime(2025, 9, 1), datetime(2025, 11, 27),
    datetime(2025, 12, 25),
    # 2026
    datetime(2026, 1, 1), datetime(2026, 1, 19), datetime(2026, 2, 16),
    datetime(2026, 4, 3), datetime(2026, 5, 25), datetime(2026, 6, 19),
    datetime(2026, 7, 3), datetime(2026, 9, 7), datetime(2026, 11, 26),
    datetime(2026, 12, 25),
    # 2027
    datetime(2027, 1, 1), datetime(2027, 1, 18), datetime(2027, 2, 15),
    datetime(2027, 3, 26), datetime(2027, 5, 31), datetime(2027, 6, 18),
    datetime(2027, 7, 5), datetime(2027, 9, 6), datetime(2027, 11, 25),
    datetime(2027, 12, 24),
}

# NYSE regular session hours (Eastern Time)
_NYSE_OPEN = time(9, 30)
_NYSE_CLOSE = time(16, 0)

# Annualization factors for Sharpe ratio calculation
# Equity: ~252 trading days/year, ~6.5 hours/day
# Crypto: 365 days/year, 24 hours/day
ANNUALIZATION_FACTORS = {
    "1Min": {AssetClass.EQUITY: np.sqrt(252 * 390), AssetClass.CRYPTO: np.sqrt(365 * 1440)},
    "5Min": {AssetClass.EQUITY: np.sqrt(252 * 78), AssetClass.CRYPTO: np.sqrt(365 * 288)},
    "15Min": {AssetClass.EQUITY: np.sqrt(252 * 26), AssetClass.CRYPTO: np.sqrt(365 * 96)},
    "1Hour": {AssetClass.EQUITY: np.sqrt(252 * 7), AssetClass.CRYPTO: np.sqrt(365 * 24)},
    "4Hour": {AssetClass.EQUITY: np.sqrt(252 * 2), AssetClass.CRYPTO: np.sqrt(365 * 6)},
    "1Day": {AssetClass.EQUITY: np.sqrt(252), AssetClass.CRYPTO: np.sqrt(365)},
}


# ══════════════════════════════════════════════════════════════════════════════
# Instrument Metadata
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Instrument:
    """
    Market instrument with calendar metadata.

    Every tradable symbol should be wrapped in an Instrument so that all
    calendar-sensitive logic can reference the correct trading schedule.
    """
    symbol: str
    asset_class: AssetClass
    exchange_calendar: ExchangeCalendar
    timezone: str = "UTC"  # Primary timezone for the exchange

    @property
    def is_crypto(self) -> bool:
        return self.asset_class == AssetClass.CRYPTO

    @property
    def is_equity(self) -> bool:
        return self.asset_class == AssetClass.EQUITY

    @property
    def trades_24_7(self) -> bool:
        return self.exchange_calendar == ExchangeCalendar.TWENTY_FOUR_SEVEN


def classify_symbol(symbol: str) -> Instrument:
    """
    Classify a symbol into an Instrument with proper calendar metadata.

    Uses the "/" convention (e.g., BTC/USD) to detect crypto assets.
    All other symbols are assumed to be US equities on NYSE calendar.

    Args:
        symbol: Trading symbol string (e.g., "AAPL", "BTC/USD")

    Returns:
        Instrument with correct asset_class and exchange_calendar
    """
    if "/" in symbol:
        return Instrument(
            symbol=symbol,
            asset_class=AssetClass.CRYPTO,
            exchange_calendar=ExchangeCalendar.TWENTY_FOUR_SEVEN,
            timezone="UTC",
        )
    else:
        return Instrument(
            symbol=symbol,
            asset_class=AssetClass.EQUITY,
            exchange_calendar=ExchangeCalendar.NYSE,
            timezone="America/New_York",
        )


def classify_symbols(symbols: list[str]) -> dict[str, Instrument]:
    """Classify a list of symbols into Instruments."""
    return {s: classify_symbol(s) for s in symbols}


# ══════════════════════════════════════════════════════════════════════════════
# Calendar Logic
# ══════════════════════════════════════════════════════════════════════════════


def is_nyse_trading_day(dt: datetime) -> bool:
    """Check if a given date is a NYSE trading day (weekday, not a holiday)."""
    if dt.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    # Check against known holidays (date only)
    dt_date = datetime(dt.year, dt.month, dt.day)
    if dt_date in _NYSE_HOLIDAYS_2024_2027:
        return False
    return True


def is_nyse_trading_hour(dt: datetime) -> bool:
    """
    Check if a given datetime falls within NYSE regular trading hours.
    Expects dt in Eastern Time or will be treated as-is.
    """
    if not is_nyse_trading_day(dt):
        return False
    t = dt.time()
    return _NYSE_OPEN <= t < _NYSE_CLOSE


def expected_gap_seconds(
    instrument: Instrument,
    timeframe: str,
    current_bar: datetime,
) -> float:
    """
    Calculate the maximum expected gap (in seconds) between two consecutive bars
    for the given instrument/calendar, starting from `current_bar`.

    For crypto (24/7): the gap should never exceed the timeframe interval.
    For equities (NYSE): accounts for overnight, weekends, and holidays.

    Args:
        instrument: The Instrument with calendar metadata.
        timeframe: Bar timeframe string (e.g., "1Hour", "1Day").
        current_bar: Timestamp of the current bar.

    Returns:
        Maximum expected seconds until the next valid bar.
    """
    tf_seconds = {
        "1Min": 60, "5Min": 300, "15Min": 900,
        "30Min": 1800, "1Hour": 3600, "4Hour": 14400,
        "1Day": 86400, "1Week": 604800,
    }
    base_interval = tf_seconds.get(timeframe, 3600)

    if instrument.trades_24_7:
        # Crypto: next bar should be exactly one interval away
        # Allow small tolerance (1.5x) for exchange maintenance windows
        return base_interval * 1.5

    # NYSE equities: calculate based on trading sessions
    if timeframe in ("1Day", "1Week"):
        # Daily bars: next bar could be up to 4 days away (Fri -> Mon, or pre-holiday)
        return base_interval * 4
    else:
        # Intraday: overnight gap (16:00 -> 09:30 next day = 17.5 hours)
        # Weekend gap (Fri 16:00 -> Mon 09:30 = 65.5 hours)
        # Holiday weekend (Fri 16:00 -> Tue 09:30 = 89.5 hours if Mon is holiday)
        # Use 90 hours as max expected gap to cover 3-day weekends + holidays
        return 90 * 3600


# ══════════════════════════════════════════════════════════════════════════════
# Calendar-Aware Gap Detection
# ══════════════════════════════════════════════════════════════════════════════


def detect_gaps(
    df: pd.DataFrame,
    instrument: Instrument,
    timeframe: str,
) -> pd.Series:
    """
    Detect genuine missing-data gaps in bar data, respecting the instrument's
    trading calendar.

    A gap is only flagged if the time between consecutive bars exceeds what
    the instrument's calendar would explain (overnight, weekends, holidays).

    Args:
        df: DataFrame with DatetimeIndex of bar timestamps.
        instrument: Instrument metadata with calendar info.
        timeframe: Bar timeframe (e.g., "1Hour").

    Returns:
        Series of gap sizes (seconds) for bars that represent genuine data gaps.
        Empty Series if no gaps found.
    """
    if len(df) < 2:
        return pd.Series(dtype=float)

    deltas = df.index.to_series().diff().dt.total_seconds().dropna()

    # Get the per-bar threshold based on calendar
    # For efficiency, compute a single threshold rather than per-bar
    # (per-bar would be ideal but expensive for large datasets)
    tf_seconds = {
        "1Min": 60, "5Min": 300, "15Min": 900,
        "30Min": 1800, "1Hour": 3600, "4Hour": 14400,
        "1Day": 86400, "1Week": 604800,
    }
    base_interval = tf_seconds.get(timeframe, 3600)

    if instrument.trades_24_7:
        # Crypto: any gap > 1.5x interval is suspicious
        threshold = base_interval * 1.5
    else:
        # NYSE equities: use calendar-appropriate thresholds
        if timeframe in ("1Day", "1Week"):
            # Daily: allow up to 4 days (3-day weekend + holiday)
            threshold = base_interval * 4
        else:
            # Intraday: allow up to 90 hours (covers 3-day weekends + holidays)
            threshold = 90 * 3600

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
        "calendar.gaps_detected",
        symbol=instrument.symbol,
        asset_class=instrument.asset_class.value,
        calendar=instrument.exchange_calendar.value,
        timeframe=timeframe,
        gap_count=len(gaps),
        max_gap_hours=round(gaps.max() / 3600, 1),
        first_gap=str(gaps.index[0]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Calendar-Aware Data Alignment (for Portfolio Backtesting)
# ══════════════════════════════════════════════════════════════════════════════


def group_by_calendar(symbols: list[str]) -> dict[ExchangeCalendar, list[str]]:
    """
    Group symbols by their exchange calendar.

    Returns:
        Dict mapping ExchangeCalendar -> list of symbols.
        e.g., {NYSE: ["AAPL", "MSFT"], "24/7": ["BTC/USD", "ETH/USD"]}
    """
    groups: dict[ExchangeCalendar, list[str]] = {}
    for symbol in symbols:
        instrument = classify_symbol(symbol)
        cal = instrument.exchange_calendar
        if cal not in groups:
            groups[cal] = []
        groups[cal].append(symbol)
    return groups


def filter_trading_hours(
    df: pd.DataFrame,
    instrument: Instrument,
) -> pd.DataFrame:
    """
    Filter a DataFrame to only include bars during valid trading hours
    for the given instrument.

    For crypto (24/7): returns df unchanged (all hours valid).
    For equities: filters to NYSE trading hours only.

    Args:
        df: DataFrame with DatetimeIndex.
        instrument: Instrument with calendar metadata.

    Returns:
        Filtered DataFrame with only valid trading-hour bars.
    """
    if instrument.trades_24_7:
        return df

    # For NYSE equities, filter to trading hours
    # Convert index to Eastern Time for hour comparison
    try:
        import pytz
        eastern = pytz.timezone("America/New_York")
        idx_et = df.index.tz_convert(eastern) if df.index.tz else df.index.tz_localize("UTC").tz_convert(eastern)
    except (ImportError, TypeError):
        # If pytz not available or timezone issues, use the index as-is
        idx_et = df.index

    mask = pd.Series(True, index=df.index)
    # Filter weekdays
    mask &= idx_et.weekday < 5
    # Filter trading hours (9:30 - 16:00 ET)
    times = idx_et.time
    mask &= (times >= _NYSE_OPEN) & (times < _NYSE_CLOSE)

    return df[mask.values]


def get_annualization_factor(
    instrument: Instrument,
    timeframe: str,
) -> float:
    """
    Get the correct Sharpe ratio annualization factor for an instrument/timeframe.

    Args:
        instrument: Instrument with asset class metadata.
        timeframe: Bar timeframe string.

    Returns:
        sqrt(N) annualization factor where N = periods per year.
    """
    tf_factors = ANNUALIZATION_FACTORS.get(timeframe)
    if tf_factors is None:
        # Default: assume daily equity
        return np.sqrt(252)
    return tf_factors.get(instrument.asset_class, np.sqrt(252))


def get_periods_per_year(
    asset_class: AssetClass,
    timeframe: str,
) -> float:
    """
    Get the number of trading periods per year for Sharpe annualization.

    Args:
        asset_class: EQUITY or CRYPTO.
        timeframe: Bar timeframe string.

    Returns:
        Number of periods per year.
    """
    periods = {
        "1Min": {AssetClass.EQUITY: 252 * 390, AssetClass.CRYPTO: 365 * 1440},
        "5Min": {AssetClass.EQUITY: 252 * 78, AssetClass.CRYPTO: 365 * 288},
        "15Min": {AssetClass.EQUITY: 252 * 26, AssetClass.CRYPTO: 365 * 96},
        "1Hour": {AssetClass.EQUITY: 252 * 7, AssetClass.CRYPTO: 365 * 24},
        "4Hour": {AssetClass.EQUITY: 252 * 2, AssetClass.CRYPTO: 365 * 6},
        "1Day": {AssetClass.EQUITY: 252, AssetClass.CRYPTO: 365},
    }
    tf_periods = periods.get(timeframe)
    if tf_periods is None:
        return 252  # default
    return tf_periods.get(asset_class, 252)


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: is_tradable_now
# ══════════════════════════════════════════════════════════════════════════════


def is_tradable_now(instrument: Instrument, now: Optional[datetime] = None) -> bool:
    """
    Check if an instrument is currently tradable based on its calendar.

    Args:
        instrument: Instrument to check.
        now: Current datetime (defaults to utcnow). Should be timezone-aware or UTC.

    Returns:
        True if the instrument can be traded right now.
    """
    if instrument.trades_24_7:
        return True

    if now is None:
        now = datetime.now(timezone.utc)

    # Convert to Eastern Time for NYSE check
    try:
        import pytz
        eastern = pytz.timezone("America/New_York")
        now_et = now.astimezone(eastern) if now.tzinfo else pytz.utc.localize(now).astimezone(eastern)
    except ImportError:
        # Without pytz, approximate: UTC-4 (EDT) or UTC-5 (EST)
        now_et = now - timedelta(hours=4)

    return is_nyse_trading_hour(now_et)

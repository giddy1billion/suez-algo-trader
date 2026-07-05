"""
Market Calendar Management — Backward-compatible wrapper.

DEPRECATED: This module is maintained for backward compatibility only.
New code should import from `src.market` directly.

The full institutional market infrastructure is now in src/market/:
    - src/market/instruments.py   — Instrument dataclass
    - src/market/registry.py      — Symbol classification & registry
    - src/market/calendars.py     — Calendar abstractions (ABC + implementations)
    - src/market/sessions.py      — Trading session definitions
    - src/market/holidays.py      — Holiday calendar management
    - src/market/timezones.py     — UTC normalization utilities
    - src/market/annualization.py  — Annualization engine
    - src/market/gap_detection.py  — Calendar-aware gap detection
    - src/market/synchronization.py — Cross-market sync policies
    - src/market/corporate_actions.py — Corporate action model
    - src/market/constraints.py   — Market trading constraints
    - src/market/exchanges.py     — Exchange definitions

This wrapper preserves the original API signatures so that existing imports
continue to work without modification.
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
# Backward-Compatible Enums & Types
# ══════════════════════════════════════════════════════════════════════════════


class AssetClass(str, Enum):
    """Supported asset classes."""
    EQUITY = "equity"
    CRYPTO = "crypto"


class ExchangeCalendar(str, Enum):
    """Supported exchange calendars."""
    NYSE = "NYSE"
    TWENTY_FOUR_SEVEN = "24/7"


@dataclass(frozen=True)
class Instrument:
    """
    Market instrument with calendar metadata.

    DEPRECATED: Use src.market.instruments.Instrument for new code.
    This class is preserved for backward compatibility.
    """
    symbol: str
    asset_class: AssetClass
    exchange_calendar: ExchangeCalendar
    timezone: str = "UTC"

    @property
    def is_crypto(self) -> bool:
        return self.asset_class == AssetClass.CRYPTO

    @property
    def is_equity(self) -> bool:
        return self.asset_class == AssetClass.EQUITY

    @property
    def trades_24_7(self) -> bool:
        return self.exchange_calendar == ExchangeCalendar.TWENTY_FOUR_SEVEN


# ══════════════════════════════════════════════════════════════════════════════
# Backward-Compatible API (delegates to src.market where appropriate)
# ══════════════════════════════════════════════════════════════════════════════

# NYSE holidays — kept for backward compatibility with direct access
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

_NYSE_OPEN = time(9, 30)
_NYSE_CLOSE = time(16, 0)

ANNUALIZATION_FACTORS = {
    "1Min": {AssetClass.EQUITY: np.sqrt(252 * 390), AssetClass.CRYPTO: np.sqrt(365 * 1440)},
    "5Min": {AssetClass.EQUITY: np.sqrt(252 * 78), AssetClass.CRYPTO: np.sqrt(365 * 288)},
    "15Min": {AssetClass.EQUITY: np.sqrt(252 * 26), AssetClass.CRYPTO: np.sqrt(365 * 96)},
    "1Hour": {AssetClass.EQUITY: np.sqrt(252 * 7), AssetClass.CRYPTO: np.sqrt(365 * 24)},
    "4Hour": {AssetClass.EQUITY: np.sqrt(252 * 2), AssetClass.CRYPTO: np.sqrt(365 * 6)},
    "1Day": {AssetClass.EQUITY: np.sqrt(252), AssetClass.CRYPTO: np.sqrt(365)},
}


def classify_symbol(symbol: str) -> Instrument:
    """
    Classify a symbol into an Instrument with proper calendar metadata.

    DEPRECATED: Use src.market.registry.classify_symbol() for new code.
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


def is_nyse_trading_day(dt: datetime) -> bool:
    """Check if a given date is a NYSE trading day (weekday, not a holiday)."""
    if dt.weekday() >= 5:
        return False
    dt_date = datetime(dt.year, dt.month, dt.day)
    if dt_date in _NYSE_HOLIDAYS_2024_2027:
        return False
    return True


def is_nyse_trading_hour(dt: datetime) -> bool:
    """Check if a given datetime falls within NYSE regular trading hours."""
    if not is_nyse_trading_day(dt):
        return False
    t = dt.time()
    return _NYSE_OPEN <= t < _NYSE_CLOSE


def expected_gap_seconds(
    instrument: Instrument,
    timeframe: str,
    current_bar: datetime,
) -> float:
    """Calculate the maximum expected gap between consecutive bars."""
    tf_seconds = {
        "1Min": 60, "5Min": 300, "15Min": 900,
        "30Min": 1800, "1Hour": 3600, "4Hour": 14400,
        "1Day": 86400, "1Week": 604800,
    }
    base_interval = tf_seconds.get(timeframe, 3600)

    if instrument.trades_24_7:
        return base_interval * 1.5

    if timeframe in ("1Day", "1Week"):
        return base_interval * 4
    else:
        return 90 * 3600


def detect_gaps(
    df: pd.DataFrame,
    instrument: Instrument,
    timeframe: str,
) -> pd.Series:
    """Detect genuine missing-data gaps in bar data."""
    if len(df) < 2:
        return pd.Series(dtype=float)

    deltas = df.index.to_series().diff().dt.total_seconds().dropna()

    tf_seconds = {
        "1Min": 60, "5Min": 300, "15Min": 900,
        "30Min": 1800, "1Hour": 3600, "4Hour": 14400,
        "1Day": 86400, "1Week": 604800,
    }
    base_interval = tf_seconds.get(timeframe, 3600)

    if instrument.trades_24_7:
        threshold = base_interval * 1.5
    else:
        if timeframe in ("1Day", "1Week"):
            threshold = base_interval * 4
        else:
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


def group_by_calendar(symbols: list[str]) -> dict[ExchangeCalendar, list[str]]:
    """Group symbols by their exchange calendar."""
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
    """Filter a DataFrame to only include bars during valid trading hours."""
    if instrument.trades_24_7:
        return df

    try:
        import pytz
        eastern = pytz.timezone("America/New_York")
        idx_et = df.index.tz_convert(eastern) if df.index.tz else df.index.tz_localize("UTC").tz_convert(eastern)
    except (ImportError, TypeError):
        idx_et = df.index

    mask = pd.Series(True, index=df.index)
    mask &= idx_et.weekday < 5
    times = idx_et.time
    mask &= (times >= _NYSE_OPEN) & (times < _NYSE_CLOSE)

    return df[mask.values]


def get_annualization_factor(
    instrument: Instrument,
    timeframe: str,
) -> float:
    """Get the correct Sharpe ratio annualization factor."""
    tf_factors = ANNUALIZATION_FACTORS.get(timeframe)
    if tf_factors is None:
        return np.sqrt(252)
    return tf_factors.get(instrument.asset_class, np.sqrt(252))


def get_periods_per_year(
    asset_class: AssetClass,
    timeframe: str,
) -> float:
    """Get the number of trading periods per year."""
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
        return 252
    return tf_periods.get(asset_class, 252)


def is_tradable_now(instrument: Instrument, now: Optional[datetime] = None) -> bool:
    """Check if an instrument is currently tradable based on its calendar."""
    if instrument.trades_24_7:
        return True

    if now is None:
        now = datetime.now(timezone.utc)

    try:
        import pytz
        eastern = pytz.timezone("America/New_York")
        now_et = now.astimezone(eastern) if now.tzinfo else pytz.utc.localize(now).astimezone(eastern)
    except ImportError:
        now_et = now - timedelta(hours=4)

    return is_nyse_trading_hour(now_et)

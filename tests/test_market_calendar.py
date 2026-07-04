"""
Tests for the Market Calendar subsystem.

Validates:
- Symbol classification (crypto vs equity)
- Calendar-aware gap detection
- NYSE trading day/hour checks
- Annualization factors
- Data alignment behavior
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.market_calendar import (
    AssetClass,
    ExchangeCalendar,
    Instrument,
    classify_symbol,
    classify_symbols,
    detect_gaps,
    expected_gap_seconds,
    filter_trading_hours,
    get_annualization_factor,
    get_periods_per_year,
    group_by_calendar,
    is_nyse_trading_day,
    is_nyse_trading_hour,
    is_tradable_now,
)


# ──────────────────────────────────────────────────────────────────────────────
# Symbol Classification
# ──────────────────────────────────────────────────────────────────────────────


class TestClassifySymbol:
    """Test symbol -> Instrument classification."""

    def test_equity_symbol(self):
        inst = classify_symbol("AAPL")
        assert inst.symbol == "AAPL"
        assert inst.asset_class == AssetClass.EQUITY
        assert inst.exchange_calendar == ExchangeCalendar.NYSE
        assert inst.is_equity is True
        assert inst.is_crypto is False
        assert inst.trades_24_7 is False

    def test_crypto_symbol(self):
        inst = classify_symbol("BTC/USD")
        assert inst.symbol == "BTC/USD"
        assert inst.asset_class == AssetClass.CRYPTO
        assert inst.exchange_calendar == ExchangeCalendar.TWENTY_FOUR_SEVEN
        assert inst.is_crypto is True
        assert inst.is_equity is False
        assert inst.trades_24_7 is True

    def test_crypto_variants(self):
        for sym in ["ETH/USD", "SOL/USD", "AAVE/USD", "ADA/USD"]:
            inst = classify_symbol(sym)
            assert inst.is_crypto
            assert inst.trades_24_7

    def test_equity_variants(self):
        for sym in ["MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]:
            inst = classify_symbol(sym)
            assert inst.is_equity
            assert not inst.trades_24_7

    def test_classify_symbols_batch(self):
        symbols = ["AAPL", "BTC/USD", "MSFT", "ETH/USD"]
        instruments = classify_symbols(symbols)
        assert len(instruments) == 4
        assert instruments["AAPL"].is_equity
        assert instruments["BTC/USD"].is_crypto


# ──────────────────────────────────────────────────────────────────────────────
# NYSE Calendar Logic
# ──────────────────────────────────────────────────────────────────────────────


class TestNYSECalendar:
    """Test NYSE trading day and hour detection."""

    def test_weekday_is_trading_day(self):
        # Wednesday, January 8, 2025
        assert is_nyse_trading_day(datetime(2025, 1, 8)) is True

    def test_saturday_not_trading_day(self):
        # Saturday, January 4, 2025
        assert is_nyse_trading_day(datetime(2025, 1, 4)) is False

    def test_sunday_not_trading_day(self):
        # Sunday, January 5, 2025
        assert is_nyse_trading_day(datetime(2025, 1, 5)) is False

    def test_holiday_not_trading_day(self):
        # MLK Day 2025 (January 20)
        assert is_nyse_trading_day(datetime(2025, 1, 20)) is False
        # Christmas 2025
        assert is_nyse_trading_day(datetime(2025, 12, 25)) is False

    def test_trading_hour_valid(self):
        # 10:00 AM on a Wednesday
        assert is_nyse_trading_hour(datetime(2025, 1, 8, 10, 0)) is True

    def test_trading_hour_at_open(self):
        # Exactly at 9:30
        assert is_nyse_trading_hour(datetime(2025, 1, 8, 9, 30)) is True

    def test_before_open_not_trading(self):
        # 9:29 AM
        assert is_nyse_trading_hour(datetime(2025, 1, 8, 9, 29)) is False

    def test_at_close_not_trading(self):
        # 16:00 (close is exclusive)
        assert is_nyse_trading_hour(datetime(2025, 1, 8, 16, 0)) is False

    def test_after_close_not_trading(self):
        # 17:00
        assert is_nyse_trading_hour(datetime(2025, 1, 8, 17, 0)) is False


# ──────────────────────────────────────────────────────────────────────────────
# Gap Detection
# ──────────────────────────────────────────────────────────────────────────────


class TestGapDetection:
    """Test calendar-aware gap detection."""

    def _make_hourly_df(self, timestamps):
        """Helper to create a minimal OHLCV DataFrame from timestamps."""
        n = len(timestamps)
        return pd.DataFrame(
            {"open": np.ones(n), "high": np.ones(n), "low": np.ones(n),
             "close": np.ones(n), "volume": np.ones(n)},
            index=pd.DatetimeIndex(timestamps),
        )

    def test_crypto_no_gaps_continuous(self):
        """Continuous hourly crypto bars should have no gaps."""
        start = datetime(2025, 6, 1, 0, 0)
        timestamps = [start + timedelta(hours=i) for i in range(48)]
        df = self._make_hourly_df(timestamps)
        instrument = classify_symbol("BTC/USD")
        gaps = detect_gaps(df, instrument, "1Hour")
        assert len(gaps) == 0

    def test_crypto_detects_missing_bars(self):
        """A 5-hour gap in crypto should be flagged."""
        start = datetime(2025, 6, 1, 0, 0)
        timestamps = [start + timedelta(hours=i) for i in range(10)]
        # Remove bars 3-7 (creates a gap from hour 2 -> hour 8 = 6 hours)
        timestamps = timestamps[:3] + timestamps[8:]
        df = self._make_hourly_df(timestamps)
        instrument = classify_symbol("ETH/USD")
        gaps = detect_gaps(df, instrument, "1Hour")
        assert len(gaps) == 1
        # Gap is from bar at hour 2 to bar at hour 8 = 6 hours
        assert gaps.iloc[0] == 6 * 3600

    def test_equity_weekend_not_flagged(self):
        """NYSE equity weekend gap (Fri 16:00 -> Mon 09:30) should NOT be flagged."""
        # Friday 15:00, then Monday 10:00 — 67 hours gap
        timestamps = [
            datetime(2025, 6, 6, 15, 0),   # Friday
            datetime(2025, 6, 9, 10, 0),   # Monday
        ]
        df = self._make_hourly_df(timestamps)
        instrument = classify_symbol("AAPL")
        gaps = detect_gaps(df, instrument, "1Hour")
        assert len(gaps) == 0  # Weekend gap is expected

    def test_equity_overnight_not_flagged(self):
        """NYSE overnight gap should NOT be flagged."""
        timestamps = [
            datetime(2025, 6, 9, 15, 0),   # Monday 3pm
            datetime(2025, 6, 10, 10, 0),  # Tuesday 10am — 19 hours
        ]
        df = self._make_hourly_df(timestamps)
        instrument = classify_symbol("MSFT")
        gaps = detect_gaps(df, instrument, "1Hour")
        assert len(gaps) == 0

    def test_equity_multi_day_gap_flagged(self):
        """A gap exceeding 90 hours (> 3-day weekend) should be flagged for equities."""
        timestamps = [
            datetime(2025, 6, 5, 15, 0),   # Thursday
            datetime(2025, 6, 10, 10, 0),  # Tuesday next week — ~115 hours
        ]
        df = self._make_hourly_df(timestamps)
        instrument = classify_symbol("AAPL")
        gaps = detect_gaps(df, instrument, "1Hour")
        assert len(gaps) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Calendar Grouping
# ──────────────────────────────────────────────────────────────────────────────


class TestGroupByCalendar:
    """Test symbol grouping by exchange calendar."""

    def test_mixed_symbols(self):
        symbols = ["AAPL", "MSFT", "BTC/USD", "ETH/USD", "NVDA", "SOL/USD"]
        groups = group_by_calendar(symbols)
        assert ExchangeCalendar.NYSE in groups
        assert ExchangeCalendar.TWENTY_FOUR_SEVEN in groups
        assert set(groups[ExchangeCalendar.NYSE]) == {"AAPL", "MSFT", "NVDA"}
        assert set(groups[ExchangeCalendar.TWENTY_FOUR_SEVEN]) == {"BTC/USD", "ETH/USD", "SOL/USD"}

    def test_all_crypto(self):
        symbols = ["BTC/USD", "ETH/USD"]
        groups = group_by_calendar(symbols)
        assert ExchangeCalendar.NYSE not in groups
        assert len(groups[ExchangeCalendar.TWENTY_FOUR_SEVEN]) == 2

    def test_all_equity(self):
        symbols = ["AAPL", "MSFT"]
        groups = group_by_calendar(symbols)
        assert ExchangeCalendar.TWENTY_FOUR_SEVEN not in groups
        assert len(groups[ExchangeCalendar.NYSE]) == 2


# ──────────────────────────────────────────────────────────────────────────────
# Annualization Factors
# ──────────────────────────────────────────────────────────────────────────────


class TestAnnualization:
    """Test correct annualization factors per asset class."""

    def test_equity_daily(self):
        inst = classify_symbol("AAPL")
        factor = get_annualization_factor(inst, "1Day")
        assert abs(factor - np.sqrt(252)) < 0.01

    def test_crypto_daily(self):
        inst = classify_symbol("BTC/USD")
        factor = get_annualization_factor(inst, "1Day")
        assert abs(factor - np.sqrt(365)) < 0.01

    def test_crypto_hourly_higher_than_equity(self):
        crypto = classify_symbol("BTC/USD")
        equity = classify_symbol("AAPL")
        crypto_factor = get_annualization_factor(crypto, "1Hour")
        equity_factor = get_annualization_factor(equity, "1Hour")
        # Crypto has more trading hours/year -> higher annualization
        assert crypto_factor > equity_factor

    def test_periods_per_year(self):
        assert get_periods_per_year(AssetClass.EQUITY, "1Day") == 252
        assert get_periods_per_year(AssetClass.CRYPTO, "1Day") == 365
        assert get_periods_per_year(AssetClass.EQUITY, "1Hour") == 252 * 7
        assert get_periods_per_year(AssetClass.CRYPTO, "1Hour") == 365 * 24


# ──────────────────────────────────────────────────────────────────────────────
# Tradability Check
# ──────────────────────────────────────────────────────────────────────────────


class TestTradability:
    """Test is_tradable_now logic."""

    def test_crypto_always_tradable(self):
        inst = classify_symbol("BTC/USD")
        # Any time, any day
        saturday_midnight = datetime(2025, 6, 7, 0, 0)
        assert is_tradable_now(inst, saturday_midnight) is True

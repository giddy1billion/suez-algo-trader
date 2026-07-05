"""
Comprehensive tests for the src/market institutional infrastructure package.

Tests cover:
- Calendar abstractions (NYSE, Crypto247, NASDAQ, CME, Forex)
- Instrument registry and classification
- Trading sessions
- Timezone normalization
- Annualization engine
- Gap detection
- Synchronization policies
- Corporate actions
- Market constraints
- Backward compatibility
"""

from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.market.instruments import AssetClass, Currency, FeeSchedule, Instrument
from src.market.registry import (
    InstrumentRegistry,
    classify_symbol,
    classify_symbols,
    get_default_registry,
)
from src.market.calendars import (
    Crypto247Calendar,
    MarketCalendar,
    NYSECalendar,
    NASDAQCalendar,
    CMECalendar,
    ForexCalendar,
    get_calendar,
    register_calendar,
)
from src.market.sessions import (
    NYSE_AFTER_HOURS,
    NYSE_PRE_MARKET,
    NYSE_REGULAR,
    SessionType,
    TradingSession,
    get_current_session_type,
    get_nyse_sessions,
)
from src.market.holidays import NoHolidayCalendar, NYSEHolidayCalendar
from src.market.timezones import (
    NaiveDatetimeError,
    ensure_utc,
    is_utc,
    normalize_index_to_utc,
    reject_naive_datetime,
    to_exchange_time,
    utc_now,
)
from src.market.annualization import (
    bars_per_year,
    get_annualization_factor,
    get_periods_per_year,
)
from src.market.gap_detection import detect_gaps, expected_gap_seconds, log_gap_report
from src.market.synchronization import (
    SyncMode,
    SynchronizationPolicy,
    align_multi_asset_data,
    group_by_calendar,
)
from src.market.corporate_actions import (
    CorporateAction,
    CorporateActionRegistry,
    CorporateActionType,
)
from src.market.constraints import MarketConstraints, get_constraints
from src.market.exchanges import Exchange, ExchangeID, EXCHANGES, get_exchange


# ══════════════════════════════════════════════════════════════════════════════
# Calendar Abstraction Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestNYSECalendar:
    """Test NYSE calendar implementation."""

    def setup_method(self):
        self.cal = NYSECalendar()

    def test_name(self):
        assert self.cal.name == "NYSE"

    def test_timezone(self):
        assert self.cal.timezone == "America/New_York"

    def test_weekday_is_trading_day(self):
        assert self.cal.is_trading_day(date(2025, 1, 8)) is True  # Wednesday

    def test_saturday_not_trading_day(self):
        assert self.cal.is_trading_day(date(2025, 1, 4)) is False

    def test_sunday_not_trading_day(self):
        assert self.cal.is_trading_day(date(2025, 1, 5)) is False

    def test_holiday_not_trading_day(self):
        assert self.cal.is_trading_day(date(2025, 1, 20)) is False  # MLK Day
        assert self.cal.is_trading_day(date(2025, 12, 25)) is False  # Christmas

    def test_trading_minutes_per_day(self):
        assert self.cal.trading_minutes_per_day() == 390

    def test_trading_days_per_year(self):
        assert self.cal.trading_days_per_year() == 252

    def test_sessions(self):
        sessions = self.cal.sessions()
        assert len(sessions) == 3
        assert sessions[0].session_type == SessionType.PRE_MARKET
        assert sessions[1].session_type == SessionType.REGULAR
        assert sessions[2].session_type == SessionType.AFTER_HOURS

    def test_max_expected_gap_daily(self):
        gap = self.cal.max_expected_gap_seconds("1Day")
        assert gap == 86400 * 4  # 4 days max

    def test_max_expected_gap_hourly(self):
        gap = self.cal.max_expected_gap_seconds("1Hour")
        assert gap == 90 * 3600  # 90 hours

    def test_annualization_factor_daily(self):
        factor = self.cal.annualization_factor("1Day")
        assert abs(factor - np.sqrt(252)) < 0.01

    def test_periods_per_year_hourly(self):
        periods = self.cal.periods_per_year("1Hour")
        # 252 days * 390 min/day / 60 min/bar = 1638
        assert abs(periods - (252 * 390 / 60)) < 1


class TestCrypto247Calendar:
    """Test 24/7 crypto calendar."""

    def setup_method(self):
        self.cal = Crypto247Calendar()

    def test_name(self):
        assert self.cal.name == "24/7"

    def test_timezone(self):
        assert self.cal.timezone == "UTC"

    def test_every_day_is_trading_day(self):
        # Test Saturday and Sunday
        assert self.cal.is_trading_day(date(2025, 6, 7)) is True  # Saturday
        assert self.cal.is_trading_day(date(2025, 6, 8)) is True  # Sunday
        assert self.cal.is_trading_day(date(2025, 12, 25)) is True  # Christmas

    def test_always_trading_time(self):
        dt = datetime(2025, 6, 7, 3, 0, tzinfo=timezone.utc)  # Saturday 3am
        assert self.cal.is_trading_time(dt) is True

    def test_always_open_session(self):
        dt = datetime(2025, 6, 7, 3, 0, tzinfo=timezone.utc)
        assert self.cal.current_session(dt) == SessionType.ALWAYS_OPEN

    def test_trading_minutes_per_day(self):
        assert self.cal.trading_minutes_per_day() == 1440

    def test_trading_days_per_year(self):
        assert self.cal.trading_days_per_year() == 365

    def test_max_expected_gap(self):
        gap = self.cal.max_expected_gap_seconds("1Hour")
        assert gap == 3600 * 1.5

    def test_annualization_factor_daily(self):
        factor = self.cal.annualization_factor("1Day")
        assert abs(factor - np.sqrt(365)) < 0.01

    def test_periods_per_year_hourly(self):
        periods = self.cal.periods_per_year("1Hour")
        assert abs(periods - (365 * 24)) < 1


class TestCalendarRegistry:
    """Test calendar lookup by name."""

    def test_get_nyse(self):
        cal = get_calendar("NYSE")
        assert isinstance(cal, NYSECalendar)

    def test_get_crypto(self):
        cal = get_calendar("24/7")
        assert isinstance(cal, Crypto247Calendar)

    def test_get_nasdaq(self):
        cal = get_calendar("NASDAQ")
        assert isinstance(cal, NASDAQCalendar)

    def test_get_cme(self):
        cal = get_calendar("CME")
        assert isinstance(cal, CMECalendar)

    def test_get_forex(self):
        cal = get_calendar("FOREX")
        assert isinstance(cal, ForexCalendar)

    def test_unknown_calendar_raises(self):
        with pytest.raises(ValueError, match="Unknown calendar"):
            get_calendar("NONEXISTENT")

    def test_register_custom_calendar(self):
        class TestCalendar(Crypto247Calendar):
            @property
            def name(self) -> str:
                return "TEST"

        register_calendar("TEST", TestCalendar)
        cal = get_calendar("TEST")
        assert cal.name == "TEST"


# ══════════════════════════════════════════════════════════════════════════════
# Instrument Registry Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestInstrumentRegistry:
    """Test the instrument registry."""

    def setup_method(self):
        self.registry = InstrumentRegistry()

    def test_register_and_get(self):
        inst = Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)
        self.registry.register(inst)
        assert self.registry.get("AAPL") == inst

    def test_auto_classify_crypto_slash(self):
        inst = self.registry.get("BTC/USD")
        assert inst.asset_class == AssetClass.CRYPTO
        assert inst.calendar == "24/7"

    def test_auto_classify_crypto_no_slash(self):
        inst = self.registry.get("BTCUSD")
        assert inst.asset_class == AssetClass.CRYPTO

    def test_auto_classify_crypto_usdt(self):
        inst = self.registry.get("ETHUSDT")
        assert inst.asset_class == AssetClass.CRYPTO

    def test_auto_classify_equity(self):
        inst = self.registry.get("AAPL")
        assert inst.asset_class == AssetClass.EQUITY
        assert inst.calendar == "NYSE"

    def test_broker_source_tracking(self):
        inst = Instrument(symbol="TSLA", asset_class=AssetClass.EQUITY)
        self.registry.register(inst, source="broker")
        assert self.registry.is_broker_populated("TSLA")

    def test_by_asset_class(self):
        self.registry.get("AAPL")
        self.registry.get("MSFT")
        self.registry.get("BTC/USD")
        equities = self.registry.by_asset_class(AssetClass.EQUITY)
        assert len(equities) == 2

    def test_by_calendar(self):
        self.registry.get("AAPL")
        self.registry.get("BTC/USD")
        crypto = self.registry.by_calendar("24/7")
        assert len(crypto) == 1


class TestClassifySymbol:
    """Test module-level classify functions."""

    def test_equity(self):
        inst = classify_symbol("AAPL")
        assert inst.asset_class == AssetClass.EQUITY
        assert inst.is_equity

    def test_crypto_with_slash(self):
        inst = classify_symbol("BTC/USD")
        assert inst.asset_class == AssetClass.CRYPTO
        assert inst.is_crypto
        assert inst.trades_24_7

    def test_crypto_without_slash(self):
        inst = classify_symbol("BTCUSD")
        assert inst.asset_class == AssetClass.CRYPTO

    def test_classify_symbols_batch(self):
        result = classify_symbols(["AAPL", "BTC/USD", "MSFT"])
        assert len(result) == 3
        assert result["AAPL"].is_equity
        assert result["BTC/USD"].is_crypto


# ══════════════════════════════════════════════════════════════════════════════
# Trading Session Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestTradingSessions:
    """Test trading session definitions and lookup."""

    def test_nyse_regular_session_duration(self):
        assert NYSE_REGULAR.duration_minutes == 390  # 6.5 hours

    def test_nyse_pre_market_duration(self):
        assert NYSE_PRE_MARKET.duration_minutes == 330  # 5.5 hours

    def test_nyse_after_hours_duration(self):
        assert NYSE_AFTER_HOURS.duration_minutes == 240  # 4 hours

    def test_session_type_during_regular(self):
        sessions = get_nyse_sessions()
        result = get_current_session_type(time(10, 0), sessions)
        assert result == SessionType.REGULAR

    def test_session_type_during_pre_market(self):
        sessions = get_nyse_sessions()
        result = get_current_session_type(time(5, 0), sessions)
        assert result == SessionType.PRE_MARKET

    def test_session_type_during_after_hours(self):
        sessions = get_nyse_sessions()
        result = get_current_session_type(time(17, 0), sessions)
        assert result == SessionType.AFTER_HOURS

    def test_session_type_closed(self):
        sessions = get_nyse_sessions()
        result = get_current_session_type(time(21, 0), sessions)
        assert result == SessionType.CLOSED


# ══════════════════════════════════════════════════════════════════════════════
# Holiday Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestHolidays:
    """Test holiday calendar implementations."""

    def test_nyse_holiday(self):
        cal = NYSEHolidayCalendar()
        assert cal.is_holiday(date(2025, 12, 25)) is True

    def test_nyse_non_holiday(self):
        cal = NYSEHolidayCalendar()
        assert cal.is_holiday(date(2025, 6, 10)) is False

    def test_nyse_holidays_in_range(self):
        cal = NYSEHolidayCalendar()
        holidays = cal.holidays_in_range(date(2025, 1, 1), date(2025, 12, 31))
        assert len(holidays) == 10  # 10 NYSE holidays per year

    def test_no_holiday_calendar(self):
        cal = NoHolidayCalendar()
        assert cal.is_holiday(date(2025, 12, 25)) is False
        assert len(cal.holidays_in_range(date(2025, 1, 1), date(2025, 12, 31))) == 0

    def test_next_holiday(self):
        cal = NYSEHolidayCalendar()
        next_h = cal.next_holiday(date(2025, 6, 1))
        assert next_h == date(2025, 6, 19)  # Juneteenth


# ══════════════════════════════════════════════════════════════════════════════
# Timezone Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestTimezones:
    """Test timezone normalization utilities."""

    def test_ensure_utc_already_utc(self):
        dt = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
        result = ensure_utc(dt)
        assert result == dt

    def test_ensure_utc_naive_raises(self):
        dt = datetime(2025, 6, 1, 12, 0)
        with pytest.raises(NaiveDatetimeError):
            ensure_utc(dt)

    def test_ensure_utc_converts_from_other_tz(self):
        import pytz
        eastern = pytz.timezone("America/New_York")
        dt = eastern.localize(datetime(2025, 6, 1, 12, 0))
        result = ensure_utc(dt)
        assert result.hour == 16  # EDT is UTC-4

    def test_reject_naive_raises(self):
        with pytest.raises(NaiveDatetimeError):
            reject_naive_datetime(datetime(2025, 1, 1))

    def test_reject_naive_passes_aware(self):
        # Should not raise
        reject_naive_datetime(datetime(2025, 1, 1, tzinfo=timezone.utc))

    def test_normalize_index_naive(self):
        idx = pd.DatetimeIndex([datetime(2025, 1, 1), datetime(2025, 1, 2)])
        df = pd.DataFrame({"a": [1, 2]}, index=idx)
        result = normalize_index_to_utc(df)
        assert str(result.index.tz) == "UTC"

    def test_normalize_index_already_utc(self):
        idx = pd.DatetimeIndex(
            [datetime(2025, 1, 1), datetime(2025, 1, 2)]
        ).tz_localize("UTC")
        df = pd.DataFrame({"a": [1, 2]}, index=idx)
        result = normalize_index_to_utc(df)
        assert str(result.index.tz) == "UTC"

    def test_is_utc(self):
        assert is_utc(datetime(2025, 1, 1, tzinfo=timezone.utc)) is True
        assert is_utc(datetime(2025, 1, 1)) is False

    def test_utc_now(self):
        now = utc_now()
        assert now.tzinfo is not None
        assert is_utc(now)

    def test_to_exchange_time(self):
        dt = datetime(2025, 6, 1, 16, 0, tzinfo=timezone.utc)
        local = to_exchange_time(dt, "America/New_York")
        assert local.hour == 12  # EDT = UTC-4


# ══════════════════════════════════════════════════════════════════════════════
# Annualization Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestAnnualization:
    """Test the annualization engine."""

    def test_equity_daily(self):
        inst = classify_symbol("AAPL")
        factor = get_annualization_factor(inst, "1Day")
        assert abs(factor - np.sqrt(252)) < 0.01

    def test_crypto_daily(self):
        inst = classify_symbol("BTC/USD")
        factor = get_annualization_factor(inst, "1Day")
        assert abs(factor - np.sqrt(365)) < 0.01

    def test_crypto_hourly_higher(self):
        crypto = classify_symbol("BTC/USD")
        equity = classify_symbol("AAPL")
        assert get_annualization_factor(crypto, "1Hour") > get_annualization_factor(equity, "1Hour")

    def test_periods_per_year_equity_daily(self):
        periods = get_periods_per_year(AssetClass.EQUITY, "1Day")
        assert periods == 252

    def test_periods_per_year_crypto_daily(self):
        periods = get_periods_per_year(AssetClass.CRYPTO, "1Day")
        assert periods == 365

    def test_bars_per_year_crypto_hourly(self):
        cal = Crypto247Calendar()
        result = bars_per_year(cal, "1Hour")
        assert abs(result - (365 * 24)) < 1

    def test_bars_per_year_nyse_daily(self):
        cal = NYSECalendar()
        result = bars_per_year(cal, "1Day")
        assert result == 252


# ══════════════════════════════════════════════════════════════════════════════
# Gap Detection Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestGapDetection:
    """Test calendar-aware gap detection."""

    def _make_df(self, timestamps):
        n = len(timestamps)
        return pd.DataFrame(
            {"close": np.ones(n)},
            index=pd.DatetimeIndex(timestamps),
        )

    def test_crypto_no_gaps(self):
        start = datetime(2025, 6, 1, 0, 0)
        timestamps = [start + timedelta(hours=i) for i in range(48)]
        df = self._make_df(timestamps)
        inst = classify_symbol("BTC/USD")
        gaps = detect_gaps(df, inst, "1Hour")
        assert len(gaps) == 0

    def test_crypto_detects_gap(self):
        start = datetime(2025, 6, 1, 0, 0)
        timestamps = [start + timedelta(hours=i) for i in range(10)]
        timestamps = timestamps[:3] + timestamps[8:]  # Remove 5 bars
        df = self._make_df(timestamps)
        inst = classify_symbol("BTC/USD")
        gaps = detect_gaps(df, inst, "1Hour")
        assert len(gaps) == 1

    def test_equity_weekend_not_flagged(self):
        timestamps = [
            datetime(2025, 6, 6, 15, 0),  # Friday
            datetime(2025, 6, 9, 10, 0),  # Monday
        ]
        df = self._make_df(timestamps)
        inst = classify_symbol("AAPL")
        gaps = detect_gaps(df, inst, "1Hour")
        assert len(gaps) == 0

    def test_expected_gap_seconds_backward_compat(self):
        inst = classify_symbol("BTC/USD")
        gap = expected_gap_seconds(inst, "1Hour")
        assert gap == 3600 * 1.5


# ══════════════════════════════════════════════════════════════════════════════
# Synchronization Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestSynchronization:
    """Test cross-market synchronization policies."""

    def test_group_by_calendar(self):
        symbols = ["AAPL", "MSFT", "BTC/USD", "ETH/USD"]
        groups = group_by_calendar(symbols)
        assert "NYSE" in groups
        assert "24/7" in groups
        assert set(groups["NYSE"]) == {"AAPL", "MSFT"}
        assert set(groups["24/7"]) == {"BTC/USD", "ETH/USD"}

    def test_strict_mode_no_forward_fill(self):
        policy = SynchronizationPolicy(mode=SyncMode.STRICT)
        inst = classify_symbol("AAPL")
        assert policy.should_forward_fill(inst) is False

    def test_carry_forward_mode(self):
        policy = SynchronizationPolicy(mode=SyncMode.CARRY_FORWARD)
        inst = classify_symbol("AAPL")
        assert policy.should_forward_fill(inst) is True

    def test_fill_limit_equity(self):
        policy = SynchronizationPolicy(max_forward_fill=7)
        inst = classify_symbol("AAPL")
        assert policy.get_fill_limit(inst) == 7

    def test_fill_limit_crypto_unlimited(self):
        policy = SynchronizationPolicy()
        inst = classify_symbol("BTC/USD")
        assert policy.get_fill_limit(inst) is None

    def test_align_multi_asset_data(self):
        # Create crypto and equity data with different timestamps
        crypto_idx = pd.DatetimeIndex([
            datetime(2025, 6, 7, i, 0) for i in range(24)  # Saturday
        ])
        equity_idx = pd.DatetimeIndex([
            datetime(2025, 6, 6, 10 + i, 0) for i in range(6)  # Friday
        ])
        data = {
            "BTC/USD": pd.DataFrame({"close": np.ones(24)}, index=crypto_idx),
            "AAPL": pd.DataFrame({"close": np.ones(6)}, index=equity_idx),
        }
        common_idx, aligned = align_multi_asset_data(data)
        assert len(common_idx) > 0
        assert "BTC/USD" in aligned
        assert "AAPL" in aligned


# ══════════════════════════════════════════════════════════════════════════════
# Corporate Actions Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCorporateActions:
    """Test corporate action model."""

    def test_stock_split(self):
        action = CorporateAction(
            symbol="AAPL",
            action_type=CorporateActionType.STOCK_SPLIT,
            effective_date=date(2020, 8, 31),
            ratio=4.0,
            description="4-for-1 stock split",
        )
        assert action.is_split
        assert action.price_adjustment_factor == 0.25

    def test_dividend(self):
        action = CorporateAction(
            symbol="AAPL",
            action_type=CorporateActionType.CASH_DIVIDEND,
            effective_date=date(2025, 5, 10),
            amount=0.25,
        )
        assert action.is_dividend
        assert not action.is_split

    def test_registry_lookup(self):
        registry = CorporateActionRegistry()
        registry.add(CorporateAction(
            symbol="AAPL",
            action_type=CorporateActionType.STOCK_SPLIT,
            effective_date=date(2020, 8, 31),
            ratio=4.0,
        ))
        registry.add(CorporateAction(
            symbol="AAPL",
            action_type=CorporateActionType.CASH_DIVIDEND,
            effective_date=date(2025, 5, 10),
            amount=0.25,
        ))
        actions = registry.get_actions("AAPL", date(2020, 1, 1), date(2021, 1, 1))
        assert len(actions) == 1
        assert actions[0].ratio == 4.0


# ══════════════════════════════════════════════════════════════════════════════
# Market Constraints Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestMarketConstraints:
    """Test market constraint validation."""

    def test_equity_constraints(self):
        inst = classify_symbol("AAPL")
        constraints = get_constraints(inst)
        assert constraints.shortable is True
        assert constraints.marginable is True
        assert constraints.fractional is False
        assert constraints.day_trade_restricted is True

    def test_crypto_constraints(self):
        inst = classify_symbol("BTC/USD")
        constraints = get_constraints(inst)
        assert constraints.shortable is False
        assert constraints.fractional is True
        assert constraints.settlement_days == 0

    def test_validate_quantity_lot_size(self):
        constraints = MarketConstraints(lot_size=1.0, fractional=False)
        assert constraints.validate_quantity(10.0) is True
        assert constraints.validate_quantity(10.5) is False

    def test_round_quantity(self):
        constraints = MarketConstraints(lot_size=0.01, fractional=False)
        assert constraints.round_quantity(1.03) == 1.03
        assert constraints.round_quantity(10.0) == 10.0

    def test_validate_price_tick(self):
        constraints = MarketConstraints(min_tick=0.01)
        assert constraints.validate_price(100.01) is True
        assert constraints.validate_price(100.005) is False

    def test_round_price(self):
        constraints = MarketConstraints(min_tick=0.05)
        assert constraints.round_price(100.03) == 100.05


# ══════════════════════════════════════════════════════════════════════════════
# Exchange Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestExchanges:
    """Test exchange definitions."""

    def test_nyse_exchange(self):
        nyse = get_exchange(ExchangeID.NYSE)
        assert nyse.name == "New York Stock Exchange"
        assert nyse.timezone == "America/New_York"
        assert nyse.calendar == "NYSE"
        assert nyse.is_24_7 is False

    def test_coinbase_exchange(self):
        coinbase = get_exchange(ExchangeID.COINBASE)
        assert coinbase.timezone == "UTC"
        assert coinbase.is_24_7 is True

    def test_all_exchanges_defined(self):
        for eid in ExchangeID:
            assert eid in EXCHANGES


# ══════════════════════════════════════════════════════════════════════════════
# Backward Compatibility Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """
    Ensure the old src.market_calendar API still works identically.
    """

    def test_old_import_still_works(self):
        from src.market_calendar import (
            AssetClass as OldAssetClass,
            ExchangeCalendar,
            Instrument as OldInstrument,
            classify_symbol as old_classify,
            classify_symbols as old_classify_many,
            detect_gaps as old_detect_gaps,
            expected_gap_seconds as old_gap_seconds,
            filter_trading_hours as old_filter,
            get_annualization_factor as old_ann_factor,
            get_periods_per_year as old_periods,
            group_by_calendar as old_group,
            is_nyse_trading_day as old_is_day,
            is_nyse_trading_hour as old_is_hour,
            is_tradable_now as old_tradable,
        )
        # All imports succeed
        assert OldAssetClass.EQUITY == "equity"
        assert ExchangeCalendar.NYSE == "NYSE"

    def test_old_classify_symbol(self):
        from src.market_calendar import classify_symbol as old_classify
        inst = old_classify("BTC/USD")
        assert inst.is_crypto
        assert inst.trades_24_7

    def test_old_annualization_factor(self):
        from src.market_calendar import classify_symbol, get_annualization_factor
        inst = classify_symbol("AAPL")
        factor = get_annualization_factor(inst, "1Day")
        assert abs(factor - np.sqrt(252)) < 0.01

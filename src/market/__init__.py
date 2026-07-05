"""
Institutional Market Infrastructure Layer.

This package is the authoritative source for all market-related definitions:
instruments, exchanges, calendars, trading sessions, holidays, timezones,
corporate actions, symbol metadata, annualization factors, and market constraints.

Every subsystem (backtesting, execution, ML, replay, risk, scheduling, broker
integration) should consume this package rather than implementing market logic
independently.

Usage:
    from src.market import (
        Instrument,
        InstrumentRegistry,
        MarketCalendar,
        NYSECalendar,
        Crypto247Calendar,
        TradingSession,
        SessionType,
        SynchronizationPolicy,
        get_annualization_factor,
        detect_gaps,
    )
"""

from src.market.instruments import (
    AssetClass,
    Currency,
    FeeSchedule,
    Instrument,
)
from src.market.registry import InstrumentRegistry, classify_symbol, classify_symbols
from src.market.calendars import (
    MarketCalendar,
    NYSECalendar,
    Crypto247Calendar,
    NASDAQCalendar,
    CMECalendar,
    ForexCalendar,
)
from src.market.sessions import SessionType, TradingSession
from src.market.exchanges import Exchange, ExchangeID
from src.market.holidays import HolidayCalendar, NYSEHolidayCalendar
from src.market.timezones import (
    ensure_utc,
    to_exchange_time,
    reject_naive_datetime,
    normalize_index_to_utc,
)
from src.market.annualization import (
    get_annualization_factor,
    get_periods_per_year,
    bars_per_year,
)
from src.market.gap_detection import detect_gaps, log_gap_report
from src.market.synchronization import SynchronizationPolicy, SyncMode
from src.market.corporate_actions import CorporateAction, CorporateActionType
from src.market.constraints import MarketConstraints
from src.market.metadata import InstrumentMetadata

__all__ = [
    # Instruments
    "AssetClass",
    "Currency",
    "FeeSchedule",
    "Instrument",
    "InstrumentMetadata",
    # Registry
    "InstrumentRegistry",
    "classify_symbol",
    "classify_symbols",
    # Calendars
    "MarketCalendar",
    "NYSECalendar",
    "Crypto247Calendar",
    "NASDAQCalendar",
    "CMECalendar",
    "ForexCalendar",
    # Sessions
    "SessionType",
    "TradingSession",
    # Exchanges
    "Exchange",
    "ExchangeID",
    # Holidays
    "HolidayCalendar",
    "NYSEHolidayCalendar",
    # Timezones
    "ensure_utc",
    "to_exchange_time",
    "reject_naive_datetime",
    "normalize_index_to_utc",
    # Annualization
    "get_annualization_factor",
    "get_periods_per_year",
    "bars_per_year",
    # Gap Detection
    "detect_gaps",
    "log_gap_report",
    # Synchronization
    "SynchronizationPolicy",
    "SyncMode",
    # Corporate Actions
    "CorporateAction",
    "CorporateActionType",
    # Constraints
    "MarketConstraints",
]

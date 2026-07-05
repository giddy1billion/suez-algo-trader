"""
Instrument definitions — the core data model for tradable assets.

An Instrument is the canonical representation of any tradable symbol. It
encapsulates all metadata needed by calendars, sessions, risk, execution,
and backtesting subsystems.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AssetClass(str, Enum):
    """Supported asset classes."""
    EQUITY = "equity"
    CRYPTO = "crypto"
    FUTURES = "futures"
    FOREX = "forex"
    OPTIONS = "options"


class Currency(str, Enum):
    """Trade settlement currencies."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    BTC = "BTC"
    USDT = "USDT"
    USDC = "USDC"


class FeeSchedule(str, Enum):
    """Fee schedule categories."""
    EQUITY_US = "equity_us"
    CRYPTO_SPOT = "crypto_spot"
    CRYPTO_PERP = "crypto_perp"
    FUTURES_US = "futures_us"
    FOREX_SPOT = "forex_spot"
    ZERO_COMMISSION = "zero_commission"


@dataclass(frozen=True)
class Instrument:
    """
    Market instrument with full metadata.

    Every tradable symbol should be wrapped in an Instrument so that all
    calendar-sensitive, session-sensitive, and risk-sensitive logic can
    reference the correct properties without heuristics.

    Attributes:
        symbol: Canonical symbol identifier (e.g., "AAPL", "BTC/USD").
        asset_class: Classification of the instrument.
        exchange: Exchange identifier string (e.g., "NYSE", "COINBASE").
        calendar: Calendar identifier for session/holiday resolution.
        timezone: IANA timezone of the primary exchange (e.g., "America/New_York").
        currency: Settlement currency.
        tick_size: Minimum price increment.
        lot_size: Minimum order size increment.
        marginable: Whether margin trading is allowed.
        shortable: Whether short selling is allowed.
        fractional: Whether fractional shares/units are supported.
        fee_schedule: Fee schedule category.
        settlement_days: Settlement period in business days (T+N).
    """
    symbol: str
    asset_class: AssetClass
    exchange: str = "UNKNOWN"
    calendar: str = "NYSE"
    timezone: str = "America/New_York"
    currency: Currency = Currency.USD
    tick_size: float = 0.01
    lot_size: float = 1.0
    marginable: bool = True
    shortable: bool = True
    fractional: bool = False
    fee_schedule: FeeSchedule = FeeSchedule.EQUITY_US
    settlement_days: int = 2

    @property
    def is_crypto(self) -> bool:
        """Whether this is a cryptocurrency instrument."""
        return self.asset_class == AssetClass.CRYPTO

    @property
    def is_equity(self) -> bool:
        """Whether this is an equity instrument."""
        return self.asset_class == AssetClass.EQUITY

    @property
    def trades_24_7(self) -> bool:
        """Whether this instrument trades continuously (24/7)."""
        return self.calendar == "24/7"

"""
Exchange definitions.

Each exchange has a canonical ID, name, timezone, and associated calendar.
The exchange is the bridge between instruments and their trading calendars.
"""

from dataclasses import dataclass
from enum import Enum


class ExchangeID(str, Enum):
    """Known exchange identifiers."""
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    CME = "CME"
    CBOE = "CBOE"
    COINBASE = "COINBASE"
    BINANCE = "BINANCE"
    KRAKEN = "KRAKEN"
    FOREX = "FOREX"
    OTC = "OTC"


@dataclass(frozen=True)
class Exchange:
    """
    Exchange definition.

    Attributes:
        id: Canonical exchange identifier.
        name: Human-readable exchange name.
        timezone: IANA timezone string.
        calendar: Calendar identifier used for session/holiday resolution.
        mic: Market Identifier Code (ISO 10383), if applicable.
    """
    id: ExchangeID
    name: str
    timezone: str
    calendar: str
    mic: str = ""

    @property
    def is_24_7(self) -> bool:
        """Whether this exchange operates continuously."""
        return self.calendar == "24/7"


# Canonical exchange definitions
EXCHANGES: dict[ExchangeID, Exchange] = {
    ExchangeID.NYSE: Exchange(
        id=ExchangeID.NYSE,
        name="New York Stock Exchange",
        timezone="America/New_York",
        calendar="NYSE",
        mic="XNYS",
    ),
    ExchangeID.NASDAQ: Exchange(
        id=ExchangeID.NASDAQ,
        name="NASDAQ Stock Market",
        timezone="America/New_York",
        calendar="NASDAQ",
        mic="XNAS",
    ),
    ExchangeID.CME: Exchange(
        id=ExchangeID.CME,
        name="Chicago Mercantile Exchange",
        timezone="America/Chicago",
        calendar="CME",
        mic="XCME",
    ),
    ExchangeID.COINBASE: Exchange(
        id=ExchangeID.COINBASE,
        name="Coinbase",
        timezone="UTC",
        calendar="24/7",
        mic="",
    ),
    ExchangeID.BINANCE: Exchange(
        id=ExchangeID.BINANCE,
        name="Binance",
        timezone="UTC",
        calendar="24/7",
        mic="",
    ),
    ExchangeID.KRAKEN: Exchange(
        id=ExchangeID.KRAKEN,
        name="Kraken",
        timezone="UTC",
        calendar="24/7",
        mic="",
    ),
    ExchangeID.FOREX: Exchange(
        id=ExchangeID.FOREX,
        name="Foreign Exchange Market",
        timezone="UTC",
        calendar="FOREX",
        mic="",
    ),
    ExchangeID.OTC: Exchange(
        id=ExchangeID.OTC,
        name="Over-the-Counter",
        timezone="America/New_York",
        calendar="NYSE",
        mic="",
    ),
    ExchangeID.CBOE: Exchange(
        id=ExchangeID.CBOE,
        name="Chicago Board Options Exchange",
        timezone="America/Chicago",
        calendar="NYSE",
        mic="XCBO",
    ),
}


def get_exchange(exchange_id: ExchangeID) -> Exchange:
    """Get exchange definition by ID."""
    return EXCHANGES[exchange_id]

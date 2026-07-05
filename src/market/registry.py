"""
Instrument Registry — authoritative source for instrument metadata.

Classification priority:
1. Broker metadata (populated via broker adapters)
2. Local registry (configured instruments)
3. Configuration file
4. Symbol heuristics (last resort fallback)

Never rely solely on "/" in symbol names for classification.
"""

from typing import Optional

from src.market.instruments import AssetClass, Currency, FeeSchedule, Instrument


# Known crypto symbols (without "/" formatting)
_CRYPTO_SYMBOLS = {
    "BTCUSD", "ETHUSD", "SOLUSD", "AAVEUSD", "ADAUSD",
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "AAVEUSDT", "ADAUSDT",
    "XBTUSD", "XBTUSDT",
    "DOGEUSD", "DOGEUSDT", "AVAXUSD", "AVAXUSDT",
    "LINKUSD", "LINKUSDT", "MATICUSD", "MATICUSDT",
    "DOTUSD", "DOTUSDT", "UNIUSD", "UNIUSDT",
}

# Known crypto base symbols
_CRYPTO_BASES = {
    "BTC", "ETH", "SOL", "AAVE", "ADA", "DOGE", "AVAX",
    "LINK", "MATIC", "DOT", "UNI", "XRP", "LTC", "ATOM",
    "ALGO", "FTM", "NEAR", "APE", "SHIB", "CRV", "MKR",
    "COMP", "SUSHI", "YFI", "SNX", "SAND", "MANA", "AXS",
}


class InstrumentRegistry:
    """
    Central registry for instrument metadata.

    Instruments can be registered from broker metadata, configuration,
    or created on-demand via heuristic classification.

    Usage:
        registry = InstrumentRegistry()
        registry.register(instrument)
        inst = registry.get("AAPL")  # Returns registered or classifies on-demand
    """

    def __init__(self):
        self._instruments: dict[str, Instrument] = {}
        self._broker_populated: set[str] = set()

    def register(self, instrument: Instrument, source: str = "config") -> None:
        """
        Register an instrument in the registry.

        Args:
            instrument: Instrument to register.
            source: Source of this registration ("broker", "config", "heuristic").
        """
        self._instruments[instrument.symbol] = instrument
        if source == "broker":
            self._broker_populated.add(instrument.symbol)

    def register_many(self, instruments: list[Instrument], source: str = "config") -> None:
        """Register multiple instruments."""
        for inst in instruments:
            self.register(inst, source)

    def get(self, symbol: str) -> Instrument:
        """
        Get an instrument by symbol.

        If not registered, performs heuristic classification and registers
        the result for future lookups.

        Args:
            symbol: Symbol string (e.g., "AAPL", "BTC/USD").

        Returns:
            Instrument with metadata.
        """
        if symbol in self._instruments:
            return self._instruments[symbol]

        # Auto-classify and register
        instrument = _classify_symbol_heuristic(symbol)
        self._instruments[symbol] = instrument
        return instrument

    def has(self, symbol: str) -> bool:
        """Check if a symbol is registered."""
        return symbol in self._instruments

    def is_broker_populated(self, symbol: str) -> bool:
        """Check if instrument metadata came from broker."""
        return symbol in self._broker_populated

    def all_symbols(self) -> list[str]:
        """Get all registered symbols."""
        return list(self._instruments.keys())

    def by_asset_class(self, asset_class: AssetClass) -> list[Instrument]:
        """Get all instruments of a given asset class."""
        return [i for i in self._instruments.values() if i.asset_class == asset_class]

    def by_calendar(self, calendar: str) -> list[Instrument]:
        """Get all instruments using a specific calendar."""
        return [i for i in self._instruments.values() if i.calendar == calendar]

    def clear(self) -> None:
        """Clear all registered instruments."""
        self._instruments.clear()
        self._broker_populated.clear()


def _classify_symbol_heuristic(symbol: str) -> Instrument:
    """
    Classify a symbol using heuristics.

    This is the LAST RESORT fallback. Prefer broker metadata or
    configuration-based classification.

    Heuristic priority:
    1. Contains "/" → likely crypto pair (BTC/USD)
    2. Symbol in known crypto list → crypto
    3. Ends with USD/USDT without "/" → crypto
    4. Otherwise → US equity on NYSE

    Args:
        symbol: Symbol string.

    Returns:
        Instrument with best-guess metadata.
    """
    normalized = symbol.upper().replace("-", "").replace("_", "")

    # Check for "/" separator (most common crypto format)
    if "/" in symbol:
        return _make_crypto_instrument(symbol)

    # Check against known crypto symbols (handles BTCUSD, ETHUSDT, etc.)
    if normalized in _CRYPTO_SYMBOLS:
        return _make_crypto_instrument(symbol)

    # Check if it starts with a known crypto base
    for base in _CRYPTO_BASES:
        if normalized.startswith(base) and normalized[len(base):] in (
            "USD", "USDT", "USDC", "EUR", "GBP", "BTC", "ETH"
        ):
            return _make_crypto_instrument(symbol)

    # Default: US equity
    return _make_equity_instrument(symbol)


def _make_crypto_instrument(symbol: str) -> Instrument:
    """Create a crypto instrument with standard defaults."""
    return Instrument(
        symbol=symbol,
        asset_class=AssetClass.CRYPTO,
        exchange="CRYPTO",
        calendar="24/7",
        timezone="UTC",
        currency=Currency.USD,
        tick_size=0.01,
        lot_size=0.0001,
        marginable=False,
        shortable=False,
        fractional=True,
        fee_schedule=FeeSchedule.CRYPTO_SPOT,
        settlement_days=0,
    )


def _make_equity_instrument(symbol: str) -> Instrument:
    """Create a US equity instrument with standard defaults."""
    return Instrument(
        symbol=symbol,
        asset_class=AssetClass.EQUITY,
        exchange="NYSE",
        calendar="NYSE",
        timezone="America/New_York",
        currency=Currency.USD,
        tick_size=0.01,
        lot_size=1.0,
        marginable=True,
        shortable=True,
        fractional=False,
        fee_schedule=FeeSchedule.EQUITY_US,
        settlement_days=2,
    )


# Module-level convenience functions using a default registry
_default_registry = InstrumentRegistry()


def classify_symbol(symbol: str) -> Instrument:
    """
    Classify a symbol into an Instrument (backward-compatible API).

    Uses the default global registry. For custom registries, use
    InstrumentRegistry.get() directly.
    """
    return _default_registry.get(symbol)


def classify_symbols(symbols: list[str]) -> dict[str, Instrument]:
    """Classify multiple symbols (backward-compatible API)."""
    return {s: classify_symbol(s) for s in symbols}


def get_default_registry() -> InstrumentRegistry:
    """Get the default global instrument registry."""
    return _default_registry

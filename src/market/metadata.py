"""
Instrument Metadata — extended metadata beyond core Instrument fields.

This module provides additional metadata that may be populated from
broker APIs, market data providers, or configuration files.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class InstrumentMetadata:
    """
    Extended metadata for an instrument.

    This data is typically populated asynchronously from broker APIs
    or market data providers. It supplements the core Instrument dataclass.

    Attributes:
        symbol: Symbol this metadata belongs to.
        full_name: Full company/asset name.
        sector: Industry sector (for equities).
        industry: Specific industry.
        market_cap: Market capitalization.
        avg_volume: Average daily trading volume.
        beta: Beta relative to market index.
        dividend_yield: Annual dividend yield.
        pe_ratio: Price-to-earnings ratio.
        ipo_date: Date of initial public offering.
        description: Brief description of the asset.
        tags: Categorization tags.
        last_updated: When this metadata was last refreshed.
    """
    symbol: str
    full_name: str = ""
    sector: str = ""
    industry: str = ""
    market_cap: Optional[float] = None
    avg_volume: Optional[float] = None
    beta: Optional[float] = None
    dividend_yield: Optional[float] = None
    pe_ratio: Optional[float] = None
    ipo_date: Optional[date] = None
    description: str = ""
    tags: list[str] = field(default_factory=list)
    last_updated: Optional[date] = None

    @property
    def is_large_cap(self) -> bool:
        """Whether this is a large-cap stock (>$10B)."""
        return self.market_cap is not None and self.market_cap > 10_000_000_000

    @property
    def is_liquid(self) -> bool:
        """Whether this has sufficient liquidity (>1M avg volume)."""
        return self.avg_volume is not None and self.avg_volume > 1_000_000

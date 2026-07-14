"""
P2-19: Crypto symbol normalization utility.

Handles both "BTC/USD" (exchange format) and "BTCUSD" (compact format)
consistently. The canonical internal format uses slash-separated pairs
(e.g., "BTC/USD") for crypto assets, matching Alpaca's crypto API.

Equity symbols (e.g., "AAPL") are passed through unchanged.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Known crypto base currencies (most common, extensible)
_CRYPTO_BASES = {
    "BTC", "ETH", "SOL", "AAVE", "ADA", "DOGE", "DOT", "AVAX",
    "MATIC", "LINK", "UNI", "SHIB", "LTC", "XRP", "ATOM", "ALGO",
    "FIL", "NEAR", "APE", "CRV", "SUSHI", "BAT", "GRT", "MKR",
    "COMP", "YFI", "SNX", "AAVE", "BCH", "XLM", "EOS", "TRX",
}

# Known quote currencies
_CRYPTO_QUOTES = {"USD", "USDT", "USDC", "EUR", "GBP", "BTC", "ETH"}

# Pattern: letters followed by slash then letters (already normalized)
_SLASH_PATTERN = re.compile(r"^([A-Z]+)/([A-Z]+)$")
# Pattern: all-caps letters that could be a concatenated crypto pair
_CONCAT_PATTERN = re.compile(r"^([A-Z]+)(USD|USDT|USDC|EUR|GBP|BTC|ETH)$")


def normalize_symbol(symbol: str) -> str:
    """Normalize a trading symbol to canonical format.

    Crypto pairs: "BTCUSD" -> "BTC/USD", "BTC/USD" -> "BTC/USD"
    Equities: "AAPL" -> "AAPL" (unchanged)

    Args:
        symbol: Raw symbol string in any format.

    Returns:
        Normalized symbol string.
    """
    if not symbol:
        return symbol

    symbol = symbol.strip().upper()

    # Already in slash format
    if "/" in symbol:
        return symbol

    # Try to split concatenated crypto pair (e.g., "BTCUSD" -> "BTC/USD")
    match = _CONCAT_PATTERN.match(symbol)
    if match:
        base, quote = match.group(1), match.group(2)
        if base in _CRYPTO_BASES or len(base) >= 3:
            normalized = f"{base}/{quote}"
            logger.debug("symbol.normalized", original=symbol, normalized=normalized)
            return normalized

    # Not a recognized crypto pair — return as-is (equity symbol)
    return symbol


def is_crypto_symbol(symbol: str) -> bool:
    """Determine if a symbol represents a crypto asset.

    Returns True for both "BTC/USD" and "BTCUSD" formats.
    """
    if "/" in symbol:
        parts = symbol.upper().split("/")
        return len(parts) == 2 and parts[1] in _CRYPTO_QUOTES
    match = _CONCAT_PATTERN.match(symbol.upper())
    return match is not None and match.group(1) in _CRYPTO_BASES


def to_compact(symbol: str) -> str:
    """Convert slash format to compact: "BTC/USD" -> "BTCUSD"."""
    return symbol.replace("/", "")


def to_slash(symbol: str) -> str:
    """Ensure symbol is in slash format (alias for normalize_symbol)."""
    return normalize_symbol(symbol)

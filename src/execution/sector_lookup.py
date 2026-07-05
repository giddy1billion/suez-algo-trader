"""
Dynamic sector lookup — replaces the static _SECTOR_MAP constant.

Priority order:
1. Crypto detection (symbol contains "/" or ends in USD/USDT patterns)
2. Static reference file (data/sector_reference.json)
3. Database cache (SectorCache table, populated via /setsector)
4. Fallback: "unmapped_equity"

In-memory caching avoids a DB round-trip per signal evaluation.
"""

import json
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

_REFERENCE_FILE = Path(__file__).resolve().parents[2] / "data" / "sector_reference.json"

# In-memory caches (module-level singletons)
_static_map: dict[str, str] = {}
_db_cache: dict[str, str] = {}
_static_loaded: bool = False


def _load_static_reference():
    """Load the static sector_reference.json file into memory."""
    global _static_map, _static_loaded
    if _REFERENCE_FILE.exists():
        try:
            with open(_REFERENCE_FILE) as f:
                _static_map = json.load(f)
            logger.info("sector_lookup.static_loaded", count=len(_static_map))
        except (json.JSONDecodeError, OSError) as e:
            logger.error("sector_lookup.static_load_failed", error=str(e))
            _static_map = {}
    _static_loaded = True


def _is_crypto(symbol: str) -> bool:
    """Heuristic: detect crypto symbols."""
    s = symbol.upper()
    return (
        "/" in s
        or s.endswith("USD")
        or s.endswith("USDT")
        or s.startswith("BTC")
        or s.startswith("ETH")
    )


def get_sector(symbol: str, db=None) -> str:
    """
    Resolve the sector for a single symbol using the priority chain.
    Optionally accepts a db (DatabaseManager) for cache lookups.
    """
    global _static_loaded
    if not _static_loaded:
        _load_static_reference()

    symbol = symbol.upper()

    # 1. Crypto detection
    if _is_crypto(symbol):
        return "crypto"

    # 2. Static reference file
    if symbol in _static_map:
        return _static_map[symbol]

    # 3. In-memory DB cache
    if symbol in _db_cache:
        return _db_cache[symbol]

    # 3b. Direct DB lookup if db provided and not in memory cache
    if db is not None:
        sector = db.get_cached_sector(symbol)
        if sector:
            _db_cache[symbol] = sector
            return sector

    # 4. Fallback
    return "unmapped_equity"


def build_sector_map(symbols: list[str], db=None) -> dict[str, str]:
    """
    Build a sector map for a list of symbols. Used per-cycle by ExecutionEngine.
    """
    return {sym.upper(): get_sector(sym, db=db) for sym in symbols}


def reload_cache(db=None):
    """
    Refresh the in-memory DB cache from the database.
    Called after /setsector and during daily risk reset.
    Also reloads the static reference file.
    """
    global _db_cache, _static_loaded
    _static_loaded = False
    _load_static_reference()

    if db is not None:
        try:
            _db_cache = db.get_all_cached_sectors()
            logger.info("sector_lookup.cache_reloaded", db_entries=len(_db_cache))
        except Exception as e:
            logger.error("sector_lookup.cache_reload_failed", error=str(e))
    else:
        _db_cache = {}

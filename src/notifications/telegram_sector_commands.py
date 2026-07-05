"""
Telegram command for manually setting sector classifications.
Used to backfill the sector cache when a symbol resolves to 'unmapped_equity'.
"""

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from src.execution.sector_lookup import reload_cache
from src.utils.logger import get_logger

logger = get_logger(__name__)

sector_router = Router()

_db = None
_authorized_users: set[int] = set()

VALID_SECTORS = {
    "technology", "communication_services", "consumer_discretionary",
    "consumer_staples", "financials", "healthcare", "industrials",
    "energy", "materials", "real_estate", "utilities", "crypto",
}


def set_sector_components(db, authorized_users: set[int]):
    """Inject dependencies — call once during bot setup, same pattern as set_config_components."""
    global _db, _authorized_users
    _db = db
    _authorized_users = authorized_users


@sector_router.message(Command("setsector"))
async def cmd_setsector(message: Message):
    """
    /setsector SYMBOL SECTOR — manually classify a symbol's sector.
    Example: /setsector PLTR technology
    """
    if message.from_user.id not in _authorized_users:
        return  # Unauthorized users receive no response, per existing bot convention

    parts = message.text.split()
    if len(parts) != 3:
        await message.answer(
            "Usage: /setsector SYMBOL SECTOR\n"
            f"Valid sectors: {', '.join(sorted(VALID_SECTORS))}"
        )
        return

    _, symbol, sector = parts
    symbol = symbol.upper()
    sector = sector.lower()

    if sector not in VALID_SECTORS:
        await message.answer(
            f"⚠️ Unknown sector '{sector}'.\nValid sectors: {', '.join(sorted(VALID_SECTORS))}"
        )
        return

    try:
        _db.set_cached_sector(symbol, sector, source="manual")
        reload_cache(_db)  # refresh in-memory map immediately, no restart needed
        logger.info("telegram.sector_set", symbol=symbol, sector=sector, user=message.from_user.id)
        await message.answer(f"✅ {symbol} → {sector}\nCache reloaded.")
    except Exception as e:
        logger.error("telegram.sector_set_failed", symbol=symbol, error=str(e))
        await message.answer(f"⚠️ Failed to set sector: {str(e)[:100]}")


@sector_router.message(Command("sectors"))
async def cmd_list_sectors(message: Message):
    """/sectors — list all manually cached sector classifications."""
    if message.from_user.id not in _authorized_users:
        return

    try:
        cached = _db.get_all_cached_sectors()
        if not cached:
            await message.answer("No manually cached sectors yet. Use /setsector SYMBOL SECTOR.")
            return
        lines = [f"{sym}: {sec}" for sym, sec in sorted(cached.items())]
        await message.answer("📊 Cached Sectors:\n" + "\n".join(lines))
    except Exception as e:
        await message.answer(f"⚠️ Failed to list sectors: {str(e)[:100]}")

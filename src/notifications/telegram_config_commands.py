"""
Telegram Config Commands — Full system configuration management via Telegram.

Adds commands for:
- Alpaca credential management (/setalpaca, /setmode)
- Full config persistence (/saveconfig, /loadconfig)
- Comprehensive config view (/configfull)
- Strategy CRUD (/newstrategy, /liststrats, /editstrat, /deletestrat, /templates)
- Config export (/export)

All commands are authorization-gated.
"""

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.enums import ParseMode

from src.utils.logger import get_logger

logger = get_logger(__name__)

config_router = Router()

# Module-level references (injected via set_config_components)
_settings = None
_strategy_store = None
_authorized_users: set[int] = set()
_runtime_lock = None
_runtime_changes = None

ENV_FILE = Path(".env")


def set_config_components(
    settings,
    strategy_store,
    authorized_users: set[int],
    runtime_lock: threading.Lock,
    runtime_changes: dict,
):
    """Inject dependencies into this module."""
    global _settings, _strategy_store, _authorized_users, _runtime_lock, _runtime_changes
    _settings = settings
    _strategy_store = strategy_store
    _authorized_users = authorized_users
    _runtime_lock = runtime_lock
    _runtime_changes = runtime_changes


def _is_authorized(message: Message) -> bool:
    """Check if user is authorized."""
    if not _authorized_users:
        return True
    return message.from_user and message.from_user.id in _authorized_users


def _mask_secret(value: str) -> str:
    """Mask a secret string, showing only last 4 chars."""
    if not value or len(value) < 8:
        return "••••••••" if value else "(not set)"
    return "•" * (len(value) - 4) + value[-4:]


# =============================================================================
# ALPACA / BROKER CONFIGURATION
# =============================================================================


@config_router.message(Command("setalpaca"))
async def cmd_setalpaca(message: Message):
    """Configure Alpaca API credentials.

    Usage:
        /setalpaca paper KEY SECRET
        /setalpaca live KEY SECRET
        /setalpaca feed iex|sip
        /setalpaca url paper|live URL
    """
    if not _is_authorized(message):
        return

    parts = message.text.split(maxsplit=3)
    if len(parts) < 3:
        await message.answer(
            "<b>🔑 Alpaca Configuration</b>\n\n"
            "<b>Set API keys:</b>\n"
            "  <code>/setalpaca paper API_KEY SECRET_KEY</code>\n"
            "  <code>/setalpaca live API_KEY SECRET_KEY</code>\n\n"
            "<b>Set data feed:</b>\n"
            "  <code>/setalpaca feed iex</code> (free)\n"
            "  <code>/setalpaca feed sip</code> (paid)\n\n"
            "<b>Set base URL:</b>\n"
            "  <code>/setalpaca url paper https://...</code>\n"
            "  <code>/setalpaca url live https://...</code>\n\n"
            "<b>Current config:</b>\n"
            f"  Mode: <code>{_settings.trading_mode.value}</code>\n"
            f"  Paper Key: <code>{_mask_secret(_settings.alpaca_paper_api_key)}</code>\n"
            f"  Live Key: <code>{_mask_secret(_settings.alpaca_live_api_key)}</code>\n"
            f"  Feed: <code>{_settings.alpaca_data_feed}</code>\n",
            parse_mode=ParseMode.HTML,
        )
        return

    sub_cmd = parts[1].lower()

    if sub_cmd in ("paper", "live"):
        if len(parts) < 4:
            await message.answer(
                f"<b>Usage:</b> <code>/setalpaca {sub_cmd} API_KEY SECRET_KEY</code>",
                parse_mode=ParseMode.HTML,
            )
            return

        api_key = parts[2].strip()
        secret_key = parts[3].strip()

        # Basic validation
        if len(api_key) < 10 or len(secret_key) < 10:
            await message.answer("❌ Keys seem too short. Alpaca keys are typically 20+ characters.")
            return

        if sub_cmd == "paper":
            _settings.alpaca_paper_api_key = api_key
            _settings.alpaca_paper_secret_key = secret_key
        else:
            _settings.alpaca_live_api_key = api_key
            _settings.alpaca_live_secret_key = secret_key

        # Delete the user's message containing secrets (security)
        try:
            await message.delete()
        except Exception:
            pass

        await message.answer(
            f"✅ <b>Alpaca {sub_cmd} credentials updated</b>\n"
            f"  Key: <code>{_mask_secret(api_key)}</code>\n\n"
            f"⚠️ Use <code>/saveconfig</code> to persist to .env\n"
            f"💡 Your message was deleted for security.",
            parse_mode=ParseMode.HTML,
        )

    elif sub_cmd == "feed":
        feed = parts[2].lower().strip()
        if feed not in ("iex", "sip"):
            await message.answer("❌ Feed must be 'iex' (free) or 'sip' (paid, all exchanges).")
            return

        old_feed = _settings.alpaca_data_feed
        _settings.alpaca_data_feed = feed

        await message.answer(
            f"✅ <b>Data feed updated</b>\n"
            f"  {old_feed} → <b>{feed}</b>\n"
            f"  {'📡 Full market data (paid)' if feed == 'sip' else '📊 IEX data (free)'}",
            parse_mode=ParseMode.HTML,
        )

    elif sub_cmd == "url":
        if len(parts) < 4:
            await message.answer(
                "<b>Usage:</b> <code>/setalpaca url paper|live URL</code>",
                parse_mode=ParseMode.HTML,
            )
            return

        # Re-parse since URL might have spaces
        remaining = message.text.split(maxsplit=2)[2]  # "url paper https://..."
        url_parts = remaining.split(maxsplit=1)
        if len(url_parts) < 2:
            await message.answer("❌ Provide mode (paper/live) and URL.")
            return

        mode = url_parts[0].lower()
        url = url_parts[1].strip() if len(url_parts) > 1 else ""

        if mode not in ("paper", "live"):
            await message.answer("❌ Mode must be 'paper' or 'live'.")
            return
        if not url.startswith("http"):
            await message.answer("❌ URL must start with http:// or https://")
            return

        if mode == "paper":
            _settings.alpaca_paper_base_url = url
        else:
            _settings.alpaca_live_base_url = url

        await message.answer(
            f"✅ <b>Alpaca {mode} URL updated</b>\n  <code>{url}</code>",
            parse_mode=ParseMode.HTML,
        )

    else:
        await message.answer(
            f"❌ Unknown subcommand: {sub_cmd}\n"
            f"Valid: paper, live, feed, url",
        )


@config_router.message(Command("setmode"))
async def cmd_setmode(message: Message):
    """Switch trading mode between paper and live.

    Usage: /setmode paper|live
    """
    if not _is_authorized(message):
        return

    from config.settings import TradingMode

    parts = message.text.split()
    if len(parts) < 2:
        current = _settings.trading_mode.value
        await message.answer(
            f"<b>🔄 Trading Mode</b>\n\n"
            f"Current: <b>{'🟢 PAPER' if current == 'paper' else '🔴 LIVE'}</b>\n\n"
            f"<b>Usage:</b> <code>/setmode paper|live</code>\n\n"
            f"⚠️ Switching to LIVE uses real money!\n"
            f"Ensure live API keys are configured first.",
            parse_mode=ParseMode.HTML,
        )
        return

    new_mode = parts[1].lower().strip()
    if new_mode not in ("paper", "live"):
        await message.answer("❌ Mode must be 'paper' or 'live'.")
        return

    old_mode = _settings.trading_mode.value
    if new_mode == old_mode:
        await message.answer(f"Already in {new_mode} mode.")
        return

    # Safety check for live mode
    if new_mode == "live":
        if not _settings.alpaca_live_api_key or len(_settings.alpaca_live_api_key) < 10:
            await message.answer(
                "❌ <b>Cannot switch to LIVE</b>\n\n"
                "Live API keys not configured.\n"
                "Use <code>/setalpaca live KEY SECRET</code> first.",
                parse_mode=ParseMode.HTML,
            )
            return

        await message.answer(
            f"⚠️ <b>SWITCHING TO LIVE TRADING</b>\n\n"
            f"🔴 This will use REAL MONEY.\n"
            f"  Key: <code>{_mask_secret(_settings.alpaca_live_api_key)}</code>\n\n"
            f"Mode changed: {old_mode} → <b>LIVE</b>\n"
            f"Use <code>/setmode paper</code> to switch back.",
            parse_mode=ParseMode.HTML,
        )
    else:
        await message.answer(
            f"✅ <b>Switched to PAPER trading</b>\n\n"
            f"🟢 Safe mode — no real money at risk.\n"
            f"Mode changed: {old_mode} → <b>PAPER</b>",
            parse_mode=ParseMode.HTML,
        )

    _settings.trading_mode = TradingMode(new_mode)


# =============================================================================
# CONFIG PERSISTENCE
# =============================================================================


@config_router.message(Command("saveconfig"))
async def cmd_saveconfig(message: Message):
    """Save current runtime configuration to .env file.

    Usage: /saveconfig
    """
    if not _is_authorized(message):
        return

    try:
        env_lines = _generate_env_content()
        with open(ENV_FILE, "w") as f:
            f.write(env_lines)

        await message.answer(
            "✅ <b>Configuration saved to .env</b>\n\n"
            f"📁 File: <code>{ENV_FILE.resolve()}</code>\n"
            f"⏰ Saved at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        await message.answer(f"❌ Failed to save config: {e}")


@config_router.message(Command("loadconfig"))
async def cmd_loadconfig(message: Message):
    """Reload configuration from .env file.

    Usage: /loadconfig
    """
    if not _is_authorized(message):
        return

    if not ENV_FILE.exists():
        await message.answer("❌ No .env file found.")
        return

    try:
        from config.settings import Settings
        new_settings = Settings()

        # Copy all fields to current settings
        for field_name in _settings.model_fields:
            if hasattr(new_settings, field_name):
                setattr(_settings, field_name, getattr(new_settings, field_name))

        await message.answer(
            "✅ <b>Configuration reloaded from .env</b>\n\n"
            f"📁 File: <code>{ENV_FILE.resolve()}</code>\n"
            "All runtime settings updated.",
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        await message.answer(f"❌ Failed to reload config: {e}")


@config_router.message(Command("configfull"))
async def cmd_configfull(message: Message):
    """Show ALL configuration including broker/Alpaca settings (secrets masked).

    Usage: /configfull [section]
    Sections: broker, risk, strategy, ml, auto, notify, execution, all
    """
    if not _is_authorized(message):
        return

    parts = message.text.split()
    section = parts[1].lower() if len(parts) > 1 else "all"

    sections = {}

    sections["broker"] = (
        f"<b>🏦 Broker (Alpaca)</b>\n"
        f"  Mode: <b>{'🟢 PAPER' if _settings.trading_mode.value == 'paper' else '🔴 LIVE'}</b>\n"
        f"  Paper Key: <code>{_mask_secret(_settings.alpaca_paper_api_key)}</code>\n"
        f"  Paper Secret: <code>{_mask_secret(_settings.alpaca_paper_secret_key)}</code>\n"
        f"  Paper URL: <code>{_settings.alpaca_paper_base_url}</code>\n"
        f"  Live Key: <code>{_mask_secret(_settings.alpaca_live_api_key)}</code>\n"
        f"  Live Secret: <code>{_mask_secret(_settings.alpaca_live_secret_key)}</code>\n"
        f"  Live URL: <code>{_settings.alpaca_live_base_url}</code>\n"
        f"  Data Feed: <code>{_settings.alpaca_data_feed}</code>\n"
    )

    sections["risk"] = (
        f"<b>🛡️ Risk Management</b>\n"
        f"  Max risk/trade: {_settings.max_position_size_pct:.1%}\n"
        f"  Daily loss limit: {_settings.max_daily_loss_pct:.1%}\n"
        f"  Max exposure: {_settings.max_portfolio_exposure:.0%}\n"
        f"  Max single stock: {_settings.max_single_stock_pct:.0%}\n"
        f"  Max leverage: {_settings.max_leverage:.1f}x\n"
        f"  Stop-loss: {_settings.default_stop_loss_pct:.1%}\n"
        f"  Take-profit: {_settings.default_take_profit_pct:.1%}\n"
        f"  Max positions: {_settings.max_open_positions}\n"
        f"  Max orders/day: {_settings.max_orders_per_day}\n"
        f"  Max correlated: {_settings.max_correlated_positions}\n"
    )

    sections["risk_layers"] = (
        f"<b>🔒 Risk Layers</b>\n"
        f"  Portfolio: {'✅' if _settings.risk_portfolio_layer_enabled else '❌'}\n"
        f"  Account: {'✅' if _settings.risk_account_layer_enabled else '❌'}\n"
        f"  Exposure: {'✅' if _settings.risk_exposure_layer_enabled else '❌'}\n"
        f"  Execution: {'✅' if _settings.risk_execution_layer_enabled else '❌'}\n"
        f"  Max drawdown: {_settings.risk_max_drawdown_pct:.0%}\n"
        f"  Max daily loss: {_settings.risk_max_daily_loss_pct:.1%}\n"
        f"  Max weekly loss: {_settings.risk_max_weekly_loss_pct:.1%}\n"
        f"  Cash reserve: {_settings.risk_min_cash_reserve_pct:.0%}\n"
        f"  Consec loss limit: {_settings.risk_consecutive_loss_limit}\n"
    )

    sections["strategy"] = (
        f"<b>📊 Strategy</b>\n"
        f"  Active: <code>{_settings.active_strategy}</code>\n"
        f"  Symbols: <code>{_settings.trading_symbols}</code>\n"
        f"  Timeframe: <code>{_settings.timeframe}</code>\n"
        f"  Lookback: {_settings.lookback_bars} bars\n"
        f"  Interval: {_settings.trading_interval}s\n"
        f"  Multi-config: <code>{_settings.multi_strategy_config or '(none)'}</code>\n"
    )

    sections["ml"] = (
        f"<b>🧠 ML</b>\n"
        f"  Model: <code>{_settings.ml_model_path}</code>\n"
        f"  Retrain: every {_settings.ml_retrain_interval_hours}h\n"
        f"  Min confidence: {_settings.ml_min_confidence:.0%}\n"
    )

    sections["auto"] = (
        f"<b>⚙️ Automation</b>\n"
        f"  Backtest: every {_settings.auto_backtest_interval_hours}h {'(off)' if _settings.auto_backtest_interval_hours == 0 else ''}\n"
        f"  Train: every {_settings.auto_train_interval_hours}h {'(off)' if _settings.auto_train_interval_hours == 0 else ''}\n"
        f"  Sweep: every {_settings.auto_sweep_interval_hours}h {'(off)' if _settings.auto_sweep_interval_hours == 0 else ''}\n"
        f"  Train bars: {_settings.auto_train_bars}\n"
    )

    sections["notify"] = (
        f"<b>🔔 Notifications</b>\n"
        f"  On trade: {'✅' if _settings.notify_on_trade else '❌'}\n"
        f"  On error: {'✅' if _settings.notify_on_error else '❌'}\n"
        f"  On signal: {'✅' if _settings.notify_on_signal else '❌'}\n"
        f"  Telegram: <code>{_mask_secret(_settings.telegram_bot_token)}</code>\n"
        f"  Chat ID: <code>{_settings.telegram_chat_id or '(not set)'}</code>\n"
    )

    sections["execution"] = (
        f"<b>⚡ Execution</b>\n"
        f"  Simulator: {'✅' if _settings.enable_execution_simulator else '❌'}\n"
        f"  Preset: <code>{_settings.execution_simulator_preset}</code>\n"
        f"  Max spread: {_settings.risk_max_spread_pct:.3%}\n"
        f"  Min volume: {_settings.risk_min_volume:,}\n"
        f"  Max slippage: {_settings.risk_max_slippage_pct:.3%}\n"
        f"  Orders/min: {_settings.risk_max_orders_per_minute}\n"
        f"  Cooldown after loss: {_settings.risk_cooldown_after_loss_minutes}min\n"
    )

    if section == "all":
        text = "\n".join(sections.values())
    elif section in sections:
        text = sections[section]
    else:
        text = (
            f"❌ Unknown section: {section}\n\n"
            f"<b>Available:</b> broker, risk, risk_layers, strategy, ml, auto, notify, execution, all"
        )

    # Telegram message limit is 4096 chars
    if len(text) > 4000:
        # Split into two messages
        mid = len(text) // 2
        split_at = text.rfind("\n", 0, mid)
        await message.answer(text[:split_at], parse_mode=ParseMode.HTML)
        await message.answer(text[split_at:], parse_mode=ParseMode.HTML)
    else:
        await message.answer(text, parse_mode=ParseMode.HTML)


@config_router.message(Command("export"))
async def cmd_export(message: Message):
    """Export full configuration as formatted text (secrets masked).

    Usage: /export
    """
    if not _is_authorized(message):
        return

    lines = []
    lines.append("📋 <b>Full Configuration Export</b>\n")
    lines.append(f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

    # Iterate all settings fields
    for field_name, field_info in _settings.model_fields.items():
        value = getattr(_settings, field_name, "")
        # Mask secrets
        if any(s in field_name for s in ("key", "secret", "token", "password", "webhook")):
            display = _mask_secret(str(value))
        else:
            display = str(value)
        lines.append(f"<code>{field_name}</code> = {display}")

    text = "\n".join(lines)

    # Split if too long
    if len(text) > 4000:
        chunks = [text[i:i+3900] for i in range(0, len(text), 3900)]
        for chunk in chunks:
            await message.answer(chunk, parse_mode=ParseMode.HTML)
    else:
        await message.answer(text, parse_mode=ParseMode.HTML)


# =============================================================================
# STRATEGY MANAGEMENT
# =============================================================================


@config_router.message(Command("newstrategy"))
async def cmd_newstrategy(message: Message):
    """Create a new custom trading strategy.

    Usage: /newstrategy NAME TEMPLATE SYMBOLS [TIMEFRAME] [INTERVAL]

    Examples:
        /newstrategy my_scalper scalping AAPL,TSLA 5Min 30
        /newstrategy btc_swing swing BTC/USD,ETH/USD 4Hour 300
        /newstrategy tech_momentum momentum NVDA,AMD,MSFT 1Hour 60
    """
    if not _is_authorized(message):
        return

    if not _strategy_store:
        await message.answer("❌ Strategy store not initialized.")
        return

    parts = message.text.split()
    if len(parts) < 4:
        templates = _strategy_store.get_templates()
        tmpl_text = "\n".join(
            f"  <b>{name}</b>: {info['description']}"
            for name, info in templates.items()
        )
        await message.answer(
            "<b>🆕 Create New Strategy</b>\n\n"
            "<b>Usage:</b>\n"
            "  <code>/newstrategy NAME TEMPLATE SYMBOLS [TIMEFRAME] [INTERVAL]</code>\n\n"
            "<b>Available templates:</b>\n"
            f"{tmpl_text}\n\n"
            "<b>Examples:</b>\n"
            "  <code>/newstrategy my_scalper scalping AAPL,TSLA 5Min 30</code>\n"
            "  <code>/newstrategy btc_swing swing BTC/USD 4Hour 300</code>\n"
            "  <code>/newstrategy tech_mom momentum NVDA,AMD 1Hour 60</code>\n\n"
            "💡 After creating, use <code>/editstrat</code> to tune parameters.",
            parse_mode=ParseMode.HTML,
        )
        return

    name = parts[1].lower()
    template = parts[2].lower()
    symbols = [s.strip().upper() for s in parts[3].split(",") if s.strip()]
    timeframe = parts[4] if len(parts) > 4 else "1Hour"
    interval = int(parts[5]) if len(parts) > 5 else 60

    success, msg = _strategy_store.create(
        name=name,
        template=template,
        symbols=symbols,
        timeframe=timeframe,
        interval=interval,
    )

    if success:
        strat = _strategy_store.get(name)
        await message.answer(
            f"✅ <b>Strategy Created: {name}</b>\n\n"
            f"  Template: <code>{template}</code>\n"
            f"  Symbols: <code>{', '.join(symbols)}</code>\n"
            f"  Timeframe: <code>{timeframe}</code>\n"
            f"  Interval: {interval}s\n"
            f"  Status: 🟢 Active\n\n"
            f"<b>Parameters:</b>\n"
            + "\n".join(f"  {k}: <code>{v}</code>" for k, v in (strat.params or {}).items())
            + "\n\n💡 Use <code>/editstrat {name} param VALUE</code> to tune.",
            parse_mode=ParseMode.HTML,
        )
    else:
        await message.answer(f"❌ {msg}")


@config_router.message(Command("liststrats"))
async def cmd_liststrats(message: Message):
    """List all saved strategies.

    Usage: /liststrats
    """
    if not _is_authorized(message):
        return

    if not _strategy_store:
        await message.answer("❌ Strategy store not initialized.")
        return

    strategies = _strategy_store.list_strategies()
    if not strategies:
        await message.answer(
            "<b>📋 No saved strategies</b>\n\n"
            "Use <code>/newstrategy</code> to create one.\n"
            "Use <code>/templates</code> to see available templates.",
            parse_mode=ParseMode.HTML,
        )
        return

    lines = ["<b>📋 Saved Strategies</b>\n"]
    for s in strategies:
        status = "🟢" if s.active else "🔴"
        lines.append(
            f"{status} <b>{s.name}</b> ({s.template})\n"
            f"    Symbols: <code>{', '.join(s.symbols[:5])}</code>"
            f"{'...' if len(s.symbols) > 5 else ''}\n"
            f"    TF: {s.timeframe} | Interval: {s.interval}s | Weight: {s.weight}\n"
        )

    lines.append(f"\n<b>Total:</b> {len(strategies)} strategies")
    lines.append(f"<b>Active:</b> {sum(1 for s in strategies if s.active)}")
    lines.append(
        f"\n<b>Multi-config string:</b>\n"
        f"<code>{_strategy_store.get_multi_config_string() or '(no active strategies)'}</code>"
    )

    await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)


@config_router.message(Command("editstrat"))
async def cmd_editstrat(message: Message):
    """Edit a strategy's parameters.

    Usage:
        /editstrat NAME symbols AAPL,MSFT
        /editstrat NAME timeframe 15Min
        /editstrat NAME interval 120
        /editstrat NAME weight 1.5
        /editstrat NAME lookback 500
        /editstrat NAME param PARAM_NAME VALUE
        /editstrat NAME activate
        /editstrat NAME deactivate
    """
    if not _is_authorized(message):
        return

    if not _strategy_store:
        await message.answer("❌ Strategy store not initialized.")
        return

    parts = message.text.split()
    if len(parts) < 3:
        await message.answer(
            "<b>✏️ Edit Strategy</b>\n\n"
            "<b>Usage:</b>\n"
            "  <code>/editstrat NAME symbols AAPL,MSFT</code>\n"
            "  <code>/editstrat NAME timeframe 15Min</code>\n"
            "  <code>/editstrat NAME interval 120</code>\n"
            "  <code>/editstrat NAME weight 1.5</code>\n"
            "  <code>/editstrat NAME lookback 500</code>\n"
            "  <code>/editstrat NAME param fast_ema 8</code>\n"
            "  <code>/editstrat NAME activate</code>\n"
            "  <code>/editstrat NAME deactivate</code>\n\n"
            "Use <code>/liststrats</code> to see strategy names.",
            parse_mode=ParseMode.HTML,
        )
        return

    name = parts[1].lower()
    action = parts[2].lower()

    strat = _strategy_store.get(name)
    if not strat:
        await message.answer(f"❌ Strategy '{name}' not found. Use /liststrats to see available.")
        return

    if action == "activate":
        success, msg = _strategy_store.activate(name)
        await message.answer(f"{'✅' if success else '❌'} {msg}")
        return

    if action == "deactivate":
        success, msg = _strategy_store.deactivate(name)
        await message.answer(f"{'✅' if success else '❌'} {msg}")
        return

    if len(parts) < 4:
        await message.answer(f"❌ Missing value for '{action}'.")
        return

    value = parts[3] if len(parts) == 4 else " ".join(parts[3:])

    if action == "symbols":
        symbols = [s.strip().upper() for s in value.split(",") if s.strip()]
        success, msg = _strategy_store.update(name, symbols=symbols)

    elif action == "timeframe":
        success, msg = _strategy_store.update(name, timeframe=value)

    elif action == "interval":
        try:
            success, msg = _strategy_store.update(name, interval=int(value))
        except ValueError:
            await message.answer("❌ Interval must be an integer (seconds).")
            return

    elif action == "weight":
        try:
            success, msg = _strategy_store.update(name, weight=float(value))
        except ValueError:
            await message.answer("❌ Weight must be a number.")
            return

    elif action == "lookback":
        try:
            success, msg = _strategy_store.update(name, lookback=int(value))
        except ValueError:
            await message.answer("❌ Lookback must be an integer.")
            return

    elif action == "param":
        if len(parts) < 5:
            await message.answer(
                f"❌ Usage: <code>/editstrat {name} param PARAM_NAME VALUE</code>\n\n"
                f"<b>Current params for {name}:</b>\n"
                + "\n".join(f"  {k}: <code>{v}</code>" for k, v in strat.params.items()),
                parse_mode=ParseMode.HTML,
            )
            return

        param_name = parts[3]
        param_value = parts[4]

        # Type coercion
        try:
            if "." in param_value:
                param_value = float(param_value)
            else:
                param_value = int(param_value)
        except ValueError:
            pass  # Keep as string

        success, msg = _strategy_store.update(name, params={param_name: param_value})

    elif action == "description":
        success, msg = _strategy_store.update(name, description=value)

    else:
        await message.answer(
            f"❌ Unknown field: '{action}'\n"
            f"Valid: symbols, timeframe, interval, weight, lookback, param, activate, deactivate, description"
        )
        return

    if success:
        updated = _strategy_store.get(name)
        await message.answer(
            f"✅ <b>{name}</b> updated: {action} → <code>{value}</code>",
            parse_mode=ParseMode.HTML,
        )
    else:
        await message.answer(f"❌ {msg}")


@config_router.message(Command("deletestrat"))
async def cmd_deletestrat(message: Message):
    """Delete a saved strategy.

    Usage: /deletestrat NAME
    """
    if not _is_authorized(message):
        return

    if not _strategy_store:
        await message.answer("❌ Strategy store not initialized.")
        return

    parts = message.text.split()
    if len(parts) < 2:
        await message.answer(
            "<b>Usage:</b> <code>/deletestrat NAME</code>\n"
            "Use <code>/liststrats</code> to see strategy names.",
            parse_mode=ParseMode.HTML,
        )
        return

    name = parts[1].lower()
    success, msg = _strategy_store.delete(name)

    if success:
        await message.answer(f"🗑️ {msg}")
    else:
        await message.answer(f"❌ {msg}")


@config_router.message(Command("copystrat"))
async def cmd_copystrat(message: Message):
    """Duplicate an existing strategy with a new name.

    Usage: /copystrat SOURCE_NAME NEW_NAME
    """
    if not _is_authorized(message):
        return

    if not _strategy_store:
        await message.answer("❌ Strategy store not initialized.")
        return

    parts = message.text.split()
    if len(parts) < 3:
        await message.answer(
            "<b>Usage:</b> <code>/copystrat SOURCE_NAME NEW_NAME</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    source = parts[1].lower()
    new_name = parts[2].lower()

    success, msg = _strategy_store.duplicate(source, new_name)
    await message.answer(f"{'✅' if success else '❌'} {msg}")


@config_router.message(Command("templates"))
async def cmd_templates(message: Message):
    """Show available strategy templates and their parameters.

    Usage: /templates [NAME]
    """
    if not _is_authorized(message):
        return

    if not _strategy_store:
        await message.answer("❌ Strategy store not initialized.")
        return

    parts = message.text.split()
    templates = _strategy_store.get_templates()

    if len(parts) > 1:
        name = parts[1].lower()
        from src.strategy.strategy_store import STRATEGY_TEMPLATES
        if name not in STRATEGY_TEMPLATES:
            await message.answer(f"❌ Unknown template: {name}\nValid: {', '.join(templates.keys())}")
            return

        tmpl = STRATEGY_TEMPLATES[name]
        params_text = "\n".join(f"  {k}: <code>{v}</code>" for k, v in tmpl["params"].items())
        await message.answer(
            f"<b>📐 Template: {name}</b>\n"
            f"{tmpl['description']}\n\n"
            f"<b>Default parameters:</b>\n{params_text}\n\n"
            f"<b>Create:</b>\n"
            f"  <code>/newstrategy my_{name} {name} AAPL,MSFT 1Hour 60</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    lines = ["<b>📐 Strategy Templates</b>\n"]
    for name, info in templates.items():
        lines.append(f"  <b>{name}</b>: {info['description']}")
        lines.append(f"    Params: {', '.join(info['params'][:5])}{'...' if len(info['params']) > 5 else ''}\n")

    lines.append("\n💡 Use <code>/templates NAME</code> for full parameter list.")
    await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)


@config_router.message(Command("applystrats"))
async def cmd_applystrats(message: Message):
    """Apply saved strategies to the multi-strategy orchestrator.

    Usage: /applystrats

    This generates the multi_strategy_config from your saved strategies
    and signals the main loop to reload.
    """
    if not _is_authorized(message):
        return

    if not _strategy_store:
        await message.answer("❌ Strategy store not initialized.")
        return

    active = _strategy_store.get_active_strategies()
    if not active:
        await message.answer("❌ No active strategies. Create or activate strategies first.")
        return

    config_str = _strategy_store.get_multi_config_string()

    # Update settings
    _settings.multi_strategy_config = config_str
    _settings.active_strategy = "multi"

    # Signal main loop
    if _runtime_lock and _runtime_changes is not None:
        with _runtime_lock:
            _runtime_changes["strategy_name"] = "multi"
            _runtime_changes["config_updates"]["multi_strategy_config"] = config_str

    await message.answer(
        f"✅ <b>Strategies Applied</b>\n\n"
        f"Active strategies: {len(active)}\n"
        f"Mode: <b>multi-strategy</b>\n\n"
        f"<b>Config:</b>\n<code>{config_str}</code>\n\n"
        f"Will take effect on next cycle.",
        parse_mode=ParseMode.HTML,
    )


# =============================================================================
# HELPER: Generate .env content from current settings
# =============================================================================


def _generate_env_content() -> str:
    """Generate .env file content from current settings."""
    lines = [
        "# ============================================================================",
        "# ALPACA ALGO TRADER — ENVIRONMENT CONFIGURATION",
        "# ============================================================================",
        f"# Auto-saved: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "# --- Account Mode ---",
        f"TRADING_MODE={_settings.trading_mode.value}",
        "",
        "# --- Alpaca Paper Trading ---",
        f"ALPACA_PAPER_API_KEY={_settings.alpaca_paper_api_key}",
        f"ALPACA_PAPER_SECRET_KEY={_settings.alpaca_paper_secret_key}",
        f"ALPACA_PAPER_BASE_URL={_settings.alpaca_paper_base_url}",
        "",
        "# --- Alpaca Live Trading ---",
        f"ALPACA_LIVE_API_KEY={_settings.alpaca_live_api_key}",
        f"ALPACA_LIVE_SECRET_KEY={_settings.alpaca_live_secret_key}",
        f"ALPACA_LIVE_BASE_URL={_settings.alpaca_live_base_url}",
        "",
        "# --- Data Feed ---",
        f"ALPACA_DATA_FEED={_settings.alpaca_data_feed}",
        "",
        "# --- Trading Loop ---",
        f"TRADING_INTERVAL={_settings.trading_interval}",
        f"MAX_CONSECUTIVE_ERRORS={_settings.max_consecutive_errors}",
        f"ERROR_COOLDOWN_SECONDS={_settings.error_cooldown_seconds}",
        "",
        "# --- Risk Management ---",
        f"MAX_POSITION_SIZE_PCT={_settings.max_position_size_pct}",
        f"MAX_DAILY_LOSS_PCT={_settings.max_daily_loss_pct}",
        f"MAX_PORTFOLIO_EXPOSURE={_settings.max_portfolio_exposure}",
        f"MAX_SINGLE_STOCK_PCT={_settings.max_single_stock_pct}",
        f"MAX_LEVERAGE={_settings.max_leverage}",
        f"MAX_OPEN_POSITIONS={_settings.max_open_positions}",
        f"MAX_ORDERS_PER_DAY={_settings.max_orders_per_day}",
        f"MAX_CORRELATED_POSITIONS={_settings.max_correlated_positions}",
        f"DEFAULT_STOP_LOSS_PCT={_settings.default_stop_loss_pct}",
        f"DEFAULT_TAKE_PROFIT_PCT={_settings.default_take_profit_pct}",
        "",
        "# --- Strategy ---",
        f"ACTIVE_STRATEGY={_settings.active_strategy}",
        f"TRADING_SYMBOLS={_settings.trading_symbols}",
        f"TIMEFRAME={_settings.timeframe}",
        f"LOOKBACK_BARS={_settings.lookback_bars}",
        f"MULTI_STRATEGY_CONFIG={_settings.multi_strategy_config}",
        "",
        "# --- Strategy Parameters (Momentum) ---",
        f"MOMENTUM_FAST_EMA={_settings.momentum_fast_ema}",
        f"MOMENTUM_SLOW_EMA={_settings.momentum_slow_ema}",
        f"MOMENTUM_RSI_PERIOD={_settings.momentum_rsi_period}",
        f"MOMENTUM_RSI_OVERSOLD={_settings.momentum_rsi_oversold}",
        f"MOMENTUM_RSI_OVERBOUGHT={_settings.momentum_rsi_overbought}",
        f"MOMENTUM_ATR_PERIOD={_settings.momentum_atr_period}",
        f"MOMENTUM_ATR_SL_MULT={_settings.momentum_atr_sl_mult}",
        f"MOMENTUM_ATR_TP_MULT={_settings.momentum_atr_tp_mult}",
        "",
        "# --- Strategy Parameters (Mean Reversion) ---",
        f"MEAN_REV_BB_PERIOD={_settings.mean_rev_bb_period}",
        f"MEAN_REV_BB_STD={_settings.mean_rev_bb_std}",
        f"MEAN_REV_ZSCORE_ENTRY={_settings.mean_rev_zscore_entry}",
        f"MEAN_REV_ZSCORE_EXIT={_settings.mean_rev_zscore_exit}",
        f"MEAN_REV_RSI_PERIOD={_settings.mean_rev_rsi_period}",
        "",
        "# --- ML Model ---",
        f"ML_MODEL_PATH={_settings.ml_model_path}",
        f"ML_RETRAIN_INTERVAL_HOURS={_settings.ml_retrain_interval_hours}",
        f"ML_MIN_CONFIDENCE={_settings.ml_min_confidence}",
        "",
        "# --- Adaptive Intelligence Layer ---",
        f"INTELLIGENCE_ENABLED={'true' if _settings.intelligence_enabled else 'false'}",
        f"INTELLIGENCE_MIN_TRADE_SCORE={_settings.intelligence_min_trade_score}",
        f"INTELLIGENCE_DRIFT_WINDOW={_settings.intelligence_drift_window}",
        f"INTELLIGENCE_DRIFT_MIN_SAMPLES={_settings.intelligence_drift_min_samples}",
        f"INTELLIGENCE_DRIFT_ALERT_DROP={_settings.intelligence_drift_alert_drop}",
        "",
        "# --- Notifications ---",
        f"TELEGRAM_BOT_TOKEN={_settings.telegram_bot_token}",
        f"TELEGRAM_CHAT_ID={_settings.telegram_chat_id}",
        f"DISCORD_WEBHOOK_URL={_settings.discord_webhook_url}",
        f"NOTIFY_ON_TRADE={'true' if _settings.notify_on_trade else 'false'}",
        f"NOTIFY_ON_ERROR={'true' if _settings.notify_on_error else 'false'}",
        f"NOTIFY_ON_SIGNAL={'true' if _settings.notify_on_signal else 'false'}",
        "",
        "# --- Database ---",
        f"DATABASE_URL={_settings.database_url}",
        "",
        "# --- Logging ---",
        f"LOG_LEVEL={_settings.log_level}",
        f"LOG_FILE={_settings.log_file}",
        "",
        "# --- Backtesting ---",
        f"BACKTEST_COMMISSION_PCT={_settings.backtest_commission_pct}",
        f"BACKTEST_SLIPPAGE_PCT={_settings.backtest_slippage_pct}",
        f"BACKTEST_INITIAL_CASH={_settings.backtest_initial_cash}",
        "",
        "# --- Automation Scheduler ---",
        f"AUTO_BACKTEST_INTERVAL_HOURS={_settings.auto_backtest_interval_hours}",
        f"AUTO_TRAIN_INTERVAL_HOURS={_settings.auto_train_interval_hours}",
        f"AUTO_SWEEP_INTERVAL_HOURS={_settings.auto_sweep_interval_hours}",
        f"AUTO_BACKTEST_SYMBOLS={_settings.auto_backtest_symbols}",
        f"AUTO_TRAIN_BARS={_settings.auto_train_bars}",
        "",
        "# --- Execution Simulator ---",
        f"ENABLE_EXECUTION_SIMULATOR={'true' if _settings.enable_execution_simulator else 'false'}",
        f"EXECUTION_SIMULATOR_PRESET={_settings.execution_simulator_preset}",
        "",
        "# --- Risk Engine (Multi-Layer) ---",
        f"RISK_PORTFOLIO_LAYER_ENABLED={'true' if _settings.risk_portfolio_layer_enabled else 'false'}",
        f"RISK_ACCOUNT_LAYER_ENABLED={'true' if _settings.risk_account_layer_enabled else 'false'}",
        f"RISK_EXPOSURE_LAYER_ENABLED={'true' if _settings.risk_exposure_layer_enabled else 'false'}",
        f"RISK_EXECUTION_LAYER_ENABLED={'true' if _settings.risk_execution_layer_enabled else 'false'}",
        f"RISK_MAX_DRAWDOWN_PCT={_settings.risk_max_drawdown_pct}",
        f"RISK_MAX_DAILY_LOSS_PCT={_settings.risk_max_daily_loss_pct}",
        f"RISK_MAX_WEEKLY_LOSS_PCT={_settings.risk_max_weekly_loss_pct}",
        f"RISK_MIN_CASH_RESERVE_PCT={_settings.risk_min_cash_reserve_pct}",
        f"RISK_CONSECUTIVE_LOSS_LIMIT={_settings.risk_consecutive_loss_limit}",
        f"RISK_DAILY_TRADE_LIMIT={_settings.risk_daily_trade_limit}",
        f"RISK_MAX_POSITIONS={_settings.risk_max_positions}",
        f"RISK_MAX_SINGLE_STOCK_PCT={_settings.risk_max_single_stock_pct}",
        f"RISK_MAX_SECTOR_EXPOSURE_PCT={_settings.risk_max_sector_exposure_pct}",
        f"RISK_MAX_CORRELATION={_settings.risk_max_correlation}",
        "",
    ]
    return "\n".join(lines) + "\n"

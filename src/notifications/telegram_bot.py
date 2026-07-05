"""
Telegram Bot — Full management interface via aiogram 3.x.
Provides interactive controls, real-time alerts, and portfolio management.
ALL settings configurable in real-time via Telegram.

Commands:
    /start        - Welcome + status overview
    /status       - Account balance, equity, positions
    /positions    - Detailed position list
    /orders       - Open orders
    /pnl          - Today's P&L summary
    /trades       - Recent trade history
    /signals      - Current active signals
    /buy          - Manual buy order (e.g., /buy AAPL 10)
    /sell         - Manual sell order (e.g., /sell AAPL 10)
    /close        - Close position (e.g., /close AAPL)
    /closeall     - Emergency: close all positions
    /cancelall    - Cancel all pending orders
    /pause        - Pause bot trading
    /resume       - Resume bot trading
    /strategy     - Show/switch active strategy
    /risk         - Show risk parameters
    /config       - View all settings (e.g., /config risk)
    /set          - Universal setter (e.g., /set momentum_fast_ema 8)
    /setrisk      - Set risk param (e.g., /setrisk max_daily_loss_pct 0.05)
    /setstrategy  - Switch strategy (momentum|mean_reversion|ml)
    /setsymbols   - Change symbols (e.g., /setsymbols AAPL,TSLA,BTC/USD)
    /setinterval  - Change cycle interval (e.g., /setinterval 120)
    /settf        - Change timeframe (e.g., /settf 15Min)
    /setlookback  - Change lookback (e.g., /setlookback 500)
    /setauto      - Configure automation (e.g., /setauto train 12)
    /setnotify    - Toggle alerts (e.g., /setnotify signal on)
    /backtest     - Backtrader backtest (e.g., /backtest AAPL 30)
    /backtestvbt  - VectorBT backtest (e.g., /backtestvbt AAPL 30)
    /sweep        - Parameter sweep (e.g., /sweep TSLA 60)
    /train        - Train ML model (e.g., /train AAPL,TSLA 1000)
    /modelinfo    - ML model metadata (type, date, features, accuracy)
    /predict      - ML prediction (e.g., /predict BTC/USD)
    /walkforward  - Walk-forward optimization (e.g., /walkforward BTC/USD 1000)
    /montecarlo   - Monte Carlo simulation (e.g., /montecarlo AAPL 1000 500)
    /portfolio    - Portfolio-level backtest (e.g., /portfolio 500)
    /models       - List model versions
    /rollback     - Rollback model (e.g., /rollback v001)
    /journal      - Recent trade journal (e.g., /journal AAPL 10)
    /journalstats - Journal performance analytics
    /help         - Show all commands
"""

import asyncio
import os
from datetime import datetime
from typing import Optional

from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.enums import ParseMode

from src.utils.logger import get_logger
from src.core.runtime_state import RuntimeState

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Per-user command rate limiter
# ──────────────────────────────────────────────────────────────────────────

from collections import defaultdict
import time as _time


class _CommandRateLimiter:
    """Per-user token bucket rate limiter."""

    def __init__(self, max_commands: int = 10, window_seconds: float = 60.0):
        self._max = max_commands
        self._window = window_seconds
        self._timestamps: dict[int, list[float]] = defaultdict(list)

    def is_allowed(self, user_id: int) -> bool:
        """Check if user is within rate limit."""
        now = _time.time()
        timestamps = self._timestamps[user_id]
        self._timestamps[user_id] = [t for t in timestamps if now - t < self._window]
        if len(self._timestamps[user_id]) >= self._max:
            return False
        self._timestamps[user_id].append(now)
        return True


_rate_limiter = _CommandRateLimiter(max_commands=10, window_seconds=60.0)

router = Router()

# These get set during bot initialization
_broker = None
_engine = None
_risk_manager = None
_strategy = None
_db = None
_runtime_state: Optional[RuntimeState] = None  # Thread-safe pause state
_authorized_users: set[int] = set()
_AUTH_PIN = os.environ.get("TELEGRAM_AUTH_PIN", "")
_health_monitor = None
_live_metrics = None
_scheduler = None
_event_bus = None
_trade_manager = None
_reconciler = None
_ops_handler = None

# Thread-safe lock for broker operations (broker is called from Telegram's async thread)
import threading
_broker_lock = threading.Lock()

# Shared runtime state — Telegram commands write, main loop reads
_runtime_changes = {
    "strategy_name": None,      # str: new strategy name to switch to
    "symbols": None,            # list: new symbols to watch
    "interval": None,           # int: new interval in seconds
    "risk_updates": {},         # dict: risk param updates (e.g., {"max_daily_loss_pct": 0.03})
    "trigger_backtest": False,  # bool: run backtest next cycle
    "trigger_train": False,     # bool: trigger ML retraining
    "timeframe": None,          # str: new timeframe
    "lookback": None,           # int: new lookback bars
    "config_updates": {},       # dict: general settings updates
}
_runtime_lock = threading.Lock()


def get_runtime_changes() -> dict:
    """Read and clear pending runtime changes (called by main loop)."""
    with _runtime_lock:
        changes = {}
        for key, val in _runtime_changes.items():
            if val and val != {} and val != False:
                changes[key] = val
        # Reset after reading
        _runtime_changes["strategy_name"] = None
        _runtime_changes["symbols"] = None
        _runtime_changes["interval"] = None
        _runtime_changes["risk_updates"] = {}
        _runtime_changes["trigger_backtest"] = False
        _runtime_changes["trigger_train"] = False
        _runtime_changes["timeframe"] = None
        _runtime_changes["lookback"] = None
        _runtime_changes["config_updates"] = {}
        return changes


def set_components(broker, engine, risk_manager, strategy, db=None, authorized_chat_ids: list[int] = None,
                   health_monitor=None, live_metrics=None, scheduler=None,
                   event_bus=None, trade_manager=None, reconciler=None, ops_handler=None,
                   runtime_state: Optional[RuntimeState] = None):
    """Inject trading components into the bot module."""
    global _broker, _engine, _risk_manager, _strategy, _db, _authorized_users
    global _health_monitor, _live_metrics, _scheduler, _event_bus, _trade_manager
    global _reconciler, _ops_handler, _runtime_state
    _broker = broker
    _engine = engine
    _risk_manager = risk_manager
    _strategy = strategy
    _db = db
    if authorized_chat_ids:
        _authorized_users = set(authorized_chat_ids)
    if health_monitor:
        _health_monitor = health_monitor
    if live_metrics:
        _live_metrics = live_metrics
    if scheduler:
        _scheduler = scheduler
    if event_bus:
        _event_bus = event_bus
    if trade_manager:
        _trade_manager = trade_manager
    if reconciler:
        _reconciler = reconciler
    if ops_handler:
        _ops_handler = ops_handler
    if runtime_state:
        _runtime_state = runtime_state
    else:
        # Create default RuntimeState if not provided
        _runtime_state = RuntimeState()


def _is_authorized(message: Message) -> bool:
    """Check if user is authorized to use the bot. Deny-by-default."""
    if not _authorized_users:
        # If no users registered and no PIN, check TELEGRAM_CHAT_ID
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if chat_id and str(message.chat.id) == chat_id:
            _authorized_users.add(message.from_user.id)
            return True
        logger.warning("telegram.unauthorized_attempt", user_id=message.from_user.id, username=getattr(message.from_user, 'username', 'unknown'))
        return False
    return message.from_user.id in _authorized_users


# ──────────────────────────────────────────────────────────────────────────
# Command Handlers
# ──────────────────────────────────────────────────────────────────────────

@router.message(Command("auth"))
async def cmd_auth(message: Message):
    """Authenticate with PIN code."""
    if not _AUTH_PIN:
        await message.answer("⚠️ No AUTH_PIN configured. Using chat ID auth.")
        return

    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Usage: /auth <PIN>")
        return

    pin = parts[1].strip()
    # Delete message immediately (contains PIN)
    try:
        await message.delete()
    except Exception:
        pass

    if pin == _AUTH_PIN:
        _authorized_users.add(message.from_user.id)
        logger.warning("telegram.user_authorized", user_id=message.from_user.id, username=message.from_user.username)
        await message.answer("✅ Authenticated successfully.")
    else:
        logger.warning("telegram.auth_failed", user_id=message.from_user.id, username=message.from_user.username)
        await message.answer("❌ Invalid PIN.")


@router.message(CommandStart())
async def cmd_start(message: Message):
    if not _is_authorized(message):
        await message.answer("Unauthorized. Use /auth <PIN> to authenticate.")
        return

    text = (
        "<b>Algo Trader Bot</b>\n\n"
        "Your trading bot is connected and ready.\n\n"
        "<b>Quick Commands:</b>\n"
        "/status - Account overview\n"
        "/positions - Open positions\n"
        "/pnl - Today's P&L\n"
        "/trades - Recent trades\n"
        "/help - All commands\n\n"
        f"<i>Mode: {'PAUSED' if _runtime_state and _runtime_state.is_paused() else 'ACTIVE'}</i>"
    )
    await message.answer(text, parse_mode=ParseMode.HTML)


@router.message(Command("help"))
async def cmd_help(message: Message):
    if not _is_authorized(message):
        return

    text = (
        "<b>Available Commands:</b>\n\n"
        "<b>Info:</b>\n"
        "/status - Account balance & equity\n"
        "/positions - All open positions\n"
        "/orders - Pending orders\n"
        "/pnl - Daily P&L summary\n"
        "/trades - Recent trade history\n"
        "/signals - Current strategy signals\n\n"
        "<b>Trading:</b>\n"
        "/buy SYMBOL QTY - Market buy\n"
        "/sell SYMBOL QTY - Market sell\n"
        "/close SYMBOL - Close position\n"
        "/closeall - EMERGENCY: close all\n"
        "/cancelall - Cancel all orders\n\n"
        "<b>Control:</b>\n"
        "/pause - Pause auto-trading\n"
        "/resume - Resume auto-trading\n"
        "/strategy - View active strategy\n"
        "/risk - View risk parameters\n\n"
        "<b>Configuration:</b>\n"
        "/config [category] - View settings\n"
        "/configfull [section] - Full config (masked)\n"
        "/set PARAM VALUE - Change any setting\n"
        "/setrisk PARAM VALUE - Set risk param\n"
        "/setstrategy NAME - Switch strategy\n"
        "/setsymbols SYM1,SYM2 - Change symbols\n"
        "/setinterval SECS - Change cycle interval\n"
        "/settf TIMEFRAME - Change timeframe\n"
        "/setlookback BARS - Change lookback\n"
        "/setauto TYPE HOURS - Configure automation\n"
        "/setnotify TYPE on|off - Toggle alerts\n\n"
        "<b>Broker & Mode:</b>\n"
        "/setalpaca paper|live KEY SECRET - Set API keys\n"
        "/setmode paper|live - Switch trading mode\n"
        "/saveconfig - Persist config to .env\n"
        "/loadconfig - Reload from .env\n"
        "/export - Export all config (masked)\n\n"
        "<b>Strategy Management:</b>\n"
        "/newstrategy NAME TPL SYMS [TF] [INT] - Create\n"
        "/liststrats - List all strategies\n"
        "/editstrat NAME FIELD VALUE - Edit strategy\n"
        "/deletestrat NAME - Delete strategy\n"
        "/copystrat SRC NEW - Duplicate strategy\n"
        "/templates [NAME] - View templates\n"
        "/applystrats - Apply to orchestrator\n\n"
        "<b>Backtesting & ML:</b>\n"
        "/backtest [SYMBOL] [DAYS] - Backtrader backtest\n"
        "/backtestvbt [SYMBOL] [DAYS] - VectorBT backtest\n"
        "/sweep [SYMBOL] [DAYS] - Parameter sweep\n"
        "/train [SYMBOLS] [BARS] - Train ML model\n"
        "/modelinfo - ML model metadata\n"
        "/predict [SYMBOL] - ML prediction + confidence\n\n"
        "<b>Asset Discovery & Market:</b>\n"
        "/assets [stocks|crypto] [exchange] [page] - Browse assets\n"
        "/search QUERY [stocks|crypto] - Search by name/symbol\n"
        "/price SYMBOL [SYMBOL2...] - Live price quotes\n"
        "/asset SYMBOL - Full asset details\n"
        "/watchlist - Current symbols + prices\n\n"
        "<b>Advanced Research:</b>\n"
        "/walkforward [SYMBOL] [BARS] - Walk-forward optimization\n"
        "/montecarlo [SYMBOL] [BARS] [SIMS] - Monte Carlo sim\n"
        "/portfolio [BARS] - Portfolio-level backtest\n"
        "/models - List model versions\n"
        "/rollback vXXX - Rollback to model version\n"
        "/journal [SYMBOL] [N] - Trade journal entries\n"
        "/journalstats - Journal analytics\n\n"
        "<b>Sector Management:</b>\n"
        "/setsector SYMBOL SECTOR - Classify symbol sector\n"
        "/sectors - List cached sector classifications\n"
    )
    await message.answer(text, parse_mode=ParseMode.HTML)


@router.message(Command("status"))
async def cmd_status(message: Message):
    if not _is_authorized(message) or not _broker:
        await message.answer("Not connected.")
        return

    try:
        with _broker_lock:
            account = _broker.get_account()
            positions = _broker.get_positions()

        pnl = account['equity'] - account['last_equity']
        pnl_pct = (pnl / account['last_equity'] * 100) if account['last_equity'] > 0 else 0

        text = (
            f"<b>Account Status</b> {'[PAUSED]' if _runtime_state and _runtime_state.is_paused() else '[ACTIVE]'}\n"
            f"{'=' * 30}\n"
            f"Equity:       <code>${account['equity']:>12,.2f}</code>\n"
            f"Cash:         <code>${account['cash']:>12,.2f}</code>\n"
            f"Buying Power: <code>${account['buying_power']:>12,.2f}</code>\n"
            f"{'─' * 30}\n"
            f"Day P&L:      <code>${pnl:>+12,.2f} ({pnl_pct:+.2f}%)</code>\n"
            f"Positions:    <code>{len(positions):>12d}</code>\n"
            f"Day Trades:   <code>{account['day_trade_count']:>12d}</code>\n"
        )
        await message.answer(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer("Operation failed. Check system logs for details.")


@router.message(Command("positions"))
async def cmd_positions(message: Message):
    if not _is_authorized(message) or not _broker:
        return

    try:
        with _broker_lock:
            positions = _broker.get_positions()
        if not positions:
            await message.answer("No open positions.")
            return

        lines = ["<b>Open Positions:</b>\n"]
        total_pnl = 0.0
        for p in positions:
            emoji = "+" if p['unrealized_pl'] >= 0 else ""
            total_pnl += p['unrealized_pl']
            lines.append(
                f"<code>{p['symbol']:8s}</code> | "
                f"{p['qty']:.4f} @ ${p['avg_entry_price']:.2f} | "
                f"PnL: <code>{emoji}${p['unrealized_pl']:.2f}</code> "
                f"({p['unrealized_plpc']:.1%})"
            )
        lines.append(f"\n<b>Total Unrealized: ${total_pnl:+,.2f}</b>")
        await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer("Operation failed. Check system logs for details.")


@router.message(Command("orders"))
async def cmd_orders(message: Message):
    if not _is_authorized(message) or not _broker:
        return

    try:
        with _broker_lock:
            orders = _broker.get_orders(status="open")
        if not orders:
            await message.answer("No pending orders.")
            return

        lines = ["<b>Open Orders:</b>\n"]
        for o in orders:
            price_str = f"@ ${o['limit_price']:.2f}" if o['limit_price'] else "MARKET"
            lines.append(
                f"<code>{o['symbol']:8s}</code> | "
                f"{o['side'].upper()} {o['qty']:.4f} {price_str} | "
                f"{o['status']}"
            )
        await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer("Operation failed. Check system logs for details.")


@router.message(Command("pnl"))
async def cmd_pnl(message: Message):
    if not _is_authorized(message) or not _risk_manager:
        return

    summary = _risk_manager.get_daily_summary()
    text = (
        f"<b>Daily P&L Summary</b>\n"
        f"{'=' * 30}\n"
        f"Trades:      {summary['trades']}\n"
        f"Win Rate:    {summary['win_rate']}\n"
        f"Daily PnL:   ${summary['daily_pnl']:.2f}\n"
        f"Return:      {summary['daily_return']}\n"
        f"Halted:      {'Yes' if summary['is_halted'] else 'No'}\n"
    )
    await message.answer(text, parse_mode=ParseMode.HTML)


@router.message(Command("trades"))
async def cmd_trades(message: Message):
    if not _is_authorized(message) or not _db:
        return

    try:
        trades = _db.get_trades(limit=10)
        if not trades:
            await message.answer("No trades recorded yet.")
            return

        lines = ["<b>Recent Trades (last 10):</b>\n"]
        for t in trades:
            side = t.get('side', '?').upper()
            symbol = t.get('symbol', '?')
            qty = t.get('qty', 0)
            price = t.get('price', 0)
            pnl = t.get('pnl')
            ts = t.get('timestamp', '')[:16]
            pnl_str = f" PnL: ${pnl:.2f}" if pnl else ""
            lines.append(f"  {side} {symbol} x{qty:.2f} @ ${price:.2f}{pnl_str} [{ts}]")

        await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer("Operation failed. Check system logs for details.")


@router.message(Command("signals"))
async def cmd_signals(message: Message):
    if not _is_authorized(message) or not _strategy or not _broker:
        return

    try:
        data = {}
        with _broker_lock:
            for symbol in _strategy.symbols[:5]:
                try:
                    df = _broker.get_bars_df(symbol, _strategy.timeframe, 200)
                    if df is not None and len(df) >= 50:
                        data[symbol] = df
                except Exception:
                    continue

        if not data:
            await message.answer("No data available for signal generation.")
            return

        signals = _strategy.generate_signals(data)
        if not signals:
            await message.answer("No actionable signals right now.")
            return

        lines = ["<b>Active Signals:</b>\n"]
        for s in signals[:10]:
            emoji = "BUY" if s.signal.value > 0 else "SELL"
            lines.append(
                f"<code>{s.symbol:8s}</code> | {emoji} | "
                f"Conf: {s.confidence:.0%} | "
                f"${s.price:.2f}"
            )
        await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer("Operation failed. Check system logs for details.")


# ──────────────────────────────────────────────────────────────────────────
# Trading Commands
# ──────────────────────────────────────────────────────────────────────────

@router.message(Command("buy"))
async def cmd_buy(message: Message):
    if not _is_authorized(message) or not _broker:
        return
    if not _rate_limiter.is_allowed(message.from_user.id):
        await message.answer("⚠️ Rate limit exceeded. Please wait before sending more commands.")
        return

    parts = message.text.split()
    if len(parts) < 3:
        await message.answer("Usage: /buy SYMBOL QTY\nExample: /buy AAPL 10")
        return

    symbol = parts[1].upper()
    try:
        qty = float(parts[2])
    except ValueError:
        await message.answer("Invalid quantity.")
        return

    # Confirmation keyboard
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Confirm BUY", callback_data=f"buy|{symbol}|{qty}"),
            InlineKeyboardButton(text="Cancel", callback_data="cancel_order"),
        ]
    ])
    await message.answer(
        f"<b>Confirm Market BUY:</b>\n{symbol} x {qty}",
        parse_mode=ParseMode.HTML, reply_markup=kb
    )


@router.message(Command("sell"))
async def cmd_sell(message: Message):
    if not _is_authorized(message) or not _broker:
        return
    if not _rate_limiter.is_allowed(message.from_user.id):
        await message.answer("⚠️ Rate limit exceeded. Please wait before sending more commands.")
        return

    parts = message.text.split()
    if len(parts) < 3:
        await message.answer("Usage: /sell SYMBOL QTY\nExample: /sell AAPL 10")
        return

    symbol = parts[1].upper()
    try:
        qty = float(parts[2])
    except ValueError:
        await message.answer("Invalid quantity.")
        return

    kb = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Confirm SELL", callback_data=f"sell|{symbol}|{qty}"),
            InlineKeyboardButton(text="Cancel", callback_data="cancel_order"),
        ]
    ])
    await message.answer(
        f"<b>Confirm Market SELL:</b>\n{symbol} x {qty}",
        parse_mode=ParseMode.HTML, reply_markup=kb
    )


@router.message(Command("close"))
async def cmd_close(message: Message):
    if not _is_authorized(message) or not _broker:
        return
    if not _rate_limiter.is_allowed(message.from_user.id):
        await message.answer("⚠️ Rate limit exceeded. Please wait before sending more commands.")
        return

    parts = message.text.split()
    if len(parts) < 2:
        await message.answer("Usage: /close SYMBOL\nExample: /close AAPL")
        return

    symbol = parts[1].upper()
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text=f"Close {symbol}", callback_data=f"close|{symbol}"),
            InlineKeyboardButton(text="Cancel", callback_data="cancel_order"),
        ]
    ])
    await message.answer(
        f"<b>Close entire position in {symbol}?</b>",
        parse_mode=ParseMode.HTML, reply_markup=kb
    )


@router.message(Command("closeall"))
async def cmd_closeall(message: Message):
    if not _is_authorized(message) or not _broker:
        return
    if not _rate_limiter.is_allowed(message.from_user.id):
        await message.answer("⚠️ Rate limit exceeded. Please wait before sending more commands.")
        return

    kb = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="YES - CLOSE ALL", callback_data="closeall"),
            InlineKeyboardButton(text="Cancel", callback_data="cancel_order"),
        ]
    ])
    await message.answer(
        "<b>EMERGENCY: Close ALL positions and cancel ALL orders?</b>",
        parse_mode=ParseMode.HTML, reply_markup=kb
    )


@router.message(Command("cancelall"))
async def cmd_cancelall(message: Message):
    if not _is_authorized(message) or not _broker:
        return

    try:
        with _broker_lock:
            _broker.cancel_all_orders()
        await message.answer("All pending orders cancelled.")
    except Exception as e:
        await message.answer("Operation failed. Check system logs for details.")


# ──────────────────────────────────────────────────────────────────────────
# Bot Control Commands
# ──────────────────────────────────────────────────────────────────────────

@router.message(Command("pause"))
async def cmd_pause(message: Message):
    if not _is_authorized(message):
        return
    global _runtime_state
    if _runtime_state:
        _runtime_state.pause()
    await message.answer("Bot PAUSED. Auto-trading disabled.\nUse /resume to restart.")


@router.message(Command("resume"))
async def cmd_resume(message: Message):
    if not _is_authorized(message):
        return
    global _runtime_state
    if _runtime_state:
        _runtime_state.resume()
    if _risk_manager:
        _risk_manager.daily_stats.is_halted = False
        _risk_manager.daily_stats.halt_reason = ""
    await message.answer("Bot RESUMED. Auto-trading active.")


@router.message(Command("strategy"))
async def cmd_strategy(message: Message):
    if not _is_authorized(message) or not _strategy:
        return

    text = (
        f"<b>Active Strategy:</b> {_strategy.name}\n"
        f"Symbols: {', '.join(_strategy.symbols[:10])}\n"
        f"Timeframe: {_strategy.timeframe}\n"
        f"Lookback: {_strategy.lookback} bars\n"
        f"Active: {'Yes' if _strategy.is_active else 'No'}\n"
    )
    await message.answer(text, parse_mode=ParseMode.HTML)


@router.message(Command("risk"))
async def cmd_risk(message: Message):
    if not _is_authorized(message) or not _risk_manager:
        return

    limits = _risk_manager.limits
    text = (
        f"<b>Risk Parameters:</b>\n"
        f"{'=' * 30}\n"
        f"Max risk/trade:    {limits.max_position_size_pct:.0%}\n"
        f"Daily loss limit:  {limits.max_daily_loss_pct:.0%}\n"
        f"Max exposure:      {limits.max_portfolio_exposure:.0%}\n"
        f"Max single stock:  {limits.max_single_stock_pct:.0%}\n"
        f"Max leverage:      {limits.max_leverage:.1f}x\n"
        f"Stop-loss:         {limits.default_stop_loss_pct:.0%}\n"
        f"Take-profit:       {limits.default_take_profit_pct:.0%}\n"
        f"Max positions:     {limits.max_open_positions}\n"
        f"Max orders/day:    {limits.max_orders_per_day}\n"
    )
    await message.answer(text, parse_mode=ParseMode.HTML)


# ──────────────────────────────────────────────────────────────────────────
# Runtime Configuration Commands
# ──────────────────────────────────────────────────────────────────────────

@router.message(Command("setrisk"))
async def cmd_setrisk(message: Message):
    """Set risk parameters at runtime.
    Usage: /setrisk max_daily_loss_pct 0.03
    """
    if not _is_authorized(message) or not _risk_manager:
        return

    parts = message.text.split()
    if len(parts) < 3:
        valid_params = [
            "max_position_size_pct", "max_daily_loss_pct", "max_portfolio_exposure",
            "max_single_stock_pct", "max_leverage", "default_stop_loss_pct",
            "default_take_profit_pct", "max_open_positions", "max_orders_per_day",
        ]
        await message.answer(
            "<b>Usage:</b> /setrisk PARAM VALUE\n\n"
            "<b>Parameters:</b>\n" + "\n".join(f"  {p}" for p in valid_params),
            parse_mode=ParseMode.HTML
        )
        return

    param_name = parts[1].lower()
    try:
        value = float(parts[2])
    except ValueError:
        await message.answer("Invalid value. Must be a number.")
        return

    # Validate and apply to risk manager
    limits = _risk_manager.limits
    if not hasattr(limits, param_name):
        await message.answer(f"Unknown parameter: {param_name}")
        return

    # Safety check for percentages
    pct_params = {"max_position_size_pct", "max_daily_loss_pct", "max_portfolio_exposure",
                  "max_single_stock_pct", "default_stop_loss_pct", "default_take_profit_pct"}
    if param_name in pct_params and not (0 < value <= 1.0):
        await message.answer(f"Percentage values must be between 0 and 1.0 (got {value})")
        return

    old_value = getattr(limits, param_name)
    setattr(limits, param_name, type(old_value)(value))

    with _runtime_lock:
        _runtime_changes["risk_updates"][param_name] = value

    await message.answer(
        f"Risk parameter updated:\n"
        f"  {param_name}: {old_value} -> {value}",
        parse_mode=ParseMode.HTML
    )


@router.message(Command("setstrategy"))
async def cmd_setstrategy(message: Message):
    """Switch active strategy at runtime.
    Usage: /setstrategy momentum|mean_reversion|ml
    """
    if not _is_authorized(message):
        return

    parts = message.text.split()
    if len(parts) < 2:
        await message.answer(
            "<b>Usage:</b> /setstrategy NAME\n\n"
            "<b>Available:</b> momentum, mean_reversion, ml",
            parse_mode=ParseMode.HTML
        )
        return

    new_strategy = parts[1].lower()
    valid = {"momentum", "mean_reversion", "ml"}
    if new_strategy not in valid:
        await message.answer(f"Unknown strategy: {new_strategy}\nValid: {', '.join(valid)}")
        return

    with _runtime_lock:
        _runtime_changes["strategy_name"] = new_strategy

    await message.answer(
        f"Strategy switch requested: {new_strategy}\n"
        f"Will take effect on next cycle.",
        parse_mode=ParseMode.HTML
    )


@router.message(Command("setsymbols"))
async def cmd_setsymbols(message: Message):
    """Change watched symbols at runtime.
    Usage: /setsymbols AAPL,TSLA,BTC/USD
    """
    if not _is_authorized(message):
        return

    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        current = ", ".join(_strategy.symbols[:10]) if _strategy else "none"
        await message.answer(
            f"<b>Usage:</b> /setsymbols SYM1,SYM2,...\n\n"
            f"<b>Current:</b> {current}",
            parse_mode=ParseMode.HTML
        )
        return

    symbols = [s.strip().upper() for s in parts[1].split(",") if s.strip()]
    if not symbols:
        await message.answer("No valid symbols provided.")
        return

    with _runtime_lock:
        _runtime_changes["symbols"] = symbols

    await message.answer(
        f"Symbol change requested: {', '.join(symbols)}\n"
        f"Will take effect on next cycle.",
        parse_mode=ParseMode.HTML
    )


@router.message(Command("setinterval"))
async def cmd_setinterval(message: Message):
    """Change trading cycle interval.
    Usage: /setinterval 120
    """
    if not _is_authorized(message):
        return

    parts = message.text.split()
    if len(parts) < 2:
        await message.answer("<b>Usage:</b> /setinterval SECONDS\nExample: /setinterval 120", parse_mode=ParseMode.HTML)
        return

    try:
        new_interval = int(parts[1])
    except ValueError:
        await message.answer("Invalid interval. Must be an integer (seconds).")
        return

    if new_interval < 10 or new_interval > 3600:
        await message.answer("Interval must be between 10 and 3600 seconds.")
        return

    with _runtime_lock:
        _runtime_changes["interval"] = new_interval

    await message.answer(f"Interval change requested: {new_interval}s\nWill take effect on next cycle.")


@router.message(Command("config"))
async def cmd_config(message: Message):
    """Show all current configuration, grouped by category.
    Usage: /config [category]
    Categories: risk, strategy, ml, auto, notify, all
    """
    if not _is_authorized(message):
        return

    from config.settings import settings

    parts = message.text.split()
    category = parts[1].lower() if len(parts) > 1 else "all"

    sections = {}

    sections["strategy"] = (
        f"<b>📊 Strategy</b>\n"
        f"  Active:    {settings.active_strategy}\n"
        f"  Symbols:   {settings.trading_symbols}\n"
        f"  Timeframe: {settings.timeframe}\n"
        f"  Lookback:  {settings.lookback_bars} bars\n"
        f"  Interval:  {settings.trading_interval}s\n"
    )

    sections["risk"] = (
        f"<b>🛡️ Risk</b>\n"
        f"  Max risk/trade:   {settings.max_position_size_pct:.0%}\n"
        f"  Daily loss limit: {settings.max_daily_loss_pct:.0%}\n"
        f"  Max exposure:     {settings.max_portfolio_exposure:.0%}\n"
        f"  Max single stock: {settings.max_single_stock_pct:.0%}\n"
        f"  Max leverage:     {settings.max_leverage:.1f}x\n"
        f"  Stop-loss:        {settings.default_stop_loss_pct:.0%}\n"
        f"  Take-profit:      {settings.default_take_profit_pct:.0%}\n"
        f"  Max positions:    {settings.max_open_positions}\n"
        f"  Max orders/day:   {settings.max_orders_per_day}\n"
    )

    sections["momentum"] = (
        f"<b>📈 Momentum Params</b>\n"
        f"  Fast EMA:       {settings.momentum_fast_ema}\n"
        f"  Slow EMA:       {settings.momentum_slow_ema}\n"
        f"  RSI Period:     {settings.momentum_rsi_period}\n"
        f"  RSI Oversold:   {settings.momentum_rsi_oversold}\n"
        f"  RSI Overbought: {settings.momentum_rsi_overbought}\n"
        f"  ATR Period:     {settings.momentum_atr_period}\n"
        f"  ATR SL Mult:    {settings.momentum_atr_sl_mult}x\n"
        f"  ATR TP Mult:    {settings.momentum_atr_tp_mult}x\n"
    )

    sections["ml"] = (
        f"<b>🧠 ML</b>\n"
        f"  Model Path:     {settings.ml_model_path}\n"
        f"  Retrain Every:  {settings.ml_retrain_interval_hours}h\n"
        f"  Min Confidence: {settings.ml_min_confidence:.0%}\n"
    )

    sections["auto"] = (
        f"<b>⚙️ Automation</b>\n"
        f"  Auto-backtest: every {settings.auto_backtest_interval_hours}h {'(disabled)' if settings.auto_backtest_interval_hours == 0 else ''}\n"
        f"  Auto-train:    every {settings.auto_train_interval_hours}h {'(disabled)' if settings.auto_train_interval_hours == 0 else ''}\n"
        f"  Auto-sweep:    every {settings.auto_sweep_interval_hours}h {'(disabled)' if settings.auto_sweep_interval_hours == 0 else ''}\n"
        f"  Train bars:    {settings.auto_train_bars}\n"
    )

    sections["notify"] = (
        f"<b>🔔 Notifications</b>\n"
        f"  On trade:  {'✅' if settings.notify_on_trade else '❌'}\n"
        f"  On error:  {'✅' if settings.notify_on_error else '❌'}\n"
        f"  On signal: {'✅' if settings.notify_on_signal else '❌'}\n"
    )

    if category == "all":
        text = "\n".join(sections.values())
    elif category in sections:
        text = sections[category]
    else:
        text = (
            f"Unknown category: {category}\n\n"
            f"<b>Available:</b> strategy, risk, momentum, ml, auto, notify, all"
        )

    await message.answer(text, parse_mode=ParseMode.HTML)


@router.message(Command("set"))
async def cmd_set(message: Message):
    """Universal config setter — change any setting in real-time.
    Usage: /set PARAM VALUE

    Examples:
        /set timeframe 15Min
        /set lookback 500
        /set momentum_fast_ema 8
        /set momentum_slow_ema 21
        /set ml_min_confidence 0.70
        /set auto_backtest_interval_hours 3
        /set notify_on_signal true
        /set max_daily_loss_pct 0.03
    """
    if not _is_authorized(message):
        return

    from config.settings import settings

    parts = message.text.split(maxsplit=2)
    if len(parts) < 3:
        await message.answer(
            "<b>Usage:</b> /set PARAM VALUE\n\n"
            "<b>Examples:</b>\n"
            "  /set timeframe 15Min\n"
            "  /set lookback 500\n"
            "  /set momentum_fast_ema 8\n"
            "  /set momentum_slow_ema 21\n"
            "  /set ml_min_confidence 0.70\n"
            "  /set auto_backtest_interval_hours 3\n"
            "  /set auto_train_interval_hours 12\n"
            "  /set notify_on_signal true\n"
            "  /set max_daily_loss_pct 0.03\n"
            "  /set default_stop_loss_pct 0.04\n\n"
            "Use /config to see all current values.",
            parse_mode=ParseMode.HTML
        )
        return

    param = parts[1].lower()
    raw_value = parts[2].strip()

    # --- Map of all settable parameters ---
    # These are the settings fields that can be changed at runtime
    SETTABLE_PARAMS = {
        # Trading Loop
        "timeframe": {"type": "choice", "choices": ["1Min", "5Min", "15Min", "30Min", "1Hour", "4Hour", "1Day"]},
        "lookback": {"type": "int", "min": 50, "max": 5000, "field": "lookback_bars"},
        "interval": {"type": "int", "min": 10, "max": 3600, "field": "trading_interval"},
        # Momentum
        "momentum_fast_ema": {"type": "int", "min": 2, "max": 100},
        "momentum_slow_ema": {"type": "int", "min": 5, "max": 500},
        "momentum_rsi_period": {"type": "int", "min": 2, "max": 100},
        "momentum_rsi_oversold": {"type": "int", "min": 5, "max": 50},
        "momentum_rsi_overbought": {"type": "int", "min": 50, "max": 95},
        "momentum_atr_period": {"type": "int", "min": 2, "max": 100},
        "momentum_atr_sl_mult": {"type": "float", "min": 0.5, "max": 10.0},
        "momentum_atr_tp_mult": {"type": "float", "min": 0.5, "max": 20.0},
        # Mean Reversion
        "mean_rev_bb_period": {"type": "int", "min": 5, "max": 100},
        "mean_rev_bb_std": {"type": "float", "min": 0.5, "max": 5.0},
        "mean_rev_zscore_entry": {"type": "float", "min": 0.5, "max": 5.0},
        "mean_rev_zscore_exit": {"type": "float", "min": 0.0, "max": 3.0},
        "mean_rev_rsi_period": {"type": "int", "min": 2, "max": 100},
        # ML
        "ml_min_confidence": {"type": "float", "min": 0.3, "max": 0.99},
        "ml_retrain_interval_hours": {"type": "int", "min": 1, "max": 168},
        # Risk
        "max_position_size_pct": {"type": "float", "min": 0.001, "max": 1.0},
        "max_daily_loss_pct": {"type": "float", "min": 0.005, "max": 1.0},
        "max_portfolio_exposure": {"type": "float", "min": 0.1, "max": 2.0},
        "max_single_stock_pct": {"type": "float", "min": 0.01, "max": 1.0},
        "max_leverage": {"type": "float", "min": 0.1, "max": 10.0},
        "max_open_positions": {"type": "int", "min": 1, "max": 200},
        "max_orders_per_day": {"type": "int", "min": 1, "max": 1000},
        "max_correlated_positions": {"type": "int", "min": 1, "max": 20},
        "default_stop_loss_pct": {"type": "float", "min": 0.005, "max": 0.5},
        "default_take_profit_pct": {"type": "float", "min": 0.005, "max": 1.0},
        # Automation
        "auto_backtest_interval_hours": {"type": "int", "min": 0, "max": 168},
        "auto_train_interval_hours": {"type": "int", "min": 0, "max": 168},
        "auto_sweep_interval_hours": {"type": "int", "min": 0, "max": 168},
        "auto_train_bars": {"type": "int", "min": 100, "max": 10000},
        # Notifications
        "notify_on_trade": {"type": "bool"},
        "notify_on_error": {"type": "bool"},
        "notify_on_signal": {"type": "bool"},
        # Backtest
        "backtest_initial_cash": {"type": "float", "min": 100, "max": 10000000},
        "backtest_commission_pct": {"type": "float", "min": 0.0, "max": 0.1},
        "backtest_slippage_pct": {"type": "float", "min": 0.0, "max": 0.1},
    }

    if param not in SETTABLE_PARAMS:
        # Check partial matches
        matches = [p for p in SETTABLE_PARAMS if param in p]
        if matches:
            await message.answer(
                f"Unknown param: <code>{param}</code>\n\nDid you mean:\n" +
                "\n".join(f"  <code>{m}</code>" for m in matches[:10]),
                parse_mode=ParseMode.HTML
            )
        else:
            await message.answer(f"Unknown param: <code>{param}</code>\nUse /config to see all params.", parse_mode=ParseMode.HTML)
        return

    spec = SETTABLE_PARAMS[param]
    field_name = spec.get("field", param)

    # Parse & validate value
    try:
        if spec["type"] == "int":
            value = int(raw_value)
            if value < spec["min"] or value > spec["max"]:
                await message.answer(f"Value must be between {spec['min']} and {spec['max']}.")
                return
        elif spec["type"] == "float":
            value = float(raw_value)
            if value < spec["min"] or value > spec["max"]:
                await message.answer(f"Value must be between {spec['min']} and {spec['max']}.")
                return
        elif spec["type"] == "bool":
            value = raw_value.lower() in ("true", "1", "yes", "on")
        elif spec["type"] == "choice":
            if raw_value not in spec["choices"]:
                await message.answer(f"Invalid value. Choose from: {', '.join(spec['choices'])}")
                return
            value = raw_value
        else:
            value = raw_value
    except ValueError:
        await message.answer(f"Invalid value for {param}. Expected type: {spec['type']}")
        return

    # Get old value
    old_value = getattr(settings, field_name, "?")

    # Apply to settings object directly (in-memory)
    try:
        setattr(settings, field_name, value)
    except Exception as e:
        await message.answer(f"Failed to set: {e}")
        return

    # Also apply to risk manager if it's a risk param
    risk_params = {
        "max_position_size_pct", "max_daily_loss_pct", "max_portfolio_exposure",
        "max_single_stock_pct", "max_leverage", "default_stop_loss_pct",
        "default_take_profit_pct", "max_open_positions", "max_orders_per_day",
        "max_correlated_positions",
    }
    if field_name in risk_params and _risk_manager:
        if hasattr(_risk_manager.limits, field_name):
            setattr(_risk_manager.limits, field_name, value)

    # Signal main loop for params that need strategy rebuild
    rebuild_params = {
        "timeframe", "lookback_bars", "momentum_fast_ema", "momentum_slow_ema",
        "momentum_rsi_period", "momentum_rsi_oversold", "momentum_rsi_overbought",
        "momentum_atr_period", "momentum_atr_sl_mult", "momentum_atr_tp_mult",
        "mean_rev_bb_period", "mean_rev_bb_std", "mean_rev_zscore_entry",
        "mean_rev_zscore_exit", "mean_rev_rsi_period", "ml_min_confidence",
    }
    if field_name in rebuild_params:
        with _runtime_lock:
            _runtime_changes["config_updates"][field_name] = value

    # Special handling for interval/timeframe/lookback
    if param == "interval":
        with _runtime_lock:
            _runtime_changes["interval"] = value
    elif param == "timeframe":
        with _runtime_lock:
            _runtime_changes["timeframe"] = value
    elif param == "lookback":
        with _runtime_lock:
            _runtime_changes["lookback"] = value

    await message.answer(
        f"✅ <b>Config updated</b>\n"
        f"  <code>{param}</code>: {old_value} → <b>{value}</b>\n"
        f"  {'(takes effect next cycle)' if field_name in rebuild_params else '(immediate)'}",
        parse_mode=ParseMode.HTML
    )


@router.message(Command("setauto"))
async def cmd_setauto(message: Message):
    """Configure automation scheduler.
    Usage: /setauto backtest|train|sweep HOURS
    Set to 0 to disable.
    """
    if not _is_authorized(message):
        return

    from config.settings import settings

    parts = message.text.split()
    if len(parts) < 3:
        await message.answer(
            "<b>Usage:</b> /setauto TYPE HOURS\n\n"
            "<b>Types:</b>\n"
            f"  backtest - currently every {settings.auto_backtest_interval_hours}h\n"
            f"  train    - currently every {settings.auto_train_interval_hours}h\n"
            f"  sweep    - currently every {settings.auto_sweep_interval_hours}h\n\n"
            "Set HOURS to 0 to disable.\n"
            "Example: /setauto train 12",
            parse_mode=ParseMode.HTML
        )
        return

    task_type = parts[1].lower()
    try:
        hours = int(parts[2])
    except ValueError:
        await message.answer("Hours must be an integer.")
        return

    if hours < 0 or hours > 168:
        await message.answer("Hours must be between 0 and 168 (1 week).")
        return

    field_map = {
        "backtest": "auto_backtest_interval_hours",
        "train": "auto_train_interval_hours",
        "sweep": "auto_sweep_interval_hours",
    }

    if task_type not in field_map:
        await message.answer(f"Unknown type: {task_type}. Use: backtest, train, sweep")
        return

    field = field_map[task_type]
    old = getattr(settings, field)
    setattr(settings, field, hours)

    status = f"every {hours}h" if hours > 0 else "DISABLED"
    await message.answer(
        f"✅ Auto-{task_type}: {old}h → <b>{status}</b>\n"
        f"<i>Note: Restart bot for scheduler changes to take full effect.</i>",
        parse_mode=ParseMode.HTML
    )


@router.message(Command("settf"))
async def cmd_settf(message: Message):
    """Change trading timeframe.
    Usage: /settf 15Min
    """
    if not _is_authorized(message):
        return

    valid_tfs = ["1Min", "5Min", "15Min", "30Min", "1Hour", "4Hour", "1Day"]
    parts = message.text.split()

    if len(parts) < 2:
        from config.settings import settings
        await message.answer(
            f"<b>Usage:</b> /settf TIMEFRAME\n\n"
            f"<b>Current:</b> {settings.timeframe}\n"
            f"<b>Valid:</b> {', '.join(valid_tfs)}",
            parse_mode=ParseMode.HTML
        )
        return

    new_tf = parts[1]
    if new_tf not in valid_tfs:
        await message.answer(f"Invalid timeframe. Choose from: {', '.join(valid_tfs)}")
        return

    from config.settings import settings
    old = settings.timeframe
    settings.timeframe = new_tf

    with _runtime_lock:
        _runtime_changes["timeframe"] = new_tf

    await message.answer(
        f"✅ Timeframe: {old} → <b>{new_tf}</b>\nStrategy will rebuild on next cycle.",
        parse_mode=ParseMode.HTML
    )


@router.message(Command("setlookback"))
async def cmd_setlookback(message: Message):
    """Change lookback bars.
    Usage: /setlookback 500
    """
    if not _is_authorized(message):
        return

    parts = message.text.split()
    if len(parts) < 2:
        from config.settings import settings
        await message.answer(
            f"<b>Usage:</b> /setlookback BARS\n"
            f"<b>Current:</b> {settings.lookback_bars}",
            parse_mode=ParseMode.HTML
        )
        return

    try:
        bars = int(parts[1])
    except ValueError:
        await message.answer("Must be an integer.")
        return

    if bars < 50 or bars > 5000:
        await message.answer("Lookback must be between 50 and 5000.")
        return

    from config.settings import settings
    old = settings.lookback_bars
    settings.lookback_bars = bars

    with _runtime_lock:
        _runtime_changes["lookback"] = bars

    await message.answer(
        f"✅ Lookback: {old} → <b>{bars}</b> bars\nStrategy will rebuild on next cycle.",
        parse_mode=ParseMode.HTML
    )


@router.message(Command("setnotify"))
async def cmd_setnotify(message: Message):
    """Toggle notification preferences.
    Usage: /setnotify trade|error|signal on|off
    """
    if not _is_authorized(message):
        return

    from config.settings import settings

    parts = message.text.split()
    if len(parts) < 3:
        await message.answer(
            f"<b>Usage:</b> /setnotify TYPE on|off\n\n"
            f"<b>Current:</b>\n"
            f"  trade:  {'ON ✅' if settings.notify_on_trade else 'OFF ❌'}\n"
            f"  error:  {'ON ✅' if settings.notify_on_error else 'OFF ❌'}\n"
            f"  signal: {'ON ✅' if settings.notify_on_signal else 'OFF ❌'}",
            parse_mode=ParseMode.HTML
        )
        return

    notify_type = parts[1].lower()
    state = parts[2].lower() in ("on", "true", "1", "yes")

    field_map = {
        "trade": "notify_on_trade",
        "error": "notify_on_error",
        "signal": "notify_on_signal",
    }

    if notify_type not in field_map:
        await message.answer(f"Unknown type. Use: trade, error, signal")
        return

    setattr(settings, field_map[notify_type], state)
    emoji = "✅ ON" if state else "❌ OFF"
    await message.answer(f"Notifications for <b>{notify_type}</b>: {emoji}", parse_mode=ParseMode.HTML)


@router.message(Command("backtest"))
async def cmd_backtest(message: Message):
    """Trigger a backtest from Telegram.
    Usage: /backtest [symbol] [days]
    """
    if not _is_authorized(message) or not _broker:
        return

    parts = message.text.split()
    symbol = parts[1].upper() if len(parts) > 1 else (_strategy.symbols[0] if _strategy else "AAPL")
    days = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 30

    await message.answer(f"Running backtest for {symbol} ({days} days)...")

    try:
        import pandas as pd
        from datetime import datetime, timedelta

        with _broker_lock:
            df = _broker.get_bars_df(symbol, "1Hour", limit=days * 7)

        if df is None or len(df) < 50:
            await message.answer(f"Insufficient data for {symbol}. Need at least 50 bars.")
            return

        try:
            from backtesting.bt_adapter import run_backtrader_backtest, _BT_AVAILABLE
            if not _BT_AVAILABLE:
                raise ImportError("backtrader not installed")

            import asyncio
            loop = asyncio.get_event_loop()
            metrics = await loop.run_in_executor(
                None, lambda: run_backtrader_backtest(df, initial_cash=10000.0)
            )
            text = (
                f"<b>Backtest Results: {symbol}</b>\n"
                f"{'=' * 30}\n"
                f"Strategy:     {metrics['strategy']}\n"
                f"Return:       {metrics['total_return']:.2%}\n"
                f"Trades:       {metrics['total_trades']}\n"
                f"Win Rate:     {metrics['win_rate']:.1%}\n"
                f"Sharpe:       {metrics['sharpe_ratio']:.2f}\n"
                f"Max DD:       {metrics['max_drawdown']:.2%}\n"
                f"Final Value:  ${metrics['final_value']:.2f}\n"
            )
            await message.answer(text, parse_mode=ParseMode.HTML)
        except ImportError:
            # Fallback to simple custom backtest using the active strategy
            from backtesting.backtest import Backtester

            if not _strategy:
                await message.answer("Backtrader not available and no strategy loaded for fallback.")
                return

            bt = Backtester.for_symbol(_strategy, symbol, initial_capital=10000.0)
            result = bt.run(df, symbol=symbol)
            text = (
                f"<b>Backtest Results: {symbol}</b>\n"
                f"Return: {result.total_return_pct:.2%}\n"
                f"Trades: {result.total_trades}\n"
                f"Win Rate: {result.win_rate:.1%}\n"
                f"Max DD: {result.max_drawdown:.2%}\n"
            )
            await message.answer(text, parse_mode=ParseMode.HTML)

    except Exception as e:
        await message.answer(f"Backtest error: {e}")


@router.message(Command("train"))
async def cmd_train(message: Message):
    """Trigger ML model retraining.
    Usage: /train [SYMBOL1,SYMBOL2] [BARS]
    Without arguments, trains on all configured symbols.
    """
    if not _is_authorized(message) or not _broker:
        return

    parts = message.text.split()
    symbols = None
    bars = 1000

    if len(parts) > 1:
        symbols = [s.strip().upper() for s in parts[1].split(",") if s.strip()]
    if len(parts) > 2:
        try:
            bars = int(parts[2])
        except ValueError:
            pass

    if not symbols:
        symbols = _strategy.symbols[:10] if _strategy else ["AAPL", "MSFT", "TSLA"]

    await message.answer(
        f"ML model training started...\n"
        f"Symbols: {', '.join(symbols)}\n"
        f"Bars: {bars}\n"
        f"This may take 1-3 minutes."
    )

    try:
        from src.strategy.ml_strategy import MLStrategy
        from config.settings import settings
        import asyncio

        strategy = MLStrategy(
            symbols=symbols,
            timeframe=_strategy.timeframe if _strategy else "1Hour",
            lookback=500,
            model_path=settings.ml_model_path,
            min_confidence=settings.ml_min_confidence,
        )

        training_data = {}
        for symbol in symbols:
            with _broker_lock:
                df = _broker.get_bars_df(symbol, _strategy.timeframe if _strategy else "1Hour", limit=bars)
            if df is not None and len(df) >= 200:
                training_data[symbol] = df

        if not training_data:
            await message.answer("Insufficient data for training. Need 200+ bars per symbol.")
            return

        # Run training in thread executor with timeout to prevent blocking indefinitely
        loop = asyncio.get_event_loop()
        _ML_TRAINING_TIMEOUT_SECONDS = 600  # 10 minutes max
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, strategy.train, training_data),
                timeout=_ML_TRAINING_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            await message.answer(
                f"⚠️ <b>Training timed out</b> after {_ML_TRAINING_TIMEOUT_SECONDS}s.\n"
                f"Consider reducing data size or using fewer symbols.",
                parse_mode=ParseMode.HTML,
            )
            return

        # Trigger strategy reload in main loop
        with _runtime_lock:
            _runtime_changes["trigger_train"] = True

        await message.answer(
            f"<b>ML Model Trained Successfully</b>\n"
            f"Symbols used: {len(training_data)}\n"
            f"Model saved. Strategy will reload on next cycle.",
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        await message.answer(f"Training failed: {e}")


@router.message(Command("backtestvbt"))
async def cmd_backtest_vbt(message: Message):
    """Run VectorBT vectorized backtest from Telegram.
    Usage: /backtestvbt [SYMBOL] [DAYS]
    """
    if not _is_authorized(message) or not _broker:
        return

    parts = message.text.split()
    symbol = parts[1].upper() if len(parts) > 1 else (_strategy.symbols[0] if _strategy else "AAPL")
    days = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 30

    await message.answer(f"Running VectorBT backtest for {symbol} ({days} days)...")

    try:
        with _broker_lock:
            df = _broker.get_bars_df(symbol, _strategy.timeframe if _strategy else "1Hour", limit=days * 7)

        if df is None or len(df) < 50:
            await message.answer(f"Insufficient data for {symbol}. Need at least 50 bars.")
            return

        try:
            from backtesting.vbt_adapter import vectorbt_momentum_backtest, vectorbt_parameter_sweep
            from src.config.backtest_params import get_backtest_config
            from config.settings import settings
            import asyncio

            # Resolve asset-class-aware parameters
            params = get_backtest_config(symbol)

            loop = asyncio.get_event_loop()
            metrics = await loop.run_in_executor(
                None, lambda: vectorbt_momentum_backtest(
                    df,
                    fast_ema=params["fast_ema"],
                    slow_ema=params["slow_ema"],
                    initial_cash=settings.backtest_initial_cash,
                    fees=params["fees"],
                    risk_per_trade=params["risk_per_trade"],
                    atr_stop_multiplier=params["atr_stop_multiplier"],
                    cooldown_bars=params["cooldown_bars"],
                    annualization_periods=params["annualization_periods"],
                )
            )
            text = (
                f"<b>VectorBT Results: {symbol}</b>\n"
                f"{'=' * 30}\n"
                f"EMA:          {params['fast_ema']}/{params['slow_ema']}\n"
                f"Fees:         {params['fees']*100:.2f}%\n"
                f"Risk/Trade:   {params['risk_per_trade']*100:.0f}%\n"
                f"Bars:         {len(df)}\n"
                f"{'─' * 30}\n"
                f"Return:       {metrics['total_return']:.2%}\n"
                f"Trades:       {metrics['total_trades']}\n"
                f"Win Rate:     {metrics['win_rate']:.1%}\n"
                f"Sharpe:       {metrics['sharpe_ratio']:.2f}\n"
                f"Max DD:       {metrics['max_drawdown']:.2%}\n"
            )

            # Quick parameter sweep (top 3) with asset-class params
            try:
                sweep_df = await loop.run_in_executor(
                    None, lambda: vectorbt_parameter_sweep(
                        df,
                        initial_cash=settings.backtest_initial_cash,
                        fees=params["fees"],
                        risk_per_trade=params["risk_per_trade"],
                        atr_stop_multiplier=params["atr_stop_multiplier"],
                        cooldown_bars=params["cooldown_bars"],
                        annualization_periods=params["annualization_periods"],
                    )
                )
                if sweep_df is not None and len(sweep_df) > 0:
                    top = sweep_df.sort_values('total_return', ascending=False).head(3)
                    text += f"\n<b>Top 3 Param Combos:</b>\n"
                    for _, row in top.iterrows():
                        text += f"  Fast={int(row['fast_window'])}, Slow={int(row['slow_window'])}: {row['total_return']:.2%}\n"
            except Exception:
                pass

            await message.answer(text, parse_mode=ParseMode.HTML)
        except ImportError:
            await message.answer(
                "VectorBT not installed.\n"
                "Install with: pip install vectorbt"
            )
    except Exception as e:
        await message.answer(f"VectorBT backtest error: {e}")


@router.message(Command("sweep"))
async def cmd_sweep(message: Message):
    """Run full parameter sweep via VectorBT.
    Usage: /sweep [SYMBOL] [DAYS]
    """
    if not _is_authorized(message) or not _broker:
        return

    parts = message.text.split()
    symbol = parts[1].upper() if len(parts) > 1 else (_strategy.symbols[0] if _strategy else "AAPL")
    days = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 60

    await message.answer(f"Running parameter sweep for {symbol} ({days} days)...\nThis may take a minute.")

    try:
        with _broker_lock:
            df = _broker.get_bars_df(symbol, _strategy.timeframe if _strategy else "1Hour", limit=days * 7)

        if df is None or len(df) < 100:
            await message.answer(f"Insufficient data for {symbol}.")
            return

        from backtesting.vbt_adapter import vectorbt_parameter_sweep
        from src.config.backtest_params import get_backtest_config
        import asyncio

        # Resolve asset-class-aware parameters
        params = get_backtest_config(symbol)

        loop = asyncio.get_event_loop()
        sweep_df = await loop.run_in_executor(
            None, lambda: vectorbt_parameter_sweep(
                df,
                fees=params["fees"],
                risk_per_trade=params["risk_per_trade"],
                atr_stop_multiplier=params["atr_stop_multiplier"],
                cooldown_bars=params["cooldown_bars"],
                annualization_periods=params["annualization_periods"],
            )
        )

        if sweep_df is None or len(sweep_df) == 0:
            await message.answer("Sweep produced no results.")
            return

        top = sweep_df.sort_values('total_return', ascending=False).head(10)
        lines = [f"<b>Parameter Sweep: {symbol}</b>\n", f"{'=' * 35}\n"]
        lines.append(f"{'Fast':<6}{'Slow':<6}{'Return':<10}{'Trades':<8}{'WinRate':<8}\n")
        for _, row in top.iterrows():
            lines.append(
                f"{int(row['fast_window']):<6}{int(row['slow_window']):<6}"
                f"{row['total_return']:.1%}{'':4}{int(row.get('total_trades', 0)):<8}"
                f"{row.get('win_rate', 0):.0%}\n"
            )

        await message.answer("".join(lines), parse_mode=ParseMode.HTML)
    except ImportError:
        await message.answer("VectorBT not installed. Install with: pip install vectorbt")
    except Exception as e:
        await message.answer(f"Sweep error: {e}")


@router.message(Command("modelinfo"))
async def cmd_modelinfo(message: Message):
    """Show ML model metadata.
    Displays: model type, training date, accuracy, features, symbols.
    """
    if not _is_authorized(message):
        return

    try:
        import os
        import joblib
        from config.settings import settings

        model_path = settings.ml_model_path
        if not os.path.exists(model_path):
            await message.answer("No trained model found.\nUse /train to train one.")
            return

        data = joblib.load(model_path)
        model = data.get('model')
        features = data.get('features', [])
        trained_at = data.get('trained_at')

        # Get model info
        model_type = type(model).__name__ if model else "Unknown"
        n_features = len(features)
        trained_str = trained_at.strftime('%Y-%m-%d %H:%M') if trained_at else "Unknown"

        # Try to get more details from the model
        n_estimators = getattr(model, 'n_estimators', '?')
        max_depth = getattr(model, 'max_depth', '?')

        # File size
        file_size = os.path.getsize(model_path)
        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024*1024):.1f} MB"

        # Current strategy status
        strategy_status = "Active" if (_strategy and _strategy.name == "ml_xgboost" and _strategy.model is not None) else "Not loaded"

        text = (
            f"<b>🧠 ML Model Info</b>\n"
            f"{'=' * 30}\n"
            f"Type:         {model_type}\n"
            f"Trained:      {trained_str}\n"
            f"Features:     {n_features}\n"
            f"Estimators:   {n_estimators}\n"
            f"Max Depth:    {max_depth}\n"
            f"File Size:    {size_str}\n"
            f"Status:       {strategy_status}\n"
            f"\n<b>Top Features:</b>\n"
        )

        # Show top features (first 10)
        for f in features[:10]:
            text += f"  • {f}\n"
        if len(features) > 10:
            text += f"  ... +{len(features) - 10} more"

        await message.answer(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer("Operation failed. Check system logs for details.")


@router.message(Command("predict"))
async def cmd_predict(message: Message):
    """Get ML prediction for a symbol.
    Usage: /predict [SYMBOL]
    Shows prediction direction, confidence, and key feature values.
    """
    if not _is_authorized(message) or not _broker:
        return

    parts = message.text.split()
    symbol = parts[1].upper() if len(parts) > 1 else (_strategy.symbols[0] if _strategy else "BTC/USD")

    # Check if ML strategy is available
    if not _strategy or _strategy.name != "ml_xgboost" or _strategy.model is None:
        # Try loading model directly
        try:
            import os
            from config.settings import settings
            from src.strategy.ml_strategy import MLStrategy

            if not os.path.exists(settings.ml_model_path):
                await message.answer(
                    "No ML model available.\n"
                    "Switch to ML strategy: /setstrategy ml\n"
                    "Or train a model: /train"
                )
                return

            ml_strat = MLStrategy(
                symbols=[symbol],
                timeframe=_strategy.timeframe if _strategy else "1Hour",
                lookback=200,
                model_path=settings.ml_model_path,
            )
        except Exception as e:
            await message.answer(f"Failed to load ML model: {e}")
            return
    else:
        ml_strat = _strategy

    try:
        with _broker_lock:
            df = _broker.get_bars_df(symbol, ml_strat.timeframe, limit=200)

        if df is None or len(df) < 100:
            await message.answer(f"Insufficient data for {symbol}.")
            return

        # Calculate features
        df_feat = ml_strat.calculate_indicators(df)
        latest = df_feat.iloc[-1:]

        feature_cols = ml_strat._feature_columns or ml_strat.get_feature_columns()
        available_cols = [c for c in feature_cols if c in df_feat.columns]
        features = latest[available_cols]

        if features.isna().any(axis=1).iloc[0]:
            await message.answer(f"Feature calculation incomplete for {symbol}. Need more data.")
            return

        # Predict
        import numpy as np
        import asyncio
        loop = asyncio.get_event_loop()
        proba = await loop.run_in_executor(None, ml_strat.model.predict_proba, features)
        proba = proba[0]
        pred_class = np.argmax(proba)
        confidence = proba[pred_class]

        direction_map = {0: "SELL 📉", 1: "HOLD ➡️", 2: "BUY 📈"}
        direction = direction_map[pred_class]

        price = float(latest['close'].iloc[0])

        # Key indicators for display
        key_features = {}
        for col in ['rsi_14', 'ret_1', 'vol_5', 'dist_sma_20', 'bb_pct', 'atr_pct']:
            if col in df_feat.columns and not np.isnan(latest[col].iloc[0]):
                key_features[col] = float(latest[col].iloc[0])

        # Determine signal strength
        if confidence >= 0.80:
            strength = "STRONG"
        elif confidence >= 0.65:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        # ATR for SL/TP
        atr = float(latest['atr_14'].iloc[0]) if 'atr_14' in df_feat.columns and not np.isnan(latest['atr_14'].iloc[0]) else price * 0.02

        text = (
            f"<b>🔮 ML Prediction: {symbol}</b>\n"
            f"{'=' * 30}\n"
            f"Direction:  <b>{direction}</b>\n"
            f"Confidence: {confidence:.0%} ({strength})\n"
            f"Price:      ${price:,.2f}\n"
            f"\n<b>Probabilities:</b>\n"
            f"  📉 Sell:  {proba[0]:.0%}\n"
            f"  ➡️ Hold:  {proba[1]:.0%}\n"
            f"  📈 Buy:   {proba[2]:.0%}\n"
        )

        if key_features:
            text += f"\n<b>Key Indicators:</b>\n"
            labels = {
                'rsi_14': 'RSI(14)',
                'ret_1': 'Return(1bar)',
                'vol_5': 'Volatility(5)',
                'dist_sma_20': 'Dist SMA20',
                'bb_pct': 'BB %B',
                'atr_pct': 'ATR %',
            }
            for col, val in key_features.items():
                label = labels.get(col, col)
                if 'pct' in col or 'ret' in col or 'dist' in col or 'vol' in col:
                    text += f"  {label}: {val:.4f}\n"
                else:
                    text += f"  {label}: {val:.2f}\n"

        if pred_class != 1:  # Not HOLD
            sl_dir = -1 if pred_class == 2 else 1
            sl = price + sl_dir * atr * 2
            tp = price - sl_dir * atr * 3
            text += (
                f"\n<b>Suggested Levels:</b>\n"
                f"  Stop Loss:    ${sl:,.2f}\n"
                f"  Take Profit:  ${tp:,.2f}\n"
            )
        else:
            text += "\n<i>No trade suggested at current confidence.</i>"

        await message.answer(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Prediction error: {e}")


# ──────────────────────────────────────────────────────────────────────────
# Advanced Research Commands
# ──────────────────────────────────────────────────────────────────────────

@router.message(Command("walkforward"))
async def cmd_walkforward(message: Message):
    """Walk-forward optimization on a symbol."""
    if not _is_authorized(message):
        return
    if not _broker:
        await message.answer("Broker not connected.")
        return

    args = message.text.split()[1:]
    # Parse: /walkforward [SYMBOL] [BARS]
    from config.settings import settings
    symbols = settings.symbols
    bars = 1000

    if args:
        if args[0].replace("/", "").isalpha() or "/" in args[0]:
            symbols = [args[0].upper()]
            if len(args) > 1:
                try:
                    bars = int(args[1])
                except ValueError:
                    pass
        else:
            try:
                bars = int(args[0])
            except ValueError:
                pass

    symbol = symbols[0]
    await message.answer(f"Running walk-forward optimization on {symbol} ({bars} bars)...\nThis may take a minute.")

    try:
        import asyncio
        loop = asyncio.get_event_loop()

        def _run_wf():
            from backtesting.walk_forward import walk_forward_ema_backtest
            from src.config.backtest_params import get_backtest_config
            params = get_backtest_config(symbol)
            with _broker_lock:
                df = _broker.get_bars_df(symbol, settings.timeframe, bars)
            return walk_forward_ema_backtest(
                df, train_window=min(500, bars // 3),
                test_window=min(100, bars // 6),
                step=min(100, bars // 6),
                initial_cash=settings.backtest_initial_cash,
                fees=params["fees"],
                cooldown_bars=params["cooldown_bars"],
                atr_stop_multiplier=params["atr_stop_multiplier"],
            )

        result = await loop.run_in_executor(None, _run_wf)

        text = (
            f"<b>Walk-Forward: {symbol}</b>\n"
            f"{'=' * 30}\n"
            f"Windows: {result['n_windows']}\n"
            f"OOS Return: {result['oos_return']:.2%}\n"
            f"OOS Sharpe: {result['oos_sharpe']:.2f}\n"
            f"OOS Max DD: {result['oos_max_drawdown']:.2%}\n"
            f"OOS Win Rate: {result['oos_win_rate']:.1%}\n"
            f"Param Stability: {result['param_stability']:.1%}\n"
            f"IS Return: {result['is_return']:.4f}\n"
            f"\n<b>Strategy:</b> {result.get('strategy', 'ema_crossover')}\n"
            f"<b>Param Ranges:</b>\n"
        )
        for k, v in result.get('param_ranges', {}).items():
            text += f"  {k}: {v}\n"

        if result['best_params_per_window']:
            last_best = result['best_params_per_window'][-1]
            text += f"\n<b>Last Window Best:</b> {last_best}"

        await message.answer(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Walk-forward error: {e}")


@router.message(Command("montecarlo"))
async def cmd_montecarlo(message: Message):
    """Monte Carlo simulation on a symbol."""
    if not _is_authorized(message):
        return
    if not _broker:
        await message.answer("Broker not connected.")
        return

    args = message.text.split()[1:]
    from config.settings import settings
    symbols = settings.symbols
    bars = 1000
    n_sims = 1000

    if args:
        if args[0].replace("/", "").isalpha() or "/" in args[0]:
            symbols = [args[0].upper()]
            if len(args) > 1:
                try:
                    bars = int(args[1])
                except ValueError:
                    pass
            if len(args) > 2:
                try:
                    n_sims = int(args[2])
                except ValueError:
                    pass
        else:
            try:
                bars = int(args[0])
            except ValueError:
                pass

    symbol = symbols[0]
    await message.answer(f"Running Monte Carlo ({n_sims} simulations) on {symbol} ({bars} bars)...")

    try:
        import asyncio
        loop = asyncio.get_event_loop()

        def _run_mc():
            from backtesting.monte_carlo import monte_carlo_from_backtest
            from src.config.backtest_params import get_backtest_config
            params = get_backtest_config(symbol)
            with _broker_lock:
                df = _broker.get_bars_df(symbol, settings.timeframe, bars)
            return monte_carlo_from_backtest(
                df,
                fast_ema=params["fast_ema"],
                slow_ema=params["slow_ema"],
                initial_cash=settings.backtest_initial_cash,
                fees=params["fees"],
                risk_per_trade=params["risk_per_trade"],
                atr_stop_multiplier=params["atr_stop_multiplier"],
                cooldown_bars=params["cooldown_bars"],
                n_simulations=n_sims,
            )

        result = await loop.run_in_executor(None, _run_mc)

        prob_profit = result['probability_of_profit'] * 100
        prob_ruin = result['probability_of_ruin'] * 100

        text = (
            f"<b>Monte Carlo: {symbol}</b>\n"
            f"{'=' * 30}\n"
            f"Simulations: {result['n_simulations']}\n"
            f"Trades Used: {result['n_trades']}\n\n"
            f"<b>Return Distribution:</b>\n"
            f"  P5 (worst):  {result['p5_return']:.2%}\n"
            f"  P25:         {result['p25_return']:.2%}\n"
            f"  Median:      {result['median_return']:.2%}\n"
            f"  P75:         {result['p75_return']:.2%}\n"
            f"  P95 (best):  {result['p95_return']:.2%}\n\n"
            f"<b>Risk Metrics:</b>\n"
            f"  Expected Return: {result['expected_return']:.2%}\n"
            f"  Std Dev: {result['return_std']:.2%}\n"
            f"  Median Max DD: {result['median_max_drawdown']:.2%}\n"
            f"  P(Profit): {prob_profit:.1f}%\n"
            f"  P(Ruin <50%): {prob_ruin:.1f}%\n"
        )
        await message.answer(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Monte Carlo error: {e}")


@router.message(Command("portfolio"))
async def cmd_portfolio_backtest(message: Message):
    """Portfolio-level backtest across all symbols."""
    if not _is_authorized(message):
        return
    if not _broker:
        await message.answer("Broker not connected.")
        return

    args = message.text.split()[1:]
    from config.settings import settings
    bars = 500

    if args:
        try:
            bars = int(args[0])
        except ValueError:
            pass

    symbols = settings.symbols
    await message.answer(f"Running portfolio backtest on {len(symbols)} symbols ({bars} bars)...\nSymbols: {', '.join(symbols)}")

    try:
        import asyncio
        loop = asyncio.get_event_loop()

        def _run_portfolio():
            from backtesting.portfolio_backtest import portfolio_backtest
            from src.config.backtest_params import get_backtest_config
            data = {}
            with _broker_lock:
                for sym in symbols:
                    try:
                        df = _broker.get_bars_df(sym, settings.timeframe, bars)
                        if df is not None and len(df) > 50:
                            data[sym] = df
                    except Exception:
                        continue
            if not data:
                return None
            # Use the maximum fee across symbols (conservative estimate)
            fees = max(get_backtest_config(sym)["fees"] for sym in data.keys())
            return portfolio_backtest(
                data, initial_cash=settings.backtest_initial_cash,
                fees=fees,
            )

        result = await loop.run_in_executor(None, _run_portfolio)

        if result is None:
            await message.answer("No data available for portfolio backtest.")
            return

        text = (
            f"<b>Portfolio Backtest</b>\n"
            f"{'=' * 30}\n"
            f"Symbols: {result.get('n_symbols', 0)}\n"
            f"Total Trades: {result.get('total_trades', 0)}\n"
            f"Final Equity: ${result.get('final_equity', 0):,.2f}\n"
            f"Return: {result.get('total_return', 0):.2%}\n"
            f"Sharpe: {result.get('sharpe_ratio', 0):.2f}\n"
            f"Max DD: {result.get('max_drawdown', 0):.2%}\n"
            f"Win Rate: {result.get('win_rate', 0):.1%}\n"
        )

        # Per-symbol breakdown
        per_symbol = result.get('per_symbol', {})
        if per_symbol:
            text += f"\n<b>Per Symbol:</b>\n"
            for sym, stats in per_symbol.items():
                pnl = stats.get('pnl', 0)
                trades = stats.get('trades', 0)
                emoji = "+" if pnl >= 0 else ""
                text += f"  {sym}: {emoji}${pnl:.2f} ({trades} trades)\n"

        await message.answer(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Portfolio backtest error: {e}")


@router.message(Command("models"))
async def cmd_models(message: Message):
    """List all model versions."""
    if not _is_authorized(message):
        return

    try:
        from src.ml.model_registry import ModelRegistry
        registry = ModelRegistry()
        versions = registry.list_versions()

        if not versions:
            await message.answer("No model versions found. Train a model first with /train")
            return

        text = "<b>Model Registry</b>\n" + "=" * 30 + "\n\n"
        for v in versions[:10]:  # Show last 10
            active = " [ACTIVE]" if v.get("is_active") else ""
            metrics = v.get("metrics", {})
            acc = metrics.get("accuracy", 0)
            text += (
                f"<b>{v['version']}</b>{active}\n"
                f"  Trained: {v.get('trained_at', '?')[:16]}\n"
                f"  Accuracy: {acc:.2%}\n"
                f"  Symbols: {', '.join(v.get('symbols', []))}\n"
                f"  Features: {v.get('n_features', '?')}\n"
                f"  Samples: {v.get('n_samples', '?')}\n"
            )
            if v.get("note"):
                text += f"  Note: {v['note']}\n"
            text += "\n"

        text += f"<i>Total versions: {len(versions)}</i>"
        await message.answer(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Registry error: {e}")


@router.message(Command("rollback"))
async def cmd_rollback(message: Message):
    """Rollback to a previous model version."""
    if not _is_authorized(message):
        return

    args = message.text.split()[1:]
    if not args:
        await message.answer("Usage: /rollback v001\nUse /models to see available versions.")
        return

    version = args[0].lower()
    if not version.startswith("v"):
        version = f"v{version}"

    try:
        from src.ml.model_registry import ModelRegistry
        registry = ModelRegistry()
        success = registry.rollback(version)

        if success:
            # Trigger strategy reload
            with _runtime_lock:
                _runtime_changes["config_updates"]["reload_model"] = True
            await message.answer(
                f"Rolled back to <b>{version}</b>\n"
                f"Model will be loaded on next trading cycle.",
                parse_mode=ParseMode.HTML,
            )
        else:
            await message.answer(f"Rollback failed. Version '{version}' not found.\nUse /models to list versions.")
    except Exception as e:
        await message.answer(f"Rollback error: {e}")


@router.message(Command("journal"))
async def cmd_journal(message: Message):
    """Show recent trade journal entries."""
    if not _is_authorized(message):
        return

    args = message.text.split()[1:]
    limit = 10
    symbol = None

    if args:
        if args[0].isdigit():
            limit = int(args[0])
        else:
            symbol = args[0].upper()
            if len(args) > 1 and args[1].isdigit():
                limit = int(args[1])

    try:
        from src.data.journal import TradeJournal
        from src.data.store import DatabaseManager
        db = _db or DatabaseManager()
        journal = TradeJournal(db)
        entries = journal.get_journal(symbol=symbol, limit=limit)

        if not entries:
            await message.answer("No journal entries found." + (f" (filter: {symbol})" if symbol else ""))
            return

        text = f"<b>Trade Journal</b>"
        if symbol:
            text += f" — {symbol}"
        text += f"\n{'=' * 30}\n\n"

        for e in entries:
            pnl = e.get("pnl")
            pnl_str = f"${pnl:.2f}" if pnl is not None else "open"
            emoji = "+" if pnl and pnl > 0 else ("" if pnl and pnl < 0 else "")
            conf = e.get("confidence")
            conf_str = f"{conf:.0%}" if conf else "?"

            text += (
                f"<b>{e.get('side', '?').upper()} {e.get('symbol', '?')}</b> "
                f"| PnL: {emoji}{pnl_str} | Conf: {conf_str}\n"
                f"  Strategy: {e.get('strategy_name', '?')} "
                f"| Model: {e.get('model_version', '?')}\n"
                f"  Entry: ${e.get('entry_price', 0):.2f}"
            )
            if e.get("exit_price"):
                text += f" → ${e['exit_price']:.2f}"
            if e.get("exit_reason"):
                text += f" ({e['exit_reason']})"
            text += f"\n  Time: {str(e.get('entry_time', ''))[:16]}\n\n"

        await message.answer(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Journal error: {e}")


@router.message(Command("journalstats"))
async def cmd_journalstats(message: Message):
    """Trade journal performance analytics."""
    if not _is_authorized(message):
        return

    try:
        from src.data.journal import TradeJournal
        from src.data.store import DatabaseManager
        db = _db or DatabaseManager()
        journal = TradeJournal(db)

        # Summary stats
        summary = journal.get_summary(days=30)

        text = (
            f"<b>Journal Analytics (30 days)</b>\n"
            f"{'=' * 30}\n\n"
            f"Total Trades: {summary.get('total_trades', 0)}\n"
            f"Win Rate: {summary.get('win_rate', 0):.1%}\n"
            f"Avg PnL: ${summary.get('avg_pnl', 0):.2f}\n"
            f"Total PnL: ${summary.get('total_pnl', 0):.2f}\n"
            f"Best Trade: ${summary.get('best_trade', 0):.2f}\n"
            f"Worst Trade: ${summary.get('worst_trade', 0):.2f}\n"
            f"Avg Holding: {summary.get('avg_holding_bars', 0):.0f} bars\n"
        )

        # Confidence analysis
        conf_stats = journal.get_performance_by_confidence()
        if conf_stats:
            text += f"\n<b>By Confidence:</b>\n"
            for bucket, stats in conf_stats.items():
                text += (
                    f"  {bucket}: WR {stats['win_rate']:.0%} "
                    f"({stats['trades']} trades, avg ${stats['avg_pnl']:.2f})\n"
                )

        # Model version analysis
        model_stats = journal.get_performance_by_model_version()
        if model_stats:
            text += f"\n<b>By Model Version:</b>\n"
            for ver, stats in model_stats.items():
                text += (
                    f"  {ver}: WR {stats['win_rate']:.0%} "
                    f"({stats['trades']} trades, PnL ${stats['total_pnl']:.2f})\n"
                )

        await message.answer(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Journal stats error: {e}")


# ──────────────────────────────────────────────────────────────────────────
# Health & Metrics Commands
# ──────────────────────────────────────────────────────────────────────────

@router.message(Command("health"))
async def cmd_health(message: Message):
    """Show full system health status."""
    if not _is_authorized(message):
        return

    if not _health_monitor:
        await message.answer("Health monitor not configured.")
        return

    try:
        report = _health_monitor.check_all(
            broker=_broker,
            db=_db,
            strategy=_strategy,
            scheduler=_scheduler,
        )

        def _status_icon(component: dict) -> str:
            s = component.get("status", "unknown")
            if s == "ok":
                return "✅"
            elif s == "warning":
                return "⚠️"
            elif s == "error":
                return "❌"
            elif s == "not_configured":
                return "⬜"
            return "❓"

        # System
        sys_info = report["system"]
        mem = sys_info.get('memory_mb')
        mem_str = f"{mem:.0f}" if isinstance(mem, (int, float)) else "?"
        sys_line = f"{_status_icon(sys_info)} System: CPU {sys_info.get('cpu_percent', '?')}%, RAM {mem_str}MB"

        # Broker
        br = report["broker"]
        if br.get("status") == "not_configured":
            br_line = "⬜ Broker: Not configured"
        else:
            latency = br.get("latency_ms", "?")
            br_line = f"{_status_icon(br)} Broker: {'Connected' if br.get('connected') else 'Disconnected'} ({latency}ms)"

        # Database
        db_info = report["database"]
        if db_info.get("status") == "not_configured":
            db_line = "⬜ Database: Not configured"
        else:
            db_line = f"{_status_icon(db_info)} Database: OK ({db_info.get('size_mb', '?')}MB, {db_info.get('query_latency_ms', '?')}ms)"

        # ML
        ml = report["ml"]
        if ml.get("status") == "not_configured":
            ml_line = "⬜ ML Model: Not configured"
        else:
            loaded = "Loaded" if ml.get("model_loaded") else "Not loaded"
            age = ml.get("model_age_hours")
            age_str = f" ({age}h old)" if age else ""
            ml_line = f"{_status_icon(ml)} ML Model: {loaded}{age_str}"

        # Telegram
        tg = report["telegram"]
        if tg.get("status") == "not_configured":
            tg_line = "⬜ Telegram: Not configured"
        else:
            polling = "Polling" if tg.get("polling_active") else "Idle"
            tg_line = f"{_status_icon(tg)} Telegram: {polling}"

        # Scheduler
        sched = report["scheduler"]
        if sched.get("status") == "not_configured":
            sched_line = "⬜ Scheduler: Not configured"
        else:
            n_jobs = len(sched.get("next_jobs", []))
            running = "active" if sched.get("running") else "stopped"
            sched_line = f"{_status_icon(sched)} Scheduler: {n_jobs} jobs {running}"

        text = (
            f"<b>System Health</b>\n"
            f"{'═' * 23}\n"
            f"{sys_line}\n"
            f"{br_line}\n"
            f"{db_line}\n"
            f"{ml_line}\n"
            f"{tg_line}\n"
            f"{sched_line}\n"
        )

        await message.answer(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Health check error: {e}")


@router.message(Command("metrics"))
async def cmd_metrics(message: Message):
    """Show live trading performance metrics."""
    if not _is_authorized(message):
        return

    if not _live_metrics:
        await message.answer("Live metrics not configured.")
        return

    try:
        args = message.text.split()[1:]
        period = 30
        if args and args[0].isdigit():
            period = int(args[0])

        text = _live_metrics.get_summary_text(period_days=period)
        await message.answer(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Metrics error: {e}")


@router.message(Command("uptime"))
async def cmd_uptime(message: Message):
    """Show bot uptime and start time."""
    if not _is_authorized(message):
        return

    try:
        if _health_monitor:
            uptime_secs = _health_monitor.uptime_seconds
        else:
            uptime_secs = 0

        # Format uptime
        days = int(uptime_secs // 86400)
        hours = int((uptime_secs % 86400) // 3600)
        minutes = int((uptime_secs % 3600) // 60)

        if days > 0:
            uptime_str = f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            uptime_str = f"{hours}h {minutes}m"
        else:
            uptime_str = f"{minutes}m"

        start_time = _health_monitor._start_time.strftime("%Y-%m-%d %H:%M UTC") if _health_monitor else "unknown"
        paused_str = "PAUSED" if _runtime_state and _runtime_state.is_paused() else "ACTIVE"

        text = (
            f"<b>Bot Uptime</b>\n"
            f"{'═' * 23}\n"
            f"Status:  {paused_str}\n"
            f"Uptime:  {uptime_str}\n"
            f"Started: {start_time}\n"
        )

        await message.answer(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Uptime error: {e}")


@router.message(Command("events"))
async def cmd_events(message: Message):
    """Show recent event bus activity."""
    if not _is_authorized(message):
        return

    if not _event_bus:
        await message.answer("Event bus not configured.")
        return

    try:
        history = _event_bus.get_history(limit=15)
        if not history:
            await message.answer("No events recorded yet.")
            return

        lines = ["<b>📡 Recent Events</b>", f"{'═' * 25}"]
        for ev in reversed(history[-15:]):
            ts = ev.timestamp.strftime("%H:%M:%S") if hasattr(ev, 'timestamp') else "?"
            etype = type(ev).__name__
            symbol = getattr(ev, 'symbol', '')
            extra = f" {symbol}" if symbol else ""
            lines.append(f"<code>{ts}</code> {etype}{extra}")

        lines.append(f"\nTotal events: {len(_event_bus.get_history(limit=9999))}")
        lines.append(f"Subscribers: {_event_bus.subscriber_count}")
        await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Events error: {e}")


@router.message(Command("activetrades"))
async def cmd_active_trades(message: Message):
    """Show active trade lifecycles from TradeManager via /activetrades."""
    if not _is_authorized(message):
        return

    if not _trade_manager:
        await message.answer("Trade manager not configured.")
        return

    try:
        active = _trade_manager.get_active_trades()
        total = _trade_manager.count

        if not active:
            await message.answer(f"No active trades. Total tracked: {total}")
            return

        lines = ["<b>🔄 Active Trades</b>", f"{'═' * 25}"]
        for t in active[:20]:
            dur = t.duration
            dur_str = f"{dur / 60:.0f}m" if dur and dur < 3600 else f"{dur / 3600:.1f}h" if dur else "?"
            lines.append(
                f"<code>{t.trade_id}</code> {t.symbol} {t.side}\n"
                f"   State: {t.state.value} | Age: {dur_str}"
            )

        lines.append(f"\nActive: {len(active)} | Total tracked: {total}")
        await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Trades error: {e}")


# ──────────────────────────────────────────────────────────────────────────
# Operational Commands (health, risk, reconcile, latency, performance, system)
# ──────────────────────────────────────────────────────────────────────────

def _get_ops_handler():
    """Lazy-create OpsCommandHandler with current components."""
    from src.monitoring.ops_commands import OpsCommandHandler
    return OpsCommandHandler(
        health_monitor=_health_monitor,
        event_bus=_event_bus,
        trade_manager=_trade_manager,
        broker=_broker,
        reconciler=None,  # No reconciler wired yet
        risk_manager=_risk_manager,
        metrics=_live_metrics,
    )


@router.message(Command("healthops"))
async def cmd_healthops(message: Message):
    """Show ops-formatted health status, uptime, memory, CPU."""
    if not _is_authorized(message):
        return
    try:
        ops = _get_ops_handler()
        text = ops.format_health()
        await message.answer(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Health error: {e}")


@router.message(Command("riskreport"))
async def cmd_riskreport(message: Message):
    """Show current risk metrics: exposure, drawdown, daily PnL."""
    if not _is_authorized(message):
        return
    try:
        ops = _get_ops_handler()
        text = ops.format_risk_report()
        await message.answer(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Risk report error: {e}")


@router.message(Command("reconcile"))
async def cmd_reconcile(message: Message):
    """Trigger portfolio reconciliation and show results."""
    if not _is_authorized(message):
        return
    try:
        ops = _get_ops_handler()
        text = ops.format_reconciliation()
        await message.answer(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Reconcile error: {e}")


@router.message(Command("latency"))
async def cmd_latency(message: Message):
    """Show broker/system latency metrics."""
    if not _is_authorized(message):
        return
    try:
        ops = _get_ops_handler()
        text = ops.format_latency()
        await message.answer(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Latency error: {e}")


@router.message(Command("performance"))
async def cmd_performance(message: Message):
    """Show live trading performance: Sharpe, Sortino, win rate, PnL."""
    if not _is_authorized(message):
        return
    try:
        ops = _get_ops_handler()
        text = ops.format_performance()
        await message.answer(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Performance error: {e}")


@router.message(Command("system"))
async def cmd_system(message: Message):
    """Show system info: Python version, uptime, disk, memory, active threads."""
    if not _is_authorized(message):
        return
    try:
        ops = _get_ops_handler()
        text = ops.format_system()
        await message.answer(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"System error: {e}")


@router.message(Command("replay"))
async def cmd_replay(message: Message):
    """Replay a session: /replay [session_id] or /replay list."""
    if not _is_authorized(message):
        return
    try:
        from src.core.event_store import EventStore
        from src.core.replay import ReplayEngine

        args = message.text.split(maxsplit=1)
        store = EventStore(db_path="data_cache/events.db")
        engine = ReplayEngine(store)

        if len(args) < 2 or args[1].strip().lower() == "list":
            sessions = engine.list_sessions(limit=10)
            if not sessions:
                await message.answer("No sessions found in event store.")
                return
            lines = ["<b>📼 Available Sessions</b>\n"]
            for s in sessions:
                lines.append(
                    f"• <code>{s['session_id'][:12]}</code> — "
                    f"{s['event_count']} events"
                )
            await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
        else:
            session_id = args[1].strip()
            summary = engine.get_session_summary(session_id)
            if not summary.get("found"):
                await message.answer(f"Session not found: {session_id}")
                return
            lines = [
                f"<b>📼 Session: {session_id[:12]}...</b>\n",
                f"Events: {summary['total_events']}",
                f"First: {summary.get('first_event', 'N/A')}",
                f"Last: {summary.get('last_event', 'N/A')}",
                "\n<b>Event Types:</b>",
            ]
            for etype, count in sorted(summary.get("event_types", {}).items()):
                lines.append(f"  {etype}: {count}")
            await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Replay error: {e}")


@router.message(Command("recover"))
async def cmd_recover(message: Message):
    """Trigger crash recovery sequence."""
    if not _is_authorized(message):
        return
    try:
        from src.core.recovery import RecoveryManager
        from src.core.event_store import EventStore

        if not _broker or not _event_bus or not _trade_manager:
            await message.answer("Components not initialized.")
            return

        store = EventStore(db_path="data_cache/events.db")
        rm = RecoveryManager(_broker, _event_bus, _trade_manager, store)
        report = rm.recover()

        lines = [
            "<b>🔄 Recovery Report</b>\n",
            f"Success: {'✅' if report.success else '❌'}",
            f"Positions recovered: {report.positions_recovered}",
            f"Orphans detected: {report.orphans_detected}",
            f"Events replayed: {report.events_replayed}",
        ]
        if report.warnings:
            lines.append(f"\n⚠️ Warnings: {len(report.warnings)}")
            for w in report.warnings[:5]:
                lines.append(f"  • {w}")
        await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Recovery error: {e}")


@router.message(Command("governance"))
async def cmd_governance(message: Message):
    """Show model governance audit report."""
    if not _is_authorized(message):
        return
    try:
        from src.ml.governance import ModelGovernance

        gov = ModelGovernance()
        report = gov.audit_report()

        lines = [
            "<b>🏛️ Model Governance</b>\n",
            f"Total versions: {report['total_versions']}",
            f"Currently deployed: {report['currently_deployed']}",
            f"Retired: {report['retired']}",
            f"With git commit: {report['with_git_commit']}",
            f"With dataset hash: {report['with_dataset_hash']}",
            f"Completeness: {report['governance_completeness']:.0%}",
        ]
        if report["versions"]:
            lines.append("\n<b>Versions:</b>")
            for v in report["versions"][:5]:
                status = "🟢" if v["deployed"] else "⚪"
                lines.append(
                    f"  {status} {v['version']} — "
                    f"acc={v['cv_accuracy']:.3f} "
                    f"({v['git_commit'] or 'no-git'})"
                )
        await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Governance error: {e}")


@router.message(Command("modelaudit"))
async def cmd_modelaudit(message: Message):
    """Validate current model for deployment readiness."""
    if not _is_authorized(message):
        return
    try:
        from src.ml.governance import ModelGovernance
        from src.ml.model_registry import ModelRegistry

        registry = ModelRegistry()
        gov = ModelGovernance()

        active = registry.get_active_version()
        if not active:
            await message.answer("No active model version found.")
            return

        is_valid, issues = gov.validate_for_deployment(active)

        lines = [
            f"<b>🔍 Model Audit: {active}</b>\n",
            f"Status: {'✅ PASS' if is_valid else '❌ FAIL'}",
        ]
        if issues:
            lines.append(f"\nIssues ({len(issues)}):")
            for issue in issues:
                lines.append(f"  ⚠️ {issue}")
        else:
            lines.append("\nAll governance checks passed.")

        lineage = gov.get_lineage(active)
        if lineage:
            lines.extend([
                f"\nGit: {lineage.git_commit[:8] if lineage.git_commit else 'N/A'}",
                f"Features: {lineage.n_features}",
                f"Dataset rows: {lineage.training_dataset_rows}",
                f"CV accuracy: {lineage.cv_accuracy:.3f}",
                f"WF Sharpe: {lineage.walk_forward_sharpe:.3f}",
                f"MC prob profit: {lineage.monte_carlo_prob_profit:.1%}",
            ])
        await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Model audit error: {e}")


# ──────────────────────────────────────────────────────────────────────────
# Asset Discovery & Market Data
# ──────────────────────────────────────────────────────────────────────────


@router.message(Command("assets"))
async def cmd_assets(message: Message):
    """List tradeable assets from Alpaca.
    Usage: /assets [stocks|crypto] [exchange] [page]
    Examples:
        /assets              - First 30 stocks
        /assets crypto       - All crypto pairs
        /assets stocks NYSE  - NYSE stocks only
        /assets stocks 2     - Page 2 of stocks
    """
    if not _is_authorized(message):
        return
    if not _broker:
        await message.answer("Broker not connected.")
        return

    parts = message.text.split()
    asset_type = "stocks"
    exchange = None
    page = 1

    if len(parts) > 1:
        arg1 = parts[1].lower()
        if arg1 in ("crypto", "c"):
            asset_type = "crypto"
        elif arg1 in ("stocks", "stock", "s"):
            asset_type = "stocks"
        else:
            # Could be exchange or page number
            if arg1.isdigit():
                page = int(arg1)
            else:
                exchange = arg1.upper()

    if len(parts) > 2:
        arg2 = parts[2]
        if arg2.isdigit():
            page = int(arg2)
        elif asset_type == "stocks":
            exchange = arg2.upper()

    if len(parts) > 3 and parts[3].isdigit():
        page = int(parts[3])

    page_size = 30
    try:
        with _broker_lock:
            if asset_type == "crypto":
                assets = _broker.get_crypto_assets()
            else:
                assets = _broker.get_stock_assets(exchange=exchange)

        total = len(assets)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))
        start = (page - 1) * page_size
        end = start + page_size
        page_assets = assets[start:end]

        lines = [
            f"<b>{'🪙 Crypto' if asset_type == 'crypto' else '📈 Stocks'}"
            f"{f' ({exchange})' if exchange else ''}</b>",
            f"Total: {total} | Page {page}/{total_pages}\n",
        ]

        if asset_type == "crypto":
            for a in page_assets:
                frac = "🔹" if a.get("fractionable") else "  "
                lines.append(f"<code>{a['symbol']:12s}</code> {frac} {(a.get('name') or '')[:30]}")
        else:
            for a in page_assets:
                frac = "🔹" if a.get("fractionable") else "  "
                short = "📉" if a.get("shortable") else "  "
                lines.append(
                    f"<code>{a['symbol']:8s}</code> {a.get('exchange', ''):5s} {frac}{short} "
                    f"{(a.get('name') or '')[:28]}"
                )

        lines.append(f"\n🔹=fractionable {'📉=shortable' if asset_type == 'stocks' else ''}")
        if total_pages > 1:
            lines.append(f"Use: /assets {asset_type}{f' {exchange}' if exchange else ''} {page+1}")

        await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer("Operation failed. Check system logs for details.")


@router.message(Command("search"))
async def cmd_search(message: Message):
    """Search for assets by name or symbol.
    Usage: /search QUERY [stocks|crypto]
    Examples:
        /search apple
        /search BTC crypto
        /search nvidia
        /search ETH
    """
    if not _is_authorized(message):
        return
    if not _broker:
        await message.answer("Broker not connected.")
        return

    parts = message.text.split(maxsplit=2)
    if len(parts) < 2:
        await message.answer(
            "<b>Usage:</b> /search QUERY [stocks|crypto]\n\n"
            "<b>Examples:</b>\n"
            "  /search apple\n"
            "  /search BTC crypto\n"
            "  /search nvidia\n"
            "  /search tesla",
            parse_mode=ParseMode.HTML,
        )
        return

    query = parts[1].upper()
    search_crypto = False
    search_stocks = True

    if len(parts) > 2:
        if parts[2].lower() in ("crypto", "c"):
            search_crypto = True
            search_stocks = False
        elif parts[2].lower() in ("both", "all"):
            search_crypto = True
            search_stocks = True

    results = []
    try:
        with _broker_lock:
            if search_stocks:
                stocks = _broker.get_stock_assets()
                for a in stocks:
                    if query in a["symbol"] or query in (a.get("name") or "").upper():
                        results.append({**a, "_type": "stock"})
            if search_crypto:
                crypto = _broker.get_crypto_assets()
                for a in crypto:
                    if query in a["symbol"] or query in (a.get("name") or "").upper():
                        results.append({**a, "_type": "crypto"})

        if not results:
            await message.answer(f"No assets found matching '<code>{query}</code>'.", parse_mode=ParseMode.HTML)
            return

        # Prioritize exact symbol matches
        results.sort(key=lambda x: (0 if x["symbol"] == query else 1, x["symbol"]))
        results = results[:25]

        lines = [f"<b>🔍 Search: '{query}'</b> — {len(results)} result(s)\n"]
        for a in results:
            type_icon = "🪙" if a["_type"] == "crypto" else "📈"
            frac = "🔹" if a.get("fractionable") else ""
            lines.append(
                f"{type_icon} <code>{a['symbol']:12s}</code> {frac} "
                f"{(a.get('name') or '')[:32]}"
            )

        lines.append("\nUse /price SYMBOL for live quote")
        lines.append("Use /asset SYMBOL for full details")
        await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Search error: {e}")


@router.message(Command("price"))
async def cmd_price(message: Message):
    """Get current price/quote for one or more symbols.
    Usage: /price SYMBOL [SYMBOL2 ...]
    Examples:
        /price AAPL
        /price BTC/USD ETH/USD
        /price AAPL MSFT NVDA
    """
    if not _is_authorized(message):
        return
    if not _broker:
        await message.answer("Broker not connected.")
        return

    parts = message.text.split()
    if len(parts) < 2:
        await message.answer(
            "<b>Usage:</b> /price SYMBOL [SYMBOL2 ...]\n\n"
            "<b>Examples:</b>\n"
            "  /price AAPL\n"
            "  /price BTC/USD ETH/USD\n"
            "  /price AAPL MSFT NVDA TSLA",
            parse_mode=ParseMode.HTML,
        )
        return

    symbols = [s.upper() for s in parts[1:6]]  # Max 5 at once

    lines = [f"<b>💰 Live Prices</b>\n"]
    for symbol in symbols:
        try:
            with _broker_lock:
                price = _broker.get_latest_price(symbol)
            is_crypto = "/" in symbol
            icon = "🪙" if is_crypto else "📈"
            lines.append(f"{icon} <code>{symbol:12s}</code> ${price:>12,.4f}" if is_crypto
                        else f"{icon} <code>{symbol:8s}</code> ${price:>10,.2f}")
        except Exception as e:
            lines.append(f"⚠️ <code>{symbol:8s}</code> Error: {str(e)[:40]}")

    lines.append(f"\n<i>Updated: {datetime.now().strftime('%H:%M:%S')}</i>")
    await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)


@router.message(Command("asset"))
async def cmd_asset_detail(message: Message):
    """Get detailed info about a specific asset.
    Usage: /asset SYMBOL
    Examples:
        /asset AAPL
        /asset BTC/USD
    """
    if not _is_authorized(message):
        return
    if not _broker:
        await message.answer("Broker not connected.")
        return

    parts = message.text.split()
    if len(parts) < 2:
        await message.answer(
            "<b>Usage:</b> /asset SYMBOL\n\n"
            "Shows full asset details including exchange, tradability, "
            "fractionable status, and current price.",
            parse_mode=ParseMode.HTML,
        )
        return

    symbol = parts[1].upper()
    is_crypto = "/" in symbol

    try:
        # Find the asset
        asset_info = None
        with _broker_lock:
            if is_crypto:
                assets = _broker.get_crypto_assets()
            else:
                assets = _broker.get_stock_assets()

        for a in assets:
            if a["symbol"] == symbol:
                asset_info = a
                break

        if not asset_info:
            await message.answer(
                f"Asset '<code>{symbol}</code>' not found.\n"
                f"Try /search {symbol.split('/')[0] if '/' in symbol else symbol}",
                parse_mode=ParseMode.HTML,
            )
            return

        # Get current price
        price_str = "N/A (market closed)"
        try:
            with _broker_lock:
                price = _broker.get_latest_price(symbol)
            price_str = f"${price:,.4f}" if is_crypto else f"${price:,.2f}"
        except Exception:
            pass

        icon = "🪙" if is_crypto else "📈"
        lines = [
            f"<b>{icon} {asset_info['symbol']}</b>",
            f"<i>{asset_info.get('name') or 'N/A'}</i>\n",
            f"Type: {'Crypto' if is_crypto else 'US Equity'}",
            f"Exchange: {asset_info.get('exchange') or 'N/A'}",
            f"Tradable: {'✅' if asset_info.get('tradable') else '❌'}",
            f"Fractionable: {'✅' if asset_info.get('fractionable') else '❌'}",
        ]

        if is_crypto:
            if asset_info.get("min_order_size"):
                lines.append(f"Min order: {asset_info['min_order_size']}")
            if asset_info.get("min_trade_increment"):
                lines.append(f"Min increment: {asset_info['min_trade_increment']}")
        else:
            lines.append(f"Shortable: {'✅' if asset_info.get('shortable') else '❌'}")

        lines.extend([
            f"\n<b>Price:</b> {price_str}",
            f"\n<b>Quick Actions:</b>",
            f"/price {symbol} — refresh price",
            f"/buy {symbol} 1 — buy 1 share/unit",
            f"/backtest {symbol} — run backtest",
            f"/predict {symbol} — ML prediction",
        ])

        await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer("Operation failed. Check system logs for details.")


@router.message(Command("watchlist"))
async def cmd_watchlist(message: Message):
    """Show current watchlist with live prices.
    Usage: /watchlist
    Shows all configured trading symbols with their current prices.
    """
    if not _is_authorized(message):
        return
    if not _broker:
        await message.answer("Broker not connected.")
        return

    from config.settings import settings

    symbols = [s.strip() for s in settings.trading_symbols.split(",") if s.strip()]
    if not symbols:
        await message.answer("No symbols configured. Use /setsymbols to add some.")
        return

    lines = [f"<b>👁 Watchlist ({len(symbols)} symbols)</b>\n"]
    for symbol in symbols:
        try:
            with _broker_lock:
                price = _broker.get_latest_price(symbol)
            is_crypto = "/" in symbol
            icon = "🪙" if is_crypto else "📈"
            lines.append(
                f"{icon} <code>{symbol:12s}</code> ${price:>12,.4f}" if is_crypto
                else f"{icon} <code>{symbol:8s}</code> ${price:>10,.2f}"
            )
        except Exception:
            lines.append(f"⚠️ <code>{symbol:8s}</code> (unavailable)")

    lines.extend([
        f"\n<i>Updated: {datetime.now().strftime('%H:%M:%S')}</i>",
        "\n/setsymbols SYM1,SYM2 — update watchlist",
        "/search QUERY — find new assets to add",
    ])
    await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)


# ──────────────────────────────────────────────────────────────────────────
# Callback Handlers (inline button confirmations)
# ──────────────────────────────────────────────────────────────────────────

@router.callback_query(F.data.startswith("buy|"))
async def callback_confirm_buy(callback: CallbackQuery):
    if callback.from_user.id not in _authorized_users:
        await callback.answer("Unauthorized", show_alert=True)
        return
    parts = callback.data.split("|")
    if len(parts) < 3:
        await callback.answer("Invalid action data", show_alert=True)
        return
    symbol = parts[1]
    try:
        qty = float(parts[2])
    except ValueError:
        await callback.answer("Invalid quantity", show_alert=True)
        return

    try:
        with _broker_lock:
            order = _broker.market_order(symbol, qty, "buy")
        await callback.message.edit_text(
            f"BUY order placed: {symbol} x {qty}\nOrder ID: {order['id'][:8]}..."
        )
    except Exception as e:
        await callback.message.edit_text(f"Order failed: {e}")
    await callback.answer()


@router.callback_query(F.data.startswith("sell|"))
async def callback_confirm_sell(callback: CallbackQuery):
    if callback.from_user.id not in _authorized_users:
        await callback.answer("Unauthorized", show_alert=True)
        return
    parts = callback.data.split("|")
    if len(parts) < 3:
        await callback.answer("Invalid action data", show_alert=True)
        return
    symbol = parts[1]
    try:
        qty = float(parts[2])
    except ValueError:
        await callback.answer("Invalid quantity", show_alert=True)
        return

    try:
        with _broker_lock:
            order = _broker.market_order(symbol, qty, "sell")
        await callback.message.edit_text(
            f"SELL order placed: {symbol} x {qty}\nOrder ID: {order['id'][:8]}..."
        )
    except Exception as e:
        await callback.message.edit_text(f"Order failed: {e}")
    await callback.answer()


@router.callback_query(F.data.startswith("close|"))
async def callback_confirm_close(callback: CallbackQuery):
    if callback.from_user.id not in _authorized_users:
        await callback.answer("Unauthorized", show_alert=True)
        return
    symbol = callback.data.split("|")[1]
    try:
        with _broker_lock:
            _broker.close_position(symbol)
        await callback.message.edit_text(f"Position in {symbol} closed.")
    except Exception as e:
        await callback.message.edit_text(f"Close failed: {e}")
    await callback.answer()


@router.callback_query(F.data == "closeall")
async def callback_confirm_closeall(callback: CallbackQuery):
    if callback.from_user.id not in _authorized_users:
        await callback.answer("Unauthorized", show_alert=True)
        return
    try:
        with _broker_lock:
            _broker.cancel_all_orders()
            _broker.close_all_positions()
        await callback.message.edit_text("ALL positions closed. ALL orders cancelled.")
    except Exception as e:
        await callback.message.edit_text(f"Emergency liquidation error: {e}")
    await callback.answer()


@router.callback_query(F.data == "cancel_order")
async def callback_cancel(callback: CallbackQuery):
    if callback.from_user.id not in _authorized_users:
        await callback.answer("Unauthorized", show_alert=True)
        return
    await callback.message.edit_text("Action cancelled.")
    await callback.answer()


# ──────────────────────────────────────────────────────────────────────────
# Bot Runner & Alert Sender
# ──────────────────────────────────────────────────────────────────────────

class TelegramBotManager:
    """Manages the Telegram bot lifecycle and provides alert methods."""

    def __init__(self, token: str, authorized_chat_ids: list[int] = None):
        self.token = token
        self.authorized_chat_ids = authorized_chat_ids or []
        self.bot = Bot(token=token)
        self.dp = Dispatcher()
        self.dp.include_router(router)
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start polling in the background."""
        logger.info("telegram.bot_starting")
        self._task = asyncio.create_task(self.dp.start_polling(self.bot))

    async def stop(self):
        """Stop the bot gracefully."""
        if self._task:
            self._task.cancel()
        await self.bot.session.close()
        logger.info("telegram.bot_stopped")

    async def send_alert(self, text: str, chat_id: int = None):
        """Send an alert message to specified or all authorized chats.
        Uses httpx directly to avoid cross-event-loop issues with aiogram's session.
        """
        import httpx
        targets = [chat_id] if chat_id else (self.authorized_chat_ids or list(_authorized_users))
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        for cid in targets:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(url, json={
                        "chat_id": cid,
                        "text": text,
                        "parse_mode": "HTML",
                    })
            except Exception as e:
                logger.error("telegram.send_failed", chat_id=cid, error=str(e))

    async def notify_trade(self, trade: dict):
        """Send trade notification to all authorized users."""
        side = trade.get('side', '?').upper()
        symbol = trade.get('symbol', '?')
        qty = trade.get('qty', 0)
        price = trade.get('price', 0)

        emoji = "BUY" if side == "BUY" else "SELL"
        text = (
            f"<b>{emoji} {symbol}</b>\n"
            f"Qty: {qty:.4f} @ ${price:.2f}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        if trade.get('stop_loss'):
            text += f"\nSL: ${trade['stop_loss']:.2f}"
        if trade.get('take_profit'):
            text += f"\nTP: ${trade['take_profit']:.2f}"

        await self.send_alert(text)

    async def notify_error(self, error: str, context: str = ""):
        """Send error notification."""
        text = f"<b>ERROR</b>\n{error}\nContext: {context}"
        await self.send_alert(text)

    async def notify_exit(self, trade: dict):
        """Send position exit notification."""
        symbol = trade.get('symbol', '?')
        pnl = trade.get('pnl', 0)
        reason = trade.get('reason', 'manual')
        pnl_emoji = "+" if pnl >= 0 else ""
        text = (
            f"<b>EXIT {symbol}</b>\n"
            f"PnL: {pnl_emoji}${pnl:.2f}\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        await self.send_alert(text)

    async def notify_risk_halt(self, reason: str):
        """Send risk halt notification."""
        text = (
            f"<b>RISK HALT</b>\n"
            f"Trading paused automatically.\n"
            f"Reason: {reason}\n"
            f"Use /resume to restart."
        )
        await self.send_alert(text)

    async def notify_signal(self, signal_info: dict):
        """Send signal notification (when configured)."""
        symbol = signal_info.get('symbol', '?')
        direction = signal_info.get('direction', '?')
        confidence = signal_info.get('confidence', 0)
        text = (
            f"<b>SIGNAL: {direction} {symbol}</b>\n"
            f"Confidence: {confidence:.0%}\n"
            f"Reason: {signal_info.get('reason', '-')}"
        )
        await self.send_alert(text)

    async def notify_daily_summary(self, summary: dict):
        """Send daily summary notification."""
        text = (
            f"<b>Daily Summary</b>\n"
            f"{'=' * 25}\n"
            f"Trades: {summary.get('trades', 0)}\n"
            f"Win Rate: {summary.get('win_rate', '0%')}\n"
            f"PnL: ${summary.get('daily_pnl', 0):.2f}\n"
            f"Return: {summary.get('daily_return', '0%')}\n"
        )
        await self.send_alert(text)

    async def notify_train_complete(self, metrics: dict):
        """Send ML training completion notification."""
        text = (
            f"<b>ML Model Retrained</b>\n"
            f"Accuracy: {metrics.get('accuracy', 0):.2%}\n"
            f"Features: {metrics.get('n_features', 0)}\n"
            f"Samples: {metrics.get('n_samples', 0)}"
        )
        await self.send_alert(text)

    def send_sync(self, text: str):
        """Thread-safe synchronous alert (called from main thread)."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.send_alert(text))
            else:
                loop.run_until_complete(self.send_alert(text))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.send_alert(text))
            loop.close()

    def is_paused(self) -> bool:
        """Check if bot is paused via Telegram command."""
        return _runtime_state.is_paused() if _runtime_state else False

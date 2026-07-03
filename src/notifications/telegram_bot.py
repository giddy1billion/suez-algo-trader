"""
Telegram Bot — Full management interface via aiogram 3.x.
Provides interactive controls, real-time alerts, and portfolio management.
ALL settings configurable in real-time via Telegram.

Commands:
    /start       - Welcome + status overview
    /status      - Account balance, equity, positions
    /positions   - Detailed position list
    /orders      - Open orders
    /pnl         - Today's P&L summary
    /trades      - Recent trade history
    /signals     - Current active signals
    /buy         - Manual buy order (e.g., /buy AAPL 10)
    /sell        - Manual sell order (e.g., /sell AAPL 10)
    /close       - Close position (e.g., /close AAPL)
    /closeall    - Emergency: close all positions
    /cancelall   - Cancel all pending orders
    /pause       - Pause bot trading
    /resume      - Resume bot trading
    /strategy    - Show/switch active strategy
    /risk        - Show risk parameters
    /config      - View all settings (e.g., /config risk)
    /set         - Universal setter (e.g., /set momentum_fast_ema 8)
    /setrisk     - Set risk param (e.g., /setrisk max_daily_loss_pct 0.05)
    /setstrategy - Switch strategy (momentum|mean_reversion|ml)
    /setsymbols  - Change symbols (e.g., /setsymbols AAPL,TSLA,BTC/USD)
    /setinterval - Change cycle interval (e.g., /setinterval 120)
    /settf       - Change timeframe (e.g., /settf 15Min)
    /setlookback - Change lookback (e.g., /setlookback 500)
    /setauto     - Configure automation (e.g., /setauto train 12)
    /setnotify   - Toggle alerts (e.g., /setnotify signal on)
    /backtest    - Backtrader backtest (e.g., /backtest AAPL 30)
    /backtestvbt - VectorBT backtest (e.g., /backtestvbt AAPL 30)
    /sweep       - Parameter sweep (e.g., /sweep TSLA 60)
    /train       - Train ML model (e.g., /train AAPL,TSLA 1000)
    /modelinfo   - ML model metadata (type, date, features, accuracy)
    /predict     - ML prediction (e.g., /predict BTC/USD)
    /help        - Show all commands
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

logger = get_logger(__name__)

router = Router()

# These get set during bot initialization
_broker = None
_engine = None
_risk_manager = None
_strategy = None
_db = None
_bot_paused = False
_authorized_users: set[int] = set()

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
        return changes


def set_components(broker, engine, risk_manager, strategy, db=None, authorized_chat_ids: list[int] = None):
    """Inject trading components into the bot module."""
    global _broker, _engine, _risk_manager, _strategy, _db, _authorized_users
    _broker = broker
    _engine = engine
    _risk_manager = risk_manager
    _strategy = strategy
    _db = db
    if authorized_chat_ids:
        _authorized_users = set(authorized_chat_ids)


def _is_authorized(message: Message) -> bool:
    """Check if user is authorized to use the bot."""
    if not _authorized_users:
        return True  # No restriction if not configured
    return message.from_user.id in _authorized_users


# ──────────────────────────────────────────────────────────────────────────
# Command Handlers
# ──────────────────────────────────────────────────────────────────────────

@router.message(CommandStart())
async def cmd_start(message: Message):
    # Auto-register first user if no chat IDs are configured
    global _authorized_users
    if not _authorized_users:
        _authorized_users.add(message.from_user.id)
        logger.info("telegram.auto_registered", user_id=message.from_user.id,
                    username=message.from_user.username)

    if not _is_authorized(message):
        await message.answer("Unauthorized.")
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
        f"<i>Mode: {'PAUSED' if _bot_paused else 'ACTIVE'}</i>"
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
        "/config [category] - View all settings\n"
        "/set PARAM VALUE - Change any setting\n"
        "/setrisk PARAM VALUE - Set risk param\n"
        "/setstrategy NAME - Switch strategy\n"
        "/setsymbols SYM1,SYM2 - Change symbols\n"
        "/setinterval SECS - Change cycle interval\n"
        "/settf TIMEFRAME - Change timeframe\n"
        "/setlookback BARS - Change lookback\n"
        "/setauto TYPE HOURS - Configure automation\n"
        "/setnotify TYPE on|off - Toggle alerts\n\n"
        "<b>Backtesting & ML:</b>\n"
        "/backtest [SYMBOL] [DAYS] - Backtrader backtest\n"
        "/backtestvbt [SYMBOL] [DAYS] - VectorBT backtest\n"
        "/sweep [SYMBOL] [DAYS] - Parameter sweep\n"
        "/train [SYMBOLS] [BARS] - Train ML model\n"
        "/modelinfo - ML model metadata\n"
        "/predict [SYMBOL] - ML prediction + confidence\n"
    )
    await message.answer(text, parse_mode=ParseMode.HTML)


@router.message(Command("status"))
async def cmd_status(message: Message):
    if not _is_authorized(message) or not _broker:
        await message.answer("Not connected.")
        return

    try:
        account = _broker.get_account()
        positions = _broker.get_positions()

        pnl = account['equity'] - account['last_equity']
        pnl_pct = (pnl / account['last_equity'] * 100) if account['last_equity'] > 0 else 0

        text = (
            f"<b>Account Status</b> {'[PAUSED]' if _bot_paused else '[ACTIVE]'}\n"
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
        await message.answer(f"Error: {e}")


@router.message(Command("positions"))
async def cmd_positions(message: Message):
    if not _is_authorized(message) or not _broker:
        return

    try:
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
        await message.answer(f"Error: {e}")


@router.message(Command("orders"))
async def cmd_orders(message: Message):
    if not _is_authorized(message) or not _broker:
        return

    try:
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
        await message.answer(f"Error: {e}")


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
        await message.answer(f"Error fetching trades: {e}")


@router.message(Command("signals"))
async def cmd_signals(message: Message):
    if not _is_authorized(message) or not _strategy or not _broker:
        return

    try:
        data = {}
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
        await message.answer(f"Error generating signals: {e}")


# ──────────────────────────────────────────────────────────────────────────
# Trading Commands
# ──────────────────────────────────────────────────────────────────────────

@router.message(Command("buy"))
async def cmd_buy(message: Message):
    if not _is_authorized(message) or not _broker:
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
        _broker.cancel_all_orders()
        await message.answer("All pending orders cancelled.")
    except Exception as e:
        await message.answer(f"Error: {e}")


# ──────────────────────────────────────────────────────────────────────────
# Bot Control Commands
# ──────────────────────────────────────────────────────────────────────────

@router.message(Command("pause"))
async def cmd_pause(message: Message):
    if not _is_authorized(message):
        return
    global _bot_paused
    _bot_paused = True
    await message.answer("Bot PAUSED. Auto-trading disabled.\nUse /resume to restart.")


@router.message(Command("resume"))
async def cmd_resume(message: Message):
    if not _is_authorized(message):
        return
    global _bot_paused
    _bot_paused = False
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
    days = int(parts[2]) if len(parts) > 2 else 30

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

            metrics = run_backtrader_backtest(df, initial_cash=10000.0)
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
            # Fallback to simple custom backtest
            from backtesting.backtest import SimpleBacktest
            bt = SimpleBacktest(df)
            results = bt.run()
            text = (
                f"<b>Backtest Results: {symbol}</b>\n"
                f"Return: {results.get('total_return', 0):.2%}\n"
                f"Trades: {results.get('total_trades', 0)}\n"
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

        strategy.train(training_data)

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
            from config.settings import settings

            metrics = vectorbt_momentum_backtest(df, initial_cash=settings.backtest_initial_cash)
            text = (
                f"<b>VectorBT Results: {symbol}</b>\n"
                f"{'=' * 30}\n"
                f"EMA:          12/26 (default)\n"
                f"Fees:         0.1%\n"
                f"Bars:         {len(df)}\n"
                f"{'─' * 30}\n"
                f"Return:       {metrics['total_return']:.2%}\n"
                f"Trades:       {metrics['total_trades']}\n"
                f"Win Rate:     {metrics['win_rate']:.1%}\n"
                f"Sharpe:       {metrics['sharpe_ratio']:.2f}\n"
                f"Max DD:       {metrics['max_drawdown']:.2%}\n"
            )

            # Quick parameter sweep (top 3)
            try:
                sweep_df = vectorbt_parameter_sweep(df)
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
        sweep_df = vectorbt_parameter_sweep(df)

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
        await message.answer(f"Error loading model info: {e}")


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
        proba = ml_strat.model.predict_proba(features)[0]
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
        atr = float(latest['atr'].iloc[0]) if 'atr' in df_feat.columns and not np.isnan(latest['atr'].iloc[0]) else price * 0.02

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
# Callback Handlers (inline button confirmations)
# ──────────────────────────────────────────────────────────────────────────

@router.callback_query(F.data.startswith("buy|"))
async def callback_confirm_buy(callback: CallbackQuery):
    parts = callback.data.split("|")
    symbol = parts[1]
    qty = float(parts[2])

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
    parts = callback.data.split("|")
    symbol = parts[1]
    qty = float(parts[2])

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
        """Send an alert message to specified or all authorized chats."""
        targets = [chat_id] if chat_id else (self.authorized_chat_ids or list(_authorized_users))
        for cid in targets:
            try:
                await self.bot.send_message(cid, text, parse_mode=ParseMode.HTML)
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
        return _bot_paused

"""
Telegram Bot — Full management interface via aiogram 3.x.
Provides interactive controls, real-time alerts, and portfolio management.

Commands:
    /start     - Welcome + status overview
    /status    - Account balance, equity, positions
    /positions - Detailed position list
    /orders    - Open orders
    /pnl       - Today's P&L summary
    /trades    - Recent trade history
    /signals   - Current active signals
    /buy       - Manual buy order (e.g., /buy AAPL 10)
    /sell      - Manual sell order (e.g., /sell AAPL 10)
    /close     - Close position (e.g., /close AAPL)
    /closeall  - Emergency: close all positions
    /cancelall - Cancel all pending orders
    /pause     - Pause bot trading
    /resume    - Resume bot trading
    /strategy  - Show/switch active strategy
    /risk      - Show risk parameters
    /backtest  - Quick backtest summary
    /help      - Show all commands
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
        "/setrisk PARAM VALUE - Set risk param\n"
        "/setstrategy NAME - Switch strategy\n"
        "/setsymbols SYM1,SYM2 - Change symbols\n"
        "/setinterval SECS - Change cycle interval\n\n"
        "<b>Tools:</b>\n"
        "/backtest [SYMBOL] [DAYS] - Run backtest\n"
        "/train - Retrain ML model\n"
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
    """Trigger ML model retraining."""
    if not _is_authorized(message):
        return

    with _runtime_lock:
        _runtime_changes["trigger_train"] = True

    await message.answer(
        "ML model retraining requested.\n"
        "Training will start on the next cycle. This may take a few minutes.\n"
        "Use /strategy to check model status."
    )


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
        targets = [chat_id] if chat_id else self.authorized_chat_ids
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

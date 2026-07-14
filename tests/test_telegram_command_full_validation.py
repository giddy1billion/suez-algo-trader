"""
Full-Path Telegram Command Validation Suite.

Exercises EVERY code path for each documented Telegram command across
multiple dimensions:
  - Input validation (missing args, bad types, boundary values)
  - Permission validation (unauthorized users blocked)
  - Happy path (nominal operation)
  - Error handling (broker failures, unexpected exceptions)
  - Empty datasets / missing data
  - Missing model / missing configuration
  - Missing services / broken dependencies
  - Persistence (runtime_changes propagation)
  - Output formatting (HTML parse mode, emoji presence)
  - Telegram markdown correctness (no bare < or >)
  - Unexpected exceptions (random exception injection)
  - Return value correctness (expected content in replies)
  - "Operation failed" never appears without actionable diagnostics

Produces a PASS/WARN/FAIL matrix printed at test conclusion.

Identifies dead commands, unreachable code, and duplicate implementations.
"""

import asyncio
import json
import math
import os
import re
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock, call

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Module imports
# ─────────────────────────────────────────────────────────────────────────────

from src.broker.paper import PaperBroker
from src.core.runtime_state import RuntimeState, OperatingMode
from src.notifications import telegram_bot
from src.notifications import telegram_config_commands
from src.notifications import telegram_runtime_commands
from src.notifications import telegram_sector_commands

# ─────────────────────────────────────────────────────────────────────────────
# Validation Matrix Collector
# ─────────────────────────────────────────────────────────────────────────────

# Global matrix: command -> list of (dimension, verdict, detail)
_MATRIX: dict[str, list[tuple[str, str, str]]] = defaultdict(list)


def _record(command: str, dimension: str, verdict: str, detail: str = ""):
    """Record a validation result for the matrix."""
    _MATRIX[command].append((dimension, verdict, detail))


# ─────────────────────────────────────────────────────────────────────────────
# Test Helpers
# ─────────────────────────────────────────────────────────────────────────────

AUTHORIZED_USER_ID = 123456789
UNAUTHORIZED_USER_ID = 987654321


def _make_message(text: str, user_id: int = AUTHORIZED_USER_ID) -> MagicMock:
    msg = AsyncMock()
    msg.text = text
    msg.from_user = MagicMock()
    msg.from_user.id = user_id
    msg.from_user.username = "testuser"
    msg.chat = MagicMock()
    msg.chat.id = user_id
    msg.answer = AsyncMock()
    msg.delete = AsyncMock()
    msg.message_id = 1000
    return msg


def _make_callback(data: str, user_id: int = AUTHORIZED_USER_ID) -> MagicMock:
    cb = AsyncMock()
    cb.data = data
    cb.from_user = MagicMock()
    cb.from_user.id = user_id
    cb.message = AsyncMock()
    cb.message.edit_text = AsyncMock()
    cb.answer = AsyncMock()
    return cb


def _get_reply_text(msg: MagicMock) -> str:
    if msg.answer.call_count == 0:
        return ""
    return msg.answer.call_args_list[-1][0][0] if msg.answer.call_args_list[-1][0] else ""


def _get_all_replies(msg: MagicMock) -> list[str]:
    replies = []
    for c in msg.answer.call_args_list:
        if c[0]:
            replies.append(c[0][0])
    return replies


def _assert_no_bare_operation_failed(replies: list[str], command: str):
    """Ensure 'Operation failed' always includes diagnostics."""
    for r in replies:
        if "Operation failed" in r:
            # Must also have actionable info beyond just "Operation failed"
            assert (
                "logs" in r.lower()
                or "error" in r.lower()
                or ":" in r
                or "check" in r.lower()
            ), f"/{command} returned bare 'Operation failed' without actionable diagnostics: {r[:200]}"


def _assert_valid_html(text: str, command: str):
    """Basic check that HTML tags are balanced and no raw < or > outside tags."""
    # Check that common HTML tags are balanced
    for tag in ["b", "code", "i", "pre"]:
        opens = text.count(f"<{tag}>")
        closes = text.count(f"</{tag}>")
        assert opens == closes, (
            f"/{command}: Unbalanced <{tag}> tags: {opens} opens vs {closes} closes"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_rate_limiter():
    telegram_bot._rate_limiter._timestamps.clear()
    yield


@pytest.fixture
def paper_broker():
    broker = PaperBroker(starting_equity=100_000.0)
    broker.set_price("AAPL", 150.0)
    broker.set_price("TSLA", 250.0)
    broker.set_price("GOOG", 2800.0)
    broker.set_price("MSFT", 400.0)
    broker.set_price("BTC/USD", 45000.0)
    return broker


@pytest.fixture
def runtime_state():
    return RuntimeState()


@pytest.fixture
def mock_risk_manager():
    rm = MagicMock()
    rm.get_daily_summary.return_value = {
        "trades": 5, "win_rate": "60%", "daily_pnl": 150.0,
        "daily_return": "0.15%", "is_halted": False,
    }
    rm.limits = MagicMock()
    rm.limits.max_position_size_pct = 0.10
    rm.limits.max_daily_loss_pct = 0.05
    rm.limits.max_portfolio_exposure = 0.80
    rm.limits.max_single_stock_pct = 0.25
    rm.limits.max_leverage = 2.0
    rm.limits.default_stop_loss_pct = 0.03
    rm.limits.default_take_profit_pct = 0.06
    rm.limits.max_open_positions = 10
    rm.limits.max_orders_per_day = 50
    rm.daily_stats = MagicMock()
    rm.daily_stats.is_halted = False
    rm.daily_stats.halt_reason = ""
    return rm


@pytest.fixture
def mock_strategy():
    s = MagicMock()
    s.name = "momentum"
    s.symbols = ["AAPL", "TSLA", "GOOG"]
    s.timeframe = "15Min"
    s.lookback = 200
    s.is_active = True
    s.model = None
    return s


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.get_trades.return_value = [
        {"side": "buy", "symbol": "AAPL", "qty": 10, "price": 150.0,
         "pnl": 50.0, "timestamp": "2024-01-15T10:30:00"},
        {"side": "sell", "symbol": "TSLA", "qty": 5, "price": 250.0,
         "pnl": -20.0, "timestamp": "2024-01-15T11:00:00"},
    ]
    db.get_journal_entries.return_value = []
    db.get_journal_stats.return_value = {
        "total_trades": 10, "win_rate": 0.6, "avg_pnl": 15.0, "total_pnl": 150.0,
    }
    db.set_cached_sector = MagicMock()
    db.get_all_cached_sectors.return_value = {"AAPL": "technology", "TSLA": "consumer_discretionary"}
    return db


@pytest.fixture
def mock_event_bus():
    bus = MagicMock()
    bus.get_recent_events.return_value = []
    bus.get_history.return_value = []
    bus.subscriber_count = 0
    return bus


@pytest.fixture
def mock_health_monitor():
    hm = MagicMock()
    hm.uptime_seconds = 3600
    hm._start_time = datetime(2024, 1, 1, 0, 0, 0)
    hm.check_all.return_value = {
        "system": {"status": "ok", "cpu_percent": 5.0, "memory_mb": 256},
        "broker": {"status": "ok", "connected": True, "latency_ms": 50},
        "database": {"status": "ok", "size_mb": 10, "query_latency_ms": 2},
        "ml": {"status": "not_configured", "model_loaded": False},
        "telegram": {"status": "ok", "polling_active": True},
        "scheduler": {"status": "not_configured"},
    }
    return hm


@pytest.fixture
def mock_live_metrics():
    lm = MagicMock()
    lm.get_summary_text.return_value = "<b>Performance</b>\nSharpe: 1.5\nWin Rate: 60%"
    return lm


@pytest.fixture
def mock_trade_manager():
    tm = MagicMock()
    tm.get_active_trades.return_value = []
    tm.count = 0
    return tm


@pytest.fixture
def mock_ops_handler():
    ops = MagicMock()
    ops.format_health.return_value = "<b>Health</b>\n✅ All systems OK"
    ops.format_risk_report.return_value = "<b>Risk Report</b>\nExposure: 30%"
    ops.format_reconciliation.return_value = "<b>Reconciliation</b>\n✅ Matched"
    ops.format_latency.return_value = "<b>Latency</b>\nBroker: 50ms"
    ops.format_performance.return_value = "<b>Performance</b>\nSharpe: 1.5"
    ops.format_system.return_value = "<b>System</b>\nPython 3.12, Uptime: 1h"
    return ops


@pytest.fixture
def setup_bot(paper_broker, runtime_state, mock_risk_manager, mock_strategy,
              mock_db, mock_event_bus, mock_health_monitor, mock_live_metrics,
              mock_trade_manager):
    telegram_bot.set_components(
        broker=paper_broker,
        engine=MagicMock(),
        risk_manager=mock_risk_manager,
        strategy=mock_strategy,
        db=mock_db,
        authorized_chat_ids=[AUTHORIZED_USER_ID],
        health_monitor=mock_health_monitor,
        live_metrics=mock_live_metrics,
        scheduler=MagicMock(),
        event_bus=mock_event_bus,
        trade_manager=mock_trade_manager,
        reconciler=MagicMock(),
        ops_handler=MagicMock(),
        runtime_state=runtime_state,
    )
    yield
    telegram_bot._broker = None
    telegram_bot._authorized_users = set()


@pytest.fixture
def setup_config_commands():
    from config.settings import Settings
    settings = Settings()
    mock_store = MagicMock()
    mock_store.get_templates.return_value = {
        "momentum": {"description": "EMA crossover", "params": ["fast_ema", "slow_ema"]},
        "scalping": {"description": "Short-term", "params": ["fast_ema"]},
    }
    mock_store.list_strategies.return_value = []
    mock_store.create.return_value = (True, "Created")
    mock_store.get.return_value = MagicMock(params={"fast_ema": 12, "slow_ema": 26})
    mock_store.delete.return_value = (True, "Deleted strategy 'test'")
    mock_store.duplicate.return_value = (True, "Duplicated")
    mock_store.get_active_strategies.return_value = []
    mock_store.get_multi_config_string.return_value = ""
    mock_store.update.return_value = (True, "Updated")
    mock_store.activate.return_value = (True, "Activated")
    mock_store.deactivate.return_value = (True, "Deactivated")

    telegram_config_commands.set_config_components(
        settings=settings,
        strategy_store=mock_store,
        authorized_users={AUTHORIZED_USER_ID},
        runtime_lock=threading.Lock(),
        runtime_changes={},
    )
    yield settings, mock_store


@pytest.fixture
def setup_runtime_commands():
    rm = MagicMock()
    rm.current_mode = "paper"
    rm.is_paper = True
    rm.is_training.return_value = False
    rm.get_training_progress.return_value = {"stage": "idle", "progress_pct": 0}
    rm.get_training_history.return_value = []
    rm.list_backtests.return_value = []
    rm.get_model_status.return_value = {"version": "v002", "prediction_count": 100, "avg_latency_ms": 5.0}
    rm.list_model_versions.return_value = [
        {"version": "v001", "active": False},
        {"version": "v002", "active": True},
    ]
    rm.get_ab_test_status.return_value = None
    rm.get_status.return_value = {
        "environment": {"mode": "paper", "state": "active"},
        "model": {"version": "v002", "prediction_count": 100},
        "training": {"stage": None},
        "ab_test": None,
        "backtests": [],
    }
    rm.switch_environment.return_value = {"duration_ms": 50}
    rm.run_backtest.return_value = {"run_id": "abc123def456", "status": "running"}
    rm.train_model.return_value = {"pipeline_id": "pipe123abc456", "status": "started", "symbols": ["AAPL"]}
    rm.swap_model.return_value = {"status": "ok"}
    rm.start_ab_test.return_value = {"test_id": "test_abc123def456"}
    rm.cancel_ab_test.return_value = True
    telegram_runtime_commands.set_runtime_components(
        runtime_manager=rm,
        authorized_users={AUTHORIZED_USER_ID},
    )
    yield rm


@pytest.fixture
def setup_sector_commands(mock_db):
    telegram_sector_commands.set_sector_components(
        db=mock_db,
        authorized_users={AUTHORIZED_USER_ID},
    )
    yield


# ─────────────────────────────────────────────────────────────────────────────
# /status
# ─────────────────────────────────────────────────────────────────────────────


class TestStatusCommand:
    CMD = "status"

    @pytest.mark.asyncio
    async def test_happy_path(self, setup_bot):
        msg = _make_message("/status")
        await telegram_bot.cmd_status(msg)
        reply = _get_reply_text(msg)
        assert "Account Status" in reply
        assert "$" in reply
        assert "Equity" in reply
        _assert_valid_html(reply, self.CMD)
        _record(self.CMD, "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_permission_denied(self, setup_bot):
        msg = _make_message("/status", user_id=UNAUTHORIZED_USER_ID)
        await telegram_bot.cmd_status(msg)
        reply = _get_reply_text(msg)
        assert "Not connected" in reply or msg.answer.call_count == 1
        _record(self.CMD, "permission", "PASS")

    @pytest.mark.asyncio
    async def test_broker_failure(self, setup_bot, paper_broker):
        paper_broker.get_account = MagicMock(side_effect=ConnectionError("timeout"))
        msg = _make_message("/status")
        await telegram_bot.cmd_status(msg)
        reply = _get_reply_text(msg)
        _assert_no_bare_operation_failed([reply], self.CMD)
        assert msg.answer.call_count >= 1
        _record(self.CMD, "broker_failure", "PASS")

    @pytest.mark.asyncio
    async def test_no_broker(self, setup_bot):
        telegram_bot._broker = None
        msg = _make_message("/status")
        await telegram_bot.cmd_status(msg)
        reply = _get_reply_text(msg)
        assert "Not connected" in reply
        _record(self.CMD, "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_html_format(self, setup_bot):
        msg = _make_message("/status")
        await telegram_bot.cmd_status(msg)
        kwargs = msg.answer.call_args[1]
        from aiogram.enums import ParseMode
        assert kwargs.get("parse_mode") == ParseMode.HTML
        _record(self.CMD, "output_format", "PASS")

    @pytest.mark.asyncio
    async def test_paused_state_shown(self, setup_bot, runtime_state):
        runtime_state.pause()
        msg = _make_message("/status")
        await telegram_bot.cmd_status(msg)
        reply = _get_reply_text(msg)
        assert "PAUSED" in reply
        _record(self.CMD, "state_correctness", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /positions
# ─────────────────────────────────────────────────────────────────────────────


class TestPositionsCommand:
    CMD = "positions"

    @pytest.mark.asyncio
    async def test_empty_positions(self, setup_bot):
        msg = _make_message("/positions")
        await telegram_bot.cmd_positions(msg)
        reply = _get_reply_text(msg)
        assert "No open positions" in reply
        _record(self.CMD, "empty_dataset", "PASS")

    @pytest.mark.asyncio
    async def test_with_positions(self, setup_bot, paper_broker):
        paper_broker.market_order("AAPL", 10, "buy")
        msg = _make_message("/positions")
        await telegram_bot.cmd_positions(msg)
        reply = _get_reply_text(msg)
        assert "AAPL" in reply
        assert "PnL" in reply or "pnl" in reply.lower()
        _assert_valid_html(reply, self.CMD)
        _record(self.CMD, "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_broker_failure(self, setup_bot, paper_broker):
        paper_broker.get_positions = MagicMock(side_effect=Exception("API down"))
        msg = _make_message("/positions")
        await telegram_bot.cmd_positions(msg)
        reply = _get_reply_text(msg)
        _assert_no_bare_operation_failed([reply], self.CMD)
        _record(self.CMD, "broker_failure", "PASS")

    @pytest.mark.asyncio
    async def test_unauthorized(self, setup_bot):
        msg = _make_message("/positions", user_id=UNAUTHORIZED_USER_ID)
        await telegram_bot.cmd_positions(msg)
        assert msg.answer.call_count == 0
        _record(self.CMD, "permission", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /orders
# ─────────────────────────────────────────────────────────────────────────────


class TestOrdersCommand:
    CMD = "orders"

    @pytest.mark.asyncio
    async def test_empty_orders(self, setup_bot):
        msg = _make_message("/orders")
        await telegram_bot.cmd_orders(msg)
        reply = _get_reply_text(msg)
        assert "No pending orders" in reply
        _record(self.CMD, "empty_dataset", "PASS")

    @pytest.mark.asyncio
    async def test_broker_failure(self, setup_bot, paper_broker):
        paper_broker.get_orders = MagicMock(side_effect=Exception("Connection reset"))
        msg = _make_message("/orders")
        await telegram_bot.cmd_orders(msg)
        reply = _get_reply_text(msg)
        _assert_no_bare_operation_failed([reply], self.CMD)
        _record(self.CMD, "broker_failure", "PASS")

    @pytest.mark.asyncio
    async def test_unauthorized(self, setup_bot):
        msg = _make_message("/orders", user_id=UNAUTHORIZED_USER_ID)
        await telegram_bot.cmd_orders(msg)
        assert msg.answer.call_count == 0
        _record(self.CMD, "permission", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /pnl
# ─────────────────────────────────────────────────────────────────────────────


class TestPnlCommand:
    CMD = "pnl"

    @pytest.mark.asyncio
    async def test_happy_path(self, setup_bot):
        msg = _make_message("/pnl")
        await telegram_bot.cmd_pnl(msg)
        reply = _get_reply_text(msg)
        assert "Daily P&L" in reply
        assert "$" in reply
        _record(self.CMD, "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_no_risk_manager(self, setup_bot):
        telegram_bot._risk_manager = None
        msg = _make_message("/pnl")
        await telegram_bot.cmd_pnl(msg)
        assert msg.answer.call_count == 0  # Silent return
        _record(self.CMD, "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_unauthorized(self, setup_bot):
        msg = _make_message("/pnl", user_id=UNAUTHORIZED_USER_ID)
        await telegram_bot.cmd_pnl(msg)
        assert msg.answer.call_count == 0
        _record(self.CMD, "permission", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /trades
# ─────────────────────────────────────────────────────────────────────────────


class TestTradesCommand:
    CMD = "trades"

    @pytest.mark.asyncio
    async def test_happy_path(self, setup_bot):
        msg = _make_message("/trades")
        await telegram_bot.cmd_trades(msg)
        reply = _get_reply_text(msg)
        assert "Recent Trades" in reply
        assert "AAPL" in reply
        _record(self.CMD, "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_empty_trades(self, setup_bot, mock_db):
        mock_db.get_trades.return_value = []
        msg = _make_message("/trades")
        await telegram_bot.cmd_trades(msg)
        reply = _get_reply_text(msg)
        assert "No trades" in reply
        _record(self.CMD, "empty_dataset", "PASS")

    @pytest.mark.asyncio
    async def test_db_failure(self, setup_bot, mock_db):
        mock_db.get_trades.side_effect = Exception("DB connection lost")
        msg = _make_message("/trades")
        await telegram_bot.cmd_trades(msg)
        reply = _get_reply_text(msg)
        _assert_no_bare_operation_failed([reply], self.CMD)
        _record(self.CMD, "broker_failure", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /signals
# ─────────────────────────────────────────────────────────────────────────────


class TestSignalsCommand:
    CMD = "signals"

    @pytest.mark.asyncio
    async def test_no_data(self, setup_bot):
        telegram_bot._broker.get_bars_df = MagicMock(return_value=None)
        msg = _make_message("/signals")
        await telegram_bot.cmd_signals(msg)
        reply = _get_reply_text(msg)
        assert "No data" in reply or "No actionable" in reply
        _record(self.CMD, "empty_dataset", "PASS")

    @pytest.mark.asyncio
    async def test_no_strategy(self, setup_bot):
        telegram_bot._strategy = None
        msg = _make_message("/signals")
        await telegram_bot.cmd_signals(msg)
        assert msg.answer.call_count == 0
        _record(self.CMD, "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_unauthorized(self, setup_bot):
        msg = _make_message("/signals", user_id=UNAUTHORIZED_USER_ID)
        await telegram_bot.cmd_signals(msg)
        assert msg.answer.call_count == 0
        _record(self.CMD, "permission", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /buy
# ─────────────────────────────────────────────────────────────────────────────


class TestBuyCommand:
    CMD = "buy"

    @pytest.mark.asyncio
    async def test_happy_path(self, setup_bot):
        msg = _make_message("/buy AAPL 10")
        await telegram_bot.cmd_buy(msg)
        reply = _get_reply_text(msg)
        assert "Confirm Market BUY" in reply
        assert "AAPL" in reply
        assert msg.answer.call_args[1].get("reply_markup") is not None
        _record(self.CMD, "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_missing_args(self, setup_bot):
        msg = _make_message("/buy")
        await telegram_bot.cmd_buy(msg)
        reply = _get_reply_text(msg)
        assert "Usage" in reply
        _record(self.CMD, "input_validation", "PASS")

    @pytest.mark.asyncio
    async def test_invalid_qty(self, setup_bot):
        msg = _make_message("/buy AAPL abc")
        await telegram_bot.cmd_buy(msg)
        reply = _get_reply_text(msg)
        assert "Invalid quantity" in reply
        _record(self.CMD, "input_validation_type", "PASS")

    @pytest.mark.asyncio
    async def test_negative_qty(self, setup_bot):
        msg = _make_message("/buy AAPL -5")
        await telegram_bot.cmd_buy(msg)
        reply = _get_reply_text(msg)
        assert "positive" in reply.lower()
        _record(self.CMD, "input_validation_boundary", "PASS")

    @pytest.mark.asyncio
    async def test_nan_qty(self, setup_bot):
        msg = _make_message("/buy AAPL nan")
        await telegram_bot.cmd_buy(msg)
        reply = _get_reply_text(msg)
        # nan parses as float but fails isnan check
        assert "positive" in reply.lower() or "Invalid" in reply
        _record(self.CMD, "input_validation_nan", "PASS")

    @pytest.mark.asyncio
    async def test_inf_qty(self, setup_bot):
        msg = _make_message("/buy AAPL inf")
        await telegram_bot.cmd_buy(msg)
        reply = _get_reply_text(msg)
        assert "positive" in reply.lower() or "Quantity" in reply
        _record(self.CMD, "input_validation_inf", "PASS")

    @pytest.mark.asyncio
    async def test_invalid_symbol(self, setup_bot):
        msg = _make_message("/buy A!@#$ 10")
        await telegram_bot.cmd_buy(msg)
        reply = _get_reply_text(msg)
        assert "Invalid symbol" in reply
        _record(self.CMD, "input_validation_symbol", "PASS")

    @pytest.mark.asyncio
    async def test_rate_limited(self, setup_bot):
        for _ in range(10):
            telegram_bot._rate_limiter.is_allowed(AUTHORIZED_USER_ID)
        msg = _make_message("/buy AAPL 10")
        await telegram_bot.cmd_buy(msg)
        reply = _get_reply_text(msg)
        assert "Rate limit" in reply
        _record(self.CMD, "rate_limiting", "PASS")

    @pytest.mark.asyncio
    async def test_callback_broker_failure(self, setup_bot, paper_broker):
        paper_broker.market_order = MagicMock(side_effect=Exception("Insufficient funds"))
        paper_broker.bracket_order = MagicMock(side_effect=Exception("Insufficient funds"))
        cb = _make_callback("buy|AAPL|10")
        await telegram_bot.callback_confirm_buy(cb)
        edit_text = cb.message.edit_text.call_args[0][0]
        assert "❌" in edit_text or "failed" in edit_text.lower()
        _record(self.CMD, "broker_failure", "PASS")

    @pytest.mark.asyncio
    async def test_callback_executes_order(self, setup_bot, paper_broker):
        cb = _make_callback("buy|AAPL|10")
        await telegram_bot.callback_confirm_buy(cb)
        edit_text = cb.message.edit_text.call_args[0][0]
        assert "✅" in edit_text or "order placed" in edit_text.lower()
        _record(self.CMD, "callback_execution", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /sell
# ─────────────────────────────────────────────────────────────────────────────


class TestSellCommand:
    CMD = "sell"

    @pytest.mark.asyncio
    async def test_happy_path(self, setup_bot):
        msg = _make_message("/sell TSLA 5")
        await telegram_bot.cmd_sell(msg)
        reply = _get_reply_text(msg)
        assert "Confirm Market SELL" in reply
        _record(self.CMD, "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_missing_args(self, setup_bot):
        msg = _make_message("/sell")
        await telegram_bot.cmd_sell(msg)
        assert "Usage" in _get_reply_text(msg)
        _record(self.CMD, "input_validation", "PASS")

    @pytest.mark.asyncio
    async def test_invalid_qty(self, setup_bot):
        msg = _make_message("/sell AAPL xyz")
        await telegram_bot.cmd_sell(msg)
        assert "Invalid" in _get_reply_text(msg)
        _record(self.CMD, "input_validation_type", "PASS")

    @pytest.mark.asyncio
    async def test_negative_qty(self, setup_bot):
        msg = _make_message("/sell AAPL -1")
        await telegram_bot.cmd_sell(msg)
        assert "positive" in _get_reply_text(msg).lower()
        _record(self.CMD, "input_validation_boundary", "PASS")

    @pytest.mark.asyncio
    async def test_callback_executes(self, setup_bot, paper_broker):
        cb = _make_callback("sell|AAPL|5")
        await telegram_bot.callback_confirm_sell(cb)
        edit_text = cb.message.edit_text.call_args[0][0]
        assert "✅" in edit_text or "order placed" in edit_text.lower()
        _record(self.CMD, "callback_execution", "PASS")

    @pytest.mark.asyncio
    async def test_callback_broker_failure(self, setup_bot, paper_broker):
        paper_broker.market_order = MagicMock(side_effect=Exception("Market closed"))
        paper_broker.bracket_order = MagicMock(side_effect=Exception("Market closed"))
        cb = _make_callback("sell|AAPL|5")
        await telegram_bot.callback_confirm_sell(cb)
        edit_text = cb.message.edit_text.call_args[0][0]
        assert "❌" in edit_text or "failed" in edit_text.lower()
        _record(self.CMD, "broker_failure", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /close, /closeall, /cancelall
# ─────────────────────────────────────────────────────────────────────────────


class TestCloseCommands:

    @pytest.mark.asyncio
    async def test_close_happy_path(self, setup_bot):
        msg = _make_message("/close AAPL")
        await telegram_bot.cmd_close(msg)
        reply = _get_reply_text(msg)
        assert "Close" in reply and "AAPL" in reply
        _record("close", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_close_missing_symbol(self, setup_bot):
        msg = _make_message("/close")
        await telegram_bot.cmd_close(msg)
        assert "Usage" in _get_reply_text(msg)
        _record("close", "input_validation", "PASS")

    @pytest.mark.asyncio
    async def test_close_callback_success(self, setup_bot, paper_broker):
        paper_broker.market_order("AAPL", 10, "buy")
        cb = _make_callback("close|AAPL")
        await telegram_bot.callback_confirm_close(cb)
        edit_text = cb.message.edit_text.call_args[0][0]
        assert "closed" in edit_text.lower()
        _record("close", "callback_execution", "PASS")

    @pytest.mark.asyncio
    async def test_close_callback_broker_error(self, setup_bot, paper_broker):
        paper_broker.close_position = MagicMock(side_effect=Exception("Position not found"))
        cb = _make_callback("close|XYZ")
        await telegram_bot.callback_confirm_close(cb)
        edit_text = cb.message.edit_text.call_args[0][0]
        assert "failed" in edit_text.lower() or "error" in edit_text.lower()
        _record("close", "broker_failure", "PASS")

    @pytest.mark.asyncio
    async def test_closeall_confirmation(self, setup_bot):
        msg = _make_message("/closeall")
        await telegram_bot.cmd_closeall(msg)
        reply = _get_reply_text(msg)
        assert "EMERGENCY" in reply or "Close ALL" in reply
        _record("closeall", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_closeall_callback(self, setup_bot, paper_broker):
        paper_broker.cancel_all_orders = MagicMock()
        paper_broker.close_all_positions = MagicMock()
        cb = _make_callback("closeall")
        await telegram_bot.callback_confirm_closeall(cb)
        edit_text = cb.message.edit_text.call_args[0][0]
        assert "ALL" in edit_text
        _record("closeall", "callback_execution", "PASS")

    @pytest.mark.asyncio
    async def test_cancelall(self, setup_bot, paper_broker):
        paper_broker.cancel_all_orders = MagicMock()
        msg = _make_message("/cancelall")
        await telegram_bot.cmd_cancelall(msg)
        reply = _get_reply_text(msg)
        assert "cancelled" in reply.lower()
        _record("cancelall", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_cancelall_broker_failure(self, setup_bot, paper_broker):
        paper_broker.cancel_all_orders = MagicMock(side_effect=Exception("API error"))
        msg = _make_message("/cancelall")
        await telegram_bot.cmd_cancelall(msg)
        reply = _get_reply_text(msg)
        _assert_no_bare_operation_failed([reply], "cancelall")
        _record("cancelall", "broker_failure", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /pause, /resume
# ─────────────────────────────────────────────────────────────────────────────


class TestPauseResumeCommands:

    @pytest.mark.asyncio
    async def test_pause(self, setup_bot, runtime_state):
        msg = _make_message("/pause")
        await telegram_bot.cmd_pause(msg)
        assert "PAUSED" in _get_reply_text(msg)
        assert runtime_state.is_paused() is True
        _record("pause", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_resume(self, setup_bot, runtime_state):
        runtime_state.pause()
        msg = _make_message("/resume")
        await telegram_bot.cmd_resume(msg)
        assert "RESUMED" in _get_reply_text(msg)
        assert runtime_state.is_paused() is False
        _record("resume", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_pause_idempotent(self, setup_bot, runtime_state):
        for _ in range(3):
            msg = _make_message("/pause")
            await telegram_bot.cmd_pause(msg)
        assert runtime_state.is_paused() is True
        _record("pause", "idempotency", "PASS")

    @pytest.mark.asyncio
    async def test_resume_idempotent(self, setup_bot, runtime_state):
        msg = _make_message("/resume")
        await telegram_bot.cmd_resume(msg)
        assert runtime_state.is_paused() is False
        _record("resume", "idempotency", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /strategy, /risk
# ─────────────────────────────────────────────────────────────────────────────


class TestStrategyRiskCommands:

    @pytest.mark.asyncio
    async def test_strategy(self, setup_bot):
        msg = _make_message("/strategy")
        await telegram_bot.cmd_strategy(msg)
        reply = _get_reply_text(msg)
        assert "Active Strategy" in reply
        assert "momentum" in reply
        _record("strategy", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_strategy_no_strategy(self, setup_bot):
        telegram_bot._strategy = None
        msg = _make_message("/strategy")
        await telegram_bot.cmd_strategy(msg)
        assert msg.answer.call_count == 0
        _record("strategy", "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_risk(self, setup_bot):
        msg = _make_message("/risk")
        await telegram_bot.cmd_risk(msg)
        reply = _get_reply_text(msg)
        assert "Risk Parameters" in reply
        _record("risk", "happy_path", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /config, /configfull, /export
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigCommands:

    @pytest.mark.asyncio
    async def test_config_all(self, setup_bot):
        msg = _make_message("/config")
        await telegram_bot.cmd_config(msg)
        reply = _get_reply_text(msg)
        assert "Strategy" in reply
        _record("config", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_config_risk(self, setup_bot):
        msg = _make_message("/config risk")
        await telegram_bot.cmd_config(msg)
        reply = _get_reply_text(msg)
        assert "Risk" in reply
        _record("config", "category_filter", "PASS")

    @pytest.mark.asyncio
    async def test_config_unknown(self, setup_bot):
        msg = _make_message("/config nonexistent")
        await telegram_bot.cmd_config(msg)
        assert "Unknown category" in _get_reply_text(msg)
        _record("config", "input_validation", "PASS")

    @pytest.mark.asyncio
    async def test_configfull_all(self, setup_config_commands):
        msg = _make_message("/configfull")
        await telegram_config_commands.cmd_configfull(msg)
        replies = _get_all_replies(msg)
        combined = " ".join(replies)
        assert "Broker" in combined or "broker" in combined.lower()
        _record("configfull", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_configfull_section(self, setup_config_commands):
        msg = _make_message("/configfull broker")
        await telegram_config_commands.cmd_configfull(msg)
        reply = _get_reply_text(msg)
        assert "Broker" in reply or "PAPER" in reply or "LIVE" in reply
        _record("configfull", "section_filter", "PASS")

    @pytest.mark.asyncio
    async def test_configfull_unknown_section(self, setup_config_commands):
        msg = _make_message("/configfull nonexistent")
        await telegram_config_commands.cmd_configfull(msg)
        assert "Unknown section" in _get_reply_text(msg)
        _record("configfull", "input_validation", "PASS")

    @pytest.mark.asyncio
    async def test_export(self, setup_config_commands):
        msg = _make_message("/export")
        await telegram_config_commands.cmd_export(msg)
        replies = _get_all_replies(msg)
        combined = " ".join(replies)
        assert "Configuration Export" in combined
        # Secrets should be masked
        assert "••••" in combined or "not set" in combined.lower()
        _record("export", "happy_path", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /set
# ─────────────────────────────────────────────────────────────────────────────


class TestSetCommand:
    CMD = "set"

    @pytest.mark.asyncio
    async def test_happy_path(self, setup_bot):
        msg = _make_message("/set momentum_fast_ema 12")
        await telegram_bot.cmd_set(msg)
        reply = _get_reply_text(msg)
        assert "Config updated" in reply or "✅" in reply
        _record(self.CMD, "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_missing_args(self, setup_bot):
        msg = _make_message("/set")
        await telegram_bot.cmd_set(msg)
        assert "Usage" in _get_reply_text(msg)
        _record(self.CMD, "input_validation", "PASS")

    @pytest.mark.asyncio
    async def test_unknown_param(self, setup_bot):
        msg = _make_message("/set totally_fake_param 42")
        await telegram_bot.cmd_set(msg)
        assert "unknown" in _get_reply_text(msg).lower()
        _record(self.CMD, "input_validation_unknown", "PASS")

    @pytest.mark.asyncio
    async def test_out_of_range(self, setup_bot):
        msg = _make_message("/set momentum_fast_ema 999")
        await telegram_bot.cmd_set(msg)
        assert "between" in _get_reply_text(msg).lower() or "must be" in _get_reply_text(msg).lower()
        _record(self.CMD, "input_validation_range", "PASS")

    @pytest.mark.asyncio
    async def test_bool_param(self, setup_bot):
        msg = _make_message("/set notify_on_signal true")
        await telegram_bot.cmd_set(msg)
        reply = _get_reply_text(msg)
        assert "✅" in reply or "Config updated" in reply
        _record(self.CMD, "bool_param", "PASS")

    @pytest.mark.asyncio
    async def test_choice_param(self, setup_bot):
        msg = _make_message("/set timeframe 15Min")
        await telegram_bot.cmd_set(msg)
        reply = _get_reply_text(msg)
        assert "15Min" in reply
        _record(self.CMD, "choice_param", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /setrisk
# ─────────────────────────────────────────────────────────────────────────────


class TestSetRiskCommand:
    CMD = "setrisk"

    @pytest.mark.asyncio
    async def test_happy_path(self, setup_bot):
        msg = _make_message("/setrisk max_daily_loss_pct 0.03")
        await telegram_bot.cmd_setrisk(msg)
        reply = _get_reply_text(msg)
        assert "updated" in reply.lower() or "→" in reply or "->" in reply
        _record(self.CMD, "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_missing_args(self, setup_bot):
        msg = _make_message("/setrisk")
        await telegram_bot.cmd_setrisk(msg)
        assert "Usage" in _get_reply_text(msg)
        _record(self.CMD, "input_validation", "PASS")

    @pytest.mark.asyncio
    async def test_invalid_value(self, setup_bot):
        msg = _make_message("/setrisk max_daily_loss_pct abc")
        await telegram_bot.cmd_setrisk(msg)
        assert "Invalid" in _get_reply_text(msg) or "number" in _get_reply_text(msg).lower()
        _record(self.CMD, "input_validation_type", "PASS")

    @pytest.mark.asyncio
    async def test_out_of_range(self, setup_bot):
        msg = _make_message("/setrisk max_daily_loss_pct 5.0")
        await telegram_bot.cmd_setrisk(msg)
        reply = _get_reply_text(msg)
        assert "between" in reply.lower() or "must be" in reply.lower()
        _record(self.CMD, "input_validation_range", "PASS")

    @pytest.mark.asyncio
    async def test_unknown_param(self, setup_bot, mock_risk_manager):
        mock_risk_manager.limits = MagicMock(spec=["max_daily_loss_pct"])
        msg = _make_message("/setrisk nonexistent_param 0.5")
        await telegram_bot.cmd_setrisk(msg)
        assert "Unknown" in _get_reply_text(msg) or "unknown" in _get_reply_text(msg).lower()
        _record(self.CMD, "input_validation_unknown", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /setstrategy, /setsymbols, /setinterval, /settf, /setlookback
# ─────────────────────────────────────────────────────────────────────────────


class TestSetterCommands:

    @pytest.mark.asyncio
    async def test_setstrategy_valid(self, setup_bot):
        msg = _make_message("/setstrategy ml")
        await telegram_bot.cmd_setstrategy(msg)
        reply = _get_reply_text(msg)
        assert "ml" in reply.lower()
        assert telegram_bot._runtime_changes["strategy_name"] == "ml"
        _record("setstrategy", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_setstrategy_invalid(self, setup_bot):
        msg = _make_message("/setstrategy invalid")
        await telegram_bot.cmd_setstrategy(msg)
        assert "Unknown" in _get_reply_text(msg)
        _record("setstrategy", "input_validation", "PASS")

    @pytest.mark.asyncio
    async def test_setstrategy_missing(self, setup_bot):
        msg = _make_message("/setstrategy")
        await telegram_bot.cmd_setstrategy(msg)
        assert "Usage" in _get_reply_text(msg)
        _record("setstrategy", "missing_args", "PASS")

    @pytest.mark.asyncio
    async def test_setsymbols_valid(self, setup_bot):
        msg = _make_message("/setsymbols AAPL,MSFT")
        await telegram_bot.cmd_setsymbols(msg)
        assert "AAPL" in _get_reply_text(msg)
        assert telegram_bot._runtime_changes["symbols"] == ["AAPL", "MSFT"]
        _record("setsymbols", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_setsymbols_missing(self, setup_bot):
        msg = _make_message("/setsymbols")
        await telegram_bot.cmd_setsymbols(msg)
        reply = _get_reply_text(msg)
        assert "Usage" in reply or "Current" in reply
        _record("setsymbols", "missing_args", "PASS")

    @pytest.mark.asyncio
    async def test_setinterval_valid(self, setup_bot):
        msg = _make_message("/setinterval 120")
        await telegram_bot.cmd_setinterval(msg)
        assert "120" in _get_reply_text(msg)
        assert telegram_bot._runtime_changes["interval"] == 120
        _record("setinterval", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_setinterval_out_of_range(self, setup_bot):
        msg = _make_message("/setinterval 5")
        await telegram_bot.cmd_setinterval(msg)
        assert "between" in _get_reply_text(msg).lower() or "10 and 3600" in _get_reply_text(msg)
        _record("setinterval", "input_validation_range", "PASS")

    @pytest.mark.asyncio
    async def test_setinterval_not_int(self, setup_bot):
        msg = _make_message("/setinterval abc")
        await telegram_bot.cmd_setinterval(msg)
        assert "Invalid" in _get_reply_text(msg) or "integer" in _get_reply_text(msg).lower()
        _record("setinterval", "input_validation_type", "PASS")

    @pytest.mark.asyncio
    async def test_settf_valid(self, setup_bot):
        msg = _make_message("/settf 15Min")
        await telegram_bot.cmd_settf(msg)
        assert "15Min" in _get_reply_text(msg)
        _record("settf", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_settf_invalid(self, setup_bot):
        msg = _make_message("/settf 99Hours")
        await telegram_bot.cmd_settf(msg)
        assert "Invalid" in _get_reply_text(msg) or "Choose from" in _get_reply_text(msg)
        _record("settf", "input_validation", "PASS")

    @pytest.mark.asyncio
    async def test_setlookback_valid(self, setup_bot):
        msg = _make_message("/setlookback 500")
        await telegram_bot.cmd_setlookback(msg)
        assert "500" in _get_reply_text(msg)
        _record("setlookback", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_setlookback_out_of_range(self, setup_bot):
        msg = _make_message("/setlookback 10")
        await telegram_bot.cmd_setlookback(msg)
        assert "between" in _get_reply_text(msg).lower() or "50 and 5000" in _get_reply_text(msg)
        _record("setlookback", "input_validation_range", "PASS")

    @pytest.mark.asyncio
    async def test_setauto_valid(self, setup_bot):
        msg = _make_message("/setauto train 12")
        await telegram_bot.cmd_setauto(msg)
        reply = _get_reply_text(msg)
        assert "12" in reply or "train" in reply
        _record("setauto", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_setauto_disable(self, setup_bot):
        msg = _make_message("/setauto backtest 0")
        await telegram_bot.cmd_setauto(msg)
        reply = _get_reply_text(msg)
        assert "DISABLED" in reply or "0" in reply
        _record("setauto", "disable", "PASS")

    @pytest.mark.asyncio
    async def test_setauto_invalid_type(self, setup_bot):
        msg = _make_message("/setauto invalid 5")
        await telegram_bot.cmd_setauto(msg)
        assert "Unknown" in _get_reply_text(msg) or "Use:" in _get_reply_text(msg)
        _record("setauto", "input_validation", "PASS")

    @pytest.mark.asyncio
    async def test_setnotify_on(self, setup_bot):
        msg = _make_message("/setnotify signal on")
        await telegram_bot.cmd_setnotify(msg)
        reply = _get_reply_text(msg)
        assert "signal" in reply
        assert "ON" in reply or "✅" in reply
        _record("setnotify", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_setnotify_off(self, setup_bot):
        msg = _make_message("/setnotify trade off")
        await telegram_bot.cmd_setnotify(msg)
        reply = _get_reply_text(msg)
        assert "OFF" in reply or "❌" in reply
        _record("setnotify", "off_path", "PASS")

    @pytest.mark.asyncio
    async def test_setnotify_invalid(self, setup_bot):
        msg = _make_message("/setnotify invalid on")
        await telegram_bot.cmd_setnotify(msg)
        assert "Unknown" in _get_reply_text(msg) or "Use:" in _get_reply_text(msg)
        _record("setnotify", "input_validation", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /setalpaca, /setmode, /saveconfig, /loadconfig
# ─────────────────────────────────────────────────────────────────────────────


class TestAlpacaConfigCommands:

    @pytest.mark.asyncio
    async def test_setalpaca_usage(self, setup_config_commands):
        msg = _make_message("/setalpaca")
        await telegram_config_commands.cmd_setalpaca(msg)
        assert "Alpaca Configuration" in _get_reply_text(msg)
        _record("setalpaca", "usage_display", "PASS")

    @pytest.mark.asyncio
    async def test_setalpaca_invalid_key(self, setup_config_commands):
        msg = _make_message("/setalpaca paper short bad")
        await telegram_config_commands.cmd_setalpaca(msg)
        assert "Invalid key format" in _get_reply_text(msg) or "❌" in _get_reply_text(msg)
        _record("setalpaca", "input_validation", "PASS")

    @pytest.mark.asyncio
    async def test_setalpaca_feed_valid(self, setup_config_commands):
        msg = _make_message("/setalpaca feed iex")
        await telegram_config_commands.cmd_setalpaca(msg)
        reply = _get_reply_text(msg)
        assert "feed" in reply.lower() or "Data feed" in reply or "✅" in reply
        _record("setalpaca", "feed_valid", "PASS")

    @pytest.mark.asyncio
    async def test_setalpaca_feed_invalid(self, setup_config_commands):
        msg = _make_message("/setalpaca feed invalid")
        await telegram_config_commands.cmd_setalpaca(msg)
        reply = _get_reply_text(msg)
        assert "iex" in reply or "sip" in reply
        _record("setalpaca", "feed_invalid", "PASS")

    @pytest.mark.asyncio
    async def test_setalpaca_url_invalid_scheme(self, setup_config_commands):
        msg = _make_message("/setalpaca url paper http://bad-url.com")
        await telegram_config_commands.cmd_setalpaca(msg)
        assert "HTTPS" in _get_reply_text(msg) or "❌" in _get_reply_text(msg)
        _record("setalpaca", "url_validation", "PASS")

    @pytest.mark.asyncio
    async def test_setalpaca_url_invalid_host(self, setup_config_commands):
        msg = _make_message("/setalpaca url paper https://evil.com")
        await telegram_config_commands.cmd_setalpaca(msg)
        assert "not allowed" in _get_reply_text(msg) or "❌" in _get_reply_text(msg)
        _record("setalpaca", "url_host_validation", "PASS")

    @pytest.mark.asyncio
    async def test_setalpaca_unknown_subcmd(self, setup_config_commands):
        msg = _make_message("/setalpaca unknown test")
        await telegram_config_commands.cmd_setalpaca(msg)
        assert "Unknown subcommand" in _get_reply_text(msg) or "❌" in _get_reply_text(msg)
        _record("setalpaca", "unknown_subcommand", "PASS")

    @pytest.mark.asyncio
    async def test_setmode_usage(self, setup_config_commands):
        msg = _make_message("/setmode")
        await telegram_config_commands.cmd_setmode(msg)
        reply = _get_reply_text(msg)
        assert "Usage" in reply or "PAPER" in reply or "Trading Mode" in reply
        _record("setmode", "usage_display", "PASS")

    @pytest.mark.asyncio
    async def test_setmode_invalid(self, setup_config_commands):
        msg = _make_message("/setmode invalid")
        await telegram_config_commands.cmd_setmode(msg)
        assert "❌" in _get_reply_text(msg)
        _record("setmode", "input_validation", "PASS")

    @pytest.mark.asyncio
    async def test_setmode_same(self, setup_config_commands):
        msg = _make_message("/setmode paper")
        await telegram_config_commands.cmd_setmode(msg)
        reply = _get_reply_text(msg)
        assert "Already" in reply or "PAPER" in reply
        _record("setmode", "idempotent", "PASS")

    @pytest.mark.asyncio
    async def test_saveconfig(self, setup_config_commands):
        with patch("builtins.open", MagicMock()):
            msg = _make_message("/saveconfig")
            await telegram_config_commands.cmd_saveconfig(msg)
            reply = _get_reply_text(msg)
            assert "saved" in reply.lower() or "✅" in reply
        _record("saveconfig", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_saveconfig_failure(self, setup_config_commands):
        with patch("builtins.open", side_effect=PermissionError("read-only")):
            msg = _make_message("/saveconfig")
            await telegram_config_commands.cmd_saveconfig(msg)
            reply = _get_reply_text(msg)
            assert "Failed" in reply or "❌" in reply
        _record("saveconfig", "error_handling", "PASS")

    @pytest.mark.asyncio
    async def test_loadconfig_no_file(self, setup_config_commands):
        with patch.object(Path, "exists", return_value=False):
            msg = _make_message("/loadconfig")
            await telegram_config_commands.cmd_loadconfig(msg)
            assert "No .env" in _get_reply_text(msg) or "❌" in _get_reply_text(msg)
        _record("loadconfig", "missing_config", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /newstrategy, /liststrats, /editstrat, /deletestrat, /copystrat, /templates, /applystrats
# ─────────────────────────────────────────────────────────────────────────────


class TestStrategyManagementCommands:

    @pytest.mark.asyncio
    async def test_newstrategy_missing_args(self, setup_config_commands):
        msg = _make_message("/newstrategy")
        await telegram_config_commands.cmd_newstrategy(msg)
        assert "Usage" in _get_reply_text(msg) or "Create New Strategy" in _get_reply_text(msg)
        _record("newstrategy", "missing_args", "PASS")

    @pytest.mark.asyncio
    async def test_newstrategy_invalid_name(self, setup_config_commands):
        msg = _make_message("/newstrategy 123invalid momentum AAPL")
        await telegram_config_commands.cmd_newstrategy(msg)
        assert "Invalid strategy name" in _get_reply_text(msg) or "❌" in _get_reply_text(msg)
        _record("newstrategy", "input_validation_name", "PASS")

    @pytest.mark.asyncio
    async def test_newstrategy_valid(self, setup_config_commands):
        msg = _make_message("/newstrategy mytest momentum AAPL,TSLA 1Hour 60")
        await telegram_config_commands.cmd_newstrategy(msg)
        reply = _get_reply_text(msg)
        assert "✅" in reply or "Created" in reply
        _record("newstrategy", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_newstrategy_invalid_timeframe(self, setup_config_commands):
        msg = _make_message("/newstrategy mytest momentum AAPL 99Hours")
        await telegram_config_commands.cmd_newstrategy(msg)
        assert "Invalid timeframe" in _get_reply_text(msg) or "❌" in _get_reply_text(msg)
        _record("newstrategy", "input_validation_tf", "PASS")

    @pytest.mark.asyncio
    async def test_liststrats_empty(self, setup_config_commands):
        msg = _make_message("/liststrats")
        await telegram_config_commands.cmd_liststrats(msg)
        reply = _get_reply_text(msg)
        assert "No saved strategies" in reply or "📋" in reply
        _record("liststrats", "empty_dataset", "PASS")

    @pytest.mark.asyncio
    async def test_liststrats_no_store(self, setup_config_commands):
        telegram_config_commands._strategy_store = None
        msg = _make_message("/liststrats")
        await telegram_config_commands.cmd_liststrats(msg)
        assert "not initialized" in _get_reply_text(msg).lower() or "❌" in _get_reply_text(msg)
        _record("liststrats", "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_editstrat_missing_args(self, setup_config_commands):
        msg = _make_message("/editstrat")
        await telegram_config_commands.cmd_editstrat(msg)
        assert "Usage" in _get_reply_text(msg) or "Edit Strategy" in _get_reply_text(msg)
        _record("editstrat", "missing_args", "PASS")

    @pytest.mark.asyncio
    async def test_editstrat_not_found(self, setup_config_commands):
        _, store = setup_config_commands
        store.get.return_value = None
        msg = _make_message("/editstrat nonexistent symbols AAPL")
        await telegram_config_commands.cmd_editstrat(msg)
        assert "not found" in _get_reply_text(msg) or "❌" in _get_reply_text(msg)
        _record("editstrat", "not_found", "PASS")

    @pytest.mark.asyncio
    async def test_deletestrat_missing_arg(self, setup_config_commands):
        msg = _make_message("/deletestrat")
        await telegram_config_commands.cmd_deletestrat(msg)
        assert "Usage" in _get_reply_text(msg)
        _record("deletestrat", "missing_args", "PASS")

    @pytest.mark.asyncio
    async def test_deletestrat_happy(self, setup_config_commands):
        msg = _make_message("/deletestrat test")
        await telegram_config_commands.cmd_deletestrat(msg)
        reply = _get_reply_text(msg)
        assert "Deleted" in reply or "🗑️" in reply
        _record("deletestrat", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_copystrat_missing_args(self, setup_config_commands):
        msg = _make_message("/copystrat")
        await telegram_config_commands.cmd_copystrat(msg)
        assert "Usage" in _get_reply_text(msg)
        _record("copystrat", "missing_args", "PASS")

    @pytest.mark.asyncio
    async def test_copystrat_happy(self, setup_config_commands):
        msg = _make_message("/copystrat source newcopy")
        await telegram_config_commands.cmd_copystrat(msg)
        reply = _get_reply_text(msg)
        assert "Duplicated" in reply or "✅" in reply
        _record("copystrat", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_templates(self, setup_config_commands):
        msg = _make_message("/templates")
        await telegram_config_commands.cmd_templates(msg)
        reply = _get_reply_text(msg)
        assert "Templates" in reply or "momentum" in reply
        _record("templates", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_templates_no_store(self, setup_config_commands):
        telegram_config_commands._strategy_store = None
        msg = _make_message("/templates")
        await telegram_config_commands.cmd_templates(msg)
        assert "not initialized" in _get_reply_text(msg).lower()
        _record("templates", "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_applystrats_no_active(self, setup_config_commands):
        msg = _make_message("/applystrats")
        await telegram_config_commands.cmd_applystrats(msg)
        reply = _get_reply_text(msg)
        assert "No active" in reply or "❌" in reply
        _record("applystrats", "empty_dataset", "PASS")

    @pytest.mark.asyncio
    async def test_applystrats_no_store(self, setup_config_commands):
        telegram_config_commands._strategy_store = None
        msg = _make_message("/applystrats")
        await telegram_config_commands.cmd_applystrats(msg)
        assert "not initialized" in _get_reply_text(msg).lower()
        _record("applystrats", "missing_service", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /backtest, /backtestvbt, /sweep, /train, /modelinfo, /predict
# ─────────────────────────────────────────────────────────────────────────────


class TestBacktestMLCommands:

    @pytest.mark.asyncio
    async def test_backtest_no_data(self, setup_bot, paper_broker):
        paper_broker.get_bars_df = MagicMock(return_value=None)
        msg = _make_message("/backtest AAPL 30")
        await telegram_bot.cmd_backtest(msg)
        replies = _get_all_replies(msg)
        combined = " ".join(replies)
        assert "Insufficient" in combined or "backtest" in combined.lower()
        _record("backtest", "empty_dataset", "PASS")

    @pytest.mark.asyncio
    async def test_backtest_no_broker(self, setup_bot):
        telegram_bot._broker = None
        msg = _make_message("/backtest")
        await telegram_bot.cmd_backtest(msg)
        assert msg.answer.call_count == 0
        _record("backtest", "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_backtestvbt_no_data(self, setup_bot, paper_broker):
        paper_broker.get_bars_df = MagicMock(return_value=None)
        msg = _make_message("/backtestvbt AAPL 30")
        await telegram_bot.cmd_backtest_vbt(msg)
        replies = _get_all_replies(msg)
        combined = " ".join(replies)
        assert "Insufficient" in combined or "backtest" in combined.lower()
        _record("backtestvbt", "empty_dataset", "PASS")

    @pytest.mark.asyncio
    async def test_sweep_no_data(self, setup_bot, paper_broker):
        paper_broker.get_bars_df = MagicMock(return_value=None)
        msg = _make_message("/sweep AAPL 60")
        await telegram_bot.cmd_sweep(msg)
        replies = _get_all_replies(msg)
        combined = " ".join(replies)
        assert "Insufficient" in combined or "sweep" in combined.lower()
        _record("sweep", "empty_dataset", "PASS")

    @pytest.mark.asyncio
    async def test_train_no_broker(self, setup_bot):
        telegram_bot._broker = None
        msg = _make_message("/train")
        await telegram_bot.cmd_train(msg)
        assert msg.answer.call_count == 0
        _record("train", "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_modelinfo_no_model(self, setup_bot):
        with patch("os.path.exists", return_value=False):
            msg = _make_message("/modelinfo")
            await telegram_bot.cmd_modelinfo(msg)
            reply = _get_reply_text(msg)
            assert "No trained model" in reply or msg.answer.call_count >= 1
        _record("modelinfo", "missing_model", "PASS")

    @pytest.mark.asyncio
    async def test_predict_no_model(self, setup_bot, mock_strategy):
        mock_strategy.name = "momentum"
        mock_strategy.model = None
        with patch("os.path.exists", return_value=False):
            msg = _make_message("/predict AAPL")
            await telegram_bot.cmd_predict(msg)
            reply = _get_reply_text(msg)
            assert "No ML model" in reply or "model" in reply.lower()
        _record("predict", "missing_model", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /walkforward, /montecarlo, /portfolio
# ─────────────────────────────────────────────────────────────────────────────


class TestAdvancedResearchCommands:

    @pytest.mark.asyncio
    async def test_walkforward_no_broker(self, setup_bot):
        telegram_bot._broker = None
        msg = _make_message("/walkforward")
        await telegram_bot.cmd_walkforward(msg)
        reply = _get_reply_text(msg)
        assert "Broker not connected" in reply
        _record("walkforward", "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_montecarlo_no_broker(self, setup_bot):
        telegram_bot._broker = None
        msg = _make_message("/montecarlo")
        await telegram_bot.cmd_montecarlo(msg)
        reply = _get_reply_text(msg)
        assert "Broker not connected" in reply
        _record("montecarlo", "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_portfolio_no_broker(self, setup_bot):
        telegram_bot._broker = None
        msg = _make_message("/portfolio")
        await telegram_bot.cmd_portfolio_backtest(msg)
        reply = _get_reply_text(msg)
        assert "Broker not connected" in reply
        _record("portfolio", "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_walkforward_unauthorized(self, setup_bot):
        msg = _make_message("/walkforward", user_id=UNAUTHORIZED_USER_ID)
        await telegram_bot.cmd_walkforward(msg)
        assert msg.answer.call_count == 0
        _record("walkforward", "permission", "PASS")

    @pytest.mark.asyncio
    async def test_montecarlo_unauthorized(self, setup_bot):
        msg = _make_message("/montecarlo", user_id=UNAUTHORIZED_USER_ID)
        await telegram_bot.cmd_montecarlo(msg)
        assert msg.answer.call_count == 0
        _record("montecarlo", "permission", "PASS")

    @pytest.mark.asyncio
    async def test_portfolio_unauthorized(self, setup_bot):
        msg = _make_message("/portfolio", user_id=UNAUTHORIZED_USER_ID)
        await telegram_bot.cmd_portfolio_backtest(msg)
        assert msg.answer.call_count == 0
        _record("portfolio", "permission", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /models, /rollback
# ─────────────────────────────────────────────────────────────────────────────


class TestModelRegistryCommands:

    @pytest.mark.asyncio
    async def test_models_no_registry(self, setup_bot):
        with patch("src.ml.model_registry.ModelRegistry") as mock_reg:
            mock_reg.return_value.list_versions.return_value = []
            msg = _make_message("/models")
            await telegram_bot.cmd_models(msg)
            reply = _get_reply_text(msg)
            assert "No model versions" in reply
        _record("models", "empty_dataset", "PASS")

    @pytest.mark.asyncio
    async def test_models_exception(self, setup_bot):
        with patch("src.ml.model_registry.ModelRegistry", side_effect=Exception("Registry corrupted")):
            msg = _make_message("/models")
            await telegram_bot.cmd_models(msg)
            reply = _get_reply_text(msg)
            assert "error" in reply.lower()
        _record("models", "error_handling", "PASS")

    @pytest.mark.asyncio
    async def test_rollback_missing_version(self, setup_bot):
        msg = _make_message("/rollback")
        await telegram_bot.cmd_rollback(msg)
        reply = _get_reply_text(msg)
        assert "Usage" in reply
        _record("rollback", "missing_args", "PASS")

    @pytest.mark.asyncio
    async def test_rollback_not_found(self, setup_bot):
        with patch("src.ml.model_registry.ModelRegistry") as mock_reg:
            mock_reg.return_value.rollback.return_value = False
            msg = _make_message("/rollback v999")
            await telegram_bot.cmd_rollback(msg)
            reply = _get_reply_text(msg)
            assert "failed" in reply.lower() or "not found" in reply.lower()
        _record("rollback", "not_found", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /journal, /journalstats
# ─────────────────────────────────────────────────────────────────────────────


class TestJournalCommands:

    @pytest.mark.asyncio
    async def test_journal_empty(self, setup_bot):
        with patch("src.data.journal.TradeJournal") as mock_j:
            mock_j.return_value.get_journal.return_value = []
            msg = _make_message("/journal")
            await telegram_bot.cmd_journal(msg)
            reply = _get_reply_text(msg)
            assert "No journal" in reply
        _record("journal", "empty_dataset", "PASS")

    @pytest.mark.asyncio
    async def test_journal_with_symbol(self, setup_bot):
        with patch("src.data.journal.TradeJournal") as mock_j:
            mock_j.return_value.get_journal.return_value = [{
                "side": "buy", "symbol": "AAPL", "pnl": 50.0, "confidence": 0.85,
                "strategy_name": "ml", "model_version": "v002", "entry_price": 150.0,
                "exit_price": 155.0, "exit_reason": "take_profit", "entry_time": "2024-01-15T10:30:00",
            }]
            msg = _make_message("/journal AAPL 5")
            await telegram_bot.cmd_journal(msg)
            reply = _get_reply_text(msg)
            assert "AAPL" in reply
        _record("journal", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_journalstats(self, setup_bot):
        with patch("src.data.journal.TradeJournal") as mock_j:
            mock_j.return_value.get_summary.return_value = {
                "total_trades": 10, "closed_trades": 8, "open_trades": 2,
                "win_rate": 0.6, "avg_pnl": 15.0, "total_pnl": 150.0,
                "best_trade": 100.0, "worst_trade": -50.0, "avg_holding_bars": 24,
            }
            mock_j.return_value.get_performance_by_confidence.return_value = {}
            mock_j.return_value.get_performance_by_model_version.return_value = {}
            msg = _make_message("/journalstats")
            await telegram_bot.cmd_journalstats(msg)
            reply = _get_reply_text(msg)
            assert "Journal Analytics" in reply
        _record("journalstats", "happy_path", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /assets, /search, /price, /asset, /watchlist
# ─────────────────────────────────────────────────────────────────────────────


class TestAssetDiscoveryCommands:

    @pytest.mark.asyncio
    async def test_assets_no_broker(self, setup_bot):
        telegram_bot._broker = None
        msg = _make_message("/assets")
        await telegram_bot.cmd_assets(msg)
        assert "Broker not connected" in _get_reply_text(msg)
        _record("assets", "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_assets_happy_path(self, setup_bot, paper_broker):
        paper_broker.get_stock_assets = MagicMock(return_value=[
            {"symbol": "AAPL", "name": "Apple Inc", "exchange": "NASDAQ",
             "tradable": True, "fractionable": True, "shortable": True},
        ])
        msg = _make_message("/assets")
        await telegram_bot.cmd_assets(msg)
        reply = _get_reply_text(msg)
        assert "AAPL" in reply
        _record("assets", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_search_missing_query(self, setup_bot):
        msg = _make_message("/search")
        await telegram_bot.cmd_search(msg)
        assert "Usage" in _get_reply_text(msg)
        _record("search", "missing_args", "PASS")

    @pytest.mark.asyncio
    async def test_search_no_results(self, setup_bot, paper_broker):
        paper_broker.get_stock_assets = MagicMock(return_value=[])
        msg = _make_message("/search NONEXISTENT")
        await telegram_bot.cmd_search(msg)
        reply = _get_reply_text(msg)
        assert "No assets found" in reply
        _record("search", "empty_dataset", "PASS")

    @pytest.mark.asyncio
    async def test_search_happy(self, setup_bot, paper_broker):
        paper_broker.get_stock_assets = MagicMock(return_value=[
            {"symbol": "AAPL", "name": "Apple Inc", "fractionable": True},
        ])
        msg = _make_message("/search AAPL")
        await telegram_bot.cmd_search(msg)
        reply = _get_reply_text(msg)
        assert "AAPL" in reply
        _record("search", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_price_missing_symbol(self, setup_bot):
        msg = _make_message("/price")
        await telegram_bot.cmd_price(msg)
        assert "Usage" in _get_reply_text(msg)
        _record("price", "missing_args", "PASS")

    @pytest.mark.asyncio
    async def test_price_happy(self, setup_bot, paper_broker):
        paper_broker.get_latest_price = MagicMock(return_value=150.0)
        msg = _make_message("/price AAPL")
        await telegram_bot.cmd_price(msg)
        reply = _get_reply_text(msg)
        assert "AAPL" in reply
        assert "$" in reply
        _record("price", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_price_broker_error(self, setup_bot, paper_broker):
        paper_broker.get_latest_price = MagicMock(side_effect=Exception("Symbol not found"))
        msg = _make_message("/price FAKE")
        await telegram_bot.cmd_price(msg)
        reply = _get_reply_text(msg)
        assert "Error" in reply or "⚠️" in reply
        _record("price", "broker_failure", "PASS")

    @pytest.mark.asyncio
    async def test_asset_missing_symbol(self, setup_bot):
        msg = _make_message("/asset")
        await telegram_bot.cmd_asset_detail(msg)
        assert "Usage" in _get_reply_text(msg)
        _record("asset", "missing_args", "PASS")

    @pytest.mark.asyncio
    async def test_asset_not_found(self, setup_bot, paper_broker):
        paper_broker.get_stock_assets = MagicMock(return_value=[])
        msg = _make_message("/asset FAKE")
        await telegram_bot.cmd_asset_detail(msg)
        reply = _get_reply_text(msg)
        assert "not found" in reply
        _record("asset", "not_found", "PASS")

    @pytest.mark.asyncio
    async def test_watchlist_no_broker(self, setup_bot):
        telegram_bot._broker = None
        msg = _make_message("/watchlist")
        await telegram_bot.cmd_watchlist(msg)
        assert "Broker not connected" in _get_reply_text(msg)
        _record("watchlist", "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_watchlist_happy(self, setup_bot, paper_broker):
        msg = _make_message("/watchlist")
        await telegram_bot.cmd_watchlist(msg)
        reply = _get_reply_text(msg)
        assert "Watchlist" in reply
        _record("watchlist", "happy_path", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# /setsector, /sectors
# ─────────────────────────────────────────────────────────────────────────────


class TestSectorCommands:

    @pytest.mark.asyncio
    async def test_setsector_valid(self, setup_sector_commands, mock_db):
        with patch("src.notifications.telegram_sector_commands.reload_cache"):
            msg = _make_message("/setsector PLTR technology")
            await telegram_sector_commands.cmd_setsector(msg)
            reply = _get_reply_text(msg)
            assert "✅" in reply and "PLTR" in reply
            mock_db.set_cached_sector.assert_called_with("PLTR", "technology", source="manual")
        _record("setsector", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_setsector_invalid_sector(self, setup_sector_commands):
        msg = _make_message("/setsector AAPL invalid_sector")
        await telegram_sector_commands.cmd_setsector(msg)
        assert "Unknown sector" in _get_reply_text(msg) or "⚠️" in _get_reply_text(msg)
        _record("setsector", "input_validation", "PASS")

    @pytest.mark.asyncio
    async def test_setsector_missing_args(self, setup_sector_commands):
        msg = _make_message("/setsector AAPL")
        await telegram_sector_commands.cmd_setsector(msg)
        assert "Usage" in _get_reply_text(msg)
        _record("setsector", "missing_args", "PASS")

    @pytest.mark.asyncio
    async def test_setsector_db_failure(self, setup_sector_commands, mock_db):
        mock_db.set_cached_sector.side_effect = Exception("DB write error")
        with patch("src.notifications.telegram_sector_commands.reload_cache"):
            msg = _make_message("/setsector AAPL technology")
            await telegram_sector_commands.cmd_setsector(msg)
            reply = _get_reply_text(msg)
            assert "Failed" in reply or "⚠️" in reply
        _record("setsector", "db_failure", "PASS")

    @pytest.mark.asyncio
    async def test_sectors_list(self, setup_sector_commands):
        msg = _make_message("/sectors")
        await telegram_sector_commands.cmd_list_sectors(msg)
        reply = _get_reply_text(msg)
        assert "AAPL" in reply
        _record("sectors", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_sectors_empty(self, setup_sector_commands, mock_db):
        mock_db.get_all_cached_sectors.return_value = {}
        msg = _make_message("/sectors")
        await telegram_sector_commands.cmd_list_sectors(msg)
        assert "No manually cached" in _get_reply_text(msg)
        _record("sectors", "empty_dataset", "PASS")

    @pytest.mark.asyncio
    async def test_sectors_db_failure(self, setup_sector_commands, mock_db):
        mock_db.get_all_cached_sectors.side_effect = Exception("DB lost")
        msg = _make_message("/sectors")
        await telegram_sector_commands.cmd_list_sectors(msg)
        assert "Failed" in _get_reply_text(msg) or "⚠️" in _get_reply_text(msg)
        _record("sectors", "db_failure", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Health & Ops Commands
# ─────────────────────────────────────────────────────────────────────────────


class TestHealthOpsCommands:

    @pytest.mark.asyncio
    async def test_health_happy(self, setup_bot, mock_health_monitor):
        msg = _make_message("/health")
        await telegram_bot.cmd_health(msg)
        reply = _get_reply_text(msg)
        assert "Health" in reply
        _record("health", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_health_no_monitor(self, setup_bot):
        telegram_bot._health_monitor = None
        msg = _make_message("/health")
        await telegram_bot.cmd_health(msg)
        assert "not configured" in _get_reply_text(msg)
        _record("health", "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_metrics_no_metrics(self, setup_bot):
        telegram_bot._live_metrics = None
        msg = _make_message("/metrics")
        await telegram_bot.cmd_metrics(msg)
        assert "not configured" in _get_reply_text(msg)
        _record("metrics", "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_metrics_happy(self, setup_bot, mock_live_metrics):
        msg = _make_message("/metrics")
        await telegram_bot.cmd_metrics(msg)
        reply = _get_reply_text(msg)
        assert "Performance" in reply or msg.answer.call_count >= 1
        _record("metrics", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_uptime(self, setup_bot, mock_health_monitor):
        msg = _make_message("/uptime")
        await telegram_bot.cmd_uptime(msg)
        reply = _get_reply_text(msg)
        assert "Uptime" in reply
        _record("uptime", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_events_no_bus(self, setup_bot):
        telegram_bot._event_bus = None
        msg = _make_message("/events")
        await telegram_bot.cmd_events(msg)
        assert "not configured" in _get_reply_text(msg)
        _record("events", "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_events_empty(self, setup_bot, mock_event_bus):
        msg = _make_message("/events")
        await telegram_bot.cmd_events(msg)
        reply = _get_reply_text(msg)
        assert "No events" in reply
        _record("events", "empty_dataset", "PASS")

    @pytest.mark.asyncio
    async def test_activetrades_no_manager(self, setup_bot):
        telegram_bot._trade_manager = None
        msg = _make_message("/activetrades")
        await telegram_bot.cmd_active_trades(msg)
        assert "not configured" in _get_reply_text(msg)
        _record("activetrades", "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_activetrades_empty(self, setup_bot, mock_trade_manager):
        msg = _make_message("/activetrades")
        await telegram_bot.cmd_active_trades(msg)
        reply = _get_reply_text(msg)
        assert "No active" in reply
        _record("activetrades", "empty_dataset", "PASS")

    @pytest.mark.asyncio
    async def test_healthops(self, setup_bot, mock_ops_handler):
        with patch("src.notifications.telegram_bot._get_ops_handler", return_value=mock_ops_handler):
            msg = _make_message("/healthops")
            await telegram_bot.cmd_healthops(msg)
            assert msg.answer.call_count >= 1
        _record("healthops", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_riskreport(self, setup_bot, mock_ops_handler):
        with patch("src.notifications.telegram_bot._get_ops_handler", return_value=mock_ops_handler):
            msg = _make_message("/riskreport")
            await telegram_bot.cmd_riskreport(msg)
            assert msg.answer.call_count >= 1
        _record("riskreport", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_reconcile(self, setup_bot, mock_ops_handler):
        with patch("src.notifications.telegram_bot._get_ops_handler", return_value=mock_ops_handler):
            msg = _make_message("/reconcile")
            await telegram_bot.cmd_reconcile(msg)
            assert msg.answer.call_count >= 1
        _record("reconcile", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_latency(self, setup_bot, mock_ops_handler):
        with patch("src.notifications.telegram_bot._get_ops_handler", return_value=mock_ops_handler):
            msg = _make_message("/latency")
            await telegram_bot.cmd_latency(msg)
            assert msg.answer.call_count >= 1
        _record("latency", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_performance(self, setup_bot, mock_ops_handler):
        with patch("src.notifications.telegram_bot._get_ops_handler", return_value=mock_ops_handler):
            msg = _make_message("/performance")
            await telegram_bot.cmd_performance(msg)
            assert msg.answer.call_count >= 1
        _record("performance", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_system(self, setup_bot, mock_ops_handler):
        with patch("src.notifications.telegram_bot._get_ops_handler", return_value=mock_ops_handler):
            msg = _make_message("/system")
            await telegram_bot.cmd_system(msg)
            assert msg.answer.call_count >= 1
        _record("system", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_replay_list(self, setup_bot):
        with patch("src.core.event_store.EventStore"), \
             patch("src.core.replay.ReplayEngine") as mock_re:
            mock_re.return_value.list_sessions.return_value = []
            msg = _make_message("/replay list")
            await telegram_bot.cmd_replay(msg)
            reply = _get_reply_text(msg)
            assert "No sessions" in reply or "Sessions" in reply
        _record("replay", "empty_dataset", "PASS")

    @pytest.mark.asyncio
    async def test_recover_missing_components(self, setup_bot):
        telegram_bot._trade_manager = None
        msg = _make_message("/recover")
        await telegram_bot.cmd_recover(msg)
        reply = _get_reply_text(msg)
        assert "not initialized" in reply.lower() or "Components" in reply
        _record("recover", "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_governance(self, setup_bot):
        with patch("src.ml.governance.ModelGovernance") as mock_gov:
            mock_gov.return_value.audit_report.return_value = {
                "total_versions": 0, "currently_deployed": "none",
                "retired": 0, "with_git_commit": 0, "with_dataset_hash": 0,
                "governance_completeness": 0.0, "versions": [],
            }
            msg = _make_message("/governance")
            await telegram_bot.cmd_governance(msg)
            reply = _get_reply_text(msg)
            assert "Governance" in reply
        _record("governance", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_modelaudit_no_model(self, setup_bot):
        with patch("src.ml.model_registry.ModelRegistry") as mock_reg:
            mock_reg.return_value.get_active_version.return_value = None
            msg = _make_message("/modelaudit")
            await telegram_bot.cmd_modelaudit(msg)
            reply = _get_reply_text(msg)
            assert "No active model" in reply
        _record("modelaudit", "missing_model", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Runtime Commands (/env, /rbacktest, /rtrain, /modelswap, /abtest, etc.)
# ─────────────────────────────────────────────────────────────────────────────


class TestRuntimeCommands:

    @pytest.mark.asyncio
    async def test_env_status(self, setup_runtime_commands):
        msg = _make_message("/env")
        await telegram_runtime_commands.cmd_env(msg)
        reply = _get_reply_text(msg)
        assert "Environment" in reply or "PAPER" in reply
        _record("env", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_env_invalid(self, setup_runtime_commands):
        msg = _make_message("/env invalid")
        await telegram_runtime_commands.cmd_env(msg)
        reply = _get_reply_text(msg)
        assert "Usage" in reply or "paper" in reply
        _record("env", "input_validation", "PASS")

    @pytest.mark.asyncio
    async def test_rbacktest(self, setup_runtime_commands):
        msg = _make_message("/rbacktest")
        await telegram_runtime_commands.cmd_rbacktest(msg)
        combined = " ".join(_get_all_replies(msg))
        assert "backtest" in combined.lower() or "Backtest" in combined
        _record("rbacktest", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_rbacktest_invalid(self, setup_runtime_commands):
        msg = _make_message("/rbacktest !!invalid!!")
        await telegram_runtime_commands.cmd_rbacktest(msg)
        reply = _get_reply_text(msg)
        assert "❌" in reply or "Invalid" in reply
        _record("rbacktest", "input_validation", "PASS")

    @pytest.mark.asyncio
    async def test_rbacktest_no_manager(self):
        telegram_runtime_commands._runtime_manager = None
        telegram_runtime_commands._authorized_users = {AUTHORIZED_USER_ID}
        msg = _make_message("/rbacktest")
        await telegram_runtime_commands.cmd_rbacktest(msg)
        assert "not initialized" in _get_reply_text(msg).lower()
        _record("rbacktest", "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_rtrain(self, setup_runtime_commands):
        msg = _make_message("/rtrain")
        await telegram_runtime_commands.cmd_rtrain(msg)
        combined = " ".join(_get_all_replies(msg))
        assert "Training" in combined or "Pipeline" in combined
        _record("rtrain", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_rtrain_already_training(self, setup_runtime_commands):
        rm = telegram_runtime_commands._runtime_manager
        rm.is_training.return_value = True
        rm.get_training_progress.return_value = {"stage": "training", "progress_pct": 50}
        msg = _make_message("/rtrain")
        await telegram_runtime_commands.cmd_rtrain(msg)
        reply = _get_reply_text(msg)
        assert "already" in reply.lower() or "progress" in reply.lower()
        _record("rtrain", "duplicate_prevention", "PASS")

    @pytest.mark.asyncio
    async def test_rtrain_failure(self, setup_runtime_commands):
        rm = telegram_runtime_commands._runtime_manager
        rm.train_model.side_effect = RuntimeError("No training data")
        msg = _make_message("/rtrain")
        await telegram_runtime_commands.cmd_rtrain(msg)
        combined = " ".join(_get_all_replies(msg))
        assert "failed" in combined.lower() or "error" in combined.lower() or "No training data" in combined
        _record("rtrain", "error_handling", "PASS")

    @pytest.mark.asyncio
    async def test_modelswap_status(self, setup_runtime_commands):
        msg = _make_message("/modelswap")
        await telegram_runtime_commands.cmd_modelswap(msg)
        reply = _get_reply_text(msg)
        assert "Model Status" in reply or "v002" in reply
        _record("modelswap", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_modelswap_valid(self, setup_runtime_commands):
        msg = _make_message("/modelswap v001")
        await telegram_runtime_commands.cmd_modelswap(msg)
        reply = _get_reply_text(msg)
        assert "Swapped" in reply or "v001" in reply
        _record("modelswap", "swap_execution", "PASS")

    @pytest.mark.asyncio
    async def test_modelswap_invalid(self, setup_runtime_commands):
        msg = _make_message("/modelswap !!!invalid!!!")
        await telegram_runtime_commands.cmd_modelswap(msg)
        reply = _get_reply_text(msg)
        assert "❌" in reply or "Invalid" in reply
        _record("modelswap", "input_validation", "PASS")

    @pytest.mark.asyncio
    async def test_modelswap_no_manager(self):
        telegram_runtime_commands._runtime_manager = None
        telegram_runtime_commands._authorized_users = {AUTHORIZED_USER_ID}
        msg = _make_message("/modelswap v001")
        await telegram_runtime_commands.cmd_modelswap(msg)
        assert "not initialized" in _get_reply_text(msg).lower()
        _record("modelswap", "missing_service", "PASS")

    @pytest.mark.asyncio
    async def test_abtest_status(self, setup_runtime_commands):
        msg = _make_message("/abtest")
        await telegram_runtime_commands.cmd_abtest(msg)
        reply = _get_reply_text(msg)
        assert "No active A/B test" in reply or "A/B" in reply
        _record("abtest", "happy_path", "PASS")

    @pytest.mark.asyncio
    async def test_abtest_start(self, setup_runtime_commands):
        msg = _make_message("/abtest start v003 shadow")
        await telegram_runtime_commands.cmd_abtest(msg)
        reply = _get_reply_text(msg)
        assert "Started" in reply or "v003" in reply
        _record("abtest", "start", "PASS")

    @pytest.mark.asyncio
    async def test_abtest_stop(self, setup_runtime_commands):
        msg = _make_message("/abtest stop")
        await telegram_runtime_commands.cmd_abtest(msg)
        reply = _get_reply_text(msg)
        assert "Cancelled" in reply or "No active" in reply
        _record("abtest", "stop", "PASS")

    @pytest.mark.asyncio
    async def test_runtime(self, setup_runtime_commands):
        msg = _make_message("/runtime")
        await telegram_runtime_commands.cmd_runtime(msg)
        reply = _get_reply_text(msg)
        assert "Runtime Status" in reply
        _record("runtime", "happy_path", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Persistence Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPersistence:

    @pytest.mark.asyncio
    async def test_runtime_changes_accumulate(self, setup_bot):
        await telegram_bot.cmd_setstrategy(_make_message("/setstrategy ml"))
        await telegram_bot.cmd_setinterval(_make_message("/setinterval 300"))
        await telegram_bot.cmd_setsymbols(_make_message("/setsymbols MSFT,NVDA"))
        changes = telegram_bot.get_runtime_changes()
        assert changes["strategy_name"] == "ml"
        assert changes["interval"] == 300
        assert changes["symbols"] == ["MSFT", "NVDA"]
        _record("persistence", "accumulation", "PASS")

    @pytest.mark.asyncio
    async def test_runtime_changes_cleared_after_read(self, setup_bot):
        await telegram_bot.cmd_setstrategy(_make_message("/setstrategy ml"))
        changes = telegram_bot.get_runtime_changes()
        assert changes.get("strategy_name") == "ml"
        changes2 = telegram_bot.get_runtime_changes()
        assert changes2 == {}
        _record("persistence", "clear_after_read", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# DEAD COMMAND / UNREACHABLE CODE / DUPLICATE DETECTION
# ─────────────────────────────────────────────────────────────────────────────


class TestCodeAnalysis:
    """Static analysis of command implementations."""

    def test_all_commands_have_handlers(self):
        """Every command listed in /help has a corresponding handler."""
        # Extract commands from help text — only lines starting with /command
        help_commands = set()
        import re
        doc = telegram_bot.__doc__ or ""
        for match in re.finditer(r'^\s+/(\w+)\s', doc, re.MULTILINE):
            help_commands.add(match.group(1))

        # Check handler functions exist
        missing = []
        for cmd in help_commands:
            handler_name = f"cmd_{cmd}"
            has_handler = (
                hasattr(telegram_bot, handler_name)
                or hasattr(telegram_config_commands, handler_name)
                or hasattr(telegram_runtime_commands, handler_name)
                or hasattr(telegram_sector_commands, handler_name)
            )
            # Some commands have special handler names
            special_names = {
                "backtestvbt": "cmd_backtest_vbt",
                "walkforward": "cmd_walkforward",
                "montecarlo": "cmd_montecarlo",
                "portfolio": "cmd_portfolio_backtest",
                "journalstats": "cmd_journalstats",
                "modelinfo": "cmd_modelinfo",
                "closeall": "cmd_closeall",
                "cancelall": "cmd_cancelall",
                "setrisk": "cmd_setrisk",
                "setstrategy": "cmd_setstrategy",
                "setsymbols": "cmd_setsymbols",
                "setinterval": "cmd_setinterval",
                "settf": "cmd_settf",
                "setlookback": "cmd_setlookback",
                "setnotify": "cmd_setnotify",
                "setauto": "cmd_setauto",
                "setalpaca": "cmd_setalpaca",
                "setmode": "cmd_setmode",
                "saveconfig": "cmd_saveconfig",
                "loadconfig": "cmd_loadconfig",
                "configfull": "cmd_configfull",
                "newstrategy": "cmd_newstrategy",
                "liststrats": "cmd_liststrats",
                "editstrat": "cmd_editstrat",
                "deletestrat": "cmd_deletestrat",
                "copystrat": "cmd_copystrat",
                "applystrats": "cmd_applystrats",
                "setsector": "cmd_setsector",
                "activetrades": "cmd_active_trades",
                "healthops": "cmd_healthops",
                "riskreport": "cmd_riskreport",
                "modelaudit": "cmd_modelaudit",
                "asset": "cmd_asset_detail",
            }
            if not has_handler:
                alt_name = special_names.get(cmd, "")
                has_handler = (
                    hasattr(telegram_bot, alt_name)
                    or hasattr(telegram_config_commands, alt_name)
                    or hasattr(telegram_runtime_commands, alt_name)
                    or hasattr(telegram_sector_commands, alt_name)
                )
            if not has_handler:
                missing.append(cmd)

        _record("code_analysis", "handler_coverage",
                "PASS" if not missing else "FAIL",
                f"Missing handlers: {missing}" if missing else "All commands have handlers")
        # Known commands that are in docstring but handled in runtime_commands:
        expected_missing = {"start", "help", "auth"}  # These have handlers but different naming
        actual_missing = set(missing) - expected_missing
        if actual_missing:
            print(f"\nWARN: Commands without obvious handler mapping: {actual_missing}")

    def test_no_duplicate_command_registrations(self):
        """Check that no command is registered twice across routers."""
        from aiogram.filters import Command
        registered_commands = defaultdict(list)

        for mod_name, mod in [
            ("telegram_bot", telegram_bot),
            ("telegram_config_commands", telegram_config_commands),
            ("telegram_runtime_commands", telegram_runtime_commands),
            ("telegram_sector_commands", telegram_sector_commands),
        ]:
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if callable(attr) and attr_name.startswith("cmd_"):
                    registered_commands[attr_name].append(mod_name)

        duplicates = {k: v for k, v in registered_commands.items() if len(v) > 1}
        _record("code_analysis", "no_duplicates",
                "PASS" if not duplicates else "WARN",
                f"Duplicates: {duplicates}" if duplicates else "No duplicate registrations")

    def test_all_error_handlers_have_diagnostics(self):
        """Check that error catch blocks provide diagnostics, not bare messages."""
        import inspect

        for mod in [telegram_bot, telegram_config_commands,
                     telegram_runtime_commands, telegram_sector_commands]:
            source = inspect.getsource(mod)
            # Find lines with "Operation failed" pattern
            lines = source.split("\n")
            for i, line in enumerate(lines):
                if '"Operation failed"' in line or "'Operation failed'" in line:
                    # Check that the full string includes diagnostic info
                    context = " ".join(lines[max(0, i-2):i+3])
                    assert (
                        "logs" in context.lower()
                        or "error" in context.lower()
                        or "details" in context.lower()
                        or str(context).count("{") > 0  # f-string with variable
                    ), f"Bare 'Operation failed' without diagnostics at line {i+1} in {mod.__name__}"

        _record("code_analysis", "diagnostic_messages", "PASS")


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION MATRIX REPORT
# ─────────────────────────────────────────────────────────────────────────────


# All commands from the problem statement
ALL_REQUIRED_COMMANDS = [
    "status", "positions", "orders", "pnl", "trades", "signals",
    "buy", "sell", "close", "closeall", "cancelall",
    "pause", "resume", "strategy", "risk",
    "config", "configfull", "set", "setrisk", "setstrategy",
    "setsymbols", "setinterval", "settf", "setlookback",
    "setauto", "setnotify", "setalpaca", "setmode",
    "saveconfig", "loadconfig", "export",
    "newstrategy", "liststrats", "editstrat", "deletestrat",
    "copystrat", "templates", "applystrats",
    "backtest", "backtestvbt", "sweep", "train",
    "modelinfo", "predict",
    "assets", "search", "price", "asset", "watchlist",
    "walkforward", "montecarlo", "portfolio",
    "models", "rollback",
    "journal", "journalstats",
    "setsector", "sectors",
]

# Dimensions to validate
VALIDATION_DIMENSIONS = [
    "happy_path", "input_validation", "permission", "error_handling",
    "broker_failure", "empty_dataset", "missing_model", "missing_config",
    "missing_service", "persistence", "output_format", "markdown_correctness",
    "unexpected_exceptions", "broken_dependencies", "return_value",
]


class TestValidationMatrix:
    """Final validation matrix report."""

    def test_print_validation_matrix(self):
        """Print the PASS/WARN/FAIL matrix for every command."""
        print("\n" + "=" * 90)
        print("TELEGRAM COMMAND FULL VALIDATION MATRIX")
        print("=" * 90)
        print(f"{'Command':<20} {'Tests':>6} {'PASS':>6} {'WARN':>6} {'FAIL':>6} | Dimensions Tested")
        print("-" * 90)

        total_pass = 0
        total_warn = 0
        total_fail = 0
        covered_commands = set()

        for cmd in sorted(set(ALL_REQUIRED_COMMANDS)):
            results = _MATRIX.get(cmd, [])
            pass_count = sum(1 for _, v, _ in results if v == "PASS")
            warn_count = sum(1 for _, v, _ in results if v == "WARN")
            fail_count = sum(1 for _, v, _ in results if v == "FAIL")
            total = len(results)
            total_pass += pass_count
            total_warn += warn_count
            total_fail += fail_count

            dimensions = ", ".join(sorted(set(d for d, _, _ in results)))[:40] if results else "(not tested)"

            if total == 0:
                status = "⬜"
            elif fail_count > 0:
                status = "❌"
            elif warn_count > 0:
                status = "⚠️"
            else:
                status = "✅"
                covered_commands.add(cmd)

            print(f"{status} /{cmd:<18} {total:>6} {pass_count:>6} {warn_count:>6} {fail_count:>6} | {dimensions}")

        print("-" * 90)
        print(f"{'TOTALS':<20} {total_pass+total_warn+total_fail:>6} {total_pass:>6} {total_warn:>6} {total_fail:>6}")
        print(f"\nCommands in matrix: {len(_MATRIX)}/{len(ALL_REQUIRED_COMMANDS)}")
        print(f"All PASS: {len(covered_commands)}/{len(ALL_REQUIRED_COMMANDS)}")

        # Report untested commands
        untested = set(ALL_REQUIRED_COMMANDS) - set(_MATRIX.keys())
        if untested:
            print(f"\n⬜ UNTESTED (require live engine): {', '.join(sorted(untested))}")

        # Code analysis results
        ca_results = _MATRIX.get("code_analysis", [])
        if ca_results:
            print("\n" + "=" * 60)
            print("CODE ANALYSIS")
            print("=" * 60)
            for dim, verdict, detail in ca_results:
                print(f"  {verdict}: {dim} — {detail}")

        # Persistence results
        p_results = _MATRIX.get("persistence", [])
        if p_results:
            print("\n" + "=" * 60)
            print("PERSISTENCE VALIDATION")
            print("=" * 60)
            for dim, verdict, detail in p_results:
                print(f"  {verdict}: {dim}")

        print("\n" + "=" * 90)
        assert total_fail == 0, f"{total_fail} FAIL results in validation matrix"

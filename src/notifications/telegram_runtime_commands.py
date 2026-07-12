"""
Telegram Runtime Commands Router -- Hot-swap, backtest, training, model management.

Provides commands for all 7 runtime capabilities without restart:
    /env          - Show or switch environment (paper/live)
    /rbacktest    - Run concurrent multi-strategy backtest
    /rtrain       - Trigger end-to-end ML training pipeline
    /modelswap    - Hot-swap active ML model version
    /abtest       - Start/stop/status A/B test
    /runtime      - Show comprehensive runtime capabilities status
    /backtests    - List recent backtest runs
    /trainhistory - Show training pipeline history

These commands require RuntimeManager to be injected via set_runtime_components().
"""

import threading
import re
from datetime import datetime

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.enums import ParseMode

from src.utils.logger import get_logger

logger = get_logger(__name__)

runtime_router = Router()

# Injected components
_runtime_manager = None
_authorized_users: set[int] = set()
_broker_lock = threading.Lock()
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9/]{1,10}$")
_STRATEGY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_MODEL_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def set_runtime_components(
    runtime_manager,
    authorized_users: set[int] = None,
):
    """Inject runtime manager into this router."""
    global _runtime_manager, _authorized_users
    _runtime_manager = runtime_manager
    if authorized_users:
        _authorized_users = authorized_users


def _is_authorized(message: Message) -> bool:
    user_id = message.from_user.id if message.from_user else None
    if not _authorized_users:
        logger.warning("telegram.runtime.unauthorized", user_id=user_id, reason="no_authorized_users")
        return False
    is_allowed = bool(user_id and user_id in _authorized_users)
    if not is_allowed:
        logger.warning("telegram.runtime.unauthorized", user_id=user_id, reason="user_not_allowlisted")
    return is_allowed


def _actor(message: Message) -> str:
    return f"telegram:{message.from_user.id}" if message.from_user else "telegram:unknown"


def _normalize_symbols(raw_symbols: list[str]) -> list[str]:
    if not raw_symbols:
        raise ValueError("At least one symbol is required.")
    if len(raw_symbols) > 100:
        raise ValueError("No more than 100 symbols are allowed.")

    normalized = []
    for raw in raw_symbols:
        symbol = (raw or "").strip().upper()
        if not _SYMBOL_PATTERN.match(symbol):
            raise ValueError(
                f"Invalid symbol '{raw}'. Use uppercase letters, digits, and / (max 10 chars)."
            )
        normalized.append(symbol)
    return normalized


def _normalize_strategy_names(raw_strategies: list[str]) -> list[str]:
    if not raw_strategies:
        raise ValueError("At least one strategy is required.")
    if len(raw_strategies) > 20:
        raise ValueError("No more than 20 strategies are allowed.")

    normalized = []
    for raw in raw_strategies:
        name = (raw or "").strip().lower()
        if not _STRATEGY_PATTERN.match(name):
            raise ValueError(
                f"Invalid strategy '{raw}'. Use lowercase letters, digits, underscores (max 32 chars)."
            )
        normalized.append(name)
    return normalized


def _normalize_model_version(version: str) -> str:
    normalized = (version or "").strip()
    if not _MODEL_VERSION_PATTERN.match(normalized):
        raise ValueError(
            "Invalid model version. Use letters, digits, dot, underscore, or hyphen (max 64 chars)."
        )
    return normalized


# ──────────────────────────────────────────────────────────────────────────
# /env - Show or switch environment
# ──────────────────────────────────────────────────────────────────────────

@runtime_router.message(Command("env"))
async def cmd_env(message: Message):
    """Show current environment or switch.
    Usage:
        /env          - Show current mode
        /env paper    - Switch to paper
        /env live     - Switch to live
    """
    if not _is_authorized(message):
        return
    if not _runtime_manager:
        await message.answer("Runtime manager not initialized.")
        return

    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        # Show current status
        mode = _runtime_manager.current_mode
        is_paper = _runtime_manager.is_paper
        icon = "📝" if is_paper else "🔴"
        text = (
            f"<b>{icon} Environment Status</b>\n"
            f"{'=' * 28}\n"
            f"Mode: <code>{mode.upper()}</code>\n"
            f"Type: {'Paper (simulated)' if is_paper else 'LIVE (real money)'}\n"
            f"\n<i>Switch with:</i>\n"
            f"  /env paper\n"
            f"  /env live"
        )
        await message.answer(text, parse_mode=ParseMode.HTML)
        return

    target = args[1].strip().lower()
    if target not in ("paper", "live"):
        await message.answer("Usage: /env paper  or  /env live")
        return

    # Confirmation for live mode
    if target == "live":
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="YES - Switch to LIVE", callback_data="env_switch|live"),
                InlineKeyboardButton(text="Cancel", callback_data="env_switch|cancel"),
            ]
        ])
        await message.answer(
            "<b>WARNING: Switch to LIVE mode?</b>\n"
            "Real money will be used for trading.\n"
            "Confirm below:",
            reply_markup=kb,
            parse_mode=ParseMode.HTML,
        )
        return

    # Paper switch - no confirmation needed
    try:
        result = _runtime_manager.switch_environment(target, reason="telegram")
        text = (
            f"<b>Environment Switched</b>\n"
            f"Mode: <code>{target.upper()}</code>\n"
            f"Duration: {result.get('duration_ms', 0):.0f}ms"
        )
        await message.answer(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Switch failed: {e}")


@runtime_router.callback_query(F.data.startswith("env_switch|"))
async def callback_env_switch(callback: CallbackQuery):
    if callback.from_user.id not in _authorized_users:
        await callback.answer("Unauthorized", show_alert=True)
        return

    action = callback.data.split("|")[1]
    if action == "cancel":
        await callback.message.edit_text("Environment switch cancelled.")
        await callback.answer()
        return

    try:
        result = _runtime_manager.switch_environment(action, reason="telegram")
        await callback.message.edit_text(
            f"<b>Environment Switched to {action.upper()}</b>\n"
            f"Duration: {result.get('duration_ms', 0):.0f}ms",
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        await callback.message.edit_text(f"Switch failed: {e}")
    await callback.answer()


# ──────────────────────────────────────────────────────────────────────────
# /rbacktest - Run concurrent multi-strategy backtest
# ──────────────────────────────────────────────────────────────────────────

@runtime_router.message(Command("rbacktest"))
async def cmd_rbacktest(message: Message):
    """Run multi-strategy backtest concurrently.
    Usage:
        /rbacktest                          - All strategies, default symbols
        /rbacktest momentum,ml              - Specific strategies
        /rbacktest momentum,ml AAPL,TSLA    - Strategies + symbols
    """
    if not _is_authorized(message):
        return
    if not _runtime_manager:
        await message.answer("Runtime manager not initialized.")
        return

    actor = _actor(message)
    parts = message.text.split(maxsplit=2)
    strategies = ["momentum", "mean_reversion", "ml"]
    symbols = None

    if len(parts) >= 2:
        strategies = [s.strip() for s in parts[1].split(",") if s.strip()]
    if len(parts) >= 3:
        symbols = [s.strip() for s in parts[2].split(",") if s.strip()]
    try:
        strategies = _normalize_strategy_names(strategies)
        if symbols is not None:
            symbols = _normalize_symbols(symbols)
    except ValueError as e:
        await message.answer(f"❌ {e}")
        return

    logger.info(
        "telegram.runtime.rbacktest.requested",
        actor=actor,
        strategies=strategies,
        symbols=symbols,
    )

    await message.answer(
        f"<b>Starting concurrent backtest...</b>\n"
        f"Strategies: {', '.join(strategies)}\n"
        f"Symbols: {', '.join(symbols) if symbols else 'default'}\n"
        f"<i>Running in background...</i>",
        parse_mode=ParseMode.HTML,
    )

    try:
        result = _runtime_manager.run_backtest(
            strategy_names=strategies,
            symbols=symbols,
            blocking=False,
        )
        logger.info(
            "telegram.runtime.rbacktest.started",
            actor=actor,
            run_id=result.get("run_id"),
            strategies=strategies,
            symbols=symbols,
        )
        await message.answer(
            f"<b>Backtest Launched</b>\n"
            f"Run ID: <code>{result['run_id'][:12]}</code>\n"
            f"Status: {result['status']}\n"
            f"\nCheck progress: /backtests",
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        logger.error("telegram.runtime.rbacktest.failed", actor=actor, error=str(e))
        await message.answer(f"Backtest failed: {e}")


# ──────────────────────────────────────────────────────────────────────────
# /backtests - List recent backtest runs
# ──────────────────────────────────────────────────────────────────────────

@runtime_router.message(Command("backtests"))
async def cmd_backtests(message: Message):
    """List recent backtest runs and their results.
    Usage: /backtests
    """
    if not _is_authorized(message):
        return
    if not _runtime_manager:
        await message.answer("Runtime manager not initialized.")
        return

    try:
        runs = _runtime_manager.list_backtests(limit=5)
        if not runs:
            await message.answer("No backtest runs yet. Use /rbacktest to start one.")
            return

        lines = ["<b>Recent Backtests</b>", f"{'=' * 28}", ""]
        for run in runs:
            status_icon = {"completed": "OK", "running": "...", "failed": "X"}.get(
                run.get("status", ""), "?"
            )
            lines.append(
                f"[{status_icon}] <code>{run.get('run_id', '?')[:10]}</code> "
                f"| {run.get('status', '?')} "
                f"| {run.get('strategies', 0)} strategies"
            )

        await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Error listing backtests: {e}")


# ──────────────────────────────────────────────────────────────────────────
# /rtrain - Trigger ML training pipeline
# ──────────────────────────────────────────────────────────────────────────

@runtime_router.message(Command("rtrain"))
async def cmd_rtrain(message: Message):
    """Trigger end-to-end ML training pipeline.
    Usage:
        /rtrain                     - Train with default symbols
        /rtrain AAPL,TSLA,BTC/USD   - Train with specific symbols
    """
    if not _is_authorized(message):
        return
    if not _runtime_manager:
        await message.answer("Runtime manager not initialized.")
        return

    actor = _actor(message)
    if _runtime_manager.is_training():
        progress = _runtime_manager.get_training_progress()
        await message.answer(
            f"<b>Training already in progress</b>\n"
            f"Stage: {progress.get('stage', '?')}\n"
            f"Progress: {progress.get('progress_pct', 0):.0f}%",
            parse_mode=ParseMode.HTML,
        )
        return

    parts = message.text.split(maxsplit=1)
    symbols = None
    if len(parts) >= 2:
        symbols = [s.strip() for s in parts[1].split(",") if s.strip()]
    try:
        if symbols is not None:
            symbols = _normalize_symbols(symbols)
    except ValueError as e:
        await message.answer(f"❌ {e}")
        return

    logger.info("telegram.runtime.rtrain.requested", actor=actor, symbols=symbols)

    await message.answer(
        f"<b>ML Training Pipeline Started</b>\n"
        f"Symbols: {', '.join(symbols) if symbols else 'default'}\n"
        f"<i>Running: data -> features -> train -> validate -> deploy</i>\n"
        f"\nCheck progress: /trainhistory",
        parse_mode=ParseMode.HTML,
    )

    try:
        result = _runtime_manager.train_model(
            symbols=symbols,
            trigger="telegram",
        )
        logger.info(
            "telegram.runtime.rtrain.started",
            actor=actor,
            pipeline_id=result.get("pipeline_id"),
            symbols=result.get("symbols"),
        )
        await message.answer(
            f"Pipeline ID: <code>{result['pipeline_id'][:12]}</code>\n"
            f"Status: {result['status']}",
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        logger.error("telegram.runtime.rtrain.failed", actor=actor, error=str(e))
        await message.answer(f"Training failed to start: {e}")


# ──────────────────────────────────────────────────────────────────────────
# /trainhistory - Show training history
# ──────────────────────────────────────────────────────────────────────────

@runtime_router.message(Command("trainhistory"))
async def cmd_trainhistory(message: Message):
    """Show training pipeline history and current progress.
    Usage: /trainhistory
    """
    if not _is_authorized(message):
        return
    if not _runtime_manager:
        await message.answer("Runtime manager not initialized.")
        return

    try:
        # Current progress
        progress = _runtime_manager.get_training_progress()
        lines = ["<b>ML Training Status</b>", f"{'=' * 28}", ""]

        if _runtime_manager.is_training():
            lines.append(
                f"<b>ACTIVE:</b> {progress.get('stage', '?')} "
                f"({progress.get('progress_pct', 0):.0f}%)"
            )
            lines.append("")

        # History
        history = _runtime_manager.get_training_history(limit=5)
        if history:
            lines.append("<b>Recent Runs:</b>")
            for h in history:
                status_icon = {"completed": "OK", "failed": "X"}.get(h.get("status", ""), "?")
                lines.append(
                    f"  [{status_icon}] {h.get('trigger', '?')} "
                    f"| {h.get('symbols_count', 0)} symbols "
                    f"| {h.get('duration_s', 0):.0f}s"
                )
        else:
            lines.append("No training history yet.")

        await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Error: {e}")


# ──────────────────────────────────────────────────────────────────────────
# /modelswap - Hot-swap ML model version
# ──────────────────────────────────────────────────────────────────────────

@runtime_router.message(Command("modelswap"))
async def cmd_modelswap(message: Message):
    """Hot-swap active ML model to a specific version.
    Usage:
        /modelswap          - Show current model + available versions
        /modelswap v003     - Swap to version v003
    """
    if not _is_authorized(message):
        return
    if not _runtime_manager:
        await message.answer("Runtime manager not initialized.")
        return

    actor = _actor(message)
    parts = message.text.split(maxsplit=1)

    if len(parts) < 2:
        # Show current model status
        try:
            status = _runtime_manager.get_model_status()
            versions = _runtime_manager.list_model_versions()
            lines = [
                "<b>ML Model Status</b>",
                f"{'=' * 28}",
                f"Active: <code>{status.get('version', 'none')}</code>",
                f"Predictions: <code>{status.get('prediction_count', 0)}</code>",
                f"Avg Latency: <code>{status.get('avg_latency_ms', 0):.1f}ms</code>",
                "",
                "<b>Available Versions:</b>",
            ]
            for v in versions[-5:]:
                active = " (ACTIVE)" if v.get("active") else ""
                lines.append(f"  <code>{v.get('version', '?')}</code>{active}")

            lines.append("\n<i>Swap: /modelswap VERSION</i>")
            await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
        except Exception as e:
            await message.answer(f"Error: {e}")
        return

    try:
        version = _normalize_model_version(parts[1])
    except ValueError as e:
        await message.answer(f"❌ {e}")
        return
    try:
        logger.info("telegram.runtime.modelswap.requested", actor=actor, version=version)
        result = _runtime_manager.swap_model(version)
        logger.info("telegram.runtime.modelswap.completed", actor=actor, version=version)
        await message.answer(
            f"<b>Model Swapped</b>\n"
            f"Version: <code>{version}</code>\n"
            f"Status: {result.get('status', 'ok')}\n"
            f"<i>All predictions now use the new model.</i>",
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        logger.error("telegram.runtime.modelswap.failed", actor=actor, version=version, error=str(e))
        await message.answer(f"Model swap failed: {e}")


# ──────────────────────────────────────────────────────────────────────────
# /abtest - A/B testing management
# ──────────────────────────────────────────────────────────────────────────

@runtime_router.message(Command("abtest"))
async def cmd_abtest(message: Message):
    """Manage A/B tests for ML model versions.
    Usage:
        /abtest                     - Show active test status
        /abtest start v004          - Start shadow test with v004
        /abtest start v004 split    - Start split-traffic test
        /abtest stop                - Cancel active test
    """
    if not _is_authorized(message):
        return
    if not _runtime_manager:
        await message.answer("Runtime manager not initialized.")
        return

    actor = _actor(message)
    parts = message.text.split()

    if len(parts) < 2:
        # Show status
        try:
            status = _runtime_manager.get_ab_test_status()
            if not status:
                await message.answer(
                    "<b>No active A/B test</b>\n"
                    "\n<i>Start one:</i>\n"
                    "  /abtest start VERSION [mode]\n"
                    "  Modes: shadow, split, interleaved",
                    parse_mode=ParseMode.HTML,
                )
                return

            lines = [
                "<b>A/B Test Active</b>",
                f"{'=' * 28}",
                f"Challenger: <code>{status.get('challenger', '?')}</code>",
                f"Mode: <code>{status.get('mode', '?')}</code>",
                f"Trades: <code>{status.get('total_trades', 0)}</code>",
                f"Min Required: <code>{status.get('min_trades', 30)}</code>",
                "",
            ]

            champion = status.get("champion_stats", {})
            challenger = status.get("challenger_stats", {})
            if champion:
                lines.append(f"<b>Champion:</b> {champion.get('win_rate', 0):.1%} win rate")
            if challenger:
                lines.append(f"<b>Challenger:</b> {challenger.get('win_rate', 0):.1%} win rate")

            conclusion = status.get("conclusion")
            if conclusion:
                lines.append(f"\n<b>Result:</b> {conclusion}")

            await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
        except Exception as e:
            await message.answer(f"Error: {e}")
        return

    action = parts[1].lower()

    if action == "stop":
        try:
            result = _runtime_manager.cancel_ab_test(reason="telegram")
            if result:
                await message.answer(
                    f"<b>A/B Test Cancelled</b>\nReason: manual (telegram)",
                    parse_mode=ParseMode.HTML,
                )
            else:
                await message.answer("No active A/B test to cancel.")
        except Exception as e:
            await message.answer(f"Error: {e}")
        return

    if action == "start":
        if len(parts) < 3:
            await message.answer("Usage: /abtest start VERSION [mode]\nModes: shadow, split, interleaved")
            return

        try:
            version = _normalize_model_version(parts[2])
        except ValueError as e:
            await message.answer(f"❌ {e}")
            return
        mode = (parts[3] if len(parts) > 3 else "shadow").lower()

        if mode not in ("shadow", "split", "interleaved"):
            await message.answer("Invalid mode. Use: shadow, split, or interleaved")
            return

        try:
            logger.info(
                "telegram.runtime.abtest.start_requested",
                actor=actor,
                version=version,
                mode=mode,
            )
            result = _runtime_manager.start_ab_test(
                challenger_version=version,
                mode=mode,
            )
            logger.info(
                "telegram.runtime.abtest.started",
                actor=actor,
                version=version,
                mode=mode,
                test_id=result.get("test_id"),
            )
            await message.answer(
                f"<b>A/B Test Started</b>\n"
                f"Challenger: <code>{version}</code>\n"
                f"Mode: <code>{mode}</code>\n"
                f"Test ID: <code>{result['test_id'][:12]}</code>\n"
                f"\nCheck progress: /abtest",
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.error(
                "telegram.runtime.abtest.start_failed",
                actor=actor,
                version=version,
                mode=mode,
                error=str(e),
            )
            await message.answer(f"A/B test start failed: {e}")
        return

    await message.answer("Usage: /abtest [start VERSION [mode] | stop]")


# ──────────────────────────────────────────────────────────────────────────
# /runtime - Comprehensive runtime status
# ──────────────────────────────────────────────────────────────────────────

@runtime_router.message(Command("runtime"))
async def cmd_runtime(message: Message):
    """Show comprehensive runtime capabilities status.
    Usage: /runtime
    """
    if not _is_authorized(message):
        return
    if not _runtime_manager:
        await message.answer("Runtime manager not initialized.")
        return

    try:
        status = _runtime_manager.get_status()
        env = status.get("environment", {})
        model = status.get("model", {})
        training = status.get("training", {})
        ab = status.get("ab_test")
        backtests = status.get("backtests", [])

        mode = env.get("mode", "?")
        mode_icon = "📝" if mode == "paper" else "🔴"

        lines = [
            f"<b>{mode_icon} Runtime Status</b>",
            f"{'=' * 28}",
            "",
            f"<b>Environment:</b> <code>{mode.upper()}</code> ({env.get('state', '?')})",
            f"<b>Model:</b> <code>{model.get('version', 'none')}</code> "
            f"({model.get('prediction_count', 0)} predictions)",
        ]

        if training and training.get("stage"):
            lines.append(
                f"<b>Training:</b> {training['stage']} "
                f"({training.get('progress_pct', 0):.0f}%)"
            )
        else:
            lines.append("<b>Training:</b> idle")

        if ab:
            lines.append(
                f"<b>A/B Test:</b> {ab.get('mode', '?')} "
                f"({ab.get('total_trades', 0)} trades)"
            )
        else:
            lines.append("<b>A/B Test:</b> none")

        running_bt = [b for b in backtests if b.get("status") == "running"]
        lines.append(f"<b>Backtests:</b> {len(running_bt)} running, {len(backtests)} total")

        lines.extend([
            "",
            "<b>Capabilities:</b>",
            "  /env - Switch paper/live",
            "  /rbacktest - Run multi-strategy backtest",
            "  /rtrain - Train ML model",
            "  /modelswap - Hot-swap model",
            "  /abtest - A/B test models",
        ])

        await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"Error: {e}")

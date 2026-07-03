"""
Alpaca Algo Trader — Main Entry Point

A fully robust and scalable algorithmic trading bot with AI/ML integration.
Supports both Paper and Live trading via Alpaca Markets API.

Usage:
    python main.py                     # Run with default strategy (paper mode)
    python main.py --live              # Run in LIVE mode (real money!)
    python main.py --strategy ml       # Use ML strategy
    python main.py --backtest          # Backtest strategy on historical data
    python main.py --train             # Train/retrain ML model
    python main.py --status            # Show account status
    python main.py --dry-run           # Generate signals without executing
"""

import argparse
import sys
import time
import signal
import threading
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from config.settings import settings, TradingMode
from src.utils.logger import setup_logging, get_logger
from src.broker.alpaca_client import AlpacaBroker
from src.risk.manager import RiskManager, RiskLimits
from src.strategy.momentum import MomentumStrategy
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.ml_strategy import MLStrategy
from src.execution.engine import ExecutionEngine
from src.data.store import DatabaseManager
from src.notifications.alerts import NotificationManager
from src.monitoring.health import HealthMonitor
from src.monitoring.metrics import LiveMetrics


# ──────────────────────────────────────────────────────────────────────────
# Globals
# ──────────────────────────────────────────────────────────────────────────

_shutdown_event = threading.Event()
logger = None


def _send_telegram_async(coro):
    """Run an async telegram coroutine from sync context safely."""
    import asyncio
    loop = None
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(coro)
    except Exception:
        pass
    finally:
        if loop:
            loop.close()


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    _shutdown_event.set()
    if logger:
        logger.info("shutdown.requested")


signal.signal(signal.SIGINT, signal_handler)
if hasattr(signal, 'SIGBREAK'):
    signal.signal(signal.SIGBREAK, signal_handler)


# ──────────────────────────────────────────────────────────────────────────
# Strategy Factory
# ──────────────────────────────────────────────────────────────────────────

def create_strategy(name: str, symbols: list[str], timeframe: str, lookback: int):
    """Factory to create strategy instances by name, using settings for params."""
    from src.strategy.composable import momentum_preset, mean_reversion_preset

    strategies = {
        "momentum": lambda: MomentumStrategy(
            symbols=symbols, timeframe=timeframe, lookback=lookback,
            fast_ema=settings.momentum_fast_ema,
            slow_ema=settings.momentum_slow_ema,
            rsi_period=settings.momentum_rsi_period,
            rsi_oversold=settings.momentum_rsi_oversold,
            rsi_overbought=settings.momentum_rsi_overbought,
            atr_period=settings.momentum_atr_period,
            atr_sl_multiplier=settings.momentum_atr_sl_mult,
            atr_tp_multiplier=settings.momentum_atr_tp_mult,
        ),
        "mean_reversion": lambda: MeanReversionStrategy(
            symbols=symbols, timeframe=timeframe, lookback=lookback,
        ),
        "ml": lambda: MLStrategy(
            symbols=symbols, timeframe=timeframe, lookback=max(lookback, 500),
            model_path=settings.ml_model_path,
            min_confidence=settings.ml_min_confidence,
        ),
        "composable": lambda: momentum_preset(
            symbols=symbols, timeframe=timeframe, lookback=lookback,
        ),
        "composable_mr": lambda: mean_reversion_preset(
            symbols=symbols, timeframe=timeframe, lookback=lookback,
        ),
    }

    if name not in strategies:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(strategies.keys())}")

    return strategies[name]()


# ──────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────

def cmd_status(broker: AlpacaBroker):
    """Show account status and positions."""
    account = broker.get_account()
    positions = broker.get_positions()

    print(f"\n{'='*50}")
    print(f"  ACCOUNT STATUS ({'PAPER' if broker.paper else '[!] LIVE'})")
    print(f"{'='*50}")
    print(f"  Equity:        ${account['equity']:>12,.2f}")
    print(f"  Cash:          ${account['cash']:>12,.2f}")
    print(f"  Buying Power:  ${account['buying_power']:>12,.2f}")
    print(f"  Portfolio Val:  ${account['portfolio_value']:>12,.2f}")
    print(f"  Day Trades:    {account['day_trade_count']}")
    print(f"  PDT Flag:      {'Yes [!]' if account['pattern_day_trader'] else 'No'}")
    print(f"{'-'*50}")

    if positions:
        print(f"  OPEN POSITIONS ({len(positions)})")
        print(f"{'-'*50}")
        for p in positions:
            pnl_mark = "+" if p['unrealized_pl'] >= 0 else "-"
            print(f"  [{pnl_mark}] {p['symbol']:8s} | {p['qty']:>8.4f} @ ${p['avg_entry_price']:.2f} "
                  f"| PnL: ${p['unrealized_pl']:>8.2f} ({p['unrealized_plpc']:.1%})")
    else:
        print("  No open positions")
    print(f"{'='*50}\n")


def cmd_backtest(broker: AlpacaBroker, strategy_name: str, symbols: list[str], timeframe: str, lookback: int):
    """Run backtest on historical data using custom engine."""
    from backtesting.backtest import Backtester

    strategy = create_strategy(strategy_name, symbols, timeframe, lookback)

    print(f"\n[*] Running backtest: {strategy.name}")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Timeframe: {timeframe}, Lookback: {lookback} bars\n")

    for symbol in symbols:
        print(f"  Fetching data for {symbol}...")
        df = broker.get_bars_df(symbol, timeframe, limit=min(lookback * 3, 1000))

        if df is None or len(df) < 100:
            print(f"  [!] Insufficient data for {symbol} (got {len(df) if df is not None else 0} bars)")
            continue

        bt = Backtester(strategy=strategy, initial_capital=10000.0)
        result = bt.run(df, symbol)
        print(result.summary())


def cmd_backtest_bt(broker: AlpacaBroker, strategy_name: str, symbols: list[str], timeframe: str, lookback: int):
    """Run backtest using Backtrader framework with full analyzers."""
    from backtesting.bt_adapter import (
        run_backtrader_backtest, BTMomentumStrategy, BTMeanReversionStrategy, compare_strategies
    )

    bt_strategies = {
        "momentum": BTMomentumStrategy,
        "mean_reversion": BTMeanReversionStrategy,
    }

    strat_class = bt_strategies.get(strategy_name, BTMomentumStrategy)

    print(f"\n[*] Running Backtrader backtest: {strategy_name}")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Timeframe: {timeframe}\n")

    for symbol in symbols:
        print(f"  Fetching data for {symbol}...")
        df = broker.get_bars_df(symbol, timeframe, limit=min(lookback * 3, 1000))

        if df is None or len(df) < 100:
            print(f"  [!] Insufficient data for {symbol}")
            continue

        metrics = run_backtrader_backtest(df, strategy_class=strat_class, initial_cash=settings.backtest_initial_cash)

        print(f"\n  {'='*50}")
        print(f"  BACKTRADER RESULTS: {strategy_name} on {symbol}")
        print(f"  {'='*50}")
        print(f"  Total Return:  {metrics['total_return']:.2%}")
        print(f"  Sharpe Ratio:  {metrics['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown:  {metrics['max_drawdown']:.2%}")
        print(f"  Total Trades:  {metrics['total_trades']}")
        print(f"  Win Rate:      {metrics['win_rate']:.1%}")
        print(f"  SQN:           {metrics['sqn']:.2f}")
        print(f"  Final Value:   ${metrics['final_value']:,.2f}")
        print(f"  {'='*50}\n")


def cmd_backtest_vbt(broker: AlpacaBroker, symbols: list[str], timeframe: str, lookback: int):
    """Run VectorBT vectorized backtest with parameter sweep."""
    from backtesting.vbt_adapter import vectorbt_momentum_backtest, vectorbt_parameter_sweep

    print(f"\n[*] Running VectorBT vectorized backtest")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Timeframe: {timeframe}\n")

    for symbol in symbols:
        print(f"  Fetching data for {symbol}...")
        df = broker.get_bars_df(symbol, timeframe, limit=min(lookback * 3, 1000))

        if df is None or len(df) < 100:
            print(f"  [!] Insufficient data for {symbol}")
            continue

        metrics = vectorbt_momentum_backtest(df, initial_cash=settings.backtest_initial_cash)

        print(f"\n  {'='*50}")
        print(f"  VECTORBT RESULTS: {symbol}")
        print(f"  {'='*50}")
        print(f"  Total Return:  {metrics['total_return']:.2%}")
        print(f"  Sharpe Ratio:  {metrics['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown:  {metrics['max_drawdown']:.2%}")
        print(f"  Total Trades:  {metrics['total_trades']}")
        print(f"  Win Rate:      {metrics['win_rate']:.1%}")
        print(f"  {'='*50}\n")

    # Parameter sweep on first symbol with enough data
    for symbol in symbols:
        df = broker.get_bars_df(symbol, timeframe, limit=min(lookback * 3, 1000))
        if df is not None and len(df) >= 100:
            print(f"\n  Running parameter sweep on {symbol}...")
            try:
                sweep_df = vectorbt_parameter_sweep(df)
                print(f"\n  Top 5 parameter combinations:")
                print(sweep_df.sort_values('total_return', ascending=False).head(5).to_string())
            except Exception as e:
                print(f"  Parameter sweep failed: {e}")
            break


def cmd_train(broker: AlpacaBroker, symbols: list[str], timeframe: str):
    """Train/retrain the ML model."""
    strategy = MLStrategy(symbols=symbols, timeframe=timeframe, lookback=500)

    print(f"\n[*] Training ML model...")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Timeframe: {timeframe}\n")

    training_data = {}
    for symbol in symbols:
        print(f"  Fetching {symbol}...")
        df = broker.get_bars_df(symbol, timeframe, limit=1000)
        if df is not None and len(df) >= 200:
            training_data[symbol] = df
            print(f"    + {len(df)} bars")
        else:
            print(f"    - Insufficient data")

    if training_data:
        strategy.train(training_data)
        print(f"\n[OK] Model trained and saved to {settings.ml_model_path}")
    else:
        print("\n[!] No sufficient data for training")


def _train_ml_model(broker: AlpacaBroker, symbols: list[str], timeframe: str):
    """Silent ML training (called from main loop / Telegram trigger)."""
    strategy = MLStrategy(symbols=symbols, timeframe=timeframe, lookback=500)
    training_data = {}
    for symbol in symbols:
        df = broker.get_bars_df(symbol, timeframe, limit=1000)
        if df is not None and len(df) >= 200:
            training_data[symbol] = df
    if training_data:
        strategy.train(training_data)
        logger.info("ml.training_complete", symbols=len(training_data))
    else:
        logger.warning("ml.training_no_data")


def cmd_run(
    broker: AlpacaBroker, strategy_name: str, symbols: list[str],
    timeframe: str, lookback: int, interval: int, dry_run: bool,
    notifier: NotificationManager, enable_telegram: bool = False,
    enable_streaming: bool = False
):
    """Main trading loop."""

    strategy = create_strategy(strategy_name, symbols, timeframe, lookback)
    risk = RiskManager(RiskLimits(
        max_position_size_pct=settings.max_position_size_pct,
        max_daily_loss_pct=settings.max_daily_loss_pct,
        max_portfolio_exposure=settings.max_portfolio_exposure,
        max_single_stock_pct=settings.max_single_stock_pct,
        max_leverage=settings.max_leverage,
        max_open_positions=settings.max_open_positions,
        max_orders_per_day=settings.max_orders_per_day,
        max_correlated_positions=settings.max_correlated_positions,
        default_stop_loss_pct=settings.default_stop_loss_pct,
        default_take_profit_pct=settings.default_take_profit_pct,
    ))
    db = DatabaseManager(settings.database_url)

    # Initialize event-driven infrastructure
    from src.core.events import EventBus
    from src.core.state_machine import TradeManager
    from src.core.subscribers import setup_default_subscribers

    event_bus = EventBus()
    trade_manager = TradeManager()

    # Execution simulator (opt-in via settings or default to realistic)
    execution_simulator = None
    if getattr(settings, 'enable_execution_simulator', True):
        from src.execution.simulator import ExecutionSimulator
        sim_preset = getattr(settings, 'execution_simulator_preset', 'realistic')
        if sim_preset == 'conservative':
            execution_simulator = ExecutionSimulator.conservative()
        elif sim_preset == 'ideal':
            execution_simulator = ExecutionSimulator.ideal()
        else:
            execution_simulator = ExecutionSimulator.realistic()
        logger.info("execution_simulator.enabled", preset=sim_preset)

    engine = ExecutionEngine(
        broker=broker,
        risk_manager=risk,
        db=db,
        dry_run=dry_run,
        event_bus=event_bus,
        trade_manager=trade_manager,
        execution_simulator=execution_simulator,
    )

    # Register default event subscribers (audit log, journal, metrics, notifications)
    def _notification_sender(message: str):
        """Send notification via Telegram/Discord if available."""
        try:
            notifier._send_telegram(message)
        except Exception:
            pass

    subscribers = setup_default_subscribers(
        bus=event_bus,
        notification_send_func=_notification_sender if notifier.telegram_token else None,
    )

    # Initialize monitoring components
    health_monitor = HealthMonitor()
    live_metrics = LiveMetrics()

    # Initialize risk manager
    account = broker.get_account()
    risk.reset_daily(account['equity'])

    # Start Telegram bot if enabled (runs in background thread)
    telegram_bot = None
    _telegram_thread = None
    if enable_telegram and settings.telegram_bot_token:
        from src.notifications.telegram_bot import (
            TelegramBotManager, set_components, get_runtime_changes
        )

        # Parse chat ID — skip if placeholder or non-numeric (auto-detect on first message)
        chat_ids = []
        if settings.telegram_chat_id and settings.telegram_chat_id.isdigit():
            chat_ids = [int(settings.telegram_chat_id)]
        set_components(broker, engine, risk, strategy, db=db,
                      authorized_chat_ids=chat_ids,
                      health_monitor=health_monitor,
                      live_metrics=live_metrics,
                      event_bus=event_bus,
                      trade_manager=trade_manager)
        telegram_bot = TelegramBotManager(
            token=settings.telegram_bot_token,
            authorized_chat_ids=chat_ids,
        )

        def _run_telegram_bot(bot):
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(bot.dp.start_polling(bot.bot))

        _telegram_thread = threading.Thread(target=_run_telegram_bot, args=(telegram_bot,), daemon=True)
        _telegram_thread.start()
        logger.info("telegram.bot_started")

    mode = "DRY RUN" if dry_run else ("PAPER" if broker.paper else "[LIVE]")
    logger.info("bot.started", mode=mode, strategy=strategy_name, symbols=len(symbols))
    notifier.notify_startup(mode, strategy_name, symbols)

    # Start WebSocket streaming if enabled
    _stream_data = {}  # Shared dict: symbol -> latest bar data
    if enable_streaming:
        import asyncio as _asyncio

        def _bar_handler(bar):
            """Store incoming bar data from WebSocket."""
            symbol = bar.symbol
            _stream_data[symbol] = {
                "timestamp": bar.timestamp,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }

        def _run_stream():
            loop = _asyncio.new_event_loop()
            _asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(broker.stream_bars(symbols, _bar_handler))
            except Exception as e:
                logger.error("stream.error", error=str(e))

        _stream_thread = threading.Thread(target=_run_stream, daemon=True)
        _stream_thread.start()
        logger.info("stream.started", symbols=len(symbols))

    print(f"\n{'='*60}")
    print(f"  ALGO TRADER RUNNING - {mode}")
    print(f"  Strategy: {strategy_name} | Symbols: {len(symbols)} | Interval: {interval}s")
    if enable_streaming:
        print(f"  WebSocket streaming: ACTIVE")
    print(f"  Ctrl+C to stop gracefully")
    if telegram_bot:
        print(f"  Telegram bot active for remote control")
    print(f"  Auto-backtest: every {settings.auto_backtest_interval_hours}h" if settings.auto_backtest_interval_hours > 0 else "  Auto-backtest: disabled")
    print(f"  Auto-train: every {settings.auto_train_interval_hours}h" if settings.auto_train_interval_hours > 0 else "  Auto-train: disabled")
    print(f"  Auto-sweep: every {settings.auto_sweep_interval_hours}h" if settings.auto_sweep_interval_hours > 0 else "  Auto-sweep: disabled")
    print(f"{'='*60}\n")

    # Thread-safe broker access for scheduled background jobs
    _broker_lock = threading.Lock()

    # Runtime state for scheduled jobs to signal the main loop
    _runtime_lock = threading.Lock()
    _runtime_changes = {"trigger_train": False}

    cycle_count = 0
    consecutive_errors = 0

    # ──────────────────────────────────────────────────────────────────────
    # Automated Scheduler — periodic backtests, training, sweeps, summary
    # ──────────────────────────────────────────────────────────────────────
    _scheduler = None
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger
        from apscheduler.executors.pool import ThreadPoolExecutor as APThreadPoolExecutor

        _auto_executor = APThreadPoolExecutor(max_workers=3)

        def _send_telegram(text: str):
            """Helper: send message via Telegram bot (thread-safe)."""
            if telegram_bot:
                import asyncio
                try:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(telegram_bot.send_alert(text))
                    loop.close()
                except Exception:
                    pass

        def _send_daily_summary():
            try:
                summary = risk.get_daily_summary()
                notifier.notify_daily_summary(summary)
                text = (
                    f"<b>📊 Daily Summary</b>\n"
                    f"{'=' * 25}\n"
                    f"Trades: {summary.get('trades', 0)}\n"
                    f"Win Rate: {summary.get('win_rate', '0%')}\n"
                    f"PnL: ${summary.get('daily_pnl', 0):.2f}\n"
                    f"Return: {summary.get('daily_return', '0%')}\n"
                )
                _send_telegram(text)
                logger.info("scheduler.daily_summary_sent")
            except Exception as e:
                logger.error("scheduler.summary_error", error=str(e))

        def _auto_backtest():
            """Run automated backtests on all symbols and report to Telegram."""
            try:
                bt_symbols = settings.auto_backtest_symbols.split(",") if settings.auto_backtest_symbols else symbols
                bt_symbols = [s.strip() for s in bt_symbols if s.strip()]

                from backtesting.vbt_adapter import vectorbt_momentum_backtest

                results_text = [f"<b>🔄 Auto-Backtest Results</b>\n{'=' * 30}\n"]
                for sym in bt_symbols:
                    try:
                        with _broker_lock:
                            df = broker.get_bars_df(sym, timeframe, limit=500)
                        if df is None or len(df) < 50:
                            continue
                        metrics = vectorbt_momentum_backtest(df, initial_cash=settings.backtest_initial_cash)
                        emoji = "✅" if metrics['total_return'] > 0 else "❌"
                        results_text.append(
                            f"{emoji} <b>{sym}</b>: {metrics['total_return']:.2%} "
                            f"| Sharpe: {metrics['sharpe_ratio']:.2f} "
                            f"| DD: {metrics['max_drawdown']:.1%} "
                            f"| Trades: {metrics['total_trades']}"
                        )
                    except Exception as e:
                        results_text.append(f"⚠️ {sym}: error - {str(e)[:50]}")

                if len(results_text) > 1:
                    _send_telegram("\n".join(results_text))
                logger.info("scheduler.auto_backtest_complete", symbols=len(bt_symbols))
            except Exception as e:
                logger.error("scheduler.auto_backtest_error", error=str(e))
                _send_telegram(f"<b>⚠️ Auto-Backtest Failed</b>\n{str(e)[:200]}")

        def _auto_train():
            """Run automated ML training and report to Telegram."""
            try:
                _send_telegram("🧠 <b>Auto ML Training Started...</b>")
                train_symbols = symbols[:10]
                bars = settings.auto_train_bars

                ml_strategy = MLStrategy(
                    symbols=train_symbols,
                    timeframe=timeframe,
                    lookback=500,
                    model_path=settings.ml_model_path,
                    min_confidence=settings.ml_min_confidence,
                )

                training_data = {}
                for sym in train_symbols:
                    try:
                        with _broker_lock:
                            df = broker.get_bars_df(sym, timeframe, limit=bars)
                        if df is not None and len(df) >= 200:
                            training_data[sym] = df
                    except Exception:
                        continue

                if not training_data:
                    _send_telegram("⚠️ <b>Auto Training:</b> No sufficient data")
                    return

                ml_strategy.train(training_data)

                # If current strategy is ML, trigger reload
                with _runtime_lock:
                    _runtime_changes["trigger_train"] = True

                _send_telegram(
                    f"✅ <b>ML Model Retrained</b>\n"
                    f"Symbols: {len(training_data)}/{len(train_symbols)}\n"
                    f"Bars/symbol: {bars}\n"
                    f"Model saved. Active strategy will reload."
                )
                logger.info("scheduler.auto_train_complete", symbols=len(training_data))
            except Exception as e:
                logger.error("scheduler.auto_train_error", error=str(e))
                _send_telegram(f"<b>⚠️ Auto Training Failed</b>\n{str(e)[:200]}")

        def _auto_sweep():
            """Run parameter sweep and report optimal params to Telegram."""
            try:
                from backtesting.vbt_adapter import vectorbt_parameter_sweep

                sweep_symbols = symbols[:3]  # Top 3 symbols for sweep
                results_text = [f"<b>🔬 Auto Parameter Sweep</b>\n{'=' * 30}\n"]

                for sym in sweep_symbols:
                    try:
                        with _broker_lock:
                            df = broker.get_bars_df(sym, timeframe, limit=700)
                        if df is None or len(df) < 100:
                            continue

                        sweep_df = vectorbt_parameter_sweep(df)
                        if sweep_df is not None and len(sweep_df) > 0:
                            best = sweep_df.sort_values('total_return', ascending=False).iloc[0]
                            results_text.append(
                                f"<b>{sym}</b>: Best EMA({int(best['fast_window'])}/{int(best['slow_window'])}) "
                                f"→ {best['total_return']:.2%} return"
                            )
                    except Exception as e:
                        results_text.append(f"⚠️ {sym}: {str(e)[:40]}")

                if len(results_text) > 1:
                    _send_telegram("\n".join(results_text))
                logger.info("scheduler.auto_sweep_complete", symbols=len(sweep_symbols))
            except Exception as e:
                logger.error("scheduler.auto_sweep_error", error=str(e))

        _scheduler = BackgroundScheduler(executors={'default': _auto_executor})

        # Daily summary at market close
        _scheduler.add_job(
            _send_daily_summary,
            CronTrigger(hour=16, minute=5, timezone="US/Eastern"),
            id="daily_summary",
        )

        # Automated backtest
        if settings.auto_backtest_interval_hours > 0:
            _scheduler.add_job(
                _auto_backtest,
                IntervalTrigger(hours=settings.auto_backtest_interval_hours),
                id="auto_backtest",
                next_run_time=datetime.now(),  # Run immediately on start
            )

        # Automated ML training
        if settings.auto_train_interval_hours > 0:
            _scheduler.add_job(
                _auto_train,
                IntervalTrigger(hours=settings.auto_train_interval_hours),
                id="auto_train",
            )

        # Automated parameter sweep
        if settings.auto_sweep_interval_hours > 0:
            _scheduler.add_job(
                _auto_sweep,
                IntervalTrigger(hours=settings.auto_sweep_interval_hours),
                id="auto_sweep",
            )

        _scheduler.start()

        active_jobs = [j.id for j in _scheduler.get_jobs()]
        logger.info("scheduler.started", jobs=active_jobs)

        # Pass scheduler to Telegram bot for /health command
        if telegram_bot:
            from src.notifications.telegram_bot import set_components as _set_components_update
            _set_components_update(broker, engine, risk, strategy, db=db,
                                  authorized_chat_ids=chat_ids,
                                  health_monitor=health_monitor,
                                  live_metrics=live_metrics,
                                  scheduler=_scheduler,
                                  event_bus=event_bus,
                                  trade_manager=trade_manager)
    except ImportError:
        logger.debug("scheduler.apscheduler_not_available")

    # Cache crypto-only strategy to avoid re-creating each cycle
    crypto_only = [s for s in symbols if "/" in s]
    crypto_strategy = create_strategy(strategy_name, crypto_only, timeframe, lookback) if crypto_only else None

    while not _shutdown_event.is_set():
        cycle_count += 1
        cycle_start = time.time()

        # Check if paused via Telegram
        if telegram_bot and telegram_bot.is_paused():
            logger.debug("cycle.paused_via_telegram")
            time.sleep(5)
            continue

        # Process runtime changes from Telegram
        if telegram_bot:
            changes = get_runtime_changes()
            if changes.get("strategy_name"):
                new_name = changes["strategy_name"]
                logger.info("runtime.strategy_change", new=new_name)
                strategy = create_strategy(new_name, strategy.symbols, strategy.timeframe, strategy.lookback)
                strategy_name = new_name
                # Rebuild crypto strategy too
                crypto_only = [s for s in strategy.symbols if "/" in s]
                crypto_strategy = create_strategy(new_name, crypto_only, timeframe, lookback) if crypto_only else None

            if changes.get("symbols"):
                new_symbols = changes["symbols"]
                logger.info("runtime.symbols_change", symbols=new_symbols)
                symbols = new_symbols
                strategy = create_strategy(strategy_name, symbols, timeframe, lookback)
                crypto_only = [s for s in symbols if "/" in s]
                crypto_strategy = create_strategy(strategy_name, crypto_only, timeframe, lookback) if crypto_only else None

            if changes.get("interval"):
                interval = changes["interval"]
                logger.info("runtime.interval_change", interval=interval)

            if changes.get("timeframe"):
                timeframe = changes["timeframe"]
                logger.info("runtime.timeframe_change", timeframe=timeframe)
                strategy = create_strategy(strategy_name, symbols, timeframe, lookback)
                crypto_only = [s for s in symbols if "/" in s]
                crypto_strategy = create_strategy(strategy_name, crypto_only, timeframe, lookback) if crypto_only else None

            if changes.get("lookback"):
                lookback = changes["lookback"]
                logger.info("runtime.lookback_change", lookback=lookback)
                strategy = create_strategy(strategy_name, symbols, timeframe, lookback)
                crypto_only = [s for s in symbols if "/" in s]
                crypto_strategy = create_strategy(strategy_name, crypto_only, timeframe, lookback) if crypto_only else None

            if changes.get("config_updates"):
                # Strategy params changed — rebuild strategy with new settings
                logger.info("runtime.config_update", params=list(changes["config_updates"].keys()))
                strategy = create_strategy(strategy_name, symbols, timeframe, lookback)
                crypto_only = [s for s in symbols if "/" in s]
                crypto_strategy = create_strategy(strategy_name, crypto_only, timeframe, lookback) if crypto_only else None

            if changes.get("trigger_train") and strategy_name == "ml":
                logger.info("runtime.ml_retrain_triggered")
                try:
                    _train_ml_model(broker, symbols, timeframe)
                    strategy = create_strategy("ml", symbols, timeframe, lookback)
                except Exception as e:
                    logger.error("runtime.train_error", error=str(e))

        # Process runtime changes from scheduler (auto_train signals reload)
        with _runtime_lock:
            if _runtime_changes.get("trigger_train") and strategy_name == "ml":
                _runtime_changes["trigger_train"] = False
                logger.info("runtime.ml_reload_from_scheduler")
                try:
                    strategy = create_strategy("ml", symbols, timeframe, lookback)
                except Exception as e:
                    logger.error("runtime.ml_reload_error", error=str(e))

        # Auto-retrain ML model if needed (check every 10 cycles)
        if strategy_name == "ml" and cycle_count % 10 == 0:
            if hasattr(strategy, 'needs_retraining') and strategy.needs_retraining():
                logger.info("ml.auto_retrain_start")
                try:
                    _train_ml_model(broker, symbols, timeframe)
                    strategy = create_strategy("ml", symbols, timeframe, lookback)
                    logger.info("ml.auto_retrain_complete")
                except Exception as e:
                    logger.error("ml.auto_retrain_error", error=str(e))

        try:
            # Record heartbeat for health monitoring
            health_monitor.heartbeat("engine")

            # Check risk halt and notify
            can_trade, halt_reason = risk.can_trade()
            if not can_trade:
                if telegram_bot and halt_reason:
                    import asyncio
                    try:
                        loop = asyncio.new_event_loop()
                        loop.run_until_complete(telegram_bot.notify_risk_halt(halt_reason))
                        loop.close()
                    except Exception:
                        pass
                logger.warning("engine.halted", reason=halt_reason)
                _shutdown_event.wait(timeout=60)
                continue

            # For stocks: check market hours
            stock_symbols = [s for s in symbols if "/" not in s]
            if stock_symbols and not broker.is_market_open():
                next_open = broker.next_market_open()
                logger.info("market.closed", next_open=str(next_open))
                if crypto_strategy:
                    with _broker_lock:
                        results = engine.run_cycle(crypto_strategy)
                else:
                    results = []
            else:
                with _broker_lock:
                    results = engine.run_cycle(strategy)

            consecutive_errors = 0

            if results:
                logger.info("cycle.complete", cycle=cycle_count, trades=len(results))
                for r in results:
                    if not r.get('dry_run'):
                        notifier.notify_trade(r)
                        # Also notify via Telegram bot
                        if telegram_bot:
                            _send_telegram_async(telegram_bot.notify_trade(r))
            else:
                logger.debug("cycle.no_signals", cycle=cycle_count)

        except KeyboardInterrupt:
            break
        except Exception as e:
            consecutive_errors += 1
            logger.error("cycle.error", error=str(e), consecutive=consecutive_errors)
            notifier.notify_error(str(e), f"Cycle {cycle_count}")

            if consecutive_errors >= settings.max_consecutive_errors:
                logger.warning("bot.too_many_errors", msg=f"Pausing for {settings.error_cooldown_seconds}s")
                _shutdown_event.wait(timeout=settings.error_cooldown_seconds)
                consecutive_errors = 0
                continue

        # Sleep until next cycle
        elapsed = time.time() - cycle_start
        sleep_time = max(1, interval - elapsed)
        logger.debug("cycle.sleeping", seconds=int(sleep_time))
        _shutdown_event.wait(timeout=sleep_time)

    # Shutdown
    logger.info("bot.stopped", cycles=cycle_count)
    if _scheduler:
        _scheduler.shutdown(wait=False)
    notifier.notify_daily_summary(risk.get_daily_summary())
    if telegram_bot:
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(telegram_bot.stop())
        except Exception:
            pass
    print(f"\n[OK] Bot stopped after {cycle_count} cycles. All state saved.")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    global logger

    parser = argparse.ArgumentParser(
        description="Alpaca Algo Trader - AI/ML Powered Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Paper trade with momentum strategy
  python main.py --strategy ml            # Use ML strategy
  python main.py --live                   # LIVE trading (real money!)
  python main.py --backtest               # Backtest on historical data
  python main.py --train                  # Train ML model
  python main.py --status                 # Show account info
  python main.py --dry-run                # Signals only, no orders
  python main.py --symbols AAPL,TSLA,BTC/USD --interval 300
        """
    )

    parser.add_argument("--live", action="store_true", help="[!] Enable LIVE trading (real money)")
    parser.add_argument("--strategy", "-s", default=settings.active_strategy,
                       choices=["momentum", "mean_reversion", "ml"],
                       help="Trading strategy to use")
    parser.add_argument("--symbols", default=settings.trading_symbols,
                       help="Comma-separated symbols (e.g., AAPL,TSLA,BTC/USD)")
    parser.add_argument("--timeframe", "-tf", default=settings.timeframe,
                       choices=["1Min", "5Min", "15Min", "30Min", "1Hour", "4Hour", "1Day"],
                       help="Candle timeframe")
    parser.add_argument("--lookback", type=int, default=settings.lookback_bars,
                       help="Number of historical bars to analyze")
    parser.add_argument("--interval", "-i", type=int, default=settings.trading_interval,
                       help="Seconds between trading cycles")
    parser.add_argument("--dry-run", action="store_true", help="Generate signals without placing orders")
    parser.add_argument("--backtest", action="store_true", help="Run custom backtest engine")
    parser.add_argument("--backtest-bt", action="store_true", help="Run Backtrader backtest")
    parser.add_argument("--backtest-vbt", action="store_true", help="Run VectorBT vectorized backtest")
    parser.add_argument("--stream", action="store_true", help="Use WebSocket streaming for real-time data")
    parser.add_argument("--train", action="store_true", help="Train/retrain ML model")
    parser.add_argument("--status", action="store_true", help="Show account status and exit")
    parser.add_argument("--telegram", action="store_true", help="Enable Telegram bot for interactive control")

    args = parser.parse_args()

    # Setup logging
    setup_logging(settings.log_level, settings.log_file)
    logger = get_logger("main")

    # Determine trading mode
    if args.live:
        if not settings.alpaca_live_api_key:
            print("[!] LIVE API key not configured. Set ALPACA_LIVE_API_KEY in .env")
            sys.exit(1)
        # Safety confirmation
        confirm = input("\n[!] LIVE TRADING MODE - Real money will be used!\n   Type 'YES' to confirm: ")
        if confirm != "YES":
            print("Cancelled.")
            sys.exit(0)
        mode = TradingMode.LIVE
    else:
        mode = TradingMode.PAPER

    # Initialize broker
    api_key = settings.alpaca_live_api_key if mode == TradingMode.LIVE else settings.alpaca_paper_api_key
    secret = settings.alpaca_live_secret_key if mode == TradingMode.LIVE else settings.alpaca_paper_secret_key

    if not api_key or not secret or api_key.startswith("your_"):
        print("[!] API keys not configured. Copy .env.example to .env and add your Alpaca keys.")
        print("   Get free keys at: https://app.alpaca.markets/signup")
        sys.exit(1)

    broker = AlpacaBroker(
        api_key=api_key,
        secret_key=secret,
        base_url=settings.alpaca_live_base_url if mode == TradingMode.LIVE else settings.alpaca_paper_base_url,
        data_feed=settings.alpaca_data_feed,
        paper=(mode == TradingMode.PAPER),
    )

    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    # Notifications
    notifier = NotificationManager(
        telegram_token=settings.telegram_bot_token,
        telegram_chat_id=settings.telegram_chat_id,
        discord_webhook=settings.discord_webhook_url,
        notify_trades=settings.notify_on_trade,
        notify_errors=settings.notify_on_error,
    )

    # Dispatch command
    if args.status:
        cmd_status(broker)
    elif args.backtest:
        cmd_backtest(broker, args.strategy, symbols, args.timeframe, args.lookback)
    elif getattr(args, 'backtest_bt', False):
        cmd_backtest_bt(broker, args.strategy, symbols, args.timeframe, args.lookback)
    elif getattr(args, 'backtest_vbt', False):
        cmd_backtest_vbt(broker, symbols, args.timeframe, args.lookback)
    elif args.train:
        cmd_train(broker, symbols, args.timeframe)
    else:
        cmd_run(broker, args.strategy, symbols, args.timeframe,
                args.lookback, args.interval, args.dry_run, notifier,
                enable_telegram=args.telegram,
                enable_streaming=getattr(args, 'stream', False))


if __name__ == "__main__":
    main()

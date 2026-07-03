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


# ──────────────────────────────────────────────────────────────────────────
# Globals
# ──────────────────────────────────────────────────────────────────────────

_shutdown_requested = False
logger = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    if logger:
        logger.info("shutdown.requested")


signal.signal(signal.SIGINT, signal_handler)


# ──────────────────────────────────────────────────────────────────────────
# Strategy Factory
# ──────────────────────────────────────────────────────────────────────────

def create_strategy(name: str, symbols: list[str], timeframe: str, lookback: int):
    """Factory to create strategy instances by name."""
    strategies = {
        "momentum": lambda: MomentumStrategy(
            symbols=symbols, timeframe=timeframe, lookback=lookback
        ),
        "mean_reversion": lambda: MeanReversionStrategy(
            symbols=symbols, timeframe=timeframe, lookback=lookback
        ),
        "ml": lambda: MLStrategy(
            symbols=symbols, timeframe=timeframe, lookback=max(lookback, 500),
            model_path=settings.ml_model_path,
            min_confidence=settings.ml_min_confidence,
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

        metrics = run_backtrader_backtest(df, strategy_class=strat_class, initial_cash=10000.0)

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


def cmd_run(
    broker: AlpacaBroker, strategy_name: str, symbols: list[str],
    timeframe: str, lookback: int, interval: int, dry_run: bool,
    notifier: NotificationManager, enable_telegram: bool = False
):
    """Main trading loop."""
    global _shutdown_requested

    strategy = create_strategy(strategy_name, symbols, timeframe, lookback)
    risk = RiskManager(RiskLimits(
        max_position_size_pct=settings.max_position_size_pct,
        max_daily_loss_pct=settings.max_daily_loss_pct,
        max_portfolio_exposure=settings.max_portfolio_exposure,
        max_single_stock_pct=settings.max_single_stock_pct,
        max_leverage=settings.max_leverage,
        default_stop_loss_pct=settings.default_stop_loss_pct,
        default_take_profit_pct=settings.default_take_profit_pct,
    ))
    db = DatabaseManager(settings.database_url)
    engine = ExecutionEngine(broker=broker, risk_manager=risk, db=db, dry_run=dry_run)

    # Initialize risk manager
    account = broker.get_account()
    risk.reset_daily(account['equity'])

    # Start Telegram bot if enabled
    telegram_bot = None
    if enable_telegram and settings.telegram_bot_token:
        from src.notifications.telegram_bot import TelegramBotManager, set_components
        chat_ids = [int(settings.telegram_chat_id)] if settings.telegram_chat_id else []
        set_components(broker, engine, risk, strategy, chat_ids)
        telegram_bot = TelegramBotManager(
            token=settings.telegram_bot_token,
            authorized_chat_ids=chat_ids,
        )
        import asyncio
        loop = asyncio.new_event_loop()
        loop.run_until_complete(telegram_bot.start())
        logger.info("telegram.bot_started")

    mode = "DRY RUN" if dry_run else ("PAPER" if broker.paper else "[LIVE]")
    logger.info("bot.started", mode=mode, strategy=strategy_name, symbols=len(symbols))
    notifier.notify_startup(mode, strategy_name, symbols)

    print(f"\n{'='*60}")
    print(f"  ALGO TRADER RUNNING - {mode}")
    print(f"  Strategy: {strategy_name} | Symbols: {len(symbols)} | Interval: {interval}s")
    print(f"  Ctrl+C to stop gracefully")
    print(f"{'='*60}\n")

    cycle_count = 0
    consecutive_errors = 0

    while not _shutdown_requested:
        cycle_count += 1
        cycle_start = time.time()

        try:
            # For stocks: check market hours
            stock_symbols = [s for s in symbols if "/" not in s]
            if stock_symbols and not broker.is_market_open():
                next_open = broker.next_market_open()
                logger.info("market.closed", next_open=str(next_open))
                # Still process crypto if any
                crypto_only = [s for s in symbols if "/" in s]
                if crypto_only:
                    strategy_copy = create_strategy(strategy_name, crypto_only, timeframe, lookback)
                    results = engine.run_cycle(strategy_copy)
                else:
                    results = []
            else:
                results = engine.run_cycle(strategy)

            consecutive_errors = 0

            if results:
                logger.info("cycle.complete", cycle=cycle_count, trades=len(results))
                for r in results:
                    if not r.get('dry_run'):
                        notifier.notify_trade(r)
            else:
                logger.info("cycle.no_signals", cycle=cycle_count)

        except KeyboardInterrupt:
            break
        except Exception as e:
            consecutive_errors += 1
            logger.error("cycle.error", error=str(e), consecutive=consecutive_errors)
            notifier.notify_error(str(e), f"Cycle {cycle_count}")

            if consecutive_errors >= 5:
                logger.warning("bot.too_many_errors", msg="Pausing for 5 minutes")
                time.sleep(300)
                consecutive_errors = 0
                continue

        # Sleep until next cycle
        elapsed = time.time() - cycle_start
        sleep_time = max(1, interval - elapsed)
        logger.debug("cycle.sleeping", seconds=int(sleep_time))

        # Interruptible sleep
        for _ in range(int(sleep_time)):
            if _shutdown_requested:
                break
            time.sleep(1)

    # Shutdown
    logger.info("bot.stopped", cycles=cycle_count)
    notifier.notify_daily_summary(risk.get_daily_summary())
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
    parser.add_argument("--interval", "-i", type=int, default=60,
                       help="Seconds between trading cycles")
    parser.add_argument("--dry-run", action="store_true", help="Generate signals without placing orders")
    parser.add_argument("--backtest", action="store_true", help="Run custom backtest engine")
    parser.add_argument("--backtest-bt", action="store_true", help="Run Backtrader backtest")
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
    elif args.train:
        cmd_train(broker, symbols, args.timeframe)
    else:
        cmd_run(broker, args.strategy, symbols, args.timeframe,
                args.lookback, args.interval, args.dry_run, notifier,
                enable_telegram=args.telegram)


if __name__ == "__main__":
    main()

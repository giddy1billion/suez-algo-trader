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
import os
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
from src.core.runtime_state import RuntimeState
from src.data.store import DatabaseManager
from src.intelligence.orchestrator import AdaptiveIntelligenceOrchestrator
from src.intelligence.confidence.gate import ConfidenceGate, ConfidenceGateConfig
from src.intelligence.confidence.orchestrator import DecisionOrchestrator
from src.intelligence.confidence.contract_store import ContractStore
from src.notifications.alerts import NotificationManager
from src.monitoring.health import HealthMonitor
from src.monitoring.metrics import LiveMetrics
from src.config.initializer import initialize_configuration_service


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
    """Factory to create strategy instances by name, using settings for params.
    
    Automatically detects asset class from symbols and applies appropriate
    parameters (crypto vs equity) for the momentum strategy.
    """
    from src.strategy.composable import momentum_preset, mean_reversion_preset

    # Detect if symbols are crypto — apply crypto-optimized parameters
    is_crypto = any("/" in s or s.endswith("USD") for s in symbols)

    strategies = {
        "momentum": lambda: MomentumStrategy(
            symbols=symbols, timeframe=timeframe, lookback=lookback,
            fast_ema=settings.crypto_momentum_fast_ema if is_crypto else settings.momentum_fast_ema,
            slow_ema=settings.crypto_momentum_slow_ema if is_crypto else settings.momentum_slow_ema,
            rsi_period=settings.momentum_rsi_period,
            rsi_oversold=settings.crypto_momentum_rsi_oversold if is_crypto else settings.momentum_rsi_oversold,
            rsi_overbought=settings.crypto_momentum_rsi_overbought if is_crypto else settings.momentum_rsi_overbought,
            atr_period=settings.momentum_atr_period,
            atr_sl_multiplier=settings.crypto_momentum_atr_sl_mult if is_crypto else settings.momentum_atr_sl_mult,
            atr_tp_multiplier=settings.crypto_momentum_atr_tp_mult if is_crypto else settings.momentum_atr_tp_mult,
            min_confirming_indicators=settings.crypto_min_confirming_indicators if is_crypto else 2,
        ),
        "mean_reversion": lambda: MeanReversionStrategy(
            symbols=symbols, timeframe=timeframe, lookback=lookback,
            min_confidence=settings.mean_rev_min_confidence,
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

        bt = Backtester.for_symbol(strategy=strategy, symbol=symbol, initial_capital=10000.0)
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
    from src.config.backtest_params import get_backtest_config

    print(f"\n[*] Running VectorBT vectorized backtest")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Timeframe: {timeframe}\n")

    for symbol in symbols:
        print(f"  Fetching data for {symbol}...")
        df = broker.get_bars_df(symbol, timeframe, limit=min(lookback * 3, 1000))

        if df is None or len(df) < 100:
            print(f"  [!] Insufficient data for {symbol}")
            continue

        params = get_backtest_config(symbol)
        metrics = vectorbt_momentum_backtest(
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

        print(f"\n  {'='*50}")
        print(f"  VECTORBT RESULTS: {symbol}")
        print(f"  {'='*50}")
        print(f"  EMA:           {params['fast_ema']}/{params['slow_ema']}")
        print(f"  Fees:          {params['fees']*100:.2f}%")
        print(f"  Risk/Trade:    {params['risk_per_trade']*100:.0f}%")
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
            params = get_backtest_config(symbol)
            print(f"\n  Running parameter sweep on {symbol}...")
            try:
                sweep_df = vectorbt_parameter_sweep(
                    df,
                    fees=params["fees"],
                    risk_per_trade=params["risk_per_trade"],
                    atr_stop_multiplier=params["atr_stop_multiplier"],
                    cooldown_bars=params["cooldown_bars"],
                    annualization_periods=params["annualization_periods"],
                )
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
    notifier: NotificationManager, enable_telegram: bool = True,
    enable_streaming: bool = True
):
    """Main trading loop."""

    # --- Multi-Strategy Orchestrator Setup ---
    from src.strategy.orchestrator import StrategyOrchestrator
    orchestrator = StrategyOrchestrator()

    # Determine multi-strategy config source:
    # Priority: CLI --strategies > settings.multi_strategy_config > auto-generate from symbols
    multi_config = []
    if hasattr(settings, '_cli_strategies') and settings._cli_strategies:
        # Parse CLI --strategies format: "name:symbols:tf:interval:weight;..."
        for entry in settings._cli_strategies.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(":")
            if len(parts) < 2:
                continue
            multi_config.append({
                "name": parts[0].strip(),
                "symbols": [s.strip() for s in parts[1].split(",")],
                "timeframe": parts[2].strip() if len(parts) > 2 else timeframe,
                "interval": int(parts[3]) if len(parts) > 3 else interval,
                "weight": float(parts[4]) if len(parts) > 4 else 1.0,
            })
    elif settings.multi_strategy_config:
        multi_config = settings.multi_strategies_parsed

    if strategy_name == "multi":
        if multi_config:
            # Use explicit multi-strategy config
            for cfg in multi_config:
                try:
                    strat = create_strategy(cfg["name"], cfg["symbols"], cfg["timeframe"], lookback)
                    orchestrator.add_strategy(
                        name=cfg["name"],
                        strategy=strat,
                        symbols=cfg["symbols"],
                        timeframe=cfg["timeframe"],
                        interval=cfg["interval"],
                        weight=cfg["weight"],
                    )
                except Exception as e:
                    logger.warning("orchestrator.strategy_config_error", name=cfg["name"], error=str(e))
        else:
            # Auto-generate multi-strategy config from symbols
            stock_symbols = [s for s in symbols if "/" not in s]
            crypto_symbols = [s for s in symbols if "/" in s]

            if stock_symbols:
                momentum_strat = create_strategy("momentum", stock_symbols, timeframe, lookback)
                orchestrator.add_strategy(
                    name="momentum",
                    strategy=momentum_strat,
                    symbols=stock_symbols,
                    timeframe=timeframe,
                    interval=interval,
                    weight=1.0,
                )

            if stock_symbols:
                mr_strat = create_strategy("mean_reversion", stock_symbols, "15Min", lookback)
                orchestrator.add_strategy(
                    name="mean_reversion",
                    strategy=mr_strat,
                    symbols=stock_symbols,
                    timeframe="15Min",
                    interval=interval * 2,
                    weight=0.7,
                )

            try:
                ml_strat = create_strategy("ml", symbols, timeframe, max(lookback, 500))
                orchestrator.add_strategy(
                    name="ml",
                    strategy=ml_strat,
                    symbols=symbols,
                    timeframe=timeframe,
                    interval=interval * 3,
                    weight=1.5,
                )
            except Exception as e:
                logger.warning("orchestrator.ml_strategy_unavailable", error=str(e))

            if crypto_symbols:
                crypto_strat = create_strategy("momentum", crypto_symbols, settings.crypto_timeframe, lookback)
                orchestrator.add_strategy(
                    name="crypto_momentum",
                    strategy=crypto_strat,
                    symbols=crypto_symbols,
                    timeframe=settings.crypto_timeframe,
                    interval=max(30, interval // 2),
                    weight=0.8,
                )

        # Use first strategy as the "primary" for backward compatibility
        if len(orchestrator) > 0:
            strategy = orchestrator._slots[next(iter(orchestrator._slots))].strategy
        else:
            strategy = create_strategy("momentum", symbols, timeframe, lookback)
            orchestrator.add_strategy(name="momentum", strategy=strategy, symbols=symbols,
                                     timeframe=timeframe, interval=interval, weight=1.0)
        logger.info("orchestrator.multi_mode", strategies=len(orchestrator), weights=orchestrator.get_weights())
    else:
        # Single-strategy mode (backward compatible)
        # Split: crypto gets its own optimized slot; equities get the primary slot
        _single_crypto = [s for s in symbols if "/" in s]
        _single_equity = [s for s in symbols if "/" not in s]

        if _single_crypto and strategy_name == "momentum":
            # Register equity-only primary (or all symbols if no equities)
            primary_symbols = _single_equity if _single_equity else symbols
            strategy = create_strategy(strategy_name, primary_symbols, timeframe, lookback)
            orchestrator.add_strategy(
                name=strategy_name,
                strategy=strategy,
                symbols=primary_symbols,
                timeframe=timeframe,
                interval=interval,
                weight=1.0,
            )
            # Dedicated crypto slot with optimized params (faster EMAs, lower timeframe)
            crypto_strat = create_strategy("momentum", _single_crypto, settings.crypto_timeframe, lookback)
            orchestrator.add_strategy(
                name="crypto_momentum",
                strategy=crypto_strat,
                symbols=_single_crypto,
                timeframe=settings.crypto_timeframe,
                interval=max(30, interval // 2),
                weight=0.8,
            )
        else:
            # Non-momentum or no crypto: single strategy handles everything
            strategy = create_strategy(strategy_name, symbols, timeframe, lookback)
            orchestrator.add_strategy(
                name=strategy_name,
                strategy=strategy,
                symbols=symbols,
                timeframe=timeframe,
                interval=interval,
                weight=1.0,
            )

    logger.info(
        "orchestrator.configured",
        strategies=len(orchestrator),
        slots=list(orchestrator.strategy_names) if hasattr(orchestrator, 'strategy_names') else [],
        weights=orchestrator.get_weights(),
    )

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
    from src.core.events import EventBus, OrderFilled, OrderRejected
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

    # Build RiskEngine explicitly from settings (not defaults) so that
    # /setrisk or .env changes take effect without code modification.
    from src.risk.engine import RiskEngine
    from src.risk.account_risk import AccountRiskLayer
    from src.risk.portfolio_risk import PortfolioRiskLayer
    from src.risk.exposure_risk import ExposureRiskLayer
    from src.risk.execution_risk import ExecutionRiskLayer

    risk_engine = RiskEngine(
        portfolio_layer=PortfolioRiskLayer(
            max_positions=settings.risk_max_positions,
            max_single_stock_pct=settings.risk_max_single_stock_pct,
            max_sector_exposure_pct=settings.risk_max_sector_exposure_pct,
            max_correlation=settings.risk_max_correlation,
            max_gross_exposure_pct=settings.risk_max_gross_exposure_pct,
            max_net_exposure_pct=settings.risk_max_net_exposure_pct,
            max_var_pct=settings.risk_max_var_pct,
            max_portfolio_heat_pct=settings.risk_max_portfolio_heat_pct,
            enabled=settings.risk_portfolio_layer_enabled,
        ),
        account_layer=AccountRiskLayer(
            max_daily_loss_pct=settings.risk_max_daily_loss_pct,
            max_weekly_loss_pct=settings.risk_max_weekly_loss_pct,
            max_drawdown_pct=settings.risk_max_drawdown_pct,
            min_cash_reserve_pct=settings.risk_min_cash_reserve_pct,
            pdt_account_threshold=settings.risk_pdt_account_threshold,
            consecutive_loss_limit=settings.risk_consecutive_loss_limit,
            daily_trade_limit=settings.risk_daily_trade_limit,
            enabled=settings.risk_account_layer_enabled,
        ),
        exposure_layer=ExposureRiskLayer(
            require_stop_loss=settings.risk_require_stop_loss,
            max_adv_pct=settings.risk_max_adv_pct,
            max_trade_concentration_pct=settings.risk_max_trade_concentration_pct,
            max_overnight_exposure_pct=settings.risk_max_overnight_exposure_pct,
            earnings_blackout_days=settings.risk_earnings_blackout_days,
            high_vol_threshold=settings.risk_high_vol_threshold,
            high_vol_size_reduction=settings.risk_high_vol_size_reduction,
            enabled=settings.risk_exposure_layer_enabled,
        ),
        execution_layer=ExecutionRiskLayer(
            max_spread_pct=settings.risk_max_spread_pct,
            min_volume=settings.risk_min_volume,
            max_slippage_pct=settings.risk_max_slippage_pct,
            max_orders_per_minute=settings.risk_max_orders_per_minute,
            cooldown_after_large_loss_minutes=settings.risk_cooldown_after_loss_minutes,
            large_loss_threshold_pct=settings.risk_large_loss_threshold_pct,
            enabled=settings.risk_execution_layer_enabled,
        ),
    )

    # Create shared RuntimeState for pause/resume state management
    runtime_state = RuntimeState()

    # Signal validation gate — enforces professional-grade signal completeness
    from src.strategy.signal_package import SignalValidationGate
    from src.strategy.signal_bridge import SignalBridgeConfig
    signal_gate = SignalValidationGate(
        require_model_provenance=False,  # Relaxed: allow rule-based strategies
        min_confidence=settings.min_signal_confidence if hasattr(settings, 'min_signal_confidence') else 0.55,
        max_position_size_pct=25.0,
        min_risk_reward=1.0,
    )

    # ─── Closed-Loop Feedback System ────────────────────────────────────────
    # Transforms every trade outcome into training data for model improvement.
    from src.ml.feedback_loop import (
        ExperienceDatabase, PostTradeValidator, ContinuousCalibrationMonitor
    )
    from src.ml.promotion_engine import ModelPromotionEngine, ModelRollbackMonitor

    experience_db = ExperienceDatabase()
    post_trade_validator = PostTradeValidator(experience_db)
    calibration_monitor = ContinuousCalibrationMonitor(experience_db, tolerance=0.12)
    promotion_engine = ModelPromotionEngine(
        min_evaluation_trades=30,
        min_improvement_pct=5.0,
    )
    rollback_monitor = ModelRollbackMonitor(
        min_sharpe=-0.5,
        max_drawdown_pct=15.0,
        min_win_rate=0.35,
        max_consecutive_losses=8,
        evaluation_window=20,
        cooldown_hours=24.0,
    )

    # Online learning loop: reward attribution + hard example mining
    from src.ml.online_learning import OnlineLearningLoop
    online_learning = OnlineLearningLoop(experience_db=experience_db)

    # Subscribe TradeClosed events to the post-trade validator
    def _on_trade_closed(event):
        """Record every trade close as a training sample with full context."""
        try:
            # Look up contract decision/confidence from contract store
            contract_decision = ""
            contract_confidence = 0.0
            if event.contract_id and contract_store:
                try:
                    replayed = contract_store.replay(event.contract_id)
                    if replayed:
                        contract_decision = replayed.get("decision", "")
                        contract_confidence = replayed.get("final_confidence", 0.0)
                except Exception:
                    pass

            trade_result = {
                "trade_id": event.trade_id,
                "symbol": event.symbol,
                "exit_price": event.exit_price,
                "pnl": event.pnl,
                "pnl_pct": event.pnl_pct,
                "reason": event.reason,
                "entry_price": event.entry_price,
                "side": event.side,
                "entry_time": event.entry_time,
                "exit_time": event.exit_time,
                "model_version": event.model_version,
                "strategy_name": event.strategy_name,
                "signal_package_id": event.signal_package_id,
                "contract_id": event.contract_id,
                "contract_decision": contract_decision,
                "contract_confidence": contract_confidence,
            }
            post_trade_validator.validate_trade(trade_result)

            # Online learning: reward attribution + hard example mining
            online_learning.process_trade_outcome(trade_result)

            # Feed rollback monitor and check for auto-rollback
            rollback_monitor.record_trade(trade_result)
            rollback_decision = rollback_monitor.check_rollback()
            if rollback_decision and rollback_decision.get("should_rollback"):
                rollback_to = rollback_decision.get("rollback_to")
                reason = rollback_decision.get("reason", "performance degradation")
                if rollback_to:
                    try:
                        runtime_manager.swap_model(rollback_to)
                        rollback_monitor.acknowledge_rollback()
                        logger.warning(
                            "auto_rollback.executed",
                            reason=reason,
                            rollback_to=rollback_to,
                        )
                    except Exception as re:
                        logger.error("auto_rollback.failed", error=str(re))
        except Exception as e:
            logger.debug("feedback_loop.score_error", error=str(e))

    from src.core.events import TradeClosed as _TradeClosed
    event_bus.subscribe(_TradeClosed, _on_trade_closed)

    confidence_gate = ConfidenceGate(ConfidenceGateConfig())

    # Decision Contract infrastructure — every trading decision is auditable
    contract_store = ContractStore(storage_path="data_cache/contracts")
    decision_orchestrator = DecisionOrchestrator(
        gate_config=ConfidenceGateConfig(),
        validity_minutes=getattr(settings, 'contract_validity_minutes', 5.0),
    )

    engine = ExecutionEngine(
        broker=broker,
        risk_manager=risk,
        risk_engine=risk_engine,
        db=db,
        dry_run=dry_run,
        event_bus=event_bus,
        trade_manager=trade_manager,
        execution_simulator=execution_simulator,
        runtime_state=runtime_state,
        signal_gate=signal_gate,
        signal_bridge_config=SignalBridgeConfig(),
        confidence_gate=confidence_gate,
        decision_orchestrator=decision_orchestrator,
        contract_store=contract_store,
        intelligence_orchestrator=AdaptiveIntelligenceOrchestrator(
            min_trade_score=settings.intelligence_min_trade_score,
            drift_window=settings.intelligence_drift_window,
            drift_min_samples=settings.intelligence_drift_min_samples,
            drift_alert_drop=settings.intelligence_drift_alert_drop,
        ) if settings.intelligence_enabled else None,
    )

    # --- Full Telegram Audit Forwarding (ALL events + WARNING+ logs) ---
    # The TelegramAuditForwarder provides rich HTML-formatted messages with
    # timestamps, source attribution, and pause-state awareness. It is the
    # single authoritative Telegram notification path.
    telegram_audit_components = {}
    _telegram_audit_active = False
    if notifier.telegram_token:
        from src.notifications.telegram_audit_forwarder import setup_telegram_full_audit

        def _telegram_html_sender(message: str):
            """Send HTML-formatted message to Telegram."""
            import httpx
            url = f"https://api.telegram.org/bot{notifier.telegram_token}/sendMessage"
            payload = {
                "chat_id": notifier.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML",
            }
            try:
                with httpx.Client(timeout=10) as client:
                    client.post(url, json=payload)
            except Exception:
                pass

        telegram_audit_components = setup_telegram_full_audit(
            event_bus=event_bus,
            send_func=_telegram_html_sender,
            runtime_state=runtime_state,
            attach_log_handler=True,
        )
        _telegram_audit_active = True
        logger.info("telegram_audit.full_forwarding_enabled")

    # Register default event subscribers (audit log, journal, metrics, notifications)
    # NOTE: When TelegramAuditForwarder is active, NotificationSubscriber's Telegram
    # sending is disabled to prevent duplicate messages. The audit forwarder already
    # handles all events with superior formatting and pause-state awareness.
    _notification_send_func = None
    if not _telegram_audit_active and notifier.telegram_token:
        # Fallback: only use NotificationSubscriber if audit forwarder is NOT active
        _send_fn = notifier._send_telegram

        def _notification_sender(message: str):
            """Send notification via Telegram (fallback when audit forwarder unavailable)."""
            try:
                _send_fn(message)
            except Exception:
                pass

        _notification_send_func = _notification_sender

    subscribers = setup_default_subscribers(
        bus=event_bus,
        notification_send_func=_notification_send_func,
    )

    # --- Event Persistence (durable event log for replay & auditing) ---
    from src.core.event_store import EventStore, EventPersistenceSubscriber
    event_store = EventStore(db_path="data_cache/events.db")
    persistence_subscriber = EventPersistenceSubscriber(event_store)
    persistence_subscriber.attach(event_bus)
    logger.info("event_store.initialized", session_id=event_store.session_id)

    # --- Crash Recovery (reconstruct state from broker on startup) ---
    from src.core.recovery import RecoveryManager
    recovery_manager = RecoveryManager(broker, event_bus, trade_manager, event_store)
    recovery_report = recovery_manager.recover()
    if recovery_report.positions_recovered > 0:
        logger.info(
            "recovery.complete",
            positions=recovery_report.positions_recovered,
            orphans=recovery_report.orphans_detected,
        )

    # --- Portfolio Reconciliation (periodic broker vs internal state check) ---
    from src.core.reconciliation import PortfolioReconciler
    reconciler = PortfolioReconciler(broker, trade_manager, event_bus)

    # --- CQRS Read Models (incremental projections for fast dashboard queries) ---
    from src.core.projections import ReadModelManager
    read_models = ReadModelManager()
    read_models.attach(event_bus)

    # --- State Snapshotting (periodic persistence for fast recovery) ---
    from src.core.snapshots import SnapshotStore, SnapshotManager
    snapshot_store = SnapshotStore(db_path="data_cache/snapshots.db")
    snapshot_manager = SnapshotManager(snapshot_store, event_store, snapshot_interval_events=500)
    # Wire snapshot manager to event bus so event-count trigger works
    event_bus.subscribe(None, snapshot_manager.on_event)

    # Initialize monitoring components
    health_monitor = HealthMonitor()
    live_metrics = LiveMetrics()

    # --- Operational Commands Handler ---
    from src.monitoring.ops_commands import OpsCommandHandler
    ops_handler = OpsCommandHandler(
        health_monitor=health_monitor,
        event_bus=event_bus,
        trade_manager=trade_manager,
        broker=broker,
        reconciler=reconciler,
        risk_manager=risk,
        metrics=live_metrics,
    )

    # --- Runtime Manager (hot-swap, concurrent backtest, training, A/B testing) ---
    from src.core.runtime import RuntimeManager
    from src.core.environment import BrokerManager

    broker_manager = BrokerManager(broker)
    runtime_manager = RuntimeManager(
        broker_manager=broker_manager,
        event_bus=event_bus,
        strategy_factory=create_strategy,
    )
    logger.info("runtime_manager.ready", mode=runtime_manager.current_mode)

    # Initialize risk manager
    account = broker.get_account()
    risk.reset_daily(account['equity'])

    # Start Telegram bot if enabled (runs in background thread)
    telegram_bot = None
    _telegram_thread = None
    if enable_telegram and settings.telegram_bot_token:
        from src.notifications.telegram_bot import (
            TelegramBotManager, set_components, get_runtime_changes,
            _runtime_lock as _tg_runtime_lock, _runtime_changes as _tg_runtime_changes,
        )
        from src.notifications.telegram_config_commands import (
            config_router, set_config_components,
        )
        from src.strategy.strategy_store import StrategyStore

        # Parse chat ID — skip if placeholder or non-numeric (auto-detect on first message)
        chat_ids = []
        if settings.telegram_chat_id and settings.telegram_chat_id.isdigit():
            chat_ids = [int(settings.telegram_chat_id)]

        # Initialize strategy store for user-defined strategies
        _strategy_store = StrategyStore()

        telegram_bot = TelegramBotManager(
            token=settings.telegram_bot_token,
            authorized_chat_ids=chat_ids,
        )

        # Register config/strategy management commands router
        telegram_bot.dp.include_router(config_router)
        set_config_components(
            settings=settings,
            strategy_store=_strategy_store,
            authorized_users=set(chat_ids),
            runtime_lock=_tg_runtime_lock,
            runtime_changes=_tg_runtime_changes,
        )

        # Register runtime capabilities commands router
        from src.notifications.telegram_runtime_commands import (
            runtime_router, set_runtime_components,
        )
        telegram_bot.dp.include_router(runtime_router)
        set_runtime_components(
            runtime_manager=runtime_manager,
            authorized_users=set(chat_ids),
        )

        # Register sector management commands router
        from src.notifications.telegram_sector_commands import (
            sector_router, set_sector_components,
        )
        telegram_bot.dp.include_router(sector_router)
        set_sector_components(
            db=db,
            authorized_users=set(chat_ids),
        )

        def _run_telegram_bot(bot):
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            max_restarts = 5
            restart_count = 0
            while restart_count < max_restarts and not _shutdown_event.is_set():
                try:
                    loop.run_until_complete(
                        bot.dp.start_polling(
                            bot.bot,
                            handle_signals=False,
                            drop_pending_updates=True,
                        )
                    )
                    break  # Normal exit
                except Exception as e:
                    restart_count += 1
                    logger.error(
                        "telegram.polling_crashed",
                        error=str(e),
                        restart_attempt=restart_count,
                    )
                    if restart_count < max_restarts:
                        import time as _time
                        _time.sleep(5)  # Wait before retry
                        # Create fresh event loop for retry
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

        _telegram_thread = threading.Thread(target=_run_telegram_bot, args=(telegram_bot,), daemon=True)
        _telegram_thread.start()
        logger.info("telegram.bot_started")

    mode = "DRY RUN" if dry_run else ("PAPER" if broker.paper else "[LIVE]")
    logger.info("bot.started", mode=mode, strategy=strategy_name, symbols=len(symbols))
    notifier.notify_startup(mode, strategy_name, symbols)

    # Start WebSocket streaming if enabled
    _stream_data = {}  # Shared dict: symbol -> latest bar data
    _stream_lock = threading.Lock()
    if enable_streaming:
        import asyncio as _asyncio

        async def _bar_handler(bar):
            """Store incoming bar data from WebSocket."""
            symbol = bar.symbol
            with _stream_lock:
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

    # Start trade update stream (real-time order fills/cancellations)
    if not dry_run:
        def _trade_update_handler(update: dict):
            """Handle real-time trade updates from Alpaca WebSocket."""
            event_type = update.get("event", "")
            order_info = update.get("order", {})
            symbol = order_info.get("symbol", "?")
            order_id = order_info.get("id", "")

            if event_type in ("fill", "partial_fill"):
                filled_qty = order_info.get("filled_qty", "0")
                filled_price = order_info.get("filled_avg_price")
                logger.info("trade_update.fill", event=event_type, symbol=symbol,
                           filled_qty=filled_qty, avg_price=filled_price, order_id=order_id)

                # Publish OrderFilled event if event bus is available
                if event_bus:
                    try:
                        event_bus.publish(OrderFilled(
                            order_id=order_id,
                            fill_price=float(filled_price) if filled_price else 0.0,
                            fill_qty=float(filled_qty) if filled_qty else 0.0,
                            fees=0.0,
                            source="trade_stream",
                        ))
                    except Exception as e:
                        logger.error("event_bus.fill_publish_failed", order_id=order_id, error=str(e))

            elif event_type in ("canceled", "rejected", "expired"):
                reason = order_info.get("status", event_type)
                logger.warning("trade_update.rejected", event=event_type, symbol=symbol,
                              reason=reason, order_id=order_id)

                if event_bus:
                    try:
                        event_bus.publish(OrderRejected(
                            order_id=order_id,
                            reason=f"{event_type}: {reason}",
                            source="trade_stream",
                        ))
                    except Exception as e:
                        logger.error("event_bus.reject_publish_failed", order_id=order_id, error=str(e))
            else:
                logger.debug("trade_update.event", event=event_type, symbol=symbol)

        broker.start_trade_stream(_trade_update_handler)
        logger.info("trade_stream.active")

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
            """Run automated backtests on all symbols with asset-class-aware params."""
            try:
                bt_symbols = settings.auto_backtest_symbols.split(",") if settings.auto_backtest_symbols else symbols
                bt_symbols = [s.strip() for s in bt_symbols if s.strip()]

                from backtesting.vbt_adapter import vectorbt_momentum_backtest
                from src.config.backtest_params import get_backtest_config

                results_text = [f"<b>🔄 Auto-Backtest Results</b>\n{'=' * 30}\n"]
                for sym in bt_symbols:
                    try:
                        # Resolve asset-class-aware parameters (includes timeframe)
                        params = get_backtest_config(sym)
                        sym_timeframe = params.get("timeframe", timeframe)
                        with _broker_lock:
                            df = broker.get_bars_df(sym, sym_timeframe, limit=500)
                        if df is None or len(df) < 50:
                            continue
                        metrics = vectorbt_momentum_backtest(
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
            """Run automated ML training through the governed TrainingPipeline."""
            try:
                _send_telegram("🧠 <b>Auto ML Training Started...</b>")
                train_symbols = symbols[:10]

                # Completion callback — runs when pipeline finishes
                def _on_pipeline_complete(progress):
                    try:
                        if progress.status.value == "completed":
                            with _runtime_lock:
                                _runtime_changes["trigger_train"] = True
                            _send_telegram(
                                f"✅ <b>ML Model Trained (Governed Pipeline)</b>\n"
                                f"Symbols: {len(train_symbols)}\n"
                                f"Model: {progress.version or 'unknown'}\n"
                                f"CV Accuracy: {progress.metrics.get('cv_accuracy', 0):.3f}\n"
                                f"Governance: {'✅ passed' if progress.auto_deployed else '⚠️ not deployed'}\n"
                                f"Duration: {progress.duration_seconds:.0f}s"
                            )
                        else:
                            _send_telegram(
                                f"⚠️ <b>ML Training Failed</b>\n"
                                f"Error: {progress.error or 'unknown'}"
                            )
                    except Exception as e:
                        logger.debug("auto_train.callback_error", error=str(e))

                # Use RuntimeManager's governed TrainingPipeline
                runtime_manager.train_model(
                    symbols=train_symbols,
                    lookback_bars=settings.auto_train_bars,
                    trigger="scheduled",
                    callback=_on_pipeline_complete,
                )
                logger.info("scheduler.auto_train_launched", symbols=len(train_symbols))
            except Exception as e:
                logger.error("scheduler.auto_train_error", error=str(e))
                _send_telegram(f"<b>⚠️ Auto Training Failed</b>\n{str(e)[:200]}")

        def _auto_sweep():
            """Run parameter sweep and report optimal params to Telegram."""
            try:
                from backtesting.vbt_adapter import vectorbt_parameter_sweep
                from src.config.backtest_params import get_backtest_config

                sweep_symbols = symbols[:3]  # Top 3 symbols for sweep
                results_text = [f"<b>🔬 Auto Parameter Sweep</b>\n{'=' * 30}\n"]

                for sym in sweep_symbols:
                    try:
                        with _broker_lock:
                            df = broker.get_bars_df(sym, timeframe, limit=700)
                        if df is None or len(df) < 100:
                            continue

                        params = get_backtest_config(sym)
                        sweep_df = vectorbt_parameter_sweep(
                            df,
                            fees=params["fees"],
                            risk_per_trade=params["risk_per_trade"],
                            atr_stop_multiplier=params["atr_stop_multiplier"],
                            cooldown_bars=params["cooldown_bars"],
                            annualization_periods=params["annualization_periods"],
                        )
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

        # Daily risk reset at market open (09:30 ET)
        def _daily_risk_reset():
            """Reset daily risk counters at market open."""
            try:
                with _broker_lock:
                    account = broker.get_account()
                risk.reset_daily(account['equity'])
                # Refresh sector cache from DB as a daily backstop
                from src.execution.sector_lookup import reload_cache as _reload_sector_cache
                _reload_sector_cache(db)
                logger.info("scheduler.daily_risk_reset", equity=account['equity'])
                _send_telegram("🔄 <b>Daily Risk Reset</b>\nCounters cleared for new trading day.")
            except Exception as e:
                logger.error("scheduler.daily_reset_error", error=str(e))

        _scheduler.add_job(
            _daily_risk_reset,
            CronTrigger(hour=9, minute=30, timezone="US/Eastern"),
            id="daily_risk_reset",
        )

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

        # Automated ML training — runs immediately if model missing, else on schedule
        if settings.auto_train_interval_hours > 0:
            # Run immediately on startup if no model exists (bootstrap)
            _needs_bootstrap = not os.path.exists(settings.ml_model_path)
            _scheduler.add_job(
                _auto_train,
                IntervalTrigger(hours=settings.auto_train_interval_hours),
                id="auto_train",
                next_run_time=datetime.now() if _needs_bootstrap else None,
            )
            if _needs_bootstrap:
                logger.info("scheduler.ml_bootstrap", msg="No model found, training immediately")

        # Automated parameter sweep
        if settings.auto_sweep_interval_hours > 0:
            _scheduler.add_job(
                _auto_sweep,
                IntervalTrigger(hours=settings.auto_sweep_interval_hours),
                id="auto_sweep",
            )

        # ─── Continuous Calibration Check (every 6 hours) ───────────────
        def _calibration_check():
            """Check model calibration and trigger retraining if needed."""
            try:
                report = calibration_monitor.check_calibration()
                if report.get("recommendation") == "needs_retraining":
                    logger.warning(
                        "calibration.drift_detected",
                        ece=report.get("ece"),
                        recommendation=report["recommendation"],
                    )
                    _send_telegram(
                        f"⚠️ <b>Model Calibration Drift</b>\n"
                        f"ECE: {report.get('ece', 0):.3f}\n"
                        f"Recommendation: {report['recommendation']}\n"
                        f"Triggering automatic retraining..."
                    )
                    # Actually trigger retraining through the governed pipeline
                    try:
                        if not runtime_manager.is_training():
                            runtime_manager.train_model(
                                trigger="calibration_drift",
                            )
                            logger.info("calibration.retrain_triggered")
                        else:
                            logger.info("calibration.retrain_skipped_already_running")
                    except Exception as te:
                        logger.error("calibration.retrain_trigger_error", error=str(te))
                elif report.get("recommendation") == "needs_recalibration":
                    logger.info("calibration.minor_drift", ece=report.get("ece"))
                else:
                    logger.info(
                        "calibration.check_ok",
                        ece=report.get("ece"),
                        total_trades=experience_db.total_trades,
                    )
            except Exception as e:
                logger.debug("calibration.check_error", error=str(e))

        _scheduler.add_job(
            _calibration_check,
            IntervalTrigger(hours=6),
            id="calibration_check",
        )

        _scheduler.start()

        active_jobs = [j.id for j in _scheduler.get_jobs()]
        logger.info("scheduler.started", jobs=active_jobs)

    except ImportError:
        logger.debug("scheduler.apscheduler_not_available")

    # Single set_components call AFTER all initialization is complete
    if telegram_bot:
        from src.notifications.telegram_bot import set_components as _set_components_update
        _set_components_update(broker, engine, risk, strategy, db=db,
                              authorized_chat_ids=chat_ids,
                              health_monitor=health_monitor,
                              live_metrics=live_metrics,
                              scheduler=_scheduler,
                              event_bus=event_bus,
                              trade_manager=trade_manager,
                              reconciler=reconciler,
                              ops_handler=ops_handler,
                              runtime_state=runtime_state)

    # Cache crypto-only strategy (fallback, used by runtime-change handlers)
    crypto_only = [s for s in symbols if "/" in s]
    crypto_strategy = create_strategy(strategy_name, crypto_only, settings.crypto_timeframe, lookback) if crypto_only else None

    while not _shutdown_event.is_set():
        cycle_count += 1
        cycle_start = time.time()

        # Periodic TradeManager cleanup to prevent unbounded memory growth
        if trade_manager and cycle_count % 100 == 0:
            removed = trade_manager.remove_terminal()
            if removed > 0:
                logger.info("trade_manager.cleanup", removed=removed, cycle=cycle_count)

        # Reap stale trades stuck in non-terminal states (every 5 cycles)
        if trade_manager and cycle_count % 5 == 0:
            reaped = trade_manager.reap_stale_trades(timeout_seconds=300.0)
            if reaped > 0:
                logger.warning("trade_manager.stale_reaped", count=reaped, cycle=cycle_count)

        # Check signal package expiry/confidence decay (every cycle)
        expired_count = engine.check_signal_expiry()
        if expired_count > 0:
            logger.info("signal_monitor.expired", count=expired_count, cycle=cycle_count)

        # Periodic portfolio reconciliation (every 5 minutes / ~5 cycles at 60s interval)
        if reconciler and cycle_count % 5 == 0:
            try:
                recon_report = reconciler.reconcile()
                if not recon_report.is_reconciled:
                    logger.warning(
                        "reconciliation.drift_detected",
                        discrepancies=len(recon_report.discrepancies),
                    )
                    # Auto-fix safe discrepancies
                    reconciler.auto_fix(recon_report)
            except Exception as e:
                logger.error("reconciliation.error", error=str(e))

        # Health heartbeats
        if health_monitor:
            health_monitor.heartbeat("engine", latency_ms=(time.time() - cycle_start) * 1000)
            # Auto-transition operating mode based on system health
            overall_status, issues = health_monitor.check_component_health()
            _health_scores = {"healthy": 1.0, "degraded": 0.5, "down": 0.1}
            _current_health = _health_scores.get(overall_status, 0.5)
            mode_changed = runtime_state.update_health(
                _current_health,
                reason="; ".join(issues[:3]) if issues else "all_healthy",
            )
            if mode_changed:
                logger.warning(
                    "system.mode_changed",
                    new_mode=mode_changed.value,
                    health=_current_health,
                )

        # Periodic snapshots (every 500 events for fast recovery)
        if snapshot_manager and snapshot_manager.should_snapshot():
            try:
                dashboard = read_models.get_dashboard()
                last_event_id = event_store.count_events()
                snapshot_manager.take_snapshot(event_store.session_id, last_event_id, dashboard)
            except Exception as e:
                logger.error("snapshot.error", error=str(e))

        # Telegram bot thread health check (every 10 cycles)
        if telegram_bot and cycle_count % 10 == 0:
            if '_telegram_thread' in dir() and not _telegram_thread.is_alive():
                logger.error("telegram.thread_dead", msg="Telegram polling thread has died — commands will not work")

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
                crypto_strategy = create_strategy(new_name, crypto_only, settings.crypto_timeframe, lookback) if crypto_only else None

            if changes.get("symbols"):
                new_symbols = changes["symbols"]
                logger.info("runtime.symbols_change", symbols=new_symbols)
                symbols = new_symbols
                strategy = create_strategy(strategy_name, symbols, timeframe, lookback)
                crypto_only = [s for s in symbols if "/" in s]
                crypto_strategy = create_strategy(strategy_name, crypto_only, settings.crypto_timeframe, lookback) if crypto_only else None

            if changes.get("interval"):
                interval = changes["interval"]
                logger.info("runtime.interval_change", interval=interval)

            if changes.get("timeframe"):
                timeframe = changes["timeframe"]
                logger.info("runtime.timeframe_change", timeframe=timeframe)
                strategy = create_strategy(strategy_name, symbols, timeframe, lookback)
                crypto_only = [s for s in symbols if "/" in s]
                crypto_strategy = create_strategy(strategy_name, crypto_only, settings.crypto_timeframe, lookback) if crypto_only else None

            if changes.get("lookback"):
                lookback = changes["lookback"]
                logger.info("runtime.lookback_change", lookback=lookback)
                strategy = create_strategy(strategy_name, symbols, timeframe, lookback)
                crypto_only = [s for s in symbols if "/" in s]
                crypto_strategy = create_strategy(strategy_name, crypto_only, settings.crypto_timeframe, lookback) if crypto_only else None

            if changes.get("config_updates"):
                # Strategy params changed — rebuild strategy with new settings
                logger.info("runtime.config_update", params=list(changes["config_updates"].keys()))
                strategy = create_strategy(strategy_name, symbols, timeframe, lookback)
                crypto_only = [s for s in symbols if "/" in s]
                crypto_strategy = create_strategy(strategy_name, crypto_only, settings.crypto_timeframe, lookback) if crypto_only else None

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

        # Sync broker reference if runtime_manager switched environments
        if runtime_manager.broker is not broker:
            broker = runtime_manager.broker
            engine._broker = broker
            logger.info("runtime.broker_synced", paper=broker.paper)

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

            # Feed account state to multi-layer risk engine every cycle
            try:
                account = broker.get_account()
                engine.risk_engine.update_account_state(
                    current_equity=account['equity'],
                    daily_pnl=risk.daily_stats.realized_pnl,
                    weekly_pnl=risk.weekly_pnl,
                    cash=account['cash'],
                )
            except Exception as e:
                logger.debug("risk_engine.state_update_error", error=str(e))

            # For stocks: check market hours
            stock_symbols = [s for s in symbols if "/" not in s]
            if stock_symbols and not broker.is_market_open():
                next_open = broker.next_market_open()
                logger.info("market.closed", next_open=str(next_open))
                # Market closed: run crypto strategies via orchestrator (has correct params)
                # Temporarily disable stock-only slots so only crypto executes
                _disabled_slots = []
                for slot_name in list(orchestrator.strategy_names):
                    slot = orchestrator._slots.get(slot_name)
                    if slot and slot.enabled and not any("/" in s for s in slot.symbols):
                        orchestrator.disable_strategy(slot_name)
                        _disabled_slots.append(slot_name)

                with _broker_lock:
                    results = orchestrator.run_due_strategies(engine)

                # Re-enable stock slots for next market-open cycle
                for slot_name in _disabled_slots:
                    orchestrator.enable_strategy(slot_name)
            else:
                # Full execution — use orchestrator for all strategies
                with _broker_lock:
                    results = orchestrator.run_due_strategies(engine)

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
    runtime_manager.shutdown()
    if _scheduler:
        _scheduler.shutdown(wait=False)
    broker.stop_trade_stream()
    notifier.notify_daily_summary(risk.get_daily_summary())

    # Close database connections
    try:
        if event_store:
            event_store.close()
    except Exception as e:
        logger.error("shutdown.event_store_close_failed", error=str(e))

    try:
        if snapshot_store:
            snapshot_store.close()
    except Exception as e:
        logger.error("shutdown.snapshot_store_close_failed", error=str(e))

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
                       choices=["momentum", "mean_reversion", "ml", "multi"],
                       help="Trading strategy (use 'multi' for concurrent strategies)")
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
    parser.add_argument("--no-stream", action="store_true", help="Disable WebSocket streaming for real-time data")
    parser.add_argument("--train", action="store_true", help="Train/retrain ML model")
    parser.add_argument("--status", action="store_true", help="Show account status and exit")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram bot")
    parser.add_argument("--strategies", default="",
                       help="Multi-strategy config (semicolon-separated): name:symbols:timeframe:interval:weight")

    args = parser.parse_args()

    # Setup logging
    setup_logging(settings.log_level, settings.log_file)
    logger = get_logger("main")

    # Initialize configuration service (load persisted DB configs)
    logger.info("main.initializing_configuration_service")
    try:
        config_service = initialize_configuration_service(
            database_url=settings.database_url,
            seed_from_env=True,  # On first run, overlay env vars over defaults
            auto_refresh=True,
        )
        logger.info(
            "main.configuration_service_initialized",
            cache_entries=len(config_service._cache) if config_service._cache else 0,
        )
    except Exception as e:
        logger.error(
            "main.configuration_service_initialization_failed",
            error=str(e),
        )
        # Non-fatal: continue with env-based settings
        config_service = None
        logger.warning("main.continuing_with_env_based_config")


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
        # Pass CLI strategies config to settings for orchestrator to pick up
        if args.strategies:
            settings._cli_strategies = args.strategies
        else:
            settings._cli_strategies = ""
        cmd_run(broker, args.strategy, symbols, args.timeframe,
                args.lookback, args.interval, args.dry_run, notifier,
                enable_telegram=not args.no_telegram,
                enable_streaming=not getattr(args, 'no_stream', False))


if __name__ == "__main__":
    main()

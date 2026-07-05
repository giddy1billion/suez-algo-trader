"""
Configuration Seed — Populates the database with default runtime configuration.

Run during initial setup or migration to move existing env-based settings
into the database-backed configuration system.
"""

from src.config.repository import ConfigurationRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Default Configuration Definitions ────────────────────────────────────────
# These define all runtime settings that should live in the database.
# Format: (category, key, default_value, value_type, description, is_editable, validation_rule)

DEFAULT_CONFIGURATIONS = [
    # ─── Trading ──────────────────────────────────────────────────────────────
    ("trading", "trading_mode", "paper", "str", "Trading mode: paper or live", True, "options:paper,live"),
    ("trading", "trading_interval", "60", "int", "Trading loop interval in seconds", True, "range:5:3600"),
    ("trading", "max_consecutive_errors", "5", "int", "Max consecutive errors before cooldown", True, "range:1:100"),
    ("trading", "error_cooldown_seconds", "300", "int", "Cooldown duration after max errors (seconds)", True, "range:30:3600"),
    ("trading", "active_strategy", "momentum", "str", "Active trading strategy", True, "options:momentum,mean_reversion,ml,multi"),
    ("trading", "trading_symbols", "AAPL,MSFT,GOOGL,AMZN,NVDA,BTC/USD,ETH/USD,SOL/USD,AAVE/USD,ADA/USD", "str", "Comma-separated trading symbols", True, ""),
    ("trading", "timeframe", "1Hour", "str", "Trading timeframe", True, "options:1Min,5Min,15Min,30Min,1Hour,4Hour,1Day"),
    ("trading", "lookback_bars", "200", "int", "Number of historical bars to fetch", True, "range:10:5000"),
    ("trading", "enable_auto_trading", "true", "bool", "Enable automatic trade execution", True, ""),
    ("trading", "multi_strategy_config", "", "str", "Multi-strategy orchestrator config", True, ""),

    # ─── Risk Management ──────────────────────────────────────────────────────
    ("risk", "max_position_size_pct", "0.02", "float", "Maximum position size as fraction of portfolio", True, "range:0.001:0.5"),
    ("risk", "max_daily_loss_pct", "0.05", "float", "Maximum daily loss percentage", True, "range:0.001:0.5"),
    ("risk", "max_portfolio_exposure", "0.80", "float", "Maximum portfolio exposure", True, "range:0.1:2.0"),
    ("risk", "max_single_stock_pct", "0.15", "float", "Maximum single stock allocation", True, "range:0.01:0.5"),
    ("risk", "max_leverage", "1.0", "float", "Maximum leverage allowed", True, "range:0.1:10.0"),
    ("risk", "max_open_positions", "20", "int", "Maximum concurrent open positions", True, "range:1:200"),
    ("risk", "max_orders_per_day", "100", "int", "Maximum orders per trading day", True, "range:1:10000"),
    ("risk", "max_correlated_positions", "3", "int", "Maximum correlated positions", True, "range:1:20"),
    ("risk", "default_stop_loss_pct", "0.03", "float", "Default stop loss percentage", True, "range:0.001:0.5"),
    ("risk", "default_take_profit_pct", "0.06", "float", "Default take profit percentage", True, "range:0.001:1.0"),
    ("risk", "emergency_stop_loss", "0.10", "float", "Emergency portfolio stop loss", True, "range:0.01:0.5"),
    ("risk", "drawdown_limit", "0.15", "float", "Maximum drawdown before halt", True, "range:0.01:0.5"),

    # ─── Risk Engine (Multi-Layer) ────────────────────────────────────────────
    ("risk_portfolio", "max_positions", "10", "int", "Max positions (portfolio layer)", True, "range:1:100"),
    ("risk_portfolio", "max_single_stock_pct", "0.20", "float", "Max single stock %", True, "range:0.01:1.0"),
    ("risk_portfolio", "max_sector_exposure_pct", "0.40", "float", "Max sector exposure %", True, "range:0.01:1.0"),
    ("risk_portfolio", "max_correlation", "0.80", "float", "Max position correlation", True, "range:0.1:1.0"),
    ("risk_portfolio", "max_gross_exposure_pct", "2.00", "float", "Max gross exposure %", True, "range:0.1:10.0"),
    ("risk_portfolio", "max_net_exposure_pct", "1.00", "float", "Max net exposure %", True, "range:0.1:5.0"),
    ("risk_portfolio", "max_var_pct", "0.05", "float", "Max Value-at-Risk %", True, "range:0.001:0.5"),
    ("risk_portfolio", "max_portfolio_heat_pct", "0.10", "float", "Max portfolio heat %", True, "range:0.01:0.5"),
    ("risk_portfolio", "layer_enabled", "true", "bool", "Enable portfolio risk layer", True, ""),

    ("risk_account", "max_daily_loss_pct", "0.03", "float", "Max daily loss (account layer)", True, "range:0.001:0.5"),
    ("risk_account", "max_weekly_loss_pct", "0.07", "float", "Max weekly loss %", True, "range:0.001:0.5"),
    ("risk_account", "max_drawdown_pct", "0.15", "float", "Max drawdown %", True, "range:0.01:0.5"),
    ("risk_account", "min_cash_reserve_pct", "0.20", "float", "Minimum cash reserve %", True, "range:0.0:1.0"),
    ("risk_account", "pdt_account_threshold", "25000.0", "float", "PDT rule threshold ($)", True, "range:0:1000000"),
    ("risk_account", "consecutive_loss_limit", "5", "int", "Consecutive loss limit", True, "range:1:50"),
    ("risk_account", "daily_trade_limit", "20", "int", "Daily trade limit", True, "range:1:1000"),
    ("risk_account", "layer_enabled", "true", "bool", "Enable account risk layer", True, ""),

    ("risk_exposure", "require_stop_loss", "true", "bool", "Require stop loss on all trades", True, ""),
    ("risk_exposure", "max_adv_pct", "0.01", "float", "Max avg daily volume %", True, "range:0.001:0.1"),
    ("risk_exposure", "max_trade_concentration_pct", "0.05", "float", "Max trade concentration %", True, "range:0.001:0.5"),
    ("risk_exposure", "max_overnight_exposure_pct", "0.60", "float", "Max overnight exposure %", True, "range:0.0:1.0"),
    ("risk_exposure", "earnings_blackout_days", "1", "int", "Earnings blackout days", True, "range:0:10"),
    ("risk_exposure", "high_vol_threshold", "0.03", "float", "High volatility threshold", True, "range:0.001:0.2"),
    ("risk_exposure", "high_vol_size_reduction", "0.50", "float", "Size reduction in high vol", True, "range:0.1:1.0"),
    ("risk_exposure", "layer_enabled", "true", "bool", "Enable exposure risk layer", True, ""),

    ("risk_execution", "max_spread_pct", "0.005", "float", "Max bid-ask spread %", True, "range:0.0001:0.05"),
    ("risk_execution", "min_volume", "10000", "int", "Minimum volume requirement", True, "range:100:1000000"),
    ("risk_execution", "max_slippage_pct", "0.003", "float", "Max acceptable slippage %", True, "range:0.0001:0.05"),
    ("risk_execution", "max_orders_per_minute", "10", "int", "Max orders per minute", True, "range:1:100"),
    ("risk_execution", "cooldown_after_loss_minutes", "5", "int", "Cooldown after large loss (minutes)", True, "range:0:60"),
    ("risk_execution", "large_loss_threshold_pct", "0.01", "float", "Large loss threshold %", True, "range:0.001:0.1"),
    ("risk_execution", "layer_enabled", "true", "bool", "Enable execution risk layer", True, ""),

    # ─── Strategy Parameters (Momentum) ───────────────────────────────────────
    ("strategy_momentum", "fast_ema", "12", "int", "Fast EMA period", True, "range:2:100"),
    ("strategy_momentum", "slow_ema", "26", "int", "Slow EMA period", True, "range:5:500"),
    ("strategy_momentum", "rsi_period", "14", "int", "RSI period", True, "range:2:100"),
    ("strategy_momentum", "rsi_oversold", "30", "int", "RSI oversold threshold", True, "range:5:50"),
    ("strategy_momentum", "rsi_overbought", "70", "int", "RSI overbought threshold", True, "range:50:95"),
    ("strategy_momentum", "atr_period", "14", "int", "ATR period", True, "range:2:100"),
    ("strategy_momentum", "atr_sl_mult", "2.0", "float", "ATR stop loss multiplier", True, "range:0.5:10.0"),
    ("strategy_momentum", "atr_tp_mult", "3.0", "float", "ATR take profit multiplier", True, "range:0.5:20.0"),

    # ─── Strategy Parameters (Mean Reversion) ─────────────────────────────────
    ("strategy_mean_reversion", "bb_period", "20", "int", "Bollinger Bands period", True, "range:5:200"),
    ("strategy_mean_reversion", "bb_std", "2.0", "float", "Bollinger Bands std deviation", True, "range:0.5:5.0"),
    ("strategy_mean_reversion", "zscore_entry", "2.0", "float", "Z-score entry threshold", True, "range:0.5:5.0"),
    ("strategy_mean_reversion", "zscore_exit", "0.5", "float", "Z-score exit threshold", True, "range:0.0:3.0"),
    ("strategy_mean_reversion", "rsi_period", "14", "int", "RSI period", True, "range:2:100"),
    ("strategy_mean_reversion", "min_confidence", "0.60", "float", "Minimum confidence threshold", True, "range:0.0:1.0"),

    # ─── ML Configuration ─────────────────────────────────────────────────────
    ("ml", "model_path", "models/latest_model.joblib", "str", "Path to ML model file", True, ""),
    ("ml", "retrain_interval_hours", "24", "int", "ML retraining interval (hours)", True, "range:1:720"),
    ("ml", "min_confidence", "0.65", "float", "ML minimum prediction confidence", True, "range:0.0:1.0"),
    ("ml", "confidence_threshold", "0.65", "float", "ML confidence threshold for trade execution", True, "range:0.0:1.0"),

    # ─── Intelligence Layer ───────────────────────────────────────────────────
    ("intelligence", "enabled", "true", "bool", "Enable adaptive intelligence layer", True, ""),
    ("intelligence", "min_trade_score", "70.0", "float", "Minimum trade score threshold", True, "range:0:100"),
    ("intelligence", "drift_window", "200", "int", "Drift detection window size", True, "range:10:5000"),
    ("intelligence", "drift_min_samples", "50", "int", "Minimum samples for drift detection", True, "range:10:1000"),
    ("intelligence", "drift_alert_drop", "0.12", "float", "Drift alert threshold drop", True, "range:0.01:0.5"),

    # ─── Notifications ────────────────────────────────────────────────────────
    ("notifications", "notify_on_trade", "true", "bool", "Send notification on trade execution", True, ""),
    ("notifications", "notify_on_error", "true", "bool", "Send notification on errors", True, ""),
    ("notifications", "notify_on_signal", "false", "bool", "Send notification on signal generation", True, ""),
    ("notifications", "polling_interval", "60", "int", "Notification polling interval (seconds)", True, "range:5:600"),

    # ─── Backtesting ──────────────────────────────────────────────────────────
    ("backtesting", "commission_pct", "0.001", "float", "Backtest commission %", True, "range:0.0:0.05"),
    ("backtesting", "slippage_pct", "0.0005", "float", "Backtest slippage %", True, "range:0.0:0.05"),
    ("backtesting", "initial_cash", "10000.0", "float", "Backtest initial cash ($)", True, "range:100:10000000"),

    # ─── Automation/Scheduler ─────────────────────────────────────────────────
    ("scheduler", "auto_backtest_interval_hours", "6", "int", "Auto backtest interval (hours, 0=disabled)", True, "range:0:168"),
    ("scheduler", "auto_train_interval_hours", "24", "int", "Auto ML training interval (hours, 0=disabled)", True, "range:0:720"),
    ("scheduler", "auto_sweep_interval_hours", "12", "int", "Auto parameter sweep interval (hours, 0=disabled)", True, "range:0:168"),
    ("scheduler", "auto_backtest_symbols", "", "str", "Override symbols for auto backtest (empty=use trading_symbols)", True, ""),
    ("scheduler", "auto_train_bars", "1000", "int", "Bars to use for auto ML training", True, "range:100:50000"),

    # ─── Execution ────────────────────────────────────────────────────────────
    ("execution", "enable_execution_simulator", "true", "bool", "Enable execution simulator", True, ""),
    ("execution", "execution_simulator_preset", "realistic", "str", "Execution simulator preset", True, "options:realistic,conservative,ideal"),
    ("execution", "request_timeout", "30", "int", "Exchange request timeout (seconds)", True, "range:5:120"),
    ("execution", "retry_count", "3", "int", "Number of retries on failure", True, "range:0:10"),
    ("execution", "websocket_reconnect_delay", "5", "int", "WebSocket reconnect delay (seconds)", True, "range:1:60"),

    # ─── Feature Flags ────────────────────────────────────────────────────────
    ("feature_flags", "enable_copy_trading", "false", "bool", "Enable copy trading feature", True, ""),
    ("feature_flags", "enable_paper_trading", "true", "bool", "Enable paper trading feature", True, ""),
    ("feature_flags", "enable_new_strategy", "false", "bool", "Enable new experimental strategy", True, ""),

    # ─── Logging ──────────────────────────────────────────────────────────────
    ("logging", "log_level", "INFO", "str", "Application log level", True, "options:DEBUG,INFO,WARNING,ERROR,CRITICAL"),
    ("logging", "log_file", "logs/trader.log", "str", "Log file path", True, ""),

    # ─── Data Feed ────────────────────────────────────────────────────────────
    ("exchange", "data_feed", "iex", "str", "Alpaca data feed (iex=free, sip=paid)", True, "options:iex,sip"),
]


def seed_default_configuration(
    database_url: str = "sqlite:///data_cache/trading.db",
    overwrite: bool = False,
    changed_by: str = "system",
) -> int:
    """
    Seed the database with default configuration values.

    Args:
        database_url: Database connection string.
        overwrite: If True, overwrite existing values; if False, only create missing entries.
        changed_by: Attribution for the seed operation.

    Returns:
        Count of entries seeded.
    """
    repo = ConfigurationRepository(database_url)
    count = 0

    for category, key, value, value_type, description, is_editable, validation_rule in DEFAULT_CONFIGURATIONS:
        existing = repo.get(category, key)
        if existing and not overwrite:
            continue

        repo.set(
            category=category,
            key=key,
            value=value,
            value_type=value_type,
            changed_by=changed_by,
            change_reason="seed_default" if not existing else "seed_overwrite",
            description=description,
            is_editable=is_editable,
            validation_rule=validation_rule,
        )
        count += 1

    logger.info("config_seed.completed", entries_seeded=count, overwrite=overwrite)
    return count


def seed_from_settings(
    database_url: str = "sqlite:///data_cache/trading.db",
    changed_by: str = "system",
) -> int:
    """
    Seed configuration from the current Settings singleton values.

    This migrates live env-loaded values into the database, preserving
    any customizations made via environment variables.
    """
    from config.settings import settings

    repo = ConfigurationRepository(database_url)

    # Map settings attributes to (category, key, value_type, description, validation_rule)
    settings_map = [
        ("trading", "trading_mode", settings.trading_mode.value, "str", "Trading mode", "options:paper,live"),
        ("trading", "trading_interval", str(settings.trading_interval), "int", "Trading loop interval (s)", "range:5:3600"),
        ("trading", "max_consecutive_errors", str(settings.max_consecutive_errors), "int", "Max consecutive errors", "range:1:100"),
        ("trading", "error_cooldown_seconds", str(settings.error_cooldown_seconds), "int", "Error cooldown (s)", "range:30:3600"),
        ("trading", "active_strategy", settings.active_strategy, "str", "Active strategy", "options:momentum,mean_reversion,ml,multi"),
        ("trading", "trading_symbols", settings.trading_symbols, "str", "Trading symbols", ""),
        ("trading", "timeframe", settings.timeframe, "str", "Timeframe", ""),
        ("trading", "lookback_bars", str(settings.lookback_bars), "int", "Lookback bars", "range:10:5000"),
        ("trading", "multi_strategy_config", settings.multi_strategy_config, "str", "Multi-strategy config", ""),

        ("risk", "max_position_size_pct", str(settings.max_position_size_pct), "float", "Max position size %", "range:0.001:0.5"),
        ("risk", "max_daily_loss_pct", str(settings.max_daily_loss_pct), "float", "Max daily loss %", "range:0.001:0.5"),
        ("risk", "max_portfolio_exposure", str(settings.max_portfolio_exposure), "float", "Max portfolio exposure", "range:0.1:2.0"),
        ("risk", "max_single_stock_pct", str(settings.max_single_stock_pct), "float", "Max single stock %", "range:0.01:0.5"),
        ("risk", "max_leverage", str(settings.max_leverage), "float", "Max leverage", "range:0.1:10.0"),
        ("risk", "max_open_positions", str(settings.max_open_positions), "int", "Max open positions", "range:1:200"),
        ("risk", "max_orders_per_day", str(settings.max_orders_per_day), "int", "Max orders/day", "range:1:10000"),
        ("risk", "max_correlated_positions", str(settings.max_correlated_positions), "int", "Max correlated positions", "range:1:20"),
        ("risk", "default_stop_loss_pct", str(settings.default_stop_loss_pct), "float", "Default stop loss %", "range:0.001:0.5"),
        ("risk", "default_take_profit_pct", str(settings.default_take_profit_pct), "float", "Default take profit %", "range:0.001:1.0"),

        ("strategy_momentum", "fast_ema", str(settings.momentum_fast_ema), "int", "Fast EMA period", "range:2:100"),
        ("strategy_momentum", "slow_ema", str(settings.momentum_slow_ema), "int", "Slow EMA period", "range:5:500"),
        ("strategy_momentum", "rsi_period", str(settings.momentum_rsi_period), "int", "RSI period", "range:2:100"),
        ("strategy_momentum", "rsi_oversold", str(settings.momentum_rsi_oversold), "int", "RSI oversold", "range:5:50"),
        ("strategy_momentum", "rsi_overbought", str(settings.momentum_rsi_overbought), "int", "RSI overbought", "range:50:95"),
        ("strategy_momentum", "atr_period", str(settings.momentum_atr_period), "int", "ATR period", "range:2:100"),
        ("strategy_momentum", "atr_sl_mult", str(settings.momentum_atr_sl_mult), "float", "ATR SL multiplier", "range:0.5:10.0"),
        ("strategy_momentum", "atr_tp_mult", str(settings.momentum_atr_tp_mult), "float", "ATR TP multiplier", "range:0.5:20.0"),

        ("strategy_mean_reversion", "bb_period", str(settings.mean_rev_bb_period), "int", "BB period", "range:5:200"),
        ("strategy_mean_reversion", "bb_std", str(settings.mean_rev_bb_std), "float", "BB std dev", "range:0.5:5.0"),
        ("strategy_mean_reversion", "zscore_entry", str(settings.mean_rev_zscore_entry), "float", "Z-score entry", "range:0.5:5.0"),
        ("strategy_mean_reversion", "zscore_exit", str(settings.mean_rev_zscore_exit), "float", "Z-score exit", "range:0.0:3.0"),
        ("strategy_mean_reversion", "rsi_period", str(settings.mean_rev_rsi_period), "int", "RSI period", "range:2:100"),
        ("strategy_mean_reversion", "min_confidence", str(settings.mean_rev_min_confidence), "float", "Min confidence", "range:0.0:1.0"),

        ("ml", "model_path", settings.ml_model_path, "str", "ML model path", ""),
        ("ml", "retrain_interval_hours", str(settings.ml_retrain_interval_hours), "int", "Retrain interval (hours)", "range:1:720"),
        ("ml", "min_confidence", str(settings.ml_min_confidence), "float", "ML min confidence", "range:0.0:1.0"),

        ("intelligence", "enabled", str(settings.intelligence_enabled).lower(), "bool", "Intelligence enabled", ""),
        ("intelligence", "min_trade_score", str(settings.intelligence_min_trade_score), "float", "Min trade score", "range:0:100"),
        ("intelligence", "drift_window", str(settings.intelligence_drift_window), "int", "Drift window", "range:10:5000"),
        ("intelligence", "drift_min_samples", str(settings.intelligence_drift_min_samples), "int", "Drift min samples", "range:10:1000"),
        ("intelligence", "drift_alert_drop", str(settings.intelligence_drift_alert_drop), "float", "Drift alert drop", "range:0.01:0.5"),

        ("notifications", "notify_on_trade", str(settings.notify_on_trade).lower(), "bool", "Notify on trade", ""),
        ("notifications", "notify_on_error", str(settings.notify_on_error).lower(), "bool", "Notify on error", ""),
        ("notifications", "notify_on_signal", str(settings.notify_on_signal).lower(), "bool", "Notify on signal", ""),

        ("backtesting", "commission_pct", str(settings.backtest_commission_pct), "float", "Commission %", "range:0.0:0.05"),
        ("backtesting", "slippage_pct", str(settings.backtest_slippage_pct), "float", "Slippage %", "range:0.0:0.05"),
        ("backtesting", "initial_cash", str(settings.backtest_initial_cash), "float", "Initial cash", "range:100:10000000"),

        ("scheduler", "auto_backtest_interval_hours", str(settings.auto_backtest_interval_hours), "int", "Auto backtest interval", "range:0:168"),
        ("scheduler", "auto_train_interval_hours", str(settings.auto_train_interval_hours), "int", "Auto train interval", "range:0:720"),
        ("scheduler", "auto_sweep_interval_hours", str(settings.auto_sweep_interval_hours), "int", "Auto sweep interval", "range:0:168"),
        ("scheduler", "auto_backtest_symbols", settings.auto_backtest_symbols, "str", "Auto backtest symbols", ""),
        ("scheduler", "auto_train_bars", str(settings.auto_train_bars), "int", "Auto train bars", "range:100:50000"),

        ("execution", "enable_execution_simulator", str(settings.enable_execution_simulator).lower(), "bool", "Execution simulator enabled", ""),
        ("execution", "execution_simulator_preset", settings.execution_simulator_preset, "str", "Simulator preset", "options:realistic,conservative,ideal"),

        ("logging", "log_level", settings.log_level, "str", "Log level", "options:DEBUG,INFO,WARNING,ERROR,CRITICAL"),
        ("logging", "log_file", settings.log_file, "str", "Log file path", ""),

        ("exchange", "data_feed", settings.alpaca_data_feed, "str", "Data feed", "options:iex,sip"),
    ]

    count = 0
    for category, key, value, value_type, description, validation_rule in settings_map:
        existing = repo.get(category, key)
        if existing:
            continue  # Don't overwrite existing DB values with env values

        repo.set(
            category=category,
            key=key,
            value=value,
            value_type=value_type,
            changed_by=changed_by,
            change_reason="migrated_from_env",
            description=description,
            is_editable=True,
            validation_rule=validation_rule,
        )
        count += 1

    logger.info("config_seed.from_settings_completed", entries_migrated=count)
    return count

"""
Centralized configuration using Pydantic settings.
Loads from .env file with validation and type coercion.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class OperationalMode(str, Enum):
    RESEARCH = "research"
    PAPER = "paper"
    LIVE = "live"


class Settings(BaseSettings):
    """Application configuration -- loaded from .env + environment variables."""

    # --- Trading Mode ---
    trading_mode: TradingMode = TradingMode.PAPER

    # --- Alpaca Paper ---
    alpaca_paper_api_key: str = ""
    alpaca_paper_secret_key: str = ""
    alpaca_paper_base_url: str = "https://paper-api.alpaca.markets"

    # --- Alpaca Live ---
    alpaca_live_api_key: str = ""
    alpaca_live_secret_key: str = ""
    alpaca_live_base_url: str = "https://api.alpaca.markets"

    # --- Data Feed ---
    alpaca_data_feed: str = "iex"

    # --- Trading Loop ---
    trading_interval: int = 60
    max_consecutive_errors: int = 5
    error_cooldown_seconds: int = 300

    # --- Risk Management ---
    max_position_size_pct: float = 0.02
    max_daily_loss_pct: float = 0.05
    max_portfolio_exposure: float = 0.80
    max_single_stock_pct: float = 0.15
    max_leverage: float = 1.0
    max_open_positions: int = 20
    max_orders_per_day: int = 100
    max_correlated_positions: int = 3
    default_stop_loss_pct: float = 0.03
    default_take_profit_pct: float = 0.06

    # --- Strategy ---
    active_strategy: str = "momentum"
    trading_symbols: str = "AAPL,MSFT,GOOGL,AMZN,NVDA,BTC/USD,ETH/USD,SOL/USD,AAVE/USD,ADA/USD"
    timeframe: str = "1Hour"
    lookback_bars: int = 200

    # --- Multi-Strategy Orchestrator ---
    # JSON-style config for multi-strategy mode (used when active_strategy="multi")
    # Format: name:symbols:timeframe:interval:weight (semicolon-separated strategies)
    # Example: "momentum:AAPL,MSFT:1Hour:60:1.0;ml:NVDA,TSLA:15Min:120:1.5"
    multi_strategy_config: str = ""

    # --- Strategy Parameters (Momentum) ---
    momentum_fast_ema: int = 12
    momentum_slow_ema: int = 26
    momentum_rsi_period: int = 14
    momentum_rsi_oversold: int = 30
    momentum_rsi_overbought: int = 70
    momentum_atr_period: int = 14
    momentum_atr_sl_mult: float = 2.0
    momentum_atr_tp_mult: float = 3.0

    # --- Strategy Parameters (Crypto Momentum) ---
    # Crypto requires different parameters due to 24/7 trading, higher volatility
    crypto_momentum_fast_ema: int = 21
    crypto_momentum_slow_ema: int = 55
    crypto_momentum_rsi_oversold: int = 25
    crypto_momentum_rsi_overbought: int = 75
    crypto_momentum_atr_sl_mult: float = 2.5
    crypto_momentum_atr_tp_mult: float = 4.0
    crypto_timeframe: str = "15Min"
    crypto_lookback_bars: int = 500
    crypto_min_confirming_indicators: int = 1

    # --- Signal Deduplication ---
    signal_dedup_enabled: bool = True
    signal_dedup_strength_threshold: float = 0.10  # Notify only if strength changes by 10%+

    # --- Strategy Parameters (Mean Reversion) ---
    mean_rev_bb_period: int = 20
    mean_rev_bb_std: float = 2.0
    mean_rev_zscore_entry: float = 2.0
    mean_rev_zscore_exit: float = 0.5
    mean_rev_rsi_period: int = 14
    mean_rev_min_confidence: float = 0.60

    # --- ML ---
    ml_model_path: str = "models/latest_model.joblib"
    ml_retrain_interval_hours: int = 24
    ml_min_confidence: float = 0.65

    # --- Adaptive Intelligence Layer ---
    intelligence_enabled: bool = True
    intelligence_min_trade_score: float = 70.0
    intelligence_drift_window: int = 200
    intelligence_drift_min_samples: int = 50
    intelligence_drift_alert_drop: float = 0.12

    # --- Notifications ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""
    notify_on_trade: bool = True
    notify_on_error: bool = True
    notify_on_signal: bool = False

    # --- Database ---
    # Supports both PostgreSQL and SQLite:
    #   PostgreSQL: "postgresql://user:pass@host:5432/dbname?sslmode=require"
    #   SQLite:     "sqlite:///data_cache/trading.db"
    database_url: str = "sqlite:///data_cache/trading.db"
    correlation_store_db_path: str = "data_cache/correlation_store.db"

    # --- Redis (optional — graceful fallback to in-memory if empty) ---
    redis_url: str = ""

    # --- Azure Blob Storage (optional — falls back to local filesystem) ---
    blob_storage_account_url: str = ""
    blob_storage_container_prefix: str = "suez-trader"

    # --- Azure Key Vault (optional — falls back to env vars) ---
    key_vault_url: str = ""

    # --- Logging ---
    log_level: str = "INFO"
    log_file: str = "logs/trader.log"

    # --- Backtesting ---
    backtest_commission_pct: float = 0.001
    backtest_slippage_pct: float = 0.0005
    backtest_initial_cash: float = 10000.0

    # --- Risk Engine (Multi-Layer) ---
    # Portfolio Risk Layer
    risk_max_positions: int = 10
    risk_max_single_stock_pct: float = 0.20
    risk_max_sector_exposure_pct: float = 0.40
    risk_max_correlation: float = 0.80
    risk_max_gross_exposure_pct: float = 2.00
    risk_max_net_exposure_pct: float = 1.00
    risk_max_var_pct: float = 0.05
    risk_max_portfolio_heat_pct: float = 0.10
    risk_portfolio_layer_enabled: bool = True

    # Account Risk Layer
    risk_max_daily_loss_pct: float = 0.03
    risk_max_weekly_loss_pct: float = 0.07
    risk_max_drawdown_pct: float = 0.15
    risk_min_cash_reserve_pct: float = 0.20
    risk_pdt_account_threshold: float = 25000.0
    risk_consecutive_loss_limit: int = 5
    risk_daily_trade_limit: int = 20
    risk_account_layer_enabled: bool = True

    # Exposure Risk Layer
    risk_require_stop_loss: bool = True
    risk_max_adv_pct: float = 0.01
    risk_max_trade_concentration_pct: float = 0.05
    risk_max_overnight_exposure_pct: float = 0.60
    risk_earnings_blackout_days: int = 1
    risk_high_vol_threshold: float = 0.03
    risk_high_vol_size_reduction: float = 0.50
    risk_exposure_layer_enabled: bool = True

    # Execution Risk Layer
    risk_max_spread_pct: float = 0.005
    risk_min_volume: int = 10000
    risk_max_slippage_pct: float = 0.003
    risk_max_orders_per_minute: int = 10
    risk_cooldown_after_loss_minutes: int = 5
    risk_large_loss_threshold_pct: float = 0.01
    risk_execution_layer_enabled: bool = True

    # --- Automation Scheduler ---
    auto_backtest_interval_hours: int = 6       # Run backtests every N hours (0=disabled)
    auto_train_interval_hours: int = 24         # Retrain ML every N hours (0=disabled)
    auto_sweep_interval_hours: int = 12         # Parameter sweep every N hours (0=disabled)
    auto_backtest_symbols: str = ""             # Override symbols for backtest (empty=use trading_symbols)
    auto_train_bars: int = 1000                 # Bars to use for training

    # --- Asset-Class Scheduler ---
    operational_mode: OperationalMode = OperationalMode.PAPER
    scheduler_equity_symbols: str = "AAPL,MSFT,GOOGL,AMZN,NVDA"
    scheduler_crypto_symbols: str = "BTC/USD,ETH/USD,SOL/USD,AAVE/USD,ADA/USD"
    scheduler_research_cycle_hours: int = 24    # Research cycle interval
    scheduler_data_accumulation_threshold: int = 100  # Bars before triggering backtest

    # --- Prediction Registry ---
    prediction_registry_storage_path: str = "data_cache/predictions"
    prediction_default_horizon_bars: int = 24   # Default prediction horizon
    prediction_metrics_window: int = 200        # Rolling window for metrics

    # --- Retraining Triggers ---
    retraining_min_outcomes: int = 500          # Min validated outcomes before retraining
    retraining_drift_threshold: float = 0.15    # Drift score threshold
    retraining_max_frequency_hours: int = 48    # Min hours between retrains
    retraining_scheduled_interval_hours: int = 168  # Fallback weekly schedule

    # --- Portfolio Allocator ---
    portfolio_correlation_threshold: float = 0.70  # Correlation filter threshold
    portfolio_max_sector_concentration: float = 0.40  # Max sector weight
    portfolio_cash_reserve_min: float = 0.10    # Minimum cash reserve
    portfolio_max_kelly_fraction: float = 0.25  # Kelly criterion cap

    # --- Shadow Deployment ---
    shadow_min_period_hours: int = 72           # Minimum shadow deployment duration
    shadow_min_predictions: int = 100           # Min predictions before evaluation
    shadow_comparison_threshold: float = 0.05   # Max performance gap allowed

    # --- Backtest Triggers ---
    backtest_trigger_data_threshold: int = 100  # New bars before auto-backtest
    backtest_trigger_param_change: bool = True  # Trigger on parameter changes
    backtest_trigger_drift_threshold: float = 0.12  # Drift score to trigger

    # --- Execution Simulator ---
    enable_execution_simulator: bool = True     # Enable realistic execution simulation
    execution_simulator_preset: str = "realistic"  # "realistic", "conservative", or "ideal"

    # --- Model Promotion Gate ---
    model_min_sharpe_ratio: float = 0.5
    model_max_drawdown_pct: float = 0.20
    model_min_expectancy: float = 0.0           # Positive expectancy required
    model_min_precision: float = 0.50
    model_min_cv_accuracy: float = 0.52
    model_min_walk_forward_sharpe: float = 0.0
    model_min_monte_carlo_prob_profit: float = 0.50
    model_min_backtest_trades: int = 30

    # --- Self-Healing ---
    model_max_retries: int = 3                  # Max training retries on failure
    model_retry_backoff_seconds: float = 60.0   # Base backoff between retries
    model_stale_threshold_hours: float = 168.0  # 7 days — trigger retraining if older
    model_underperformance_window_hours: float = 24.0  # Window to detect underperformance after promotion

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # --- Computed Properties ---

    @property
    def api_key(self) -> str:
        if self.trading_mode == TradingMode.LIVE:
            return self.alpaca_live_api_key
        return self.alpaca_paper_api_key

    @property
    def secret_key(self) -> str:
        if self.trading_mode == TradingMode.LIVE:
            return self.alpaca_live_secret_key
        return self.alpaca_paper_secret_key

    @property
    def base_url(self) -> str:
        if self.trading_mode == TradingMode.LIVE:
            return self.alpaca_live_base_url
        return self.alpaca_paper_base_url

    @property
    def symbols_list(self) -> list[str]:
        return [s.strip() for s in self.trading_symbols.split(",") if s.strip()]

    @property
    def multi_strategies_parsed(self) -> list[dict]:
        """Parse multi_strategy_config into list of strategy dicts.
        
        Format: "name:symbols:timeframe:interval:weight;..." 
        Returns: [{"name": ..., "symbols": [...], "timeframe": ..., "interval": int, "weight": float}]
        """
        if not self.multi_strategy_config:
            return []
        strategies = []
        for entry in self.multi_strategy_config.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(":")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            symbols = [s.strip() for s in parts[1].split(",") if s.strip()]
            timeframe = parts[2].strip()
            interval = int(parts[3]) if len(parts) > 3 else 60
            weight = float(parts[4]) if len(parts) > 4 else 1.0
            strategies.append({
                "name": name,
                "symbols": symbols,
                "timeframe": timeframe,
                "interval": interval,
                "weight": weight,
            })
        return strategies

    @property
    def is_paper(self) -> bool:
        return self.trading_mode == TradingMode.PAPER

    @field_validator("trading_mode", mode="before")
    @classmethod
    def validate_trading_mode(cls, v) -> str:
        if not v or (isinstance(v, str) and not v.strip()):
            return TradingMode.PAPER
        return v

    @field_validator("database_url", mode="before")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v or not v.strip():
            return "sqlite:///data_cache/trading.db"
        return v

    @field_validator("max_position_size_pct", "max_daily_loss_pct", "max_portfolio_exposure", "max_single_stock_pct", "default_stop_loss_pct", "default_take_profit_pct")
    @classmethod
    def validate_percentage(cls, v: float) -> float:
        if not 0 < v <= 1.0:
            raise ValueError(f"Percentage must be between 0 and 1.0, got {v}")
        return v

    @field_validator("max_leverage")
    @classmethod
    def validate_leverage(cls, v: float) -> float:
        if v <= 0 or v > 10:
            raise ValueError(f"Leverage must be between 0 and 10, got {v}")
        return v

    @field_validator("intelligence_min_trade_score")
    @classmethod
    def validate_trade_score_threshold(cls, v: float) -> float:
        if v < 0 or v > 100:
            raise ValueError(f"Intelligence trade score must be in [0,100], got {v}")
        return v


# Singleton instance
settings = Settings()

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
    trading_symbols: str = "AAPL,MSFT,GOOGL,AMZN,NVDA"
    timeframe: str = "1Hour"
    lookback_bars: int = 200

    # --- Strategy Parameters (Momentum) ---
    momentum_fast_ema: int = 12
    momentum_slow_ema: int = 26
    momentum_rsi_period: int = 14
    momentum_rsi_oversold: int = 30
    momentum_rsi_overbought: int = 70
    momentum_atr_period: int = 14
    momentum_atr_sl_mult: float = 2.0
    momentum_atr_tp_mult: float = 3.0

    # --- Strategy Parameters (Mean Reversion) ---
    mean_rev_bb_period: int = 20
    mean_rev_bb_std: float = 2.0
    mean_rev_zscore_entry: float = 2.0
    mean_rev_zscore_exit: float = 0.5
    mean_rev_rsi_period: int = 14

    # --- ML ---
    ml_model_path: str = "models/latest_model.joblib"
    ml_retrain_interval_hours: int = 24
    ml_min_confidence: float = 0.65

    # --- Notifications ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""
    notify_on_trade: bool = True
    notify_on_error: bool = True
    notify_on_signal: bool = False

    # --- Database ---
    database_url: str = "sqlite:///data_cache/trading.db"

    # --- Logging ---
    log_level: str = "INFO"
    log_file: str = "logs/trader.log"

    # --- Backtesting ---
    backtest_commission_pct: float = 0.001
    backtest_slippage_pct: float = 0.0005
    backtest_initial_cash: float = 10000.0

    # --- Automation Scheduler ---
    auto_backtest_interval_hours: int = 6       # Run backtests every N hours (0=disabled)
    auto_train_interval_hours: int = 24         # Retrain ML every N hours (0=disabled)
    auto_sweep_interval_hours: int = 12         # Parameter sweep every N hours (0=disabled)
    auto_backtest_symbols: str = ""             # Override symbols for backtest (empty=use trading_symbols)
    auto_train_bars: int = 1000                 # Bars to use for training

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
    def is_paper(self) -> bool:
        return self.trading_mode == TradingMode.PAPER

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


# Singleton instance
settings = Settings()

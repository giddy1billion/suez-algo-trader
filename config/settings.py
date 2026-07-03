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
    """Application configuration — loaded from .env + environment variables."""

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

    # --- Risk Management ---
    max_position_size_pct: float = 0.02
    max_daily_loss_pct: float = 0.05
    max_portfolio_exposure: float = 0.80
    max_single_stock_pct: float = 0.15
    max_leverage: float = 1.0
    default_stop_loss_pct: float = 0.03
    default_take_profit_pct: float = 0.06

    # --- Strategy ---
    active_strategy: str = "momentum"
    trading_symbols: str = "AAPL,MSFT,GOOGL,AMZN,NVDA"
    timeframe: str = "1Hour"
    lookback_bars: int = 200

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

    # --- Database ---
    database_url: str = "sqlite:///data_cache/trading.db"

    # --- Logging ---
    log_level: str = "INFO"
    log_file: str = "logs/trader.log"

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

    @field_validator("max_position_size_pct", "max_daily_loss_pct", "max_portfolio_exposure")
    @classmethod
    def validate_percentage(cls, v: float) -> float:
        if not 0 < v <= 1.0:
            raise ValueError(f"Percentage must be between 0 and 1.0, got {v}")
        return v


# Singleton instance
settings = Settings()

"""
Strongly Typed Configuration Objects — Immutable configuration models.

Rather than scattered string lookups like `runtime_config("risk.max_daily_loss")`,
services depend on typed, validated configuration objects that provide:
- Type safety
- IDE autocompletion
- Validation at construction time
- Immutability after construction
"""

from dataclasses import dataclass
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RiskConfig:
    """Strongly typed risk management configuration."""

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
    emergency_stop_loss: float = 0.10
    drawdown_limit: float = 0.15


@dataclass(frozen=True)
class TradingConfig:
    """Strongly typed trading configuration."""

    trading_mode: str = "paper"
    trading_interval: int = 60
    max_consecutive_errors: int = 5
    error_cooldown_seconds: int = 300
    active_strategy: str = "momentum"
    trading_symbols: str = "AAPL,MSFT,GOOGL"
    timeframe: str = "1Hour"
    lookback_bars: int = 200
    enable_auto_trading: bool = True

    @property
    def symbols_list(self) -> list[str]:
        """Parse comma-separated symbols into list."""
        return [s.strip() for s in self.trading_symbols.split(",") if s.strip()]


@dataclass(frozen=True)
class MLConfig:
    """Strongly typed ML/AI configuration."""

    model_path: str = "models/latest_model.joblib"
    retrain_interval_hours: int = 24
    min_confidence: float = 0.65
    intelligence_enabled: bool = True
    intelligence_min_trade_score: float = 70.0
    intelligence_drift_window: int = 200
    intelligence_drift_min_samples: int = 50
    intelligence_drift_alert_drop: float = 0.12


@dataclass(frozen=True)
class TelegramConfig:
    """Strongly typed Telegram notification configuration."""

    bot_token: str = ""
    chat_id: str = ""
    notify_on_trade: bool = True
    notify_on_error: bool = True
    notify_on_signal: bool = False

    @property
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured."""
        return bool(self.bot_token and self.chat_id)


@dataclass(frozen=True)
class ExchangeConfig:
    """Strongly typed exchange configuration."""

    paper_api_key: str = ""
    paper_secret_key: str = ""
    paper_base_url: str = "https://paper-api.alpaca.markets"
    live_api_key: str = ""
    live_secret_key: str = ""
    live_base_url: str = "https://api.alpaca.markets"
    data_feed: str = "iex"

    @property
    def is_live_configured(self) -> bool:
        """Check if live trading credentials are configured."""
        return bool(self.live_api_key and self.live_secret_key)

    @property
    def is_paper_configured(self) -> bool:
        """Check if paper trading credentials are configured."""
        return bool(self.paper_api_key and self.paper_secret_key)


@dataclass(frozen=True)
class RiskEngineConfig:
    """Strongly typed risk engine multi-layer configuration."""

    # Portfolio Layer
    portfolio_max_positions: int = 10
    portfolio_max_single_stock_pct: float = 0.20
    portfolio_max_sector_exposure_pct: float = 0.40
    portfolio_max_correlation: float = 0.80
    portfolio_layer_enabled: bool = True

    # Account Layer
    account_max_daily_loss_pct: float = 0.03
    account_max_weekly_loss_pct: float = 0.07
    account_max_drawdown_pct: float = 0.15
    account_min_cash_reserve_pct: float = 0.20
    account_layer_enabled: bool = True

    # Exposure Layer
    exposure_require_stop_loss: bool = True
    exposure_max_overnight_pct: float = 0.60
    exposure_layer_enabled: bool = True

    # Execution Layer
    execution_max_spread_pct: float = 0.005
    execution_min_volume: int = 10000
    execution_max_slippage_pct: float = 0.003
    execution_layer_enabled: bool = True


@dataclass(frozen=True)
class BacktestConfig:
    """Strongly typed backtesting configuration."""

    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    initial_cash: float = 10000.0
    enable_execution_simulator: bool = True
    execution_simulator_preset: str = "realistic"


# ─── Factory Functions ────────────────────────────────────────────────────────


def build_risk_config(config_service: Optional[Any] = None, settings: Optional[Any] = None) -> RiskConfig:
    """Build a RiskConfig from ConfigurationService or Settings."""
    if config_service:
        return RiskConfig(
            max_position_size_pct=config_service.get_float("risk", "max_position_size_pct", 0.02),
            max_daily_loss_pct=config_service.get_float("risk", "max_daily_loss_pct", 0.05),
            max_portfolio_exposure=config_service.get_float("risk", "max_portfolio_exposure", 0.80),
            max_single_stock_pct=config_service.get_float("risk", "max_single_stock_pct", 0.15),
            max_leverage=config_service.get_float("risk", "max_leverage", 1.0),
            max_open_positions=config_service.get_int("risk", "max_open_positions", 20),
            max_orders_per_day=config_service.get_int("risk", "max_orders_per_day", 100),
            max_correlated_positions=config_service.get_int("risk", "max_correlated_positions", 3),
            default_stop_loss_pct=config_service.get_float("risk", "default_stop_loss_pct", 0.03),
            default_take_profit_pct=config_service.get_float("risk", "default_take_profit_pct", 0.06),
            emergency_stop_loss=config_service.get_float("risk", "emergency_stop_loss", 0.10),
            drawdown_limit=config_service.get_float("risk", "drawdown_limit", 0.15),
        )
    if settings:
        return RiskConfig(
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
        )
    return RiskConfig()


def build_trading_config(config_service: Optional[Any] = None, settings: Optional[Any] = None) -> TradingConfig:
    """Build a TradingConfig from ConfigurationService or Settings."""
    if config_service:
        return TradingConfig(
            trading_mode=config_service.get_str("trading", "trading_mode", "paper"),
            trading_interval=config_service.get_int("trading", "trading_interval", 60),
            max_consecutive_errors=config_service.get_int("trading", "max_consecutive_errors", 5),
            error_cooldown_seconds=config_service.get_int("trading", "error_cooldown_seconds", 300),
            active_strategy=config_service.get_str("trading", "active_strategy", "momentum"),
            trading_symbols=config_service.get_str("trading", "trading_symbols", "AAPL,MSFT,GOOGL"),
            timeframe=config_service.get_str("trading", "timeframe", "1Hour"),
            lookback_bars=config_service.get_int("trading", "lookback_bars", 200),
            enable_auto_trading=config_service.get_bool("trading", "enable_auto_trading", True),
        )
    if settings:
        return TradingConfig(
            trading_mode=settings.trading_mode.value if hasattr(settings.trading_mode, "value") else str(settings.trading_mode),
            trading_interval=settings.trading_interval,
            max_consecutive_errors=settings.max_consecutive_errors,
            error_cooldown_seconds=settings.error_cooldown_seconds,
            active_strategy=settings.active_strategy,
            trading_symbols=settings.trading_symbols,
            timeframe=settings.timeframe,
            lookback_bars=settings.lookback_bars,
        )
    return TradingConfig()


def build_ml_config(config_service: Optional[Any] = None, settings: Optional[Any] = None) -> MLConfig:
    """Build an MLConfig from ConfigurationService or Settings."""
    if config_service:
        return MLConfig(
            model_path=config_service.get_str("ml", "model_path", "models/latest_model.joblib"),
            retrain_interval_hours=config_service.get_int("ml", "retrain_interval_hours", 24),
            min_confidence=config_service.get_float("ml", "min_confidence", 0.65),
            intelligence_enabled=config_service.get_bool("intelligence", "enabled", True),
            intelligence_min_trade_score=config_service.get_float("intelligence", "min_trade_score", 70.0),
            intelligence_drift_window=config_service.get_int("intelligence", "drift_window", 200),
            intelligence_drift_min_samples=config_service.get_int("intelligence", "drift_min_samples", 50),
            intelligence_drift_alert_drop=config_service.get_float("intelligence", "drift_alert_drop", 0.12),
        )
    if settings:
        return MLConfig(
            model_path=settings.ml_model_path,
            retrain_interval_hours=settings.ml_retrain_interval_hours,
            min_confidence=settings.ml_min_confidence,
            intelligence_enabled=settings.intelligence_enabled,
            intelligence_min_trade_score=settings.intelligence_min_trade_score,
            intelligence_drift_window=settings.intelligence_drift_window,
            intelligence_drift_min_samples=settings.intelligence_drift_min_samples,
            intelligence_drift_alert_drop=settings.intelligence_drift_alert_drop,
        )
    return MLConfig()


def build_telegram_config(config_service: Optional[Any] = None, settings: Optional[Any] = None) -> TelegramConfig:
    """Build a TelegramConfig from ConfigurationService or Settings."""
    if config_service:
        return TelegramConfig(
            bot_token=config_service.get_str("notifications", "telegram_bot_token", ""),
            chat_id=config_service.get_str("notifications", "telegram_chat_id", ""),
            notify_on_trade=config_service.get_bool("notifications", "notify_on_trade", True),
            notify_on_error=config_service.get_bool("notifications", "notify_on_error", True),
            notify_on_signal=config_service.get_bool("notifications", "notify_on_signal", False),
        )
    if settings:
        return TelegramConfig(
            bot_token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
            notify_on_trade=settings.notify_on_trade,
            notify_on_error=settings.notify_on_error,
            notify_on_signal=settings.notify_on_signal,
        )
    return TelegramConfig()


def build_exchange_config(config_service: Optional[Any] = None, settings: Optional[Any] = None) -> ExchangeConfig:
    """Build an ExchangeConfig from ConfigurationService or Settings."""
    if settings:
        return ExchangeConfig(
            paper_api_key=settings.alpaca_paper_api_key,
            paper_secret_key=settings.alpaca_paper_secret_key,
            paper_base_url=settings.alpaca_paper_base_url,
            live_api_key=settings.alpaca_live_api_key,
            live_secret_key=settings.alpaca_live_secret_key,
            live_base_url=settings.alpaca_live_base_url,
            data_feed=settings.alpaca_data_feed,
        )
    return ExchangeConfig()

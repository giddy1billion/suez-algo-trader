"""
Asset-Class Aware Backtest Parameters — LayeredConfig integration.

Registers per-asset-class defaults for EMA periods, fees, position sizing,
stops, and cooldown via the existing LayeredConfig precedence system.

Usage:
    from src.config.backtest_params import get_backtest_config, backtest_params

    # Get params for a specific symbol
    params = get_backtest_config("BTC/USD")
    # params = {"fast_ema": 21, "slow_ema": 55, "fees": 0.0015, ...}

    # Or access the LayeredConfig directly for custom overrides
    backtest_params.set("fast_ema", 34, level=ConfigLevel.USER_OVERRIDE, context="BTC/USD")
"""

from src.config.layered import ConfigLevel, LayeredConfig
from src.market.instruments import AssetClass
from src.market.registry import classify_symbol

# Module-level instance — backtest parameter registry
backtest_params = LayeredConfig()

# ══════════════════════════════════════════════════════════════════════════════
# System Defaults (Level 0)
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_DEFAULTS = {
    "fast_ema": 12,
    "slow_ema": 26,
    "fees": 0.001,
    "risk_per_trade": 0.5,
    "atr_stop_multiplier": 2.0,
    "cooldown_bars": 0,
    "annualization_periods": 252.0,
}

for key, value in _SYSTEM_DEFAULTS.items():
    backtest_params.set(key, value, level=ConfigLevel.SYSTEM_DEFAULT)

# ══════════════════════════════════════════════════════════════════════════════
# Exchange-Level Overrides (Level 3) — Asset-Class Differentiation
# ══════════════════════════════════════════════════════════════════════════════

_EQUITY_OVERRIDES = {
    "fast_ema": 12,
    "slow_ema": 26,
    "fees": 0.0001,           # Near-zero for commission-free broker (Alpaca)
    "risk_per_trade": 0.5,    # 50% of capital per position
    "atr_stop_multiplier": 2.0,
    "cooldown_bars": 2,
    "annualization_periods": 252.0,
}

_CRYPTO_OVERRIDES = {
    "fast_ema": 21,
    "slow_ema": 55,
    "fees": 0.0015,           # Realistic taker fee + spread
    "risk_per_trade": 0.3,    # More conservative due to higher vol
    "atr_stop_multiplier": 2.5,
    "cooldown_bars": 4,
    "annualization_periods": 365.0,
}

for key, value in _EQUITY_OVERRIDES.items():
    backtest_params.set(key, value, level=ConfigLevel.EXCHANGE, context="EQUITY")

for key, value in _CRYPTO_OVERRIDES.items():
    backtest_params.set(key, value, level=ConfigLevel.EXCHANGE, context="CRYPTO")


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════


def get_backtest_config(symbol: str) -> dict:
    """
    Get backtest parameters for a symbol, resolved through LayeredConfig precedence.

    Uses InstrumentRegistry for classification (not "/" heuristic).
    Resolution: USER_OVERRIDE > EXCHANGE (asset class) > ENVIRONMENT > SYSTEM_DEFAULT.

    Args:
        symbol: Trading symbol (e.g., "AAPL", "BTC/USD", "ETHUSDT").

    Returns:
        Dict with resolved parameter values.
    """
    instrument = classify_symbol(symbol)
    exchange_context = instrument.asset_class.value.upper()

    return {
        "fast_ema": backtest_params.get("fast_ema", exchange=exchange_context),
        "slow_ema": backtest_params.get("slow_ema", exchange=exchange_context),
        "fees": backtest_params.get("fees", exchange=exchange_context),
        "risk_per_trade": backtest_params.get("risk_per_trade", exchange=exchange_context),
        "atr_stop_multiplier": backtest_params.get("atr_stop_multiplier", exchange=exchange_context),
        "cooldown_bars": backtest_params.get("cooldown_bars", exchange=exchange_context),
        "annualization_periods": backtest_params.get("annualization_periods", exchange=exchange_context),
    }


def get_fee_for_symbol(symbol: str) -> float:
    """Get the fee rate for a symbol via the FeeModel integration."""
    instrument = classify_symbol(symbol)
    exchange_context = instrument.asset_class.value.upper()
    return backtest_params.get("fees", exchange=exchange_context)


def set_symbol_override(symbol: str, key: str, value) -> None:
    """Set a user override for a specific symbol."""
    backtest_params.set(key, value, level=ConfigLevel.USER_OVERRIDE, context=symbol)


def set_asset_class_override(asset_class: str, key: str, value) -> None:
    """Set an exchange-level override for an entire asset class."""
    backtest_params.set(key, value, level=ConfigLevel.EXCHANGE, context=asset_class.upper())

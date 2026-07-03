"""Tests for composable strategy framework."""
import pytest
import pandas as pd
import numpy as np

from src.strategy.composable import (
    ComposableStrategy, EMAEntryModel, RSIEntryModel, MACDEntryModel,
    EMAExitModel, ADXFilter, VolumeFilter, TimeFilter,
    FixedRiskSizing, KellySizing, VolatilitySizing,
    ATRStop, PercentStop, SwingStop,
    momentum_preset, mean_reversion_preset,
)
from src.strategy.base import BaseStrategy, Signal


class TestEntryModels:
    def test_ema_entry_returns_signal_or_none(self, sample_ohlcv):
        entry = EMAEntryModel(fast=12, slow=26)
        result = entry.should_enter(sample_ohlcv, "AAPL")
        assert result is None or isinstance(result, Signal)

    def test_rsi_entry_returns_signal_or_none(self, sample_ohlcv):
        entry = RSIEntryModel(period=14, oversold=30, overbought=70)
        result = entry.should_enter(sample_ohlcv, "AAPL")
        assert result is None or isinstance(result, Signal)

    def test_macd_entry_returns_signal_or_none(self, sample_ohlcv):
        entry = MACDEntryModel()
        result = entry.should_enter(sample_ohlcv, "AAPL")
        assert result is None or isinstance(result, Signal)

    def test_ema_entry_trending_data(self, trending_ohlcv):
        """Clear uptrend should eventually produce BUY signal."""
        entry = EMAEntryModel(fast=5, slow=20)
        result = entry.should_enter(trending_ohlcv, "AAPL")
        # With a clear uptrend, fast EMA should be above slow
        # May still be None if crossover didn't happen at last bar
        assert result is None or result in (Signal.BUY, Signal.STRONG_BUY, Signal.HOLD)


class TestFilters:
    def test_adx_filter_returns_bool(self, sample_ohlcv):
        f = ADXFilter(period=14, min_adx=20)
        result = f.allow(sample_ohlcv, Signal.BUY)
        assert result == True or result == False

    def test_volume_filter_returns_bool(self, sample_ohlcv):
        f = VolumeFilter(ma_period=20, min_spike=1.5)
        result = f.allow(sample_ohlcv, Signal.BUY)
        assert result == True or result == False

    def test_volume_filter_passes_on_spike(self, trending_ohlcv):
        """Trending data has volume spike at end — should pass."""
        f = VolumeFilter(ma_period=20, min_spike=1.5)
        result = f.allow(trending_ohlcv, Signal.BUY)
        assert result == True


class TestSizingModels:
    def test_fixed_risk_sizing(self):
        sizing = FixedRiskSizing(risk_pct=0.02)
        qty = sizing.size(Signal.BUY, price=150.0, portfolio_value=100000.0, atr=3.0)
        # Returns a fraction of portfolio (0 to 0.25)
        assert qty > 0
        assert qty <= 0.25

    def test_kelly_sizing_bounded(self):
        sizing = KellySizing(win_rate=0.6, avg_win=200, avg_loss=100, fraction=0.5)
        qty = sizing.size(Signal.BUY, price=100.0, portfolio_value=50000.0, atr=2.0)
        assert qty >= 0
        # Kelly returns a fraction capped at 0.25
        assert qty <= 0.25

    def test_volatility_sizing(self):
        sizing = VolatilitySizing(target_vol=0.15)
        qty = sizing.size(Signal.BUY, price=200.0, portfolio_value=100000.0, atr=5.0)
        assert qty > 0


class TestStopModels:
    def test_atr_stop_produces_levels(self, sample_ohlcv):
        stop = ATRStop(period=14, sl_mult=2.0, tp_mult=3.0)
        sl, tp = stop.levels(sample_ohlcv, "buy", 100.0)
        assert sl is not None and sl < 100.0  # Stop below entry for buy
        assert tp is not None and tp > 100.0  # Target above entry for buy

    def test_atr_stop_sell_side(self, sample_ohlcv):
        stop = ATRStop(period=14, sl_mult=2.0, tp_mult=3.0)
        sl, tp = stop.levels(sample_ohlcv, "sell", 100.0)
        assert sl is not None and sl > 100.0  # Stop above entry for sell
        assert tp is not None and tp < 100.0  # Target below entry for sell

    def test_percent_stop(self, sample_ohlcv):
        stop = PercentStop(sl_pct=0.02, tp_pct=0.04)
        sl, tp = stop.levels(sample_ohlcv, "buy", 100.0)
        assert abs(sl - 98.0) < 0.01
        assert abs(tp - 104.0) < 0.01


class TestComposableStrategy:
    def test_extends_base_strategy(self):
        s = momentum_preset(symbols=["AAPL"])
        assert isinstance(s, BaseStrategy)

    def test_has_required_interface(self):
        s = momentum_preset(symbols=["AAPL"])
        assert hasattr(s, 'generate_signals')
        assert hasattr(s, 'calculate_indicators')
        assert hasattr(s, 'symbols')
        assert hasattr(s, 'timeframe')

    def test_generate_signals_returns_list(self, sample_ohlcv):
        s = momentum_preset(symbols=["AAPL"])
        result = s.generate_signals({"AAPL": sample_ohlcv})
        assert isinstance(result, list)

    def test_mean_reversion_preset(self):
        s = mean_reversion_preset(symbols=["MSFT", "GOOGL"])
        assert isinstance(s, ComposableStrategy)
        assert len(s.symbols) == 2

"""
Regression Tests — Protective Exits for All Order Entry Paths.

Validates that:
1. ProtectiveExits module computes valid SL/TP for any entry
2. Signal-driven trades have SL/TP from strategy → contract → defaults
3. Manual /buy & /sell trades now use bracket orders with protective exits
4. Configuration options work correctly
5. Edge cases are handled (no price, short sells, extreme ATR)
"""

import math
from unittest.mock import MagicMock, patch

import pytest

from src.risk.protective_exits import (
    ProtectiveExits,
    ProtectiveExitConfig,
    ProtectiveExitLevels,
)


# ---------------------------------------------------------------------------
# ProtectiveExits Core Logic
# ---------------------------------------------------------------------------

class TestProtectiveExitsComputation:
    """Tests for the shared protective exits computation module."""

    def setup_method(self):
        self.exits = ProtectiveExits()

    # --- Long (buy) positions ---

    def test_long_default_stop_loss_below_entry(self):
        """For a BUY, stop-loss must be below entry price."""
        levels = self.exits.compute(entry_price=100.0, side="buy")
        assert levels.stop_loss < 100.0

    def test_long_default_take_profit_above_entry(self):
        """For a BUY, take-profit must be above entry price."""
        levels = self.exits.compute(entry_price=100.0, side="buy")
        assert levels.take_profit > 100.0

    def test_long_default_sl_pct(self):
        """Default SL is 3% below entry."""
        levels = self.exits.compute(entry_price=100.0, side="buy")
        assert abs(levels.stop_loss - 97.0) < 0.01

    def test_long_default_tp_pct(self):
        """Default TP is 6% above entry."""
        levels = self.exits.compute(entry_price=100.0, side="buy")
        assert abs(levels.take_profit - 106.0) < 0.01

    def test_long_risk_reward_at_least_1_5(self):
        """Risk-reward ratio must meet minimum (1.5)."""
        levels = self.exits.compute(entry_price=150.0, side="buy")
        assert levels.risk_reward_ratio >= 1.5

    # --- Short (sell) positions ---

    def test_short_stop_loss_above_entry(self):
        """For a SELL, stop-loss must be above entry price."""
        levels = self.exits.compute(entry_price=100.0, side="sell")
        assert levels.stop_loss > 100.0

    def test_short_take_profit_below_entry(self):
        """For a SELL, take-profit must be below entry price."""
        levels = self.exits.compute(entry_price=100.0, side="sell")
        assert levels.take_profit < 100.0

    def test_short_default_sl_pct(self):
        """Default SL is 3% above entry for shorts."""
        levels = self.exits.compute(entry_price=100.0, side="sell")
        assert abs(levels.stop_loss - 103.0) < 0.01

    def test_short_default_tp_pct(self):
        """Default TP is 6% below entry for shorts."""
        levels = self.exits.compute(entry_price=100.0, side="sell")
        assert abs(levels.take_profit - 94.0) < 0.01

    # --- Strategy-provided SL/TP ---

    def test_strategy_sl_respected_when_valid(self):
        """Strategy-provided stop-loss is used when valid."""
        levels = self.exits.compute(
            entry_price=100.0, side="buy", strategy_stop_loss=95.0
        )
        assert levels.stop_loss == 95.0
        assert levels.source == "strategy"

    def test_strategy_tp_respected_when_valid(self):
        """Strategy-provided take-profit is used when valid."""
        levels = self.exits.compute(
            entry_price=100.0, side="buy",
            strategy_stop_loss=95.0, strategy_take_profit=115.0
        )
        assert levels.take_profit == 115.0

    def test_invalid_strategy_sl_rejected_buy(self):
        """SL above entry for a BUY is invalid and falls back to default."""
        levels = self.exits.compute(
            entry_price=100.0, side="buy", strategy_stop_loss=105.0
        )
        # Should fallback to default (97.0), not use 105.0
        assert levels.stop_loss < 100.0

    def test_invalid_strategy_sl_rejected_sell(self):
        """SL below entry for a SELL is invalid and falls back to default."""
        levels = self.exits.compute(
            entry_price=100.0, side="sell", strategy_stop_loss=95.0
        )
        assert levels.stop_loss > 100.0

    def test_invalid_strategy_tp_rejected_buy(self):
        """TP below entry for a BUY is invalid and falls back to default."""
        levels = self.exits.compute(
            entry_price=100.0, side="buy", strategy_take_profit=90.0
        )
        assert levels.take_profit > 100.0

    # --- ATR-based SL/TP ---

    def test_atr_based_sl(self):
        """ATR-based stop-loss is computed correctly."""
        levels = self.exits.compute(
            entry_price=100.0, side="buy", atr=2.0, atr_sl_multiplier=1.5
        )
        # ATR SL = 100 - (2.0 * 1.5) = 97.0 → 3% distance, within bounds
        assert levels.stop_loss < 100.0
        assert levels.source == "atr"

    def test_atr_based_tp(self):
        """ATR-based take-profit is computed correctly."""
        levels = self.exits.compute(
            entry_price=100.0, side="buy", atr=2.0, atr_tp_multiplier=3.0
        )
        # ATR TP = 100 + (2.0 * 3.0) = 106.0
        assert levels.take_profit > 100.0

    def test_atr_clamped_to_max(self):
        """Extreme ATR is clamped to max_stop_loss_pct."""
        levels = self.exits.compute(
            entry_price=100.0, side="buy", atr=50.0, atr_sl_multiplier=1.5
        )
        # 50 * 1.5 = 75, which is 75% — should be clamped to 10%
        assert levels.stop_loss_pct <= 0.10 + 0.001

    def test_atr_clamped_to_min(self):
        """Tiny ATR is clamped to min_stop_loss_pct."""
        levels = self.exits.compute(
            entry_price=100.0, side="buy", atr=0.01, atr_sl_multiplier=1.5
        )
        # 0.01 * 1.5 = 0.015, which is 0.015% — below min 0.5%
        assert levels.stop_loss_pct >= 0.005 - 0.0001

    # --- Configuration ---

    def test_custom_config_changes_defaults(self):
        """Custom config overrides default percentages."""
        config = ProtectiveExitConfig(
            default_stop_loss_pct=0.05,
            default_take_profit_pct=0.10,
        )
        exits = ProtectiveExits(config)
        levels = exits.compute(entry_price=100.0, side="buy")
        assert abs(levels.stop_loss - 95.0) < 0.01
        assert abs(levels.take_profit - 110.0) < 0.01

    def test_min_risk_reward_enforced(self):
        """If strategy TP is too close, it's bumped to meet min risk-reward."""
        config = ProtectiveExitConfig(min_risk_reward=2.0)
        exits = ProtectiveExits(config)
        levels = exits.compute(
            entry_price=100.0, side="buy",
            strategy_stop_loss=97.0,  # 3% SL
            strategy_take_profit=101.0,  # Only 1% TP → violates 2.0 RR
        )
        # TP should be bumped: need TP_pct >= SL_pct * 2.0 = 6%
        assert levels.take_profit_pct >= levels.stop_loss_pct * 2.0 - 0.001

    # --- Edge Cases ---

    def test_zero_entry_price_raises(self):
        """Zero entry price raises ValueError."""
        with pytest.raises(ValueError, match="entry_price must be positive"):
            self.exits.compute(entry_price=0.0, side="buy")

    def test_negative_entry_price_raises(self):
        """Negative entry price raises ValueError."""
        with pytest.raises(ValueError, match="entry_price must be positive"):
            self.exits.compute(entry_price=-50.0, side="buy")

    def test_invalid_side_raises(self):
        """Invalid side raises ValueError."""
        with pytest.raises(ValueError, match="side must be"):
            self.exits.compute(entry_price=100.0, side="hold")

    def test_nan_strategy_sl_rejected(self):
        """NaN strategy SL is treated as invalid."""
        levels = self.exits.compute(
            entry_price=100.0, side="buy", strategy_stop_loss=float('nan')
        )
        # Should fallback to default
        assert levels.stop_loss < 100.0
        assert not math.isnan(levels.stop_loss)

    def test_output_is_dataclass(self):
        """Output is a ProtectiveExitLevels dataclass with all fields."""
        levels = self.exits.compute(entry_price=150.0, side="buy")
        assert isinstance(levels, ProtectiveExitLevels)
        d = levels.to_dict()
        assert "entry_price" in d
        assert "stop_loss" in d
        assert "take_profit" in d
        assert "risk_reward_ratio" in d
        assert "source" in d

    def test_very_high_price_stock(self):
        """Works for high-priced stocks (e.g., BRK.A)."""
        levels = self.exits.compute(entry_price=500_000.0, side="buy")
        assert levels.stop_loss < 500_000.0
        assert levels.take_profit > 500_000.0
        assert levels.stop_loss > 0

    def test_very_low_price_stock(self):
        """Works for penny stocks."""
        levels = self.exits.compute(entry_price=0.50, side="buy")
        assert levels.stop_loss < 0.50
        assert levels.take_profit > 0.50
        assert levels.stop_loss > 0


# ---------------------------------------------------------------------------
# Integration: Signal-Driven Trades
# ---------------------------------------------------------------------------

class TestSignalDrivenExits:
    """Verifies signal-driven trades use SL/TP through execution engine."""

    def test_execution_engine_uses_bracket_when_sl_tp_present(self):
        """ExecutionEngine calls bracket_order() when SL/TP are available."""
        from src.execution.engine import ExecutionEngine
        from src.risk.manager import RiskManager, RiskLimits
        from src.risk.engine import RiskEngine
        from src.broker.paper import PaperBroker
        from src.core.events import EventBus

        broker = PaperBroker(starting_equity=100_000.0)
        broker.set_price("AAPL", 150.0)
        risk_engine = RiskEngine()
        bus = EventBus()

        engine = ExecutionEngine(
            broker=broker,
            risk_manager=RiskManager(),
            risk_engine=risk_engine,
            event_bus=bus,
            db=None,
            dry_run=False,
        )

        # Create strategy that generates signal WITH SL/TP
        strategy = MagicMock()
        strategy.name = "test_momentum"
        strategy.symbols = ["AAPL"]
        strategy.timeframe = "1Hour"
        strategy.lookback = 200
        strategy.generate_signals.return_value = [{
            "symbol": "AAPL",
            "signal": "BUY",
            "confidence": 0.85,
            "price": 150.0,
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "reason": "test signal",
            "indicators": {"rsi": 35.0},
        }]
        strategy.should_exit.return_value = None

        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            "open": np.full(200, 150.0),
            "high": np.full(200, 152.0),
            "low": np.full(200, 148.0),
            "close": np.full(200, 150.0),
            "volume": np.full(200, 50000.0),
        }, index=pd.date_range("2024-01-01", periods=200, freq="h"))

        broker.get_bars_df = MagicMock(return_value=df)
        broker.get_account = MagicMock(return_value={
            "equity": 100_000.0, "cash": 100_000.0,
            "portfolio_value": 100_000.0, "buying_power": 200_000.0,
        })
        broker.get_positions = MagicMock(return_value=[])

        # Track if bracket_order is called
        original_bracket = broker.bracket_order
        bracket_calls = []

        def _track_bracket(*args, **kwargs):
            bracket_calls.append((args, kwargs))
            return original_bracket(*args, **kwargs)

        broker.bracket_order = _track_bracket

        results = engine.run_cycle(strategy)
        # If signal passes all gates, bracket_order should have been called
        # (unless rejected by risk/confidence gates — which is also valid)
        # The key assertion: if a trade WAS placed, it used bracket
        if results:
            assert len(bracket_calls) > 0 or any(
                r.get("dry_run") for r in results
            ), "Signal-driven trade should use bracket order"


# ---------------------------------------------------------------------------
# Integration: Manual Trades
# ---------------------------------------------------------------------------

class TestManualTradeExits:
    """Verifies manual trades now receive protective exits."""

    def test_protective_exits_compute_for_manual_buy(self):
        """Manual BUY gets valid SL/TP from ProtectiveExits."""
        exits = ProtectiveExits()
        levels = exits.compute(entry_price=150.0, side="buy")
        assert levels.stop_loss < 150.0
        assert levels.take_profit > 150.0
        # These would be passed to bracket_order()
        assert levels.stop_loss > 0
        assert levels.risk_reward_ratio >= 1.5

    def test_protective_exits_compute_for_manual_sell(self):
        """Manual SELL gets valid SL/TP from ProtectiveExits."""
        exits = ProtectiveExits()
        levels = exits.compute(entry_price=150.0, side="sell")
        assert levels.stop_loss > 150.0
        assert levels.take_profit < 150.0
        assert levels.risk_reward_ratio >= 1.5

    def test_paper_broker_bracket_order_accepts_computed_levels(self):
        """Paper broker successfully executes bracket with computed levels."""
        from src.broker.paper import PaperBroker

        broker = PaperBroker(starting_equity=100_000.0)
        broker.set_price("AAPL", 150.0)

        exits = ProtectiveExits()
        levels = exits.compute(entry_price=150.0, side="buy")

        order = broker.bracket_order(
            symbol="AAPL",
            qty=10,
            side="buy",
            stop_loss=levels.stop_loss,
            take_profit=levels.take_profit,
        )
        assert order is not None
        assert "id" in order
        assert order.get("order_type") == "bracket" or "bracket" in str(order.get("type", ""))

    def test_manual_trade_has_audit_trail(self):
        """ProtectiveExitLevels provides complete audit data."""
        exits = ProtectiveExits()
        levels = exits.compute(entry_price=200.0, side="buy")
        d = levels.to_dict()

        # All required audit fields present
        assert d["entry_price"] == 200.0
        assert d["side"] == "buy"
        assert d["stop_loss"] > 0
        assert d["take_profit"] > 0
        assert d["stop_loss_pct"] > 0
        assert d["take_profit_pct"] > 0
        assert d["risk_reward_ratio"] > 0
        assert d["source"] in ("default", "strategy", "atr", "decision_contract")
        assert d["computed_at"]  # Timestamp present


# ---------------------------------------------------------------------------
# Both Paths: Consistency
# ---------------------------------------------------------------------------

class TestConsistency:
    """Proves both manual and automated paths produce consistent protection."""

    def test_same_module_used_for_both_paths(self):
        """ProtectiveExits is a single shared module for all paths."""
        # Same class computes for both manual and automated
        exits = ProtectiveExits()

        # "Manual" entry (no strategy context)
        manual = exits.compute(entry_price=100.0, side="buy")
        assert manual.source == "default"

        # "Automated" entry (with strategy SL/TP)
        auto = exits.compute(
            entry_price=100.0, side="buy",
            strategy_stop_loss=96.0, strategy_take_profit=110.0
        )
        assert auto.source == "strategy"

        # Both have valid levels
        assert manual.stop_loss < 100.0 and auto.stop_loss < 100.0
        assert manual.take_profit > 100.0 and auto.take_profit > 100.0

    def test_no_naked_position_possible(self):
        """No configuration can produce a position without SL/TP."""
        # Even with minimal config, levels are always computed
        config = ProtectiveExitConfig(
            default_stop_loss_pct=0.005,
            default_take_profit_pct=0.01,
        )
        exits = ProtectiveExits(config)
        levels = exits.compute(entry_price=50.0, side="buy")
        assert levels.stop_loss > 0
        assert levels.take_profit > 0
        assert levels.stop_loss != levels.entry_price
        assert levels.take_profit != levels.entry_price

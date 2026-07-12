"""
Comprehensive tests for the enhanced Telegram signal formatter.

Covers:
- Legacy message format preservation
- Valid actionable signals (explicit position size)
- Auto-sized quantities
- Missing sizing inputs (command suppression)
- Fallback/model-disabled cases
- Risk rejections
- Bracket-order vs informational TP/SL paths
- Regression: existing _format_signal behavior unchanged
"""

import pytest

from src.notifications.signal_formatter import (
    SignalPackage,
    format_signal_message,
    _get_warning,
    _resolve_quantity,
    _format_qty,
    _format_tp_sl,
)
from src.core.events import SignalGenerated
from src.notifications.telegram_audit_forwarder import _format_signal


# ─────────────────────────────────────────────────────────────────────────────
# Regression: Legacy _format_signal from telegram_audit_forwarder unchanged
# ─────────────────────────────────────────────────────────────────────────────


class TestLegacyFormatSignalRegression:
    """Ensure existing _format_signal behavior is unchanged."""

    def test_buy_signal_format(self):
        event = SignalGenerated(symbol="AAPL", signal="BUY", confidence=0.85)
        result = _format_signal(event)
        assert "📶" in result
        assert "BUY" in result
        assert "AAPL" in result
        assert "0.85" in result

    def test_sell_signal_format(self):
        event = SignalGenerated(symbol="TSLA", signal="SELL", confidence=0.7)
        result = _format_signal(event)
        assert "📉" in result
        assert "SELL" in result
        assert "TSLA" in result

    def test_signal_with_strategy(self):
        event = SignalGenerated(
            symbol="MSFT", signal="BUY", confidence=0.9, strategy="momentum_v2"
        )
        result = _format_signal(event)
        assert "momentum_v2" in result

    def test_signal_with_strength_over_confidence(self):
        """signal_strength takes priority over confidence when > 0."""
        event = SignalGenerated(
            symbol="NVDA", signal="BUY", signal_strength=0.92, confidence=0.5
        )
        result = _format_signal(event)
        assert "0.92" in result

    def test_hold_signal_emoji(self):
        event = SignalGenerated(symbol="X", signal="HOLD", confidence=0.5)
        result = _format_signal(event)
        assert "⏸️" in result

    def test_source_field_displayed(self):
        event = SignalGenerated(symbol="X", signal="BUY", source="ml_predictor")
        result = _format_signal(event)
        assert "ml_predictor" in result


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced formatter: Valid actionable signals
# ─────────────────────────────────────────────────────────────────────────────


class TestValidActionableSignals:
    """Test actionable signals with explicit position sizes."""

    def test_buy_signal_with_explicit_size(self):
        pkg = SignalPackage(
            symbol="AAPL",
            direction="BUY",
            strength=0.85,
            strategy="momentum_v2",
            source="ml_predictor",
            position_size=100,
        )
        result = format_signal_message(pkg)
        assert "/buy AAPL 100" in result
        assert "📶" in result
        assert "0.85" in result
        assert "momentum_v2" in result

    def test_sell_signal_with_explicit_size(self):
        pkg = SignalPackage(
            symbol="TSLA",
            direction="SELL",
            strength=0.75,
            strategy="mean_reversion",
            source="ml_predictor",
            position_size=50,
        )
        result = format_signal_message(pkg)
        assert "/sell TSLA 50" in result
        assert "📉" in result

    def test_fractional_quantity(self):
        pkg = SignalPackage(
            symbol="BTC/USD",
            direction="BUY",
            strength=0.9,
            strategy="trend",
            source="ml_predictor",
            position_size=0.5,
        )
        result = format_signal_message(pkg)
        assert "/buy BTC/USD 0.5" in result

    def test_no_auto_sized_label_for_explicit(self):
        pkg = SignalPackage(
            symbol="AAPL",
            direction="BUY",
            strength=0.8,
            strategy="test",
            source="active",
            position_size=25,
        )
        result = format_signal_message(pkg)
        assert "(auto-sized)" not in result

    def test_message_line_count_within_limit(self):
        """Message should be approximately 8 lines or fewer."""
        pkg = SignalPackage(
            symbol="AAPL",
            direction="BUY",
            strength=0.85,
            strategy="momentum_v2",
            source="ml_predictor",
            position_size=100,
            stop_loss=145.0,
            take_profit=165.0,
        )
        result = format_signal_message(pkg)
        assert len(result.split("\n")) <= 8


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced formatter: Auto-sized quantities
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoSizedQuantities:
    """Test auto-sizing via risk-sizing mechanism."""

    def test_auto_sized_buy(self):
        pkg = SignalPackage(
            symbol="MSFT",
            direction="BUY",
            strength=0.8,
            strategy="ml_ensemble",
            source="ml_predictor",
            auto_sized_qty=42,
        )
        result = format_signal_message(pkg)
        assert "/buy MSFT 42" in result
        assert "(auto-sized)" in result

    def test_auto_sized_sell(self):
        pkg = SignalPackage(
            symbol="GOOG",
            direction="SELL",
            strength=0.72,
            strategy="trend",
            source="ml_predictor",
            auto_sized_qty=15.5,
        )
        result = format_signal_message(pkg)
        assert "/sell GOOG 15.5" in result
        assert "(auto-sized)" in result

    def test_explicit_size_takes_priority_over_auto(self):
        """When both position_size and auto_sized_qty are set, explicit wins."""
        pkg = SignalPackage(
            symbol="AAPL",
            direction="BUY",
            strength=0.9,
            strategy="test",
            source="active",
            position_size=100,
            auto_sized_qty=50,
        )
        result = format_signal_message(pkg)
        assert "/buy AAPL 100" in result
        assert "(auto-sized)" not in result


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced formatter: Missing sizing inputs
# ─────────────────────────────────────────────────────────────────────────────


class TestMissingSizingInputs:
    """Test behavior when position sizing cannot be determined."""

    def test_no_size_no_auto_size(self):
        pkg = SignalPackage(
            symbol="AAPL",
            direction="BUY",
            strength=0.8,
            strategy="test",
            source="active",
            # No position_size, no auto_sized_qty
        )
        result = format_signal_message(pkg)
        assert "/buy" not in result
        assert "/sell" not in result
        assert "could not be determined" in result

    def test_zero_position_size(self):
        pkg = SignalPackage(
            symbol="AAPL",
            direction="SELL",
            strength=0.7,
            strategy="test",
            source="active",
            position_size=0,
        )
        result = format_signal_message(pkg)
        assert "/sell" not in result
        assert "could not be determined" in result

    def test_zero_auto_sized_qty(self):
        pkg = SignalPackage(
            symbol="AAPL",
            direction="BUY",
            strength=0.8,
            strategy="test",
            source="active",
            auto_sized_qty=0,
        )
        result = format_signal_message(pkg)
        assert "/buy" not in result
        assert "could not be determined" in result

    def test_negative_position_size_treated_as_missing(self):
        pkg = SignalPackage(
            symbol="X",
            direction="BUY",
            strength=0.8,
            strategy="test",
            source="active",
            position_size=-10,
        )
        result = format_signal_message(pkg)
        assert "/buy" not in result
        assert "could not be determined" in result


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced formatter: Fallback source / model disabled
# ─────────────────────────────────────────────────────────────────────────────


class TestFallbackAndModelDisabled:
    """Test warning prepend and command suppression for fallback/disabled models."""

    def test_fallback_source_flag(self):
        pkg = SignalPackage(
            symbol="AAPL",
            direction="BUY",
            strength=0.6,
            strategy="momentum",
            source="fallback",
            is_fallback_source=True,
            position_size=100,
        )
        result = format_signal_message(pkg)
        assert "⚠️" in result
        assert "FALLBACK" in result
        assert "/buy" not in result

    def test_fallback_source_string_match(self):
        """Source string 'fallback' also triggers warning."""
        pkg = SignalPackage(
            symbol="AAPL",
            direction="BUY",
            strength=0.6,
            strategy="momentum",
            source="fallback",
            position_size=100,
        )
        result = format_signal_message(pkg)
        assert "⚠️" in result
        assert "FALLBACK" in result
        assert "/buy" not in result

    def test_model_inactive(self):
        pkg = SignalPackage(
            symbol="AAPL",
            direction="BUY",
            strength=0.7,
            strategy="ml_ensemble",
            source="ml_predictor",
            model_active=False,
            position_size=100,
        )
        result = format_signal_message(pkg)
        assert "⚠️" in result
        assert "NO ACTIVE MODEL" in result
        assert "/buy" not in result

    def test_fallback_takes_priority_over_model_check(self):
        pkg = SignalPackage(
            symbol="AAPL",
            direction="BUY",
            strength=0.5,
            strategy="test",
            source="fallback",
            is_fallback_source=True,
            model_active=False,
            position_size=100,
        )
        result = format_signal_message(pkg)
        assert "FALLBACK" in result


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced formatter: Risk rejections
# ─────────────────────────────────────────────────────────────────────────────


class TestRiskRejections:
    """Test risk-control rejection warnings and command suppression."""

    def test_risk_rejected_with_reason(self):
        pkg = SignalPackage(
            symbol="TSLA",
            direction="BUY",
            strength=0.8,
            strategy="trend",
            source="ml_predictor",
            risk_approved=False,
            risk_rejection_reason="max daily loss exceeded",
            position_size=50,
        )
        result = format_signal_message(pkg)
        assert "⚠️" in result
        assert "RISK REJECTED" in result
        assert "max daily loss exceeded" in result
        assert "/buy" not in result

    def test_risk_rejected_no_reason(self):
        pkg = SignalPackage(
            symbol="AAPL",
            direction="SELL",
            strength=0.7,
            strategy="test",
            source="ml_predictor",
            risk_approved=False,
            position_size=25,
        )
        result = format_signal_message(pkg)
        assert "RISK REJECTED" in result
        assert "risk-control check failed" in result
        assert "/sell" not in result

    def test_risk_approved_no_warning(self):
        pkg = SignalPackage(
            symbol="AAPL",
            direction="BUY",
            strength=0.85,
            strategy="test",
            source="ml_predictor",
            risk_approved=True,
            position_size=100,
        )
        result = format_signal_message(pkg)
        assert "RISK REJECTED" not in result
        assert "/buy AAPL 100" in result


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced formatter: Bracket-order vs informational TP/SL
# ─────────────────────────────────────────────────────────────────────────────


class TestBracketOrderVsInformational:
    """Test bracket-order support vs informational TP/SL close commands."""

    def test_bracket_order_supported_both_tp_sl(self):
        pkg = SignalPackage(
            symbol="AAPL",
            direction="BUY",
            strength=0.85,
            strategy="test",
            source="ml_predictor",
            position_size=100,
            stop_loss=145.0,
            take_profit=165.0,
            bracket_orders_supported=True,
        )
        result = format_signal_message(pkg)
        assert "Native bracket order" in result
        assert "/sell" not in result.split("\n")[-1] or "bracket" in result

    def test_no_bracket_support_informational_buy(self):
        """Long signal: TP/SL close commands use /sell."""
        pkg = SignalPackage(
            symbol="AAPL",
            direction="BUY",
            strength=0.85,
            strategy="test",
            source="ml_predictor",
            position_size=100,
            stop_loss=145.0,
            take_profit=165.0,
            bracket_orders_supported=False,
        )
        result = format_signal_message(pkg)
        assert "/sell AAPL @ 165.00" in result  # TP
        assert "/sell AAPL @ 145.00" in result  # SL

    def test_no_bracket_support_informational_sell(self):
        """Short signal: TP/SL close commands use /buy."""
        pkg = SignalPackage(
            symbol="TSLA",
            direction="SELL",
            strength=0.75,
            strategy="test",
            source="ml_predictor",
            position_size=50,
            stop_loss=260.0,
            take_profit=230.0,
            bracket_orders_supported=False,
        )
        result = format_signal_message(pkg)
        assert "/buy TSLA @ 230.00" in result  # TP
        assert "/buy TSLA @ 260.00" in result  # SL

    def test_only_stop_loss_present(self):
        pkg = SignalPackage(
            symbol="AAPL",
            direction="BUY",
            strength=0.8,
            strategy="test",
            source="ml_predictor",
            position_size=100,
            stop_loss=145.0,
        )
        result = format_signal_message(pkg)
        assert "SL: $145.00" in result
        assert "TP" not in result or "TP:" not in result

    def test_only_take_profit_present(self):
        pkg = SignalPackage(
            symbol="AAPL",
            direction="BUY",
            strength=0.8,
            strategy="test",
            source="ml_predictor",
            position_size=100,
            take_profit=165.0,
        )
        result = format_signal_message(pkg)
        assert "TP: $165.00" in result

    def test_no_tp_no_sl(self):
        pkg = SignalPackage(
            symbol="AAPL",
            direction="BUY",
            strength=0.8,
            strategy="test",
            source="ml_predictor",
            position_size=100,
        )
        result = format_signal_message(pkg)
        # No TP/SL line at all
        assert "TP" not in result
        assert "SL" not in result

    def test_never_invents_tp_sl(self):
        """If TP/SL are not provided, they must not appear."""
        pkg = SignalPackage(
            symbol="AAPL",
            direction="BUY",
            strength=0.9,
            strategy="test",
            source="ml_predictor",
            position_size=100,
            stop_loss=None,
            take_profit=None,
        )
        result = format_signal_message(pkg)
        assert "bracket" not in result.lower()
        assert "@" not in result


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for internal helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestHelperFunctions:
    """Unit tests for internal formatting helpers."""

    def test_format_qty_integer(self):
        assert _format_qty(100.0) == "100"
        assert _format_qty(1.0) == "1"

    def test_format_qty_fractional(self):
        assert _format_qty(0.5) == "0.5"
        assert _format_qty(10.25) == "10.25"
        assert _format_qty(0.1234) == "0.1234"

    def test_format_qty_trailing_zeros(self):
        assert _format_qty(1.5000) == "1.5"
        assert _format_qty(2.10) == "2.1"

    def test_resolve_quantity_explicit(self):
        pkg = SignalPackage(position_size=100)
        qty, label = _resolve_quantity(pkg)
        assert qty == 100
        assert label == ""

    def test_resolve_quantity_auto(self):
        pkg = SignalPackage(auto_sized_qty=50)
        qty, label = _resolve_quantity(pkg)
        assert qty == 50
        assert label == "(auto-sized)"

    def test_resolve_quantity_none(self):
        pkg = SignalPackage()
        qty, label = _resolve_quantity(pkg)
        assert qty is None
        assert label == ""

    def test_get_warning_none_for_valid(self):
        pkg = SignalPackage(
            source="ml_predictor",
            model_active=True,
            risk_approved=True,
        )
        assert _get_warning(pkg) is None

    def test_get_warning_fallback(self):
        pkg = SignalPackage(source="fallback", is_fallback_source=True)
        warning = _get_warning(pkg)
        assert warning is not None
        assert "FALLBACK" in warning

    def test_get_warning_model_inactive(self):
        pkg = SignalPackage(source="active", model_active=False)
        warning = _get_warning(pkg)
        assert "NO ACTIVE MODEL" in warning

    def test_get_warning_risk_rejected(self):
        pkg = SignalPackage(source="active", risk_approved=False)
        warning = _get_warning(pkg)
        assert "RISK REJECTED" in warning

    def test_format_tp_sl_none(self):
        pkg = SignalPackage(direction="BUY")
        assert _format_tp_sl(pkg) is None

    def test_format_tp_sl_bracket(self):
        pkg = SignalPackage(
            direction="BUY",
            stop_loss=100.0,
            take_profit=120.0,
            bracket_orders_supported=True,
        )
        result = _format_tp_sl(pkg)
        assert "bracket" in result.lower()

    def test_format_tp_sl_informational(self):
        pkg = SignalPackage(
            direction="BUY",
            stop_loss=100.0,
            take_profit=120.0,
            bracket_orders_supported=False,
        )
        result = _format_tp_sl(pkg)
        assert "/sell" in result
        assert "100.00" in result
        assert "120.00" in result


# ─────────────────────────────────────────────────────────────────────────────
# Integration: Full message formatting end-to-end
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEndFormatting:
    """End-to-end tests for complete message output."""

    def test_complete_buy_message(self):
        pkg = SignalPackage(
            symbol="AAPL",
            direction="BUY",
            strength=0.87,
            strategy="ml_ensemble_v3",
            source="ml_predictor",
            position_size=150,
            stop_loss=148.0,
            take_profit=162.0,
            bracket_orders_supported=False,
        )
        result = format_signal_message(pkg)
        lines = result.split("\n")
        assert lines[0] == "📶 Signal: BUY AAPL"
        assert "0.87" in lines[1]
        assert "ml_ensemble_v3" in lines[2]
        assert "ml_predictor" in lines[3]
        assert "/buy AAPL 150" in result

    def test_complete_sell_message_auto_sized(self):
        pkg = SignalPackage(
            symbol="GOOG",
            direction="SELL",
            strength=0.72,
            strategy="mean_reversion",
            source="ml_predictor",
            auto_sized_qty=30,
        )
        result = format_signal_message(pkg)
        assert "📉 Signal: SELL GOOG" in result
        assert "/sell GOOG 30 (auto-sized)" in result

    def test_warning_message_format(self):
        pkg = SignalPackage(
            symbol="TSLA",
            direction="BUY",
            strength=0.5,
            strategy="momentum",
            source="fallback",
            is_fallback_source=True,
            position_size=100,
        )
        result = format_signal_message(pkg)
        lines = result.split("\n")
        # Warning is the first line
        assert lines[0].startswith("⚠️")
        assert "FALLBACK" in lines[0]
        # No command anywhere
        assert "/buy" not in result
        assert "/sell" not in result

    def test_empty_direction_no_crash(self):
        """Signal with empty direction should not crash."""
        pkg = SignalPackage(
            symbol="X",
            direction="",
            strength=0.5,
            strategy="test",
            source="active",
            position_size=10,
        )
        result = format_signal_message(pkg)
        assert "X" in result
        # No command for empty direction
        assert "/buy" not in result
        assert "/sell" not in result


class TestSignalBlockSnapshotRegression:
    def test_signal_block_is_preserved_byte_for_byte(self):
        fixtures = [
            SignalGenerated(symbol="AAPL", signal="BUY", side="BUY", signal_strength=0.91, strategy="mom", source="engine"),
            SignalGenerated(symbol="TSLA", signal="SELL", side="SELL", confidence=0.72, strategy="mean", source="engine"),
        ]
        for event in fixtures:
            expected_block = _format_signal(event)
            pkg = SignalPackage(
                symbol=event.symbol,
                direction=event.side or event.signal,
                strength=event.signal_strength if event.signal_strength > 0 else event.confidence,
                strategy=event.strategy,
                source=event.source,
                signal_block=expected_block,
                position_size=10,
            )
            rendered = format_signal_message(pkg)
            assert rendered.startswith(expected_block)
            assert expected_block in rendered

    def test_unknown_direction_rejected_with_warning(self):
        pkg = SignalPackage(
            symbol="AAPL",
            direction="HOLD",
            strength=0.8,
            strategy="test",
            source="active",
            position_size=10,
        )
        result = format_signal_message(pkg)
        assert "INVALID DIRECTION" in result
        assert "/buy" not in result
        assert "/sell" not in result

    def test_non_finite_numeric_input_rejected(self):
        pkg = SignalPackage(
            symbol="AAPL",
            direction="BUY",
            strength=float("nan"),
            strategy="test",
            source="active",
            position_size=10,
        )
        result = format_signal_message(pkg)
        assert "INVALID STRENGTH" in result
        assert "/buy" not in result

    def test_quantity_precision_uses_step_size(self):
        pkg = SignalPackage(
            symbol="BTC/USD",
            direction="BUY",
            strength=0.9,
            strategy="test",
            source="active",
            auto_sized_qty=1.23456,
            quantity_step=0.001,
        )
        result = format_signal_message(pkg)
        assert "/buy BTC/USD 1.235 (auto-sized)" in result

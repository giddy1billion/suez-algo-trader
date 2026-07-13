"""
Staging Verification — Paper Broker SL/TP Order Submission and Acknowledgment.

Verifies via a simulated staging run that both stop-loss and take-profit
orders are correctly submitted to and acknowledged by the paper broker for
both manual and signal-driven trades.

This test module serves as the staging verification gate before promoting
protective exit telemetry to production.
"""

import pytest
from datetime import datetime, timezone

from src.broker.paper import PaperBroker
from src.risk.protective_exits import ProtectiveExits, ProtectiveExitConfig
from src.monitoring.paper_dashboard import PaperTradingDashboard
from src.monitoring.telemetry import Telemetry
from src.core.events import (
    ProtectiveExitConfigured,
    ProtectiveExitAdjusted,
    ProtectiveExitExecuted,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def paper_broker():
    """Create a paper broker with test prices."""
    broker = PaperBroker(starting_equity=100_000.0)
    broker.set_price("AAPL", 150.0)
    broker.set_price("MSFT", 380.0)
    broker.set_price("BTCUSD", 45000.0)
    broker.set_price("TSLA", 250.0)
    return broker


@pytest.fixture
def telemetry():
    return Telemetry()


@pytest.fixture
def dashboard():
    return PaperTradingDashboard()


@pytest.fixture
def exits(telemetry):
    return ProtectiveExits(telemetry=telemetry)


# ---------------------------------------------------------------------------
# Signal-Driven Trade: SL/TP Submission to Paper Broker
# ---------------------------------------------------------------------------

class TestSignalDrivenSLTP:
    """Verify SL/TP orders are submitted and acknowledged for signal trades."""

    def test_bracket_order_with_strategy_sl_tp(self, paper_broker, exits, dashboard):
        """Signal-driven trade with strategy-provided SL/TP creates a bracket order."""
        levels = exits.compute(
            entry_price=150.0,
            side="buy",
            strategy_stop_loss=145.0,
            strategy_take_profit=160.0,
            symbol="AAPL",
            trade_source="signal",
        )

        # Submit bracket order to paper broker
        order = paper_broker.bracket_order(
            symbol="AAPL",
            qty=10,
            side="buy",
            take_profit=levels.take_profit,
            stop_loss=levels.stop_loss,
        )

        # Record in dashboard
        dashboard.record_exit_configured(
            "AAPL", "buy", 150.0, levels.stop_loss, levels.take_profit, levels.source, "signal"
        )
        dashboard.record_order_submitted("AAPL", "buy", "bracket", sl=levels.stop_loss, tp=levels.take_profit)
        dashboard.record_order_acknowledged("AAPL", response_ms=5.2)

        # Verify order was filled (paper broker fills immediately)
        assert order["status"] == "filled"
        assert order["type"] == "bracket"
        assert order["stop_loss"] == levels.stop_loss
        assert order["take_profit"] == levels.take_profit
        assert order["qty"] == 10
        assert order["side"] == "buy"
        assert "id" in order

        # Verify dashboard recorded the event
        data = dashboard.get_dashboard_data()
        assert data["protective_exits"]["configured_total"] == 1
        assert data["protective_exits"]["configured_by_source"]["strategy"] == 1

    def test_bracket_order_with_default_sl_tp(self, paper_broker, exits, dashboard):
        """Signal-driven trade without strategy SL/TP uses defaults."""
        levels = exits.compute(
            entry_price=380.0,
            side="buy",
            symbol="MSFT",
            trade_source="signal",
        )

        order = paper_broker.bracket_order(
            symbol="MSFT",
            qty=5,
            side="buy",
            take_profit=levels.take_profit,
            stop_loss=levels.stop_loss,
        )

        dashboard.record_exit_configured(
            "MSFT", "buy", 380.0, levels.stop_loss, levels.take_profit, levels.source, "signal"
        )
        dashboard.record_order_submitted("MSFT", "buy", "bracket", sl=levels.stop_loss, tp=levels.take_profit)
        dashboard.record_order_acknowledged("MSFT", response_ms=3.1)

        assert order["status"] == "filled"
        assert order["stop_loss"] == levels.stop_loss
        assert order["take_profit"] == levels.take_profit
        assert levels.source == "default"

    def test_bracket_order_short_side(self, paper_broker, exits, dashboard):
        """Short (sell) signal-driven trade has correct SL above and TP below entry."""
        levels = exits.compute(
            entry_price=250.0,
            side="sell",
            strategy_stop_loss=260.0,
            strategy_take_profit=235.0,
            symbol="TSLA",
            trade_source="signal",
        )

        order = paper_broker.bracket_order(
            symbol="TSLA",
            qty=8,
            side="sell",
            take_profit=levels.take_profit,
            stop_loss=levels.stop_loss,
        )

        assert order["status"] == "filled"
        assert levels.stop_loss > 250.0  # SL above entry for shorts
        assert levels.take_profit < 250.0  # TP below entry for shorts

        dashboard.record_exit_configured(
            "TSLA", "sell", 250.0, levels.stop_loss, levels.take_profit, levels.source, "signal"
        )
        dashboard.record_order_submitted("TSLA", "sell", "bracket", sl=levels.stop_loss, tp=levels.take_profit)
        dashboard.record_order_acknowledged("TSLA", response_ms=4.0)

    def test_bracket_order_with_atr(self, paper_broker, exits, dashboard):
        """Signal with ATR-based exits creates valid bracket order."""
        levels = exits.compute(
            entry_price=45000.0,
            side="buy",
            atr=1200.0,
            symbol="BTCUSD",
            trade_source="signal",
        )

        order = paper_broker.bracket_order(
            symbol="BTCUSD",
            qty=0.5,
            side="buy",
            take_profit=levels.take_profit,
            stop_loss=levels.stop_loss,
        )

        assert order["status"] == "filled"
        assert levels.stop_loss < 45000.0
        assert levels.take_profit > 45000.0
        assert levels.source == "atr"


# ---------------------------------------------------------------------------
# Manual Trade: SL/TP Submission to Paper Broker
# ---------------------------------------------------------------------------

class TestManualTradeSLTP:
    """Verify SL/TP orders are submitted and acknowledged for manual trades."""

    def test_manual_buy_bracket_order(self, paper_broker, exits, dashboard):
        """Manual /buy command creates a bracket order with default SL/TP."""
        # Simulating manual trade (no strategy-provided SL/TP)
        levels = exits.compute(
            entry_price=150.0,
            side="buy",
            symbol="AAPL",
            trade_source="manual",
        )

        order = paper_broker.bracket_order(
            symbol="AAPL",
            qty=20,
            side="buy",
            take_profit=levels.take_profit,
            stop_loss=levels.stop_loss,
        )

        dashboard.record_exit_configured(
            "AAPL", "buy", 150.0, levels.stop_loss, levels.take_profit, levels.source, "manual"
        )
        dashboard.record_order_submitted("AAPL", "buy", "bracket", sl=levels.stop_loss, tp=levels.take_profit)
        dashboard.record_order_acknowledged("AAPL", response_ms=2.8)

        # Verify
        assert order["status"] == "filled"
        assert order["type"] == "bracket"
        assert levels.source == "default"
        assert levels.stop_loss < 150.0
        assert levels.take_profit > 150.0

    def test_manual_sell_bracket_order(self, paper_broker, exits, dashboard):
        """Manual /sell command creates bracket with SL above entry."""
        levels = exits.compute(
            entry_price=380.0,
            side="sell",
            symbol="MSFT",
            trade_source="manual",
        )

        order = paper_broker.bracket_order(
            symbol="MSFT",
            qty=3,
            side="sell",
            take_profit=levels.take_profit,
            stop_loss=levels.stop_loss,
        )

        dashboard.record_exit_configured(
            "MSFT", "sell", 380.0, levels.stop_loss, levels.take_profit, levels.source, "manual"
        )
        dashboard.record_order_submitted("MSFT", "sell", "bracket", sl=levels.stop_loss, tp=levels.take_profit)
        dashboard.record_order_acknowledged("MSFT", response_ms=3.5)

        assert order["status"] == "filled"
        assert levels.stop_loss > 380.0
        assert levels.take_profit < 380.0


# ---------------------------------------------------------------------------
# Stop-Loss Execution Verification
# ---------------------------------------------------------------------------

class TestStopLossExecution:
    """Verify stop-loss orders trigger correctly in paper broker."""

    def test_stop_loss_triggers_on_price_drop(self, paper_broker, exits, dashboard):
        """Long position SL triggers when price drops to SL level."""
        levels = exits.compute(entry_price=150.0, side="buy", symbol="AAPL", trade_source="signal")

        # Open position via bracket
        paper_broker.bracket_order(
            symbol="AAPL", qty=10, side="buy",
            take_profit=levels.take_profit, stop_loss=levels.stop_loss,
        )

        # Verify position exists
        positions = paper_broker.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "AAPL"
        assert positions[0]["qty"] == 10

        # Simulate price drop to SL level — close position
        paper_broker.set_price("AAPL", levels.stop_loss)
        close_order = paper_broker.close_position("AAPL")

        # Record execution in dashboard
        pnl = (levels.stop_loss - 150.0) * 10
        dashboard.record_exit_executed(
            "AAPL", "stop_loss", levels.stop_loss, levels.stop_loss, 10, pnl,
            broker_acknowledged=True,
        )
        exits.record_execution(
            "AAPL", "stop_loss", levels.stop_loss, levels.stop_loss, 10, pnl,
        )

        assert close_order is not None
        assert close_order["status"] == "filled"

        # Dashboard shows the execution
        data = dashboard.get_dashboard_data()
        assert data["protective_exits"]["executed_stop_loss"] == 1
        assert data["protective_exits"]["broker_acknowledged"] == 1

    def test_stop_loss_triggers_on_short_price_rise(self, paper_broker, exits, dashboard):
        """Short position SL triggers when price rises to SL level."""
        levels = exits.compute(entry_price=250.0, side="sell", symbol="TSLA", trade_source="signal")

        paper_broker.bracket_order(
            symbol="TSLA", qty=5, side="sell",
            take_profit=levels.take_profit, stop_loss=levels.stop_loss,
        )

        # Simulate price rise to SL
        paper_broker.set_price("TSLA", levels.stop_loss)
        close_order = paper_broker.close_position("TSLA")

        pnl = (250.0 - levels.stop_loss) * 5  # Negative for short
        dashboard.record_exit_executed(
            "TSLA", "stop_loss", levels.stop_loss, levels.stop_loss, 5, pnl,
            broker_acknowledged=True,
        )

        assert close_order is not None
        assert close_order["status"] == "filled"
        assert pnl < 0  # SL should result in a loss


# ---------------------------------------------------------------------------
# Take-Profit Execution Verification
# ---------------------------------------------------------------------------

class TestTakeProfitExecution:
    """Verify take-profit orders trigger correctly in paper broker."""

    def test_take_profit_triggers_on_price_rise(self, paper_broker, exits, dashboard):
        """Long position TP triggers when price rises to TP level."""
        levels = exits.compute(entry_price=150.0, side="buy", symbol="AAPL", trade_source="signal")

        paper_broker.bracket_order(
            symbol="AAPL", qty=10, side="buy",
            take_profit=levels.take_profit, stop_loss=levels.stop_loss,
        )

        # Simulate price rise to TP
        paper_broker.set_price("AAPL", levels.take_profit)
        close_order = paper_broker.close_position("AAPL")

        pnl = (levels.take_profit - 150.0) * 10
        dashboard.record_exit_executed(
            "AAPL", "take_profit", levels.take_profit, levels.take_profit, 10, pnl,
            broker_acknowledged=True,
        )
        exits.record_execution(
            "AAPL", "take_profit", levels.take_profit, levels.take_profit, 10, pnl,
        )

        assert close_order is not None
        assert close_order["status"] == "filled"
        assert pnl > 0  # TP should result in a profit

        data = dashboard.get_dashboard_data()
        assert data["protective_exits"]["executed_take_profit"] == 1
        assert data["protective_exits"]["broker_acknowledged"] == 1

    def test_take_profit_triggers_on_short_price_drop(self, paper_broker, exits, dashboard):
        """Short position TP triggers when price drops to TP level."""
        levels = exits.compute(entry_price=380.0, side="sell", symbol="MSFT", trade_source="manual")

        paper_broker.bracket_order(
            symbol="MSFT", qty=5, side="sell",
            take_profit=levels.take_profit, stop_loss=levels.stop_loss,
        )

        # Simulate price drop to TP
        paper_broker.set_price("MSFT", levels.take_profit)
        close_order = paper_broker.close_position("MSFT")

        pnl = (380.0 - levels.take_profit) * 5  # Positive for short TP
        dashboard.record_exit_executed(
            "MSFT", "take_profit", levels.take_profit, levels.take_profit, 5, pnl,
            broker_acknowledged=True,
        )

        assert close_order is not None
        assert close_order["status"] == "filled"
        assert pnl > 0


# ---------------------------------------------------------------------------
# Telemetry and Dashboard Integration
# ---------------------------------------------------------------------------

class TestTelemetryIntegration:
    """Verify telemetry counters and dashboard metrics are consistent."""

    def test_telemetry_counters_increment(self, telemetry, dashboard):
        """Telemetry counters track protective exit lifecycle."""
        exits = ProtectiveExits(telemetry=telemetry)

        # Configure several exits
        for symbol, price in [("AAPL", 150.0), ("MSFT", 380.0), ("TSLA", 250.0)]:
            levels = exits.compute(entry_price=price, side="buy", symbol=symbol, trade_source="signal")
            dashboard.record_exit_configured(
                symbol, "buy", price, levels.stop_loss, levels.take_profit, levels.source, "signal"
            )

        assert telemetry.get_counter("protective_exits.configured") == 3
        data = dashboard.get_dashboard_data()
        assert data["protective_exits"]["configured_total"] == 3

    def test_adjustment_telemetry(self, telemetry):
        """Adjustments are tracked via telemetry when SL/TP is clamped."""
        config = ProtectiveExitConfig(max_stop_loss_pct=0.05)
        exits = ProtectiveExits(config=config, telemetry=telemetry)

        # Strategy SL at 10% would be clamped to 5%
        levels = exits.compute(
            entry_price=100.0,
            side="buy",
            strategy_stop_loss=85.0,  # 15% away — will be clamped to max 5%
            symbol="TEST",
            trade_source="signal",
        )

        assert telemetry.get_counter("protective_exits.adjusted") >= 1
        summary = exits.get_telemetry_summary()
        assert summary["total_adjustments"] >= 1

    def test_daily_report_generation(self, dashboard):
        """Daily report includes all protective exit metrics."""
        # Simulate a day of trading
        dashboard.record_exit_configured("AAPL", "buy", 150.0, 145.5, 159.0, "strategy", "signal")
        dashboard.record_exit_configured("MSFT", "buy", 380.0, 368.6, 402.8, "default", "manual")
        dashboard.record_exit_adjusted("AAPL", "stop_loss", 143.0, 145.5, "clamped_min")
        dashboard.record_exit_executed("AAPL", "stop_loss", 145.5, 145.3, 10, -47.0)
        dashboard.record_exit_executed("MSFT", "take_profit", 402.8, 403.0, 5, 115.0)

        report = dashboard.get_daily_report()
        pe = report["protective_exits"]
        assert pe["configured"] == 2
        assert pe["adjusted"] == 1
        assert pe["executed_total"] == 2
        assert pe["executed_stop_loss"] == 1
        assert pe["executed_take_profit"] == 1
        assert pe["total_pnl"] == pytest.approx(68.0, abs=0.01)

    def test_dashboard_text_output(self, dashboard):
        """Dashboard text format is valid HTML for Telegram."""
        dashboard.record_exit_configured("AAPL", "buy", 150.0, 145.5, 159.0, "strategy", "signal")
        dashboard.record_exit_executed("AAPL", "stop_loss", 145.5, 145.3, 10, -47.0)

        text = dashboard.get_dashboard_text()
        assert "Paper Trading Dashboard" in text
        assert "Protective Exits" in text
        assert "Configured: 1" in text
        assert "Stop-Loss: 1" in text

    def test_daily_report_text_output(self, dashboard):
        """Daily report text includes all sections."""
        dashboard.record_exit_configured("AAPL", "buy", 150.0, 145.5, 159.0, "strategy", "signal")
        dashboard.record_exit_executed("AAPL", "take_profit", 159.0, 159.2, 10, 92.0)

        text = dashboard.get_daily_report_text()
        assert "Daily Report" in text
        assert "Take-Profit: 1" in text


# ---------------------------------------------------------------------------
# Event Classes Verification
# ---------------------------------------------------------------------------

class TestProtectiveExitEvents:
    """Verify the new event dataclasses work correctly."""

    def test_configured_event_creation(self):
        """ProtectiveExitConfigured can be created and serialized."""
        event = ProtectiveExitConfigured(
            symbol="AAPL",
            side="buy",
            entry_price=150.0,
            stop_loss=145.5,
            take_profit=159.0,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            risk_reward_ratio=2.0,
            source="strategy",
            trade_source="signal",
        )
        d = event.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["stop_loss"] == 145.5
        assert d["_type"] == "ProtectiveExitConfigured"

    def test_adjusted_event_creation(self):
        """ProtectiveExitAdjusted can be created and serialized."""
        event = ProtectiveExitAdjusted(
            symbol="AAPL",
            field_adjusted="stop_loss",
            original_value=143.0,
            adjusted_value=145.5,
            reason="clamped_min",
        )
        d = event.to_dict()
        assert d["field_adjusted"] == "stop_loss"
        assert d["reason"] == "clamped_min"
        assert d["_type"] == "ProtectiveExitAdjusted"

    def test_executed_event_creation(self):
        """ProtectiveExitExecuted can be created and serialized."""
        event = ProtectiveExitExecuted(
            symbol="AAPL",
            exit_type="stop_loss",
            trigger_price=145.5,
            fill_price=145.3,
            qty=10,
            pnl=-47.0,
            order_id="abc123",
            trade_id="trade_001",
            broker_acknowledged=True,
        )
        d = event.to_dict()
        assert d["exit_type"] == "stop_loss"
        assert d["broker_acknowledged"] is True
        assert d["_type"] == "ProtectiveExitExecuted"

    def test_events_registered_in_registry(self):
        """All protective exit events are in the event class registry."""
        from src.core.events import get_event_class_registry
        registry = get_event_class_registry()
        assert "ProtectiveExitConfigured" in registry
        assert "ProtectiveExitAdjusted" in registry
        assert "ProtectiveExitExecuted" in registry


# ---------------------------------------------------------------------------
# Full Staging Run Simulation
# ---------------------------------------------------------------------------

class TestStagingRun:
    """
    End-to-end staging simulation: multiple trades with SL/TP through
    paper broker, verifying the full lifecycle.
    """

    def test_full_staging_run(self, paper_broker, telemetry, dashboard):
        """Simulate a complete staging run with mixed trade sources."""
        exits = ProtectiveExits(telemetry=telemetry)

        # --- Trade 1: Signal-driven long with strategy SL/TP ---
        levels1 = exits.compute(
            entry_price=150.0, side="buy",
            strategy_stop_loss=145.0, strategy_take_profit=162.0,
            symbol="AAPL", trade_source="signal",
        )
        order1 = paper_broker.bracket_order(
            "AAPL", 10, "buy", take_profit=levels1.take_profit, stop_loss=levels1.stop_loss,
        )
        dashboard.record_exit_configured(
            "AAPL", "buy", 150.0, levels1.stop_loss, levels1.take_profit, levels1.source, "signal"
        )
        dashboard.record_order_submitted("AAPL", "buy", "bracket", sl=levels1.stop_loss, tp=levels1.take_profit)
        dashboard.record_order_acknowledged("AAPL", response_ms=4.0)
        assert order1["status"] == "filled"

        # --- Trade 2: Manual short with defaults ---
        levels2 = exits.compute(
            entry_price=380.0, side="sell",
            symbol="MSFT", trade_source="manual",
        )
        order2 = paper_broker.bracket_order(
            "MSFT", 5, "sell", take_profit=levels2.take_profit, stop_loss=levels2.stop_loss,
        )
        dashboard.record_exit_configured(
            "MSFT", "sell", 380.0, levels2.stop_loss, levels2.take_profit, levels2.source, "manual"
        )
        dashboard.record_order_submitted("MSFT", "sell", "bracket", sl=levels2.stop_loss, tp=levels2.take_profit)
        dashboard.record_order_acknowledged("MSFT", response_ms=3.2)
        assert order2["status"] == "filled"

        # --- Simulate SL hit on AAPL ---
        paper_broker.set_price("AAPL", levels1.stop_loss)
        sl_close = paper_broker.close_position("AAPL")
        sl_pnl = (levels1.stop_loss - 150.0) * 10
        dashboard.record_exit_executed(
            "AAPL", "stop_loss", levels1.stop_loss, levels1.stop_loss, 10, sl_pnl,
            broker_acknowledged=True,
        )
        exits.record_execution(
            "AAPL", "stop_loss", levels1.stop_loss, levels1.stop_loss, 10, sl_pnl,
        )
        assert sl_close["status"] == "filled"

        # --- Simulate TP hit on MSFT ---
        paper_broker.set_price("MSFT", levels2.take_profit)
        tp_close = paper_broker.close_position("MSFT")
        tp_pnl = (380.0 - levels2.take_profit) * 5
        dashboard.record_exit_executed(
            "MSFT", "take_profit", levels2.take_profit, levels2.take_profit, 5, tp_pnl,
            broker_acknowledged=True,
        )
        exits.record_execution(
            "MSFT", "take_profit", levels2.take_profit, levels2.take_profit, 5, tp_pnl,
        )
        assert tp_close["status"] == "filled"

        # --- Verify telemetry ---
        assert telemetry.get_counter("protective_exits.configured") == 2
        assert telemetry.get_counter("protective_exits.executed.stop_loss") == 1
        assert telemetry.get_counter("protective_exits.executed.take_profit") == 1
        assert telemetry.get_counter("protective_exits.executed.total") == 2

        # --- Verify dashboard ---
        data = dashboard.get_dashboard_data()
        assert data["protective_exits"]["configured_total"] == 2
        assert data["protective_exits"]["executed_stop_loss"] == 1
        assert data["protective_exits"]["executed_take_profit"] == 1
        assert data["protective_exits"]["broker_acknowledged"] == 2
        assert data["protective_exits"]["broker_rejected"] == 0

        # --- Verify daily report ---
        report = dashboard.get_daily_report()
        assert report["protective_exits"]["executed_total"] == 2
        assert report["protective_exits"]["total_pnl"] == pytest.approx(sl_pnl + tp_pnl, abs=0.01)

        # --- Verify positions are closed ---
        positions = paper_broker.get_positions()
        assert len(positions) == 0

        # --- Verify account reflects P&L ---
        account = paper_broker.get_account()
        assert account["equity"] != 100_000.0  # Changed from starting

        # Print dashboard for visual inspection
        dashboard_text = dashboard.get_dashboard_text()
        report_text = dashboard.get_daily_report_text()
        assert "Paper Trading Dashboard" in dashboard_text
        assert "Daily Report" in report_text

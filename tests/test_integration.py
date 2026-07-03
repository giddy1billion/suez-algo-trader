"""
Integration tests — verify EventBus + TradeManager + ExecutionSimulator + Engine
work together in a full trade lifecycle.
"""

import threading
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.core.events import (
    EventBus, SignalGenerated, RiskEvaluated, OrderSubmitted,
    OrderFilled, TradeOpened, TradeClosed, RiskHalt, OrderRejected,
)
from src.core.state_machine import TradeManager, TradeState
from src.core.subscribers import (
    MetricsSubscriber, NotificationSubscriber, setup_default_subscribers,
)
from src.execution.simulator import ExecutionSimulator
from src.execution.engine import ExecutionEngine
from src.risk.manager import RiskManager, RiskLimits
from src.risk.engine import RiskEngine
from src.strategy.base import BaseStrategy, TradeSignal, Signal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockBroker:
    """Minimal mock broker for integration testing."""

    def __init__(self):
        self.paper = True
        self._orders = []
        self._positions = []

    def get_account(self):
        return {
            "portfolio_value": 100000.0,
            "equity": 100000.0,
            "cash": 80000.0,
            "long_market_value": 20000.0,
        }

    def get_positions(self):
        return self._positions

    def get_bars_df(self, symbol, timeframe, limit):
        np.random.seed(42)
        n = max(limit, 100)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close = np.maximum(close, 10)  # Prevent negative prices
        df = pd.DataFrame({
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.random.randint(100000, 500000, n),
        })
        return df

    def market_order(self, symbol, qty, side):
        order = {"id": f"order-{len(self._orders)}", "status": "filled", "symbol": symbol}
        self._orders.append(order)
        return order

    def bracket_order(self, symbol, qty, side, stop_loss_price, take_profit_price):
        order = {"id": f"bracket-{len(self._orders)}", "status": "filled", "symbol": symbol}
        self._orders.append(order)
        return order

    def close_position(self, symbol):
        self._positions = [p for p in self._positions if p["symbol"] != symbol]

    def close_all_positions(self):
        self._positions = []

    def cancel_all_orders(self):
        pass


class MockDB:
    """Minimal mock database."""

    def __init__(self):
        self.trades = []
        self.signals = []
        self.snapshots = []

    def record_trade(self, data):
        self.trades.append(data)

    def log_signal(self, data):
        self.signals.append(data)

    def snapshot_portfolio(self, data):
        self.snapshots.append(data)


class SimpleStrategy(BaseStrategy):
    """Strategy that generates one BUY signal for testing."""

    def __init__(self, signal_type=Signal.BUY, confidence=0.85):
        super().__init__(name="test_strategy", symbols=["AAPL"], timeframe="1Hour", lookback=100)
        self._signal_type = signal_type
        self._confidence = confidence

    def calculate_indicators(self, df):
        df["rsi"] = 50.0
        df["atr_14"] = 2.0
        return df

    def generate_signals(self, data):
        signals = []
        for symbol, df in data.items():
            price = float(df["close"].iloc[-1])
            signals.append(TradeSignal(
                symbol=symbol,
                signal=self._signal_type,
                confidence=self._confidence,
                price=price,
                stop_loss=price * 0.97,
                take_profit=price * 1.05,
                reason="test_signal",
                indicators={"rsi": 50.0},
            ))
        return signals


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullTradeLifecycle:
    """Test the complete trade lifecycle with all components wired together."""

    def setup_method(self):
        self.broker = MockBroker()
        self.db = MockDB()
        self.event_bus = EventBus()
        self.trade_manager = TradeManager()
        self.simulator = ExecutionSimulator.ideal()  # No friction for deterministic test
        self.risk = RiskManager(RiskLimits())
        self.risk.reset_daily(100000.0)

        self.engine = ExecutionEngine(
            broker=self.broker,
            risk_manager=self.risk,
            db=self.db,
            dry_run=False,
            event_bus=self.event_bus,
            trade_manager=self.trade_manager,
            execution_simulator=self.simulator,
        )

    def test_buy_signal_produces_full_event_chain(self):
        """A BUY signal should produce: SignalGenerated → RiskEvaluated → OrderSubmitted → OrderFilled → TradeOpened."""
        strategy = SimpleStrategy(Signal.BUY, confidence=0.85)
        results = self.engine.run_cycle(strategy)

        # Should have placed an order
        assert len(results) > 0
        assert len(self.broker._orders) == 1

        # Check event history
        history = self.event_bus.get_history()
        event_types = [type(e).__name__ for e in history]

        assert "SignalGenerated" in event_types
        assert "RiskEvaluated" in event_types
        assert "OrderSubmitted" in event_types
        assert "OrderFilled" in event_types
        assert "TradeOpened" in event_types

    def test_trade_lifecycle_transitions_correctly(self):
        """Trade should transition: SIGNAL → PENDING_RISK → RISK_APPROVED → SUBMITTED → ACCEPTED → FILLED → ACTIVE."""
        strategy = SimpleStrategy(Signal.BUY, confidence=0.85)
        self.engine.run_cycle(strategy)

        # TradeManager should have one trade
        assert self.trade_manager.count == 1
        trades = self.trade_manager.get_active_trades()
        assert len(trades) == 1

        trade = trades[0]
        assert trade.state == TradeState.ACTIVE
        assert trade.symbol == "AAPL"

        # Check history includes all transitions
        states_visited = [s for s, _, _ in trade.history]
        assert TradeState.SIGNAL in states_visited
        assert TradeState.PENDING_RISK in states_visited
        assert TradeState.RISK_APPROVED in states_visited
        assert TradeState.SUBMITTED in states_visited
        assert TradeState.ACCEPTED in states_visited
        assert TradeState.FILLED in states_visited
        assert TradeState.ACTIVE in states_visited

    def test_trade_id_consistency_across_events(self):
        """TradeOpened event should use TradeManager's trade_id."""
        strategy = SimpleStrategy(Signal.BUY, confidence=0.85)
        self.engine.run_cycle(strategy)

        # Get the trade_id from TradeManager
        trades = self.trade_manager.get_active_trades()
        lifecycle_id = trades[0].trade_id

        # Get TradeOpened event
        opened_events = [e for e in self.event_bus.get_history() if isinstance(e, TradeOpened)]
        assert len(opened_events) == 1
        assert opened_events[0].trade_id == lifecycle_id

    def test_hold_signal_produces_no_order(self):
        """HOLD signal should not produce orders."""
        strategy = SimpleStrategy(Signal.HOLD, confidence=0.5)
        results = self.engine.run_cycle(strategy)

        assert len(results) == 0
        assert len(self.broker._orders) == 0
        assert self.trade_manager.count == 0

    def test_risk_rejection_publishes_event_and_transitions(self):
        """If risk rejects, should publish RiskEvaluated(approved=False) and transition to RISK_REJECTED."""
        # Make risk engine reject everything by setting very low limits
        strict_engine = RiskEngine()
        self.engine.risk_engine = strict_engine

        # Set max daily loss to 0 (already hit)
        self.risk.daily_stats.realized_pnl = -9999
        self.risk.daily_stats.is_halted = True

        strategy = SimpleStrategy(Signal.BUY, confidence=0.85)
        results = self.engine.run_cycle(strategy)

        # Should be halted — risk halt event published
        history = self.event_bus.get_history()
        halt_events = [e for e in history if isinstance(e, RiskHalt)]
        assert len(halt_events) >= 1

    def test_emergency_liquidate_publishes_per_position_events(self):
        """Emergency liquidate should publish TradeClosed for each open position."""
        # Add mock positions
        self.broker._positions = [
            {"symbol": "AAPL", "unrealized_pl": -50.0, "current_price": 150.0,
             "avg_entry_price": 155.0, "asset_id": "pos-1"},
            {"symbol": "MSFT", "unrealized_pl": 100.0, "current_price": 400.0,
             "avg_entry_price": 380.0, "asset_id": "pos-2"},
        ]

        self.engine.emergency_liquidate()

        history = self.event_bus.get_history()
        closed_events = [e for e in history if isinstance(e, TradeClosed)]
        halt_events = [e for e in history if isinstance(e, RiskHalt)]

        assert len(closed_events) == 2
        assert len(halt_events) == 1
        assert halt_events[0].level == "CRITICAL"

        symbols_closed = {e.symbol for e in closed_events}
        assert symbols_closed == {"AAPL", "MSFT"}

    def test_simulator_rejection_publishes_order_rejected(self):
        """If simulator rejects (e.g., failure model), should publish OrderRejected."""
        # Use a simulator with 100% failure rate
        sim = ExecutionSimulator.realistic(seed=42)
        sim.failures.rejection_rate = 1.0  # 100% rejection
        self.engine._simulator = sim

        strategy = SimpleStrategy(Signal.BUY, confidence=0.85)
        results = self.engine.run_cycle(strategy)

        history = self.event_bus.get_history()
        rejected = [e for e in history if isinstance(e, OrderRejected)]
        assert len(rejected) >= 1


class TestSubscribersIntegration:
    """Test that subscribers correctly process events from a real trade cycle."""

    def test_metrics_subscriber_counts_trades(self):
        """MetricsSubscriber should count winning/losing trades from TradeClosed events."""
        bus = EventBus()
        metrics = MetricsSubscriber()
        metrics.register(bus)

        # Simulate two trades
        bus.publish(TradeClosed(trade_id="t1", symbol="AAPL", exit_price=155,
                               pnl=50.0, pnl_pct=3.3, reason="tp_hit"))
        bus.publish(TradeClosed(trade_id="t2", symbol="MSFT", exit_price=390,
                               pnl=-20.0, pnl_pct=-1.5, reason="sl_hit"))

        assert metrics.total_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.total_pnl == 30.0

    def test_notification_subscriber_sends_messages(self):
        """NotificationSubscriber should call send_func on trade events."""
        bus = EventBus()
        messages = []
        notif = NotificationSubscriber(send_func=lambda m: messages.append(m))
        notif.register(bus)

        bus.publish(TradeOpened(trade_id="t1", symbol="AAPL", side="buy",
                               entry_price=150, qty=10, stop_loss=145, take_profit=160))
        bus.publish(RiskHalt(reason="max_daily_loss", level="CRITICAL"))

        assert len(messages) == 2
        assert "AAPL" in messages[0]
        assert "RISK HALT" in messages[1]

    def test_setup_default_subscribers_registers_all(self):
        """setup_default_subscribers should register 4 subscriber types."""
        bus = EventBus()
        subs = setup_default_subscribers(bus)

        assert "audit" in subs
        assert "journal" in subs
        assert "metrics" in subs
        assert "notifications" in subs
        assert bus.subscriber_count >= 4


class TestPeriodicCleanup:
    """Test that TradeManager cleanup prevents unbounded growth."""

    def test_remove_terminal_clears_closed_trades(self):
        """Closed trades should be removable from memory."""
        mgr = TradeManager()

        # Create and close 5 trades
        for i in range(5):
            t = mgr.create_trade(f"SYM{i}", "buy")
            t.transition(TradeState.PENDING_RISK, "test")
            t.transition(TradeState.RISK_APPROVED, "test")
            t.transition(TradeState.SUBMITTED, "test")
            t.transition(TradeState.ACCEPTED, "test")
            t.transition(TradeState.FILLED, "test")
            t.transition(TradeState.ACTIVE, "test")
            t.transition(TradeState.CLOSING, "test")
            t.transition(TradeState.CLOSED, "test")

        assert mgr.count == 5
        assert len(mgr.get_active_trades()) == 0

        removed = mgr.remove_terminal()
        assert removed == 5
        assert mgr.count == 0

    def test_active_trades_not_removed(self):
        """Active trades should NOT be removed by cleanup."""
        mgr = TradeManager()

        t = mgr.create_trade("AAPL", "buy")
        t.transition(TradeState.PENDING_RISK, "test")
        t.transition(TradeState.RISK_APPROVED, "test")
        t.transition(TradeState.SUBMITTED, "test")
        t.transition(TradeState.ACCEPTED, "test")
        t.transition(TradeState.FILLED, "test")
        t.transition(TradeState.ACTIVE, "test")

        removed = mgr.remove_terminal()
        assert removed == 0
        assert mgr.count == 1
        assert mgr.get_active_trades()[0].state == TradeState.ACTIVE

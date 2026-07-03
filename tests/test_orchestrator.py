"""Tests for multi-strategy orchestrator."""

import time
import threading
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.strategy.orchestrator import StrategyOrchestrator, StrategySlot


@pytest.fixture
def mock_strategy():
    """Create a mock strategy."""
    strat = MagicMock()
    strat.name = "test_momentum"
    strat.symbols = ["AAPL", "MSFT"]
    strat.timeframe = "1Hour"
    strat.is_active = True
    strat.generate_signals.return_value = []
    return strat


@pytest.fixture
def orchestrator():
    return StrategyOrchestrator()


@pytest.fixture
def mock_engine():
    """Mock execution engine."""
    engine = MagicMock()
    engine.run_cycle.return_value = [
        {"symbol": "AAPL", "action": "buy", "pnl": 50.0},
    ]
    return engine


class TestStrategySlot:
    def test_is_due_when_never_run(self, mock_strategy):
        slot = StrategySlot(
            name="momentum",
            strategy=mock_strategy,
            symbols=["AAPL"],
            timeframe="1Hour",
            interval=60,
        )
        assert slot.is_due is True

    def test_is_due_when_disabled(self, mock_strategy):
        slot = StrategySlot(
            name="momentum",
            strategy=mock_strategy,
            symbols=["AAPL"],
            timeframe="1Hour",
            interval=60,
            enabled=False,
        )
        assert slot.is_due is False

    def test_is_due_respects_interval(self, mock_strategy):
        slot = StrategySlot(
            name="momentum",
            strategy=mock_strategy,
            symbols=["AAPL"],
            timeframe="1Hour",
            interval=60,
        )
        # Just ran
        slot.last_cycle = datetime.now(timezone.utc)
        assert slot.is_due is False

        # Ran 61 seconds ago
        slot.last_cycle = datetime.now(timezone.utc) - timedelta(seconds=61)
        assert slot.is_due is True

    def test_record_cycle(self, mock_strategy):
        slot = StrategySlot(
            name="momentum",
            strategy=mock_strategy,
            symbols=["AAPL"],
            timeframe="1Hour",
        )
        trades = [
            {"pnl": 100.0},
            {"pnl": -30.0},
            {"pnl": 50.0},
        ]
        slot.record_cycle(signals=5, trades=trades)

        assert slot.cycle_count == 1
        assert slot.total_signals == 5
        assert slot.total_trades == 3
        assert slot.win_count == 2
        assert slot.loss_count == 1
        assert slot.realized_pnl == 120.0
        assert slot.last_cycle is not None

    def test_win_rate(self, mock_strategy):
        slot = StrategySlot(
            name="momentum",
            strategy=mock_strategy,
            symbols=["AAPL"],
            timeframe="1Hour",
        )
        slot.win_count = 7
        slot.loss_count = 3
        assert slot.win_rate == 0.7

    def test_win_rate_zero_trades(self, mock_strategy):
        slot = StrategySlot(
            name="momentum",
            strategy=mock_strategy,
            symbols=["AAPL"],
            timeframe="1Hour",
        )
        assert slot.win_rate == 0.0

    def test_get_stats(self, mock_strategy):
        slot = StrategySlot(
            name="momentum",
            strategy=mock_strategy,
            symbols=["AAPL", "MSFT"],
            timeframe="1Hour",
            interval=60,
            weight=1.5,
        )
        slot.record_cycle(3, [{"pnl": 100}])
        stats = slot.get_stats()

        assert stats["name"] == "momentum"
        assert stats["enabled"] is True
        assert stats["symbols"] == ["AAPL", "MSFT"]
        assert stats["timeframe"] == "1Hour"
        assert stats["interval"] == 60
        assert stats["weight"] == 1.5
        assert stats["cycle_count"] == 1
        assert stats["total_signals"] == 3
        assert stats["total_trades"] == 1
        assert stats["realized_pnl"] == 100.0
        assert stats["win_rate"] == 100.0


class TestStrategyOrchestrator:
    def test_add_strategy(self, orchestrator, mock_strategy):
        orchestrator.add_strategy(
            name="momentum",
            strategy=mock_strategy,
            symbols=["AAPL", "MSFT"],
            timeframe="1Hour",
            interval=60,
        )
        assert len(orchestrator) == 1
        assert "momentum" in orchestrator.strategy_names

    def test_remove_strategy(self, orchestrator, mock_strategy):
        orchestrator.add_strategy(name="momentum", strategy=mock_strategy,
                                  symbols=["AAPL"], timeframe="1Hour")
        assert orchestrator.remove_strategy("momentum") is True
        assert len(orchestrator) == 0
        assert orchestrator.remove_strategy("nonexistent") is False

    def test_enable_disable(self, orchestrator, mock_strategy):
        orchestrator.add_strategy(name="momentum", strategy=mock_strategy,
                                  symbols=["AAPL"], timeframe="1Hour")
        assert orchestrator.disable_strategy("momentum") is True
        assert orchestrator._slots["momentum"].enabled is False
        assert orchestrator.active_count == 0

        assert orchestrator.enable_strategy("momentum") is True
        assert orchestrator._slots["momentum"].enabled is True
        assert orchestrator.active_count == 1

    def test_enable_nonexistent(self, orchestrator):
        assert orchestrator.enable_strategy("nonexistent") is False
        assert orchestrator.disable_strategy("nonexistent") is False

    def test_get_due_strategies(self, orchestrator, mock_strategy):
        orchestrator.add_strategy(name="fast", strategy=mock_strategy,
                                  symbols=["AAPL"], timeframe="1Hour", interval=10)
        orchestrator.add_strategy(name="slow", strategy=mock_strategy,
                                  symbols=["MSFT"], timeframe="1Hour", interval=3600)

        # Both are due (never run)
        due = orchestrator.get_due_strategies()
        assert len(due) == 2

        # Mark "fast" as just run
        orchestrator._slots["fast"].last_cycle = datetime.now(timezone.utc)
        due = orchestrator.get_due_strategies()
        assert len(due) == 1
        assert due[0].name == "slow"

    def test_run_due_strategies(self, orchestrator, mock_strategy, mock_engine):
        orchestrator.add_strategy(name="momentum", strategy=mock_strategy,
                                  symbols=["AAPL"], timeframe="1Hour", interval=10)

        results = orchestrator.run_due_strategies(mock_engine)
        assert len(results) == 1
        assert results[0]["_strategy"] == "momentum"
        mock_engine.run_cycle.assert_called_once_with(mock_strategy)

    def test_run_due_strategies_empty(self, orchestrator, mock_engine):
        # No strategies registered
        results = orchestrator.run_due_strategies(mock_engine)
        assert results == []

    def test_run_due_strategies_not_due(self, orchestrator, mock_strategy, mock_engine):
        orchestrator.add_strategy(name="momentum", strategy=mock_strategy,
                                  symbols=["AAPL"], timeframe="1Hour", interval=3600)
        # Mark as just run
        orchestrator._slots["momentum"].last_cycle = datetime.now(timezone.utc)
        results = orchestrator.run_due_strategies(mock_engine)
        assert results == []

    def test_run_due_handles_errors(self, orchestrator, mock_strategy, mock_engine):
        mock_engine.run_cycle.side_effect = Exception("Broker error")
        orchestrator.add_strategy(name="failing", strategy=mock_strategy,
                                  symbols=["AAPL"], timeframe="1Hour")
        # Should not raise
        results = orchestrator.run_due_strategies(mock_engine)
        assert results == []

    def test_get_weights(self, orchestrator, mock_strategy):
        orchestrator.add_strategy(name="a", strategy=mock_strategy,
                                  symbols=["AAPL"], timeframe="1Hour", weight=1.0)
        orchestrator.add_strategy(name="b", strategy=mock_strategy,
                                  symbols=["MSFT"], timeframe="1Hour", weight=3.0)

        weights = orchestrator.get_weights()
        assert abs(weights["a"] - 0.25) < 0.01
        assert abs(weights["b"] - 0.75) < 0.01

    def test_set_weight(self, orchestrator, mock_strategy):
        orchestrator.add_strategy(name="a", strategy=mock_strategy,
                                  symbols=["AAPL"], timeframe="1Hour", weight=1.0)
        assert orchestrator.set_weight("a", 2.5) is True
        assert orchestrator._slots["a"].weight == 2.5
        assert orchestrator.set_weight("nonexistent", 1.0) is False
        assert orchestrator.set_weight("a", 0) is False  # weight must be > 0

    def test_get_all_stats(self, orchestrator, mock_strategy, mock_engine):
        orchestrator.add_strategy(name="momentum", strategy=mock_strategy,
                                  symbols=["AAPL", "MSFT"], timeframe="1Hour", weight=1.0)
        orchestrator.add_strategy(name="ml", strategy=mock_strategy,
                                  symbols=["NVDA"], timeframe="15Min", weight=1.5)

        # Run one cycle
        orchestrator.run_due_strategies(mock_engine)
        stats = orchestrator.get_all_stats()

        assert stats["total_strategies"] == 2
        assert stats["active_strategies"] == 2
        assert stats["total_cycles"] == 2
        assert "momentum" in stats["strategies"]
        assert "ml" in stats["strategies"]

    def test_repr(self, orchestrator, mock_strategy):
        orchestrator.add_strategy(name="a", strategy=mock_strategy,
                                  symbols=["AAPL"], timeframe="1Hour")
        assert "strategies=1" in repr(orchestrator)
        assert "active=1" in repr(orchestrator)

    def test_multiple_cycles_accumulate(self, orchestrator, mock_strategy, mock_engine):
        orchestrator.add_strategy(name="fast", strategy=mock_strategy,
                                  symbols=["AAPL"], timeframe="1Hour", interval=0)

        # First cycle
        orchestrator.run_due_strategies(mock_engine)
        # Second cycle (interval=0, always due)
        orchestrator._slots["fast"].last_cycle = datetime.now(timezone.utc) - timedelta(seconds=1)
        orchestrator.run_due_strategies(mock_engine)

        assert orchestrator._slots["fast"].cycle_count == 2
        assert orchestrator._slots["fast"].total_trades == 2

    def test_thread_safety(self, orchestrator, mock_strategy, mock_engine):
        """Ensure concurrent access doesn't crash."""
        orchestrator.add_strategy(name="a", strategy=mock_strategy,
                                  symbols=["AAPL"], timeframe="1Hour", interval=0)

        errors = []

        def run_cycles():
            try:
                for _ in range(50):
                    orchestrator.run_due_strategies(mock_engine)
                    orchestrator._slots["a"].last_cycle = datetime.now(timezone.utc) - timedelta(seconds=1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_cycles) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

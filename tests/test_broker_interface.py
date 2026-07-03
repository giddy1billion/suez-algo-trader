"""
Tests for the broker interface, PaperBroker, and ReplayBroker.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from src.broker.base import BrokerProtocol
from src.broker.paper import PaperBroker
from src.broker.replay_broker import ReplayBroker


# ──────────────────────────────────────────────────────────────────────
# 1. Protocol conformance
# ──────────────────────────────────────────────────────────────────────

class TestProtocolConformance:
    """Test that broker implementations satisfy BrokerProtocol."""

    def test_alpaca_broker_satisfies_protocol(self):
        """AlpacaBroker structurally satisfies BrokerProtocol (no instantiation needed)."""
        from src.broker.alpaca_client import AlpacaBroker
        # Protocol uses structural typing — check the class has all required methods
        required_methods = [
            "get_account", "get_positions", "market_order", "limit_order",
            "bracket_order", "cancel_order", "close_position", "get_orders",
            "get_bars", "get_bars_df",
        ]
        for method in required_methods:
            assert hasattr(AlpacaBroker, method), f"AlpacaBroker missing {method}"
        # Check paper attribute exists on class (set in __init__)
        assert "paper" in AlpacaBroker.__init__.__code__.co_varnames

    def test_paper_broker_is_protocol_instance(self):
        """PaperBroker satisfies BrokerProtocol via isinstance check."""
        broker = PaperBroker()
        assert isinstance(broker, BrokerProtocol)

    def test_replay_broker_is_protocol_instance(self):
        """ReplayBroker satisfies BrokerProtocol via isinstance check."""
        broker = ReplayBroker(historical_data={})
        assert isinstance(broker, BrokerProtocol)


# ──────────────────────────────────────────────────────────────────────
# 2. PaperBroker tests
# ──────────────────────────────────────────────────────────────────────

class TestPaperBrokerOrders:
    """Test PaperBroker order execution and position tracking."""

    def test_market_order_fills(self):
        """Market order fills instantly and creates a position."""
        broker = PaperBroker(starting_equity=100_000)
        broker.set_price("AAPL", 150.0)

        order = broker.market_order("AAPL", 10, "buy")

        assert order["status"] == "filled"
        assert order["filled_avg_price"] == 150.0
        assert order["symbol"] == "AAPL"
        assert order["qty"] == 10

    def test_positions_track_after_order(self):
        """Positions reflect filled orders."""
        broker = PaperBroker()
        broker.set_price("AAPL", 150.0)
        broker.market_order("AAPL", 10, "buy")

        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "AAPL"
        assert positions[0]["qty"] == 10
        assert positions[0]["side"] == "buy"
        assert positions[0]["avg_entry_price"] == 150.0

    def test_unrealized_pl_calculation(self):
        """Unrealized P&L updates when price changes."""
        broker = PaperBroker()
        broker.set_price("AAPL", 150.0)
        broker.market_order("AAPL", 10, "buy")

        # Price goes up
        broker.set_price("AAPL", 160.0)
        positions = broker.get_positions()
        assert positions[0]["unrealized_pl"] == pytest.approx(100.0)  # (160-150)*10

    def test_account_equity_updates(self):
        """Account equity reflects position value changes."""
        broker = PaperBroker(starting_equity=100_000)
        broker.set_price("AAPL", 100.0)
        broker.market_order("AAPL", 100, "buy")

        # Price doubles
        broker.set_price("AAPL", 200.0)
        account = broker.get_account()
        # Cash was 100k, spent 10k (100*100), now have 90k cash + 20k market value
        assert account["equity"] == pytest.approx(110_000.0)

    def test_close_position_removes_position(self):
        """close_position removes the position entirely."""
        broker = PaperBroker()
        broker.set_price("TSLA", 250.0)
        broker.market_order("TSLA", 5, "buy")
        assert len(broker.get_positions()) == 1

        result = broker.close_position("TSLA")
        assert result is not None
        assert result["status"] == "filled"
        assert len(broker.get_positions()) == 0

    def test_close_position_nonexistent_returns_none(self):
        """close_position returns None for non-existent symbol."""
        broker = PaperBroker()
        assert broker.close_position("NOPE") is None

    def test_limit_order_pending(self):
        """Limit order stays pending when price doesn't satisfy."""
        broker = PaperBroker()
        broker.set_price("AAPL", 150.0)

        # Buy limit below market — shouldn't fill
        order = broker.limit_order("AAPL", 10, "buy", limit_price=140.0)
        assert order["status"] == "pending"

    def test_limit_order_fills_on_price_update(self):
        """Limit order fills when price crosses limit."""
        broker = PaperBroker()
        broker.set_price("AAPL", 150.0)

        order = broker.limit_order("AAPL", 10, "buy", limit_price=145.0)
        assert order["status"] == "pending"

        # Price drops below limit
        broker.set_price("AAPL", 140.0)
        assert order["status"] == "filled"
        assert order["filled_avg_price"] == 145.0

    def test_cancel_order(self):
        """Pending orders can be cancelled."""
        broker = PaperBroker()
        broker.set_price("AAPL", 150.0)
        order = broker.limit_order("AAPL", 10, "buy", limit_price=140.0)

        result = broker.cancel_order(order["id"])
        assert result is not None
        assert result["status"] == "cancelled"

    def test_get_orders(self):
        """get_orders filters correctly."""
        broker = PaperBroker()
        broker.set_price("AAPL", 150.0)
        broker.market_order("AAPL", 5, "buy")
        broker.limit_order("AAPL", 5, "buy", limit_price=140.0)

        open_orders = broker.get_orders(status="open")
        assert len(open_orders) == 1
        assert open_orders[0]["type"] == "limit"

    def test_paper_property(self):
        """Paper property returns True."""
        broker = PaperBroker()
        assert broker.paper is True

    def test_name_property(self):
        """Name property returns 'paper'."""
        broker = PaperBroker()
        assert broker.name == "paper"


class TestPaperBrokerThreadSafety:
    """Test PaperBroker under concurrent access."""

    def test_concurrent_market_orders(self):
        """Multiple threads can place orders without corruption."""
        broker = PaperBroker(starting_equity=1_000_000)
        broker.set_price("AAPL", 100.0)

        num_orders = 100
        results = []

        def place_order(i):
            order = broker.market_order("AAPL", 1, "buy")
            results.append(order)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(place_order, i) for i in range(num_orders)]
            for f in futures:
                f.result()

        assert len(results) == num_orders
        assert all(o["status"] == "filled" for o in results)

        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0]["qty"] == pytest.approx(num_orders)


# ──────────────────────────────────────────────────────────────────────
# 3. ReplayBroker tests
# ──────────────────────────────────────────────────────────────────────

class TestReplayBroker:
    """Test ReplayBroker historical data replay."""

    @pytest.fixture
    def sample_data(self):
        """Create sample historical data."""
        df = pd.DataFrame({
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1000, 1100, 1200, 1300, 1400],
        })
        return {"AAPL": df}

    def test_get_bars_df_returns_historical_data(self, sample_data):
        """get_bars_df returns the historical data up to current index."""
        broker = ReplayBroker(historical_data=sample_data)
        # At index 0, should get 1 bar
        df = broker.get_bars_df("AAPL")
        assert len(df) == 1
        assert df.iloc[0]["close"] == 100.5

        # Advance and check more data
        broker.advance("AAPL", 3)
        df = broker.get_bars_df("AAPL")
        assert len(df) == 4
        assert df.iloc[-1]["close"] == 103.5

    def test_get_bars_df_unknown_symbol(self, sample_data):
        """get_bars_df returns empty DataFrame for unknown symbol."""
        broker = ReplayBroker(historical_data=sample_data)
        df = broker.get_bars_df("NOPE")
        assert len(df) == 0
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_market_order_fills_at_close_price(self, sample_data):
        """Market orders fill at current bar's close price."""
        broker = ReplayBroker(historical_data=sample_data)
        # Index 0, close = 100.5
        order = broker.market_order("AAPL", 10, "buy")
        assert order["status"] == "filled"
        assert order["filled_avg_price"] == 100.5

    def test_order_fills_at_advanced_price(self, sample_data):
        """After advancing, orders fill at the new bar's close."""
        broker = ReplayBroker(historical_data=sample_data)
        broker.advance("AAPL", 2)
        # Now at index 2, close = 102.5
        order = broker.market_order("AAPL", 5, "buy")
        assert order["filled_avg_price"] == 102.5

    def test_positions_track_correctly(self, sample_data):
        """Positions are maintained after orders."""
        broker = ReplayBroker(historical_data=sample_data)
        broker.market_order("AAPL", 10, "buy")

        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "AAPL"
        assert positions[0]["qty"] == 10

    def test_replay_broker_properties(self, sample_data):
        """ReplayBroker has correct property values."""
        broker = ReplayBroker(historical_data=sample_data)
        assert broker.paper is True
        assert broker.name == "replay"

    def test_close_position(self, sample_data):
        """Closing position works in replay broker."""
        broker = ReplayBroker(historical_data=sample_data)
        broker.market_order("AAPL", 10, "buy")
        assert len(broker.get_positions()) == 1

        broker.close_position("AAPL")
        assert len(broker.get_positions()) == 0

"""Tests for execution realism simulator."""
import pytest
import numpy as np
from src.execution.simulator import (
    ExecutionSimulator, FixedSlippage, VolumeImpactSlippage,
    InstantFill, PartialFill, SpreadModel, FeeModel, OrderFailureModel,
)


class TestFixedSlippage:
    def test_buy_slippage_increases_price(self):
        slip = FixedSlippage(bps=10.0)
        result = slip.apply(100.0, 100, "buy", 1000000)
        assert result > 100.0  # Buy = pay more

    def test_sell_slippage_decreases_price(self):
        slip = FixedSlippage(bps=10.0)
        result = slip.apply(100.0, 100, "sell", 1000000)
        assert result < 100.0  # Sell = receive less

    def test_zero_slippage(self):
        slip = FixedSlippage(bps=0)
        assert slip.apply(100.0, 100, "buy", 1000000) == 100.0


class TestVolumeImpactSlippage:
    def test_larger_order_more_slippage(self):
        slip = VolumeImpactSlippage(impact_factor=0.1)
        small = slip.apply(100.0, 100, "buy", 1000000)
        large = slip.apply(100.0, 10000, "buy", 1000000)
        assert large > small  # Larger order = more slippage


class TestSpreadModel:
    def test_buy_at_ask(self):
        spread = SpreadModel(base_spread_bps=10.0, volatility_scaling=False)
        price = spread.execution_price(100.0, "buy")
        assert price > 100.0  # Buy at ask (above mid)

    def test_sell_at_bid(self):
        spread = SpreadModel(base_spread_bps=10.0, volatility_scaling=False)
        price = spread.execution_price(100.0, "sell")
        assert price < 100.0  # Sell at bid (below mid)


class TestFeeModel:
    def test_equity_fees(self):
        fees = FeeModel()
        cost = fees.calculate(100, 150.0, "sell", "equity")
        assert cost >= 0

    def test_crypto_fees(self):
        fees = FeeModel(taker_fee_pct=0.001)
        cost = fees.calculate(1.0, 45000.0, "buy", "crypto")
        assert cost > 0  # Should be ~$45


class TestInstantFill:
    def test_fills_full_qty(self):
        fill = InstantFill()
        rng = np.random.default_rng(42)
        fills = fill.simulate_fill(100, 150.0, 5000000, rng)
        assert len(fills) >= 1
        total = sum(f['qty'] for f in fills)
        assert abs(total - 100) < 0.01


class TestPartialFill:
    def test_large_order_may_partial_fill(self):
        fill = PartialFill(max_participation_rate=0.01)
        rng = np.random.default_rng(42)
        fills = fill.simulate_fill(100000, 150.0, 500000, rng)  # 20% of volume
        total = sum(f['qty'] for f in fills)
        # Should not fill entire order in one go (multiple fills)
        assert len(fills) > 1


class TestExecutionSimulator:
    def test_ideal_no_friction(self):
        sim = ExecutionSimulator.ideal(seed=42)
        r = sim.simulate_execution("AAPL", "buy", 100, 150.0, volume=5000000)
        assert r['executed'] is True
        assert abs(r['avg_price'] - 150.0) < 0.01
        assert r['slippage_bps'] < 0.1

    def test_realistic_has_slippage(self):
        sim = ExecutionSimulator.realistic(seed=42)
        r = sim.simulate_execution("AAPL", "buy", 100, 150.0, volume=5000000, atr=3.0)
        assert r['executed'] is True
        assert r['avg_price'] != 150.0  # Should have some slippage

    def test_deterministic_with_seed(self):
        sim1 = ExecutionSimulator.realistic(seed=123)
        sim2 = ExecutionSimulator.realistic(seed=123)
        r1 = sim1.simulate_execution("AAPL", "buy", 50, 200.0, volume=3000000, atr=4.0)
        r2 = sim2.simulate_execution("AAPL", "buy", 50, 200.0, volume=3000000, atr=4.0)
        assert r1['executed'] == r2['executed']
        assert r1['avg_price'] == r2['avg_price']

    def test_result_structure(self):
        sim = ExecutionSimulator.realistic(seed=42)
        r = sim.simulate_execution("TSLA", "sell", 25, 250.0, volume=10000000, atr=8.0)
        assert 'executed' in r
        assert 'fills' in r
        assert 'avg_price' in r
        assert 'slippage_bps' in r
        assert 'fees' in r
        assert 'latency_ms' in r

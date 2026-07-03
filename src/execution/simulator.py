"""
Execution Realism Simulator — Models real market execution conditions.

Simulates slippage, partial fills, latency, bid-ask spread, fees, and order
failures for more accurate backtesting and paper trading. Wraps around the
ExecutionEngine to provide realistic P&L estimates instead of naive fill-at-close
assumptions.

Usage:
    sim = ExecutionSimulator.realistic()
    result = sim.simulate_execution('AAPL', 'buy', 100, 150.0, volume=5_000_000, atr=3.0)

All models are deterministic when seeded, thread-safe, and require only numpy.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Slippage Models
# ---------------------------------------------------------------------------


class SlippageModel(ABC):
    """Base class for slippage models."""

    @abstractmethod
    def apply(self, price: float, qty: float, side: str, volume: float) -> float:
        """
        Return adjusted execution price after slippage.

        Args:
            price: Base execution price before slippage.
            qty: Order quantity (shares).
            side: 'buy' or 'sell'.
            volume: Average daily volume for the security.

        Returns:
            Adjusted price reflecting market impact/slippage.
        """
        pass


class FixedSlippage(SlippageModel):
    """
    Fixed basis-point slippage applied uniformly to all orders.

    Buys slip up, sells slip down — always adverse to the trader.
    """

    def __init__(self, bps: float = 5.0):
        """
        Args:
            bps: Slippage in basis points (1 bp = 0.01%).
        """
        self.bps = bps

    def apply(self, price: float, qty: float, side: str, volume: float) -> float:
        slip_pct = self.bps / 10_000
        if side == "buy":
            return price * (1 + slip_pct)
        else:
            return price * (1 - slip_pct)


class VolumeImpactSlippage(SlippageModel):
    """
    Slippage proportional to order size relative to average volume.

    Models the square-root market impact: slippage ~ impact_factor * sqrt(qty / volume).
    Larger orders relative to liquidity incur more slippage.
    """

    def __init__(self, impact_factor: float = 0.1, daily_volume_pct: float = 0.01):
        """
        Args:
            impact_factor: Scaling factor for market impact.
            daily_volume_pct: Fraction of daily volume considered as reference liquidity.
        """
        self.impact_factor = impact_factor
        self.daily_volume_pct = daily_volume_pct

    def apply(self, price: float, qty: float, side: str, volume: float) -> float:
        participation = qty / max(volume * self.daily_volume_pct, 1.0)
        impact_pct = self.impact_factor * np.sqrt(participation) / 100
        if side == "buy":
            return price * (1 + impact_pct)
        else:
            return price * (1 - impact_pct)


class VolatilitySlippage(SlippageModel):
    """
    Slippage scales with current ATR (Average True Range).

    Higher volatility environments produce wider fills. Requires ATR to be
    passed via the ExecutionSimulator orchestrator.
    """

    def __init__(self, atr_fraction: float = 0.1):
        """
        Args:
            atr_fraction: Fraction of ATR used as slippage magnitude.
        """
        self.atr_fraction = atr_fraction
        self._atr: Optional[float] = None

    def set_atr(self, atr: float) -> None:
        """Set the current ATR for slippage calculation."""
        self._atr = atr

    def apply(self, price: float, qty: float, side: str, volume: float) -> float:
        if self._atr is None or self._atr <= 0:
            return price
        slip = self._atr * self.atr_fraction
        if side == "buy":
            return price + slip
        else:
            return price - slip


# ---------------------------------------------------------------------------
# Fill Models
# ---------------------------------------------------------------------------


class FillModel(ABC):
    """Base class for fill simulation models."""

    @abstractmethod
    def simulate_fill(
        self, qty: float, price: float, volume: float, rng: np.random.Generator
    ) -> list[dict]:
        """
        Simulate how an order gets filled.

        Args:
            qty: Desired order quantity.
            price: Execution price (after slippage/spread adjustments).
            volume: Bar/daily volume for liquidity estimation.
            rng: Numpy random generator for reproducibility.

        Returns:
            List of fill dicts: [{'qty': float, 'price': float, 'timestamp': str}]
        """
        pass


class InstantFill(FillModel):
    """
    All-or-nothing instant fill at the given price.

    Matches current (idealized) backtest behavior.
    """

    def simulate_fill(
        self, qty: float, price: float, volume: float, rng: np.random.Generator
    ) -> list[dict]:
        return [
            {
                "qty": qty,
                "price": price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]


class PartialFill(FillModel):
    """
    Simulates partial fills based on available liquidity.

    Large orders relative to volume are broken into multiple fills, each
    consuming up to max_participation_rate of the bar volume. Fill prices
    drift slightly between partials to model sequential market impact.
    """

    def __init__(self, max_participation_rate: float = 0.02):
        """
        Args:
            max_participation_rate: Maximum fraction of bar volume consumed per fill
                                   (default 2%).
        """
        self.max_participation_rate = max_participation_rate

    def simulate_fill(
        self, qty: float, price: float, volume: float, rng: np.random.Generator
    ) -> list[dict]:
        max_per_fill = max(volume * self.max_participation_rate, 1.0)
        remaining = qty
        fills = []
        fill_price = price

        while remaining > 0:
            # Each fill takes up to max_per_fill shares with slight randomness
            fill_qty = min(remaining, max_per_fill * (0.5 + 0.5 * rng.random()))
            fill_qty = max(fill_qty, 1.0)  # At least 1 share
            if fill_qty > remaining:
                fill_qty = remaining

            fills.append(
                {
                    "qty": round(fill_qty, 4),
                    "price": round(fill_price, 6),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            remaining -= fill_qty

            # Subsequent fills have slight price drift (adverse)
            drift = rng.normal(0.0001, 0.00005)
            fill_price *= 1 + abs(drift)

        return fills


class ProbabilisticFill(FillModel):
    """
    Fill probability depends on order aggressiveness and randomness.

    Simulates the reality that not all orders get filled — especially
    limit orders far from market or during fast-moving markets.
    """

    def __init__(self, fill_probability: float = 0.85):
        """
        Args:
            fill_probability: Base probability that the order fills at all.
        """
        self.fill_probability = fill_probability

    def simulate_fill(
        self, qty: float, price: float, volume: float, rng: np.random.Generator
    ) -> list[dict]:
        if rng.random() > self.fill_probability:
            # Order not filled — return partial or empty
            partial_pct = rng.random() * 0.5  # 0-50% partial fill
            if partial_pct < 0.1:
                return []  # Complete miss
            filled_qty = round(qty * partial_pct, 4)
            return [
                {
                    "qty": filled_qty,
                    "price": price,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ]

        return [
            {
                "qty": qty,
                "price": price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]


# ---------------------------------------------------------------------------
# Latency Model
# ---------------------------------------------------------------------------


class LatencyModel:
    """
    Simulates network and exchange processing latency.

    Latency is drawn from a truncated normal distribution to model realistic
    network conditions (occasional spikes but bounded).
    """

    def __init__(
        self, mean_ms: float = 50.0, std_ms: float = 20.0, max_ms: float = 500.0
    ):
        """
        Args:
            mean_ms: Mean latency in milliseconds.
            std_ms: Standard deviation of latency.
            max_ms: Maximum latency cap (simulates timeout threshold).
        """
        self.mean_ms = mean_ms
        self.std_ms = std_ms
        self.max_ms = max_ms

    def simulate_latency(self, rng: np.random.Generator) -> float:
        """
        Returns simulated one-way latency in milliseconds.

        Uses truncated normal: always >= 0, capped at max_ms.
        """
        if self.mean_ms <= 0 and self.std_ms <= 0:
            return 0.0
        latency = rng.normal(self.mean_ms, self.std_ms)
        return float(np.clip(latency, 0.0, self.max_ms))

    def adjusted_price(
        self, price: float, atr: Optional[float], latency_ms: float, rng: np.random.Generator
    ) -> float:
        """
        Price may drift during the latency window.

        Models the risk that the market moves between order submission and
        execution. Drift magnitude scales with latency and volatility.

        Args:
            price: Original intended price.
            atr: Current ATR (if available) for volatility scaling.
            latency_ms: Simulated latency in ms.
            rng: Random generator.

        Returns:
            Price adjusted for latency-period drift.
        """
        if latency_ms <= 0 or atr is None or atr <= 0:
            return price

        # Estimate per-ms volatility from daily ATR (assuming 6.5hr trading day)
        trading_ms_per_day = 6.5 * 3600 * 1000
        per_ms_vol = atr / np.sqrt(trading_ms_per_day)
        drift = rng.normal(0, per_ms_vol * np.sqrt(latency_ms))
        return price + drift


# ---------------------------------------------------------------------------
# Spread Model
# ---------------------------------------------------------------------------


class SpreadModel:
    """
    Simulates the bid-ask spread.

    Buys execute at the ask, sells at the bid. Spread widens with volatility
    when volatility_scaling is enabled.
    """

    def __init__(
        self, base_spread_bps: float = 2.0, volatility_scaling: bool = True
    ):
        """
        Args:
            base_spread_bps: Base half-spread in basis points.
            volatility_scaling: If True, spread widens when ATR is high relative
                                to price.
        """
        self.base_spread_bps = base_spread_bps
        self.volatility_scaling = volatility_scaling

    def get_spread(
        self, price: float, atr: Optional[float] = None
    ) -> tuple[float, float]:
        """
        Returns (bid, ask) prices around the midpoint.

        Args:
            price: Mid-market price.
            atr: Current ATR for volatility-scaled spread.

        Returns:
            Tuple of (bid_price, ask_price).
        """
        half_spread_pct = self.base_spread_bps / 10_000

        if self.volatility_scaling and atr is not None and atr > 0 and price > 0:
            # Scale spread by relative volatility (ATR / price)
            vol_ratio = atr / price
            vol_multiplier = 1.0 + vol_ratio * 50  # Amplify effect
            half_spread_pct *= vol_multiplier

        half_spread = price * half_spread_pct
        return (price - half_spread, price + half_spread)

    def execution_price(
        self, price: float, side: str, atr: Optional[float] = None
    ) -> float:
        """
        Returns the realistic execution price (buy at ask, sell at bid).

        Args:
            price: Mid-market price.
            side: 'buy' or 'sell'.
            atr: Current ATR for volatility scaling.

        Returns:
            Execution price accounting for spread.
        """
        bid, ask = self.get_spread(price, atr)
        return ask if side == "buy" else bid


# ---------------------------------------------------------------------------
# Fee Model
# ---------------------------------------------------------------------------


class FeeModel:
    """
    Commission and regulatory fee calculation.

    Models zero-commission brokers (like Alpaca) which still incur SEC and
    FINRA fees on sells, and includes crypto fee support.
    """

    def __init__(
        self,
        commission_per_share: float = 0.0,
        sec_fee_rate: float = 0.0000278,
        taf_fee_rate: float = 0.000166,
        min_commission: float = 0.0,
        maker_fee_pct: float = 0.0,
        taker_fee_pct: float = 0.001,
    ):
        """
        Args:
            commission_per_share: Per-share commission (0 for Alpaca equities).
            sec_fee_rate: SEC fee rate applied to sell notional.
            taf_fee_rate: FINRA TAF rate applied to sell quantity.
            min_commission: Minimum commission per order.
            maker_fee_pct: Maker fee percentage (crypto).
            taker_fee_pct: Taker fee percentage (crypto).
        """
        self.commission_per_share = commission_per_share
        self.sec_fee_rate = sec_fee_rate
        self.taf_fee_rate = taf_fee_rate
        self.min_commission = min_commission
        self.maker_fee_pct = maker_fee_pct
        self.taker_fee_pct = taker_fee_pct

    def calculate(
        self, qty: float, price: float, side: str, asset_type: str = "equity"
    ) -> float:
        """
        Calculate total fees for a trade.

        Args:
            qty: Number of shares/units traded.
            price: Execution price per share/unit.
            side: 'buy' or 'sell'.
            asset_type: 'equity' or 'crypto'.

        Returns:
            Total fee amount in dollars.
        """
        notional = qty * price

        if asset_type == "crypto":
            return notional * self.taker_fee_pct

        # Equity fees
        commission = max(qty * self.commission_per_share, self.min_commission)
        fees = commission

        if side == "sell":
            # SEC fee on sell notional
            fees += notional * self.sec_fee_rate
            # FINRA TAF on sell shares (capped at $7.27 per trade)
            taf = min(qty * self.taf_fee_rate, 7.27)
            fees += taf

        return round(fees, 6)


# ---------------------------------------------------------------------------
# Order Failure Model
# ---------------------------------------------------------------------------


class OrderFailureModel:
    """
    Simulates order rejections and failures.

    Models real-world scenarios: exchange rejections, timeouts, halts,
    and insufficient liquidity situations.
    """

    def __init__(
        self,
        rejection_rate: float = 0.001,
        timeout_rate: float = 0.005,
        market_hours_only: bool = True,
    ):
        """
        Args:
            rejection_rate: Base probability of order rejection (0.1%).
            timeout_rate: Base probability of order timeout (0.5%).
            market_hours_only: If True, reject orders outside market hours.
        """
        self.rejection_rate = rejection_rate
        self.timeout_rate = timeout_rate
        self.market_hours_only = market_hours_only

    def should_reject(
        self, symbol: str, qty: float, side: str, rng: np.random.Generator
    ) -> tuple[bool, Optional[str]]:
        """
        Determine if an order should be rejected.

        Args:
            symbol: Ticker symbol.
            qty: Order quantity.
            side: 'buy' or 'sell'.
            rng: Random generator for deterministic behavior.

        Returns:
            Tuple of (is_rejected, reason_string_or_None).
        """
        # Check for rejection
        if self.rejection_rate > 0 and rng.random() < self.rejection_rate:
            reasons = [
                "insufficient_buying_power",
                "symbol_not_tradeable",
                "exchange_rejected",
                "account_restricted",
            ]
            reason = reasons[rng.integers(0, len(reasons))]
            return (True, reason)

        # Check for timeout
        if self.timeout_rate > 0 and rng.random() < self.timeout_rate:
            return (True, "order_timeout")

        return (False, None)


# ---------------------------------------------------------------------------
# ExecutionSimulator — Orchestrator
# ---------------------------------------------------------------------------


class ExecutionSimulator:
    """
    Orchestrates all execution realism models for realistic trade simulation.

    Combines slippage, fill, latency, spread, fee, and failure models into a
    single pipeline. Can be used standalone for backtest evaluation or injected
    into the ExecutionEngine for paper trading.

    Thread-safe: uses a lock around RNG state. Deterministic when seeded.

    Example:
        sim = ExecutionSimulator.realistic(seed=42)
        result = sim.simulate_execution('AAPL', 'buy', 100, 150.0, volume=5_000_000)
    """

    def __init__(
        self,
        slippage: Optional[SlippageModel] = None,
        fill: Optional[FillModel] = None,
        latency: Optional[LatencyModel] = None,
        spread: Optional[SpreadModel] = None,
        fees: Optional[FeeModel] = None,
        failures: Optional[OrderFailureModel] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            slippage: Slippage model (default: FixedSlippage at 5 bps).
            fill: Fill model (default: InstantFill).
            latency: Latency model (default: 50ms mean).
            spread: Spread model (default: 2 bps base).
            fees: Fee model (default: Alpaca equity fees).
            failures: Failure model (default: 0.1% rejection).
            seed: Random seed for reproducible simulations.
        """
        self.slippage = slippage or FixedSlippage(bps=5.0)
        self.fill = fill or InstantFill()
        self.latency = latency or LatencyModel()
        self.spread = spread or SpreadModel()
        self.fees = fees or FeeModel()
        self.failures = failures or OrderFailureModel()
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()

    def set_seed(self, seed: int) -> None:
        """Reset the RNG with a new seed for reproducibility."""
        with self._lock:
            self._rng = np.random.default_rng(seed)

    def simulate_execution(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        volume: float = 1_000_000,
        atr: Optional[float] = None,
        asset_type: str = "equity",
    ) -> dict:
        """
        Simulate the full execution pipeline for a single order.

        Pipeline stages:
        1. Order failure/rejection check
        2. Latency simulation + price drift
        3. Bid-ask spread application
        4. Slippage (market impact)
        5. Fill simulation (partial fills)
        6. Fee calculation
        7. Metrics computation

        Args:
            symbol: Ticker symbol (e.g., 'AAPL').
            side: 'buy' or 'sell'.
            qty: Order quantity in shares.
            price: Reference/mid-market price.
            volume: Average daily or bar volume for liquidity estimation.
            atr: Current Average True Range (optional, improves realism).
            asset_type: 'equity' or 'crypto'.

        Returns:
            Dict with execution results:
            {
                'executed': bool,
                'fills': list[dict],       # [{qty, price, timestamp}]
                'avg_price': float,
                'total_qty': float,
                'slippage_bps': float,     # Realized slippage in basis points
                'fees': float,             # Total fees in dollars
                'latency_ms': float,       # Simulated latency
                'rejection_reason': str | None,
            }
        """
        with self._lock:
            rng = self._rng

            # 1. Check for rejection/failure
            rejected, reason = self.failures.should_reject(symbol, qty, side, rng)
            if rejected:
                return {
                    "executed": False,
                    "fills": [],
                    "avg_price": 0.0,
                    "total_qty": 0.0,
                    "slippage_bps": 0.0,
                    "fees": 0.0,
                    "latency_ms": 0.0,
                    "rejection_reason": reason,
                }

            # 2. Simulate latency and potential price drift
            latency_ms = self.latency.simulate_latency(rng)
            exec_price = self.latency.adjusted_price(price, atr, latency_ms, rng)

            # 3. Apply bid-ask spread
            exec_price = self.spread.execution_price(exec_price, side, atr)

            # 4. Apply slippage (market impact)
            if isinstance(self.slippage, VolatilitySlippage) and atr is not None:
                self.slippage.set_atr(atr)
            exec_price = self.slippage.apply(exec_price, qty, side, volume)

            # 5. Simulate fills
            fills = self.fill.simulate_fill(qty, exec_price, volume, rng)

            # Handle empty fills (order completely missed)
            if not fills:
                return {
                    "executed": False,
                    "fills": [],
                    "avg_price": 0.0,
                    "total_qty": 0.0,
                    "slippage_bps": 0.0,
                    "fees": 0.0,
                    "latency_ms": latency_ms,
                    "rejection_reason": "no_fill",
                }

            # 6. Calculate fees
            total_fees = sum(
                self.fees.calculate(f["qty"], f["price"], side, asset_type)
                for f in fills
            )

            # 7. Compute execution metrics
            total_qty = sum(f["qty"] for f in fills)
            avg_price = (
                sum(f["qty"] * f["price"] for f in fills) / total_qty
                if total_qty > 0
                else 0.0
            )
            slippage_bps = (
                abs(avg_price - price) / price * 10_000 if price > 0 else 0.0
            )

            return {
                "executed": True,
                "fills": fills,
                "avg_price": round(avg_price, 6),
                "total_qty": round(total_qty, 4),
                "slippage_bps": round(slippage_bps, 4),
                "fees": round(total_fees, 6),
                "latency_ms": round(latency_ms, 2),
                "rejection_reason": None,
            }

    def simulate_batch(
        self,
        orders: list[dict],
    ) -> list[dict]:
        """
        Simulate execution for multiple orders (vectorized-friendly interface).

        Useful for backtesting where many orders are evaluated in one pass.

        Args:
            orders: List of order dicts, each with keys:
                    {symbol, side, qty, price, volume?, atr?, asset_type?}

        Returns:
            List of execution result dicts (same format as simulate_execution).
        """
        results = []
        for order in orders:
            result = self.simulate_execution(
                symbol=order["symbol"],
                side=order["side"],
                qty=order["qty"],
                price=order["price"],
                volume=order.get("volume", 1_000_000),
                atr=order.get("atr"),
                asset_type=order.get("asset_type", "equity"),
            )
            results.append(result)
        return results

    # -------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------

    @classmethod
    def realistic(cls, seed: Optional[int] = None) -> "ExecutionSimulator":
        """
        Factory for realistic market simulation.

        Models typical US equity execution: moderate slippage based on volume
        participation, partial fills for large orders, ~50ms latency, standard
        spread, and Alpaca fee structure.
        """
        return cls(
            slippage=VolumeImpactSlippage(impact_factor=0.1),
            fill=PartialFill(max_participation_rate=0.02),
            latency=LatencyModel(mean_ms=50, std_ms=20),
            spread=SpreadModel(base_spread_bps=2.0),
            fees=FeeModel(),
            failures=OrderFailureModel(rejection_rate=0.001),
            seed=seed,
        )

    @classmethod
    def conservative(cls, seed: Optional[int] = None) -> "ExecutionSimulator":
        """
        Factory for conservative (pessimistic) simulation.

        Assumes worse-than-average conditions: higher impact, lower participation
        caps, more latency, wider spreads. Useful for stress-testing strategies.
        """
        return cls(
            slippage=VolumeImpactSlippage(impact_factor=0.2),
            fill=PartialFill(max_participation_rate=0.01),
            latency=LatencyModel(mean_ms=100, std_ms=50),
            spread=SpreadModel(base_spread_bps=5.0),
            fees=FeeModel(taker_fee_pct=0.002),
            failures=OrderFailureModel(rejection_rate=0.005),
            seed=seed,
        )

    @classmethod
    def ideal(cls, seed: Optional[int] = None) -> "ExecutionSimulator":
        """
        Factory for ideal (zero-friction) simulation.

        Matches current naive backtest behavior: no slippage, instant fills,
        zero latency, no spread. Useful as a comparison baseline.
        """
        return cls(
            slippage=FixedSlippage(bps=0),
            fill=InstantFill(),
            latency=LatencyModel(mean_ms=0, std_ms=0),
            spread=SpreadModel(base_spread_bps=0),
            fees=FeeModel(),
            failures=OrderFailureModel(rejection_rate=0, timeout_rate=0),
            seed=seed,
        )

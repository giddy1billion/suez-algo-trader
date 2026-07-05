"""
Trade Signal Package — A comprehensive execution package for professional-grade signals.

Every signal must be a fully validated execution package that includes:
- Verified entry zone with slippage tolerance
- Stop loss and multiple take-profit levels
- Expected holding period and expiry time
- Confidence backed by a validated model with full provenance
- Time-based exits (entry window, max holding, hard exit)
- Confidence decay schedule
- Strategy contributor attribution
- Market regime and volatility context
- Complete audit trail linking to trained model, backtests, and walk-forward validation

Anything less is treated as an incomplete signal and blocked from execution.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Optional

from src.strategy.base import Signal
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SignalStatus(str, Enum):
    """Signal lifecycle status."""
    PENDING_VALIDATION = "pending_validation"
    READY_FOR_EXECUTION = "ready_for_execution"
    EXECUTING = "executing"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"
    CANCELLED = "cancelled"


class MarketRegime(str, Enum):
    """Current market regime classification."""
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    MEAN_REVERTING = "mean_reverting"


class VolatilityLevel(str, Enum):
    """Volatility classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class TrailingStopMode(str, Enum):
    """Trailing stop activation mode."""
    DISABLED = "disabled"
    IMMEDIATE = "immediate"
    AFTER_TP1 = "after_tp1"
    AFTER_TP2 = "after_tp2"
    AFTER_BREAKEVEN = "after_breakeven"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


@dataclass
class EntryZone:
    """Entry zone specification with slippage tolerance."""
    preferred_min: float
    preferred_max: float
    max_slippage_pct: float = 0.10  # 0.10%

    @property
    def midpoint(self) -> float:
        return (self.preferred_min + self.preferred_max) / 2

    def is_price_acceptable(self, price: float) -> bool:
        """Check if a fill price is within acceptable range including slippage."""
        slippage_buffer = self.midpoint * (self.max_slippage_pct / 100)
        return (self.preferred_min - slippage_buffer) <= price <= (self.preferred_max + slippage_buffer)


@dataclass
class ModelInfo:
    """Complete model provenance and audit trail."""
    model_version: str
    training_run_id: str = ""
    dataset_version: str = ""
    backtest_id: str = ""
    walk_forward_validation_id: str = ""
    training_timestamp: str = ""
    feature_set_version: str = ""
    validation_metrics: dict = field(default_factory=dict)

    def is_complete(self) -> bool:
        """Check if all required provenance fields are present."""
        return bool(
            self.model_version
            and self.training_run_id
            and self.dataset_version
            and self.backtest_id
            and self.walk_forward_validation_id
        )


@dataclass
class StrategyContributor:
    """A strategy that contributed to the signal with its weight."""
    name: str
    weight_pct: float  # Percentage contribution (0-100)
    confirmed: bool = True  # Whether this strategy confirmed the signal


@dataclass
class TakeProfitLevel:
    """A single take-profit target with allocation."""
    price: float
    allocation_pct: float  # Percentage of position to close at this level (0-100)
    expected_time_minutes: Optional[int] = None  # Expected time to reach this level

    def __post_init__(self):
        if not 0 < self.allocation_pct <= 100:
            raise ValueError(f"allocation_pct must be between 0 and 100, got {self.allocation_pct}")


@dataclass
class TimeBasedExit:
    """Time-based exit configuration."""
    entry_window_start: Optional[datetime] = None
    entry_window_end: Optional[datetime] = None
    max_holding_minutes: int = 480  # 8 hours default
    hard_exit_time: Optional[datetime] = None
    max_adverse_excursion_minutes: int = 60  # Max time trade can stay unfavorable

    @property
    def entry_window_duration_minutes(self) -> Optional[int]:
        if self.entry_window_start and self.entry_window_end:
            delta = self.entry_window_end - self.entry_window_start
            return int(delta.total_seconds() / 60)
        return None


@dataclass
class ConfidenceDecay:
    """Confidence decay schedule — signal confidence decreases over time."""
    initial_confidence: float
    decay_rate_per_minute: float = 0.001  # ~6% per hour
    invalidation_threshold: float = 0.65  # Auto-invalidate below this

    def confidence_at(self, elapsed_minutes: float) -> float:
        """Calculate confidence at a given elapsed time."""
        decayed = self.initial_confidence - (self.decay_rate_per_minute * elapsed_minutes)
        return max(0.0, decayed)

    def minutes_until_invalidation(self) -> float:
        """Calculate minutes until confidence drops below threshold."""
        if self.initial_confidence <= self.invalidation_threshold:
            return 0.0
        return (self.initial_confidence - self.invalidation_threshold) / self.decay_rate_per_minute

    def is_valid_at(self, elapsed_minutes: float) -> bool:
        """Check if signal is still valid at given elapsed time."""
        return self.confidence_at(elapsed_minutes) >= self.invalidation_threshold


# ---------------------------------------------------------------------------
# Main Signal Package
# ---------------------------------------------------------------------------


def _generate_signal_id() -> str:
    """Generate a unique signal ID with timestamp prefix."""
    now = datetime.now(timezone.utc)
    return f"SIG-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"


@dataclass
class TradeSignalPackage:
    """
    A complete trade execution package — professional-grade signal.

    This is the fundamental unit of trade execution. No trade should be
    executed unless all required fields are present and validated.
    A signal that fails validation is blocked from execution.
    """

    # --- Identity ---
    signal_id: str = field(default_factory=_generate_signal_id)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # --- Core Signal ---
    symbol: str = ""
    direction: Signal = Signal.HOLD
    entry_zone: Optional[EntryZone] = None

    # --- Confidence ---
    confidence: float = 0.0
    confidence_decay: Optional[ConfidenceDecay] = None

    # --- Model Provenance ---
    model_info: Optional[ModelInfo] = None

    # --- Strategy Attribution ---
    strategy_contributors: list[StrategyContributor] = field(default_factory=list)

    # --- Timeframe ---
    signal_timeframe: str = ""  # e.g., "15m", "1h"
    trade_horizon_minutes: int = 0  # Expected trade duration
    max_holding_minutes: int = 0  # Maximum holding time

    # --- Risk Management ---
    stop_loss: float = 0.0
    take_profit_levels: list[TakeProfitLevel] = field(default_factory=list)
    trailing_stop_mode: TrailingStopMode = TrailingStopMode.DISABLED
    trailing_stop_distance_pct: float = 0.0

    # --- Expected Metrics ---
    expected_risk_reward: float = 0.0  # e.g., 3.4 means 1:3.4
    expected_win_probability: float = 0.0
    expected_return_pct: float = 0.0
    max_expected_drawdown_pct: float = 0.0

    # --- Market Context ---
    market_regime: MarketRegime = MarketRegime.RANGING
    volatility: VolatilityLevel = VolatilityLevel.MEDIUM

    # --- Position Sizing ---
    position_size_pct: float = 0.0  # % of portfolio

    # --- Time-Based Exit ---
    time_based_exit: Optional[TimeBasedExit] = None
    signal_expiry_minutes: int = 90  # Cancel if entry hasn't occurred

    # --- Reasons / Audit ---
    reasons: list[str] = field(default_factory=list)
    indicators: dict[str, Any] = field(default_factory=dict)

    # --- Status ---
    status: SignalStatus = SignalStatus.PENDING_VALIDATION

    # ──────────────────────────────────────────────────────────────────────
    # Derived Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def is_buy(self) -> bool:
        return self.direction in (Signal.BUY, Signal.STRONG_BUY)

    @property
    def is_sell(self) -> bool:
        return self.direction in (Signal.SELL, Signal.STRONG_SELL)

    @property
    def side(self) -> str:
        if self.is_buy:
            return "buy"
        elif self.is_sell:
            return "sell"
        return "hold"

    @property
    def expiry_time(self) -> datetime:
        """Time at which signal expires if entry hasn't triggered."""
        return self.generated_at + timedelta(minutes=self.signal_expiry_minutes)

    @property
    def is_expired(self) -> bool:
        """Check if signal has expired based on current time."""
        return datetime.now(timezone.utc) > self.expiry_time

    @property
    def current_confidence(self) -> float:
        """Get confidence adjusted for time decay."""
        if self.confidence_decay is None:
            return self.confidence
        elapsed = (datetime.now(timezone.utc) - self.generated_at).total_seconds() / 60
        return self.confidence_decay.confidence_at(elapsed)

    @property
    def total_take_profit_allocation(self) -> float:
        """Sum of all TP level allocations (should be 100%)."""
        return sum(tp.allocation_pct for tp in self.take_profit_levels)

    @property
    def total_strategy_weight(self) -> float:
        """Sum of all strategy contributor weights (should be 100%)."""
        return sum(s.weight_pct for s in self.strategy_contributors)

    # ──────────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────────

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate that this signal package is complete and executable.

        Returns:
            (is_valid, list_of_errors)

        A signal is only executable if ALL validations pass.
        """
        errors: list[str] = []

        # Core fields
        if not self.symbol:
            errors.append("Missing symbol")
        if self.direction in (Signal.HOLD, Signal.NO_SIGNAL):
            errors.append("Direction must be BUY/SELL, not HOLD/NO_SIGNAL")
        if self.confidence <= 0:
            errors.append("Confidence must be positive")

        # Entry zone
        if self.entry_zone is None:
            errors.append("Missing entry zone")
        elif self.entry_zone.preferred_min >= self.entry_zone.preferred_max:
            errors.append("Entry zone min must be less than max")

        # Stop loss
        if self.stop_loss <= 0:
            errors.append("Missing or invalid stop loss")
        elif self.entry_zone:
            mid = self.entry_zone.midpoint
            if self.is_buy and self.stop_loss >= mid:
                errors.append("Stop loss must be below entry for BUY")
            elif self.is_sell and self.stop_loss <= mid:
                errors.append("Stop loss must be above entry for SELL")

        # Take profit levels
        if not self.take_profit_levels:
            errors.append("At least one take-profit level required")
        else:
            tp_alloc = self.total_take_profit_allocation
            if abs(tp_alloc - 100.0) > 0.01:
                errors.append(f"TP allocations must sum to 100%, got {tp_alloc:.1f}%")
            # Validate TP direction
            if self.entry_zone:
                mid = self.entry_zone.midpoint
                for i, tp in enumerate(self.take_profit_levels):
                    if self.is_buy and tp.price <= mid:
                        errors.append(f"TP{i+1} must be above entry for BUY")
                    elif self.is_sell and tp.price >= mid:
                        errors.append(f"TP{i+1} must be below entry for SELL")

        # Model provenance
        if self.model_info is None:
            errors.append("Missing model information")
        elif not self.model_info.is_complete():
            errors.append("Incomplete model provenance (requires version, training_run, dataset, backtest, walk_forward)")

        # Strategy contributors
        if not self.strategy_contributors:
            errors.append("At least one strategy contributor required")
        elif abs(self.total_strategy_weight - 100.0) > 0.01:
            errors.append(f"Strategy weights must sum to 100%, got {self.total_strategy_weight:.1f}%")

        # Timeframe
        if not self.signal_timeframe:
            errors.append("Missing signal timeframe")
        if self.max_holding_minutes <= 0:
            errors.append("Max holding time must be positive")

        # Expected metrics
        if self.expected_risk_reward <= 0:
            errors.append("Expected risk/reward must be positive")
        if not (0 < self.expected_win_probability <= 1.0):
            errors.append("Expected win probability must be between 0 and 1")

        # Position sizing
        if self.position_size_pct <= 0:
            errors.append("Position size must be positive")
        if self.position_size_pct > 25.0:
            errors.append("Position size exceeds 25% safety limit")

        # Time-based exit
        if self.time_based_exit is None:
            errors.append("Missing time-based exit configuration")

        # Confidence decay
        if self.confidence_decay is None:
            errors.append("Missing confidence decay configuration")

        # Reasons
        if not self.reasons:
            errors.append("At least one reason/justification required")

        is_valid = len(errors) == 0
        if is_valid:
            self.status = SignalStatus.READY_FOR_EXECUTION
        return is_valid, errors

    def mark_ready(self) -> bool:
        """Validate and mark signal as ready for execution. Returns True if valid."""
        is_valid, errors = self.validate()
        if not is_valid:
            logger.warning(
                "signal_package.validation_failed",
                signal_id=self.signal_id,
                symbol=self.symbol,
                errors=errors,
            )
        return is_valid

    def invalidate(self, reason: str = ""):
        """Invalidate this signal (e.g., due to confidence decay or expiry)."""
        self.status = SignalStatus.INVALIDATED
        if reason:
            self.reasons.append(f"INVALIDATED: {reason}")
        logger.info(
            "signal_package.invalidated",
            signal_id=self.signal_id,
            symbol=self.symbol,
            reason=reason,
        )

    def check_expiry(self) -> bool:
        """
        Check if signal should be invalidated due to expiry or confidence decay.
        Returns True if signal is still valid.
        """
        # Time expiry
        if self.is_expired:
            self.invalidate("Signal expired — entry not triggered within window")
            return False

        # Confidence decay
        if self.confidence_decay:
            elapsed = (datetime.now(timezone.utc) - self.generated_at).total_seconds() / 60
            if not self.confidence_decay.is_valid_at(elapsed):
                current = self.confidence_decay.confidence_at(elapsed)
                self.invalidate(
                    f"Confidence decayed below threshold: "
                    f"{current:.1%} < {self.confidence_decay.invalidation_threshold:.1%}"
                )
                return False

        return True

    # ──────────────────────────────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage/transmission."""
        return {
            "signal_id": self.signal_id,
            "generated_at": self.generated_at.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction.name,
            "entry_zone": {
                "preferred_min": self.entry_zone.preferred_min,
                "preferred_max": self.entry_zone.preferred_max,
                "max_slippage_pct": self.entry_zone.max_slippage_pct,
            } if self.entry_zone else None,
            "confidence": self.confidence,
            "model_info": {
                "model_version": self.model_info.model_version,
                "training_run_id": self.model_info.training_run_id,
                "dataset_version": self.model_info.dataset_version,
                "backtest_id": self.model_info.backtest_id,
                "walk_forward_validation_id": self.model_info.walk_forward_validation_id,
            } if self.model_info else None,
            "strategy_contributors": [
                {"name": s.name, "weight_pct": s.weight_pct, "confirmed": s.confirmed}
                for s in self.strategy_contributors
            ],
            "signal_timeframe": self.signal_timeframe,
            "trade_horizon_minutes": self.trade_horizon_minutes,
            "max_holding_minutes": self.max_holding_minutes,
            "stop_loss": self.stop_loss,
            "take_profit_levels": [
                {"price": tp.price, "allocation_pct": tp.allocation_pct,
                 "expected_time_minutes": tp.expected_time_minutes}
                for tp in self.take_profit_levels
            ],
            "trailing_stop_mode": self.trailing_stop_mode.value,
            "trailing_stop_distance_pct": self.trailing_stop_distance_pct,
            "expected_risk_reward": self.expected_risk_reward,
            "expected_win_probability": self.expected_win_probability,
            "expected_return_pct": self.expected_return_pct,
            "max_expected_drawdown_pct": self.max_expected_drawdown_pct,
            "market_regime": self.market_regime.value,
            "volatility": self.volatility.value,
            "position_size_pct": self.position_size_pct,
            "time_based_exit": {
                "entry_window_start": self.time_based_exit.entry_window_start.isoformat()
                    if self.time_based_exit.entry_window_start else None,
                "entry_window_end": self.time_based_exit.entry_window_end.isoformat()
                    if self.time_based_exit.entry_window_end else None,
                "max_holding_minutes": self.time_based_exit.max_holding_minutes,
                "hard_exit_time": self.time_based_exit.hard_exit_time.isoformat()
                    if self.time_based_exit.hard_exit_time else None,
                "max_adverse_excursion_minutes": self.time_based_exit.max_adverse_excursion_minutes,
            } if self.time_based_exit else None,
            "signal_expiry_minutes": self.signal_expiry_minutes,
            "confidence_decay": {
                "initial_confidence": self.confidence_decay.initial_confidence,
                "decay_rate_per_minute": self.confidence_decay.decay_rate_per_minute,
                "invalidation_threshold": self.confidence_decay.invalidation_threshold,
            } if self.confidence_decay else None,
            "reasons": self.reasons,
            "indicators": self.indicators,
            "status": self.status.value,
        }

    def summary(self) -> str:
        """Human-readable signal summary."""
        lines = [
            f"Signal ID: {self.signal_id}",
            f"Symbol: {self.symbol}",
            f"Direction: {self.direction.name}",
            "",
        ]
        if self.entry_zone:
            lines.append(f"Entry Zone: {self.entry_zone.preferred_min:,.2f}–{self.entry_zone.preferred_max:,.2f}")
            lines.append(f"Max Slippage: {self.entry_zone.max_slippage_pct:.2f}%")
        lines.append(f"Confidence: {self.confidence:.1%}")
        lines.append(f"Stop Loss: {self.stop_loss:,.2f}")
        for i, tp in enumerate(self.take_profit_levels, 1):
            lines.append(f"TP{i}: {tp.price:,.2f} ({tp.allocation_pct:.0f}%)")
        lines.append(f"Risk/Reward: 1:{self.expected_risk_reward:.1f}")
        lines.append(f"Win Prob: {self.expected_win_probability:.0%}")
        lines.append(f"Market Regime: {self.market_regime.value}")
        lines.append(f"Status: {self.status.value}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Signal Validation Gate
# ---------------------------------------------------------------------------


class SignalValidationGate:
    """
    Enforces that only complete, validated signal packages can proceed to execution.

    This is the single checkpoint that blocks incomplete signals from reaching
    the execution engine. Any signal that fails validation is rejected with
    a detailed reason.
    """

    def __init__(
        self,
        require_model_provenance: bool = True,
        min_confidence: float = 0.55,
        max_position_size_pct: float = 25.0,
        min_risk_reward: float = 1.0,
    ):
        self.require_model_provenance = require_model_provenance
        self.min_confidence = min_confidence
        self.max_position_size_pct = max_position_size_pct
        self.min_risk_reward = min_risk_reward

    def evaluate(self, package: TradeSignalPackage) -> tuple[bool, list[str]]:
        """
        Evaluate a signal package for execution readiness.

        Returns:
            (approved, list_of_rejection_reasons)
        """
        # First run the package's internal validation
        is_valid, errors = package.validate()
        if not is_valid:
            return False, errors

        # Additional gate-level checks
        gate_errors: list[str] = []

        # Confidence check
        if package.confidence < self.min_confidence:
            gate_errors.append(
                f"Confidence {package.confidence:.1%} below gate minimum {self.min_confidence:.1%}"
            )

        # Risk/reward check
        if package.expected_risk_reward < self.min_risk_reward:
            gate_errors.append(
                f"Risk/reward {package.expected_risk_reward:.1f} below minimum {self.min_risk_reward:.1f}"
            )

        # Position size cap
        if package.position_size_pct > self.max_position_size_pct:
            gate_errors.append(
                f"Position size {package.position_size_pct:.1f}% exceeds max {self.max_position_size_pct:.1f}%"
            )

        # Expiry check
        if package.is_expired:
            gate_errors.append("Signal has already expired")

        # Confidence decay check
        if package.confidence_decay:
            elapsed = (datetime.now(timezone.utc) - package.generated_at).total_seconds() / 60
            if not package.confidence_decay.is_valid_at(elapsed):
                gate_errors.append("Signal confidence has decayed below threshold")

        # Model provenance (can be relaxed for non-ML strategies)
        if self.require_model_provenance:
            if package.model_info and not package.model_info.is_complete():
                gate_errors.append("Incomplete model provenance — cannot verify signal origin")

        approved = len(gate_errors) == 0
        if approved:
            package.status = SignalStatus.READY_FOR_EXECUTION
            logger.info(
                "signal_gate.approved",
                signal_id=package.signal_id,
                symbol=package.symbol,
                confidence=package.confidence,
            )
        else:
            logger.warning(
                "signal_gate.rejected",
                signal_id=package.signal_id,
                symbol=package.symbol,
                errors=gate_errors,
            )

        return approved, gate_errors

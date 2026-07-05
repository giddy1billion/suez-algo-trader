"""
Signal Bridge — Converts basic TradeSignal into professional-grade TradeSignalPackage.

This module is the integration layer between:
  1. Strategy output (basic TradeSignal with price, direction, confidence)
  2. Full execution package (TradeSignalPackage with entry zone, TP levels, decay, etc.)

The builder enriches signals using available context:
  - Market regime from intelligence orchestrator
  - Model provenance from ML strategy metadata
  - Position sizing from risk engine
  - Volatility/ATR from market data
  - Time-based exits from strategy configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import numpy as np

from src.strategy.base import TradeSignal, Signal
from src.strategy.signal_package import (
    TradeSignalPackage,
    EntryZone,
    ModelInfo,
    StrategyContributor,
    TakeProfitLevel,
    TimeBasedExit,
    ConfidenceDecay,
    SignalStatus,
    MarketRegime,
    VolatilityLevel,
    TrailingStopMode,
    SignalValidationGate,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Regime mapping from intelligence layer strings to signal_package enums
# ---------------------------------------------------------------------------

_REGIME_MAP = {
    "trending_bullish": MarketRegime.TRENDING_BULLISH,
    "trending_bearish": MarketRegime.TRENDING_BEARISH,
    "bullish": MarketRegime.TRENDING_BULLISH,
    "bearish": MarketRegime.TRENDING_BEARISH,
    "ranging": MarketRegime.RANGING,
    "sideways": MarketRegime.RANGING,
    "high_volatility": MarketRegime.HIGH_VOLATILITY,
    "low_volatility": MarketRegime.LOW_VOLATILITY,
    "breakout": MarketRegime.BREAKOUT,
    "mean_reverting": MarketRegime.MEAN_REVERTING,
}


# ---------------------------------------------------------------------------
# Builder Configuration
# ---------------------------------------------------------------------------


@dataclass
class SignalBridgeConfig:
    """Configuration for signal enrichment."""

    # Entry zone
    entry_zone_spread_pct: float = 0.05  # +/- from signal price
    max_slippage_pct: float = 0.10

    # TP levels (default 3 levels)
    tp1_multiplier: float = 1.5  # 1.5x risk
    tp2_multiplier: float = 2.5  # 2.5x risk
    tp3_multiplier: float = 4.0  # 4.0x risk
    tp1_allocation: float = 30.0
    tp2_allocation: float = 40.0
    tp3_allocation: float = 30.0

    # Time-based exit defaults
    max_holding_minutes: int = 480  # 8 hours
    signal_expiry_minutes: int = 90
    max_adverse_excursion_minutes: int = 60

    # Confidence decay
    decay_rate_per_minute: float = 0.001  # ~6% per hour
    invalidation_threshold: float = 0.65

    # Trailing stop
    trailing_stop_mode: TrailingStopMode = TrailingStopMode.AFTER_TP1
    trailing_stop_distance_pct: float = 1.5

    # Strict mode: require all fields for validation
    strict_mode: bool = False


# ---------------------------------------------------------------------------
# Signal Package Builder
# ---------------------------------------------------------------------------


class SignalPackageBuilder:
    """
    Builds a complete TradeSignalPackage from a basic TradeSignal
    by enriching it with context from the trading system.

    Usage:
        builder = SignalPackageBuilder(config)
        package = builder.build(
            signal=trade_signal,
            strategy_name="momentum",
            market_data=df,
            intelligence_decision=decision,
            model_metadata=model_info_dict,
            portfolio_value=100000,
            position_size_pct=2.5,
        )
    """

    def __init__(self, config: Optional[SignalBridgeConfig] = None):
        self.config = config or SignalBridgeConfig()

    def build(
        self,
        signal: TradeSignal,
        strategy_name: str = "unknown",
        market_data=None,
        intelligence_decision=None,
        model_metadata: Optional[dict] = None,
        portfolio_value: float = 0.0,
        position_size_pct: float = 0.0,
        additional_contributors: Optional[list[dict]] = None,
    ) -> TradeSignalPackage:
        """
        Build a complete TradeSignalPackage from a basic TradeSignal.

        Args:
            signal: Basic TradeSignal from strategy
            strategy_name: Name of the primary strategy
            market_data: DataFrame with OHLCV for ATR/vol calculation
            intelligence_decision: IntelligenceDecision from orchestrator
            model_metadata: Dict with model version, training_run_id, etc.
            portfolio_value: Current portfolio value for sizing
            position_size_pct: Pre-calculated position size as % of portfolio
            additional_contributors: Extra strategy contributors [{name, weight_pct}]

        Returns:
            Enriched TradeSignalPackage ready for validation
        """
        now = datetime.now(timezone.utc)

        # --- Entry Zone ---
        entry_zone = self._build_entry_zone(signal, market_data)

        # --- Market context ---
        regime, volatility = self._extract_market_context(
            intelligence_decision, market_data
        )

        # --- Model info ---
        model_info = self._build_model_info(model_metadata)

        # --- Strategy contributors ---
        contributors = self._build_contributors(
            strategy_name, signal, intelligence_decision, additional_contributors
        )

        # --- Take profit levels ---
        tp_levels = self._build_tp_levels(signal, market_data)

        # --- Time-based exit ---
        time_exit = self._build_time_exit(now)

        # --- Confidence decay ---
        confidence_decay = ConfidenceDecay(
            initial_confidence=signal.confidence,
            decay_rate_per_minute=self.config.decay_rate_per_minute,
            invalidation_threshold=self.config.invalidation_threshold,
        )

        # --- Expected metrics ---
        risk_reward = self._calculate_risk_reward(signal, tp_levels)
        win_prob = self._estimate_win_probability(signal, intelligence_decision)

        # --- Position sizing ---
        if position_size_pct <= 0 and portfolio_value > 0 and signal.price > 0:
            # Fallback: estimate from risk params
            position_size_pct = self._estimate_position_size(
                signal, portfolio_value
            )

        # --- Build package ---
        # Determine if model provenance should be required
        has_full_provenance = model_info is not None and model_info.is_complete()

        package = TradeSignalPackage(
            generated_at=now,
            symbol=signal.symbol,
            direction=signal.signal,
            entry_zone=entry_zone,
            confidence=signal.confidence,
            confidence_decay=confidence_decay,
            model_info=model_info,
            strategy_contributors=contributors,
            signal_timeframe=self._infer_timeframe(market_data),
            trade_horizon_minutes=self.config.max_holding_minutes // 2,
            max_holding_minutes=self.config.max_holding_minutes,
            stop_loss=signal.stop_loss or 0.0,
            take_profit_levels=tp_levels,
            trailing_stop_mode=self.config.trailing_stop_mode,
            trailing_stop_distance_pct=self.config.trailing_stop_distance_pct,
            expected_risk_reward=risk_reward,
            expected_win_probability=win_prob,
            expected_return_pct=self._estimate_return(signal, tp_levels),
            max_expected_drawdown_pct=self._estimate_max_drawdown(signal),
            market_regime=regime,
            volatility=volatility,
            position_size_pct=position_size_pct,
            time_based_exit=time_exit,
            signal_expiry_minutes=self.config.signal_expiry_minutes,
            reasons=self._build_reasons(signal, intelligence_decision),
            indicators=signal.indicators or {},
            status=SignalStatus.PENDING_VALIDATION,
            require_model_provenance=has_full_provenance,
        )

        return package

    # ──────────────────────────────────────────────────────────────────────
    # Private builders
    # ──────────────────────────────────────────────────────────────────────

    def _build_entry_zone(self, signal: TradeSignal, market_data) -> EntryZone:
        """Build entry zone from signal price and ATR-based spread."""
        price = signal.price
        spread_pct = self.config.entry_zone_spread_pct

        # Use ATR for smarter spread if market data available
        if market_data is not None and len(market_data) >= 14:
            atr = self._calculate_atr(market_data, period=14)
            if atr > 0:
                # ATR-based spread: half ATR on each side
                spread_pct = (atr / price) * 50  # Convert to percentage

        half_spread = price * (spread_pct / 100)
        return EntryZone(
            preferred_min=round(price - half_spread, 6),
            preferred_max=round(price + half_spread, 6),
            max_slippage_pct=self.config.max_slippage_pct,
        )

    def _extract_market_context(
        self, intelligence_decision, market_data
    ) -> tuple[MarketRegime, VolatilityLevel]:
        """Extract market regime and volatility from intelligence or data."""
        regime = MarketRegime.RANGING
        volatility = VolatilityLevel.MEDIUM

        if intelligence_decision:
            # Map from intelligence layer regime string
            raw_regime = getattr(
                intelligence_decision.market_state, "overall_regime", ""
            )
            regime = _REGIME_MAP.get(raw_regime.lower(), MarketRegime.RANGING)

            # Volatility from diagnostics
            diag = getattr(intelligence_decision.market_state, "diagnostics", {})
            vol_level = diag.get("volatility_level", "")
            if "high" in vol_level.lower():
                volatility = VolatilityLevel.HIGH
            elif "low" in vol_level.lower():
                volatility = VolatilityLevel.LOW
            elif "extreme" in vol_level.lower():
                volatility = VolatilityLevel.EXTREME

        elif market_data is not None and len(market_data) >= 20:
            # Fallback: estimate from recent returns
            returns = market_data["close"].pct_change().dropna()
            if len(returns) >= 10:
                daily_vol = returns.tail(20).std()
                if daily_vol > 0.04:
                    volatility = VolatilityLevel.EXTREME
                elif daily_vol > 0.025:
                    volatility = VolatilityLevel.HIGH
                elif daily_vol < 0.008:
                    volatility = VolatilityLevel.LOW

        return regime, volatility

    def _build_model_info(self, model_metadata: Optional[dict]) -> Optional[ModelInfo]:
        """Build model provenance info from metadata dict."""
        if not model_metadata:
            # Return a minimal model info for non-ML strategies
            return ModelInfo(model_version="rule-based-v1")

        return ModelInfo(
            model_version=model_metadata.get("model_version", "unknown"),
            training_run_id=model_metadata.get("training_run_id", ""),
            dataset_version=model_metadata.get("dataset_version", ""),
            backtest_id=model_metadata.get("backtest_id", ""),
            walk_forward_validation_id=model_metadata.get(
                "walk_forward_validation_id", ""
            ),
            training_timestamp=model_metadata.get("training_timestamp", ""),
            feature_set_version=model_metadata.get("feature_set_version", ""),
            validation_metrics=model_metadata.get("validation_metrics", {}),
        )

    def _build_contributors(
        self,
        strategy_name: str,
        signal: TradeSignal,
        intelligence_decision,
        additional_contributors: Optional[list[dict]],
    ) -> list[StrategyContributor]:
        """Build strategy contributor list."""
        contributors = []

        if additional_contributors:
            for c in additional_contributors:
                contributors.append(
                    StrategyContributor(
                        name=c.get("name", "unknown"),
                        weight_pct=c.get("weight_pct", 0.0),
                        confirmed=c.get("confirmed", True),
                    )
                )
        else:
            # Single strategy as sole contributor
            contributors.append(
                StrategyContributor(
                    name=strategy_name,
                    weight_pct=100.0,
                    confirmed=True,
                )
            )

        return contributors

    def _build_tp_levels(
        self, signal: TradeSignal, market_data
    ) -> list[TakeProfitLevel]:
        """Build multi-level take profit targets."""
        if not signal.stop_loss or not signal.price:
            # Can't calculate TP without risk reference
            if signal.take_profit:
                return [TakeProfitLevel(price=signal.take_profit, allocation_pct=100.0)]
            return []

        price = signal.price
        is_buy = signal.signal in (Signal.BUY, Signal.STRONG_BUY)
        risk = abs(price - signal.stop_loss)

        if risk <= 0:
            if signal.take_profit:
                return [TakeProfitLevel(price=signal.take_profit, allocation_pct=100.0)]
            return []

        # Calculate TP levels as multiples of risk
        levels = []
        multipliers = [
            (self.config.tp1_multiplier, self.config.tp1_allocation),
            (self.config.tp2_multiplier, self.config.tp2_allocation),
            (self.config.tp3_multiplier, self.config.tp3_allocation),
        ]

        for mult, alloc in multipliers:
            if is_buy:
                tp_price = price + (risk * mult)
            else:
                tp_price = price - (risk * mult)

            # Estimate time to reach based on ATR
            expected_minutes = None
            if market_data is not None and len(market_data) >= 14:
                atr = self._calculate_atr(market_data, period=14)
                if atr > 0:
                    bars_to_target = abs(tp_price - price) / atr
                    expected_minutes = int(bars_to_target * 60)  # Assume 1h bars

            levels.append(
                TakeProfitLevel(
                    price=round(tp_price, 6),
                    allocation_pct=alloc,
                    expected_time_minutes=expected_minutes,
                )
            )

        return levels

    def _build_time_exit(self, now: datetime) -> TimeBasedExit:
        """Build time-based exit configuration."""
        return TimeBasedExit(
            entry_window_start=now,
            entry_window_end=now + timedelta(minutes=self.config.signal_expiry_minutes),
            max_holding_minutes=self.config.max_holding_minutes,
            hard_exit_time=now + timedelta(minutes=self.config.max_holding_minutes),
            max_adverse_excursion_minutes=self.config.max_adverse_excursion_minutes,
        )

    def _build_reasons(
        self, signal: TradeSignal, intelligence_decision
    ) -> list[str]:
        """Build reasons list from signal and intelligence."""
        reasons = []

        if signal.reason:
            reasons.append(signal.reason)

        # Extract indicator-based reasons
        indicators = signal.indicators or {}
        if indicators.get("ema_cross"):
            reasons.append("EMA crossover confirmed")
        if indicators.get("rsi_oversold"):
            reasons.append("RSI in oversold territory")
        if indicators.get("rsi_overbought"):
            reasons.append("RSI in overbought territory")
        if indicators.get("volume_surge"):
            reasons.append("Volume surge detected")
        if indicators.get("macd_signal"):
            reasons.append("MACD confirmation")
        if indicators.get("bb_squeeze"):
            reasons.append("Bollinger Band squeeze")

        if intelligence_decision:
            if hasattr(intelligence_decision, "explanation") and intelligence_decision.explanation:
                reasons.append(f"Intelligence: {intelligence_decision.explanation}")

        # Ensure at least one reason
        if not reasons:
            reasons.append(f"Signal from {signal.signal.name} with confidence {signal.confidence:.1%}")

        return reasons

    def _calculate_risk_reward(
        self, signal: TradeSignal, tp_levels: list[TakeProfitLevel]
    ) -> float:
        """Calculate expected risk/reward ratio."""
        if not signal.stop_loss or not signal.price or not tp_levels:
            return 0.0

        risk = abs(signal.price - signal.stop_loss)
        if risk <= 0:
            return 0.0

        # Weighted average reward across TP levels
        total_reward = 0.0
        total_weight = 0.0
        for tp in tp_levels:
            reward = abs(tp.price - signal.price)
            total_reward += reward * (tp.allocation_pct / 100)
            total_weight += tp.allocation_pct / 100

        if total_weight <= 0:
            return 0.0

        avg_reward = total_reward / total_weight
        return round(avg_reward / risk, 2)

    def _estimate_win_probability(
        self, signal: TradeSignal, intelligence_decision
    ) -> float:
        """Estimate win probability from confidence and intelligence score."""
        base_prob = signal.confidence

        if intelligence_decision:
            # Blend signal confidence with intelligence score
            intel_score = intelligence_decision.final_score
            base_prob = (base_prob * 0.6) + (intel_score * 0.4)

        return min(max(round(base_prob, 3), 0.01), 0.99)

    def _estimate_return(
        self, signal: TradeSignal, tp_levels: list[TakeProfitLevel]
    ) -> float:
        """Estimate expected return percentage."""
        if not signal.price or not tp_levels:
            return 0.0

        # Weighted average TP distance
        total = 0.0
        for tp in tp_levels:
            dist_pct = abs(tp.price - signal.price) / signal.price * 100
            total += dist_pct * (tp.allocation_pct / 100)

        return round(total, 2)

    def _estimate_max_drawdown(self, signal: TradeSignal) -> float:
        """Estimate max drawdown from stop loss."""
        if not signal.stop_loss or not signal.price:
            return 5.0  # Default conservative estimate
        return round(abs(signal.price - signal.stop_loss) / signal.price * 100, 2)

    def _estimate_position_size(
        self, signal: TradeSignal, portfolio_value: float
    ) -> float:
        """Estimate position size as % of portfolio from risk parameters."""
        if not signal.stop_loss or signal.price <= 0:
            return 2.0  # Default 2%

        risk_per_share = abs(signal.price - signal.stop_loss)
        # Risk 2% of portfolio
        risk_amount = portfolio_value * 0.02
        qty = risk_amount / risk_per_share if risk_per_share > 0 else 0
        position_value = qty * signal.price
        return round(min(position_value / portfolio_value * 100, 25.0), 2)

    def _infer_timeframe(self, market_data) -> str:
        """Infer timeframe from market data index spacing."""
        if market_data is None or len(market_data) < 2:
            return "1Hour"

        try:
            idx = market_data.index
            if hasattr(idx, "to_series"):
                deltas = idx.to_series().diff().dropna()
                if len(deltas) > 0:
                    median_seconds = deltas.dt.total_seconds().median()
                    if median_seconds <= 120:
                        return "1Min"
                    elif median_seconds <= 400:
                        return "5Min"
                    elif median_seconds <= 1200:
                        return "15Min"
                    elif median_seconds <= 5400:
                        return "1Hour"
                    else:
                        return "1Day"
        except Exception:
            pass
        return "1Hour"

    @staticmethod
    def _calculate_atr(df, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(df) < period or "high" not in df.columns or "low" not in df.columns:
            return 0.0

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )

        if len(tr) < period:
            return float(np.mean(tr)) if len(tr) > 0 else 0.0

        return float(np.mean(tr[-period:]))


# ---------------------------------------------------------------------------
# Active Signal Monitor
# ---------------------------------------------------------------------------


class ActiveSignalMonitor:
    """
    Monitors active signal packages for expiry and confidence decay.

    Call `check_all()` periodically (e.g., every 30s in main loop) to
    auto-invalidate stale signals.
    """

    def __init__(self):
        self._active_packages: dict[str, TradeSignalPackage] = {}

    @property
    def active_count(self) -> int:
        return len(self._active_packages)

    def register(self, package: TradeSignalPackage) -> None:
        """Register a validated package for monitoring."""
        self._active_packages[package.signal_id] = package

    def unregister(self, signal_id: str) -> None:
        """Remove a package from monitoring (filled or cancelled)."""
        self._active_packages.pop(signal_id, None)

    def check_all(self) -> list[TradeSignalPackage]:
        """
        Check all active packages for expiry/decay.

        Returns list of packages that were invalidated.
        """
        invalidated = []
        to_remove = []

        for signal_id, package in self._active_packages.items():
            if package.status in (
                SignalStatus.FILLED,
                SignalStatus.CANCELLED,
                SignalStatus.INVALIDATED,
            ):
                to_remove.append(signal_id)
                continue

            if not package.check_expiry():
                invalidated.append(package)
                to_remove.append(signal_id)

        for signal_id in to_remove:
            self._active_packages.pop(signal_id, None)

        if invalidated:
            logger.info(
                "signal_monitor.invalidated_signals",
                count=len(invalidated),
                symbols=[p.symbol for p in invalidated],
            )

        return invalidated

    def get_active(self) -> list[TradeSignalPackage]:
        """Get all currently active signal packages."""
        return list(self._active_packages.values())

    def get_by_symbol(self, symbol: str) -> Optional[TradeSignalPackage]:
        """Get active package for a symbol."""
        for pkg in self._active_packages.values():
            if pkg.symbol == symbol and pkg.status == SignalStatus.READY_FOR_EXECUTION:
                return pkg
        return None

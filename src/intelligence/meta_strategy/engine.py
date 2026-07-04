"""
Meta Strategy Engine — Ranks and selects strategies based on market state + performance.

Instead of statically running all strategies:
  1. Scores each registered strategy against the current MarketFingerprint.
  2. Ranks by regime alignment + recent risk-adjusted performance.
  3. Returns which strategies should be active and their capital weight.

This is the "strategy of strategies" layer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.intelligence.market_state.engine import MarketFingerprint


@dataclass
class StrategyProfile:
    """Metadata about a strategy's characteristics."""
    name: str
    preferred_regimes: list[str] = field(default_factory=list)
    avoid_regimes: list[str] = field(default_factory=list)
    preferred_volatility: list[str] = field(default_factory=lambda: ["Normal", "Expansion"])
    preferred_trend: list[str] = field(default_factory=list)
    min_liquidity: str = "Low"


@dataclass
class StrategyPerformance:
    """Rolling performance tracker for a strategy."""
    name: str
    recent_pnl: deque = field(default_factory=lambda: deque(maxlen=50))
    win_count: int = 0
    loss_count: int = 0
    total_pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.5

    @property
    def avg_pnl(self) -> float:
        return sum(self.recent_pnl) / len(self.recent_pnl) if self.recent_pnl else 0.0

    @property
    def sharpe_proxy(self) -> float:
        """Simple Sharpe-like ratio from recent PnL."""
        if len(self.recent_pnl) < 5:
            return 0.0
        pnls = list(self.recent_pnl)
        mean = sum(pnls) / len(pnls)
        variance = sum((p - mean) ** 2 for p in pnls) / len(pnls)
        std = variance ** 0.5
        return mean / std if std > 0 else 0.0


@dataclass
class StrategyRanking:
    """Result of meta-strategy evaluation."""
    name: str
    score: float
    regime_fit: float
    performance_score: float
    active: bool
    weight: float
    reason: str


# Default profiles for common strategies
DEFAULT_PROFILES: dict[str, StrategyProfile] = {
    "momentum": StrategyProfile(
        name="momentum",
        preferred_regimes=["TRENDING_LOW_VOL", "TRENDING_HIGH_VOL"],
        avoid_regimes=["RANGING_LOW_VOL", "RANGING_HIGH_VOL_PANIC"],
        preferred_volatility=["Normal", "Expansion"],
        preferred_trend=["Strong Uptrend", "Weak Uptrend", "Strong Downtrend", "Weak Downtrend"],
    ),
    "mean_reversion": StrategyProfile(
        name="mean_reversion",
        preferred_regimes=["RANGING_LOW_VOL", "RANGING_HIGH_VOL"],
        avoid_regimes=["TRENDING_LOW_VOL", "TRENDING_HIGH_VOL"],
        preferred_volatility=["Normal", "Compression"],
        preferred_trend=["Sideways"],
    ),
    "breakout": StrategyProfile(
        name="breakout",
        preferred_regimes=["RANGING_HIGH_VOL", "TRENDING_HIGH_VOL"],
        avoid_regimes=["RANGING_LOW_VOL"],
        preferred_volatility=["Expansion", "Extreme"],
        preferred_trend=["Sideways", "Weak Uptrend", "Weak Downtrend"],
    ),
    "ml": StrategyProfile(
        name="ml",
        preferred_regimes=[],  # ML adapts; no strong preference
        avoid_regimes=[],
        preferred_volatility=["Normal", "Compression", "Expansion"],
        preferred_trend=[],
    ),
}


class MetaStrategyEngine:
    """Evaluate, rank, and activate strategies based on market state."""

    def __init__(
        self,
        min_activation_score: float = 40.0,
        max_active_strategies: int = 4,
        performance_weight: float = 0.4,
        regime_weight: float = 0.6,
    ):
        self.min_activation_score = min_activation_score
        self.max_active_strategies = max_active_strategies
        self.performance_weight = performance_weight
        self.regime_weight = regime_weight
        self._profiles: dict[str, StrategyProfile] = dict(DEFAULT_PROFILES)
        self._performance: dict[str, StrategyPerformance] = {}

    def register_strategy(self, name: str, profile: Optional[StrategyProfile] = None) -> None:
        if profile:
            self._profiles[name] = profile
        elif name not in self._profiles:
            self._profiles[name] = StrategyProfile(name=name)
        if name not in self._performance:
            self._performance[name] = StrategyPerformance(name=name)

    def record_trade_result(self, strategy_name: str, pnl: float) -> None:
        if strategy_name not in self._performance:
            self._performance[strategy_name] = StrategyPerformance(name=strategy_name)
        perf = self._performance[strategy_name]
        perf.recent_pnl.append(pnl)
        perf.total_pnl += pnl
        if pnl > 0:
            perf.win_count += 1
        elif pnl < 0:
            perf.loss_count += 1

    def rank(self, fingerprint: MarketFingerprint) -> list[StrategyRanking]:
        """Rank all registered strategies against the current market fingerprint."""
        rankings: list[StrategyRanking] = []

        for name, profile in self._profiles.items():
            regime_score = self._score_regime_fit(profile, fingerprint)
            perf_score = self._score_performance(name)
            combined = (self.regime_weight * regime_score) + (self.performance_weight * perf_score)
            combined = max(0.0, min(100.0, combined))
            rankings.append(StrategyRanking(
                name=name,
                score=combined,
                regime_fit=regime_score,
                performance_score=perf_score,
                active=False,
                weight=0.0,
                reason="",
            ))

        # Sort descending by score
        rankings.sort(key=lambda r: r.score, reverse=True)

        # Activate top N above threshold
        total_weight = 0.0
        activated = 0
        for r in rankings:
            if activated >= self.max_active_strategies:
                r.active = False
                r.reason = "Capacity limit reached"
                continue

            if r.score >= self.min_activation_score:
                r.active = True
                r.weight = r.score  # Proportional weight
                total_weight += r.score
                activated += 1
                r.reason = f"Score {r.score:.1f} >= threshold {self.min_activation_score:.1f}"
            else:
                r.active = False
                r.reason = f"Score {r.score:.1f} below threshold {self.min_activation_score:.1f}"

        # Normalize weights to sum to 1.0
        if total_weight > 0:
            for r in rankings:
                if r.active:
                    r.weight = r.weight / total_weight

        return rankings

    def get_active_strategies(self, fingerprint: MarketFingerprint) -> list[StrategyRanking]:
        """Convenience: return only active strategies."""
        return [r for r in self.rank(fingerprint) if r.active]

    def _score_regime_fit(self, profile: StrategyProfile, fp: MarketFingerprint) -> float:
        score = 50.0  # Base neutral score

        # Regime match/avoid
        if profile.preferred_regimes:
            if any(pref in fp.overall_regime for pref in profile.preferred_regimes):
                score += 25.0
            elif any(avoid in fp.overall_regime for avoid in profile.avoid_regimes):
                score -= 30.0

        # Trend alignment
        if profile.preferred_trend:
            if fp.trend.label in profile.preferred_trend:
                score += 15.0
            else:
                score -= 10.0

        # Volatility alignment
        if fp.volatility.label in profile.preferred_volatility:
            score += 10.0
        elif fp.volatility.label == "Extreme" and "Extreme" not in profile.preferred_volatility:
            score -= 15.0

        # Stress penalty for all strategies
        if fp.stress.label == "Panic":
            score -= 20.0
        elif fp.stress.label == "Elevated":
            score -= 8.0

        # Liquidity penalty
        liq_order = {"High": 3, "Medium": 2, "Low": 1, "Unknown": 0}
        min_liq = liq_order.get(profile.min_liquidity, 0)
        cur_liq = liq_order.get(fp.liquidity.label, 2)
        if cur_liq < min_liq:
            score -= 15.0

        return max(0.0, min(100.0, score))

    def _score_performance(self, name: str) -> float:
        perf = self._performance.get(name)
        if not perf or len(perf.recent_pnl) < 3:
            return 50.0  # Neutral until enough data

        score = 50.0
        # Win rate contribution
        wr_bonus = (perf.win_rate - 0.5) * 40.0  # ±20 from neutral
        score += wr_bonus

        # Sharpe-like contribution
        sharpe = perf.sharpe_proxy
        score += max(-20.0, min(20.0, sharpe * 15.0))

        # Recent streak
        recent = list(perf.recent_pnl)[-5:]
        if all(p > 0 for p in recent):
            score += 10.0
        elif all(p < 0 for p in recent):
            score -= 15.0

        return max(0.0, min(100.0, score))

    def get_performance_summary(self) -> dict[str, dict]:
        return {
            name: {
                "win_rate": perf.win_rate,
                "avg_pnl": perf.avg_pnl,
                "sharpe_proxy": perf.sharpe_proxy,
                "total_pnl": perf.total_pnl,
                "trades": perf.win_count + perf.loss_count,
            }
            for name, perf in self._performance.items()
        }

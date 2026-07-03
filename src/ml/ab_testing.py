"""
A/B Testing Framework — Compare ML model versions in production.

Provides:
- Concurrent model testing with configurable capital allocation
- Statistical significance testing (Welch's t-test)
- Auto-promotion of winning model
- Shadow mode (challenger predictions logged but not traded)
- Full audit trail of all test decisions
- Integration with ModelPredictor for transparent routing
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from src.ml.model_registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ABTestStatus(str, Enum):
    """Status of an A/B test."""
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    INSUFFICIENT_DATA = "insufficient_data"


class ABTestMode(str, Enum):
    """How the challenger model is tested."""
    SHADOW = "shadow"       # Challenger predictions logged, not traded
    SPLIT = "split"         # Capital split between champion and challenger
    INTERLEAVED = "interleaved"  # Alternate between models


@dataclass
class ModelPerformance:
    """Performance tracking for a single model in an A/B test."""
    version: str
    trades: list[dict] = field(default_factory=list)
    predictions: list[dict] = field(default_factory=list)
    total_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def avg_pnl(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades

    @property
    def sharpe_ratio(self) -> float:
        if not self.trades:
            return 0.0
        returns = [t.get("pnl_pct", 0.0) for t in self.trades]
        if len(returns) < 2:
            return 0.0
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        if std_ret == 0:
            return 0.0
        return float(mean_ret / std_ret * np.sqrt(252))

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 2),
            "avg_pnl": round(self.avg_pnl, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "prediction_count": len(self.predictions),
        }


@dataclass
class ABTest:
    """A single A/B test comparing champion vs challenger."""
    test_id: str
    champion_version: str
    challenger_version: str
    mode: ABTestMode
    allocation_pct: float  # % of decisions routed to challenger
    min_trades: int
    max_duration_hours: float
    auto_promote: bool
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    status: ABTestStatus = ABTestStatus.ACTIVE
    winner: Optional[str] = None
    champion_perf: ModelPerformance = field(default_factory=lambda: ModelPerformance(version=""))
    challenger_perf: ModelPerformance = field(default_factory=lambda: ModelPerformance(version=""))
    significance_p_value: Optional[float] = None

    def __post_init__(self):
        self.champion_perf = ModelPerformance(version=self.champion_version)
        self.challenger_perf = ModelPerformance(version=self.challenger_version)

    @property
    def duration_hours(self) -> float:
        end = self.completed_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds() / 3600

    @property
    def is_expired(self) -> bool:
        return self.duration_hours >= self.max_duration_hours

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "status": self.status.value,
            "mode": self.mode.value,
            "champion": self.champion_perf.to_dict(),
            "challenger": self.challenger_perf.to_dict(),
            "allocation_pct": self.allocation_pct,
            "winner": self.winner,
            "duration_hours": round(self.duration_hours, 1),
            "max_duration_hours": self.max_duration_hours,
            "significance_p_value": self.significance_p_value,
            "auto_promote": self.auto_promote,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class ABTestManager:
    """
    Manages A/B testing of ML model versions in production.

    Usage:
        ab_manager = ABTestManager(registry, predictor, event_bus)

        # Start a test
        test_id = ab_manager.start_test(
            challenger_version="v004",
            mode=ABTestMode.SHADOW,
            allocation_pct=0.2,
        )

        # Record outcomes (called by execution engine)
        ab_manager.record_trade(model_version="v003", trade_result={...})

        # Check results
        result = ab_manager.get_test_status(test_id)

        # Or auto-promote on statistical significance
    """

    def __init__(
        self,
        registry: ModelRegistry,
        predictor=None,
        event_bus=None,
        significance_threshold: float = 0.05,
        min_trades_default: int = 30,
        max_duration_hours_default: float = 168.0,  # 1 week
    ):
        self._registry = registry
        self._predictor = predictor  # ModelPredictor instance
        self._event_bus = event_bus
        self._significance_threshold = significance_threshold
        self._min_trades_default = min_trades_default
        self._max_duration_default = max_duration_hours_default

        self._active_test: Optional[ABTest] = None
        self._completed_tests: list[ABTest] = []
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────────
    # Test Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def start_test(
        self,
        challenger_version: str,
        mode: ABTestMode = ABTestMode.SHADOW,
        allocation_pct: float = 0.2,
        min_trades: Optional[int] = None,
        max_duration_hours: Optional[float] = None,
        auto_promote: bool = True,
    ) -> str:
        """
        Start a new A/B test.

        Args:
            challenger_version: Version to test against current champion.
            mode: Testing mode (shadow, split, interleaved).
            allocation_pct: Fraction of decisions for challenger (0-1).
            min_trades: Minimum trades before declaring winner.
            max_duration_hours: Maximum test duration.
            auto_promote: Auto-deploy winner to production.

        Returns:
            test_id for tracking.

        Raises:
            RuntimeError: If a test is already active.
            ValueError: If challenger version doesn't exist.
        """
        with self._lock:
            if self._active_test and self._active_test.status == ABTestStatus.ACTIVE:
                raise RuntimeError(
                    f"Test already active: {self._active_test.test_id} "
                    f"({self._active_test.challenger_version} vs {self._active_test.champion_version})"
                )

            # Validate challenger exists
            try:
                self._registry.get_version(challenger_version)
            except (KeyError, FileNotFoundError) as e:
                raise ValueError(f"Challenger version {challenger_version} not found: {e}")

            # Get current champion
            champion = self._registry.get_active_version()
            if champion is None:
                raise RuntimeError("No active model version to test against")
            if champion == challenger_version:
                raise ValueError("Challenger and champion are the same version")

            test_id = uuid.uuid4().hex[:10]
            test = ABTest(
                test_id=test_id,
                champion_version=champion,
                challenger_version=challenger_version,
                mode=mode,
                allocation_pct=allocation_pct,
                min_trades=min_trades or self._min_trades_default,
                max_duration_hours=max_duration_hours or self._max_duration_default,
                auto_promote=auto_promote,
            )
            self._active_test = test

            # Set up shadow model in predictor if available
            if self._predictor and mode == ABTestMode.SHADOW:
                try:
                    self._predictor.set_shadow_model(challenger_version)
                except Exception as e:
                    logger.warning("ab_test.shadow_setup_failed", error=str(e))

            # Publish event
            if self._event_bus:
                from src.core.events import ABTestStarted
                self._event_bus.publish(ABTestStarted(
                    test_id=test_id,
                    champion_version=champion,
                    challenger_version=challenger_version,
                    allocation_pct=allocation_pct,
                    source="ab_test_manager",
                ))

            logger.info(
                "ab_test.started",
                test_id=test_id,
                champion=champion,
                challenger=challenger_version,
                mode=mode.value,
                allocation=allocation_pct,
            )
            return test_id

    def cancel_test(self, reason: str = "manual") -> Optional[dict]:
        """Cancel the active test."""
        with self._lock:
            if not self._active_test:
                return None

            self._active_test.status = ABTestStatus.CANCELLED
            self._active_test.completed_at = datetime.now(timezone.utc)
            result = self._active_test.to_dict()
            self._completed_tests.append(self._active_test)
            self._active_test = None

            # Clean up shadow
            if self._predictor:
                self._predictor.clear_shadow_model()

            logger.info("ab_test.cancelled", reason=reason)
            return result

    # ──────────────────────────────────────────────────────────────────────
    # Trade Recording
    # ──────────────────────────────────────────────────────────────────────

    def record_trade(
        self,
        model_version: str,
        trade_result: dict,
    ) -> None:
        """
        Record a trade outcome for A/B test evaluation.

        Args:
            model_version: Which model version produced this trade.
            trade_result: Dict with at least {pnl, pnl_pct, symbol, side}.
        """
        with self._lock:
            if not self._active_test or self._active_test.status != ABTestStatus.ACTIVE:
                return

            test = self._active_test
            pnl = trade_result.get("pnl", 0.0)

            if model_version == test.champion_version:
                perf = test.champion_perf
            elif model_version == test.challenger_version:
                perf = test.challenger_perf
            else:
                return  # Not part of this test

            perf.trades.append(trade_result)
            perf.total_trades += 1
            perf.total_pnl += pnl
            if pnl > 0:
                perf.winning_trades += 1

            # Check if test should conclude
            self._maybe_conclude_test(test)

    def record_prediction(
        self,
        model_version: str,
        prediction: Any,
        features_hash: Optional[str] = None,
    ) -> None:
        """Record a prediction for shadow-mode evaluation."""
        with self._lock:
            if not self._active_test or self._active_test.status != ABTestStatus.ACTIVE:
                return

            test = self._active_test
            record = {
                "version": model_version,
                "prediction": prediction,
                "features_hash": features_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if model_version == test.champion_version:
                test.champion_perf.predictions.append(record)
            elif model_version == test.challenger_version:
                test.challenger_perf.predictions.append(record)

    # ──────────────────────────────────────────────────────────────────────
    # Routing
    # ──────────────────────────────────────────────────────────────────────

    def should_use_challenger(self) -> bool:
        """
        Determine if this prediction should use the challenger model.

        Used in SPLIT mode to route a fraction of decisions to challenger.
        """
        with self._lock:
            if not self._active_test or self._active_test.status != ABTestStatus.ACTIVE:
                return False
            if self._active_test.mode != ABTestMode.SPLIT:
                return False
            return np.random.random() < self._active_test.allocation_pct

    def get_active_version_for_prediction(self) -> str:
        """
        Get which model version to use for this prediction.

        In SPLIT mode, randomly routes based on allocation.
        In SHADOW mode, always returns champion (shadow runs separately).
        """
        with self._lock:
            if not self._active_test or self._active_test.status != ABTestStatus.ACTIVE:
                return self._registry.get_active_version() or ""

            test = self._active_test
            if test.mode == ABTestMode.SPLIT:
                if np.random.random() < test.allocation_pct:
                    return test.challenger_version
                return test.champion_version
            elif test.mode == ABTestMode.INTERLEAVED:
                # Alternate based on trade count
                total = test.champion_perf.total_trades + test.challenger_perf.total_trades
                if total % 2 == 0:
                    return test.champion_version
                return test.challenger_version
            else:
                # Shadow mode — always use champion
                return test.champion_version

    # ──────────────────────────────────────────────────────────────────────
    # Status & Results
    # ──────────────────────────────────────────────────────────────────────

    def get_test_status(self, test_id: Optional[str] = None) -> Optional[dict]:
        """Get status of active or specific test."""
        with self._lock:
            if test_id:
                if self._active_test and self._active_test.test_id == test_id:
                    return self._active_test.to_dict()
                for test in self._completed_tests:
                    if test.test_id == test_id:
                        return test.to_dict()
                return None
            elif self._active_test:
                return self._active_test.to_dict()
            return None

    def get_active_test(self) -> Optional[ABTest]:
        """Get the active test object."""
        with self._lock:
            return self._active_test

    def list_tests(self, limit: int = 10) -> list[dict]:
        """List recent tests."""
        with self._lock:
            tests = []
            if self._active_test:
                tests.append(self._active_test)
            tests.extend(reversed(self._completed_tests[-limit:]))
            return [t.to_dict() for t in tests[:limit]]

    # ──────────────────────────────────────────────────────────────────────
    # Statistical Testing
    # ──────────────────────────────────────────────────────────────────────

    def _maybe_conclude_test(self, test: ABTest):
        """Check if test should conclude and declare winner."""
        # Check expiration
        if test.is_expired:
            self._conclude_test(test, reason="expired")
            return

        # Check minimum trades met for both
        champion_trades = test.champion_perf.total_trades
        challenger_trades = test.challenger_perf.total_trades

        if champion_trades < test.min_trades or challenger_trades < test.min_trades:
            return  # Not enough data yet

        # Run statistical test
        winner, p_value = self._statistical_test(test)
        test.significance_p_value = p_value

        if p_value is not None and p_value < self._significance_threshold:
            test.winner = winner
            self._conclude_test(test, reason="statistical_significance")

    def _statistical_test(self, test: ABTest) -> tuple[Optional[str], Optional[float]]:
        """
        Welch's t-test comparing champion and challenger returns.

        Returns (winner_version, p_value) or (None, None) if inconclusive.
        """
        from scipy import stats

        champ_returns = [t.get("pnl_pct", 0.0) for t in test.champion_perf.trades]
        chall_returns = [t.get("pnl_pct", 0.0) for t in test.challenger_perf.trades]

        if len(champ_returns) < 5 or len(chall_returns) < 5:
            return None, None

        try:
            t_stat, p_value = stats.ttest_ind(
                chall_returns, champ_returns, equal_var=False
            )

            # Determine winner based on mean returns
            champ_mean = np.mean(champ_returns)
            chall_mean = np.mean(chall_returns)

            if chall_mean > champ_mean:
                winner = test.challenger_version
            else:
                winner = test.champion_version

            return winner, float(p_value)
        except Exception as e:
            logger.debug("ab_test.stats_error", error=str(e))
            return None, None

    def _conclude_test(self, test: ABTest, reason: str):
        """Conclude a test and optionally promote winner."""
        test.status = ABTestStatus.COMPLETED
        test.completed_at = datetime.now(timezone.utc)

        # If no winner determined yet, pick by Sharpe
        if test.winner is None:
            champ_sharpe = test.champion_perf.sharpe_ratio
            chall_sharpe = test.challenger_perf.sharpe_ratio
            if chall_sharpe > champ_sharpe:
                test.winner = test.challenger_version
            else:
                test.winner = test.champion_version

        logger.info(
            "ab_test.concluded",
            test_id=test.test_id,
            winner=test.winner,
            reason=reason,
            champion_sharpe=test.champion_perf.sharpe_ratio,
            challenger_sharpe=test.challenger_perf.sharpe_ratio,
            p_value=test.significance_p_value,
        )

        # Auto-promote if enabled and challenger won
        if test.auto_promote and test.winner == test.challenger_version:
            try:
                self._registry.rollback(test.challenger_version)
                logger.info(
                    "ab_test.auto_promoted",
                    version=test.challenger_version,
                )
            except Exception as e:
                logger.error("ab_test.promotion_failed", error=str(e))

        # Publish event
        if self._event_bus:
            from src.core.events import ABTestCompleted
            self._event_bus.publish(ABTestCompleted(
                test_id=test.test_id,
                winner=test.winner or "",
                champion_sharpe=test.champion_perf.sharpe_ratio,
                challenger_sharpe=test.challenger_perf.sharpe_ratio,
                trades_evaluated=test.champion_perf.total_trades + test.challenger_perf.total_trades,
                source="ab_test_manager",
            ))

        # Clean up
        if self._predictor:
            self._predictor.clear_shadow_model()

        self._completed_tests.append(test)
        self._active_test = None

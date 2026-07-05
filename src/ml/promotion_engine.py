"""Champion-Challenger auto-promotion engine with governance gates."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)

REPORTS_PATH = Path("data_cache/promotions/reports.jsonl")


@dataclass
class PromotionGate:
    """Defines what a model must satisfy to be promoted."""

    name: str
    passed: bool
    metric_name: str
    threshold: float
    actual_value: float
    details: str = ""


@dataclass
class PromotionReport:
    """Complete audit trail for every promotion decision."""

    timestamp: datetime
    champion_version: str
    challenger_version: str
    decision: str  # "promoted" | "rejected" | "inconclusive"

    # Performance comparison
    champion_metrics: dict
    challenger_metrics: dict

    # Gate results
    gates: list[PromotionGate]
    gates_passed: int
    gates_total: int

    # Context
    evaluation_trades: int
    evaluation_period_days: int
    market_regimes_tested: list[str]

    # Explanation
    improvement_summary: str
    risk_areas: list[str]
    rollback_criteria: str


class ModelPromotionEngine:
    """Core promotion logic for champion-challenger model evaluation."""

    def __init__(
        self,
        min_evaluation_trades: int = 30,
        min_evaluation_days: int = 5,
        significance_level: float = 0.05,
        min_improvement_pct: float = 5.0,
        canary_allocation_pct: float = 10.0,
    ):
        self.min_evaluation_trades = min_evaluation_trades
        self.min_evaluation_days = min_evaluation_days
        self.significance_level = significance_level
        self.min_improvement_pct = min_improvement_pct
        self.canary_allocation_pct = canary_allocation_pct
        self._reports: list[PromotionReport] = []

    def evaluate_promotion(
        self,
        champion_metrics: dict,
        challenger_metrics: dict,
        champion_trades: list[dict],
        challenger_trades: list[dict],
    ) -> PromotionReport:
        """
        Evaluate whether challenger should replace champion.

        Runs through governance gates:
        1. Minimum sample size (enough trades)
        2. Statistical significance (t-test on returns)
        3. Risk-adjusted return (Sharpe improvement >= min_improvement_pct)
        4. Maximum drawdown (challenger <= champion)
        5. Win rate (challenger >= champion - 5%)
        6. Calibration (challenger ECE <= champion ECE)
        7. Regime stability (performs in >= 2 regimes)
        8. No latency degradation
        """
        champion_returns = [t.get("pnl_pct", t.get("return", 0.0)) for t in champion_trades]
        challenger_returns = [t.get("pnl_pct", t.get("return", 0.0)) for t in challenger_trades]

        gates: list[PromotionGate] = [
            # Critical gates (1-6)
            self._check_gate_sample_size(challenger_trades),
            self._check_gate_significance(champion_returns, challenger_returns),
            self._check_gate_sharpe(champion_metrics, challenger_metrics),
            self._check_gate_drawdown(champion_metrics, challenger_metrics),
            self._check_gate_win_rate(champion_metrics, challenger_metrics),
            self._check_gate_cvar(champion_trades, challenger_trades),
            # Optional gates (7-12)
            self._check_gate_calibration(champion_metrics, challenger_metrics),
            self._check_gate_brier_score(champion_metrics, challenger_metrics),
            self._check_gate_return_stability(champion_trades, challenger_trades),
            self._check_gate_regime_stability(challenger_trades),
            self._check_gate_feature_drift(challenger_metrics),
            self._check_gate_latency(champion_metrics, challenger_metrics),
        ]

        gates_passed = sum(1 for g in gates if g.passed)
        gates_total = len(gates)

        # Determine regimes tested
        regimes = list(
            {t.get("regime", "unknown") for t in challenger_trades if "regime" in t}
        )

        # Determine evaluation period
        if challenger_trades:
            timestamps = [
                t.get("timestamp", 0) for t in challenger_trades if "timestamp" in t
            ]
            if len(timestamps) >= 2:
                period_days = max(
                    1,
                    int(
                        (max(timestamps) - min(timestamps)) / 86400
                    ),
                )
            else:
                period_days = 0
        else:
            period_days = 0

        # Build improvement summary
        sharpe_champ = champion_metrics.get("sharpe", 0.0)
        sharpe_chall = challenger_metrics.get("sharpe", 0.0)
        if sharpe_champ != 0:
            sharpe_improvement = ((sharpe_chall - sharpe_champ) / abs(sharpe_champ)) * 100
        else:
            sharpe_improvement = 0.0 if sharpe_chall == 0 else 100.0

        improvement_summary = (
            f"Sharpe: {sharpe_champ:.3f} -> {sharpe_chall:.3f} "
            f"({sharpe_improvement:+.1f}%), "
            f"Win rate: {champion_metrics.get('win_rate', 0):.1%} -> "
            f"{challenger_metrics.get('win_rate', 0):.1%}"
        )

        # Identify risk areas
        risk_areas: list[str] = []
        if challenger_metrics.get("max_dd", 0) > champion_metrics.get("max_dd", 0):
            risk_areas.append("Higher maximum drawdown")
        if challenger_metrics.get("win_rate", 0) < champion_metrics.get("win_rate", 0):
            risk_areas.append("Lower win rate")
        if challenger_metrics.get("calibration_ece", 0) > champion_metrics.get(
            "calibration_ece", 0
        ):
            risk_areas.append("Worse calibration")

        report = PromotionReport(
            timestamp=datetime.now(timezone.utc),
            champion_version=champion_metrics.get("version", "unknown"),
            challenger_version=challenger_metrics.get("version", "unknown"),
            decision="inconclusive",
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            gates=gates,
            gates_passed=gates_passed,
            gates_total=gates_total,
            evaluation_trades=len(challenger_trades),
            evaluation_period_days=period_days,
            market_regimes_tested=regimes,
            improvement_summary=improvement_summary,
            risk_areas=risk_areas,
            rollback_criteria=(
                "Revert if Sharpe drops >10% or max drawdown exceeds "
                f"{champion_metrics.get('max_dd', 0) * 1.2:.2%} within 48h"
            ),
        )

        # Make decision
        if self.should_promote(report):
            report.decision = "promoted"
        elif not gates[0].passed:
            report.decision = "inconclusive"
        else:
            report.decision = "rejected"

        self._reports.append(report)
        self._persist_report(report)

        logger.info(
            "promotion_evaluated",
            decision=report.decision,
            gates_passed=gates_passed,
            gates_total=gates_total,
            challenger=report.challenger_version,
        )

        return report

    def _check_gate_sample_size(self, trades: list) -> PromotionGate:
        """Gate 1: Enough trades to evaluate."""
        actual = len(trades)
        passed = actual >= self.min_evaluation_trades
        return PromotionGate(
            name="sample_size",
            passed=passed,
            metric_name="trade_count",
            threshold=float(self.min_evaluation_trades),
            actual_value=float(actual),
            details=f"Need {self.min_evaluation_trades} trades, have {actual}",
        )

    def _check_gate_significance(
        self, champion_returns: list, challenger_returns: list
    ) -> PromotionGate:
        """Gate 2: Statistical significance via t-test."""
        if len(champion_returns) < 2 or len(challenger_returns) < 2:
            return PromotionGate(
                name="statistical_significance",
                passed=False,
                metric_name="p_value",
                threshold=self.significance_level,
                actual_value=1.0,
                details="Insufficient data for t-test",
            )

        t_stat, p_value = stats.ttest_ind(
            challenger_returns, champion_returns, alternative="greater"
        )
        # Handle NaN from t-test (e.g., zero variance in one sample)
        if np.isnan(p_value):
            p_value = 1.0
            t_stat = 0.0
        passed = p_value < self.significance_level
        return PromotionGate(
            name="statistical_significance",
            passed=passed,
            metric_name="p_value",
            threshold=self.significance_level,
            actual_value=float(p_value),
            details=f"t={t_stat:.3f}, p={p_value:.4f}",
        )

    def _check_gate_sharpe(
        self, champion_metrics: dict, challenger_metrics: dict
    ) -> PromotionGate:
        """Gate 3: Risk-adjusted return improvement."""
        champ_sharpe = champion_metrics.get("sharpe", 0.0)
        chall_sharpe = challenger_metrics.get("sharpe", 0.0)

        if champ_sharpe != 0:
            improvement = ((chall_sharpe - champ_sharpe) / abs(champ_sharpe)) * 100
        else:
            improvement = 0.0 if chall_sharpe == 0 else 100.0

        passed = improvement >= self.min_improvement_pct
        return PromotionGate(
            name="sharpe_improvement",
            passed=passed,
            metric_name="sharpe_improvement_pct",
            threshold=self.min_improvement_pct,
            actual_value=improvement,
            details=f"Sharpe {champ_sharpe:.3f} -> {chall_sharpe:.3f} ({improvement:+.1f}%)",
        )

    def _check_gate_drawdown(
        self, champion_metrics: dict, challenger_metrics: dict
    ) -> PromotionGate:
        """Gate 4: Max drawdown not worse."""
        champ_dd = champion_metrics.get("max_dd", 0.0)
        chall_dd = challenger_metrics.get("max_dd", 0.0)
        passed = chall_dd <= champ_dd
        return PromotionGate(
            name="max_drawdown",
            passed=passed,
            metric_name="max_drawdown",
            threshold=champ_dd,
            actual_value=chall_dd,
            details=f"Drawdown {champ_dd:.4f} -> {chall_dd:.4f}",
        )

    def _check_gate_win_rate(
        self, champion_metrics: dict, challenger_metrics: dict
    ) -> PromotionGate:
        """Gate 5: Win rate acceptable."""
        champ_wr = champion_metrics.get("win_rate", 0.0)
        chall_wr = challenger_metrics.get("win_rate", 0.0)
        threshold = champ_wr - 0.05  # Allow up to 5% worse
        passed = chall_wr >= threshold
        return PromotionGate(
            name="win_rate",
            passed=passed,
            metric_name="win_rate",
            threshold=threshold,
            actual_value=chall_wr,
            details=f"Win rate {champ_wr:.3f} -> {chall_wr:.3f} (min: {threshold:.3f})",
        )

    def _check_gate_calibration(
        self, champion_metrics: dict, challenger_metrics: dict
    ) -> PromotionGate:
        """Gate 6: Calibration not degraded."""
        champ_ece = champion_metrics.get("calibration_ece", 1.0)
        chall_ece = challenger_metrics.get("calibration_ece", 1.0)
        passed = chall_ece <= champ_ece
        return PromotionGate(
            name="calibration",
            passed=passed,
            metric_name="calibration_ece",
            threshold=champ_ece,
            actual_value=chall_ece,
            details=f"ECE {champ_ece:.4f} -> {chall_ece:.4f}",
        )

    def _check_gate_regime_stability(self, trades: list) -> PromotionGate:
        """Gate 7: Works across multiple regimes."""
        regimes = set()
        for t in trades:
            regime = t.get("regime") or t.get("market_regime")
            if regime:
                regimes.add(regime)
        actual = len(regimes)
        threshold = 2.0
        passed = actual >= threshold
        return PromotionGate(
            name="regime_stability",
            passed=passed,
            metric_name="regimes_tested",
            threshold=threshold,
            actual_value=float(actual),
            details=f"Tested in {actual} regimes: {sorted(regimes) if regimes else 'none'}",
        )

    def _check_gate_latency(
        self, champion_metrics: dict, challenger_metrics: dict
    ) -> PromotionGate:
        """Gate: No latency degradation."""
        champ_latency = champion_metrics.get("avg_latency_ms", 0.0)
        chall_latency = challenger_metrics.get("avg_latency_ms", 0.0)
        # Allow up to 20% latency increase
        threshold = champ_latency * 1.2 if champ_latency > 0 else float("inf")
        passed = chall_latency <= threshold
        return PromotionGate(
            name="latency",
            passed=passed,
            metric_name="avg_latency_ms",
            threshold=threshold,
            actual_value=chall_latency,
            details=f"Latency {champ_latency:.1f}ms -> {chall_latency:.1f}ms",
        )

    def _check_gate_cvar(
        self, champion_trades: list, challenger_trades: list
    ) -> PromotionGate:
        """Gate: Conditional Value at Risk (Expected Shortfall).
        Challenger's CVaR-5% must not be worse than champion's.
        CVaR = average of worst 5% of returns."""
        champ_returns = [t.get("pnl_pct", t.get("return", 0.0)) for t in champion_trades]
        chall_returns = [t.get("pnl_pct", t.get("return", 0.0)) for t in challenger_trades]

        champ_cvar = self._compute_cvar(champ_returns)
        chall_cvar = self._compute_cvar(chall_returns)

        # Challenger CVaR must not be worse (more negative) than champion's
        passed = chall_cvar >= champ_cvar
        return PromotionGate(
            name="cvar",
            passed=passed,
            metric_name="cvar_5pct",
            threshold=champ_cvar,
            actual_value=chall_cvar,
            details=f"CVaR-5% {champ_cvar:.4f} -> {chall_cvar:.4f}",
        )

    def _check_gate_return_stability(
        self, champion_trades: list, challenger_trades: list
    ) -> PromotionGate:
        """Gate: Return variance/stability.
        Challenger's return std must be <= 1.5x champion's."""
        champ_returns = [t.get("pnl_pct", t.get("return", 0.0)) for t in champion_trades]
        chall_returns = [t.get("pnl_pct", t.get("return", 0.0)) for t in challenger_trades]

        champ_std = float(np.std(champ_returns)) if len(champ_returns) > 1 else 0.0
        chall_std = float(np.std(chall_returns)) if len(chall_returns) > 1 else 0.0

        threshold = champ_std * 1.5 if champ_std > 0 else float("inf")
        passed = chall_std <= threshold
        return PromotionGate(
            name="return_stability",
            passed=passed,
            metric_name="return_std",
            threshold=threshold,
            actual_value=chall_std,
            details=f"Std {champ_std:.4f} -> {chall_std:.4f} (max: {threshold:.4f})",
        )

    def _check_gate_brier_score(
        self, champion_metrics: dict, challenger_metrics: dict
    ) -> PromotionGate:
        """Gate: Brier score (prediction probability accuracy).
        Challenger's Brier score must be <= champion's.
        Brier = mean((predicted_prob - actual_outcome)^2)"""
        champ_brier = champion_metrics.get("brier_score", 1.0)
        chall_brier = challenger_metrics.get("brier_score", 1.0)

        passed = chall_brier <= champ_brier
        return PromotionGate(
            name="brier_score",
            passed=passed,
            metric_name="brier_score",
            threshold=champ_brier,
            actual_value=chall_brier,
            details=f"Brier {champ_brier:.4f} -> {chall_brier:.4f}",
        )

    def _check_gate_feature_drift(self, challenger_metrics: dict) -> PromotionGate:
        """Gate: Feature drift (PSI).
        If PSI > 0.25 for >20% of features, reject promotion
        (model trained on drifted data is unreliable)."""
        psi_values = challenger_metrics.get("feature_psi", {})

        if not psi_values:
            # No PSI data available, pass by default
            return PromotionGate(
                name="feature_drift",
                passed=True,
                metric_name="psi_drift_pct",
                threshold=20.0,
                actual_value=0.0,
                details="No PSI data available, gate passes by default",
            )

        total_features = len(psi_values)
        drifted_features = sum(1 for v in psi_values.values() if v > 0.25)
        drift_pct = (drifted_features / total_features) * 100 if total_features > 0 else 0.0

        passed = drift_pct <= 20.0
        return PromotionGate(
            name="feature_drift",
            passed=passed,
            metric_name="psi_drift_pct",
            threshold=20.0,
            actual_value=drift_pct,
            details=f"{drifted_features}/{total_features} features drifted (PSI>0.25): {drift_pct:.1f}%",
        )

    def _compute_cvar(self, returns: list[float], percentile: float = 5.0) -> float:
        """Conditional Value at Risk (Expected Shortfall).
        Average of the worst `percentile`% of returns."""
        if not returns:
            return 0.0
        sorted_returns = sorted(returns)
        cutoff_idx = max(1, int(len(sorted_returns) * percentile / 100))
        return float(np.mean(sorted_returns[:cutoff_idx]))

    def should_promote(self, report: PromotionReport) -> bool:
        """
        Final decision: promote if all critical gates pass + majority of optional gates.

        Critical gates (indices 0-5): sample_size, significance, sharpe, drawdown, win_rate, cvar
        Optional gates (indices 6-11): calibration, brier_score, return_stability,
                                       regime_stability, feature_drift, latency
        Promote if all critical pass AND at least 3 of 6 optional gates pass.
        """
        if len(report.gates) < 6:
            return False

        critical_gates = report.gates[:6]
        optional_gates = report.gates[6:]

        all_critical_pass = all(g.passed for g in critical_gates)
        optional_passed = sum(1 for g in optional_gates if g.passed)

        return all_critical_pass and optional_passed >= 3

    def get_latest_report(self) -> Optional[PromotionReport]:
        """Get most recent promotion report."""
        return self._reports[-1] if self._reports else None

    def get_history(self) -> list[PromotionReport]:
        """Get all promotion reports."""
        return list(self._reports)

    def _persist_report(self, report: PromotionReport) -> None:
        """Persist report to JSONL file."""
        try:
            REPORTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": report.timestamp.isoformat(),
                "champion_version": report.champion_version,
                "challenger_version": report.challenger_version,
                "decision": report.decision,
                "gates_passed": report.gates_passed,
                "gates_total": report.gates_total,
                "evaluation_trades": report.evaluation_trades,
                "evaluation_period_days": report.evaluation_period_days,
                "market_regimes_tested": report.market_regimes_tested,
                "improvement_summary": report.improvement_summary,
                "risk_areas": report.risk_areas,
                "rollback_criteria": report.rollback_criteria,
                "champion_metrics": report.champion_metrics,
                "challenger_metrics": report.challenger_metrics,
                "gates": [asdict(g) for g in report.gates],
            }
            with open(REPORTS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.warning("failed_to_persist_report", error=str(e))


class CanaryDeployment:
    """Manages gradual rollout of challenger models."""

    def __init__(
        self,
        canary_pct: float = 10.0,
        evaluation_trades: int = 20,
        max_days: int = 7,
    ):
        self.canary_pct = canary_pct
        self.evaluation_trades = evaluation_trades
        self.max_days = max_days
        self._active: Optional[dict] = None

    def start_canary(self, challenger_version: str) -> None:
        """Start canary deployment for challenger."""
        self._active = {
            "challenger_version": challenger_version,
            "start_time": datetime.now(timezone.utc),
            "trades": [],
        }
        logger.info(
            "canary_started",
            challenger=challenger_version,
            allocation_pct=self.canary_pct,
        )

    def is_active(self) -> bool:
        """True if a canary deployment is in progress."""
        return self._active is not None

    def get_allocation_pct(self) -> float:
        """Current canary allocation percentage."""
        return self.canary_pct if self._active else 0.0

    def record_canary_trade(self, trade_result: dict) -> None:
        """Record a trade executed by the canary model."""
        if self._active is None:
            logger.warning("canary_trade_recorded_without_active_deployment")
            return
        self._active["trades"].append(trade_result)

    def evaluate_canary(self) -> dict:
        """
        Evaluate canary performance.

        Returns dict with ready_for_evaluation, trades_completed,
        canary_return, should_promote, should_abort.
        """
        if self._active is None:
            return {
                "ready_for_evaluation": False,
                "trades_completed": 0,
                "canary_return": 0.0,
                "should_promote": False,
                "should_abort": False,
            }

        trades = self._active["trades"]
        trades_completed = len(trades)
        ready = trades_completed >= self.evaluation_trades

        # Calculate canary return
        returns = [t.get("return", 0.0) for t in trades]
        canary_return = float(np.sum(returns)) if returns else 0.0

        # Check elapsed time
        elapsed = (datetime.now(timezone.utc) - self._active["start_time"]).days
        timed_out = elapsed >= self.max_days

        # Abort if losing badly (> 5% drawdown in canary)
        cumulative = np.cumsum(returns) if returns else np.array([0.0])
        peak = np.maximum.accumulate(cumulative)
        drawdown = float(np.max(peak - cumulative)) if len(cumulative) > 0 else 0.0
        should_abort = drawdown > 0.05

        # Promote if enough trades and positive return
        should_promote = ready and canary_return > 0 and not should_abort

        result = {
            "ready_for_evaluation": ready or timed_out,
            "trades_completed": trades_completed,
            "canary_return": canary_return,
            "should_promote": should_promote,
            "should_abort": should_abort,
        }

        logger.info("canary_evaluated", **result)
        return result


class ModelRollbackMonitor:
    """
    Monitors live model performance and auto-rollbacks if degradation detected.

    Rollback triggers:
    - Rolling Sharpe drops below threshold
    - Drawdown exceeds maximum
    - Win rate drops below minimum
    - Consecutive losses exceed limit
    - Calibration drift exceeds tolerance
    """

    def __init__(
        self,
        min_sharpe: float = -0.5,
        max_drawdown_pct: float = 15.0,
        min_win_rate: float = 0.35,
        max_consecutive_losses: int = 8,
        evaluation_window: int = 20,
        cooldown_hours: float = 24.0,
    ):
        self._min_sharpe = min_sharpe
        self._max_drawdown_pct = max_drawdown_pct
        self._min_win_rate = min_win_rate
        self._max_consecutive_losses = max_consecutive_losses
        self._evaluation_window = evaluation_window
        self._cooldown_hours = cooldown_hours

        self._trades: list[dict] = []
        self._max_trades: int = 500
        self._current_model: Optional[str] = None
        self._previous_model: Optional[str] = None
        self._last_rollback: Optional[datetime] = None
        self._rollback_history: list[dict] = []
        self._lock = threading.Lock()

    def set_models(self, current: str, previous: str) -> None:
        """Set current and previous model versions for rollback target."""
        with self._lock:
            self._current_model = current
            self._previous_model = previous

    def record_trade(self, trade_result: dict) -> None:
        """Record a live trade result. Check if rollback needed."""
        with self._lock:
            self._trades.append(trade_result)
            if len(self._trades) > self._max_trades:
                self._trades = self._trades[-self._max_trades:]

    def check_rollback(self) -> Optional[dict]:
        """
        Evaluate if rollback is needed.
        Returns None if OK, or dict with rollback details.
        """
        with self._lock:
            if self._in_cooldown():
                return None

            if not self._trades:
                return None

            # Consecutive losses checked regardless of evaluation window
            consecutive_losses = self._compute_consecutive_losses()
            if consecutive_losses > self._max_consecutive_losses:
                reason = (
                    f"Consecutive losses {consecutive_losses} exceed limit "
                    f"{self._max_consecutive_losses}"
                )
                result = {
                    "should_rollback": True,
                    "reason": reason,
                    "current_model": self._current_model,
                    "rollback_to": self._previous_model,
                    "metrics": {
                        "rolling_sharpe": self._compute_rolling_sharpe(),
                        "current_drawdown": self._compute_current_drawdown(),
                        "recent_win_rate": 0.0,
                        "consecutive_losses": consecutive_losses,
                    },
                }
                logger.warning(
                    "rollback_triggered",
                    reason=reason,
                    current_model=self._current_model,
                    rollback_to=self._previous_model,
                )
                return result

            if len(self._trades) < self._evaluation_window:
                return None

            recent_trades = self._trades[-self._evaluation_window:]
            rolling_sharpe = self._compute_rolling_sharpe()
            current_drawdown = self._compute_current_drawdown()

            # Compute recent win rate
            wins = sum(
                1 for t in recent_trades
                if t.get("pnl_pct", t.get("pnl", 0.0)) > 0
            )
            recent_win_rate = wins / len(recent_trades) if recent_trades else 0.0

            # Check triggers
            reason = None
            if rolling_sharpe < self._min_sharpe:
                reason = f"Rolling Sharpe {rolling_sharpe:.3f} below threshold {self._min_sharpe}"
            elif current_drawdown > self._max_drawdown_pct:
                reason = f"Drawdown {current_drawdown:.2f}% exceeds max {self._max_drawdown_pct}%"
            elif recent_win_rate < self._min_win_rate:
                reason = f"Win rate {recent_win_rate:.3f} below minimum {self._min_win_rate}"

            if reason is None:
                return None

            result = {
                "should_rollback": True,
                "reason": reason,
                "current_model": self._current_model,
                "rollback_to": self._previous_model,
                "metrics": {
                    "rolling_sharpe": rolling_sharpe,
                    "current_drawdown": current_drawdown,
                    "recent_win_rate": recent_win_rate,
                    "consecutive_losses": consecutive_losses,
                },
            }

            logger.warning(
                "rollback_triggered",
                reason=reason,
                current_model=self._current_model,
                rollback_to=self._previous_model,
            )

            return result

    def _compute_rolling_sharpe(self) -> float:
        """Compute Sharpe ratio over evaluation window."""
        recent = self._trades[-self._evaluation_window:]
        returns = [t.get("pnl_pct", t.get("pnl", 0.0)) for t in recent]
        if len(returns) < 2:
            return 0.0
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        if std_ret == 0:
            return 0.0
        return float(mean_ret / std_ret)

    def _compute_current_drawdown(self) -> float:
        """Current drawdown from peak equity (in %)."""
        returns = [t.get("pnl_pct", t.get("pnl", 0.0)) for t in self._trades]
        if not returns:
            return 0.0
        cumulative = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = peak - cumulative
        return float(drawdowns[-1]) if len(drawdowns) > 0 else 0.0

    def _compute_consecutive_losses(self) -> int:
        """Count of most recent consecutive losing trades."""
        count = 0
        for trade in reversed(self._trades):
            pnl = trade.get("pnl_pct", trade.get("pnl", 0.0))
            if pnl < 0:
                count += 1
            else:
                break
        return count

    def _in_cooldown(self) -> bool:
        """True if a rollback happened recently (prevent rapid oscillation)."""
        if self._last_rollback is None:
            return False
        elapsed = datetime.now(timezone.utc) - self._last_rollback
        return elapsed < timedelta(hours=self._cooldown_hours)

    def acknowledge_rollback(self) -> None:
        """Record that rollback was executed. Start cooldown."""
        with self._lock:
            self._last_rollback = datetime.now(timezone.utc)
            self._rollback_history.append({
                "timestamp": self._last_rollback.isoformat(),
                "from_model": self._current_model,
                "to_model": self._previous_model,
            })
            logger.info(
                "rollback_acknowledged",
                from_model=self._current_model,
                to_model=self._previous_model,
            )

    @property
    def rollback_count(self) -> int:
        """Total rollbacks performed."""
        return len(self._rollback_history)

    def get_status(self) -> dict:
        """Current monitoring status."""
        with self._lock:
            return {
                "current_model": self._current_model,
                "previous_model": self._previous_model,
                "total_trades": len(self._trades),
                "rollback_count": self.rollback_count,
                "in_cooldown": self._in_cooldown(),
                "last_rollback": (
                    self._last_rollback.isoformat() if self._last_rollback else None
                ),
            }


class ShadowTrader:
    """
    Runs challenger model predictions on live market data without execution.
    Records virtual trades for fair performance comparison.
    """

    def __init__(self, challenger_version: str, max_shadow_trades: int = 200):
        self._challenger = challenger_version
        self._max_shadow_trades = max_shadow_trades
        self._shadow_trades: list[dict] = []
        self._pending_signals: dict[str, dict] = {}
        self._lock = threading.Lock()

    def record_shadow_signal(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> str:
        """Record a shadow prediction. Returns shadow_trade_id."""
        trade_id = f"shadow_{uuid.uuid4().hex[:12]}"
        signal = {
            "trade_id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_time": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._pending_signals[symbol] = signal

        logger.debug(
            "shadow_signal_recorded",
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
        )
        return trade_id

    def update_prices(self, symbol: str, current_price: float) -> list[dict]:
        """
        Update pending shadow trades with current prices.
        Close trades that hit SL/TP or expire.
        Returns list of completed shadow trades.
        """
        completed: list[dict] = []

        with self._lock:
            if symbol not in self._pending_signals:
                return completed

            signal = self._pending_signals[symbol]
            direction = signal["direction"]
            entry_price = signal["entry_price"]
            stop_loss = signal["stop_loss"]
            take_profit = signal["take_profit"]

            hit_tp = False
            hit_sl = False

            if direction == "long":
                hit_tp = current_price >= take_profit
                hit_sl = current_price <= stop_loss
            elif direction == "short":
                hit_tp = current_price <= take_profit
                hit_sl = current_price >= stop_loss

            if hit_tp or hit_sl:
                # Calculate PnL
                if direction == "long":
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100

                trade_result = {
                    **signal,
                    "exit_price": current_price,
                    "exit_time": datetime.now(timezone.utc).isoformat(),
                    "pnl_pct": pnl_pct,
                    "exit_reason": "take_profit" if hit_tp else "stop_loss",
                    "won": hit_tp,
                }

                self._shadow_trades.append(trade_result)
                del self._pending_signals[symbol]
                completed.append(trade_result)

                # Cap total stored trades
                if len(self._shadow_trades) > self._max_shadow_trades:
                    self._shadow_trades = self._shadow_trades[-self._max_shadow_trades:]

                logger.debug(
                    "shadow_trade_closed",
                    trade_id=signal["trade_id"],
                    pnl_pct=pnl_pct,
                    reason=trade_result["exit_reason"],
                )

        return completed

    def get_shadow_performance(self) -> dict:
        """Return shadow performance metrics."""
        with self._lock:
            trades = self._shadow_trades

        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "vs_champion": 0.0,
            }

        returns = [t["pnl_pct"] for t in trades]
        wins = sum(1 for t in trades if t.get("won", False))
        win_rate = wins / len(trades)
        avg_return = float(np.mean(returns))

        # Sharpe
        std_ret = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        sharpe = avg_return / std_ret if std_ret > 0 else 0.0

        # Max drawdown
        cumulative = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = peak - cumulative
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        return {
            "total_trades": len(trades),
            "win_rate": win_rate,
            "avg_return": avg_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "vs_champion": 0.0,  # Set externally when comparing
        }

    def get_comparison_trades(self) -> list[dict]:
        """Return completed shadow trades for promotion evaluation."""
        with self._lock:
            return list(self._shadow_trades)

    @property
    def is_active(self) -> bool:
        """True if shadow trader is active (has pending signals or trades)."""
        with self._lock:
            return len(self._pending_signals) > 0 or len(self._shadow_trades) > 0

    @property
    def trade_count(self) -> int:
        """Total completed shadow trades."""
        with self._lock:
            return len(self._shadow_trades)

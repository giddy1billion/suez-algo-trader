"""Champion-Challenger auto-promotion engine with governance gates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
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
            self._check_gate_sample_size(challenger_trades),
            self._check_gate_significance(champion_returns, challenger_returns),
            self._check_gate_sharpe(champion_metrics, challenger_metrics),
            self._check_gate_drawdown(champion_metrics, challenger_metrics),
            self._check_gate_win_rate(champion_metrics, challenger_metrics),
            self._check_gate_calibration(champion_metrics, challenger_metrics),
            self._check_gate_regime_stability(challenger_trades),
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
        """Gate 8: No latency degradation."""
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

    def should_promote(self, report: PromotionReport) -> bool:
        """
        Final decision: promote if all critical gates pass + majority of optional gates.

        Critical gates (indices 0-4): sample_size, significance, sharpe, drawdown, win_rate
        Optional gates (indices 5-7): calibration, regime_stability, latency
        Promote if all critical pass AND at least 1 optional passes.
        """
        if len(report.gates) < 5:
            return False

        critical_gates = report.gates[:5]
        optional_gates = report.gates[5:]

        all_critical_pass = all(g.passed for g in critical_gates)
        optional_passed = sum(1 for g in optional_gates if g.passed)

        return all_critical_pass and optional_passed >= 1

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

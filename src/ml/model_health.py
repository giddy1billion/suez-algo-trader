"""
Model Health Score — Continuous monitoring and composite scoring for deployed models.

Every active model receives a health score (0-100) computed from:
- Prediction accuracy (direction correctness)
- Risk-adjusted returns (Sharpe, Sortino)
- Calibration error (ECE)
- Feature drift (PSI)
- Prediction drift (distribution shift)
- Live-vs-backtest deviation
- Execution quality (slippage vs expected)
- Latency (inference time)
- Data completeness

Health Grades:
    96-100  Excellent   — Model performing optimally
    82-95   Healthy     — Normal operations
    71-81   Warning     — Monitoring closely, possible degradation
    55-70   Degraded    — Consider replacement, reduce allocation
    0-54    Critical    — Retire immediately, halt new signals

Integrates with:
- DriftMonitor for feature/prediction drift
- CalibrationAnalyzer for ECE
- ExperienceDatabase for live trade outcomes
- PromotionEngine for governance decisions
- CircuitBreaker for emergency halt
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Enums & Data Classes
# ──────────────────────────────────────────────────────────────────────────


class HealthGrade(Enum):
    EXCELLENT = "excellent"
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> "HealthGrade":
        if score >= 96:
            return cls.EXCELLENT
        elif score >= 82:
            return cls.HEALTHY
        elif score >= 71:
            return cls.WARNING
        elif score >= 55:
            return cls.DEGRADED
        else:
            return cls.CRITICAL


@dataclass
class HealthDimension:
    """Individual health dimension score."""
    name: str
    score: float  # 0-100
    weight: float  # contribution weight
    raw_value: float  # the underlying metric
    threshold_warning: float
    threshold_critical: float
    details: str = ""


@dataclass
class ModelHealthReport:
    """Comprehensive health report for a single model version."""
    model_version: str
    timestamp: float
    composite_score: float
    grade: HealthGrade
    dimensions: List[HealthDimension]
    n_predictions: int
    n_resolved_trades: int
    evaluation_window_hours: float
    should_retire: bool
    should_warn: bool
    should_reduce_allocation: bool
    retirement_reasons: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'═'*60}",
            f"MODEL HEALTH REPORT: {self.model_version}",
            f"{'═'*60}",
            f"Score: {self.composite_score:.1f}/100  Grade: {self.grade.value.upper()}",
            f"Predictions: {self.n_predictions}  |  Resolved Trades: {self.n_resolved_trades}",
            f"Window: {self.evaluation_window_hours:.1f}h",
            f"{'─'*60}",
        ]
        for dim in sorted(self.dimensions, key=lambda d: d.score):
            status = "✅" if dim.score >= 71 else "⚠️" if dim.score >= 55 else "❌"
            lines.append(f"  {status} {dim.name:<25} {dim.score:5.1f}/100  (raw: {dim.raw_value:.4f})")
        if self.retirement_reasons:
            lines.append(f"{'─'*60}")
            lines.append("⛔ RETIREMENT TRIGGERS:")
            for r in self.retirement_reasons:
                lines.append(f"  • {r}")
        if self.recommendations:
            lines.append(f"{'─'*60}")
            lines.append("📋 Recommendations:")
            for r in self.recommendations:
                lines.append(f"  • {r}")
        lines.append(f"{'═'*60}")
        return "\n".join(lines)


@dataclass
class HealthHistory:
    """Rolling history of health scores for trend detection."""
    scores: deque = field(default_factory=lambda: deque(maxlen=100))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))

    def add(self, score: float, ts: float):
        self.scores.append(score)
        self.timestamps.append(ts)

    @property
    def trend(self) -> float:
        """Returns slope of recent health scores (-1 to +1 normalized)."""
        if len(self.scores) < 5:
            return 0.0
        recent = list(self.scores)[-10:]
        x = np.arange(len(recent), dtype=float)
        y = np.array(recent)
        if np.std(x) == 0:
            return 0.0
        slope = np.polyfit(x, y, 1)[0]
        # Normalize: -1 means losing ~10 pts per evaluation, +1 means gaining
        return float(np.clip(slope / 10, -1.0, 1.0))

    @property
    def is_declining(self) -> bool:
        """True if health has been declining over recent evaluations."""
        return self.trend < -0.2 and len(self.scores) >= 5


# ──────────────────────────────────────────────────────────────────────────
# Health Dimensions — Individual Metric Evaluators
# ──────────────────────────────────────────────────────────────────────────


def _score_accuracy(correct: int, total: int) -> HealthDimension:
    """Score directional accuracy."""
    if total < 5:
        return HealthDimension(
            name="Accuracy", score=75.0, weight=0.20,
            raw_value=0.0, threshold_warning=0.45, threshold_critical=0.35,
            details="Insufficient data (< 5 predictions)",
        )
    accuracy = correct / total
    # Score: 60%+ acc → 100, 50% → 75, 40% → 50, 30% → 0
    score = np.clip((accuracy - 0.30) / 0.30 * 100, 0, 100)
    return HealthDimension(
        name="Accuracy", score=float(score), weight=0.20,
        raw_value=accuracy, threshold_warning=0.45, threshold_critical=0.35,
        details=f"{correct}/{total} correct ({accuracy:.1%})",
    )


def _score_sharpe(sharpe: float) -> HealthDimension:
    """Score risk-adjusted returns via Sharpe ratio."""
    # Score: Sharpe 2+ → 100, 1 → 80, 0 → 50, -1 → 20, -2 → 0
    score = np.clip((sharpe + 2) / 4 * 100, 0, 100)
    return HealthDimension(
        name="Risk-Adjusted Return", score=float(score), weight=0.15,
        raw_value=sharpe, threshold_warning=0.5, threshold_critical=0.0,
        details=f"Sharpe: {sharpe:.2f}",
    )


def _score_calibration(ece: float) -> HealthDimension:
    """Score calibration error (lower is better)."""
    # Score: ECE 0 → 100, 0.05 → 80, 0.10 → 60, 0.20 → 20, 0.30+ → 0
    score = np.clip((0.30 - ece) / 0.30 * 100, 0, 100)
    return HealthDimension(
        name="Calibration", score=float(score), weight=0.12,
        raw_value=ece, threshold_warning=0.10, threshold_critical=0.20,
        details=f"ECE: {ece:.4f}",
    )


def _score_feature_drift(psi: float) -> HealthDimension:
    """Score feature drift via Population Stability Index."""
    # PSI < 0.1 → stable, 0.1-0.2 → moderate shift, > 0.2 → significant
    # Score: 0 → 100, 0.1 → 80, 0.2 → 50, 0.4 → 0
    score = np.clip((0.4 - psi) / 0.4 * 100, 0, 100)
    return HealthDimension(
        name="Feature Drift", score=float(score), weight=0.10,
        raw_value=psi, threshold_warning=0.10, threshold_critical=0.25,
        details=f"PSI: {psi:.4f}",
    )


def _score_prediction_drift(kl_divergence: float) -> HealthDimension:
    """Score prediction distribution drift via KL divergence."""
    # KL < 0.05 → stable, 0.05-0.15 → moderate, > 0.15 → severe
    score = np.clip((0.3 - kl_divergence) / 0.3 * 100, 0, 100)
    return HealthDimension(
        name="Prediction Drift", score=float(score), weight=0.10,
        raw_value=kl_divergence, threshold_warning=0.05, threshold_critical=0.15,
        details=f"KL Divergence: {kl_divergence:.4f}",
    )


def _score_backtest_deviation(deviation_pct: float) -> HealthDimension:
    """Score deviation between live and backtest performance."""
    # Deviation 0% → 100, 10% → 80, 25% → 50, 50%+ → 0
    abs_dev = abs(deviation_pct)
    score = np.clip((50 - abs_dev) / 50 * 100, 0, 100)
    return HealthDimension(
        name="Live-vs-Backtest", score=float(score), weight=0.12,
        raw_value=deviation_pct, threshold_warning=15.0, threshold_critical=30.0,
        details=f"Deviation: {deviation_pct:+.1f}%",
    )


def _score_slippage(actual_slippage_bps: float, expected_slippage_bps: float) -> HealthDimension:
    """Score execution quality via slippage comparison."""
    if expected_slippage_bps == 0:
        expected_slippage_bps = 5.0  # assume 5bps baseline
    ratio = actual_slippage_bps / expected_slippage_bps
    # Ratio 1.0 → 90 (some slippage is normal), 1.5 → 70, 2.0 → 50, 3.0 → 0
    score = np.clip((3.0 - ratio) / 2.0 * 100, 0, 100)
    return HealthDimension(
        name="Execution Quality", score=float(score), weight=0.08,
        raw_value=actual_slippage_bps, threshold_warning=10.0, threshold_critical=20.0,
        details=f"Slippage: {actual_slippage_bps:.1f}bps (expected: {expected_slippage_bps:.1f}bps)",
    )


def _score_latency(avg_latency_ms: float, p99_latency_ms: float) -> HealthDimension:
    """Score inference latency."""
    # p99 < 100ms → 100, 200ms → 80, 500ms → 50, 1000ms → 0
    score = np.clip((1000 - p99_latency_ms) / 900 * 100, 0, 100)
    return HealthDimension(
        name="Latency", score=float(score), weight=0.05,
        raw_value=p99_latency_ms, threshold_warning=200.0, threshold_critical=500.0,
        details=f"p99: {p99_latency_ms:.0f}ms (avg: {avg_latency_ms:.0f}ms)",
    )


def _score_data_completeness(completeness_pct: float) -> HealthDimension:
    """Score data availability/completeness."""
    # 100% → 100, 95% → 85, 90% → 70, 80% → 40, <70% → 0
    score = np.clip((completeness_pct - 70) / 30 * 100, 0, 100)
    return HealthDimension(
        name="Data Completeness", score=float(score), weight=0.08,
        raw_value=completeness_pct, threshold_warning=92.0, threshold_critical=85.0,
        details=f"Completeness: {completeness_pct:.1f}%",
    )


# ──────────────────────────────────────────────────────────────────────────
# Model Health Monitor — Main Class
# ──────────────────────────────────────────────────────────────────────────


class ModelHealthMonitor:
    """
    Continuous model health scoring and lifecycle governance.

    Evaluates deployed models on multiple dimensions and provides
    composite health scores that drive promotion/retirement decisions.

    Usage:
        monitor = ModelHealthMonitor()
        report = monitor.evaluate(model_version, metrics)
        if report.should_retire:
            # trigger model retirement
            ...
    """

    def __init__(
        self,
        retire_threshold: float = 55.0,
        warn_threshold: float = 71.0,
        reduce_threshold: float = 65.0,
        min_predictions_for_evaluation: int = 20,
        evaluation_window_hours: float = 168.0,  # 7 days
    ):
        self.retire_threshold = retire_threshold
        self.warn_threshold = warn_threshold
        self.reduce_threshold = reduce_threshold
        self.min_predictions = min_predictions_for_evaluation
        self.evaluation_window_hours = evaluation_window_hours

        # Track history per model version
        self._histories: Dict[str, HealthHistory] = {}

    def evaluate(self, model_version: str, metrics: Dict[str, Any]) -> ModelHealthReport:
        """
        Evaluate model health and return comprehensive report.

        Args:
            model_version: Version identifier (e.g., "v014.3").
            metrics: Dict containing all available health metrics:
                - correct_predictions (int)
                - total_predictions (int)
                - sharpe_ratio (float)
                - calibration_ece (float)
                - feature_psi (float)
                - prediction_kl_divergence (float)
                - backtest_deviation_pct (float)
                - actual_slippage_bps (float)
                - expected_slippage_bps (float)
                - avg_latency_ms (float)
                - p99_latency_ms (float)
                - data_completeness_pct (float)
                - n_resolved_trades (int)

        Returns:
            ModelHealthReport with composite score and recommendations.
        """
        dimensions = []

        # 1. Accuracy
        dimensions.append(_score_accuracy(
            metrics.get("correct_predictions", 0),
            metrics.get("total_predictions", 0),
        ))

        # 2. Risk-adjusted returns
        dimensions.append(_score_sharpe(
            metrics.get("sharpe_ratio", 0.0),
        ))

        # 3. Calibration
        dimensions.append(_score_calibration(
            metrics.get("calibration_ece", 0.10),
        ))

        # 4. Feature drift
        dimensions.append(_score_feature_drift(
            metrics.get("feature_psi", 0.0),
        ))

        # 5. Prediction drift
        dimensions.append(_score_prediction_drift(
            metrics.get("prediction_kl_divergence", 0.0),
        ))

        # 6. Live-vs-backtest deviation
        dimensions.append(_score_backtest_deviation(
            metrics.get("backtest_deviation_pct", 0.0),
        ))

        # 7. Execution quality
        dimensions.append(_score_slippage(
            metrics.get("actual_slippage_bps", 5.0),
            metrics.get("expected_slippage_bps", 5.0),
        ))

        # 8. Latency
        dimensions.append(_score_latency(
            metrics.get("avg_latency_ms", 50.0),
            metrics.get("p99_latency_ms", 100.0),
        ))

        # 9. Data completeness
        dimensions.append(_score_data_completeness(
            metrics.get("data_completeness_pct", 100.0),
        ))

        # Compute weighted composite score
        total_weight = sum(d.weight for d in dimensions)
        composite = sum(d.score * d.weight for d in dimensions) / total_weight if total_weight > 0 else 0.0
        composite = float(np.clip(composite, 0, 100))

        # Track history
        now = time.time()
        if model_version not in self._histories:
            self._histories[model_version] = HealthHistory()
        self._histories[model_version].add(composite, now)

        # Apply trend penalty: declining models get score reduction
        history = self._histories[model_version]
        if history.is_declining:
            trend_penalty = abs(history.trend) * 5  # up to 5 point penalty
            composite = max(0, composite - trend_penalty)

        # Determine actions
        grade = HealthGrade.from_score(composite)
        should_retire = composite < self.retire_threshold
        should_warn = composite < self.warn_threshold
        should_reduce = composite < self.reduce_threshold

        # Collect retirement reasons
        retirement_reasons = []
        if should_retire:
            for dim in dimensions:
                if dim.score < 40:
                    retirement_reasons.append(
                        f"{dim.name} critically low: {dim.score:.0f}/100 ({dim.details})"
                    )
            if history.is_declining:
                retirement_reasons.append(
                    f"Sustained decline: trend={history.trend:.2f} over {len(history.scores)} evaluations"
                )

        # Generate recommendations
        recommendations = self._generate_recommendations(dimensions, history, composite)

        report = ModelHealthReport(
            model_version=model_version,
            timestamp=now,
            composite_score=composite,
            grade=grade,
            dimensions=dimensions,
            n_predictions=metrics.get("total_predictions", 0),
            n_resolved_trades=metrics.get("n_resolved_trades", 0),
            evaluation_window_hours=self.evaluation_window_hours,
            should_retire=should_retire,
            should_warn=should_warn,
            should_reduce_allocation=should_reduce,
            retirement_reasons=retirement_reasons,
            recommendations=recommendations,
        )

        logger.info(
            "model_health.evaluated",
            model_version=model_version,
            score=round(composite, 1),
            grade=grade.value,
            should_retire=should_retire,
        )

        return report

    def get_history(self, model_version: str) -> Optional[HealthHistory]:
        """Get health score history for a model version."""
        return self._histories.get(model_version)

    def compare_models(
        self,
        metrics_a: Dict[str, Any],
        metrics_b: Dict[str, Any],
        version_a: str = "champion",
        version_b: str = "challenger",
    ) -> Tuple[ModelHealthReport, ModelHealthReport, str]:
        """
        Compare two models' health and recommend which should be active.

        Returns:
            Tuple of (report_a, report_b, recommendation)
        """
        report_a = self.evaluate(version_a, metrics_a)
        report_b = self.evaluate(version_b, metrics_b)

        if report_a.should_retire and not report_b.should_retire:
            recommendation = f"Promote {version_b}: {version_a} is degraded"
        elif report_b.composite_score > report_a.composite_score + 5:
            recommendation = f"Consider promoting {version_b}: +{report_b.composite_score - report_a.composite_score:.1f} health advantage"
        elif report_a.composite_score > report_b.composite_score + 5:
            recommendation = f"Keep {version_a}: +{report_a.composite_score - report_b.composite_score:.1f} health advantage"
        else:
            recommendation = "No significant difference — keep current champion"

        return report_a, report_b, recommendation

    def _generate_recommendations(
        self,
        dimensions: List[HealthDimension],
        history: HealthHistory,
        composite: float,
    ) -> List[str]:
        """Generate actionable recommendations based on health dimensions."""
        recs = []

        # Find weakest dimensions
        sorted_dims = sorted(dimensions, key=lambda d: d.score)
        for dim in sorted_dims[:3]:  # top 3 weakest
            if dim.score < 60:
                if dim.name == "Accuracy":
                    recs.append("Retrain model with recent data — accuracy below acceptable threshold")
                elif dim.name == "Calibration":
                    recs.append("Recalibrate confidence scores — ECE indicates miscalibration")
                elif dim.name == "Feature Drift":
                    recs.append("Feature distributions have shifted — validate feature pipeline")
                elif dim.name == "Prediction Drift":
                    recs.append("Prediction distribution drifting — model may be stale")
                elif dim.name == "Live-vs-Backtest":
                    recs.append("Significant live/backtest gap — investigate execution or data differences")
                elif dim.name == "Execution Quality":
                    recs.append("Slippage exceeding expectations — review order routing and sizing")
                elif dim.name == "Latency":
                    recs.append("Inference latency high — optimize model or infrastructure")
                elif dim.name == "Data Completeness":
                    recs.append("Missing data detected — check data pipeline health")

        if history.is_declining:
            recs.append(f"Health trending down (slope={history.trend:.2f}) — consider preemptive retraining")

        if composite >= 82 and not recs:
            recs.append("Model healthy — no immediate action required")

        return recs


# ──────────────────────────────────────────────────────────────────────────
# Convenience: Collect Metrics from Existing Infrastructure
# ──────────────────────────────────────────────────────────────────────────


def collect_model_metrics(
    model_version: str,
    experience_db: Any = None,
    drift_monitor: Any = None,
    calibration_analyzer: Any = None,
    health_monitor: Any = None,
) -> Dict[str, Any]:
    """
    Collect all health metrics from existing infrastructure components.

    This bridges the various monitoring systems into a single metrics dict
    suitable for ModelHealthMonitor.evaluate().

    Args:
        model_version: Model version to collect metrics for.
        experience_db: ExperienceDatabase instance (from feedback_loop).
        drift_monitor: DriftMonitor instance (from intelligence/drift).
        calibration_analyzer: CalibrationAnalyzer instance (from predictions).
        health_monitor: HealthMonitor instance (from monitoring).

    Returns:
        Dict of metrics ready for ModelHealthMonitor.evaluate().
    """
    metrics: Dict[str, Any] = {
        "correct_predictions": 0,
        "total_predictions": 0,
        "sharpe_ratio": 0.0,
        "calibration_ece": 0.10,
        "feature_psi": 0.0,
        "prediction_kl_divergence": 0.0,
        "backtest_deviation_pct": 0.0,
        "actual_slippage_bps": 5.0,
        "expected_slippage_bps": 5.0,
        "avg_latency_ms": 50.0,
        "p99_latency_ms": 100.0,
        "data_completeness_pct": 100.0,
        "n_resolved_trades": 0,
    }

    # From ExperienceDatabase
    if experience_db is not None:
        try:
            perf = experience_db.get_model_performance(model_version)
            if perf:
                metrics["correct_predictions"] = perf.get("correct", 0)
                metrics["total_predictions"] = perf.get("total", 0)
                metrics["sharpe_ratio"] = perf.get("sharpe_ratio", 0.0)
                metrics["n_resolved_trades"] = perf.get("n_trades", 0)
                metrics["actual_slippage_bps"] = perf.get("avg_slippage_bps", 5.0)
        except Exception as e:
            logger.warning(f"Failed to collect experience metrics: {e}")

    # From DriftMonitor
    if drift_monitor is not None:
        try:
            state = drift_monitor.get_state()
            if state:
                metrics["feature_psi"] = getattr(state, "psi", 0.0)
                metrics["prediction_kl_divergence"] = getattr(state, "kl_divergence", 0.0)
        except Exception as e:
            logger.warning(f"Failed to collect drift metrics: {e}")

    # From CalibrationAnalyzer
    if calibration_analyzer is not None:
        try:
            cal_report = calibration_analyzer.analyze()
            if cal_report:
                metrics["calibration_ece"] = getattr(cal_report, "ece", 0.10)
        except Exception as e:
            logger.warning(f"Failed to collect calibration metrics: {e}")

    # From HealthMonitor (system health)
    if health_monitor is not None:
        try:
            report = health_monitor.get_report()
            if report:
                # Data completeness from pipeline health
                metrics["data_completeness_pct"] = report.get("data_completeness", 100.0)
                metrics["avg_latency_ms"] = report.get("avg_latency_ms", 50.0)
                metrics["p99_latency_ms"] = report.get("p99_latency_ms", 100.0)
        except Exception as e:
            logger.warning(f"Failed to collect system health metrics: {e}")

    return metrics

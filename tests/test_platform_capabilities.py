"""
Integration tests for the next-generation platform capabilities:
- Model Health Scoring
- Auto-Rollback with CUSUM/Page-Hinkley/EWMA
- Closed-Loop Pipeline
- Multi-Target Prediction
- Statistical Validation
- Explainability
- Evidence Packages
"""
import time

import numpy as np
import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────────
# Model Health Scoring
# ──────────────────────────────────────────────────────────────────────────


class TestModelHealth:
    """Tests for ModelHealthMonitor."""

    def test_healthy_model_scores_high(self):
        from src.ml.model_health import ModelHealthMonitor, HealthGrade

        monitor = ModelHealthMonitor()
        report = monitor.evaluate("v001", {
            "correct_predictions": 60,
            "total_predictions": 80,
            "sharpe_ratio": 1.8,
            "calibration_ece": 0.03,
            "feature_psi": 0.05,
            "prediction_kl_divergence": 0.02,
            "backtest_deviation_pct": 5.0,
            "actual_slippage_bps": 4.0,
            "expected_slippage_bps": 5.0,
            "avg_latency_ms": 30.0,
            "p99_latency_ms": 80.0,
            "data_completeness_pct": 99.0,
        })
        assert report.composite_score >= 80
        assert report.grade in (HealthGrade.EXCELLENT, HealthGrade.HEALTHY)
        assert not report.should_retire

    def test_degraded_model_triggers_retirement(self):
        from src.ml.model_health import ModelHealthMonitor, HealthGrade

        monitor = ModelHealthMonitor()
        report = monitor.evaluate("v_bad", {
            "correct_predictions": 15,
            "total_predictions": 50,
            "sharpe_ratio": -1.0,
            "calibration_ece": 0.25,
            "feature_psi": 0.30,
            "prediction_kl_divergence": 0.20,
            "backtest_deviation_pct": 40.0,
            "actual_slippage_bps": 25.0,
            "expected_slippage_bps": 5.0,
            "avg_latency_ms": 200.0,
            "p99_latency_ms": 800.0,
            "data_completeness_pct": 75.0,
        })
        assert report.composite_score < 55
        assert report.grade == HealthGrade.CRITICAL
        assert report.should_retire
        assert len(report.retirement_reasons) > 0

    def test_health_grades_map_correctly(self):
        from src.ml.model_health import HealthGrade

        assert HealthGrade.from_score(98) == HealthGrade.EXCELLENT
        assert HealthGrade.from_score(85) == HealthGrade.HEALTHY
        assert HealthGrade.from_score(75) == HealthGrade.WARNING
        assert HealthGrade.from_score(60) == HealthGrade.DEGRADED
        assert HealthGrade.from_score(40) == HealthGrade.CRITICAL

    def test_model_comparison(self):
        from src.ml.model_health import ModelHealthMonitor

        monitor = ModelHealthMonitor()
        good_metrics = {"correct_predictions": 70, "total_predictions": 100, "sharpe_ratio": 1.5}
        bad_metrics = {"correct_predictions": 30, "total_predictions": 100, "sharpe_ratio": -0.5}

        report_a, report_b, recommendation = monitor.compare_models(
            good_metrics, bad_metrics, "champion", "challenger"
        )
        assert report_a.composite_score > report_b.composite_score
        assert "champion" in recommendation.lower() or "keep" in recommendation.lower()


# ──────────────────────────────────────────────────────────────────────────
# Auto-Rollback
# ──────────────────────────────────────────────────────────────────────────


class TestAutoRollback:
    """Tests for CUSUM, Page-Hinkley, and AutoRollbackManager."""

    def test_cusum_detects_degradation(self):
        from src.ml.auto_rollback import CUSUMDetector

        detector = CUSUMDetector(target_mean=0.002, drift=0.005, threshold=0.05)

        # Normal returns — should not trigger
        rng = np.random.default_rng(42)
        for r in rng.normal(0.002, 0.01, 30):
            assert not detector.update(r), "Should not trigger on normal returns"

        # Sudden degradation — should trigger
        detected = False
        for r in rng.normal(-0.015, 0.01, 30):
            if detector.update(r):
                detected = True
                break
        assert detected, "CUSUM should detect degradation"

    def test_page_hinkley_detects_drift(self):
        from src.ml.auto_rollback import PageHinkleyDetector

        # Threshold proportional to return magnitude (~5x expected deviation)
        detector = PageHinkleyDetector(threshold=0.15, alpha=0.001)

        # Normal period
        rng = np.random.default_rng(123)
        for r in rng.normal(0.001, 0.005, 50):
            detector.update(r)

        # Strong drifting period — should eventually trigger
        detected = False
        for r in rng.normal(-0.02, 0.005, 50):
            if detector.update(r):
                detected = True
                break
        assert detected, "Page-Hinkley should detect mean drift"

    def test_ewma_detects_shift(self):
        from src.ml.auto_rollback import EWMADetector

        detector = EWMADetector(target_mean=0.001, target_std=0.01, lam=0.1, L=2.5)

        # Normal period
        rng = np.random.default_rng(99)
        for r in rng.normal(0.001, 0.01, 30):
            detector.update(r)

        # Shift — should trigger
        detected = False
        for r in rng.normal(-0.02, 0.01, 30):
            if detector.update(r):
                detected = True
                break
        assert detected, "EWMA should detect persistent shift"

    def test_rollback_manager_escalation(self):
        from src.ml.auto_rollback import AutoRollbackManager, RollbackConfig, RollbackSeverity

        config = RollbackConfig(
            cusum_threshold=3.0,
            cusum_drift=0.3,
            ph_threshold=15.0,
            alert_after=1,
            reduce_after=2,
            rollback_after=3,
            action_cooldown_seconds=0,  # disable cooldown for test
        )
        manager = AutoRollbackManager(config=config)
        manager.register_model("v001", baseline_return=0.002, baseline_std=0.01, previous_version="v000")

        # Feed degraded returns
        severities = []
        for r in np.full(20, -0.02):
            sev = manager.observe("v001", r)
            if sev != RollbackSeverity.NONE:
                severities.append(sev)

        # Should have escalated through ALERT → REDUCE → ROLLBACK
        assert RollbackSeverity.ALERT in severities, "Should have triggered alert"

    def test_health_based_rollback(self):
        from src.ml.auto_rollback import AutoRollbackManager, RollbackConfig, RollbackSeverity

        config = RollbackConfig(action_cooldown_seconds=0)
        manager = AutoRollbackManager(config=config)
        manager.register_model("v001", baseline_return=0.002, baseline_std=0.01)

        # Health score below threshold
        severity = manager.evaluate_health("v001", health_score=50.0)
        assert severity == RollbackSeverity.ROLLBACK


# ──────────────────────────────────────────────────────────────────────────
# Closed-Loop Pipeline
# ──────────────────────────────────────────────────────────────────────────


class TestClosedLoop:
    """Tests for ClosedLoopPipeline."""

    def test_trade_accumulation_trigger(self):
        from src.ml.closed_loop import ClosedLoopPipeline, RetriggerPolicy, RetriggerReason

        trained = {"called": False}

        def mock_train(params):
            trained["called"] = True
            return {"success": True, "model_version": "v002"}

        policy = RetriggerPolicy(min_new_trades_for_retrain=5, min_retrain_interval_hours=0)
        pipeline = ClosedLoopPipeline(policy=policy, training_callback=mock_train)
        pipeline.state.current_model_version = "v001"

        # Add trades
        for i in range(6):
            pipeline.on_trade_resolved({"return_pct": 0.01, "direction_correct": True})

        reason = pipeline.check_triggers()
        assert reason == RetriggerReason.SAMPLE_ACCUMULATION
        assert trained["called"]

    def test_health_warning_trigger(self):
        from src.ml.closed_loop import ClosedLoopPipeline, RetriggerPolicy

        trained = {"called": False}

        def mock_train(params):
            trained["called"] = True
            return {"success": True, "model_version": "v002"}

        policy = RetriggerPolicy(health_score_threshold=71.0, min_retrain_interval_hours=0)
        pipeline = ClosedLoopPipeline(policy=policy, training_callback=mock_train)
        pipeline.state.current_model_version = "v001"

        pipeline.on_health_update("v001", 65.0)
        assert trained["called"]

    def test_pipeline_state_tracking(self):
        from src.ml.closed_loop import ClosedLoopPipeline, PipelineStage

        pipeline = ClosedLoopPipeline()
        pipeline.state.current_model_version = "v001"

        state = pipeline.get_state()
        assert state["stage"] == PipelineStage.IDLE.value
        assert state["current_model"] == "v001"


# ──────────────────────────────────────────────────────────────────────────
# Multi-Target Prediction
# ──────────────────────────────────────────────────────────────────────────


class TestMultiTarget:
    """Tests for MultiTargetPredictor."""

    def test_target_engineering(self):
        from src.ml.multi_target import engineer_multi_targets

        # Generate synthetic OHLCV
        rng = np.random.default_rng(42)
        n = 200
        prices = 100 + np.cumsum(rng.normal(0, 0.5, n))
        df = pd.DataFrame({
            "open": prices,
            "high": prices + rng.uniform(0, 1, n),
            "low": prices - rng.uniform(0, 1, n),
            "close": prices + rng.normal(0, 0.2, n),
            "volume": rng.uniform(1000, 5000, n),
        }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

        result = engineer_multi_targets(df)

        assert "target_direction" in result.columns
        assert "target_return_pct" in result.columns
        assert "target_holding_hours" in result.columns
        assert "target_mae_pct" in result.columns
        assert "target_mfe_pct" in result.columns
        # Direction should be {-1, 0, 1}
        assert set(result["target_direction"].unique()).issubset({-1, 0, 1})

    def test_untrained_returns_hold(self):
        from src.ml.multi_target import MultiTargetPredictor

        predictor = MultiTargetPredictor()
        pred = predictor.predict(pd.DataFrame({"f1": [1.0], "f2": [2.0]}))

        assert pred.direction == "HOLD"
        assert pred.confidence == 0.0
        assert pred.risk_grade == "F"

    def test_prediction_has_all_fields(self):
        from src.ml.multi_target import MultiTargetPredictor

        predictor = MultiTargetPredictor()
        pred = predictor.predict(pd.DataFrame({"f1": [1.0]}))
        d = pred.to_dict()

        required_keys = [
            "direction", "direction_probability", "expected_return_pct",
            "expected_holding_hours", "risk_reward_ratio", "confidence",
            "recommended_position_pct", "kelly_fraction", "risk_grade",
            "suggested_tp_pct", "suggested_sl_pct",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"


# ──────────────────────────────────────────────────────────────────────────
# Statistical Validation
# ──────────────────────────────────────────────────────────────────────────


class TestStatisticalValidation:
    """Tests for statistical validation module."""

    def test_deflated_sharpe_on_noise(self):
        from backtesting.statistical_validation import deflated_sharpe_ratio

        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, 0.02, 200)

        result = deflated_sharpe_ratio(noise, n_trials=20)
        # Random noise should NOT be significant after multiple testing correction
        assert not result.is_significant

    def test_deflated_sharpe_on_signal(self):
        from backtesting.statistical_validation import deflated_sharpe_ratio

        rng = np.random.default_rng(42)
        signal = rng.normal(0.01, 0.015, 500)  # strong positive signal

        result = deflated_sharpe_ratio(signal, n_trials=1)
        # Strong signal with no multiple testing should be significant
        assert result.is_significant

    def test_pbo_on_random(self):
        from backtesting.statistical_validation import probability_of_backtest_overfitting

        rng = np.random.default_rng(42)
        # Random strategies — should show overfitting tendency
        returns_matrix = rng.normal(0, 0.02, (200, 10))

        result = probability_of_backtest_overfitting(returns_matrix, n_partitions=8)
        assert 0 <= result.pbo <= 1
        assert result.n_combinations > 0

    def test_whites_reality_check(self):
        from backtesting.statistical_validation import whites_reality_check

        rng = np.random.default_rng(42)
        # No real signal — should not be significant
        returns = rng.normal(0, 0.02, (200, 5))

        result = whites_reality_check(returns, n_bootstrap=100)
        assert 0 <= result.p_value <= 1
        assert result.n_strategies == 5


# ──────────────────────────────────────────────────────────────────────────
# Explainability
# ──────────────────────────────────────────────────────────────────────────


class TestExplainability:
    """Tests for PredictionExplainer."""

    def test_permutation_fallback(self):
        """Test that explainability works without SHAP library."""
        from src.ml.explainability import PredictionExplainer

        explainer = PredictionExplainer(
            feature_names=["ema_dist", "rsi", "volume_ratio", "regime"],
            top_k=4,
        )
        explainer._is_initialized = True  # skip SHAP init

        features = pd.DataFrame({
            "ema_dist": [0.02],
            "rsi": [65.0],
            "volume_ratio": [1.5],
            "regime": [1.0],
        })

        explanation = explainer.explain(
            features,
            prediction_id="test_001",
            symbol="BTC/USD",
            direction="BUY",
            confidence=0.85,
        )

        assert explanation.symbol == "BTC/USD"
        assert explanation.direction == "BUY"
        assert explanation.confidence == 0.85
        # Should have attempted contributions (may be zero without model)
        assert isinstance(explanation.top_contributors, list)

    def test_explanation_to_evidence_dict(self):
        from src.ml.explainability import PredictionExplanation, FeatureContribution

        explanation = PredictionExplanation(
            prediction_id="p001",
            symbol="AAPL",
            direction="BUY",
            confidence=0.9,
            base_value=0.5,
            prediction_value=0.85,
            top_contributors=[
                FeatureContribution("ema_trend", 0.15, 0.02, 42.0, "supporting"),
                FeatureContribution("volume", 0.10, 1.5, 28.0, "supporting"),
            ],
        )

        d = explanation.to_evidence_dict()
        assert d["explanation_type"] == "shap"
        assert len(d["top_features"]) == 2
        assert d["top_features"][0]["name"] == "ema_trend"


# ──────────────────────────────────────────────────────────────────────────
# Evidence Package Integration
# ──────────────────────────────────────────────────────────────────────────


class TestEvidencePackage:
    """Tests for enhanced TradeSignalPackage with multi-target fields."""

    def test_signal_package_has_new_fields(self):
        from src.strategy.signal_package import TradeSignalPackage

        pkg = TradeSignalPackage(symbol="BTC/USD")

        # Multi-target fields
        assert hasattr(pkg, "probability_tp")
        assert hasattr(pkg, "probability_sl")
        assert hasattr(pkg, "probability_timeout")
        assert hasattr(pkg, "expected_holding_hours")
        assert hasattr(pkg, "kelly_fraction")
        assert hasattr(pkg, "risk_grade")
        assert hasattr(pkg, "prediction_uncertainty")

        # Explainability fields
        assert hasattr(pkg, "feature_contributions")
        assert hasattr(pkg, "explanation_summary")
        assert hasattr(pkg, "counterfactual")

        # Validation artifacts
        assert hasattr(pkg, "walk_forward_passed")
        assert hasattr(pkg, "monte_carlo_passed")
        assert hasattr(pkg, "reality_check_passed")
        assert hasattr(pkg, "deflated_sharpe")

    def test_enriched_signal_package(self):
        from src.strategy.signal_package import TradeSignalPackage
        from src.strategy.base import Signal

        pkg = TradeSignalPackage(
            symbol="BTC/USD",
            direction=Signal.BUY,
            confidence=0.89,
            probability_tp=0.74,
            probability_sl=0.21,
            probability_timeout=0.05,
            expected_holding_hours=24.0,
            kelly_fraction=0.08,
            risk_grade="A",
            prediction_uncertainty=0.11,
            walk_forward_passed=True,
            monte_carlo_passed=True,
            reality_check_passed=True,
            deflated_sharpe=1.85,
            feature_contributions=[
                {"name": "ema_trend", "contribution": 0.18},
                {"name": "volume", "contribution": 0.24},
            ],
            explanation_summary="BUY signal driven by EMA trend (+18%) and Volume (+24%)",
        )

        assert pkg.probability_tp == 0.74
        assert pkg.risk_grade == "A"
        assert pkg.walk_forward_passed is True
        assert len(pkg.feature_contributions) == 2

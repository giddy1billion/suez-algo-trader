"""Tests for the adaptive intelligence bounded context."""

import numpy as np
import pandas as pd

from src.intelligence.orchestrator import AdaptiveIntelligenceOrchestrator


def _sample_df(n: int = 120, drift: float = 0.2, vol: float = 1.0) -> pd.DataFrame:
    np.random.seed(7)
    base = 100 + np.cumsum(np.random.randn(n) * vol + drift)
    base = np.maximum(base, 1)
    return pd.DataFrame(
        {
            "open": base - 0.3,
            "high": base + 0.6,
            "low": base - 0.8,
            "close": base,
            "volume": np.random.randint(100_000, 300_000, n),
        }
    )


class TestAdaptiveIntelligenceOrchestrator:
    def test_accepts_high_quality_momentum_signal(self):
        orch = AdaptiveIntelligenceOrchestrator(min_trade_score=55)
        df = _sample_df(drift=0.35, vol=0.7)

        decision = orch.evaluate_signal(
            strategy_name="momentum",
            signal_confidence=0.86,
            indicators={"rsi": 64, "rr": 2.8},
            df=df,
            portfolio_context={"correlation_risk": 0.2},
            execution_context={"spread_pct": 0.001},
        )

        assert decision.accepted is True
        assert decision.final_score >= 55
        assert decision.market_state.overall_regime in {"TRENDING_LOW_VOL", "TRENDING_HIGH_VOL"}
        assert decision.qty_multiplier > 0

    def test_blocks_regime_mismatch(self):
        orch = AdaptiveIntelligenceOrchestrator(min_trade_score=70)
        # Deterministic sideways series intended to fail momentum routing
        n = 120
        x = np.arange(n)
        close = 100 + (np.sin(x / 3.0) * 1.2)
        df = pd.DataFrame(
            {
                "open": close - 0.2,
                "high": close + 0.4,
                "low": close - 0.4,
                "close": close,
                "volume": np.full(n, 120_000),
            }
        )

        decision = orch.evaluate_signal(
            strategy_name="momentum",
            signal_confidence=0.72,
            indicators={"rsi": 49, "rr": 1.1},
            df=df,
            portfolio_context={"correlation_risk": 0.8},
            execution_context={"spread_pct": 0.006},
        )

        assert decision.accepted is False
        assert "Routing: BLOCK" in decision.explanation
        assert decision.qty_multiplier == 0.0

    def test_drift_monitor_triggers_degrading_state(self):
        orch = AdaptiveIntelligenceOrchestrator(
            min_trade_score=40,
            drift_window=60,
            drift_min_samples=30,
            drift_alert_drop=0.15,
        )

        # First half: high accuracy
        for _ in range(30):
            orch.record_outcome(True)
        # Second half: poor accuracy
        for i in range(30):
            orch.record_outcome(i % 4 == 0)

        state = orch.drift_monitor.get_state()
        assert state.sample_size == 60
        assert state.degrading is True
        assert state.accuracy_drop >= 0.15

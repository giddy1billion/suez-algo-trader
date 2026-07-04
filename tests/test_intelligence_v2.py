"""Tests for Market State Engine, Meta Strategy, Decision Journal, and Counterfactual Engine."""

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from src.intelligence.market_state.engine import MarketStateEngine, MarketFingerprint
from src.intelligence.meta_strategy.engine import MetaStrategyEngine
from src.intelligence.journal.decision_journal import DecisionJournal, DecisionRecord
from src.intelligence.counterfactual.engine import CounterfactualEngine


def _trending_df(n: int = 120) -> pd.DataFrame:
    np.random.seed(42)
    base = 100 + np.cumsum(np.ones(n) * 0.5 + np.random.randn(n) * 0.2)
    return pd.DataFrame({
        "open": base - 0.3,
        "high": base + 0.6,
        "low": base - 0.5,
        "close": base,
        "volume": np.random.randint(150_000, 400_000, n),
    })


def _sideways_df(n: int = 120) -> pd.DataFrame:
    x = np.arange(n)
    close = 100.0 + np.sin(x / 3.0) * 1.0
    return pd.DataFrame({
        "open": close - 0.2,
        "high": close + 0.3,
        "low": close - 0.3,
        "close": close,
        "volume": np.full(n, 200_000),
    })


class TestMarketStateEngine:
    def test_trending_market_detection(self):
        engine = MarketStateEngine()
        df = _trending_df()
        fp = engine.compute(df)
        assert isinstance(fp, MarketFingerprint)
        assert "Uptrend" in fp.trend.label
        assert "TRENDING" in fp.overall_regime
        assert fp.overall_confidence > 0.4

    def test_sideways_market_detection(self):
        engine = MarketStateEngine()
        df = _sideways_df()
        fp = engine.compute(df)
        assert fp.trend.label == "Sideways"
        assert "RANGING" in fp.overall_regime

    def test_insufficient_data_returns_unknown(self):
        engine = MarketStateEngine()
        fp = engine.compute(pd.DataFrame({"close": [100, 101], "volume": [1000, 1000]}))
        assert fp.overall_regime == "UNKNOWN"
        assert fp.overall_confidence == 0.3

    def test_none_df_returns_unknown(self):
        engine = MarketStateEngine()
        fp = engine.compute(None)
        assert fp.overall_regime == "UNKNOWN"

    def test_fingerprint_to_dict(self):
        engine = MarketStateEngine()
        fp = engine.compute(_trending_df())
        d = fp.to_dict()
        assert "trend" in d
        assert "volatility" in d
        assert "overall_regime" in d
        assert "timestamp" in d

    def test_stress_detection(self):
        engine = MarketStateEngine()
        # Simulate a drawdown: goes up then sharply down
        np.random.seed(10)
        up = 100 + np.cumsum(np.ones(60) * 0.3)
        down = up[-1] + np.cumsum(-np.ones(60) * 0.8)
        close = np.concatenate([up, down])
        df = pd.DataFrame({
            "open": close - 0.2,
            "high": close + 0.3,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(120, 200_000),
        })
        fp = engine.compute(df)
        assert fp.stress.label in ("Elevated", "Panic")


class TestMetaStrategyEngine:
    def test_rank_returns_sorted_strategies(self):
        engine = MetaStrategyEngine()
        fp = MarketStateEngine().compute(_trending_df())
        rankings = engine.rank(fp)
        assert len(rankings) > 0
        scores = [r.score for r in rankings]
        assert scores == sorted(scores, reverse=True)

    def test_trending_market_favors_momentum(self):
        engine = MetaStrategyEngine()
        fp = MarketStateEngine().compute(_trending_df())
        rankings = engine.rank(fp)
        names = [r.name for r in rankings]
        # momentum should rank higher than mean_reversion in a trending market
        momentum_idx = names.index("momentum") if "momentum" in names else 999
        mean_rev_idx = names.index("mean_reversion") if "mean_reversion" in names else -1
        assert momentum_idx < mean_rev_idx

    def test_record_performance_updates_scores(self):
        engine = MetaStrategyEngine()
        # Record good performance for momentum
        for _ in range(10):
            engine.record_trade_result("momentum", pnl=100.0)
        fp = MarketStateEngine().compute(_trending_df())
        rankings = engine.rank(fp)
        momentum = next((r for r in rankings if r.name == "momentum"), None)
        assert momentum is not None
        assert momentum.score > 50


class TestDecisionJournal:
    def test_record_and_query(self):
        journal = DecisionJournal()
        record = DecisionRecord(
            decision_id="test-1",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC",
            strategy="momentum",
            side="buy",
            accepted=True,
            signal_confidence=0.85,
            regime="TRENDING_LOW_VOL",
            trade_score=88.0,
        )
        journal.record(record)
        assert journal.count == 1

        results = journal.query(symbol="BTC")
        assert len(results) == 1
        assert results[0].decision_id == "test-1"

    def test_query_filters(self):
        journal = DecisionJournal()
        for i in range(5):
            journal.record(DecisionRecord(
                decision_id=f"t-{i}",
                timestamp=datetime.now(timezone.utc),
                symbol="BTC" if i < 3 else "ETH",
                strategy="momentum",
                side="buy",
                accepted=i < 4,
                signal_confidence=0.7 + i * 0.05,
                trade_score=70 + i * 5,
            ))

        assert len(journal.query(symbol="BTC")) == 3
        assert len(journal.query(symbol="ETH")) == 2
        assert len(journal.query(accepted=True)) == 4
        assert len(journal.query(min_score=80)) == 3

    def test_update_outcome(self):
        journal = DecisionJournal()
        journal.record(DecisionRecord(
            decision_id="outcome-1",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC",
            strategy="momentum",
            side="buy",
            accepted=True,
            signal_confidence=0.8,
            trade_score=85,
        ))
        assert journal.update_outcome("outcome-1", pnl=250.0, pnl_pct=2.5)
        results = journal.query(outcome_positive=True)
        assert len(results) == 1
        assert results[0].outcome_pnl == 250.0

    def test_analytics(self):
        journal = DecisionJournal()
        for i in range(10):
            r = DecisionRecord(
                decision_id=f"a-{i}",
                timestamp=datetime.now(timezone.utc),
                symbol="BTC",
                strategy="momentum",
                side="buy",
                accepted=True,
                signal_confidence=0.8,
                trade_score=80,
            )
            journal.record(r)
            journal.update_outcome(f"a-{i}", pnl=100 if i < 7 else -50, pnl_pct=1.0 if i < 7 else -0.5)

        analytics = journal.get_analytics()
        assert analytics["total"] == 10
        assert analytics["win_rate"] == 0.7
        assert analytics["profit_factor"] > 1.0

    def test_regime_breakdown(self):
        journal = DecisionJournal()
        for regime, pnl in [("TRENDING_LOW_VOL", 100), ("TRENDING_LOW_VOL", 50), ("RANGING_LOW_VOL", -30)]:
            r = DecisionRecord(
                decision_id=f"rb-{regime}-{pnl}",
                timestamp=datetime.now(timezone.utc),
                symbol="BTC",
                strategy="momentum",
                side="buy",
                accepted=True,
                signal_confidence=0.8,
                regime=regime,
                trade_score=80,
            )
            journal.record(r)
            journal.update_outcome(r.decision_id, pnl=pnl, pnl_pct=pnl / 1000)

        breakdown = journal.get_regime_breakdown()
        assert "TRENDING_LOW_VOL" in breakdown
        assert breakdown["TRENDING_LOW_VOL"]["win_rate"] == 1.0


class TestCounterfactualEngine:
    def test_open_and_resolve(self):
        engine = CounterfactualEngine()
        record = engine.open_record(
            decision_id="cf-1",
            symbol="BTC",
            side="buy",
            accepted=True,
            entry_price=100.0,
            qty=10.0,
        )
        assert record is not None
        assert engine.count == 1

        resolved = engine.resolve("cf-1", exit_price=110.0, actual_pnl=100.0)
        assert resolved is not None
        # half_size and double_size scenarios should be added
        scenario_names = [s.scenario for s in resolved.scenarios]
        assert "no_trade" in scenario_names
        assert "half_size" in scenario_names
        assert "double_size" in scenario_names
        assert resolved.actual_pnl == 100.0

    def test_rejected_trade_counterfactual(self):
        engine = CounterfactualEngine()
        engine.open_record(
            decision_id="cf-rejected",
            symbol="ETH",
            side="buy",
            accepted=False,
            entry_price=50.0,
            qty=5.0,
        )
        resolved = engine.resolve("cf-rejected", exit_price=60.0, actual_pnl=0.0)
        assert resolved is not None
        # Should have "if_accepted" scenario
        scenario_names = [s.scenario for s in resolved.scenarios]
        assert "if_accepted" in scenario_names
        # If accepted, PnL = (60-50) * 5 = 50
        accepted_scenario = next(s for s in resolved.scenarios if s.scenario == "if_accepted")
        assert accepted_scenario.hypothetical_pnl == 50.0

    def test_regret_calculation(self):
        engine = CounterfactualEngine()
        engine.open_record("cf-2", "BTC", "buy", True, 100.0, 10.0)
        engine.add_alternative_scenario("cf-2", "alt_breakout", 500.0)
        resolved = engine.resolve("cf-2", exit_price=110.0, actual_pnl=100.0)
        assert resolved is not None
        assert resolved.regret == 400.0  # 500 - 100
        assert resolved.was_optimal is False

    def test_analytics(self):
        engine = CounterfactualEngine()
        for i in range(5):
            engine.open_record(f"a-{i}", "BTC", "buy", True, 100.0, 10.0)
            engine.resolve(f"a-{i}", exit_price=110.0, actual_pnl=100.0)

        analytics = engine.get_analytics()
        assert analytics["total_resolved"] == 5
        assert analytics["total_actual_pnl"] == 500.0

    def test_scenario_comparison(self):
        engine = CounterfactualEngine()
        for i in range(3):
            engine.open_record(f"sc-{i}", "BTC", "buy", True, 100.0, 10.0)
            engine.resolve(f"sc-{i}", exit_price=110.0, actual_pnl=100.0)

        comparison = engine.get_scenario_comparison()
        assert "actual" in comparison
        assert "half_size" in comparison
        assert comparison["actual"]["avg_pnl"] == 100.0
        assert comparison["half_size"]["avg_pnl"] == 50.0


class TestOrchestratorIntegration:
    """Test the full orchestrator with journal + counterfactual wired in."""

    def test_evaluate_records_in_journal(self):
        from src.intelligence.orchestrator import AdaptiveIntelligenceOrchestrator

        orch = AdaptiveIntelligenceOrchestrator(min_trade_score=55)
        df = _trending_df()

        decision = orch.evaluate_signal(
            strategy_name="momentum",
            signal_confidence=0.85,
            symbol="BTC",
            side="buy",
            indicators={"rsi": 64, "rr": 2.5, "close": 100.0},
            df=df,
            portfolio_context={"correlation_risk": 0.2},
            execution_context={"spread_pct": 0.001, "qty": 5.0},
        )

        assert orch.journal.count == 1
        assert orch.counterfactual.count == 1
        assert hasattr(decision, "decision_id")

    def test_record_outcome_updates_journal_and_counterfactual(self):
        from src.intelligence.orchestrator import AdaptiveIntelligenceOrchestrator

        orch = AdaptiveIntelligenceOrchestrator(min_trade_score=55)
        df = _trending_df()

        decision = orch.evaluate_signal(
            strategy_name="momentum",
            signal_confidence=0.85,
            symbol="BTC",
            side="buy",
            indicators={"rsi": 64, "rr": 2.5, "close": 100.0},
            df=df,
            execution_context={"qty": 5.0},
        )

        decision_id = decision.decision_id  # type: ignore[attr-defined]
        orch.record_outcome(
            prediction_correct=True,
            decision_id=decision_id,
            pnl=200.0,
            pnl_pct=2.0,
            exit_price=110.0,
            bars_held=12,
            exit_reason="take_profit",
        )

        # Journal should have outcome
        records = orch.journal.query(symbol="BTC")
        assert records[0].outcome_pnl == 200.0

        # Counterfactual should be resolved
        analytics = orch.get_counterfactual_analytics()
        assert analytics["total_resolved"] == 1

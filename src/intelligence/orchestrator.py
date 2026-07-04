"""Top-level coordinator for adaptive intelligence decisions."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from src.intelligence.allocator.capital_allocator import CapitalAllocator
from src.intelligence.counterfactual.engine import CounterfactualEngine
from src.intelligence.drift.monitor import DriftMonitor
from src.intelligence.explainability.explainer import DecisionExplainer
from src.intelligence.journal.decision_journal import DecisionJournal, DecisionRecord
from src.intelligence.market_state.engine import MarketStateEngine, MarketFingerprint
from src.intelligence.meta_strategy.engine import MetaStrategyEngine
from src.intelligence.models import IntelligenceDecision
from src.intelligence.regime.classifier import RegimeClassifier
from src.intelligence.routing.strategy_router import StrategyRouter
from src.intelligence.scoring.trade_quality import TradeQualityEngine


class AdaptiveIntelligenceOrchestrator:
    """Single entrypoint for strategy routing + quality scoring + allocation.

    Integrates:
      - Market State Engine (full fingerprint)
      - Regime Classifier (backward-compat MarketState)
      - Meta Strategy Engine (strategy ranking)
      - Drift Monitor (concept drift detection)
      - Strategy Router (enable/disable per regime)
      - Trade Quality Scorer (composite score)
      - Capital Allocator (dynamic sizing)
      - Decision Explainer (human-readable reasons)
      - Decision Journal (full audit trail)
      - Counterfactual Engine (what-if analysis)
    """

    def __init__(
        self,
        min_trade_score: float = 70.0,
        drift_window: int = 200,
        drift_min_samples: int = 50,
        drift_alert_drop: float = 0.12,
    ):
        self.classifier = RegimeClassifier()
        self.market_state_engine = MarketStateEngine()
        self.meta_strategy = MetaStrategyEngine()
        self.drift_monitor = DriftMonitor(
            window=drift_window,
            min_samples=drift_min_samples,
            alert_drop=drift_alert_drop,
        )
        self.scorer = TradeQualityEngine(threshold=min_trade_score)
        self.allocator = CapitalAllocator()
        self.router = StrategyRouter()
        self.explainer = DecisionExplainer()
        self.journal = DecisionJournal()
        self.counterfactual = CounterfactualEngine()

    def evaluate_signal(
        self,
        strategy_name: str,
        signal_confidence: float,
        symbol: str = "",
        side: str = "buy",
        indicators: Optional[dict] = None,
        df: Optional[pd.DataFrame] = None,
        market_context: Optional[dict] = None,
        portfolio_context: Optional[dict] = None,
        execution_context: Optional[dict] = None,
    ) -> IntelligenceDecision:
        # 1. Market State (rich fingerprint + backward-compat regime)
        fingerprint = self.market_state_engine.compute(df=df, context=market_context)
        market_state = self.classifier.classify(df=df, context=market_context)

        # 2. Meta strategy ranking (inform routing decisions)
        rankings = self.meta_strategy.rank(fingerprint)

        # 3. Drift detection
        drift_state = self.drift_monitor.get_state()

        # 4. Strategy routing (is this strategy suitable for the current regime?)
        routing = self.router.evaluate(
            strategy_name=strategy_name,
            market_state=market_state,
            drift_state=drift_state,
        )

        # 5. Trade quality scoring
        quality = self.scorer.score(
            strategy_name=strategy_name,
            signal_confidence=signal_confidence,
            market_state=market_state,
            indicators=indicators,
            portfolio_context=portfolio_context,
            execution_context=execution_context,
        )

        # 6. Capital allocation
        allocation = self.allocator.allocate(
            market_state=market_state,
            drift_state=drift_state,
            portfolio_context=portfolio_context,
        )

        # 7. Final decision
        accepted = routing.enabled and quality.accepted
        adjusted_confidence = min(1.0, max(0.0, signal_confidence * (0.6 + 0.4 * (quality.score / 100.0))))
        qty_multiplier = allocation.qty_multiplier if accepted else 0.0

        # 8. Explanation
        explanation = self.explainer.build(
            accepted=accepted,
            market_state=market_state,
            routing=routing,
            quality=quality,
            allocation=allocation,
        )

        # 9. Record in Decision Journal
        decision_id = str(uuid.uuid4())
        portfolio_ctx = portfolio_context or {}
        journal_record = DecisionRecord(
            decision_id=decision_id,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            strategy=strategy_name,
            side=side,
            accepted=accepted,
            signal_confidence=signal_confidence,
            indicators=indicators or {},
            regime=market_state.overall_regime,
            trend=market_state.trend,
            volatility=market_state.volatility,
            stress=market_state.stress,
            liquidity=market_state.liquidity,
            trade_score=quality.score,
            allocation_multiplier=qty_multiplier,
            explanation=explanation,
            portfolio_value=portfolio_ctx.get("portfolio_value", 0.0),
            open_positions=portfolio_ctx.get("open_positions", 0),
            daily_pnl=portfolio_ctx.get("daily_pnl", 0.0),
            correlation_risk=portfolio_ctx.get("correlation_risk", 0.0),
        )
        self.journal.record(journal_record)

        # 10. Open counterfactual record
        entry_price = (indicators or {}).get("close", 0.0)
        qty_hint = (execution_context or {}).get("qty", 1.0)
        self.counterfactual.open_record(
            decision_id=decision_id,
            symbol=symbol,
            side=side,
            accepted=accepted,
            entry_price=entry_price,
            qty=qty_hint,
        )

        decision = IntelligenceDecision(
            accepted=accepted,
            final_score=quality.score,
            adjusted_confidence=adjusted_confidence,
            qty_multiplier=qty_multiplier,
            explanation=explanation,
            market_state=market_state,
            drift_state=drift_state,
            quality=quality,
            routing=routing,
            allocation=allocation,
        )
        # Attach IDs for downstream tracing
        decision.decision_id = decision_id  # type: ignore[attr-defined]
        decision.fingerprint = fingerprint  # type: ignore[attr-defined]
        decision.meta_rankings = rankings  # type: ignore[attr-defined]
        return decision

    def record_outcome(
        self,
        prediction_correct: bool,
        decision_id: Optional[str] = None,
        pnl: float = 0.0,
        pnl_pct: float = 0.0,
        exit_price: float = 0.0,
        bars_held: int = 0,
        exit_reason: str = "",
    ) -> None:
        """Record trade outcome for drift, journal, and counterfactual tracking."""
        self.drift_monitor.record_prediction(prediction_correct)

        if decision_id:
            self.journal.update_outcome(
                decision_id=decision_id,
                pnl=pnl,
                pnl_pct=pnl_pct,
                bars_held=bars_held,
                exit_reason=exit_reason,
            )
            if exit_price > 0:
                self.counterfactual.resolve(
                    decision_id=decision_id,
                    exit_price=exit_price,
                    actual_pnl=pnl,
                )

        # Feed outcome to meta strategy performance tracker
        # (requires knowing which strategy produced the outcome)

    def get_journal_analytics(self, **kwargs: Any) -> dict:
        return self.journal.get_analytics(**kwargs)

    def get_counterfactual_analytics(self, **kwargs: Any) -> dict:
        return self.counterfactual.get_analytics(**kwargs)

    def get_regime_breakdown(self) -> dict:
        return self.journal.get_regime_breakdown()

    def get_scenario_comparison(self) -> dict:
        return self.counterfactual.get_scenario_comparison()


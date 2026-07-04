"""
Decision Journal — Records every trade decision (accepted + rejected) with full context.

Captures:
  - Market state at decision time
  - Signal indicators
  - ML probabilities
  - Risk state
  - Portfolio state
  - Intelligence score/explanation
  - Outcome (filled later)

Supports querying for post-hoc analysis:
  "Show all losing BTC trades during high volatility when ML confidence > 80%"
"""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class DecisionRecord:
    """Complete record of a single trade decision."""
    decision_id: str
    timestamp: datetime
    symbol: str
    strategy: str
    side: str  # "buy" | "sell" | "hold"
    accepted: bool

    # Signal context
    signal_confidence: float
    indicators: dict[str, Any] = field(default_factory=dict)

    # Market state
    regime: str = ""
    trend: str = ""
    volatility: str = ""
    stress: str = ""
    liquidity: str = ""
    momentum: str = ""

    # Intelligence layer
    trade_score: float = 0.0
    routing_reason: str = ""
    allocation_multiplier: float = 1.0
    explanation: str = ""

    # Risk/Portfolio state
    portfolio_value: float = 0.0
    open_positions: int = 0
    daily_pnl: float = 0.0
    correlation_risk: float = 0.0

    # Execution (filled later)
    executed_qty: float = 0.0
    executed_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Outcome (filled later when trade closes)
    outcome_pnl: Optional[float] = None
    outcome_pnl_pct: Optional[float] = None
    outcome_bars_held: Optional[int] = None
    outcome_exit_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


class DecisionJournal:
    """Thread-safe journal that stores and queries decision records."""

    def __init__(self, max_records: int = 5000):
        self._records: deque[DecisionRecord] = deque(maxlen=max_records)
        self._index_by_id: dict[str, DecisionRecord] = {}
        self._lock = threading.Lock()

    def record(self, decision: DecisionRecord) -> None:
        with self._lock:
            self._records.append(decision)
            self._index_by_id[decision.decision_id] = decision
            # Trim index if it grows beyond deque
            if len(self._index_by_id) > len(self._records) * 2:
                valid_ids = {r.decision_id for r in self._records}
                self._index_by_id = {k: v for k, v in self._index_by_id.items() if k in valid_ids}

    def update_outcome(
        self,
        decision_id: str,
        pnl: float,
        pnl_pct: float,
        bars_held: int = 0,
        exit_reason: str = "",
    ) -> bool:
        with self._lock:
            record = self._index_by_id.get(decision_id)
            if record:
                record.outcome_pnl = pnl
                record.outcome_pnl_pct = pnl_pct
                record.outcome_bars_held = bars_held
                record.outcome_exit_reason = exit_reason
                return True
            return False

    def query(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        accepted: Optional[bool] = None,
        regime: Optional[str] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        volatility: Optional[str] = None,
        stress: Optional[str] = None,
        outcome_positive: Optional[bool] = None,
        limit: int = 100,
    ) -> list[DecisionRecord]:
        """Query journal with flexible filters."""
        with self._lock:
            results = []
            for record in reversed(self._records):
                if symbol and record.symbol != symbol:
                    continue
                if strategy and record.strategy != strategy:
                    continue
                if accepted is not None and record.accepted != accepted:
                    continue
                if regime and regime not in record.regime:
                    continue
                if min_confidence is not None and record.signal_confidence < min_confidence:
                    continue
                if max_confidence is not None and record.signal_confidence > max_confidence:
                    continue
                if min_score is not None and record.trade_score < min_score:
                    continue
                if max_score is not None and record.trade_score > max_score:
                    continue
                if volatility and record.volatility != volatility:
                    continue
                if stress and record.stress != stress:
                    continue
                if outcome_positive is not None:
                    if record.outcome_pnl is None:
                        continue
                    if outcome_positive and record.outcome_pnl <= 0:
                        continue
                    if not outcome_positive and record.outcome_pnl > 0:
                        continue

                results.append(record)
                if len(results) >= limit:
                    break
            return results

    def get_analytics(self, symbol: Optional[str] = None, strategy: Optional[str] = None) -> dict[str, Any]:
        """Compute analytics over journal entries."""
        records = self.query(symbol=symbol, strategy=strategy, limit=5000)
        if not records:
            return {"total": 0}

        accepted = [r for r in records if r.accepted]
        rejected = [r for r in records if not r.accepted]
        with_outcome = [r for r in accepted if r.outcome_pnl is not None]

        wins = [r for r in with_outcome if r.outcome_pnl > 0]
        losses = [r for r in with_outcome if r.outcome_pnl < 0]

        # Score distribution for accepted vs outcomes
        accepted_scores = [r.trade_score for r in accepted]
        winning_scores = [r.trade_score for r in wins]
        losing_scores = [r.trade_score for r in losses]

        return {
            "total": len(records),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "acceptance_rate": len(accepted) / len(records) if records else 0,
            "with_outcome": len(with_outcome),
            "win_rate": len(wins) / len(with_outcome) if with_outcome else 0,
            "avg_win_pnl": sum(r.outcome_pnl for r in wins) / len(wins) if wins else 0,
            "avg_loss_pnl": sum(r.outcome_pnl for r in losses) / len(losses) if losses else 0,
            "avg_accepted_score": sum(accepted_scores) / len(accepted_scores) if accepted_scores else 0,
            "avg_winning_score": sum(winning_scores) / len(winning_scores) if winning_scores else 0,
            "avg_losing_score": sum(losing_scores) / len(losing_scores) if losing_scores else 0,
            "profit_factor": (
                abs(sum(r.outcome_pnl for r in wins)) / abs(sum(r.outcome_pnl for r in losses))
                if losses and sum(r.outcome_pnl for r in losses) != 0 else float("inf")
            ),
        }

    def get_regime_breakdown(self) -> dict[str, dict]:
        """Break down performance by regime."""
        with self._lock:
            regimes: dict[str, list[DecisionRecord]] = {}
            for r in self._records:
                if r.outcome_pnl is None:
                    continue
                regime = r.regime or "UNKNOWN"
                regimes.setdefault(regime, []).append(r)

        breakdown = {}
        for regime, records in regimes.items():
            wins = [r for r in records if r.outcome_pnl > 0]
            losses = [r for r in records if r.outcome_pnl < 0]
            total_pnl = sum(r.outcome_pnl for r in records)
            breakdown[regime] = {
                "trades": len(records),
                "win_rate": len(wins) / len(records) if records else 0,
                "total_pnl": total_pnl,
                "avg_pnl": total_pnl / len(records) if records else 0,
            }
        return breakdown

    @property
    def count(self) -> int:
        return len(self._records)

    def export_json(self, limit: int = 500) -> str:
        with self._lock:
            records = list(self._records)[-limit:]
        return json.dumps([r.to_dict() for r in records], indent=2)

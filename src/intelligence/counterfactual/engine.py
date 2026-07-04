"""
Counterfactual Engine — Answers "what would have happened if...?"

For every trade decision, tracks alternative scenarios:
  - What if we didn't trade? (no-trade baseline)
  - What if we used half size?
  - What if we used an alternative strategy?

Reports opportunity cost and regret to improve future decision-making.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class Counterfactual:
    """A single what-if scenario."""
    scenario: str  # "no_trade", "half_size", "alternative:mean_reversion"
    hypothetical_pnl: float
    hypothetical_pnl_pct: float = 0.0


@dataclass
class CounterfactualRecord:
    """Complete counterfactual analysis for one decision."""
    decision_id: str
    symbol: str
    timestamp: datetime
    side: str
    accepted: bool
    actual_pnl: Optional[float] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    qty: float = 0.0
    scenarios: list[Counterfactual] = field(default_factory=list)

    @property
    def best_scenario(self) -> Optional[Counterfactual]:
        if not self.scenarios:
            return None
        return max(self.scenarios, key=lambda s: s.hypothetical_pnl)

    @property
    def regret(self) -> float:
        """Opportunity cost: best counterfactual PnL minus actual PnL."""
        best = self.best_scenario
        if best is None or self.actual_pnl is None:
            return 0.0
        return max(0.0, best.hypothetical_pnl - self.actual_pnl)

    @property
    def was_optimal(self) -> bool:
        """Whether the actual decision was the best option."""
        if self.actual_pnl is None:
            return True
        best = self.best_scenario
        if best is None:
            return True
        return self.actual_pnl >= best.hypothetical_pnl

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "side": self.side,
            "accepted": self.accepted,
            "actual_pnl": self.actual_pnl,
            "entry_price": self.entry_price,
            "qty": self.qty,
            "regret": self.regret,
            "was_optimal": self.was_optimal,
            "scenarios": [
                {"scenario": s.scenario, "pnl": s.hypothetical_pnl}
                for s in self.scenarios
            ],
        }


class CounterfactualEngine:
    """Track and analyze counterfactual trade scenarios."""

    def __init__(self, max_records: int = 2000):
        self._records: deque[CounterfactualRecord] = deque(maxlen=max_records)
        self._index: dict[str, CounterfactualRecord] = {}
        self._lock = threading.Lock()

    def open_record(
        self,
        decision_id: str,
        symbol: str,
        side: str,
        accepted: bool,
        entry_price: float,
        qty: float,
        timestamp: Optional[datetime] = None,
    ) -> CounterfactualRecord:
        """Create a new counterfactual record when a decision is made."""
        record = CounterfactualRecord(
            decision_id=decision_id,
            symbol=symbol,
            timestamp=timestamp or datetime.now(timezone.utc),
            side=side,
            accepted=accepted,
            entry_price=entry_price,
            qty=qty,
        )

        # Auto-generate the no-trade scenario
        record.scenarios.append(Counterfactual(scenario="no_trade", hypothetical_pnl=0.0))

        with self._lock:
            self._records.append(record)
            self._index[decision_id] = record
        return record

    def resolve(
        self,
        decision_id: str,
        exit_price: float,
        actual_pnl: float,
    ) -> Optional[CounterfactualRecord]:
        """
        Close out a counterfactual record with the actual outcome.
        Automatically computes alternative scenario PnLs.
        """
        with self._lock:
            record = self._index.get(decision_id)
            if not record:
                return None

            record.exit_price = exit_price
            record.actual_pnl = actual_pnl

            # Update no-trade scenario (it stays at 0)
            # Compute half-size scenario
            if record.accepted and record.qty > 0:
                half_pnl = actual_pnl * 0.5
                record.scenarios.append(
                    Counterfactual(scenario="half_size", hypothetical_pnl=half_pnl)
                )

                # Compute double-size scenario (aggressive alternative)
                double_pnl = actual_pnl * 2.0
                record.scenarios.append(
                    Counterfactual(scenario="double_size", hypothetical_pnl=double_pnl)
                )

            # For rejected trades, compute what-if-accepted
            if not record.accepted and record.entry_price > 0 and exit_price > 0:
                if record.side == "buy":
                    hyp_pnl = (exit_price - record.entry_price) * record.qty
                else:
                    hyp_pnl = (record.entry_price - exit_price) * record.qty
                record.scenarios.append(
                    Counterfactual(scenario="if_accepted", hypothetical_pnl=hyp_pnl)
                )

            return record

    def add_alternative_scenario(
        self,
        decision_id: str,
        scenario_name: str,
        hypothetical_pnl: float,
    ) -> bool:
        """Add a custom counterfactual (e.g., alternative strategy result)."""
        with self._lock:
            record = self._index.get(decision_id)
            if record:
                record.scenarios.append(
                    Counterfactual(scenario=scenario_name, hypothetical_pnl=hypothetical_pnl)
                )
                return True
            return False

    def get_analytics(self, symbol: Optional[str] = None) -> dict[str, Any]:
        """Compute aggregate counterfactual analytics."""
        with self._lock:
            records = [r for r in self._records if r.actual_pnl is not None]
            if symbol:
                records = [r for r in records if r.symbol == symbol]

        if not records:
            return {"total_resolved": 0}

        total_regret = sum(r.regret for r in records)
        optimal_count = sum(1 for r in records if r.was_optimal)
        total_actual = sum(r.actual_pnl for r in records if r.actual_pnl is not None)

        # Best missed opportunity (highest regret record)
        worst_miss = max(records, key=lambda r: r.regret) if records else None

        return {
            "total_resolved": len(records),
            "optimal_decisions_pct": optimal_count / len(records) if records else 0,
            "total_regret": total_regret,
            "avg_regret": total_regret / len(records) if records else 0,
            "total_actual_pnl": total_actual,
            "worst_miss": worst_miss.to_dict() if worst_miss and worst_miss.regret > 0 else None,
        }

    def get_scenario_comparison(self) -> dict[str, dict]:
        """Compare average PnL across all scenario types."""
        with self._lock:
            records = [r for r in self._records if r.actual_pnl is not None]

        scenario_pnls: dict[str, list[float]] = {"actual": []}
        for r in records:
            scenario_pnls["actual"].append(r.actual_pnl)
            for s in r.scenarios:
                scenario_pnls.setdefault(s.scenario, []).append(s.hypothetical_pnl)

        comparison = {}
        for name, pnls in scenario_pnls.items():
            if pnls:
                comparison[name] = {
                    "avg_pnl": sum(pnls) / len(pnls),
                    "total_pnl": sum(pnls),
                    "count": len(pnls),
                }
        return comparison

    @property
    def count(self) -> int:
        return len(self._records)

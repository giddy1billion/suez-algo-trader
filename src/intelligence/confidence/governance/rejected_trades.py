"""Rejected trade tracking — counterfactual learning from missed opportunities.

Key insight: "The signal was rejected at confidence=0.62. The stock went up 4%
in 2 hours. Were we too strict?" This module enables threshold optimization
by recording what WOULD have happened with rejected signals.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RejectedTradeRecord:
    """A rejected signal + what would have happened."""

    lineage_id: str
    symbol: str
    direction: str  # "BUY" | "SELL"
    confidence: float
    rejection_reason: str
    rejected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Counterfactual outcome (populated later)
    price_at_rejection: float = 0.0
    price_after_1h: Optional[float] = None
    price_after_4h: Optional[float] = None
    price_after_1d: Optional[float] = None
    would_have_won: Optional[bool] = None
    counterfactual_pnl_pct: Optional[float] = None

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "lineage_id": self.lineage_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "rejection_reason": self.rejection_reason,
            "rejected_at": self.rejected_at.isoformat(),
            "price_at_rejection": self.price_at_rejection,
            "price_after_1h": self.price_after_1h,
            "price_after_4h": self.price_after_4h,
            "price_after_1d": self.price_after_1d,
            "would_have_won": self.would_have_won,
            "counterfactual_pnl_pct": self.counterfactual_pnl_pct,
        }


@dataclass
class ThresholdAnalysis:
    """Analysis of rejection outcomes by confidence bin."""

    confidence_bin: str  # e.g., "0.55-0.60"
    bin_lower: float
    bin_upper: float
    total_rejections: int
    outcomes_recorded: int
    would_have_won: int
    win_rate: float  # % that would have profited
    avg_counterfactual_pnl: float  # Average P&L if we had traded
    is_overtight: bool  # True if win_rate > 50% (threshold too strict)

    def to_dict(self) -> dict:
        """Serialize."""
        return {
            "confidence_bin": self.confidence_bin,
            "total_rejections": self.total_rejections,
            "outcomes_recorded": self.outcomes_recorded,
            "would_have_won": self.would_have_won,
            "win_rate": round(self.win_rate, 4),
            "avg_counterfactual_pnl": round(self.avg_counterfactual_pnl, 4),
            "is_overtight": self.is_overtight,
        }


class RejectedTradeTracker:
    """Tracks rejected signals and their counterfactual outcomes.

    Thread-safe. Maintains a rolling window of the last N rejections
    to enable threshold optimization without unbounded memory growth.
    """

    def __init__(self, max_records: int = 1000) -> None:
        """Initialize tracker.

        Args:
            max_records: Maximum rejection records to retain (rolling window)
        """
        self._lock = threading.Lock()
        self._records: deque[RejectedTradeRecord] = deque(maxlen=max_records)
        self._by_lineage: dict[str, RejectedTradeRecord] = {}
        self._max_records = max_records

    def record_rejection(
        self,
        lineage_id: str,
        symbol: str,
        direction: str,
        confidence: float,
        rejection_reason: str,
        price_at_rejection: float = 0.0,
    ) -> RejectedTradeRecord:
        """Record a rejected trade signal.

        Args:
            lineage_id: Decision lineage ID
            symbol: Trading symbol
            direction: "BUY" or "SELL"
            confidence: Model confidence at rejection time
            rejection_reason: Why it was rejected
            price_at_rejection: Price when rejection occurred

        Returns:
            The created RejectedTradeRecord
        """
        record = RejectedTradeRecord(
            lineage_id=lineage_id,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            rejection_reason=rejection_reason,
            price_at_rejection=price_at_rejection,
        )

        with self._lock:
            # If deque is full, remove oldest from index
            if len(self._records) >= self._max_records:
                oldest = self._records[0]
                self._by_lineage.pop(oldest.lineage_id, None)

            self._records.append(record)
            self._by_lineage[lineage_id] = record

        return record

    def record_outcome(
        self,
        lineage_id: str,
        price_after_1h: Optional[float] = None,
        price_after_4h: Optional[float] = None,
        price_after_1d: Optional[float] = None,
    ) -> bool:
        """Populate counterfactual outcome for a previously rejected trade.

        Args:
            lineage_id: The lineage ID of the rejected trade
            price_after_1h: Price 1 hour after rejection
            price_after_4h: Price 4 hours after rejection
            price_after_1d: Price 1 day after rejection

        Returns:
            True if record was found and updated
        """
        with self._lock:
            record = self._by_lineage.get(lineage_id)
            if record is None:
                logger.warning(f"Rejected trade record not found: {lineage_id}")
                return False

            record.price_after_1h = price_after_1h
            record.price_after_4h = price_after_4h
            record.price_after_1d = price_after_1d

            # Compute counterfactual P&L using best available horizon
            if record.price_at_rejection > 0:
                # Use 4h as primary evaluation horizon, fall back to 1h or 1d
                eval_price = price_after_4h or price_after_1h or price_after_1d
                if eval_price is not None:
                    if record.direction == "BUY":
                        pnl_pct = (eval_price - record.price_at_rejection) / record.price_at_rejection
                    else:
                        pnl_pct = (record.price_at_rejection - eval_price) / record.price_at_rejection

                    record.counterfactual_pnl_pct = pnl_pct
                    record.would_have_won = pnl_pct > 0

            return True

    def compute_threshold_analysis(self, bin_width: float = 0.05) -> list[ThresholdAnalysis]:
        """Analyze rejection outcomes by confidence bin.

        For each confidence bin, computes what percentage of rejected
        trades would have been profitable.

        Args:
            bin_width: Width of each confidence bin (default 5%)

        Returns:
            List of ThresholdAnalysis per bin, sorted by confidence
        """
        with self._lock:
            records = list(self._records)

        if not records:
            return []

        # Create bins from 0.0 to 1.0
        bins: dict[str, list[RejectedTradeRecord]] = {}
        bin_edges = np.arange(0.0, 1.0, bin_width)

        for edge in bin_edges:
            bin_key = f"{edge:.2f}-{edge + bin_width:.2f}"
            bins[bin_key] = []

        for record in records:
            bin_idx = int(record.confidence / bin_width)
            bin_lower = bin_idx * bin_width
            bin_key = f"{bin_lower:.2f}-{bin_lower + bin_width:.2f}"
            if bin_key in bins:
                bins[bin_key].append(record)

        results = []
        for bin_key, bin_records in sorted(bins.items()):
            if not bin_records:
                continue

            parts = bin_key.split("-")
            bin_lower = float(parts[0])
            bin_upper = float(parts[1])

            outcomes = [r for r in bin_records if r.would_have_won is not None]
            won_count = sum(1 for r in outcomes if r.would_have_won)
            win_rate = won_count / len(outcomes) if outcomes else 0.0

            pnls = [r.counterfactual_pnl_pct for r in outcomes if r.counterfactual_pnl_pct is not None]
            avg_pnl = float(np.mean(pnls)) if pnls else 0.0

            results.append(
                ThresholdAnalysis(
                    confidence_bin=bin_key,
                    bin_lower=bin_lower,
                    bin_upper=bin_upper,
                    total_rejections=len(bin_records),
                    outcomes_recorded=len(outcomes),
                    would_have_won=won_count,
                    win_rate=win_rate,
                    avg_counterfactual_pnl=avg_pnl,
                    is_overtight=win_rate > 0.5 and len(outcomes) >= 5,
                )
            )

        return results

    def get_overtight_thresholds(self) -> list[ThresholdAnalysis]:
        """Return confidence bins where the threshold appears too strict.

        A threshold is 'overtight' if rejected trades at that confidence
        level would have won more than 50% of the time (with sufficient sample).

        Returns:
            List of ThresholdAnalysis entries where is_overtight=True
        """
        analysis = self.compute_threshold_analysis()
        return [a for a in analysis if a.is_overtight]

    @property
    def total_rejections(self) -> int:
        """Total rejections currently tracked."""
        with self._lock:
            return len(self._records)

    @property
    def outcomes_recorded(self) -> int:
        """Number of rejections that have outcome data."""
        with self._lock:
            return sum(1 for r in self._records if r.would_have_won is not None)

    def get_summary(self) -> dict:
        """Get summary statistics."""
        with self._lock:
            records = list(self._records)

        if not records:
            return {"total_rejections": 0, "outcomes_recorded": 0}

        outcomes = [r for r in records if r.would_have_won is not None]
        won = sum(1 for r in outcomes if r.would_have_won)

        return {
            "total_rejections": len(records),
            "outcomes_recorded": len(outcomes),
            "overall_would_have_won_rate": round(won / len(outcomes), 4) if outcomes else 0.0,
            "avg_confidence_at_rejection": round(
                float(np.mean([r.confidence for r in records])), 4
            ),
            "top_rejection_reasons": self._top_reasons(records),
        }

    @staticmethod
    def _top_reasons(records: list[RejectedTradeRecord], top_n: int = 5) -> list[dict]:
        """Get most common rejection reasons."""
        reason_counts: dict[str, int] = {}
        for r in records:
            reason_counts[r.rejection_reason] = reason_counts.get(r.rejection_reason, 0) + 1

        sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"reason": r, "count": c} for r, c in sorted_reasons[:top_n]]

"""Decision lineage — immutable tracing of every decision artifact.

Every decision gets a unique ID that links to all downstream artifacts:
prediction → confidence gate → risk decision → order → fill → P&L → retraining.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DecisionLineage:
    """Immutable lineage record linking all artifacts of a decision."""

    lineage_id: str  # D-20260705-000234 format
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Upstream references
    prediction_id: str = ""
    model_version: str = ""
    feature_snapshot_hash: str = ""
    dataset_version: str = ""
    calibration_table_version: str = ""

    # Decision references
    contract_id: str = ""  # DecisionContract.contract_id
    confidence_gate_result: str = ""  # "pass" | "reject"
    risk_decision: str = ""  # "approved" | "rejected"

    # Downstream references (populated as they happen)
    order_id: Optional[str] = None
    trade_id: Optional[str] = None
    fill_price: Optional[float] = None
    pnl: Optional[float] = None
    outcome: Optional[str] = None  # "win" | "loss" | "timeout" | "stopped"

    # Model lifecycle (populated much later)
    included_in_dataset: Optional[str] = None  # Which retraining dataset
    contributed_to_model: Optional[str] = None  # Which model version

    def to_dict(self) -> dict:
        """Serialize for audit storage."""
        return {
            "lineage_id": self.lineage_id,
            "created_at": self.created_at.isoformat(),
            "prediction_id": self.prediction_id,
            "model_version": self.model_version,
            "feature_snapshot_hash": self.feature_snapshot_hash,
            "dataset_version": self.dataset_version,
            "calibration_table_version": self.calibration_table_version,
            "contract_id": self.contract_id,
            "confidence_gate_result": self.confidence_gate_result,
            "risk_decision": self.risk_decision,
            "order_id": self.order_id,
            "trade_id": self.trade_id,
            "fill_price": self.fill_price,
            "pnl": self.pnl,
            "outcome": self.outcome,
            "included_in_dataset": self.included_in_dataset,
            "contributed_to_model": self.contributed_to_model,
        }


class LineageRegistry:
    """Thread-safe registry for decision lineage records.

    Generates monotonically increasing lineage IDs in format D-YYYYMMDD-NNNNNN.
    Supports lookup by any reference field and downstream updates.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, DecisionLineage] = {}
        self._daily_counter: int = 0
        self._current_date: str = ""

        # Secondary indexes for fast lookup
        self._by_order_id: dict[str, str] = {}
        self._by_trade_id: dict[str, str] = {}
        self._by_prediction_id: dict[str, str] = {}
        self._by_model_version: dict[str, list[str]] = {}

    def create(
        self,
        prediction_id: str = "",
        model_version: str = "",
        feature_snapshot_hash: str = "",
        dataset_version: str = "",
        calibration_table_version: str = "",
        contract_id: str = "",
        confidence_gate_result: str = "",
        risk_decision: str = "",
    ) -> DecisionLineage:
        """Create a new lineage record with a unique ID.

        Returns:
            New DecisionLineage with generated lineage_id
        """
        with self._lock:
            lineage_id = self._generate_id()

            record = DecisionLineage(
                lineage_id=lineage_id,
                prediction_id=prediction_id,
                model_version=model_version,
                feature_snapshot_hash=feature_snapshot_hash,
                dataset_version=dataset_version,
                calibration_table_version=calibration_table_version,
                contract_id=contract_id,
                confidence_gate_result=confidence_gate_result,
                risk_decision=risk_decision,
            )

            self._records[lineage_id] = record

            # Update indexes
            if prediction_id:
                self._by_prediction_id[prediction_id] = lineage_id
            if model_version:
                self._by_model_version.setdefault(model_version, []).append(lineage_id)

            return record

    def get(self, lineage_id: str) -> Optional[DecisionLineage]:
        """Retrieve a lineage record by ID."""
        with self._lock:
            return self._records.get(lineage_id)

    def lookup_by_order_id(self, order_id: str) -> Optional[DecisionLineage]:
        """Find lineage record by order ID."""
        with self._lock:
            lineage_id = self._by_order_id.get(order_id)
            if lineage_id:
                return self._records.get(lineage_id)
            return None

    def lookup_by_trade_id(self, trade_id: str) -> Optional[DecisionLineage]:
        """Find lineage record by trade ID."""
        with self._lock:
            lineage_id = self._by_trade_id.get(trade_id)
            if lineage_id:
                return self._records.get(lineage_id)
            return None

    def lookup_by_prediction_id(self, prediction_id: str) -> Optional[DecisionLineage]:
        """Find lineage record by prediction ID."""
        with self._lock:
            lineage_id = self._by_prediction_id.get(prediction_id)
            if lineage_id:
                return self._records.get(lineage_id)
            return None

    def lookup_by_model_version(self, model_version: str) -> list[DecisionLineage]:
        """Find all lineage records for a given model version."""
        with self._lock:
            lineage_ids = self._by_model_version.get(model_version, [])
            return [self._records[lid] for lid in lineage_ids if lid in self._records]

    def update_downstream(
        self,
        lineage_id: str,
        order_id: Optional[str] = None,
        trade_id: Optional[str] = None,
        fill_price: Optional[float] = None,
        pnl: Optional[float] = None,
        outcome: Optional[str] = None,
        included_in_dataset: Optional[str] = None,
        contributed_to_model: Optional[str] = None,
    ) -> bool:
        """Update downstream references after execution.

        Args:
            lineage_id: The lineage record to update
            order_id: Broker order ID
            trade_id: Executed trade ID
            fill_price: Actual fill price
            pnl: Realized P&L
            outcome: "win" | "loss" | "timeout" | "stopped"
            included_in_dataset: Retraining dataset reference
            contributed_to_model: Model version this fed into

        Returns:
            True if record was found and updated
        """
        with self._lock:
            record = self._records.get(lineage_id)
            if record is None:
                logger.warning(f"Lineage record not found: {lineage_id}")
                return False

            if order_id is not None:
                record.order_id = order_id
                self._by_order_id[order_id] = lineage_id

            if trade_id is not None:
                record.trade_id = trade_id
                self._by_trade_id[trade_id] = lineage_id

            if fill_price is not None:
                record.fill_price = fill_price

            if pnl is not None:
                record.pnl = pnl

            if outcome is not None:
                record.outcome = outcome

            if included_in_dataset is not None:
                record.included_in_dataset = included_in_dataset

            if contributed_to_model is not None:
                record.contributed_to_model = contributed_to_model

            return True

    def export_all(self) -> list[dict]:
        """Export all records as dicts for audit storage."""
        with self._lock:
            return [record.to_dict() for record in self._records.values()]

    @property
    def count(self) -> int:
        """Number of records in the registry."""
        with self._lock:
            return len(self._records)

    def _generate_id(self) -> str:
        """Generate monotonically increasing lineage ID: D-YYYYMMDD-NNNNNN."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")

        if today != self._current_date:
            self._current_date = today
            self._daily_counter = 0

        self._daily_counter += 1
        return f"D-{today}-{self._daily_counter:06d}"

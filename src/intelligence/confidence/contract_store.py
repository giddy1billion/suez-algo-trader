"""
Contract Store — SQLAlchemy ORM-backed persistence for immutable DecisionContracts.

Every contract produced by DecisionOrchestrator (executed, rejected, expired)
is stored permanently for:
  1. Complete audit trail
  2. Replay capability (reconstruct any historical decision)
  3. Model governance analytics
  4. Calibration analysis
  5. Regulatory compliance

The store also tracks execution outcomes against contracts, enabling:
  - Contract accuracy analysis (how often did EXECUTE decisions profit?)
  - Stage-level diagnostics (which gates are too aggressive/permissive?)
  - Temporal analysis (confidence drift over time)

Usage:
    store = ContractStore()
    store.store(contract)  # Persist any contract
    store.record_outcome(contract_id, trade_id, pnl_pct, ...)  # Link result

    # Replay
    contract_data = store.replay(contract_id)

    # Analytics
    df = store.query_contracts(symbol="BTC/USD", decision="execute", since=...)
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from src.intelligence.confidence.decision_contract import (
    DecisionContract,
    Decision,
)
from src.intelligence.confidence.orm_models import (
    ContractBase,
    CTContract,
    CTContractStage,
    CTContractOutcome,
)
from src.utils.database import create_db_engine
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ContractStore:
    """
    Persistent storage for DecisionContracts via SQLAlchemy ORM.

    Thread-safe. Write operations are serialized via lock.
    Read operations use independent sessions.

    Tables:
        ct_contracts        — Full contract snapshot (one row per contract)
        ct_contract_stages  — Individual stage assessments (N rows per contract)
        ct_contract_outcomes — Execution results linked back to contracts
    """

    def __init__(self, storage_path: str = "data_cache/contracts", database_url: str = None) -> None:
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

        if database_url:
            self._engine = create_db_engine(database_url)
        else:
            db_path = self._storage_path / "contracts.db"
            self._engine = create_db_engine(f"sqlite:///{db_path}")

        ContractBase.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        logger.info("contract_store.initialized", path=str(self._storage_path))

    # ──────────────────────────────────────────────────────────────────────
    # Write Operations
    # ──────────────────────────────────────────────────────────────────────

    def store(self, contract: DecisionContract) -> str:
        """
        Store a DecisionContract permanently.

        Called by ExecutionEngine for EVERY contract produced (not just executed ones).
        This ensures the audit trail includes rejected/vetoed decisions too.

        Returns:
            The contract_id that was stored.
        """
        full_json = json.dumps(contract.to_dict(), default=str)

        with self._write_lock:
            with self._Session() as session:
                record = CTContract(
                    contract_id=contract.contract_id,
                    symbol=contract.symbol,
                    direction=contract.direction,
                    decision=contract.decision.value,
                    final_confidence=contract.final_confidence,
                    recommendation=contract.recommendation,
                    vetoed=contract.vetoed,
                    vetoed_by=contract.vetoed_by.value if contract.vetoed_by else None,
                    veto_reason=contract.veto_reason,
                    recommended_position_pct=contract.recommended_position_pct,
                    kelly_fraction=contract.kelly_fraction,
                    risk_grade=contract.risk_grade,
                    integrity_hash=contract.integrity_hash,
                    created_at=contract.created_at,
                    valid_until=contract.valid_until,
                    model_version=contract.provenance.model_version,
                    feature_set_version=contract.provenance.feature_set_version,
                    dataset_version=contract.provenance.dataset_version,
                    walk_forward_passed=contract.provenance.walk_forward_passed,
                    monte_carlo_passed=contract.provenance.monte_carlo_passed,
                    model_health_score=contract.provenance.model_health_score,
                    full_contract_json=full_json,
                    execution_status="pending" if contract.is_executable else "rejected",
                    stored_at=datetime.now(timezone.utc),
                )
                session.merge(record)

                # Insert stage assessments
                for stage in contract.stages:
                    stage_record = CTContractStage(
                        contract_id=contract.contract_id,
                        stage_name=stage.stage,
                        score=stage.score,
                        passed=stage.passed,
                        weight=stage.weight,
                        severity=stage.severity.value,
                        veto=stage.veto,
                        veto_reason=stage.veto_reason,
                        evidence_json=json.dumps(stage.evidence, default=str),
                        warnings_json=json.dumps(stage.warnings),
                        blockers_json=json.dumps(stage.blockers),
                        evaluation_ms=stage.evaluation_ms,
                    )
                    session.merge(stage_record)

                session.commit()

        logger.debug(
            "contract_store.stored",
            contract_id=contract.contract_id,
            symbol=contract.symbol,
            decision=contract.decision.value,
        )
        return contract.contract_id

    def mark_executed(self, contract_id: str, trade_id: str) -> None:
        """Mark a contract as having been executed (order placed)."""
        with self._write_lock:
            with self._Session() as session:
                record = session.query(CTContract).filter(
                    CTContract.contract_id == contract_id
                ).first()
                if record:
                    record.execution_status = "executed"
                    record.trade_id = trade_id
                    session.commit()

    def mark_expired(self, contract_id: str) -> None:
        """Mark a contract as expired (validity window passed without execution)."""
        with self._write_lock:
            with self._Session() as session:
                record = session.query(CTContract).filter(
                    CTContract.contract_id == contract_id,
                    CTContract.execution_status == "pending",
                ).first()
                if record:
                    record.execution_status = "expired"
                    session.commit()

    def record_outcome(
        self,
        contract_id: str,
        trade_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        holding_minutes: int = 0,
        max_favorable_excursion: float = 0.0,
        max_adverse_excursion: float = 0.0,
        slippage_pct: float = 0.0,
        fees: float = 0.0,
        exit_reason: str = "",
    ) -> None:
        """
        Record the trade outcome linked to a contract.

        Called when a trade closes. Links the P&L result back to the
        original DecisionContract that authorized the trade.
        """
        actual_profitable = pnl_pct > 0

        with self._write_lock:
            with self._Session() as session:
                outcome_record = CTContractOutcome(
                    contract_id=contract_id,
                    trade_id=trade_id,
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    actual_profitable=actual_profitable,
                    holding_minutes=holding_minutes,
                    max_favorable_excursion=max_favorable_excursion,
                    max_adverse_excursion=max_adverse_excursion,
                    slippage_pct=slippage_pct,
                    fees=fees,
                    exit_reason=exit_reason,
                    closed_at=datetime.now(timezone.utc),
                    recorded_at=datetime.now(timezone.utc),
                )
                session.merge(outcome_record)
                session.commit()

        logger.info(
            "contract_store.outcome_recorded",
            contract_id=contract_id,
            trade_id=trade_id,
            pnl_pct=round(pnl_pct, 2),
            profitable=actual_profitable,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Replay — Reconstruct any historical decision
    # ──────────────────────────────────────────────────────────────────────

    def replay(self, contract_id: str) -> Optional[dict]:
        """
        Replay a contract: reconstruct the full decision context.

        Returns the complete contract data including:
        - All stage assessments with evidence
        - Provenance metadata
        - Execution outcome (if trade was placed)
        - Original integrity hash for verification

        Returns None if contract_id not found.
        """
        with self._Session() as session:
            record = session.query(CTContract).filter(
                CTContract.contract_id == contract_id
            ).first()

            if not record:
                return None

            contract_data = json.loads(record.full_contract_json)
            contract_data["_replay_metadata"] = {
                "execution_status": record.execution_status,
                "trade_id": record.trade_id,
                "replayed_at": datetime.now(timezone.utc).isoformat(),
            }

            # Attach outcome if available
            outcome = session.query(CTContractOutcome).filter(
                CTContractOutcome.contract_id == contract_id
            ).first()

            if outcome:
                contract_data["_outcome"] = {
                    "trade_id": outcome.trade_id,
                    "entry_price": outcome.entry_price,
                    "exit_price": outcome.exit_price,
                    "pnl": outcome.pnl,
                    "pnl_pct": outcome.pnl_pct,
                    "actual_profitable": outcome.actual_profitable,
                    "holding_minutes": outcome.holding_minutes,
                    "max_favorable_excursion": outcome.max_favorable_excursion,
                    "max_adverse_excursion": outcome.max_adverse_excursion,
                    "slippage_pct": outcome.slippage_pct,
                    "fees": outcome.fees,
                    "exit_reason": outcome.exit_reason,
                    "closed_at": str(outcome.closed_at) if outcome.closed_at else None,
                }

            # Attach per-stage details from structured table
            stages = session.query(CTContractStage).filter(
                CTContractStage.contract_id == contract_id
            ).order_by(CTContractStage.stage_name).all()

            if stages:
                contract_data["_stage_details"] = [
                    {
                        "stage": s.stage_name,
                        "score": s.score,
                        "passed": s.passed,
                        "weight": s.weight,
                        "severity": s.severity,
                        "veto": s.veto,
                        "veto_reason": s.veto_reason,
                        "evidence": json.loads(s.evidence_json) if s.evidence_json else {},
                        "warnings": json.loads(s.warnings_json) if s.warnings_json else [],
                        "blockers": json.loads(s.blockers_json) if s.blockers_json else [],
                        "evaluation_ms": s.evaluation_ms,
                    }
                    for s in stages
                ]

        return contract_data

    def replay_by_trade(self, trade_id: str) -> Optional[dict]:
        """Replay a contract by its linked trade_id (reverse lookup)."""
        with self._Session() as session:
            record = session.query(CTContract).filter(
                CTContract.trade_id == trade_id
            ).first()

            if not record:
                # Try outcomes table
                outcome = session.query(CTContractOutcome).filter(
                    CTContractOutcome.trade_id == trade_id
                ).first()
                if outcome:
                    return self.replay(outcome.contract_id)
                return None

            return self.replay(record.contract_id)

    # ──────────────────────────────────────────────────────────────────────
    # Query / Analytics
    # ──────────────────────────────────────────────────────────────────────

    def query_contracts(
        self,
        symbol: Optional[str] = None,
        decision: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        model_version: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Query contracts with optional filters.

        Returns list of contract summaries (not full JSON).
        """
        with self._Session() as session:
            query = session.query(CTContract)

            if symbol:
                query = query.filter(CTContract.symbol == symbol)
            if decision:
                query = query.filter(CTContract.decision == decision)
            if since:
                query = query.filter(CTContract.created_at >= since)
            if until:
                query = query.filter(CTContract.created_at <= until)
            if model_version:
                query = query.filter(CTContract.model_version == model_version)

            rows = query.order_by(CTContract.created_at.desc()).limit(limit).all()

            return [
                {
                    "contract_id": r.contract_id,
                    "symbol": r.symbol,
                    "direction": r.direction,
                    "decision": r.decision,
                    "confidence": r.final_confidence,
                    "vetoed": r.vetoed,
                    "position_pct": r.recommended_position_pct,
                    "risk_grade": r.risk_grade,
                    "model_version": r.model_version,
                    "execution_status": r.execution_status,
                    "trade_id": r.trade_id,
                    "created_at": str(r.created_at),
                }
                for r in rows
            ]

    def get_contract_accuracy(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> dict:
        """
        Compute accuracy metrics: how often did EXECUTE contracts profit?

        Returns:
            Dict with win_rate, avg_pnl, total_contracts, etc.
        """
        conditions = ["c.decision = 'execute'", "o.contract_id IS NOT NULL"]
        params: dict[str, Any] = {}

        if symbol:
            conditions.append("c.symbol = :symbol")
            params["symbol"] = symbol
        if since:
            conditions.append("c.created_at >= :since")
            params["since"] = since

        where = "WHERE " + " AND ".join(conditions)

        sql = text(f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN o.actual_profitable THEN 1 ELSE 0 END) AS wins,
                AVG(o.pnl_pct) AS avg_pnl_pct,
                AVG(c.final_confidence) AS avg_confidence,
                AVG(o.holding_minutes) AS avg_holding
            FROM ct_contracts c
            LEFT JOIN ct_contract_outcomes o ON c.contract_id = o.contract_id
            {where}
        """)

        with self._engine.connect() as conn:
            row = conn.execute(sql, params).fetchone()

        if not row or row[0] == 0:
            return {"total": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0}

        total = int(row[0])
        wins = int(row[1]) if row[1] else 0
        return {
            "total": total,
            "wins": wins,
            "win_rate": wins / total if total > 0 else 0.0,
            "avg_pnl_pct": float(row[2]) if row[2] else 0.0,
            "avg_confidence": float(row[3]) if row[3] else 0.0,
            "avg_holding_minutes": float(row[4]) if row[4] else 0.0,
        }

    def get_stage_diagnostics(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> list[dict]:
        """
        Per-stage diagnostics: average scores, pass rates, veto counts.

        Useful for identifying which stages are too aggressive or permissive.
        """
        conditions = []
        params: dict[str, Any] = {}

        if symbol:
            conditions.append("c.symbol = :symbol")
            params["symbol"] = symbol
        if since:
            conditions.append("c.created_at >= :since")
            params["since"] = since

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        join_clause = (
            f"JOIN ct_contracts c ON cs.contract_id = c.contract_id {where}"
            if conditions
            else ""
        )

        sql = text(f"""
            SELECT
                cs.stage_name,
                COUNT(*) AS evaluations,
                AVG(cs.score) AS avg_score,
                SUM(CASE WHEN cs.passed THEN 1 ELSE 0 END) AS pass_count,
                SUM(CASE WHEN cs.veto THEN 1 ELSE 0 END) AS veto_count,
                AVG(cs.evaluation_ms) AS avg_eval_ms
            FROM ct_contract_stages cs
            {join_clause}
            GROUP BY cs.stage_name
            ORDER BY avg_score ASC
        """)

        with self._engine.connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            {
                "stage": r[0],
                "evaluations": int(r[1]),
                "avg_score": round(float(r[2]), 3),
                "pass_rate": round(int(r[3]) / int(r[1]), 3) if r[1] else 0.0,
                "veto_count": int(r[4]),
                "avg_eval_ms": round(float(r[5]), 2) if r[5] else 0.0,
            }
            for r in rows
        ]

    def get_confidence_calibration(self, bins: int = 10) -> list[dict]:
        """
        Calibration curve: predicted confidence vs actual win rate.

        Groups contracts by confidence bin and computes actual profitability.
        """
        sql = text("""
            SELECT
                CAST(c.final_confidence * :bins AS INTEGER) * 1.0 / :bins AS bin_low,
                COUNT(*) AS cnt,
                AVG(CASE WHEN o.actual_profitable THEN 1.0 ELSE 0.0 END) AS actual_win_rate,
                AVG(c.final_confidence) AS avg_predicted
            FROM ct_contracts c
            JOIN ct_contract_outcomes o ON c.contract_id = o.contract_id
            WHERE c.decision = 'execute'
            GROUP BY bin_low
            ORDER BY bin_low
        """)

        with self._engine.connect() as conn:
            rows = conn.execute(sql, {"bins": bins}).fetchall()

        return [
            {
                "bin": f"{float(r[0]):.2f}-{float(r[0]) + 1.0/bins:.2f}",
                "count": int(r[1]),
                "actual_win_rate": round(float(r[2]), 3) if r[2] else 0.0,
                "avg_predicted": round(float(r[3]), 3) if r[3] else 0.0,
            }
            for r in rows
        ]

    # ──────────────────────────────────────────────────────────────────────
    # Maintenance
    # ──────────────────────────────────────────────────────────────────────

    def count(self) -> dict:
        """Quick counts for monitoring."""
        with self._Session() as session:
            total = session.query(CTContract).count()
            executed = session.query(CTContract).filter(
                CTContract.execution_status == "executed"
            ).count()
            rejected = session.query(CTContract).filter(
                CTContract.execution_status == "rejected"
            ).count()
            with_outcome = session.query(CTContractOutcome).count()

        return {
            "total": total,
            "executed": executed,
            "rejected": rejected,
            "with_outcome": with_outcome,
        }

    def close(self) -> None:
        """Dispose the SQLAlchemy engine and release connection pool."""
        self._engine.dispose()

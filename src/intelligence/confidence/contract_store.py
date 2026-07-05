"""
Contract Store — DuckDB-backed persistence for immutable DecisionContracts.

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

import duckdb

from src.intelligence.confidence.decision_contract import (
    DecisionContract,
    Decision,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ContractStore:
    """
    Persistent storage for DecisionContracts in DuckDB.

    Thread-safe. Write operations are serialized via lock.
    Read operations are lock-free (DuckDB handles concurrency).

    Tables:
        contracts       — Full contract snapshot (one row per contract)
        contract_stages — Individual stage assessments (N rows per contract)
        contract_outcomes — Execution results linked back to contracts
    """

    _CREATE_TABLES_SQL = """
    CREATE TABLE IF NOT EXISTS contracts (
        contract_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        direction TEXT NOT NULL,
        decision TEXT NOT NULL,
        final_confidence DOUBLE NOT NULL,
        recommendation TEXT,
        vetoed BOOLEAN DEFAULT FALSE,
        vetoed_by TEXT,
        veto_reason TEXT,
        recommended_position_pct DOUBLE DEFAULT 0.0,
        kelly_fraction DOUBLE DEFAULT 0.0,
        risk_grade TEXT DEFAULT '',
        integrity_hash TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL,
        valid_until TIMESTAMP NOT NULL,
        -- Provenance
        model_version TEXT DEFAULT '',
        feature_set_version TEXT DEFAULT '',
        dataset_version TEXT DEFAULT '',
        walk_forward_passed BOOLEAN,
        monte_carlo_passed BOOLEAN,
        model_health_score DOUBLE,
        -- Full JSON for replay
        full_contract_json TEXT NOT NULL,
        -- Execution tracking
        execution_status TEXT DEFAULT 'pending',  -- pending, executed, rejected, expired
        trade_id TEXT,
        stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS contract_stages (
        contract_id TEXT NOT NULL,
        stage_name TEXT NOT NULL,
        score DOUBLE NOT NULL,
        passed BOOLEAN NOT NULL,
        weight DOUBLE NOT NULL,
        severity TEXT DEFAULT 'none',
        veto BOOLEAN DEFAULT FALSE,
        veto_reason TEXT DEFAULT '',
        evidence_json TEXT DEFAULT '{}',
        warnings_json TEXT DEFAULT '[]',
        blockers_json TEXT DEFAULT '[]',
        evaluation_ms DOUBLE DEFAULT 0.0,
        PRIMARY KEY (contract_id, stage_name)
    );

    CREATE TABLE IF NOT EXISTS contract_outcomes (
        contract_id TEXT PRIMARY KEY,
        trade_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        entry_price DOUBLE,
        exit_price DOUBLE,
        pnl DOUBLE,
        pnl_pct DOUBLE,
        actual_profitable BOOLEAN,
        holding_minutes INTEGER,
        max_favorable_excursion DOUBLE,
        max_adverse_excursion DOUBLE,
        slippage_pct DOUBLE DEFAULT 0.0,
        fees DOUBLE DEFAULT 0.0,
        exit_reason TEXT DEFAULT '',
        closed_at TIMESTAMP,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_contracts_symbol ON contracts(symbol);
    CREATE INDEX IF NOT EXISTS idx_contracts_decision ON contracts(decision);
    CREATE INDEX IF NOT EXISTS idx_contracts_created ON contracts(created_at);
    CREATE INDEX IF NOT EXISTS idx_contracts_model ON contracts(model_version);
    CREATE INDEX IF NOT EXISTS idx_outcomes_trade ON contract_outcomes(trade_id);
    """

    def __init__(self, storage_path: str = "data_cache/contracts") -> None:
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._db_path = self._storage_path / "contracts.duckdb"
        self._write_lock = threading.Lock()
        self._conn = duckdb.connect(str(self._db_path))
        self._init_tables()
        logger.info("contract_store.initialized", path=str(self._db_path))

    def _init_tables(self) -> None:
        """Create tables if they don't exist."""
        self._conn.execute(self._CREATE_TABLES_SQL)

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
            # Insert contract header
            self._conn.execute(
                """
                INSERT OR REPLACE INTO contracts (
                    contract_id, symbol, direction, decision, final_confidence,
                    recommendation, vetoed, vetoed_by, veto_reason,
                    recommended_position_pct, kelly_fraction, risk_grade,
                    integrity_hash, created_at, valid_until,
                    model_version, feature_set_version, dataset_version,
                    walk_forward_passed, monte_carlo_passed, model_health_score,
                    full_contract_json, execution_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    contract.contract_id,
                    contract.symbol,
                    contract.direction,
                    contract.decision.value,
                    contract.final_confidence,
                    contract.recommendation,
                    contract.vetoed,
                    contract.vetoed_by.value if contract.vetoed_by else None,
                    contract.veto_reason,
                    contract.recommended_position_pct,
                    contract.kelly_fraction,
                    contract.risk_grade,
                    contract.integrity_hash,
                    contract.created_at,
                    contract.valid_until,
                    contract.provenance.model_version,
                    contract.provenance.feature_set_version,
                    contract.provenance.dataset_version,
                    contract.provenance.walk_forward_passed,
                    contract.provenance.monte_carlo_passed,
                    contract.provenance.model_health_score,
                    full_json,
                    "pending" if contract.is_executable else "rejected",
                ],
            )

            # Insert stage assessments
            for stage in contract.stages:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO contract_stages (
                        contract_id, stage_name, score, passed, weight,
                        severity, veto, veto_reason, evidence_json,
                        warnings_json, blockers_json, evaluation_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        contract.contract_id,
                        stage.stage,
                        stage.score,
                        stage.passed,
                        stage.weight,
                        stage.severity.value,
                        stage.veto,
                        stage.veto_reason,
                        json.dumps(stage.evidence, default=str),
                        json.dumps(stage.warnings),
                        json.dumps(stage.blockers),
                        stage.evaluation_ms,
                    ],
                )

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
            self._conn.execute(
                """
                UPDATE contracts
                SET execution_status = 'executed', trade_id = ?
                WHERE contract_id = ?
                """,
                [trade_id, contract_id],
            )

    def mark_expired(self, contract_id: str) -> None:
        """Mark a contract as expired (validity window passed without execution)."""
        with self._write_lock:
            self._conn.execute(
                """
                UPDATE contracts SET execution_status = 'expired'
                WHERE contract_id = ? AND execution_status = 'pending'
                """,
                [contract_id],
            )

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
            self._conn.execute(
                """
                INSERT OR REPLACE INTO contract_outcomes (
                    contract_id, trade_id, symbol, side,
                    entry_price, exit_price, pnl, pnl_pct,
                    actual_profitable, holding_minutes,
                    max_favorable_excursion, max_adverse_excursion,
                    slippage_pct, fees, exit_reason, closed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    contract_id, trade_id, symbol, side,
                    entry_price, exit_price, pnl, pnl_pct,
                    actual_profitable, holding_minutes,
                    max_favorable_excursion, max_adverse_excursion,
                    slippage_pct, fees, exit_reason,
                    datetime.now(timezone.utc),
                ],
            )

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
        row = self._conn.execute(
            "SELECT full_contract_json, execution_status, trade_id FROM contracts WHERE contract_id = ?",
            [contract_id],
        ).fetchone()

        if not row:
            return None

        contract_data = json.loads(row[0])
        contract_data["_replay_metadata"] = {
            "execution_status": row[1],
            "trade_id": row[2],
            "replayed_at": datetime.now(timezone.utc).isoformat(),
        }

        # Attach outcome if available
        outcome_row = self._conn.execute(
            """
            SELECT trade_id, entry_price, exit_price, pnl, pnl_pct,
                   actual_profitable, holding_minutes,
                   max_favorable_excursion, max_adverse_excursion,
                   slippage_pct, fees, exit_reason, closed_at
            FROM contract_outcomes WHERE contract_id = ?
            """,
            [contract_id],
        ).fetchone()

        if outcome_row:
            contract_data["_outcome"] = {
                "trade_id": outcome_row[0],
                "entry_price": outcome_row[1],
                "exit_price": outcome_row[2],
                "pnl": outcome_row[3],
                "pnl_pct": outcome_row[4],
                "actual_profitable": outcome_row[5],
                "holding_minutes": outcome_row[6],
                "max_favorable_excursion": outcome_row[7],
                "max_adverse_excursion": outcome_row[8],
                "slippage_pct": outcome_row[9],
                "fees": outcome_row[10],
                "exit_reason": outcome_row[11],
                "closed_at": str(outcome_row[12]) if outcome_row[12] else None,
            }

        # Attach per-stage details from structured table
        stages = self._conn.execute(
            """
            SELECT stage_name, score, passed, weight, severity,
                   veto, veto_reason, evidence_json, warnings_json,
                   blockers_json, evaluation_ms
            FROM contract_stages WHERE contract_id = ?
            ORDER BY stage_name
            """,
            [contract_id],
        ).fetchall()

        if stages:
            contract_data["_stage_details"] = [
                {
                    "stage": s[0],
                    "score": s[1],
                    "passed": s[2],
                    "weight": s[3],
                    "severity": s[4],
                    "veto": s[5],
                    "veto_reason": s[6],
                    "evidence": json.loads(s[7]) if s[7] else {},
                    "warnings": json.loads(s[8]) if s[8] else [],
                    "blockers": json.loads(s[9]) if s[9] else [],
                    "evaluation_ms": s[10],
                }
                for s in stages
            ]

        return contract_data

    def replay_by_trade(self, trade_id: str) -> Optional[dict]:
        """Replay a contract by its linked trade_id (reverse lookup)."""
        row = self._conn.execute(
            "SELECT contract_id FROM contracts WHERE trade_id = ?",
            [trade_id],
        ).fetchone()

        if not row:
            # Try outcomes table
            row = self._conn.execute(
                "SELECT contract_id FROM contract_outcomes WHERE trade_id = ?",
                [trade_id],
            ).fetchone()

        if row:
            return self.replay(row[0])
        return None

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
        conditions = []
        params = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if decision:
            conditions.append("decision = ?")
            params.append(decision)
        if since:
            conditions.append("created_at >= ?")
            params.append(since)
        if until:
            conditions.append("created_at <= ?")
            params.append(until)
        if model_version:
            conditions.append("model_version = ?")
            params.append(model_version)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)

        rows = self._conn.execute(
            f"""
            SELECT contract_id, symbol, direction, decision, final_confidence,
                   vetoed, recommended_position_pct, risk_grade,
                   model_version, execution_status, trade_id, created_at
            FROM contracts
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        return [
            {
                "contract_id": r[0],
                "symbol": r[1],
                "direction": r[2],
                "decision": r[3],
                "confidence": r[4],
                "vetoed": r[5],
                "position_pct": r[6],
                "risk_grade": r[7],
                "model_version": r[8],
                "execution_status": r[9],
                "trade_id": r[10],
                "created_at": str(r[11]),
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
        params = []

        if symbol:
            conditions.append("c.symbol = ?")
            params.append(symbol)
        if since:
            conditions.append("c.created_at >= ?")
            params.append(since)

        where = "WHERE " + " AND ".join(conditions)

        row = self._conn.execute(
            f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN o.actual_profitable THEN 1 ELSE 0 END) AS wins,
                AVG(o.pnl_pct) AS avg_pnl_pct,
                AVG(c.final_confidence) AS avg_confidence,
                AVG(o.holding_minutes) AS avg_holding
            FROM contracts c
            LEFT JOIN contract_outcomes o ON c.contract_id = o.contract_id
            {where}
            """,
            params,
        ).fetchone()

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
        params = []

        if symbol:
            conditions.append("c.symbol = ?")
            params.append(symbol)
        if since:
            conditions.append("c.created_at >= ?")
            params.append(since)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        join_where = f"JOIN contracts c ON cs.contract_id = c.contract_id {where}" if conditions else ""

        rows = self._conn.execute(
            f"""
            SELECT
                cs.stage_name,
                COUNT(*) AS evaluations,
                AVG(cs.score) AS avg_score,
                SUM(CASE WHEN cs.passed THEN 1 ELSE 0 END) AS pass_count,
                SUM(CASE WHEN cs.veto THEN 1 ELSE 0 END) AS veto_count,
                AVG(cs.evaluation_ms) AS avg_eval_ms
            FROM contract_stages cs
            {join_where}
            GROUP BY cs.stage_name
            ORDER BY avg_score ASC
            """,
            params,
        ).fetchall()

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
        rows = self._conn.execute(
            """
            SELECT
                FLOOR(c.final_confidence * ?) / ? AS bin_low,
                COUNT(*) AS cnt,
                AVG(CASE WHEN o.actual_profitable THEN 1.0 ELSE 0.0 END) AS actual_win_rate,
                AVG(c.final_confidence) AS avg_predicted
            FROM contracts c
            JOIN contract_outcomes o ON c.contract_id = o.contract_id
            WHERE c.decision = 'execute'
            GROUP BY bin_low
            ORDER BY bin_low
            """,
            [bins, bins],
        ).fetchall()

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
        total = self._conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
        executed = self._conn.execute(
            "SELECT COUNT(*) FROM contracts WHERE execution_status = 'executed'"
        ).fetchone()[0]
        rejected = self._conn.execute(
            "SELECT COUNT(*) FROM contracts WHERE execution_status = 'rejected'"
        ).fetchone()[0]
        with_outcome = self._conn.execute(
            "SELECT COUNT(*) FROM contract_outcomes"
        ).fetchone()[0]
        return {
            "total": total,
            "executed": executed,
            "rejected": rejected,
            "with_outcome": with_outcome,
        }

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()

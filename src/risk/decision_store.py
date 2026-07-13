"""
Risk Decision Store — Persistent, timestamped audit trail for risk-layer decisions.

Provides durable storage of every risk evaluation with full layer breakdown,
enabling post-hoc audit, compliance review, and failure analysis.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RiskDecisionStore:
    """
    SQLite-backed persistent store for risk engine decisions.

    Each decision is stored with:
    - UTC timestamp (wall-clock and monotonic for ordering)
    - Full layer breakdown (per-layer action, reason, metadata)
    - Request context (symbol, side, qty, confidence, strategy)
    - Final verdict and risk score

    Thread-safe. Uses WAL mode for concurrent read/write.
    """

    def __init__(self, db_path: str = "data_cache/risk_decisions.db"):
        self._db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()

    def _create_schema(self) -> None:
        """Create the decisions table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                timestamp_epoch REAL NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                requested_qty REAL NOT NULL,
                adjusted_qty REAL NOT NULL,
                approved INTEGER NOT NULL,
                risk_score REAL NOT NULL,
                confidence REAL,
                strategy TEXT,
                reasons TEXT NOT NULL,
                layer_details TEXT NOT NULL,
                signal_id TEXT,
                decision_contract_id TEXT
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_risk_decisions_timestamp
            ON risk_decisions(timestamp_utc)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_risk_decisions_symbol
            ON risk_decisions(symbol)
        """)
        self._conn.commit()

    def persist(
        self,
        symbol: str,
        side: str,
        requested_qty: float,
        adjusted_qty: float,
        approved: bool,
        risk_score: float,
        reasons: list[str],
        layer_details: dict[str, Any],
        confidence: Optional[float] = None,
        strategy: Optional[str] = None,
        signal_id: Optional[str] = None,
        decision_contract_id: Optional[str] = None,
    ) -> int:
        """
        Persist a risk decision. Returns the row ID.

        Args:
            symbol: Instrument symbol.
            side: Trade side (BUY/SELL).
            requested_qty: Originally requested quantity.
            adjusted_qty: Quantity after risk adjustments.
            approved: Whether the trade was approved.
            risk_score: Aggregate risk score (0-100).
            reasons: List of human-readable reasons.
            layer_details: Per-layer breakdown dict.
            confidence: Effective confidence score.
            strategy: Strategy name.
            signal_id: Originating signal ID.
            decision_contract_id: Associated decision contract ID.

        Returns:
            Database row ID of the persisted decision.
        """
        now = datetime.now(timezone.utc)
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO risk_decisions (
                    timestamp_utc, timestamp_epoch, symbol, side,
                    requested_qty, adjusted_qty, approved, risk_score,
                    confidence, strategy, reasons, layer_details,
                    signal_id, decision_contract_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now.isoformat(),
                    now.timestamp(),
                    symbol,
                    side,
                    requested_qty,
                    adjusted_qty,
                    1 if approved else 0,
                    risk_score,
                    confidence,
                    strategy,
                    json.dumps(reasons),
                    json.dumps(layer_details, default=str),
                    signal_id,
                    decision_contract_id,
                ),
            )
            self._conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def query(
        self,
        symbol: Optional[str] = None,
        approved: Optional[bool] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query persisted risk decisions with optional filters.

        Returns list of decision dicts ordered by timestamp descending.
        """
        conditions = []
        params: list[Any] = []

        if symbol is not None:
            conditions.append("symbol = ?")
            params.append(symbol)
        if approved is not None:
            conditions.append("approved = ?")
            params.append(1 if approved else 0)
        if since is not None:
            conditions.append("timestamp_utc >= ?")
            params.append(since.isoformat())
        if until is not None:
            conditions.append("timestamp_utc <= ?")
            params.append(until.isoformat())

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        with self._lock:
            cursor = self._conn.execute(
                f"""
                SELECT * FROM risk_decisions
                {where_clause}
                ORDER BY timestamp_utc DESC
                LIMIT ?
                """,
                params,
            )
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

        results = []
        for row in rows:
            entry = dict(zip(columns, row))
            entry["reasons"] = json.loads(entry["reasons"])
            entry["layer_details"] = json.loads(entry["layer_details"])
            entry["approved"] = bool(entry["approved"])
            results.append(entry)
        return results

    def count(self, approved: Optional[bool] = None) -> int:
        """Count total decisions, optionally filtered by approval status."""
        if approved is None:
            sql = "SELECT COUNT(*) FROM risk_decisions"
            params: tuple = ()
        else:
            sql = "SELECT COUNT(*) FROM risk_decisions WHERE approved = ?"
            params = (1 if approved else 0,)
        with self._lock:
            cursor = self._conn.execute(sql, params)
            return cursor.fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

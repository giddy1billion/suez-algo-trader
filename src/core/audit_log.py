"""
Immutable Audit Trail.

Records every trade action as JSON-lines files for compliance,
debugging, and post-trade analysis.
"""

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeAuditTrail:
    """
    Full audit trail linking a trade through its entire ML lifecycle.

    Provides complete traceability:
    trade → signal → prediction → model → training run → dataset → features
    """
    trade_id: str = ""
    signal_id: str = ""
    prediction_id: str = ""
    model_version: str = ""
    training_run_id: str = ""
    backtest_run_id: str = ""
    dataset_snapshot_hash: str = ""
    feature_snapshot_hash: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_complete(self) -> bool:
        """Check if the audit trail has all required links."""
        return bool(self.trade_id and self.prediction_id and self.model_version)

    @classmethod
    def from_dict(cls, data: dict) -> "TradeAuditTrail":
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known_fields})


class AuditLogger:
    """
    Writes immutable audit entries to daily JSON-lines files.

    File naming: audit_YYYY-MM-DD.jsonl
    Each line is a self-contained JSON object with:
        timestamp, event_type, trade_id, symbol, data, source
    """

    def __init__(self, log_dir: str = "data_cache/audit") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._current_date: Optional[str] = None
        self._current_file = None

    def log(
        self,
        event_type: str,
        data: dict[str, Any],
        trade_id: str = "",
        symbol: str = "",
        source: str = "",
    ) -> None:
        """
        Write an immutable audit entry.

        Args:
            event_type: Type of event (e.g., "SignalGenerated", "OrderFilled").
            data: Full event payload.
            trade_id: Associated trade ID, if any.
            symbol: Trading symbol, if applicable.
            source: Source component name.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "trade_id": trade_id,
            "symbol": symbol,
            "source": source,
            "data": data,
        }

        line = json.dumps(entry, default=str)

        with self._lock:
            try:
                self._ensure_file()
                self._current_file.write(line + "\n")
                self._current_file.flush()
            except Exception:
                logger.exception("Failed to write audit entry: %s", event_type)

    def query(
        self,
        event_type: Optional[str] = None,
        symbol: Optional[str] = None,
        trade_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query audit log entries.

        Scans log files in reverse chronological order. Filtering is done
        in-memory (suitable for debugging, not high-frequency queries).

        Args:
            event_type: Filter by event type.
            symbol: Filter by symbol.
            trade_id: Filter by trade ID.
            since: Only include entries after this timestamp.
            limit: Maximum number of results.

        Returns:
            List of matching audit entries (most recent first).
        """
        results: list[dict[str, Any]] = []
        log_files = sorted(self._log_dir.glob("audit_*.jsonl"), reverse=True)

        for log_file in log_files:
            if len(results) >= limit:
                break

            try:
                lines = log_file.read_text(encoding="utf-8").strip().split("\n")
            except Exception:
                logger.warning("Failed to read audit file: %s", log_file)
                continue

            for line in reversed(lines):
                if len(results) >= limit:
                    break
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Apply filters
                if event_type and entry.get("event_type") != event_type:
                    continue
                if symbol and entry.get("symbol") != symbol:
                    continue
                if trade_id and entry.get("trade_id") != trade_id:
                    continue
                if since:
                    entry_ts = datetime.fromisoformat(entry["timestamp"])
                    if entry_ts < since:
                        continue

                results.append(entry)

        return results

    def close(self) -> None:
        """Close the current file handle."""
        with self._lock:
            if self._current_file:
                try:
                    self._current_file.close()
                except Exception:
                    pass
                self._current_file = None
                self._current_date = None

    def _ensure_file(self) -> None:
        """Ensure the correct daily file is open."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if self._current_date != today:
            # Close previous file
            if self._current_file:
                try:
                    self._current_file.close()
                except Exception:
                    pass

            filename = f"audit_{today}.jsonl"
            filepath = self._log_dir / filename
            self._current_file = open(filepath, "a", encoding="utf-8")
            self._current_date = today

    def __del__(self) -> None:
        self.close()

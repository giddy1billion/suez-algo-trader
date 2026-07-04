"""
Event Store Snapshotting — Periodic state persistence for fast recovery.

Instead of replaying the entire event history:
  50,000 events → replay → slow startup

Use snapshots:
  Latest snapshot + events since → fast startup
"""

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SnapshotStore:
    """
    Persists periodic state snapshots for fast recovery.

    Stores snapshots in SQLite with:
    - snapshot_id (auto)
    - session_id
    - timestamp
    - last_event_id (the event store row ID of the last processed event)
    - state (JSON blob of the ReadModelManager state)
    """

    def __init__(self, db_path: str = "data_cache/snapshots.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                last_event_id INTEGER NOT NULL,
                state TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshots_session ON snapshots(session_id)
        """)
        self._conn.commit()

    def save_snapshot(self, session_id: str, last_event_id: int, state: dict) -> int:
        """Save a state snapshot. Returns the snapshot ID."""
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO snapshots (session_id, timestamp, last_event_id, state)
                   VALUES (?, ?, ?, ?)""",
                (
                    session_id,
                    datetime.now(timezone.utc).isoformat(),
                    last_event_id,
                    json.dumps(state, default=str),
                )
            )
            self._conn.commit()
            snapshot_id = cursor.lastrowid
            logger.info("snapshot.saved", snapshot_id=snapshot_id, last_event_id=last_event_id)
            return snapshot_id

    def get_latest_snapshot(self, session_id: Optional[str] = None) -> Optional[dict]:
        """Get the most recent snapshot, optionally filtered by session."""
        with self._lock:
            if session_id:
                cursor = self._conn.execute(
                    "SELECT * FROM snapshots WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                    (session_id,)
                )
            else:
                cursor = self._conn.execute(
                    "SELECT * FROM snapshots ORDER BY id DESC LIMIT 1"
                )
            row = cursor.fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "session_id": row[1],
                "timestamp": row[2],
                "last_event_id": row[3],
                "state": json.loads(row[4]),
            }

    def get_snapshot_count(self, session_id: Optional[str] = None) -> int:
        with self._lock:
            if session_id:
                cursor = self._conn.execute(
                    "SELECT COUNT(*) FROM snapshots WHERE session_id = ?", (session_id,)
                )
            else:
                cursor = self._conn.execute("SELECT COUNT(*) FROM snapshots")
            return cursor.fetchone()[0]

    def cleanup_old_snapshots(self, keep_latest: int = 10) -> int:
        """Delete old snapshots, keeping the N most recent."""
        with self._lock:
            cursor = self._conn.execute("SELECT COUNT(*) FROM snapshots")
            total = cursor.fetchone()[0]
            if total <= keep_latest:
                return 0
            delete_count = total - keep_latest
            self._conn.execute(
                "DELETE FROM snapshots WHERE id IN (SELECT id FROM snapshots ORDER BY id ASC LIMIT ?)",
                (delete_count,)
            )
            self._conn.commit()
            logger.info("snapshot.cleanup", deleted=delete_count, remaining=keep_latest)
            return delete_count

    def close(self):
        self._conn.close()


class SnapshotManager:
    """
    Coordinates snapshot creation and recovery.

    - Creates snapshots periodically (every N events or T minutes)
    - On recovery, loads latest snapshot + replays only events since
    """

    def __init__(self, snapshot_store: SnapshotStore, event_store=None,
                 snapshot_interval_events: int = 500,
                 snapshot_interval_seconds: float = 300.0):
        self.store = snapshot_store
        self.event_store = event_store
        self.snapshot_interval = snapshot_interval_events
        self._snapshot_interval_seconds = snapshot_interval_seconds
        self._events_since_snapshot = 0
        self._last_snapshot_time = time.time()
        self._lock = threading.Lock()

    def on_event(self, event) -> None:
        """Call after each event is processed. Triggers snapshot if interval reached."""
        with self._lock:
            self._events_since_snapshot += 1

    def should_snapshot(self) -> bool:
        """Check if it's time for a new snapshot (adaptive: events OR time)."""
        with self._lock:
            events_trigger = self._events_since_snapshot >= self.snapshot_interval
            time_trigger = (time.time() - self._last_snapshot_time) >= self._snapshot_interval_seconds
            return events_trigger or time_trigger

    def take_snapshot(self, session_id: str, last_event_id: int, state: dict) -> int:
        """Take snapshot and reset both counters."""
        snapshot_id = self.store.save_snapshot(session_id, last_event_id, state)
        with self._lock:
            self._events_since_snapshot = 0
            self._last_snapshot_time = time.time()
        return snapshot_id

    def force_snapshot(self, session_id: str, last_event_id: int, state: dict, reason: str = "") -> int:
        """Force a snapshot regardless of interval (e.g., graceful shutdown, pre-deployment)."""
        logger.info("snapshot.forced", reason=reason)
        return self.take_snapshot(session_id, last_event_id, state)

    def recover_from_snapshot(self, session_id: Optional[str] = None) -> Optional[dict]:
        """
        Load the latest snapshot for recovery.

        Returns dict with 'state' and 'last_event_id', or None if no snapshot exists
        or if the snapshot fails validation.
        The caller should then replay events with id > last_event_id.
        """
        snapshot = self.store.get_latest_snapshot(session_id)
        if snapshot is None:
            logger.info("snapshot.recovery_none_found")
            return None

        # Validate snapshot integrity before using it
        if not self._validate_snapshot(snapshot):
            logger.error(
                "snapshot.validation_failed",
                snapshot_id=snapshot["id"],
                msg="Corrupt or incomplete snapshot — falling back to full replay",
            )
            return None

        logger.info(
            "snapshot.recovered",
            snapshot_id=snapshot["id"],
            last_event_id=snapshot["last_event_id"],
            timestamp=snapshot["timestamp"],
        )
        return snapshot

    @staticmethod
    def _validate_snapshot(snapshot: dict) -> bool:
        """Validate snapshot has required fields and sane values."""
        try:
            # Required top-level keys
            if not all(k in snapshot for k in ("id", "last_event_id", "state", "timestamp")):
                return False
            # last_event_id must be a positive integer
            if not isinstance(snapshot["last_event_id"], int) or snapshot["last_event_id"] < 0:
                return False
            # state must be a dict
            if not isinstance(snapshot["state"], dict):
                return False
            # timestamp must parse
            if not snapshot.get("timestamp"):
                return False
            return True
        except Exception:
            return False

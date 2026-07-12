"""
Event Store Snapshotting — Periodic state persistence for fast recovery.

Instead of replaying the entire event history:
  50,000 events → replay → slow startup

Use snapshots:
  Latest snapshot + events since → fast startup

Supports both PostgreSQL and SQLite via SQLAlchemy.
"""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import Column, Integer, String, Text, Index
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SnapshotBase(DeclarativeBase):
    pass


class SnapshotRecord(SnapshotBase):
    """Persisted state snapshot."""
    __tablename__ = "snapshots"
    __table_args__ = (
        Index("idx_snapshots_session", "session_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False)
    timestamp = Column(String(64), nullable=False)
    last_event_id = Column(Integer, nullable=False)
    state = Column(Text, nullable=False)
    schema_version = Column(String(20), nullable=False, default="1")
    engine_version = Column(String(20), nullable=False, default="1.0.0")
    config_hash = Column(String(64), nullable=False, default="")
    created_at = Column(String(64), default=lambda: datetime.now(timezone.utc).isoformat())


class SnapshotStore:
    """
    Persists periodic state snapshots for fast recovery.

    Supports PostgreSQL and SQLite via the shared engine factory.
    """

    def __init__(self, db_path: str = "data_cache/snapshots.db", database_url: Optional[str] = None):
        self._lock = threading.Lock()

        if database_url:
            url = database_url
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{db_path}"

        from src.utils.database import create_db_engine
        self._engine = create_db_engine(url)
        SnapshotBase.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)

    def save_snapshot(
        self,
        session_id: str,
        last_event_id: int,
        state: dict,
        schema_version: str = "1",
        engine_version: str = "1.0.0",
        config_hash: str = "",
    ) -> int:
        """Save a state snapshot with versioning metadata. Returns the snapshot ID."""
        with self._lock:
            with self._session_factory() as session:
                record = SnapshotRecord(
                    session_id=session_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    last_event_id=last_event_id,
                    state=json.dumps(state, default=str),
                    schema_version=schema_version,
                    engine_version=engine_version,
                    config_hash=config_hash,
                )
                session.add(record)
                session.commit()
                session.refresh(record)
                snapshot_id = record.id
                logger.info("snapshot.saved", snapshot_id=snapshot_id, last_event_id=last_event_id)
                return snapshot_id

    def get_latest_snapshot(self, session_id: Optional[str] = None) -> Optional[dict]:
        """Get the most recent snapshot, optionally filtered by session."""
        with self._lock:
            with self._session_factory() as session:
                q = session.query(SnapshotRecord)
                if session_id:
                    q = q.filter_by(session_id=session_id)
                record = q.order_by(SnapshotRecord.id.desc()).first()
                if record is None:
                    return None
                return {
                    "id": record.id,
                    "session_id": record.session_id,
                    "timestamp": record.timestamp,
                    "last_event_id": record.last_event_id,
                    "state": json.loads(record.state),
                    "schema_version": record.schema_version or "1",
                    "engine_version": record.engine_version or "1.0.0",
                    "config_hash": record.config_hash or "",
                }

    def get_snapshot_count(self, session_id: Optional[str] = None) -> int:
        with self._lock:
            with self._session_factory() as session:
                q = session.query(SnapshotRecord)
                if session_id:
                    q = q.filter_by(session_id=session_id)
                return q.count()

    def cleanup_old_snapshots(self, keep_latest: int = 10) -> int:
        """Delete old snapshots, keeping the N most recent."""
        with self._lock:
            with self._session_factory() as session:
                total = session.query(SnapshotRecord).count()
                if total <= keep_latest:
                    return 0
                delete_count = total - keep_latest
                # Get IDs of records to delete (oldest first)
                old_ids = [
                    r.id for r in session.query(SnapshotRecord.id)
                    .order_by(SnapshotRecord.id.asc())
                    .limit(delete_count)
                    .all()
                ]
                if old_ids:
                    session.query(SnapshotRecord).filter(
                        SnapshotRecord.id.in_(old_ids)
                    ).delete(synchronize_session=False)
                    session.commit()
                logger.info("snapshot.cleanup", deleted=delete_count, remaining=keep_latest)
                return delete_count

    def close(self):
        self._engine.dispose()


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

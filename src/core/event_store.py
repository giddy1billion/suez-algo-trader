"""
Persistent Event Store — SQLite-backed durable event log.

Persists ALL events published through the EventBus for auditing,
replay, and debugging purposes.
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from src.core.events import (
    Event,
    OrderAccepted,
    OrderFilled,
    OrderPartialFill,
    OrderRejected,
    OrderSubmitted,
    RiskEvaluated,
    RiskHalt,
    SchedulerEvent,
    SignalGenerated,
    SystemHealth,
    TradeClosed,
    TradeOpened,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Event Class Registry
# ---------------------------------------------------------------------------

EVENT_REGISTRY: dict[str, type] = {
    "Event": Event,
    "SignalGenerated": SignalGenerated,
    "RiskEvaluated": RiskEvaluated,
    "OrderSubmitted": OrderSubmitted,
    "OrderAccepted": OrderAccepted,
    "OrderPartialFill": OrderPartialFill,
    "OrderFilled": OrderFilled,
    "OrderRejected": OrderRejected,
    "TradeOpened": TradeOpened,
    "TradeClosed": TradeClosed,
    "RiskHalt": RiskHalt,
    "SchedulerEvent": SchedulerEvent,
    "SystemHealth": SystemHealth,
}


def register_event_class(cls: type) -> None:
    """Register an additional event class for deserialization."""
    EVENT_REGISTRY[cls.__name__] = cls


def _reconstruct_event(event_type: str, payload: dict) -> Event:
    """Reconstruct a typed Event object from its type name and payload dict."""
    cls = EVENT_REGISTRY.get(event_type, Event)
    try:
        return cls.from_dict(payload)
    except Exception:
        # Fallback: return base Event if subclass deserialization fails
        logger.warning("Failed to reconstruct %s, falling back to base Event", event_type)
        return Event.from_dict(payload)


# ---------------------------------------------------------------------------
# EventStore
# ---------------------------------------------------------------------------

class EventStore:
    """
    SQLite-backed persistent event store.

    Stores all events with metadata for later retrieval, replay, and auditing.
    Thread-safe with WAL mode for concurrent read/write performance.
    """

    def __init__(self, db_path: str = "data_cache/events.db", session_id: Optional[str] = None):
        self.db_path = db_path
        self.session_id = session_id or uuid.uuid4().hex
        self._lock = threading.Lock()

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Create connection (thread-safe)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._create_tables()
        logger.info("EventStore initialized", db_path=db_path, session_id=self.session_id)

    def _create_tables(self) -> None:
        """Create the events table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                event_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                source TEXT DEFAULT '',
                payload TEXT NOT NULL,
                session_id TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)
        """)
        self._conn.commit()

    def persist(self, event: Event) -> None:
        """Persist an event to the database."""
        try:
            data = event.to_dict()
            event_type = data.pop("_type", type(event).__name__)
            payload_json = json.dumps(data, default=str)

            with self._lock:
                self._conn.execute(
                    """INSERT INTO events (event_type, event_id, timestamp, source, payload, session_id)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        event_type,
                        event.event_id,
                        event.timestamp.isoformat() if isinstance(event.timestamp, datetime) else str(event.timestamp),
                        event.source,
                        payload_json,
                        self.session_id,
                    ),
                )
                self._conn.commit()
        except Exception:
            logger.exception("Failed to persist event %s", type(event).__name__)
            raise

    def get_session_events(self, session_id: str) -> List[dict]:
        """Get all events from a specific session."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM events WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_events_by_type(self, event_type: str, limit: int = 100) -> List[dict]:
        """Get events filtered by type."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM events WHERE event_type = ? ORDER BY id DESC LIMIT ?",
                (event_type, limit),
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_events_since(self, timestamp: datetime) -> List[dict]:
        """Get all events since a given timestamp."""
        ts_str = timestamp.isoformat()
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM events WHERE timestamp >= ? ORDER BY id ASC",
                (ts_str,),
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_latest_events(self, limit: int = 50) -> List[dict]:
        """Get the most recent events."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def replay_session(self, session_id: str) -> List[Event]:
        """Replay a session by deserializing events back to typed objects."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM events WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            )
            rows = cursor.fetchall()

        events = []
        for row in rows:
            row_dict = self._row_to_dict(row)
            payload = json.loads(row_dict["payload"])
            # Ensure _type is in payload for from_dict
            payload["_type"] = row_dict["event_type"]
            event = _reconstruct_event(row_dict["event_type"], payload)
            events.append(event)
        return events

    def count_events(self, session_id: Optional[str] = None) -> int:
        """Count total events, optionally filtered by session."""
        with self._lock:
            if session_id:
                cursor = self._conn.execute(
                    "SELECT COUNT(*) FROM events WHERE session_id = ?",
                    (session_id,),
                )
            else:
                cursor = self._conn.execute("SELECT COUNT(*) FROM events")
            return cursor.fetchone()[0]

    def cleanup_old_events(self, days: int = 30) -> None:
        """Delete events older than N days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = cutoff.isoformat()
        with self._lock:
            self._conn.execute(
                "DELETE FROM events WHERE timestamp < ?",
                (cutoff_str,),
            )
            self._conn.commit()
        logger.info("Cleaned up events older than %d days", days)

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict."""
        return dict(row)


# ---------------------------------------------------------------------------
# EventPersistenceSubscriber
# ---------------------------------------------------------------------------

class EventPersistenceSubscriber:
    """
    Subscribes to ALL events on the EventBus (wildcard) and persists them
    to the EventStore. Handles serialization errors gracefully.
    """

    def __init__(self, event_store: EventStore):
        self.event_store = event_store

    def attach(self, event_bus) -> None:
        """Subscribe to all events on the given EventBus."""
        event_bus.subscribe(None, self.handle_event)
        logger.info("EventPersistenceSubscriber attached to EventBus")

    def handle_event(self, event: Event) -> None:
        """Persist event, logging warnings on failure without crashing."""
        try:
            self.event_store.persist(event)
        except Exception:
            logger.warning(
                "Failed to persist event %s (id=%s) — skipping",
                type(event).__name__,
                getattr(event, "event_id", "unknown"),
                exc_info=True,
            )

"""
Persistent Event Store — Database-backed durable event log.

Persists ALL events published through the EventBus for auditing,
replay, and debugging purposes. Supports both PostgreSQL and SQLite
backends via SQLAlchemy.
"""

import json
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from sqlalchemy import Column, Integer, String, Text, Index, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

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
# ORM Model
# ---------------------------------------------------------------------------

class EventBase(DeclarativeBase):
    pass


class EventRecord(EventBase):
    """Persisted event record."""
    __tablename__ = "events"
    __table_args__ = (
        Index("idx_events_session", "session_id"),
        Index("idx_events_type", "event_type"),
        Index("idx_events_timestamp", "timestamp"),
        Index("idx_events_event_id", "event_id", unique=True),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(100), nullable=False)
    event_id = Column(String(64), nullable=False, unique=True)
    timestamp = Column(String(64), nullable=False)
    source = Column(String(100), default="")
    payload = Column(Text, nullable=False)
    session_id = Column(String(64), nullable=False)


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
        logger.warning("Failed to reconstruct %s, falling back to base Event", event_type)
        return Event.from_dict(payload)


# ---------------------------------------------------------------------------
# EventStore
# ---------------------------------------------------------------------------

class EventStore:
    """
    Database-backed persistent event store.

    Stores all events with metadata for later retrieval, replay, and auditing.
    Supports PostgreSQL and SQLite via the shared engine factory.
    """

    def __init__(self, db_path: str = "data_cache/events.db", session_id: Optional[str] = None,
                 database_url: Optional[str] = None):
        self.session_id = session_id or uuid.uuid4().hex
        self._lock = threading.Lock()

        # If an explicit database_url is provided (e.g., postgresql://), use it.
        # Otherwise fall back to SQLite file path for backward compatibility.
        if database_url:
            url = database_url
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{db_path}"

        from src.utils.database import create_db_engine
        self._engine = create_db_engine(url)
        # For PostgreSQL, schema is managed by Alembic migrations (bootstrap_database).
        if not url.startswith("postgresql"):
            EventBase.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)

        logger.info("EventStore initialized", session_id=self.session_id)

    def _get_session(self) -> Session:
        return self._session_factory()

    def persist(self, event: Event) -> None:
        """Persist an event to the database. Deduplicates by event_id.

        Uses INSERT ... ON CONFLICT DO NOTHING on PostgreSQL to avoid TOCTOU race
        conditions in multi-process deployments. Falls back to check-then-insert
        for SQLite (single-writer, lock-protected).
        """
        try:
            data = event.to_dict()
            event_type = data.pop("_type", type(event).__name__)
            payload_json = json.dumps(data, default=str)

            record = EventRecord(
                event_type=event_type,
                event_id=event.event_id,
                timestamp=event.timestamp.isoformat() if isinstance(event.timestamp, datetime) else str(event.timestamp),
                source=event.source,
                payload=payload_json,
                session_id=self.session_id,
            )

            with self._lock:
                with self._get_session() as session:
                    dialect = session.bind.dialect.name if session.bind else "sqlite"
                    if dialect == "postgresql":
                        from sqlalchemy.dialects.postgresql import insert as pg_insert
                        stmt = pg_insert(EventRecord).values(
                            event_type=record.event_type,
                            event_id=record.event_id,
                            timestamp=record.timestamp,
                            source=record.source,
                            payload=record.payload,
                            session_id=record.session_id,
                        ).on_conflict_do_nothing(index_elements=["event_id"])
                        session.execute(stmt)
                        session.commit()
                    else:
                        # SQLite: check-then-insert under the threading lock
                        existing = session.query(EventRecord.id).filter_by(
                            event_id=event.event_id
                        ).first()
                        if existing:
                            return
                        session.add(record)
                        session.commit()
        except Exception:
            logger.exception("Failed to persist event %s", type(event).__name__)
            raise

    def get_session_events(self, session_id: str) -> List[dict]:
        """Get all events from a specific session."""
        with self._lock:
            with self._get_session() as session:
                records = session.query(EventRecord).filter_by(
                    session_id=session_id
                ).order_by(EventRecord.id.asc()).all()
                return [self._record_to_dict(r) for r in records]

    def get_events_by_type(self, event_type: str, limit: int = 100) -> List[dict]:
        """Get events filtered by type."""
        with self._lock:
            with self._get_session() as session:
                records = session.query(EventRecord).filter_by(
                    event_type=event_type
                ).order_by(EventRecord.id.desc()).limit(limit).all()
                return [self._record_to_dict(r) for r in records]

    def get_events_since(self, timestamp: datetime) -> List[dict]:
        """Get all events since a given timestamp."""
        ts_str = timestamp.isoformat()
        with self._lock:
            with self._get_session() as session:
                records = session.query(EventRecord).filter(
                    EventRecord.timestamp >= ts_str
                ).order_by(EventRecord.id.asc()).all()
                return [self._record_to_dict(r) for r in records]

    def get_latest_events(self, limit: int = 50) -> List[dict]:
        """Get the most recent events."""
        with self._lock:
            with self._get_session() as session:
                records = session.query(EventRecord).order_by(
                    EventRecord.id.desc()
                ).limit(limit).all()
                return [self._record_to_dict(r) for r in records]

    def replay_session(self, session_id: str) -> List[Event]:
        """Replay a session by deserializing events back to typed objects."""
        with self._lock:
            with self._get_session() as session:
                records = session.query(EventRecord).filter_by(
                    session_id=session_id
                ).order_by(EventRecord.id.asc()).all()
                rows = [self._record_to_dict(r) for r in records]

        events = []
        for row_dict in rows:
            payload = json.loads(row_dict["payload"])
            payload["_type"] = row_dict["event_type"]
            event = _reconstruct_event(row_dict["event_type"], payload)
            events.append(event)
        return events

    def count_events(self, session_id: Optional[str] = None) -> int:
        """Count total events, optionally filtered by session."""
        with self._lock:
            with self._get_session() as session:
                q = session.query(EventRecord)
                if session_id:
                    q = q.filter_by(session_id=session_id)
                return q.count()

    def cleanup_old_events(self, days: int = 30) -> None:
        """Delete events older than N days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = cutoff.isoformat()
        with self._lock:
            with self._get_session() as session:
                session.query(EventRecord).filter(
                    EventRecord.timestamp < cutoff_str
                ).delete(synchronize_session=False)
                session.commit()
        logger.info("Cleaned up events older than %d days", days)

    def list_sessions(self, limit: int = 20) -> List[dict]:
        """List available sessions with event counts and time ranges."""
        from sqlalchemy import func
        with self._lock:
            with self._get_session() as session:
                results = session.query(
                    EventRecord.session_id,
                    func.count(EventRecord.id).label("event_count"),
                    func.min(EventRecord.timestamp).label("first_event"),
                    func.max(EventRecord.timestamp).label("last_event"),
                ).group_by(EventRecord.session_id).order_by(
                    func.max(EventRecord.id).desc()
                ).limit(limit).all()
                return [
                    {
                        "session_id": r.session_id,
                        "event_count": r.event_count,
                        "first_event": r.first_event,
                        "last_event": r.last_event,
                    }
                    for r in results
                ]

    def close(self) -> None:
        """Dispose of the engine connection pool."""
        self._engine.dispose()

    @staticmethod
    def _record_to_dict(record: EventRecord) -> dict:
        """Convert an EventRecord ORM object to a plain dict."""
        return {
            "id": record.id,
            "event_type": record.event_type,
            "event_id": record.event_id,
            "timestamp": record.timestamp,
            "source": record.source,
            "payload": record.payload,
            "session_id": record.session_id,
        }


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

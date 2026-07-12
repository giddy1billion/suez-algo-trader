"""
Shared Database Engine Factory — supports PostgreSQL and SQLite.

Provides a centralized engine creation function that configures the
appropriate driver, pooling, and pragmas based on the DATABASE_URL scheme.

Usage:
    from src.utils.database import create_db_engine, get_engine

    # Create a one-off engine
    engine = create_db_engine("postgresql://user:pass@host:5432/db")

    # Get/share the application-wide singleton engine
    engine = get_engine()
"""

import threading
from typing import Optional

from sqlalchemy import create_engine as sa_create_engine, event as sa_event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.utils.logger import get_logger

logger = get_logger(__name__)

_engine_lock = threading.Lock()
_shared_engine: Optional[Engine] = None
_shared_session_factory: Optional[sessionmaker] = None


def create_db_engine(database_url: str, echo: bool = False) -> Engine:
    """
    Create a SQLAlchemy engine configured for the given database URL.

    Supports:
      - postgresql:// or postgresql+psycopg2://  → connection pool
      - sqlite:///  → StaticPool with WAL pragmas

    Returns a fully configured Engine instance.
    """
    engine_kwargs = {"echo": echo}

    if database_url.startswith("sqlite"):
        # SQLite: single-connection pool, thread-safe configuration
        from sqlalchemy.pool import StaticPool

        engine_kwargs["connect_args"] = {
            "check_same_thread": False,
            "timeout": 30,
        }
        engine_kwargs["poolclass"] = StaticPool

        engine = sa_create_engine(database_url, **engine_kwargs)

        @sa_event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA cache_size=-64000")
            cursor.close()

        logger.info("database.engine_created", backend="sqlite", url=database_url[:50])

    elif database_url.startswith("postgresql"):
        # PostgreSQL: connection pool sized for single-container deployment
        engine_kwargs["pool_size"] = 5
        engine_kwargs["max_overflow"] = 3
        engine_kwargs["pool_timeout"] = 30
        engine_kwargs["pool_recycle"] = 1800  # Recycle connections every 30 min
        engine_kwargs["pool_pre_ping"] = True  # Verify connections before use

        engine = sa_create_engine(database_url, **engine_kwargs)

        logger.info("database.engine_created", backend="postgresql", pool_size=5)

    else:
        # Fallback: unknown scheme — let SQLAlchemy figure it out
        engine = sa_create_engine(database_url, **engine_kwargs)
        logger.warning("database.engine_created", backend="unknown", url=database_url[:50])

    return engine


def get_engine(database_url: Optional[str] = None) -> Engine:
    """
    Get or create the shared application-wide database engine.

    On first call, creates the engine from the provided URL or settings.
    Subsequent calls return the same engine instance (singleton).
    """
    global _shared_engine

    if _shared_engine is not None:
        return _shared_engine

    with _engine_lock:
        # Double-checked locking
        if _shared_engine is not None:
            return _shared_engine

        if database_url is None:
            from config.settings import settings
            database_url = settings.database_url

        _shared_engine = create_db_engine(database_url)
        return _shared_engine


def get_session_factory(database_url: Optional[str] = None) -> sessionmaker:
    """
    Get or create the shared session factory bound to the application engine.
    """
    global _shared_session_factory

    if _shared_session_factory is not None:
        return _shared_session_factory

    with _engine_lock:
        if _shared_session_factory is not None:
            return _shared_session_factory

        engine = get_engine(database_url)
        _shared_session_factory = sessionmaker(bind=engine)
        return _shared_session_factory


def is_postgres(engine: Optional[Engine] = None) -> bool:
    """Check if the current engine is PostgreSQL."""
    if engine is None:
        engine = get_engine()
    return engine.dialect.name == "postgresql"


def is_sqlite(engine: Optional[Engine] = None) -> bool:
    """Check if the current engine is SQLite."""
    if engine is None:
        engine = get_engine()
    return engine.dialect.name == "sqlite"


def reset_engine() -> None:
    """Reset the shared engine (for testing or reconfiguration)."""
    global _shared_engine, _shared_session_factory
    with _engine_lock:
        if _shared_engine is not None:
            _shared_engine.dispose()
        _shared_engine = None
        _shared_session_factory = None

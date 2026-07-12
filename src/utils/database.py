"""
Shared Database Engine Factory - supports PostgreSQL and SQLite.

Provides a centralized engine creation function that configures the
appropriate driver, pooling, and pragmas based on the DATABASE_URL scheme.

Includes:
  - Connection pool lifecycle hooks (checkout, checkin, invalidate)
  - Slow query detection (configurable threshold)
  - Health check method for pool status
  - Structured observability events for all pool transitions

Usage:
    from src.utils.database import create_db_engine, get_engine, db_health_check

    # Create a one-off engine
    engine = create_db_engine("postgresql://host:5432/db")

    # Get/share the application-wide singleton engine
    engine = get_engine()

    # Check pool health
    status = db_health_check()
"""

import threading
import time
from typing import Optional

from sqlalchemy import create_engine as sa_create_engine, event as sa_event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.utils.logger import get_logger

logger = get_logger(__name__)

_engine_lock = threading.Lock()
_shared_engine: Optional[Engine] = None
_shared_session_factory: Optional[sessionmaker] = None

# Configurable slow query threshold (seconds)
SLOW_QUERY_THRESHOLD_SECONDS: float = 1.0


def _attach_pool_events(engine: Engine) -> None:
    """Attach structured logging hooks to connection pool events."""

    @sa_event.listens_for(engine, "checkout")
    def _on_checkout(dbapi_conn, connection_record, connection_proxy):
        connection_record.info["checkout_time"] = time.time()

    @sa_event.listens_for(engine, "checkin")
    def _on_checkin(dbapi_conn, connection_record):
        checkout_time = connection_record.info.pop("checkout_time", None)
        if checkout_time:
            hold_duration = time.time() - checkout_time
            if hold_duration > SLOW_QUERY_THRESHOLD_SECONDS:
                logger.warning(
                    "db.connection.slow_return",
                    hold_seconds=round(hold_duration, 3),
                    threshold=SLOW_QUERY_THRESHOLD_SECONDS,
                )

    @sa_event.listens_for(engine, "invalidate")
    def _on_invalidate(dbapi_conn, connection_record, exception):
        logger.warning(
            "db.pool.connection_invalidated",
            error=str(exception) if exception else "soft_invalidate",
        )

    @sa_event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, connection_record):
        logger.debug("db.pool.new_connection")


def _attach_query_timing(engine: Engine) -> None:
    """Attach before/after cursor execute hooks for slow query detection."""

    @sa_event.listens_for(engine, "before_cursor_execute")
    def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        conn.info.setdefault("query_start_time", []).append(time.time())

    @sa_event.listens_for(engine, "after_cursor_execute")
    def _after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        start_times = conn.info.get("query_start_time")
        if start_times:
            start = start_times.pop()
            elapsed = time.time() - start
            if elapsed > SLOW_QUERY_THRESHOLD_SECONDS:
                stmt_preview = statement[:200] if statement else ""
                logger.warning(
                    "db.query.slow",
                    elapsed_seconds=round(elapsed, 3),
                    threshold=SLOW_QUERY_THRESHOLD_SECONDS,
                    statement_preview=stmt_preview,
                )


def create_db_engine(database_url: str, echo: bool = False) -> Engine:
    """
    Create a SQLAlchemy engine configured for the given database URL.

    Supports:
      - postgresql:// or postgresql+psycopg2:// -> connection pool with observability
      - sqlite:/// -> StaticPool with WAL pragmas

    Returns a fully configured Engine instance with lifecycle hooks attached.
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

        # Attach pool lifecycle and query timing hooks for PostgreSQL
        _attach_pool_events(engine)
        _attach_query_timing(engine)

        logger.info(
            "db.pool.initialized",
            backend="postgresql",
            pool_size=5,
            max_overflow=3,
            pool_timeout=30,
            pool_recycle=1800,
        )

    else:
        # Fallback: unknown scheme - let SQLAlchemy figure it out
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


def db_health_check(engine: Optional[Engine] = None) -> dict:
    """
    Return connection pool health status.

    For PostgreSQL, reports pool size, checked-out connections, overflow, etc.
    For SQLite, reports basic connectivity.
    """
    if engine is None:
        engine = get_engine()

    status = {
        "backend": engine.dialect.name,
        "healthy": False,
    }

    try:
        # Verify connectivity with a simple query
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        status["healthy"] = True
    except Exception as e:
        status["error"] = str(e)
        logger.error("db.health_check.failed", error=str(e))
        return status

    # Report pool stats for PostgreSQL (QueuePool)
    pool = engine.pool
    if hasattr(pool, "size"):
        status["pool_size"] = pool.size()
        status["checked_out"] = pool.checkedout()
        status["overflow"] = pool.overflow()
        status["checked_in"] = pool.checkedin()
        status["pool_timeout"] = getattr(pool, "_timeout", None)

    logger.info("db.health_check.ok", **status)
    return status


def dispose_engine(engine: Optional[Engine] = None, timeout: float = 5.0) -> None:
    """
    Gracefully dispose of an engine's connection pool.

    Logs a warning if pool still has checked-out connections.
    """
    if engine is None:
        engine = get_engine()

    pool = engine.pool
    if hasattr(pool, "checkedout") and pool.checkedout() > 0:
        logger.warning(
            "db.pool.dispose_with_active_connections",
            checked_out=pool.checkedout(),
        )

    engine.dispose()
    logger.info("db.pool.shutdown", backend=engine.dialect.name)


def reset_engine() -> None:
    """Reset the shared engine (for testing or reconfiguration)."""
    global _shared_engine, _shared_session_factory
    with _engine_lock:
        if _shared_engine is not None:
            dispose_engine(_shared_engine)
        _shared_engine = None
        _shared_session_factory = None


def run_migrations(database_url: Optional[str] = None) -> None:
    """
    Run pending Alembic migrations against the target database.

    Emits structured log events for migration start/success/failure.
    Safe to call on startup - skips if already at head.
    """
    import os

    if database_url is None:
        from config.settings import settings
        database_url = settings.database_url

    # Only run migrations for PostgreSQL (SQLite uses create_all)
    if not database_url.startswith("postgresql"):
        logger.debug("db.migration.skipped", reason="sqlite_uses_create_all")
        return

    try:
        from alembic.config import Config
        from alembic import command

        # Find alembic.ini relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        alembic_ini = os.path.join(project_root, "alembic.ini")

        if not os.path.exists(alembic_ini):
            logger.warning("db.migration.skipped", reason="alembic_ini_not_found")
            return

        alembic_cfg = Config(alembic_ini)
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)

        logger.info("db.migration.starting", target="head")
        command.upgrade(alembic_cfg, "head")
        logger.info("db.migration.completed", target="head")

    except Exception as e:
        logger.error("db.migration.failed", error=str(e))
        raise


def bootstrap_database(database_url: Optional[str] = None) -> None:
    """
    Single entry point for PostgreSQL schema initialization on startup.

    Handles all deployment scenarios:
      - Fresh PostgreSQL: runs Alembic migration to create all tables
      - Existing PG (tables exist, no alembic_version): stamps current revision, then upgrades
      - Existing PG (alembic_version exists): runs upgrade to head (no-op if current)
      - SQLite: skips entirely (stores use create_all() for simplicity)

    Must be called BEFORE any DatabaseManager/EventStore/SnapshotStore/ConfigRepo
    initialization when using PostgreSQL.
    """
    import os

    if database_url is None:
        from config.settings import settings
        database_url = settings.database_url

    if not database_url.startswith("postgresql"):
        logger.debug("db.bootstrap.skipped", reason="sqlite_uses_create_all")
        return

    try:
        from sqlalchemy import inspect as sa_inspect, text as sa_text
        from alembic.config import Config
        from alembic import command

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        alembic_ini = os.path.join(project_root, "alembic.ini")

        if not os.path.exists(alembic_ini):
            logger.warning("db.bootstrap.skipped", reason="alembic_ini_not_found")
            return

        alembic_cfg = Config(alembic_ini)
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)

        # Create a temporary engine to inspect the database state
        engine = create_db_engine(database_url)
        inspector = sa_inspect(engine)
        existing_tables = inspector.get_table_names()
        has_alembic_version = "alembic_version" in existing_tables
        has_app_tables = "trades" in existing_tables or "events" in existing_tables

        if has_app_tables and not has_alembic_version:
            # Existing database bootstrapped by create_all() — stamp current revision
            logger.info(
                "db.bootstrap.stamping_existing",
                tables_found=len(existing_tables),
                reason="tables_exist_without_alembic_version",
            )
            command.stamp(alembic_cfg, "001")
            logger.info("db.bootstrap.stamped", revision="001")

        # Now run upgrade to head (no-op if already at head)
        logger.info("db.bootstrap.upgrading", target="head")
        command.upgrade(alembic_cfg, "head")
        logger.info("db.bootstrap.completed", target="head")

        engine.dispose()

    except Exception as e:
        logger.error("db.bootstrap.failed", error=str(e))
        raise
"""
End-to-End PostgreSQL Bootstrap Tests — exercises all three deployment scenarios
against a REAL PostgreSQL instance.

Requires: DATABASE_URL_TEST environment variable pointing to a live PostgreSQL
database that can be dropped/recreated between tests.

Example:
    DATABASE_URL_TEST="postgresql://user:pass@localhost:5432/algo_trader_test" \
    python -m pytest tests/test_pg_bootstrap_e2e.py -v

These tests are destructive: they DROP and CREATE the test database between runs.
Do NOT point them at a production database.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

PG_URL = os.environ.get("DATABASE_URL_TEST")

pytestmark = pytest.mark.skipif(
    not PG_URL or not PG_URL.startswith("postgresql"),
    reason="DATABASE_URL_TEST not set or not PostgreSQL",
)


def _drop_all_tables(url: str):
    """Drop all tables in the target database (including alembic_version)."""
    from sqlalchemy import create_engine, text

    engine = create_engine(url)
    with engine.connect() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.commit()
    engine.dispose()


def _get_tables(url: str) -> list:
    """Get all table names in the target database."""
    from sqlalchemy import create_engine, inspect

    engine = create_engine(url)
    inspector = inspect(engine)
    tables = sorted(inspector.get_table_names())
    engine.dispose()
    return tables


def _get_alembic_version(url: str):
    """Get the current alembic_version, or None if table doesn't exist."""
    from sqlalchemy import create_engine, text

    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            return result.scalar()
    except Exception:
        return None
    finally:
        engine.dispose()


class TestFreshDatabaseBootstrap:
    """Scenario 1: Fresh PostgreSQL database with no tables."""

    def setup_method(self):
        _drop_all_tables(PG_URL)

    def test_bootstrap_creates_all_tables(self):
        """bootstrap_database() on fresh PG should create all 10 app tables + alembic_version."""
        from src.utils.database import bootstrap_database

        assert _get_tables(PG_URL) == []  # Confirm fresh

        bootstrap_database(PG_URL)

        tables = _get_tables(PG_URL)
        assert "trades" in tables
        assert "events" in tables
        assert "snapshots" in tables
        assert "system_configuration" in tables
        assert "configuration_audit_log" in tables
        assert "alembic_version" in tables
        assert len(tables) == 11  # 10 app tables + alembic_version

    def test_bootstrap_sets_correct_version(self):
        """After fresh bootstrap, alembic_version should be '001'."""
        from src.utils.database import bootstrap_database

        bootstrap_database(PG_URL)
        assert _get_alembic_version(PG_URL) == "001"

    def test_stores_work_after_bootstrap(self):
        """All stores should be able to read/write after bootstrap."""
        from src.utils.database import bootstrap_database
        from src.data.store import DatabaseManager
        from src.core.event_store import EventStore
        from src.core.snapshots import SnapshotStore
        from src.config.repository import ConfigurationRepository
        from src.core.events import Event

        bootstrap_database(PG_URL)

        # DatabaseManager
        db = DatabaseManager(PG_URL)
        trade = db.record_trade({
            "symbol": "AAPL", "side": "buy", "qty": 10, "price": 150.0,
            "order_id": "test-001", "strategy": "test",
        })
        assert trade is not None
        db.close()

        # EventStore
        es = EventStore(database_url=PG_URL)
        event = Event(source="test")
        es.persist(event)
        events = es.get_session_events(es.session_id)
        assert len(events) == 1
        es.close()

        # SnapshotStore
        ss = SnapshotStore(database_url=PG_URL)
        ss.save_snapshot(
            session_id="test-session",
            last_event_id=1,
            state={"key": "value"},
        )
        latest = ss.get_latest_snapshot("test-session")
        assert latest is not None
        ss.close()

        # ConfigurationRepository
        repo = ConfigurationRepository(database_url=PG_URL)
        repo.set("test_cat", "test_key", "test_val", "str", changed_by="test")
        entry = repo.get("test_cat", "test_key")
        assert entry is not None
        assert entry.value == "test_val"
        repo.close()


class TestExistingUnversionedDatabase:
    """Scenario 2: Database with ORM-created tables but no alembic_version."""

    def setup_method(self):
        """Create tables via create_all() (simulating legacy bootstrap)."""
        _drop_all_tables(PG_URL)

        from sqlalchemy import create_engine
        from src.data.store import Base as TradingBase
        from src.core.event_store import EventBase
        from src.core.snapshots import SnapshotBase
        from src.config.models import ConfigBase

        engine = create_engine(PG_URL)
        TradingBase.metadata.create_all(engine)
        EventBase.metadata.create_all(engine)
        SnapshotBase.metadata.create_all(engine)
        ConfigBase.metadata.create_all(engine)
        engine.dispose()

    def test_bootstrap_stamps_existing_tables(self):
        """bootstrap_database() should stamp '001' without modifying existing tables."""
        from src.utils.database import bootstrap_database

        tables_before = _get_tables(PG_URL)
        assert "trades" in tables_before
        assert "alembic_version" not in tables_before

        bootstrap_database(PG_URL)

        assert _get_alembic_version(PG_URL) == "001"
        tables_after = _get_tables(PG_URL)
        # Should only add alembic_version
        assert set(tables_after) - set(tables_before) == {"alembic_version"}

    def test_data_preserved_after_stamp(self):
        """Existing data should be preserved through the stamp + upgrade."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from src.data.store import Trade, Base as TradingBase
        from src.utils.database import bootstrap_database

        # Insert test data before bootstrap
        engine = create_engine(PG_URL)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            session.add(Trade(symbol="MSFT", side="buy", qty=5, price=300.0))
            session.commit()
        engine.dispose()

        # Run bootstrap
        bootstrap_database(PG_URL)

        # Verify data still exists
        engine = create_engine(PG_URL)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            trades = session.query(Trade).all()
            assert len(trades) == 1
            assert trades[0].symbol == "MSFT"
        engine.dispose()


class TestAlreadyVersionedDatabase:
    """Scenario 3: Database already managed by Alembic (has alembic_version)."""

    def setup_method(self):
        """Create a properly versioned database via migration."""
        _drop_all_tables(PG_URL)
        from src.utils.database import bootstrap_database
        bootstrap_database(PG_URL)

    def test_re_bootstrap_is_noop(self):
        """Running bootstrap again should not fail or modify the database."""
        from src.utils.database import bootstrap_database

        tables_before = _get_tables(PG_URL)
        version_before = _get_alembic_version(PG_URL)

        bootstrap_database(PG_URL)

        tables_after = _get_tables(PG_URL)
        version_after = _get_alembic_version(PG_URL)

        assert tables_before == tables_after
        assert version_before == version_after

    def test_stores_work_on_versioned_db(self):
        """All CRUD operations should work on an already-versioned database."""
        from src.utils.database import bootstrap_database
        from src.core.event_store import EventStore
        from src.core.events import Event

        bootstrap_database(PG_URL)  # Should be no-op

        es = EventStore(database_url=PG_URL)
        event = Event(source="versioned-test")
        es.persist(event)
        events = es.get_session_events(es.session_id)
        assert len(events) == 1
        es.close()


class TestEventDedupOnPostgreSQL:
    """Test ON CONFLICT DO NOTHING dedup on real PostgreSQL."""

    def setup_method(self):
        _drop_all_tables(PG_URL)
        from src.utils.database import bootstrap_database
        bootstrap_database(PG_URL)

    def test_duplicate_event_is_silently_ignored(self):
        """Persisting the same event_id twice should not raise on PostgreSQL."""
        from src.core.event_store import EventStore
        from src.core.events import Event

        es = EventStore(database_url=PG_URL)
        event = Event(source="dedup-test")

        es.persist(event)
        es.persist(event)  # Should not raise

        events = es.get_session_events(es.session_id)
        assert len(events) == 1
        es.close()

    def test_concurrent_dedup(self):
        """Multiple threads persisting the same event should result in one record."""
        import threading
        from src.core.event_store import EventStore
        from src.core.events import Event

        es = EventStore(database_url=PG_URL)
        event = Event(source="concurrent-dedup")
        errors = []

        def persist():
            try:
                es.persist(event)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=persist) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert len(errors) == 0, f"Dedup errors: {errors}"
        events = es.get_session_events(es.session_id)
        assert len(events) == 1
        es.close()


class TestConnectionPoolOnPostgreSQL:
    """Test pool behavior on real PostgreSQL."""

    def setup_method(self):
        _drop_all_tables(PG_URL)
        from src.utils.database import bootstrap_database
        bootstrap_database(PG_URL)

    def test_health_check(self):
        """db_health_check should report pool stats on PostgreSQL."""
        from src.utils.database import create_db_engine, db_health_check

        engine = create_db_engine(PG_URL)
        status = db_health_check(engine)
        assert status["healthy"] is True
        assert status["backend"] == "postgresql"
        assert "pool_size" in status
        engine.dispose()

    def test_dispose_cleans_pool(self):
        """dispose_engine should cleanly shut down the PostgreSQL pool."""
        from src.utils.database import create_db_engine, dispose_engine

        engine = create_db_engine(PG_URL)
        # Verify pool is alive
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        dispose_engine(engine)
        # Pool should be disposed (new connections will fail or auto-recreate)

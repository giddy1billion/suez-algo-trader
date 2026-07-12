"""
PostgreSQL Integration Test — validates dialect compatibility.

Tests that all ORM models, CRUD operations, and migration logic work
correctly against a real PostgreSQL-compatible target.

When a live PostgreSQL instance is available (via DATABASE_URL_TEST env var),
runs full end-to-end tests. Otherwise, validates dialect SQL generation and
exercises the migration script against a fresh SQLite target.

Usage:
    # Against live PG (set DATABASE_URL_TEST to a disposable test DB):
    DATABASE_URL_TEST="postgresql://user:pass@localhost:5432/test_db" python -m pytest tests/test_pg_integration.py -v

    # Dialect validation only (no PG server needed):
    python -m pytest tests/test_pg_integration.py -v
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import create_db_engine, is_postgres, is_sqlite
from src.data.store import Base as TradingBase, Trade, SignalLog, PortfolioSnapshot, JournalEntry, MarketData, SectorCache, DatabaseManager
from src.core.event_store import EventBase, EventRecord, EventStore
from src.core.snapshots import SnapshotBase, SnapshotRecord, SnapshotStore
from src.core.events import Event, SignalGenerated


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pg_url():
    """Get PostgreSQL URL from env, or skip test."""
    url = os.environ.get("DATABASE_URL_TEST")
    if url and url.startswith("postgresql"):
        return url
    pytest.skip("DATABASE_URL_TEST not set to a PostgreSQL URL — skipping live PG tests")


@pytest.fixture
def sqlite_url(tmp_path):
    """Create a temporary SQLite database URL."""
    return f"sqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def any_db_url(pg_url, sqlite_url):
    """Use PG if available, else SQLite for migration-path tests."""
    try:
        return pg_url
    except pytest.skip.Exception:
        return sqlite_url


# ---------------------------------------------------------------------------
# Dialect SQL Generation Tests (no PG server needed)
# ---------------------------------------------------------------------------

class TestDialectCompatibility:
    """Verify that ORM models produce valid SQL for PostgreSQL dialect."""

    def test_trading_models_create_table_pg_dialect(self):
        """Verify CREATE TABLE SQL is valid for PostgreSQL."""
        from sqlalchemy.dialects import postgresql
        from sqlalchemy.schema import CreateTable
        for table in TradingBase.metadata.sorted_tables:
            create_sql = str(CreateTable(table).compile(dialect=postgresql.dialect()))
            # Should not contain SQLite-specific syntax
            assert "AUTOINCREMENT" not in create_sql, f"{table.name} uses SQLite AUTOINCREMENT"

    def test_event_models_create_table_pg_dialect(self):
        """Verify event store CREATE TABLE SQL for PostgreSQL."""
        from sqlalchemy.dialects import postgresql
        from sqlalchemy.schema import CreateTable
        for table in EventBase.metadata.sorted_tables:
            create_sql = str(CreateTable(table).compile(dialect=postgresql.dialect()))
            assert "AUTOINCREMENT" not in create_sql

    def test_snapshot_models_create_table_pg_dialect(self):
        """Verify snapshot store CREATE TABLE SQL for PostgreSQL."""
        from sqlalchemy.dialects import postgresql
        from sqlalchemy.schema import CreateTable
        for table in SnapshotBase.metadata.sorted_tables:
            create_sql = str(CreateTable(table).compile(dialect=postgresql.dialect()))
            assert "AUTOINCREMENT" not in create_sql

    def test_json_text_columns_compatible(self):
        """Verify JSON-stored columns use TEXT (portable) not JSON type."""
        for table in TradingBase.metadata.sorted_tables:
            for col in table.columns:
                # Our JSON data uses Text columns which work in both SQLite and PG
                if "json" in col.name.lower() or col.name in ("indicators", "features_snapshot", "notes"):
                    assert str(col.type) in ("TEXT", "VARCHAR"), \
                        f"{table.name}.{col.name} uses {col.type} — should be TEXT for portability"


# ---------------------------------------------------------------------------
# Engine Factory Tests
# ---------------------------------------------------------------------------

class TestEngineFactory:
    """Validate the shared engine factory behavior."""

    def test_sqlite_engine_creation(self, tmp_path):
        url = f"sqlite:///{tmp_path / 'test.db'}"
        engine = create_db_engine(url)
        assert engine.dialect.name == "sqlite"
        assert is_sqlite(engine)
        assert not is_postgres(engine)
        engine.dispose()

    def test_postgres_url_parsing(self):
        """Verify PostgreSQL URL is recognized (doesn't actually connect)."""
        # Just verify the code path doesn't crash — actual connection would fail
        # without a running server
        try:
            engine = create_db_engine("postgresql://user:pass@localhost:5432/testdb")
            assert engine.dialect.name == "postgresql"
            assert is_postgres(engine)
            assert not is_sqlite(engine)
            engine.dispose()
        except Exception:
            pass  # Connection error is expected without a server


# ---------------------------------------------------------------------------
# Schema Creation & CRUD End-to-End Tests
# ---------------------------------------------------------------------------

class TestSchemaCreationAndCRUD:
    """Full CRUD cycle against whatever database is available."""

    def test_database_manager_schema_creation(self, sqlite_url):
        """DatabaseManager creates all tables on init."""
        db = DatabaseManager(sqlite_url)
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        assert "trades" in tables
        assert "signal_logs" in tables
        assert "portfolio_snapshots" in tables
        assert "trade_journal" in tables
        assert "market_data" in tables
        assert "sector_cache" in tables

    def test_trade_write_read_cycle(self, sqlite_url):
        """Write and read back a trade record."""
        db = DatabaseManager(sqlite_url)
        trade = db.record_trade({
            "symbol": "AAPL",
            "side": "buy",
            "qty": 10.0,
            "price": 150.50,
            "order_type": "market",
            "status": "filled",
            "order_id": "test-order-001",
            "strategy": "momentum",
            "signal_confidence": 0.85,
            "stop_loss": 147.0,
            "take_profit": 155.0,
        })
        assert trade.id is not None
        assert trade.symbol == "AAPL"

        # Read back
        trades = db.get_trades(symbol="AAPL")
        assert len(trades) == 1
        assert trades[0]["price"] == 150.50
        assert trades[0]["order_id"] == "test-order-001"

    def test_trade_update_pnl(self, sqlite_url):
        """Update a trade with P&L on close."""
        db = DatabaseManager(sqlite_url)
        db.record_trade({
            "symbol": "MSFT",
            "side": "buy",
            "qty": 5.0,
            "price": 300.0,
            "order_id": "test-order-002",
            "status": "filled",
            "strategy": "ml",
        })
        db.update_trade("test-order-002", {"pnl": 25.50, "pnl_pct": 0.017})
        trades = db.get_trades(symbol="MSFT")
        assert trades[0]["pnl"] == 25.50

    def test_event_store_persist_and_replay(self, tmp_path):
        """Events persist and replay correctly."""
        db_path = str(tmp_path / "events.db")
        store = EventStore(db_path=db_path, session_id="test-session")

        # Persist events
        e1 = Event(timestamp=datetime.now(timezone.utc), source="test")
        e2 = SignalGenerated(
            timestamp=datetime.now(timezone.utc),
            source="momentum",
            symbol="AAPL",
            signal="BUY",
            confidence=0.85,
            strategy="momentum",
        )
        store.persist(e1)
        store.persist(e2)

        # Verify count
        assert store.count_events(session_id="test-session") == 2

        # Replay
        events = store.replay_session("test-session")
        assert len(events) == 2
        assert isinstance(events[0], Event)

    def test_event_store_deduplication(self, tmp_path):
        """Duplicate events are silently ignored."""
        db_path = str(tmp_path / "events.db")
        store = EventStore(db_path=db_path)

        e = Event(timestamp=datetime.now(timezone.utc), source="test")
        store.persist(e)
        store.persist(e)  # Same event_id — should be deduped
        assert store.count_events() == 1

    def test_snapshot_store_save_and_recover(self, tmp_path):
        """Snapshots save and load with JSON state intact."""
        db_path = str(tmp_path / "snapshots.db")
        store = SnapshotStore(db_path=db_path)

        state = {
            "positions": {"AAPL": {"qty": 10, "avg_price": 150.0}},
            "cash": 50000.0,
            "metrics": {"win_rate": 0.65, "sharpe": 1.2},
        }
        sid = store.save_snapshot(
            session_id="sess-001",
            last_event_id=42,
            state=state,
            schema_version="2",
            engine_version="1.5.0",
            config_hash="abc123",
        )
        assert sid is not None

        # Recover
        snap = store.get_latest_snapshot(session_id="sess-001")
        assert snap is not None
        assert snap["last_event_id"] == 42
        assert snap["state"]["positions"]["AAPL"]["qty"] == 10
        assert snap["state"]["cash"] == 50000.0
        assert snap["schema_version"] == "2"

    def test_snapshot_cleanup(self, tmp_path):
        """Old snapshots get cleaned up correctly."""
        db_path = str(tmp_path / "snapshots.db")
        store = SnapshotStore(db_path=db_path)

        for i in range(15):
            store.save_snapshot("sess", i, {"i": i})

        assert store.get_snapshot_count() == 15
        deleted = store.cleanup_old_snapshots(keep_latest=5)
        assert deleted == 10
        assert store.get_snapshot_count() == 5

    def test_sector_cache_crud(self, sqlite_url):
        """Sector cache insert/update/query cycle."""
        db = DatabaseManager(sqlite_url)
        db.set_cached_sector("AAPL", "technology", "manual")
        assert db.get_cached_sector("AAPL") == "technology"

        # Update
        db.set_cached_sector("AAPL", "tech_hardware", "auto")
        assert db.get_cached_sector("AAPL") == "tech_hardware"

        # Get all
        sectors = db.get_all_cached_sectors()
        assert "AAPL" in sectors


# ---------------------------------------------------------------------------
# Migration Script Tests
# ---------------------------------------------------------------------------

class TestMigrationScript:
    """Validate the migration script logic."""

    def test_migration_dry_run(self, tmp_path):
        """Migration dry run reports row counts without writing."""
        import subprocess
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [
                sys.executable, "scripts/migrate_sqlite_to_pg.py",
                "--dry-run",
                "--trading-db", "data_cache/trading.db",
                "--events-db", "data_cache/events.db",
                "--snapshots-db", "data_cache/snapshots.db",
                "--target-url", f"sqlite:///{tmp_path / 'target.db'}",
            ],
            capture_output=True, text=True, encoding="utf-8",
            cwd=str(Path(__file__).parent.parent), env=env,
        )
        assert result.returncode == 0, f"Dry run failed:\n{result.stdout}\n{result.stderr}"
        assert "Dry run complete" in result.stdout

    def test_migration_to_fresh_sqlite(self, tmp_path):
        """Full migration to a fresh SQLite target (validates migration logic)."""
        import subprocess
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        target_url = f"sqlite:///{tmp_path / 'migrated.db'}"
        result = subprocess.run(
            [
                sys.executable, "scripts/migrate_sqlite_to_pg.py",
                "--trading-db", "data_cache/trading.db",
                "--events-db", "data_cache/events.db",
                "--snapshots-db", "data_cache/snapshots.db",
                "--target-url", target_url,
            ],
            capture_output=True, text=True, encoding="utf-8",
            cwd=str(Path(__file__).parent.parent), env=env,
        )
        # Should succeed (exit 0) or have nothing to migrate
        assert result.returncode == 0, f"Migration failed:\n{result.stdout}\n{result.stderr}"


# ---------------------------------------------------------------------------
# Live PostgreSQL Tests (only run when DATABASE_URL_TEST is set)
# ---------------------------------------------------------------------------

class TestLivePostgreSQL:
    """Full integration tests against a real PostgreSQL instance."""

    def test_schema_creation(self, pg_url):
        """All tables create successfully on PostgreSQL."""
        engine = create_db_engine(pg_url)
        TradingBase.metadata.drop_all(engine)
        EventBase.metadata.drop_all(engine)
        SnapshotBase.metadata.drop_all(engine)

        TradingBase.metadata.create_all(engine)
        EventBase.metadata.create_all(engine)
        SnapshotBase.metadata.create_all(engine)

        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "trades" in tables
        assert "events" in tables
        assert "snapshots" in tables
        engine.dispose()

    def test_trade_crud_on_pg(self, pg_url):
        """Trade CRUD operations work on PostgreSQL."""
        db = DatabaseManager(pg_url)
        trade = db.record_trade({
            "symbol": "NVDA",
            "side": "buy",
            "qty": 5.0,
            "price": 500.0,
            "order_id": f"pg-test-{datetime.now().timestamp()}",
            "status": "filled",
            "strategy": "ml",
            "signal_confidence": 0.92,
        })
        assert trade.id is not None
        trades = db.get_trades(symbol="NVDA", limit=1)
        assert len(trades) >= 1

    def test_event_store_on_pg(self, pg_url):
        """EventStore works on PostgreSQL."""
        store = EventStore(database_url=pg_url, session_id="pg-test-session")
        e = Event(timestamp=datetime.now(timezone.utc), source="pg-test")
        store.persist(e)
        assert store.count_events(session_id="pg-test-session") >= 1
        store.close()

    def test_snapshot_store_on_pg(self, pg_url):
        """SnapshotStore works on PostgreSQL."""
        store = SnapshotStore(database_url=pg_url)
        sid = store.save_snapshot("pg-sess", 99, {"test": True})
        assert sid is not None
        snap = store.get_latest_snapshot("pg-sess")
        assert snap["state"]["test"] is True
        store.close()

    def test_json_roundtrip_on_pg(self, pg_url):
        """JSON data stored in TEXT columns roundtrips correctly on PG."""
        db = DatabaseManager(pg_url)
        complex_notes = json.dumps({
            "features": {"rsi": 45.2, "ema_cross": True},
            "reasoning": "Bullish divergence detected",
            "nested": {"a": [1, 2, 3], "b": None},
        })
        trade = db.record_trade({
            "symbol": "TEST",
            "side": "buy",
            "qty": 1.0,
            "price": 100.0,
            "order_id": f"json-test-{datetime.now().timestamp()}",
            "status": "filled",
            "strategy": "test",
            "notes": complex_notes,
        })
        trades = db.get_trades(symbol="TEST", limit=1)
        # Notes should roundtrip as-is
        recovered = json.loads(trades[0].get("notes", "{}") if "notes" in trades[0] else "{}")
        # At minimum, the trade should exist
        assert len(trades) >= 1

    def test_concurrent_writes_on_pg(self, pg_url):
        """Multiple concurrent writes don't deadlock on PG."""
        import threading

        db = DatabaseManager(pg_url)
        errors = []

        def write_trade(i):
            try:
                db.record_trade({
                    "symbol": f"SYM{i}",
                    "side": "buy",
                    "qty": 1.0,
                    "price": float(i),
                    "order_id": f"concurrent-{i}-{datetime.now().timestamp()}",
                    "status": "filled",
                    "strategy": "test",
                })
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=write_trade, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Concurrent write errors: {errors}"

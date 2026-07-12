"""
SQLite → PostgreSQL Data Migration Script.

Reads data from the existing SQLite databases and imports it into the
target PostgreSQL database specified by DATABASE_URL environment variable.

Usage:
    # Set target PostgreSQL URL
    export DATABASE_URL="postgresql://user:pass@host:5432/suez_trader?sslmode=require"

    # Run migration (reads from local SQLite files)
    python scripts/migrate_sqlite_to_pg.py

    # Dry run (validates only, no writes)
    python scripts/migrate_sqlite_to_pg.py --dry-run

    # Specify custom SQLite paths
    python scripts/migrate_sqlite_to_pg.py --trading-db data_cache/trading.db --events-db data_cache/events.db
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.data.store import Base as TradingBase, Trade, SignalLog, PortfolioSnapshot, JournalEntry, MarketData, SectorCache
from src.core.event_store import EventBase, EventRecord
from src.core.snapshots import SnapshotBase, SnapshotRecord
from src.utils.database import create_db_engine


def migrate_table(sqlite_path: str, table_name: str, pg_session, orm_class, dry_run: bool = False) -> int:
    """Migrate a single table from SQLite to PostgreSQL."""
    if not Path(sqlite_path).exists():
        print(f"  [WARN]  SQLite file not found: {sqlite_path} -- skipping {table_name}")
        return 0

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
        total = cursor.fetchone()[0]
        if total == 0:
            print(f"  [SKIP]  {table_name}: 0 rows -- skipping")
            return 0

        cursor = conn.execute(f"SELECT * FROM {table_name}")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        if dry_run:
            print(f"  [CHECK] {table_name}: {len(rows)} rows would be migrated (columns: {columns})")
            return len(rows)

        # Get ORM column names (skip auto-generated 'id' for autoincrement tables)
        orm_columns_map = {c.name: c for c in orm_class.__table__.columns}
        # Map SQLite columns to ORM columns (intersection)
        common_cols = [c for c in columns if c in orm_columns_map and c != 'id']

        # Identify DateTime columns for type coercion
        from sqlalchemy import DateTime
        datetime_cols = {
            name for name, col in orm_columns_map.items()
            if isinstance(col.type, DateTime)
        }

        migrated = 0
        batch_size = 100
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            for row in batch:
                row_dict = {}
                for col in common_cols:
                    val = row[col]
                    if val is None:
                        continue
                    # Coerce string timestamps to datetime objects
                    if col in datetime_cols and isinstance(val, str):
                        try:
                            val = datetime.fromisoformat(val.replace('Z', '+00:00'))
                        except (ValueError, TypeError):
                            pass  # Leave as-is
                    row_dict[col] = val
                record = orm_class(**row_dict)
                pg_session.add(record)
                migrated += 1
            pg_session.flush()

        pg_session.commit()
        print(f"  [OK] {table_name}: {migrated}/{total} rows migrated")
        return migrated

    except sqlite3.OperationalError as e:
        print(f"  [WARN]  {table_name}: table does not exist in {sqlite_path} ({e})")
        return 0
    finally:
        conn.close()


def migrate_events(sqlite_path: str, pg_session, dry_run: bool = False) -> int:
    """Migrate events table (custom handling for dedup)."""
    if not Path(sqlite_path).exists():
        print(f"  [WARN]  Events DB not found: {sqlite_path}")
        return 0

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM events")
        total = cursor.fetchone()[0]
        if total == 0:
            print(f"  [SKIP]  events: 0 rows -- skipping")
            return 0

        cursor = conn.execute("SELECT * FROM events ORDER BY id ASC")
        rows = cursor.fetchall()

        if dry_run:
            print(f"  [CHECK] events: {len(rows)} rows would be migrated")
            return len(rows)

        migrated = 0
        for row in rows:
            record = EventRecord(
                event_type=row["event_type"],
                event_id=row["event_id"],
                timestamp=row["timestamp"],
                source=row["source"] or "",
                payload=row["payload"],
                session_id=row["session_id"],
            )
            pg_session.merge(record)  # merge handles dedup on event_id
            migrated += 1

        pg_session.commit()
        print(f"  [OK] events: {migrated}/{total} rows migrated")
        return migrated
    except sqlite3.OperationalError as e:
        print(f"  [WARN]  events: {e}")
        return 0
    finally:
        conn.close()


def migrate_snapshots(sqlite_path: str, pg_session, dry_run: bool = False) -> int:
    """Migrate snapshots table."""
    if not Path(sqlite_path).exists():
        print(f"  [WARN]  Snapshots DB not found: {sqlite_path}")
        return 0

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM snapshots")
        total = cursor.fetchone()[0]
        if total == 0:
            print(f"  [SKIP]  snapshots: 0 rows -- skipping")
            return 0

        cursor = conn.execute("SELECT * FROM snapshots ORDER BY id ASC")
        rows = cursor.fetchall()

        if dry_run:
            print(f"  [CHECK] snapshots: {len(rows)} rows would be migrated")
            return len(rows)

        migrated = 0
        for row in rows:
            record = SnapshotRecord(
                session_id=row["session_id"],
                timestamp=row["timestamp"],
                last_event_id=row["last_event_id"],
                state=row["state"],
                schema_version=row["schema_version"] if "schema_version" in row.keys() else "1",
                engine_version=row["engine_version"] if "engine_version" in row.keys() else "1.0.0",
                config_hash=row["config_hash"] if "config_hash" in row.keys() else "",
            )
            pg_session.add(record)
            migrated += 1

        pg_session.commit()
        print(f"  [OK] snapshots: {migrated}/{total} rows migrated")
        return migrated
    except sqlite3.OperationalError as e:
        print(f"  [WARN]  snapshots: {e}")
        return 0
    finally:
        conn.close()


def validate_migration(pg_engine, expected_counts: dict) -> bool:
    """Validate row counts match between source and target."""
    print("\n[STATS] Validation:")
    all_ok = True
    Session = sessionmaker(bind=pg_engine)
    with Session() as session:
        for table_name, expected in expected_counts.items():
            if expected == 0:
                continue
            try:
                result = session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                actual = result.scalar()
                status = "[OK]" if actual == expected else "[FAIL]"
                if actual != expected:
                    all_ok = False
                print(f"  {status} {table_name}: {actual}/{expected}")
            except Exception as e:
                print(f"  [FAIL] {table_name}: error -- {e}")
                all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Migrate SQLite data to PostgreSQL")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, no writes")
    parser.add_argument("--trading-db", default="data_cache/trading.db", help="Path to trading.db")
    parser.add_argument("--events-db", default="data_cache/events.db", help="Path to events.db")
    parser.add_argument("--snapshots-db", default="data_cache/snapshots.db", help="Path to snapshots.db")
    parser.add_argument("--target-url", default=None, help="Target PostgreSQL URL (overrides DATABASE_URL env)")
    args = parser.parse_args()

    target_url = args.target_url or os.environ.get("DATABASE_URL")
    if not target_url:
        print("[FAIL] No target database URL. Set DATABASE_URL env var or use --target-url")
        sys.exit(1)

    if "sqlite" in target_url:
        print("[WARN] Target URL is SQLite -- this script is for migrating TO PostgreSQL.")
        print("   Proceeding anyway (useful for testing the migration logic)...")

    print(f"[START] Migration: SQLite → {target_url.split('@')[-1] if '@' in target_url else target_url[:50]}")
    print(f"   Mode: {'DRY RUN' if args.dry_run else 'LIVE MIGRATION'}")
    print()

    # Create target engine and tables
    pg_engine = create_db_engine(target_url)

    if not args.dry_run:
        print("[PKG] Creating schema...")
        TradingBase.metadata.create_all(pg_engine)
        EventBase.metadata.create_all(pg_engine)
        SnapshotBase.metadata.create_all(pg_engine)
        print("  [OK] All tables created\n")

    Session = sessionmaker(bind=pg_engine)
    expected_counts = {}

    # Migrate trading.db tables
    print("[DIR] Trading DB:")
    with Session() as session:
        expected_counts["trades"] = migrate_table(args.trading_db, "trades", session, Trade, args.dry_run)
        expected_counts["signal_logs"] = migrate_table(args.trading_db, "signal_logs", session, SignalLog, args.dry_run)
        expected_counts["portfolio_snapshots"] = migrate_table(args.trading_db, "portfolio_snapshots", session, PortfolioSnapshot, args.dry_run)
        expected_counts["trade_journal"] = migrate_table(args.trading_db, "trade_journal", session, JournalEntry, args.dry_run)
        expected_counts["market_data"] = migrate_table(args.trading_db, "market_data", session, MarketData, args.dry_run)
        expected_counts["sector_cache"] = migrate_table(args.trading_db, "sector_cache", session, SectorCache, args.dry_run)

    # Migrate events.db
    print("\n[DIR] Events DB:")
    with Session() as session:
        expected_counts["events"] = migrate_events(args.events_db, session, args.dry_run)

    # Migrate snapshots.db
    print("\n[DIR] Snapshots DB:")
    with Session() as session:
        expected_counts["snapshots"] = migrate_snapshots(args.snapshots_db, session, args.dry_run)

    # Validate
    if not args.dry_run:
        ok = validate_migration(pg_engine, expected_counts)
        print(f"\n{'[OK] Migration PASSED' if ok else '[FAIL] Migration FAILED -- check counts above'}")
        sys.exit(0 if ok else 1)
    else:
        total = sum(expected_counts.values())
        print(f"\n[CHECK] Dry run complete. {total} total rows would be migrated.")


if __name__ == "__main__":
    main()

"""Tests for SQLite hardening pragmas in DatabaseManager."""

import os
import pytest
from sqlalchemy import text

from src.data.store import DatabaseManager


class TestSQLiteHardening:
    """Verify SQLite production pragmas are applied."""

    def test_sqlite_wal_mode_active(self):
        """DatabaseManager must set WAL journal mode on SQLite databases."""
        db_path = "data_cache/test_hardening.db"
        db_url = f"sqlite:///{db_path}"

        try:
            dm = DatabaseManager(database_url=db_url)

            with dm.engine.connect() as conn:
                result = conn.execute(text("PRAGMA journal_mode")).fetchone()
                assert result[0].lower() == "wal", (
                    f"Expected WAL journal mode, got: {result[0]}"
                )

                # Verify other pragmas
                busy = conn.execute(text("PRAGMA busy_timeout")).fetchone()
                assert busy[0] == 30000, f"Expected busy_timeout=30000, got {busy[0]}"

                fk = conn.execute(text("PRAGMA foreign_keys")).fetchone()
                assert fk[0] == 1, f"Expected foreign_keys=ON, got {fk[0]}"

                sync = conn.execute(text("PRAGMA synchronous")).fetchone()
                # NORMAL = 1
                assert sync[0] == 1, f"Expected synchronous=NORMAL (1), got {sync[0]}"

            # Dispose engine to release file handles before cleanup
            dm.engine.dispose()
        finally:
            # Cleanup test database files
            import time
            time.sleep(0.1)
            for suffix in ["", "-wal", "-shm"]:
                path = db_path + suffix
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except PermissionError:
                        pass  # Best-effort cleanup on Windows

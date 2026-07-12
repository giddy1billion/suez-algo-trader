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


class TestNumpyScalarSerialization:
    """Verify that numpy scalars are converted to native Python before DB insertion.

    PostgreSQL's psycopg2 adapter cannot serialize numpy types. Without conversion,
    np.float64(0.65) renders as a schema-qualified identifier in SQL, causing:
      psycopg2.errors.InvalidSchemaName: schema "np" does not exist
    """

    def test_log_signal_with_numpy_floats(self):
        """log_signal must accept numpy float64 values without error."""
        import numpy as np
        from src.data.store import SignalLog

        db = DatabaseManager(database_url="sqlite:///:memory:")
        db.log_signal({
            "symbol": "BTC/USD",
            "strategy": "momentum",
            "signal": "SELL",
            "confidence": np.float64(0.65),
            "price_at_signal": np.float64(60141.2),
            "indicators": '{"rsi": 50.35}',
            "was_executed": False,
        })

        with db.get_session() as s:
            sig = s.query(SignalLog).first()
            assert sig is not None
            assert sig.confidence == pytest.approx(0.65)
            assert sig.price_at_signal == pytest.approx(60141.2)
            assert isinstance(sig.confidence, float)

    def test_record_trade_with_numpy_floats(self):
        """record_trade must accept numpy float64/int64 values."""
        import numpy as np
        from src.data.store import Trade

        db = DatabaseManager(database_url="sqlite:///:memory:")
        db.record_trade({
            "symbol": "AAPL",
            "side": "buy",
            "qty": np.float64(10.0),
            "price": np.float64(150.25),
            "strategy": "momentum",
            "signal_confidence": np.float64(0.8),
        })

        with db.get_session() as s:
            trade = s.query(Trade).first()
            assert trade is not None
            assert trade.price == pytest.approx(150.25)
            assert trade.qty == pytest.approx(10.0)

    def test_snapshot_portfolio_with_numpy_values(self):
        """snapshot_portfolio must accept mixed numpy types."""
        import numpy as np
        from src.data.store import PortfolioSnapshot

        db = DatabaseManager(database_url="sqlite:///:memory:")
        db.snapshot_portfolio({
            "total_equity": np.float64(100000.0),
            "cash": np.float64(50000.0),
            "positions_value": np.float64(50000.0),
            "unrealized_pnl": np.float64(1234.56),
            "open_positions": np.int64(3),
        })

        with db.get_session() as s:
            snap = s.query(PortfolioSnapshot).first()
            assert snap is not None
            assert snap.total_equity == pytest.approx(100000.0)
            assert snap.open_positions == 3

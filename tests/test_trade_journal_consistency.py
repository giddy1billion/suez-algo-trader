"""
Regression & integration tests for /trades ↔ /journalstats consistency.

Covers:
  - Open trades journaling
  - Closed trades journaling with exit data
  - Journal summary analytics calculations
  - Confidence bucket analysis
  - Model version comparison analytics
  - Replay/recovery scenarios (orphaned journal entries)
  - Edge cases (empty DB, single trade, boundary conditions)

See docs/production-readiness-audit.md for the full audit report.
"""

import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.data.store import Base, Trade, JournalEntry, DatabaseManager
from src.data.journal import TradeJournal


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    """In-memory SQLite DatabaseManager for isolated tests."""
    url = f"sqlite:///{tmp_path / 'test.db'}"
    mgr = DatabaseManager(url)
    return mgr


@pytest.fixture
def journal(db):
    """TradeJournal backed by the test DB."""
    return TradeJournal(db)


def _make_trade(db, *, symbol="AAPL", side="buy", qty=10, price=150.0,
                status="filled", pnl=None, pnl_pct=None, strategy="momentum",
                order_id=None, closed_at=None, created_at=None):
    """Helper: insert a trade into the trades table and return it."""
    data = {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": price,
        "status": status,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "strategy": strategy,
        "order_id": order_id or f"ord-{symbol}-{id(data) if False else hash((symbol, side, qty, price, pnl))}",
    }
    trade = db.record_trade(data)
    if closed_at or created_at:
        with db.get_session() as session:
            t = session.query(Trade).filter_by(id=trade.id).first()
            if closed_at:
                t.closed_at = closed_at
            if created_at:
                t.created_at = created_at
            session.commit()
    return trade


def _journal_open(journal, *, symbol="AAPL", side="buy", entry_price=150.0,
                  qty=10, strategy_name="momentum", model_version="v1.0",
                  confidence=0.75, features=None, market_regime="trending",
                  trade_id=None, contract_id=""):
    """Helper: log a journal entry (open trade)."""
    return journal.log_entry({
        "trade_id": trade_id,
        "symbol": symbol,
        "side": side,
        "entry_price": entry_price,
        "qty": qty,
        "strategy_name": strategy_name,
        "model_version": model_version,
        "prediction": side,
        "confidence": confidence,
        "features_snapshot": features or {"rsi": 45.0, "macd": 0.5},
        "market_regime": market_regime,
        "contract_id": contract_id,
    })


def _journal_close(journal, journal_id, *, exit_price=155.0, pnl=50.0,
                   pnl_pct=3.33, exit_reason="strategy_exit", holding_bars=5):
    """Helper: log a journal exit (close trade)."""
    journal.log_exit(journal_id, {
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "holding_bars": holding_bars,
    })


# ── Test: Open Trades ────────────────────────────────────────────────────

class TestOpenTrades:
    """Verify that open trades are journaled and appear correctly."""

    def test_open_trade_appears_in_journal(self, db, journal):
        """An open trade should have a journal entry with no exit data."""
        _make_trade(db, symbol="TSLA", pnl=None)
        jid = _journal_open(journal, symbol="TSLA")

        entries = journal.get_journal(symbol="TSLA")
        assert len(entries) == 1
        entry = entries[0]
        assert entry["symbol"] == "TSLA"
        assert entry["exit_price"] is None
        assert entry["pnl"] is None

    def test_open_trade_counted_in_summary_total(self, db, journal):
        """Open trades should be included in total_trades but not in closed stats."""
        _journal_open(journal, symbol="AAPL")

        summary = journal.get_summary(days=30)
        assert summary["total_trades"] == 1
        assert summary["closed_trades"] == 0
        assert summary["open_trades"] == 1
        assert summary["win_rate"] == 0.0

    def test_open_trade_excluded_from_pnl_analytics(self, db, journal):
        """Open trades (pnl=None) should not affect confidence or model analytics."""
        _journal_open(journal, confidence=0.85, model_version="v2.0")

        conf_stats = journal.get_performance_by_confidence()
        assert conf_stats == {}  # No closed trades → no stats

        model_stats = journal.get_performance_by_model_version()
        assert model_stats == {}


# ── Test: Closed Trades ──────────────────────────────────────────────────

class TestClosedTrades:
    """Verify that closed trades have complete journal entries."""

    def test_closed_trade_has_exit_data(self, db, journal):
        """A closed trade's journal entry should have exit price, pnl, reason."""
        _make_trade(db, symbol="GOOG", pnl=100.0)
        jid = _journal_open(journal, symbol="GOOG", entry_price=2800.0)
        _journal_close(journal, jid, exit_price=2810.0, pnl=100.0, pnl_pct=0.36)

        entries = journal.get_journal(symbol="GOOG")
        assert len(entries) == 1
        entry = entries[0]
        assert entry["exit_price"] == 2810.0
        assert entry["pnl"] == 100.0
        assert entry["exit_reason"] == "strategy_exit"

    def test_closed_trade_contributes_to_summary(self, db, journal):
        """Closed trades should contribute to win_rate and pnl calculations."""
        jid = _journal_open(journal, symbol="AAPL")
        _journal_close(journal, jid, pnl=50.0, pnl_pct=3.33)

        summary = journal.get_summary(days=30)
        assert summary["total_trades"] == 1
        assert summary["closed_trades"] == 1
        assert summary["win_rate"] == 1.0
        assert summary["total_pnl"] == 50.0
        assert summary["avg_pnl"] == 50.0

    def test_win_loss_classification(self, db, journal):
        """Positive pnl = win, zero/negative pnl = loss."""
        jid1 = _journal_open(journal, symbol="AAPL")
        _journal_close(journal, jid1, pnl=50.0)

        jid2 = _journal_open(journal, symbol="GOOG")
        _journal_close(journal, jid2, pnl=-30.0)

        jid3 = _journal_open(journal, symbol="MSFT")
        _journal_close(journal, jid3, pnl=0.0)

        summary = journal.get_summary(days=30)
        assert summary["closed_trades"] == 3
        # pnl > 0 = win; pnl <= 0 = loss
        assert summary["win_rate"] == pytest.approx(1 / 3, abs=0.01)
        assert summary["total_pnl"] == 20.0
        assert summary["best_trade"] == 50.0
        assert summary["worst_trade"] == -30.0


# ── Test: Analytics Calculations ─────────────────────────────────────────

class TestAnalytics:
    """Verify analytics aggregation logic."""

    def test_confidence_buckets(self, db, journal):
        """Trades should be bucketed by confidence into 0.1-wide ranges."""
        # Low confidence losing trade
        jid1 = _journal_open(journal, confidence=0.35)
        _journal_close(journal, jid1, pnl=-20.0)

        # High confidence winning trade
        jid2 = _journal_open(journal, confidence=0.85)
        _journal_close(journal, jid2, pnl=100.0)

        stats = journal.get_performance_by_confidence()
        assert "0.3-0.4" in stats
        assert "0.8-0.9" in stats
        assert stats["0.3-0.4"]["trades"] == 1
        assert stats["0.3-0.4"]["win_rate"] == 0.0
        assert stats["0.8-0.9"]["trades"] == 1
        assert stats["0.8-0.9"]["win_rate"] == 1.0

    def test_model_version_comparison(self, db, journal):
        """Stats should be grouped by model_version."""
        jid1 = _journal_open(journal, model_version="v1.0")
        _journal_close(journal, jid1, pnl=50.0)

        jid2 = _journal_open(journal, model_version="v2.0")
        _journal_close(journal, jid2, pnl=-10.0)

        jid3 = _journal_open(journal, model_version="v2.0")
        _journal_close(journal, jid3, pnl=30.0)

        stats = journal.get_performance_by_model_version()
        assert stats["v1.0"]["trades"] == 1
        assert stats["v1.0"]["win_rate"] == 1.0
        assert stats["v2.0"]["trades"] == 2
        assert stats["v2.0"]["win_rate"] == 0.5
        assert stats["v2.0"]["total_pnl"] == 20.0

    def test_summary_win_rate_by_strategy(self, db, journal):
        """Win rate should be broken down by strategy."""
        jid1 = _journal_open(journal, strategy_name="momentum")
        _journal_close(journal, jid1, pnl=50.0)

        jid2 = _journal_open(journal, strategy_name="momentum")
        _journal_close(journal, jid2, pnl=-20.0)

        jid3 = _journal_open(journal, strategy_name="mean_reversion")
        _journal_close(journal, jid3, pnl=80.0)

        summary = journal.get_summary(days=30)
        strats = summary["win_rate_by_strategy"]
        assert strats["momentum"]["trades"] == 2
        assert strats["momentum"]["win_rate"] == 0.5
        assert strats["mean_reversion"]["trades"] == 1
        assert strats["mean_reversion"]["win_rate"] == 1.0

    def test_summary_30_day_window_excludes_old_trades(self, db, journal):
        """Trades older than 30 days should not appear in get_summary."""
        # Insert a journal entry with old created_at
        with db.get_journal_session() as session:
            old_entry = JournalEntry(
                symbol="OLD",
                side="buy",
                entry_price=100.0,
                qty=1,
                strategy_name="momentum",
                pnl=999.0,
                created_at=datetime.utcnow() - timedelta(days=60),
            )
            session.add(old_entry)
            session.commit()

        # Insert a recent journal entry
        jid = _journal_open(journal, symbol="NEW")
        _journal_close(journal, jid, pnl=10.0)

        summary = journal.get_summary(days=30)
        assert summary["total_trades"] == 1  # Only the recent one
        assert summary["total_pnl"] == 10.0  # Not 999.0

    def test_avg_holding_bars(self, db, journal):
        """Average holding bars should be calculated from closed trades only."""
        jid1 = _journal_open(journal)
        _journal_close(journal, jid1, pnl=10.0, holding_bars=5)

        jid2 = _journal_open(journal)
        _journal_close(journal, jid2, pnl=20.0, holding_bars=15)

        # Open trade (no holding_bars)
        _journal_open(journal)

        summary = journal.get_summary(days=30)
        assert summary["avg_holding_bars"] == 10.0


# ── Test: Replay / Recovery Scenarios ────────────────────────────────────

class TestRecoveryScenarios:
    """Verify behavior under replay and recovery conditions."""

    def test_orphaned_journal_entry_stays_open(self, db, journal):
        """A journal entry never closed stays in 'open' state indefinitely."""
        _journal_open(journal, symbol="AAPL")

        entries = journal.get_journal(symbol="AAPL")
        assert len(entries) == 1
        assert entries[0]["pnl"] is None
        assert entries[0]["exit_price"] is None

        # It should count as open in summary
        summary = journal.get_summary(days=30)
        assert summary["open_trades"] == 1
        assert summary["closed_trades"] == 0

    def test_log_exit_nonexistent_journal_id(self, db, journal):
        """log_exit with a nonexistent journal_id should not raise."""
        # Should not raise, just log a warning
        journal.log_exit(99999, {
            "exit_price": 100.0,
            "exit_reason": "test",
            "pnl": 0.0,
        })

    def test_multiple_open_entries_same_symbol(self, db, journal):
        """Multiple open entries for the same symbol should all be tracked."""
        jid1 = _journal_open(journal, symbol="AAPL", entry_price=150.0)
        jid2 = _journal_open(journal, symbol="AAPL", entry_price=155.0)

        entries = journal.get_journal(symbol="AAPL")
        assert len(entries) == 2

        # Close only the first one
        _journal_close(journal, jid1, exit_price=160.0, pnl=100.0)

        entries = journal.get_journal(symbol="AAPL")
        open_entries = [e for e in entries if e["pnl"] is None]
        closed_entries = [e for e in entries if e["pnl"] is not None]
        assert len(open_entries) == 1
        assert len(closed_entries) == 1

    def test_trade_without_journal_entry(self, db, journal):
        """A trade in the trades table without a journal entry should still show in /trades."""
        _make_trade(db, symbol="TSLA", pnl=42.0)

        trades = db.get_trades(symbol="TSLA")
        assert len(trades) == 1
        assert trades[0]["pnl"] == 42.0

        # Journal should be empty
        entries = journal.get_journal(symbol="TSLA")
        assert len(entries) == 0

    def test_journal_entry_without_trade_record(self, db, journal):
        """A journal entry without a trades record should still appear in journal/stats."""
        jid = _journal_open(journal, symbol="MSFT")
        _journal_close(journal, jid, pnl=25.0)

        entries = journal.get_journal(symbol="MSFT")
        assert len(entries) == 1

        summary = journal.get_summary(days=30)
        assert summary["closed_trades"] == 1


# ── Test: Edge Cases ─────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case coverage."""

    def test_empty_database_trades(self, db):
        """get_trades on empty DB returns empty list."""
        trades = db.get_trades()
        assert trades == []

    def test_empty_database_journal_summary(self, db, journal):
        """get_summary on empty DB returns zero-trade summary."""
        summary = journal.get_summary(days=30)
        assert summary["total_trades"] == 0

    def test_empty_database_confidence_stats(self, db, journal):
        """get_performance_by_confidence on empty DB returns empty dict."""
        assert journal.get_performance_by_confidence() == {}

    def test_empty_database_model_stats(self, db, journal):
        """get_performance_by_model_version on empty DB returns empty dict."""
        assert journal.get_performance_by_model_version() == {}

    def test_single_winning_trade(self, db, journal):
        """Single winning trade should produce 100% win rate."""
        jid = _journal_open(journal)
        _journal_close(journal, jid, pnl=100.0)

        summary = journal.get_summary(days=30)
        assert summary["win_rate"] == 1.0
        assert summary["total_pnl"] == 100.0
        assert summary["best_trade"] == 100.0
        assert summary["worst_trade"] == 100.0

    def test_single_losing_trade(self, db, journal):
        """Single losing trade should produce 0% win rate."""
        jid = _journal_open(journal)
        _journal_close(journal, jid, pnl=-50.0)

        summary = journal.get_summary(days=30)
        assert summary["win_rate"] == 0.0
        assert summary["total_pnl"] == -50.0

    def test_feature_importance_needs_minimum_entries(self, db, journal):
        """Feature importance requires at least 5 entries."""
        # Add 3 entries (below threshold)
        for i in range(3):
            jid = _journal_open(journal, features={"rsi": float(40 + i)})
            _journal_close(journal, jid, pnl=float(10 * (i + 1)))

        result = journal.get_feature_importance_from_outcomes()
        assert result == {}

    def test_feature_importance_with_sufficient_entries(self, db, journal):
        """Feature importance should return correlations with 5+ entries."""
        for i in range(6):
            jid = _journal_open(journal, features={"rsi": float(40 + i * 5)})
            _journal_close(journal, jid, pnl=float(10 * (i - 2)))

        result = journal.get_feature_importance_from_outcomes()
        assert "rsi" in result
        assert -1.0 <= result["rsi"] <= 1.0

    def test_confidence_bucket_boundary(self, db, journal):
        """Confidence of exactly 0.5 should go into 0.5-0.6 bucket."""
        jid = _journal_open(journal, confidence=0.5)
        _journal_close(journal, jid, pnl=10.0)

        stats = journal.get_performance_by_confidence()
        assert "0.5-0.6" in stats

    def test_get_trades_returns_all_statuses(self, db):
        """get_trades should return trades of all statuses."""
        _make_trade(db, symbol="A", status="filled", order_id="o1")
        _make_trade(db, symbol="B", status="cancelled", order_id="o2")
        _make_trade(db, symbol="C", status="rejected", order_id="o3")

        trades = db.get_trades()
        assert len(trades) == 3
        statuses = {t["status"] for t in trades}
        assert statuses == {"filled", "cancelled", "rejected"}


# ── Test: /trades vs /journalstats Consistency ───────────────────────────

class TestTradesJournalConsistency:
    """Verify the documented relationship between /trades and /journalstats."""

    def test_both_populated_on_complete_flow(self, db, journal):
        """When both record_trade and journal.log_entry/exit are called,
        both /trades and /journalstats reflect the trade."""
        trade = _make_trade(db, symbol="AAPL", price=150.0, pnl=50.0)
        jid = _journal_open(journal, symbol="AAPL", entry_price=150.0)
        _journal_close(journal, jid, exit_price=155.0, pnl=50.0)

        # /trades sees it
        trades = db.get_trades(symbol="AAPL")
        assert len(trades) == 1
        assert trades[0]["pnl"] == 50.0

        # /journalstats sees it
        summary = journal.get_summary(days=30)
        assert summary["closed_trades"] == 1
        assert summary["total_pnl"] == 50.0

    def test_journal_stats_excludes_non_journaled_trades(self, db, journal):
        """A trade in the trades table but not journaled should NOT appear
        in /journalstats. This is expected behavior."""
        _make_trade(db, symbol="NVDA", pnl=200.0)
        # Intentionally not calling journal.log_entry

        trades = db.get_trades(symbol="NVDA")
        assert len(trades) == 1

        summary = journal.get_summary(days=30)
        assert summary["total_trades"] == 0  # Not in journal

    def test_performance_summary_vs_journal_summary(self, db, journal):
        """DatabaseManager.get_performance_summary and TradeJournal.get_summary
        should agree on closed trade statistics when data is consistent."""
        # Create consistent data in both tables
        trade = _make_trade(db, symbol="AAPL", pnl=50.0)
        jid = _journal_open(journal, symbol="AAPL")
        _journal_close(journal, jid, pnl=50.0)

        trade2 = _make_trade(db, symbol="GOOG", pnl=-20.0, order_id="o2")
        jid2 = _journal_open(journal, symbol="GOOG")
        _journal_close(journal, jid2, pnl=-20.0)

        db_summary = db.get_performance_summary(days=30)
        journal_summary = journal.get_summary(days=30)

        # Both should agree on core metrics
        assert db_summary["total_trades"] == journal_summary["closed_trades"]
        assert db_summary["total_pnl"] == journal_summary["total_pnl"]
        assert db_summary["win_rate"] == pytest.approx(journal_summary["win_rate"], abs=0.01)

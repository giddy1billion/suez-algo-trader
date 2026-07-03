"""
Trade Journal — Logs every executed trade with full context for analysis and retraining.
"""

import json
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import func

from src.data.store import JournalEntry, DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TradeJournal:
    """Journals trade entries/exits with ML context for performance analysis and retraining."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self._ensure_table()

    def _ensure_table(self):
        """Create journal table if it doesn't exist."""
        JournalEntry.__table__.create(bind=self.db.engine, checkfirst=True)

    def log_entry(self, trade_data: dict) -> int:
        """
        Log when a trade is OPENED. Returns journal entry ID.

        trade_data keys:
            symbol, side, entry_price, qty, strategy_name, model_version,
            prediction, confidence, features_snapshot (dict), volatility_at_entry,
            market_regime, trade_id (optional)
        """
        features = trade_data.get("features_snapshot")
        if isinstance(features, dict):
            features = json.dumps(features)

        entry = JournalEntry(
            trade_id=trade_data.get("trade_id"),
            symbol=trade_data["symbol"],
            side=trade_data["side"],
            entry_price=trade_data.get("entry_price"),
            entry_time=trade_data.get("entry_time", datetime.utcnow()),
            qty=trade_data.get("qty"),
            strategy_name=trade_data.get("strategy_name"),
            model_version=trade_data.get("model_version"),
            prediction=trade_data.get("prediction"),
            confidence=trade_data.get("confidence"),
            features_snapshot=features,
            volatility_at_entry=trade_data.get("volatility_at_entry"),
            market_regime=trade_data.get("market_regime"),
        )

        with self.db.get_journal_session() as session:
            session.add(entry)
            session.commit()
            session.refresh(entry)
            journal_id = entry.id

        logger.info("journal_entry_logged", journal_id=journal_id, symbol=entry.symbol, side=entry.side)
        return journal_id

    def log_exit(self, journal_id: int, exit_data: dict):
        """
        Update when trade is CLOSED.

        exit_data keys:
            exit_price, exit_reason, pnl, pnl_pct, holding_bars
        """
        with self.db.get_journal_session() as session:
            entry = session.query(JournalEntry).filter_by(id=journal_id).first()
            if not entry:
                logger.warning("journal_exit_not_found", journal_id=journal_id)
                return

            entry.exit_price = exit_data.get("exit_price")
            entry.exit_time = exit_data.get("exit_time", datetime.utcnow())
            entry.exit_reason = exit_data.get("exit_reason")
            entry.pnl = exit_data.get("pnl")
            entry.pnl_pct = exit_data.get("pnl_pct")
            entry.holding_bars = exit_data.get("holding_bars")
            session.commit()

        logger.info("journal_exit_logged", journal_id=journal_id, pnl=exit_data.get("pnl"))

    def get_journal(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """Query journal entries with optional filters."""
        with self.db.get_journal_session() as session:
            q = session.query(JournalEntry)
            if symbol:
                q = q.filter(JournalEntry.symbol == symbol)
            if strategy:
                q = q.filter(JournalEntry.strategy_name == strategy)
            entries = (
                q.order_by(JournalEntry.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [self._entry_to_dict(e) for e in entries]

    def get_performance_by_confidence(self, min_conf: float = 0.0, max_conf: float = 1.0) -> dict:
        """
        Analyze how confidence correlates with win rate.
        Returns bucketed stats: {bucket_label: {trades: N, win_rate: X, avg_pnl: Y}}
        """
        with self.db.get_journal_session() as session:
            entries = (
                session.query(JournalEntry)
                .filter(
                    JournalEntry.confidence >= min_conf,
                    JournalEntry.confidence <= max_conf,
                    JournalEntry.pnl.isnot(None),
                )
                .all()
            )

        buckets = {}
        for e in entries:
            bucket = self._confidence_bucket(e.confidence)
            if bucket not in buckets:
                buckets[bucket] = {"trades": 0, "wins": 0, "total_pnl": 0.0}
            buckets[bucket]["trades"] += 1
            if e.pnl and e.pnl > 0:
                buckets[bucket]["wins"] += 1
            buckets[bucket]["total_pnl"] += e.pnl or 0.0

        result = {}
        for bucket in sorted(buckets.keys()):
            stats = buckets[bucket]
            result[bucket] = {
                "trades": stats["trades"],
                "win_rate": stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0.0,
                "avg_pnl": stats["total_pnl"] / stats["trades"] if stats["trades"] > 0 else 0.0,
            }
        return result

    def get_performance_by_model_version(self) -> dict:
        """
        Compare model versions by actual trade outcomes.
        Returns: {version: {trades: N, win_rate: X, avg_pnl: Y, total_pnl: Z}}
        """
        with self.db.get_journal_session() as session:
            entries = (
                session.query(JournalEntry)
                .filter(
                    JournalEntry.model_version.isnot(None),
                    JournalEntry.pnl.isnot(None),
                )
                .all()
            )

        versions: dict = {}
        for e in entries:
            v = e.model_version
            if v not in versions:
                versions[v] = {"trades": 0, "wins": 0, "total_pnl": 0.0}
            versions[v]["trades"] += 1
            if e.pnl and e.pnl > 0:
                versions[v]["wins"] += 1
            versions[v]["total_pnl"] += e.pnl or 0.0

        result = {}
        for v, stats in versions.items():
            result[v] = {
                "trades": stats["trades"],
                "win_rate": stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0.0,
                "avg_pnl": stats["total_pnl"] / stats["trades"] if stats["trades"] > 0 else 0.0,
                "total_pnl": stats["total_pnl"],
            }
        return result

    def get_feature_importance_from_outcomes(self) -> dict:
        """
        Correlate feature values at entry with trade outcome (simple Pearson correlation).
        Returns: {feature_name: correlation_with_pnl}
        """
        with self.db.get_journal_session() as session:
            entries = (
                session.query(JournalEntry)
                .filter(
                    JournalEntry.features_snapshot.isnot(None),
                    JournalEntry.pnl.isnot(None),
                )
                .all()
            )

        if len(entries) < 5:
            return {}

        # Collect feature vectors and outcomes
        feature_values: dict[str, list[float]] = {}
        pnls: list[float] = []

        for e in entries:
            try:
                features = json.loads(e.features_snapshot)
            except (json.JSONDecodeError, TypeError):
                continue
            pnls.append(e.pnl)
            for feat_name, feat_val in features.items():
                if isinstance(feat_val, (int, float)):
                    feature_values.setdefault(feat_name, []).append(float(feat_val))
                else:
                    feature_values.setdefault(feat_name, []).append(None)

        # Calculate correlations
        correlations = {}
        for feat_name, values in feature_values.items():
            # Align: only use entries where both feature and pnl are numeric
            paired = [
                (v, p) for v, p in zip(values, pnls) if v is not None and p is not None
            ]
            if len(paired) < 5:
                continue
            feat_arr = [p[0] for p in paired]
            pnl_arr = [p[1] for p in paired]
            corr = self._pearson(feat_arr, pnl_arr)
            if corr is not None:
                correlations[feat_name] = round(corr, 4)

        return dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))

    def get_summary(self, days: int = 30) -> dict:
        """
        Overall journal summary: total trades, win rate by strategy,
        avg holding period, best/worst trades.
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        with self.db.get_journal_session() as session:
            entries = (
                session.query(JournalEntry)
                .filter(JournalEntry.created_at >= cutoff)
                .all()
            )

        if not entries:
            return {"total_trades": 0, "days": days, "message": "No journal entries in period"}

        closed = [e for e in entries if e.pnl is not None]
        wins = [e for e in closed if e.pnl > 0]
        losses = [e for e in closed if e.pnl <= 0]

        # Win rate by strategy
        strategy_stats: dict = {}
        for e in closed:
            s = e.strategy_name or "unknown"
            if s not in strategy_stats:
                strategy_stats[s] = {"trades": 0, "wins": 0, "total_pnl": 0.0}
            strategy_stats[s]["trades"] += 1
            if e.pnl > 0:
                strategy_stats[s]["wins"] += 1
            strategy_stats[s]["total_pnl"] += e.pnl

        win_rate_by_strategy = {
            s: {
                "trades": stats["trades"],
                "win_rate": stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0.0,
                "total_pnl": round(stats["total_pnl"], 2),
            }
            for s, stats in strategy_stats.items()
        }

        holding_bars = [e.holding_bars for e in closed if e.holding_bars is not None]
        pnls = [e.pnl for e in closed]

        return {
            "days": days,
            "total_trades": len(entries),
            "closed_trades": len(closed),
            "open_trades": len(entries) - len(closed),
            "win_rate": len(wins) / len(closed) if closed else 0.0,
            "total_pnl": round(sum(pnls), 2) if pnls else 0.0,
            "avg_pnl": round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
            "best_trade": round(max(pnls), 2) if pnls else 0.0,
            "worst_trade": round(min(pnls), 2) if pnls else 0.0,
            "avg_holding_bars": round(sum(holding_bars) / len(holding_bars), 1) if holding_bars else 0,
            "win_rate_by_strategy": win_rate_by_strategy,
        }

    def export_training_data(self, min_trades: int = 100) -> pd.DataFrame:
        """
        Export journal entries as a DataFrame suitable for retraining.
        Columns: all features + outcome (1=profit, 0=loss).
        Creates a feedback loop from live trading back into model training.
        """
        with self.db.get_journal_session() as session:
            entries = (
                session.query(JournalEntry)
                .filter(
                    JournalEntry.features_snapshot.isnot(None),
                    JournalEntry.pnl.isnot(None),
                )
                .all()
            )

        if len(entries) < min_trades:
            logger.warning(
                "insufficient_training_data",
                available=len(entries),
                required=min_trades,
            )
            return pd.DataFrame()

        rows = []
        for e in entries:
            try:
                features = json.loads(e.features_snapshot)
            except (json.JSONDecodeError, TypeError):
                continue

            row = {
                "symbol": e.symbol,
                "side": e.side,
                "strategy_name": e.strategy_name,
                "model_version": e.model_version,
                "confidence": e.confidence,
                "market_regime": e.market_regime,
                "volatility_at_entry": e.volatility_at_entry,
                "holding_bars": e.holding_bars,
                "pnl": e.pnl,
                "pnl_pct": e.pnl_pct,
                "outcome": 1 if e.pnl > 0 else 0,
            }
            # Flatten features into columns
            for feat_name, feat_val in features.items():
                if isinstance(feat_val, (int, float)):
                    row[f"feat_{feat_name}"] = feat_val

            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info("training_data_exported", rows=len(df))
        return df

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _entry_to_dict(entry: JournalEntry) -> dict:
        features = None
        if entry.features_snapshot:
            try:
                features = json.loads(entry.features_snapshot)
            except (json.JSONDecodeError, TypeError):
                features = entry.features_snapshot

        return {
            "id": entry.id,
            "trade_id": entry.trade_id,
            "symbol": entry.symbol,
            "side": entry.side,
            "entry_price": entry.entry_price,
            "entry_time": str(entry.entry_time) if entry.entry_time else None,
            "qty": entry.qty,
            "strategy_name": entry.strategy_name,
            "model_version": entry.model_version,
            "prediction": entry.prediction,
            "confidence": entry.confidence,
            "features_snapshot": features,
            "exit_price": entry.exit_price,
            "exit_time": str(entry.exit_time) if entry.exit_time else None,
            "exit_reason": entry.exit_reason,
            "pnl": entry.pnl,
            "pnl_pct": entry.pnl_pct,
            "holding_bars": entry.holding_bars,
            "market_regime": entry.market_regime,
            "volatility_at_entry": entry.volatility_at_entry,
            "created_at": str(entry.created_at) if entry.created_at else None,
        }

    @staticmethod
    def _confidence_bucket(conf: float) -> str:
        if conf is None:
            return "unknown"
        lower = int(conf * 10) / 10
        upper = lower + 0.1
        return f"{lower:.1f}-{upper:.1f}"

    @staticmethod
    def _pearson(x: list[float], y: list[float]) -> Optional[float]:
        """Simple Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return None
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
        den_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
        if den_x == 0 or den_y == 0:
            return None
        return num / (den_x * den_y)

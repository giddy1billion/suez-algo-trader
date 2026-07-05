from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeScorecard:
    """Every completed trade produces a scorecard for ML feedback."""

    # Identity
    trade_id: str
    signal_id: str
    symbol: str
    timestamp: datetime

    # Prediction vs Reality
    predicted_direction: str  # "buy" or "sell"
    actual_profitable: bool
    predicted_confidence: float
    predicted_win_probability: float
    predicted_return_pct: float
    actual_return_pct: float
    predicted_risk_reward: float
    actual_risk_reward: float
    predicted_duration_minutes: int
    actual_duration_minutes: int

    # Price accuracy
    entry_price: float
    exit_price: float
    stop_loss_price: float
    stop_loss_hit: bool
    take_profit_prices: list[float]
    take_profit_reached: list[bool]

    # Excursion metrics
    max_favorable_excursion: float  # Best price seen during trade (MFE)
    max_adverse_excursion: float  # Worst price seen during trade (MAE)

    # Context
    model_version: str
    strategy_name: str
    market_regime: str
    volatility_level: str

    # Feature snapshot (for retraining)
    feature_vector: dict = field(default_factory=dict)

    # Decision Contract linkage (for governance feedback loop)
    contract_id: str = ""  # Links to DecisionContract that approved this trade
    contract_decision: str = ""  # "execute" | "reduce" — what the contract decided
    contract_confidence: float = 0.0  # Contract's final confidence at decision time

    # Execution quality
    slippage_pct: float = 0.0
    fees: float = 0.0

    # Scoring
    direction_score: float = 0.0  # 1.0 if correct, 0.0 if wrong
    confidence_calibration_error: float = 0.0
    timing_score: float = 0.0  # How close predicted vs actual duration
    exit_efficiency: float = 0.0  # actual_profit / max_possible_profit (MFE)
    overall_score: float = 0.0  # Composite 0-100


def _compute_feature_hash(feature_vector: dict) -> str:
    """Deterministic hash of feature keys for lineage tracking."""
    keys_str = ",".join(sorted(feature_vector.keys()))
    return hashlib.md5(keys_str.encode("utf-8")).hexdigest()[:12]


class ExperienceDatabase:
    """Stores all scorecards and provides query interface for the training pipeline.

    Uses DuckDB for efficient columnar storage and SQL-based analytics.
    """

    _CREATE_TABLES_SQL = """
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id TEXT PRIMARY KEY,
        trade_id TEXT,
        symbol TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        model_version TEXT,
        strategy_name TEXT,
        predicted_direction TEXT,
        predicted_confidence DOUBLE,
        predicted_win_probability DOUBLE,
        predicted_return_pct DOUBLE,
        predicted_duration_minutes INTEGER,
        predicted_risk_reward DOUBLE,
        market_regime TEXT,
        volatility_level TEXT,
        feature_hash TEXT,
        contract_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS outcomes (
        trade_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        actual_profitable BOOLEAN,
        actual_return_pct DOUBLE,
        actual_duration_minutes INTEGER,
        actual_risk_reward DOUBLE,
        entry_price DOUBLE,
        exit_price DOUBLE,
        stop_loss_price DOUBLE,
        stop_loss_hit BOOLEAN,
        max_favorable_excursion DOUBLE,
        max_adverse_excursion DOUBLE,
        slippage_pct DOUBLE,
        fees DOUBLE,
        contract_id TEXT,
        contract_decision TEXT,
        contract_confidence DOUBLE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS scorecards (
        trade_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        direction_score DOUBLE,
        confidence_calibration_error DOUBLE,
        timing_score DOUBLE,
        exit_efficiency DOUBLE,
        overall_score DOUBLE,
        model_version TEXT,
        strategy_name TEXT,
        market_regime TEXT,
        contract_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS feature_snapshots (
        trade_id TEXT NOT NULL,
        feature_name TEXT NOT NULL,
        feature_value DOUBLE,
        PRIMARY KEY (trade_id, feature_name)
    );
    """

    def __init__(self, storage_path: str = "data_cache/experience") -> None:
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._db_path = self._storage_path / "experience.duckdb"
        self._write_lock = threading.Lock()
        self._conn = duckdb.connect(str(self._db_path))
        self._init_tables()
        self._check_legacy_migration()
        logger.info("experience_db_initialized", path=str(self._db_path))

    def _init_tables(self) -> None:
        """Create tables if they do not exist."""
        self._conn.execute(self._CREATE_TABLES_SQL)

    def _check_legacy_migration(self) -> None:
        """Warn if legacy JSONL file exists and migrate schema if needed."""
        legacy_path = self._storage_path / "scorecards.jsonl"
        if legacy_path.exists():
            logger.warning(
                "legacy_jsonl_detected",
                path=str(legacy_path),
                message="Legacy JSONL file found. Consider migrating with "
                "ExperienceDatabase.migrate_from_jsonl().",
            )
        # Schema migration: add contract_id columns if missing
        self._migrate_contract_columns()

    def _migrate_contract_columns(self) -> None:
        """Add contract_id columns to existing tables (idempotent)."""
        migrations = [
            ("predictions", "contract_id", "TEXT"),
            ("outcomes", "contract_id", "TEXT"),
            ("outcomes", "contract_decision", "TEXT"),
            ("outcomes", "contract_confidence", "DOUBLE"),
            ("scorecards", "contract_id", "TEXT"),
        ]
        for table, column, dtype in migrations:
            try:
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {dtype}"
                )
            except Exception:
                pass  # Column already exists

    def migrate_from_jsonl(self) -> int:
        """Migrate legacy JSONL data into DuckDB. Returns count of migrated records."""
        legacy_path = self._storage_path / "scorecards.jsonl"
        if not legacy_path.exists():
            return 0
        count = 0
        with open(legacy_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sc = self._dict_to_scorecard(record)
                self.record_trade(sc)
                count += 1
        logger.info("jsonl_migration_complete", migrated=count)
        return count

    @staticmethod
    def _dict_to_scorecard(record: dict) -> TradeScorecard:
        """Convert a legacy dict record to TradeScorecard."""
        ts = record.get("timestamp", datetime.now(tz=timezone.utc))
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return TradeScorecard(
            trade_id=record.get("trade_id", str(uuid.uuid4())[:12]),
            signal_id=record.get("signal_id", ""),
            symbol=record.get("symbol", "UNKNOWN"),
            timestamp=ts,
            predicted_direction=record.get("predicted_direction", "buy"),
            actual_profitable=record.get("actual_profitable", False),
            predicted_confidence=record.get("predicted_confidence", 0.5),
            predicted_win_probability=record.get("predicted_win_probability", 0.5),
            predicted_return_pct=record.get("predicted_return_pct", 0.0),
            actual_return_pct=record.get("actual_return_pct", 0.0),
            predicted_risk_reward=record.get("predicted_risk_reward", 1.0),
            actual_risk_reward=record.get("actual_risk_reward", 0.0),
            predicted_duration_minutes=record.get("predicted_duration_minutes", 0),
            actual_duration_minutes=record.get("actual_duration_minutes", 0),
            entry_price=record.get("entry_price", 0.0),
            exit_price=record.get("exit_price", 0.0),
            stop_loss_price=record.get("stop_loss_price", 0.0),
            stop_loss_hit=record.get("stop_loss_hit", False),
            take_profit_prices=record.get("take_profit_prices", []),
            take_profit_reached=record.get("take_profit_reached", []),
            max_favorable_excursion=record.get("max_favorable_excursion", 0.0),
            max_adverse_excursion=record.get("max_adverse_excursion", 0.0),
            model_version=record.get("model_version", "unknown"),
            strategy_name=record.get("strategy_name", "unknown"),
            market_regime=record.get("market_regime", "unknown"),
            volatility_level=record.get("volatility_level", "unknown"),
            feature_vector=record.get("feature_vector", {}),
            slippage_pct=record.get("slippage_pct", 0.0),
            fees=record.get("fees", 0.0),
            direction_score=record.get("direction_score", 0.0),
            confidence_calibration_error=record.get("confidence_calibration_error", 0.0),
            timing_score=record.get("timing_score", 0.0),
            exit_efficiency=record.get("exit_efficiency", 0.0),
            overall_score=record.get("overall_score", 0.0),
        )

    def record_trade(self, scorecard: TradeScorecard) -> None:
        """Insert into predictions, outcomes, scorecards, feature_snapshots tables."""
        ts = scorecard.timestamp
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)

        feature_hash = _compute_feature_hash(scorecard.feature_vector)
        prediction_id = f"{scorecard.trade_id}_{scorecard.signal_id}"

        with self._write_lock:
            # Insert prediction
            self._conn.execute(
                """
                INSERT OR REPLACE INTO predictions (
                    prediction_id, trade_id, symbol, timestamp, model_version,
                    strategy_name, predicted_direction, predicted_confidence,
                    predicted_win_probability, predicted_return_pct,
                    predicted_duration_minutes, predicted_risk_reward,
                    market_regime, volatility_level, feature_hash, contract_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    prediction_id, scorecard.trade_id, scorecard.symbol, ts,
                    scorecard.model_version, scorecard.strategy_name,
                    scorecard.predicted_direction, scorecard.predicted_confidence,
                    scorecard.predicted_win_probability, scorecard.predicted_return_pct,
                    scorecard.predicted_duration_minutes, scorecard.predicted_risk_reward,
                    scorecard.market_regime, scorecard.volatility_level, feature_hash,
                    scorecard.contract_id,
                ],
            )

            # Insert outcome
            self._conn.execute(
                """
                INSERT OR REPLACE INTO outcomes (
                    trade_id, symbol, actual_profitable, actual_return_pct,
                    actual_duration_minutes, actual_risk_reward, entry_price,
                    exit_price, stop_loss_price, stop_loss_hit,
                    max_favorable_excursion, max_adverse_excursion,
                    slippage_pct, fees, contract_id, contract_decision,
                    contract_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    scorecard.trade_id, scorecard.symbol,
                    scorecard.actual_profitable, scorecard.actual_return_pct,
                    scorecard.actual_duration_minutes, scorecard.actual_risk_reward,
                    scorecard.entry_price, scorecard.exit_price,
                    scorecard.stop_loss_price, scorecard.stop_loss_hit,
                    scorecard.max_favorable_excursion, scorecard.max_adverse_excursion,
                    scorecard.slippage_pct, scorecard.fees,
                    scorecard.contract_id, scorecard.contract_decision,
                    scorecard.contract_confidence,
                ],
            )

            # Insert scorecard
            self._conn.execute(
                """
                INSERT OR REPLACE INTO scorecards (
                    trade_id, symbol, timestamp, direction_score,
                    confidence_calibration_error, timing_score, exit_efficiency,
                    overall_score, model_version, strategy_name, market_regime,
                    contract_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    scorecard.trade_id, scorecard.symbol, ts,
                    scorecard.direction_score, scorecard.confidence_calibration_error,
                    scorecard.timing_score, scorecard.exit_efficiency,
                    scorecard.overall_score, scorecard.model_version,
                    scorecard.strategy_name, scorecard.market_regime,
                    scorecard.contract_id,
                ],
            )

            # Insert feature snapshots
            if scorecard.feature_vector:
                for fname, fval in scorecard.feature_vector.items():
                    try:
                        fval_float = float(fval)
                    except (TypeError, ValueError):
                        continue
                    self._conn.execute(
                        """
                        INSERT OR REPLACE INTO feature_snapshots
                            (trade_id, feature_name, feature_value)
                        VALUES (?, ?, ?)
                        """,
                        [scorecard.trade_id, fname, fval_float],
                    )

        logger.info(
            "trade_recorded",
            trade_id=scorecard.trade_id,
            symbol=scorecard.symbol,
            score=scorecard.overall_score,
        )

    def get_training_samples(
        self, min_trades: int = 50, since: Optional[datetime] = None
    ) -> pd.DataFrame:
        """SQL query joining outcomes + features, returns training-ready DataFrame."""
        where_clause = ""
        params: list[Any] = []
        if since is not None:
            where_clause = "WHERE o.created_at >= ?"
            params.append(since)

        count_result = self._conn.execute(
            f"SELECT COUNT(*) FROM outcomes o {where_clause}", params
        ).fetchone()
        count = count_result[0] if count_result else 0

        if count < min_trades:
            logger.warning(
                "insufficient_training_data", available=count, required=min_trades
            )
            return pd.DataFrame()

        df = self._conn.execute(
            f"""
            SELECT
                o.trade_id, o.symbol, o.actual_profitable, o.actual_return_pct,
                o.actual_duration_minutes, o.actual_risk_reward, o.entry_price,
                o.exit_price, o.stop_loss_price, o.stop_loss_hit,
                o.max_favorable_excursion, o.max_adverse_excursion,
                o.slippage_pct, o.fees,
                p.predicted_direction, p.predicted_confidence,
                p.predicted_win_probability, p.predicted_return_pct AS predicted_return_pct,
                p.predicted_duration_minutes, p.predicted_risk_reward,
                p.model_version, p.strategy_name, p.market_regime,
                p.volatility_level, p.feature_hash,
                s.direction_score, s.confidence_calibration_error,
                s.timing_score, s.exit_efficiency, s.overall_score,
                p.timestamp
            FROM outcomes o
            LEFT JOIN predictions p ON o.trade_id = p.trade_id
            LEFT JOIN scorecards s ON o.trade_id = s.trade_id
            {where_clause}
            ORDER BY o.created_at
            """,
            params,
        ).fetchdf()

        # Add derived label columns
        df["profitable"] = df["actual_profitable"].astype(int)
        df["confidence_target"] = df["actual_profitable"].apply(
            lambda x: 1.0 if x else 0.0
        )
        df["duration_target"] = df["actual_duration_minutes"]

        logger.info("training_samples_exported", count=len(df))
        return df

    def get_training_samples_weighted(
        self, min_trades: int = 50, half_life_days: float = 30.0
    ) -> pd.DataFrame:
        """Like get_training_samples but with time-decay weights.

        Weight = exp(-ln(2) * age_days / half_life_days)
        Returns DataFrame with 'sample_weight' column.
        """
        df = self.get_training_samples(min_trades=min_trades)
        if df.empty:
            return df

        now = datetime.now(tz=timezone.utc)
        timestamps = pd.to_datetime(df["timestamp"], utc=True)
        age_days = (now - timestamps).dt.total_seconds() / 86400.0
        ln2 = math.log(2)
        df["sample_weight"] = np.exp(-ln2 * age_days / half_life_days)
        return df

    def get_calibration_data(self) -> dict:
        """SQL query: GROUP BY confidence_bin, compute actual win rates."""
        total_result = self._conn.execute(
            "SELECT COUNT(*) FROM outcomes"
        ).fetchone()
        total_trades = total_result[0] if total_result else 0

        if total_trades == 0:
            return {"bins": [], "total_trades": 0}

        rows = self._conn.execute(
            """
            SELECT
                FLOOR(p.predicted_confidence * 10) / 10.0 AS bin_low,
                COUNT(*) AS cnt,
                AVG(CASE WHEN o.actual_profitable THEN 1.0 ELSE 0.0 END) AS actual_win_rate
            FROM predictions p
            JOIN outcomes o ON p.trade_id = o.trade_id
            GROUP BY bin_low
            ORDER BY bin_low
            """
        ).fetchall()

        result_bins = []
        for row in rows:
            bin_low = float(row[0])
            bin_high = bin_low + 0.1
            result_bins.append(
                {
                    "bin": f"{bin_low:.1f}-{bin_high:.1f}",
                    "predicted_avg": (bin_low + bin_high) / 2,
                    "actual_win_rate": float(row[2]),
                    "count": int(row[1]),
                }
            )

        return {"bins": result_bins, "total_trades": total_trades}

    def get_model_performance(self, model_version: str) -> dict:
        """SQL aggregation for a specific model version."""
        row = self._conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN o.actual_profitable THEN 1 ELSE 0 END) AS wins,
                AVG(o.actual_return_pct) AS avg_return,
                AVG(s.overall_score) AS avg_score
            FROM outcomes o
            JOIN predictions p ON o.trade_id = p.trade_id
            LEFT JOIN scorecards s ON o.trade_id = s.trade_id
            WHERE p.model_version = ?
            """,
            [model_version],
        ).fetchone()

        if not row or row[0] == 0:
            return {"model_version": model_version, "total_trades": 0}

        total = int(row[0])
        wins = int(row[1])
        return {
            "model_version": model_version,
            "total_trades": total,
            "win_rate": wins / total,
            "avg_return_pct": float(row[2]) if row[2] is not None else 0.0,
            "avg_overall_score": float(row[3]) if row[3] is not None else 0.0,
            "wins": wins,
            "losses": total - wins,
        }

    def get_regime_performance(self) -> dict:
        """SQL GROUP BY market_regime."""
        rows = self._conn.execute(
            """
            SELECT
                COALESCE(p.market_regime, 'unknown') AS regime,
                COUNT(*) AS total,
                SUM(CASE WHEN o.actual_profitable THEN 1 ELSE 0 END) AS wins,
                AVG(o.actual_return_pct) AS avg_return
            FROM outcomes o
            JOIN predictions p ON o.trade_id = p.trade_id
            GROUP BY regime
            """
        ).fetchall()

        result = {}
        for row in rows:
            regime = row[0]
            total = int(row[1])
            wins = int(row[2])
            result[regime] = {
                "total_trades": total,
                "win_rate": wins / total,
                "avg_return_pct": float(row[3]) if row[3] is not None else 0.0,
            }
        return result

    def get_recent_accuracy(self, n_trades: int = 50) -> float:
        """SQL: SELECT from outcomes ORDER BY created_at DESC LIMIT n."""
        row = self._conn.execute(
            """
            SELECT AVG(CASE WHEN actual_profitable THEN 1.0 ELSE 0.0 END)
            FROM (
                SELECT actual_profitable
                FROM outcomes
                ORDER BY created_at DESC
                LIMIT ?
            )
            """,
            [n_trades],
        ).fetchone()
        if not row or row[0] is None:
            return 0.0
        return float(row[0])

    @property
    def total_trades(self) -> int:
        """SELECT COUNT(*) FROM outcomes"""
        row = self._conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()
        return int(row[0]) if row else 0

    @property
    def overall_accuracy(self) -> float:
        """SELECT AVG(actual_profitable::INT) FROM outcomes"""
        row = self._conn.execute(
            "SELECT AVG(CASE WHEN actual_profitable THEN 1.0 ELSE 0.0 END) FROM outcomes"
        ).fetchone()
        if not row or row[0] is None:
            return 0.0
        return float(row[0])

    def get_model_lineage(self, trade_id: str) -> dict:
        """Full lineage: trade -> prediction -> model_version -> feature_hash."""
        row = self._conn.execute(
            """
            SELECT p.prediction_id, p.model_version, p.feature_hash,
                   p.timestamp, p.strategy_name
            FROM predictions p
            WHERE p.trade_id = ?
            """,
            [trade_id],
        ).fetchone()

        if not row:
            return {"trade_id": trade_id, "found": False}

        features = self._conn.execute(
            "SELECT feature_name, feature_value FROM feature_snapshots WHERE trade_id = ?",
            [trade_id],
        ).fetchall()

        return {
            "trade_id": trade_id,
            "found": True,
            "prediction_id": row[0],
            "model_version": row[1],
            "feature_hash": row[2],
            "timestamp": row[3],
            "strategy_name": row[4],
            "feature_count": len(features),
            "features": {f[0]: f[1] for f in features},
        }

    def get_drift_stats(self, window_days: int = 7) -> dict:
        """Compare recent feature distributions vs historical.

        Returns PSI (Population Stability Index) per feature.
        """
        recent_df = self._conn.execute(
            """
            SELECT fs.feature_name, fs.feature_value
            FROM feature_snapshots fs
            JOIN outcomes o ON fs.trade_id = o.trade_id
            WHERE o.created_at >= CURRENT_TIMESTAMP - INTERVAL ? DAY
            """,
            [window_days],
        ).fetchdf()

        historical_df = self._conn.execute(
            """
            SELECT fs.feature_name, fs.feature_value
            FROM feature_snapshots fs
            JOIN outcomes o ON fs.trade_id = o.trade_id
            WHERE o.created_at < CURRENT_TIMESTAMP - INTERVAL ? DAY
            """,
            [window_days],
        ).fetchdf()

        if recent_df.empty or historical_df.empty:
            return {"psi": {}, "window_days": window_days, "status": "insufficient_data"}

        psi_results: dict[str, float] = {}
        features = recent_df["feature_name"].unique()

        for feat in features:
            recent_vals = recent_df[recent_df["feature_name"] == feat]["feature_value"].values
            hist_vals = historical_df[historical_df["feature_name"] == feat]["feature_value"].values

            if len(hist_vals) < 10 or len(recent_vals) < 5:
                continue

            # Compute PSI using 10 quantile bins from historical
            bins = np.quantile(hist_vals, np.linspace(0, 1, 11))
            bins = np.unique(bins)
            if len(bins) < 3:
                continue

            hist_counts, _ = np.histogram(hist_vals, bins=bins)
            recent_counts, _ = np.histogram(recent_vals, bins=bins)

            # Normalize to proportions with smoothing
            eps = 1e-6
            hist_prop = (hist_counts + eps) / (hist_counts.sum() + eps * len(hist_counts))
            recent_prop = (recent_counts + eps) / (recent_counts.sum() + eps * len(recent_counts))

            psi = float(np.sum((recent_prop - hist_prop) * np.log(recent_prop / hist_prop)))
            psi_results[feat] = psi

        return {"psi": psi_results, "window_days": window_days, "status": "ok"}


class PostTradeValidator:
    """Called when a trade closes — computes the scorecard."""

    def __init__(self, experience_db: ExperienceDatabase) -> None:
        self._db = experience_db

    def validate_trade(
        self,
        trade_result: dict,
        signal_package: Optional[Any] = None,
        feature_snapshot: Optional[dict] = None,
        price_history: Optional[list] = None,
    ) -> TradeScorecard:
        """
        Create scorecard from trade result.

        trade_result should contain:
            symbol, side, entry_price, exit_price, pnl, pnl_pct,
            entry_time, exit_time, fees, model_version, strategy

        signal_package (if available) provides predicted metrics.
        price_history provides intra-trade prices for MFE/MAE.
        """
        symbol = trade_result.get("symbol", "UNKNOWN")
        side = trade_result.get("side", "buy")
        entry_price = trade_result.get("entry_price", 0.0)
        exit_price = trade_result.get("exit_price", 0.0)
        pnl_pct = trade_result.get("pnl_pct", 0.0)
        entry_time = trade_result.get("entry_time", datetime.now())
        exit_time = trade_result.get("exit_time", datetime.now())
        fees = trade_result.get("fees", 0.0)
        model_version = trade_result.get("model_version", "unknown")
        strategy = trade_result.get("strategy", "unknown")

        # Parse times
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        if isinstance(exit_time, str):
            exit_time = datetime.fromisoformat(exit_time)

        actual_duration_minutes = int(
            (exit_time - entry_time).total_seconds() / 60
        )
        actual_profitable = pnl_pct > 0

        # Extract predicted values from signal_package
        predicted_confidence = 0.5
        predicted_win_probability = 0.5
        predicted_return_pct = 0.0
        predicted_risk_reward = 1.0
        predicted_duration_minutes = 0
        signal_id = ""
        stop_loss_price = 0.0
        take_profit_prices: list[float] = []
        market_regime = "unknown"
        volatility_level = "unknown"

        if signal_package is not None:
            predicted_confidence = getattr(
                signal_package, "confidence", predicted_confidence
            )
            predicted_win_probability = getattr(
                signal_package, "win_probability", predicted_win_probability
            )
            predicted_return_pct = getattr(
                signal_package, "predicted_return_pct", predicted_return_pct
            )
            predicted_risk_reward = getattr(
                signal_package, "risk_reward_ratio", predicted_risk_reward
            )
            predicted_duration_minutes = getattr(
                signal_package, "predicted_duration_minutes", predicted_duration_minutes
            )
            signal_id = getattr(signal_package, "signal_id", "")
            stop_loss_price = getattr(signal_package, "stop_loss", 0.0)
            take_profit_prices = getattr(signal_package, "take_profit_prices", [])
            market_regime = getattr(signal_package, "market_regime", "unknown")
            volatility_level = getattr(signal_package, "volatility_level", "unknown")

        if not signal_id:
            signal_id = str(uuid.uuid4())[:8]

        # Compute MFE/MAE from price history
        mfe = 0.0
        mae = 0.0
        if price_history:
            prices = [float(p) for p in price_history]
            if side == "buy":
                mfe = max(prices) - entry_price
                mae = entry_price - min(prices)
            else:
                mfe = entry_price - min(prices)
                mae = max(prices) - entry_price

        # Check stop loss hit
        stop_loss_hit = False
        if stop_loss_price > 0:
            if side == "buy":
                stop_loss_hit = (exit_price <= stop_loss_price) or (
                    price_history
                    and min(float(p) for p in price_history) <= stop_loss_price
                )
            else:
                stop_loss_hit = (exit_price >= stop_loss_price) or (
                    price_history
                    and max(float(p) for p in price_history) >= stop_loss_price
                )

        # Check which take profits were reached
        take_profit_reached: list[bool] = []
        for tp in take_profit_prices:
            if price_history:
                if side == "buy":
                    reached = max(float(p) for p in price_history) >= tp
                else:
                    reached = min(float(p) for p in price_history) <= tp
                take_profit_reached.append(reached)
            else:
                if side == "buy":
                    take_profit_reached.append(exit_price >= tp)
                else:
                    take_profit_reached.append(exit_price <= tp)

        # Compute actual risk/reward
        actual_risk_reward = 0.0
        if mae > 0:
            actual_profit = abs(exit_price - entry_price)
            actual_risk_reward = actual_profit / mae

        # Compute slippage
        slippage_pct = trade_result.get("slippage_pct", 0.0)

        # Compute scores
        direction_score = 1.0 if actual_profitable else 0.0

        confidence_calibration_error = abs(
            predicted_confidence - (1.0 if actual_profitable else 0.0)
        )

        # Timing score: 1.0 if exact match, decays with error
        if predicted_duration_minutes > 0:
            duration_error = abs(
                predicted_duration_minutes - actual_duration_minutes
            )
            timing_score = max(
                0.0, 1.0 - (duration_error / max(predicted_duration_minutes, 1))
            )
        else:
            timing_score = 0.5

        # Exit efficiency: how much of MFE was captured
        if mfe > 0:
            actual_capture = abs(exit_price - entry_price)
            exit_efficiency = min(1.0, actual_capture / mfe)
        else:
            exit_efficiency = 0.0 if not actual_profitable else 1.0

        # Overall score (composite 0-100)
        overall_score = (
            direction_score * 40
            + (1.0 - confidence_calibration_error) * 20
            + timing_score * 15
            + exit_efficiency * 25
        )

        trade_id = trade_result.get("trade_id", str(uuid.uuid4())[:12])

        scorecard = TradeScorecard(
            trade_id=trade_id,
            signal_id=signal_id,
            symbol=symbol,
            timestamp=exit_time,
            predicted_direction=side,
            actual_profitable=actual_profitable,
            predicted_confidence=predicted_confidence,
            predicted_win_probability=predicted_win_probability,
            predicted_return_pct=predicted_return_pct,
            actual_return_pct=pnl_pct,
            predicted_risk_reward=predicted_risk_reward,
            actual_risk_reward=actual_risk_reward,
            predicted_duration_minutes=predicted_duration_minutes,
            actual_duration_minutes=actual_duration_minutes,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss_price=stop_loss_price,
            stop_loss_hit=stop_loss_hit,
            take_profit_prices=take_profit_prices,
            take_profit_reached=take_profit_reached,
            max_favorable_excursion=mfe,
            max_adverse_excursion=mae,
            model_version=model_version,
            strategy_name=strategy,
            market_regime=market_regime,
            volatility_level=volatility_level,
            feature_vector=feature_snapshot or {},
            slippage_pct=slippage_pct,
            fees=fees,
            direction_score=direction_score,
            confidence_calibration_error=confidence_calibration_error,
            timing_score=timing_score,
            exit_efficiency=exit_efficiency,
            overall_score=overall_score,
        )

        self._db.record_trade(scorecard)
        logger.info(
            "trade_validated",
            trade_id=trade_id,
            symbol=symbol,
            profitable=actual_profitable,
            overall_score=round(overall_score, 1),
        )

        return scorecard


class ContinuousCalibrationMonitor:
    """Periodically checks if model confidence is well-calibrated."""

    def __init__(
        self, experience_db: ExperienceDatabase, tolerance: float = 0.10
    ) -> None:
        self._db = experience_db
        self._tolerance = tolerance

    def check_calibration(self) -> dict:
        """
        Returns calibration report with ECE, bin details, and recommendation.
        """
        cal_data = self._db.get_calibration_data()
        bins = cal_data.get("bins", [])
        total_trades = cal_data.get("total_trades", 0)

        if not bins or total_trades == 0:
            return {
                "is_calibrated": True,
                "ece": 0.0,
                "bins": [],
                "recommendation": "insufficient_data",
                "worst_bin": None,
            }

        # Compute Expected Calibration Error (ECE)
        ece = 0.0
        worst_bin = None
        worst_error = 0.0

        for b in bins:
            bin_error = abs(b["predicted_avg"] - b["actual_win_rate"])
            weight = b["count"] / total_trades
            ece += weight * bin_error

            if bin_error > worst_error:
                worst_error = bin_error
                worst_bin = {**b, "error": bin_error}

        is_calibrated = ece <= self._tolerance

        # Determine recommendation
        if is_calibrated:
            recommendation = "well_calibrated"
        elif ece <= self._tolerance * 2:
            recommendation = "needs_recalibration"
        else:
            recommendation = "needs_retraining"

        report = {
            "is_calibrated": is_calibrated,
            "ece": float(ece),
            "bins": bins,
            "recommendation": recommendation,
            "worst_bin": worst_bin,
        }

        logger.info(
            "calibration_checked",
            ece=round(ece, 4),
            is_calibrated=is_calibrated,
            recommendation=recommendation,
        )

        return report

    def needs_recalibration(self) -> bool:
        """True if calibration has drifted beyond tolerance."""
        report = self.check_calibration()
        return not report["is_calibrated"]

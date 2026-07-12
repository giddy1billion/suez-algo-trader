from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy import func, text
from sqlalchemy.orm import sessionmaker

from src.ml.models import (
    MLBase,
    MLFeatureSnapshotExp,
    MLOutcome,
    MLPrediction,
    MLScorecard,
)
from src.utils.database import create_db_engine
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

    Uses SQLAlchemy ORM for portable storage across SQLite and PostgreSQL.
    """

    def __init__(
        self,
        storage_path: str = "data_cache/experience",
        database_url: str = None,
    ) -> None:
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

        if database_url:
            self._engine = create_db_engine(database_url)
        else:
            db_path = self._storage_path / "experience.db"
            self._engine = create_db_engine(f"sqlite:///{db_path}")

        MLBase.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        self._check_legacy_migration()
        logger.info("experience_db_initialized", path=str(self._storage_path))

    def _check_legacy_migration(self) -> None:
        """Warn if legacy JSONL file exists."""
        legacy_path = self._storage_path / "scorecards.jsonl"
        if legacy_path.exists():
            logger.warning(
                "legacy_jsonl_detected",
                path=str(legacy_path),
                message="Legacy JSONL file found. Consider migrating with "
                "ExperienceDatabase.migrate_from_jsonl().",
            )

    def migrate_from_jsonl(self) -> int:
        """Migrate legacy JSONL data into the database. Returns count of migrated records."""
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
        """Upsert into ml_predictions, ml_outcomes, ml_scorecards, ml_feature_snapshots_exp."""
        ts = scorecard.timestamp
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)

        feature_hash = _compute_feature_hash(scorecard.feature_vector)
        prediction_id = f"{scorecard.trade_id}_{scorecard.signal_id}"

        with self._write_lock:
            with self._Session() as session:
                # Upsert prediction
                pred = MLPrediction(
                    prediction_id=prediction_id,
                    trade_id=scorecard.trade_id,
                    symbol=scorecard.symbol,
                    timestamp=ts,
                    model_version=scorecard.model_version,
                    strategy_name=scorecard.strategy_name,
                    predicted_direction=scorecard.predicted_direction,
                    predicted_confidence=scorecard.predicted_confidence,
                    predicted_win_probability=scorecard.predicted_win_probability,
                    predicted_return_pct=scorecard.predicted_return_pct,
                    predicted_duration_minutes=scorecard.predicted_duration_minutes,
                    predicted_risk_reward=scorecard.predicted_risk_reward,
                    market_regime=scorecard.market_regime,
                    volatility_level=scorecard.volatility_level,
                    feature_hash=feature_hash,
                    contract_id=scorecard.contract_id,
                    created_at=datetime.now(timezone.utc),
                )
                session.merge(pred)

                # Upsert outcome
                outcome = MLOutcome(
                    trade_id=scorecard.trade_id,
                    symbol=scorecard.symbol,
                    actual_profitable=scorecard.actual_profitable,
                    actual_return_pct=scorecard.actual_return_pct,
                    actual_duration_minutes=scorecard.actual_duration_minutes,
                    actual_risk_reward=scorecard.actual_risk_reward,
                    entry_price=scorecard.entry_price,
                    exit_price=scorecard.exit_price,
                    stop_loss_price=scorecard.stop_loss_price,
                    stop_loss_hit=scorecard.stop_loss_hit,
                    max_favorable_excursion=scorecard.max_favorable_excursion,
                    max_adverse_excursion=scorecard.max_adverse_excursion,
                    slippage_pct=scorecard.slippage_pct,
                    fees=scorecard.fees,
                    contract_id=scorecard.contract_id,
                    contract_decision=scorecard.contract_decision,
                    contract_confidence=scorecard.contract_confidence,
                    created_at=datetime.now(timezone.utc),
                )
                session.merge(outcome)

                # Upsert scorecard
                sc = MLScorecard(
                    trade_id=scorecard.trade_id,
                    symbol=scorecard.symbol,
                    timestamp=ts,
                    direction_score=scorecard.direction_score,
                    confidence_calibration_error=scorecard.confidence_calibration_error,
                    timing_score=scorecard.timing_score,
                    exit_efficiency=scorecard.exit_efficiency,
                    overall_score=scorecard.overall_score,
                    model_version=scorecard.model_version,
                    strategy_name=scorecard.strategy_name,
                    market_regime=scorecard.market_regime,
                    contract_id=scorecard.contract_id,
                    created_at=datetime.now(timezone.utc),
                )
                session.merge(sc)

                # Upsert feature snapshots
                if scorecard.feature_vector:
                    for fname, fval in scorecard.feature_vector.items():
                        try:
                            fval_float = float(fval)
                        except (TypeError, ValueError):
                            continue
                        fs = MLFeatureSnapshotExp(
                            trade_id=scorecard.trade_id,
                            feature_name=fname,
                            feature_value=fval_float,
                        )
                        session.merge(fs)

                session.commit()

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
        with self._Session() as session:
            query = session.query(func.count(MLOutcome.trade_id))
            if since is not None:
                query = query.filter(MLOutcome.created_at >= since)
            count = query.scalar() or 0

        if count < min_trades:
            if count == 0:
                logger.info(
                    "training_data.bootstrapping",
                    available=count,
                    required=min_trades,
                    msg="No trade outcomes yet — experience enrichment skipped",
                )
            else:
                logger.warning(
                    "insufficient_training_data", available=count, required=min_trades
                )
            return pd.DataFrame()

        sql = text("""
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
            FROM ml_outcomes o
            LEFT JOIN ml_predictions p ON o.trade_id = p.trade_id
            LEFT JOIN ml_scorecards s ON o.trade_id = s.trade_id
            {where_clause}
            ORDER BY o.created_at
        """.format(where_clause="WHERE o.created_at >= :since_param" if since else ""))

        with self._engine.connect() as conn:
            if since is not None:
                df = pd.read_sql(sql, conn, params={"since_param": since})
            else:
                df = pd.read_sql(sql, conn)

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
        with self._Session() as session:
            total_trades = session.query(func.count(MLOutcome.trade_id)).scalar() or 0

        if total_trades == 0:
            return {"bins": [], "total_trades": 0}

        sql = text("""
            SELECT
                FLOOR(p.predicted_confidence * 10) / 10.0 AS bin_low,
                COUNT(*) AS cnt,
                AVG(CASE WHEN o.actual_profitable THEN 1.0 ELSE 0.0 END) AS actual_win_rate
            FROM ml_predictions p
            JOIN ml_outcomes o ON p.trade_id = o.trade_id
            GROUP BY bin_low
            ORDER BY bin_low
        """)

        with self._engine.connect() as conn:
            result = conn.execute(sql)
            rows = result.fetchall()

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
        sql = text("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN o.actual_profitable THEN 1 ELSE 0 END) AS wins,
                AVG(o.actual_return_pct) AS avg_return,
                AVG(s.overall_score) AS avg_score
            FROM ml_outcomes o
            JOIN ml_predictions p ON o.trade_id = p.trade_id
            LEFT JOIN ml_scorecards s ON o.trade_id = s.trade_id
            WHERE p.model_version = :model_ver
        """)

        with self._engine.connect() as conn:
            row = conn.execute(sql, {"model_ver": model_version}).fetchone()

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
        sql = text("""
            SELECT
                COALESCE(p.market_regime, 'unknown') AS regime,
                COUNT(*) AS total,
                SUM(CASE WHEN o.actual_profitable THEN 1 ELSE 0 END) AS wins,
                AVG(o.actual_return_pct) AS avg_return
            FROM ml_outcomes o
            JOIN ml_predictions p ON o.trade_id = p.trade_id
            GROUP BY regime
        """)

        with self._engine.connect() as conn:
            rows = conn.execute(sql).fetchall()

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
        """SELECT from outcomes ORDER BY created_at DESC LIMIT n."""
        sql = text("""
            SELECT AVG(CASE WHEN actual_profitable THEN 1.0 ELSE 0.0 END)
            FROM (
                SELECT actual_profitable
                FROM ml_outcomes
                ORDER BY created_at DESC
                LIMIT :n_limit
            ) sub
        """)

        with self._engine.connect() as conn:
            row = conn.execute(sql, {"n_limit": n_trades}).fetchone()
        if not row or row[0] is None:
            return 0.0
        return float(row[0])

    @property
    def total_trades(self) -> int:
        """SELECT COUNT(*) FROM ml_outcomes"""
        with self._Session() as session:
            count = session.query(func.count(MLOutcome.trade_id)).scalar()
        return int(count) if count else 0

    @property
    def overall_accuracy(self) -> float:
        """Average win rate across all outcomes."""
        sql = text(
            "SELECT AVG(CASE WHEN actual_profitable THEN 1.0 ELSE 0.0 END) FROM ml_outcomes"
        )
        with self._engine.connect() as conn:
            row = conn.execute(sql).fetchone()
        if not row or row[0] is None:
            return 0.0
        return float(row[0])

    def get_model_lineage(self, trade_id: str) -> dict:
        """Full lineage: trade -> prediction -> model_version -> feature_hash."""
        with self._Session() as session:
            pred = (
                session.query(MLPrediction)
                .filter(MLPrediction.trade_id == trade_id)
                .first()
            )

            if not pred:
                return {"trade_id": trade_id, "found": False}

            features = (
                session.query(
                    MLFeatureSnapshotExp.feature_name,
                    MLFeatureSnapshotExp.feature_value,
                )
                .filter(MLFeatureSnapshotExp.trade_id == trade_id)
                .all()
            )

        return {
            "trade_id": trade_id,
            "found": True,
            "prediction_id": pred.prediction_id,
            "model_version": pred.model_version,
            "feature_hash": pred.feature_hash,
            "timestamp": pred.timestamp,
            "strategy_name": pred.strategy_name,
            "feature_count": len(features),
            "features": {f[0]: f[1] for f in features},
        }

    def get_drift_stats(self, window_days: int = 7) -> dict:
        """Compare recent feature distributions vs historical.

        Returns PSI (Population Stability Index) per feature.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

        recent_sql = text("""
            SELECT fs.feature_name, fs.feature_value
            FROM ml_feature_snapshots_exp fs
            JOIN ml_outcomes o ON fs.trade_id = o.trade_id
            WHERE o.created_at >= :cutoff
        """)

        historical_sql = text("""
            SELECT fs.feature_name, fs.feature_value
            FROM ml_feature_snapshots_exp fs
            JOIN ml_outcomes o ON fs.trade_id = o.trade_id
            WHERE o.created_at < :cutoff
        """)

        with self._engine.connect() as conn:
            recent_df = pd.read_sql(recent_sql, conn, params={"cutoff": cutoff})
            historical_df = pd.read_sql(historical_sql, conn, params={"cutoff": cutoff})

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

        # Extract contract information from trade_result
        contract_id = trade_result.get("contract_id", "")
        contract_decision = trade_result.get("contract_decision", "")
        contract_confidence = trade_result.get("contract_confidence", 0.0)

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
            contract_id=contract_id,
            contract_decision=contract_decision,
            contract_confidence=contract_confidence,
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

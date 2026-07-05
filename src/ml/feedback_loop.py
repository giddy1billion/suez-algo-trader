from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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

    # Execution quality
    slippage_pct: float = 0.0
    fees: float = 0.0

    # Scoring
    direction_score: float = 0.0  # 1.0 if correct, 0.0 if wrong
    confidence_calibration_error: float = 0.0
    timing_score: float = 0.0  # How close predicted vs actual duration
    exit_efficiency: float = 0.0  # actual_profit / max_possible_profit (MFE)
    overall_score: float = 0.0  # Composite 0-100


class ExperienceDatabase:
    """Stores all scorecards and provides query interface for the training pipeline."""

    def __init__(self, storage_path: str = "data_cache/experience") -> None:
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._file_path = self._storage_path / "scorecards.jsonl"
        logger.info("experience_db_initialized", path=str(self._file_path))

    def record_trade(self, scorecard: TradeScorecard) -> None:
        """Append a completed trade scorecard."""
        record = asdict(scorecard)
        # Serialize datetime to ISO format
        if isinstance(record.get("timestamp"), datetime):
            record["timestamp"] = record["timestamp"].isoformat()
        with open(self._file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(
            "trade_recorded",
            trade_id=scorecard.trade_id,
            symbol=scorecard.symbol,
            score=scorecard.overall_score,
        )

    def _load_all(self) -> list[dict]:
        """Load all scorecards from JSONL file."""
        if not self._file_path.exists():
            return []
        records = []
        with open(self._file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("skipping_malformed_line")
        return records

    def get_training_samples(
        self, min_trades: int = 50, since: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Export scorecards as training-ready DataFrame.
        Returns features + labels (profitable, confidence_target, duration_target).
        """
        records = self._load_all()
        if since is not None:
            since_iso = since.isoformat()
            records = [r for r in records if r.get("timestamp", "") >= since_iso]

        if len(records) < min_trades:
            logger.warning(
                "insufficient_training_data",
                available=len(records),
                required=min_trades,
            )
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Add derived label columns
        df["profitable"] = df["actual_profitable"].astype(int)
        df["confidence_target"] = df.apply(
            lambda row: 1.0 if row["actual_profitable"] else 0.0, axis=1
        )
        df["duration_target"] = df["actual_duration_minutes"]

        logger.info("training_samples_exported", count=len(df))
        return df

    def get_calibration_data(self) -> dict:
        """Return confidence bins with actual win rates for calibration."""
        records = self._load_all()
        if not records:
            return {"bins": [], "total_trades": 0}

        bins = np.arange(0.0, 1.1, 0.1)
        bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
        result_bins = []

        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            in_bin = [
                r
                for r in records
                if low <= r.get("predicted_confidence", 0) < high
            ]
            if in_bin:
                actual_win_rate = sum(
                    1 for r in in_bin if r.get("actual_profitable", False)
                ) / len(in_bin)
                result_bins.append(
                    {
                        "bin": bin_labels[i],
                        "predicted_avg": (low + high) / 2,
                        "actual_win_rate": actual_win_rate,
                        "count": len(in_bin),
                    }
                )

        return {"bins": result_bins, "total_trades": len(records)}

    def get_model_performance(self, model_version: str) -> dict:
        """Get aggregate performance metrics for a specific model version."""
        records = self._load_all()
        model_records = [
            r for r in records if r.get("model_version") == model_version
        ]

        if not model_records:
            return {"model_version": model_version, "total_trades": 0}

        wins = sum(1 for r in model_records if r.get("actual_profitable", False))
        total = len(model_records)
        avg_return = np.mean([r.get("actual_return_pct", 0) for r in model_records])
        avg_score = np.mean([r.get("overall_score", 0) for r in model_records])

        return {
            "model_version": model_version,
            "total_trades": total,
            "win_rate": wins / total,
            "avg_return_pct": float(avg_return),
            "avg_overall_score": float(avg_score),
            "wins": wins,
            "losses": total - wins,
        }

    def get_regime_performance(self) -> dict:
        """Performance breakdown by market regime."""
        records = self._load_all()
        if not records:
            return {}

        regimes: dict[str, list[dict]] = {}
        for r in records:
            regime = r.get("market_regime", "unknown")
            regimes.setdefault(regime, []).append(r)

        result = {}
        for regime, regime_records in regimes.items():
            wins = sum(
                1 for r in regime_records if r.get("actual_profitable", False)
            )
            total = len(regime_records)
            avg_return = np.mean(
                [r.get("actual_return_pct", 0) for r in regime_records]
            )
            result[regime] = {
                "total_trades": total,
                "win_rate": wins / total,
                "avg_return_pct": float(avg_return),
            }

        return result

    def get_recent_accuracy(self, n_trades: int = 50) -> float:
        """Rolling accuracy of recent predictions."""
        records = self._load_all()
        if not records:
            return 0.0

        recent = records[-n_trades:]
        wins = sum(1 for r in recent if r.get("actual_profitable", False))
        return wins / len(recent)

    @property
    def total_trades(self) -> int:
        return len(self._load_all())

    @property
    def overall_accuracy(self) -> float:
        records = self._load_all()
        if not records:
            return 0.0
        wins = sum(1 for r in records if r.get("actual_profitable", False))
        return wins / len(records)


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

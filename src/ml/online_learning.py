"""
Online Learning Loop — Transforms trade outcomes into adaptive intelligence.

Pipeline:
    Trade Closes
        ↓
    Outcome Labeled (ExperienceDB)
        ↓
    Confidence Recalibration
        ↓
    Reward Attribution (multi-objective)
        ↓
    Hard Example Mining
        ↓
    Next Training Cycle (weighted samples)

This module does NOT do online gradient updates (too dangerous for trading).
Instead, it curates the HIGHEST-VALUE training samples for the next batch
retrain, and recalibrates confidence between retrains.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RewardAttribution:
    """Multi-objective reward for a completed trade."""

    trade_id: str = ""
    # Component rewards (each -1.0 to +1.0)
    pnl_reward: float = 0.0           # risk-adjusted return
    drawdown_penalty: float = 0.0      # penalize large adverse excursion
    duration_reward: float = 0.0       # bonus for hitting predicted duration
    confidence_accuracy: float = 0.0   # was confidence calibrated?
    capital_efficiency: float = 0.0    # return per unit time in market
    slippage_penalty: float = 0.0      # penalize high slippage

    # Composite
    composite_reward: float = 0.0

    # Weights used
    weights: dict = field(default_factory=lambda: {
        "pnl": 0.35,
        "drawdown": 0.20,
        "duration": 0.10,
        "confidence": 0.15,
        "capital_efficiency": 0.10,
        "slippage": 0.10,
    })

    def compute_composite(self) -> float:
        """Weighted sum of all reward components."""
        w = self.weights
        self.composite_reward = (
            w["pnl"] * self.pnl_reward
            + w["drawdown"] * self.drawdown_penalty
            + w["duration"] * self.duration_reward
            + w["confidence"] * self.confidence_accuracy
            + w["capital_efficiency"] * self.capital_efficiency
            + w["slippage"] * self.slippage_penalty
        )
        return self.composite_reward


@dataclass
class HardExample:
    """A training sample identified as high-value for learning."""

    trade_id: str = ""
    symbol: str = ""
    reason: str = ""  # why this is a hard example
    priority: float = 0.0  # higher = more important
    feature_snapshot_id: str = ""
    predicted_confidence: float = 0.0
    actual_outcome: float = 0.0
    regime: str = ""


class OnlineLearningLoop:
    """
    Curates high-value training samples and recalibrates confidence.

    NOT online gradient descent. Instead:
    1. Identifies which trades the model got WRONG (hard examples)
    2. Identifies calibration gaps (overconfident/underconfident)
    3. Computes multi-objective rewards for experience weighting
    4. Prepares priority-weighted sample set for next training cycle
    """

    def __init__(
        self,
        experience_db=None,
        max_hard_examples: int = 500,
        confidence_recalibration_window: int = 100,
    ):
        self._experience_db = experience_db
        self._max_hard_examples = max_hard_examples
        self._calibration_window = confidence_recalibration_window

        self._hard_examples: list = []
        self._reward_history: list = []
        self._confidence_calibration: dict = {}  # bucket → [predicted, actual]
        self._lock = threading.Lock()

    def process_trade_outcome(self, trade_result: dict) -> Optional[RewardAttribution]:
        """
        Process a completed trade through the online learning pipeline.

        Returns the multi-objective reward attribution.
        """
        reward = self._compute_reward(trade_result)

        # Mine hard examples
        hard_example = self._identify_hard_example(trade_result, reward)
        if hard_example:
            with self._lock:
                self._hard_examples.append(hard_example)
                if len(self._hard_examples) > self._max_hard_examples:
                    # Keep highest priority examples
                    self._hard_examples.sort(key=lambda x: x.priority, reverse=True)
                    self._hard_examples = self._hard_examples[:self._max_hard_examples]

        # Update confidence calibration
        self._update_calibration(trade_result)

        # Store reward
        with self._lock:
            self._reward_history.append(reward)
            if len(self._reward_history) > 1000:
                self._reward_history = self._reward_history[-1000:]

        return reward

    def _compute_reward(self, trade_result: dict) -> RewardAttribution:
        """Compute multi-objective reward for a trade."""
        reward = RewardAttribution(trade_id=trade_result.get("trade_id", ""))

        # PnL reward (risk-adjusted: normalize by confidence)
        pnl_pct = trade_result.get("pnl_pct", 0.0)
        confidence = trade_result.get("predicted_confidence", trade_result.get("confidence", 0.5))

        # Higher confidence should produce higher returns; penalize overconfident losses
        if pnl_pct > 0:
            reward.pnl_reward = min(pnl_pct / 5.0, 1.0)  # Cap at 1.0 for +5%
        else:
            # Penalize more harshly when confidence was high
            reward.pnl_reward = max(pnl_pct / 3.0 * confidence, -1.0)

        # Drawdown penalty (from MAE if available)
        mae_pct = trade_result.get("max_adverse_excursion_pct", abs(min(pnl_pct, 0)))
        reward.drawdown_penalty = -min(mae_pct / 10.0, 1.0)  # -1.0 at 10% MAE

        # Duration reward (bonus if actual matches predicted)
        predicted_duration = trade_result.get("predicted_duration_hours", 0)
        actual_duration = trade_result.get("actual_duration_hours", 0)
        if predicted_duration > 0 and actual_duration > 0:
            ratio = actual_duration / predicted_duration
            # 0.8-1.2x is good
            if 0.8 <= ratio <= 1.2:
                reward.duration_reward = 0.5
            elif 0.5 <= ratio <= 2.0:
                reward.duration_reward = 0.0
            else:
                reward.duration_reward = -0.3

        # Confidence accuracy
        won = pnl_pct > 0
        predicted_win = confidence > 0.5
        if won == predicted_win:
            reward.confidence_accuracy = confidence if won else (1 - confidence)
        else:
            # Model was wrong — penalize proportional to confidence
            reward.confidence_accuracy = -(confidence if not won else (1 - confidence))

        # Capital efficiency (return per hour)
        duration_hours = trade_result.get("actual_duration_hours", 1.0) or 1.0
        reward.capital_efficiency = min(pnl_pct / max(duration_hours, 0.5), 0.5)

        # Slippage penalty
        slippage_bps = trade_result.get("slippage_bps", 0)
        reward.slippage_penalty = -min(slippage_bps / 50.0, 0.5)  # -0.5 at 50bps

        reward.compute_composite()
        return reward

    def _identify_hard_example(self, trade_result: dict, reward: RewardAttribution) -> Optional[HardExample]:
        """Identify if this trade is a high-value training example."""
        confidence = trade_result.get("predicted_confidence", trade_result.get("confidence", 0.5))
        pnl_pct = trade_result.get("pnl_pct", 0.0)
        won = pnl_pct > 0

        reason = None
        priority = 0.0

        # High confidence wrong — MOST valuable for calibration
        if confidence > 0.8 and not won:
            reason = f"High confidence ({confidence:.2f}) loss ({pnl_pct:.2f}%)"
            priority = confidence * abs(pnl_pct)

        # Low confidence win — model underestimated opportunity
        elif confidence < 0.6 and won and pnl_pct > 2.0:
            reason = f"Low confidence ({confidence:.2f}) big win ({pnl_pct:.2f}%)"
            priority = pnl_pct * (1 - confidence)

        # Extreme loss regardless of confidence
        elif pnl_pct < -3.0:
            reason = f"Extreme loss ({pnl_pct:.2f}%) at confidence {confidence:.2f}"
            priority = abs(pnl_pct)

        # Regime mismatch (predicted regime != actual regime)
        elif trade_result.get("regime_mismatch", False):
            reason = "Regime mismatch during trade"
            priority = abs(pnl_pct) * 1.5

        if reason is None:
            return None

        return HardExample(
            trade_id=trade_result.get("trade_id", ""),
            symbol=trade_result.get("symbol", ""),
            reason=reason,
            priority=priority,
            feature_snapshot_id=trade_result.get("feature_snapshot_id", ""),
            predicted_confidence=confidence,
            actual_outcome=pnl_pct,
            regime=trade_result.get("market_regime", ""),
        )

    def _update_calibration(self, trade_result: dict) -> None:
        """Update running confidence calibration statistics."""
        confidence = trade_result.get("predicted_confidence", trade_result.get("confidence", 0.0))
        if confidence <= 0:
            return

        won = 1.0 if trade_result.get("pnl_pct", 0.0) > 0 else 0.0

        # Bucket by confidence decile
        bucket = f"{int(confidence * 10) / 10:.1f}"
        with self._lock:
            if bucket not in self._confidence_calibration:
                self._confidence_calibration[bucket] = []
            self._confidence_calibration[bucket].append({
                "predicted": confidence,
                "actual": won,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            # Keep window
            if len(self._confidence_calibration[bucket]) > self._calibration_window:
                self._confidence_calibration[bucket] = self._confidence_calibration[bucket][-self._calibration_window:]

    def get_calibration_report(self) -> dict:
        """Get current confidence calibration by bucket."""
        report = {}
        with self._lock:
            for bucket, entries in self._confidence_calibration.items():
                if not entries:
                    continue
                predicted_avg = np.mean([e["predicted"] for e in entries])
                actual_avg = np.mean([e["actual"] for e in entries])
                report[bucket] = {
                    "predicted_avg": round(float(predicted_avg), 3),
                    "actual_win_rate": round(float(actual_avg), 3),
                    "gap": round(float(predicted_avg - actual_avg), 3),
                    "sample_count": len(entries),
                    "overconfident": float(predicted_avg) > float(actual_avg) + 0.05,
                }
        return report

    def get_hard_examples(self, limit: int = 50) -> list:
        """Get highest-priority hard examples for next training."""
        with self._lock:
            sorted_examples = sorted(self._hard_examples, key=lambda x: x.priority, reverse=True)
            return sorted_examples[:limit]

    def get_sample_weights(self, trade_ids: list) -> dict:
        """
        Get sample weights for training based on reward attribution.

        Hard examples get higher weight. Well-predicted easy examples get lower weight.
        This focuses the next training cycle on where the model struggles.
        """
        with self._lock:
            reward_map = {r.trade_id: r for r in self._reward_history}
            hard_ids = {h.trade_id for h in self._hard_examples}

        weights = {}
        for tid in trade_ids:
            if tid in hard_ids:
                # Hard examples get 2-3x weight
                weights[tid] = 2.5
            elif tid in reward_map:
                reward = reward_map[tid]
                # Poorly predicted trades get higher weight
                if reward.confidence_accuracy < -0.3:
                    weights[tid] = 2.0
                elif reward.composite_reward < -0.5:
                    weights[tid] = 1.8
                else:
                    weights[tid] = 1.0
            else:
                weights[tid] = 1.0

        return weights

    def get_status(self) -> dict:
        """Current online learning status."""
        with self._lock:
            return {
                "hard_examples_count": len(self._hard_examples),
                "reward_history_count": len(self._reward_history),
                "calibration_buckets": len(self._confidence_calibration),
                "avg_composite_reward": (
                    round(float(np.mean([r.composite_reward for r in self._reward_history])), 4)
                    if self._reward_history else 0.0
                ),
                "calibration_report": self.get_calibration_report(),
            }

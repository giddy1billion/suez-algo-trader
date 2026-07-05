"""Confidence drift monitoring — early warning for distribution shifts.

Key insight: If average confidence collapses from 0.81 to 0.49, something
changed BEFORE losses materialize. Detect the shift early and alert.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default window sizes (in number of observations, not time)
DEFAULT_LONG_WINDOW = 5000  # ~30 days of signals
DEFAULT_SHORT_WINDOW = 200  # ~1 day of signals


@dataclass
class ConfidenceDriftAlert:
    """Alert when confidence distribution shifts significantly."""

    alert_type: str  # "mean_drop" | "variance_spike" | "distribution_shift"
    severity: str  # "warning" | "critical"
    current_mean: float
    baseline_mean: float
    shift_magnitude: float
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""

    def to_dict(self) -> dict:
        """Serialize for logging/alerting."""
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "current_mean": round(self.current_mean, 4),
            "baseline_mean": round(self.baseline_mean, 4),
            "shift_magnitude": round(self.shift_magnitude, 4),
            "triggered_at": self.triggered_at.isoformat(),
            "message": self.message,
        }


class ConfidenceDriftMonitor:
    """Monitors confidence distribution for drift and anomalies.

    Maintains rolling long-window (baseline) and short-window (recent)
    distributions. Periodically compares them to detect shifts in mean,
    variance, or overall distribution shape.

    Lightweight: no heavy computation on every record call. Checks are
    batched and run only when explicitly called or at intervals.
    """

    def __init__(
        self,
        long_window_size: int = DEFAULT_LONG_WINDOW,
        short_window_size: int = DEFAULT_SHORT_WINDOW,
        mean_drop_warning: float = 0.15,
        mean_drop_critical: float = 0.30,
        variance_spike_factor: float = 2.0,
        kl_divergence_threshold: float = 0.5,
    ) -> None:
        """Initialize the monitor.

        Args:
            long_window_size: Number of observations for baseline window (~30 days)
            short_window_size: Number of observations for recent window (~1 day)
            mean_drop_warning: Fractional mean drop for warning alert
            mean_drop_critical: Fractional mean drop for critical alert
            variance_spike_factor: Variance multiplier for spike alert
            kl_divergence_threshold: KL divergence threshold for distribution shift alert
        """
        self._long_window: deque[float] = deque(maxlen=long_window_size)
        self._short_window: deque[float] = deque(maxlen=short_window_size)

        self.mean_drop_warning = mean_drop_warning
        self.mean_drop_critical = mean_drop_critical
        self.variance_spike_factor = variance_spike_factor
        self.kl_divergence_threshold = kl_divergence_threshold

        # Cache baseline stats to avoid recomputing on every check
        self._baseline_mean: Optional[float] = None
        self._baseline_variance: Optional[float] = None
        self._baseline_histogram: Optional[np.ndarray] = None
        self._records_since_baseline_update: int = 0
        self._baseline_update_interval: int = 100  # Recompute baseline every N records

    def record(self, confidence: float) -> None:
        """Add a confidence observation to the history.

        Lightweight operation — no heavy computation here.

        Args:
            confidence: Confidence score (0.0-1.0) from a prediction
        """
        self._long_window.append(confidence)
        self._short_window.append(confidence)
        self._records_since_baseline_update += 1

        # Periodically refresh baseline cache
        if self._records_since_baseline_update >= self._baseline_update_interval:
            self._update_baseline_cache()

    def check(self) -> Optional[ConfidenceDriftAlert]:
        """Check for confidence distribution drift.

        Compares short window (recent) against long window (baseline).
        Returns the most severe alert found, or None if no drift detected.

        Returns:
            ConfidenceDriftAlert if drift detected, None otherwise
        """
        # Need minimum data in both windows
        min_long = 100
        min_short = 20

        if len(self._long_window) < min_long or len(self._short_window) < min_short:
            return None

        # Ensure baseline is fresh
        if self._baseline_mean is None:
            self._update_baseline_cache()

        baseline_mean = self._baseline_mean
        baseline_var = self._baseline_variance

        if baseline_mean is None or baseline_var is None or baseline_mean == 0:
            return None

        # Current (short window) stats
        short_arr = np.array(self._short_window)
        current_mean = float(np.mean(short_arr))
        current_var = float(np.var(short_arr))

        # --- CHECK 1: Mean drop ---
        mean_drop_pct = (baseline_mean - current_mean) / baseline_mean
        if mean_drop_pct > self.mean_drop_critical:
            return ConfidenceDriftAlert(
                alert_type="mean_drop",
                severity="critical",
                current_mean=current_mean,
                baseline_mean=baseline_mean,
                shift_magnitude=mean_drop_pct,
                message=(
                    f"CRITICAL: Confidence mean dropped {mean_drop_pct:.1%} "
                    f"from baseline {baseline_mean:.3f} to {current_mean:.3f}"
                ),
            )

        if mean_drop_pct > self.mean_drop_warning:
            return ConfidenceDriftAlert(
                alert_type="mean_drop",
                severity="warning",
                current_mean=current_mean,
                baseline_mean=baseline_mean,
                shift_magnitude=mean_drop_pct,
                message=(
                    f"WARNING: Confidence mean dropped {mean_drop_pct:.1%} "
                    f"from baseline {baseline_mean:.3f} to {current_mean:.3f}"
                ),
            )

        # --- CHECK 2: Variance spike ---
        if baseline_var > 0:
            variance_ratio = current_var / baseline_var
            if variance_ratio > self.variance_spike_factor:
                return ConfidenceDriftAlert(
                    alert_type="variance_spike",
                    severity="warning",
                    current_mean=current_mean,
                    baseline_mean=baseline_mean,
                    shift_magnitude=variance_ratio,
                    message=(
                        f"WARNING: Confidence variance spiked {variance_ratio:.1f}x "
                        f"(baseline={baseline_var:.4f}, current={current_var:.4f})"
                    ),
                )

        # --- CHECK 3: Distribution shift (KL divergence) ---
        kl_div = self._compute_kl_divergence(short_arr)
        if kl_div is not None and kl_div > self.kl_divergence_threshold:
            return ConfidenceDriftAlert(
                alert_type="distribution_shift",
                severity="warning",
                current_mean=current_mean,
                baseline_mean=baseline_mean,
                shift_magnitude=kl_div,
                message=(
                    f"WARNING: Confidence distribution shifted (KL={kl_div:.3f}). "
                    f"Shape change detected beyond threshold {self.kl_divergence_threshold}"
                ),
            )

        return None

    def get_stats(self) -> dict:
        """Get current monitoring statistics."""
        long_arr = np.array(self._long_window) if self._long_window else np.array([])
        short_arr = np.array(self._short_window) if self._short_window else np.array([])

        stats = {
            "long_window_size": len(self._long_window),
            "short_window_size": len(self._short_window),
            "baseline_mean": round(self._baseline_mean, 4) if self._baseline_mean else None,
            "baseline_variance": round(self._baseline_variance, 6) if self._baseline_variance else None,
        }

        if len(long_arr) > 0:
            stats["long_mean"] = round(float(np.mean(long_arr)), 4)
            stats["long_std"] = round(float(np.std(long_arr)), 4)

        if len(short_arr) > 0:
            stats["short_mean"] = round(float(np.mean(short_arr)), 4)
            stats["short_std"] = round(float(np.std(short_arr)), 4)

        return stats

    def _update_baseline_cache(self) -> None:
        """Recompute cached baseline statistics from long window."""
        if len(self._long_window) < 50:
            return

        long_arr = np.array(self._long_window)
        self._baseline_mean = float(np.mean(long_arr))
        self._baseline_variance = float(np.var(long_arr))

        # Compute histogram for KL divergence comparison
        self._baseline_histogram, _ = np.histogram(
            long_arr, bins=20, range=(0.0, 1.0), density=True
        )

        self._records_since_baseline_update = 0

    def _compute_kl_divergence(self, short_arr: np.ndarray) -> Optional[float]:
        """Compute KL divergence between short window and baseline distribution.

        Uses histogram-based approximation for computational efficiency.

        Returns:
            KL divergence value, or None if baseline histogram unavailable
        """
        if self._baseline_histogram is None:
            return None

        # Compute short window histogram with same bins
        short_hist, _ = np.histogram(short_arr, bins=20, range=(0.0, 1.0), density=True)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        p = self._baseline_histogram + eps
        q = short_hist + eps

        # Normalize to proper distributions
        p = p / p.sum()
        q = q / q.sum()

        # KL(P || Q) = sum(P * log(P/Q))
        kl = float(np.sum(p * np.log(p / q)))

        return max(0.0, kl)

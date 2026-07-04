"""Concept drift monitor using rolling prediction correctness windows."""

from __future__ import annotations

from collections import deque

from src.intelligence.models import DriftState


class DriftMonitor:
    def __init__(self, window: int = 200, min_samples: int = 50, alert_drop: float = 0.12):
        self.window = max(window, 20)
        self.min_samples = max(min_samples, 10)
        self.alert_drop = max(alert_drop, 0.01)
        self._history: deque[int] = deque(maxlen=self.window)

    def record_prediction(self, is_correct: bool) -> None:
        self._history.append(1 if is_correct else 0)

    def get_state(self) -> DriftState:
        n = len(self._history)
        if n == 0:
            return DriftState(
                sample_size=0,
                baseline_accuracy=0.0,
                recent_accuracy=0.0,
                accuracy_drop=0.0,
                degrading=False,
            )

        baseline = sum(self._history) / n
        cut = max(int(n * 0.5), 1)
        recent_slice = list(self._history)[cut:]
        recent = sum(recent_slice) / len(recent_slice)
        drop = baseline - recent
        degrading = n >= self.min_samples and drop >= self.alert_drop
        return DriftState(
            sample_size=n,
            baseline_accuracy=baseline,
            recent_accuracy=recent,
            accuracy_drop=drop,
            degrading=degrading,
        )


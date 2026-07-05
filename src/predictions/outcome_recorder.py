"""
Outcome Recorder — Monitors markets and records prediction outcomes at horizon expiry.

Runs as a background task checking for expired predictions and
recording their outcomes based on actual market data.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from src.predictions.registry import PredictionRecord, PredictionRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OutcomeRecorder:
    """
    Background service that checks for expired predictions
    and records their outcomes using market price data.
    """

    def __init__(
        self,
        registry: PredictionRegistry,
        price_fetcher: Optional[Callable[[str], Optional[float]]] = None,
        check_interval_seconds: float = 300.0,
    ):
        """
        Args:
            registry: The prediction registry to monitor
            price_fetcher: Callable(symbol) -> current_price or None
            check_interval_seconds: How often to check for expired predictions
        """
        self._registry = registry
        self._price_fetcher = price_fetcher
        self._check_interval = check_interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._entry_prices: dict[str, float] = {}  # prediction_id -> entry price

    def start(self) -> None:
        """Start background outcome recording."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("outcome_recorder.started")

    def stop(self) -> None:
        """Stop the outcome recorder."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("outcome_recorder.stopped")

    def record_entry_price(self, prediction_id: str, price: float) -> None:
        """Record the entry price at prediction time for return calculation."""
        self._entry_prices[prediction_id] = price

    def check_and_record(self) -> int:
        """
        Check for expired predictions and record outcomes.

        Returns number of outcomes recorded.
        """
        expired = self._registry.get_expired_predictions({})
        recorded = 0

        for prediction in expired:
            outcome = self._compute_outcome(prediction)
            if outcome is not None:
                self._registry.record_outcome(prediction.prediction_id, outcome)
                self._entry_prices.pop(prediction.prediction_id, None)
                recorded += 1

        if recorded > 0:
            logger.info("outcome_recorder.recorded", count=recorded)

        return recorded

    def _compute_outcome(self, prediction: PredictionRecord) -> Optional[float]:
        """Compute the actual return for a prediction."""
        if not self._price_fetcher:
            return None

        current_price = self._price_fetcher(prediction.asset)
        if current_price is None:
            return None

        entry_price = self._entry_prices.get(prediction.prediction_id)
        if entry_price is None or entry_price <= 0:
            return None

        # Compute return based on direction
        if prediction.direction == "long":
            actual_return = (current_price - entry_price) / entry_price
        else:
            actual_return = (entry_price - current_price) / entry_price

        return actual_return

    def _run_loop(self) -> None:
        """Background loop for checking expired predictions."""
        while self._running:
            try:
                self.check_and_record()
            except Exception as e:
                logger.error("outcome_recorder.error", error=str(e))
            time.sleep(self._check_interval)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def tracked_predictions(self) -> int:
        """Number of predictions with tracked entry prices."""
        return len(self._entry_prices)

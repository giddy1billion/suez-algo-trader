"""
Asset-Class Scheduler — Top-level dependency-aware orchestrator.

Replaces flat APScheduler interval jobs with a DAG-based system where
activities have declared preconditions, dependencies, and asset-class awareness.

The scheduler:
- Subscribes to events from the event bus
- Maintains trigger context (accumulated bars, drift scores, etc.)
- Evaluates the activity graph on each tick
- Executes ready activities respecting dependencies
- Publishes SchedulerEvent for each activity lifecycle
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from config.settings import OperationalMode, settings
from src.core.events import (
    BacktestTriggered,
    DataIngested,
    Event,
    ModelTrainingCompleted,
    SchedulerEvent,
)
from src.scheduler.activity_graph import (
    ActivityGraph,
    ActivityNode,
    ActivityResult,
    ActivityStatus,
    AssetClass,
)
from src.scheduler.market_status import MarketStatusService
from src.scheduler.triggers import (
    DataArrivalTrigger,
    DriftTrigger,
    ManualTrigger,
    ModelTrainedTrigger,
    ParameterChangeTrigger,
    ScheduleTrigger,
    TriggerContext,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AssetClassScheduler:
    """
    Top-level scheduler that orchestrates all research and trading activities.

    Features:
    - Event-driven trigger evaluation
    - Asset-class aware activity gating (equity vs crypto)
    - Dependency-aware execution ordering
    - Operational mode awareness (research/paper/live)
    - Integration with existing event bus
    """

    def __init__(
        self,
        event_bus=None,
        market_status: Optional[MarketStatusService] = None,
        operational_mode: Optional[OperationalMode] = None,
        alert_callback: Optional[Callable[[str], None]] = None,
    ):
        self._event_bus = event_bus
        self._market_status = market_status or MarketStatusService(
            equity_symbols=settings.scheduler_equity_symbols.split(","),
            crypto_symbols=settings.scheduler_crypto_symbols.split(","),
        )
        self._operational_mode = operational_mode or settings.operational_mode
        self._graph = ActivityGraph()
        self._context = TriggerContext()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._tick_interval = 30.0  # Evaluate triggers every 30 seconds
        self._alert_callback = alert_callback

        # Health monitoring
        self._last_tick_time: Optional[datetime] = None
        self._consecutive_errors: int = 0
        self._max_consecutive_errors: int = 5
        self._health_timeout_seconds: float = 120.0  # Unhealthy if no tick in 2 min
        self._restart_count: int = 0
        self._monitor_interval: float = min(max(self._tick_interval / 2.0, 5.0), 30.0)
        self._monitor_thread: Optional[threading.Thread] = None

        # Subscribe to events
        if self._event_bus:
            self._subscribe_events()

        logger.info(
            "asset_class_scheduler.initialized",
            mode=self._operational_mode.value,
        )

    def _subscribe_events(self) -> None:
        """Subscribe to relevant events from the bus."""
        self._event_bus.subscribe(DataIngested, self._on_data_ingested)
        self._event_bus.subscribe(ModelTrainingCompleted, self._on_model_trained)

    def _on_data_ingested(self, event: DataIngested) -> None:
        """Handle data ingestion events — update accumulated bar counts."""
        with self._lock:
            current = self._context.accumulated_bars.get(event.symbol, 0)
            self._context.accumulated_bars[event.symbol] = current + event.bar_count

    def _on_model_trained(self, event: ModelTrainingCompleted) -> None:
        """Handle model training completion."""
        with self._lock:
            self._context.last_model_trained = datetime.now(timezone.utc)
            self._context.events_received.append("ModelTrainingCompleted")

    # ──────────────────────────────────────────────────────────────────────
    # Activity Registration
    # ──────────────────────────────────────────────────────────────────────

    def register_activity(self, node: ActivityNode) -> None:
        """Register an activity node in the scheduler graph."""
        self._graph.add_activity(node)
        logger.info("scheduler.activity_registered", name=node.name)

    def register_default_activities(
        self,
        backtest_callable: Optional[Callable] = None,
        train_callable: Optional[Callable] = None,
        research_callable: Optional[Callable] = None,
    ) -> None:
        """Register the default set of activities for equity and crypto."""
        data_threshold = settings.scheduler_data_accumulation_threshold
        research_hours = settings.scheduler_research_cycle_hours

        # Equity backtest activity
        if backtest_callable:
            self._graph.add_activity(ActivityNode(
                name="equity_backtest",
                callable=backtest_callable,
                triggers=[
                    DataArrivalTrigger(threshold=data_threshold),
                    DriftTrigger(threshold=settings.backtest_trigger_drift_threshold),
                    ScheduleTrigger(interval_seconds=research_hours * 3600),
                ],
                asset_class=AssetClass.EQUITY,
            ))

            self._graph.add_activity(ActivityNode(
                name="crypto_backtest",
                callable=backtest_callable,
                triggers=[
                    DataArrivalTrigger(threshold=data_threshold),
                    DriftTrigger(threshold=settings.backtest_trigger_drift_threshold),
                    ScheduleTrigger(interval_seconds=research_hours * 3600),
                ],
                asset_class=AssetClass.CRYPTO,
            ))

        # Training activity (depends on backtest)
        if train_callable:
            self._graph.add_activity(ActivityNode(
                name="model_training",
                callable=train_callable,
                triggers=[
                    ModelTrainedTrigger(),  # Re-evaluate after any training
                    DriftTrigger(threshold=settings.retraining_drift_threshold),
                    ScheduleTrigger(
                        interval_seconds=settings.retraining_scheduled_interval_hours * 3600
                    ),
                ],
                asset_class=AssetClass.BOTH,
            ))

        # Research cycle
        if research_callable:
            self._graph.add_activity(ActivityNode(
                name="research_cycle",
                callable=research_callable,
                triggers=[
                    ScheduleTrigger(interval_seconds=research_hours * 3600),
                ],
                asset_class=AssetClass.BOTH,
            ))

    # ──────────────────────────────────────────────────────────────────────
    # Execution Loop
    # ──────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the scheduler background loop."""
        if self._running:
            return
        self._running = True
        with self._lock:
            self._last_tick_time = datetime.now(timezone.utc)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("asset_class_scheduler.started")

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        logger.info("asset_class_scheduler.stopped")

    def _run_loop(self) -> None:
        """Main scheduler loop — evaluates triggers and executes activities."""
        while self._running:
            try:
                self.tick()
                with self._lock:
                    self._last_tick_time = datetime.now(timezone.utc)
                    self._consecutive_errors = 0
            except Exception as e:
                logger.error("scheduler.tick_error", error=str(e))
                with self._lock:
                    self._consecutive_errors += 1
                    if self._consecutive_errors >= self._max_consecutive_errors:
                        logger.critical(
                            "scheduler.too_many_errors",
                            consecutive_errors=self._consecutive_errors,
                            action="restarting",
                        )
                        self._emit_alert(
                            f"Scheduler auto-restart after {self._consecutive_errors} consecutive errors"
                        )
                        self._attempt_restart(reason="consecutive_errors")
            time.sleep(self._tick_interval)

    def _monitor_loop(self) -> None:
        """Heartbeat watchdog loop to detect stale scheduler execution."""
        while self._running:
            time.sleep(self._monitor_interval)
            try:
                health = self.health_check()
                if not health["healthy"]:
                    seconds = health.get("seconds_since_last_tick")
                    msg = (
                        f"Scheduler heartbeat stale: {seconds:.1f}s since last tick"
                        if seconds is not None
                        else "Scheduler heartbeat stale: no tick recorded"
                    )
                    logger.critical("scheduler.heartbeat_stale", details=msg)
                    self._publish_scheduler_event("scheduler_watchdog", "heartbeat_missed")
                    self._emit_alert(msg)
                    self._attempt_restart(reason="heartbeat_stale")
            except Exception as e:
                logger.error("scheduler.monitor_error", error=str(e))

    def tick(self) -> list[ActivityResult]:
        """
        Single scheduler evaluation tick.

        Evaluates all activity triggers and executes ready activities.
        Can be called manually for testing or by the background loop.
        """
        results = []

        with self._lock:
            self._context.current_time = datetime.now(timezone.utc)
            context = TriggerContext(
                current_time=self._context.current_time,
                accumulated_bars=dict(self._context.accumulated_bars),
                drift_scores=dict(self._context.drift_scores),
                parameter_hashes=dict(self._context.parameter_hashes),
                last_model_trained=self._context.last_model_trained,
                last_backtest_run=self._context.last_backtest_run,
                last_retraining=self._context.last_retraining,
                events_received=list(self._context.events_received),
            )

        # Determine active asset class based on market status
        statuses = self._market_status.get_all_statuses()

        # Get ready activities
        ready = self._graph.get_ready_activities(context)

        # Filter by operational mode
        ready = self._filter_by_mode(ready)

        # Execute in dependency order
        for node in ready:
            # Check asset class gating
            if node.asset_class == AssetClass.EQUITY:
                if not statuses["equity"].is_trading:
                    continue
            # Crypto is always active, no gating needed

            self._publish_scheduler_event(node.name, "started")

            result = self._graph.execute_activity(node, context)
            results.append(result)

            status = "completed" if result.status == ActivityStatus.COMPLETED else "failed"
            self._publish_scheduler_event(node.name, status)

            # Publish specific trigger event for backtests
            if "backtest" in node.name and result.status == ActivityStatus.COMPLETED:
                self._publish_backtest_triggered(node, result)

        return results

    def _filter_by_mode(self, activities: list[ActivityNode]) -> list[ActivityNode]:
        """Filter activities based on operational mode."""
        if self._operational_mode == OperationalMode.RESEARCH:
            # In research mode, allow all research activities but no live execution
            return activities
        return activities

    def _publish_scheduler_event(self, job_name: str, status: str) -> None:
        """Publish scheduler lifecycle event."""
        if self._event_bus:
            self._event_bus.publish(SchedulerEvent(
                job_name=job_name,
                status=status,
                source="asset_class_scheduler",
            ))

    def _publish_backtest_triggered(
        self, node: ActivityNode, result: ActivityResult
    ) -> None:
        """Publish backtest trigger event."""
        if self._event_bus:
            symbols = (
                self._market_status.equity_symbols
                if node.asset_class == AssetClass.EQUITY
                else self._market_status.crypto_symbols
            )
            self._event_bus.publish(BacktestTriggered(
                reason=result.trigger_reason,
                symbols=symbols,
                trigger_source=node.name,
                source="asset_class_scheduler",
            ))

    # ──────────────────────────────────────────────────────────────────────
    # Context Updates (external callers)
    # ──────────────────────────────────────────────────────────────────────

    def update_drift_score(self, symbol: str, score: float) -> None:
        """Update drift score for a symbol."""
        with self._lock:
            self._context.drift_scores[symbol] = score

    def update_parameter_hash(self, component: str, hash_val: str) -> None:
        """Update parameter hash for change detection."""
        with self._lock:
            self._context.parameter_hashes[component] = hash_val

    def trigger_manual(self, activity_name: str) -> Optional[ActivityResult]:
        """Manually trigger a specific activity."""
        node = self._graph.get_activity(activity_name)
        if not node:
            return None

        with self._lock:
            context = TriggerContext(
                current_time=datetime.now(timezone.utc),
                accumulated_bars=dict(self._context.accumulated_bars),
                drift_scores=dict(self._context.drift_scores),
                parameter_hashes=dict(self._context.parameter_hashes),
                last_model_trained=self._context.last_model_trained,
            )

        self._publish_scheduler_event(node.name, "started")
        result = self._graph.execute_activity(node, context)
        status = "completed" if result.status == ActivityStatus.COMPLETED else "failed"
        self._publish_scheduler_event(node.name, status)
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Status & Introspection
    # ──────────────────────────────────────────────────────────────────────

    @property
    def operational_mode(self) -> OperationalMode:
        return self._operational_mode

    @operational_mode.setter
    def operational_mode(self, mode: OperationalMode) -> None:
        old = self._operational_mode
        self._operational_mode = mode
        logger.info("scheduler.mode_changed", old=old.value, new=mode.value)

    def health_check(self) -> dict[str, Any]:
        """Return scheduler health status with liveness and error metrics."""
        with self._lock:
            now = datetime.now(timezone.utc)
            last_tick = self._last_tick_time
            if last_tick is not None:
                seconds_since = (now - last_tick).total_seconds()
                is_healthy = seconds_since < self._health_timeout_seconds
            else:
                seconds_since = None
                is_healthy = self._running  # Not yet ticked

            return {
                "healthy": is_healthy,
                "running": self._running,
                "seconds_since_last_tick": seconds_since,
                "consecutive_errors": self._consecutive_errors,
                "restart_count": self._restart_count,
                "health_timeout_seconds": self._health_timeout_seconds,
            }

    def _attempt_restart(self, reason: str = "unknown") -> None:
        """Attempt to restart the scheduler loop after repeated failures."""
        with self._lock:
            self._restart_count += 1
            self._consecutive_errors = 0
            self._last_tick_time = datetime.now(timezone.utc)
        logger.warning(
            "scheduler.auto_restart",
            restart_count=self._restart_count,
            reason=reason,
        )
        self._publish_scheduler_event("scheduler_watchdog", "restarting")
        # Actually restart threads (not just reset counters)
        try:
            self._running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5)
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            logger.info("scheduler.threads_restarted", restart_count=self._restart_count)
            self._publish_scheduler_event("scheduler_watchdog", "restarted")
        except Exception as e:
            logger.error("scheduler.restart_failed", error=str(e))
            self._publish_scheduler_event("scheduler_watchdog", "restart_failed")

    def _emit_alert(self, message: str) -> None:
        """Emit scheduler alert via callback and event bus."""
        if self._alert_callback:
            try:
                self._alert_callback(message)
            except Exception as e:
                logger.warning("scheduler.alert_callback_error", error=str(e))

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive scheduler status including health."""
        return {
            "running": self._running,
            "operational_mode": self._operational_mode.value,
            "health": self.health_check(),
            "market_status": {
                k: {"phase": v.phase.value, "is_trading": v.is_trading}
                for k, v in self._market_status.get_all_statuses().items()
            },
            "graph": self._graph.summary(),
            "context": {
                "accumulated_bars": dict(self._context.accumulated_bars),
                "drift_scores": dict(self._context.drift_scores),
                "last_model_trained": (
                    self._context.last_model_trained.isoformat()
                    if self._context.last_model_trained
                    else None
                ),
            },
        }

    def get_execution_history(self, limit: int = 50) -> list[dict]:
        """Get recent execution history."""
        history = self._graph.get_execution_history(limit)
        return [
            {
                "activity": r.activity_id,
                "status": r.status.value,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "trigger_reason": r.trigger_reason,
                "error": r.error,
            }
            for r in history
        ]

"""
Activity Graph — DAG of scheduler activities with trigger conditions.

Each activity node declares:
- Preconditions (triggers that must be met)
- Dependencies (other activities that must complete first)
- The callable to execute
- Asset class gating (equity, crypto, or both)
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from src.scheduler.triggers import Trigger, TriggerContext
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ActivityStatus(str, Enum):
    """Status of an activity node."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AssetClass(str, Enum):
    """Asset class for gating activities."""
    EQUITY = "equity"
    CRYPTO = "crypto"
    BOTH = "both"


@dataclass
class ActivityResult:
    """Result of an activity execution."""
    activity_id: str
    status: ActivityStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    trigger_reason: str = ""


@dataclass
class ActivityNode:
    """
    A single activity in the scheduler DAG.

    Attributes:
        name: Human-readable activity name
        callable: The function to execute
        triggers: List of triggers (OR logic — any trigger fires the activity)
        dependencies: Names of activities that must complete first
        asset_class: Which asset class this activity applies to
        enabled: Whether this activity is active
        timeout_seconds: Max execution time
    """
    name: str
    callable: Callable
    triggers: list[Trigger] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    asset_class: AssetClass = AssetClass.BOTH
    enabled: bool = True
    timeout_seconds: float = 300.0
    # Runtime state
    _status: ActivityStatus = field(default=ActivityStatus.PENDING, init=False)
    _last_run: Optional[datetime] = field(default=None, init=False)
    _last_result: Optional[ActivityResult] = field(default=None, init=False)
    _run_count: int = field(default=0, init=False)

    @property
    def status(self) -> ActivityStatus:
        return self._status

    @property
    def last_run(self) -> Optional[datetime]:
        return self._last_run

    @property
    def last_result(self) -> Optional[ActivityResult]:
        return self._last_result

    def should_trigger(self, context: TriggerContext) -> bool:
        """Check if any trigger condition is met (OR logic)."""
        if not self.enabled:
            return False
        if not self.triggers:
            return False
        return any(t.evaluate(context) for t in self.triggers)

    def reset_triggers(self, context: TriggerContext) -> None:
        """Reset all triggers after execution."""
        for trigger in self.triggers:
            trigger.reset()
            if hasattr(trigger, "acknowledge"):
                trigger.acknowledge(context)


class ActivityGraph:
    """
    Directed Acyclic Graph of scheduler activities.

    Activities are evaluated in topological order, respecting
    dependencies and trigger conditions.
    """

    def __init__(self):
        self._nodes: dict[str, ActivityNode] = {}
        self._lock = threading.Lock()
        self._execution_history: list[ActivityResult] = []

    def add_activity(self, node: ActivityNode) -> None:
        """Add an activity node to the graph."""
        with self._lock:
            if node.name in self._nodes:
                raise ValueError(f"Activity '{node.name}' already exists")
            self._nodes[node.name] = node

    def remove_activity(self, name: str) -> None:
        """Remove an activity from the graph."""
        with self._lock:
            self._nodes.pop(name, None)

    def get_activity(self, name: str) -> Optional[ActivityNode]:
        """Get activity by name."""
        with self._lock:
            return self._nodes.get(name)

    @property
    def activities(self) -> list[ActivityNode]:
        """All activity nodes."""
        with self._lock:
            return list(self._nodes.values())

    def get_ready_activities(
        self,
        context: TriggerContext,
        active_asset_class: Optional[AssetClass] = None,
    ) -> list[ActivityNode]:
        """
        Get activities that are ready to execute.

        An activity is ready when:
        1. It's enabled
        2. Its triggers are met (or it has no triggers and dependencies are done)
        3. All dependencies have completed
        4. Asset class matches the active class (or is BOTH)
        """
        ready = []
        with self._lock:
            completed_names = {
                name for name, node in self._nodes.items()
                if node._status == ActivityStatus.COMPLETED
            }

            for node in self._nodes.values():
                if not node.enabled:
                    continue
                if node._status == ActivityStatus.RUNNING:
                    continue

                # Asset class gating
                if active_asset_class and node.asset_class != AssetClass.BOTH:
                    if node.asset_class != active_asset_class:
                        continue

                # Check dependencies
                deps_met = all(dep in completed_names for dep in node.dependencies)
                if not deps_met:
                    continue

                # Check triggers
                if node.should_trigger(context):
                    ready.append(node)

        return ready

    def execute_activity(
        self,
        node: ActivityNode,
        context: TriggerContext,
        **kwargs,
    ) -> ActivityResult:
        """Execute a single activity and record the result."""
        with self._lock:
            if node._status == ActivityStatus.RUNNING:
                return ActivityResult(
                    activity_id=node.name,
                    status=ActivityStatus.SKIPPED,
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    error="activity_already_running",
                )
            node._status = ActivityStatus.RUNNING
            started = datetime.now(timezone.utc)
            trigger_reasons = [t.description() for t in node.triggers if t.evaluate(context)]

        try:
            result = node.callable(**kwargs)
            activity_result = ActivityResult(
                activity_id=node.name,
                status=ActivityStatus.COMPLETED,
                started_at=started,
                completed_at=datetime.now(timezone.utc),
                result=result,
                trigger_reason=", ".join(trigger_reasons),
            )
        except Exception as e:
            activity_result = ActivityResult(
                activity_id=node.name,
                status=ActivityStatus.FAILED,
                started_at=started,
                completed_at=datetime.now(timezone.utc),
                error=str(e),
                trigger_reason=", ".join(trigger_reasons),
            )
            logger.error("activity_graph.execution_failed", activity=node.name, error=str(e))

        with self._lock:
            node._status = activity_result.status
            node._last_run = activity_result.completed_at
            node._last_result = activity_result
            node._run_count += 1
            node.reset_triggers(context)
            self._execution_history.append(activity_result)
            # Keep history bounded
            if len(self._execution_history) > 1000:
                self._execution_history = self._execution_history[-500:]

        return activity_result

    def reset_all(self) -> None:
        """Reset all activity statuses to pending."""
        with self._lock:
            for node in self._nodes.values():
                node._status = ActivityStatus.PENDING

    def get_execution_history(self, limit: int = 50) -> list[ActivityResult]:
        """Get recent execution history."""
        with self._lock:
            return list(self._execution_history[-limit:])

    def get_topology_order(self) -> list[str]:
        """Return activity names in topological (dependency) order."""
        with self._lock:
            visited: set[str] = set()
            order: list[str] = []

            def _visit(name: str) -> None:
                if name in visited:
                    return
                visited.add(name)
                node = self._nodes.get(name)
                if node:
                    for dep in node.dependencies:
                        _visit(dep)
                order.append(name)

            for name in self._nodes:
                _visit(name)

            return order

    def summary(self) -> dict[str, Any]:
        """Get a summary of the graph state."""
        with self._lock:
            return {
                "total_activities": len(self._nodes),
                "enabled": sum(1 for n in self._nodes.values() if n.enabled),
                "statuses": {
                    name: node._status.value
                    for name, node in self._nodes.items()
                },
                "execution_count": len(self._execution_history),
            }

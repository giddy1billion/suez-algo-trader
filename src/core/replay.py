"""
Deterministic Replay Engine — Reconstruct historical trading sessions.

Replays persisted events through the full pipeline to reproduce exact
system behavior for debugging, verification, and analysis.

Replay guarantees:
- Events are replayed in exact chronological order (by DB sequence ID)
- Subscribers receive events identically to the live session
- State machines are reconstructed step by step
- Output metrics can be compared against live session for consistency
"""

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional

from src.core.event_store import EventStore, _reconstruct_event
from src.core.events import (
    Event,
    EventBus,
    OrderFilled,
    OrderRejected,
    OrderSubmitted,
    RiskEvaluated,
    SignalGenerated,
    TradeClosed,
    TradeOpened,
)
from src.core.state_machine import TradeLifecycle, TradeManager, TradeState
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Replay Report
# ---------------------------------------------------------------------------


@dataclass
class ReplayReport:
    """Summary of a replay session."""

    session_id: str
    events_replayed: int = 0
    signals_count: int = 0
    trades_opened: int = 0
    trades_closed: int = 0
    orders_rejected: int = 0
    final_pnl: float = 0.0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    event_timeline: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Replay Engine
# ---------------------------------------------------------------------------


class ReplayEngine:
    """
    Deterministic replay of historical trading sessions.

    Reconstructs a session by replaying persisted events through a fresh
    EventBus and TradeManager, allowing subscribers to observe the same
    event sequence as the live session.

    Usage:
        store = EventStore(db_path="data_cache/events.db")
        engine = ReplayEngine(store)
        report = engine.replay("session_abc123")
        print(report.trades_opened, report.final_pnl)

    Advanced usage with custom subscribers:
        engine = ReplayEngine(store)
        engine.add_observer(my_analysis_handler)
        report = engine.replay("session_abc123")
    """

    def __init__(self, event_store: EventStore):
        self._store = event_store
        self._observers: List[Callable[[Event], None]] = []
        self._lock = threading.Lock()

    def add_observer(self, handler: Callable[[Event], None]) -> None:
        """Add a custom observer that receives each replayed event."""
        self._observers.append(handler)

    def remove_observer(self, handler: Callable[[Event], None]) -> None:
        """Remove a previously added observer."""
        self._observers = [h for h in self._observers if h is not handler]

    def list_sessions(self, limit: int = 20) -> List[dict]:
        """List available sessions with event counts and time ranges."""
        with self._lock:
            conn = self._store._conn
            cursor = conn.execute("""
                SELECT session_id, 
                       COUNT(*) as event_count,
                       MIN(timestamp) as first_event,
                       MAX(timestamp) as last_event
                FROM events 
                GROUP BY session_id 
                ORDER BY MAX(id) DESC 
                LIMIT ?
            """, (limit,))
            return [
                {
                    "session_id": row[0],
                    "event_count": row[1],
                    "first_event": row[2],
                    "last_event": row[3],
                }
                for row in cursor.fetchall()
            ]

    def replay(
        self,
        session_id: str,
        event_bus: Optional[EventBus] = None,
        trade_manager: Optional[TradeManager] = None,
        stop_after: Optional[int] = None,
        event_filter: Optional[Callable[[Event], bool]] = None,
        mode: str = "full",
    ) -> ReplayReport:
        """
        Replay a historical session event by event.

        Args:
            session_id: The session to replay.
            event_bus: Optional EventBus to publish events into (creates fresh one if None).
            trade_manager: Optional TradeManager for lifecycle tracking.
            stop_after: Stop after N events (for partial replays / debugging).
            event_filter: Optional filter — only replay events where filter returns True.
            mode: "full" | "strategy" | "execution"
                - full: replay all events (default)
                - strategy: only SignalGenerated and RiskEvaluated
                - execution: only OrderSubmitted, OrderFilled, OrderRejected, TradeOpened, TradeClosed

        Returns:
            ReplayReport with summary statistics.
        """
        start_time = datetime.now(timezone.utc)
        report = ReplayReport(session_id=session_id)

        # Build mode filter
        if mode == "strategy":
            _mode_filter = lambda e: isinstance(e, (SignalGenerated, RiskEvaluated))
        elif mode == "execution":
            _mode_filter = lambda e: isinstance(e, (OrderSubmitted, OrderFilled, OrderRejected, TradeOpened, TradeClosed))
        else:
            _mode_filter = None

        # Combine mode filter with user-provided filter
        def _combined_filter(e):
            if _mode_filter and not _mode_filter(e):
                return False
            if event_filter and not event_filter(e):
                return False
            return True

        # Create fresh infrastructure for replay
        if event_bus is None:
            event_bus = EventBus(max_history=10000)
        if trade_manager is None:
            trade_manager = TradeManager()

        # Subscribe internal tracker
        tracker = _ReplayTracker(trade_manager, report)
        event_bus.subscribe(None, tracker.handle_event)

        # Register custom observers on the bus
        for obs in self._observers:
            event_bus.subscribe(None, obs)

        # Load events from store
        raw_events = self._store.get_session_events(session_id)
        if not raw_events:
            report.errors.append(f"No events found for session {session_id}")
            logger.warning("replay.no_events", session_id=session_id)
            return report

        logger.info(
            "replay.start",
            session_id=session_id,
            total_events=len(raw_events),
        )

        # Replay events in order
        for i, raw in enumerate(raw_events):
            if stop_after is not None and i >= stop_after:
                break

            try:
                # Parse payload — it comes as JSON string from get_session_events
                payload = raw["payload"]
                if isinstance(payload, str):
                    payload = json.loads(payload)

                event = _reconstruct_event(raw["event_type"], payload)

                if not _combined_filter(event):
                    continue

                # Publish into the replay bus (triggers all subscribers)
                event_bus.publish(event)
                report.events_replayed += 1

                # Record timeline entry
                report.event_timeline.append({
                    "seq": i,
                    "type": raw["event_type"],
                    "timestamp": raw["timestamp"],
                    "event_id": raw.get("event_id", ""),
                })

            except Exception as e:
                error_msg = f"Event {i} ({raw.get('event_type', '?')}): {e}"
                report.errors.append(error_msg)
                logger.warning("replay.event_error", error=str(e), seq=i)

        # Finalize report
        end_time = datetime.now(timezone.utc)
        report.duration_seconds = (end_time - start_time).total_seconds()

        logger.info(
            "replay.complete",
            session_id=session_id,
            events=report.events_replayed,
            trades_opened=report.trades_opened,
            trades_closed=report.trades_closed,
            pnl=report.final_pnl,
            errors=len(report.errors),
        )

        return report

    def compare_sessions(self, session_a: str, session_b: str) -> dict:
        """
        Compare two sessions for consistency.

        Useful for verifying that replay produces identical results to live.
        """
        report_a = self.replay(session_a)
        report_b = self.replay(session_b)

        return {
            "session_a": session_a,
            "session_b": session_b,
            "events_match": report_a.events_replayed == report_b.events_replayed,
            "signals_match": report_a.signals_count == report_b.signals_count,
            "trades_opened_match": report_a.trades_opened == report_b.trades_opened,
            "trades_closed_match": report_a.trades_closed == report_b.trades_closed,
            "pnl_match": abs(report_a.final_pnl - report_b.final_pnl) < 0.01,
            "a_events": report_a.events_replayed,
            "b_events": report_b.events_replayed,
            "a_pnl": report_a.final_pnl,
            "b_pnl": report_b.final_pnl,
        }

    def get_session_summary(self, session_id: str) -> dict:
        """Get a quick summary without full replay."""
        raw_events = self._store.get_session_events(session_id)
        if not raw_events:
            return {"session_id": session_id, "found": False}

        type_counts = {}
        for raw in raw_events:
            t = raw.get("event_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "session_id": session_id,
            "found": True,
            "total_events": len(raw_events),
            "event_types": type_counts,
            "first_event": raw_events[0].get("timestamp"),
            "last_event": raw_events[-1].get("timestamp"),
        }


# ---------------------------------------------------------------------------
# Internal Replay Tracker
# ---------------------------------------------------------------------------


class _ReplayTracker:
    """Tracks replay metrics by observing events."""

    def __init__(self, trade_manager: TradeManager, report: ReplayReport):
        self._tm = trade_manager
        self._report = report
        self._trade_pnls: dict[str, float] = {}

    def handle_event(self, event: Event) -> None:
        """Handle a replayed event and update metrics."""
        if isinstance(event, SignalGenerated):
            self._report.signals_count += 1

        elif isinstance(event, TradeOpened):
            self._report.trades_opened += 1
            # Create lifecycle in trade manager
            lifecycle = self._tm.create_trade(
                symbol=event.symbol, side=event.side, trade_id=event.trade_id
            )
            if lifecycle:
                # Fast-forward to ACTIVE state
                lifecycle.transition(TradeState.PENDING_RISK, "replay")
                lifecycle.transition(TradeState.RISK_APPROVED, "replay")
                lifecycle.transition(TradeState.SUBMITTED, "replay")
                lifecycle.transition(TradeState.ACCEPTED, "replay")
                lifecycle.transition(TradeState.FILLED, "replay")
                lifecycle.transition(TradeState.ACTIVE, "replay")

        elif isinstance(event, TradeClosed):
            self._report.trades_closed += 1
            self._trade_pnls[event.trade_id] = event.pnl
            self._report.final_pnl = sum(self._trade_pnls.values())
            # Transition lifecycle to closed
            lifecycle = self._tm.get_trade(event.trade_id)
            if lifecycle and not lifecycle.is_terminal:
                lifecycle.transition(TradeState.CLOSING, "replay")
                lifecycle.transition(TradeState.CLOSED, "replay")

        elif isinstance(event, OrderRejected):
            self._report.orders_rejected += 1

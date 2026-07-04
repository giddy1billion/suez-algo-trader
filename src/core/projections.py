"""
CQRS Read Model Projections — Incrementally updated from events.

Maintains fast-access views of:
- Open positions with current P&L
- Portfolio exposure and risk metrics
- Aggregated trade history
- Real-time performance metrics
- System state summary

Each projection subscribes to relevant events and updates itself
incrementally, avoiding expensive replay operations for dashboards.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.core.events import (
    Event, EventBus, SignalGenerated, RiskEvaluated,
    OrderSubmitted, OrderFilled, OrderRejected,
    TradeOpened, TradeClosed, RiskHalt, SystemHealth,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PositionView:
    """Read model for a single open position."""
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    qty: float
    stop_loss: float
    take_profit: float
    opened_at: str
    unrealized_pnl: float = 0.0


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state."""
    timestamp: datetime
    positions: List[PositionView]
    total_equity: float = 0.0
    total_exposure: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0


class PositionProjection:
    """Maintains current open positions, updated from TradeOpened/TradeClosed events."""

    def __init__(self):
        self._positions: Dict[str, PositionView] = {}
        self._lock = threading.Lock()

    def handle_event(self, event: Event):
        if isinstance(event, TradeOpened):
            with self._lock:
                self._positions[event.trade_id] = PositionView(
                    trade_id=event.trade_id,
                    symbol=event.symbol,
                    side=event.side,
                    entry_price=event.entry_price,
                    qty=event.qty,
                    stop_loss=event.stop_loss,
                    take_profit=event.take_profit,
                    opened_at=event.timestamp.isoformat() if isinstance(event.timestamp, datetime) else str(event.timestamp),
                )
        elif isinstance(event, TradeClosed):
            with self._lock:
                self._positions.pop(event.trade_id, None)

    def get_positions(self) -> List[PositionView]:
        with self._lock:
            return list(self._positions.values())

    def get_position(self, trade_id: str) -> Optional[PositionView]:
        with self._lock:
            return self._positions.get(trade_id)

    def count(self) -> int:
        with self._lock:
            return len(self._positions)

    def get_exposure(self) -> float:
        """Total notional exposure."""
        with self._lock:
            return sum(p.entry_price * p.qty for p in self._positions.values())


class PerformanceProjection:
    """Maintains aggregated performance metrics from trade events."""

    def __init__(self):
        self._lock = threading.Lock()
        self._realized_pnl: float = 0.0
        self._trade_count: int = 0
        self._win_count: int = 0
        self._loss_count: int = 0
        self._pnl_history: List[float] = []
        self._signal_count: int = 0
        self._rejection_count: int = 0
        self._risk_halt_count: int = 0

    def handle_event(self, event: Event):
        if isinstance(event, TradeClosed):
            with self._lock:
                self._trade_count += 1
                self._realized_pnl += event.pnl
                self._pnl_history.append(event.pnl)
                if event.pnl > 0:
                    self._win_count += 1
                elif event.pnl < 0:
                    self._loss_count += 1
        elif isinstance(event, SignalGenerated):
            with self._lock:
                self._signal_count += 1
        elif isinstance(event, OrderRejected):
            with self._lock:
                self._rejection_count += 1
        elif isinstance(event, RiskHalt):
            with self._lock:
                self._risk_halt_count += 1

    def get_metrics(self) -> dict:
        with self._lock:
            win_rate = self._win_count / self._trade_count if self._trade_count > 0 else 0.0
            avg_pnl = self._realized_pnl / self._trade_count if self._trade_count > 0 else 0.0
            return {
                "realized_pnl": self._realized_pnl,
                "trade_count": self._trade_count,
                "win_count": self._win_count,
                "loss_count": self._loss_count,
                "win_rate": win_rate,
                "avg_pnl": avg_pnl,
                "signal_count": self._signal_count,
                "rejection_count": self._rejection_count,
                "risk_halt_count": self._risk_halt_count,
                "profit_factor": self._compute_profit_factor(),
            }

    def _compute_profit_factor(self) -> float:
        """Gross profit / gross loss."""
        gross_profit = sum(p for p in self._pnl_history if p > 0)
        gross_loss = abs(sum(p for p in self._pnl_history if p < 0))
        if gross_loss == 0:
            return 999.99 if gross_profit > 0 else 0.0
        return min(gross_profit / gross_loss, 999.99)


class ActivityProjection:
    """Tracks recent activity: signals, orders, events for dashboards."""

    def __init__(self, max_recent: int = 100):
        self._lock = threading.Lock()
        self._recent_events: List[dict] = []
        self._max_recent = max_recent

    def handle_event(self, event: Event):
        entry = {
            "type": type(event).__name__,
            "timestamp": event.timestamp.isoformat() if isinstance(event.timestamp, datetime) else str(event.timestamp),
            "source": event.source,
            "event_id": event.event_id,
        }
        # Add key fields per type
        if isinstance(event, SignalGenerated):
            entry["symbol"] = event.symbol
            entry["signal"] = event.signal
            entry["confidence"] = event.confidence
        elif isinstance(event, TradeOpened):
            entry["trade_id"] = event.trade_id
            entry["symbol"] = event.symbol
            entry["side"] = event.side
        elif isinstance(event, TradeClosed):
            entry["trade_id"] = event.trade_id
            entry["symbol"] = event.symbol
            entry["pnl"] = event.pnl
        elif isinstance(event, OrderRejected):
            entry["reason"] = event.reason

        with self._lock:
            self._recent_events.append(entry)
            if len(self._recent_events) > self._max_recent:
                self._recent_events = self._recent_events[-self._max_recent:]

    def get_recent(self, n: int = 20) -> List[dict]:
        with self._lock:
            return self._recent_events[-n:]

    def count(self) -> int:
        with self._lock:
            return len(self._recent_events)


class ReadModelManager:
    """
    Manages all CQRS read model projections.

    Subscribes to EventBus (wildcard) and dispatches to all projections.
    Provides a unified query interface for dashboards.
    """

    def __init__(self):
        self.positions = PositionProjection()
        self.performance = PerformanceProjection()
        self.activity = ActivityProjection()
        self._projections = [self.positions, self.performance, self.activity]

    def attach(self, event_bus: EventBus) -> None:
        """Subscribe to all events on the bus."""
        event_bus.subscribe(None, self._dispatch)
        logger.info("ReadModelManager attached to EventBus")

    def _dispatch(self, event: Event) -> None:
        """Dispatch event to all projections with isolation.

        Each projection is applied independently. If one projection fails,
        only that projection is affected — others still receive the event.
        Failed projections log the error and continue (no state corruption
        because each projection holds its own lock).
        """
        for proj in self._projections:
            try:
                proj.handle_event(event)
            except Exception:
                logger.exception(
                    "Projection error in %s for event %s (event_id=%s). "
                    "Projection may be out-of-sync — monitor for drift.",
                    type(proj).__name__,
                    type(event).__name__,
                    getattr(event, "event_id", "?"),
                )

    def get_dashboard(self) -> dict:
        """Get full dashboard data in a single call."""
        return {
            "positions": [vars(p) for p in self.positions.get_positions()],
            "position_count": self.positions.count(),
            "exposure": self.positions.get_exposure(),
            "performance": self.performance.get_metrics(),
            "recent_activity": self.activity.get_recent(10),
        }

    def get_snapshot(self) -> PortfolioSnapshot:
        """Get a point-in-time snapshot for persistence."""
        metrics = self.performance.get_metrics()
        positions = self.positions.get_positions()
        unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        total_exposure = self.positions.get_exposure()
        return PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            positions=positions,
            total_equity=total_exposure + unrealized_pnl,
            total_exposure=total_exposure,
            realized_pnl=metrics["realized_pnl"],
            unrealized_pnl=unrealized_pnl,
            trade_count=metrics["trade_count"],
            win_count=metrics["win_count"],
            loss_count=metrics["loss_count"],
        )

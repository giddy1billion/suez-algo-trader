"""
Crash Recovery Engine.

Recovers system state after an unexpected shutdown by reloading
open positions from the broker, reconstructing trade lifecycles,
and reconciling internal state.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.utils.logger import get_logger
from src.core.events import SystemHealth, EventBus
from src.core.state_machine import TradeLifecycle, TradeManager, TradeState

logger = get_logger(__name__)


@dataclass
class RecoveryReport:
    """Summary of a crash recovery operation."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    positions_recovered: int = 0
    orphans_detected: int = 0
    events_replayed: int = 0
    warnings: list[str] = field(default_factory=list)
    success: bool = False


class RecoveryManager:
    """
    Crash recovery engine.

    On startup, queries the broker for open positions and recent orders,
    reconstructs TradeLifecycle objects, registers them with TradeManager,
    and publishes a SystemHealth event indicating recovery status.
    """

    def __init__(
        self,
        broker,
        event_bus: EventBus,
        trade_manager: TradeManager,
        event_store=None,
    ):
        self.broker = broker
        self.event_bus = event_bus
        self.trade_manager = trade_manager
        self.event_store = event_store

    def recover(self) -> RecoveryReport:
        """Full recovery sequence on startup."""
        report = RecoveryReport()
        logger.info("Starting crash recovery...")

        try:
            # 1. Query broker for open positions
            positions = self._get_broker_positions()

            # 2. Query broker for recent orders (last 24h)
            recent_orders = self._get_recent_orders()

            # 3. Load persisted events from event_store (if available)
            events_replayed = self._replay_events()
            report.events_replayed = events_replayed

            # 4. Reconstruct TradeLifecycle objects for open positions
            lifecycles = self._reconstruct_lifecycles(positions)
            report.positions_recovered = len(lifecycles)

            # 5. Register them with TradeManager
            for lc in lifecycles:
                self._register_lifecycle(lc)

            # 6. Detect orphans
            internal_trades = self.trade_manager.get_active_trades()
            orphans = self._detect_orphans(positions, internal_trades)
            report.orphans_detected = len(orphans)
            if orphans:
                report.warnings.append(
                    f"Found {len(orphans)} orphan position(s): "
                    f"{[o.get('symbol', 'unknown') for o in orphans]}"
                )

            report.success = True
            logger.info(
                "Recovery complete: %d positions recovered, %d orphans detected",
                report.positions_recovered,
                report.orphans_detected,
            )

        except Exception as e:
            report.success = False
            report.warnings.append(f"Recovery failed: {str(e)}")
            logger.error("Recovery failed: %s", str(e), exc_info=True)

        # 7. Publish SystemHealth event indicating recovery complete
        self.event_bus.publish(
            SystemHealth(
                component="recovery_manager",
                status="healthy" if report.success else "degraded",
                metrics={
                    "positions_recovered": report.positions_recovered,
                    "orphans_detected": report.orphans_detected,
                    "events_replayed": report.events_replayed,
                    "success": report.success,
                },
                source="RecoveryManager",
            )
        )

        return report

    def _get_broker_positions(self) -> list:
        """Get all open positions from broker."""
        try:
            positions = self.broker.get_positions()
            logger.info("Retrieved %d positions from broker", len(positions))
            return positions
        except Exception as e:
            logger.error("Failed to get broker positions: %s", str(e))
            raise

    def _get_recent_orders(self) -> list:
        """Get recent orders from broker."""
        try:
            orders = self.broker.get_orders(status="open")
            logger.info("Retrieved %d recent orders from broker", len(orders))
            return orders
        except Exception as e:
            logger.warning("Failed to get recent orders: %s", str(e))
            return []

    def _replay_events(self) -> int:
        """Replay persisted events from event store if available."""
        if self.event_store is None:
            return 0

        try:
            events = self.event_store.get_recent_events()
            count = 0
            for event in events:
                self.event_bus.publish(event)
                count += 1
            logger.info("Replayed %d events from event store", count)
            return count
        except Exception as e:
            logger.warning("Failed to replay events: %s", str(e))
            return 0

    def _reconstruct_lifecycles(self, positions: list) -> list:
        """Create TradeLifecycle objects for each broker position."""
        lifecycles = []

        for pos in positions:
            symbol = pos.get("symbol", "")
            side = pos.get("side", "long").upper()
            qty = pos.get("qty", pos.get("quantity", 0))
            trade_id = pos.get("asset_id", pos.get("id", f"recovered-{symbol}"))

            # Check if already tracked
            existing = self.trade_manager.get_trade(trade_id)
            if existing and not existing.is_terminal:
                continue

            # Create lifecycle starting at ACTIVE state (position already open)
            lifecycle = TradeLifecycle(
                trade_id=trade_id,
                symbol=symbol,
                side=side,
            )

            # Fast-forward to ACTIVE state
            lifecycle.transition(TradeState.PENDING_RISK, "recovery")
            lifecycle.transition(TradeState.RISK_APPROVED, "recovery")
            lifecycle.transition(TradeState.SUBMITTED, "recovery")
            lifecycle.transition(TradeState.ACCEPTED, "recovery")
            lifecycle.transition(TradeState.FILLED, "recovery")
            lifecycle.transition(TradeState.ACTIVE, "recovery: reconstructed from broker")

            # Attach broker metadata
            lifecycle.metadata["recovered"] = True
            lifecycle.metadata["recovery_timestamp"] = datetime.now(timezone.utc).isoformat()
            lifecycle.metadata["broker_qty"] = qty
            lifecycle.metadata["broker_position"] = pos

            lifecycles.append(lifecycle)

        return lifecycles

    def _register_lifecycle(self, lifecycle: TradeLifecycle) -> None:
        """Register a recovered lifecycle with TradeManager."""
        with self.trade_manager._lock:
            self.trade_manager._trades[lifecycle.trade_id] = lifecycle

    def _detect_orphans(self, broker_positions: list, internal_trades: list) -> list:
        """Find positions in broker not tracked internally (orphans)."""
        internal_symbols = {t.symbol for t in internal_trades}
        internal_ids = {t.trade_id for t in internal_trades}

        orphans = []
        for pos in broker_positions:
            symbol = pos.get("symbol", "")
            asset_id = pos.get("asset_id", pos.get("id", ""))

            # A position is orphaned if neither its ID nor symbol is tracked
            if asset_id not in internal_ids and symbol not in internal_symbols:
                orphans.append(pos)

        return orphans

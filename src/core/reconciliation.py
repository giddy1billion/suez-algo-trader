"""
Portfolio Reconciliation Engine.

Periodically compares internal portfolio state against broker reality
and detects discrepancies for alerting or auto-correction.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.utils.logger import get_logger
from src.core.events import SystemHealth, EventBus
from src.core.state_machine import TradeLifecycle, TradeManager, TradeState

logger = get_logger(__name__)


# Discrepancy types
MISSING_INTERNAL = "MISSING_INTERNAL"
MISSING_BROKER = "MISSING_BROKER"
QTY_MISMATCH = "QTY_MISMATCH"
SIDE_MISMATCH = "SIDE_MISMATCH"
BROKER_ERROR = "BROKER_ERROR"


@dataclass
class Discrepancy:
    """A single discrepancy between broker and internal state."""

    symbol: str
    type: str  # MISSING_INTERNAL, MISSING_BROKER, QTY_MISMATCH, SIDE_MISMATCH, BROKER_ERROR
    broker_state: dict = field(default_factory=dict)
    internal_state: dict = field(default_factory=dict)
    severity: str = "MEDIUM"  # LOW, MEDIUM, HIGH


@dataclass
class ReconciliationReport:
    """Summary of a reconciliation run."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    broker_positions: int = 0
    internal_positions: int = 0
    discrepancies: list = field(default_factory=list)
    is_reconciled: bool = True


class PortfolioReconciler:
    """
    Compares internal portfolio state with broker and detects discrepancies.

    Can run periodically (every interval_seconds) or on-demand.
    """

    def __init__(
        self,
        broker,
        trade_manager: TradeManager,
        event_bus: EventBus,
        interval_seconds: int = 300,
    ):
        self.broker = broker
        self.trade_manager = trade_manager
        self.event_bus = event_bus
        self.interval_seconds = interval_seconds

    def reconcile(self) -> ReconciliationReport:
        """Compare internal state with broker and detect discrepancies."""
        report = ReconciliationReport()

        try:
            # 1. Get broker positions
            broker_positions = self.broker.get_positions()
            report.broker_positions = len(broker_positions)
        except Exception as e:
            logger.error("Failed to get broker positions for reconciliation: %s", str(e))
            report.is_reconciled = False
            report.discrepancies.append(
                Discrepancy(
                    symbol="N/A",
                    type=BROKER_ERROR,
                    broker_state={"error": str(e)},
                    internal_state={},
                    severity="HIGH",
                )
            )
            return report

        # 2. Get internal active trades from TradeManager
        internal_trades = self.trade_manager.get_active_trades()
        report.internal_positions = len(internal_trades)

        # Build lookup maps
        broker_by_symbol = {}
        for pos in broker_positions:
            symbol = pos.get("symbol", "")
            broker_by_symbol[symbol] = pos

        internal_by_symbol = {}
        for trade in internal_trades:
            internal_by_symbol[trade.symbol] = trade

        # 3. Compare: find mismatches
        discrepancies = []

        # Check for positions in broker but not internal (MISSING_INTERNAL)
        for symbol, pos in broker_by_symbol.items():
            if symbol not in internal_by_symbol:
                discrepancies.append(
                    Discrepancy(
                        symbol=symbol,
                        type=MISSING_INTERNAL,
                        broker_state=pos,
                        internal_state={},
                        severity="HIGH",
                    )
                )
            else:
                # Check qty and side mismatches
                trade = internal_by_symbol[symbol]
                broker_qty = abs(float(pos.get("qty", pos.get("quantity", 0))))
                internal_qty = abs(float(trade.metadata.get("broker_qty", 0)))
                broker_side = pos.get("side", "long").upper()
                internal_side = trade.side.upper()

                # Normalize side values
                if broker_side in ("LONG", "BUY"):
                    broker_side = "BUY"
                elif broker_side in ("SHORT", "SELL"):
                    broker_side = "SELL"
                if internal_side in ("LONG", "BUY"):
                    internal_side = "BUY"
                elif internal_side in ("SHORT", "SELL"):
                    internal_side = "SELL"

                if broker_side != internal_side:
                    discrepancies.append(
                        Discrepancy(
                            symbol=symbol,
                            type=SIDE_MISMATCH,
                            broker_state={"side": pos.get("side", ""), "qty": broker_qty},
                            internal_state={"side": trade.side, "qty": internal_qty},
                            severity="HIGH",
                        )
                    )
                elif broker_qty != internal_qty and internal_qty > 0:
                    discrepancies.append(
                        Discrepancy(
                            symbol=symbol,
                            type=QTY_MISMATCH,
                            broker_state={"qty": broker_qty, "side": pos.get("side", "")},
                            internal_state={"qty": internal_qty, "side": trade.side},
                            severity="MEDIUM",
                        )
                    )

        # Check for positions tracked internally but not in broker (MISSING_BROKER)
        for symbol, trade in internal_by_symbol.items():
            if symbol not in broker_by_symbol:
                discrepancies.append(
                    Discrepancy(
                        symbol=symbol,
                        type=MISSING_BROKER,
                        broker_state={},
                        internal_state={
                            "trade_id": trade.trade_id,
                            "side": trade.side,
                            "state": trade.state.value,
                        },
                        severity="HIGH",
                    )
                )

        report.discrepancies = discrepancies
        report.is_reconciled = len(discrepancies) == 0

        # 5. Publish warnings for each discrepancy
        for disc in discrepancies:
            self.event_bus.publish(
                SystemHealth(
                    component="portfolio_reconciler",
                    status="degraded",
                    metrics={
                        "symbol": disc.symbol,
                        "discrepancy_type": disc.type,
                        "severity": disc.severity,
                    },
                    source="PortfolioReconciler",
                )
            )

        if report.is_reconciled:
            logger.info("Reconciliation complete: all positions match")
        else:
            logger.warning(
                "Reconciliation found %d discrepancies", len(discrepancies)
            )

        return report

    def auto_fix(self, report: ReconciliationReport) -> list[str]:
        """
        Attempt automatic fixes for safe discrepancies.

        Only fixes MISSING_INTERNAL (creates lifecycle for broker position).
        Never auto-closes broker positions.
        """
        fixes = []

        for disc in report.discrepancies:
            if disc.type != MISSING_INTERNAL:
                continue

            symbol = disc.symbol
            pos = disc.broker_state
            side = pos.get("side", "long").upper()
            if side in ("LONG",):
                side = "BUY"
            elif side in ("SHORT",):
                side = "SELL"

            try:
                # Create a new lifecycle for this broker position
                trade = self.trade_manager.create_trade(
                    symbol=symbol,
                    side=side,
                    trade_id=pos.get("asset_id", f"reconciled-{symbol}"),
                )

                # Fast-forward to ACTIVE
                trade.transition(TradeState.PENDING_RISK, "auto-fix reconciliation")
                trade.transition(TradeState.RISK_APPROVED, "auto-fix reconciliation")
                trade.transition(TradeState.SUBMITTED, "auto-fix reconciliation")
                trade.transition(TradeState.ACCEPTED, "auto-fix reconciliation")
                trade.transition(TradeState.FILLED, "auto-fix reconciliation")
                trade.transition(TradeState.ACTIVE, "auto-fix: created from broker position")

                # Attach metadata
                qty = pos.get("qty", pos.get("quantity", 0))
                trade.metadata["recovered"] = True
                trade.metadata["auto_fixed"] = True
                trade.metadata["broker_qty"] = qty
                trade.metadata["broker_position"] = pos

                fixes.append(f"Created lifecycle for {symbol} (side={side})")
                logger.info("Auto-fix: created lifecycle for %s", symbol)

            except Exception as e:
                logger.error("Auto-fix failed for %s: %s", symbol, str(e))
                fixes.append(f"FAILED to fix {symbol}: {str(e)}")

        return fixes

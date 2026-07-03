"""
Core module — Event-driven architecture and trade lifecycle management.
"""

from .audit_log import AuditLogger
from .events import (
    Event,
    EventBus,
    OrderAccepted,
    OrderFilled,
    OrderPartialFill,
    OrderRejected,
    OrderSubmitted,
    RiskEvaluated,
    RiskHalt,
    SchedulerEvent,
    SignalGenerated,
    SystemHealth,
    TradeClosed,
    TradeOpened,
)
from .state_machine import TradeLifecycle, TradeManager, TradeState
from .subscribers import setup_default_subscribers

__all__ = [
    # Events
    "Event",
    "EventBus",
    "SignalGenerated",
    "RiskEvaluated",
    "OrderSubmitted",
    "OrderAccepted",
    "OrderPartialFill",
    "OrderFilled",
    "OrderRejected",
    "TradeOpened",
    "TradeClosed",
    "RiskHalt",
    "SchedulerEvent",
    "SystemHealth",
    # State Machine
    "TradeState",
    "TradeLifecycle",
    "TradeManager",
    # Audit
    "AuditLogger",
    # Setup
    "setup_default_subscribers",
]

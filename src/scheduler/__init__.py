"""
Research & Trading Scheduler — Dependency-aware activity orchestration.

Provides:
- MarketStatusService: unified market state per asset class
- AssetClassScheduler: top-level DAG-based orchestrator
- ActivityGraph: DAG of activities with trigger conditions
- Triggers: event-driven trigger types
"""

from src.scheduler.market_status import MarketStatusService
from src.scheduler.asset_class_scheduler import AssetClassScheduler
from src.scheduler.activity_graph import ActivityGraph, ActivityNode
from src.scheduler.triggers import (
    Trigger,
    DataArrivalTrigger,
    DriftTrigger,
    ScheduleTrigger,
    ParameterChangeTrigger,
    ManualTrigger,
    ModelTrainedTrigger,
)

__all__ = [
    "MarketStatusService",
    "AssetClassScheduler",
    "ActivityGraph",
    "ActivityNode",
    "Trigger",
    "DataArrivalTrigger",
    "DriftTrigger",
    "ScheduleTrigger",
    "ParameterChangeTrigger",
    "ManualTrigger",
    "ModelTrainedTrigger",
]

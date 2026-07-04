"""Shared models for the adaptive intelligence bounded context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MarketState:
    trend: str
    volatility: str
    liquidity: str
    correlation_env: str
    stress: str
    overall_regime: str
    confidence: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftState:
    sample_size: int
    baseline_accuracy: float
    recent_accuracy: float
    accuracy_drop: float
    degrading: bool


@dataclass
class TradeQualityResult:
    score: float
    threshold: float
    accepted: bool
    contributions: dict[str, float] = field(default_factory=dict)
    positives: list[str] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)


@dataclass
class AllocationDecision:
    qty_multiplier: float
    max_exposure_multiplier: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class RoutingDecision:
    enabled: bool
    reason: str


@dataclass
class IntelligenceDecision:
    accepted: bool
    final_score: float
    adjusted_confidence: float
    qty_multiplier: float
    explanation: str
    market_state: MarketState
    drift_state: DriftState
    quality: TradeQualityResult
    routing: RoutingDecision
    allocation: AllocationDecision

